"""
ML surrogate / proxy modeling for fast EUR estimation.

Two complementary uses of the same underlying "train a regressor on a
table of {features -> EUR}" utility (`train_regressor` below):

1. SYNTHETIC surrogate (`generate_training_data` + `train_regressor`):
   sample the existing physics engine across a parameter-uncertainty range
   (Latin hypercube via scipy.stats.qmc), run the full RTA/material-balance
   forecast for each sample, and train a scikit-learn regressor on
   {inputs -> EUR}. This is explicitly NOT an independent estimate - it is
   a fast APPROXIMATION of this app's own physics engine (a "proxy model"
   in reservoir-engineering terms), useful for near-instant sensitivity
   scanning and Monte-Carlo-style probabilistic EUR distributions
   (P10/P50/P90) that would be too slow to get by rerunning the full
   simulator for every one of thousands of Monte Carlo draws.
2. REAL-DATA model (`train_regressor` called directly on a user-supplied
   table): trains the same kind of regressor on a table of historical/
   offset wells (design parameters -> actual EUR) the user uploads. This
   IS a genuinely independent estimate, sourced from real outcomes rather
   than this app's own physics - but its quality is entirely limited by
   how much (and how representative) data is supplied. With the small
   sample sizes typical of an offset-well dataset (tens, not thousands, of
   wells), cross-validated metrics should be read as rough guidance, not a
   precise accuracy figure.

Neither path is a substitute for the physics engine or the history-match
calibration in `history_match.py` - they are complementary: the synthetic
surrogate is a fast stand-in for THIS model's own behavior, and the
real-data model is only as good as the offset-well analogy holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from .simulator import summarize
from .history_match import WELL_BUILDERS, _build_case  # reuse the same case-building wiring

try:
    from scipy.stats import qmc
    _HAVE_QMC = True
except Exception:  # pragma: no cover
    _HAVE_QMC = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score, mean_absolute_error
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


# Default sampling ranges for the synthetic surrogate. Each entry is
# (lo, hi, scale) where scale is "log" (sample log-uniform - appropriate
# for permeability, which spans orders of magnitude) or "linear".
# Integer-valued parameters (n_frac, n_branches) are rounded after sampling.
DEFAULT_RANGES = {
    "common": {
        "k_md": (1e-4, 1.0, "log"),
        "area_acres": (80.0, 2000.0, "linear"),
        "h_ft": (20.0, 300.0, "linear"),
        "pi": (4000.0, 12000.0, "linear"),
        "pwf": (300.0, 3000.0, "linear"),
    },
    "vertical": {
        "rw_ft": (0.2, 0.6, "linear"),
        "skin": (-2.0, 20.0, "linear"),
    },
    "mfhw": {
        "xf_ft": (50.0, 500.0, "linear"),
        "fcd": (1.0, 100.0, "linear"),
        "n_frac": (10, 60, "linear"),
        "lateral_length_ft": (3000.0, 15000.0, "linear"),
    },
    "fishbone": {
        "branch_length_ft": (200.0, 1500.0, "linear"),
        "n_branches": (4, 20, "linear"),
        "main_bore_length_ft": (3000.0, 12000.0, "linear"),
    },
}


def full_ranges(well_type: str) -> dict:
    """Common + well-type-specific sampling ranges, merged."""
    r = dict(DEFAULT_RANGES["common"])
    r.update(DEFAULT_RANGES[well_type])
    return r


def sample_params(ranges: dict, n_samples: int, seed: int | None = None) -> pd.DataFrame:
    """Latin-hypercube sample the given {name: (lo, hi, scale)} ranges into
    an (n_samples x n_params) DataFrame. Falls back to plain uniform random
    sampling if scipy's qmc module isn't available.
    """
    names = list(ranges.keys())
    d = len(names)
    rng = np.random.default_rng(seed)
    if _HAVE_QMC and n_samples >= 2:
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        unit = sampler.random(n=n_samples)
    else:
        unit = rng.uniform(0.0, 1.0, size=(n_samples, d))

    cols = {}
    for j, name in enumerate(names):
        lo, hi, scale = ranges[name]
        u = unit[:, j]
        if scale == "log":
            vals = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
        else:
            vals = lo + u * (hi - lo)
        cols[name] = vals
    df = pd.DataFrame(cols)
    for name in ("n_frac", "n_branches"):
        if name in df.columns:
            df[name] = df[name].round().astype(int).clip(lower=1)
    return df


def generate_training_data(pvt, well_type: str, ranges: dict, n_samples: int, fixed: dict,
                              t_max_days: float, n_steps: int = 180,
                              seed: int | None = None) -> pd.DataFrame:
    """Sample `ranges`, run the full forecast for each sample, and return a
    DataFrame with one row per sample: the sampled feature columns plus
    EUR_Bcf, Initial_rate_MMscfd, and OGIP_Bcf targets. Any well-design
    field not present in `ranges` is held at its value in `fixed`.

    `fixed` needs the same keys as `history_match._build_case` expects:
    area_acres, k_md, h_ft, phi, sw, pi, cf, pwf, p_abandon, q_min_mscfd,
    q_max_mscfd, well_params (dict) - used as the base/default for anything
    not being sampled.
    """
    samples = sample_params(ranges, n_samples, seed=seed)
    rows = []
    for _, s in samples.iterrows():
        area = float(s.get("area_acres", fixed["area_acres"]))
        k_md = float(s.get("k_md", fixed["k_md"]))
        h_ft = float(s.get("h_ft", fixed["h_ft"]))
        pi = float(s.get("pi", fixed["pi"]))
        pwf = float(s.get("pwf", fixed["pwf"]))
        well_params = dict(fixed["well_params"])
        for name in DEFAULT_RANGES.get(well_type, {}):
            if name in s.index:
                v = s[name]
                well_params[name] = int(v) if name in ("n_frac", "n_branches") else float(v)
        try:
            df, G, mb = _build_case(pvt, well_type, area, k_md, h_ft, fixed["phi"], fixed["sw"], pi,
                                       fixed["cf"], pwf, fixed["p_abandon"], fixed["q_min_mscfd"],
                                       fixed["q_max_mscfd"], well_params, t_max_days, n_steps=n_steps)
            summ = summarize(df, G)
            row = dict(s)
            row["EUR_Bcf"] = summ["EUR_Bcf"]
            row["Initial_rate_MMscfd"] = summ["Initial_rate_MMscfd"]
            row["OGIP_Bcf"] = summ["OGIP_Bcf"]
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


@dataclass
class RegressorResult:
    model: object
    feature_cols: list
    target_col: str
    cv_r2: float
    cv_mae: float
    n_train: int
    cv_folds: int
    feature_importance: pd.DataFrame
    oof_predictions: np.ndarray = dc_field(default=None)
    oof_actual: np.ndarray = dc_field(default=None)


def train_regressor(df: pd.DataFrame, feature_cols: list, target_col: str,
                      n_estimators: int = 300, max_cv_folds: int = 5,
                      seed: int = 0) -> RegressorResult:
    """Train a RandomForestRegressor on df[feature_cols] -> df[target_col],
    with k-fold cross-validation (folds auto-shrunk for small datasets - a
    real-data upload might only have a handful of offset wells) to give an
    honest out-of-sample R^2/MAE rather than an optimistic training-set score.
    """
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required for the ML surrogate / offset-well model.")

    data = df.dropna(subset=feature_cols + [target_col])
    n = len(data)
    if n < 8:
        raise ValueError(f"Need at least 8 complete rows to train a model reliably (got {n}).")

    X = data[feature_cols].values.astype(float)
    y = data[target_col].values.astype(float)

    n_folds = max(2, min(max_cv_folds, n // 4))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    model_cv = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, min_samples_leaf=2)
    oof_pred = cross_val_predict(model_cv, X, y, cv=kf)
    cv_r2 = float(r2_score(y, oof_pred))
    cv_mae = float(mean_absolute_error(y, oof_pred))

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, min_samples_leaf=2)
    model.fit(X, y)
    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    return RegressorResult(model=model, feature_cols=feature_cols, target_col=target_col,
                             cv_r2=cv_r2, cv_mae=cv_mae, n_train=n, cv_folds=n_folds,
                             feature_importance=importance, oof_predictions=oof_pred, oof_actual=y)


def predict_batch(reg: RegressorResult, samples: pd.DataFrame) -> np.ndarray:
    X = samples[reg.feature_cols].values.astype(float)
    return reg.model.predict(X)


def monte_carlo_eur(reg: RegressorResult, distributions: dict, base_values: dict,
                      n_draws: int = 5000, seed: int | None = None) -> np.ndarray:
    """Draw `n_draws` samples of each feature per `distributions` (feature
    not listed there is held fixed at `base_values[feature]`), run them all
    through the trained surrogate at once (vectorized - this is the whole
    point of a surrogate: a Monte Carlo run that would take many minutes
    through the full physics engine takes well under a second here), and
    return the resulting array of predicted EUR (or whatever `reg.target_col`
    is) draws.

    distributions: {feature: ("uniform", lo, hi) | ("triangular", lo, mode, hi)
                     | ("normal", mean, std) | ("lognormal", mean_ln, sigma_ln)}
    """
    rng = np.random.default_rng(seed)
    n = len(reg.feature_cols)
    X = np.zeros((n_draws, n))
    for j, feat in enumerate(reg.feature_cols):
        if feat in distributions:
            spec = distributions[feat]
            kind = spec[0]
            if kind == "uniform":
                _, lo, hi = spec
                X[:, j] = rng.uniform(lo, hi, n_draws)
            elif kind == "triangular":
                _, lo, mode, hi = spec
                X[:, j] = rng.triangular(lo, mode, hi, n_draws)
            elif kind == "normal":
                _, mean, std = spec
                X[:, j] = rng.normal(mean, std, n_draws)
            elif kind == "lognormal":
                _, mean_ln, sigma_ln = spec
                X[:, j] = rng.lognormal(mean_ln, sigma_ln, n_draws)
            else:
                raise ValueError(f"Unknown distribution kind: {kind}")
        else:
            X[:, j] = base_values.get(feat, 0.0)
    return reg.model.predict(X)
