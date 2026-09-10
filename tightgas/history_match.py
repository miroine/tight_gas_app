"""
Automated history-match calibration.

Given a well's actual production history (t_days, q_mscfd - optionally
p_avg_psia, though only rate is used for the fit itself), solves for the
best-fit values of a chosen subset of otherwise-uncertain reservoir /
completion parameters against the EXISTING rate-transient / material-
balance engine (simulator.run_forecast) - not a separate empirical curve.
This is the standard RTA workflow of calibrating an analytical model to
observed production before trusting its forward EUR extrapolation, and it
is the most defensible way this app can offer a "better estimate": it
improves the SAME validated physics with real data, rather than
substituting a black-box prediction for it.

Nonlinear least squares (scipy.optimize.least_squares) is run in log-rate
space - production data typically spans 1-3 decades of decline over a
well's life, so fitting linear rate residuals would let the first few
high-rate points dominate the fit and starve out the late-time behavior
that usually matters most for EUR.

t=0 observations are always excluded from the fit: the forward model's
time grid carries a synthetic t=0 seed point (q=0, an artifact of the
marching scheme, not a real "day zero" forecast - see simulator.py) that
would otherwise be compared against a real, nonzero first-day rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .pvt import GasPVT
from .material_balance import ogip_volumetric, MaterialBalanceTank
from .flow_models import ReservoirRock, VerticalWell, MultiFracHorizontalWell, FishboneWell
from .simulator import SimulationInputs, run_forecast, summarize
from .type_curves import goodness_of_fit

try:
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


WELL_BUILDERS = {
    "vertical": lambda p, area: VerticalWell(
        rw_ft=p.get("rw_ft", 0.35), skin=p.get("skin", 0.0), drainage_area_acres=area),
    "mfhw": lambda p, area: MultiFracHorizontalWell(
        lateral_length_ft=p.get("lateral_length_ft", 8000.0), n_frac=int(p.get("n_frac", 25)),
        xf_ft=p.get("xf_ft", 200.0), fcd=p.get("fcd", 20.0), drainage_area_acres=area,
        linear_calib=p.get("linear_calib", 1.0)),
    "fishbone": lambda p, area: FishboneWell(
        main_bore_length_ft=p.get("main_bore_length_ft", 6000.0), n_branches=int(p.get("n_branches", 8)),
        branch_length_ft=p.get("branch_length_ft", 600.0), drainage_area_acres=area,
        linear_calib=p.get("linear_calib", 0.5)),
}

# Parameters eligible for calibration per well type: name -> (kind, lo, hi).
# kind = "rock" (ReservoirRock.k_md), "area" (drainage_area_acres - feeds
# both OGIP and the well's own PSS geometry), or "well" (a completion
# parameter of that well type's dataclass / params dict).
CALIBRATABLE = {
    "vertical": {
        "k_md": ("rock", 1e-5, 50.0),
        "skin": ("well", -5.0, 50.0),
        "rw_ft": ("well", 0.1, 1.0),
        "area_acres": ("area", 20.0, 5000.0),
    },
    "mfhw": {
        "k_md": ("rock", 1e-5, 50.0),
        "xf_ft": ("well", 20.0, 1000.0),
        "fcd": ("well", 0.1, 500.0),
        "linear_calib": ("well", 0.1, 3.0),
        "area_acres": ("area", 20.0, 5000.0),
    },
    "fishbone": {
        "k_md": ("rock", 1e-5, 50.0),
        "branch_length_ft": ("well", 50.0, 3000.0),
        "linear_calib": ("well", 0.05, 3.0),
        "area_acres": ("area", 20.0, 5000.0),
    },
}


def _build_case(pvt, well_type, area_acres, k_md, h_ft, phi, sw, pi, cf, pwf, p_abandon,
                  q_min_mscfd, q_max_mscfd, well_params, t_max_days, n_steps=200):
    rock = ReservoirRock(k_md=k_md, phi=phi, h_ft=h_ft, stress_gamma=0.0)
    well = WELL_BUILDERS[well_type](well_params, area_acres)
    bgi = float(pvt.bg(pi))
    G = ogip_volumetric(area_acres, h_ft, phi, sw, bgi)
    mb = MaterialBalanceTank(pvt=pvt, G_scf=G, pi=pi, cf=cf)
    sim_in = SimulationInputs(pvt=pvt, rock=rock, well=well, mb=mb, pwf=pwf,
                                t_max_days=max(t_max_days, 1.0), p_abandon=p_abandon,
                                q_min_mscfd=q_min_mscfd, n_steps=n_steps, q_max_mscfd=q_max_mscfd)
    df = run_forecast(sim_in)
    return df, G, mb


@dataclass
class HistoryMatchResult:
    fitted: dict            # {param_name: calibrated value, field units}
    success: bool
    message: str
    df: object               # forecast DataFrame at fitted params, extended to the full forecast horizon
    summary: dict
    r2: float
    log_rmse: float
    n_eval: int
    t_obs: np.ndarray = dc_field(default=None)
    q_obs: np.ndarray = dc_field(default=None)
    q_sim_at_obs: np.ndarray = dc_field(default=None)


def calibrate(pvt: GasPVT, well_type: str, fixed: dict, param_names: list[str],
               initial_guess: dict, t_obs, q_obs, t_forecast_days: float,
               n_steps_search: int = 150, n_steps_final: int = 350,
               max_nfev: int = 60) -> HistoryMatchResult:
    """
    fixed: dict with keys area_acres, k_md, h_ft, phi, sw, pi, cf, pwf,
           p_abandon, q_min_mscfd, q_max_mscfd, well_params (dict) - base/
           current values for everything; entries named in `param_names`
           are overridden during the search.
    param_names: subset of CALIBRATABLE[well_type] keys to solve for
                  (1-4 recommended; more parameters than the data can
                  actually resolve will just wander within their bounds).
    initial_guess: {name: starting value} for each entry in param_names,
                    field units.
    t_obs, q_obs: observed production history, days and Mscf/d.
    t_forecast_days: horizon for the final returned forecast (independent
                      of how far the observed history itself extends).
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for history-match calibration.")

    t_obs = np.asarray(t_obs, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)
    mask = (t_obs > 0) & np.isfinite(q_obs) & (q_obs > 0)
    t_obs, q_obs = t_obs[mask], q_obs[mask]
    if len(t_obs) < 4:
        raise ValueError("Need at least 4 valid (t>0, q>0) observations to calibrate against.")

    order = np.argsort(t_obs)
    t_obs, q_obs = t_obs[order], q_obs[order]
    t_max_hist = float(t_obs.max())

    specs = CALIBRATABLE[well_type]
    unknown = [p for p in param_names if p not in specs]
    if unknown:
        raise ValueError(f"Not calibratable for well_type={well_type}: {unknown}")

    bounds_lo = [specs[n][1] for n in param_names]
    bounds_hi = [specs[n][2] for n in param_names]
    x0 = [float(np.clip(initial_guess[n], specs[n][1], specs[n][2])) for n in param_names]
    scales = [max(abs(v), 1e-9) for v in x0]

    n_eval = 0

    def unpack(x):
        k_md = fixed["k_md"]
        area = fixed["area_acres"]
        well_over = dict(fixed["well_params"])
        vals = {}
        for n, v in zip(param_names, x):
            v = float(v)
            vals[n] = v
            kind = specs[n][0]
            if kind == "rock":
                k_md = v
            elif kind == "area":
                area = v
            else:
                well_over[n] = v
        return k_md, area, well_over, vals

    def residuals(x):
        nonlocal n_eval
        n_eval += 1
        k_md, area, well_over, _ = unpack(x)
        try:
            df, G, mb = _build_case(pvt, well_type, area, k_md, fixed["h_ft"], fixed["phi"], fixed["sw"],
                                       fixed["pi"], fixed["cf"], fixed["pwf"], fixed["p_abandon"],
                                       fixed["q_min_mscfd"], fixed["q_max_mscfd"], well_over,
                                       t_max_hist, n_steps=n_steps_search)
        except Exception:
            return np.full(len(t_obs), 5.0)
        q_sim = np.interp(t_obs, df["t_days"].values, df["q_mscfd"].values)
        q_sim = np.where(q_sim > 0, q_sim, 1e-6)
        return np.log(q_sim) - np.log(q_obs)

    result = least_squares(residuals, x0=x0, bounds=(bounds_lo, bounds_hi), x_scale=scales,
                             max_nfev=max_nfev)

    k_md_f, area_f, well_over_f, fitted_vals = unpack(result.x)
    df_full, G_full, mb_full = _build_case(pvt, well_type, area_f, k_md_f, fixed["h_ft"], fixed["phi"],
                                              fixed["sw"], fixed["pi"], fixed["cf"], fixed["pwf"],
                                              fixed["p_abandon"], fixed["q_min_mscfd"], fixed["q_max_mscfd"],
                                              well_over_f, t_forecast_days, n_steps=n_steps_final)
    summ = summarize(df_full, G_full)

    q_sim_obs = np.interp(t_obs, df_full["t_days"].values, df_full["q_mscfd"].values)
    gof = goodness_of_fit(q_obs, q_sim_obs)

    return HistoryMatchResult(
        fitted=fitted_vals, success=bool(result.success), message=str(result.message),
        df=df_full, summary=summ, r2=gof["r2"], log_rmse=gof["log_rmse"], n_eval=n_eval,
        t_obs=t_obs, q_obs=q_obs, q_sim_at_obs=q_sim_obs,
    )
