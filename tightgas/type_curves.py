"""
Empirical decline (Arps) utilities and Blasingame-style rate-transient
type-curve diagnostics.

Arps (1945) equations are the industry-standard empirical decline family:

    exponential (b=0):      q(t) = qi * exp(-Di*t)
    hyperbolic (0<b<1..2):  q(t) = qi / (1 + b*Di*t)^(1/b)
    harmonic (b=1):         q(t) = qi / (1 + Di*t)

Blasingame (1991) / Agarwal-Gardner normalization is the standard way
tight-gas rate-transient analysts diagnose flow regime and compare wells
without needing an independent, possibly-uncertain re/rwa geometry:

    normalized rate      = q / (m(pi) - m(pwf))
    material balance time = Gp / q

On a log-log plot of normalized rate vs material balance time, the
Fetkovich-McCray depletion (boundary-dominated) stems collapse onto the
same curves as the dimensionless Arps family qDd = 1/(1+b*tDd)^(1/b), which
is what is used below as the reference "known type curve" overlay.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def arps_rate(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if b <= 1e-6:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def arps_cum(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if b <= 1e-6:
        return qi / di * (1.0 - np.exp(-di * t))
    if abs(b - 1.0) < 1e-6:
        return qi / di * np.log(1.0 + di * t)
    return qi / (di * (1.0 - b)) * (1.0 - np.power(1.0 + b * di * t, 1.0 - 1.0 / b))


def fit_arps(t: np.ndarray, q: np.ndarray, b_bounds=(0.0, 2.0)):
    """Least-squares fit of qi, Di, b to a rate-vs-time series.

    Returns (qi, di, b, success: bool). Falls back to a coarse grid search
    if scipy is unavailable or the fit fails to converge.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (t > 0) & (q > 0)
    t, q = t[mask], q[mask]
    if len(t) < 3:
        return float(q[0]) if len(q) else 0.0, 0.1, 0.5, False

    qi0 = float(q[0])

    if _HAVE_SCIPY:
        try:
            popt, _ = curve_fit(
                arps_rate, t, q,
                p0=[qi0, 0.01, 0.5],
                bounds=([qi0 * 0.5, 1e-6, b_bounds[0]], [qi0 * 2.0, 5.0, b_bounds[1]]),
                maxfev=20000,
            )
            return float(popt[0]), float(popt[1]), float(popt[2]), True
        except Exception:
            pass

    # coarse grid-search fallback
    best = (qi0, 0.01, 0.5, np.inf)
    for b in np.linspace(b_bounds[0], b_bounds[1], 9):
        for di in np.geomspace(1e-4, 3.0, 30):
            qhat = arps_rate(t, qi0, di, b)
            sse = np.sum((np.log(qhat + 1e-9) - np.log(q + 1e-9)) ** 2)
            if sse < best[3]:
                best = (qi0, di, b, sse)
    return best[0], best[1], best[2], False


def duong_rate(t: np.ndarray, q1: float, a: float, m: float) -> np.ndarray:
    """Duong (2011) rate model - built for linear-flow-dominated
    unconventional/tight wells where transient linear flow persists for
    most of the well's producing life (SPE 137748, "An Unconventional
    Rate Decline Approach for Tight and Fracture-Dominated Gas Wells").

    Derived from the empirical observation that q(t)/Gp(t) ~ a*t^-m holds
    closely for this well class:  q(t) = q1 * t^-m * exp[a/(1-m)*(t^(1-m)-1)]
    (t in days). m is typically in (1, 2) for tight/unconventional wells -
    m -> 1 approaches a boundary case handled separately below. q1 is a
    fitted rate constant, not simply q at t=1 day.

    Not defined at t=0 (by construction, since it comes from a t>0 ratio);
    callers should clip t away from 0 before evaluating.
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, 1e-3, None)
    if abs(1.0 - m) < 1e-6:
        expo = a * np.log(t)
    else:
        expo = a / (1.0 - m) * (np.power(t, 1.0 - m) - 1.0)
    expo = np.clip(expo, -700.0, 700.0)
    return q1 * np.power(t, -m) * np.exp(expo)


def fit_duong(t: np.ndarray, q: np.ndarray):
    """Fit (q1, a, m) via Duong's own ratio-method for the initial guess
    (linear regression of ln(q/Gp) vs ln(t), since q/Gp ~ a*t^-m is linear
    in log-log space) followed by a nonlinear log-space refinement.
    Returns a dict(q1, a, m, success).
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (t > 0) & (q > 0)
    t, q = t[mask], q[mask]
    if len(t) < 5:
        return dict(q1=float(q[0]) if len(q) else 0.0, a=1.0, m=1.1, success=False)

    gp = np.concatenate(([0.0], np.cumsum((q[1:] + q[:-1]) / 2.0 * np.diff(t))))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(gp > 0, q / np.where(gp > 0, gp, np.nan), np.nan)
    valid = np.isfinite(ratio) & (ratio > 0)
    if valid.sum() >= 3:
        slope, intercept = np.polyfit(np.log(t[valid]), np.log(ratio[valid]), 1)
        m0 = float(np.clip(-slope, 1.01, 1.99))
        a0 = float(np.clip(np.exp(intercept), 1e-4, 50.0))
    else:
        m0, a0 = 1.1, 1.0
    q10 = float(q[0])

    if _HAVE_SCIPY:
        try:
            def _resid(tt, q1, a, m):
                return np.log(duong_rate(tt, q1, a, m) + 1e-9)
            popt, _ = curve_fit(_resid, t, np.log(q + 1e-9), p0=[q10, a0, m0],
                                  bounds=([q10 * 0.1, 1e-4, 1.001], [q10 * 10.0, 50.0, 1.999]),
                                  maxfev=20000)
            return dict(q1=float(popt[0]), a=float(popt[1]), m=float(popt[2]), success=True)
        except Exception:
            pass
    return dict(q1=q10, a=a0, m=m0, success=False)


def ple_rate(t: np.ndarray, qi: float, dinf: float, delta_d: float, n_hat: float) -> np.ndarray:
    """Power-Law Exponential model (Ilk, Currie & Blasingame, 2008 - SPE
    116731), reparameterized so the "terminal decline >= 0" and "early
    decline >= terminal decline" physical constraints are automatically
    satisfied by simple non-negativity bounds on dinf and delta_d:

        D(t) = dinf + delta_d * t^(n_hat-1)      (instantaneous decline)
        q(t) = qi * exp[-dinf*t - (delta_d/n_hat)*t^n_hat]

    dinf = terminal (minimum, late-time) exponential decline rate, 1/day -
    the power-law term decays away and D(t) -> dinf, giving Arps-exponential
    behavior at late time (consistent with this app's own boundary-dominated
    regime). delta_d >= 0 is the excess early-time decline above dinf.
    n_hat in (0,1) controls how quickly that excess decays with time.
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, 0.0, None)
    n_hat = max(n_hat, 1e-6)
    expo = -dinf * t - (delta_d / n_hat) * np.power(t, n_hat)
    expo = np.clip(expo, -700.0, 700.0)
    return qi * np.exp(expo)


def fit_ple(t: np.ndarray, q: np.ndarray):
    """Fit (qi, dinf, delta_d, n_hat) by nonlinear least squares in log-rate
    space. Returns a dict with `di` (= dinf + delta_d, the model's implied
    early-time decline constant) added for readability alongside the raw
    fitted parameters.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (t >= 0) & (q > 0)
    t, q = t[mask], q[mask]
    if len(t) < 5:
        qi0 = float(q[0]) if len(q) else 0.0
        return dict(qi=qi0, dinf=0.01, delta_d=0.05, n_hat=0.5, di=0.06, success=False)

    qi0 = float(q[0]) if t[0] == 0 else float(q[0]) * 1.05

    if _HAVE_SCIPY:
        try:
            def _resid(tt, qi, dinf, delta_d, n_hat):
                return np.log(ple_rate(tt, qi, dinf, delta_d, n_hat) + 1e-9)
            popt, _ = curve_fit(_resid, t, np.log(q + 1e-9),
                                  p0=[qi0, 0.001, 0.05, 0.5],
                                  bounds=([qi0 * 0.5, 0.0, 0.0, 0.05],
                                          [qi0 * 2.0, 2.0, 5.0, 0.999]),
                                  maxfev=20000)
            qi_f, dinf_f, dd_f, nh_f = [float(x) for x in popt]
            return dict(qi=qi_f, dinf=dinf_f, delta_d=dd_f, n_hat=nh_f,
                         di=dinf_f + dd_f, success=True)
        except Exception:
            pass
    return dict(qi=qi0, dinf=0.001, delta_d=0.05, n_hat=0.5, di=0.051, success=False)


def sedm_rate(t: np.ndarray, qi: float, tau: float, n: float) -> np.ndarray:
    """Stretched Exponential Decline Model (Valko & Lee, 2010 - SPE 134231):
    q(t) = qi * exp[-(t/tau)^n], 0 < n <= 1 typically for unconventional
    wells (n=1 reduces to ordinary Arps-exponential decline). tau is a
    characteristic time constant, days.
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, 0.0, None)
    tau = max(tau, 1e-6)
    expo = -np.power(t / tau, n)
    expo = np.clip(expo, -700.0, 700.0)
    return qi * np.exp(expo)


def fit_sedm(t: np.ndarray, q: np.ndarray):
    """Fit (qi, tau, n) by nonlinear least squares in log-rate space."""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (t >= 0) & (q > 0)
    t, q = t[mask], q[mask]
    if len(t) < 4:
        qi0 = float(q[0]) if len(q) else 0.0
        return dict(qi=qi0, tau=365.0, n=0.5, success=False)

    qi0 = float(q[0]) if t[0] == 0 else float(q[0]) * 1.05
    tau0 = float(np.median(t)) if len(t) else 365.0

    if _HAVE_SCIPY:
        try:
            def _resid(tt, qi, tau, n):
                return np.log(sedm_rate(tt, qi, tau, n) + 1e-9)
            popt, _ = curve_fit(_resid, t, np.log(q + 1e-9), p0=[qi0, tau0, 0.5],
                                  bounds=([qi0 * 0.5, 1e-2, 0.05], [qi0 * 2.0, max(t) * 20.0, 1.5]),
                                  maxfev=20000)
            return dict(qi=float(popt[0]), tau=float(popt[1]), n=float(popt[2]), success=True)
        except Exception:
            pass
    return dict(qi=qi0, tau=tau0, n=0.5, success=False)


def goodness_of_fit(q_obs: np.ndarray, q_pred: np.ndarray) -> dict:
    """R^2, RMSE (linear rate space) and log-RMSE (relative-error-flavored,
    fairer when a series spans several decades of decline, which raw RMSE
    would otherwise let the early high-rate points dominate)."""
    q_obs = np.asarray(q_obs, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    resid = q_obs - q_pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((q_obs - np.mean(q_obs)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_resid = np.log(np.where(q_obs > 0, q_obs, np.nan)) - np.log(np.where(q_pred > 0, q_pred, np.nan))
    log_rmse = float(np.sqrt(np.nanmean(log_resid ** 2))) if np.isfinite(log_resid).any() else float("nan")
    return dict(r2=r2, rmse=rmse, log_rmse=log_rmse)


# Registry used by the "decline-curve library" UI to fit every family in one
# pass and rank them - each entry maps a display name to (fit_fn, rate_fn,
# param_names). fit_fn(t, q) -> params dict (with a "success" flag);
# rate_fn(t, **params) -> fitted rate array.
DECLINE_MODELS = {
    "Arps": dict(
        fit=lambda t, q: dict(zip(("qi", "di", "b", "success"), fit_arps(t, q))),
        rate=lambda t, p: arps_rate(t, p["qi"], p["di"], p["b"]),
        param_names=["qi", "di", "b"],
        citation="Arps (1945)",
    ),
    "Duong": dict(
        fit=fit_duong,
        rate=lambda t, p: duong_rate(t, p["q1"], p["a"], p["m"]),
        param_names=["q1", "a", "m"],
        citation="Duong (2011), SPE 137748",
    ),
    "Power-Law Exponential (PLE)": dict(
        fit=fit_ple,
        rate=lambda t, p: ple_rate(t, p["qi"], p["dinf"], p["delta_d"], p["n_hat"]),
        param_names=["qi", "dinf", "delta_d", "n_hat"],
        citation="Ilk, Currie & Blasingame (2008), SPE 116731",
    ),
    "Stretched Exponential (SEDM)": dict(
        fit=fit_sedm,
        rate=lambda t, p: sedm_rate(t, p["qi"], p["tau"], p["n"]),
        param_names=["qi", "tau", "n"],
        citation="Valko & Lee (2010), SPE 134231",
    ),
}


def fit_all_models(t: np.ndarray, q: np.ndarray) -> dict:
    """Fit every registered decline-curve family to the same (t, q) series
    and score each by goodness-of-fit, for side-by-side comparison /
    benchmarking. Returns {model_name: {params, success, rate_fn(t),
    r2, rmse, log_rmse, citation}}, sorted by no particular order (the
    caller sorts by whichever metric it prefers - log_rmse is the more
    robust ranking criterion for decline data spanning multiple decades).
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (t >= 0) & (q > 0)
    t_m, q_m = t[mask], q[mask]

    results = {}
    for name, spec in DECLINE_MODELS.items():
        try:
            params = spec["fit"](t_m, q_m)
            q_fit = spec["rate"](t_m, params)
            gof = goodness_of_fit(q_m, q_fit)
        except Exception:
            params, gof = dict(success=False), dict(r2=float("nan"), rmse=float("nan"), log_rmse=float("nan"))
        results[name] = dict(
            params=params, success=bool(params.get("success", False)),
            rate_fn=(lambda tt, _spec=spec, _p=params: _spec["rate"](tt, _p)),
            citation=spec["citation"], **gof,
        )
    return results


def fetkovich_mccray_qDd(tDd: np.ndarray, b: float) -> np.ndarray:
    """Dimensionless depletion-stem rate of the Fetkovich-McCray / Arps
    family in normalized (Blasingame) space: qDd = 1/(1+b*tDd)^(1/b).
    """
    tDd = np.asarray(tDd, dtype=float)
    if b <= 1e-6:
        return np.exp(-tDd)
    return np.power(1.0 + b * tDd, -1.0 / b)


def blasingame_normalize(t_days: np.ndarray, q: np.ndarray, dm: np.ndarray,
                          gp: np.ndarray):
    """Compute Blasingame-style normalized rate and material balance time.

    q        : rate series, Mscf/d
    dm       : m(pi)-m(pwf) series, psi^2/cp
    gp       : cumulative production series, scf, aligned with q

    Returns (t_mb [days], q_over_dm [Mscf/d per psi^2/cp]).
    """
    q = np.asarray(q, dtype=float)
    dm = np.asarray(dm, dtype=float)
    gp = np.asarray(gp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mb = np.where(q > 0, gp / 1000.0 / np.where(q > 0, q, np.nan), np.nan)  # Mscf/(Mscf/d) = days
        q_over_dm = np.where(dm > 0, q / dm, np.nan)
    return t_mb, q_over_dm
