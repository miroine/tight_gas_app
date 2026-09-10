"""
Calendar-style period resampling for reserves-report-style output.

Periods are anniversary-based (Year 1 = days [0, 365.25), Year 2 =
[365.25, 730.5), ... ; Month 1 = days [0, 30.4375), ...) relative to the
start of the forecast (t=0), not aligned to real calendar dates. This is
the standard convention in type-curve / reserves work (wells that came
on-stream on different real dates are compared and reported on "years on
production"), and it avoids needing a first-production-date input.

Works generically on any set of cumulative-volume series against the same
irregular time grid, via interpolation on the (monotonic) cumulative
curve at each period boundary - so it applies equally to a single well's
gas+condensate cumulative or a field's aggregate cumulative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_DAYS = 365.25
MONTH_DAYS = 365.25 / 12.0


def resample_periods(t_days: np.ndarray, cum_series: dict[str, np.ndarray],
                       period: str = "year") -> pd.DataFrame:
    """cum_series: {name: cumulative_array} - each array MUST be a monotonic
    non-decreasing cumulative volume aligned to t_days (same length).
    period: "year" or "month".
    Returns one row per period with volume produced and average rate over
    that period, per series (columns f"{name}_volume", f"{name}_avg_rate"),
    in whatever units the input cumulative arrays were in (volume units;
    avg_rate comes out as volume-unit / day).
    """
    t_days = np.asarray(t_days, dtype=float)
    if len(t_days) == 0:
        return pd.DataFrame()
    period_days = YEAR_DAYS if period == "year" else MONTH_DAYS
    t_max = float(t_days[-1])
    n_periods = max(int(np.ceil(t_max / period_days)), 1)

    rows = []
    for n in range(1, n_periods + 1):
        t0 = (n - 1) * period_days
        t1 = min(n * period_days, t_max)
        if t1 <= t0:
            break
        row = {
            "period": n,
            "label": f"Year {n}" if period == "year" else f"Month {n}",
            "start_day": t0,
            "end_day": t1,
            "days_in_period": t1 - t0,
        }
        for name, arr in cum_series.items():
            arr = np.asarray(arr, dtype=float)
            c0 = float(np.interp(t0, t_days, arr))
            c1 = float(np.interp(t1, t_days, arr))
            vol = max(c1 - c0, 0.0)
            row[f"{name}_volume"] = vol
            row[f"{name}_avg_rate"] = vol / (t1 - t0) if t1 > t0 else 0.0
        rows.append(row)
        if t1 >= t_max:
            break
    return pd.DataFrame(rows)
