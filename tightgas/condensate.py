"""
Condensate-gas ratio (CGR) and condensate volume post-processing.

This is intentionally a simple yield-based model, not a compositional /
EOS retrograde-condensate simulation: it multiplies the already-computed
gas rate by a condensate-gas ratio (STB/MMscf) to get a condensate rate,
optionally letting that CGR decline linearly with average reservoir
pressure between the initial value and an abandonment-pressure value (a
common, transparent way to approximate the drop in producible liquid
yield as a retrograde-condensate reservoir depletes below its dewpoint,
without a full PVT/EOS liquid-dropout calculation).

Because it only needs a well's already-computed gas rate and average
reservoir pressure at each time step, it is applied as a POST-PROCESSING
step on the DataFrames returned by simulator.run_forecast() or
field.run_field_forecast()'s per-well DataFrames - the core rate-transient
/ material-balance engine is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CondensateModel:
    cgr_i: float          # CGR at initial pressure, STB/MMscf
    pi: float              # initial reservoir pressure, psia (for interpolation)
    p_abandon: float        # abandonment pressure, psia (for interpolation)
    cgr_abandon: float | None = None  # CGR at abandonment pressure; None = constant CGR

    def cgr_at(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.cgr_abandon is None:
            return np.full_like(p, self.cgr_i, dtype=float)
        span = max(self.pi - self.p_abandon, 1e-6)
        frac = np.clip((self.pi - p) / span, 0.0, 1.0)
        return self.cgr_i + (self.cgr_abandon - self.cgr_i) * frac


def apply_condensate(df: pd.DataFrame, model: CondensateModel,
                       p_col: str = "p_avg_psia", q_gas_col: str = "q_mscfd",
                       t_col: str = "t_days") -> pd.DataFrame:
    """Return a copy of df with condensate columns added:
        cgr_stb_mmscf   - instantaneous CGR, STB/MMscf
        q_cond_bbld      - condensate rate, bbl/d (STB/d)
        Np_bbl           - cumulative condensate, bbl
        Np_mstb          - cumulative condensate, thousand bbl (Mstb)
    """
    df = df.copy()
    if p_col in df.columns:
        p = df[p_col].values
    else:
        p = np.full(len(df), model.pi)
    cgr = model.cgr_at(p)
    q_gas = df[q_gas_col].values
    q_cond_bbld = q_gas / 1000.0 * cgr  # (Mscf/d)/1000 = MMscf/d, * STB/MMscf = STB/d

    t = df[t_col].values
    if len(t) > 1:
        cum = np.concatenate(([0.0], np.cumsum(
            (q_cond_bbld[1:] + q_cond_bbld[:-1]) / 2.0 * np.diff(t))))
    else:
        cum = np.zeros(len(t))

    df["cgr_stb_mmscf"] = cgr
    df["q_cond_bbld"] = q_cond_bbld
    df["Np_bbl"] = cum
    df["Np_mstb"] = cum / 1000.0
    return df


def sum_condensate_across_wells(well_dfs: dict, t_grid: np.ndarray) -> pd.DataFrame:
    """Aggregate per-well condensate (each already run through
    apply_condensate) onto the shared field time grid to get field-level
    condensate rate and cumulative volume.
    """
    q_total = np.zeros(len(t_grid))
    for df in well_dfs.values():
        if "q_cond_bbld" in df.columns:
            q_total += df["q_cond_bbld"].values
    if len(t_grid) > 1:
        cum = np.concatenate(([0.0], np.cumsum(
            (q_total[1:] + q_total[:-1]) / 2.0 * np.diff(t_grid))))
    else:
        cum = np.zeros(len(t_grid))
    return pd.DataFrame({
        "t_days": t_grid,
        "q_cond_bbld": q_total,
        "Np_bbl": cum,
        "Np_mstb": cum / 1000.0,
    })
