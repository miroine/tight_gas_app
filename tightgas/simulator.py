"""
Couples a well deliverability model (flow_models.py) to the p/z tank
material balance (material_balance.py) to march a full-life production
profile forward in time under constant flowing-bottomhole-pressure (BHP)
operation.

Numerical scheme
-----------------
An explicit (lagged) time-marching scheme is used: the rate at step i is
evaluated using the average reservoir pressure obtained from the material
balance at the END of step i-1. With a log-spaced time grid (fine near
t=0, where the solution changes fastest, coarser at late time) this lag
introduces negligible error while keeping the model simple, fast, and easy
to audit. Each step's incremental cumulative production is added to the
running Gp used by the p/z material balance, closing the loop between
deliverability and depletion.

Stopping criteria: reservoir pressure at/below the abandonment pressure,
rate at/below the economic rate limit, cumulative production reaching
(numerically) the full OGIP, or the requested forecast horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pvt import GasPVT
from .material_balance import MaterialBalanceTank
from .flow_models import ReservoirRock


@dataclass
class SimulationInputs:
    pvt: GasPVT
    rock: ReservoirRock
    well: object                 # VerticalWell | MultiFracHorizontalWell | FishboneWell
    mb: MaterialBalanceTank
    pwf: float                   # constant flowing bottomhole pressure, psia
    t_max_days: float = 365.25 * 30
    p_abandon: float = 500.0
    q_min_mscfd: float = 5.0
    n_steps: int = 400
    q_max_mscfd: float | None = None
    """Facility/choke-constrained rate ceiling, Mscf/d. Real wells are almost
    always produced against a surface constraint (choke, pipeline, facility
    capacity) rather than the fully unconstrained reservoir potential,
    especially early in life. This also regularizes the analytical transient
    linear-flow solution used for fractured/multilateral wells, which is
    mathematically unbounded as t -> 0 (a well-known property of the
    idealized constant-pwf, infinite-conductivity solution); setting a
    realistic facility limit is standard practice and keeps the forecast
    physically sensible without needing a full wellbore-storage model."""


def run_forecast(inputs: SimulationInputs) -> pd.DataFrame:
    pvt = inputs.pvt
    rock = inputs.rock
    well = inputs.well
    mb = inputs.mb
    pi = mb.pi

    mu_i = float(pvt.mu(pi))
    cti = float(pvt.cg(pi))

    t_grid = np.concatenate((
        [0.0],
        np.geomspace(0.01, max(inputs.t_max_days, 1.0), inputs.n_steps),
    ))

    n = len(t_grid)
    q = np.zeros(n)
    p_avg = np.zeros(n)
    gp = np.zeros(n)
    regime = np.empty(n, dtype=object)

    p_avg[0] = pi
    gp[0] = 0.0

    last_valid = 0
    for i in range(1, n):
        p_prev = p_avg[i - 1]
        if p_prev <= inputs.pwf + 1.0:
            # reservoir pressure has fallen to (near) flowing pressure: dead well
            q[i] = 0.0
            regime[i] = "depleted"
            p_avg[i] = p_prev
            gp[i] = gp[i - 1]
            last_valid = i
            break

        qi, reg = well.rate(pvt, rock, pi, p_prev, inputs.pwf, t_grid[i], mu_i, cti)
        qi = max(qi, 0.0)
        if inputs.q_max_mscfd is not None:
            qi = min(qi, inputs.q_max_mscfd)
        q[i] = qi
        regime[i] = reg

        dt = t_grid[i] - t_grid[i - 1]
        gp[i] = gp[i - 1] + q[i] * 1000.0 * dt  # Mscf/d * d * 1000 = scf
        gp[i] = min(gp[i], 0.999999 * mb.G_scf)
        p_avg[i] = mb.p_from_gp(gp[i])

        last_valid = i

        if p_avg[i] <= inputs.p_abandon or q[i] <= inputs.q_min_mscfd:
            break

    t_grid = t_grid[: last_valid + 1]
    q = q[: last_valid + 1]
    p_avg = p_avg[: last_valid + 1]
    gp = gp[: last_valid + 1]
    regime = regime[: last_valid + 1]
    regime[0] = regime[1] if n > 1 else "transient"

    dm = np.array([pvt.dm(p, inputs.pwf) for p in p_avg])
    poz = np.array([mb.poz(p) for p in p_avg])
    regime = np.array([r.value if hasattr(r, "value") else str(r) for r in regime], dtype=object)

    df = pd.DataFrame({
        "t_days": t_grid,
        "t_years": t_grid / 365.25,
        "q_mscfd": q,
        "q_mmscfd": q / 1000.0,
        "Gp_scf": gp,
        "Gp_bcf": gp / 1e9,
        "p_avg_psia": p_avg,
        "p_over_z": poz,
        "dm_psi2_cp": dm,
        "regime": regime,
    })
    return df


def summarize(df: pd.DataFrame, G_scf: float) -> dict:
    gp_final = float(df["Gp_scf"].iloc[-1])
    return {
        "EUR_Bcf": gp_final / 1e9,
        "OGIP_Bcf": G_scf / 1e9,
        "Recovery_factor_pct": 100.0 * gp_final / G_scf,
        "Initial_rate_MMscfd": float(df["q_mmscfd"].iloc[1]) if len(df) > 1 else 0.0,
        "Life_years": float(df["t_years"].iloc[-1]),
        "Final_reservoir_pressure_psia": float(df["p_avg_psia"].iloc[-1]),
    }
