"""
Multi-well field / drainage-strategy layer.

Builds on the single-well engine (pvt.py, material_balance.py,
flow_models.py) to forecast a *portfolio* of wells brought online at
different times, aggregated to a field-level production profile that is
fitted to a target facility/production capacity and a field-level
abandonment rate.

Modeling approach
------------------
Each well is modeled as its own INDEPENDENT tank (own OGIP, own p/z
material balance, own deliverability) - i.e. no explicit pressure
communication / interference between wells is solved for. This is the
standard simplifying assumption for tight-gas development planning at
typical spacing (permeability is low enough that wells are usually
designed not to meaningfully interfere within the forecast horizon); it
is NOT appropriate for tightly-spaced infill wells where interference is
significant. Wells share a common produced-fluid PVT system (gas
composition, temperature) and, optionally, common rock-compressibility /
stress-sensitivity behavior, but each well has its own permeability, net
pay, drainage area, initial pressure, completion design, and operating
constraints.

Field-level coupling happens through two mechanisms only:

  1. A shared facility/production-capacity ceiling. At every point in
     time, if the sum of all online wells' deliverable potential exceeds
     the capacity, every online well is curtailed by the SAME proportional
     factor so the field total exactly equals the capacity (proportional
     allocation - the simplest and most common capacity-sharing rule).
     Curtailment reduces a well's *produced* rate but its own material
     balance still only depletes by what it actually produced, so a
     curtailed well's life is correspondingly extended.
  2. A field-level abandonment rate. The reported forecast stops once the
     total field rate falls to/below this threshold (or once every well
     has individually reached its own abandonment pressure / economic
     rate limit, whichever comes first).

A lightweight, explicitly "first-pass, not an optimizer" greedy scheduler
(`suggest_schedule`) is also provided: given a single representative
"type well" rate profile, it proposes how many wells and roughly when to
bring them online to reach and hold a target plateau capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from .pvt import GasPVT
from .material_balance import ogip_volumetric, MaterialBalanceTank
from .flow_models import ReservoirRock, VerticalWell, MultiFracHorizontalWell, FishboneWell


# --------------------------------------------------------------------------
# Well specification & construction
# --------------------------------------------------------------------------
@dataclass
class WellSpec:
    name: str
    well_type: str  # "vertical" | "mfhw" | "fishbone"
    start_day: float
    area_acres: float
    k_md: float
    h_ft: float
    pi: float
    pwf: float
    p_abandon: float
    q_min_mscfd: float
    q_max_mmscfd: float | None
    params: dict = dc_field(default_factory=dict)


def build_well_object(spec: WellSpec):
    p = spec.params
    if spec.well_type == "vertical":
        return VerticalWell(rw_ft=p.get("rw_ft", 0.35), skin=p.get("skin", 0.0),
                              drainage_area_acres=spec.area_acres)
    if spec.well_type == "mfhw":
        return MultiFracHorizontalWell(
            lateral_length_ft=p.get("lateral_length_ft", 8000),
            n_frac=int(p.get("n_frac", 25)),
            xf_ft=p.get("xf_ft", 200),
            fcd=p.get("fcd", 20.0),
            drainage_area_acres=spec.area_acres,
            linear_calib=p.get("linear_calib", 1.0),
            transition_tD=p.get("transition_tD", 0.25),
        )
    if spec.well_type == "fishbone":
        return FishboneWell(
            main_bore_length_ft=p.get("main_bore_length_ft", 6000),
            n_branches=int(p.get("n_branches", 8)),
            branch_length_ft=p.get("branch_length_ft", 600),
            drainage_area_acres=spec.area_acres,
            linear_calib=p.get("linear_calib", 0.5),
            transition_tD=p.get("transition_tD", 0.25),
        )
    raise ValueError(f"Unknown well_type: {spec.well_type}")


# --------------------------------------------------------------------------
# Time grid: fine near t=0 (field start) AND fine near each well's own start
# --------------------------------------------------------------------------
def build_field_time_grid(t_max_days: float, well_starts: list[float],
                            n_base: int = 300, max_total_points: int = 4000) -> np.ndarray:
    n_wells = max(len(well_starts), 1)
    n_per_well = max(30, min(150, max_total_points // n_wells))

    grid = {0.0, float(t_max_days)}
    grid.update(np.geomspace(0.01, t_max_days, n_base).tolist())
    for s in well_starts:
        s = float(s)
        grid.add(s)
        remaining = t_max_days - s
        if remaining > 0.02:
            local = s + np.geomspace(0.01, remaining, n_per_well)
            grid.update(local.tolist())
    arr = np.array(sorted(g for g in grid if 0.0 <= g <= t_max_days))
    return arr


# --------------------------------------------------------------------------
# Field-level coupled forecast
# --------------------------------------------------------------------------
@dataclass
class FieldInputs:
    pvt: GasPVT
    phi: float
    sw: float
    cf: float
    stress_gamma: float
    wells: list[WellSpec]
    capacity_mmscfd: float | None   # None / 0 = uncapped
    field_abandon_mmscfd: float
    t_max_days: float
    n_base_steps: int = 300


def run_field_forecast(inp: FieldInputs):
    pvt = inp.pvt
    wells = inp.wells
    n = len(wells)

    rocks, mus_i, ctis, mbs, well_objs, G_list = [], [], [], [], [], []
    for w in wells:
        rock = ReservoirRock(k_md=w.k_md, phi=inp.phi, h_ft=w.h_ft, stress_gamma=inp.stress_gamma)
        bgi = float(pvt.bg(w.pi))
        G = ogip_volumetric(w.area_acres, w.h_ft, inp.phi, inp.sw, bgi)
        mb = MaterialBalanceTank(pvt=pvt, G_scf=G, pi=w.pi, cf=inp.cf)
        rocks.append(rock)
        mus_i.append(float(pvt.mu(w.pi)))
        ctis.append(float(pvt.cg(w.pi)))
        mbs.append(mb)
        well_objs.append(build_well_object(w))
        G_list.append(G)

    starts = [w.start_day for w in wells]
    t_grid = build_field_time_grid(inp.t_max_days, starts, n_base=inp.n_base_steps)
    nt = len(t_grid)

    Gp = np.zeros(n)
    q_actual = np.zeros((nt, n))
    p_avg_arr = np.full((nt, n), np.nan)
    status = ["pending"] * n
    regime_arr = np.empty((nt, n), dtype=object)
    regime_arr[:] = ""

    field_abandon_mscfd = inp.field_abandon_mmscfd * 1000.0
    cap_mscfd = inp.capacity_mmscfd * 1000.0 if inp.capacity_mmscfd else None

    last_valid = 0
    for it in range(1, nt):
        t = t_grid[it]
        potentials = np.zeros(n)
        for i, w in enumerate(wells):
            if status[i] == "shut-in":
                continue
            if t < w.start_day:
                status[i] = "pending"
                continue
            status[i] = "online" if status[i] == "pending" else status[i]

            p_prev = p_avg_arr[it - 1, i] if it > 1 and np.isfinite(p_avg_arr[it - 1, i]) else w.pi
            if p_prev <= w.pwf + 1.0:
                status[i] = "shut-in"
                continue

            t_local = t - w.start_day
            q_pot, reg = well_objs[i].rate(pvt, rocks[i], w.pi, p_prev, w.pwf, t_local, mus_i[i], ctis[i])
            q_pot = max(q_pot, 0.0)
            if w.q_max_mmscfd:
                q_pot = min(q_pot, w.q_max_mmscfd * 1000.0)

            # individual economic / abandonment check uses the UNCURTAILED potential
            if p_prev <= w.p_abandon or q_pot <= w.q_min_mscfd:
                status[i] = "shut-in"
                continue

            potentials[i] = q_pot
            regime_arr[it, i] = reg

        total_potential = potentials.sum()
        curtail = 1.0
        if cap_mscfd is not None and total_potential > cap_mscfd > 0:
            curtail = cap_mscfd / total_potential

        dt = t - t_grid[it - 1]
        for i in range(n):
            if potentials[i] <= 0:
                q_actual[it, i] = 0.0
                p_avg_arr[it, i] = p_avg_arr[it - 1, i] if it > 1 else wells[i].pi
                continue
            actual = potentials[i] * curtail
            q_actual[it, i] = actual
            Gp[i] += actual * 1000.0 * dt
            Gp[i] = min(Gp[i], 0.999999 * G_list[i])
            p_avg_arr[it, i] = mbs[i].p_from_gp(Gp[i])

        field_rate = q_actual[it].sum()
        last_valid = it

        any_active = any(s in ("online", "pending") for s in status)
        if not any_active:
            break
        if field_rate <= field_abandon_mscfd and any(s == "online" for s in status) and it > 2:
            # require the field to have actually ramped up at least once before honoring
            # the abandonment cutoff, so an initial ramp-up period isn't mistaken for decline
            if q_actual[:it + 1].sum(axis=1).max() > field_abandon_mscfd:
                break

    t_grid = t_grid[: last_valid + 1]
    q_actual = q_actual[: last_valid + 1]
    p_avg_arr = p_avg_arr[: last_valid + 1]
    regime_arr = regime_arr[: last_valid + 1]

    field_q_mscfd = q_actual.sum(axis=1)
    field_gp_scf = np.concatenate(([0.0], np.cumsum(
        (field_q_mscfd[1:] + field_q_mscfd[:-1]) / 2.0 * np.diff(t_grid) * 1000.0)))

    field_df = pd.DataFrame({
        "t_days": t_grid,
        "t_years": t_grid / 365.25,
        "q_mscfd": field_q_mscfd,
        "q_mmscfd": field_q_mscfd / 1000.0,
        "Gp_scf": field_gp_scf,
        "Gp_bcf": field_gp_scf / 1e9,
        "n_wells_online": (q_actual > 0).sum(axis=1),
    })

    well_dfs = {}
    well_summaries = {}
    for i, w in enumerate(wells):
        gp_i = np.concatenate(([0.0], np.cumsum(
            (q_actual[1:, i] + q_actual[:-1, i]) / 2.0 * np.diff(t_grid) * 1000.0)))
        df_i = pd.DataFrame({
            "t_days": t_grid,
            "t_years": t_grid / 365.25,
            "q_mscfd": q_actual[:, i],
            "q_mmscfd": q_actual[:, i] / 1000.0,
            "Gp_scf": gp_i,
            "Gp_bcf": gp_i / 1e9,
            "p_avg_psia": p_avg_arr[:, i],
            "regime": regime_arr[:, i],
        })
        well_dfs[w.name] = df_i
        eur = float(gp_i[-1])
        well_summaries[w.name] = {
            "well_type": w.well_type,
            "start_day": w.start_day,
            "OGIP_Bcf": G_list[i] / 1e9,
            "EUR_Bcf": eur / 1e9,
            "Recovery_factor_pct": 100.0 * eur / G_list[i] if G_list[i] > 0 else 0.0,
            "peak_rate_MMscfd": float(np.nanmax(q_actual[:, i])) / 1000.0,
            "final_status": status[i],
        }

    field_summary = {
        "n_wells": n,
        "field_EUR_Bcf": float(field_gp_scf[-1]) / 1e9,
        "peak_field_rate_MMscfd": float(field_q_mscfd.max()) / 1000.0 if len(field_q_mscfd) else 0.0,
        "field_life_years": float(t_grid[-1] / 365.25) if len(t_grid) else 0.0,
        "capacity_MMscfd": inp.capacity_mmscfd,
    }

    return field_df, well_dfs, well_summaries, field_summary


# --------------------------------------------------------------------------
# Simple (non-optimizing) well-count / timing suggestion helper
# --------------------------------------------------------------------------
def suggest_schedule(type_well_t_days: np.ndarray, type_well_q_mscfd: np.ndarray,
                       capacity_mmscfd: float, min_cadence_days: float,
                       max_wells: int, t_max_days: float, tolerance: float = 0.02) -> list[float]:
    """Greedy heuristic: starting with one well at t=0, whenever the combined
    (uncurtailed) potential of already-scheduled wells - each following the
    same representative "type well" rate curve, offset by its own start day -
    is projected to fall more than `tolerance` below the target capacity at
    a candidate decision point, add another well at that point. Candidate
    decision points are spaced at `min_cadence_days` (a stand-in for
    rig/completion-crew cadence). This is a first-pass sizing heuristic, not
    a rigorous scheduling optimizer - it will not, for example, anticipate a
    decline before it happens or re-optimize spacing after the fact.
    """
    cap_mscfd = capacity_mmscfd * 1000.0

    def q_type(local_t: float) -> float:
        if local_t < 0:
            return 0.0
        return float(np.interp(local_t, type_well_t_days, type_well_q_mscfd,
                                 left=0.0, right=0.0))

    starts = [0.0]
    t = 0.0
    while t < t_max_days and len(starts) < max_wells:
        t += min_cadence_days
        if t >= t_max_days:
            break
        total = sum(q_type(t - s) for s in starts)
        if total < cap_mscfd * (1.0 - tolerance):
            starts.append(t)
    return starts
