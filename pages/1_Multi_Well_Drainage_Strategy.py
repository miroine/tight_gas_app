"""
Multi-Well Drainage Strategy
============================
Field-level planning page: stack any number of wells (each with its own
completion design, reservoir properties, online date, and optional
condensate yield), fit the aggregate profile to a target facility
capacity via proportional curtailment, and cut the forecast off at a
field-level abandonment rate.

See tightgas/field.py for the full methodology and assumptions, and
app.py / tightgas/condensate.py, tightgas/reporting.py, tightgas/units.py
for the condensate, period-export, and unit-system conventions shared
with the single-well page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tightgas.pvt import GasComposition, GasPVT
from tightgas.field import (
    WellSpec, FieldInputs, run_field_forecast, suggest_schedule, build_well_object,
)
from tightgas.material_balance import ogip_volumetric, MaterialBalanceTank
from tightgas.flow_models import ReservoirRock
from tightgas.simulator import SimulationInputs, run_forecast
from tightgas.condensate import CondensateModel, apply_condensate, sum_condensate_across_wells
from tightgas.reporting import resample_periods
from tightgas.units import to_display, unit_label, FIELD, METRIC
from tightgas.ui_helpers import init_unit_system, num_in, select_in, text_in, unit_num_in
from tightgas.viz import CATEGORICAL, style_fig

st.set_page_config(page_title="Multi-Well Drainage Strategy", layout="wide")


@st.cache_resource(show_spinner="Building shared field PVT tables...")
def build_pvt(gamma_g, y_co2, y_n2, y_h2s, temp_f, p_max):
    comp = GasComposition(gamma_g=gamma_g, y_co2=y_co2, y_n2=y_n2, y_h2s=y_h2s)
    return GasPVT(comp, temp_f=temp_f, p_min=25.0, p_max=p_max, n=400)


WELL_TYPE_LABELS = ["Vertical well", "Horizontal multi-frac well", "Fishbone / multilateral well"]
WELL_TYPE_CODE = {"Vertical well": "vertical", "Horizontal multi-frac well": "mfhw",
                    "Fishbone / multilateral well": "fishbone"}

GENERIC_DEFAULTS = {
    "vertical": dict(rw_ft=0.35, skin=0.0),
    "mfhw": dict(lateral_length_ft=8000.0, n_frac=25, xf_ft=200.0, fcd=20.0, linear_calib=1.0),
    "fishbone": dict(main_bore_length_ft=6000.0, n_branches=8, branch_length_ft=600.0, linear_calib=0.5),
}

st.title("Multi-Well Drainage Strategy")
st.caption("Stack multiple wells - each with its own design, online date, and optional condensate yield - "
           "into a field-level production plan fitted to a target facility capacity and abandonment rate.")

# ==========================================================================
# UNITS
# ==========================================================================
system, sys_changed = init_unit_system()

# ==========================================================================
# SHARED FIELD-LEVEL PVT & ROCK PROPERTIES
# ==========================================================================
st.sidebar.title("Field-wide fluid & rock properties")
st.sidebar.caption("Shared across every well (same produced-fluid system / formation). "
                     "Permeability, net pay, initial pressure, drainage area, completion "
                     "design, and CGR are set per well below.")

with st.sidebar.expander("Gas composition & PVT", expanded=True):
    gamma_g = st.number_input("Gas specific gravity (air = 1)", 0.55, 1.20, 0.65, 0.01)
    y_co2 = st.number_input("CO2 mole fraction", 0.0, 0.30, 0.02, 0.01)
    y_n2 = st.number_input("N2 mole fraction", 0.0, 0.30, 0.01, 0.01)
    y_h2s = st.number_input("H2S mole fraction", 0.0, 0.30, 0.00, 0.01)
    temp_f = unit_num_in(st.sidebar, "Reservoir temperature", "shared_temp", 210.0, "temperature",
                           system, sys_changed, min_field=100.0, max_field=400.0, step_field=5.0)

with st.sidebar.expander("Shared rock behavior", expanded=True):
    phi = st.number_input("Porosity, fraction", 0.02, 0.30, 0.08, 0.005, format="%.3f")
    sw = st.number_input("Water saturation, fraction", 0.05, 0.80, 0.35, 0.01)
    cf_mb = unit_num_in(st.sidebar, "Apparent compaction / water-drive correction cf", "shared_cf",
                          2e-5, "inv_pressure", system, sys_changed, min_field=0.0, max_field=5e-4,
                          step_field=5e-6, fmt="%.3e",
                          help="(p/z)*(1+cf*(pi-p)) = (pi/zi)*(1-Gp/G). 0 = classical volumetric tank.")
    stress_gamma = unit_num_in(st.sidebar, "Permeability modulus (stress sensitivity)", "shared_gamma",
                                 0.0, "inv_pressure", system, sys_changed, min_field=0.0, max_field=2e-3,
                                 step_field=1e-5, fmt="%.3e",
                                 help="k(p) = ki * exp(-gamma*(pi-p)). 0 = pressure-independent permeability.")

st.sidebar.markdown("---")
st.sidebar.title("Field targets")
capacity_on = st.sidebar.checkbox("Cap field production at a facility capacity", value=True)
capacity_mmscfd = unit_num_in(st.sidebar, "Target facility capacity", "cap", 25.0, "rate_gas_mmscfd",
                                system, sys_changed, min_field=0.1, max_field=1000.0, step_field=1.0)
field_abandon_mmscfd = unit_num_in(st.sidebar, "Field abandonment rate", "fieldab", 1.0, "rate_gas_mmscfd",
                                     system, sys_changed, min_field=0.0, max_field=500.0, step_field=0.1,
                                     help="Forecast stops once total field production falls "
                                          "to/below this rate (or once every well is individually "
                                          "depleted, whichever comes first).")
t_max_years = st.sidebar.number_input("Forecast horizon, years", 1, 60, 30, 1)

st.sidebar.markdown("---")
st.sidebar.title("Type-well template")
st.sidebar.caption("Defaults used to pre-fill new wells and to drive the schedule-suggestion tool below. "
                     "Every well can still be edited individually.")
tmpl_type_label = st.sidebar.selectbox("Template well type", WELL_TYPE_LABELS, key="tmpl_type")
tmpl_area = unit_num_in(st.sidebar, "Drainage area", "tmpl_area", 640.0, "area", system, sys_changed,
                          min_field=20.0, max_field=5000.0, step_field=10.0)
tmpl_k = st.sidebar.number_input("Matrix permeability, md", 1e-5, 50.0, 0.05, 0.001, format="%.5f", key="tmpl_k",
                                   help="md is a universal unit (unaffected by Field/Metric).")
tmpl_h = unit_num_in(st.sidebar, "Net pay thickness", "tmpl_h", 80.0, "length", system, sys_changed,
                       min_field=5.0, max_field=1000.0, step_field=5.0)
tmpl_pi = unit_num_in(st.sidebar, "Initial reservoir pressure", "tmpl_pi", 8000.0, "pressure",
                        system, sys_changed, min_field=500.0, max_field=20000.0, step_field=50.0)
tmpl_pwf = unit_num_in(st.sidebar, "Flowing bottomhole pressure", "tmpl_pwf", 1000.0, "pressure",
                         system, sys_changed, min_field=100.0, max_field=10000.0, step_field=50.0)
tmpl_p_abandon = unit_num_in(st.sidebar, "Abandonment pressure", "tmpl_pab", 500.0, "pressure",
                               system, sys_changed, min_field=50.0, max_field=5000.0, step_field=50.0)
tmpl_q_min = unit_num_in(st.sidebar, "Economic rate limit", "tmpl_qmin", 50.0, "rate_gas_mscfd",
                           system, sys_changed, min_field=0.0, max_field=5000.0, step_field=10.0)
tmpl_q_max = unit_num_in(st.sidebar, "Per-well facility/choke cap (0 = uncapped)", "tmpl_qmax", 10.0,
                           "rate_gas_mmscfd", system, sys_changed, min_field=0.0, max_field=200.0, step_field=0.5)

tmpl_wt = WELL_TYPE_CODE[tmpl_type_label]
tmpl_params = dict(GENERIC_DEFAULTS[tmpl_wt])
if tmpl_wt == "vertical":
    c1, c2 = st.sidebar.columns(2)
    tmpl_params["rw_ft"] = unit_num_in(c1, "Wellbore radius", "tmpl_rw", 0.35, "length", system, sys_changed,
                                         min_field=0.1, max_field=1.0, step_field=0.01)
    tmpl_params["skin"] = c2.number_input("Skin", -5.0, 50.0, 0.0, 0.5, key="tmpl_skin")
elif tmpl_wt == "mfhw":
    c1, c2 = st.sidebar.columns(2)
    tmpl_params["lateral_length_ft"] = unit_num_in(c1, "Lateral length", "tmpl_lat", 8000.0, "length",
                                                      system, sys_changed, min_field=500.0, max_field=20000.0,
                                                      step_field=100.0)
    tmpl_params["n_frac"] = c2.number_input("Number of frac stages", 1, 200, 25, 1, key="tmpl_nfrac")
    c3, c4 = st.sidebar.columns(2)
    tmpl_params["xf_ft"] = unit_num_in(c3, "Fracture half-length", "tmpl_xf", 200.0, "length", system, sys_changed,
                                         min_field=20.0, max_field=1000.0, step_field=10.0)
    tmpl_params["fcd"] = c4.number_input("Frac conductivity FcD", 0.1, 500.0, 20.0, 0.5, key="tmpl_fcd")
    tmpl_params["linear_calib"] = st.sidebar.number_input("Linear-flow calibration", 0.1, 3.0, 1.0, 0.05, key="tmpl_lc")
else:
    c1, c2 = st.sidebar.columns(2)
    tmpl_params["main_bore_length_ft"] = unit_num_in(c1, "Main bore length", "tmpl_mb", 6000.0, "length",
                                                        system, sys_changed, min_field=500.0, max_field=20000.0,
                                                        step_field=100.0)
    tmpl_params["n_branches"] = c2.number_input("Number of branches", 1, 40, 8, 1, key="tmpl_nb")
    c3, c4 = st.sidebar.columns(2)
    tmpl_params["branch_length_ft"] = unit_num_in(c3, "Branch length", "tmpl_bl", 600.0, "length",
                                                     system, sys_changed, min_field=50.0, max_field=3000.0,
                                                     step_field=50.0)
    tmpl_params["linear_calib"] = c4.number_input("Linear-flow calibration", 0.05, 3.0, 0.5, 0.05, key="tmpl_lcf")

with st.sidebar.expander("Condensate / CGR template", expanded=False):
    tmpl_cgr_on = st.checkbox("New wells track condensate by default", value=False, key="tmpl_cgr_on")
    tmpl_cgr_i = unit_num_in(st.sidebar, "CGR at initial pressure", "tmpl_cgri", 30.0, "cgr", system, sys_changed,
                               min_field=0.0, max_field=500.0, step_field=1.0)
    tmpl_cgr_declines = st.checkbox("CGR declines with reservoir pressure", value=False, key="tmpl_cgr_decl")
    tmpl_cgr_abandon = unit_num_in(st.sidebar, "CGR at abandonment pressure", "tmpl_cgrab", 10.0, "cgr",
                                     system, sys_changed, min_field=0.0, max_field=500.0, step_field=1.0)

# ==========================================================================
# SCHEDULE SUGGESTION TOOL (first-pass heuristic, not an optimizer)
# ==========================================================================
st.markdown("### 1. (Optional) Suggest a well count & timing to hold the target plateau")
st.caption("A simple greedy heuristic - not an optimizer: it walks forward in time and adds another "
           "type-well whenever the combined potential of already-scheduled wells is projected to fall "
           "below the target capacity. Use it to get a reasonable starting schedule, then fine-tune each "
           "well below.")

sug1, sug2, sug3, sug4 = st.columns(4)
sug_cadence = sug1.number_input("Minimum spud cadence, days", 30, 720, 120, 10,
                                  help="Stand-in for rig/completion-crew availability.")
sug_max_wells = sug2.number_input("Max wells to suggest", 1, 60, 12, 1)
sug_horizon = sug3.number_input("Suggestion horizon, years", 1, 40, 10, 1)
suggest_clicked = sug4.button("Suggest schedule", use_container_width=True)

if suggest_clicked:
    pvt_tmpl = build_pvt(gamma_g, y_co2, y_n2, y_h2s, temp_f, p_max=max(tmpl_pi * 1.1, 2000))
    rock_tmpl = ReservoirRock(k_md=tmpl_k, phi=phi, h_ft=tmpl_h, stress_gamma=stress_gamma)
    well_tmpl = build_well_object(WellSpec(name="type well", well_type=tmpl_wt, start_day=0.0,
                                             area_acres=tmpl_area, k_md=tmpl_k, h_ft=tmpl_h, pi=tmpl_pi,
                                             pwf=tmpl_pwf, p_abandon=tmpl_p_abandon, q_min_mscfd=tmpl_q_min,
                                             q_max_mmscfd=(tmpl_q_max or None), params=tmpl_params))
    bgi_tmpl = float(pvt_tmpl.bg(tmpl_pi))
    G_tmpl = ogip_volumetric(tmpl_area, tmpl_h, phi, sw, bgi_tmpl)
    mb_tmpl = MaterialBalanceTank(pvt=pvt_tmpl, G_scf=G_tmpl, pi=tmpl_pi, cf=cf_mb)
    sim_tmpl = SimulationInputs(pvt=pvt_tmpl, rock=rock_tmpl, well=well_tmpl, mb=mb_tmpl, pwf=tmpl_pwf,
                                  t_max_days=sug_horizon * 365.25, p_abandon=tmpl_p_abandon,
                                  q_min_mscfd=tmpl_q_min, n_steps=300,
                                  q_max_mscfd=(tmpl_q_max * 1000.0 if tmpl_q_max else None))
    df_tmpl = run_forecast(sim_tmpl)

    target_cap = capacity_mmscfd if capacity_on else tmpl_q_max * 3
    starts = suggest_schedule(df_tmpl["t_days"].values, df_tmpl["q_mscfd"].values,
                                capacity_mmscfd=target_cap, min_cadence_days=sug_cadence,
                                max_wells=sug_max_wells, t_max_days=sug_horizon * 365.25)
    st.session_state["suggested_starts"] = starts
    st.session_state["n_wells_input"] = len(starts)
    for i, s in enumerate(starts):
        # Plain (non-unit-aware) widgets are keyed directly; unit-aware
        # per-well widgets (area/h/pi) store their canonical field-unit
        # value under f"{key}__field" (see ui_helpers.unit_num_in), so the
        # programmatic seed must target that suffixed key instead of the
        # bare one, or it would silently have no effect on the widget.
        st.session_state[f"w{i}_start"] = float(round(s))
        st.session_state[f"w{i}_type"] = tmpl_type_label
        st.session_state[f"w{i}_area__field"] = float(tmpl_area)
        st.session_state[f"w{i}_k"] = float(tmpl_k)
        st.session_state[f"w{i}_h__field"] = float(tmpl_h)
        st.session_state[f"w{i}_pi__field"] = float(tmpl_pi)

if "suggested_starts" in st.session_state:
    starts = st.session_state["suggested_starts"]
    st.success(f"Suggested {len(starts)} wells at start days: "
               f"{', '.join(str(int(s)) for s in starts)} (applied to the well list below - "
               f"adjust freely).")

st.markdown("### 2. Well list")
if "n_wells_input" not in st.session_state:
    st.session_state["n_wells_input"] = 3
n_wells = st.number_input("Number of wells", 1, 60, key="n_wells_input")

wells: list[WellSpec] = []
well_cond_models: dict[str, CondensateModel] = {}
for i in range(int(n_wells)):
    with st.expander(f"Well {i + 1}" + (f" - {st.session_state.get(f'w{i}_name', '')}" if st.session_state.get(f'w{i}_name') else ""),
                       expanded=(i == 0)):
        c1, c2, c3 = st.columns(3)
        name = text_in(c1, "Name", f"w{i}_name", f"Well {i+1}")
        wt_label = select_in(c2, "Well type", WELL_TYPE_LABELS, f"w{i}_type", tmpl_type_label)
        start_day = num_in(c3, "Start day (days from field start)", f"w{i}_start", float(i * 180),
                             min_value=0.0, max_value=20000.0, step=10.0)

        c4, c5, c6, c7 = st.columns(4)
        area = unit_num_in(c4, "Drainage area", f"w{i}_area", float(tmpl_area), "area", system, sys_changed,
                             min_field=20.0, max_field=5000.0, step_field=10.0)
        k_md = num_in(c5, "Permeability, md", f"w{i}_k", float(tmpl_k),
                        min_value=1e-5, max_value=50.0, step=0.001, format="%.5f")
        h_ft = unit_num_in(c6, "Net pay", f"w{i}_h", float(tmpl_h), "length", system, sys_changed,
                             min_field=5.0, max_field=1000.0, step_field=5.0)
        pi_w = unit_num_in(c7, "Initial pressure", f"w{i}_pi", float(tmpl_pi), "pressure", system, sys_changed,
                             min_field=500.0, max_field=20000.0, step_field=50.0)

        wt = WELL_TYPE_CODE[wt_label]
        params = {}
        if wt == "vertical":
            c8, c9 = st.columns(2)
            params["rw_ft"] = unit_num_in(c8, "Wellbore radius", f"w{i}_rw",
                                            float(tmpl_params.get("rw_ft", 0.35)), "length", system, sys_changed,
                                            min_field=0.1, max_field=1.0, step_field=0.01)
            params["skin"] = num_in(c9, "Skin", f"w{i}_skin", float(tmpl_params.get("skin", 0.0)),
                                       min_value=-5.0, max_value=50.0, step=0.5)
        elif wt == "mfhw":
            c8, c9 = st.columns(2)
            params["lateral_length_ft"] = unit_num_in(c8, "Lateral length", f"w{i}_lat",
                                                         float(tmpl_params.get("lateral_length_ft", 8000)),
                                                         "length", system, sys_changed,
                                                         min_field=500.0, max_field=20000.0, step_field=100.0)
            params["n_frac"] = num_in(c9, "Number of frac stages", f"w{i}_nfrac",
                                         int(tmpl_params.get("n_frac", 25)),
                                         min_value=1, max_value=200, step=1)
            c10, c11, c12 = st.columns(3)
            params["xf_ft"] = unit_num_in(c10, "Fracture half-length", f"w{i}_xf",
                                            float(tmpl_params.get("xf_ft", 200)), "length", system, sys_changed,
                                            min_field=20.0, max_field=1000.0, step_field=10.0)
            params["fcd"] = num_in(c11, "Frac conductivity FcD", f"w{i}_fcd",
                                      float(tmpl_params.get("fcd", 20.0)),
                                      min_value=0.1, max_value=500.0, step=0.5)
            params["linear_calib"] = num_in(c12, "Linear-flow calibration", f"w{i}_lc",
                                               float(tmpl_params.get("linear_calib", 1.0)),
                                               min_value=0.1, max_value=3.0, step=0.05)
        else:
            c8, c9 = st.columns(2)
            params["main_bore_length_ft"] = unit_num_in(c8, "Main bore length", f"w{i}_mb",
                                                           float(tmpl_params.get("main_bore_length_ft", 6000)),
                                                           "length", system, sys_changed,
                                                           min_field=500.0, max_field=20000.0, step_field=100.0)
            params["n_branches"] = num_in(c9, "Number of branches", f"w{i}_nb",
                                             int(tmpl_params.get("n_branches", 8)),
                                             min_value=1, max_value=40, step=1)
            c10, c11 = st.columns(2)
            params["branch_length_ft"] = unit_num_in(c10, "Branch length", f"w{i}_bl",
                                                        float(tmpl_params.get("branch_length_ft", 600)),
                                                        "length", system, sys_changed,
                                                        min_field=50.0, max_field=3000.0, step_field=50.0)
            params["linear_calib"] = num_in(c11, "Linear-flow calibration", f"w{i}_lcf",
                                               float(tmpl_params.get("linear_calib", 0.5)),
                                               min_value=0.05, max_value=3.0, step=0.05)

        c13, c14, c15, c16 = st.columns(4)
        pwf = unit_num_in(c13, "Flowing BHP", f"w{i}_pwf", float(tmpl_pwf), "pressure", system, sys_changed,
                            min_field=100.0, max_field=10000.0, step_field=50.0)
        p_abandon = unit_num_in(c14, "Abandonment pressure", f"w{i}_pab", float(tmpl_p_abandon), "pressure",
                                  system, sys_changed, min_field=50.0, max_field=5000.0, step_field=50.0)
        q_min = unit_num_in(c15, "Economic rate limit", f"w{i}_qmin", float(tmpl_q_min), "rate_gas_mscfd",
                              system, sys_changed, min_field=0.0, max_field=5000.0, step_field=10.0)
        q_max = unit_num_in(c16, "Per-well facility cap (0=uncapped)", f"w{i}_qmax", float(tmpl_q_max),
                              "rate_gas_mmscfd", system, sys_changed, min_field=0.0, max_field=200.0, step_field=0.5)

        st.markdown("**Condensate / CGR** (optional, this well)")
        c17, c18, c19, c20 = st.columns(4)
        cgr_on = c17.checkbox("Track condensate", value=bool(tmpl_cgr_on), key=f"w{i}_cgr_on")
        cgr_i_w = unit_num_in(c18, "CGR at pi", f"w{i}_cgri", float(tmpl_cgr_i), "cgr", system, sys_changed,
                                min_field=0.0, max_field=500.0, step_field=1.0)
        cgr_declines_w = c19.checkbox("CGR declines w/ pressure", value=bool(tmpl_cgr_declines),
                                        key=f"w{i}_cgr_decl")
        cgr_abandon_w = unit_num_in(c20, "CGR at abandonment", f"w{i}_cgrab", float(tmpl_cgr_abandon), "cgr",
                                      system, sys_changed, min_field=0.0, max_field=500.0, step_field=1.0)

        wells.append(WellSpec(name=name, well_type=wt, start_day=float(start_day), area_acres=area,
                                k_md=k_md, h_ft=h_ft, pi=pi_w, pwf=pwf, p_abandon=p_abandon,
                                q_min_mscfd=q_min, q_max_mmscfd=(q_max or None), params=params))
        if cgr_on:
            well_cond_models[name] = CondensateModel(
                cgr_i=cgr_i_w, pi=pi_w, p_abandon=p_abandon,
                cgr_abandon=(cgr_abandon_w if cgr_declines_w else None))

run_clicked = st.button("Run field forecast", type="primary", use_container_width=True)

# ==========================================================================
# RUN & DISPLAY
# ==========================================================================
if run_clicked:
    max_pi = max(w.pi for w in wells)
    pvt = build_pvt(gamma_g, y_co2, y_n2, y_h2s, temp_f, p_max=max(max_pi * 1.1, 2000))
    field_in = FieldInputs(pvt=pvt, phi=phi, sw=sw, cf=cf_mb, stress_gamma=stress_gamma, wells=wells,
                             capacity_mmscfd=(capacity_mmscfd if capacity_on else None),
                             field_abandon_mmscfd=field_abandon_mmscfd,
                             t_max_days=t_max_years * 365.25, n_base_steps=300)
    field_df, well_dfs, well_summ, field_summ = run_field_forecast(field_in)

    # --- condensate: post-processed per well, then aggregated onto the
    # shared field time grid (every well_dfs entry already shares field_df's
    # t_days grid by construction, so no re-interpolation is needed).
    for wname, cm in well_cond_models.items():
        if wname in well_dfs:
            well_dfs[wname] = apply_condensate(well_dfs[wname], cm)
    has_condensate = len(well_cond_models) > 0
    if has_condensate:
        cond_wells = {n: d for n, d in well_dfs.items() if n in well_cond_models}
        field_cond = sum_condensate_across_wells(cond_wells, field_df["t_days"].values)
        field_df["q_cond_bbld"] = field_cond["q_cond_bbld"].values
        field_df["Np_bbl"] = field_cond["Np_bbl"].values
        field_df["Np_mstb"] = field_cond["Np_mstb"].values

    st.session_state["field_result"] = dict(field_df=field_df, well_dfs=well_dfs,
                                              well_summ=well_summ, field_summ=field_summ,
                                              has_condensate=has_condensate)

if "field_result" not in st.session_state:
    st.info("Configure your wells above, then click **Run field forecast**.")
    st.stop()

res = st.session_state["field_result"]
field_df, well_dfs, well_summ, field_summ, has_condensate = (
    res["field_df"], res["well_dfs"], res["well_summ"], res["field_summ"], res["has_condensate"])

# --- unit-aware display helpers (mirrors app.py) ---
gvol_lbl = unit_label("volume_gas_bcf", system)
grate_lbl = unit_label("rate_gas_mmscfd", system)
lvol_lbl = unit_label("volume_liquid_mstb", system)
lrate_lbl = unit_label("rate_liquid_bbld", system)


def gvol(bcf):
    return to_display(bcf, "volume_gas_bcf", system)


def grate(mmscfd):
    return to_display(mmscfd, "rate_gas_mmscfd", system)


def lvol(mstb):
    return to_display(mstb, "volume_liquid_mstb", system)


def lrate(bbld):
    return to_display(bbld, "rate_liquid_bbld", system)


st.markdown("### 3. Results")
metric_cols = st.columns(6 if has_condensate else 5)
metric_cols[0].metric("Field EUR", f"{gvol(field_summ['field_EUR_Bcf']):.2f} {gvol_lbl}")
metric_cols[1].metric("Peak field rate", f"{grate(field_summ['peak_field_rate_MMscfd']):.2f} {grate_lbl}")
metric_cols[2].metric("Field life", f"{field_summ['field_life_years']:.1f} yr")
metric_cols[3].metric("Wells", f"{field_summ['n_wells']}")
metric_cols[4].metric("Capacity target",
                        f"{grate(field_summ['capacity_MMscfd']):.1f} {grate_lbl}" if field_summ["capacity_MMscfd"] else "uncapped")
if has_condensate:
    cond_eur_mstb = float(field_df["Np_bbl"].iloc[-1]) / 1000.0
    metric_cols[5].metric("Condensate EUR", f"{lvol(cond_eur_mstb):.1f} {lvol_lbl}")

tab_names = ["Field Profile", "Per-Well Detail"]
if has_condensate:
    tab_names.append("Condensate")
tab_names += ["Yearly / Monthly", "Data & Export"]
tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))

with tab_map["Field Profile"]:
    col1, col2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=field_df["t_years"], y=grate(field_df["q_mmscfd"]), mode="lines",
                               name="Field rate", line=dict(color=CATEGORICAL[0], width=3)))
    if field_summ["capacity_MMscfd"]:
        fig.add_hline(y=grate(field_summ["capacity_MMscfd"]), line_dash="dash", line_color=CATEGORICAL[7],
                       annotation_text="Target capacity")
    fig.add_hline(y=grate(field_abandon_mmscfd), line_dash="dot", line_color="#898781",
                   annotation_text="Field abandonment rate")
    style_fig(fig, height=420)
    fig.update_layout(title="Field production rate vs time", xaxis_title="Time, years",
                        yaxis_title=f"Rate, {grate_lbl}")
    col1.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=field_df["t_years"], y=gvol(field_df["Gp_bcf"]), mode="lines",
                                name="Field cumulative", line=dict(color=CATEGORICAL[2], width=3)))
    style_fig(fig2, height=420)
    fig2.update_layout(title="Field cumulative production vs time", xaxis_title="Time, years",
                         yaxis_title=f"Cumulative, {gvol_lbl}")
    col2.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    for idx, (wname, wdf) in enumerate(well_dfs.items()):
        fig3.add_trace(go.Scatter(x=wdf["t_years"], y=grate(wdf["q_mmscfd"]), mode="lines", name=wname,
                                    line=dict(color=CATEGORICAL[idx % len(CATEGORICAL)]),
                                    stackgroup="one"))
    style_fig(fig3, height=460)
    fig3.update_layout(title="Field rate by well (stacked)", xaxis_title="Time, years",
                         yaxis_title=f"Rate, {grate_lbl}")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=field_df["t_years"], y=field_df["n_wells_online"], mode="lines",
                                line=dict(shape="hv", color=CATEGORICAL[3])))
    style_fig(fig4, height=300)
    fig4.update_layout(title="Wells online vs time", xaxis_title="Time, years",
                         yaxis_title="Number of wells producing")
    st.plotly_chart(fig4, use_container_width=True)

with tab_map["Per-Well Detail"]:
    summary_rows = []
    for wname, s in well_summ.items():
        summary_rows.append({
            "Well": wname, "Type": s["well_type"], "Start day": round(s["start_day"]),
            f"OGIP, {gvol_lbl}": round(gvol(s["OGIP_Bcf"]), 2),
            f"EUR, {gvol_lbl}": round(gvol(s["EUR_Bcf"]), 2),
            "RF, %": round(s["Recovery_factor_pct"], 1),
            f"Peak rate, {grate_lbl}": round(grate(s["peak_rate_MMscfd"]), 2),
            "Status at cutoff": s["final_status"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    fig5 = go.Figure()
    for idx, (wname, wdf) in enumerate(well_dfs.items()):
        fig5.add_trace(go.Scatter(x=wdf["t_years"], y=grate(wdf["q_mmscfd"]), mode="lines", name=wname,
                                    line=dict(color=CATEGORICAL[idx % len(CATEGORICAL)])))
    style_fig(fig5, height=460)
    fig5.update_layout(title="Individual well rates (as actually produced, i.e. after any curtailment)",
                         xaxis_title="Time, years", yaxis_title=f"Rate, {grate_lbl}")
    st.plotly_chart(fig5, use_container_width=True)

if has_condensate:
    with tab_map["Condensate"]:
        st.caption("Yield-based (CGR x gas rate) condensate estimate, summed only over wells with "
                   "condensate tracking enabled. Not a compositional/EOS retrograde-liquid model.")
        col1, col2 = st.columns(2)
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=field_df["t_years"], y=lrate(field_df["q_cond_bbld"]), mode="lines",
                                    name="Field condensate rate", line=dict(color=CATEGORICAL[4], width=3)))
        style_fig(fig6, height=400)
        fig6.update_layout(title="Field condensate rate vs time", xaxis_title="Time, years",
                             yaxis_title=f"Condensate rate, {lrate_lbl}")
        col1.plotly_chart(fig6, use_container_width=True)

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=field_df["t_years"], y=lvol(field_df["Np_bbl"] / 1000.0), mode="lines",
                                    name="Field cumulative condensate", line=dict(color=CATEGORICAL[5], width=3)))
        style_fig(fig7, height=400)
        fig7.update_layout(title="Field cumulative condensate vs time", xaxis_title="Time, years",
                             yaxis_title=f"Cumulative condensate, {lvol_lbl}")
        col2.plotly_chart(fig7, use_container_width=True)

        fig8 = go.Figure()
        for idx, (wname, wdf) in enumerate(well_dfs.items()):
            if "q_cond_bbld" not in wdf.columns:
                continue
            fig8.add_trace(go.Scatter(x=wdf["t_years"], y=lrate(wdf["q_cond_bbld"]), mode="lines", name=wname,
                                        line=dict(color=CATEGORICAL[idx % len(CATEGORICAL)]),
                                        stackgroup="one"))
        style_fig(fig8, height=420)
        fig8.update_layout(title="Field condensate rate by well (stacked)", xaxis_title="Time, years",
                             yaxis_title=f"Condensate rate, {lrate_lbl}")
        st.plotly_chart(fig8, use_container_width=True)

with tab_map["Yearly / Monthly"]:
    period_choice = st.radio("Period", ["Yearly", "Monthly"], horizontal=True, key="field_period_choice")
    period_key = "year" if period_choice == "Yearly" else "month"

    cum_series = {"gas": field_df["Gp_scf"].values}
    if has_condensate:
        cum_series["cond"] = field_df["Np_bbl"].values
    table = resample_periods(field_df["t_days"].values, cum_series, period=period_key)

    disp = pd.DataFrame({"Period": table["label"], "Days": table["days_in_period"].round(1)})
    disp[f"Gas volume, {gvol_lbl}"] = gvol(table["gas_volume"] / 1e9).round(3)
    disp[f"Gas avg rate, {grate_lbl}"] = grate(table["gas_avg_rate"] / 1e6).round(3)
    if has_condensate:
        disp[f"Condensate volume, {lvol_lbl}"] = lvol(table["cond_volume"] / 1000.0).round(2)
        disp[f"Condensate avg rate, {lrate_lbl}"] = lrate(table["cond_avg_rate"]).round(2)

    st.dataframe(disp, use_container_width=True, hide_index=True, height=min(420, 45 + 35 * len(disp)))

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=table["label"], y=gvol(table["gas_volume"] / 1e9),
                               marker_color=CATEGORICAL[0], name="Gas"))
    style_fig(fig_bar, height=380, hovermode="x unified")
    fig_bar.update_layout(title=f"Field gas volume by {period_choice.lower()[:-2] if period_choice=='Yearly' else 'month'} period",
                            xaxis_title=None, yaxis_title=f"Gas volume, {gvol_lbl}")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.download_button(f"Download field {period_choice.lower()} profile (CSV)",
                        disp.to_csv(index=False).encode("utf-8"),
                        file_name=f"field_{period_key}ly_profile.csv", mime="text/csv")

with tab_map["Data & Export"]:
    st.markdown("##### Field-level time series")
    st.dataframe(field_df, use_container_width=True, height=300)
    st.download_button("Download field profile (CSV)", field_df.to_csv(index=False).encode("utf-8"),
                        file_name="field_profile.csv", mime="text/csv")
    st.caption("Exports are always in base field units (psia, ft, Mscf/d, bbl, ...) regardless of the "
               "display unit system above.")

    well_pick = st.selectbox("Well detail to export", list(well_dfs.keys()))
    st.dataframe(well_dfs[well_pick], use_container_width=True, height=300)
    st.download_button(f"Download {well_pick} profile (CSV)",
                        well_dfs[well_pick].to_csv(index=False).encode("utf-8"),
                        file_name=f"{well_pick.replace(' ', '_')}_profile.csv", mime="text/csv")

st.markdown("---")
with st.expander("Methodology & assumptions for this page"):
    st.markdown("""
- Every well is modeled as its own **independent tank** (own OGIP, own p/z
  material balance, own deliverability) - there is no explicit
  well-to-well pressure interference. This is standard practice for
  tight-gas development planning at typical spacing, and is *not*
  appropriate for tightly-spaced infill wells with significant
  interference.
- All wells share the same produced-fluid PVT system and the same
  rock-compaction / stress-sensitivity behavior; permeability, net pay,
  drainage area, initial pressure, completion design, operating
  constraints, and condensate yield (CGR) are all independently set per
  well.
- When the combined deliverable potential of all online wells exceeds the
  target facility capacity, every online well is curtailed by the **same
  proportional factor** so the field total exactly equals the capacity.
  A well's material balance only depletes by what it actually produced, so
  curtailment extends a well's life.
- A well is judged individually "dead" (pressure <= its abandonment
  pressure, or potential rate <= its economic limit) using its own
  **uncurtailed** potential, not its curtailed/allocated share - being
  choked back for facility reasons is not the same as being depleted.
- The forecast stops at the field level once total field production falls
  to/below the field abandonment rate (after having ramped up at least
  once) or once every well is individually dead, whichever comes first.
- The schedule-suggestion tool is a simple greedy heuristic (add a well
  whenever projected potential falls below the target), not a scheduling
  optimizer - it will not re-optimize spacing in hindsight or anticipate a
  decline before it happens.
- **Condensate** is a per-well, yield-based (CGR x gas rate) add-on -
  optionally declining linearly between an initial-pressure CGR and an
  abandonment-pressure CGR - summed across only the wells that have it
  enabled to get the field condensate profile; it is not a compositional
  liquid-dropout simulation.
- **Units.** All internal physics runs in field units; the Field/Metric
  toggle only converts what widgets, plots, tables, and on-screen metrics
  show. CSV exports are always in field units. Switching units re-converts
  each field's last entered value rather than resetting it.
- **Yearly/monthly export** periods are anniversary-based (Year 1 = days
  0-365.25 from the start of the field forecast), matching the single-well
  page's convention.
    """)
