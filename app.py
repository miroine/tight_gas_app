"""
Tight Gas Production Profile Builder
=====================================
Streamlit front-end for the `tightgas` rate-transient / material-balance
engine. See tightgas/*.py module docstrings for the underlying petroleum
engineering methodology, assumptions, and references.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tightgas.pvt import GasComposition, GasPVT
from tightgas.material_balance import ogip_volumetric, MaterialBalanceTank
from tightgas.flow_models import (
    ReservoirRock, VerticalWell, MultiFracHorizontalWell, FishboneWell,
)
from tightgas.simulator import SimulationInputs, run_forecast, summarize
from tightgas.type_curves import fit_arps, arps_rate, blasingame_normalize, fetkovich_mccray_qDd
from tightgas.condensate import CondensateModel, apply_condensate
from tightgas.reporting import resample_periods
from tightgas.units import to_display, to_field, unit_label, FIELD, METRIC
from tightgas.ui_helpers import init_unit_system, unit_num_in
from tightgas.viz import CATEGORICAL, DIVERGING_POS, DIVERGING_NEG, style_fig

st.set_page_config(page_title="Tight Gas Production Profile Builder", layout="wide")


# --------------------------------------------------------------------------
# Cached PVT builder
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building real-gas PVT tables (Z, viscosity, pseudo-pressure)...")
def build_pvt(gamma_g: float, y_co2: float, y_n2: float, y_h2s: float,
               temp_f: float, p_max: float) -> GasPVT:
    comp = GasComposition(gamma_g=gamma_g, y_co2=y_co2, y_n2=y_n2, y_h2s=y_h2s)
    return GasPVT(comp, temp_f=temp_f, p_min=25.0, p_max=p_max, n=400)


def run_case(pvt, rock, well, pi, area_acres, phi, sw, cf, pwf, t_max_years,
             p_abandon, q_min_mscfd, q_max_mmscfd, n_steps, condensate_model=None):
    bgi = float(pvt.bg(pi))
    G = ogip_volumetric(area_acres, rock.h_ft, phi, sw, bgi)
    mb = MaterialBalanceTank(pvt=pvt, G_scf=G, pi=pi, cf=cf)
    sim_in = SimulationInputs(
        pvt=pvt, rock=rock, well=well, mb=mb, pwf=pwf,
        t_max_days=t_max_years * 365.25, p_abandon=p_abandon,
        q_min_mscfd=q_min_mscfd, n_steps=n_steps,
        q_max_mscfd=(q_max_mmscfd * 1000.0 if q_max_mmscfd else None),
    )
    df = run_forecast(sim_in)
    if condensate_model is not None:
        df = apply_condensate(df, condensate_model)
    summ = summarize(df, G)
    return df, summ, G, mb


def build_variant(base_well, base_rock, base_area, param_key, new_val):
    """Return (well, rock, area_acres) with exactly one parameter changed,
    for the sensitivity tornado - area is special since it drives both the
    OGIP calculation and the well's own drainage radius, so both must move
    together for a physically consistent perturbation.
    """
    if param_key == "k_md":
        return base_well, dc_replace(base_rock, k_md=new_val), base_area
    if param_key == "area_acres":
        return dc_replace(base_well, drainage_area_acres=new_val), base_rock, new_val
    return dc_replace(base_well, **{param_key: new_val}), base_rock, base_area


# ==========================================================================
# SIDEBAR - INPUTS
# ==========================================================================
system, sys_changed = init_unit_system()
st.sidebar.title("Reservoir & Well Inputs")

with st.sidebar.expander("Gas composition & PVT", expanded=True):
    gamma_g = st.number_input("Gas specific gravity (air = 1)", 0.55, 1.20, 0.65, 0.01)
    y_co2 = st.number_input("CO2 mole fraction", 0.0, 0.30, 0.02, 0.01)
    y_n2 = st.number_input("N2 mole fraction", 0.0, 0.30, 0.01, 0.01)
    y_h2s = st.number_input("H2S mole fraction", 0.0, 0.30, 0.00, 0.01)

with st.sidebar.expander("Reservoir geometry, depth, P & T", expanded=True):
    depth_ft = unit_num_in(st.sidebar, "True vertical depth", "depth", 9000.0, "length",
                             system, sys_changed, min_field=3000.0, max_field=25000.0, step_field=100.0,
                             help="Contextual only; enter initial pressure/temperature directly below.")
    temp_f = unit_num_in(st.sidebar, "Reservoir temperature", "temp", 210.0, "temperature",
                           system, sys_changed, min_field=100.0, max_field=400.0, step_field=5.0)
    _grad_lo = to_display(0.60 * depth_ft, "pressure", system)
    _grad_hi = to_display(0.65 * depth_ft, "pressure", system)
    pi = unit_num_in(st.sidebar, "Initial reservoir pressure", "pi", 8000.0, "pressure",
                       system, sys_changed, min_field=500.0, max_field=20000.0, step_field=50.0,
                       help=f"Normal gradient reference (~0.60-0.65 psi/ft) at this depth would be "
                            f"~{_grad_lo:.0f}-{_grad_hi:.0f} {unit_label('pressure', system)}; many "
                            f"tight-gas plays are overpressured.")
    area_acres = unit_num_in(st.sidebar, "Drainage area per well", "area", 640.0, "area",
                               system, sys_changed, min_field=20.0, max_field=5000.0, step_field=10.0,
                               help="Also used as the OGIP tank volume and as the boundary-dominated-flow drainage area (re).")
    h_ft = unit_num_in(st.sidebar, "Net pay thickness", "h", 80.0, "length",
                         system, sys_changed, min_field=5.0, max_field=1000.0, step_field=5.0)
    phi = st.number_input("Porosity, fraction", 0.02, 0.30, 0.08, 0.005, format="%.3f")
    sw = st.number_input("Water saturation, fraction", 0.05, 0.80, 0.35, 0.01)

with st.sidebar.expander("Matrix permeability & rock properties", expanded=True):
    k_md = st.number_input("Matrix permeability, md", 1e-5, 50.0, 0.05, 0.001, format="%.5f",
                             help="Tight gas is typically 0.0001-0.1 md. md is a universal unit (unaffected by Field/Metric).")
    stress_gamma = unit_num_in(st.sidebar, "Permeability modulus (stress sensitivity)", "gamma",
                                 0.0, "inv_pressure", system, sys_changed, min_field=0.0, max_field=2e-3,
                                 step_field=1e-5, fmt="%.3e",
                                 help="k(p) = ki * exp(-gamma*(pi-p)). 0 = pressure-independent permeability.")
    cf_mb = unit_num_in(st.sidebar, "Apparent compaction / water-drive correction cf", "cf",
                          2e-5, "inv_pressure", system, sys_changed, min_field=0.0, max_field=5e-4,
                          step_field=5e-6, fmt="%.3e",
                          help="(p/z)*(1+cf*(pi-p)) = (pi/zi)*(1-Gp/G). 0 = classical volumetric tank.")

st.sidebar.markdown("---")
st.sidebar.subheader("Well configuration")
well_type = st.sidebar.selectbox(
    "Primary well type for the forecast",
    ["Vertical well", "Horizontal multi-frac well", "Fishbone / multilateral well"],
)
compare_all = st.sidebar.checkbox("Compare all three configurations on one plot", value=False)

with st.sidebar.expander("Vertical well parameters", expanded=(well_type == "Vertical well")):
    rw_ft = unit_num_in(st.sidebar, "Wellbore radius", "rw", 0.35, "length", system, sys_changed,
                          min_field=0.1, max_field=1.0, step_field=0.01)
    skin = st.number_input("Skin factor", -5.0, 50.0, 0.0, 0.5)

with st.sidebar.expander("Horizontal multi-frac well parameters",
                          expanded=(well_type == "Horizontal multi-frac well")):
    lateral_ft = unit_num_in(st.sidebar, "Lateral (horizontal) length", "lateral", 8000.0, "length",
                               system, sys_changed, min_field=500.0, max_field=20000.0, step_field=100.0)
    n_frac = st.number_input("Number of fracture stages", 1, 200, 25, 1)
    xf_ft = unit_num_in(st.sidebar, "Fracture half-length xf", "xf", 200.0, "length", system, sys_changed,
                          min_field=20.0, max_field=1000.0, step_field=10.0)
    fcd = st.number_input("Dimensionless fracture conductivity FcD", 0.1, 500.0, 20.0, 0.5)
    linear_calib_mfhw = st.number_input("Linear-flow calibration multiplier", 0.1, 3.0, 1.0, 0.05,
                                          help="Scales the analytical transient linear-flow rate; use to "
                                               "history-match real data when xf/k are uncertain.")

with st.sidebar.expander("Fishbone / multilateral well parameters",
                          expanded=(well_type == "Fishbone / multilateral well")):
    main_bore_ft = unit_num_in(st.sidebar, "Main bore (horizontal) length", "mainbore", 6000.0, "length",
                                 system, sys_changed, min_field=500.0, max_field=20000.0, step_field=100.0)
    n_branches = st.number_input("Number of branches (fishbones)", 1, 40, 8, 1)
    branch_len_ft = unit_num_in(st.sidebar, "Branch length", "branchlen", 600.0, "length",
                                  system, sys_changed, min_field=50.0, max_field=3000.0, step_field=50.0)
    linear_calib_fb = st.number_input("Linear-flow calibration multiplier ", 0.05, 3.0, 0.5, 0.05,
                                        help="Branches are typically unstimulated/open-hole -> lower default than MFHW.")

st.sidebar.markdown("---")
with st.sidebar.expander("Operating constraints & forecast horizon", expanded=True):
    pwf = unit_num_in(st.sidebar, "Flowing bottomhole pressure (constant)", "pwf", 1000.0, "pressure",
                        system, sys_changed, min_field=100.0, max_field=10000.0, step_field=50.0)
    q_max_mmscfd = unit_num_in(st.sidebar, "Facility / choke-constrained max rate", "qmax", 10.0,
                                 "rate_gas_mmscfd", system, sys_changed, min_field=0.0, max_field=200.0,
                                 step_field=0.5,
                                 help="0 = uncapped. Also regularizes the early-time transient linear-flow solution.")
    p_abandon = unit_num_in(st.sidebar, "Abandonment pressure", "pab", 500.0, "pressure", system, sys_changed,
                              min_field=50.0, max_field=5000.0, step_field=50.0)
    q_min_mscfd = unit_num_in(st.sidebar, "Economic rate limit", "qmin", 50.0, "rate_gas_mscfd",
                                system, sys_changed, min_field=0.0, max_field=5000.0, step_field=10.0)
    t_max_years = st.number_input("Forecast horizon, years", 1, 60, 30, 1)

with st.sidebar.expander("Condensate / CGR", expanded=False):
    cgr_on = st.checkbox("Track condensate production", value=False)
    cgr_i = unit_num_in(st.sidebar, "CGR at initial pressure", "cgri", 30.0, "cgr", system, sys_changed,
                          min_field=0.0, max_field=500.0, step_field=1.0, help="Condensate yield per unit gas produced.")
    cgr_declines = st.checkbox("CGR declines with reservoir pressure (retrograde condensate)", value=False,
                                 help="Simple two-point linear interpolation between the initial-pressure CGR and "
                                      "an abandonment-pressure CGR - not a compositional/EOS liquid-dropout model.")
    cgr_abandon = None
    if cgr_declines:
        cgr_abandon = unit_num_in(st.sidebar, "CGR at abandonment pressure", "cgrab", 10.0, "cgr",
                                    system, sys_changed, min_field=0.0, max_field=500.0, step_field=1.0)

with st.sidebar.expander("Type-curve / decline overlay", expanded=False):
    show_arps = st.checkbox("Overlay auto-fit Arps decline", value=True)
    fmc_b = st.slider("Fetkovich-McCray / Arps depletion-stem b (Blasingame overlay)", 0.0, 1.0, 0.5, 0.05)

run_button = st.sidebar.button("Run forecast", type="primary", use_container_width=True)


# ==========================================================================
# MAIN PANEL
# ==========================================================================
st.title("Tight Gas Production Profile Builder")
st.caption("Rate-transient analysis coupled to p/z material balance, for vertical, "
           "multi-stage-fractured horizontal, and fishbone/multilateral tight-gas wells.")
st.caption("Planning a multi-well development instead of a single well? See **Multi-Well Drainage "
           "Strategy** in the page navigation at the top of the sidebar.")

if not run_button and "last_result" not in st.session_state:
    st.info("Set your inputs in the sidebar, then click **Run forecast**.")
    st.stop()

if run_button:
    pvt = build_pvt(gamma_g, y_co2, y_n2, y_h2s, temp_f, p_max=max(pi * 1.1, 2000))
    rock = ReservoirRock(k_md=k_md, phi=phi, h_ft=h_ft, stress_gamma=stress_gamma)
    cond_model = CondensateModel(cgr_i=cgr_i, pi=pi, p_abandon=p_abandon,
                                   cgr_abandon=cgr_abandon) if cgr_on else None

    wells = {}
    if well_type == "Vertical well" or compare_all:
        wells["Vertical well"] = VerticalWell(rw_ft=rw_ft, skin=skin, drainage_area_acres=area_acres)
    if well_type == "Horizontal multi-frac well" or compare_all:
        wells["Horizontal multi-frac well"] = MultiFracHorizontalWell(
            lateral_length_ft=lateral_ft, n_frac=int(n_frac), xf_ft=xf_ft, fcd=fcd,
            drainage_area_acres=area_acres, linear_calib=linear_calib_mfhw)
    if well_type == "Fishbone / multilateral well" or compare_all:
        wells["Fishbone / multilateral well"] = FishboneWell(
            main_bore_length_ft=main_bore_ft, n_branches=int(n_branches),
            branch_length_ft=branch_len_ft, drainage_area_acres=area_acres,
            linear_calib=linear_calib_fb)

    results = {}
    for name, well in wells.items():
        df, summ, G, mb = run_case(pvt, rock, well, pi, area_acres, phi, sw, cf_mb,
                                     pwf, t_max_years, p_abandon, q_min_mscfd,
                                     q_max_mmscfd, n_steps=350, condensate_model=cond_model)
        results[name] = dict(df=df, summ=summ, G=G, mb=mb)

    st.session_state["last_result"] = dict(results=results, primary=well_type, pvt=pvt, rock=rock)
    st.session_state["run_inputs"] = dict(
        pvt=pvt, rock=rock, well=wells[well_type], area_acres=area_acres, pi=pi, phi=phi, sw=sw,
        cf_mb=cf_mb, pwf=pwf, t_max_years=t_max_years, p_abandon=p_abandon, q_min_mscfd=q_min_mscfd,
        q_max_mmscfd=q_max_mmscfd, cond_model=cond_model, well_type=well_type,
    )
    st.session_state.pop("sens_result", None)

state = st.session_state["last_result"]
results = state["results"]
primary = state["primary"] if state["primary"] in results else list(results.keys())[0]
primary_df = results[primary]["df"]
primary_summ = results[primary]["summ"]
primary_G = results[primary]["G"]
has_condensate = "q_cond_bbld" in primary_df.columns

# --- unit-aware display helpers ---
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


# --- headline metrics ---
metric_cols = st.columns(6 if has_condensate else 5)
metric_cols[0].metric("EUR", f"{gvol(primary_summ['EUR_Bcf']):.2f} {gvol_lbl}")
metric_cols[1].metric("OGIP (volumetric)", f"{gvol(primary_summ['OGIP_Bcf']):.2f} {gvol_lbl}")
metric_cols[2].metric("Recovery factor", f"{primary_summ['Recovery_factor_pct']:.1f} %")
metric_cols[3].metric("Initial rate", f"{grate(primary_summ['Initial_rate_MMscfd']):.2f} {grate_lbl}")
metric_cols[4].metric("Forecast life", f"{primary_summ['Life_years']:.1f} yr")
if has_condensate:
    cond_eur_mstb = primary_df["Np_mstb"].iloc[-1]
    metric_cols[5].metric("Condensate EUR", f"{lvol(cond_eur_mstb):.1f} {lvol_lbl}")

tab_names = ["Production Profile", "Material Balance"]
if has_condensate:
    tab_names.append("Condensate")
tab_names += ["Type-Curve Diagnostics", "Sensitivity", "Yearly / Monthly", "Data & Export", "Methodology"]
tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))

colors = {name: CATEGORICAL[i] for i, name in enumerate(results.keys())}

with tab_map["Production Profile"]:
    col1, col2 = st.columns(2)

    fig_rate = go.Figure()
    for name, r in results.items():
        df = r["df"]
        fig_rate.add_trace(go.Scatter(x=df["t_years"], y=grate(df["q_mmscfd"]), mode="lines",
                                        name=name, line=dict(color=colors.get(name), width=2.5)))
    if show_arps and not compare_all:
        t = primary_df["t_days"].values
        q = primary_df["q_mscfd"].values
        qi_a, di_a, b_a, ok = fit_arps(t, q)
        q_fit = arps_rate(t, qi_a, di_a, b_a) / 1000.0
        fig_rate.add_trace(go.Scatter(x=primary_df["t_years"], y=grate(q_fit), mode="lines",
                                        name=f"Arps fit (b={b_a:.2f}, Di={di_a*365.25:.2f}/yr)",
                                        line=dict(color="#31333F", dash="dot", width=2)))
    style_fig(fig_rate, height=420)
    fig_rate.update_layout(title="Gas rate vs time", xaxis_title="Time, years",
                             yaxis_title=f"Rate, {grate_lbl}")
    col1.plotly_chart(fig_rate, use_container_width=True)

    fig_rate_log = go.Figure()
    for name, r in results.items():
        df = r["df"].iloc[1:]
        fig_rate_log.add_trace(go.Scatter(x=df["t_days"], y=grate(df["q_mmscfd"]), mode="lines",
                                            name=name, line=dict(color=colors.get(name), width=2.5)))
    style_fig(fig_rate_log, height=420)
    fig_rate_log.update_layout(title="Gas rate vs time (log-log)", xaxis_title="Time, days",
                                 yaxis_title=f"Rate, {grate_lbl}", xaxis_type="log", yaxis_type="log")
    col2.plotly_chart(fig_rate_log, use_container_width=True)

    col3, col4 = st.columns(2)
    fig_cum = go.Figure()
    for name, r in results.items():
        df = r["df"]
        fig_cum.add_trace(go.Scatter(x=df["t_years"], y=gvol(df["Gp_bcf"]), mode="lines",
                                       name=name, line=dict(color=colors.get(name), width=2.5)))
    style_fig(fig_cum, height=380)
    fig_cum.update_layout(title="Cumulative production vs time", xaxis_title="Time, years",
                            yaxis_title=f"Cumulative gas, {gvol_lbl}")
    col3.plotly_chart(fig_cum, use_container_width=True)

    fig_press = go.Figure()
    for name, r in results.items():
        df = r["df"]
        fig_press.add_trace(go.Scatter(x=df["t_years"], y=to_display(df["p_avg_psia"], "pressure", system),
                                         mode="lines", name=name, line=dict(color=colors.get(name), width=2.5)))
    style_fig(fig_press, height=380)
    fig_press.update_layout(title="Average reservoir pressure vs time", xaxis_title="Time, years",
                              yaxis_title=f"Pressure, {unit_label('pressure', system)}")
    col4.plotly_chart(fig_press, use_container_width=True)

    st.markdown("##### Summary by configuration")
    summary_rows = []
    for name, r in results.items():
        s = r["summ"]
        summary_rows.append({
            "Well configuration": name,
            f"EUR, {gvol_lbl}": round(gvol(s["EUR_Bcf"]), 2),
            "Recovery factor, %": round(s["Recovery_factor_pct"], 1),
            f"Initial rate, {grate_lbl}": round(grate(s["Initial_rate_MMscfd"]), 2),
            "Life, yr": round(s["Life_years"], 1),
            f"Final p_avg, {unit_label('pressure', system)}": round(to_display(s["Final_reservoir_pressure_psia"], "pressure", system), 0),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

with tab_map["Material Balance"]:
    mb = results[primary]["mb"]
    df = primary_df
    fig_poz = go.Figure()
    fig_poz.add_trace(go.Scatter(x=gvol(df["Gp_bcf"]), y=to_display(df["p_over_z"], "pressure", system),
                                   mode="lines+markers", name="p/z (simulated depletion path)",
                                   marker=dict(size=4), line=dict(color=CATEGORICAL[0], width=2.5)))
    gp_line = np.linspace(0, mb.G_scf, 50)
    poz_line = mb.poz_i * (1 - gp_line / mb.G_scf)
    fig_poz.add_trace(go.Scatter(x=gvol(gp_line / 1e9), y=to_display(poz_line, "pressure", system), mode="lines",
                                   name="Straight-line volumetric trend (cf=0 reference)",
                                   line=dict(color="#898781", dash="dash", width=2)))
    style_fig(fig_poz, height=480)
    fig_poz.update_layout(title=f"p/z material balance - {primary}",
                            xaxis_title=f"Cumulative gas, {gvol_lbl}",
                            yaxis_title=f"p/z, {unit_label('pressure', system)}")
    st.plotly_chart(fig_poz, use_container_width=True)
    st.caption(f"OGIP (volumetric) = {gvol(primary_G/1e9):.2f} {gvol_lbl} | "
               f"pi = {to_display(mb.pi,'pressure',system):.0f} {unit_label('pressure',system)} | "
               f"zi = {mb.zi:.3f} | cf = {to_display(mb.cf,'inv_pressure',system):.2e} {unit_label('inv_pressure',system)}")

if has_condensate:
    with tab_map["Condensate"]:
        df = primary_df
        col1, col2 = st.columns(2)
        fig_cr = go.Figure()
        fig_cr.add_trace(go.Scatter(x=df["t_years"], y=lrate(df["q_cond_bbld"]), mode="lines",
                                      name=primary, line=dict(color=CATEGORICAL[2], width=2.5)))
        style_fig(fig_cr, height=400)
        fig_cr.update_layout(title="Condensate rate vs time", xaxis_title="Time, years",
                               yaxis_title=f"Condensate rate, {lrate_lbl}")
        col1.plotly_chart(fig_cr, use_container_width=True)

        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(x=df["t_years"], y=lvol(df["Np_mstb"]), mode="lines",
                                      name=primary, line=dict(color=CATEGORICAL[2], width=2.5)))
        style_fig(fig_cc, height=400)
        fig_cc.update_layout(title="Cumulative condensate vs time", xaxis_title="Time, years",
                               yaxis_title=f"Cumulative condensate, {lvol_lbl}")
        col2.plotly_chart(fig_cc, use_container_width=True)

        fig_cgr = go.Figure()
        fig_cgr.add_trace(go.Scatter(x=to_display(df["p_avg_psia"], "pressure", system),
                                       y=to_display(df["cgr_stb_mmscf"], "cgr", system), mode="lines",
                                       line=dict(color=CATEGORICAL[4], width=2.5)))
        style_fig(fig_cgr, height=380)
        fig_cgr.update_layout(title="CGR vs average reservoir pressure",
                                xaxis_title=f"Pressure, {unit_label('pressure', system)}",
                                yaxis_title=f"CGR, {unit_label('cgr', system)}")
        st.plotly_chart(fig_cgr, use_container_width=True)
        st.caption("Simple yield-based model: condensate rate = gas rate x CGR(p). Not a compositional / "
                   "EOS retrograde-liquid-dropout simulation - see Methodology tab.")

with tab_map["Type-Curve Diagnostics"]:
    st.markdown("##### Blasingame-normalized rate-transient diagnostic")
    st.caption("Normalized rate q/Δm(p) vs material-balance time Gp/q, with the Fetkovich-McCray / "
               "Arps depletion-stem family qDd = 1/(1+b·tDd)^(1/b) shown for reference decline exponents. "
               "This diagnostic's axes are always shown in field-unit pseudo-pressure terms "
               "(Mscf/d per psi²/cp, days) regardless of the Field/Metric toggle above - normalized "
               "rate-transient diagnostics are conventionally read in these terms even in metric practice.")

    t_mb, q_over_dm = blasingame_normalize(primary_df["t_days"].values, primary_df["q_mscfd"].values,
                                              primary_df["dm_psi2_cp"].values, primary_df["Gp_scf"].values)
    valid = np.isfinite(t_mb) & np.isfinite(q_over_dm) & (t_mb > 0) & (q_over_dm > 0)

    fig_bl = go.Figure()
    fig_bl.add_trace(go.Scatter(x=t_mb[valid], y=q_over_dm[valid], mode="markers",
                                  name=f"Simulated - {primary}", marker=dict(size=5, color=CATEGORICAL[0])))

    if valid.any():
        anchor_t = np.median(t_mb[valid])
        anchor_q = np.interp(anchor_t, t_mb[valid], q_over_dm[valid])
        tDd_ref = np.geomspace(t_mb[valid].min() / anchor_t, t_mb[valid].max() / anchor_t, 100)
        for b_ref, dash in [(0.0, "dot"), (0.5, "dash"), (1.0, "solid")]:
            qDd = fetkovich_mccray_qDd(tDd_ref, b_ref)
            fig_bl.add_trace(go.Scatter(x=tDd_ref * anchor_t, y=qDd * anchor_q, mode="lines",
                                          name=f"Depletion stem b={b_ref}", line=dict(dash=dash, color="#898781")))
        qDd_user = fetkovich_mccray_qDd(tDd_ref, fmc_b)
        fig_bl.add_trace(go.Scatter(x=tDd_ref * anchor_t, y=qDd_user * anchor_q, mode="lines",
                                      name=f"Selected b={fmc_b:.2f}", line=dict(color=CATEGORICAL[7], width=3)))

    style_fig(fig_bl, height=500, hovermode="closest")
    fig_bl.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Material balance time, days",
                           yaxis_title="q / Δm(p), Mscf/d per psi²/cp")
    st.plotly_chart(fig_bl, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Decline-rate diagnostic")
        st.caption("Instantaneous nominal decline rate Di(t) = -d(ln q)/dt - flattening indicates the "
                    "shift from transient to boundary-dominated flow; a rising trend after that is typical "
                    "harmonic-to-exponential behavior as depletion progresses.")
        q = primary_df["q_mscfd"].values
        t = primary_df["t_days"].values
        mask = q > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            di = -np.gradient(np.log(np.where(mask, q, np.nan)), t) * 365.25  # 1/yr
        fig_di = go.Figure()
        fig_di.add_trace(go.Scatter(x=primary_df["t_years"].values[1:], y=di[1:], mode="lines",
                                      line=dict(color=CATEGORICAL[3], width=2.5)))
        style_fig(fig_di, height=380)
        fig_di.update_layout(title="Nominal decline rate vs time", xaxis_title="Time, years",
                               yaxis_title="Di, 1/yr", yaxis_range=[0, np.nanpercentile(di[np.isfinite(di)], 98) * 1.2
                                                                       if np.isfinite(di).any() else 1])
        st.plotly_chart(fig_di, use_container_width=True)

    with col_b:
        st.markdown("##### Rate vs cumulative")
        st.caption("Removes time from the decline diagnostic - a common EUR-extrapolation cross-check.")
        fig_qgp = go.Figure()
        fig_qgp.add_trace(go.Scatter(x=gvol(primary_df["Gp_bcf"]), y=grate(primary_df["q_mmscfd"]), mode="lines",
                                       line=dict(color=CATEGORICAL[6], width=2.5)))
        style_fig(fig_qgp, height=380)
        fig_qgp.update_layout(title="Rate vs cumulative production", xaxis_title=f"Cumulative gas, {gvol_lbl}",
                                yaxis_title=f"Rate, {grate_lbl}")
        st.plotly_chart(fig_qgp, use_container_width=True)

    st.markdown("##### Arps decline fit")
    t = primary_df["t_days"].values
    q = primary_df["q_mscfd"].values
    qi_a, di_a, b_a, ok = fit_arps(t, q)
    cA, cB, cC = st.columns(3)
    cA.metric("qi (fit)", f"{grate(qi_a/1000):.2f} {grate_lbl}")
    cB.metric("Di (fit)", f"{di_a*365.25*100:.1f} %/yr")
    cC.metric("b (fit)", f"{b_a:.2f}")
    if not ok:
        st.warning("Nonlinear fit did not fully converge; result is from a coarse grid search fallback.")

with tab_map["Sensitivity"]:
    st.caption("Perturbs the primary well configuration's key uncertain drivers one at a time (holding "
               "everything else at its base value) and reruns the forecast to show each driver's impact "
               "on gas EUR - a classic reservoir-engineering tornado chart. Not a probabilistic/Monte "
               "Carlo uncertainty analysis.")
    pct = st.slider("Perturbation, ± %", 5, 50, 20, 5)
    run_sens = st.button("Run sensitivity analysis")

    if run_sens and "run_inputs" in st.session_state:
        ri = st.session_state["run_inputs"]
        base_well, base_rock = ri["well"], ri["rock"]
        wt_label = ri["well_type"]
        if wt_label == "Vertical well":
            completion_spec = ("Wellbore radius", "rw_ft", base_well.rw_ft)
        elif wt_label == "Horizontal multi-frac well":
            completion_spec = ("Fracture half-length", "xf_ft", base_well.xf_ft)
        else:
            completion_spec = ("Branch length", "branch_length_ft", base_well.branch_length_ft)
        specs = [("Matrix permeability", "k_md", base_rock.k_md),
                 ("Drainage area", "area_acres", ri["area_acres"]),
                 completion_spec]

        rows = []
        for label, pkey, base_val in specs:
            deltas = {}
            for tag, mult in [("low", 1.0 - pct / 100.0), ("high", 1.0 + pct / 100.0)]:
                new_val = base_val * mult
                w_v, r_v, a_v = build_variant(base_well, base_rock, ri["area_acres"], pkey, new_val)
                df_v, summ_v, _, _ = run_case(ri["pvt"], r_v, w_v, ri["pi"], a_v, ri["phi"], ri["sw"],
                                                ri["cf_mb"], ri["pwf"], ri["t_max_years"], ri["p_abandon"],
                                                ri["q_min_mscfd"], ri["q_max_mmscfd"], n_steps=200)
                deltas[tag] = summ_v["EUR_Bcf"]
            rows.append({"label": label, "low": deltas["low"], "high": deltas["high"]})

        base_eur = primary_summ["EUR_Bcf"]
        st.session_state["sens_result"] = dict(rows=rows, base_eur=base_eur, pct=pct)

    if "sens_result" in st.session_state:
        sr = st.session_state["sens_result"]
        rows, base_eur = sr["rows"], sr["base_eur"]
        rows_sorted = sorted(rows, key=lambda r: abs(r["high"] - r["low"]))
        labels = [r["label"] for r in rows_sorted]
        low_delta = [gvol(r["low"]) - gvol(base_eur) for r in rows_sorted]
        high_delta = [gvol(r["high"]) - gvol(base_eur) for r in rows_sorted]

        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(y=labels, x=low_delta, base=gvol(base_eur), orientation="h",
                                 name=f"-{sr['pct']}%", marker_color=DIVERGING_NEG))
        fig_t.add_trace(go.Bar(y=labels, x=high_delta, base=gvol(base_eur), orientation="h",
                                 name=f"+{sr['pct']}%", marker_color=DIVERGING_POS))
        fig_t.add_vline(x=gvol(base_eur), line_color="#898781", line_dash="dash")
        style_fig(fig_t, height=320)
        fig_t.update_layout(title=f"Gas EUR sensitivity (± {sr['pct']}%)", barmode="overlay",
                              xaxis_title=f"Gas EUR, {gvol_lbl}", yaxis_title=None)
        st.plotly_chart(fig_t, use_container_width=True)
        st.caption(f"Base case gas EUR = {gvol(base_eur):.2f} {gvol_lbl}.")
    elif not run_sens:
        st.info("Click **Run sensitivity analysis** to generate the tornado chart (a handful of extra "
                "forecast runs, so it isn't computed automatically).")

with tab_map["Yearly / Monthly"]:
    period_choice = st.radio("Period", ["Yearly", "Monthly"], horizontal=True)
    period_key = "year" if period_choice == "Yearly" else "month"

    cum_series = {"gas": primary_df["Gp_scf"].values}
    if has_condensate:
        cum_series["cond"] = primary_df["Np_bbl"].values
    table = resample_periods(primary_df["t_days"].values, cum_series, period=period_key)

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
    fig_bar.update_layout(title=f"Gas volume by {period_choice.lower()[:-2] if period_choice=='Yearly' else 'month'} period",
                            xaxis_title=None, yaxis_title=f"Gas volume, {gvol_lbl}")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.download_button(f"Download {period_choice.lower()} profile (CSV)", disp.to_csv(index=False).encode("utf-8"),
                        file_name=f"{primary.replace(' ', '_')}_{period_key}ly_profile.csv", mime="text/csv")

with tab_map["Data & Export"]:
    st.markdown(f"##### Full time series - {primary}")
    export_df = primary_df.copy()
    st.dataframe(export_df, use_container_width=True, height=400)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download daily production profile (CSV)", csv,
                        file_name=f"tight_gas_profile_{primary.replace(' ', '_')}.csv",
                        mime="text/csv")
    if has_condensate:
        st.caption("Includes gas (q_mscfd, Gp_scf, ...) and condensate (q_cond_bbld, Np_bbl, cgr_stb_mmscf) "
                    "columns, always in base field units regardless of the display unit system.")

with tab_map["Methodology"]:
    st.markdown("""
This tool combines a **real-gas PVT engine**, a **p/z tank material balance**,
**rate-transient deliverability models** for three completion styles, and a
simple **condensate yield model**. Full equations, constants, and literature
references are documented in the Python module docstrings shipped with this
app (`tightgas/pvt.py`, `tightgas/material_balance.py`, `tightgas/flow_models.py`,
`tightgas/type_curves.py`, `tightgas/simulator.py`, `tightgas/condensate.py`,
`tightgas/reporting.py`, `tightgas/units.py`) - see the project README for a
consolidated summary.

**Units.** All internal physics runs in U.S. field units - the PVT correlations
and flow equations are field-unit correlations by construction. The Field/Metric
toggle only converts what widgets display/accept and what plots/tables/exports
show; switching units re-converts your last entered value rather than resetting
it, but does so independently per field (there is no single hidden "master"
value you're fighting with).

**Condensate / CGR.** A yield-based model, not compositional PVT: condensate
rate = gas rate x CGR(p), with CGR either constant or linearly interpolated
between an initial-pressure value and an abandonment-pressure value. This
approximates the drop in produced liquid yield as a retrograde-condensate
reservoir depletes below its dewpoint, without a full EOS liquid-dropout
calculation.

**Yearly / monthly export.** Periods are anniversary-based (Year 1 = days
0-365.25 from the start of the forecast, not aligned to real calendar dates) -
the standard convention for comparing wells that came on-stream on different
real dates.

**Sensitivity tornado.** One-at-a-time perturbation of matrix permeability,
drainage area, and the well's key completion parameter, each re-run through
the full forecast - not a probabilistic (Monte Carlo) uncertainty analysis.

**Key assumptions / limitations**

- Single-phase dry gas (plus the yield-based condensate add-on above), tank-type
  (0-D) material balance - no reservoir simulation grid, no water banking.
- Constant flowing bottomhole pressure operation; a facility/choke rate cap is
  applied as a practical ceiling, which also regularizes the early-time
  transient linear-flow solution for fractured/multilateral wells.
- Multi-fractured horizontal and fishbone wells use a simplified
  transient-linear-flow-to-effective-wellbore-radius-PSS construction, not a
  full trilinear or numerical simulation.
- Fishbone geometry is approximated as a horizontal well with one
  low-conductivity "pseudo-fracture" per branch - treat as directional only.
- No non-Darcy (turbulent) flow term, no liquid loading / deliquification
  effects, no explicit hydraulic-fracture proppant conductivity decline over time.
    """)

st.markdown("---")
st.caption("Engineering scoping tool - verify against offset well data, nodal analysis, "
           "and/or full numerical simulation before use in investment decisions.")
