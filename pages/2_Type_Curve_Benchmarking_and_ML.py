"""
Type-Curve Benchmarking & ML
============================
Four complementary ways to sanity-check or sharpen the single-well
forecast built on the main page, in increasing order of how much they
depend on real data you supply:

1. **Decline-curve library & offset-well benchmark** - fit Arps plus three
   other published unconventional-well decline families (Duong, Power-Law
   Exponential, Stretched Exponential) to the current forecast or to a
   real offset well's production you upload, and compare goodness of fit.
2. **History-match calibration** - given a well's actual rate history,
   solve for the best-fit values of a few uncertain reservoir/completion
   parameters against the SAME physics engine used elsewhere in this app
   (not a separate curve fit) - the standard RTA way to get a better
   estimate from real data.
3. **ML surrogate (synthetic)** - train a fast regression model on many
   synthetic runs of the physics engine, for instant EUR sensitivity
   scans and Monte-Carlo-style probabilistic EUR (P10/P50/P90) that would
   be too slow to get by rerunning the full simulator thousands of times.
   This approximates this app's OWN physics, not an independent estimate.
4. **ML from your own data** - upload a table of historical/offset wells
   (design parameters + actual EUR) and train a model directly on it - a
   genuinely independent estimate, bounded entirely by how much and how
   representative your data is.

Tabs 2 and 3 need a single-well forecast to already exist (run one on the
main "Tight Gas Production Profile Builder" page first) since they
calibrate against / sample around that well's setup. Tabs 1 and 4 work
standalone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tightgas.type_curves import fit_all_models, DECLINE_MODELS
from tightgas.history_match import calibrate, CALIBRATABLE
from tightgas.surrogate import (
    full_ranges, generate_training_data, train_regressor, predict_batch, monte_carlo_eur, _HAVE_SKLEARN,
)
from tightgas.units import to_display, to_field, unit_label
from tightgas.ui_helpers import init_unit_system, num_in, select_in
from tightgas.viz import CATEGORICAL, SEQUENTIAL_BLUE, style_fig

st.set_page_config(page_title="Type-Curve Benchmarking & ML", layout="wide")

WELL_TYPE_CODE = {"Vertical well": "vertical", "Horizontal multi-frac well": "mfhw",
                    "Fishbone / multilateral well": "fishbone"}
WELL_TYPE_LABEL_PARAMS = {
    "vertical": ["rw_ft", "skin"],
    "mfhw": ["lateral_length_ft", "n_frac", "xf_ft", "fcd", "linear_calib"],
    "fishbone": ["main_bore_length_ft", "n_branches", "branch_length_ft", "linear_calib"],
}
PARAM_LABELS = {
    "k_md": "Matrix permeability, md", "area_acres": "Drainage area, acres",
    "rw_ft": "Wellbore radius, ft", "skin": "Skin factor",
    "xf_ft": "Fracture half-length, ft", "fcd": "Frac conductivity FcD",
    "linear_calib": "Linear-flow calibration multiplier",
    "branch_length_ft": "Branch length, ft",
}


def extract_well_params(wt_code: str, well_obj) -> dict:
    return {name: getattr(well_obj, name) for name in WELL_TYPE_LABEL_PARAMS[wt_code]}


def read_history_csv(uploaded, t_col_hint=None, q_col_hint=None):
    """Flexibly parse an uploaded production-history CSV: accepts a handful
    of common column-name spellings for time (days) and gas rate (Mscf/d),
    or explicit column names if the caller already knows them (used once
    the user has picked them from a selectbox)."""
    df = pd.read_csv(uploaded)
    if t_col_hint and q_col_hint:
        return df, t_col_hint, q_col_hint
    t_candidates = ["t_days", "days", "day", "t", "time_days"]
    q_candidates = ["q_mscfd", "rate_mscfd", "q", "rate", "gas_rate_mscfd", "mscfd"]
    t_col = next((c for c in df.columns if c.strip().lower() in t_candidates), df.columns[0])
    q_col = next((c for c in df.columns if c.strip().lower() in q_candidates),
                  df.columns[1] if len(df.columns) > 1 else df.columns[0])
    return df, t_col, q_col


st.title("Type-Curve Benchmarking & ML")
st.caption("Compare this tool's forecast against published decline-curve families and real offset-well "
           "data, calibrate its physics to a well's actual production, and get fast ML-based EUR "
           "estimates - synthetic (approximates this app's own engine) or trained on your own data "
           "(a genuinely independent estimate, as good as the data behind it).")

system, sys_changed = init_unit_system()

run_inputs = st.session_state.get("run_inputs")
last_result = st.session_state.get("last_result")
has_base_well = run_inputs is not None

if not has_base_well:
    st.info("Tabs 2 (History Match) and 3 (ML Surrogate) need a single-well forecast to calibrate/sample "
            "around - run one on the **Tight Gas Production Profile Builder** page first, then come back "
            "here. Tabs 1 and 4 below work without it.")

gvol_lbl = unit_label("volume_gas_bcf", system)
grate_lbl = unit_label("rate_gas_mmscfd", system)


def gvol(bcf):
    return to_display(bcf, "volume_gas_bcf", system)


def grate(mmscfd):
    return to_display(mmscfd, "rate_gas_mmscfd", system)


tab1, tab2, tab3, tab4 = st.tabs([
    "Decline-Curve Library & Offset Benchmark", "History-Match Calibration",
    "ML Surrogate (Synthetic)", "ML From Your Data",
])

# ==========================================================================
# TAB 1 - decline-curve library + offset-well benchmark
# ==========================================================================
with tab1:
    st.markdown("##### Fit every decline family to a rate-time series")
    st.caption("Arps (1945) is the industry-standard empirical family already used elsewhere in this "
               "app. The other three are published families built specifically for unconventional / "
               "tight-flow-dominated wells, included here so you can see whether one of them tracks "
               "your well's actual shape better than Arps does - not because any one of them is "
               "presumed 'right'. All four are pure curve fits; none use the reservoir-engineering "
               "model in `tightgas/flow_models.py`.")

    source = st.radio("Data source", ["My last single-well forecast", "Upload an offset well's production CSV"],
                        key="t1_source")

    t_fit, q_fit, series_label = None, None, None
    offset_df = None
    if source == "My last single-well forecast":
        if last_result is not None:
            primary = last_result["primary"]
            primary_df = last_result["results"][primary]["df"]
            t_fit = primary_df["t_days"].values
            q_fit = primary_df["q_mscfd"].values
            series_label = f"Simulated - {primary}"
        else:
            st.warning("No forecast yet - run one on the main page, or switch to uploading an offset well.")
    else:
        uploaded = st.file_uploader("Offset well CSV (needs a days column and a Mscf/d rate column)",
                                      type=["csv"], key="t1_upload")
        if uploaded is not None:
            offset_df, t_col, q_col = read_history_csv(uploaded)
            st.caption(f"Using `{t_col}` as time (days) and `{q_col}` as rate (Mscf/d). "
                       f"Rename columns in your CSV if this guess is wrong.")
            t_fit = offset_df[t_col].values.astype(float)
            q_fit = offset_df[q_col].values.astype(float)
            series_label = "Offset well (uploaded)"

    if t_fit is not None and len(t_fit) >= 5:
        results = fit_all_models(t_fit, q_fit)
        rows = []
        for name, r in results.items():
            rows.append({"Model": name, "Fit converged": r["success"],
                          "R²": round(r["r2"], 4) if np.isfinite(r["r2"]) else None,
                          "log-RMSE (lower=better)": round(r["log_rmse"], 4) if np.isfinite(r["log_rmse"]) else None,
                          "Reference": r["citation"]})
        table = pd.DataFrame(rows).sort_values("log-RMSE (lower=better)", na_position="last")
        st.dataframe(table, use_container_width=True, hide_index=True)
        best_name = table.iloc[0]["Model"] if len(table) else None
        if best_name:
            st.caption(f"Best log-RMSE fit: **{best_name}**. A close race between models usually just "
                       "means the data doesn't yet run long enough to tell them apart - all four agree "
                       "closely near the fitted region and diverge most at long-term extrapolation.")

        t_plot = np.geomspace(max(float(np.min(t_fit[t_fit > 0])), 0.5), float(np.max(t_fit)) * 1.5, 200)
        fig = go.Figure()
        mask = t_fit > 0
        fig.add_trace(go.Scatter(x=t_fit[mask], y=q_fit[mask], mode="markers", name=series_label,
                                    marker=dict(size=5, color=CATEGORICAL[0])))
        for i, (name, r) in enumerate(results.items()):
            try:
                q_line = r["rate_fn"](t_plot)
            except Exception:
                continue
            fig.add_trace(go.Scatter(x=t_plot, y=q_line, mode="lines", name=name,
                                        line=dict(color=CATEGORICAL[(i + 1) % len(CATEGORICAL)], width=2)))
        style_fig(fig, height=480, hovermode="closest")
        fig.update_layout(title="Decline-curve family comparison", xaxis_title="Time, days",
                            yaxis_title="Rate, Mscf/d", xaxis_type="log", yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)

        if last_result is not None and offset_df is not None:
            st.markdown("##### Offset-well benchmark vs. this app's simulated forecast")
            primary = last_result["primary"]
            sim_df = last_result["results"][primary]["df"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=sim_df["t_days"], y=sim_df["q_mscfd"], mode="lines",
                                         name=f"Simulated - {primary}", line=dict(color=CATEGORICAL[0], width=2.5)))
            fig2.add_trace(go.Scatter(x=t_fit[mask], y=q_fit[mask], mode="markers", name="Offset well (uploaded)",
                                         marker=dict(size=5, color=CATEGORICAL[1])))
            style_fig(fig2, height=440, hovermode="closest")
            fig2.update_layout(title="Simulated forecast vs. offset well", xaxis_title="Time, days",
                                 yaxis_title="Rate, Mscf/d", xaxis_type="log", yaxis_type="log")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("A visual/analog check, not a calibration - if the shapes disagree systematically, "
                       "that's a signal to revisit inputs on the main page or use the History-Match tab.")
    elif t_fit is not None:
        st.warning("Need at least 5 data points to fit the decline-curve library.")

# ==========================================================================
# TAB 2 - history-match calibration
# ==========================================================================
with tab2:
    if not has_base_well:
        st.info("Run a forecast on the main page first - this tab calibrates around its well type and "
                "base parameters.")
    else:
        wt_label = run_inputs["well_type"]
        wt_code = WELL_TYPE_CODE[wt_label]
        base_well = run_inputs["well"]
        base_well_params = extract_well_params(wt_code, base_well)
        base_k_md = run_inputs["rock"].k_md

        st.markdown(f"##### Calibrate the **{wt_label}** model to real production")
        st.caption("Upload the well's actual history (a days column and a Mscf/d rate column). Nonlinear "
                   "least squares (in log-rate space) solves for the parameters you pick below by "
                   "re-running the same physics engine used on the main page - not a separate curve fit. "
                   "Pick 1-2 parameters when possible: history-matching more than that from rate data "
                   "alone is often non-unique (several k/xf combinations can produce a very similar rate "
                   "curve) - EUR still comes out reasonably robust even when the individual parameters "
                   "trade off against each other, but don't over-interpret any single fitted value from "
                   "a 3-4-parameter simultaneous fit.")

        hm_upload = st.file_uploader("Actual production history CSV", type=["csv"], key="t2_upload")
        param_choices = list(CALIBRATABLE[wt_code].keys())
        chosen = st.multiselect("Parameters to calibrate", param_choices,
                                  default=param_choices[:2], key="t2_params",
                                  format_func=lambda p: PARAM_LABELS.get(p, p))
        run_hm = st.button("Run calibration", key="t2_run")

        if run_hm and hm_upload is not None and chosen:
            hist_df, t_col, q_col = read_history_csv(hm_upload)
            st.caption(f"Using `{t_col}` (days) and `{q_col}` (Mscf/d) from the upload.")
            t_obs = hist_df[t_col].values.astype(float)
            q_obs = hist_df[q_col].values.astype(float)

            fixed = dict(area_acres=run_inputs["area_acres"], k_md=base_k_md,
                          h_ft=run_inputs["rock"].h_ft, phi=run_inputs["phi"], sw=run_inputs["sw"],
                          pi=run_inputs["pi"], cf=run_inputs["cf_mb"], pwf=run_inputs["pwf"],
                          p_abandon=run_inputs["p_abandon"], q_min_mscfd=run_inputs["q_min_mscfd"],
                          q_max_mscfd=(run_inputs["q_max_mmscfd"] * 1000.0 if run_inputs["q_max_mmscfd"] else None),
                          well_params=base_well_params)
            initial_guess = {"k_md": base_k_md, "area_acres": run_inputs["area_acres"], **base_well_params}
            try:
                with st.spinner("Calibrating (a few dozen forecast re-runs)..."):
                    res = calibrate(run_inputs["pvt"], wt_code, fixed, chosen, initial_guess,
                                      t_obs, q_obs, t_forecast_days=run_inputs["t_max_years"] * 365.25)
                st.session_state["hm_result"] = dict(res=res, wt_label=wt_label, fixed=fixed, chosen=chosen)
            except Exception as e:
                st.error(f"Calibration failed: {e}")

        if "hm_result" in st.session_state:
            hm = st.session_state["hm_result"]
            res = hm["res"]
            st.success(f"Converged: {res.success} ({res.message}) - {res.n_eval} forecast evaluations, "
                       f"R²={res.r2:.3f}, log-RMSE={res.log_rmse:.3f}")

            fit_rows = []
            for p in hm["chosen"]:
                base_v = hm["fixed"]["k_md"] if p == "k_md" else \
                    (hm["fixed"]["area_acres"] if p == "area_acres" else hm["fixed"]["well_params"].get(p))
                fit_rows.append({"Parameter": PARAM_LABELS.get(p, p), "Base value": round(base_v, 4),
                                   "Calibrated value": round(res.fitted[p], 4)})
            st.dataframe(pd.DataFrame(fit_rows), use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Calibrated gas EUR", f"{gvol(res.summary['EUR_Bcf']):.2f} {gvol_lbl}")
            c2.metric("Recovery factor", f"{res.summary['Recovery_factor_pct']:.1f} %")
            c3.metric("Forecast life", f"{res.summary['Life_years']:.1f} yr")

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=res.t_obs, y=grate(res.q_obs / 1000.0), mode="markers",
                                         name="Observed", marker=dict(size=6, color=CATEGORICAL[0])))
            fig3.add_trace(go.Scatter(x=res.df["t_days"], y=grate(res.df["q_mmscfd"]), mode="lines",
                                         name="Calibrated forecast", line=dict(color=CATEGORICAL[1], width=2.5)))
            style_fig(fig3, height=460, hovermode="closest")
            fig3.update_layout(title="Calibrated forecast vs. observed history", xaxis_title="Time, days",
                                 yaxis_title=f"Rate, {grate_lbl}", xaxis_type="log", yaxis_type="log")
            st.plotly_chart(fig3, use_container_width=True)

            st.download_button("Download calibrated forecast (CSV)",
                                res.df.to_csv(index=False).encode("utf-8"),
                                file_name="calibrated_forecast.csv", mime="text/csv")
        elif run_hm and hm_upload is None:
            st.warning("Upload a production history CSV first.")

# ==========================================================================
# TAB 3 - synthetic ML surrogate + Monte Carlo
# ==========================================================================
with tab3:
    if not _HAVE_SKLEARN:
        st.error("scikit-learn isn't installed in this environment - run `pip install scikit-learn` "
                 "(see requirements.txt) to use this tab.")
    elif not has_base_well:
        st.info("Run a forecast on the main page first - the surrogate is trained by sampling around "
                "its well type.")
    else:
        wt_label = run_inputs["well_type"]
        wt_code = WELL_TYPE_CODE[wt_label]
        base_well = run_inputs["well"]
        base_well_params = extract_well_params(wt_code, base_well)

        st.markdown(f"##### Train a fast EUR proxy model for the **{wt_label}** configuration")
        st.caption("Samples the parameter ranges below (Latin hypercube), runs the full physics engine "
                   "for each sample, and trains a random-forest regressor on {parameters -> gas EUR}. "
                   "This approximates this app's OWN engine - it is a speed trade, not an independent "
                   "estimate - but once trained it can score thousands of what-if combinations "
                   "(a Monte Carlo scan, below) in well under a second.")

        ranges = full_ranges(wt_code)
        st.markdown("**Sampling ranges** (edit if the defaults don't match your play)")
        edited_ranges = {}
        rcols = st.columns(3)
        for i, (name, (lo, hi, scale)) in enumerate(ranges.items()):
            col = rcols[i % 3]
            lo_e = num_in(col, f"{name} min", f"t3_{name}_lo", float(lo))
            hi_e = num_in(col, f"{name} max", f"t3_{name}_hi", float(hi))
            edited_ranges[name] = (lo_e, hi_e, scale)

        n_samples = st.slider("Number of synthetic training runs", 100, 1000, 350, 50, key="t3_nsamp")
        gen_clicked = st.button("Generate & train", key="t3_gen")

        if gen_clicked:
            fixed = dict(area_acres=run_inputs["area_acres"], k_md=run_inputs["rock"].k_md,
                          h_ft=run_inputs["rock"].h_ft, phi=run_inputs["phi"], sw=run_inputs["sw"],
                          pi=run_inputs["pi"], cf=run_inputs["cf_mb"], pwf=run_inputs["pwf"],
                          p_abandon=run_inputs["p_abandon"], q_min_mscfd=run_inputs["q_min_mscfd"],
                          q_max_mscfd=(run_inputs["q_max_mmscfd"] * 1000.0 if run_inputs["q_max_mmscfd"] else None),
                          well_params=base_well_params)
            with st.spinner(f"Running {n_samples} synthetic forecasts and training the surrogate..."):
                df_train = generate_training_data(run_inputs["pvt"], wt_code, edited_ranges, n_samples,
                                                     fixed, t_max_days=run_inputs["t_max_years"] * 365.25)
                feature_cols = list(edited_ranges.keys())
                reg = train_regressor(df_train, feature_cols, "EUR_Bcf")
            st.session_state["surrogate"] = dict(reg=reg, wt_code=wt_code, fixed=fixed,
                                                    feature_cols=feature_cols, df_train=df_train)

        if "surrogate" in st.session_state and st.session_state["surrogate"]["wt_code"] == wt_code:
            sur = st.session_state["surrogate"]
            reg = sur["reg"]
            st.success(f"Trained on {reg.n_train} synthetic runs, {reg.cv_folds}-fold cross-validated "
                       f"R²={reg.cv_r2:.3f}, MAE={reg.cv_mae:.2f} Bcf.")

            col1, col2 = st.columns(2)
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(x=reg.feature_importance["importance"], y=reg.feature_importance["feature"],
                                        orientation="h", marker_color=SEQUENTIAL_BLUE[3]))
            style_fig(fig_imp, height=380)
            fig_imp.update_layout(title="Feature importance (drivers of EUR)", xaxis_title="Importance",
                                     yaxis_title=None)
            col1.plotly_chart(fig_imp, use_container_width=True)

            fig_oof = go.Figure()
            fig_oof.add_trace(go.Scatter(x=gvol(reg.oof_actual), y=gvol(reg.oof_predictions), mode="markers",
                                            marker=dict(size=5, color=CATEGORICAL[0]), name="Out-of-fold"))
            lims = [float(np.min(gvol(reg.oof_actual))), float(np.max(gvol(reg.oof_actual)))]
            fig_oof.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect fit",
                                            line=dict(color="#898781", dash="dash")))
            style_fig(fig_oof, height=380, hovermode="closest")
            fig_oof.update_layout(title="Cross-validated predicted vs. actual EUR",
                                     xaxis_title=f"Actual EUR, {gvol_lbl}", yaxis_title=f"Predicted EUR, {gvol_lbl}")
            col2.plotly_chart(fig_oof, use_container_width=True)

            st.markdown("##### Monte Carlo probabilistic EUR")
            st.caption("Define an uncertainty distribution for one or more parameters; everything else "
                       "stays fixed at this well's base value. Draws are scored by the trained surrogate "
                       "(near-instant), not by rerunning the full simulator.")
            mc_vary = st.multiselect("Parameters to vary", sur["feature_cols"], key="t3_mc_vary",
                                        default=sur["feature_cols"][:1])
            distributions = {}
            for name in mc_vary:
                lo, hi, _ = edited_ranges[name]
                mode_default = float(sur["fixed"].get(name, sur["fixed"]["well_params"].get(name, (lo + hi) / 2)))
                mc_lo = num_in(st, f"{name}: low (P10-ish)", f"t3_mc_{name}_lo", lo)
                mc_mode = num_in(st, f"{name}: most likely", f"t3_mc_{name}_mode",
                                    float(np.clip(mode_default, mc_lo, hi)))
                mc_hi = num_in(st, f"{name}: high (P90-ish)", f"t3_mc_{name}_hi", hi)
                distributions[name] = ("triangular", mc_lo, mc_mode, mc_hi)

            n_draws = st.slider("Monte Carlo draws", 500, 20000, 5000, 500, key="t3_ndraws")
            run_mc = st.button("Run Monte Carlo", key="t3_mc_run")

            if run_mc:
                base_values = {c: sur["fixed"].get(c, sur["fixed"]["well_params"].get(c, 0.0))
                                for c in sur["feature_cols"]}
                draws = monte_carlo_eur(reg, distributions, base_values, n_draws=n_draws)
                st.session_state["mc_draws"] = draws

            if "mc_draws" in st.session_state:
                draws = st.session_state["mc_draws"]
                p10, p50, p90 = np.percentile(draws, [10, 50, 90])
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("P90 (conservative)", f"{gvol(p10):.2f} {gvol_lbl}")
                mcol2.metric("P50 (median)", f"{gvol(p50):.2f} {gvol_lbl}")
                mcol3.metric("P10 (upside)", f"{gvol(p90):.2f} {gvol_lbl}")
                st.caption("Reserves convention: P90 = 90% probability of exceeding this value "
                           "(conservative/low case), P10 = 10% probability of exceeding (high case).")
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(x=gvol(draws), nbinsx=50, marker_color=CATEGORICAL[0]))
                for val, lbl in [(p10, "P90"), (p50, "P50"), (p90, "P10")]:
                    fig_mc.add_vline(x=gvol(val), line_dash="dash", line_color="#898781", annotation_text=lbl)
                style_fig(fig_mc, height=380)
                fig_mc.update_layout(title="Monte Carlo gas EUR distribution", xaxis_title=f"Gas EUR, {gvol_lbl}",
                                        yaxis_title="Draws")
                st.plotly_chart(fig_mc, use_container_width=True)

# ==========================================================================
# TAB 4 - ML trained on the user's own offset-well dataset
# ==========================================================================
with tab4:
    if not _HAVE_SKLEARN:
        st.error("scikit-learn isn't installed in this environment - run `pip install scikit-learn` "
                 "(see requirements.txt) to use this tab.")
    else:
        st.markdown("##### Train a model on your own historical / offset wells")
        st.caption("Upload a table with one row per well: design/reservoir parameters as columns, plus "
                   "an actual-EUR (or actual-recovery) column. This is a genuinely independent estimate "
                   "- it doesn't touch this app's physics engine at all - but with the small sample "
                   "sizes typical of an offset-well set (tens of wells, not thousands), read the "
                   "cross-validated score as rough guidance, not a precise accuracy figure.")

        data_upload = st.file_uploader("Offset-well dataset CSV", type=["csv"], key="t4_upload")
        if data_upload is not None:
            df_data = pd.read_csv(data_upload)
            st.dataframe(df_data.head(20), use_container_width=True)
            numeric_cols = [c for c in df_data.columns if pd.api.types.is_numeric_dtype(df_data[c])]
            target_col = st.selectbox("Target column (actual EUR or similar)", numeric_cols, key="t4_target")
            feature_cols = st.multiselect("Feature columns (design/reservoir parameters)",
                                             [c for c in numeric_cols if c != target_col],
                                             default=[c for c in numeric_cols if c != target_col],
                                             key="t4_features")
            train_clicked = st.button("Train model", key="t4_train")

            if train_clicked and feature_cols:
                try:
                    reg2 = train_regressor(df_data, feature_cols, target_col)
                    st.session_state["offset_model"] = dict(reg=reg2, df=df_data)
                except Exception as e:
                    st.error(str(e))

            if "offset_model" in st.session_state:
                reg2 = st.session_state["offset_model"]["reg"]
                st.success(f"Trained on {reg2.n_train} wells, {reg2.cv_folds}-fold cross-validated "
                           f"R²={reg2.cv_r2:.3f}, MAE={reg2.cv_mae:.2f}.")
                if reg2.n_train < 20:
                    st.warning(f"Only {reg2.n_train} wells - cross-validated metrics on a set this small "
                               "are noisy; treat this as a rough starting point, not a validated model.")

                col1, col2 = st.columns(2)
                fig_imp2 = go.Figure()
                fig_imp2.add_trace(go.Bar(x=reg2.feature_importance["importance"],
                                             y=reg2.feature_importance["feature"], orientation="h",
                                             marker_color=SEQUENTIAL_BLUE[3]))
                style_fig(fig_imp2, height=360)
                fig_imp2.update_layout(title="Feature importance", xaxis_title="Importance", yaxis_title=None)
                col1.plotly_chart(fig_imp2, use_container_width=True)

                fig_oof2 = go.Figure()
                fig_oof2.add_trace(go.Scatter(x=reg2.oof_actual, y=reg2.oof_predictions, mode="markers",
                                                 marker=dict(size=6, color=CATEGORICAL[0]), name="Out-of-fold"))
                lims2 = [float(np.min(reg2.oof_actual)), float(np.max(reg2.oof_actual))]
                fig_oof2.add_trace(go.Scatter(x=lims2, y=lims2, mode="lines", name="Perfect fit",
                                                 line=dict(color="#898781", dash="dash")))
                style_fig(fig_oof2, height=360, hovermode="closest")
                fig_oof2.update_layout(title="Cross-validated predicted vs. actual",
                                          xaxis_title=f"Actual {target_col}", yaxis_title=f"Predicted {target_col}")
                col2.plotly_chart(fig_oof2, use_container_width=True)

                st.markdown("##### Predict for a new well design")
                new_vals = {}
                pcols = st.columns(min(4, max(len(feature_cols), 1)))
                for i, c in enumerate(feature_cols):
                    default_v = float(df_data[c].median())
                    new_vals[c] = num_in(pcols[i % len(pcols)], c, f"t4_pred_{c}", default_v)
                if st.button("Predict", key="t4_predict"):
                    pred_df = pd.DataFrame({c: [v] for c, v in new_vals.items()})
                    pred = predict_batch(reg2, pred_df)
                    st.metric(f"Predicted {target_col}", f"{pred[0]:.2f}")
        else:
            st.caption("No file uploaded yet.")

st.markdown("---")
with st.expander("Methodology & caveats for this page"):
    st.markdown("""
- **Decline-curve library**: Arps (1945), Duong (2011, SPE 137748), Power-Law
  Exponential (Ilk, Currie & Blasingame 2008, SPE 116731), and Stretched
  Exponential (Valko & Lee 2010, SPE 134231) are all pure empirical curve
  fits to rate-vs-time data - none of them use this app's reservoir-
  engineering model. They're offered for benchmarking / cross-checking,
  not as a replacement for the physics-based forecast on the main page.
- **History-match calibration** re-runs the SAME physics engine used
  elsewhere in this app (`tightgas/simulator.py`) under a nonlinear
  least-squares search over 1-4 chosen parameters, in log-rate space.
  Calibrating more than 1-2 parameters at once from rate data alone is
  often non-unique (e.g. permeability and fracture half-length can trade
  off against each other and still match the same rate curve reasonably
  well) - EUR tends to stay fairly robust to this even when the
  individual parameter values don't recover exactly, but don't
  over-interpret any single fitted value from a heavily multi-parameter fit.
- **ML surrogate (synthetic)** trains a random-forest regressor on this
  app's OWN physics engine, sampled across a parameter range via Latin
  hypercube sampling. It is a speed trade (near-instant Monte Carlo)
  for an approximation of the existing model - it cannot be more
  accurate than the physics engine it was trained to imitate, and its
  accuracy away from the sampled range (extrapolation) is not guaranteed.
- **ML from your own data** trains the same kind of regressor directly on
  a dataset you supply (real wells, real outcomes) - a genuinely
  independent estimate, but entirely bounded by how much and how
  representative that data is. With small offset-well counts (a common
  real-world situation), cross-validated R²/MAE are rough guidance, not a
  precise accuracy figure; a model like this also cannot know about a well
  design that falls well outside the range of wells it was trained on.
- None of the four tools on this page override or feed back into the
  physics engine on the main page - each produces its own independent
  view, meant to be read alongside (not instead of) the RTA/material-
  balance forecast, for triangulation.
    """)
