"""
Quick end-to-end sanity check of the tightgas engineering library (not a
formal pytest suite - run directly with `python tests/smoke_test.py` from
the project root after any change to the core modules).

Checks:
  - Z-factor correlation lands in a physically sane range and matches a
    well-known textbook reference case (Tpr=1.5, Ppr=3.0 -> Z~0.78).
  - p/z material balance is monotonically decreasing with cumulative
    production.
  - All three well types produce a monotonically non-increasing rate
    profile (after the synthetic t=0 seed point) and sane EUR/recovery
    factors.
  - Arps fitting and Blasingame normalization run without error.
  - Multi-well field module: a 1-well field run roughly reconciles with the
    single-well engine, a capacity cap is never exceeded, per-well EURs sum
    to the field EUR, and the schedule-suggestion heuristic proposes a
    sensible well count.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tightgas.pvt import GasComposition, GasPVT, z_factor_dak
from tightgas.material_balance import ogip_volumetric, MaterialBalanceTank
from tightgas.flow_models import ReservoirRock, VerticalWell, MultiFracHorizontalWell, FishboneWell
from tightgas.simulator import SimulationInputs, run_forecast, summarize
from tightgas.type_curves import fit_arps, blasingame_normalize, fetkovich_mccray_qDd
from tightgas.field import WellSpec, FieldInputs, run_field_forecast, suggest_schedule
from tightgas.units import to_display, to_field, to_display_delta, to_field_delta, unit_label, FIELD, METRIC
from tightgas.condensate import CondensateModel, apply_condensate, sum_condensate_across_wells
from tightgas.reporting import resample_periods
from tightgas.type_curves import arps_rate, duong_rate, fit_all_models
from tightgas.history_match import calibrate, _build_case
from tightgas.surrogate import (
    full_ranges, generate_training_data, train_regressor, monte_carlo_eur, _HAVE_SKLEARN,
)


def main():
    z_ref = z_factor_dak(3.0, 1.5, 1.0, 1.0)
    assert abs(z_ref - 0.776) < 0.02, f"DAK reference case mismatch: {z_ref}"
    print(f"[ok] DAK reference case Tpr=1.5, Ppr=3.0 -> Z={z_ref:.3f} (expect ~0.776)")

    comp = GasComposition(gamma_g=0.65, y_co2=0.02, y_n2=0.01, y_h2s=0.0)
    pvt = GasPVT(comp, temp_f=210, p_min=25, p_max=9000, n=300)
    assert 0.5 < float(pvt.z(8000)) < 1.8
    assert 0.005 < float(pvt.mu(8000)) < 0.1
    print("[ok] PVT tables built and in sane range")

    bgi = float(pvt.bg(8000))
    G = ogip_volumetric(area_acres=640, h_ft=80, phi=0.08, sw=0.35, bgi_rcf_per_scf=bgi)
    assert G > 0
    mb = MaterialBalanceTank(pvt=pvt, G_scf=G, pi=8000.0, cf=2e-5)
    ps = [mb.p_from_gp(f * G) for f in np.linspace(0, 0.95, 20)]
    assert all(ps[i] >= ps[i + 1] - 1e-6 for i in range(len(ps) - 1)), "p/z not monotonic"
    print(f"[ok] OGIP={G/1e9:.2f} Bcf, p/z material balance monotonic")

    rock = ReservoirRock(k_md=0.05, phi=0.08, h_ft=80, stress_gamma=0.0)
    wells = {
        "vertical": VerticalWell(rw_ft=0.35, skin=0.0, drainage_area_acres=640),
        "mfhw": MultiFracHorizontalWell(lateral_length_ft=8000, n_frac=25, xf_ft=200, fcd=20,
                                          drainage_area_acres=640, linear_calib=1.0),
        "fishbone": FishboneWell(main_bore_length_ft=6000, n_branches=8, branch_length_ft=600,
                                   drainage_area_acres=640),
    }
    for name, well in wells.items():
        mb_i = MaterialBalanceTank(pvt=pvt, G_scf=G, pi=8000.0, cf=2e-5)
        sim_in = SimulationInputs(pvt=pvt, rock=rock, well=well, mb=mb_i, pwf=1000.0,
                                    t_max_days=365.25 * 25, p_abandon=500, q_min_mscfd=5,
                                    n_steps=300, q_max_mscfd=15000)
        df = run_forecast(sim_in)
        assert df["q_mmscfd"].iloc[1] > 0, f"{name}: zero initial rate"
        assert (df["q_mmscfd"].iloc[1:].diff().dropna() <= 1e-6).all(), f"{name}: rate not declining"
        summ = summarize(df, G)
        assert 0 < summ["Recovery_factor_pct"] < 100
        print(f"[ok] {name}: qi={summ['Initial_rate_MMscfd']:.2f} MMscf/d, "
              f"EUR={summ['EUR_Bcf']:.2f} Bcf, RF={summ['Recovery_factor_pct']:.1f}%")

    qi, di, b, ok = fit_arps(df["t_days"].values, df["q_mscfd"].values)
    t_mb, q_over_dm = blasingame_normalize(df["t_days"].values, df["q_mscfd"].values,
                                             df["dm_psi2_cp"].values, df["Gp_scf"].values)
    _ = fetkovich_mccray_qDd(np.array([0.1, 1, 10]), b=0.5)
    print(f"[ok] Arps fit qi={qi:.0f} Mscf/d Di={di:.4f}/d b={b:.2f}; Blasingame normalization ran")

    # --- multi-well field module ---
    mfhw_params = dict(lateral_length_ft=8000, n_frac=25, xf_ft=200, fcd=20.0, linear_calib=1.0)
    single = WellSpec(name="W1", well_type="mfhw", start_day=0.0, area_acres=640, k_md=0.05, h_ft=80,
                        pi=8000.0, pwf=1000.0, p_abandon=500.0, q_min_mscfd=50.0, q_max_mmscfd=15.0,
                        params=mfhw_params)
    field_in1 = FieldInputs(pvt=pvt, phi=0.08, sw=0.35, cf=2e-5, stress_gamma=0.0, wells=[single],
                              capacity_mmscfd=None, field_abandon_mmscfd=0.05, t_max_days=365.25 * 25)
    _, _, _, fsumm1 = run_field_forecast(field_in1)
    assert abs(fsumm1["field_EUR_Bcf"] - df["Gp_bcf"].iloc[-1]) / df["Gp_bcf"].iloc[-1] < 0.15, \
        "1-well field engine diverges too much from the single-well engine"
    print(f"[ok] 1-well field cross-check: field EUR={fsumm1['field_EUR_Bcf']:.2f} Bcf "
          f"vs single-well EUR={df['Gp_bcf'].iloc[-1]:.2f} Bcf")

    wells6 = [WellSpec(name=f"W{i+1}", well_type="mfhw", start_day=i * 180.0, area_acres=640,
                         k_md=0.05, h_ft=80, pi=8000.0, pwf=1000.0, p_abandon=500.0, q_min_mscfd=50.0,
                         q_max_mmscfd=15.0, params=mfhw_params) for i in range(6)]
    field_in6 = FieldInputs(pvt=pvt, phi=0.08, sw=0.35, cf=2e-5, stress_gamma=0.0, wells=wells6,
                              capacity_mmscfd=25.0, field_abandon_mmscfd=0.5, t_max_days=365.25 * 20)
    field_df6, well_dfs6, well_summ6, fsumm6 = run_field_forecast(field_in6)
    assert field_df6["q_mmscfd"].max() <= 25.0 * 1.001, "field rate exceeded the capacity cap"
    sum_eur = sum(s["EUR_Bcf"] for s in well_summ6.values())
    assert abs(sum_eur - fsumm6["field_EUR_Bcf"]) < 0.05, "per-well EUR does not reconcile with field EUR"
    assert fsumm6["peak_field_rate_MMscfd"] <= 25.0 * 1.001, "peak_field_rate_MMscfd unit/scale bug"
    print(f"[ok] 6-well staggered field: peak={fsumm6['peak_field_rate_MMscfd']:.2f} MMscf/d "
          f"(cap=25), field EUR={fsumm6['field_EUR_Bcf']:.2f} Bcf, sum of well EURs={sum_eur:.2f} Bcf")

    starts = suggest_schedule(df["t_days"].values, df["q_mscfd"].values, capacity_mmscfd=25.0,
                                min_cadence_days=120, max_wells=15, t_max_days=365.25 * 10)
    assert len(starts) >= 2, "suggestion tool should propose more than 1 well for a 25 MMscf/d target"
    print(f"[ok] schedule suggestion: {len(starts)} wells, starts={[round(s) for s in starts]}")

    # --- unit conversion registry ---
    cases = [
        ("pressure", 1000.0), ("temperature", 68.0), ("length", 1.0), ("area", 640.0),
        ("rate_gas_mscfd", 500.0), ("rate_gas_mmscfd", 25.0), ("volume_gas_bcf", 15.0),
        ("rate_liquid_bbld", 100.0), ("volume_liquid_mstb", 50.0), ("cgr", 30.0),
        ("perm", 0.05), ("dimensionless", 0.5),
    ]
    for qty, val in cases:
        disp = to_display(val, qty, METRIC)
        back = to_field(disp, qty, METRIC)
        assert abs(back - val) < 1e-6, f"{qty} round-trip failed: {val} -> {disp} -> {back}"
        assert to_display(val, qty, FIELD) == val and to_field(val, qty, FIELD) == val, \
            f"{qty}: Field system should be a no-op"
    assert abs(to_display(1000.0, "pressure", METRIC) - 6894.757) < 0.01
    assert abs(to_display(68.0, "temperature", METRIC) - 20.0) < 1e-9
    assert abs(to_display(1.0, "length", METRIC) - 0.3048) < 1e-9
    assert abs(to_display(640.0, "area", METRIC) - 259.0) < 0.5
    assert abs(to_display(25.0, "rate_gas_mmscfd", METRIC) - 708.42) < 0.5
    assert abs(to_display(15.0, "volume_gas_bcf", METRIC) - 424.75) < 0.5
    assert abs(to_display(30.0, "cgr", METRIC) - 0.16844) < 0.001
    d_disp = to_display_delta(10.0, "temperature", METRIC)
    assert abs(d_disp - 10.0 * 5 / 9) < 1e-9, "temperature delta should have no +32 offset"
    assert abs(to_field_delta(d_disp, "temperature", METRIC) - 10.0) < 1e-9
    print("[ok] units: Field/Metric round-trip, known-value spot checks, Field no-op, "
          "temperature delta has no offset")

    # --- condensate / CGR ---
    t_c = np.linspace(0, 365, 50)
    q_gas_c = np.linspace(15000, 5000, 50)
    p_avg_c = np.linspace(8000, 4000, 50)
    import pandas as pd
    df_c = pd.DataFrame({"t_days": t_c, "q_mscfd": q_gas_c, "p_avg_psia": p_avg_c})

    model_const = CondensateModel(cgr_i=30.0, pi=8000, p_abandon=500)
    dfc = apply_condensate(df_c, model_const)
    assert np.allclose(dfc["q_cond_bbld"].values, q_gas_c / 1000.0 * 30.0), \
        "constant CGR condensate rate mismatch"
    assert dfc["Np_bbl"].iloc[-1] > 0

    model_decl = CondensateModel(cgr_i=40.0, pi=8000, p_abandon=500, cgr_abandon=10.0)
    assert abs(model_decl.cgr_at(np.array([8000.0]))[0] - 40.0) < 1e-9
    assert abs(model_decl.cgr_at(np.array([500.0]))[0] - 10.0) < 1e-9
    mid_cgr = model_decl.cgr_at(np.array([(8000 + 500) / 2]))[0]
    assert abs(mid_cgr - 25.0) < 0.5, f"expected ~25 midpoint CGR, got {mid_cgr}"

    well_dfs_c = {"W1": apply_condensate(df_c, model_const), "W2": apply_condensate(df_c, model_decl)}
    field_cond = sum_condensate_across_wells(well_dfs_c, t_c)
    assert np.allclose(field_cond["q_cond_bbld"].values,
                        well_dfs_c["W1"]["q_cond_bbld"].values + well_dfs_c["W2"]["q_cond_bbld"].values)
    print(f"[ok] condensate: constant + pressure-declining CGR correct, field aggregation sums "
          f"across wells (peak={field_cond['q_cond_bbld'].max():.1f} bbl/d)")

    # --- anniversary-based period resampling ---
    t2 = np.linspace(0, 365.25 * 3, 2000)
    q_const = 1000.0
    gp_scf = t2 * q_const * 1000.0
    yearly = resample_periods(t2, {"gas": gp_scf}, period="year")
    assert len(yearly) == 3, f"expected 3 full years, got {len(yearly)}"
    assert abs(yearly["gas_volume"].sum() - gp_scf[-1]) / gp_scf[-1] < 1e-6, \
        "yearly volumes don't sum to total"
    assert all(abs(r["gas_avg_rate"] - q_const * 1000.0) < 1.0 for _, r in yearly.iterrows()), \
        "constant-rate case should show ~constant average rate per year"

    monthly = resample_periods(t2, {"gas": gp_scf}, period="month")
    assert abs(monthly["gas_volume"].sum() - gp_scf[-1]) / gp_scf[-1] < 1e-6, \
        "monthly volumes don't sum to total"

    t3 = np.linspace(0, 400, 500)
    yearly3 = resample_periods(t3, {"gas": t3 * 500.0}, period="year")
    assert len(yearly3) == 2, f"expected 2 periods (1 full + 1 partial), got {len(yearly3)}"
    assert yearly3["days_in_period"].iloc[-1] < 365.25, "last period should be truncated"
    print(f"[ok] reporting: yearly ({len(yearly)} periods) and monthly ({len(monthly)} periods) "
          f"resampling reconcile with totals; partial final period handled")

    # --- decline-curve library (Duong/PLE/SEDM alongside Arps) ---
    t_dc = np.linspace(0.5, 3650, 300)
    q_dc = arps_rate(t_dc, 15000.0, 0.003, 1.3) * (1 + np.random.default_rng(0).normal(0, 0.02, len(t_dc)))
    fits = fit_all_models(t_dc, q_dc)
    assert set(fits.keys()) == {"Arps", "Duong", "Power-Law Exponential (PLE)", "Stretched Exponential (SEDM)"}
    best = min(fits.items(), key=lambda kv: kv[1]["log_rmse"] if np.isfinite(kv[1]["log_rmse"]) else np.inf)
    assert best[0] == "Arps", f"expected Arps to win on Arps-generated data, got {best[0]}"
    q_duong = duong_rate(t_dc, 500.0, 2.0, 1.3)
    assert np.all(np.isfinite(q_duong)) and np.all(q_duong >= 0), "Duong model produced invalid rates"
    print(f"[ok] decline-curve library: all 4 families fit; best log-RMSE on Arps-generated "
          f"data correctly identifies Arps ({fits['Arps']['log_rmse']:.4f})")

    # --- history-match calibration against the real physics engine ---
    true_well = dict(lateral_length_ft=8000, n_frac=25, xf_ft=180.0, fcd=20.0, linear_calib=1.0)
    fixed_hm = dict(area_acres=640.0, k_md=0.03, h_ft=80.0, phi=0.08, sw=0.35, pi=8000.0, cf=2e-5,
                      pwf=1000.0, p_abandon=500.0, q_min_mscfd=20.0, q_max_mscfd=15000.0, well_params=true_well)
    df_true_hm, G_hm, _ = _build_case(pvt, "mfhw", n_steps=300, t_max_days=365.25 * 10, **fixed_hm)
    rng_hm = np.random.default_rng(1)
    t_obs = np.arange(30, 365.25 * 3, 30.0)
    q_obs = np.interp(t_obs, df_true_hm["t_days"].values, df_true_hm["q_mscfd"].values)
    q_obs *= 1 + rng_hm.normal(0, 0.03, len(t_obs))
    fixed_wrong = dict(fixed_hm, k_md=0.08)
    res_hm = calibrate(pvt, "mfhw", fixed_wrong, ["k_md"], initial_guess={"k_md": 0.08},
                         t_obs=t_obs, q_obs=q_obs, t_forecast_days=365.25 * 10, max_nfev=40)
    assert abs(res_hm.fitted["k_md"] - 0.03) / 0.03 < 0.15, f"k_md recovery off: {res_hm.fitted}"
    assert res_hm.log_rmse < 0.08, f"calibration fit quality too low: {res_hm.log_rmse}"
    print(f"[ok] history-match calibration recovers k_md={res_hm.fitted['k_md']:.4f} (true 0.03) "
          f"from a wrong 0.08 start, log_rmse={res_hm.log_rmse:.4f}")

    # --- ML surrogate: synthetic training + cross-validation + Monte Carlo ---
    if _HAVE_SKLEARN:
        fixed_sur = dict(area_acres=640.0, k_md=0.05, h_ft=80.0, phi=0.08, sw=0.35, pi=8000.0, cf=2e-5,
                           pwf=1000.0, p_abandon=500.0, q_min_mscfd=20.0, q_max_mscfd=15000.0,
                           well_params=dict(lateral_length_ft=8000, n_frac=25, xf_ft=200.0, fcd=20.0,
                                              linear_calib=1.0))
        ranges = full_ranges("mfhw")
        df_train = generate_training_data(pvt, "mfhw", ranges, n_samples=80, fixed=fixed_sur,
                                             t_max_days=365.25 * 15, n_steps=120, seed=42)
        feature_cols = list(ranges.keys())
        reg = train_regressor(df_train, feature_cols, "EUR_Bcf", n_estimators=150)
        assert reg.cv_r2 > 0.4, f"surrogate cv_r2 too low even for a small smoke-test sample: {reg.cv_r2}"
        base_values = {c: fixed_sur.get(c, fixed_sur["well_params"].get(c, 0.0)) for c in feature_cols}
        draws = monte_carlo_eur(reg, {"k_md": ("uniform", 0.01, 0.2)}, base_values, n_draws=2000, seed=2)
        p10, p50, p90 = np.percentile(draws, [10, 50, 90])
        assert p10 < p50 < p90, "Monte Carlo percentiles not monotonic"
        print(f"[ok] ML surrogate: {reg.n_train}-sample synthetic training, cv_r2={reg.cv_r2:.2f}; "
              f"Monte Carlo P10/P50/P90 = {p10:.1f}/{p50:.1f}/{p90:.1f} Bcf (monotonic)")
    else:
        print("[skip] scikit-learn not installed - ML surrogate checks skipped")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
