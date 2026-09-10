# Tight Gas Production Profile Builder

A Streamlit app that builds a full-life production forecast for a tight-gas
well from first-principles reservoir engineering: real-gas PVT, a p/z tank
material balance, and rate-transient deliverability models for **vertical**,
**multi-stage-fractured horizontal**, and **fishbone / multilateral** wells —
plus Arps and Fetkovich-McCray/Blasingame type-curve diagnostics, a
sensitivity tornado chart, an optional condensate/CGR add-on, and
yearly/monthly profile export. A second page extends this to a **multi-well
drainage strategy**: stack any number of wells, each with its own design,
online date, and condensate yield, fit the aggregate field profile to a
target facility capacity, and cut it off at a field-level abandonment rate.
Every input can be shown in **Field or Metric units** (the underlying physics
always runs in field units — see below).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`). Use the
page navigation at the top of the sidebar to switch between the single-well
tool ("app") and **Multi-Well Drainage Strategy**.

## What it does

1. **PVT** (`tightgas/pvt.py`) — builds Z-factor (Dranchuk & Abu-Kassem,
   1975, with Wichert-Aziz sour-gas correction), gas viscosity (Lee,
   Gonzalez & Eakin, 1966), Bg, gas compressibility, and the real-gas
   pseudo-pressure function m(p) (Al-Hussainy, Ramey & Crawford, 1966) for
   your gas composition, temperature, and CO2/N2/H2S content.
2. **Volumetrics & material balance** (`tightgas/material_balance.py`) —
   computes original gas in place (OGIP) from reservoir geometry, then
   tracks average reservoir pressure vs. cumulative production with a p/z
   tank model (with an optional compaction/water-drive correction term for
   the concave-down p/z trends common in overpressured tight-gas plays).
3. **Well deliverability** (`tightgas/flow_models.py`) — for the selected
   completion (vertical / multi-frac horizontal / fishbone), computes rate
   at any point in time as a hard switch between a **transient** stem
   (early time, before the pressure disturbance reaches the drainage
   boundary) and a **boundary-dominated / pseudo-steady-state** stem
   (governed directly by the material balance). This mirrors how
   Fetkovich's classic combined type curves are constructed.
4. **Coupled simulator** (`tightgas/simulator.py`) — marches the well
   forward in time under constant flowing bottomhole pressure, updating
   average reservoir pressure via material balance at every step, until an
   abandonment pressure, economic rate limit, or the forecast horizon is
   reached.
5. **Type-curve diagnostics** (`tightgas/type_curves.py`) — Arps
   exponential/hyperbolic/harmonic decline fitting, and a Blasingame-style
   normalized-rate vs. material-balance-time plot with the Fetkovich-McCray
   depletion-stem family (qDd = 1/(1+b·tDd)^(1/b)) overlaid for reference.

Full equations, unit systems, and literature references are documented in
each module's docstring — read those before trusting a number from this
tool for anything beyond scoping.

## Well configuration methodology (read this before using the horizontal/fishbone results)

- **Vertical well**: classical Darcy radial flow, transient (log-approximation,
  field units) switching to pseudo-steady-state at the standard tDA = 0.1
  criterion.
- **Multi-frac horizontal well**: early/mid-life is dominated by transient
  linear flow into the fracture faces (the well-known √t behavior — Wattenbarger
  et al., 1998; Nobakht & Clarkson, 2012). A **linear-flow calibration
  multiplier** is exposed because in practice fracture half-length and matrix
  permeability are rarely both known independently — this is exactly the
  parameter analysts history-match in commercial RTA software. Once the
  transient reaches the edge of the well's drainage element, the model
  switches to a boundary-dominated stem using an **effective wellbore radius**
  (Cinco-Ley-type asymptotic behavior, r'w → xf/2 at high fracture
  conductivity) plugged into the same radial PSS equation as the vertical
  well. This is a standard engineering simplification for long-term
  forecasting, **not** a full trilinear or numerical simulation.
- **Fishbone / multilateral well**: approximated as a horizontal well with
  one low-conductivity "pseudo-fracture" per branch (half-length = branch
  length / 2). Rigorous analytical fishbone RTA is not standardized in the
  literature — treat these results as directional/order-of-magnitude only.
- A **facility/choke-constrained maximum rate** is applied in all cases.
  Real wells are almost always produced against a surface constraint, and
  this also regularizes the transient linear-flow solution, which is
  mathematically unbounded as t → 0 for an idealized infinite-conductivity
  fracture at constant pwf.

## Multi-well drainage strategy (`pages/1_Multi_Well_Drainage_Strategy.py`)

Builds on the single-well engine via `tightgas/field.py`:

- Each well is its own **independent tank** (own OGIP, own p/z material
  balance, own deliverability) — there is no explicit well-to-well pressure
  interference solved for. This is standard practice for tight-gas
  development planning at typical spacing; it is **not** appropriate for
  tightly-spaced infill wells with significant interference.
- Wells share the field's produced-fluid PVT system and rock-compaction /
  stress-sensitivity behavior, but each well independently sets its own
  permeability, net pay, drainage area, initial pressure, completion
  design, online date, and operating constraints ("each well takes its own
  parameters").
- **Capacity fitting**: when the combined deliverable potential of all
  online wells exceeds the target facility capacity, every online well is
  curtailed by the **same proportional factor** so the field total exactly
  equals the capacity (proportional allocation — the simplest and most
  common capacity-sharing rule). A curtailed well's material balance only
  depletes by what it actually produced, so curtailment extends its life.
  A well is judged individually "dead" (below its own abandonment pressure
  or economic rate) using its **uncurtailed** potential — being choked back
  for facility reasons isn't the same as being depleted.
- **Field abandonment**: the forecast stops once total field production
  falls to/below a field-level abandonment rate (after having ramped up at
  least once), or once every well is individually dead, whichever comes
  first.
- **Schedule suggestion tool**: an optional, explicitly first-pass (not an
  optimizer) greedy heuristic that proposes a well count and timing to
  reach and hold a target plateau, using one representative "type well"
  design. It walks forward in time and adds another well whenever the
  combined potential of already-scheduled wells is projected to fall below
  the target. Apply it to pre-fill the well list, then fine-tune (or fully
  replace) any individual well.

## Units, condensate/CGR, sensitivity, and yearly/monthly export

- **Field / Metric toggle** (`tightgas/units.py`, `tightgas/ui_helpers.py`) —
  every widget, plot, table, and on-screen metric can be shown in either
  system; the underlying PVT correlations and flow equations are field-unit
  correlations by construction and are **never** converted — only the
  presentation layer is. Switching units re-converts your last entered value
  per field (rather than resetting to a default), so edits survive a switch.
  Metric conventions used: kPa(a), °C, m, hectares, 10³ m³/d, 10⁶ m³, m³/d,
  m³/10³ m³ gas (see the module docstring for the full table). CSV exports
  are always in field units regardless of the display system, so a file
  opened later is unambiguous.
- **Condensate / CGR** (`tightgas/condensate.py`) — an optional, per-well
  yield-based add-on: condensate rate = gas rate × CGR(p), where CGR is
  either constant or linearly interpolated between an initial-pressure value
  and an abandonment-pressure value (a simple, transparent approximation of
  falling liquid yield as a retrograde-condensate reservoir depletes below
  its dewpoint). This is **not** a compositional/EOS liquid-dropout model.
  On the multi-well page, condensate is summed only across the wells that
  have it enabled.
- **Sensitivity tornado** (single-well page) — one-at-a-time ± X% perturbation
  of matrix permeability, drainage area, and the well type's key completion
  parameter (wellbore radius / fracture half-length / branch length), each
  re-run through the full forecast to show its impact on gas EUR. Not a
  probabilistic (Monte Carlo) uncertainty analysis.
- **Yearly / monthly export** (`tightgas/reporting.py`) — resamples the daily
  forecast onto anniversary-based periods (Year 1 = days 0–365.25 from the
  start of the forecast, not aligned to a real calendar date — the standard
  convention for comparing wells that came online on different real dates)
  and offers both an on-screen table/bar chart and a CSV download, gas and
  (when enabled) condensate.
- **Chart styling** (`tightgas/viz.py`) — a fixed-order, colorblind-validated
  categorical palette (never reassigned when the number of series changes)
  plus a diverging blue/red pair for the tornado chart, applied consistently
  across both pages.

## Type-curve benchmarking & ML (`pages/2_Type_Curve_Benchmarking_and_ML.py`)

A third page with four complementary ways to sanity-check or sharpen the
single-well forecast, in increasing order of how much they depend on real
data you supply:

1. **Decline-curve library & offset-well benchmark** (`tightgas/type_curves.py`) -
   fits Arps plus three published unconventional-well decline families -
   **Duong** (2011, SPE 137748; built for linear-flow-dominated tight/
   fracture wells), **Power-Law Exponential / PLE** (Ilk, Currie &
   Blasingame, 2008, SPE 116731; blends early power-law decline into a
   late-time exponential floor), and **Stretched Exponential / SEDM**
   (Valko & Lee, 2010, SPE 134231) - to either the current forecast or an
   uploaded offset well's real production CSV, and ranks them by
   goodness-of-fit (R², log-RMSE). All four are pure curve fits; none use
   the reservoir-engineering model. If you upload a real offset well while
   a forecast already exists, the page also overlays them directly for a
   visual "does this look like the analog" benchmark.
2. **History-match calibration** (`tightgas/history_match.py`) - given a
   well's actual rate-vs-time history, solves for the best-fit values of
   1-4 chosen uncertain parameters (matrix permeability, drainage area,
   and the well type's key completion parameter) by nonlinear least
   squares **against the same physics engine used elsewhere in this app**
   (`tightgas/simulator.py`) - not a separate empirical curve. This is the
   standard RTA workflow for getting a better estimate from real
   production, and it's the most defensible of the four since it improves
   the validated physics rather than replacing it with something opaque.
   Fitting more than 1-2 parameters at once from rate data alone can be
   non-unique (permeability and fracture half-length routinely trade off
   against each other while still matching the same rate curve) - the
   page surfaces this caveat and generally still recovers EUR accurately
   even when individual parameters don't.
3. **ML surrogate, synthetic** (`tightgas/surrogate.py`) - samples the
   physics engine across a parameter-uncertainty range (Latin hypercube via
   `scipy.stats.qmc`), runs a full forecast per sample, and trains a
   scikit-learn random-forest regressor on {parameters -> gas EUR}. This
   is an explicit speed trade, not an independent estimate: it approximates
   this app's own engine so that a Monte Carlo probabilistic EUR scan
   (P10/P50/P90) that would take many minutes through the full simulator
   runs in well under a second once trained.
4. **ML from your own data** (also `tightgas/surrogate.py`, same training
   function) - trains the same kind of regressor directly on a table of
   historical/offset wells you upload (design parameters + actual EUR). A
   genuinely independent estimate, bounded entirely by how much and how
   representative that data is; cross-validated R²/MAE are reported so a
   too-small or too-noisy dataset is visible rather than silently trusted.

None of the four feed back into or override the main page's forecast -
each is its own independent cross-check, meant to be read alongside the
physics-based forecast for triangulation, not as a replacement for it.

## Units audit (fourth round)

A full pass was made through every unit-tagged input, plot, table, and
export on both pages, cross-checking each `qty` string against
`tightgas/units.py`'s conversion registry and re-deriving the conversion
factors by hand (psi->kPa, ft->m, acres->ha, Mscf/d & MMscf/d & Bcf ->
10³/10⁶ m³, bbl->m³, STB/MMscf->m³/10³m³, and the temperature affine
transform). No incorrect conversions or field/display unit mix-ups were
found. Three cosmetic gaps were fixed:

- The initial-pressure widget's "normal pressure gradient" help text used
  to always show a psia reference number even in Metric mode; it now
  converts to the active display system.
- The Blasingame-normalized rate-transient diagnostic (q/Δm(p) vs.
  material-balance time) intentionally stays in field-unit pseudo-pressure
  terms in both systems — this is normal RTA practice, not a bug — but the
  page now says so explicitly in a caption instead of leaving it unexplained.
- The compaction/water-drive correction (`cf`) and permeability-modulus
  (`stress_gamma`) inputs, which are tiny inverse-pressure numbers, used a
  fixed `%.6f` display format that could round to 1 significant digit once
  converted to Metric (smaller numbers); switched to scientific notation
  (`%.3e`) so precision reads correctly in either system.

## Known limitations

- Single-phase dry gas, 0-D tank material balance (no reservoir simulation
  grid, no water/condensate banking, no aquifer support beyond the lumped
  compaction/water-drive correction factor).
- Constant flowing bottomhole pressure operation only (no rate-then-pwf-decline
  schedules, no user-defined pwf(t) table).
- No non-Darcy (turbulent) flow term, no liquid loading, no time-dependent
  proppant conductivity decline.
- The effective-wellbore-radius fit for fracture conductivity is a smooth
  engineering approximation of the Cinco-Ley & Samaniego (1981) tabulated
  solution, not the tabulated solution itself.
- Condensate is a yield-based (CGR) add-on, not a compositional/EOS
  liquid-dropout calculation — see above.

## Project layout

```
app.py                       Streamlit UI - single well
pages/
  1_Multi_Well_Drainage_Strategy.py   Streamlit UI - multi-well field planning
  2_Type_Curve_Benchmarking_and_ML.py Streamlit UI - decline-curve library, offset
                                       benchmarking, history-match calibration, ML
tightgas/
  pvt.py                     Real-gas PVT correlations
  material_balance.py        OGIP + p/z tank material balance
  flow_models.py             Vertical / MFHW / fishbone deliverability
  simulator.py                Time-marching coupled simulator (single well)
  field.py                    Multi-well field aggregation, capacity fitting,
                               schedule-suggestion heuristic
  type_curves.py              Arps/Duong/PLE/SEDM fitting + Blasingame/Fetkovich-McCray
                               diagnostics
  condensate.py                CGR / condensate yield post-processing
  reporting.py                 Anniversary-based yearly/monthly resampling
  units.py                     Field <-> Metric conversion registry
  ui_helpers.py                 Shared Streamlit widget helpers (incl. unit-aware inputs)
  viz.py                        Shared chart color palette & styling
  history_match.py              Nonlinear least-squares calibration of the physics
                                 engine against real production history
  surrogate.py                   ML surrogate training (synthetic or user-uploaded
                                 data) + Monte Carlo EUR sampling
tests/smoke_test.py           End-to-end sanity check - rerun after any edit
requirements.txt
```

## Disclaimer

This is an engineering **scoping** tool built from published, standard
correlations and simplified analytical flow models. Validate against offset
well data, nodal analysis, and/or full numerical simulation before using
results for investment decisions.
