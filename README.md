# ProspectRisk Studio

Exploration prospect assessment in Streamlit: probabilistic volumetrics, geological and commercial
chance of success, multi-segment prospects and portfolio aggregation with shared play risk.

The engine (`prospectrisk/`) is pure Python and fully separate from the UI (`ui/`, `app.py`).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens on a fictitious two-play, two-prospect demo project. Tests:

```bash
python tests/run_all.py
```

## What it does

| Area | Capability |
|---|---|
| Inputs | Constant, uniform, triangular, PERT, normal, lognormal, beta, discrete, percentile table; P90/P10 fitting; exact truncation |
| Sampling | Latin Hypercube or Monte Carlo, fixed seed, Iman–Conover rank correlation with Spearman-exact targets |
| Dependencies | Correlations within and across segments; links (same percentile); formula-defined inputs with optional uncertainty and fixed formula units |
| Volume cut-off | Per-segment minimum (reduces Pg, or renormalises) and maximum (renormalises) on any output |
| GRV | Area × thickness × geometric factor; direct GRV; area–depth table with thickness or base surface; contact by depth, column height or fill-to-spill; spill-point limit |
| Fluids | Oil, gas/condensate, or oil with gas cap by probability; Bo, Bg or E, GOR, CGR, separate recovery factors and gas saturation |
| Outputs | GRV, HCPV, STOIIP, GIIP, solution gas, condensate, recoverable liquids/gas, oil equivalent (NCS or SPE convention) |
| Chance | Play / prospect / segment factors in five geological categories; probability or chance-adequacy matrix (editable); risk class; key risk |
| Commercial | Minimum economic field size on the prospect total; Pc, conditional commercial volumes |
| Aggregation | Segments into prospects, prospects into portfolios; shared play and prospect risk drawn once per trial; copula dependency of remaining risks; success-volume rank correlation; number of discoveries |
| Analysis | Histograms, success and risked expectation curves, risk-versus-volume crossplot, tornado, rank-correlation contribution to variance |
| QC | P10/P90 ratio, Swanson's-mean check, physical-limit checks, clipped samples, Pg range, simulated vs input chance, MEFS impact |
| Files | Project YAML, Excel workbook (stats, chance, trials), PDF report, FieldVista bridge YAML |
| Units | Metric (NCS) or field presets, per-quantity overrides, per-input units (ha, km², acres, mi², m, ft, Sm³…Tscf, rb/Mscf, %…); oilfield units internally |

## Conventions

* **P90 is the low case, P10 the high case** everywhere in this app. The FieldVista bridge file also
  writes `percentile_10/50/90` in statistical convention so the two apps cannot be confused.
* Pg(segment) = Π play factors × Π prospect factors × Π segment factors.
* Risked mean = Pg × success-case mean. Dependencies change spread and discovery counts, not the risked mean.

## Structure

```
app.py                      navigation, theme, unit switch
prospectrisk/
  distributions.py          distribution families, truncation, fitting, serialisation
  correlation.py            LHS, Iman–Conover, nearest correlation matrix
  expressions.py            safe AST-based formula evaluation
  volumetrics.py            GRV methods, phases, in-place and recoverable volumes
  risk.py                   chance factors, adequacy matrix, commercial chance, curves
  aggregation.py            shared-event and copula aggregation
  sensitivity.py            tornado, rank correlation
  project.py                data model (plays, prospects, segments), run orchestration
  qc.py                     consistency checks with editable thresholds
  io_export.py              YAML, Excel, PDF, FieldVista bridge
  defaults.py, examples.py  placeholder inputs, anonymised demo project
ui/                          Streamlit pages, editors, charts
tests/                       engine suites + headless UI smoke test
```

## Validation

The engine suites check against closed-form results: distribution means and percentiles, truncated normal
against SciPy, 7758 bbl/acre-ft hand calculations, metric integrity (1 km² × 100 m × φ 0.2 × Sh 0.8 = 16.0 MSm³),
cone and slab area–depth GRV, gas-cap splitting, Iman–Conover target correlations with unchanged marginals,
P(any success) for independent and shared-play cases, bivariate-normal copula probabilities, and risked-mean
conservation through aggregation.

`tests/smoke_ui.py` runs every page with Streamlit and Plotly stubbed, across all distribution types, GRV methods
and contact modes, plus the add/duplicate/delete, export and portfolio buttons. It does not test layout.

## Scope

Implements published, industry-standard exploration assessment methods. It is not a copy of any commercial
product and does not reproduce proprietary algorithms, so results will differ where methods or defaults differ.
Not included: map-based GRV from grids, prospect-level economics beyond MEFS (use FieldVista), drilling-sequence
optimisation, and multi-user databases.

Screening-level decision support; results must be reviewed by qualified professionals.
Not affiliated with or endorsed by Equinor or SLB. MIT licence.
