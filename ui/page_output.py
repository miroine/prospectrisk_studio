"""Quality control, export and method documentation pages."""
from __future__ import annotations

from dataclasses import fields

import pandas as pd
import streamlit as st

from prospectrisk import qc
from prospectrisk.io_export import fieldvista_bridge_yaml, pdf_report, project_to_yaml, results_to_excel
from ui import state
from ui.page_model import _slug


def page_qc() -> None:
    p = state.project()
    st.title("Quality control")
    t: qc.QCThresholds = st.session_state.qc_thresholds
    with st.expander("Check thresholds"):
        st.caption("Screening heuristics. Adjust them to your organisation's calibration.")
        cols = st.columns(2)
        for i, f in enumerate(fields(t)):
            with cols[i % 2]:
                setattr(t, f.name, st.number_input(f.name.replace("_", " ").capitalize(), value=float(getattr(t, f.name)),
                                                   format="%.4g", key=f"qc_{f.name}"))
    runs = state.valid_runs()
    if not runs:
        st.info("Run the prospects to check them.")
        if st.button("Run all prospects", type="primary", key="qc_run"):
            state.run_all()
            st.rerun()
        return
    findings = []
    for name, r in runs.items():
        try:
            findings.extend(qc.check_prospect(p, r, t))
        except ValueError as exc:
            findings.append(qc.Finding("error", name, f"Could not check: {exc}"))
        if state.is_stale(name):
            findings.append(qc.Finding("warning", name, "Results are out of date with the inputs; re-run before relying on them."))
    missing = [pr.name for pr in p.prospects if pr.name not in runs]
    for m in missing:
        findings.append(qc.Finding("info", m, "Not run yet."))
    df = pd.DataFrame([f.as_row() for f in findings])
    counts = df["level"].value_counts()
    c = st.columns(3)
    c[0].metric("Errors", int(counts.get("error", 0)))
    c[1].metric("Warnings", int(counts.get("warning", 0)))
    c[2].metric("Notes", int(counts.get("info", 0)))
    show = st.multiselect("Show", ["error", "warning", "info"], default=["error", "warning"], key="qc_show")
    for f in findings:
        if f.level not in show:
            continue
        msg = f"**{f.scope}**: {f.message}"
        {"error": st.error, "warning": st.warning, "info": st.info}[f.level](msg)


def page_export() -> None:
    p = state.project()
    st.title("Export")
    slug = _slug(p.name)
    st.subheader("Project file")
    st.download_button("Download project (YAML)", project_to_yaml(p), f"{slug}.yaml", "application/x-yaml", key="exp_yaml")
    runs = state.valid_runs()
    if not runs:
        st.info("Run the prospects to export results and reports.")
        return
    stale = [n for n in runs if state.is_stale(n)]
    if stale:
        st.warning(f"Out-of-date results for: {', '.join(stale)}. Re-run before exporting.")

    st.subheader("Results workbook")
    if st.button("Prepare Excel workbook", key="prep_xlsx"):
        with st.spinner("Writing workbook"):
            st.session_state.xlsx = results_to_excel(p, runs)
    if st.session_state.get("xlsx"):
        st.download_button("Download workbook (.xlsx)", st.session_state.xlsx, f"{slug}_results.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")

    st.subheader("Printable report")
    pf = st.session_state.get("portfolio")
    include_pf = st.checkbox("Include the last portfolio aggregation", value=pf is not None, disabled=pf is None,
                             key="pdf_pf")
    if st.button("Prepare PDF report", key="prep_pdf"):
        with st.spinner("Building report"):
            st.session_state.pdf = pdf_report(p, runs, pf.summary() if (pf is not None and include_pf) else None)
    if st.session_state.get("pdf"):
        st.download_button("Download report (.pdf)", st.session_state.pdf, f"{slug}_report.pdf", "application/pdf",
                           key="dl_pdf")

    st.subheader("FieldVista bridge")
    st.markdown("Success-case recoverable volumes, Pg and Pc per prospect, for decision-tree leaves. Both percentile "
                "conventions are written explicitly (`low/mid/high` and `percentile_10/50/90`) because FieldVista's "
                "Monte Carlo views use the statistical convention.")
    y = fieldvista_bridge_yaml(p, runs)
    st.code(y[:1500] + ("\n..." if len(y) > 1500 else ""), language="yaml")
    st.download_button("Download bridge file (YAML)", y, f"{slug}_fieldvista_bridge.yaml", "application/x-yaml",
                       key="dl_bridge")


METHOD = r"""
### Workflow
1. **Project**: simulation settings, plays with their play-level chance factors, and the chance-adequacy matrix.
2. **Prospects**: for each prospect, its play, MEFS and shared chance factors; then one or more **segments**
   (stacked reservoirs, fault blocks, compartments), each with volumetric inputs and its own chance factors.
3. **Results**, **Sensitivity**, **Portfolio**, **Quality control**, **Export**.

### Volumetrics (per segment, per trial)
GRV comes from area × gross thickness × geometric factor, a direct GRV distribution, or an area–depth table
integrated between the top surface and either a base surface or a vertical thickness, down to a contact
defined by depth, column height, or fill fraction of the spill-point relief.

GRV is split into oil leg and gas cap according to the sampled phase. Then
HCPV = GRV × N/G × φ × Sh, STOIIP = HCPV / Bo, GIIP = HCPV / Bg, solution gas = STOIIP × GOR,
condensate = GIIP × CGR, and recoverable volumes apply recovery factors.
Oil equivalent uses the selected convention (NCS 1000 Sm³ gas = 1 Sm³ o.e., or SPE 6 Mscf = 1 boe).

Inputs are sampled by Latin Hypercube or plain Monte Carlo. Rank correlations are imposed with the
Iman–Conover method, which leaves every marginal distribution unchanged. Samples outside physical bounds
(for example negative porosity) are clipped and reported; truncating the distribution is the better fix.

### Dependencies between parameters
* **Rank correlation** between uncertain inputs of a segment (Correlations tab), or between inputs of different
  segments (prospect, Segment dependencies tab). Targets are Spearman rank correlations; pairs you leave out are
  treated as uncorrelated, so inconsistent chains are adjusted to the nearest valid set and flagged.
* **Links**: input B takes the same percentile as input A in every trial, while keeping its own distribution.
* **Formulas**: any input can be defined as a formula of other inputs, e.g. `1 - 0.08 / porosity` or
  `0.42 * exp(-0.00035 * contact_depth)`, optionally multiplied by or added to an uncertainty distribution.
  Each formula stores the units of the names it uses when it is written, so changing display units later does
  not change its meaning.
* Segments with cross-segment dependencies are simulated jointly and stay trial-aligned through aggregation.

### Volume cut-off
Per segment, on any output. A **minimum** in *chance* mode defines geological success as reaching that volume:
effective Pg = Pg × P(V ≥ minimum) and the success-case distribution is truncated. In *renormalise* mode, and for
any **maximum**, out-of-range trials are excluded and replaced by resampling accepted trials; Pg is unchanged.

### Units
Choose a preset (Metric NCS or Field), override the unit of any quantity in the sidebar, and give individual
inputs their own unit with the unit box next to each input. Gas volume factor can be entered as Bg or as the
expansion factor E. Units only affect display and entry; the engine computes in oilfield units.

### Chance of success
Pg(segment) = Π play factors × Π prospect factors × Π segment factors. Factors are entered as probabilities
or through the chance-adequacy matrix. Pc adds the requirement that the prospect's total recoverable oil
equivalent is at least the MEFS in that trial.

### Aggregation
Segments and prospects are combined trial by trial. Each play and prospect factor is drawn **once** per
trial and shared by every segment that depends on it, which models common geological risk exactly.
Remaining segment risks can be made dependent through a Gaussian copula (dependency 0 to 1), and success
volumes can be rank-correlated. The risked mean always equals Σ Pg × success mean; dependencies change the
spread, the chance of at least one discovery, and the number of discoveries.

### Conventions
P90 is the low case and P10 the high case, throughout. The engine computes in oilfield units and converts
only for display and files.

### Scope and limitations
The tool implements published, industry-standard methods for volumes and chance of success. It is not a
reproduction of any commercial product's proprietary algorithms, and results will differ from other tools
where methods or defaults differ. Economic evaluation beyond a minimum economic field size is left to a
dedicated economics tool.
"""


def page_method() -> None:
    st.title("Method")
    st.markdown(METHOD)
    st.caption("Screening-level estimates for exploration decision support. Results depend entirely on input "
               "assumptions and must be reviewed by qualified professionals. Not affiliated with or endorsed by "
               "Equinor or SLB.")
