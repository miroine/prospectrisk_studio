"""File input/output: project YAML, Excel workbook, FieldVista bridge YAML, PDF report."""
from __future__ import annotations

import io
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .project import Project, ProspectRun
from .risk import exceedance_stats
from .units import from_internal
from .volumetrics import KEY_OUTPUT, OUTPUTS, PHASES, PHASE_LABELS, VARIABLES, input_label, input_quantity


# ---- project files -------------------------------------------------------------------

def project_to_yaml(project: Project) -> str:
    return yaml.safe_dump(project.to_dict(), sort_keys=False, allow_unicode=True)


def project_from_yaml(text: str | bytes) -> Project:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"File is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or "prospects" not in data:
        raise ValueError("File does not look like a project file (no 'prospects' section)")
    return Project.from_dict(data)


# ---- tables in display units ------------------------------------------------------------

def output_stats_table(project: Project, prun: ProspectRun) -> pd.DataFrame:
    """Success-case output statistics per segment, after any volume truncation."""
    st = project.settings
    rows = []
    for seg_name in prun.segments:
        for key, (label, qty) in OUTPUTS.items():
            if qty == "none":
                continue
            unit = st.unit(qty)
            s = exceedance_stats(prun.segment_success_case(seg_name, key))
            if not np.isfinite(s["Max"]) or s["Max"] == 0:
                continue
            rows.append({"Segment": seg_name, "Output": label, "Unit": unit,
                         **{k: from_internal(v, qty, unit) for k, v in s.items()}})
    return pd.DataFrame(rows)


def input_stats_table(project: Project, prun: ProspectRun) -> pd.DataFrame:
    st = project.settings
    pr = project.prospect(prun.prospect)
    rows = []
    for seg in pr.segments:
        vol = seg.volumetrics
        sampled = prun.segments[seg.name].result.inputs
        defs = vol.sampled_inputs()
        exprs = vol.active_expressions()
        for key in list(defs) + [k for k in exprs]:
            qty = input_quantity(key)
            unit = st.input_unit(key) if key in VARIABLES else "-"
            if key in exprs:
                how = f"Formula: {exprs[key].expr}"
            else:
                how = defs[key].label
            x = sampled.get(key)
            if x is None:
                continue
            q = np.quantile(x, [0.1, 0.5, 0.9])
            rows.append({"Segment": seg.name, "Input": input_label(key), "Unit": unit, "Definition": how,
                         "P90": from_internal(q[0], qty, unit), "P50": from_internal(q[1], qty, unit),
                         "P10": from_internal(q[2], qty, unit), "Mean": from_internal(float(x.mean()), qty, unit)})
    return pd.DataFrame(rows)


def chance_table(project: Project, prun: ProspectRun) -> pd.DataFrame:
    pr = project.prospect(prun.prospect)
    play = project.play(pr.play)
    m = project.adequacy_matrix
    rows = []
    for scope, factors, owner in (("Play", play.factors, play.name), ("Prospect", pr.factors, pr.name)):
        for f in factors:
            rows.append({"Scope": scope, "Owner": owner, "Category": f.category, "Factor": f.name,
                         "Mode": f.mode, "Chance": f.value(m)})
    for seg in pr.segments:
        for f in seg.factors:
            rows.append({"Scope": "Segment", "Owner": seg.name, "Category": f.category, "Factor": f.name,
                         "Mode": f.mode, "Chance": f.value(m)})
    return pd.DataFrame(rows)


def prospect_summary(project: Project, prun: ProspectRun) -> dict[str, Any]:
    unit = project.settings.unit("oe")
    a = prun.aggregation
    pr = project.prospect(prun.prospect)
    tot = prun.success_total()
    s = exceedance_stats(tot) if tot.size else exceedance_stats(np.array([0.0]))
    pc = float(a.group_commercial(pr.name).mean())
    return {
        "Prospect": pr.name, "Play": pr.play, "Status": pr.status, "Pg": prun.pg, "Pc": pc,
        f"MEFS [{unit}]": from_internal(pr.mefs, "oe", unit),
        f"Success mean [{unit}]": from_internal(s["Mean"], "oe", unit),
        f"Success P90 [{unit}]": from_internal(s["P90"], "oe", unit),
        f"Success P50 [{unit}]": from_internal(s["P50"], "oe", unit),
        f"Success P10 [{unit}]": from_internal(s["P10"], "oe", unit),
        f"Risked mean [{unit}]": from_internal(float(a.group_total(pr.name, KEY_OUTPUT).mean()), "oe", unit),
    }


def _sheet(name: str) -> str:
    return re.sub(r"[\[\]\*\?/\\:]", "_", str(name))[:31]


def results_to_excel(project: Project, runs: dict[str, ProspectRun], max_trial_rows: int = 5000) -> bytes:
    buf = io.BytesIO()
    st = project.settings
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([prospect_summary(project, r) for r in runs.values()]).to_excel(xw, sheet_name=_sheet("Summary"), index=False)
        for name, r in runs.items():
            tag = name[:20]
            input_stats_table(project, r).to_excel(xw, sheet_name=_sheet(f"{tag} inputs"), index=False)
            output_stats_table(project, r).to_excel(xw, sheet_name=_sheet(f"{tag} outputs"), index=False)
            chance_table(project, r).to_excel(xw, sheet_name=_sheet(f"{tag} chance"), index=False)
            frames = []
            for seg_name, srun in r.segments.items():
                n = min(srun.result.n, max_trial_rows)
                df = pd.DataFrame({"Segment": seg_name, "Trial": np.arange(1, n + 1),
                                   "Phase": [PHASE_LABELS[PHASES[i]] for i in srun.result.phase[:n]]})
                for k, arr in srun.result.inputs.items():
                    qty = input_quantity(k)
                    unit = st.input_unit(k) if k in VARIABLES else "-"
                    df[f"{input_label(k)} [{unit}]"] = from_internal(arr[:n], qty, unit)
                for k in ("stoiip", "giip_free", "rec_oil", "rec_gas", "rec_condensate", KEY_OUTPUT):
                    label, qty = OUTPUTS[k]
                    unit = st.unit(qty)
                    df[f"{label} [{unit}]"] = from_internal(srun.result.outputs[k][:n], qty, unit)
                frames.append(df)
            pd.concat(frames).to_excel(xw, sheet_name=_sheet(f"{tag} trials"), index=False)
    return buf.getvalue()


# ---- FieldVista bridge ------------------------------------------------------------------

def fieldvista_bridge_yaml(project: Project, runs: dict[str, ProspectRun]) -> str:
    """Success-case recoverable volumes and chances for use as decision-tree leaf inputs.

    Both percentile conventions are written explicitly: ``low/mid/high`` (exploration
    P90/P50/P10) and ``percentile_10/50/90`` (statistical, 10th percentile = low), so the
    receiving application cannot confuse them.
    """
    st = project.settings
    out: dict[str, Any] = {"source": "ProspectRisk Studio", "generated": date.today().isoformat(),
                           "project": project.name, "unit_system": st.unit_system, "prospects": []}
    for name, r in runs.items():
        a = r.aggregation
        succ = a.group_success(name)
        entry: dict[str, Any] = {"name": name, "pg": round(r.pg, 5),
                                 "pc": round(float(a.group_commercial(name).mean()), 5), "volumes": {}}
        for key in ("rec_liquids", "rec_gas", KEY_OUTPUT):
            label, qty = OUTPUTS[key]
            unit = st.unit(qty)
            tot = a.group_total(name, key)[succ]
            if tot.size == 0 or tot.max() == 0:
                continue
            s = exceedance_stats(tot)
            conv = {k: round(float(from_internal(s[k], qty, unit)), 6) for k in ("P90", "P50", "P10", "Mean")}
            entry["volumes"][key] = {
                "label": label, "unit": unit, "basis": "success case",
                "low": conv["P90"], "mid": conv["P50"], "high": conv["P10"], "mean": conv["Mean"],
                "percentile_10": conv["P90"], "percentile_50": conv["P50"], "percentile_90": conv["P10"],
            }
        out["prospects"].append(entry)
    return yaml.safe_dump(out, sort_keys=False, allow_unicode=True)


# ---- PDF report ---------------------------------------------------------------------------

def pdf_report(project: Project, runs: dict[str, ProspectRun], portfolio_summary: dict | None = None) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, red = colors.HexColor("#00243D"), colors.HexColor("#EB0037")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=navy)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=navy)
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=7.5, textColor=colors.grey)

    def table(df: pd.DataFrame, fmt="{:,.4g}") -> Table:
        data = [list(df.columns)] + [[fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in row] for row in df.itertuples(index=False)]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F9")]),
        ]))
        return t

    def fig_img(fig, width=170 * mm):
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        b.seek(0)
        w, h = fig.get_size_inches()
        return Image(b, width=width, height=width * h / w)

    unit = project.settings.unit("oe")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=project.name)
    el = [Paragraph(project.name, h1),
          Paragraph(f"Prospect volumes and chance of success — {date.today():%d %B %Y}", body),
          Paragraph(f"{project.settings.n_trials:,} trials, {project.settings.sampling.upper()} sampling, "
                    f"seed {project.settings.seed}, {project.settings.boe_convention}. "
                    "Percentiles use the exploration convention: P90 low, P10 high.", body),
          Spacer(1, 6)]
    summ = pd.DataFrame([prospect_summary(project, r) for r in runs.values()])
    el += [Paragraph("Summary", h2), table(summ)]
    if portfolio_summary:
        ps = pd.DataFrame([{"Metric": k, "Value": v if "success)" in k or "discover" in k
                            else from_internal(v, "oe", unit)} for k, v in portfolio_summary.items()])
        el += [Spacer(1, 8), Paragraph(f"Portfolio (volumes in {unit})", h2), table(ps)]

    for name, r in runs.items():
        el += [PageBreak(), Paragraph(f"Prospect {name}", h1)]
        tot = from_internal(r.success_total(), "oe", unit)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        if tot.size:
            ax[0].hist(tot, bins=50, color="#007079", alpha=0.85)
            v = np.sort(tot)
            exc = 1 - np.arange(v.size) / v.size
            ax[1].plot(v, exc, color="#00243D", label="Success case")
            ax[1].plot(v, exc * r.pg, color="#EB0037", label=f"Risked (Pg {r.pg:.2f})")
            ax[1].legend(fontsize=8)
        ax[0].set_title("Success-case recoverable", fontsize=9)
        ax[0].set_xlabel(unit, fontsize=8)
        ax[1].set_title("Expectation curves", fontsize=9)
        ax[1].set_xlabel(unit, fontsize=8)
        ax[1].set_ylabel("Probability ≥ volume", fontsize=8)
        for a_ in ax:
            a_.tick_params(labelsize=7)
            a_.grid(alpha=0.3)
        el += [fig_img(fig), Paragraph("Chance of success", h2), table(chance_table(project, r))]
        el += [Paragraph("Inputs", h2), table(input_stats_table(project, r))]
        out = output_stats_table(project, r)
        out = out[out["Output"].isin([OUTPUTS[k][0] for k in ("stoiip", "giip_free", "rec_liquids", "rec_gas", KEY_OUTPUT)])]
        el += [Paragraph("Success-case outputs", h2), table(out.drop(columns=["Min", "Max"]))]

    el += [Spacer(1, 12), Paragraph(
        "Screening-level estimates for engineering and exploration decision support. Results depend entirely on "
        "input assumptions and must be reviewed by qualified professionals. Not affiliated with or endorsed by "
        "Equinor or SLB.", small)]
    doc.build(el)
    return buf.getvalue()
