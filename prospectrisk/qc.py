"""Consistency and quality-control checks.

All thresholds are screening heuristics exposed in :class:`QCThresholds` so a
team can replace them with its own calibration. They flag things to look at;
they do not reject a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .project import Project, ProspectRun, SegmentRun
from .risk import exceedance_stats, swanson_mean
from .volumetrics import KEY_OUTPUT, VARIABLES


@dataclass
class QCThresholds:
    min_p10_p90_ratio: float = 3.0        # narrower recoverable range suggests over-confidence
    max_p10_p90_ratio: float = 100.0      # wider range suggests poorly constrained inputs
    max_porosity: float = 0.40
    max_hc_saturation: float = 0.95
    min_bo: float = 1.0
    max_rf_oil: float = 0.70
    max_rf_gas: float = 0.90
    max_factor_for_frontier: float = 0.95
    swanson_tolerance: float = 0.15        # relative difference simulated mean vs Swanson's mean
    min_pg_warning: float = 0.05
    max_pg_warning: float = 0.60


@dataclass
class Finding:
    level: str      # "error" | "warning" | "info"
    scope: str
    message: str

    def as_row(self) -> dict:
        return asdict(self)


def check_segment(run: SegmentRun, seg_inputs, t: QCThresholds) -> list[Finding]:
    f: list[Finding] = []
    scope = run.label
    for w in run.result.diagnostics.get("warnings", []):
        f.append(Finding("warning", scope, w))

    act = seg_inputs.volumetrics.active_variables()
    checks = [("porosity", t.max_porosity, "above"), ("hc_saturation", t.max_hc_saturation, "above"),
              ("gas_saturation", t.max_hc_saturation, "above"), ("rf_oil", t.max_rf_oil, "above"),
              ("rf_gas", t.max_rf_gas, "above"), ("bo", t.min_bo, "below")]
    for key, lim, side in checks:
        if key not in act:
            continue
        d = act[key]
        val = d.p10 if side == "above" else d.p90
        if (side == "above" and val > lim) or (side == "below" and val < lim):
            f.append(Finding("warning", scope, f"{VARIABLES[key][0]}: "
                             f"{'P10' if side == 'above' else 'P90'} = {val:.3g} is {side} the typical limit {lim:.3g}"))

    v = run.result.outputs[KEY_OUTPUT]
    s = exceedance_stats(v[v > 0]) if np.any(v > 0) else None
    if s is None:
        f.append(Finding("error", scope, "Recoverable volume is zero in every trial — check inputs and phase settings"))
    else:
        ratio = s["P10"] / s["P90"] if s["P90"] > 0 else np.inf
        if ratio < t.min_p10_p90_ratio:
            f.append(Finding("warning", scope, f"Recoverable P10/P90 ratio {ratio:.1f} is below {t.min_p10_p90_ratio:g}: "
                                               "the volume range may be over-confident for an undrilled prospect"))
        elif ratio > t.max_p10_p90_ratio:
            f.append(Finding("warning", scope, f"Recoverable P10/P90 ratio {ratio:.0f} exceeds {t.max_p10_p90_ratio:g}: "
                                               "inputs may be poorly constrained"))
        else:
            f.append(Finding("info", scope, f"Recoverable P10/P90 ratio {ratio:.1f}"))
        sw = swanson_mean(s["P90"], s["P50"], s["P10"])
        rel = abs(s["Mean"] - sw) / s["Mean"] if s["Mean"] > 0 else 0.0
        if rel > t.swanson_tolerance:
            f.append(Finding("info", scope, f"Simulated mean differs from Swanson's mean by {100 * rel:.0f} % — "
                                            "distribution is strongly skewed or multi-modal (e.g. phase uncertainty)"))
        zero = float(np.mean(v <= 0))
        if zero > 0.01:
            f.append(Finding("warning", scope, f"{100 * zero:.1f} % of success-case trials have zero recoverable volume; "
                                               "a geological success should normally imply a non-zero accumulation"))

    ch = run.chance
    if ch.pg < t.min_pg_warning:
        f.append(Finding("warning", scope, f"Pg {ch.pg:.3f} is below {t.min_pg_warning:g} — confirm the prospect is mature enough to rank"))
    if ch.pg > t.max_pg_warning:
        f.append(Finding("warning", scope, f"Pg {ch.pg:.2f} exceeds {t.max_pg_warning:g} — unusually high for an undrilled prospect; "
                                           "check for double-counted de-risking"))
    for sc, name, val in ch.critical:
        if val > t.max_factor_for_frontier:
            f.append(Finding("info", scope, f"{sc.title()} factor '{name}' = {val:.2f} is effectively certain"))
    if ch.critical:
        sc, name, val = ch.critical[0]
        f.append(Finding("info", scope, f"Key risk: {name} ({sc}) = {val:.2f}"))
    return f


def check_prospect(project: Project, prun: ProspectRun, t: QCThresholds) -> list[Finding]:
    pr = project.prospect(prun.prospect)
    f: list[Finding] = []
    for seg in pr.segments:
        f.extend(check_segment(prun.segments[seg.name], seg, t))
    for w in prun.aggregation.diagnostics.get("warnings", []):
        f.append(Finding("warning", pr.name, w))
    a = prun.aggregation
    for r in a.item_table():
        pe = r["Pg (effective)"]
        if pe > 0 and abs(r["Pg (simulated)"] - pe) > max(0.02, 3 * np.sqrt(pe * (1 - pe) / a.n)):
            f.append(Finding("warning", pr.name, f"{r['Item']}: simulated Pg {r['Pg (simulated)']:.3f} differs from "
                                                 f"expected {pe:.3f} — increase the number of trials"))
    for seg in pr.segments:
        t = seg.truncation
        if not t.active:
            continue
        j = a.item_index(pr.name, seg.name)
        if t.minimum is not None and t.min_mode == "chance":
            p_ok = a.p_volume_ok(j)
            level = "warning" if p_ok < 0.5 else "info"
            f.append(Finding(level, f"{pr.name} / {seg.name}",
                             f"Minimum volume removes {100 * (1 - p_ok):.1f} % of success-case trials; "
                             f"Pg {prun.segments[seg.name].chance.pg:.3f} becomes {prun.segments[seg.name].chance.pg * p_ok:.3f}"))
        rej = a.diagnostics.get("rejected_fraction", {}).get(seg.name)
        if rej:
            f.append(Finding("warning" if rej > 0.5 else "info", f"{pr.name} / {seg.name}",
                             f"Volume truncation rejected and resampled {100 * rej:.1f} % of trials"))
    if pr.mefs > 0 and prun.pg > 0:
        pc = float(prun.aggregation.group_commercial(pr.name).mean())
        if pc < 0.5 * prun.pg:
            f.append(Finding("warning", pr.name, f"Less than half of geological successes exceed the MEFS "
                                                 f"(Pc {pc:.3f} vs Pg {prun.pg:.3f})"))
    if not pr.mefs:
        f.append(Finding("info", pr.name, "No minimum economic field size set — commercial chance equals Pg"))
    return f
