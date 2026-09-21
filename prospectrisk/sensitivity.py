"""Sensitivity analysis.

* **Tornado** — one-at-a-time swing: every stochastic input moved to its low
  and high percentile with the others held at the base case (P50 or mean),
  for a fixed fluid phase.
* **Rank correlation / contribution to variance** — Spearman correlation of
  each sampled input with the output across all trials, and the normalised
  squared rank correlation as an approximate contribution to variance.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .volumetrics import SegmentVolumetrics, constant_inputs, deterministic, input_bounds, input_label


def rank_sensitivity(inputs: dict[str, np.ndarray], output: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    y = np.asarray(output, float)
    if mask is not None:
        y = y[mask]
    rows = []
    for k, x in inputs.items():
        x = np.asarray(x, float)
        if mask is not None:
            x = x[mask]
        if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        rho = float(stats.spearmanr(x, y).statistic)
        if np.isfinite(rho):
            rows.append({"Variable": input_label(k), "key": k, "Rank correlation": rho})
    total = sum(r["Rank correlation"] ** 2 for r in rows) or 1.0
    for r in rows:
        r["Contribution to variance"] = np.sign(r["Rank correlation"]) * r["Rank correlation"] ** 2 / total
    rows.sort(key=lambda r: -abs(r["Rank correlation"]))
    return rows


def tornado(seg: SegmentVolumetrics, output_key: str, phase: str, base: str = "p50",
            low_pct: float = 90.0, high_pct: float = 10.0, scf_per_boe: float | None = None) -> tuple[float, list[dict]]:
    kw = {} if scf_per_boe is None else {"scf_per_boe": scf_per_boe}
    active = seg.sampled_inputs()
    base_vals = constant_inputs(seg, base)
    base_out = deterministic(seg, base_vals, phase, **kw)[output_key]
    rows = []
    for k, d in active.items():
        if d.is_constant:
            continue
        lo_in, hi_in = d.exceedance(low_pct), d.exceedance(high_pct)
        lo_b, hi_b = input_bounds(k)
        lo_in, hi_in = float(np.clip(lo_in, lo_b, hi_b)), float(np.clip(hi_in, lo_b, hi_b))
        out_lo = deterministic(seg, {**base_vals, k: lo_in}, phase, **kw)[output_key]
        out_hi = deterministic(seg, {**base_vals, k: hi_in}, phase, **kw)[output_key]
        rows.append({"Variable": input_label(k), "key": k, "Input low": lo_in, "Input high": hi_in,
                     "Output at low input": out_lo, "Output at high input": out_hi,
                     "Swing": abs(out_hi - out_lo)})
    rows.sort(key=lambda r: -r["Swing"])
    return base_out, rows
