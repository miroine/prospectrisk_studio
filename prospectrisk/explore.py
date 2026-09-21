"""Trial-level exploration: input/result tables, filters and rank correlations.

All segments of a prospect are simulated jointly, so trial *i* of every segment is
the same realisation. The trial table puts every input and result of every
segment side by side (display units), plus the prospect sum, so any input can be
plotted against any result and trials can be filtered in or out.

Values are success-case trial values before chance of success is applied; a
column flags whether each trial passes a segment's volume cut-off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .project import Project, ProspectRun
from .units import from_internal
from .volumetrics import KEY_OUTPUT, OUTPUTS, PHASE_LABELS, PHASES, VARIABLES, input_label, input_quantity

RESULT_KEYS = ["grv_total", "stoiip", "giip_free", "inplace_oe", "rec_oil", "rec_gas", "rec_liquids", "rec_oe"]


@dataclass
class ColumnInfo:
    segment: str | None
    key: str
    kind: str          # "input" | "result" | "phase" | "flag" | "total"
    unit: str

    @property
    def numeric(self) -> bool:
        return self.kind not in ("phase", "flag")


def trial_table(project: Project, prun: ProspectRun) -> tuple[pd.DataFrame, dict[str, ColumnInfo]]:
    st = project.settings
    segs = list(prun.segments)
    multi = len(segs) > 1
    cols: dict[str, np.ndarray] = {}
    meta: dict[str, ColumnInfo] = {}

    def name(seg: str | None, label: str, unit: str) -> str:
        u = "" if unit in ("fraction", "-", "") else f" [{unit}]"
        return f"{seg} | {label}{u}" if (seg and multi) else f"{label}{u}"

    n = next(iter(prun.segments.values())).result.n
    total = np.zeros(n)
    for seg in segs:
        res = prun.segments[seg].result
        for k, arr in res.inputs.items():
            q = input_quantity(k)
            unit = st.input_unit(k) if k in VARIABLES else "-"
            c = name(seg, input_label(k), unit)
            cols[c] = from_internal(np.asarray(arr, float), q, unit)
            meta[c] = ColumnInfo(seg, k, "input", unit)
        for k in RESULT_KEYS:
            arr = np.asarray(res.outputs[k], float)
            if not np.any(arr):
                continue
            label, q = OUTPUTS[k]
            unit = st.unit(q)
            c = name(seg, label, unit)
            cols[c] = from_internal(arr, q, unit)
            meta[c] = ColumnInfo(seg, k, "result", unit)
        if "contact" in res.outputs:
            unit = st.unit("length")
            c = name(seg, "Hydrocarbon contact", unit)
            cols[c] = from_internal(np.asarray(res.outputs["contact"], float), "length", unit)
            meta[c] = ColumnInfo(seg, "contact", "result", unit)
        if len({int(p) for p in np.unique(res.phase)}) > 1 or multi:
            c = name(seg, "Fluid phase", "")
            cols[c] = np.array([PHASE_LABELS[PHASES[int(i)]] for i in res.phase], object)
            meta[c] = ColumnInfo(seg, "phase", "phase", "")
        t = project.prospect(prun.prospect).segment(seg).truncation
        if t.active:
            v = np.asarray(res.outputs[t.output], float)
            ok = np.ones(n, bool)
            if t.minimum is not None:
                ok &= v >= t.minimum
            if t.maximum is not None:
                ok &= v <= t.maximum
            c = name(seg, "Passes volume cut-off", "")
            cols[c] = ok
            meta[c] = ColumnInfo(seg, "cutoff", "flag", "")
        total = total + np.asarray(res.outputs[KEY_OUTPUT], float)
    if multi:
        unit = st.unit("oe")
        c = f"Prospect | Recoverable oil equivalent, all segments [{unit}]"
        cols[c] = from_internal(total, "oe", unit)
        meta[c] = ColumnInfo(None, KEY_OUTPUT, "total", unit)
    df = pd.DataFrame(cols)
    df.insert(0, "Trial", np.arange(1, n + 1))
    return df, meta


# ---- filters -------------------------------------------------------------------------------

@dataclass
class TrialFilter:
    column: str
    action: str = "keep"               # "keep" | "exclude"
    low: float | None = None
    high: float | None = None
    values: list[Any] | None = None    # categorical / flag columns

    def matches(self, df: pd.DataFrame) -> np.ndarray:
        if self.column not in df.columns:
            raise KeyError(f"Unknown column '{self.column}'")
        s = df[self.column]
        if self.values is not None:
            return s.isin(self.values).to_numpy()
        v = s.to_numpy(float)
        m = np.ones(v.size, bool)
        if self.low is not None:
            m &= v >= self.low
        if self.high is not None:
            m &= v <= self.high
        return m

    def describe(self) -> str:
        verb = "Keep" if self.action == "keep" else "Exclude"
        if self.values is not None:
            return f"{verb} {self.column} in {{{', '.join(map(str, self.values))}}}"
        rng = []
        if self.low is not None:
            rng.append(f"≥ {self.low:.4g}")
        if self.high is not None:
            rng.append(f"≤ {self.high:.4g}")
        return f"{verb} {self.column} " + " and ".join(rng or ["(any)"])


def apply_filters(df: pd.DataFrame, filters: list[TrialFilter]) -> np.ndarray:
    """Trials that satisfy every *keep* filter and no *exclude* filter."""
    mask = np.ones(len(df), bool)
    for f in filters:
        if f.action not in ("keep", "exclude"):
            raise ValueError(f"Unknown filter action '{f.action}'")
        m = f.matches(df)
        mask &= m if f.action == "keep" else ~m
    return mask


def stats_row(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"Trials": 0, "Mean": np.nan, "P90": np.nan, "P50": np.nan, "P10": np.nan}
    q = np.quantile(x, [0.1, 0.5, 0.9])
    return {"Trials": int(x.size), "Mean": float(x.mean()), "P90": q[0], "P50": q[1], "P10": q[2]}


def compare(df: pd.DataFrame, column: str, mask: np.ndarray) -> pd.DataFrame:
    v = df[column].to_numpy(float)
    rows = [{"Set": "All trials", **stats_row(v)}, {"Set": "Filtered in", **stats_row(v[mask])},
            {"Set": "Filtered out", **stats_row(v[~mask])}]
    return pd.DataFrame(rows)


def spearman_matrix(df: pd.DataFrame, columns: list[str], mask: np.ndarray | None = None) -> pd.DataFrame:
    sub = df[columns] if mask is None else df.loc[mask, columns]
    sub = sub.astype(float)
    keep = [c for c in columns if sub[c].nunique() > 1]
    out = pd.DataFrame(np.nan, index=columns, columns=columns)
    if len(keep) >= 2 and len(sub) >= 3:
        r = stats.spearmanr(sub[keep].to_numpy(), nan_policy="omit").statistic
        r = np.atleast_2d(r)
        out.loc[keep, keep] = r
    elif len(keep) == 1:
        out.loc[keep, keep] = 1.0
    return out


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def binned_trend(x: np.ndarray, y: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and P90/P10 band of y within equal-count bins of x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < bins * 3:
        bins = max(x.size // 3, 1)
    order = np.argsort(x)
    chunks = np.array_split(order, bins)
    xm = np.array([np.median(x[c]) for c in chunks if c.size])
    ym = np.array([np.median(y[c]) for c in chunks if c.size])
    lo = np.array([np.quantile(y[c], 0.1) for c in chunks if c.size])
    hi = np.array([np.quantile(y[c], 0.9) for c in chunks if c.size])
    return xm, ym, lo, hi
