"""Reusable Streamlit editors bound to the project model.

Widget-state rules used throughout (learned the hard way):

* keyed widgets ignore ``value=`` once their state exists, so every key embeds a
  *generation* counter that is bumped whenever the model changes from outside the
  widget (file load, add/delete, unit-system switch);
* keys are md5-derived, never ``hash()`` (per-process randomised);
* ``st.data_editor`` receives a frozen base frame stored once per key, never a
  frame rebuilt from the model, otherwise added rows are applied twice.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from prospectrisk import distributions as D
from prospectrisk import risk as R
from prospectrisk import units as U
from prospectrisk.volumetrics import AreaDepthTable, VolumetricsError

KIND_OPTIONS = ["constant", "uniform", "triangular", "pert", "normal", "normal_p90p10", "lognormal",
                "lognormal_p90p10", "beta", "discrete", "percentile_table"]
KIND_LABELS = {
    "constant": "Constant", "uniform": "Uniform", "triangular": "Triangular", "pert": "PERT",
    "normal": "Normal (mean, sd)", "normal_p90p10": "Normal (P90, P10)", "lognormal": "Lognormal (mean, sd)",
    "lognormal_p90p10": "Lognormal (P90, P10)", "beta": "Beta", "discrete": "Discrete",
    "percentile_table": "Percentile table",
}


# ---- keys and state --------------------------------------------------------------------

def gen() -> int:
    return st.session_state.setdefault("gen", 0)


def bump() -> None:
    st.session_state["gen"] = gen() + 1


def wkey(*parts: Any) -> str:
    raw = "|".join(str(p) for p in (gen(),) + parts)
    return "w_" + hashlib.md5(raw.encode()).hexdigest()[:20]


def frozen_base(key: str, builder) -> pd.DataFrame:
    bk = key + "_base"
    if bk not in st.session_state:
        st.session_state[bk] = builder()
    return st.session_state[bk]


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def num(label: str, value: float, key: str, help: str | None = None, min_value=None, max_value=None) -> float:
    return st.number_input(label, value=float(value), key=key, format="%.6g", step=None, help=help,
                           min_value=min_value, max_value=max_value)


# ---- distributions --------------------------------------------------------------------

def _spread(d: D.Distribution) -> tuple[float, float, float]:
    lo, mid, hi = d.p90, d.p50, d.p10
    if not hi > lo:
        w = abs(mid) * 0.1 or 1.0
        lo, hi = mid - w, mid + w
    return lo, mid, hi


def _defaults(kind: str, d: D.Distribution) -> dict[str, Any]:
    lo, mid, hi = _spread(d)
    mean = d.mean
    sd = d.std if d.std > 0 else (hi - lo) / (2 * D.Z90)
    if kind == "constant":
        return {"value": mid}
    if kind == "uniform":
        return {"min": lo, "max": hi}
    if kind in ("triangular", "pert"):
        out = {"min": float(d.ppf(0.01)) if d.std > 0 else lo, "mode": mid, "max": float(d.ppf(0.99)) if d.std > 0 else hi}
        if not out["min"] < out["max"]:
            out["min"], out["max"] = lo, hi
        out["mode"] = min(max(out["mode"], out["min"]), out["max"])
        if kind == "pert":
            out["shape"] = 4.0
        return out
    if kind == "normal":
        return {"mean": mean, "sd": sd}
    if kind == "lognormal":
        return {"mean": mean if mean > 0 else 1.0, "sd": sd if sd > 0 else 0.5}
    if kind in ("normal_p90p10", "lognormal_p90p10"):
        p90, p10 = lo, hi
        if kind == "lognormal_p90p10" and p90 <= 0:
            p90, p10 = max(abs(mid) * 0.5, 1e-6), max(abs(mid) * 2.0, 2e-6)
        return {"p90": p90, "p10": p10}
    if kind == "beta":
        return {"alpha": 2.0, "beta": 2.0, "min": float(d.ppf(0.01)) if d.std > 0 else lo,
                "max": float(d.ppf(0.99)) if d.std > 0 else hi}
    if kind == "discrete":
        return {"values": [lo, mid, hi], "probs": [0.3, 0.4, 0.3]}
    if kind == "percentile_table":
        return {"points": [[100, float(d.ppf(0.001)) if d.std > 0 else lo - (hi - lo)], [90, lo], [50, mid],
                           [10, hi], [0, float(d.ppf(0.999)) if d.std > 0 else hi + (hi - lo)]]}
    raise ValueError(kind)


def dist_editor(path: str, label: str, dist: D.Distribution, quantity: str, unit: str,
                help: str | None = None, show_preview: bool = True) -> D.Distribution:
    """Edit a distribution in ``unit``; returns a distribution in internal units."""
    f = U.factor(quantity, unit)
    disp = D.scaled(dist, 1.0 / f) if f != 1.0 else dist
    unit_txt = "" if unit in ("fraction", "-") else f" [{unit}]"

    top = st.columns([3, 2])
    kind = top[1].selectbox("Distribution", KIND_OPTIONS, index=KIND_OPTIONS.index(disp.kind),
                            format_func=KIND_LABELS.get, key=wkey(path, unit, "kind"),
                            label_visibility="collapsed")
    top[0].markdown(f"**{label}**{unit_txt}", help=help)

    base = disp.params() if disp.kind == kind else _defaults(kind, disp)
    k = lambda *p: wkey(path, unit, kind, *p)  # noqa: E731
    new: D.Distribution | None = None
    try:
        if kind in D.PARAM_SCHEMA:
            schema = D.PARAM_SCHEMA[kind]
            cols = st.columns(len(schema))
            vals: dict[str, float] = {}
            for (name, lab), c in zip(schema, cols):
                with c:
                    vals[name] = num(lab, base[name], k(name))
            trunc = _truncation_inputs(k, disp if disp.kind == kind else None) if kind != "constant" else {}
            new = D.REGISTRY[kind](**vals, **trunc) if kind != "constant" else D.Constant(vals["value"])
        elif kind in ("normal_p90p10", "lognormal_p90p10"):
            c1, c2 = st.columns(2)
            with c1:
                p90 = num("P90 (low)", base["p90"], k("p90"))
            with c2:
                p10 = num("P10 (high)", base["p10"], k("p10"))
            trunc = _truncation_inputs(k, None)
            cls = D.Normal if kind == "normal_p90p10" else D.Lognormal
            new = cls.from_p90_p10(p90, p10, **trunc)
        elif kind == "discrete":
            ek = k("table")
            df0 = frozen_base(ek, lambda: pd.DataFrame({"Value": base["values"], "Probability": base["probs"]}, dtype=float))
            ed = st.data_editor(df0, key=ek, num_rows="dynamic", width="stretch", hide_index=True)
            ed = ed.dropna()
            new = D.Discrete(ed["Value"].astype(float).tolist(), ed["Probability"].astype(float).tolist())
        elif kind == "percentile_table":
            ek = k("table")
            df0 = frozen_base(ek, lambda: pd.DataFrame(base["points"], columns=["Exceedance %", "Value"], dtype=float))
            ed = st.data_editor(df0, key=ek, num_rows="dynamic", width="stretch", hide_index=True,
                                column_config={"Exceedance %": st.column_config.NumberColumn(min_value=0, max_value=100)})
            ed = ed.dropna()
            new = D.PercentileTable([[float(a), float(b)] for a, b in zip(ed["Exceedance %"], ed["Value"])])
    except (D.DistributionError, ValueError, KeyError, TypeError) as exc:
        st.error(f"{label}: {exc}")
        return dist

    if new is None:
        return dist
    if show_preview:
        try:
            st.caption(f"P90 {_f(new.p90)}  |  P50 {_f(new.p50)}  |  P10 {_f(new.p10)}  |  mean {_f(new.mean)}{unit_txt}")
        except D.DistributionError as exc:
            st.error(f"{label}: {exc}")
            return dist
    if _close(new.to_dict(), disp.to_dict()):
        return dist
    return D.scaled(new, f) if f != 1.0 else new


def _truncation_inputs(k, current: D.Distribution | None) -> dict[str, float]:
    has_lo = current is not None and current.lower is not None
    has_hi = current is not None and current.upper is not None
    on = st.checkbox("Truncate", value=has_lo or has_hi, key=k("trunc"))
    if not on:
        return {}
    c1, c2 = st.columns(2)
    out = {}
    with c1:
        if st.checkbox("Lower bound", value=has_lo or not has_hi, key=k("tlo_on")):
            out["lower"] = num("Lower", current.lower if has_lo else 0.0, k("tlo"))
    with c2:
        if st.checkbox("Upper bound", value=has_hi, key=k("thi_on")):
            out["upper"] = num("Upper", current.upper if has_hi else 1.0, k("thi"))
    return out


def _f(x: float) -> str:
    a = abs(x)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{x:,.0f}"
    return f"{x:.4g}"


# ---- chance factors -----------------------------------------------------------------------

FACTOR_COLS = ["Factor", "Category", "Mode", "Probability", "Interpretation", "Confidence", "Comment"]


def factor_editor(path: str, factors: list[R.ChanceFactor], matrix) -> list[R.ChanceFactor]:
    key = wkey(path, "factors")

    def build():
        return pd.DataFrame([{"Factor": f.name, "Category": f.category, "Mode": f.mode, "Probability": f.probability,
                              "Interpretation": f.interpretation, "Confidence": f.confidence, "Comment": f.comment}
                             for f in factors], columns=FACTOR_COLS).astype({"Probability": float})

    ed = st.data_editor(
        frozen_base(key, build), key=key, num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn(required=True, width="medium"),
            "Category": st.column_config.SelectboxColumn(options=list(R.CATEGORIES), required=True),
            "Mode": st.column_config.SelectboxColumn(options=["probability", "adequacy"], required=True,
                                                     help="Direct probability, or looked up from the chance-adequacy matrix"),
            "Probability": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                                                         help="Used when Mode = probability"),
            "Interpretation": st.column_config.SelectboxColumn(options=list(R.INTERPRETATIONS),
                                                               help="Used when Mode = adequacy"),
            "Confidence": st.column_config.SelectboxColumn(options=list(R.CONFIDENCES), help="Used when Mode = adequacy"),
            "Comment": st.column_config.TextColumn(width="large"),
        })
    out: list[R.ChanceFactor] = []
    for row in ed.to_dict("records"):
        name = str(row.get("Factor") or "").strip()
        if not name:
            continue
        prob = row.get("Probability")
        prob = 0.5 if prob is None or (isinstance(prob, float) and math.isnan(prob)) else float(prob)
        out.append(R.ChanceFactor(
            name=name, category=row.get("Category") or "Other", mode=row.get("Mode") or "probability",
            probability=prob, interpretation=row.get("Interpretation") or "neutral",
            confidence=row.get("Confidence") or "medium", comment=str(row.get("Comment") or "")))
    if out:
        try:
            vals = [f.value(matrix) for f in out]
            st.caption("Chance per factor: " + ", ".join(f"{f.name} {v:.2f}" for f, v in zip(out, vals))
                       + f"  |  product {np.prod(vals):.3f}")
        except R.RiskError as exc:
            st.error(str(exc))
    else:
        st.caption("No factors at this level (chance 1.0).")
    return out


def adequacy_editor(matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    key = wkey("adequacy")
    df = frozen_base(key, lambda: pd.DataFrame(matrix).T.loc[list(R.INTERPRETATIONS), list(R.CONFIDENCES)])
    ed = st.data_editor(df, key=key, width="stretch",
                        column_config={c: st.column_config.NumberColumn(f"{c.title()} confidence", min_value=0.0,
                                                                        max_value=1.0, step=0.05, format="%.2f")
                                       for c in R.CONFIDENCES})
    out = {i: {c: float(ed.loc[i, c]) if pd.notna(ed.loc[i, c]) else matrix[i][c] for c in R.CONFIDENCES}
           for i in R.INTERPRETATIONS}
    return out


# ---- tables -----------------------------------------------------------------------------------

def area_depth_editor(path: str, table: AreaDepthTable | None, du: str, au: str) -> AreaDepthTable | None:
    key = wkey(path, du, au, "adt")

    def build():
        if table is None:
            return pd.DataFrame({f"Depth [{du}]": [2500.0, 2550.0, 2600.0, 2650.0],
                                 f"Area [{au}]": [0.0, 3.0, 8.0, 14.0]})
        return pd.DataFrame({f"Depth [{du}]": U.from_internal(np.array(table.depth), "length", du),
                             f"Area [{au}]": U.from_internal(np.array(table.area), "area", au)})

    ed = st.data_editor(frozen_base(key, build), key=key, num_rows="dynamic", width="stretch", hide_index=True).dropna()
    try:
        ed = ed.sort_values(ed.columns[0])
        new = AreaDepthTable(list(U.to_internal(ed.iloc[:, 0].to_numpy(float), "length", du)),
                             list(U.to_internal(ed.iloc[:, 1].to_numpy(float), "area", au)))
    except VolumetricsError as exc:
        st.error(str(exc))
        return table
    if table is not None and _close(new.to_dict(), table.to_dict()):
        return table
    return new


def pair_editor(path: str, names: list[str], matrix: np.ndarray | None, label: str, lo: float = 0.0,
                help: str | None = None) -> np.ndarray | None:
    k = len(names)
    if k < 2:
        st.caption("Needs at least two items.")
        return None
    m = np.eye(k) if matrix is None or np.shape(matrix) != (k, k) else np.asarray(matrix, float)
    key = wkey(path, "pairs", *names)
    rows = [{"A": names[i], "B": names[j], label: float(m[i, j])} for i in range(k) for j in range(i + 1, k)]
    ed = st.data_editor(frozen_base(key, lambda: pd.DataFrame(rows)), key=key, width="stretch", hide_index=True,
                        disabled=["A", "B"],
                        column_config={label: st.column_config.NumberColumn(min_value=lo, max_value=1.0, step=0.05,
                                                                            format="%.2f", help=help)})
    out = np.eye(k)
    idx = {n: i for i, n in enumerate(names)}
    for r in ed.to_dict("records"):
        v = r.get(label)
        v = 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        out[idx[r["A"]], idx[r["B"]]] = out[idx[r["B"]], idx[r["A"]]] = v
    return None if np.allclose(out, np.eye(k)) else out


def correlation_editor(path: str, keys: list[str], labels: dict[str, str],
                       pairs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    key = wkey(path, "corr")
    inv = {labels[k]: k for k in keys}
    opts = [labels[k] for k in keys]

    def build():
        return pd.DataFrame([{"Variable A": labels.get(a, a), "Variable B": labels.get(b, b), "Rank correlation": r}
                             for a, b, r in pairs],
                            columns=["Variable A", "Variable B", "Rank correlation"]).astype({"Rank correlation": float})

    ed = st.data_editor(frozen_base(key, build), key=key, num_rows="dynamic", width="stretch", hide_index=True,
                        column_config={
                            "Variable A": st.column_config.SelectboxColumn(options=opts),
                            "Variable B": st.column_config.SelectboxColumn(options=opts),
                            "Rank correlation": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05,
                                                                              format="%.2f")})
    out = []
    for r in ed.to_dict("records"):
        a, b, rho = inv.get(r.get("Variable A")), inv.get(r.get("Variable B")), r.get("Rank correlation")
        if a and b and a != b and rho is not None and not (isinstance(rho, float) and math.isnan(rho)):
            out.append((a, b, float(rho)))
    return out


# ---- units ------------------------------------------------------------------------------------

def unit_customiser(settings) -> bool:
    """Per-quantity unit overrides. Returns True when something changed."""
    changed = False
    token = settings.unit_system
    for q, label in U.QUANTITY_LABELS.items():
        opts = U.units_for(q)
        cur = settings.unit(q)
        new = st.selectbox(label, opts, index=opts.index(cur), key=wkey("unit_q", q, token))
        if new != cur:
            if new == U.UNIT_SYSTEMS[settings.unit_system][q]:
                settings.unit_overrides.pop(q, None)
            else:
                settings.unit_overrides[q] = new
            changed = True
    return changed


def input_unit_picker(path: str, settings, variable: str, quantity: str) -> bool:
    """Compact unit selector for one input (applies to that input in every segment)."""
    opts = U.units_for(quantity)
    if len(opts) < 2:
        return False
    cur = settings.input_unit(variable)
    new = st.selectbox("Unit", opts, index=opts.index(cur), key=wkey(path, "iu", cur),
                       label_visibility="collapsed")
    if new == cur:
        return False
    if new == settings.unit(quantity):
        settings.input_units.pop(variable, None)
    else:
        settings.input_units[variable] = new
    return True


# ---- input dependencies across segments -------------------------------------------------------------

DEP_KINDS = {"correlation": "Rank correlation", "link": "Link (same percentile)"}


def dependency_editor(path: str, segments: list[str], inputs: dict[str, list[str]], labels: dict[str, str],
                      deps: list[dict]) -> list[dict]:
    """Editable list of input dependencies; ``inputs`` maps segment -> uncertain input keys."""
    key = wkey(path, "deps", *segments)
    inv = {v: k for k, v in labels.items()}
    all_labels = sorted({labels[i] for ins in inputs.values() for i in ins})
    kind_inv = {v: k for k, v in DEP_KINDS.items()}

    def build():
        return pd.DataFrame([{"Segment A": d.get("seg_a"), "Input A": labels.get(d.get("var_a"), d.get("var_a")),
                              "Segment B": d.get("seg_b"), "Input B": labels.get(d.get("var_b"), d.get("var_b")),
                              "Type": DEP_KINDS.get(d.get("kind", "correlation")), "Rank correlation": d.get("rho", 0.0)}
                             for d in deps],
                            columns=["Segment A", "Input A", "Segment B", "Input B", "Type", "Rank correlation"]
                            ).astype({"Rank correlation": float})

    ed = st.data_editor(frozen_base(key, build), key=key, num_rows="dynamic", width="stretch", hide_index=True,
                        column_config={
                            "Segment A": st.column_config.SelectboxColumn(options=segments),
                            "Input A": st.column_config.SelectboxColumn(options=all_labels),
                            "Segment B": st.column_config.SelectboxColumn(options=segments),
                            "Input B": st.column_config.SelectboxColumn(options=all_labels),
                            "Type": st.column_config.SelectboxColumn(options=list(DEP_KINDS.values())),
                            "Rank correlation": st.column_config.NumberColumn(
                                min_value=-1.0, max_value=1.0, step=0.05, format="%.2f",
                                help="Used for rank correlation; ignored for links"),
                        })
    out = []
    for r in ed.to_dict("records"):
        sa, sb = r.get("Segment A"), r.get("Segment B")
        va, vb = inv.get(r.get("Input A")), inv.get(r.get("Input B"))
        kind = kind_inv.get(r.get("Type") or "", "correlation")
        rho = r.get("Rank correlation")
        rho = 0.0 if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
        if not (sa and sb and va and vb):
            continue
        problems = [f"{s_}: {lab} is currently not an uncertain, distribution-defined input"
                    for s_, v_, lab in ((sa, va, r.get("Input A")), (sb, vb, r.get("Input B")))
                    if v_ not in inputs.get(s_, [])]
        if problems:
            st.warning("; ".join(problems) + " — this row is kept but has no effect until that changes.")
        if (sa, va) == (sb, vb):
            st.warning(f"{sa}: {r.get('Input A')} cannot depend on itself — row ignored.")
            continue
        out.append({"seg_a": sa, "var_a": va, "seg_b": sb, "var_b": vb, "kind": kind, "rho": rho})
    return out
