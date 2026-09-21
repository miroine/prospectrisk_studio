"""Explore: plot any input against any result, correlation matrices, and filter trials in or out."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from prospectrisk import explore as E
from ui import charts as C
from ui import state
from ui.editors import frozen_base, selected_box, wkey

ACTIONS = {"Keep": "keep", "Exclude": "exclude"}


def _table(p, name: str):
    """Trial table cached per run and unit settings."""
    run = st.session_state.runs[name]
    key = (name, st.session_state.run_fp.get(name), p.settings.unit_token, id(run))
    cache = st.session_state.setdefault("_trial_tables", {})
    if key not in cache:
        cache.clear()
        cache[key] = E.trial_table(p, run)
    return cache[key]


def _short(col: str) -> str:
    return col.split(" [")[0].replace(" | ", ": ")


def page_explore() -> None:
    p = state.project()
    st.title("Explore trials")
    names = [pr.name for pr in p.prospects]
    if not names:
        st.info("Add a prospect first.")
        return
    default = st.session_state.get("sel_prospect")
    name = st.selectbox("Prospect", names, index=names.index(default) if default in names else 0, key="exp_prospect")
    st.session_state.sel_prospect = name
    if name not in st.session_state.runs:
        st.info("Run the prospect to explore its trials.")
        if st.button(f"Run {name}", type="primary", key="exp_run"):
            state.run_one(name)
            st.rerun()
        return
    if state.is_stale(name):
        st.warning("Inputs have changed since the last run; these trials are out of date.")
    df, meta = _table(p, name)
    numeric = [c for c, m in meta.items() if m.numeric]
    inputs = [c for c, m in meta.items() if m.kind == "input"]
    results = [c for c, m in meta.items() if m.kind in ("result", "total")]
    st.caption("Every row is one Monte Carlo trial of the success case (before chance of success). All segments of "
               "a prospect share the same trials, so inputs of one segment can be compared with results of another.")

    filters = _filter_editor(name, df, meta)
    try:
        mask = E.apply_filters(df, filters)
    except (KeyError, ValueError) as exc:
        st.error(str(exc))
        mask = np.ones(len(df), bool)
    kept = int(mask.sum())
    c1, c2 = st.columns([1, 3])
    c1.metric("Trials kept", f"{kept:,}", f"{kept / len(df):.1%} of {len(df):,}", delta_color="off")
    if filters:
        c2.markdown("  \n".join(f"- {f.describe()}" for f in filters))
    if kept == 0:
        st.warning("No trial passes the filters.")

    t_sc, t_mx, t_cor, t_dist, t_tab = st.tabs(["Scatter", "Scatter matrix", "Correlations", "Distributions", "Trials"])
    with t_sc:
        _scatter(name, df, meta, mask, numeric, inputs, results)
    with t_mx:
        default_cols = (inputs[:3] + results[-1:]) if results else inputs[:4]
        cols = st.multiselect("Variables (up to 6)", numeric, default=default_cols, max_selections=6,
                              key=wkey("exp_mx", name))
        if len(cols) >= 2:
            st.plotly_chart(C.scatter_matrix(df, cols, mask, {c: _short(c) for c in cols}), width="stretch",
                            key=wkey("exp_mx_fig", name))
            st.caption("Teal: filtered in. Grey: filtered out.")
    with t_cor:
        _correlations(name, df, mask, numeric, inputs, results)
    with t_dist:
        _distributions(name, df, mask, numeric, results)
    with t_tab:
        view = df[mask]
        st.dataframe(view.head(2000), width="stretch", hide_index=True)
        st.download_button("Download filtered trials (CSV)", view.to_csv(index=False).encode(),
                           f"{name}_filtered_trials.csv", "text/csv", key=wkey("exp_dl", name))


def _filter_editor(name: str, df: pd.DataFrame, meta) -> list[E.TrialFilter]:
    skey = f"filters:{name}"
    stored: list[dict] = st.session_state.setdefault(skey, [])
    with st.expander(f"Filters ({len(stored)})", expanded=bool(stored)):
        st.caption("Keep rows keep only matching trials; Exclude rows drop matching trials. Rows combine with AND. "
                   "For numeric columns give Min and/or Max; for phase or cut-off columns list the values to match, "
                   "separated by commas (e.g. Oil, Gas / condensate or True).")
        cols = [c for c in df.columns if c != "Trial"]
        ek = wkey("exp_filters", name, st.session_state.get(f"filters_rev:{name}", 0))

        def build():
            return pd.DataFrame([{"Column": f["column"], "Action": "Keep" if f["action"] == "keep" else "Exclude",
                                  "Min": f.get("low"), "Max": f.get("high"),
                                  "Values": ", ".join(map(str, f["values"])) if f.get("values") is not None else ""}
                                 for f in stored],
                                columns=["Column", "Action", "Min", "Max", "Values"]).astype({"Min": float, "Max": float})

        ed = st.data_editor(frozen_base(ek, build), key=ek, num_rows="dynamic", hide_index=True, width="stretch",
                            column_config={
                                "Column": st.column_config.SelectboxColumn(options=cols, width="large", required=True),
                                "Action": st.column_config.SelectboxColumn(options=list(ACTIONS), required=True),
                                "Min": st.column_config.NumberColumn(format="%.6g"),
                                "Max": st.column_config.NumberColumn(format="%.6g"),
                                "Values": st.column_config.TextColumn()})
        out_dicts, out = [], []
        for r in ed.to_dict("records"):
            col = r.get("Column")
            if not col or col not in df.columns:
                continue
            action = ACTIONS.get(r.get("Action") or "Keep", "keep")
            info = meta.get(col)
            lo, hi = r.get("Min"), r.get("Max")
            lo = None if lo is None or (isinstance(lo, float) and math.isnan(lo)) else float(lo)
            hi = None if hi is None or (isinstance(hi, float) and math.isnan(hi)) else float(hi)
            vals = None
            if info is not None and not info.numeric:
                raw = [v.strip() for v in str(r.get("Values") or "").split(",") if v.strip()]
                if not raw:
                    st.warning(f"{col}: list the values to match.")
                    continue
                if info.kind == "flag":
                    vals = [v.lower() in ("true", "yes", "1", "passes") for v in raw]
                else:
                    known = set(map(str, df[col].unique()))
                    bad = [v for v in raw if v not in known]
                    if bad:
                        st.warning(f"{col}: unknown value(s) {', '.join(bad)}; known: {', '.join(sorted(known))}.")
                    vals = raw
            elif lo is None and hi is None:
                continue
            elif lo is not None and hi is not None and lo > hi:
                st.warning(f"{col}: Min is above Max; row ignored.")
                continue
            f = E.TrialFilter(col, action, lo, hi, vals)
            out.append(f)
            out_dicts.append({"column": col, "action": action, "low": lo, "high": hi, "values": vals})
        st.session_state[skey] = out_dicts
        if stored and st.button("Clear all filters", key=wkey("exp_clear", name)):
            st.session_state[skey] = []
            st.session_state[f"filters_rev:{name}"] = st.session_state.get(f"filters_rev:{name}", 0) + 1
            st.rerun()
    return out


def _add_filters(name: str, new: list[dict]) -> None:
    skey = f"filters:{name}"
    st.session_state[skey] = st.session_state.get(skey, []) + new
    st.session_state[f"filters_rev:{name}"] = st.session_state.get(f"filters_rev:{name}", 0) + 1


def _scatter(name, df, meta, mask, numeric, inputs, results) -> None:
    c1, c2, c3 = st.columns([3, 3, 2])
    x_default = inputs[0] if inputs else numeric[0]
    y_default = results[-1] if results else numeric[-1]
    xcol = c1.selectbox("X", numeric, index=numeric.index(x_default), key=wkey("exp_x", name))
    ycol = c2.selectbox("Y", numeric, index=numeric.index(y_default), key=wkey("exp_y", name))
    colour_opts = ["Filter status"] + numeric
    ccol = c3.selectbox("Colour", colour_opts, key=wkey("exp_c", name))
    o1, o2, o3 = st.columns(3)
    log_x = o1.checkbox("Log X", key=wkey("exp_lx", name))
    log_y = o2.checkbox("Log Y", key=wkey("exp_ly", name))
    trend = o3.checkbox("Median trend and P90–P10 band", value=True, key=wkey("exp_tr", name))
    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)
    colour = None if ccol == "Filter status" else df[ccol].to_numpy(float)
    tr = E.binned_trend(x[mask], y[mask], 20) if trend and mask.sum() >= 30 else None
    fig = C.scatter_xy(x, y, mask, xcol, ycol, colour, _short(ccol) if colour is not None else None, tr, log_x, log_y)
    ev = st.plotly_chart(fig, key=wkey("exp_sc", name, xcol, ycol), on_select="rerun", selection_mode=("box",),
                         width="stretch")
    r_all = E.rank_correlation(x, y)
    r_f = E.rank_correlation(x[mask], y[mask]) if mask.any() else float("nan")
    st.caption(f"Spearman ρ: all trials {r_all:+.2f}" + ("" if mask.all() else f", filtered in {r_f:+.2f}")
               + ". Drag a box on the chart to select a region.")
    box = selected_box(ev)
    if box is not None:
        (x0, x1), (y0, y1) = box
        if st.button(f"Keep trials inside the box ({_short(xcol)} {x0:.4g}–{x1:.4g}, {_short(ycol)} {y0:.4g}–{y1:.4g})",
                     key=wkey("exp_keepbox", name)):
            _add_filters(name, [{"column": xcol, "action": "keep", "low": x0, "high": x1, "values": None},
                                {"column": ycol, "action": "keep", "low": y0, "high": y1, "values": None}])
            st.rerun()


def _correlations(name, df, mask, numeric, inputs, results) -> None:
    target = st.selectbox("Result", results or numeric, index=len(results or numeric) - 1, key=wkey("exp_cor_t", name))
    rows = [(c, E.rank_correlation(df[c].to_numpy(float), df[target].to_numpy(float))) for c in inputs]
    rows = [r for r in rows if np.isfinite(r[1])]
    rows.sort(key=lambda r: -abs(r[1]))
    if rows:
        labels = [_short(c) for c, _ in rows]
        filt = None if mask.all() else [E.rank_correlation(df.loc[mask, c].to_numpy(float),
                                                           df.loc[mask, target].to_numpy(float)) for c, _ in rows]
        st.plotly_chart(C.rank_compare(labels, [r for _, r in rows], filt, f"Inputs against {_short(target)}"),
                        width="stretch", key=wkey("exp_rank", name, target))
        st.caption("Rank correlation of every uncertain input with the selected result. With filters, the red bars "
                   "show how the relationships change inside the filtered set.")
    st.markdown("**Correlation matrix**")
    default = (inputs[:6] + results[-2:]) if len(inputs) > 6 else inputs + results[-2:]
    cols = st.multiselect("Variables", numeric, default=default, key=wkey("exp_cor_cols", name))
    which = st.radio("Trials", ["All", "Filtered in"], horizontal=True, key=wkey("exp_cor_which", name))
    if len(cols) >= 2:
        m = E.spearman_matrix(df, cols, None if which == "All" else mask)
        m.index = m.columns = [_short(c) for c in cols]
        st.plotly_chart(C.correlation_heatmap(m), width="stretch", key=wkey("exp_heat", name))


def _distributions(name, df, mask, numeric, results) -> None:
    col = st.selectbox("Variable", numeric, index=numeric.index(results[-1]) if results else 0,
                       key=wkey("exp_dist", name))
    v = df[col].to_numpy(float)
    series = [("All trials", v, C.GREY)]
    if not mask.all():
        series += [("Filtered in", v[mask], C.MOSS), ("Filtered out", v[~mask], C.RED)]
    st.plotly_chart(C.overlay_histograms(series, col), width="stretch", key=wkey("exp_hist", name, col))
    st.dataframe(E.compare(df, col, mask), hide_index=True, width="stretch")
    unit = col.split(" [")[-1].rstrip("]") if " [" in col else ""
    curves = [("All trials", v, 1.0, C.GREY)]
    if not mask.all() and mask.any():
        curves.append(("Filtered in", v[mask], 1.0, C.MOSS))
    st.plotly_chart(C.expectation_curves(curves, unit, f"{_short(col)}: probability of exceeding"), width="stretch",
                    key=wkey("exp_curve", name, col))
