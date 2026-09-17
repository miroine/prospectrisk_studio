"""Analysis pages: prospect results, sensitivity and portfolio aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from prospectrisk import risk as R
from prospectrisk import sensitivity as S
from prospectrisk import units as U
from prospectrisk.io_export import input_stats_table, output_stats_table
from prospectrisk.volumetrics import KEY_OUTPUT, OUTPUTS, PHASE_LABELS, PHASES, VARIABLES, input_label, input_quantity
from ui import charts as C
from ui import state
from ui.editors import pair_editor, wkey

VOLUME_OUTPUTS = [k for k, (_, q) in OUTPUTS.items() if q != "none"]


def _unit(p, key: str) -> tuple[str, str]:
    qty = OUTPUTS[key][1]
    return qty, p.settings.unit(qty)


def _prospect_picker(label_key: str) -> str | None:
    p = state.project()
    names = [pr.name for pr in p.prospects]
    if not names:
        st.info("Add a prospect first.")
        return None
    default = st.session_state.get("sel_prospect")
    sel = st.selectbox("Prospect", names, index=names.index(default) if default in names else 0, key=label_key)
    st.session_state.sel_prospect = sel
    return sel


def _run_bar(name: str) -> None:
    c1, c2 = st.columns(2)
    if c1.button(f"Run {name}", type="primary", key=f"run_{label_safe(name)}", width="stretch"):
        with st.spinner("Simulating"):
            state.run_one(name)
    if c2.button("Run all prospects", key=f"runall_{label_safe(name)}", width="stretch"):
        state.run_all()
    if name in st.session_state.runs and state.is_stale(name):
        st.warning("Inputs have changed since this prospect was last run. The results below are out of date.")


def label_safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)


# =============================================================================================
# Results
# =============================================================================================

def page_results() -> None:
    p = state.project()
    st.title("Results")
    name = _prospect_picker("res_prospect")
    if name is None:
        return
    _run_bar(name)
    run = st.session_state.runs.get(name)
    if run is None:
        st.info("Run the simulation to see volumes, chance of success and expectation curves.")
        return
    pr = p.prospect(name)
    a = run.aggregation
    oe = p.settings.unit("oe")
    tot = U.from_internal(run.success_total(), "oe", oe)
    pc = float(a.group_commercial(name).mean())
    stats_tot = R.exceedance_stats(tot) if tot.size else R.exceedance_stats(np.array([0.0]))
    risked_mean = U.from_internal(float(a.group_total(name, KEY_OUTPUT).mean()), "oe", oe)

    m = st.columns(4)
    m[0].metric("Geological chance (Pg)", f"{run.pg:.1%}")
    m[1].metric("Commercial chance (Pc)", f"{pc:.1%}", help=f"Pg and total recoverable ≥ MEFS "
                                                            f"({C.fmt(U.from_internal(pr.mefs, 'oe', oe))} {oe})")
    m[2].metric(f"Success P50 [{oe}]", C.fmt(stats_tot["P50"]),
                f"P90 {C.fmt(stats_tot['P90'])} to P10 {C.fmt(stats_tot['P10'])}", delta_color="off")
    m[3].metric(f"Risked mean [{oe}]", C.fmt(risked_mean))

    seg_names = list(run.segments)
    t_vol, t_curve, t_ch, t_ph, t_stats, t_trials = st.tabs(
        ["Volumes", "Expectation curves", "Chance", "Phase & inputs", "Statistics", "Trials"])

    with t_vol:
        c1, c2 = st.columns(2)
        out_key = c1.selectbox("Output", VOLUME_OUTPUTS, index=VOLUME_OUTPUTS.index(KEY_OUTPUT),
                               format_func=lambda k: OUTPUTS[k][0], key=wkey("res_out", name))
        level = c2.selectbox("Level", ["Prospect total"] + seg_names, key=wkey("res_level", name))
        qty, unit = _unit(p, out_key)
        if level == "Prospect total":
            succ = a.group_success(name)
            vals = U.from_internal(a.group_total(name, out_key)[succ], qty, unit)
            title = f"{OUTPUTS[out_key][0]}, prospect total given at least one segment succeeds"
        else:
            vals = U.from_internal(run.segment_success_case(level, out_key), qty, unit)
            title = f"{OUTPUTS[out_key][0]}, {level} success case"
            if pr.segment(level).truncation.active:
                title += " (after volume cut-off)"
        if vals.size == 0 or np.all(vals == 0):
            st.info("This output is zero for the selected level.")
        else:
            st.plotly_chart(C.histogram(vals, unit, title), width="stretch", key=wkey("res_hist", name))
            s = R.exceedance_stats(vals)
            st.caption(f"Mean {C.fmt(s['Mean'])}  |  P90 {C.fmt(s['P90'])}  |  P50 {C.fmt(s['P50'])}  |  "
                       f"P10 {C.fmt(s['P10'])}  |  Swanson's mean {C.fmt(R.swanson_mean(s['P90'], s['P50'], s['P10']))} {unit}")

    with t_curve:
        c1, c2 = st.columns([3, 1])
        out_key = c1.selectbox("Output", VOLUME_OUTPUTS, index=VOLUME_OUTPUTS.index(KEY_OUTPUT),
                               format_func=lambda k: OUTPUTS[k][0], key=wkey("curve_out", name))
        log_x = c2.checkbox("Log volume axis", key=wkey("curve_log", name))
        qty, unit = _unit(p, out_key)
        curves = []
        for i, sn in enumerate(seg_names):
            v = U.from_internal(run.segment_success_case(sn, out_key), qty, unit)
            curves.append((f"{sn} success", v, 1.0, C.SERIES[(i + 2) % len(C.SERIES)]))
        succ = a.group_success(name)
        total_s = U.from_internal(a.group_total(name, out_key)[succ], qty, unit)
        curves.append(("Prospect success", total_s, 1.0, C.MOSS))
        curves.append((f"Prospect risked (Pg {run.pg:.2f})", total_s, run.pg, C.RED))
        mefs = U.from_internal(pr.mefs, "oe", oe) if out_key == KEY_OUTPUT and pr.mefs > 0 else None
        if log_x:
            curves = [(n, v[v > 0], ch, col) for n, v, ch, col in curves]
        st.plotly_chart(C.expectation_curves(curves, unit, f"{OUTPUTS[out_key][0]} expectation curves", mefs, log_x),
                        width="stretch", key=wkey("curve_chart", name))
        st.caption("Solid lines are success-case curves; the dashed line is scaled by the prospect Pg "
                   "and ends at the chance of success.")

    with t_ch:
        rows = []
        for sn, srun in run.segments.items():
            b = srun.chance
            j = run.segment_index(sn)
            rows.append({"Segment": sn, "Play chance": b.p_play, "Prospect chance": b.p_prospect,
                         "Segment chance": b.p_segment, "Pg": b.pg, "P(V ≥ minimum)": a.p_volume_ok(j),
                         "Effective Pg": run.segment_effective_pg(sn), "Risk class": b.risk_class[0],
                         "Key risk": b.critical[0][1] if b.critical else ""})
        df = pd.DataFrame(rows)
        st.dataframe(df.style.format({c: "{:.3f}" for c in ("Play chance", "Prospect chance", "Segment chance", "Pg", "P(V ≥ minimum)",
                                              "Effective Pg")}),
                     width="stretch", hide_index=True)
        k = len(seg_names)
        if k > 1:
            both = float(a.success[:, a.group_index(name)].all(axis=1).mean())
            st.markdown(f"Chance that **at least one** segment succeeds: **{run.pg:.3f}**. "
                        f"Chance that **all {k}** succeed: **{both:.3f}**.")
        sel = st.selectbox("Breakdown for segment", seg_names, key=wkey("ch_seg", name))
        b = run.segments[sel].chance
        st.plotly_chart(C.chance_bars(b.by_category, {"Play": b.p_play, "Prospect": b.p_prospect, "Segment": b.p_segment},
                                      b.pg), width="stretch", key=wkey("ch_chart", name))
        st.markdown("**Factors from lowest to highest chance**")
        st.dataframe(pd.DataFrame(b.critical, columns=["Scope", "Factor", "Chance"]).style.format({"Chance": "{:.2f}"}),
                     width="stretch", hide_index=True)

    with t_ph:
        sel = st.selectbox("Segment", seg_names, key=wkey("ph_seg", name))
        res = run.segments[sel].result
        present = [i for i, ph in enumerate(PHASES)
                   if p.prospect(name).segment(sel).volumetrics.config.phase_probabilities.get(ph, 0) > 0]
        st.plotly_chart(C.phase_bars([PHASE_LABELS[PHASES[i]] for i in present],
                                     [float(np.mean(res.phase == i)) for i in present]),
                        width="stretch", key=wkey("ph_chart", name))
        var = st.selectbox("Sampled input", list(res.inputs), format_func=input_label, key=wkey("ph_var", name, sel))
        vq = input_quantity(var)
        vu = p.settings.input_unit(var) if var in VARIABLES else "-"
        st.plotly_chart(C.histogram(U.from_internal(res.inputs[var], vq, vu), vu, f"Sampled {input_label(var)}", C.NAVY),
                        width="stretch", key=wkey("ph_hist", name))
        for w in res.diagnostics.get("warnings", []):
            st.warning(w)

    with t_stats:
        st.markdown("**Prospect (aggregated)**")
        gt = pd.DataFrame(a.group_table())
        for col in [c for c in gt.columns if c.startswith("Success") or c in ("Risked mean", "MEFS")]:
            gt[col] = U.from_internal(gt[col].to_numpy(float), "oe", oe)
        gt.columns = [f"{c} [{oe}]" if c.startswith("Success") or c in ("Risked mean", "MEFS") else c for c in gt.columns]
        st.dataframe(gt, width="stretch", hide_index=True)
        st.markdown("**Success-case outputs by segment**")
        st.dataframe(output_stats_table(p, run), width="stretch", hide_index=True)
        st.markdown("**Inputs**")
        st.dataframe(input_stats_table(p, run), width="stretch", hide_index=True)

    with t_trials:
        sel = st.selectbox("Segment", seg_names, key=wkey("tr_seg", name))
        res = run.segments[sel].result
        df = pd.DataFrame({"Phase": [PHASE_LABELS[PHASES[i]] for i in res.phase]})
        for k2, arr in res.inputs.items():
            q = input_quantity(k2)
            u = p.settings.input_unit(k2) if k2 in VARIABLES else "-"
            df[f"{input_label(k2)} [{u}]"] = U.from_internal(arr, q, u)
        for k2 in ("grv_total", "stoiip", "giip_free", "rec_liquids", "rec_gas", KEY_OUTPUT):
            q, u = _unit(p, k2)
            df[f"{OUTPUTS[k2][0]} [{u}]"] = U.from_internal(res.outputs[k2], q, u)
        st.dataframe(df.head(1000), width="stretch")
        st.download_button("Download all trials (CSV)", df.to_csv(index=False).encode(), f"{label_safe(name)}_{label_safe(sel)}_trials.csv",
                           "text/csv", key=wkey("tr_dl", name, sel))


# =============================================================================================
# Sensitivity
# =============================================================================================

def page_sensitivity() -> None:
    p = state.project()
    st.title("Sensitivity")
    name = _prospect_picker("sens_prospect")
    if name is None:
        return
    pr = p.prospect(name)
    seg_names = [s.name for s in pr.segments]
    c1, c2 = st.columns(2)
    sel = c1.selectbox("Segment", seg_names, key=wkey("sens_seg", name))
    out_key = c2.selectbox("Output", VOLUME_OUTPUTS, index=VOLUME_OUTPUTS.index(KEY_OUTPUT),
                           format_func=lambda k: OUTPUTS[k][0], key=wkey("sens_out", name))
    seg = pr.segment(sel)
    phases = [ph for ph in PHASES if seg.volumetrics.config.phase_probabilities.get(ph, 0) > 0]
    c3, c4 = st.columns(2)
    phase = c3.selectbox("Fluid phase", phases, format_func=PHASE_LABELS.get, key=wkey("sens_phase", name, sel))
    base = c4.selectbox("Base case", ["p50", "mean"], format_func={"p50": "Inputs at P50", "mean": "Inputs at mean"}.get,
                        key=wkey("sens_base", name))
    qty, unit = _unit(p, out_key)
    scale = U.factor(qty, unit)

    st.subheader("Tornado")
    try:
        base_out, rows = S.tornado(seg.volumetrics, out_key, phase, base, scf_per_boe=p.settings.scf_per_boe)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not rows or base_out == 0:
        st.info("No uncertain inputs affect this output for the selected phase.")
    else:
        st.plotly_chart(C.tornado(rows, base_out, unit, scale, f"{OUTPUTS[out_key][0]}, {PHASE_LABELS[phase]}"),
                        width="stretch", key=wkey("tornado", name, sel))
        st.caption(f"Each input swung from its P90 to its P10 with all others at the base case "
                   f"({C.fmt(base_out / scale)} {unit}).")
        with st.expander("Tornado table"):
            df = pd.DataFrame(rows).drop(columns=["key"])
            for c in ("Output at low input", "Output at high input", "Swing"):
                df[c] = df[c] / scale
            st.dataframe(df, width="stretch", hide_index=True)

    st.subheader("Contribution to variance")
    run = st.session_state.runs.get(name)
    if run is None:
        st.info("Run the prospect to compute rank-correlation sensitivity from the Monte Carlo trials.")
        return
    if state.is_stale(name):
        st.warning("Based on the last run, which is out of date.")
    res = run.segments[sel].result
    only_phase = st.checkbox("Only trials with the selected phase", value=len(phases) > 1, key=wkey("sens_mask", name))
    mask = res.phase == PHASES.index(phase) if only_phase else None
    rr = S.rank_sensitivity(dict(res.inputs), res.outputs[out_key], mask)
    if rr:
        st.plotly_chart(C.rank_bars(rr, f"{OUTPUTS[out_key][0]}, {sel}"), width="stretch", key=wkey("rank", name, sel))
        st.caption("Signed squared Spearman rank correlation, normalised. Unlike the tornado, this includes "
                   "the effect of input correlations and of the full distribution shapes.")
    else:
        st.info("No varying inputs for this selection.")


# =============================================================================================
# Portfolio
# =============================================================================================

def page_portfolio() -> None:
    p = state.project()
    st.title("Portfolio")
    names = [pr.name for pr in p.prospects]
    if not names:
        st.info("Add prospects first.")
        return
    prev = [n for n in st.session_state.get("portfolio_names", names) if n in names]
    sel = st.multiselect("Prospects in portfolio", names, default=prev or names, key=wkey("pf_sel"))
    if not sel:
        return
    with st.expander("Dependencies between prospects", expanded=False):
        st.markdown("Prospects in the **same play already share the play chance factors exactly**. Use these tables "
                    "for additional dependency between prospect-specific risks, and for correlated success volumes.")
        dep_m = _pairs_to_matrix(sel, p.portfolio_dependency)
        new_dep = pair_editor("pf_dep", sel, dep_m, "Dependency", 0.0)
        vol_m = _pairs_to_matrix(sel, p.portfolio_volume_correlation)
        new_vol = pair_editor("pf_vol", sel, vol_m, "Rank correlation", -1.0)
        p.portfolio_dependency = _merge_pairs(p.portfolio_dependency, sel, new_dep)
        p.portfolio_volume_correlation = _merge_pairs(p.portfolio_volume_correlation, sel, new_vol)

    if st.button("Aggregate portfolio", type="primary", key="pf_run"):
        with st.spinner("Aggregating"):
            state.run_portfolio_now(sel)
    res = st.session_state.portfolio
    if res is None:
        st.info("Aggregate the selected prospects to see portfolio chance and volumes.")
        return
    if st.session_state.get("portfolio_names") != sel or st.session_state.portfolio_fp != state.portfolio_fp(p, sel):
        st.warning("Selection or inputs changed since the last aggregation. Results are out of date.")
    oe = p.settings.unit("oe")
    s = res.summary()
    m = st.columns(4)
    m[0].metric("P(at least one discovery)", f"{s['P(at least one success)']:.1%}")
    m[1].metric("Expected discoveries", f"{s['Expected discoveries']:.2f}",
                f"{s['Expected commercial discoveries']:.2f} commercial", delta_color="off")
    m[2].metric(f"Risked mean [{oe}]", C.fmt(U.from_internal(s["Risked mean"], "oe", oe)))
    m[3].metric(f"P50 given ≥1 discovery [{oe}]", C.fmt(U.from_internal(s["P50 | ≥1 success"], "oe", oe)))

    t1, t2, t3 = st.tabs(["Expectation curve", "Discoveries", "Tables"])
    with t1:
        tot = U.from_internal(res.total(KEY_OUTPUT), "oe", oe)
        any_s = res.success.any(axis=1)
        p_any = float(any_s.mean())
        curves = [("Portfolio given ≥1 discovery", tot[any_s], 1.0, C.MOSS),
                  (f"Portfolio risked (P≥1 {p_any:.2f})", tot[any_s], p_any, C.RED)]
        for i, g in enumerate(res.groups):
            gs = res.group_success(g)
            gt = U.from_internal(res.group_total(g, KEY_OUTPUT)[gs], "oe", oe)
            curves.append((f"{g} risked", gt, float(gs.mean()), C.SERIES[(i + 2) % len(C.SERIES)]))
        st.plotly_chart(C.expectation_curves(curves, oe, "Recoverable oil equivalent"), width="stretch", key="pf_curve")
    with t2:
        st.plotly_chart(C.count_bars(res.discovery_counts("group"), "Number of prospect discoveries"),
                        width="stretch", key="pf_counts")
        st.plotly_chart(C.count_bars(res.discovery_counts("group", commercial=True), "Number of commercial discoveries"),
                        width="stretch", key="pf_counts_c")
    with t3:
        gt = pd.DataFrame(res.group_table())
        vol_cols = [c for c in gt.columns if c.startswith("Success") or c in ("Risked mean", "MEFS")]
        for c in vol_cols:
            gt[c] = U.from_internal(gt[c].to_numpy(float), "oe", oe)
        gt = gt.rename(columns={c: f"{c} [{oe}]" for c in vol_cols})
        st.dataframe(gt, width="stretch", hide_index=True)
        it = pd.DataFrame(res.item_table())
        for c in ("Success mean", "Risked mean"):
            it[c] = U.from_internal(it[c].to_numpy(float), "oe", oe)
        it = it.rename(columns={"Group": "Prospect", "Item": "Segment", "Success mean": f"Success mean [{oe}]",
                                "Risked mean": f"Risked mean [{oe}]"})
        st.dataframe(it, width="stretch", hide_index=True)
        sm = pd.DataFrame([{"Metric": k, "Value": v if ("P(" in k or "discover" in k) else U.from_internal(v, "oe", oe)}
                           for k, v in s.items()])
        st.dataframe(sm, width="stretch", hide_index=True)
        for w in res.diagnostics.get("warnings", []):
            st.warning(w)


def _pairs_to_matrix(names: list[str], pairs: list) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    m = np.eye(len(names))
    for a, b, r in pairs:
        if a in idx and b in idx and a != b:
            m[idx[a], idx[b]] = m[idx[b], idx[a]] = float(r)
    return m


def _merge_pairs(pairs: list, names: list[str], matrix: np.ndarray | None) -> list:
    keep = [[a, b, r] for a, b, r in pairs if not (a in names and b in names)]
    if matrix is not None:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if matrix[i, j] != 0:
                    keep.append([names[i], names[j], float(matrix[i, j])])
    return keep
