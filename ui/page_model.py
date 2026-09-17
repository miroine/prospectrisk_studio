"""Model-building pages: project settings, plays, prospects and segments."""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prospectrisk import distributions as D
from prospectrisk import expressions as ex
from prospectrisk import risk as R
from prospectrisk import units as U
from prospectrisk.aggregation import VolumeTruncation
from prospectrisk.defaults import (VARIABLE_HELP, default_area_depth_table, default_distribution, ensure_required,
                                   new_prospect, new_segment, unique_name)
from prospectrisk.io_export import project_from_yaml, project_to_yaml
from prospectrisk.examples import example_project
from prospectrisk.project import Play, Project, Settings
from prospectrisk.units import BOE_CONVENTIONS, UNIT_SYSTEMS
from prospectrisk.volumetrics import (CONTACT_MODES, GRV_METHODS, OUTPUTS, PHASE_LABELS, PHASES, VARIABLES, Expression,
                                     constant_inputs)
from ui import charts as C
from ui import state
from ui.editors import (adequacy_editor, area_depth_editor, bump, correlation_editor, dependency_editor, dist_editor,
                        factor_editor, input_unit_picker, num, pair_editor, wkey)

STATUSES = ["Lead", "Prospect", "Drilled – dry", "Drilled – discovery"]


# =============================================================================================
# Project page
# =============================================================================================

def page_project() -> None:
    p = state.project()
    st.title(p.name)
    st.caption(f"{len(p.plays)} plays, {len(p.prospects)} prospects, "
               f"{sum(len(pr.segments) for pr in p.prospects)} segments. {p.notes}")

    runs = state.valid_runs()
    if runs:
        _risk_volume_overview(p, runs)
    else:
        st.info("Run the simulation to see the risk–volume overview of the portfolio.")
        if st.button("Run all prospects", type="primary", key="proj_run_all"):
            state.run_all()
            st.rerun()

    t_set, t_play, t_mat, t_file = st.tabs(["Settings", "Plays", "Chance-adequacy matrix", "Files"])
    with t_set:
        _settings(p)
    with t_play:
        _plays(p)
    with t_mat:
        st.markdown("Chance assigned to a factor entered in *adequacy* mode, by interpretation of the data "
                    "(rows) and confidence in that interpretation (columns). Replace with your company calibration.")
        p.adequacy_matrix = adequacy_editor(p.adequacy_matrix)
        if st.button("Reset to default matrix", key="reset_matrix"):
            p.adequacy_matrix = copy.deepcopy(R.DEFAULT_ADEQUACY_MATRIX)
            bump()
            st.rerun()
    with t_file:
        _files(p)


def _risk_volume_overview(p: Project, runs: dict) -> None:
    unit = p.settings.unit("oe")
    rows = []
    for name, r in runs.items():
        pr = p.prospect(name)
        tot = r.success_total()
        mean = U.from_internal(float(tot.mean()) if tot.size else 0.0, "oe", unit)
        rows.append({"Prospect": name, "Play": pr.play, "Status": pr.status, "Pg": r.pg,
                     "Pc": float(r.aggregation.group_commercial(name).mean()),
                     f"Success mean [{unit}]": mean, f"Risked mean [{unit}]": r.pg * mean,
                     "Stale": state.is_stale(name)})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    plays = list(dict.fromkeys(df["Play"]))
    xmax = max(df[f"Success mean [{unit}]"].max() * 1.3, 1e-9)
    for level in sorted({float(f"{v:.1g}") for v in df[f"Risked mean [{unit}]"] if v > 0}):
        xs = np.linspace(level / 1.0, xmax, 100)
        fig.add_trace(go.Scatter(x=xs, y=level / xs, mode="lines", line=dict(color=C.LINE, width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
    for i, play in enumerate(plays):
        d = df[df["Play"] == play]
        size = 14 + 30 * np.sqrt(d[f"Risked mean [{unit}]"] / max(df[f"Risked mean [{unit}]"].max(), 1e-12))
        fig.add_trace(go.Scatter(
            x=d[f"Success mean [{unit}]"], y=d["Pg"], mode="markers+text", text=d["Prospect"], textposition="top center",
            marker=dict(size=size, color=C.SERIES[i % len(C.SERIES)], line=dict(color="white", width=1.5), opacity=0.9),
            name=play, hovertemplate="%{text}<br>Pg %{y:.2f}<br>Success mean %{x:.3g} " + unit + "<extra></extra>"))
    fig.update_yaxes(range=[0, min(1.0, max(0.6, df["Pg"].max() * 1.3))], tickformat=".0%")
    fig.update_xaxes(range=[0, xmax])
    C.eq_layout(fig, "Risk versus volume", height=380, xtitle=f"Success-case mean recoverable [{unit}]",
                ytitle="Geological chance of success")
    st.plotly_chart(fig, width="stretch")
    st.caption("Bubble size scales with risked mean volume; dotted lines are constant risked volume.")
    st.dataframe(df.style.format({"Pg": "{:.3f}", "Pc": "{:.3f}", f"Success mean [{unit}]": "{:.3g}",
                                  f"Risked mean [{unit}]": "{:.3g}"}), width="stretch", hide_index=True)
    if df["Stale"].any():
        st.warning("Some results are out of date with the inputs. Re-run to refresh them.")


def _settings(p: Project) -> None:
    s: Settings = p.settings
    p.name = st.text_input("Project name", p.name, key=wkey("proj_name"))
    p.notes = st.text_area("Notes", p.notes, key=wkey("proj_notes"), height=80)
    c1, c2 = st.columns(2)
    with c1:
        s.n_trials = int(st.number_input("Monte Carlo trials", 100, 1_000_000, int(s.n_trials), step=1000,
                                         key=wkey("n_trials"),
                                         help="10,000 is adequate for most prospects; use 50,000+ for portfolios with small chances."))
        s.sampling = st.selectbox("Sampling", ["lhs", "mc"], index=["lhs", "mc"].index(s.sampling), key=wkey("sampling"),
                                  format_func={"lhs": "Latin Hypercube", "mc": "Plain Monte Carlo"}.get)
    with c2:
        s.seed = int(st.number_input("Random seed", 0, 2**31 - 1, int(s.seed), key=wkey("seed"),
                                     help="Same seed and inputs give identical results."))
        s.boe_convention = st.selectbox("Gas to oil-equivalent", list(BOE_CONVENTIONS),
                                        index=list(BOE_CONVENTIONS).index(s.boe_convention), key=wkey("boe"))
    st.caption("Percentiles follow the exploration convention: P90 is the low case, P10 the high case.")


def _plays(p: Project) -> None:
    names = [pl.name for pl in p.plays]
    c1, c2, c3 = st.columns([3, 1, 1])
    idx = 0
    if names:
        sel = c1.selectbox("Play", names, key=wkey("play_sel"))
        idx = names.index(sel)
    if c2.button("Add play", key="add_play", width="stretch"):
        p.plays.append(Play(unique_name("New play", names)))
        bump()
        st.rerun()
    if names and c3.button("Delete", key="del_play", width="stretch"):
        used = [pr.name for pr in p.prospects if pr.play == names[idx]]
        if used:
            st.error(f"Play is used by: {', '.join(used)}. Move those prospects to another play first.")
        else:
            p.plays.pop(idx)
            bump()
            st.rerun()
    if not names:
        st.info("Add a play to start.")
        return
    play = p.plays[idx]
    new = st.text_input("Play name", play.name, key=wkey("play_name", idx))
    if new != play.name:
        if not new.strip() or new in names:
            st.error("Play names must be unique and not empty.")
        else:
            for pr in p.prospects:
                if pr.play == play.name:
                    pr.play = new
            play.name = new
    play.notes = st.text_area("Notes", play.notes, key=wkey("play_notes", idx), height=70)
    st.markdown("**Play chance factors**, shared by every segment of every prospect in this play")
    play.factors = factor_editor(f"play{idx}", play.factors, p.adequacy_matrix)


def _files(p: Project) -> None:
    st.download_button("Save project (YAML)", project_to_yaml(p), file_name=f"{_slug(p.name)}.yaml",
                       mime="application/x-yaml", key="dl_yaml_proj")
    up = st.file_uploader("Open project file", type=["yaml", "yml"], key="upload_project")
    if up is not None:
        data = up.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if st.session_state.get("loaded_digest") != digest:
            try:
                newp = project_from_yaml(data)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.loaded_digest = digest
                state.replace_project(newp)
                st.success(f"Opened '{newp.name}'.")
                st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("Load demo project", key="load_demo", width="stretch"):
        state.replace_project(example_project())
        st.rerun()
    if c2.button("Start empty project", key="new_proj", width="stretch"):
        play = Play("Play 1")
        state.replace_project(Project("New project", plays=[play], prospects=[new_prospect("Prospect 1", play.name)]))
        st.rerun()


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")[:60] or "project"


# =============================================================================================
# Prospects page
# =============================================================================================

def page_prospects() -> None:
    p = state.project()
    us = p.settings
    st.title("Prospects")
    names = [pr.name for pr in p.prospects]
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    pi = 0
    if names:
        default = st.session_state.get("sel_prospect")
        sel = c1.selectbox("Prospect", names, index=names.index(default) if default in names else 0,
                           key=wkey("pros_sel"))
        st.session_state.sel_prospect = sel
        pi = names.index(sel)
    if c2.button("Add", key="add_pros", width="stretch"):
        if not p.plays:
            st.error("Create a play first (Project page).")
        else:
            nm = unique_name("New prospect", names)
            p.prospects.append(new_prospect(nm, p.plays[0].name))
            st.session_state.sel_prospect = nm
            bump()
            st.rerun()
    if names and c3.button("Duplicate", key="dup_pros", width="stretch"):
        cp = copy.deepcopy(p.prospects[pi])
        cp.name = unique_name(cp.name + " copy", names)
        p.prospects.append(cp)
        st.session_state.sel_prospect = cp.name
        bump()
        st.rerun()
    if names and c4.button("Delete", key="del_pros", width="stretch"):
        gone = p.prospects.pop(pi).name
        st.session_state.runs.pop(gone, None)
        st.session_state.sel_prospect = None
        bump()
        st.rerun()
    if not names:
        st.info("Add a prospect to start.")
        return

    pr = p.prospects[pi]
    t_gen, t_seg, t_dep = st.tabs(["General & shared chance", "Segments", "Segment dependencies"])
    with t_gen:
        new = st.text_input("Prospect name", pr.name, key=wkey("pr_name", pi))
        if new != pr.name:
            if not new.strip() or new in names:
                st.error("Prospect names must be unique and not empty.")
            else:
                st.session_state.runs.pop(pr.name, None)
                pr.name = new
                st.session_state.sel_prospect = new
        c1, c2 = st.columns(2)
        plays = [pl.name for pl in p.plays]
        pr.play = c1.selectbox("Play", plays, index=plays.index(pr.play) if pr.play in plays else 0,
                               key=wkey("pr_play", pi))
        pr.status = c2.selectbox("Status", STATUSES, index=STATUSES.index(pr.status) if pr.status in STATUSES else 1,
                                 key=wkey("pr_status", pi))
        oe = us.unit("oe")
        pr.mefs = U.to_internal(num(f"Minimum economic field size [{oe}]", U.from_internal(pr.mefs, "oe", oe),
                                    wkey("pr_mefs", pi, oe), min_value=0.0,
                                    help="Applied to total recoverable oil equivalent of the prospect in each trial."),
                                "oe", oe)
        pr.notes = st.text_area("Notes", pr.notes, key=wkey("pr_notes", pi), height=70)
        st.markdown("**Prospect chance factors**, shared by all segments of this prospect")
        pr.factors = factor_editor(f"pros{pi}", pr.factors, p.adequacy_matrix)
        try:
            play = p.play(pr.play)
            st.caption(f"Play chance {R.product(play.factors, p.adequacy_matrix):.3f} "
                       f"× prospect shared chance {R.product(pr.factors, p.adequacy_matrix):.3f}, "
                       "before segment factors.")
        except ValueError:
            pass
    with t_seg:
        _segments(p, pi, pr, us)
    with t_dep:
        seg_names = [s.name for s in pr.segments]
        st.markdown("**Chance dependency** between segment-level risks (0 = independent, 1 = fully dependent). "
                    "Play and prospect factors are already shared exactly; this covers the remaining segment risks.")
        pr.segment_dependency = _as_list(pair_editor(f"dep{pi}", seg_names, pr.segment_dependency, "Dependency", 0.0))
        st.markdown("**Success-volume rank correlation** between segments")
        pr.segment_volume_correlation = _as_list(pair_editor(f"vcorr{pi}", seg_names, pr.segment_volume_correlation,
                                                             "Rank correlation", -1.0))
        st.caption("Ignored between segments that have input dependencies below, which already make them "
                   "dependent trial by trial.")
        st.markdown("**Input dependencies**: correlate or link uncertain inputs across segments (or within one). "
                    "A *link* makes input B take the same percentile as input A in every trial, e.g. the same "
                    "fluid properties in two stacked sands; B keeps its own distribution.")
        inputs = {sg.name: [k for k, d in sg.volumetrics.active_variables().items() if not d.is_constant]
                  for sg in pr.segments}
        pr.input_dependencies = dependency_editor(f"indep{pi}", seg_names, inputs,
                                                  {k: v[0] for k, v in VARIABLES.items()}, pr.input_dependencies)


def _as_list(m):
    return None if m is None else np.asarray(m, float).tolist()


def _segments(p: Project, pi: int, pr, us: str) -> None:
    names = [s.name for s in pr.segments]
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    si = 0
    if names:
        key = f"sel_segment_{pi}"
        default = st.session_state.get(key)
        sel = c1.selectbox("Segment", names, index=names.index(default) if default in names else 0,
                           key=wkey("seg_sel", pi))
        st.session_state[key] = sel
        si = names.index(sel)

    def _reset_pairs():
        pr.segment_dependency = None
        pr.segment_volume_correlation = None

    if c2.button("Add", key=f"add_seg{pi}", width="stretch"):
        nm = unique_name("New segment", names)
        pr.segments.append(new_segment(nm))
        st.session_state[f"sel_segment_{pi}"] = nm
        _reset_pairs()
        bump()
        st.rerun()
    if names and c3.button("Duplicate", key=f"dup_seg{pi}", width="stretch"):
        cp = copy.deepcopy(pr.segments[si])
        cp.name = unique_name(cp.name + " copy", names)
        pr.segments.append(cp)
        st.session_state[f"sel_segment_{pi}"] = cp.name
        _reset_pairs()
        bump()
        st.rerun()
    if c4.button("Delete", key=f"del_seg{pi}", width="stretch", disabled=len(names) < 2) and len(names) > 1:
        gone = pr.segments.pop(si).name
        pr.input_dependencies = [d for d in pr.input_dependencies if gone not in (d.get("seg_a"), d.get("seg_b"))]
        _reset_pairs()
        bump()
        st.rerun()
    if not names:
        st.info("Add a segment.")
        return
    seg = pr.segments[si]
    new = st.text_input("Segment name", seg.name, key=wkey("seg_name", pi, si))
    if new != seg.name:
        if not new.strip() or new in names:
            st.error("Segment names must be unique within the prospect and not empty.")
        else:
            for d in pr.input_dependencies:
                for side in ("seg_a", "seg_b"):
                    if d.get(side) == seg.name:
                        d[side] = new
            seg.name = new
            st.session_state[f"sel_segment_{pi}"] = new
    segment_editor(p, pi, si, pr, seg, us)


def _var(path: str, vol, name: str, settings, optional: str | None = None) -> None:
    label, qty, _ = VARIABLES[name]
    if optional is not None:
        on = st.checkbox(optional, value=name in vol.variables or name in vol.expressions, key=wkey(path, name, "on"))
        if not on:
            vol.variables.pop(name, None)
            vol.expressions.pop(name, None)
            return
    if name not in vol.variables:
        vol.variables[name] = default_distribution(name, vol.config)
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        mode = c1.selectbox("Defined by", ["Distribution", "Formula"], index=1 if name in vol.expressions else 0,
                            key=wkey(path, name, "mode"), help="A formula makes this input depend on other inputs.")
        with c2:
            st.caption("Unit")
            if input_unit_picker(f"{path}:{name}", settings, name, qty):
                bump()
                st.rerun()
        unit = settings.input_unit(name)
        if mode == "Distribution":
            vol.expressions.pop(name, None)
            vol.variables[name] = dist_editor(f"{path}:{name}", label, vol.variables[name], qty, unit,
                                              help=VARIABLE_HELP.get(name))
        else:
            _formula(path, vol, name, settings, unit)


def _formula(path: str, vol, name: str, settings, unit: str) -> None:
    label, qty, _ = VARIABLES[name]
    e = vol.expressions.get(name)
    unit_txt = "" if unit in ("fraction", "-") else f" [{unit}]"
    st.markdown(f"**{label}**{unit_txt} =")
    avail = [k for k in list(vol.active_variables()) + list(vol.active_expressions()) if k != name]
    default = e.expr if e else f"{U.from_internal(vol.variables[name].p50, qty, unit):.6g}"
    expr = st.text_input("Formula", default, key=wkey(path, name, "expr"), label_visibility="collapsed")
    st.caption("Inputs: " + ", ".join(f"`{k}` [{settings.input_unit(k)}]" for k in avail)
               + ". Functions: " + ", ".join(sorted(ex.FUNCTIONS)) + ".")
    if e is None or expr != e.expr:
        try:
            names = ex.parse(expr)[1]
        except ex.ExpressionError as exc:
            st.error(str(exc))
            return
        units = {k: settings.input_unit(k) for k in names | {name} if k in VARIABLES}
        e = Expression(expr, units, e.noise if e else None, e.noise_mode if e else "multiply")
    shown = {k: u for k, u in e.units.items() if u not in ("fraction", "-")}
    if shown:
        st.caption("Units inside this formula (fixed when it was written): "
                   + ", ".join(f"{k} in {u}" for k, u in shown.items()))
    if st.checkbox("Add uncertainty to the formula result", value=e.noise is not None, key=wkey(path, name, "noise_on")):
        modes = {"multiply": "Multiply by a factor", "add": "Add a term"}
        mode = st.selectbox("Uncertainty", list(modes), index=list(modes).index(e.noise_mode), format_func=modes.get,
                            key=wkey(path, name, "noise_mode"))
        if e.noise is None or mode != e.noise_mode:
            base = abs(U.from_internal(vol.variables[name].p50, qty, e.units.get(name, unit))) or 1.0
            e.noise = D.Triangular(0.9, 1.0, 1.1) if mode == "multiply" else D.Normal(0.0, 0.05 * base)
            e.noise_mode = mode
        tunit = e.units.get(name, unit)
        lab = "Uncertainty factor" if mode == "multiply" else \
            f"Uncertainty term{'' if tunit in ('fraction', '-') else f' [{tunit}]'}"
        e.noise = dist_editor(f"{path}:{name}:noise:{mode}", lab, e.noise, "none", "-")
    else:
        e.noise = None
    vol.expressions[name] = e
    try:
        base_vals = constant_inputs(vol, "p50")
        out = vol.evaluate_expressions({k: np.array([v]) for k, v in base_vals.items()}, 1)
        st.caption(f"With inputs at P50: {C.fmt(U.from_internal(float(out[name][0]), qty, unit))}{unit_txt}")
    except ValueError as exc:
        st.error(str(exc))


def segment_editor(p: Project, pi: int, si: int, pr, seg, us: str) -> None:
    vol = seg.volumetrics
    cfg = vol.config
    path = f"p{pi}s{si}"
    t_grv, t_res, t_fl, t_cor, t_ch, t_cut = st.tabs(["Gross rock volume", "Reservoir", "Fluids", "Correlations",
                                                     "Chance", "Volume cut-off"])

    with t_grv:
        methods = list(GRV_METHODS)
        cfg.grv_method = st.selectbox("GRV method", methods, index=methods.index(cfg.grv_method),
                                      format_func=GRV_METHODS.get, key=wkey(path, "grv_method"))
        if cfg.grv_method == "area_thickness":
            for v in ("area", "gross_thickness", "geometric_factor"):
                _var(path, vol, v, us)
        elif cfg.grv_method == "direct":
            _var(path, vol, "grv", us)
        else:
            _area_depth(path, vol, us)

    with t_res:
        for v in ("net_to_gross", "porosity", "hc_saturation"):
            _var(path, vol, v, us)

    with t_fl:
        _fluids(path, vol, us)

    ensure_required(vol)

    with t_cor:
        active = vol.active_variables()
        keys = [k for k, d in active.items() if not d.is_constant]
        st.markdown("Rank correlations between uncertain inputs of this segment (Iman–Conover). "
                    "Typical examples: porosity with saturation, net-to-gross with porosity.")
        if len(keys) < 2:
            st.caption("Needs at least two uncertain inputs.")
        else:
            vol.correlations = correlation_editor(path, keys, {k: VARIABLES[k][0] for k in keys}, vol.correlations)

    with t_ch:
        st.markdown("**Segment chance factors**, specific to this segment")
        seg.factors = factor_editor(f"{path}:chance", seg.factors, p.adequacy_matrix)
        try:
            b = R.breakdown(p.play(pr.play).factors, pr.factors, seg.factors, p.adequacy_matrix)
        except ValueError as exc:
            st.error(str(exc))
            return
        label, colour = b.risk_class
        m1, m2, m3 = st.columns(3)
        m1.metric("Segment Pg", f"{b.pg:.3f}")
        m2.metric("Chance given play works", f"{b.p_conditional:.3f}")
        m3.metric("Key risk", b.critical[0][1] if b.critical else "–",
                  f"{b.critical[0][2]:.2f}" if b.critical else None, delta_color="off")
        st.markdown(f"<span class='risk-chip' style='background:{colour}'>{label}</span>", unsafe_allow_html=True)
        st.plotly_chart(C.chance_bars(b.by_category, {"Play": b.p_play, "Prospect": b.p_prospect,
                                                      "Segment": b.p_segment}, b.pg),
                        width="stretch", key=wkey(path, "chance_chart"))

    with t_cut:
        _cutoff(path, p, pr, seg, us)


def _cutoff(path: str, p: Project, pr, seg, settings) -> None:
    t: VolumeTruncation = seg.truncation
    st.markdown("Truncate the success-case volume. A **minimum** usually defines what counts as a geological "
                "success (for example the smallest accumulation that would flow or be called a discovery). "
                "A **maximum** removes physically impossible outcomes.")
    outs = [k for k, (_, q) in OUTPUTS.items() if q != "none"]
    t.output = st.selectbox("Applied to", outs, index=outs.index(t.output) if t.output in outs else 0,
                            format_func=lambda k: OUTPUTS[k][0], key=wkey(path, "trunc_out"))
    q = OUTPUTS[t.output][1]
    u = settings.unit(q)
    if st.checkbox("Minimum volume", value=t.minimum is not None, key=wkey(path, "tmin_on")):
        t.minimum = U.to_internal(num(f"Minimum [{u}]", U.from_internal(t.minimum or 0.0, q, u),
                                      wkey(path, "tmin", u, t.output), min_value=0.0), q, u)
        modes = {"chance": "Count as failures: Pg is reduced by P(V ≥ minimum)",
                 "renormalise": "Exclude them: distribution renormalised, Pg unchanged"}
        t.min_mode = st.radio("Trials below the minimum", list(modes), index=list(modes).index(t.min_mode),
                              format_func=modes.get, key=wkey(path, "tmin_mode"))
    else:
        t.minimum = None
    if st.checkbox("Maximum volume", value=t.maximum is not None, key=wkey(path, "tmax_on")):
        default = t.maximum if t.maximum is not None else (t.minimum or 0.0) + U.to_internal(1.0, q, u)
        t.maximum = U.to_internal(num(f"Maximum [{u}]", U.from_internal(default, q, u),
                                      wkey(path, "tmax", u, t.output), min_value=0.0), q, u)
        st.caption("Trials above the maximum are excluded and the distribution renormalised; Pg is unchanged.")
    else:
        t.maximum = None
    try:
        t.validate()
    except ValueError as exc:
        st.error(str(exc))
    run = st.session_state.runs.get(pr.name)
    if run is not None and seg.name in run.segments and t.active:
        raw = run.segments[seg.name].result.outputs[t.output]
        pg = run.segments[seg.name].chance.pg
        msg = []
        if t.minimum is not None:
            p_ok = float(np.mean(raw >= t.minimum))
            msg.append(f"{p_ok:.1%} of success-case trials reach the minimum"
                       + (f", so Pg {pg:.3f} becomes {pg * p_ok:.3f}" if t.min_mode == "chance" else ""))
        if t.maximum is not None:
            msg.append(f"{float(np.mean(raw > t.maximum)):.1%} of trials exceed the maximum")
        st.info("From the last run: " + "; ".join(msg) + ". Re-run to apply.")


def _area_depth(path: str, vol, us: str) -> None:
    cfg = vol.config
    if cfg.top_table is None:
        cfg.top_table = default_area_depth_table()
    st.markdown("**Top-structure area–depth table**")
    du, au = us.unit("length"), us.unit("area")
    cfg.top_table = area_depth_editor(f"{path}:top", cfg.top_table, du, au) or cfg.top_table
    use_base = st.checkbox("Use a separate base-surface table instead of a thickness", value=cfg.base_table is not None,
                           key=wkey(path, "use_base"))
    if use_base:
        st.markdown("**Base-surface area–depth table**")
        cfg.base_table = area_depth_editor(f"{path}:base", cfg.base_table, du, au)
    else:
        cfg.base_table = None
        _var(path, vol, "gross_thickness", us)
    modes = list(CONTACT_MODES)
    cfg.contact_mode = st.selectbox("Contact defined by", modes, index=modes.index(cfg.contact_mode),
                                    format_func=CONTACT_MODES.get, key=wkey(path, "contact_mode"))
    _var(path, vol, {"contact_depth": "contact_depth", "column_height": "column_height",
                     "fill_fraction": "fill_fraction"}[cfg.contact_mode], us)
    _var(path, vol, "spill_depth", us, optional="Specify spill-point depth (otherwise deepest contour)")
    cfg.limit_to_spill = st.checkbox("Limit contact to spill point", value=cfg.limit_to_spill,
                                     key=wkey(path, "limit_spill"))

    tables = [("Top", list(U.from_internal(np.array(cfg.top_table.depth), "length", du)),
               list(U.from_internal(np.array(cfg.top_table.area), "area", au)), C.NAVY)]
    if cfg.base_table is not None:
        tables.append(("Base", list(U.from_internal(np.array(cfg.base_table.depth), "length", du)),
                       list(U.from_internal(np.array(cfg.base_table.area), "area", au)), C.MOSS))
    contacts = {}
    crest = cfg.top_table.crest
    if cfg.contact_mode == "contact_depth" and "contact_depth" in vol.variables:
        d = vol.variables["contact_depth"]
        contacts = {"Contact P90": d.p90, "Contact P10": d.p10}
    elif cfg.contact_mode == "column_height" and "column_height" in vol.variables:
        d = vol.variables["column_height"]
        contacts = {"Contact P90": crest + d.p90, "Contact P10": crest + d.p10}
    elif cfg.contact_mode == "fill_fraction" and "fill_fraction" in vol.variables:
        spill = vol.variables["spill_depth"].p50 if "spill_depth" in vol.variables else cfg.top_table.base
        d = vol.variables["fill_fraction"]
        contacts = {"Contact P90": crest + d.p90 * (spill - crest), "Contact P10": crest + d.p10 * (spill - crest)}
    if "spill_depth" in vol.variables:
        contacts["Spill"] = vol.variables["spill_depth"].p50
    contacts = {k: U.from_internal(v, "length", du) for k, v in contacts.items()}
    st.plotly_chart(C.area_depth_plot(tables, contacts, au, du), width="stretch", key=wkey(path, "adt_chart"))


def _fluids(path: str, vol, us: str) -> None:
    cfg = vol.config
    st.markdown("**Fluid phase probabilities** in case of geological success")
    cols = st.columns(3)
    probs = {}
    for c, ph in zip(cols, PHASES):
        with c:
            probs[ph] = st.number_input(PHASE_LABELS[ph], 0.0, 1.0, float(cfg.phase_probabilities.get(ph, 0.0)),
                                        step=0.05, format="%.3f", key=wkey(path, "phase", ph))
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-6:
        st.error(f"Phase probabilities sum to {total:.3f}; they must sum to 1. The previous values are kept.")
    else:
        cfg.phase_probabilities = probs

    if cfg.phase_probabilities.get("oil_gascap", 0) > 0:
        _var(path, vol, "gas_cap_fraction", us)
    if cfg.has_oil:
        st.markdown("#### Oil")
        for v in ("bo", "rf_oil"):
            _var(path, vol, v, us)
        _var(path, vol, "gor", us, optional="Include solution gas (GOR)")
        if "gor" in vol.variables:
            _var(path, vol, "rf_solution_gas", us, optional="Separate recovery factor for solution gas")
    if cfg.has_gas:
        st.markdown("#### Gas")
        fvf = {"bg": "Bg (formation volume factor)", "eg": "E (expansion factor = 1/Bg)"}
        new_fvf = st.radio("Gas volume factor entered as", list(fvf), index=list(fvf).index(cfg.gas_fvf),
                           format_func=fvf.get, horizontal=True, key=wkey(path, "gas_fvf"))
        if new_fvf != cfg.gas_fvf:
            other = "bg" if new_fvf == "eg" else "eg"
            if new_fvf not in vol.variables and other in vol.variables and not vol.variables[other].is_constant:
                d = vol.variables[other]
                pts = {100: 1 / d.ppf(0.999), 90: 1 / d.p10, 50: 1 / d.p50, 10: 1 / d.p90, 0: 1 / d.ppf(0.001)}
                try:
                    vol.variables[new_fvf] = D.PercentileTable(pts)
                except D.DistributionError:
                    pass
            elif new_fvf not in vol.variables and other in vol.variables:
                vol.variables[new_fvf] = D.Constant(1.0 / vol.variables[other].p50)
            cfg.gas_fvf = new_fvf
            bump()
            st.rerun()
        for v in (cfg.gas_fvf, "rf_gas"):
            _var(path, vol, v, us)
        _var(path, vol, "gas_saturation", us, optional="Separate hydrocarbon saturation for the gas zone")
        _var(path, vol, "cgr", us, optional="Include condensate (CGR)")
        if "cgr" in vol.variables:
            _var(path, vol, "rf_condensate", us, optional="Separate recovery factor for condensate")
    if not cfg.has_oil and not cfg.has_gas:
        st.warning("Set at least one phase probability above zero.")
