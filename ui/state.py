"""Session state, result caching and run helpers."""
from __future__ import annotations

import hashlib
import json

import streamlit as st

from prospectrisk import qc
from prospectrisk.defaults import ensure_required
from prospectrisk.examples import example_project
from prospectrisk.project import Project, run_portfolio, run_prospect
from ui.editors import bump


def init() -> None:
    ss = st.session_state
    if "project" not in ss:
        ss.project = example_project()
        ss.runs = {}
        ss.run_fp = {}
        ss.portfolio = None
        ss.portfolio_fp = None
        ss.qc_thresholds = qc.QCThresholds()


def project() -> Project:
    return st.session_state.project


def replace_project(p: Project) -> None:
    ss = st.session_state
    ss.project = p
    ss.runs, ss.run_fp, ss.portfolio, ss.portfolio_fp = {}, {}, None, None
    for k in [k for k in ss.keys() if str(k).startswith("w_") or str(k) in ("unit_system_sel", "xlsx", "pdf")]:
        del ss[k]
    bump()


def _md5(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def prospect_fp(p: Project, name: str) -> str:
    pr = p.prospect(name)
    d = p.to_dict()
    return _md5([d["settings"], d["adequacy_matrix"], p.play(pr.play).to_dict(), pr.to_dict()])


def portfolio_fp(p: Project, names: list[str]) -> str:
    return _md5([prospect_fp(p, n) for n in names] + [names, p.portfolio_dependency, p.portfolio_volume_correlation])


def is_stale(name: str) -> bool:
    p = project()
    try:
        return st.session_state.run_fp.get(name) != prospect_fp(p, name)
    except ValueError:
        return True


def run_one(name: str) -> bool:
    p = project()
    try:
        for seg in p.prospect(name).segments:
            ensure_required(seg.volumetrics)
        st.session_state.runs[name] = run_prospect(p, name)
        st.session_state.run_fp[name] = prospect_fp(p, name)
        return True
    except ValueError as exc:
        st.error(f"Could not run prospect '{name}': {exc}")
        return False


def run_all() -> None:
    p = project()
    with st.spinner(f"Simulating {len(p.prospects)} prospects × {p.settings.n_trials:,} trials"):
        for pr in p.prospects:
            run_one(pr.name)


def run_portfolio_now(names: list[str]) -> None:
    p = project()
    for n in names:
        if n not in st.session_state.runs or is_stale(n):
            if not run_one(n):
                return
    try:
        st.session_state.portfolio = run_portfolio(p, names, {n: st.session_state.runs[n] for n in names})
        st.session_state.portfolio_fp = portfolio_fp(p, names)
        st.session_state.portfolio_names = list(names)
    except ValueError as exc:
        st.error(f"Could not aggregate the portfolio: {exc}")


def valid_runs() -> dict:
    """Runs for prospects that still exist (possibly stale)."""
    names = {pr.name for pr in project().prospects}
    return {k: v for k, v in st.session_state.runs.items() if k in names}
