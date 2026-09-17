"""ProspectRisk Studio — exploration prospect volumes and chance of success.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="ProspectRisk Studio", page_icon="🎯", layout="wide")

from prospectrisk.units import UNIT_SYSTEMS  # noqa: E402
from ui import state  # noqa: E402
from ui.editors import bump, unit_customiser  # noqa: E402
from ui.page_analysis import page_portfolio, page_results, page_sensitivity  # noqa: E402
from ui.page_model import page_project, page_prospects  # noqa: E402
from ui.page_output import page_export, page_method, page_qc  # noqa: E402

CSS = """
<style>
:root { --navy:#00243D; --red:#EB0037; --karry:#FFE7D6; --moss:#007079; --mist:#D5EAF4; }
html, body, [data-testid="stAppViewContainer"] {
  font-family: Equinor, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: var(--navy);
}
h1 { font-weight: 600; letter-spacing: -0.01em; color: var(--navy); }
h2, h3, h4 { color: var(--navy); font-weight: 600; }
[data-testid="stSidebar"] { background: var(--navy); }
[data-testid="stSidebar"] * { color: #FFFFFF; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--navy); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary * { color: #FFFFFF; }
[data-testid="stSidebar"] .brand { font-size: 1.25rem; font-weight: 600; line-height: 1.2; margin-bottom: 0.1rem; }
[data-testid="stSidebar"] .brand span { display:block; font-size: 0.8rem; font-weight: 400; opacity: 0.75; }
[data-testid="stMetric"] { border-left: 3px solid var(--red); padding: 0.25rem 0 0.25rem 0.75rem; }
[data-testid="stMetricValue"] { color: var(--navy); font-weight: 600; }
.risk-chip { display:inline-block; padding: 0.15rem 0.6rem; border-radius: 1rem; color: #FFFFFF;
             font-size: 0.85rem; font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #E6E6E6; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--red); }
</style>
"""

PAGES = {
    "Project": page_project,
    "Prospects": page_prospects,
    "Results": page_results,
    "Sensitivity": page_sensitivity,
    "Portfolio": page_portfolio,
    "Quality control": page_qc,
    "Export": page_export,
    "Method": page_method,
}


def sidebar() -> str:
    p = state.project()
    with st.sidebar:
        st.markdown("<div class='brand'>ProspectRisk Studio<span>Volumes and chance of success</span></div>",
                    unsafe_allow_html=True)
        st.caption(p.name)
        page = st.radio("Go to", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        systems = list(UNIT_SYSTEMS)
        new_sys = st.selectbox("Units", systems, index=systems.index(p.settings.unit_system), key="unit_system_sel")
        if new_sys != p.settings.unit_system:
            p.settings.unit_system = new_sys
            p.settings.unit_overrides.clear()
            bump()
            st.rerun()
        n_custom = len(p.settings.unit_overrides) + len(p.settings.input_units)
        with st.expander(f"Customise units ({n_custom} custom)" if n_custom else "Customise units"):
            if unit_customiser(p.settings):
                bump()
                st.rerun()
            st.caption("Individual inputs can also use their own unit from the unit box next to each input.")
            if n_custom and st.button("Reset to preset", key="reset_units", width="stretch"):
                p.settings.unit_overrides.clear()
                p.settings.input_units.clear()
                bump()
                st.rerun()
        n_runs = len(state.valid_runs())
        st.caption(f"{n_runs} of {len(p.prospects)} prospects simulated")
        if st.button("Run all prospects", key="sidebar_run", width="stretch"):
            state.run_all()
        st.divider()
        st.caption("Engineering decision support. Not affiliated with Equinor or SLB. MIT licence.")
    return page


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    state.init()
    page = sidebar()
    PAGES[page]()


main()
