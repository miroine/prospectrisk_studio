"""Headless smoke test of the Streamlit UI.

Streamlit and Plotly are replaced by permissive stubs so every page function runs
end-to-end against the real engine. This catches NameErrors, wrong call
signatures into the engine, unit plumbing and data-frame handling. It does not
check layout or real widget behaviour: run `streamlit run app.py` for that.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from _harness import check, run  # noqa: E402


class Anything:
    """Absorbs any attribute access / call; usable as a context manager."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return Anything()

    def __call__(self, *a, **k):
        return Anything()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([])


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]


class Rerun(Exception):
    pass


def make_streamlit(script):
    st = types.ModuleType("streamlit")
    st.session_state = SessionState()
    st.column_config = Anything()
    calls = {"plots": 0, "errors": [], "buttons": set()}
    st._calls = calls

    def _widget(default):
        def f(label="", *a, key=None, **k):
            if key is not None and key in script.get("values", {}):
                return script["values"][key]
            return default(label, a, k)
        return f

    def selectbox(label, options, index=0, key=None, format_func=None, **k):
        options = list(options)
        forced = script.get("selectbox", {}).get(label)
        if forced is not None:
            val = forced(options) if callable(forced) else forced
            if val in options:
                return val
        return options[index] if options else None

    def radio(label, options, index=0, key=None, **k):
        options = list(options)
        if label == "Go to":
            return script.get("page", options[0])
        forced = script.get("radio", {}).get(label)
        return forced if forced in options else options[index]

    def button(label, key=None, **k):
        calls["buttons"].add(key)
        return key in script.get("press", set())

    def number_input(label, *a, value=None, key=None, **k):
        if value is None:
            value = a[2] if len(a) >= 3 else (a[0] if a else 0.0)
        return value

    def text_input(label, value="", key=None, **k):
        return value

    def checkbox(label, value=False, key=None, **k):
        return script.get("checkbox", {}).get(label, value)

    def multiselect(label, options, default=None, key=None, **k):
        return list(default or [])

    def data_editor(df, key=None, **k):
        return df.copy()

    def columns(spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [make_block() for _ in range(n)]

    def tabs(names):
        return [make_block() for _ in names]

    def error(msg, *a, **k):
        calls["errors"].append(str(msg))

    def plotly_chart(fig, *a, **k):
        calls["plots"] += 1

    def rerun():
        raise Rerun()

    def make_block():
        b = Anything()
        for name, fn in funcs.items():
            setattr(b, name, fn)
        return b

    funcs = dict(selectbox=selectbox, radio=radio, button=button, number_input=number_input, text_input=text_input,
                 text_area=text_input, checkbox=checkbox, multiselect=multiselect, data_editor=data_editor,
                 columns=columns, tabs=tabs, error=error, plotly_chart=plotly_chart)
    for name, fn in funcs.items():
        setattr(st, name, fn)
    st.rerun = rerun
    for name in ("set_page_config", "markdown", "title", "caption", "info", "warning", "success", "dataframe",
                 "metric", "download_button", "divider", "subheader", "code"):
        setattr(st, name, lambda *a, **k: None)
    st.file_uploader = lambda *a, **k: None
    st.expander = lambda *a, **k: make_block()
    st.container = lambda *a, **k: make_block()
    st.spinner = lambda *a, **k: Anything()
    st.sidebar = make_block()
    return st


def install(script):
    st = make_streamlit(script)
    sys.modules["streamlit"] = st
    plotly = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")

    class Figure(Anything):
        pass

    go.Figure = Figure
    for n in ("Histogram", "Scatter", "Bar"):
        setattr(go, n, Anything)
    plotly.graph_objects = go
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = go
    for m in [m for m in list(sys.modules) if m == "app" or m.startswith("ui")]:
        del sys.modules[m]
    return st


def run_page(st, page_fn, tries=4):
    for _ in range(tries):
        try:
            page_fn()
            return
        except Rerun:
            continue


PAGES = ["Project", "Prospects", "Results", "Sensitivity", "Portfolio", "Quality control", "Export", "Method"]


def test_all_pages_demo_project():
    st = install({})
    from ui import state
    from ui import page_model, page_analysis, page_output
    state.init()
    st.session_state.qc_thresholds  # noqa: B018
    state.run_all()
    check("run_all produced results for both prospects", len(st.session_state.runs) == 2, str(st.session_state.runs.keys()))
    fns = {"Project": page_model.page_project, "Prospects": page_model.page_prospects,
           "Results": page_analysis.page_results, "Sensitivity": page_analysis.page_sensitivity,
           "Portfolio": page_analysis.page_portfolio, "Quality control": page_output.page_qc,
           "Export": page_output.page_export, "Method": page_output.page_method}
    for prospect in ("Alpha", "Bravo"):
        st.session_state.sel_prospect = prospect
        for name, fn in fns.items():
            before = len(st._calls["errors"])
            try:
                run_page(st, fn)
                ok = True
                detail = ""
            except Exception as exc:  # noqa: BLE001
                import traceback
                ok, detail = False, traceback.format_exc()
            check(f"page {name} ({prospect}) runs", ok, detail)
            new_err = st._calls["errors"][before:]
            check(f"page {name} ({prospect}) shows no error messages", not new_err, "; ".join(new_err))
    check("charts were produced", st._calls["plots"] > 10, str(st._calls["plots"]))


def test_segment_editor_all_segments_and_kinds():
    from prospectrisk.volumetrics import GRV_METHODS, CONTACT_MODES
    kinds = ["constant", "uniform", "triangular", "pert", "normal", "normal_p90p10", "lognormal",
             "lognormal_p90p10", "beta", "discrete", "percentile_table"]
    for kind in kinds:
        for method in GRV_METHODS:
            for mode in CONTACT_MODES:
                script = {"selectbox": {"Distribution": kind, "GRV method": method, "Contact defined by": mode},
                          "checkbox": {"Include condensate (CGR)": True, "Include solution gas (GOR)": True,
                                       "Truncate": kind in ("normal", "lognormal")}}
                st = install(script)
                from ui import state
                from ui import page_model
                state.init()
                p = state.project()
                errs = []
                for pr in p.prospects:
                    for si, seg in enumerate(pr.segments):
                        st.session_state.sel_prospect = pr.name
                        st.session_state[f"sel_segment_{[x.name for x in p.prospects].index(pr.name)}"] = seg.name
                        try:
                            run_page(st, page_model.page_prospects)
                        except Exception as exc:  # noqa: BLE001
                            import traceback
                            errs.append(traceback.format_exc())
                check(f"segment editor: {kind} / {method} / {mode}", not errs, errs[0] if errs else "")
                ui_errors = [e for e in st._calls["errors"] if "Phase probabilities" not in e]
                check(f"no editor errors: {kind} / {method} / {mode}", not ui_errors, "; ".join(ui_errors[:3]))
                ok_run = True
                try:
                    state.run_all()
                except Exception:  # noqa: BLE001
                    ok_run = False
                check(f"edited project still simulates: {kind} / {method} / {mode}",
                      ok_run and len(st.session_state.runs) == 2, "; ".join(st._calls["errors"][:3]))
                if method != "area_depth":
                    break


def test_app_entry_and_unit_switch():
    st = install({"page": "Results", "selectbox": {"Units": "Field"}})
    import importlib
    try:
        importlib.import_module("app")
        ok, detail = True, ""
    except Rerun:
        ok, detail = True, ""
    except Exception:  # noqa: BLE001
        import traceback
        ok, detail = False, traceback.format_exc()
    check("app.py imports and renders", ok, detail)
    check("unit system switched to Field", st.session_state.project.settings.unit_system == "Field")


def test_button_actions():
    for press, page in ((("add_pros", "dup_pros", "add_play"), "model"), (("add_seg0", "dup_seg0"), "model"),
                        (("pf_run",), "portfolio"), (("prep_xlsx", "prep_pdf"), "export"),
                        (("del_seg0",), "model"), (("del_pros",), "model"), (("new_proj",), "project"),
                        (("load_demo",), "project"), (("reset_matrix", "proj_run_all"), "project"),
                        (("qc_run",), "qc")):
        st = install({"press": set(press)})
        from ui import state, page_model, page_analysis, page_output
        state.init()
        if page != "qc":
            state.run_all()
        fn = {"model": page_model.page_prospects, "portfolio": page_analysis.page_portfolio,
              "export": page_output.page_export, "project": page_model.page_project, "qc": page_output.page_qc}[page]
        try:
            run_page(st, fn, tries=2)
            if page == "model":
                run_page(st, page_model.page_prospects, tries=1)
            state.run_all()
            ok, detail = True, ""
        except Exception:  # noqa: BLE001
            import traceback
            ok, detail = False, traceback.format_exc()
        check(f"buttons {press} work", ok and not st._calls["errors"], detail + "; ".join(st._calls["errors"][:3]))
        if "pf_run" in press:
            check("portfolio aggregated", st.session_state.portfolio is not None)
        if "prep_pdf" in press:
            check("PDF prepared", st.session_state.get("pdf", b"")[:4] == b"%PDF")
        if "add_pros" in press:
            check("prospect added and duplicated", len(state.project().prospects) >= 4)


def test_new_controls():
    variants = [
        {"selectbox": {"Defined by": "Formula", "Unit": lambda o: o[-1], "Uncertainty": "add"},
         "checkbox": {"Add uncertainty to the formula result": True, "Minimum volume": True, "Maximum volume": True},
         "radio": {"Trials below the minimum": "renormalise", "Gas volume factor entered as": "eg"}},
        {"selectbox": {"Defined by": "Formula", "Uncertainty": "multiply", "Area": "ha", "Oil equivalent": "MMboe",
                       "Depth & thickness": "ft", "Fractions (N/G, porosity, saturation, RF)": "%"},
         "checkbox": {"Add uncertainty to the formula result": True, "Minimum volume": True},
         "radio": {"Trials below the minimum": "chance"}},
        {"selectbox": {"Unit": lambda o: o[0]}, "checkbox": {"Maximum volume": True}},
    ]
    for vi, script in enumerate(variants):
        st = install(script)
        from ui import state, page_model, page_analysis, page_output
        from prospectrisk.aggregation import VolumeTruncation
        state.init()
        p = state.project()
        alpha = p.prospect("Alpha")
        alpha.input_dependencies = [
            {"seg_a": "Upper sand", "var_a": "bo", "seg_b": "Lower sand", "var_b": "bo", "kind": "link", "rho": 0.0},
            {"seg_a": "Upper sand", "var_a": "porosity", "seg_b": "Lower sand", "var_b": "porosity",
             "kind": "correlation", "rho": 0.6}]
        errs = []
        try:
            import app  # noqa: F401  (sidebar + unit customiser)
        except Rerun:
            pass
        except Exception:  # noqa: BLE001
            import traceback
            errs.append(traceback.format_exc())
        for pi, pr in enumerate(p.prospects):
            for seg in pr.segments:
                st.session_state.sel_prospect = pr.name
                st.session_state[f"sel_segment_{pi}"] = seg.name
                try:
                    run_page(st, page_model.page_prospects, tries=80)  # each unit change reruns once
                except Exception:  # noqa: BLE001
                    import traceback
                    errs.append(traceback.format_exc())
        check(f"variant {vi}: editors run", not errs, errs[0] if errs else "")
        exprs = sum(len(sg.volumetrics.expressions) for pr in p.prospects for sg in pr.segments)
        if "Formula" in str(script["selectbox"].get("Defined by")):
            check(f"variant {vi}: formula mode stored expressions", exprs > 0, str(exprs))
        if script["checkbox"].get("Minimum volume"):
            check(f"variant {vi}: minimum volume stored",
                  all(sg.truncation.minimum is not None for pr in p.prospects for sg in pr.segments))
        if script.get("radio", {}).get("Gas volume factor entered as") == "eg":
            check(f"variant {vi}: E input selected for gas segments",
                  all(sg.volumetrics.config.gas_fvf == "eg" for pr in p.prospects for sg in pr.segments
                      if sg.volumetrics.config.has_gas))
        check(f"variant {vi}: dependencies kept", len(p.prospect("Alpha").input_dependencies) == 2,
              str(p.prospect("Alpha").input_dependencies))
        # maximum defaults to min + 1 unit which may reject everything; use a generous maximum for the run
        for pr in p.prospects:
            for sg in pr.segments:
                if sg.truncation.maximum is not None:
                    sg.truncation.maximum = 1e12
                if sg.truncation.minimum is not None:
                    sg.truncation.minimum = 1e5
        st._calls["errors"].clear()
        state.run_all()
        check(f"variant {vi}: project with new features simulates", len(st.session_state.runs) == 2,
              "; ".join(st._calls["errors"][:3]))
        for fn in (page_analysis.page_results, page_analysis.page_sensitivity, page_analysis.page_portfolio,
                   page_output.page_qc, page_output.page_export):
            try:
                run_page(st, fn)
                ok, detail = True, ""
            except Exception:  # noqa: BLE001
                import traceback
                ok, detail = False, traceback.format_exc()
            check(f"variant {vi}: {fn.__name__} runs", ok, detail)
        check(f"variant {vi}: no error messages on analysis pages", not st._calls["errors"],
              "; ".join(st._calls["errors"][:3]))
        if vi == 1:
            check("unit overrides applied from sidebar", p.settings.unit("area") == "ha" and p.settings.unit("oe") == "MMboe",
                  str(p.settings.unit_overrides))


if __name__ == "__main__":
    sys.exit(run("smoke_ui", [test_all_pages_demo_project, test_segment_editor_all_segments_and_kinds,
                              test_app_entry_and_unit_switch, test_button_actions,
                              test_new_controls]))
