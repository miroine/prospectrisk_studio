"""Headless smoke test of the Streamlit UI.

Streamlit and Plotly are replaced by permissive stubs so every page function runs
end-to-end against the real engine. This catches NameErrors, wrong call
signatures into the engine, unit plumbing and data-frame handling. It does not
check layout or real widget behaviour: run `streamlit run app.py` for that.
"""
from __future__ import annotations

import sys

import numpy as np
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
        calls.setdefault("button_labels", []).append(label)
        if k.get("disabled"):
            return False
        if key in script.get("press", set()):
            return True
        for pref in script.get("press_labels", []):
            # like a real click: true on one run only, whatever the widget key becomes afterwards
            if str(label).startswith(pref) and pref not in calls.setdefault("pressed_once", set()):
                calls["pressed_once"].add(pref)
                return True
        return False

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
        if k.get("on_select") and script.get("events"):
            return script["events"].pop(0)
        return None

    def rerun():
        raise Rerun()

    def slider(label, lo=None, hi=None, value=None, *a, **k):
        return value if value is not None else lo

    def make_block():
        b = Anything()
        for name, fn in funcs.items():
            setattr(b, name, fn)
        return b

    funcs = dict(selectbox=selectbox, radio=radio, button=button, number_input=number_input, text_input=text_input,
                 text_area=text_input, checkbox=checkbox, multiselect=multiselect, data_editor=data_editor,
                 columns=columns, tabs=tabs, error=error, plotly_chart=plotly_chart, slider=slider)
    for name, fn in funcs.items():
        setattr(st, name, fn)
    st.rerun = rerun
    for name in ("set_page_config", "markdown", "title", "caption", "info", "warning", "success", "dataframe",
                 "metric", "download_button", "divider", "subheader", "code"):
        setattr(st, name, lambda *a, **k: None)
    class Upload:
        def __init__(self, name, data):
            self.name, self._data = name, data

        def getvalue(self):
            return self._data

    def file_uploader(label, *a, **k):
        item = script.get("uploads", {}).get(label)
        return Upload(*item) if item else None

    st.file_uploader = file_uploader
    st.expander = lambda *a, **k: make_block()
    st.container = lambda *a, **k: make_block()
    st.spinner = lambda *a, **k: Anything()
    st.sidebar = make_block()
    return st


# Real Plotly trace types and the top-level properties the app uses (all valid in Plotly 5).
TRACE_PROPS = {
    "Scatter": {"x", "y", "mode", "name", "line", "marker", "fill", "fillcolor", "hoverinfo", "hovertemplate",
                "legendgroup", "showlegend", "text", "textposition", "opacity", "customdata"},
    "Bar": {"x", "y", "orientation", "name", "marker", "text", "textposition", "hovertemplate", "base", "customdata",
            "opacity"},
    "Histogram": {"x", "name", "marker", "opacity", "nbinsx", "histnorm", "hovertemplate"},
    "Heatmap": {"x", "y", "z", "zmin", "zmax", "zmid", "showscale", "opacity", "hoverinfo", "colorscale", "name",
                "text", "texttemplate", "colorbar", "hovertemplate"},
    "Contour": {"x", "y", "z", "colorscale", "reversescale", "contours", "line", "colorbar", "hovertemplate", "name",
                "showscale"},
    "Splom": {"dimensions", "showupperhalf", "diagonal", "marker"},
}
TRACE_PROPS["Scattergl"] = TRACE_PROPS["Scatter"]
FIGURE_METHODS = {"add_trace", "add_hline", "add_vline", "add_hrect", "add_vrect", "add_layout_image",
                  "update_layout", "update_xaxes", "update_yaxes"}


def _trace_class(kind):
    allowed = TRACE_PROPS[kind]

    class Trace:
        def __init__(self, *a, **k):
            for key in k:
                if key.split("_")[0] not in allowed:
                    raise TypeError(f"go.{kind} has no property '{key}' (as used by the app)")
    Trace.__name__ = kind
    return Trace


class StrictFigure:
    def __init__(self, data=None, *a, **k):
        pass

    def __getattr__(self, name):
        if name not in FIGURE_METHODS:
            raise AttributeError(f"plotly Figure has no method '{name}' (in the app's usage)")
        return lambda *a, **k: None


def install(script):
    st = make_streamlit(script)
    sys.modules["streamlit"] = st
    plotly = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    go.Figure = StrictFigure
    for n in TRACE_PROPS:
        setattr(go, n, _trace_class(n))
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
    alpha_fp = state.prospect_fp(state.project(), "Alpha")
    state.project().prospect("Bravo").segments[0].notes = "unrelated change"
    check("prospect fingerprint ignores unrelated prospects",
          state.prospect_fp(state.project(), "Alpha") == alpha_fp)
    state.project().prospect("Alpha").segments[0].notes = "relevant change"
    check("prospect fingerprint tracks its own model",
          state.prospect_fp(state.project(), "Alpha") != alpha_fp)
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


def _png_bytes(w=800, h=600):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (235, 235, 230)).save(buf, format="PNG")
    return buf.getvalue()


def _page_ok(st, fn, label, tries=40):
    before = len(st._calls["errors"])
    try:
        run_page(st, fn, tries)
        ok, detail = True, ""
    except Exception:  # noqa: BLE001
        import traceback
        ok, detail = False, traceback.format_exc()
    check(f"{label}: runs", ok, detail)
    errs = st._calls["errors"][before:]
    check(f"{label}: no error messages", not errs, "; ".join(errs[:3]))


def test_maps_page():
    from prospectrisk import maps as M
    # empty segment, every tab
    st = install({})
    from ui import state, page_maps
    state.init()
    st.session_state.sel_prospect = "Alpha"
    _page_ok(st, page_maps.page_maps, "maps page without a map")

    # demonstration surface -> closure -> apply to segment -> sections
    st = install({"radio": {"Source": "demo"},
                  "press_labels": ["Load demonstration surface", "Use this area–depth table"]})
    from ui import state, page_maps
    state.init()
    p = state.project()
    st.session_state.sel_prospect = "Alpha"
    seg = p.prospect("Alpha").segments[0]
    _page_ok(st, page_maps.page_maps, "demo surface loaded and applied")
    check("demo map attached", seg.structure_map is not None and seg.structure_map.top is not None)
    cfg = seg.volumetrics.config
    check("segment switched to area-depth from the map", cfg.grv_method == "area_depth" and cfg.top_table is not None)
    check("spill depth set from the map", "spill_depth" in seg.volumetrics.variables
          and seg.volumetrics.variables["spill_depth"].is_constant)
    st._calls["errors"].clear()
    state.run_all()
    check("project with the map runs", len(st.session_state.runs) == 2, "; ".join(st._calls["errors"][:3]))
    for sec in ("map", "synthetic"):
        for case in ("pct", "trial"):
            st2 = install({"radio": {"Section": sec, "Case": case}})
            from ui import state as state2, page_maps as pm2
            st2.session_state.update(st.session_state)
            _page_ok(st2, pm2.page_maps, f"cross-section {sec} / {case}")

    # grid files, points and contours through the uploader
    dome = M.synthetic_dome(nx=81, ny=61, cell=80.0)
    rng = np.random.default_rng(0)
    ii, jj = rng.integers(0, dome.ny, 1500), rng.integers(0, dome.nx, 1500)
    xyz = "x y z\n" + "\n".join(f"{dome.x[j]} {dome.y[i]} {dome.z[i, j]}" for i, j in zip(ii, jj))
    circles = "id,depth,x,y\n" + "\n".join(
        f"{k},{2000 + 50 * k},{3000 + (300 + 200 * k) * np.cos(t)},{2500 + (300 + 200 * k) * np.sin(t)}"
        for k in range(1, 4) for t in np.linspace(0, 2 * np.pi, 40, endpoint=False))
    cases = [("grid", "top.irap", M.write_irap_ascii(dome), {"top"}), ("grid", "top.zmap", M.write_zmap(dome), {"top"}),
             ("points", "pts.xyz", xyz, {"points"}), ("contours", "cont.csv", circles, {"contours"})]
    for src, fname, text, _ in cases:
        st = install({"radio": {"Source": src}, "uploads": {"File": (fname, text.encode())},
                      "press_labels": ["Use this area–depth table"]})
        from ui import state, page_maps
        state.init()
        st.session_state.sel_prospect = "Alpha"
        _page_ok(st, page_maps.page_maps, f"upload {fname}")
        sm = state.project().prospect("Alpha").segments[0].structure_map
        check(f"{fname}: map stored with source {src}", sm is not None and sm.source == src,
              str(sm.source if sm else None))
        check(f"{fname}: applied as area-depth",
              state.project().prospect("Alpha").segments[0].volumetrics.config.grv_method == "area_depth")
    # base grid after top grid
    st = install({"radio": {"Source": "grid", "Surface": "base"},
                  "uploads": {"File": ("base.irap", M.write_irap_ascii(M.Grid(dome.x0, dome.y0, dome.dx, dome.dy,
                                                                                dome.z + 50, "m", "m")).encode())}})
    from ui import state, page_maps
    state.init()
    st.session_state.sel_prospect = "Alpha"
    state.project().prospect("Alpha").segments[0].structure_map = M.StructureMap("grid", top=dome)
    _page_ok(st, page_maps.page_maps, "base grid upload")
    check("base grid stored", state.project().prospect("Alpha").segments[0].structure_map.base is not None)

    # digitising: image, three clicks, save
    clicks = [{"selection": {"points": [{"x": x, "y": y, "curve_number": 0}]}}
              for x, y in ((1000.0, 1000.0), (3000.0, 1000.0), (2000.0, 3000.0))]
    st = install({"uploads": {"Map image": ("map.png", _png_bytes())}, "events": clicks,
                  "press_labels": ["Use this image", "Save contour at"]})
    from ui import state, page_maps
    state.init()
    st.session_state.sel_prospect = "Alpha"
    _page_ok(st, page_maps.page_maps, "digitising", tries=40)
    sm = state.project().prospect("Alpha").segments[0].structure_map
    check("image stored", sm is not None and sm.image is not None)
    check("clicked contour saved with three vertices", sm is not None and len(sm.contours) == 1
          and sm.contours[0].n == 3, str(sm.contours if sm else None))
    check("crest set above the contour", sm is not None and sm.crest_depth is not None
          and sm.crest_depth < sm.contours[0].depth)


def test_explore_page():
    from prospectrisk import explore as E
    st = install({"press_labels": ["Keep trials inside the box"]})
    from ui import state, page_explore
    state.init()
    state.run_all()
    p = state.project()
    st.session_state.sel_prospect = "Alpha"
    df, meta = E.trial_table(p, st.session_state.runs["Alpha"])
    phase = [c for c, m in meta.items() if m.kind == "phase" and m.segment == "Lower sand"][0]
    st.session_state["filters:Alpha"] = [{"column": phase, "action": "keep", "low": None, "high": None,
                                          "values": ["Oil"]}]
    area = [c for c, m in meta.items() if m.key == "area"][0]
    st.session_state["_events_placeholder"] = None
    st2 = install({"press_labels": ["Keep trials inside the box"],
                   "events": [{"selection": {"box": [{"x": [9.0, 14.0], "y": [0.0, 1e9]}], "points": []}}]})
    from ui import page_explore as pe2
    st2.session_state.update(st.session_state)
    _page_ok(st2, pe2.page_explore, "explore page with a phase filter and a box selection", tries=10)
    filt = st2.session_state["filters:Alpha"]
    check("box selection added two keep filters", len(filt) == 3 and filt[1]["column"] == area, str(filt))
    mask = E.apply_filters(df, [E.TrialFilter(f["column"], f["action"], f["low"], f["high"], f["values"]) for f in filt])
    check("filtered set respects the box and the phase", mask.any() and df.loc[mask, area].between(9, 14).all()
          and (df.loc[mask, phase] == "Oil").all())
    st3 = install({})
    from ui import page_explore as pe3
    st3.session_state.update(st2.session_state)
    st3.session_state["filters:Alpha"] = [{"column": phase, "action": "keep", "low": None, "high": None,
                                           "values": ["Nonexistent"]}]
    _page_ok(st3, pe3.page_explore, "explore page when no trial passes")


def test_benchmark_page():
    from prospectrisk import recovery as RF
    st = install({"press_labels": ["Add this permeability to the segment", "Use as the oil recovery factor"],
                  "uploads": {"Analogue table": ("a.csv", RF.ANALOG_TEMPLATE.encode())}})
    from ui import state, page_benchmark
    state.init()
    p = state.project()
    st.session_state.sel_prospect = "Alpha"
    seg = p.prospect("Alpha").segments[0]
    _page_ok(st, page_benchmark.page_benchmark, "benchmark page (oil)", tries=10)
    check("permeability added to the segment", "permeability" in seg.volumetrics.variables)
    check("correlation applied as the oil RF formula", "rf_oil" in seg.volumetrics.expressions
          and "permeability" in seg.volumetrics.expressions["rf_oil"].expr)
    check("benchmark parameters saved on the segment", seg.rf_benchmark is not None and "mu_o" in seg.rf_benchmark)
    st._calls["errors"].clear()
    state.run_all()
    check("prospect runs with the RF formula", "Alpha" in st.session_state.runs, "; ".join(st._calls["errors"][:3]))
    res = st.session_state.runs["Alpha"].segments[seg.name].result
    from scipy import stats as _s
    check("oil RF now follows permeability", _s.spearmanr(res.inputs["permeability"], res.inputs["rf_oil"]).statistic > 0.4)
    st2 = install({"selectbox": {"Segment": "Lower sand"}})
    from ui import page_benchmark as pb2
    st2.session_state.update(st.session_state)
    _page_ok(st2, pb2.page_benchmark, "benchmark page with a run and a gas segment")
    from prospectrisk.io_export import project_from_yaml, project_to_yaml
    back = project_from_yaml(project_to_yaml(p)).prospect("Alpha").segments[0]
    check("RF formula and benchmark parameters saved in the project file",
          "rf_oil" in back.volumetrics.expressions and back.rf_benchmark == seg.rf_benchmark)


def test_new_pages_through_app():
    for page in ("Maps & sections", "Explore trials", "RF benchmark"):
        st = install({"page": page})
        import importlib
        try:
            importlib.import_module("app")
            ok, detail = True, ""
        except Rerun:
            ok, detail = True, ""
        except Exception:  # noqa: BLE001
            import traceback
            ok, detail = False, traceback.format_exc()
        check(f"app renders '{page}'", ok, detail)
        check(f"app '{page}' shows no errors", not st._calls["errors"], "; ".join(st._calls["errors"][:3]))


def test_more_branches():
    from prospectrisk import maps as M
    # digitising: two clicks then undo -> vertex table shown, nothing saved
    clicks = [{"selection": {"points": [{"x": 1000.0, "y": 1000.0}]}}, {"selection": {"points": [{"x": 2000.0, "y": 1500.0}]}}]
    st = install({"uploads": {"Map image": ("map.png", _png_bytes())}, "events": clicks,
                  "press_labels": ["Use this image"]})
    from ui import state, page_maps
    state.init()
    st.session_state.sel_prospect = "Alpha"
    _page_ok(st, page_maps.page_maps, "digitising in progress")
    pts = [v for k, v in st.session_state.items() if str(k).startswith("dig_pts:")]
    check("two vertices held while digitising", pts and len(pts[0]) == 2, str(pts))
    st2 = install({"press_labels": ["Undo last point"]})
    from ui import page_maps as pm2
    st2.session_state.update(st.session_state)
    _page_ok(st2, pm2.page_maps, "undo a vertex")
    pts = [v for k, v in st2.session_state.items() if str(k).startswith("dig_pts:")]
    check("undo removes the last vertex", pts and len(pts[0]) == 1, str(pts))
    # closure with a picked crest near the second high, then section line buttons
    st = install({"radio": {"Crest": "pick", "Section": "map"}, "press_labels": ["Use this area–depth table",
                                                                              "South–north through crest"]})
    from ui import state, page_maps
    state.init()
    st.session_state.sel_prospect = "Alpha"
    g = M.synthetic_dome(nx=121, ny=101, cell=50.0)
    seg = state.project().prospect("Alpha").segments[0]
    seg.structure_map = M.StructureMap("grid", top=g, crest_xy=(g.x[int(g.nx * 0.8)] + 60, g.y[g.ny // 2]))
    _page_ok(st, page_maps.page_maps, "picked crest, applied, section line buttons")
    tab = seg.volumetrics.config.top_table
    check("picked crest used (second high at 2100 m)", tab is not None and abs(tab.crest - 2100 / 0.3048) < 1e-6,
          str(tab.crest if tab else None))
    sec = seg.structure_map.section
    check("south–north line through the crest", sec is not None and abs(sec[0] - sec[2]) < 1e-9, str(sec))
    # explore: colour by a variable, log axes, no trend
    st = install({"selectbox": {"Colour": lambda o: o[2]}, "checkbox": {"Log X": True, "Log Y": True,
                                                                        "Median trend and P90–P10 band": False}})
    from ui import state, page_explore
    state.init()
    state.run_all()
    st.session_state.sel_prospect = "Bravo"
    _page_ok(st, page_explore.page_explore, "explore single-segment prospect, colour and log axes")
    # benchmark: manual permeability range, no run, gas tab on a gas segment
    st = install({})
    from ui import state, page_benchmark
    state.init()
    st.session_state.sel_prospect = "Bravo"
    _page_ok(st, page_benchmark.page_benchmark, "benchmark on a gas-only prospect")
    st.session_state.sel_prospect = "Alpha"
    _page_ok(st, page_benchmark.page_benchmark, "benchmark with a manual permeability range")


if __name__ == "__main__":
    sys.exit(run("smoke_ui", [test_all_pages_demo_project, test_segment_editor_all_segments_and_kinds,
                              test_app_entry_and_unit_switch, test_button_actions,
                              test_new_controls, test_maps_page, test_explore_page, test_benchmark_page,
                              test_new_pages_through_app, test_more_branches]))
