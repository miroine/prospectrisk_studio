"""Maps & sections: load or digitise a structure map, derive area-depth, draw filled cross-sections."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from prospectrisk import distributions as D
from prospectrisk import maps as M
from prospectrisk import sections as S
from prospectrisk import units as U
from prospectrisk.defaults import ensure_required
from prospectrisk.volumetrics import OUTPUTS, PHASE_LABELS, PHASES
from ui import charts as C
from ui import state
from ui.editors import bump, frozen_base, num, selected_points as _selected_points, wkey

SOURCE_OPTIONS = {
    "grid": "Grid file (IRAP classic ASCII or ZMAP+)",
    "points": "Scattered points (x, y, z)",
    "contours": "Contour file (depth, x, y)",
    "demo": "Demonstration surface",
}


# =============================================================================================
# helpers
# =============================================================================================

def _pick_segment(p):
    names = [pr.name for pr in p.prospects]
    if not names:
        st.info("Add a prospect first.")
        return None, None
    c1, c2 = st.columns(2)
    default = st.session_state.get("sel_prospect")
    pname = c1.selectbox("Prospect", names, index=names.index(default) if default in names else 0, key="maps_prospect")
    st.session_state.sel_prospect = pname
    pr = p.prospect(pname)
    segs = [s.name for s in pr.segments]
    sname = c2.selectbox("Segment", segs, key=wkey("maps_segment", pname))
    return pr, pr.segment(sname)


def _digest(data: bytes, *extra) -> str:
    return hashlib.md5(data + json.dumps(extra, default=str).encode()).hexdigest()


def _map_key(smap: M.StructureMap) -> str:
    parts = [smap.source, smap.xy_unit, smap.z_unit, smap.n_levels, smap.min_relief, smap.crest_xy, smap.crest_depth,
             [c.to_dict() for c in smap.contours]]
    for g in (smap.top, smap.base):
        parts.append(None if g is None else hashlib.md5(np.ascontiguousarray(g.z).tobytes()).hexdigest()
                     + str((g.x0, g.y0, g.dx, g.dy)))
    return hashlib.md5(json.dumps(parts, default=str).encode()).hexdigest()


def _closure(smap: M.StructureMap, spill_override: float | None):
    """Area-depth tables and closure info, cached per map state."""
    cache = st.session_state.setdefault("_closure_cache", {})
    key = _map_key(smap) + str(spill_override)
    if key not in cache:
        if len(cache) > 20:
            cache.clear()
        cache[key] = smap.area_depth(spill_override)
    return cache[key]


def _units_row(prefix: str, xy: str = "m", z: str = "m") -> tuple[str, str]:
    c1, c2 = st.columns(2)
    xy_u = c1.selectbox("Map coordinates in", M.LENGTH_UNITS, index=M.LENGTH_UNITS.index(xy), key=wkey(prefix, "xyu"))
    z_u = c2.selectbox("Depths in", M.LENGTH_UNITS, index=M.LENGTH_UNITS.index(z), key=wkey(prefix, "zu"))
    return xy_u, z_u


# =============================================================================================
# page
# =============================================================================================

def page_maps() -> None:
    p = state.project()
    st.title("Maps & sections")
    pr, seg = _pick_segment(p)
    if seg is None:
        return
    path = f"map:{pr.name}:{seg.name}"
    t_load, t_dig, t_close, t_sec = st.tabs(["Load map", "Digitise contours", "Closure & area–depth", "Cross-section"])
    with t_load:
        _load_tab(path, seg)
    with t_dig:
        _digitise_tab(path, seg)
    with t_close:
        _closure_tab(p, path, pr, seg)
    with t_sec:
        _section_tab(p, path, pr, seg)


# ---- load --------------------------------------------------------------------------------------

def _load_tab(path: str, seg) -> None:
    smap = seg.structure_map
    if smap is not None:
        _map_summary(smap)
        if st.button("Remove map from this segment", key=wkey(path, "rm_map")):
            seg.structure_map = None
            bump()
            st.rerun()
        st.divider()
    src = st.radio("Source", list(SOURCE_OPTIONS), format_func=SOURCE_OPTIONS.get, horizontal=True,
                   key=wkey(path, "src"))
    xy_u, z_u = _units_row(path + ":load", smap.xy_unit if smap else "m", smap.z_unit if smap else "m")
    negative = st.checkbox("Depth values are negative (elevations); convert to positive depths",
                           key=wkey(path, "neg"))
    if src == "demo":
        st.caption("Two elliptical highs joined by a saddle; the closure around the western high spills over "
                   "the saddle. Useful for trying the workflow.")
        if st.button("Load demonstration surface", key=wkey(path, "demo")):
            seg.structure_map = M.StructureMap("grid", top=M.synthetic_dome(noise=4.0), n_levels=60)
            bump()
            st.rerun()
        return
    role = "top"
    if src == "grid":
        role = st.radio("Surface", ["top", "base"], format_func={"top": "Top reservoir", "base": "Base reservoir"}.get,
                        horizontal=True, key=wkey(path, "role"),
                        help="A base grid replaces the gross-thickness model: GRV is the rock between the two surfaces.")
    cell = None
    if src == "points":
        cell = num(f"Grid cell size [{xy_u}] (0 = automatic)", 0.0, wkey(path, "cell"), min_value=0.0) or None
    types = {"grid": ["txt", "asc", "irap", "irapasc", "dat", "zmap", "grd", "gri"], "points": ["txt", "csv", "xyz", "dat"],
             "contours": ["txt", "csv", "dat"]}[src]
    up = st.file_uploader("File", type=types, key=wkey(path, "file", src))
    if up is None:
        st.caption({"grid": "IRAP classic ASCII (header starts with -996) or ZMAP+ (header between @ lines) are "
                            "detected automatically. Binary IRAP is not supported; export as ASCII.",
                    "points": "Any text file whose rows start with x, y, z; header lines are skipped.",
                    "contours": "Rows of depth, x, y or id, depth, x, y. With a header, use columns named "
                                "x, y, depth (or z) and optionally id."}[src])
        return
    data = up.getvalue()
    dig = _digest(data, src, xy_u, z_u, negative, role, cell)
    if st.session_state.get(f"loaded:{path}") == dig:
        st.success(f"Loaded {up.name}.")
        return
    try:
        new = _ingest(seg.structure_map, src, data, up.name, xy_u, z_u, negative, role, cell)
    except (M.MapError, ValueError) as exc:
        st.error(str(exc))
        return
    seg.structure_map = new
    st.session_state[f"loaded:{path}"] = dig
    bump()
    st.rerun()


def _ingest(current, src, data, fname, xy_u, z_u, negative, role, cell) -> M.StructureMap:
    sign = -1.0 if negative else 1.0
    if src == "grid":
        text = M._decode(data)
        head = text.lstrip()[:400]
        g = M.read_irap_ascii(text, xy_u, z_u, fname) if head.startswith("-996") else \
            M.read_zmap(text, xy_u, z_u, fname) if "@" in head else None
        if g is None:
            raise M.MapError("Unrecognised grid format: expected IRAP classic ASCII (-996 header) or ZMAP+ (@ header)")
        g.z = g.z * sign
        if role == "base":
            if current is None or current.top is None:
                raise M.MapError("Load the top-structure grid before the base grid")
            if (current.top.xy_unit, current.top.z_unit) != (xy_u, z_u):
                raise M.MapError("Base and top grids must use the same units")
            current.base = g
            return current
        return M.StructureMap("grid", top=g, xy_unit=xy_u, z_unit=z_u,
                              base=None, image=current.image if current else None)
    if src == "points":
        xyz = M.read_xyz(data)
        xyz[:, 2] *= sign
        g = M.grid_from_points(xyz, cell, xy_u, z_u, fname)
        return M.StructureMap("points", top=g, xy_unit=xy_u, z_unit=z_u, image=current.image if current else None)
    cs = M.read_contours(data)
    for c in cs:
        c.depth *= sign
    crest = min(c.depth for c in cs) - (np.diff(sorted({c.depth for c in cs})).min()
                                        if len({c.depth for c in cs}) > 1 else 10.0)
    return M.StructureMap("contours", contours=cs, crest_depth=float(crest), xy_unit=xy_u, z_unit=z_u,
                          image=current.image if current else None)


def _map_summary(smap: M.StructureMap) -> None:
    g = smap.top_grid()
    cols = st.columns(4)
    cols[0].metric("Source", M.SOURCES.get(smap.source, smap.source))
    if smap.source == "contours":
        cols[1].metric("Contours", len(smap.contours))
        depths = [c.depth for c in smap.contours]
        cols[2].metric(f"Depth range [{smap.z_unit}]", f"{min(depths):g}–{max(depths):g}" if depths else "–")
        cols[3].metric(f"Crest [{smap.z_unit}]", C.fmt(smap.crest_depth))
    elif g is not None:
        cols[1].metric("Nodes", f"{g.nx} × {g.ny}")
        cols[2].metric(f"Depth range [{g.z_unit}]", f"{np.nanmin(g.z):.0f}–{np.nanmax(g.z):.0f}")
        cols[3].metric("Base surface", "Grid" if smap.base is not None else "Thickness")
        if abs(g.rotation) > 1e-9:
            st.caption(f"The grid is rotated {g.rotation:g}°. Areas and volumes are unaffected; the plan view is "
                       "drawn unrotated.")


# ---- digitise ----------------------------------------------------------------------------------------

def _digitise_tab(path: str, seg) -> None:
    smap = seg.structure_map
    img = smap.image if smap else None
    with st.expander("Map image and its coordinates", expanded=img is None):
        st.caption("Upload a picture of the depth map (PNG or JPG) and enter the map coordinates of its left, right, "
                   "bottom and top edges. Contours you click are stored in those coordinates.")
        c = st.columns(4)
        xmin = c[0].number_input("Left x", value=float(img.xmin) if img else 0.0, key=wkey(path, "ix0"), format="%.6g")
        xmax = c[1].number_input("Right x", value=float(img.xmax) if img else 5000.0, key=wkey(path, "ix1"), format="%.6g")
        ymin = c[2].number_input("Bottom y", value=float(img.ymin) if img else 0.0, key=wkey(path, "iy0"), format="%.6g")
        ymax = c[3].number_input("Top y", value=float(img.ymax) if img else 4000.0, key=wkey(path, "iy1"), format="%.6g")
        xy_u, z_u = _units_row(path + ":dig", smap.xy_unit if smap else "m", smap.z_unit if smap else "m")
        up = st.file_uploader("Map image", type=["png", "jpg", "jpeg"], key=wkey(path, "img"))
        b1, b2 = st.columns(2)
        if b1.button("Use this image", key=wkey(path, "use_img"), disabled=up is None, width="stretch"):
            try:
                im = M.load_image(up.getvalue(), xmin, xmax, ymin, ymax)
            except M.MapError as exc:
                st.error(str(exc))
            else:
                if smap is None:
                    seg.structure_map = M.StructureMap("contours", xy_unit=xy_u, z_unit=z_u, image=im)
                else:
                    smap.image = im
                    if smap.source == "contours":
                        smap.xy_unit, smap.z_unit = xy_u, z_u
                bump()
                st.rerun()
        if img is not None and b2.button("Update coordinates only", key=wkey(path, "img_ext"), width="stretch"):
            if xmax > xmin and ymax > ymin:
                img.xmin, img.xmax, img.ymin, img.ymax = xmin, xmax, ymin, ymax
                bump()
                st.rerun()
            else:
                st.error("Right must exceed left and top must exceed bottom.")
    if img is None:
        st.info("Load a map image to start digitising, or load a contour file on the Load map tab.")
        return
    if smap.source != "contours" and smap.top is not None:
        st.warning("This segment currently uses an imported grid. Saving a digitised contour switches the "
                   "segment's map to contours (the grid is kept and can be restored below).")
    pts_key = f"dig_pts:{path}"  # stable across widget refreshes so a half-digitised contour is kept
    pts: list[tuple[float, float]] = st.session_state.setdefault(pts_key, [])
    c1, c2, c3 = st.columns([2, 2, 3])
    depth = c1.number_input(f"Depth of this contour [{smap.z_unit}]", value=float(
        st.session_state.get(f"last_depth:{path}", 2000.0)), key=wkey(path, "cdepth"), format="%.6g")
    grid_n = c2.slider("Click precision", 40, 180, 100, 10, key=wkey(path, "gridn"),
                       help="Number of clickable positions across the image. Higher is more precise but slower.")
    c3.caption("Click the map to add a vertex. The click snaps to the nearest grid position; edit the coordinates "
               "in the table for exact values. Save the contour when it goes all the way round.")
    fig = C.digitise_figure(img, [c for c in smap.contours], pts, grid_n)
    ev = st.plotly_chart(fig, key=wkey(path, "digfig", len(pts), grid_n), on_select="rerun",
                         selection_mode=("points",), width="stretch")
    new = _selected_points(ev)
    if len(new) == 1 and (not pts or new[0] != pts[-1]):
        pts.append(new[0])
        st.session_state[pts_key] = pts
        st.rerun()
    elif len(new) > 1:
        st.info("Several points were selected; click single positions to add vertices.")
    b = st.columns(4)
    if b[0].button("Undo last point", key=wkey(path, "undo"), disabled=not pts, width="stretch"):
        pts.pop()
        st.rerun()
    if b[1].button("Clear points", key=wkey(path, "clear"), disabled=not pts, width="stretch"):
        st.session_state[pts_key] = []
        st.rerun()
    if b[2].button(f"Save contour at {depth:g}", key=wkey(path, "save_c"), type="primary", disabled=len(pts) < 3,
                   width="stretch"):
        smap.contours.append(M.Contour(depth, [q[0] for q in pts], [q[1] for q in pts]))
        smap.source = "contours"
        if smap.crest_depth is None or smap.crest_depth >= min(c.depth for c in smap.contours):
            smap.crest_depth = float(min(c.depth for c in smap.contours)) - 10.0
        st.session_state[pts_key] = []
        st.session_state[f"last_depth:{path}"] = depth
        bump()
        st.rerun()
    if smap.source == "contours" and smap.top is not None and b[3].button("Use the imported grid again",
                                                                          key=wkey(path, "back_grid"), width="stretch"):
        smap.source = "grid"
        bump()
        st.rerun()
    if pts:
        ek = wkey(path, "pts_table", len(pts))
        ed = st.data_editor(frozen_base(ek, lambda: pd.DataFrame(pts, columns=["x", "y"])), key=ek,
                            num_rows="dynamic", hide_index=True, width="stretch")
        edited = [(float(a), float(b_)) for a, b_ in ed.dropna().itertuples(index=False)]
        if edited != pts:
            st.session_state[pts_key] = edited
    _contour_list(path, smap)


def _contour_list(path: str, smap: M.StructureMap) -> None:
    if not smap.contours:
        return
    area_u = "km2" if smap.xy_unit == "m" else "acres"
    f = 1e-6 if smap.xy_unit == "m" else 1 / 43560.0
    df = pd.DataFrame([{"#": k + 1, f"Depth [{smap.z_unit}]": c.depth, "Vertices": c.n,
                        f"Area [{area_u}]": c.area() * f} for k, c in enumerate(smap.contours)])
    st.markdown("**Saved contours**")
    st.dataframe(df, hide_index=True, width="stretch")
    c1, c2, c3 = st.columns([2, 2, 2])
    drop = c1.multiselect("Delete contours", df["#"].tolist(), key=wkey(path, "drop_c"))
    if drop and c2.button("Delete selected", key=wkey(path, "drop_btn"), width="stretch"):
        smap.contours = [c for k, c in enumerate(smap.contours) if k + 1 not in drop]
        bump()
        st.rerun()
    shallowest = min(c.depth for c in smap.contours)
    cd = num(f"Crest depth [{smap.z_unit}]", smap.crest_depth if smap.crest_depth is not None else shallowest - 10,
             wkey(path, "crest_d"), help="Depth of the structural crest, where the enclosed area is zero.")
    if cd < shallowest:
        smap.crest_depth = cd
    else:
        st.error(f"The crest must be shallower than the shallowest contour ({shallowest:g}).")


# ---- closure -------------------------------------------------------------------------------------------

def _closure_tab(p, path: str, pr, seg) -> None:
    smap = seg.structure_map
    if smap is None:
        st.info("Load or digitise a map first.")
        return
    try:
        smap.validate()
    except M.MapError as exc:
        st.info(str(exc))
        return
    g = smap.top_grid()
    spill_override = None
    with st.expander("Closure settings", expanded=False):
        if smap.source in ("grid", "points") and g is not None:
            mode = st.radio("Crest", ["auto", "pick"], horizontal=True, index=1 if smap.crest_xy else 0,
                            format_func={"auto": "Shallowest point on the map", "pick": "High nearest to a location"}.get,
                            key=wkey(path, "crest_mode"))
            if mode == "pick":
                cx0, cy0 = smap.crest_xy or g.xy_of(*g.shallowest())
                c1, c2 = st.columns(2)
                smap.crest_xy = (c1.number_input("x", value=float(cx0), key=wkey(path, "cx"), format="%.6g"),
                                 c2.number_input("y", value=float(cy0), key=wkey(path, "cy"), format="%.6g"))
            else:
                smap.crest_xy = None
            relief = float(np.nanmax(g.z) - np.nanmin(g.z))
            smap.min_relief = num(f"Ignore highs with less relief than [{smap.z_unit}]",
                                  smap.min_relief if smap.min_relief is not None else round(0.005 * relief, 3),
                                  wkey(path, "relief"), min_value=0.0,
                                  help="Small bumps from noise or gridding are not treated as separate highs "
                                       "when searching for the spill point.")
        smap.n_levels = int(st.slider("Depth levels in the area–depth table", 20, 200, int(smap.n_levels), 10,
                                      key=wkey(path, "nlev")))
        if st.checkbox("Enter the spill depth myself", key=wkey(path, "spill_on")):
            spill_override = num(f"Spill depth [{smap.z_unit}]", 0.0, wkey(path, "spill_val"))
    try:
        top, base, info = _closure(smap, spill_override)
    except (M.MapError, ValueError) as exc:
        st.error(str(exc))
        return
    zu, lu, au = smap.z_unit, p.settings.unit("length"), p.settings.unit("area")
    m = st.columns(4)
    m[0].metric(f"Crest [{zu}]", C.fmt(info.crest_depth))
    m[1].metric(f"Spill [{zu}]", C.fmt(info.spill_depth) if info.spill_depth is not None else "Below map")
    m[2].metric(f"Maximum column [{zu}]", C.fmt((info.spill_depth - info.crest_depth) if info.spill_depth else np.nan))
    m[3].metric(f"Closure area [{au}]", C.fmt(U.from_internal(top.area[-1], "area", au)))
    for w in info.warnings:
        st.caption(w)
    if g is not None:
        case = S.SectionCase("Closure to spill", "oil", top.base, None, None, top.base, top.crest)
        try:
            pv = S.plan_view(smap, case, info.spill_xy)
            st.plotly_chart(C.plan_figure(pv, smap.xy_unit, smap.z_unit), width="stretch",
                            key=wkey(path, "plan_closure"))
            st.caption("Green: the closure filled to the spill point. Star: crest. Cross: spill point.")
        except (M.MapError, S.SectionError) as exc:
            st.caption(str(exc))
    d = U.from_internal(np.asarray(top.depth), "length", lu)
    a = U.from_internal(np.asarray(top.area), "area", au)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(C.area_depth_fill(a, d, None, None, au, lu), width="stretch", key=wkey(path, "ad_fill"))
    with c2:
        st.dataframe(pd.DataFrame({f"Depth [{lu}]": d, f"Area [{au}]": a}), hide_index=True, width="stretch", height=300)
    cfg = seg.volumetrics.config
    in_use = cfg.grv_method == "area_depth" and cfg.top_table is not None and \
        np.allclose(cfg.top_table.depth, top.depth) and np.allclose(cfg.top_table.area, top.area)
    if in_use:
        st.success("This segment uses this area–depth table.")
    if st.button("Use this area–depth table in the segment", type="primary", key=wkey(path, "apply"),
                 disabled=in_use):
        msgs = _apply_to_segment(seg, top, base, info, smap)
        for msg in msgs:
            st.info(msg)
        bump()
        st.rerun()


def _apply_to_segment(seg, top, base, info, smap) -> list[str]:
    vol = seg.volumetrics
    cfg = vol.config
    msgs = []
    cfg.grv_method = "area_depth"
    cfg.top_table = top
    cfg.base_table = base
    if info.spill_depth is not None:
        vol.variables["spill_depth"] = D.Constant(U.to_internal(info.spill_depth, "length", smap.z_unit))
        vol.expressions.pop("spill_depth", None)
    ensure_required(vol)
    spill = top.base
    for k in ("contact_depth",):
        d = vol.variables.get(k)
        if d is not None and (d.p10 < top.crest or d.p90 > spill + (spill - top.crest)):
            vol.variables[k] = D.Triangular(top.crest + 0.3 * (spill - top.crest), top.crest + 0.6 * (spill - top.crest),
                                            spill)
            msgs.append("The contact depth distribution lay outside the new structure and was reset between crest and "
                        "spill; review it on the Prospects page.")
    return msgs


# ---- sections --------------------------------------------------------------------------------------------

def _section_tab(p, path: str, pr, seg) -> None:
    vol = seg.volumetrics
    lu = p.settings.unit("length")
    if vol.config.grv_method != "area_depth" or vol.config.top_table is None:
        st.info("Cross-sections need the area–depth GRV method. Derive a table from a map on the Closure tab, or "
                "enter one on the Prospects page.")
        return
    smap = seg.structure_map
    has_map = smap is not None and smap.top_grid() is not None
    c1, c2, c3 = st.columns(3)
    kinds = (["map"] if has_map else []) + ["synthetic"]
    kind = c1.radio("Section", kinds, format_func={"map": "Along a line on the map",
                                                    "synthetic": "Idealised (from area–depth)"}.get,
                    key=wkey(path, "sec_kind"))
    source = c2.radio("Case", ["pct", "trial"], format_func={"pct": "Input percentiles",
                                                            "trial": "A Monte Carlo trial"}.get,
                      key=wkey(path, "sec_src"))
    phases = [ph for ph in PHASES if vol.config.phase_probabilities.get(ph, 0) > 0]
    run = st.session_state.runs.get(pr.name)
    try:
        if source == "pct":
            pct = c3.selectbox("Percentile", list(S.PERCENTILES), index=1, key=wkey(path, "sec_pct"),
                               help="P90 is the low case: shallow contact, thin reservoir.")
            phase = st.selectbox("Fluid", phases, format_func=PHASE_LABELS.get, key=wkey(path, "sec_phase"))
            case = S.case_from_percentile(vol, pct, phase)
        else:
            if run is None or seg.name not in run.segments:
                st.info("Run the prospect to draw individual trials.")
                return
            if state.is_stale(pr.name):
                st.warning("The last run is out of date with the inputs.")
            res = run.segments[seg.name].result
            how = c3.selectbox("Trial", ["P90", "P50", "P10", "number"], index=1, key=wkey(path, "sec_trial"),
                               format_func=lambda v: "By number" if v == "number" else f"Closest to recoverable {v}")
            if how == "number":
                idx = int(st.number_input("Trial number", 1, res.n, 1, key=wkey(path, "sec_tn"))) - 1
            else:
                idx = S.trial_near_percentile(res, how)
            case = S.case_from_trial(vol, res, idx)
            rec_q, rec_u = OUTPUTS["rec_oe"][1], p.settings.unit("oe")
            st.caption(f"Trial {idx + 1}: recoverable {C.fmt(U.from_internal(res.outputs['rec_oe'][idx], rec_q, rec_u))} "
                       f"{rec_u}.")
        line = None
        if kind == "map":
            g = smap.top_grid()
            line = _line_controls(path, smap, g)
            geom = S.map_section(smap, vol, case, line)
        else:
            geom = S.synthetic_section(vol, case)
    except (S.SectionError, M.MapError, ValueError) as exc:
        st.error(str(exc))
        return
    st.plotly_chart(C.section_figure(geom, lu, geom.case.label), width="stretch", key=wkey(path, "sec_fig"))
    for n_ in geom.notes:
        st.caption(n_)
    _case_metrics(p, vol, case, lu)
    if kind == "map":
        try:
            info = _closure(smap, None)[2]
            pv = S.plan_view(smap, case, info.spill_xy)
            pv.line = line
            st.plotly_chart(C.plan_figure(pv, smap.xy_unit, smap.z_unit), width="stretch", key=wkey(path, "sec_plan"))
            st.caption("Plan view of the same case: red gas, green oil, section line A–A′.")
        except (M.MapError, S.SectionError, ValueError) as exc:
            st.caption(str(exc))
    with st.expander("Compare P90, P50 and P10"):
        cols = st.columns(3)
        for col, pct in zip(cols, S.PERCENTILES):
            try:
                cc = S.case_from_percentile(vol, pct, case.phase)
                gg = S.map_section(smap, vol, cc, line) if kind == "map" else S.synthetic_section(vol, cc)
                with col:
                    st.plotly_chart(C.section_figure(gg, lu, pct, height=300, show_legend=False), width="stretch",
                                    key=wkey(path, "cmp", pct))
            except (S.SectionError, M.MapError, ValueError) as exc:
                col.caption(str(exc))


def _line_controls(path: str, smap: M.StructureMap, g: M.Grid) -> list[float]:
    line = smap.section or S.default_line(smap, g)
    i, j = smap.crest_index(g)
    cx, cy = g.xy_of(i, j)
    x1, x2, y1, y2 = g.extent
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("West–east through crest", key=wkey(path, "we"), width="stretch"):
        smap.section = [x1, cy, x2, cy]
        bump()
        st.rerun()
    if b2.button("South–north through crest", key=wkey(path, "sn"), width="stretch"):
        smap.section = [cx, y1, cx, y2]
        bump()
        st.rerun()
    c = st.columns(4)
    new = [c[0].number_input("A x", value=float(line[0]), key=wkey(path, "lx1"), format="%.6g"),
           c[1].number_input("A y", value=float(line[1]), key=wkey(path, "ly1"), format="%.6g"),
           c[2].number_input("A′ x", value=float(line[2]), key=wkey(path, "lx2"), format="%.6g"),
           c[3].number_input("A′ y", value=float(line[3]), key=wkey(path, "ly2"), format="%.6g")]
    smap.section = new
    return new


def _case_metrics(p, vol, case: S.SectionCase, lu: str) -> None:
    f = lambda v: C.fmt(U.from_internal(v, "length", lu)) if v is not None else "–"  # noqa: E731
    m = st.columns(4)
    m[0].metric(f"Contact [{lu}]", f(case.contact))
    m[1].metric(f"Gas–oil contact [{lu}]", f(case.goc) if case.phase == "oil_gascap" else "–")
    m[2].metric(f"Column [{lu}]", f(case.column))
    top = vol.config.top_table
    au = p.settings.unit("area")
    m[3].metric(f"Area at contact [{au}]",
                C.fmt(U.from_internal(float(np.interp(case.contact, top.depth, top.area)), "area", au)))
