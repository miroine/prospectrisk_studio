"""Map import, closure analysis, area-depth from maps and contours, cross-sections — against analytic geometry."""
import io
import sys

import numpy as np

from _harness import check, close, raises, run
from prospectrisk import distributions as D
from prospectrisk import maps as M
from prospectrisk import sections as S
from prospectrisk import units as U
from prospectrisk import volumetrics as V
from prospectrisk.examples import example_project
from prospectrisk.io_export import project_from_yaml, project_to_yaml

C = D.Constant


def single_dome(cell=25.0):
    n = int(8000 / cell) + 1
    return M.synthetic_dome(nx=n, ny=int(6000 / cell) + 1, cell=cell, second_crest=None)


def dome_area_m2(level, crest=2000.0):
    return np.pi * 1400.0 * 1000.0 * (level - crest) / 300.0


def test_irap_and_zmap_orientation():
    # hand-written IRAP: 3 x 3, x fastest from the south-west corner, z = x + 100 y
    irap = "-996 3 10 5\n0 20 0 10\n3 0 0 0\n0 0 0 0 0 0 0\n0 10 20 500 510 520 1000 1010 9999900\n"
    g = M.read_irap_ascii(irap)
    check("IRAP: south-west node", g.z[0, 0] == 0 and g.xy_of(0, 0) == (0.0, 0.0))
    check("IRAP: x varies along a row", g.z[0, 2] == 20)
    check("IRAP: y varies along a column", g.z[2, 0] == 1000 and g.xy_of(2, 0) == (0.0, 10.0))
    check("IRAP: undefined value read as NaN", np.isnan(g.z[2, 2]))
    # hand-written ZMAP+: columns from the top-left node downwards
    zmap = ("! comment\n@TEST, GRID, 3\n15, 1E+30, , 2, 1\n3, 3, 0.0, 20.0, 0.0, 10.0\n0.0, 0.0, 0.0\n@\n"
            "1000 500 0\n1010 510 10\n1E+30 520 20\n")
    z = M.read_zmap(zmap)
    check("ZMAP: same surface as IRAP", np.array_equal(np.nan_to_num(z.z), np.nan_to_num(g.z)), str(z.z))
    check("ZMAP: increments", z.dx == 10 and z.dy == 5)
    big = M.synthetic_dome(nx=31, ny=21, cell=100.0)
    big.z[3, 4] = np.nan
    for name, w, r in (("IRAP", M.write_irap_ascii, M.read_irap_ascii), ("ZMAP", M.write_zmap, M.read_zmap)):
        back = r(w(big))
        check(f"{name} round trip values", np.allclose(np.nan_to_num(back.z, nan=-1), np.nan_to_num(big.z, nan=-1), atol=1e-3))
        check(f"{name} round trip geometry", np.allclose([back.x0, back.y0, back.dx, back.dy], [big.x0, big.y0, big.dx, big.dy]))
    raises("non-IRAP text rejected", lambda: M.read_irap_ascii("1 2 3"), M.MapError)
    raises("ZMAP without header rejected", lambda: M.read_zmap("1 2 3"), M.MapError)
    raises("truncated IRAP rejected", lambda: M.read_irap_ascii(irap.rsplit(" ", 3)[0]), M.MapError)


def test_grid_area_vs_analytic():
    g = single_dome(25.0)
    crest = g.shallowest()
    d, a = M.grid_area_depth(g, crest, 2250.0, 26)
    a_m2 = a * U.ACRE_M2
    for lv in (2050.0, 2100.0, 2200.0):
        k = int(np.argmin(np.abs(d - lv)))
        close(f"dome area at {lv:g} m (node count vs π·a·b·h/300)", float(a_m2[k]), dome_area_m2(d[k]), rel=0.015)
    check("area starts at zero at the crest", a[0] == 0 and d[0] == 2000.0)
    check("area non-decreasing", np.all(np.diff(a) >= 0))


def test_spill_detection():
    for cell in (50.0, 25.0):
        n_x, n_y = int(6000 / cell) + 1, int(5000 / cell) + 1
        g = M.synthetic_dome(nx=n_x, ny=n_y, cell=cell)
        ci = M.find_spill(g, g.shallowest())
        # exact saddle of min(f1, f2) found on a fine sampling of the intersection curve
        x = np.linspace(0, 6000, 3001)
        y = np.linspace(0, 5000, 2501)
        X, Y = np.meshgrid(x, y)
        c1, c2, cy = g.x[g.nx // 3], g.x[int(g.nx * 0.8)], g.y[g.ny // 2]
        f1 = 2000 + 300 * (((X - c1) / 1400) ** 2 + ((Y - cy) / 1000) ** 2)
        f2 = 2100 + 300 * (((X - c2) / 1000) ** 2 + ((Y - cy) / 800) ** 2)
        m = np.abs(f1 - f2) < 0.5
        k = np.argmin(np.where(m, f1, np.inf))
        sad, sx = f1.ravel()[k], X.ravel()[k]
        tol = 0.4 * cell  # half a cell × ~0.6-0.8 m/m slope on each side of the ridge
        close(f"spill depth over saddle (cell {cell:g} m)", ci.spill_depth, sad, abs_=tol)
        close(f"spill location at the saddle (cell {cell:g} m)", ci.spill_xy[0], sx, abs_=2 * cell)
        check(f"saddle spill not flagged as map edge (cell {cell:g} m)", not ci.open_at_map_edge)
    single = single_dome(50.0)
    ci = M.find_spill(single, single.shallowest())
    x_left_edge = 2000 + 300 * ((single.x[single.nx // 3] - single.x[0]) / 1400) ** 2
    check("single dome spills at the map edge", ci.open_at_map_edge and abs(ci.spill_depth - x_left_edge) < 1e-6,
          f"{ci.spill_depth} vs {x_left_edge}")
    # crest picked near the second high climbs to that high
    g = M.synthetic_dome(nx=121, ny=101, cell=50.0)
    smap = M.StructureMap("grid", top=g, crest_xy=(g.x[int(g.nx * 0.8)] + 120, g.y[g.ny // 2] + 80))
    i, j = smap.crest_index(g)
    close("crest picker climbs to the second high", g.z[i, j], 2100.0, abs_=1e-9)
    hole = M.synthetic_dome(nx=61, ny=51, cell=100.0, second_crest=None)
    hole.z[25, 45:] = np.nan
    ci = M.find_spill(hole, hole.shallowest())
    check("undefined nodes act as an open boundary", ci.open_at_map_edge)


def test_map_to_grv_and_units():
    g = single_dome(25.0)
    smap = M.StructureMap("grid", top=g, n_levels=80)
    top, base, info = smap.area_depth()
    close("crest converted to ft", top.crest, U.to_internal(2000.0, "length", "m"), rel=1e-12)
    check("table ends at the spill depth", abs(top.base - U.to_internal(info.spill_depth, "length", "m")) < 1e-6)
    contact_m = 2150.0
    seg = V.SegmentVolumetrics(
        V.SegmentConfig("area_depth", "contact_depth", True, top),
        {"gross_thickness": C(U.to_internal(1000.0, "length", "m")),
         "contact_depth": C(U.to_internal(contact_m, "length", "m")),
         "net_to_gross": C(1), "porosity": C(1), "hc_saturation": C(1), "bo": C(1), "rf_oil": C(1)})
    r = V.deterministic(seg, V.constant_inputs(seg), "oil")
    grv_m3 = U.from_internal(r["grv_total"], "grv", "m3")
    want = np.pi * 1400 * 1000 * (contact_m - 2000.0) ** 2 / (2 * 300.0)
    close("GRV from the map vs analytic paraboloid", grv_m3, want, rel=0.02)
    smap_ft = M.StructureMap("grid", top=M.Grid(g.x0 / U.FT_M, g.y0 / U.FT_M, g.dx / U.FT_M, g.dy / U.FT_M,
                                                 g.z / U.FT_M, "ft", "ft"), n_levels=80)
    top_ft, _, _ = smap_ft.area_depth()
    close("same map in feet gives the same area", float(np.interp(top.crest + 300, top_ft.depth, top_ft.area)),
          float(np.interp(top.crest + 300, top.depth, top.area)), rel=1e-6)
    base = M.Grid(g.x0, g.y0, g.dx, g.dy, g.z + 40.0, "m", "m")
    smap_b = M.StructureMap("grid", top=g, base=base, n_levels=80)
    t2, b2, _ = smap_b.area_depth()
    seg_b = V.SegmentVolumetrics(V.SegmentConfig("area_depth", "contact_depth", True, t2, b2),
                                 {"contact_depth": C(U.to_internal(2150.0, "length", "m")), "net_to_gross": C(1),
                                  "porosity": C(1), "hc_saturation": C(1), "bo": C(1), "rf_oil": C(1)})
    seg_t = V.SegmentVolumetrics(V.SegmentConfig("area_depth", "contact_depth", True, t2),
                                 {"gross_thickness": C(U.to_internal(40.0, "length", "m")),
                                  "contact_depth": C(U.to_internal(2150.0, "length", "m")), "net_to_gross": C(1),
                                  "porosity": C(1), "hc_saturation": C(1), "bo": C(1), "rf_oil": C(1)})
    gb = V.deterministic(seg_b, V.constant_inputs(seg_b), "oil")["grv_total"]
    gt = V.deterministic(seg_t, V.constant_inputs(seg_t), "oil")["grv_total"]
    close("base grid 40 m below top = 40 m thickness model", gb, gt, rel=0.02)
    raises("mixed grid units rejected",
           lambda: M.StructureMap("grid", top=g, base=M.Grid(0, 0, 25, 25, g.z, "ft", "ft")).area_depth(), M.MapError)


def circle(depth, r, cx=0.0, cy=0.0, n=720):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return M.Contour(depth, (cx + r * np.cos(t)).tolist(), (cy + r * np.sin(t)).tolist())


def test_contours():
    cs = [circle(2050, 500), circle(2100, 800), circle(2150, 1000)]
    d, a, w = M.contour_area_depth(cs, 2000.0)
    close("circle polygon area", a[2], np.pi * 800 ** 2, rel=1e-4)
    check("crest row has zero area", d[0] == 2000 and a[0] == 0)
    raises("crest must be shallower than contours", lambda: M.contour_area_depth(cs, 2060.0), M.MapError)
    d2, a2, w2 = M.contour_area_depth([circle(2050, 500), circle(2100, 300)], 2000.0)
    check("decreasing area corrected with a warning", a2[2] == a2[1] and any("decreases" in x for x in w2))
    _, a3, w3 = M.contour_area_depth(cs + [circle(2150, 200, 5000, 5000)], 2000.0)
    check("two polygons at one depth summed with a warning",
          abs(a3[3] - np.pi * (1000 ** 2 + 200 ** 2)) / a3[3] < 1e-3 and any("share" in x for x in w3))
    text = "id,depth,x,y\n" + "\n".join(f"{k},{c.depth},{x},{y}" for k, c in enumerate(cs) for x, y in zip(c.x, c.y))
    back = M.read_contours(text)
    check("contour CSV with header and id", len(back) == 3 and back[1].depth == 2100 and back[1].n == 720)
    plain = "\n".join(f"{c.depth} {x} {y}" for c in cs for x, y in zip(c.x, c.y))
    check("contour file without id splits on depth change", len(M.read_contours(plain)) == 3)
    raises("polygon with two depths rejected", lambda: M.read_contours("id,depth,x,y\n1,5,0,0\n1,6,1,0\n1,5,1,1\n"),
           M.MapError)
    smap = M.StructureMap("contours", contours=cs, crest_depth=2000.0, crest_xy=(0.0, 0.0), xy_unit="m", z_unit="m")
    top, _, info = smap.area_depth()
    close("contour table in acres", top.area[-1], np.pi * 1000 ** 2 / U.ACRE_M2, rel=1e-4)
    close("deepest contour used as spill", info.spill_depth, 2150.0)
    g = smap.top_grid()
    check("contours interpolated to a surface for sections", g is not None and abs(np.nanmin(g.z) - 2000.0) < 1.0)


def test_points_and_image():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 1000, (400, 2))
    pts = np.column_stack([xy, 2000 + 0.1 * xy[:, 0] - 0.05 * xy[:, 1]])
    text = "x y z\n" + "\n".join(f"{a:.3f}, {b:.3f}; {c:.4f}" for a, b, c in pts)
    arr = M.read_xyz(text)
    check("XYZ reader skips header, mixed separators", arr.shape == (400, 3))
    g = M.grid_from_points(arr, cell=20.0)
    zz = 2000 + 0.1 * g.x[None, :] - 0.05 * g.y[:, None]
    ok = np.isfinite(g.z)
    check("linear gridding reproduces a plane", ok.mean() > 0.8 and np.allclose(g.z[ok], zz[ok], atol=1e-3))
    raises("too few points rejected", lambda: M.grid_from_points(np.zeros((2, 3))), M.MapError)
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (3000, 1500), (240, 240, 240)).save(buf, format="PNG")
    im = M.load_image(buf.getvalue(), 450000, 460000, 6700000, 6705000)
    check("large images downscaled", im.width == 1400 and im.height == 700)
    wx, wy = im.world(700, 350)
    close("pixel centre maps to world centre (x)", wx, 455000.0, rel=1e-12)
    close("pixel centre maps to world centre (y)", wy, 6702500.0, rel=1e-12)
    px, py = im.pixel(451000, 6704000)
    close("world to pixel round trip", im.world(px, py)[1], 6704000.0, rel=1e-12)
    raises("bad image bytes rejected", lambda: M.load_image(b"notapng", 0, 1, 0, 1), M.MapError)


def _area_depth_segment(phase="oil", contact_m=2150.0, gcf=0.4):
    g = single_dome(50.0)
    smap = M.StructureMap("grid", top=g, n_levels=60)
    top, _, info = smap.area_depth()
    probs = {"oil": 0.0, "gas": 0.0, "oil_gascap": 0.0}
    probs[phase] = 1.0
    seg = V.SegmentVolumetrics(
        V.SegmentConfig("area_depth", "contact_depth", True, top, phase_probabilities=probs),
        {"gross_thickness": C(U.to_internal(60.0, "length", "m")),
         "contact_depth": D.Triangular(*(U.to_internal(v, "length", "m") for v in (2080.0, contact_m, 2220.0))),
         "gas_cap_fraction": C(gcf), "net_to_gross": C(1), "porosity": C(0.2), "hc_saturation": C(0.8),
         "bo": C(1.2), "rf_oil": C(0.3), "bg": C(0.004), "rf_gas": C(0.7)})
    return smap, seg


def test_sections():
    smap, seg = _area_depth_segment("oil_gascap")
    p90, p50, p10 = (S.case_from_percentile(seg, p) for p in ("P90", "P50", "P10"))
    check("P90 contact shallower than P10", p90.contact < p50.contact < p10.contact)
    close("gas-oil contact at crest + fraction × column", p50.goc, p50.crest + 0.4 * (p50.contact - p50.crest), rel=1e-12)
    geo = S.synthetic_section(seg, p50)
    mid = len(geo.x) // 2
    close("synthetic section crest at x = 0", geo.top[mid], p50.crest, rel=1e-9)
    total = sum(lo - up for up, lo in geo.bands.values())
    check("gas + oil + water fill the whole reservoir", np.allclose(total, geo.base - geo.top))
    hc_mid = geo.hc_thickness()[mid]
    close("HC thickness at crest = min(column, thickness)", hc_mid, min(p50.column, p50.thickness), rel=1e-9)
    far = geo.top > p50.contact
    check("no hydrocarbons where the top is below the contact", np.all(geo.hc_thickness()[far] == 0))
    oil_only = S.SectionCase("x", "oil", p50.contact, None, p50.thickness, p50.spill, p50.crest)
    check("oil case has no gas band", np.all(S.synthetic_section(seg, oil_only).bands["gas"][1]
                                             == S.synthetic_section(seg, oil_only).bands["gas"][0]))
    gas_case = S.SectionCase("x", "gas", p50.contact, p50.contact, p50.thickness, p50.spill, p50.crest)
    gg = S.synthetic_section(seg, gas_case)
    check("gas case has no oil band", np.allclose(gg.bands["oil"][0], gg.bands["oil"][1]))
    # map section along the crest row
    ms = S.map_section(smap, seg, p50)
    g = smap.top
    i, j = smap.crest_index(g)
    close("map section passes through the crest (within sampling)", float(np.nanmin(ms.top)),
          U.to_internal(2000.0, "length", "m"), abs_=1.0)
    res = V.simulate(seg, 2000, 5)
    k = S.trial_near_percentile(res, "P50")
    tc = S.case_from_trial(seg, res, k)
    close("trial case uses that trial's contact", tc.contact, float(res.outputs["contact"][k]))
    pv = S.plan_view(smap, p50)
    area_hc = pv.hc.sum() * g.cell_area_acres
    close("plan-view HC area = area-depth table at the contact", area_hc,
          float(np.interp(p50.contact, seg.config.top_table.depth, seg.config.top_table.area)), rel=0.03)
    two = M.StructureMap("grid", top=M.synthetic_dome(nx=121, ny=101, cell=50.0), n_levels=60)
    t2, _, info2 = two.area_depth()
    seg2 = V.SegmentVolumetrics(V.SegmentConfig("area_depth", "contact_depth", True, t2),
                                {"gross_thickness": C(200.0), "contact_depth": C(t2.base - 1.0), "net_to_gross": C(1),
                                 "porosity": C(0.2), "hc_saturation": C(0.8), "bo": C(1.2), "rf_oil": C(0.3)})
    case2 = S.case_from_percentile(seg2, "P50")
    sec2 = S.map_section(two, seg2, case2)
    x_ft = sec2.x + U.to_internal(two.top.x0, "length", "m")
    second_high = x_ft > U.to_internal(4600.0, "length", "m")
    shallow = sec2.top < case2.contact
    check("second high above the contact holds no hydrocarbons (not connected to this crest)",
          np.any(second_high & shallow) and np.all(sec2.hc_thickness()[second_high & shallow] == 0))
    oil_seg = V.SegmentVolumetrics(V.SegmentConfig("area_thickness"), {})
    raises("sections need area-depth", lambda: S.case_from_percentile(oil_seg), S.SectionError)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    S.render_section_mpl(ax, geo, "m")
    plt.close(fig)
    check("static section renders", True)


def test_map_serialisation_in_project():
    p = example_project()
    seg = p.prospect("Alpha").segment("Lower sand")
    g = M.synthetic_dome(nx=61, ny=51, cell=100.0)
    g.z[0, 0] = np.nan
    seg.structure_map = M.StructureMap("grid", top=g, contours=[circle(2100, 300)], crest_depth=2000.0,
                                       crest_xy=(1000.0, 2500.0), section=[0, 2500, 6000, 2500])
    back = project_from_yaml(project_to_yaml(p)).prospect("Alpha").segment("Lower sand").structure_map
    check("map survives YAML round trip", back is not None and back.section == [0, 2500, 6000, 2500]
          and np.allclose(np.nan_to_num(back.top.z, nan=-9), np.nan_to_num(g.z.astype(np.float32), nan=-9)))
    check("contours survive YAML round trip", len(back.contours) == 1 and back.contours[0].n == 720)
    size = len(project_to_yaml(p))
    check("compressed grid keeps the project file small", size < 200_000, f"{size} bytes")


if __name__ == "__main__":
    sys.exit(run("test_maps_sections", [test_irap_and_zmap_orientation, test_grid_area_vs_analytic,
                                        test_spill_detection, test_map_to_grv_and_units, test_contours,
                                        test_points_and_image, test_sections, test_map_serialisation_in_project]))
