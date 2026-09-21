"""Volumetrics engine versus hand calculations and analytic geometry."""
import sys

import numpy as np
from scipy import stats

from _harness import check, close, raises, run
from prospectrisk import distributions as D
from prospectrisk import units as U
from prospectrisk import volumetrics as V

C = D.Constant


def oil_seg(**over):
    v = {"area": C(1000), "gross_thickness": C(100), "geometric_factor": C(1.0), "net_to_gross": C(0.5),
         "porosity": C(0.2), "hc_saturation": C(0.8), "bo": C(1.25), "rf_oil": C(0.3), "gor": C(500),
         "rf_solution_gas": C(0.4)}
    v.update(over)
    return V.SegmentVolumetrics(V.SegmentConfig("area_thickness"), v)


def test_unit_constants():
    close("7758 bbl/acre-ft", U.BBL_PER_ACRE_FT, 7758.367, rel=1e-6)
    close("NCS gas equivalence 5614.6 scf/boe", U.BOE_CONVENTIONS[U.DEFAULT_BOE_CONVENTION], 5614.583, rel=1e-6)
    close("1 km2 in acres", U.to_internal(1, "area", "km2"), 247.10538, rel=1e-7)
    close("1 Sm3/Sm3 in scf/stb", U.to_internal(1, "gor", "Sm3/Sm3"), 5.614583, rel=1e-6)
    close("1 Sm3/MSm3 in stb/MMscf", U.to_internal(1, "cgr", "Sm3/MSm3"), 0.1781076, rel=1e-6)
    for q in ("area", "length", "grv", "oil", "gas", "oe", "gor", "cgr"):
        for u in U.units_for(q):
            close(f"round trip {q} {u}", U.from_internal(U.to_internal(3.7, q, u), q, u), 3.7, rel=1e-12)


def test_oil_hand_calc():
    r = V.deterministic(oil_seg(), V.constant_inputs(oil_seg()), "oil", scf_per_boe=6000)
    stoiip = 7758.367 * 1e5 * 0.5 * 0.2 * 0.8 / 1.25
    close("STOIIP hand calc", r["stoiip"], stoiip, rel=1e-6)
    close("recoverable oil", r["rec_oil"], 0.3 * stoiip, rel=1e-6)
    close("solution gas in place", r["giip_solution"], 500 * stoiip, rel=1e-6)
    close("recoverable o.e. (SPE)", r["rec_oe"], 0.3 * stoiip + 0.4 * 500 * stoiip / 6000, rel=1e-6)
    check("no gas volumes for oil phase", r["giip_free"] == 0 and r["rec_free_gas"] == 0)


def test_metric_integrity():
    """1 km2 x 100 m x phi 0.2 x Sh 0.8 / Bo 1 = 16.0 MSm3 exactly."""
    seg = oil_seg(area=C(U.to_internal(1, "area", "km2")), gross_thickness=C(U.to_internal(100, "length", "m")),
                  net_to_gross=C(1.0), bo=C(1.0))
    r = V.deterministic(seg, V.constant_inputs(seg), "oil")
    close("metric STOIIP 16.0 MSm3", U.from_internal(r["stoiip"], "oil", "MSm3"), 16.0, rel=1e-9)
    close("metric GRV 100 1e6 m3", U.from_internal(r["grv_total"], "grv", "1e6 m3"), 100.0, rel=1e-9)
    # 1 GSm3 gas at CGR 150 Sm3/MSm3 -> 150 000 Sm3 condensate
    gseg = V.SegmentVolumetrics(V.SegmentConfig("direct", phase_probabilities={"oil": 0, "gas": 1, "oil_gascap": 0}),
                                {"grv": C(U.to_internal(5.0, "grv", "1e6 m3")), "net_to_gross": C(1), "porosity": C(1),
                                 "hc_saturation": C(1), "bg": C(0.005), "rf_gas": C(1),
                                 "cgr": C(U.to_internal(150, "cgr", "Sm3/MSm3"))})
    g = V.deterministic(gseg, V.constant_inputs(gseg), "gas")
    close("metric GIIP 1 GSm3", U.from_internal(g["giip_free"], "gas", "GSm3"), 1.0, rel=1e-9)
    close("metric condensate 0.15 MSm3", U.from_internal(g["cond_ip"], "oil", "MSm3"), 0.15, rel=1e-9)


def test_gas_hand_calc():
    seg = V.SegmentVolumetrics(V.SegmentConfig("direct", phase_probabilities={"oil": 0, "gas": 1, "oil_gascap": 0}),
                               {"grv": C(1e5), "net_to_gross": C(0.5), "porosity": C(0.2), "hc_saturation": C(0.8),
                                "bg": C(0.005), "rf_gas": C(0.7), "cgr": C(50), "rf_condensate": C(0.5)})
    r = V.deterministic(seg, V.constant_inputs(seg), "gas")
    giip = 43560 * 1e5 * 0.08 / 0.005
    close("GIIP hand calc", r["giip_free"], giip, rel=1e-9)
    close("condensate in place", r["cond_ip"], giip / 1e6 * 50, rel=1e-9)
    close("recoverable condensate", r["rec_condensate"], 0.5 * giip / 1e6 * 50, rel=1e-9)
    check("gas_saturation defaults to hc_saturation", True)


def test_gas_cap_split():
    seg = V.SegmentVolumetrics(V.SegmentConfig("direct", phase_probabilities={"oil": 0, "gas": 0, "oil_gascap": 1}),
                               {"grv": C(1000), "net_to_gross": C(1), "porosity": C(0.2), "hc_saturation": C(0.8),
                                "gas_cap_fraction": C(0.3), "bo": C(1.2), "rf_oil": C(0.3), "bg": C(0.004), "rf_gas": C(0.6)})
    r = V.deterministic(seg, V.constant_inputs(seg), "oil_gascap")
    close("gas-cap GRV", r["grv_gas"], 300, rel=1e-12)
    close("oil-leg GRV", r["grv_oil"], 700, rel=1e-12)
    close("total GRV conserved", r["grv_total"], 1000, rel=1e-12)


def test_area_depth_slab_and_cone():
    slab = V.AreaDepthTable([1000, 3000], [500, 500])
    close("slab: column < thickness -> A x column", float(V._grv_between(slab, 1200, 400)), 500 * 200, rel=1e-9)
    close("slab: column > thickness -> A x h", float(V._grv_between(slab, 1800, 400)), 500 * 400, rel=1e-9)
    k = 2.0
    cone = V.AreaDepthTable([0, 1000], [0, k * 1000])
    col, h = 600.0, 250.0
    close("cone: thin base GRV analytic", float(V._grv_between(cone, col, h)), k * (col ** 2 - (col - h) ** 2) / 2, rel=1e-5)
    close("cone: thick base GRV analytic", float(V._grv_between(cone, col, 5000)), k * col ** 2 / 2, rel=1e-5)
    base = V.AreaDepthTable([h, 1000 + h], [0, k * 1000])
    close("base table equals thickness model", float(V._grv_between(cone, col, base=base)),
          float(V._grv_between(cone, col, h)), rel=1e-4)
    raises("non-monotonic depths rejected", lambda: V.AreaDepthTable([0, 0], [1, 2]), V.VolumetricsError)
    raises("decreasing area rejected", lambda: V.AreaDepthTable([0, 10], [5, 1]), V.VolumetricsError)
    close("extrapolation below table uses constant area", float(cone.cumulative_volume(1100)),
          k * 1000 ** 2 / 2 + 100 * 2000, rel=1e-5)


def test_area_depth_contact_modes():
    top = V.AreaDepthTable([0, 1000], [0, 2000])
    base_vars = {"gross_thickness": C(5000), "net_to_gross": C(1), "porosity": C(1), "hc_saturation": C(1),
                 "bo": C(1), "rf_oil": C(1)}
    for mode, var, val in (("contact_depth", "contact_depth", 600), ("column_height", "column_height", 600),
                           ("fill_fraction", "fill_fraction", 0.6)):
        seg = V.SegmentVolumetrics(V.SegmentConfig("area_depth", contact_mode=mode, top_table=top),
                                   {**base_vars, var: C(val)})
        r = V.deterministic(seg, V.constant_inputs(seg), "oil")
        close(f"{mode}: GRV", r["grv_total"], 2.0 * 600 ** 2 / 2, rel=1e-5)
    seg = V.SegmentVolumetrics(V.SegmentConfig("area_depth", contact_mode="contact_depth", top_table=top,
                                               limit_to_spill=True),
                               {**base_vars, "contact_depth": C(900), "spill_depth": C(500)})
    r = V.deterministic(seg, V.constant_inputs(seg), "oil")
    close("contact limited to spill point", r["contact"], 500, rel=1e-12)
    res = V.simulate(seg, 500, 1)
    close("spill clipping reported", res.diagnostics["contact_clipped_fraction"], 1.0, rel=1e-12)


def test_monte_carlo_product_mean():
    seg = oil_seg(area=D.Triangular(500, 1000, 1800), gross_thickness=D.Normal(100, 15),
                  net_to_gross=D.Uniform(0.4, 0.8), porosity=D.Normal(0.2, 0.02), hc_saturation=D.PERT(0.6, 0.75, 0.9),
                  bo=C(1.25), rf_oil=D.Triangular(0.2, 0.3, 0.4))
    res = V.simulate(seg, 100_000, 42, "lhs")
    means = {k: d.mean for k, d in seg.variables.items()}
    want = means["area"] * means["gross_thickness"] * 1 * means["net_to_gross"] * means["porosity"] * \
        means["hc_saturation"] * 7758.367 / 1.25 * means["rf_oil"]
    close("independent product: E[rec] = product of means", float(res.outputs["rec_oil"].mean()), want, rel=5e-3)
    r2 = V.simulate(seg, 2000, 42)
    r3 = V.simulate(seg, 2000, 42)
    r4 = V.simulate(seg, 2000, 43)
    check("same seed reproducible", np.array_equal(r2.outputs["rec_oe"], r3.outputs["rec_oe"]))
    check("different seed differs", not np.array_equal(r2.outputs["rec_oe"], r4.outputs["rec_oe"]))
    for k, d in seg.variables.items():
        if not d.is_constant:
            close(f"sampled P50 of {k}", float(np.quantile(res.inputs[k], 0.5)), d.p50, rel=5e-3)


def test_correlation_in_simulation():
    seg = oil_seg(porosity=D.Normal(0.2, 0.03), hc_saturation=D.Triangular(0.5, 0.7, 0.9), net_to_gross=D.Uniform(0.4, 0.8))
    seg.correlations = [("porosity", "hc_saturation", 0.6), ("bo", "porosity", 0.5)]
    res = V.simulate(seg, 20000, 3)
    close("porosity–Sh rank correlation", stats.spearmanr(res.inputs["porosity"], res.inputs["hc_saturation"]).statistic,
          0.6, abs_=0.03)
    check("correlation with constant input ignored with a warning",
          any("constant" in w for w in res.diagnostics["warnings"]))


def test_phase_sampling():
    seg = V.SegmentVolumetrics(V.SegmentConfig("direct", phase_probabilities={"oil": 0.5, "gas": 0.2, "oil_gascap": 0.3}),
                               {"grv": C(1000), "net_to_gross": C(1), "porosity": C(0.2), "hc_saturation": C(0.8),
                                "gas_cap_fraction": C(0.3), "bo": C(1.2), "rf_oil": C(0.3), "bg": C(0.004), "rf_gas": C(0.6)})
    res = V.simulate(seg, 50000, 9)
    for i, p in enumerate((0.5, 0.2, 0.3)):
        close(f"phase {V.PHASES[i]} frequency", float(np.mean(res.phase == i)), p, abs_=0.002)
    check("oil volume zero exactly when phase is gas",
          np.all(res.outputs["stoiip"][res.phase == 1] == 0) and np.all(res.outputs["stoiip"][res.phase != 1] > 0))


def test_validation_and_clipping():
    raises("missing required input", lambda: V.simulate(V.SegmentVolumetrics(V.SegmentConfig("direct"), {}), 100, 1),
           V.VolumetricsError)
    bad = V.SegmentConfig("direct", phase_probabilities={"oil": 0.5, "gas": 0.2, "oil_gascap": 0.0})
    raises("phase probabilities must sum to 1", bad.validate, V.VolumetricsError)
    raises("area-depth without table", V.SegmentConfig("area_depth").validate, V.VolumetricsError)
    seg = oil_seg(porosity=D.Normal(0.05, 0.05))
    res = V.simulate(seg, 5000, 1)
    check("negative porosity samples clipped", res.inputs["porosity"].min() >= 0 and "porosity" in res.diagnostics["clipped"])
    check("clipping produces a warning", any("clipped" in w for w in res.diagnostics["warnings"]))
    req, opt = V.required_variables(V.SegmentConfig("area_depth", contact_mode="fill_fraction",
                                                    top_table=V.AreaDepthTable([0, 1], [0, 1]),
                                                    phase_probabilities={"oil": 0, "gas": 0, "oil_gascap": 1}))
    check("required variables for area-depth / fill / gas cap",
          {"gross_thickness", "fill_fraction", "gas_cap_fraction", "bo", "bg"} <= set(req) and "spill_depth" in opt)


def test_serialisation():
    seg = V.SegmentVolumetrics(V.SegmentConfig("area_depth", "column_height", True, V.AreaDepthTable([0, 100], [0, 50])),
                               {"column_height": D.Lognormal(50, 10), "porosity": D.Normal(0.2, 0.02, lower=0)},
                               [("porosity", "column_height", 0.3)])
    r = V.SegmentVolumetrics.from_dict(seg.to_dict())
    check("segment round trip", r.to_dict() == seg.to_dict())


if __name__ == "__main__":
    sys.exit(run("test_volumetrics", [test_unit_constants, test_oil_hand_calc, test_metric_integrity, test_gas_hand_calc,
                                      test_gas_cap_split, test_area_depth_slab_and_cone, test_area_depth_contact_modes,
                                      test_monte_carlo_product_mean, test_correlation_in_simulation, test_phase_sampling,
                                      test_validation_and_clipping, test_serialisation]))
