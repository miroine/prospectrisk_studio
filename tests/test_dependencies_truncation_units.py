"""Volume truncation, parameter dependencies (formulas, links, cross-segment correlation) and flexible units."""
import sys

import numpy as np
from scipy import stats

from _harness import check, close, raises, run
from prospectrisk import aggregation as A
from prospectrisk import distributions as D
from prospectrisk import expressions as ex
from prospectrisk import units as U
from prospectrisk import volumetrics as V
from prospectrisk.examples import example_project
from prospectrisk.io_export import project_from_yaml, project_to_yaml
from prospectrisk.project import Project, ProjectError, Settings, run_portfolio, run_prospect
from prospectrisk.sensitivity import tornado

C = D.Constant
RNG = np.random.default_rng(21)
N = 100_000


def oil_seg(**over):
    v = {"area": D.Triangular(500, 1000, 2000), "gross_thickness": C(100), "geometric_factor": C(1.0),
         "net_to_gross": C(0.6), "porosity": D.Normal(0.2, 0.03, lower=0.02), "hc_saturation": C(0.75),
         "bo": C(1.25), "rf_oil": C(0.3)}
    v.update(over)
    return V.SegmentVolumetrics(V.SegmentConfig("area_thickness"), v)


# ---- volume truncation ----------------------------------------------------------------------------

def _item(vol, p=0.4, trunc=None, name="a", group="G", align=None):
    return A.AggItem(name, group, {"rec_oe": vol, "rec_liquids": 0.5 * vol}, {}, p,
                     trunc or A.VolumeTruncation(), align)


def test_minimum_volume_chance_mode():
    vol = RNG.lognormal(np.log(20), 0.8, N)
    vmin = 10.0
    res = A.aggregate([_item(vol, 0.4, A.VolumeTruncation("rec_oe", minimum=vmin, min_mode="chance"))], seed=1)
    p_ok = float(np.mean(vol >= vmin))
    close("P(V ≥ min) reported", res.p_volume_ok(0), p_ok, abs_=1e-12)
    close("effective chance = Pg × P(V ≥ min)", float(res.success[:, 0].mean()), 0.4 * p_ok, abs_=0.004)
    sc = res.item_success_case(0, "rec_oe")
    check("success case is truncated at the minimum", sc.min() >= vmin)
    close("success-case P50 = conditional median", float(np.median(sc)), float(np.median(vol[vol >= vmin])), rel=0.01)
    close("risked mean = Pg × E[V; V ≥ min]", res.summary()["Risked mean"], 0.4 * float(np.mean(vol * (vol >= vmin))),
          rel=0.02)
    check("row integrity kept (liquids = 0.5 × oe)", np.allclose(res.volumes["rec_liquids"], 0.5 * res.volumes["rec_oe"]))


def test_renormalised_truncation():
    vol = RNG.lognormal(np.log(20), 0.8, N)
    t = A.VolumeTruncation("rec_oe", minimum=8.0, maximum=60.0, min_mode="renormalise")
    res = A.aggregate([_item(vol, 0.4, t)], seed=2)
    v = res.item_success_case(0, "rec_oe")
    check("all success volumes within [min, max]", v.min() >= 8.0 and v.max() <= 60.0)
    close("chance unchanged by renormalising truncation", float(res.success[:, 0].mean()), 0.4, abs_=0.004)
    cond = vol[(vol >= 8) & (vol <= 60)]
    close("renormalised mean = conditional mean", float(v.mean()), float(cond.mean()), rel=0.01)
    check("rejected fraction recorded", abs(res.diagnostics["rejected_fraction"]["a"] - (1 - cond.size / N)) < 1e-9)
    raises("max <= min rejected", lambda: A.VolumeTruncation("rec_oe", 5, 5).validate(), A.AggregationError)
    raises("truncation removing everything rejected",
           lambda: A.aggregate([_item(vol, 0.4, A.VolumeTruncation("rec_oe", maximum=-1.0))], seed=1), A.AggregationError)


def test_aligned_items_resample_jointly():
    a = RNG.lognormal(0, 0.5, N)
    b = 2.0 * a  # perfectly dependent realisations
    t = A.VolumeTruncation("rec_oe", maximum=float(np.quantile(a, 0.8)))
    res = A.aggregate([_item(a, 1, t, "a", align="X"), _item(b, 1, None, "b", align="X")], seed=3)
    check("aligned items keep their trial pairing through resampling",
          np.allclose(res.volumes["rec_oe"][:, 1], 2.0 * res.volumes["rec_oe"][:, 0]))
    res2 = A.aggregate([_item(a, 1, None, "a"), _item(b, 1, None, "b")], seed=3)
    check("unaligned items are decorrelated",
          abs(stats.spearmanr(res2.volumes["rec_oe"][:, 0], res2.volumes["rec_oe"][:, 1]).statistic) < 0.03)


# ---- formulas ----------------------------------------------------------------------------------------

def test_expression_parser_safety():
    for bad in ("__import__('os').system('ls')", "porosity.real", "[1,2]", "lambda x: x", "open('f')",
                "porosity if True else 1", "a == b"):
        raises(f"rejects unsafe/unsupported: {bad}", lambda b=bad: ex.parse(b), ex.ExpressionError)
    tree, names = ex.parse("0.42 * exp(-0.00035 * contact_depth) + max(porosity, 0.1) - pi*0")
    check("names extracted", names == {"contact_depth", "porosity"})
    out = ex.evaluate(tree, {"contact_depth": np.array([0.0, 1000.0]), "porosity": np.array([0.05, 0.3])}, 2)
    close("evaluation matches NumPy", out[1], 0.42 * np.exp(-0.35) + 0.3, rel=1e-12)


def test_formula_dependency_in_simulation():
    seg = oil_seg()
    seg.expressions["hc_saturation"] = V.Expression("1 - 0.05 / porosity")
    res = V.simulate(seg, 20000, 4)
    want = np.clip(1 - 0.05 / res.inputs["porosity"], 0, 1)
    check("formula applied trial by trial", np.allclose(res.inputs["hc_saturation"], want))
    check("formula input excluded from sampled inputs", "hc_saturation" not in seg.sampled_inputs())
    check("perfect positive rank dependency Sh–porosity",
          stats.spearmanr(res.inputs["porosity"], res.inputs["hc_saturation"]).statistic > 0.999)


def test_formula_units_are_fixed():
    """Porosity–depth trend written in metres must not change when display units change."""
    top = V.AreaDepthTable([0, 5000], [0, 5000])
    seg = V.SegmentVolumetrics(V.SegmentConfig("area_depth", "contact_depth", top_table=top),
                               {"gross_thickness": C(5000), "contact_depth": C(U.to_internal(2000, "length", "m")),
                                "net_to_gross": C(1), "porosity": C(0.2), "hc_saturation": C(0.8), "bo": C(1), "rf_oil": C(1)})
    seg.expressions["porosity"] = V.Expression("0.4 * exp(-0.0005 * contact_depth)", units={"contact_depth": "m"})
    r = V.simulate(seg, 200, 1)
    close("porosity from metric depth trend", float(r.inputs["porosity"][0]), 0.4 * np.exp(-1.0), rel=1e-9)
    seg.expressions["gross_thickness"] = V.Expression("30", units={"gross_thickness": "m"})
    r = V.simulate(seg, 200, 1)
    close("formula result converted from its unit (30 m)", float(r.inputs["gross_thickness"][0]),
          U.to_internal(30, "length", "m"), rel=1e-12)


def test_formula_noise_and_errors():
    seg = oil_seg()
    seg.expressions["hc_saturation"] = V.Expression("0.5 + porosity", noise=D.Triangular(0.9, 1.0, 1.1))
    res = V.simulate(seg, 20000, 5)
    ratio = res.inputs["hc_saturation"] / (0.5 + res.inputs["porosity"])
    close("multiplicative uncertainty mean", float(ratio.mean()), 1.0, abs_=0.002)
    check("uncertainty is a sampled input", "hc_saturation@noise" in seg.sampled_inputs())
    base, rows = tornado(seg, "rec_oe", "oil")
    check("tornado includes formula uncertainty", any(r["key"] == "hc_saturation@noise" for r in rows))
    s2 = oil_seg()
    s2.expressions["porosity"] = V.Expression("hc_saturation * 0.3")
    s2.expressions["hc_saturation"] = V.Expression("porosity + 0.5")
    raises("circular formulas rejected", lambda: V.simulate(s2, 100, 1), V.VolumetricsError)
    s3 = oil_seg()
    s3.expressions["hc_saturation"] = V.Expression("log(porosity - 1)")
    raises("NaN results rejected with a clear error", lambda: V.simulate(s3, 100, 1), V.VolumetricsError)
    s4 = oil_seg()
    s4.expressions["hc_saturation"] = V.Expression("contact_depth * 0")
    raises("unavailable input rejected", lambda: V.simulate(s4, 100, 1), V.VolumetricsError)
    s5 = oil_seg()
    s5.expressions["hc_saturation"] = V.Expression("0.5 + porosity", noise=D.Normal(1, 0.1))
    check("expressions survive serialisation",
          V.SegmentVolumetrics.from_dict(s5.to_dict()).to_dict() == s5.to_dict())


# ---- links and cross-segment correlation ----------------------------------------------------------------

def test_links_and_cross_segment_correlation():
    s1 = oil_seg(bo=D.Triangular(1.2, 1.3, 1.5))
    s2 = oil_seg(bo=D.Uniform(1.1, 1.6), porosity=D.Normal(0.18, 0.02, lower=0.02))
    deps = [V.InputDependency(0, "bo", 1, "bo", "link"),
            V.InputDependency(0, "porosity", 1, "porosity", "correlation", 0.8)]
    r1, r2 = V.simulate_many([s1, s2], 20000, [1, 2], dependencies=deps)
    close("linked input: same percentile in every trial",
          stats.spearmanr(r1.inputs["bo"], r2.inputs["bo"]).statistic, 1.0, abs_=1e-9)
    check("linked input keeps its own distribution", np.quantile(r2.inputs["bo"], 0.999) <= 1.6 and
          abs(np.median(r2.inputs["bo"]) - 1.35) < 0.01)
    close("cross-segment rank correlation", stats.spearmanr(r1.inputs["porosity"], r2.inputs["porosity"]).statistic,
          0.8, abs_=0.03)
    raises("self dependency rejected",
           lambda: V.simulate_many([s1], 100, [1], dependencies=[V.InputDependency(0, "bo", 0, "bo", "link")]),
           V.VolumetricsError)
    raises("two leaders rejected",
           lambda: V.simulate_many([s1, s2], 100, [1, 2], dependencies=[V.InputDependency(0, "bo", 1, "bo", "link"),
                                                                        V.InputDependency(0, "area", 1, "bo", "link")]),
           V.VolumetricsError)
    r = V.simulate_many([s1, s2], 200, [1, 2], dependencies=[V.InputDependency(0, "rf_oil", 1, "bo", "link")])
    check("dependency on a constant is ignored with a warning", any("Ignored dependency" in w
                                                                    for w in r[1].diagnostics["warnings"]))


def test_project_level_dependencies_and_truncation():
    p = example_project()
    pr = p.prospect("Alpha")
    for sg in pr.segments:
        sg.volumetrics.correlations = []  # keep the combined correlation matrix consistent for an exact check
    pr.input_dependencies = [{"seg_a": "Upper sand", "var_a": "porosity", "seg_b": "Lower sand", "var_b": "porosity",
                              "kind": "correlation", "rho": 0.7},
                             {"seg_a": "Upper sand", "var_a": "bo", "seg_b": "Lower sand", "var_b": "bo", "kind": "link"}]
    pr.segment("Lower sand").truncation = A.VolumeTruncation("rec_oe", minimum=U.to_internal(2.0, "oe", "MSm3 o.e."))
    run1 = run_prospect(p, "Alpha")
    a = run1.aggregation
    j_up, j_lo = run1.segment_index("Upper sand"), run1.segment_index("Lower sand")
    check("segments with cross dependencies are trial-aligned",
          all(it.align == "prospect::Alpha" for it in a.items))
    close("cross-segment porosity correlation survives into the prospect run",
          stats.spearmanr(run1.segments["Upper sand"].result.inputs["porosity"],
                          run1.segments["Lower sand"].result.inputs["porosity"]).statistic, 0.7, abs_=0.03)
    close("effective Pg = factor Pg × P(V ≥ min)", run1.segment_effective_pg("Lower sand"),
          float(a.success[:, j_lo].mean()), abs_=0.012)
    check("segment success case truncated", run1.segment_success_case("Lower sand").min()
          >= U.to_internal(2.0, "oe", "MSm3 o.e."))
    b = run_prospect(p, "Bravo")
    pf = run_portfolio(p, ["Alpha", "Bravo"], {"Alpha": run1, "Bravo": b})
    close("portfolio risked mean ≈ analytic with truncation", pf.summary()["Risked mean"],
          pf.summary()["Analytic risked mean"], rel=0.05)
    rt = project_from_yaml(project_to_yaml(p))
    check("dependencies and truncation round-trip through YAML", rt.to_dict() == p.to_dict())
    pr.input_dependencies = [{"seg_a": "Nope", "var_a": "bo", "seg_b": "Lower sand", "var_b": "bo", "kind": "link"}]
    raises("unknown segment in dependency rejected", lambda: run_prospect(p, "Alpha"), ProjectError)


# ---- units --------------------------------------------------------------------------------------------------

def test_expansion_factor_equals_bg():
    base = {"grv": C(1e5), "net_to_gross": C(0.8), "porosity": C(0.2), "hc_saturation": C(0.8), "rf_gas": C(0.7)}
    g1 = V.SegmentVolumetrics(V.SegmentConfig("direct", phase_probabilities={"oil": 0, "gas": 1, "oil_gascap": 0}),
                              {**base, "bg": C(0.0045)})
    g2 = V.SegmentVolumetrics(V.SegmentConfig("direct", phase_probabilities={"oil": 0, "gas": 1, "oil_gascap": 0},
                                              gas_fvf="eg"), {**base, "eg": C(1 / 0.0045)})
    close("E input gives same GIIP as Bg", V.deterministic(g2, V.constant_inputs(g2), "gas")["giip_free"],
          V.deterministic(g1, V.constant_inputs(g1), "gas")["giip_free"], rel=1e-12)


def test_flexible_units():
    close("1 ha", U.to_internal(1, "area", "ha"), 2.4710538, rel=1e-7)
    close("1 mi2 = 640 acres", U.to_internal(1, "area", "mi2"), 640.0)
    close("1 rb/Mscf in rcf/scf", U.to_internal(1, "bg", "rb/Mscf"), 0.0056145833, rel=1e-8)
    close("1 Tscf", U.from_internal(1e12, "gas", "Tscf"), 1.0)
    close("CGR 1 Sm3/Sm3 = 1e6 Sm3/MSm3", U.to_internal(1, "cgr", "Sm3/Sm3"), 1e6 * U.to_internal(1, "cgr", "Sm3/MSm3"),
          rel=1e-12)
    for q in U.QUANTITY_LABELS:
        check(f"internal unit of {q} has factor 1", U.factor(q, U.internal_unit(q)) == 1.0)
        for sys_ in U.UNIT_SYSTEMS:
            check(f"preset {sys_} defines {q}", U.is_valid_unit(q, U.UNIT_SYSTEMS[sys_][q]))
    s = Settings(unit_system="Metric (NCS)", unit_overrides={"area": "acres", "oe": "MMboe", "gas": "nonsense"},
                 input_units={"gross_thickness": "ft", "porosity": "%", "bo": "bad"})
    check("quantity override applied", s.unit("area") == "acres" and s.unit("oe") == "MMboe")
    check("invalid override falls back to preset", s.unit("gas") == "GSm3")
    check("per-input override applied", s.input_unit("gross_thickness") == "ft" and s.input_unit("porosity") == "%")
    check("per-input fallback to quantity unit", s.input_unit("contact_depth") == "m" and s.input_unit("bo") == "Rm3/Sm3")
    p = example_project()
    p.settings = s
    check("unit settings round-trip", Project.from_dict(p.to_dict()).settings.unit_overrides == s.unit_overrides)
    pct = D.scaled(D.Normal(20, 3), U.factor("fraction", "%"))
    close("porosity entered in % becomes a fraction", pct.p50, 0.20, rel=1e-12)


if __name__ == "__main__":
    sys.exit(run("test_dependencies_truncation_units", [
        test_minimum_volume_chance_mode, test_renormalised_truncation, test_aligned_items_resample_jointly,
        test_expression_parser_safety, test_formula_dependency_in_simulation, test_formula_units_are_fixed,
        test_formula_noise_and_errors, test_links_and_cross_segment_correlation,
        test_project_level_dependencies_and_truncation, test_expansion_factor_equals_bg, test_flexible_units]))
