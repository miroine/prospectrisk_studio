"""Chance of success, commercial chance, aggregation with dependencies, project runs and exports."""
import sys

import numpy as np
from scipy import stats

from _harness import check, close, raises, run
from prospectrisk import aggregation as A
from prospectrisk import io_export as X
from prospectrisk import qc
from prospectrisk import risk as R
from prospectrisk.examples import example_project
from prospectrisk.project import Project, ProjectError, run_portfolio, run_prospect, segment_seed

RNG = np.random.default_rng(11)
N = 200_000


def test_chance_factors():
    m = R.DEFAULT_ADEQUACY_MATRIX
    close("favourable/high = 0.9", R.ChanceFactor("x", mode="adequacy", interpretation="favourable",
                                                  confidence="high").value(m), 0.9)
    close("neutral always 0.5", R.ChanceFactor("x", mode="adequacy", interpretation="neutral", confidence="low").value(m), 0.5)
    raises("probability > 1 rejected", lambda: R.ChanceFactor("x", probability=1.2).value(), R.RiskError)
    raises("unknown adequacy class rejected",
           lambda: R.ChanceFactor("x", mode="adequacy", interpretation="great").value(), R.RiskError)
    b = R.breakdown([R.ChanceFactor("s", "Source", probability=0.9)],
                    [R.ChanceFactor("t", "Trap", probability=0.6)],
                    [R.ChanceFactor("r", "Reservoir", probability=0.8), R.ChanceFactor("q", "Reservoir", probability=0.5)])
    close("Pg = product of all scopes", b.pg, 0.9 * 0.6 * 0.8 * 0.5)
    close("conditional chance excludes play", b.p_conditional, 0.6 * 0.8 * 0.5)
    close("category product (Reservoir)", b.by_category["Reservoir"], 0.4)
    check("critical factor is the lowest", b.critical[0][1] == "q")
    check("risk class assigned", b.risk_class[0] == "High risk", str(b.risk_class))


def test_commercial_and_curves():
    v = np.arange(1, 101, dtype=float)
    c = R.commercial_chance(0.3, v, 51)
    close("P(V>=MEFS | success)", c["p_above_mefs_given_success"], 0.5)
    close("Pc = Pg x P(V>=MEFS)", c["pc"], 0.15)
    close("mean commercial success", c["mean_commercial_success"], 75.5)
    x, p = R.expectation_curve(v, 0.3)
    close("risked curve starts at Pg", p[0], 0.3)
    check("curve monotone non-increasing", np.all(np.diff(p) <= 1e-12))
    close("Swanson's mean", R.swanson_mean(10, 20, 40), 23.0)
    s = R.exceedance_stats(v)
    check("P90 low / P10 high", s["P90"] < s["P50"] < s["P10"])


def _item(name, group, p_res, shared=None, scale=1.0):
    vol = RNG.lognormal(np.log(10 * scale), 0.5, N)
    return A.AggItem(name, group, {"rec_oe": vol, "rec_liquids": vol * 0.8, "rec_gas": vol * 0.2 * 5614.6},
                     dict(shared or {}), p_res)


def test_independent_aggregation():
    items = [_item("a", "A", 0.3), _item("b", "B", 0.2), _item("c", "C", 0.5)]
    res = A.aggregate(items, seed=1)
    close("P(any) independent", res.summary()["P(at least one success)"], 1 - 0.7 * 0.8 * 0.5, abs_=0.004)
    for row in res.item_table():
        close(f"simulated Pg {row['Item']}", row["Pg (simulated)"], row["Pg (effective)"], abs_=0.004)
    s = res.summary()
    close("risked mean = Σ Pg x mean", s["Risked mean"], s["Analytic risked mean"], rel=0.02)
    close("expected discoveries = Σ Pg", s["Expected discoveries"], 1.0, abs_=0.01)


def test_shared_play_risk():
    play = {"play::P": 0.5}
    items = [_item("a", "A", 0.6, play), _item("b", "B", 0.6, play)]
    res = A.aggregate(items, seed=2)
    close("segment Pg includes play chance", res.item_table()[0]["Pg (simulated)"], 0.3, abs_=0.004)
    want = 0.5 * (1 - 0.4 * 0.4)
    close("P(any) with shared play = Pplay x (1-Π(1-Pcond))", res.summary()["P(at least one success)"], want, abs_=0.004)
    both = np.mean(res.success.all(axis=1))
    close("P(both) = Pplay x Pcond²", both, 0.5 * 0.36, abs_=0.004)
    bad = [_item("a", "A", 1, {"play::P": 0.5}), _item("b", "B", 1, {"play::P": 0.6})]
    raises("inconsistent shared chance rejected", lambda: A.aggregate(bad, seed=1), A.AggregationError)


def test_residual_dependency():
    items = [_item("a", "A", 0.4), _item("b", "A", 0.4)]
    full = A.aggregate(items, seed=3, dependency=np.array([[1, 1.0], [1.0, 1]]))
    close("full dependency: P(both) = p", float(full.success.all(axis=1).mean()), 0.4, abs_=0.005)
    half = A.aggregate(items, seed=3, dependency=np.array([[1, 0.5], [0.5, 1]]))
    want = stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]).cdf([stats.norm.ppf(0.4)] * 2)
    close("copula dependency 0.5: P(both) = bivariate normal CDF", float(half.success.all(axis=1).mean()), want, abs_=0.005)
    close("dependency does not change marginal Pg", float(half.success[:, 0].mean()), 0.4, abs_=0.004)
    raises("asymmetric dependency rejected",
           lambda: A.aggregate(items, seed=1, dependency=np.array([[1, 0.2], [0.3, 1]])), A.AggregationError)


def test_volume_correlation_and_row_integrity():
    items = [_item("a", "A", 1.0), _item("b", "A", 1.0)]
    res = A.aggregate(items, seed=4, volume_correlation=np.array([[1, 0.7], [0.7, 1]]))
    rho = stats.spearmanr(res.volumes["rec_oe"][:, 0], res.volumes["rec_oe"][:, 1]).statistic
    close("success volume rank correlation", rho, 0.7, abs_=0.03)
    check("trial rows kept intact when reordered",
          np.allclose(res.volumes["rec_liquids"], 0.8 * res.volumes["rec_oe"]))


def test_group_commercial():
    items = [_item("a", "G", 0.5), _item("b", "G", 0.5)]
    res = A.aggregate(items, seed=5, group_mefs={"G": 1e9})
    close("MEFS above all volumes -> Pc = 0", float(res.group_commercial("G").mean()), 0.0)
    res0 = A.aggregate(items, seed=5, group_mefs={"G": 0.0})
    close("MEFS 0 -> Pc = P(any)", float(res0.group_commercial("G").mean()), float(res0.group_success("G").mean()))


def test_project_runs_and_determinism():
    p = example_project()
    close("md5 seed is deterministic", segment_seed(1, "A", "B"), segment_seed(1, "A", "B"))
    check("seed differs by segment", segment_seed(1, "A", "B") != segment_seed(1, "A", "C"))
    r1 = run_prospect(p, "Alpha")
    r2 = run_prospect(p, "Alpha")
    check("prospect run reproducible", np.array_equal(r1.aggregation.success, r2.aggregation.success))
    for sname, srun in r1.segments.items():
        close(f"simulated vs effective segment Pg ({sname})",
              float(r1.aggregation.success[:, r1.segment_index(sname)].mean()), r1.segment_effective_pg(sname), abs_=0.015)
    b = run_prospect(p, "Bravo")
    pf = run_portfolio(p, ["Alpha", "Bravo"], {"Alpha": r1, "Bravo": b})
    s = pf.summary()
    close("portfolio risked mean ≈ analytic", s["Risked mean"], s["Analytic risked mean"], rel=0.05)
    check("portfolio P(any) ≥ max prospect Pg", s["P(at least one success)"] >= max(r1.pg, b.pg) - 0.01)
    findings = qc.check_prospect(p, r1, qc.QCThresholds())
    check("QC runs and produces findings", len(findings) > 0 and all(f.level in ("error", "warning", "info") for f in findings))


def test_project_validation_and_files():
    p = example_project()
    d = p.to_dict()
    d["prospects"][1]["name"] = "Alpha"
    raises("duplicate prospect names rejected", lambda: Project.from_dict(d), ProjectError)
    d = p.to_dict()
    d["prospects"][0]["play"] = "No such play"
    raises("unknown play rejected", lambda: Project.from_dict(d), ProjectError)
    raises("non-project YAML rejected", lambda: X.project_from_yaml("a: 1"), ValueError)
    check("YAML round trip", X.project_from_yaml(X.project_to_yaml(p)).to_dict() == p.to_dict())
    runs = {pr.name: run_prospect(p, pr.name) for pr in p.prospects}
    import yaml
    bridge = yaml.safe_load(X.fieldvista_bridge_yaml(p, runs))
    vol = bridge["prospects"][0]["volumes"]["rec_oe"]
    check("bridge: low < mid < high", vol["low"] < vol["mid"] < vol["high"])
    check("bridge: percentile_10 is the low case", vol["percentile_10"] == vol["low"])
    check("Excel export produces a workbook", X.results_to_excel(p, runs)[:2] == b"PK")
    check("PDF report produced", X.pdf_report(p, runs)[:4] == b"%PDF")


if __name__ == "__main__":
    sys.exit(run("test_risk_aggregation", [test_chance_factors, test_commercial_and_curves, test_independent_aggregation,
                                           test_shared_play_risk, test_residual_dependency,
                                           test_volume_correlation_and_row_integrity, test_group_commercial,
                                           test_project_runs_and_determinism, test_project_validation_and_files]))
