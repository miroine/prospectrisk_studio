"""Distributions, sampling designs and rank correlation."""
import sys

import numpy as np
from scipy import stats

from _harness import check, close, raises, run
from prospectrisk import correlation as corr
from prospectrisk import distributions as D

RNG = np.random.default_rng(7)
N = 200_000


def test_analytic_means():
    cases = [
        ("uniform", D.Uniform(2, 8), 5.0),
        ("triangular", D.Triangular(1, 2, 6), 3.0),
        ("pert", D.PERT(1, 2, 7), (1 + 4 * 2 + 7) / 6),
        ("normal", D.Normal(10, 2), 10.0),
        ("lognormal", D.Lognormal(50, 20), 50.0),
        ("beta", D.Beta(2, 5, 0, 1), 2 / 7),
        ("discrete", D.Discrete([1, 2, 3], [0.2, 0.5, 0.3]), 2.1),
        ("constant", D.Constant(4.2), 4.2),
    ]
    for name, d, want in cases:
        close(f"mean(quadrature) {name}", d.mean, want, rel=2e-3, abs_=1e-9)
        close(f"mean(sample) {name}", float(d.sample(N, RNG).mean()), want, rel=1e-2, abs_=1e-9)


def test_lognormal_sd_and_percentiles():
    d = D.Lognormal(50, 20)
    close("lognormal sd", d.std, 20.0, rel=5e-3)
    check("P90 < P50 < P10 (exploration convention)", d.p90 < d.p50 < d.p10)
    close("P90 is the 10th statistical percentile", d.p90, float(stats.lognorm(s=d.sigma, scale=np.exp(d.mu)).ppf(0.1)))


def test_fits():
    ln = D.Lognormal.from_p90_p10(10, 90)
    close("lognormal fit P90", ln.p90, 10, rel=1e-9)
    close("lognormal fit P10", ln.p10, 90, rel=1e-9)
    close("lognormal fit P50 = geometric mean", ln.p50, 30, rel=1e-9)
    nm = D.Normal.from_p90_p10(4, 16)
    close("normal fit P90", nm.p90, 4, rel=1e-9)
    close("normal fit mean", nm.mu, 10, rel=1e-12)
    d3, asym = D.Lognormal.from_p90_p50_p10(10, 30, 90)
    close("3-point lognormal symmetric input -> asymmetry 0", asym, 0.0, abs_=1e-12)
    _, asym2 = D.Lognormal.from_p90_p50_p10(10, 20, 90)
    check("3-point lognormal flags asymmetric input", asym2 > 0.2, f"asym={asym2}")


def test_truncation():
    d = D.Normal(0.2, 0.1, lower=0.05, upper=0.3)
    x = d.sample(N, RNG)
    check("truncated samples within bounds", x.min() >= 0.05 and x.max() <= 0.3)
    a, b = (0.05 - 0.2) / 0.1, (0.3 - 0.2) / 0.1
    want = float(stats.truncnorm(a, b, loc=0.2, scale=0.1).mean())
    close("truncated normal mean vs scipy.truncnorm", d.mean, want, rel=1e-3)
    raises("truncation excluding all mass raises", lambda: D.Normal(0, 1, lower=50, upper=60).p50, D.DistributionError)
    raises("discrete cannot be truncated",
           lambda: D.Distribution.__init__(D.Discrete([1], [1]), lower=0), D.DistributionError)


def test_discrete_and_table():
    d = D.Discrete([3, 1, 2], [0.3, 0.2, 0.5])
    x = d.sample(N, RNG)
    for v, p in ((1, 0.2), (2, 0.5), (3, 0.3)):
        close(f"discrete frequency of {v}", float(np.mean(x == v)), p, abs_=0.005)
    t = D.PercentileTable({100: 5, 90: 10, 50: 20, 10: 45, 0: 80})
    for pct, v in ((90, 10), (50, 20), (10, 45)):
        close(f"table P{pct}", t.exceedance(pct), v, rel=1e-9)
    raises("table without P0/P100 raises", lambda: D.PercentileTable({90: 1, 10: 2}), D.DistributionError)
    raises("table decreasing values raises", lambda: D.PercentileTable({100: 5, 50: 3, 0: 8}), D.DistributionError)


def test_validation():
    raises("triangular mode outside", lambda: D.Triangular(1, 9, 5), D.DistributionError)
    raises("uniform max<=min", lambda: D.Uniform(3, 3), D.DistributionError)
    raises("normal sd<=0", lambda: D.Normal(1, 0), D.DistributionError)
    raises("lognormal mean<=0", lambda: D.Lognormal(-1, 1), D.DistributionError)
    raises("discrete probs not summing", lambda: D.Discrete([1, 2], [0.5, 0.6]), D.DistributionError)
    raises("unknown kind", lambda: D.from_dict({"kind": "weibull"}), D.DistributionError)


def test_serialisation_and_scaling():
    ds = [D.Constant(2), D.Uniform(1, 2, lower=1.2), D.Triangular(0, 1, 3), D.PERT(0, 1, 3, 6),
          D.Normal(1, 0.2, upper=1.3), D.Lognormal(5, 2), D.Beta(2, 3, 1, 4),
          D.Discrete([1, 2], [0.4, 0.6]), D.PercentileTable({100: 1, 50: 2, 0: 5})]
    for d in ds:
        r = D.from_dict(d.to_dict())
        close(f"round-trip P50 {d.kind}", r.p50, d.p50, rel=1e-12, abs_=1e-12)
        s = D.scaled(d, 3.5)
        close(f"scaled P10 {d.kind}", s.p10, 3.5 * d.p10, rel=1e-9, abs_=1e-12)
        close(f"scaled mean {d.kind}", s.mean, 3.5 * d.mean, rel=1e-6, abs_=1e-12)


def test_lhs():
    n = 1000
    u = corr.uniform_design(n, 3, RNG, "lhs")
    strata_ok = all(np.array_equal(np.sort(np.floor(u[:, j] * n)), np.arange(n)) for j in range(3))
    check("LHS: exactly one sample per stratum in every column", strata_ok)
    raises("unknown sampling method", lambda: corr.uniform_design(10, 1, RNG, "sobol"), ValueError)


def test_iman_conover():
    n = 20000
    x = np.column_stack([D.Lognormal(10, 5).sample(n, RNG), D.Normal(0.2, 0.03).sample(n, RNG),
                         D.Triangular(0, 1, 2).sample(n, RNG)])
    target = np.array([[1, 0.7, -0.3], [0.7, 1, 0.0], [-0.3, 0.0, 1]])
    y = corr.iman_conover(x, target, RNG)
    for j in range(3):
        check(f"Iman-Conover preserves marginal {j}", np.array_equal(np.sort(x[:, j]), np.sort(y[:, j])))
    rs = stats.spearmanr(y).statistic
    close("Iman-Conover rho(0,1)", rs[0, 1], 0.7, abs_=0.02)
    close("Iman-Conover rho(0,2)", rs[0, 2], -0.3, abs_=0.02)
    close("Iman-Conover rho(1,2)", rs[1, 2], 0.0, abs_=0.02)
    bad = np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
    fixed, modified = corr.nearest_correlation(bad)
    check("non-PD matrix detected and repaired", modified and np.linalg.eigvalsh(fixed).min() > 0
          and np.allclose(np.diag(fixed), 1))
    raises("self-correlation rejected", lambda: corr.build_matrix(["a", "b"], [("a", "a", 0.5)]), corr.CorrelationError)
    raises("unknown variable rejected", lambda: corr.build_matrix(["a"], [("a", "z", 0.5)]), corr.CorrelationError)


def test_row_orders():
    n = 10000
    keys = np.column_stack([RNG.lognormal(0, 1, n), RNG.lognormal(0, 1, n)])
    orders = corr.row_orders_for_rank_correlation(keys, np.array([[1, 0.6], [0.6, 1]]), RNG)
    check("row orders are permutations", all(np.array_equal(np.sort(o), np.arange(n)) for o in orders))
    close("row-order rank correlation", corr.spearman(keys[orders[0], 0], keys[orders[1], 1]), 0.6, abs_=0.03)


if __name__ == "__main__":
    sys.exit(run("test_distributions", [test_analytic_means, test_lognormal_sd_and_percentiles, test_fits,
                                        test_truncation, test_discrete_and_table, test_validation,
                                        test_serialisation_and_scaling, test_lhs, test_iman_conover, test_row_orders]))
