"""Tornado and rank-correlation sensitivity."""
import sys

import numpy as np

from _harness import check, close, run
from prospectrisk import distributions as D
from prospectrisk import sensitivity as S
from prospectrisk import volumetrics as V


def seg():
    return V.SegmentVolumetrics(V.SegmentConfig("area_thickness"), {
        "area": D.Triangular(500, 1000, 3000), "gross_thickness": D.Constant(100), "geometric_factor": D.Constant(1),
        "net_to_gross": D.Uniform(0.69, 0.71), "porosity": D.Normal(0.2, 0.02), "hc_saturation": D.Constant(0.8),
        "bo": D.Uniform(1.1, 1.4), "rf_oil": D.Constant(0.3)})


def test_tornado():
    s = seg()
    base, rows = S.tornado(s, "rec_oe", "oil")
    close("tornado base = deterministic at P50", base,
          V.deterministic(s, V.constant_inputs(s, "p50"), "oil")["rec_oe"], rel=1e-12)
    check("tornado ranks area first (widest input)", rows[0]["key"] == "area", rows[0]["key"])
    check("tornado ranks N/G last (narrowest input)", rows[-1]["key"] == "net_to_gross")
    bo = next(r for r in rows if r["key"] == "bo")
    check("Bo acts inversely", bo["Output at high input"] < bo["Output at low input"])
    area = next(r for r in rows if r["key"] == "area")
    close("area swing is linear", area["Output at high input"] / area["Output at low input"],
          area["Input high"] / area["Input low"], rel=1e-9)
    check("constants excluded", all(r["key"] not in ("rf_oil", "hc_saturation") for r in rows))


def test_rank_sensitivity():
    s = seg()
    res = V.simulate(s, 20000, 5)
    rows = S.rank_sensitivity(res.inputs, res.outputs["rec_oe"])
    close("contributions sum to 1 in magnitude", sum(abs(r["Contribution to variance"]) for r in rows), 1.0, rel=1e-9)
    check("area dominates", rows[0]["key"] == "area")
    check("Bo negative correlation", next(r for r in rows if r["key"] == "bo")["Rank correlation"] < 0)


if __name__ == "__main__":
    sys.exit(run("test_sensitivity", [test_tornado, test_rank_sensitivity]))
