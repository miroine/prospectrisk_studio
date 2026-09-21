"""Recovery-factor correlations and the trial explorer (filters, correlations)."""
import sys

import numpy as np
from scipy import stats

from _harness import check, close, raises, run
from prospectrisk import distributions as D
from prospectrisk import explore as E
from prospectrisk import recovery as RF
from prospectrisk import volumetrics as V
from prospectrisk.examples import example_project
from prospectrisk.project import run_prospect

C = D.Constant


def test_api_water_drive_matches_published_per_acre_ft_form():
    phi, sw, bo, k, mu_w, mu_o, pi, pa = 0.2, 0.25, 1.3, 100.0, 0.5, 2.0, 4000.0, 1000.0
    rf = float(RF.api_water_drive(phi, sw, bo, k, mu_w, mu_o, pi, pa))
    per_acre_ft = 4259 * (phi * (1 - sw) / bo) ** 1.0422 * (k / 1000 * mu_w / mu_o) ** 0.0770 * sw ** -0.1903 \
        * (pi / pa) ** -0.2159
    close("API water drive: fraction form × 7758 φ(1−Sw)/Bo = published 4259 form",
          rf * 7758.367 * phi * (1 - sw) / bo, per_acre_ft, rel=2e-4)
    check("API water drive in the textbook water-drive range (35–75 %)", 0.35 <= rf <= 0.75, f"{rf:.3f}")


def test_guthrie_greenberger_and_solution_gas():
    k, sw, mu_o, phi, h = 100.0, 0.25, 2.0, 0.2, 100.0
    gg = float(RF.guthrie_greenberger(k, sw, mu_o, phi, h))
    rounded = 0.114 + 0.272 * np.log10(k) + 0.256 * sw - 0.136 * np.log10(mu_o) - 1.538 * phi - 0.00035 * h
    close("Guthrie–Greenberger vs published rounded coefficients", gg, rounded, abs_=0.002)
    sg = float(RF.api_solution_gas(0.2, 0.25, 1.3, 100.0, 1.0, 3000.0, 500.0))
    check("API solution gas in the textbook range (5–30 %)", 0.05 <= sg <= 0.30, f"{sg:.3f}")
    ks = np.logspace(-1, 4, 50)
    p = RF.OilBenchmarkInputs()
    for nm in RF.CORRELATIONS:
        v = RF.correlation_rf(nm, ks, p)
        check(f"{nm}: RF increases with permeability", np.all(np.diff(v) >= -1e-12))
        check(f"{nm}: RF within [0, 1]", v.min() >= 0 and v.max() <= 1)
    check("GG clipped at zero for tight rock", float(RF.guthrie_greenberger(0.01, 0.2, 5, 0.3, 300)) == 0.0)
    raises("negative permeability rejected", lambda: RF.api_water_drive(0.2, 0.2, 1.2, -1, 0.5, 1, 4000, 1000),
           RF.RecoveryError)
    raises("abandonment above initial pressure rejected",
           lambda: RF.OilBenchmarkInputs(p_i=1000, p_a=2000).validate(), RF.RecoveryError)
    df = RF.curves(p, 1, 1000, 30)
    check("curve table has one column per correlation", df.shape == (30, 4))


def test_gas_recovery():
    close("p/z depletion", float(RF.gas_depletion(5000, 0.95, 1000, 0.90)), 1 - (1000 / 0.9) / (5000 / 0.95), rel=1e-12)
    wd = float(RF.gas_water_drive(5000, 0.95, 3000, 0.9, 0.8, 0.3, 1.0))
    close("water-drive gas with trapped gas", wd, 1 - (0.3 / 0.8) * (0.95 * 3000) / (0.9 * 5000), rel=1e-12)
    check("partial sweep lowers water-drive gas recovery",
          float(RF.gas_water_drive(5000, 0.95, 3000, 0.9, 0.8, 0.3, 0.7)) < wd)


def test_correlation_as_formula_matches_function():
    p = RF.OilBenchmarkInputs(mu_o=1.5, mu_w=0.4, p_i=4500, p_b=3500, p_a=1200, net_pay_ft=80)
    for nm in RF.CORRELATIONS:
        seg = V.SegmentVolumetrics(V.SegmentConfig("area_thickness"), {
            "area": C(1000), "gross_thickness": C(100), "geometric_factor": C(1), "net_to_gross": C(0.8),
            "porosity": D.Normal(0.22, 0.02, lower=0.05, upper=0.35), "hc_saturation": D.Triangular(0.6, 0.72, 0.85),
            "bo": D.Triangular(1.2, 1.3, 1.45), "rf_oil": C(0.3),
            "permeability": D.Lognormal.from_p90_p10(30, 900)})
        seg.expressions["rf_oil"] = V.Expression(RF.formula(nm, p))
        res = V.simulate(seg, 3000, 7)
        want = RF.correlation_rf(nm, res.inputs["permeability"], p, res.inputs["porosity"],
                                 1 - res.inputs["hc_saturation"], res.inputs["bo"])
        check(f"{nm}: formula in the segment = benchmark function trial by trial",
              np.allclose(res.inputs["rf_oil"], want, rtol=1e-10, atol=1e-12))
        check(f"{nm}: RF uncertainty now follows permeability",
              stats.spearmanr(res.inputs["permeability"], res.inputs["rf_oil"]).statistic > 0.5)


def test_analog_reader():
    df = RF.read_analogs("Name;Permeability_mD;RF (%);Drive\nA;100;35;water\nB;1000;52;water\nC;-5;20;x\n")
    check("analogue CSV: semicolons, % converted, invalid rows dropped",
          len(df) == 2 and abs(df["RF"].iloc[1] - 0.52) < 1e-12 and df["Name"].iloc[0] == "A")
    raises("missing columns rejected", lambda: RF.read_analogs("a,b\n1,2\n"), RF.RecoveryError)
    check("template parses", len(RF.read_analogs(RF.ANALOG_TEMPLATE)) == 1)


def test_trial_table_and_filters():
    p = example_project()
    r = run_prospect(p, "Alpha")
    df, meta = E.trial_table(p, r)
    check("one row per trial", len(df) == p.settings.n_trials)
    tot = [c for c, m in meta.items() if m.kind == "total"][0]
    parts = [c for c, m in meta.items() if m.kind == "result" and m.key == "rec_oe"]
    check("prospect total = sum of segment results in every trial",
          np.allclose(df[tot], df[parts].sum(axis=1), rtol=1e-12))
    phi = [c for c, m in meta.items() if m.key == "porosity" and m.segment == "Upper sand"][0]
    close("porosity in display units matches the simulation", float(df[phi].mean()),
          float(r.segments["Upper sand"].result.inputs["porosity"].mean()), rel=1e-12)
    cut = float(df[phi].median())
    keep = E.apply_filters(df, [E.TrialFilter(phi, "keep", low=cut)])
    check("keep filter: all kept trials satisfy it", df.loc[keep, phi].min() >= cut and abs(keep.mean() - 0.5) < 0.01)
    excl = E.apply_filters(df, [E.TrialFilter(phi, "exclude", low=cut)])
    check("exclude is the complement of keep", np.array_equal(excl, ~keep))
    phase = [c for c, m in meta.items() if m.kind == "phase" and m.segment == "Lower sand"][0]
    both = E.apply_filters(df, [E.TrialFilter(phi, "keep", low=cut), E.TrialFilter(phase, "keep", values=["Oil"])])
    check("filters combine with AND", np.array_equal(both, keep & (df[phase] == "Oil").to_numpy()))
    comp = E.compare(df, tot, keep)
    check("comparison counts add up", comp["Trials"].iloc[1] + comp["Trials"].iloc[2] == comp["Trials"].iloc[0])
    upper_rec = parts[0]
    check("high-porosity trials have higher Upper sand volume",
          df.loc[keep, upper_rec].median() > df.loc[~keep, upper_rec].median())
    cols = [phi, upper_rec, tot]
    m = E.spearman_matrix(df, cols)
    check("Spearman matrix symmetric with unit diagonal",
          np.allclose(m.to_numpy(), m.to_numpy().T) and np.allclose(np.diag(m.to_numpy()), 1))
    close("Spearman entry matches scipy", float(m.loc[phi, upper_rec]),
          float(stats.spearmanr(df[phi], df[upper_rec]).statistic), rel=1e-12)
    xm, ym, lo, hi = E.binned_trend(df[phi].to_numpy(), df[upper_rec].to_numpy(), 10)
    check("binned trend rises with porosity", ym[-1] > ym[0] and np.all(lo <= ym) and np.all(ym <= hi))
    check("rank correlation of constant column is NaN", np.isnan(E.rank_correlation(np.ones(10), np.arange(10))))
    flag = [c for c, m_ in meta.items() if m_.kind == "flag"]
    check("volume cut-off flag column present for the truncated segment", len(flag) == 1)
    raises("unknown filter column rejected", lambda: E.apply_filters(df, [E.TrialFilter("nope")]), KeyError)


if __name__ == "__main__":
    sys.exit(run("test_recovery_explore", [
        test_api_water_drive_matches_published_per_acre_ft_form, test_guthrie_greenberger_and_solution_gas,
        test_gas_recovery, test_correlation_as_formula_matches_function, test_analog_reader,
        test_trial_table_and_filters]))
