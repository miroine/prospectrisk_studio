"""Recovery-factor benchmark against permeability: correlations, analogues and the segment's own RF input."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from prospectrisk import distributions as D
from prospectrisk import recovery as RF
from prospectrisk import units as U
from prospectrisk.defaults import default_distribution
from prospectrisk.volumetrics import Expression, constant_inputs
from ui import charts as C
from ui import state
from ui.editors import bump, num, wkey

PRESSURE_UNITS = ["bar", "psia"]


def _pick(p):
    names = [pr.name for pr in p.prospects]
    if not names:
        st.info("Add a prospect first.")
        return None, None
    c1, c2 = st.columns(2)
    default = st.session_state.get("sel_prospect")
    pname = c1.selectbox("Prospect", names, index=names.index(default) if default in names else 0, key="rf_prospect")
    st.session_state.sel_prospect = pname
    pr = p.prospect(pname)
    sname = c2.selectbox("Segment", [s.name for s in pr.segments], key=wkey("rf_segment", pname))
    return pr, pr.segment(sname)


def _p50s(vol) -> dict[str, float]:
    vals = constant_inputs(vol, "p50")
    try:
        out = vol.evaluate_expressions({k: np.array([v]) for k, v in vals.items()}, 1)
        return {k: float(np.asarray(v).ravel()[0]) for k, v in out.items()}
    except ValueError:
        return vals


def page_benchmark() -> None:
    p = state.project()
    st.title("Recovery-factor benchmark")
    pr, seg = _pick(p)
    if seg is None:
        return
    vol = seg.volumetrics
    path = f"rf:{pr.name}:{seg.name}"
    t_oil, t_gas, t_ref = st.tabs(["Oil", "Gas", "References"])
    with t_oil:
        if vol.config.has_oil:
            _oil(p, path, pr, seg)
        else:
            st.info("This segment has no oil phase.")
    with t_gas:
        if vol.config.has_gas:
            _gas(p, path, pr, seg)
        else:
            st.info("This segment has no gas phase.")
    with t_ref:
        st.markdown(
            "**API water drive** (Arps et al., 1967): RF = 0.54898 [φ(1−Sw)/Boi]^0.0422 [k μw/μo]^0.0770 "
            "Sw^−0.1903 (pi/pa)^−0.2159, k in darcy.  \n"
            "**API solution-gas drive** (Arps et al., 1967): RF = 0.41815 [φ(1−Sw)/Bob]^0.1611 [k/μob]^0.0979 "
            "Sw^0.3722 (pb/pa)^0.1741, k in darcy.  \n"
            "**Guthrie & Greenberger** (1955): RF = 0.2719 log k + 0.25569 Sw − 0.1355 log μo − 1.538 φ − 0.00035 h "
            "+ 0.11403, k in mD, h net pay in ft.  \n"
            "**Gas depletion**: RF = 1 − (pa/za)/(pi/zi). **Gas water drive**: RF = 1 − [Ev Sgr/Sgi + (1 − Ev)]·"
            "(zi pa)/(za pi).")
        st.caption("The oil correlations are regressions on mostly US sandstone reservoirs from the 1950s–60s. "
                   "They show how recovery trends with permeability; calibrate against your own analogues before "
                   "relying on them.")
        st.dataframe(pd.DataFrame(RF.DRIVE_RANGES).drop(columns=["Fluid"]).style.format({"Low": "{:.0%}",
                                                                                           "High": "{:.0%}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Source: {RF.DRIVE_RANGES_SOURCE}.")


def _params(path: str, seg, vol) -> RF.OilBenchmarkInputs | None:
    saved = dict(seg.rf_benchmark or {})
    p50 = _p50s(vol)
    base = RF.OilBenchmarkInputs()
    if "porosity" in p50:
        base.porosity = float(np.clip(p50["porosity"], 0.01, 0.6))
    if "hc_saturation" in p50:
        base.sw = float(np.clip(1.0 - p50["hc_saturation"], 0.01, 0.99))
    if "bo" in p50:
        base.bo = float(np.clip(p50["bo"], 0.8, 5.0))
    if "gross_thickness" in p50:
        base.net_pay_ft = float(np.clip(p50["gross_thickness"] * p50.get("net_to_gross", 1.0), 1.0, 16000.0))
    punit = saved.get("pressure_unit", "bar")
    to_psia = 14.5037738 if punit == "bar" else 1.0
    st.markdown("**Reservoir and fluid parameters** (porosity, Sw and Bo default to the segment's P50 inputs)")
    c = st.columns(4)
    phi = c[0].number_input("Porosity", 0.01, 0.6, float(saved.get("porosity", base.porosity)), 0.01,
                            key=wkey(path, "phi"), format="%.3f")
    sw = c[1].number_input("Water saturation", 0.01, 0.99, float(saved.get("sw", base.sw)), 0.01,
                           key=wkey(path, "sw"), format="%.3f")
    bo = c[2].number_input("Bo [rb/stb = Rm3/Sm3]", 0.8, 5.0, float(saved.get("bo", base.bo)), 0.01,
                           key=wkey(path, "bo"), format="%.3f")
    h_m = c[3].number_input("Net pay [m]", 0.1, 5000.0, float(saved.get("net_pay_ft", base.net_pay_ft)) * U.FT_M,
                            key=wkey(path, "h"), format="%.4g")
    c = st.columns(5)
    punit = c[0].selectbox("Pressure unit", PRESSURE_UNITS, index=PRESSURE_UNITS.index(punit), key=wkey(path, "pu"))
    to_psia = 14.5037738 if punit == "bar" else 1.0
    pi = c[1].number_input(f"Initial pressure [{punit}]", 1.0, 2e5, float(saved.get("p_i", base.p_i)) / to_psia,
                           key=wkey(path, "pi", punit), format="%.5g")
    pb = c[2].number_input(f"Bubble point [{punit}]", 1.0, 2e5, float(saved.get("p_b", base.p_b)) / to_psia,
                           key=wkey(path, "pb", punit), format="%.5g")
    pa = c[3].number_input(f"Abandonment [{punit}]", 0.1, 2e5, float(saved.get("p_a", base.p_a)) / to_psia,
                           key=wkey(path, "pa", punit), format="%.5g")
    c4a, c4b = c[4].columns(2)
    mu_o = c4a.number_input("μo [cP]", 0.01, 1e5, float(saved.get("mu_o", base.mu_o)), key=wkey(path, "muo"),
                            format="%.3g")
    mu_w = c4b.number_input("μw [cP]", 0.01, 100.0, float(saved.get("mu_w", base.mu_w)), key=wkey(path, "muw"),
                            format="%.3g")
    params = RF.OilBenchmarkInputs(phi, sw, bo, mu_o, mu_w, pi * to_psia, pb * to_psia, pa * to_psia, h_m / U.FT_M)
    try:
        params.validate()
    except RF.RecoveryError as exc:
        st.error(str(exc))
        return None
    seg.rf_benchmark = {**saved, **params.to_dict(), "pressure_unit": punit}
    return params


def _oil(p, path: str, pr, seg) -> None:
    vol = seg.volumetrics
    params = _params(path, seg, vol)
    if params is None:
        return
    has_k = "permeability" in vol.active_variables()
    run = st.session_state.runs.get(pr.name)
    res = run.segments[seg.name].result if run is not None and seg.name in run.segments else None
    if has_k:
        kd = vol.variables["permeability"]
        k_band = (kd.p90, kd.p50, kd.p10)
        st.caption(f"Permeability from the segment input: P90 {kd.p90:.3g}, P50 {kd.p50:.3g}, P10 {kd.p10:.3g} mD.")
    else:
        st.info("This segment has no permeability input. Enter a range here, or add permeability to the segment so "
                "it is sampled with the other inputs (and can drive the recovery factor through a formula).")
        c1, c2, c3 = st.columns([1, 1, 2])
        k90 = c1.number_input("Permeability P90 [mD]", 1e-4, 1e6, 20.0, key=wkey(path, "k90"), format="%.4g")
        k10 = c2.number_input("Permeability P10 [mD]", 1e-4, 1e6, 800.0, key=wkey(path, "k10"), format="%.4g")
        if k10 <= k90:
            st.error("P10 (high) must exceed P90 (low).")
            return
        kd = D.Lognormal.from_p90_p10(k90, k10)
        k_band = (k90, kd.p50, k10)
        if c3.button("Add this permeability to the segment", key=wkey(path, "addk")):
            vol.variables["permeability"] = kd
            bump()
            st.rerun()
    names = st.multiselect("Correlations", list(RF.CORRELATIONS), default=list(RF.CORRELATIONS),
                           format_func=RF.CORRELATIONS.get, key=wkey(path, "corrs"))
    show_ranges = st.checkbox("Show typical ranges by drive mechanism", value=True, key=wkey(path, "ranges"))
    analogs = _analogs(path)
    rf_band = None
    rf_formula = "rf_oil" in vol.active_expressions()
    if not rf_formula and "rf_oil" in vol.variables:
        d = vol.variables["rf_oil"]
        rf_band = (d.p90, d.p50, d.p10)
    elif res is not None and "rf_oil" in res.inputs:
        q = np.quantile(res.inputs["rf_oil"], [0.1, 0.5, 0.9])
        rf_band = (q[0], q[1], q[2])
    lo_k = max(min(k_band[0], analogs["Permeability [mD]"].min() if analogs is not None else k_band[0]) / 10, 1e-3)
    hi_k = max(k_band[2], analogs["Permeability [mD]"].max() if analogs is not None else k_band[2]) * 10
    try:
        curves = RF.curves(params, lo_k, hi_k, 200, names or None)
    except RF.RecoveryError as exc:
        st.error(str(exc))
        return
    if not names:
        curves = curves[["Permeability [mD]"]]
    st.plotly_chart(C.rf_benchmark_figure(curves, rf_band, k_band, analogs, RF.DRIVE_RANGES if show_ranges else None),
                    width="stretch", key=wkey(path, "rf_fig"))
    st.caption("Curves: correlations at the parameters above. Red band: the segment's oil recovery factor input "
               "(P90–P10, line at P50). Shaded column: permeability P90–P10.")
    if names:
        _trial_comparison(path, vol, params, names, res, kd, has_k)
    _apply_formula(path, vol, params, has_k)


def _analogs(path: str):
    with st.expander("Analogue fields"):
        st.caption("Upload a CSV with permeability (mD) and recovery factor (fraction or %) columns; name, drive and "
                   "fluid columns are optional.")
        st.download_button("Download template", RF.ANALOG_TEMPLATE, "rf_analogues_template.csv", "text/csv",
                           key=wkey(path, "tmpl"))
        up = st.file_uploader("Analogue table", type=["csv", "txt"], key=wkey(path, "analog_up"))
        if up is None:
            return st.session_state.get(f"analogs:{path}")
        try:
            df = RF.read_analogs(up.getvalue())
        except RF.RecoveryError as exc:
            st.error(str(exc))
            return None
        st.session_state[f"analogs:{path}"] = df
        st.dataframe(df, hide_index=True, width="stretch")
        return df


def _trial_comparison(path, vol, params, names, res, kd, has_k) -> None:
    st.markdown("**Recovery factor: input versus correlations, trial by trial**")
    if res is not None and has_k and "permeability" in res.inputs:
        k = res.inputs["permeability"]
        phi = res.inputs.get("porosity")
        sw = 1.0 - res.inputs["hc_saturation"] if "hc_saturation" in res.inputs else None
        bo = res.inputs.get("bo")
        rf_in = res.inputs["rf_oil"]
        st.caption("Uses the last run: each trial's permeability, porosity, saturation and Bo.")
    else:
        rng = np.random.default_rng(1)
        k = kd.sample(5000, rng)
        phi = sw = bo = None
        rf_d = vol.variables.get("rf_oil")
        if rf_d is None or "rf_oil" in vol.active_expressions():
            st.caption("Run the prospect to compare with a formula-defined recovery factor.")
            return
        rf_in = rf_d.sample(5000, rng)
        st.caption("Samples the permeability range with the other parameters fixed at the values above. Run the "
                   "prospect with a permeability input for a trial-by-trial comparison.")
    try:
        tab = RF.compare_trials(rf_in, k, params, names, phi, sw, bo)
    except RF.RecoveryError as exc:
        st.error(str(exc))
        return
    st.dataframe(tab.style.format({c: "{:.1%}" for c in ("P90", "P50", "P10", "Mean")}), hide_index=True,
                 width="stretch")
    series = [("Input", rf_in, C.RED)] + [
        (RF.CORRELATIONS[n], RF.correlation_rf(n, k, params, phi, sw, bo), C.SERIES[i % len(C.SERIES)])
        for i, n in enumerate(names)]
    st.plotly_chart(C.overlay_histograms(series, "Oil recovery factor"), width="stretch", key=wkey(path, "rf_hist"))


def _apply_formula(path, vol, params, has_k) -> None:
    with st.expander("Use a correlation as the segment's oil recovery factor"):
        st.caption("Replaces the oil recovery factor distribution with the correlation, evaluated in every trial "
                   "from that trial's permeability, porosity, saturation and Bo. Recovery uncertainty then follows "
                   "the reservoir inputs, and permeability appears in the sensitivity and explore views.")
        name = st.selectbox("Correlation", list(RF.CORRELATIONS), format_func=RF.CORRELATIONS.get,
                            key=wkey(path, "apply_corr"))
        spread = st.slider("Extra uncertainty around the correlation (± %)", 0, 50, 15, 5, key=wkey(path, "spread"),
                           help="Multiplies the correlation by a triangular factor 1 ± this value, to cover the "
                                "scatter of the regression.")
        st.code(RF.formula(name, params), language="text")
        c1, c2 = st.columns(2)
        if c1.button("Use as the oil recovery factor", type="primary", key=wkey(path, "apply")):
            if not has_k:
                vol.variables["permeability"] = default_distribution("permeability")
            noise = D.Triangular(1 - spread / 100, 1.0, 1 + spread / 100) if spread else None
            vol.expressions["rf_oil"] = Expression(
                RF.formula(name, params),
                units={"permeability": "mD", "porosity": "fraction", "hc_saturation": "fraction", "bo": "rb/stb",
                       "rf_oil": "fraction"},
                noise=noise, noise_mode="multiply")
            bump()
            st.rerun()
        if "rf_oil" in vol.expressions and c2.button("Back to a recovery-factor distribution", key=wkey(path, "unapply")):
            vol.expressions.pop("rf_oil", None)
            bump()
            st.rerun()
        if "rf_oil" in vol.expressions:
            st.success(f"The oil recovery factor is defined by: {vol.expressions['rf_oil'].expr}")


def _gas(p, path: str, pr, seg) -> None:
    vol = seg.volumetrics
    saved = dict(seg.rf_benchmark or {})
    p50 = _p50s(vol)
    st.markdown("**Gas recovery from pressure depletion and water drive**")
    punit = st.selectbox("Pressure unit", PRESSURE_UNITS, index=PRESSURE_UNITS.index(saved.get("pressure_unit", "bar")),
                         key=wkey(path, "gpu"))
    to_psia = 14.5037738 if punit == "bar" else 1.0
    c = st.columns(4)
    pi = c[0].number_input(f"Initial pressure [{punit}]", 1.0, 2e5, float(saved.get("g_pi", 4500.0)) / to_psia,
                           key=wkey(path, "gpi", punit), format="%.5g")
    zi = c[1].number_input("zi", 0.2, 2.0, float(saved.get("g_zi", 0.95)), 0.01, key=wkey(path, "gzi"))
    pa = c[2].number_input(f"Abandonment pressure [{punit}]", 0.1, 2e5, float(saved.get("g_pa", 1000.0)) / to_psia,
                           key=wkey(path, "gpa", punit), format="%.5g")
    za = c[3].number_input("za", 0.2, 2.0, float(saved.get("g_za", 0.92)), 0.01, key=wkey(path, "gza"))
    sgi0 = p50.get("gas_saturation", p50.get("hc_saturation", 0.8))
    c = st.columns(3)
    sgi = c[0].number_input("Initial gas saturation", 0.05, 1.0, float(saved.get("g_sgi", sgi0)), 0.01,
                            key=wkey(path, "gsgi"))
    sgr = c[1].number_input("Trapped (residual) gas saturation", 0.0, 0.9, float(saved.get("g_sgr", 0.3)), 0.01,
                            key=wkey(path, "gsgr"))
    ev = c[2].number_input("Volumetric sweep Ev", 0.0, 1.0, float(saved.get("g_ev", 0.8)), 0.05, key=wkey(path, "gev"))
    if pa >= pi:
        st.error("Abandonment pressure must be below the initial pressure.")
        return
    seg.rf_benchmark = {**saved, "pressure_unit": punit, "g_pi": pi * to_psia, "g_zi": zi, "g_pa": pa * to_psia,
                        "g_za": za, "g_sgi": sgi, "g_sgr": sgr, "g_ev": ev}
    dep = float(RF.gas_depletion(pi, zi, pa, za))
    wd = float(RF.gas_water_drive(pi, zi, pa, za, sgi, sgr, ev))
    rows = [{"Source": "Volumetric depletion (p/z)", "RF": dep},
            {"Source": f"Water drive, Ev {ev:.2f}, Sgr {sgr:.2f}", "RF": wd}]
    if "rf_gas" in vol.variables and "rf_gas" not in vol.active_expressions():
        d = vol.variables["rf_gas"]
        rows.append({"Source": f"Segment input (P90 / P50 / P10: {d.p90:.0%} / {d.p50:.0%} / {d.p10:.0%})", "RF": d.p50})
    st.dataframe(pd.DataFrame(rows).style.format({"RF": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption("Permeability acts mainly through the abandonment pressure: tight or poorly connected gas reservoirs "
               "are abandoned at higher pressure, which lowers depletion recovery. Try a higher abandonment pressure "
               "for the low-permeability case.")
