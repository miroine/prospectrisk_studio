"""Anonymised demonstration project (fictitious plays, prospects and numbers)."""
from __future__ import annotations

from . import distributions as D
from .aggregation import VolumeTruncation
from .project import Play, Project, Prospect, Segment, Settings
from .risk import ChanceFactor
from .units import to_internal
from .volumetrics import AreaDepthTable, Expression, SegmentConfig, SegmentVolumetrics


def _m(dist: D.Distribution, quantity: str, unit: str) -> D.Distribution:
    """Distribution entered in a display unit -> internal units."""
    return D.scaled(dist, to_internal(1.0, quantity, unit))


def example_project() -> Project:
    play_a = Play("Play A – Jurassic shallow marine", [
        ChanceFactor("Source rock presence & maturity", "Source", "adequacy", interpretation="favourable", confidence="high"),
        ChanceFactor("Regional reservoir presence", "Reservoir", probability=0.9),
        ChanceFactor("Regional top seal", "Seal", probability=0.9),
    ], notes="Proven play; producing fields in the area.")
    play_b = Play("Play B – Palaeocene deep-marine fan", [
        ChanceFactor("Source rock presence & maturity", "Source", probability=0.9),
        ChanceFactor("Fan sand delivery to basin", "Reservoir", "adequacy", interpretation="neutral", confidence="medium"),
        ChanceFactor("Regional top seal", "Seal", probability=0.85),
    ], notes="Semi-proven play; one technical discovery.")

    upper = SegmentVolumetrics(
        config=SegmentConfig("area_thickness", phase_probabilities={"oil": 1.0, "gas": 0.0, "oil_gascap": 0.0}),
        variables={
            "area": _m(D.Triangular(8, 12, 18), "area", "km2"),
            "gross_thickness": _m(D.Normal(40, 8, lower=10), "length", "m"),
            "geometric_factor": D.Triangular(0.6, 0.75, 0.9),
            "net_to_gross": D.PERT(0.5, 0.7, 0.85),
            "porosity": D.Normal(0.22, 0.02, lower=0.05, upper=0.35),
            "hc_saturation": D.Triangular(0.6, 0.72, 0.85),
            "bo": D.Triangular(1.2, 1.35, 1.5),
            "rf_oil": D.PERT(0.25, 0.35, 0.45),
            "gor": _m(D.Lognormal(100, 20), "gor", "Sm3/Sm3"),
            "rf_solution_gas": D.PERT(0.25, 0.35, 0.45),
        },
        correlations=[("porosity", "hc_saturation", 0.5), ("net_to_gross", "porosity", 0.4)],
    )
    top = AreaDepthTable(
        depth=list(to_internal([2800, 2850, 2900, 2950, 3000, 3050], "length", "m")),
        area=list(to_internal([0.0, 2.0, 5.5, 9.0, 13.0, 18.0], "area", "km2")),
    )
    lower = SegmentVolumetrics(
        config=SegmentConfig("area_depth", contact_mode="fill_fraction", limit_to_spill=True, top_table=top,
                             phase_probabilities={"oil": 0.5, "gas": 0.2, "oil_gascap": 0.3}),
        variables={
            "gross_thickness": _m(D.Normal(60, 10, lower=20), "length", "m"),
            "fill_fraction": D.PERT(0.3, 0.6, 1.0),
            "spill_depth": _m(D.Constant(3050), "length", "m"),
            "gas_cap_fraction": D.Uniform(0.2, 0.5),
            "net_to_gross": D.Triangular(0.4, 0.6, 0.8),
            "porosity": D.Normal(0.19, 0.025, lower=0.05, upper=0.32),
            "hc_saturation": D.Triangular(0.55, 0.68, 0.8),
            "bo": D.Triangular(1.3, 1.38, 1.45),
            "rf_oil": D.PERT(0.2, 0.3, 0.4),
            "gor": _m(D.Lognormal(120, 25), "gor", "Sm3/Sm3"),
            "bg": D.Triangular(0.0038, 0.0045, 0.0052),
            "rf_gas": D.PERT(0.55, 0.65, 0.75),
            "cgr": _m(D.Lognormal(150, 50), "cgr", "Sm3/MSm3"),
        },
        correlations=[("porosity", "hc_saturation", 0.5)],
    )
    alpha = Prospect(
        "Alpha", play_a.name, status="Prospect", mefs=to_internal(5.0, "oe", "MSm3 o.e."),
        factors=[ChanceFactor("Migration & timing", "Charge", probability=0.8),
                 ChanceFactor("Trap geometry (4-way closure)", "Trap", "adequacy",
                              interpretation="favourable", confidence="medium")],
        segments=[
            Segment("Upper sand", upper, [ChanceFactor("Reservoir effectiveness", "Reservoir", probability=0.85),
                                          ChanceFactor("Top seal integrity", "Seal", probability=0.8)]),
            Segment("Lower sand", lower, [ChanceFactor("Reservoir effectiveness", "Reservoir", probability=0.7),
                                          ChanceFactor("Fault seal", "Seal", "adequacy",
                                                       interpretation="neutral", confidence="low")],
                    truncation=VolumeTruncation("rec_oe", minimum=to_internal(3.0, "oe", "MSm3 o.e."),
                                                min_mode="chance"),
                    notes="Accumulations below 3 MSm3 o.e. would not be recognised as a discovery."),
        ],
        input_dependencies=[{"seg_a": "Upper sand", "var_a": "bo", "seg_b": "Lower sand", "var_b": "bo",
                             "kind": "link", "rho": 0.0}],
        segment_dependency=[[1.0, 0.5], [0.5, 1.0]],
        notes="Stacked Jurassic targets in a tilted fault block.",
    )

    bravo_vol = SegmentVolumetrics(
        config=SegmentConfig("direct", phase_probabilities={"oil": 0.0, "gas": 1.0, "oil_gascap": 0.0}),
        variables={
            "grv": _m(D.Lognormal.from_p90_p10(150, 900), "grv", "1e6 m3"),
            "net_to_gross": D.Triangular(0.8, 0.88, 0.95),
            "porosity": D.Triangular(0.18, 0.24, 0.30),
            "hc_saturation": D.Triangular(0.7, 0.8, 0.88),
            "bg": D.Normal(0.0042, 0.0003, lower=0.003),
            "rf_gas": D.PERT(0.6, 0.72, 0.8),
            "cgr": _m(D.Lognormal(40, 15), "cgr", "Sm3/MSm3"),
            "rf_condensate": D.PERT(0.4, 0.5, 0.6),
        },
        expressions={"hc_saturation": Expression("0.9 - 1.2 * (0.30 - porosity)", units={},
                                                 noise=D.Triangular(0.95, 1.0, 1.05), noise_mode="multiply")},
    )
    bravo = Prospect(
        "Bravo", play_b.name, status="Lead", mefs=to_internal(3.0, "oe", "MSm3 o.e."),
        factors=[ChanceFactor("Migration into fan", "Charge", probability=0.6),
                 ChanceFactor("Stratigraphic trap", "Trap", probability=0.5)],
        segments=[Segment("Main fan", bravo_vol, [ChanceFactor("Reservoir quality", "Reservoir", probability=0.8)])],
        notes="Seismic amplitude-supported fan lobe.",
    )
    return Project("Demo licence – exploration portfolio", Settings(n_trials=10000, seed=20260916),
                   [play_a, play_b], [alpha, bravo],
                   portfolio_dependency=[["Alpha", "Bravo", 0.0]],
                   notes="Fictitious data for demonstration only.")
