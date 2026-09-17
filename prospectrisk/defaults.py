"""Default inputs used when a variable is first switched on or a new segment is created.

Values are neutral placeholders in internal units, chosen only so that a new
model runs; every one should be replaced with prospect-specific data.
"""
from __future__ import annotations

from . import distributions as D
from .project import Prospect, Segment
from .volumetrics import AreaDepthTable, SegmentConfig, SegmentVolumetrics, required_variables

VARIABLE_HELP = {
    "area": "Area of the closure (or of the mapped hydrocarbon accumulation) in plan view.",
    "gross_thickness": "Vertical gross reservoir thickness. For area-depth GRV this is the vertical offset of the base surface.",
    "geometric_factor": "Correction from area × thickness to the true trap volume (shape of the closure vs thickness).",
    "grv": "Gross rock volume of the hydrocarbon-bearing interval, when computed outside the tool.",
    "contact_depth": "Depth of the deepest hydrocarbon contact (OWC or GWC).",
    "column_height": "Hydrocarbon column height measured from the crest.",
    "fill_fraction": "Fraction of the crest-to-spill relief filled with hydrocarbons.",
    "spill_depth": "Spill-point depth; defaults to the deepest contour of the table.",
    "gas_cap_fraction": "Share of the hydrocarbon column (area-depth) or of GRV (other methods) occupied by the gas cap.",
    "net_to_gross": "Net reservoir thickness divided by gross thickness.",
    "porosity": "Average effective porosity of net reservoir.",
    "hc_saturation": "Average hydrocarbon saturation (1 − Sw) of the oil leg.",
    "gas_saturation": "Average hydrocarbon saturation of the gas zone if different from the oil leg.",
    "bo": "Oil formation volume factor.",
    "rf_oil": "Recovery factor for oil.",
    "gor": "Solution gas-oil ratio of the oil.",
    "rf_solution_gas": "Recovery factor for solution gas; defaults to the oil recovery factor.",
    "bg": "Gas formation volume factor (reservoir volume per standard volume) = 1 / expansion factor.",
    "eg": "Gas expansion factor E (standard volume per reservoir volume) = 1 / Bg.",
    "rf_gas": "Recovery factor for free gas.",
    "cgr": "Condensate-gas ratio of the free gas.",
    "rf_condensate": "Recovery factor for condensate; defaults to the gas recovery factor.",
}


def default_distribution(name: str, cfg: SegmentConfig | None = None) -> D.Distribution:
    top = cfg.top_table if cfg is not None else None
    relief = (top.base - top.crest) if top is not None else 500.0
    crest = top.crest if top is not None else 8000.0
    table = {
        "area": lambda: D.Triangular(1000, 2500, 5000),
        "gross_thickness": lambda: D.Normal(150, 30, lower=1),
        "geometric_factor": lambda: D.Triangular(0.6, 0.75, 0.9),
        "grv": lambda: D.Lognormal.from_p90_p10(1e5, 6e5),
        "contact_depth": lambda: D.Triangular(crest + 0.3 * relief, crest + 0.6 * relief, crest + relief),
        "column_height": lambda: D.Triangular(0.2 * relief, 0.5 * relief, 0.9 * relief),
        "fill_fraction": lambda: D.PERT(0.3, 0.6, 1.0),
        "spill_depth": lambda: D.Constant(crest + relief),
        "gas_cap_fraction": lambda: D.Uniform(0.2, 0.5),
        "net_to_gross": lambda: D.PERT(0.5, 0.7, 0.9),
        "porosity": lambda: D.Normal(0.2, 0.03, lower=0.01, upper=0.4),
        "hc_saturation": lambda: D.Triangular(0.6, 0.7, 0.85),
        "gas_saturation": lambda: D.Triangular(0.65, 0.75, 0.88),
        "bo": lambda: D.Triangular(1.2, 1.35, 1.5),
        "rf_oil": lambda: D.PERT(0.2, 0.3, 0.45),
        "gor": lambda: D.Lognormal(600, 150),
        "rf_solution_gas": lambda: D.PERT(0.2, 0.3, 0.45),
        "bg": lambda: D.Triangular(0.004, 0.0045, 0.005),
        "eg": lambda: D.Triangular(200, 222, 250),
        "rf_gas": lambda: D.PERT(0.55, 0.7, 0.8),
        "cgr": lambda: D.Lognormal(30, 10),
        "rf_condensate": lambda: D.PERT(0.4, 0.5, 0.6),
    }
    return table[name]()


def default_area_depth_table() -> AreaDepthTable:
    return AreaDepthTable([8000, 8200, 8400, 8600, 8800], [0, 800, 2000, 3500, 5000])


def ensure_required(vol: SegmentVolumetrics) -> list[str]:
    """Add default distributions for required inputs that are missing. Returns names added."""
    cfg = vol.config
    if cfg.grv_method == "area_depth" and cfg.top_table is None:
        cfg.top_table = default_area_depth_table()
    req, _ = required_variables(cfg)
    added = []
    for name in req:
        if name not in vol.variables:
            vol.variables[name] = default_distribution(name, cfg)
            added.append(name)
    return added


def new_segment(name: str) -> Segment:
    vol = SegmentVolumetrics(SegmentConfig("area_thickness"))
    ensure_required(vol)
    return Segment(name, vol)


def new_prospect(name: str, play: str) -> Prospect:
    return Prospect(name, play, segments=[new_segment("Main segment")])


def unique_name(base: str, existing: list[str]) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"
