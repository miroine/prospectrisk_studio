"""Cross-section and plan-view geometry with hydrocarbon fill.

Two kinds of section, both returned in internal feet:

* **Synthetic** — from the area-depth table alone. Each contour is replaced by a
  circle of the same area (equivalent radius r = sqrt(A / pi)), giving a
  symmetric section through an idealised trap. Available for every area-depth
  segment, with or without a map.
* **Map section** — along a line across the top-structure grid (imported,
  gridded from points, or interpolated from contours), with the base from a
  base grid or the top shifted down by the gross thickness.

The fluid fill is drawn from a *case*: contact, gas-oil contact, thickness and
phase taken either from input percentiles or from a single Monte Carlo trial
(e.g. the trial whose recoverable volume is closest to the success-case P50).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .maps import Grid, MapError, StructureMap, _ft_per, closure_region
from .volumetrics import (KEY_OUTPUT, PHASE_LABELS, PHASES, AreaDepthTable, SegmentVolumetrics, VolumetricResult,
                          constant_inputs)

FLUID_COLOURS = {"gas": "#E0413A", "oil": "#3B8C3A", "water": "#8DC3E8"}
PERCENTILES = {"P90": 90.0, "P50": 50.0, "P10": 10.0}


class SectionError(ValueError):
    pass


@dataclass
class SectionCase:
    label: str
    phase: str
    contact: float                   # ft, deepest hydrocarbon contact
    goc: float | None                # ft, gas-oil contact (None = no gas)
    thickness: float | None          # ft, used when there is no base surface
    spill: float | None              # ft
    crest: float                     # ft

    @property
    def column(self) -> float:
        return max(self.contact - self.crest, 0.0)


def _need_area_depth(vol: SegmentVolumetrics) -> AreaDepthTable:
    cfg = vol.config
    if cfg.grv_method != "area_depth" or cfg.top_table is None:
        raise SectionError("Sections need the area-depth GRV method (a top-structure table or a map)")
    return cfg.top_table


def _goc(phase: str, crest: float, contact: float, gcf: float) -> float | None:
    if phase == "gas":
        return contact
    if phase == "oil_gascap":
        return crest + gcf * (contact - crest)
    return None


def _default_phase(vol: SegmentVolumetrics) -> str:
    p = vol.config.phase_probabilities
    return max(PHASES, key=lambda k: p.get(k, 0.0))


def case_from_percentile(vol: SegmentVolumetrics, pct: str = "P50", phase: str | None = None) -> SectionCase:
    """Geometry inputs at one exceedance percentile (P90 = low case), the rest at P50."""
    top = _need_area_depth(vol)
    cfg = vol.config
    phase = phase or _default_phase(vol)
    if phase not in PHASES:
        raise SectionError(f"Unknown phase '{phase}'")
    q = PERCENTILES[pct]
    vals = constant_inputs(vol, "p50")
    sampled = vol.sampled_inputs()
    for k in ("contact_depth", "column_height", "fill_fraction", "gross_thickness"):
        if k in sampled:
            vals[k] = sampled[k].exceedance(q)
    vals = {k: float(np.asarray(v).ravel()[0])
            for k, v in vol.evaluate_expressions({k: np.array([v]) for k, v in vals.items()}, 1).items()}
    crest = top.crest
    spill = vals.get("spill_depth", top.base)
    if cfg.contact_mode == "contact_depth":
        contact = vals["contact_depth"]
    elif cfg.contact_mode == "column_height":
        contact = crest + vals["column_height"]
    else:
        contact = crest + vals["fill_fraction"] * (spill - crest)
    contact = max(contact, crest)
    if cfg.limit_to_spill:
        contact = min(contact, spill)
    gcf = float(np.clip(vals.get("gas_cap_fraction", 0.0), 0.0, 1.0))
    thick = None if cfg.base_table is not None else vals.get("gross_thickness")
    return SectionCase(f"{pct} inputs, {PHASE_LABELS[phase].lower()}", phase, contact, _goc(phase, crest, contact, gcf),
                       thick, spill, crest)


def trial_near_percentile(result: VolumetricResult, pct: str = "P50", output: str = KEY_OUTPUT) -> int:
    v = np.asarray(result.outputs[output], float)
    target = np.quantile(v, 1.0 - PERCENTILES[pct] / 100.0)
    return int(np.argmin(np.abs(v - target)))


def case_from_trial(vol: SegmentVolumetrics, result: VolumetricResult, index: int, label: str | None = None) -> SectionCase:
    top = _need_area_depth(vol)
    if "contact" not in result.outputs:
        raise SectionError("This run has no contact output; re-run the prospect")
    if not 0 <= index < result.n:
        raise SectionError(f"Trial {index + 1} does not exist")
    phase = PHASES[int(result.phase[index])]
    contact = float(result.outputs["contact"][index])
    gcf = float(result.inputs["gas_cap_fraction"][index]) if "gas_cap_fraction" in result.inputs else 0.0
    thick = None if vol.config.base_table is not None else float(result.inputs["gross_thickness"][index]) \
        if "gross_thickness" in result.inputs else None
    spill = float(result.inputs["spill_depth"][index]) if "spill_depth" in result.inputs else top.base
    return SectionCase(label or f"Trial {index + 1}, {PHASE_LABELS[phase].lower()}", phase, contact,
                       _goc(phase, top.crest, contact, gcf), thick, spill, top.crest)


# =====================================================================================
# Geometry
# =====================================================================================

@dataclass
class SectionGeometry:
    x: np.ndarray                     # ft along section
    top: np.ndarray                   # ft depth
    base: np.ndarray                  # ft depth
    bands: dict[str, tuple[np.ndarray, np.ndarray]]
    lines: dict[str, float]
    case: SectionCase
    kind: str                         # "synthetic" | "map"
    notes: list[str] = field(default_factory=list)

    def hc_thickness(self) -> np.ndarray:
        return sum(lo - up for k, (up, lo) in self.bands.items() if k != "water")


def _radius(tab: AreaDepthTable) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(tab.depth, float)
    r = np.sqrt(np.asarray(tab.area, float) * 43560.0 / np.pi)
    r = r + np.arange(r.size) * 1e-9 * max(r[-1], 1.0)  # strictly increasing for interpolation
    return r, d


def _surface(r_abs: np.ndarray, r: np.ndarray, d: np.ndarray) -> np.ndarray:
    z = np.interp(r_abs, r, d)
    if r.size >= 2 and r[-1] > r[-2]:
        slope = (d[-1] - d[-2]) / (r[-1] - r[-2])
        z = np.where(r_abs > r[-1], d[-1] + (r_abs - r[-1]) * slope, z)
    return z


def synthetic_section(vol: SegmentVolumetrics, case: SectionCase, n: int = 401) -> SectionGeometry:
    top_tab = _need_area_depth(vol)
    r, d = _radius(top_tab)
    if r[-1] <= 0:
        raise SectionError("The area-depth table has zero area")
    half = 1.15 * r[-1]
    x = np.linspace(-half, half, n)
    top = _surface(np.abs(x), r, d)
    notes = ["Idealised symmetric section: each contour is a circle with the mapped area."]
    if vol.config.base_table is not None:
        rb, db = _radius(vol.config.base_table)
        base = _surface(np.abs(x), rb, db)
    else:
        if case.thickness is None:
            raise SectionError("Gross thickness is needed to draw the base of the reservoir")
        base = top + case.thickness
    lines = {"Crest": case.crest, "Contact": case.contact}
    if case.goc is not None and case.phase == "oil_gascap":
        lines["GOC"] = case.goc
    if case.spill is not None:
        lines["Spill"] = case.spill
    contact = np.full_like(x, case.contact)
    goc = None if case.goc is None else np.full_like(x, case.goc)
    return SectionGeometry(x, top, base, _bands_varying(top, base, contact, goc), lines, case, "synthetic", notes)


def default_line(smap: StructureMap, g: Grid) -> list[float]:
    """West-east line through the crest, across the full map."""
    i, j = smap.crest_index(g)
    cx, cy = g.xy_of(i, j)
    x1, x2, _, _ = g.extent
    return [x1, cy, x2, cy]


def map_section(smap: StructureMap, vol: SegmentVolumetrics, case: SectionCase, line: list[float] | None = None,
                n: int = 401) -> SectionGeometry:
    g = smap.top_grid()
    if g is None:
        raise SectionError("The map has no surface to section")
    line = line or smap.section or default_line(smap, g)
    x1, y1, x2, y2 = (float(v) for v in line)
    if (x1, y1) == (x2, y2):
        raise SectionError("Section line end points must differ")
    xs, ys = np.linspace(x1, x2, n), np.linspace(y1, y2, n)
    xy_ft, z_ft = _ft_per(smap.xy_unit), _ft_per(smap.z_unit)
    top = g.sample(xs, ys) * z_ft
    if np.all(~np.isfinite(top)):
        raise SectionError("The section line does not cross the defined part of the map")
    dist = np.hypot(xs - x1, ys - y1) * xy_ft
    notes = []
    if smap.base is not None:
        base = smap.base.sample(xs, ys) * z_ft
    else:
        if case.thickness is None:
            raise SectionError("Gross thickness is needed to draw the base of the reservoir")
        base = top + case.thickness
        notes.append("Base drawn as the top shifted down by the gross thickness.")
    top_region = closure_region(g.z, case.contact / z_ft, smap.crest_index(g))
    in_closure = _closure_along(g, top_region, xs, ys)
    # hydrocarbons only inside the closure connected to the crest
    contact_line = np.where(in_closure, case.contact, -np.inf)
    goc_line = None if case.goc is None else np.where(in_closure, case.goc, -np.inf)
    bands = _bands_varying(top, base, contact_line, goc_line)
    lines = {"Contact": case.contact}
    if case.goc is not None and case.phase == "oil_gascap":
        lines["GOC"] = case.goc
    if case.spill is not None:
        lines["Spill"] = case.spill
    return SectionGeometry(dist, top, base, bands, lines, case, "map", notes)


def _closure_along(g: Grid, region: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    ii = np.clip(np.rint((ys - g.y0) / g.dy).astype(int), 0, g.ny - 1)
    jj = np.clip(np.rint((xs - g.x0) / g.dx).astype(int), 0, g.nx - 1)
    return region[ii, jj]


def _bands_varying(top, base, contact, goc) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Gas / oil / water intervals between top and base for (possibly position-dependent) contacts."""
    top, base = np.asarray(top, float), np.asarray(base, float)
    base = np.where(np.isfinite(base) & np.isfinite(top), np.maximum(base, top), np.nan)
    contact = np.asarray(contact, float)
    goc = np.full_like(top, -np.inf) if goc is None else np.asarray(goc, float)
    gas_lo = np.clip(np.minimum(goc, contact), top, base)
    oil_lo = np.clip(contact, gas_lo, base)
    return {"gas": (top, gas_lo), "oil": (gas_lo, oil_lo), "water": (oil_lo, base)}


# =====================================================================================
# Plan view
# =====================================================================================

@dataclass
class PlanView:
    grid: Grid
    hc: np.ndarray                    # bool, hydrocarbon-bearing top area
    gas: np.ndarray                   # bool, gas-bearing top area
    crest_xy: tuple[float, float]
    spill_xy: tuple[float, float] | None
    line: list[float]
    contact_native: float
    goc_native: float | None


def plan_view(smap: StructureMap, case: SectionCase, spill_xy: tuple[float, float] | None = None) -> PlanView:
    g = smap.top_grid()
    if g is None:
        raise MapError("The map has no surface")
    zf = _ft_per(smap.z_unit)
    crest = smap.crest_index(g)
    hc = closure_region(g.z, case.contact / zf, crest)
    gas = closure_region(g.z, case.goc / zf, crest) if case.goc is not None else np.zeros_like(hc)
    return PlanView(g, hc, gas, g.xy_of(*crest), spill_xy, smap.section or default_line(smap, g),
                    case.contact / zf, None if case.goc is None else case.goc / zf)


# =====================================================================================
# Static rendering (PDF)
# =====================================================================================

def render_section_mpl(ax, geom: SectionGeometry, length_unit: str = "m", title: str | None = None) -> None:
    from .units import from_internal

    f = lambda a: from_internal(np.asarray(a, float), "length", length_unit)  # noqa: E731
    x = f(geom.x)
    for name in ("water", "oil", "gas"):
        up, lo = geom.bands[name]
        ok = np.isfinite(up) & np.isfinite(lo)
        if not np.any((lo - up)[ok] > 0):
            continue  # no such fluid in this case: keep it out of the legend
        ax.fill_between(x, f(up), f(lo), where=ok, color=FLUID_COLOURS[name], linewidth=0, label=name.capitalize())
    ax.plot(x, f(geom.top), color="#00243D", lw=1.4, label="Top reservoir")
    ax.plot(x, f(geom.base), color="#00243D", lw=0.9, ls="--", label="Base reservoir")
    for name, z in geom.lines.items():
        if name == "Crest":
            continue
        ax.axhline(float(f(z)), color="#6F6F6F" if name == "Spill" else "#00243D", lw=0.8, ls=":")
        ax.text(x[-1], float(f(z)), f" {name}", va="center", ha="left", fontsize=7, color="#00243D")
    ax.invert_yaxis()
    ax.set_xlabel(f"Distance [{length_unit}]", fontsize=8)
    ax.set_ylabel(f"Depth [{length_unit}]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.set_title(title or geom.case.label, fontsize=9)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5, frameon=False)
