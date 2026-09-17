"""Success-case volumetrics for one reservoir segment.

Pipeline per Monte Carlo trial::

    GRV  (area x thickness x geometric factor | direct | area-depth + contact)
     -> split into oil leg / gas cap according to the sampled fluid phase
     -> HCPV = GRV x N/G x porosity x Sh
     -> STOIIP = HCPV / Bo,   GIIP = HCPV / Bg
     -> associated gas (GOR), condensate (CGR)
     -> recoverable = in-place x recovery factor
     -> oil equivalent

Internal units are oilfield units (see :mod:`prospectrisk.units`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import correlation as corr
from . import expressions as ex
from .distributions import Constant, Distribution, from_dict
from .units import (BBL_PER_ACRE_FT, BOE_CONVENTIONS, DEFAULT_BOE_CONVENTION, FT3_PER_ACRE_FT, from_internal,
                    internal_unit, is_valid_unit, to_internal)

PHASES = ("oil", "gas", "oil_gascap")
PHASE_LABELS = {"oil": "Oil", "gas": "Gas / condensate", "oil_gascap": "Oil with gas cap"}
GRV_METHODS = {
    "area_thickness": "Area × thickness × geometric factor",
    "direct": "Direct GRV",
    "area_depth": "Area-depth table + contact",
}
CONTACT_MODES = {
    "contact_depth": "Hydrocarbon contact depth",
    "column_height": "Hydrocarbon column height",
    "fill_fraction": "Fill fraction of spill-point closure",
}

# Variable catalogue: name -> (label, quantity, bounds)
VARIABLES: dict[str, tuple[str, str, tuple[float, float]]] = {
    "area": ("Closure area", "area", (0.0, np.inf)),
    "gross_thickness": ("Gross thickness", "length", (0.0, np.inf)),
    "geometric_factor": ("Geometric factor", "fraction", (0.0, 1.0)),
    "grv": ("Gross rock volume", "grv", (0.0, np.inf)),
    "contact_depth": ("HC contact depth", "length", (-np.inf, np.inf)),
    "column_height": ("HC column height", "length", (0.0, np.inf)),
    "fill_fraction": ("Fill fraction", "fraction", (0.0, 1.0)),
    "spill_depth": ("Spill-point depth", "length", (-np.inf, np.inf)),
    "gas_cap_fraction": ("Gas-cap fraction", "fraction", (0.0, 1.0)),
    "net_to_gross": ("Net-to-gross", "fraction", (0.0, 1.0)),
    "porosity": ("Porosity", "fraction", (0.0, 1.0)),
    "hc_saturation": ("HC saturation (oil)", "fraction", (0.0, 1.0)),
    "gas_saturation": ("HC saturation (gas)", "fraction", (0.0, 1.0)),
    "bo": ("Bo", "bo", (1e-6, np.inf)),
    "rf_oil": ("Oil recovery factor", "fraction", (0.0, 1.0)),
    "gor": ("Solution GOR", "gor", (0.0, np.inf)),
    "rf_solution_gas": ("Solution gas recovery factor", "fraction", (0.0, 1.0)),
    "bg": ("Bg", "bg", (1e-9, np.inf)),
    "eg": ("Gas expansion factor E", "eg", (1e-9, np.inf)),
    "rf_gas": ("Gas recovery factor", "fraction", (0.0, 1.0)),
    "cgr": ("Condensate-gas ratio", "cgr", (0.0, np.inf)),
    "rf_condensate": ("Condensate recovery factor", "fraction", (0.0, 1.0)),
}

OUTPUTS: dict[str, tuple[str, str]] = {
    "grv_total": ("GRV (HC-bearing)", "grv"),
    "grv_oil": ("GRV oil leg", "grv"),
    "grv_gas": ("GRV gas", "grv"),
    "hcpv_oil": ("HCPV oil [rb]", "none"),
    "hcpv_gas": ("HCPV gas [rcf]", "none"),
    "stoiip": ("STOIIP", "oil"),
    "giip_free": ("Free GIIP", "gas"),
    "giip_solution": ("Solution gas in place", "gas"),
    "cond_ip": ("Condensate in place", "oil"),
    "inplace_liquids": ("Liquids in place", "oil"),
    "inplace_gas": ("Gas in place (total)", "gas"),
    "inplace_oe": ("In place oil equivalent", "oe"),
    "rec_oil": ("Recoverable oil", "oil"),
    "rec_condensate": ("Recoverable condensate", "oil"),
    "rec_free_gas": ("Recoverable free gas", "gas"),
    "rec_solution_gas": ("Recoverable solution gas", "gas"),
    "rec_liquids": ("Recoverable liquids", "oil"),
    "rec_gas": ("Recoverable gas (total)", "gas"),
    "rec_oe": ("Recoverable oil equivalent", "oe"),
}
KEY_OUTPUT = "rec_oe"

FRACTION_DEFAULTS = {"gas_saturation": "hc_saturation", "rf_solution_gas": "rf_oil", "rf_condensate": "rf_gas"}


class VolumetricsError(ValueError):
    pass


@dataclass
class AreaDepthTable:
    """Area enclosed by a depth contour. Depth increasing downwards, internal units (ft, acres)."""

    depth: list[float]
    area: list[float]
    _grid: np.ndarray = field(init=False, repr=False)
    _cum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        d, a = np.asarray(self.depth, float), np.asarray(self.area, float)
        if d.size < 2 or d.size != a.size:
            raise VolumetricsError("Area-depth table needs at least two matching depth/area points")
        if np.any(np.diff(d) <= 0):
            raise VolumetricsError("Area-depth table depths must be strictly increasing")
        if np.any(a < 0) or np.any(np.diff(a) < 0):
            raise VolumetricsError("Area-depth table areas must be non-negative and non-decreasing with depth")
        self.depth, self.area = d.tolist(), a.tolist()
        grid = np.linspace(d[0], d[-1], 4001)
        ag = np.interp(grid, d, a)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (ag[1:] + ag[:-1]) * np.diff(grid))])
        self._grid, self._cum = grid, cum

    @property
    def crest(self) -> float:
        return self.depth[0]

    @property
    def base(self) -> float:
        return self.depth[-1]

    def cumulative_volume(self, z) -> np.ndarray:
        """Rock volume above depth ``z`` enclosed by the surface [acre-ft]."""
        z = np.asarray(z, float)
        inside = np.interp(np.clip(z, self.crest, self.base), self._grid, self._cum)
        below = np.clip(z - self.base, 0.0, None) * self.area[-1]  # constant-area extrapolation
        return inside + below

    def to_dict(self) -> dict[str, Any]:
        return {"depth": list(self.depth), "area": list(self.area)}


def _grv_between(top: AreaDepthTable, contact, thickness=None, base: AreaDepthTable | None = None):
    if base is not None:
        return np.clip(top.cumulative_volume(contact) - base.cumulative_volume(contact), 0.0, None)
    return np.clip(top.cumulative_volume(contact) - top.cumulative_volume(np.asarray(contact) - thickness), 0.0, None)


@dataclass
class SegmentConfig:
    grv_method: str = "area_thickness"
    contact_mode: str = "contact_depth"
    limit_to_spill: bool = True
    top_table: AreaDepthTable | None = None
    base_table: AreaDepthTable | None = None
    phase_probabilities: dict[str, float] = field(default_factory=lambda: {"oil": 1.0, "gas": 0.0, "oil_gascap": 0.0})
    gas_fvf: str = "bg"  # "bg" (formation volume factor) or "eg" (expansion factor = 1 / Bg)

    def validate(self) -> None:
        if self.grv_method not in GRV_METHODS:
            raise VolumetricsError(f"Unknown GRV method '{self.grv_method}'")
        if self.grv_method == "area_depth":
            if self.top_table is None:
                raise VolumetricsError("Area-depth GRV method requires a top-structure area-depth table")
            if self.contact_mode not in CONTACT_MODES:
                raise VolumetricsError(f"Unknown contact mode '{self.contact_mode}'")
        if self.gas_fvf not in ("bg", "eg"):
            raise VolumetricsError("Gas volume factor input must be 'bg' or 'eg'")
        p = self.phase_probabilities
        if set(p) - set(PHASES):
            raise VolumetricsError(f"Unknown phase(s): {set(p) - set(PHASES)}")
        if any(v < 0 for v in p.values()) or abs(sum(p.values()) - 1.0) > 1e-6:
            raise VolumetricsError(f"Phase probabilities must be non-negative and sum to 1 (sum={sum(p.values()):.3f})")

    @property
    def has_oil(self) -> bool:
        p = self.phase_probabilities
        return p.get("oil", 0) + p.get("oil_gascap", 0) > 0

    @property
    def has_gas(self) -> bool:
        p = self.phase_probabilities
        return p.get("gas", 0) + p.get("oil_gascap", 0) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grv_method": self.grv_method,
            "contact_mode": self.contact_mode,
            "limit_to_spill": self.limit_to_spill,
            "top_table": None if self.top_table is None else self.top_table.to_dict(),
            "base_table": None if self.base_table is None else self.base_table.to_dict(),
            "phase_probabilities": dict(self.phase_probabilities),
            "gas_fvf": self.gas_fvf,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SegmentConfig":
        d = dict(d)
        for k in ("top_table", "base_table"):
            if d.get(k):
                d[k] = AreaDepthTable(**d[k])
        return cls(**d)


def required_variables(cfg: SegmentConfig) -> tuple[list[str], list[str]]:
    """(required, optional) variable names for a configuration."""
    req: list[str] = []
    opt: list[str] = []
    if cfg.grv_method == "area_thickness":
        req += ["area", "gross_thickness", "geometric_factor"]
    elif cfg.grv_method == "direct":
        req += ["grv"]
    else:
        if cfg.base_table is None:
            req += ["gross_thickness"]
        req += {"contact_depth": ["contact_depth"], "column_height": ["column_height"],
                "fill_fraction": ["fill_fraction"]}[cfg.contact_mode]
        opt += ["spill_depth"]
    req += ["net_to_gross", "porosity", "hc_saturation"]
    if cfg.phase_probabilities.get("oil_gascap", 0) > 0:
        req += ["gas_cap_fraction"]
    if cfg.has_oil:
        req += ["bo", "rf_oil"]
        opt += ["gor", "rf_solution_gas"]
    if cfg.has_gas:
        req += [cfg.gas_fvf, "rf_gas"]
        opt += ["cgr", "rf_condensate", "gas_saturation"]
    return req, opt


def compute(cfg: SegmentConfig, v: dict[str, np.ndarray], phase: np.ndarray,
            scf_per_boe: float = BOE_CONVENTIONS[DEFAULT_BOE_CONVENTION]) -> dict[str, np.ndarray]:
    """Deterministic, vectorised volumetric calculation for arrays of trial values."""
    phase = np.asarray(phase, int)
    n = phase.size
    zeros = np.zeros(n)

    def g(name, default=None):
        if name in v:
            return np.asarray(v[name], float)
        if name in FRACTION_DEFAULTS and FRACTION_DEFAULTS[name] in v:
            return np.asarray(v[FRACTION_DEFAULTS[name]], float)
        if default is not None:
            return np.full(n, default, float)
        raise VolumetricsError(f"Missing input variable '{name}'")

    out: dict[str, np.ndarray] = {}
    is_oil, is_gas, is_both = phase == 0, phase == 1, phase == 2
    gcf = g("gas_cap_fraction", 0.0)

    if cfg.grv_method == "area_thickness":
        grv = g("area") * g("gross_thickness") * g("geometric_factor")
        grv_cap = gcf * grv
    elif cfg.grv_method == "direct":
        grv = g("grv")
        grv_cap = gcf * grv
    else:
        top = cfg.top_table
        spill = g("spill_depth", top.base)
        if cfg.contact_mode == "contact_depth":
            contact = g("contact_depth")
        elif cfg.contact_mode == "column_height":
            contact = top.crest + g("column_height")
        else:
            contact = top.crest + g("fill_fraction") * (spill - top.crest)
        contact = np.maximum(contact, top.crest)
        if cfg.limit_to_spill:
            out["_contact_clipped"] = (contact > spill).astype(float)
            contact = np.minimum(contact, spill)
        thick = None if cfg.base_table is not None else g("gross_thickness")
        grv = _grv_between(top, contact, thick, cfg.base_table)
        goc = top.crest + gcf * (contact - top.crest)
        grv_cap = np.minimum(_grv_between(top, goc, thick, cfg.base_table), grv)
        out["contact"] = contact

    grv_gas = np.where(is_gas, grv, np.where(is_both, grv_cap, 0.0))
    grv_oil = np.where(is_oil, grv, np.where(is_both, grv - grv_cap, 0.0))
    pore = g("net_to_gross") * g("porosity")

    if cfg.has_oil:
        hcpv_oil = BBL_PER_ACRE_FT * grv_oil * pore * g("hc_saturation")
        stoiip = hcpv_oil / g("bo")
        sol = stoiip * g("gor", 0.0)
        rec_oil = stoiip * g("rf_oil")
        rec_sol = sol * g("rf_solution_gas")
    else:
        hcpv_oil = stoiip = sol = rec_oil = rec_sol = zeros
    if cfg.has_gas:
        hcpv_gas = FT3_PER_ACRE_FT * grv_gas * pore * g("gas_saturation")
        giip = hcpv_gas / (g("bg") if cfg.gas_fvf == "bg" else 1.0 / g("eg"))
        cond = giip / 1.0e6 * g("cgr", 0.0)
        rec_gas = giip * g("rf_gas")
        rec_cond = cond * g("rf_condensate")
    else:
        hcpv_gas = giip = cond = rec_gas = rec_cond = zeros

    out.update({
        "grv_total": grv_oil + grv_gas, "grv_oil": grv_oil, "grv_gas": grv_gas,
        "hcpv_oil": hcpv_oil, "hcpv_gas": hcpv_gas,
        "stoiip": stoiip, "giip_free": giip, "giip_solution": sol, "cond_ip": cond,
        "inplace_liquids": stoiip + cond, "inplace_gas": giip + sol,
        "rec_oil": rec_oil, "rec_condensate": rec_cond, "rec_free_gas": rec_gas,
        "rec_solution_gas": rec_sol, "rec_liquids": rec_oil + rec_cond, "rec_gas": rec_gas + rec_sol,
    })
    out["inplace_oe"] = out["inplace_liquids"] + out["inplace_gas"] / scf_per_boe
    out["rec_oe"] = out["rec_liquids"] + out["rec_gas"] / scf_per_boe
    return out


NOISE_SUFFIX = "@noise"


@dataclass
class Expression:
    """Dependent parameter: target = formula(other inputs) with optional uncertainty.

    ``units`` fixes the unit of every name in the formula (target included) so the
    formula keeps its meaning whatever display units are chosen later. Names not
    listed use internal oilfield units.
    """

    expr: str
    units: dict[str, str] = field(default_factory=dict)
    noise: Distribution | None = None
    noise_mode: str = "multiply"  # "multiply" | "add" (additive noise is in the target's formula unit)

    def names(self) -> set[str]:
        return ex.parse(self.expr)[1]

    def to_dict(self) -> dict[str, Any]:
        return {"expr": self.expr, "units": dict(self.units),
                "noise": None if self.noise is None else self.noise.to_dict(), "noise_mode": self.noise_mode}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Expression":
        return cls(expr=d["expr"], units=dict(d.get("units") or {}),
                   noise=from_dict(d["noise"]) if d.get("noise") else None,
                   noise_mode=d.get("noise_mode", "multiply"))


def input_label(name: str) -> str:
    if name.endswith(NOISE_SUFFIX):
        base = name[: -len(NOISE_SUFFIX)]
        return f"{VARIABLES.get(base, (base,))[0]} (formula uncertainty)"
    return VARIABLES.get(name, (name,))[0]


def input_quantity(name: str) -> str:
    return "none" if name.endswith(NOISE_SUFFIX) else VARIABLES[name][1]


def input_bounds(name: str) -> tuple[float, float]:
    return (-np.inf, np.inf) if name.endswith(NOISE_SUFFIX) else VARIABLES[name][2]


@dataclass
class SegmentVolumetrics:
    config: SegmentConfig = field(default_factory=SegmentConfig)
    variables: dict[str, Distribution] = field(default_factory=dict)
    correlations: list[tuple[str, str, float]] = field(default_factory=list)
    expressions: dict[str, Expression] = field(default_factory=dict)

    def _needed(self) -> set[str]:
        req, opt = required_variables(self.config)
        return set(req) | set(opt)

    def active_expressions(self) -> dict[str, Expression]:
        need = self._needed()
        return {k: e for k, e in self.expressions.items() if k in need}

    def active_variables(self) -> dict[str, Distribution]:
        """Sampled (distribution-defined) inputs used by the configuration."""
        need, exprs = self._needed(), self.active_expressions()
        return {k: d for k, d in self.variables.items() if k in need and k not in exprs}

    def sampled_inputs(self) -> dict[str, Distribution]:
        """Distribution-defined inputs plus formula-uncertainty terms."""
        out = dict(self.active_variables())
        for t, e in self.active_expressions().items():
            if e.noise is not None:
                out[t + NOISE_SUFFIX] = e.noise
        return out

    def expression_order(self) -> list[str]:
        """Topological order of active expressions; raises on cycles or unknown names."""
        exprs = self.active_expressions()
        available = set(self.active_variables()) | set(exprs)
        deps: dict[str, set[str]] = {}
        for t, e in exprs.items():
            if t not in VARIABLES:
                raise VolumetricsError(f"Formula target '{t}' is not an input variable")
            try:
                names = e.names()
            except ex.ExpressionError as exc:
                raise VolumetricsError(f"Formula for {VARIABLES[t][0]}: {exc}") from exc
            if t in names:
                raise VolumetricsError(f"Formula for {VARIABLES[t][0]} refers to itself")
            unknown = sorted(names - available)
            if unknown:
                raise VolumetricsError(f"Formula for {VARIABLES[t][0]} uses input(s) not available in this "
                                       f"configuration: {', '.join(unknown)}")
            if e.noise_mode not in ("multiply", "add"):
                raise VolumetricsError(f"Formula for {VARIABLES[t][0]}: unknown uncertainty mode '{e.noise_mode}'")
            for name, unit in e.units.items():
                if name in VARIABLES and not is_valid_unit(VARIABLES[name][1], unit):
                    raise VolumetricsError(f"Formula for {VARIABLES[t][0]}: invalid unit '{unit}' for {name}")
            deps[t] = names & set(exprs)
        order, done, visiting = [], set(), set()

        def visit(t):
            if t in done:
                return
            if t in visiting:
                raise VolumetricsError(f"Circular formula dependency involving {VARIABLES[t][0]}")
            visiting.add(t)
            for d in sorted(deps[t]):
                visit(d)
            visiting.discard(t)
            done.add(t)
            order.append(t)

        for t in sorted(exprs):
            visit(t)
        return order

    def validate(self) -> list[str]:
        """Raise on hard errors; return soft warnings."""
        self.config.validate()
        req, _ = required_variables(self.config)
        missing = [r for r in req if r not in self.variables and r not in self.expressions]
        if missing:
            raise VolumetricsError("Missing required inputs: " + ", ".join(VARIABLES[m][0] for m in missing))
        unknown = [k for k in self.variables if k not in VARIABLES]
        if unknown:
            raise VolumetricsError("Unknown input variable(s): " + ", ".join(unknown))
        self.expression_order()
        return []

    def evaluate_expressions(self, values: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
        """Add formula-defined inputs (internal units) to ``values``; returns the updated dict."""
        exprs = self.active_expressions()
        out = dict(values)
        for t in self.expression_order():
            e = exprs[t]
            tree, names = ex.parse(e.expr)
            env = {}
            for nm in names:
                q = VARIABLES[nm][1]
                env[nm] = from_internal(np.asarray(out[nm], float), q, e.units.get(nm, internal_unit(q)))
            try:
                val = ex.evaluate(tree, env, n)
            except ex.ExpressionError as exc:
                raise VolumetricsError(f"Formula for {VARIABLES[t][0]}: {exc}") from exc
            if e.noise is not None:
                nz = np.asarray(out.get(t + NOISE_SUFFIX, np.full(n, e.noise.p50)), float)
                val = val * nz if e.noise_mode == "multiply" else val + nz
            bad = ~np.isfinite(val)
            if bad.any():
                raise VolumetricsError(f"Formula for {VARIABLES[t][0]} gives invalid values (NaN/inf) in "
                                       f"{int(bad.sum())} of {n} trials, e.g. log or division by zero")
            q = VARIABLES[t][1]
            out[t] = to_internal(val, q, e.units.get(t, internal_unit(q)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "variables": {k: d.to_dict() for k, d in self.variables.items()},
            "correlations": [[a, b, float(r)] for a, b, r in self.correlations],
            "expressions": {k: e.to_dict() for k, e in self.expressions.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SegmentVolumetrics":
        return cls(
            config=SegmentConfig.from_dict(d.get("config", {})),
            variables={k: from_dict(v) for k, v in d.get("variables", {}).items()},
            correlations=[(a, b, float(r)) for a, b, r in d.get("correlations", [])],
            expressions={k: Expression.from_dict(v) for k, v in (d.get("expressions") or {}).items()},
        )


@dataclass
class VolumetricResult:
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    phase: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def n(self) -> int:
        return self.phase.size


@dataclass
class InputDependency:
    """Dependency between sampled inputs, possibly in different segments (indices into the segment list).

    kind "correlation": rank correlation ``rho`` (Iman-Conover).
    kind "link": variable B takes the same percentile as variable A in every trial (B follows A).
    """

    seg_a: int
    var_a: str
    seg_b: int
    var_b: str
    kind: str = "correlation"
    rho: float = 0.0


def sample_phase(probs: dict[str, float], u: np.ndarray) -> np.ndarray:
    p = np.array([probs.get(ph, 0.0) for ph in PHASES], float)
    cum = np.cumsum(p)
    cum[-1] = 1.0
    idx = np.searchsorted(cum, u, side="right")
    return np.minimum(idx, len(PHASES) - 1)


def simulate_many(segs: list[SegmentVolumetrics], n_trials: int, seeds: list[int], sampling: str = "lhs",
                  scf_per_boe: float = BOE_CONVENTIONS[DEFAULT_BOE_CONVENTION],
                  dependencies: list[InputDependency] | None = None, joint_seed: int | None = None,
                  labels: list[str] | None = None) -> list[VolumetricResult]:
    """Joint simulation of several segments with trial-aligned input dependencies."""
    k = len(segs)
    labels = labels or [f"Segment {i + 1}" for i in range(k)]
    warns: list[list[str]] = [list(s.validate()) for s in segs]
    blocks = []
    for i, seg in enumerate(segs):
        sampled = seg.sampled_inputs()
        stoch = [v for v, d in sampled.items() if not d.is_constant]
        rng = np.random.default_rng(seeds[i])
        u = corr.uniform_design(n_trials, len(stoch) + 1, rng, sampling)
        blocks.append({
            "sampled": sampled, "stoch": stoch,
            "cols": {v: sampled[v].ppf(u[:, j]) for j, v in enumerate(stoch)},
            "consts": {v: np.full(n_trials, d.p50) for v, d in sampled.items() if d.is_constant},
            "u_phase": u[:, -1],
        })

    pos = {(i, v): None for i, b in enumerate(blocks) for v in b["stoch"]}
    index = list(pos)
    pairs: list[tuple[tuple[int, str], tuple[int, str], float]] = []
    for i, seg in enumerate(segs):
        dropped = []
        for a, b, r in seg.correlations:
            if r == 0:
                continue
            if (i, a) in pos and (i, b) in pos:
                pairs.append(((i, a), (i, b), float(r)))
            else:
                dropped.append(f"{input_label(a)}–{input_label(b)}")
        if dropped:
            warns[i].append("Ignored correlations involving constant, formula-defined or inactive inputs: "
                            + ", ".join(dropped))
    links = []
    for d in dependencies or []:
        if d.seg_a >= k or d.seg_b >= k:
            raise VolumetricsError("Input dependency refers to a segment that does not exist")
        a, b = (d.seg_a, d.var_a), (d.seg_b, d.var_b)
        desc = f"{labels[d.seg_a]}: {input_label(d.var_a)} → {labels[d.seg_b]}: {input_label(d.var_b)}"
        if a == b:
            raise VolumetricsError(f"Dependency of an input on itself ({desc})")
        if a not in pos or b not in pos:
            warns[d.seg_b].append(f"Ignored dependency {desc}: both inputs must be uncertain, distribution-defined "
                                  "and used by their segment")
            continue
        if d.kind == "correlation":
            if not -1 <= d.rho <= 1:
                raise VolumetricsError(f"Correlation {desc} = {d.rho} is outside [-1, 1]")
            if d.rho != 0:
                pairs.append((a, b, float(d.rho)))
        elif d.kind == "link":
            links.append((a, b))
        else:
            raise VolumetricsError(f"Unknown dependency type '{d.kind}'")

    if pairs:
        idx = {key: j for j, key in enumerate(index)}
        target = np.eye(len(index))
        for a, b, r in pairs:
            target[idx[a], idx[b]] = target[idx[b], idx[a]] = r
        fixed, modified = corr.nearest_correlation(target)
        if modified:
            warns[0].append("The requested correlations are mutually inconsistent (pairs you did not specify are "
                            "treated as uncorrelated, which can contradict chains such as A–B and B–C). They were "
                            "adjusted to the nearest valid set, so achieved values may differ slightly; specifying "
                            "the implied pairs avoids this")
        x = np.column_stack([blocks[i]["cols"][v] for i, v in index])
        x = corr.iman_conover(x, fixed, np.random.default_rng(seeds[0] + 7919 if joint_seed is None else joint_seed))
        for j, (i, v) in enumerate(index):
            blocks[i]["cols"][v] = x[:, j]

    if links:
        leader_of: dict[tuple[int, str], tuple[int, str]] = {}
        for a, b in links:
            if b in leader_of and leader_of[b] != a:
                raise VolumetricsError(f"{labels[b[0]]}: {input_label(b[1])} is linked to more than one input")
            leader_of[b] = a
        order, state = [], {}

        def visit(node):
            if state.get(node) == 2:
                return
            if state.get(node) == 1:
                raise VolumetricsError("Circular chain of linked inputs")
            state[node] = 1
            if node in leader_of:
                visit(leader_of[node])
                order.append(node)
            state[node] = 2

        for b in leader_of:
            visit(b)
        corr_nodes = {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
        for b in order:
            a = leader_of[b]
            lead = blocks[a[0]]["cols"][a[1]]
            ranks = np.argsort(np.argsort(lead, kind="stable"), kind="stable")
            blocks[b[0]]["cols"][b[1]] = blocks[b[0]]["sampled"][b[1]].ppf((ranks + 0.5) / n_trials)
            if b in corr_nodes:
                warns[b[0]].append(f"{input_label(b[1])} is linked to another input; its own correlations are overridden")

    results = []
    for i, (seg, blk) in enumerate(zip(segs, blocks)):
        inputs = {**blk["consts"], **blk["cols"]}
        inputs = seg.evaluate_expressions(inputs, n_trials)
        clipped: dict[str, int] = {}
        for v in list(inputs):
            lo, hi = input_bounds(v)
            bad = int(np.count_nonzero((inputs[v] < lo) | (inputs[v] > hi)))
            if bad:
                clipped[v] = bad
                inputs[v] = np.clip(inputs[v], lo, hi)
        for v, cnt in clipped.items():
            how = "formula result" if v in seg.active_expressions() else "samples"
            warns[i].append(f"{input_label(v)}: {cnt} of {n_trials} {how} ({100 * cnt / n_trials:.1f} %) fell outside "
                            "physical bounds and were clipped")
        phase = sample_phase(seg.config.phase_probabilities, blk["u_phase"])
        outputs = compute(seg.config, inputs, phase, scf_per_boe)
        diag: dict[str, Any] = {"warnings": warns[i], "clipped": clipped}
        if "_contact_clipped" in outputs:
            frac = float(outputs.pop("_contact_clipped").mean())
            diag["contact_clipped_fraction"] = frac
            if frac > 0:
                warns[i].append(f"Contact deeper than spill point in {100 * frac:.1f} % of trials — limited to spill depth")
        results.append(VolumetricResult(inputs=inputs, outputs=outputs, phase=phase, diagnostics=diag))
    return results


def simulate(seg: SegmentVolumetrics, n_trials: int, seed: int, sampling: str = "lhs",
             scf_per_boe: float = BOE_CONVENTIONS[DEFAULT_BOE_CONVENTION]) -> VolumetricResult:
    return simulate_many([seg], n_trials, [seed], sampling, scf_per_boe)[0]


def deterministic(seg: SegmentVolumetrics, values: dict[str, float], phase: str,
                  scf_per_boe: float = BOE_CONVENTIONS[DEFAULT_BOE_CONVENTION]) -> dict[str, float]:
    arr = {k: np.array([float(val)]) for k, val in values.items()}
    arr = seg.evaluate_expressions(arr, 1)
    for v in list(arr):
        lo, hi = input_bounds(v)
        arr[v] = np.clip(arr[v], lo, hi)
    res = compute(seg.config, arr, np.array([PHASES.index(phase)]), scf_per_boe)
    return {k: float(val[0]) for k, val in res.items() if not k.startswith("_")}


def constant_inputs(seg: SegmentVolumetrics, which: str = "p50") -> dict[str, float]:
    return {k: (d.mean if which == "mean" else getattr(d, which)) for k, d in seg.sampled_inputs().items()}


__all__ = [
    "AreaDepthTable", "SegmentConfig", "SegmentVolumetrics", "VolumetricResult", "Expression", "InputDependency",
    "simulate", "simulate_many", "compute", "deterministic", "constant_inputs", "required_variables", "input_label",
    "input_quantity", "VARIABLES", "OUTPUTS", "PHASES", "PHASE_LABELS", "GRV_METHODS", "CONTACT_MODES", "KEY_OUTPUT",
    "NOISE_SUFFIX", "Constant",
]
