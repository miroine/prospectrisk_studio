"""Project data model and simulation orchestration.

Project
 ├─ settings (trials, seed, sampling, unit system, gas-equivalence convention)
 ├─ chance-adequacy matrix
 ├─ plays[]      — play-scope chance factors
 └─ prospects[]  — prospect-scope chance factors, MEFS, segment dependencies
      └─ segments[] — volumetrics + segment-scope chance factors
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import aggregation as agg
from . import risk
from .units import BOE_CONVENTIONS, DEFAULT_BOE_CONVENTION, UNIT_SYSTEMS, is_valid_unit, resolve_unit
from .volumetrics import (KEY_OUTPUT, OUTPUTS, VARIABLES, InputDependency, SegmentVolumetrics, VolumetricResult,
                          simulate_many)

SCHEMA_VERSION = 1


class ProjectError(ValueError):
    pass


@dataclass
class Settings:
    n_trials: int = 10000
    seed: int = 12345
    sampling: str = "lhs"
    unit_system: str = "Metric (NCS)"
    boe_convention: str = DEFAULT_BOE_CONVENTION
    unit_overrides: dict[str, str] = field(default_factory=dict)   # quantity -> unit
    input_units: dict[str, str] = field(default_factory=dict)      # input variable -> unit

    def unit(self, quantity: str) -> str:
        """Display unit for a quantity (preset system, overridden per quantity)."""
        return resolve_unit(self.unit_system, quantity, self.unit_overrides)

    def input_unit(self, variable: str) -> str:
        """Display unit for one input variable (per-input override, else its quantity's unit)."""
        q = VARIABLES[variable][1] if variable in VARIABLES else "none"
        u = self.input_units.get(variable)
        return u if u and is_valid_unit(q, u) else self.unit(q)

    @property
    def unit_token(self) -> str:
        return f"{self.unit_system}|{sorted(self.unit_overrides.items())}|{sorted(self.input_units.items())}"

    @property
    def scf_per_boe(self) -> float:
        return BOE_CONVENTIONS[self.boe_convention]

    def validate(self) -> None:
        if not 100 <= int(self.n_trials) <= 1_000_000:
            raise ProjectError("Number of trials must be between 100 and 1,000,000")
        if self.sampling not in ("lhs", "mc"):
            raise ProjectError("Sampling must be 'lhs' or 'mc'")
        if self.boe_convention not in BOE_CONVENTIONS:
            raise ProjectError(f"Unknown oil-equivalent convention '{self.boe_convention}'")
        if self.unit_system not in UNIT_SYSTEMS:
            raise ProjectError(f"Unknown unit system '{self.unit_system}'")


@dataclass
class Segment:
    name: str
    volumetrics: SegmentVolumetrics = field(default_factory=SegmentVolumetrics)
    factors: list[risk.ChanceFactor] = field(default_factory=risk.default_segment_factors)
    truncation: agg.VolumeTruncation = field(default_factory=agg.VolumeTruncation)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "notes": self.notes, "volumetrics": self.volumetrics.to_dict(),
                "factors": [f.to_dict() for f in self.factors], "truncation": self.truncation.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return cls(name=d["name"], notes=d.get("notes", ""),
                   volumetrics=SegmentVolumetrics.from_dict(d.get("volumetrics", {})),
                   factors=[risk.ChanceFactor.from_dict(f) for f in d.get("factors", [])],
                   truncation=agg.VolumeTruncation.from_dict(d.get("truncation")))


@dataclass
class Prospect:
    name: str
    play: str
    segments: list[Segment] = field(default_factory=list)
    factors: list[risk.ChanceFactor] = field(default_factory=risk.default_prospect_factors)
    mefs: float = 0.0                     # boe, applied to the prospect total recoverable o.e.
    status: str = "Prospect"
    segment_dependency: list[list[float]] | None = None
    segment_volume_correlation: list[list[float]] | None = None
    input_dependencies: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def segment(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise ProjectError(f"Prospect '{self.name}' has no segment '{name}'")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "play": self.play, "status": self.status, "mefs": self.mefs,
                "notes": self.notes, "factors": [f.to_dict() for f in self.factors],
                "segment_dependency": self.segment_dependency,
                "segment_volume_correlation": self.segment_volume_correlation,
                "input_dependencies": [dict(d) for d in self.input_dependencies],
                "segments": [s.to_dict() for s in self.segments]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prospect":
        return cls(name=d["name"], play=d["play"], status=d.get("status", "Prospect"),
                   mefs=float(d.get("mefs", 0.0)), notes=d.get("notes", ""),
                   factors=[risk.ChanceFactor.from_dict(f) for f in d.get("factors", [])],
                   segment_dependency=d.get("segment_dependency"),
                   segment_volume_correlation=d.get("segment_volume_correlation"),
                   input_dependencies=[dict(x) for x in d.get("input_dependencies") or []],
                   segments=[Segment.from_dict(s) for s in d.get("segments", [])])


@dataclass
class Play:
    name: str
    factors: list[risk.ChanceFactor] = field(default_factory=risk.default_play_factors)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "notes": self.notes, "factors": [f.to_dict() for f in self.factors]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Play":
        return cls(name=d["name"], notes=d.get("notes", ""),
                   factors=[risk.ChanceFactor.from_dict(f) for f in d.get("factors", [])])


@dataclass
class Project:
    name: str = "New project"
    settings: Settings = field(default_factory=Settings)
    plays: list[Play] = field(default_factory=list)
    prospects: list[Prospect] = field(default_factory=list)
    adequacy_matrix: dict[str, dict[str, float]] = field(
        default_factory=lambda: copy.deepcopy(risk.DEFAULT_ADEQUACY_MATRIX))
    portfolio_dependency: list[list[Any]] = field(default_factory=list)          # [prospect_a, prospect_b, rho]
    portfolio_volume_correlation: list[list[Any]] = field(default_factory=list)  # [prospect_a, prospect_b, rho]
    notes: str = ""

    # ---- lookup --------------------------------------------------------------------
    def play(self, name: str) -> Play:
        for p in self.plays:
            if p.name == name:
                return p
        raise ProjectError(f"Unknown play '{name}'")

    def prospect(self, name: str) -> Prospect:
        for p in self.prospects:
            if p.name == name:
                return p
        raise ProjectError(f"Unknown prospect '{name}'")

    def validate_names(self) -> None:
        for what, names in (("play", [p.name for p in self.plays]), ("prospect", [p.name for p in self.prospects])):
            dup = {n for n in names if names.count(n) > 1}
            if dup:
                raise ProjectError(f"Duplicate {what} names: {', '.join(sorted(dup))}")
            if any(not str(n).strip() for n in names):
                raise ProjectError(f"Every {what} needs a name")
        for pr in self.prospects:
            self.play(pr.play)
            names = [s.name for s in pr.segments]
            dup = {n for n in names if names.count(n) > 1}
            if dup:
                raise ProjectError(f"Prospect '{pr.name}' has duplicate segment names: {', '.join(sorted(dup))}")

    # ---- serialisation ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        s = self.settings
        return {
            "schema_version": SCHEMA_VERSION, "name": self.name, "notes": self.notes,
            "settings": {"n_trials": s.n_trials, "seed": s.seed, "sampling": s.sampling,
                         "unit_system": s.unit_system, "boe_convention": s.boe_convention,
                         "unit_overrides": dict(s.unit_overrides), "input_units": dict(s.input_units)},
            "adequacy_matrix": self.adequacy_matrix,
            "plays": [p.to_dict() for p in self.plays],
            "prospects": [p.to_dict() for p in self.prospects],
            "portfolio_dependency": self.portfolio_dependency,
            "portfolio_volume_correlation": self.portfolio_volume_correlation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Project":
        ver = d.get("schema_version", 1)
        if ver > SCHEMA_VERSION:
            raise ProjectError(f"Project file schema v{ver} is newer than this application (v{SCHEMA_VERSION})")
        proj = cls(
            name=d.get("name", "Project"), notes=d.get("notes", ""),
            settings=Settings(**{k: v for k, v in (d.get("settings") or {}).items()
                                 if k in Settings.__dataclass_fields__}),
            adequacy_matrix=d.get("adequacy_matrix") or copy.deepcopy(risk.DEFAULT_ADEQUACY_MATRIX),
            plays=[Play.from_dict(p) for p in d.get("plays", [])],
            prospects=[Prospect.from_dict(p) for p in d.get("prospects", [])],
            portfolio_dependency=d.get("portfolio_dependency", []),
            portfolio_volume_correlation=d.get("portfolio_volume_correlation", []),
        )
        proj.validate_names()
        return proj


# ---- runs ------------------------------------------------------------------------------

def segment_seed(base_seed: int, prospect: str, segment: str) -> int:
    """Deterministic, process-independent seed per segment (md5, not hash())."""
    h = hashlib.md5(f"{int(base_seed)}|{prospect}|{segment}".encode()).hexdigest()
    return int(h[:15], 16)


@dataclass
class SegmentRun:
    prospect: str
    segment: str
    result: VolumetricResult
    chance: risk.ChanceBreakdown

    @property
    def label(self) -> str:
        return f"{self.prospect} / {self.segment}"


@dataclass
class ProspectRun:
    prospect: str
    segments: dict[str, SegmentRun]
    aggregation: agg.AggregationResult

    @property
    def pg(self) -> float:
        return float(self.aggregation.group_success(self.prospect).mean())

    def segment_index(self, segment: str) -> int:
        return self.aggregation.item_index(self.prospect, segment)

    def segment_success_case(self, segment: str, output: str = KEY_OUTPUT) -> np.ndarray:
        """Success-case volumes of a segment after volume truncation."""
        return self.aggregation.item_success_case(self.segment_index(segment), output)

    def segment_effective_pg(self, segment: str) -> float:
        j = self.segment_index(segment)
        return self.segments[segment].chance.pg * self.aggregation.p_volume_ok(j)

    def success_total(self, output: str = KEY_OUTPUT) -> np.ndarray:
        a = self.aggregation
        succ = a.group_success(self.prospect)
        return a.group_total(self.prospect, output)[succ]


def _dependencies(pr: Prospect) -> list[InputDependency]:
    idx = {seg.name: i for i, seg in enumerate(pr.segments)}
    out = []
    for d in pr.input_dependencies:
        for side in ("seg_a", "seg_b"):
            if d.get(side) not in idx:
                raise ProjectError(f"Prospect '{pr.name}': input dependency refers to unknown segment '{d.get(side)}'")
        for side in ("var_a", "var_b"):
            if d.get(side) not in VARIABLES:
                raise ProjectError(f"Prospect '{pr.name}': input dependency refers to unknown input '{d.get(side)}'")
        out.append(InputDependency(idx[d["seg_a"]], d["var_a"], idx[d["seg_b"]], d["var_b"],
                                   d.get("kind", "correlation"), float(d.get("rho", 0.0))))
    return out


def simulate_prospect_segments(project: Project, prospect: Prospect) -> dict[str, SegmentRun]:
    s = project.settings
    s.validate()
    play = project.play(prospect.play)
    for seg in prospect.segments:
        seg.truncation.validate()
        if seg.truncation.active and seg.truncation.output not in OUTPUTS:
            raise ProjectError(f"{seg.name}: unknown truncation output '{seg.truncation.output}'")
    results = simulate_many(
        [seg.volumetrics for seg in prospect.segments], int(s.n_trials),
        [segment_seed(s.seed, prospect.name, seg.name) for seg in prospect.segments], s.sampling, s.scf_per_boe,
        dependencies=_dependencies(prospect), joint_seed=segment_seed(s.seed, prospect.name, "__inputs__"),
        labels=[seg.name for seg in prospect.segments])
    return {seg.name: SegmentRun(prospect.name, seg.name, res,
                                 risk.breakdown(play.factors, prospect.factors, seg.factors, project.adequacy_matrix))
            for seg, res in zip(prospect.segments, results)}


def run_segment(project: Project, prospect: Prospect, segment: Segment) -> SegmentRun:
    """Stand-alone simulation of one segment (ignores cross-segment input dependencies)."""
    tmp = Prospect(prospect.name, prospect.play, [segment], prospect.factors,
                   input_dependencies=[d for d in prospect.input_dependencies
                                       if d.get("seg_a") == segment.name and d.get("seg_b") == segment.name])
    return simulate_prospect_segments(project, tmp)[segment.name]


def _items_for(project: Project, prospect: Prospect, runs: dict[str, SegmentRun]) -> list[agg.AggItem]:
    items = []
    play = project.play(prospect.play)
    m = project.adequacy_matrix
    cross = any(d.get("seg_a") != d.get("seg_b") for d in prospect.input_dependencies)
    for seg in prospect.segments:
        run = runs[seg.name]
        shared = {f"play::{play.name}::{i}::{f.name}": f.value(m) for i, f in enumerate(play.factors)}
        shared.update({f"prospect::{prospect.name}::{i}::{f.name}": f.value(m) for i, f in enumerate(prospect.factors)})
        items.append(agg.AggItem(seg.name, prospect.name, run.result.outputs, shared, risk.product(seg.factors, m),
                                 truncation=seg.truncation, align=f"prospect::{prospect.name}" if cross else None))
    return items


def _matrix_or_none(m, k):
    if m is None:
        return None
    arr = np.asarray(m, float)
    return arr if arr.shape == (k, k) else None


def run_prospect(project: Project, name: str, segment_runs: dict[str, SegmentRun] | None = None) -> ProspectRun:
    pr = project.prospect(name)
    if not pr.segments:
        raise ProjectError(f"Prospect '{name}' has no segments")
    runs = segment_runs or simulate_prospect_segments(project, pr)
    items = _items_for(project, pr, runs)
    k = len(items)
    res = agg.aggregate(items, segment_seed(project.settings.seed, pr.name, "__aggregate__"),
                        dependency=_matrix_or_none(pr.segment_dependency, k),
                        volume_correlation=_matrix_or_none(pr.segment_volume_correlation, k),
                        group_mefs={pr.name: pr.mefs})
    return ProspectRun(pr.name, runs, res)


def run_portfolio(project: Project, names: list[str],
                  prospect_runs: dict[str, ProspectRun] | None = None) -> agg.AggregationResult:
    if not names:
        raise ProjectError("Select at least one prospect for the portfolio")
    prospect_runs = prospect_runs or {}
    items: list[agg.AggItem] = []
    owners: list[tuple[str, int]] = []
    for nm in names:
        pr = project.prospect(nm)
        runs = prospect_runs[nm].segments if nm in prospect_runs else simulate_prospect_segments(project, pr)
        its = _items_for(project, pr, runs)
        for i, it in enumerate(its):
            owners.append((nm, i))
        items.extend(its)

    k = len(items)
    dep = np.eye(k)
    vol = np.eye(k)
    pd_map = {frozenset((a, b)): float(r) for a, b, r in project.portfolio_dependency if a != b}
    pv_map = {frozenset((a, b)): float(r) for a, b, r in project.portfolio_volume_correlation if a != b}
    for i in range(k):
        for j in range(i + 1, k):
            (pa, ia), (pb, ib) = owners[i], owners[j]
            if pa == pb:
                pr = project.prospect(pa)
                n_seg = len(pr.segments)
                d = _matrix_or_none(pr.segment_dependency, n_seg)
                v = _matrix_or_none(pr.segment_volume_correlation, n_seg)
                dep[i, j] = dep[j, i] = 0.0 if d is None else d[ia, ib]
                vol[i, j] = vol[j, i] = 0.0 if v is None else v[ia, ib]
            else:
                dep[i, j] = dep[j, i] = pd_map.get(frozenset((pa, pb)), 0.0)
                vol[i, j] = vol[j, i] = pv_map.get(frozenset((pa, pb)), 0.0)

    mefs = {nm: project.prospect(nm).mefs for nm in names}
    return agg.aggregate(items, segment_seed(project.settings.seed, "__portfolio__", "|".join(names)),
                         dependency=dep, volume_correlation=vol, group_mefs=mefs)
