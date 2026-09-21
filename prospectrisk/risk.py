"""Geological chance of success (Pg), commercial chance and risked volumes.

Chance factors are organised in geological categories (source, charge,
reservoir, trap, seal) and in three *scopes*:

* **play**     — shared by every segment of every prospect in the play
* **prospect** — shared by all segments of one prospect
* **segment**  — unique to one segment

Pg(segment) = Π play factors × Π prospect factors × Π segment factors.

A factor is given either as a direct probability or through a chance-adequacy
matrix (interpretation × confidence). The default matrix is the symmetric
3 × 3 scheme common in industry practice; it is fully editable so it can be
replaced by a company calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CATEGORIES = ("Source", "Charge", "Reservoir", "Trap", "Seal", "Other")
SCOPES = ("play", "prospect", "segment")
INTERPRETATIONS = ("favourable", "neutral", "unfavourable")
CONFIDENCES = ("high", "medium", "low")

DEFAULT_ADEQUACY_MATRIX: dict[str, dict[str, float]] = {
    "favourable": {"high": 0.9, "medium": 0.8, "low": 0.7},
    "neutral": {"high": 0.5, "medium": 0.5, "low": 0.5},
    "unfavourable": {"high": 0.1, "medium": 0.2, "low": 0.3},
}

# Industry-style verbal risk classes for Pg (editable thresholds, lower bound inclusive)
RISK_CLASSES = [
    (0.5, "Low risk", "#9DBA00"),
    (0.25, "Moderate risk", "#F7D117"),
    (0.125, "High risk", "#FF9200"),
    (0.0, "Very high risk", "#EB0037"),
]


class RiskError(ValueError):
    pass


@dataclass
class ChanceFactor:
    name: str
    category: str = "Other"
    mode: str = "probability"  # "probability" | "adequacy"
    probability: float = 0.5
    interpretation: str = "neutral"
    confidence: str = "medium"
    comment: str = ""

    def value(self, matrix: dict[str, dict[str, float]] | None = None) -> float:
        if self.mode == "adequacy":
            m = matrix or DEFAULT_ADEQUACY_MATRIX
            try:
                p = float(m[self.interpretation][self.confidence])
            except KeyError as exc:
                raise RiskError(f"Chance factor '{self.name}': unknown adequacy class "
                                f"{self.interpretation}/{self.confidence}") from exc
        elif self.mode == "probability":
            p = float(self.probability)
        else:
            raise RiskError(f"Chance factor '{self.name}': unknown mode '{self.mode}'")
        if not 0.0 <= p <= 1.0:
            raise RiskError(f"Chance factor '{self.name}' = {p} is outside [0, 1]")
        return p

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("name", "category", "mode", "probability", "interpretation", "confidence", "comment")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChanceFactor":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def product(factors: list[ChanceFactor], matrix=None) -> float:
    return float(np.prod([f.value(matrix) for f in factors])) if factors else 1.0


@dataclass
class ChanceBreakdown:
    p_play: float
    p_prospect: float
    p_segment: float
    by_category: dict[str, float]
    critical: list[tuple[str, str, float]]  # (scope, factor name, value) sorted ascending

    @property
    def pg(self) -> float:
        return self.p_play * self.p_prospect * self.p_segment

    @property
    def p_conditional(self) -> float:
        """Chance of the prospect given that the play works."""
        return self.p_prospect * self.p_segment

    @property
    def risk_class(self) -> tuple[str, str]:
        for lo, label, colour in RISK_CLASSES:
            if self.pg >= lo:
                return label, colour
        return RISK_CLASSES[-1][1], RISK_CLASSES[-1][2]


def breakdown(play: list[ChanceFactor], prospect: list[ChanceFactor], segment: list[ChanceFactor],
              matrix=None) -> ChanceBreakdown:
    by_cat = {c: 1.0 for c in CATEGORIES}
    crit = []
    for scope, fs in (("play", play), ("prospect", prospect), ("segment", segment)):
        for f in fs:
            v = f.value(matrix)
            cat = f.category if f.category in by_cat else "Other"
            by_cat[cat] *= v
            crit.append((scope, f.name, v))
    crit.sort(key=lambda t: t[2])
    return ChanceBreakdown(product(play, matrix), product(prospect, matrix), product(segment, matrix),
                           {k: v for k, v in by_cat.items()}, crit)


def default_play_factors() -> list[ChanceFactor]:
    return [
        ChanceFactor("Source rock presence & maturity", "Source", probability=0.9),
        ChanceFactor("Regional reservoir presence", "Reservoir", probability=0.9),
        ChanceFactor("Regional top seal", "Seal", probability=0.9),
    ]


def default_prospect_factors() -> list[ChanceFactor]:
    return [
        ChanceFactor("Migration & timing", "Charge", probability=0.7),
        ChanceFactor("Trap geometry", "Trap", probability=0.7),
    ]


def default_segment_factors() -> list[ChanceFactor]:
    return [
        ChanceFactor("Reservoir effectiveness", "Reservoir", probability=0.8),
        ChanceFactor("Trap retention / fault seal", "Seal", probability=0.7),
    ]


# ---- volumes ---------------------------------------------------------------------

def exceedance_stats(x: np.ndarray) -> dict[str, float]:
    """Summary statistics in exploration convention (P90 low)."""
    x = np.asarray(x, float)
    if x.size == 0:
        return {k: float("nan") for k in ("Mean", "Std", "P90", "P50", "P10", "Min", "Max")}
    q = np.quantile(x, [0.10, 0.50, 0.90])
    return {"Mean": float(x.mean()), "Std": float(x.std()), "P90": float(q[0]), "P50": float(q[1]),
            "P10": float(q[2]), "Min": float(x.min()), "Max": float(x.max())}


def swanson_mean(p90: float, p50: float, p10: float) -> float:
    return 0.3 * p90 + 0.4 * p50 + 0.3 * p10


def commercial_chance(pg: float, success_volumes: np.ndarray, mefs: float) -> dict[str, float]:
    """Commercial chance and conditional statistics for a minimum economic field size."""
    v = np.asarray(success_volumes, float)
    p_above = float(np.mean(v >= mefs)) if v.size else 0.0
    above = v[v >= mefs]
    return {
        "mefs": float(mefs),
        "p_above_mefs_given_success": p_above,
        "pc": pg * p_above,
        "mean_commercial_success": float(above.mean()) if above.size else 0.0,
        "risked_commercial_mean": pg * p_above * (float(above.mean()) if above.size else 0.0),
    }


def expectation_curve(success_volumes: np.ndarray, chance: float = 1.0, n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Volumes and exceedance probabilities P(V >= x) scaled by ``chance``.

    ``chance=1`` gives the success-case curve, ``chance=Pg`` the risked curve.
    """
    v = np.sort(np.asarray(success_volumes, float))
    n = v.size
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    exc = 1.0 - np.arange(n) / n
    if n > n_points:
        idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
        v, exc = v[idx], exc[idx]
    return v, chance * exc


def risked_summary(pg: float, success_volumes: np.ndarray) -> dict[str, float]:
    s = exceedance_stats(success_volumes)
    return {"Pg": pg, "Success mean": s["Mean"], "Risked mean": pg * s["Mean"],
            "Success P90": s["P90"], "Success P50": s["P50"], "Success P10": s["P10"]}
