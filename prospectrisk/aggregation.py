"""Stochastic aggregation of segments into prospects and portfolios.

Each item (a reservoir segment) carries

* its success-case Monte Carlo outputs (one row per trial),
* the chance events it depends on, identified by *keys* — play- and
  prospect-scope chance factors produce keys shared by several items, so a
  single draw decides them for all items at once (exact geological dependency),
* a residual (segment-scope) chance, optionally dependent across items through
  a Gaussian copula on a latent dependency matrix
  (0 = independent, 1 = fully dependent / nested outcomes),
* optional volume truncation (minimum and/or maximum on any output),
* an optional *alignment* key: items simulated jointly with input dependencies
  keep their trials aligned (row i of each belongs to the same realisation),
* optionally a rank correlation of success-case volumes between items.

Volume truncation
-----------------
* Minimum, mode ``"chance"`` — geological success defined as "at least the
  minimum volume": a trial below the minimum is a failure. Effective chance =
  Pg × P(V ≥ Vmin) and the success-case distribution is truncated at Vmin.
* Minimum, mode ``"renormalise"`` and any maximum — the distribution is
  truncated and renormalised; chance is unchanged. Rejected trials are replaced
  by resampling accepted trials (jointly for aligned items).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from . import correlation as corr
from .risk import exceedance_stats


class AggregationError(ValueError):
    pass


@dataclass
class VolumeTruncation:
    output: str = "rec_oe"
    minimum: float | None = None     # internal units of the output
    maximum: float | None = None
    min_mode: str = "chance"         # "chance" | "renormalise"

    @property
    def active(self) -> bool:
        return self.minimum is not None or self.maximum is not None

    def validate(self) -> None:
        if self.min_mode not in ("chance", "renormalise"):
            raise AggregationError(f"Unknown minimum-volume mode '{self.min_mode}'")
        if self.minimum is not None and self.minimum < 0:
            raise AggregationError("Minimum volume must not be negative")
        if self.minimum is not None and self.maximum is not None and self.maximum <= self.minimum:
            raise AggregationError("Maximum volume must exceed the minimum volume")

    def to_dict(self) -> dict[str, Any]:
        return {"output": self.output, "minimum": self.minimum, "maximum": self.maximum, "min_mode": self.min_mode}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "VolumeTruncation":
        if not d:
            return cls()
        return cls(output=d.get("output", "rec_oe"),
                   minimum=None if d.get("minimum") is None else float(d["minimum"]),
                   maximum=None if d.get("maximum") is None else float(d["maximum"]),
                   min_mode=d.get("min_mode", "chance"))


@dataclass
class AggItem:
    name: str
    group: str
    outputs: dict[str, np.ndarray]
    shared_events: dict[str, float] = field(default_factory=dict)
    p_residual: float = 1.0
    truncation: VolumeTruncation = field(default_factory=VolumeTruncation)
    align: str | None = None

    @property
    def pg(self) -> float:
        """Geological chance from chance factors only (before any minimum-volume cut-off)."""
        return float(np.prod(list(self.shared_events.values()))) * self.p_residual if self.shared_events \
            else self.p_residual


@dataclass
class AggregationResult:
    items: list[AggItem]
    success: np.ndarray                    # n x k bool (chance events AND volume >= minimum)
    volumes: dict[str, np.ndarray]         # output -> n x k (success-case, reordered, renormalised)
    volume_ok: np.ndarray                  # n x k bool: passes the chance-mode minimum volume
    groups: list[str]
    group_mefs: dict[str, float]
    key_output: str
    diagnostics: dict[str, Any]

    @property
    def n(self) -> int:
        return self.success.shape[0]

    def risked(self, output: str) -> np.ndarray:
        return self.volumes[output] * self.success

    def total(self, output: str) -> np.ndarray:
        return self.risked(output).sum(axis=1)

    def item_index(self, group: str, name: str) -> int:
        for j, it in enumerate(self.items):
            if it.group == group and it.name == name:
                return j
        raise KeyError(f"{group}/{name}")

    def item_success_case(self, j: int, output: str) -> np.ndarray:
        """Success-case distribution of one item after truncation."""
        return self.volumes[output][:, j][self.volume_ok[:, j]]

    def p_volume_ok(self, j: int) -> float:
        return float(self.volume_ok[:, j].mean())

    def group_index(self, group: str) -> np.ndarray:
        return np.array([i for i, it in enumerate(self.items) if it.group == group], int)

    def group_success(self, group: str) -> np.ndarray:
        return self.success[:, self.group_index(group)].any(axis=1)

    def group_total(self, group: str, output: str) -> np.ndarray:
        return self.risked(output)[:, self.group_index(group)].sum(axis=1)

    def group_commercial(self, group: str) -> np.ndarray:
        mefs = self.group_mefs.get(group, 0.0)
        return self.group_success(group) & (self.group_total(group, self.key_output) >= mefs)

    def item_table(self) -> list[dict[str, Any]]:
        rows = []
        for j, it in enumerate(self.items):
            v = self.volumes[self.key_output][:, j]
            p_ok = self.p_volume_ok(j)
            ok = self.volume_ok[:, j]
            rows.append({
                "Group": it.group, "Item": it.name, "Pg (factors)": it.pg,
                "P(V ≥ minimum)": p_ok, "Pg (effective)": it.pg * p_ok,
                "Pg (simulated)": float(self.success[:, j].mean()),
                "Success mean": float(v[ok].mean()) if ok.any() else 0.0,
                "Risked mean": float((v * self.success[:, j]).mean()),
            })
        return rows

    def group_table(self) -> list[dict[str, Any]]:
        rows = []
        for g in self.groups:
            succ = self.group_success(g)
            tot = self.group_total(g, self.key_output)
            s = exceedance_stats(tot[succ]) if succ.any() else exceedance_stats(np.array([]))
            rows.append({
                "Group": g, "Segments": int(self.group_index(g).size),
                "P(any segment success)": float(succ.mean()),
                "P(commercial)": float(self.group_commercial(g).mean()),
                "MEFS": self.group_mefs.get(g, 0.0),
                "Success mean": s["Mean"], "Success P90": s["P90"], "Success P50": s["P50"],
                "Success P10": s["P10"], "Risked mean": float(tot.mean()),
            })
        return rows

    def discovery_counts(self, level: str = "group", commercial: bool = False) -> np.ndarray:
        if level == "item":
            return self.success.sum(axis=1)
        fn = self.group_commercial if commercial else self.group_success
        return np.column_stack([fn(g) for g in self.groups]).sum(axis=1)

    def summary(self) -> dict[str, float]:
        tot = self.total(self.key_output)
        any_s = self.success.any(axis=1)
        cond = exceedance_stats(tot[any_s]) if any_s.any() else exceedance_stats(np.array([]))
        analytic = sum(it.pg * float((self.volumes[self.key_output][:, j] * self.volume_ok[:, j]).mean())
                       for j, it in enumerate(self.items))
        return {
            "P(at least one success)": float(any_s.mean()),
            "Expected discoveries": float(self.discovery_counts("group").mean()),
            "Expected commercial discoveries": float(self.discovery_counts("group", True).mean()),
            "Risked mean": float(tot.mean()),
            "Analytic risked mean": analytic,
            "Mean | ≥1 success": cond["Mean"], "P90 | ≥1 success": cond["P90"],
            "P50 | ≥1 success": cond["P50"], "P10 | ≥1 success": cond["P10"],
        }


def _check_matrix(m: np.ndarray | None, k: int, what: str, lo: float) -> np.ndarray | None:
    if m is None:
        return None
    m = np.asarray(m, float)
    if m.shape != (k, k):
        raise AggregationError(f"{what} matrix must be {k}×{k}")
    if not np.allclose(m, m.T):
        raise AggregationError(f"{what} matrix must be symmetric")
    if np.any(m < lo) or np.any(m > 1):
        raise AggregationError(f"{what} coefficients must lie in [{lo}, 1]")
    if np.allclose(m, np.eye(k)):
        return None
    return m


def aggregate(items: list[AggItem], seed: int, dependency: np.ndarray | None = None,
              volume_correlation: np.ndarray | None = None, key_output: str = "rec_oe",
              group_mefs: dict[str, float] | None = None) -> AggregationResult:
    if not items:
        raise AggregationError("Nothing to aggregate")
    k = len(items)
    n = items[0].outputs[key_output].size
    outputs = list(items[0].outputs.keys())
    for it in items:
        if it.outputs[key_output].size != n:
            raise AggregationError("All items must have the same number of trials")
        if not 0 <= it.p_residual <= 1:
            raise AggregationError(f"{it.name}: residual chance outside [0, 1]")
        it.truncation.validate()
        if it.truncation.active and it.truncation.output not in it.outputs:
            raise AggregationError(f"{it.name}: unknown truncation output '{it.truncation.output}'")
        outputs = [o for o in outputs if o in it.outputs]

    shared: dict[str, float] = {}
    for it in items:
        for key, p in it.shared_events.items():
            if not 0 <= p <= 1:
                raise AggregationError(f"{it.name}: chance '{key}' outside [0, 1]")
            if key in shared and abs(shared[key] - p) > 1e-9:
                raise AggregationError(f"Shared chance '{key}' has inconsistent values ({shared[key]} vs {p})")
            shared[key] = p

    dependency = _check_matrix(dependency, k, "Dependency", 0.0)
    volume_correlation = _check_matrix(volume_correlation, k, "Volume correlation", -1.0)
    rng = np.random.default_rng(seed)
    diag: dict[str, Any] = {"warnings": [], "rejected_fraction": {}}

    # ---- units of trial alignment -------------------------------------------------------
    units: list[list[int]] = []
    unit_of_key: dict[str, int] = {}
    for j, it in enumerate(items):
        if it.align is None:
            units.append([j])
        elif it.align in unit_of_key:
            units[unit_of_key[it.align]].append(j)
        else:
            unit_of_key[it.align] = len(units)
            units.append([j])
    orders: list[np.ndarray] = [np.empty(0, int)] * k
    for u in units:
        perm = rng.permutation(n)
        for j in u:
            orders[j] = perm.copy()

    # ---- renormalising truncation (reject and resample accepted trials) ---------------------
    for u in units:
        valid = np.ones(n, bool)
        involved = False
        for j in u:
            t = items[j].truncation
            if t.maximum is None and not (t.minimum is not None and t.min_mode == "renormalise"):
                continue
            v = items[j].outputs[t.output][orders[j]]
            if t.maximum is not None:
                valid &= v <= t.maximum
                involved = True
            if t.minimum is not None and t.min_mode == "renormalise":
                valid &= v >= t.minimum
                involved = True
        if not involved:
            continue
        n_valid = int(valid.sum())
        names = ", ".join(items[j].name for j in u)
        if n_valid == 0:
            raise AggregationError(f"Volume truncation rejects every trial for {names}")
        rej = 1.0 - n_valid / n
        for j in u:
            diag["rejected_fraction"][items[j].name] = rej
        if rej > 0.5:
            diag["warnings"].append(f"Volume truncation rejects {100 * rej:.0f} % of trials for {names}; "
                                    "the truncated distribution rests on few distinct trials")
        if rej > 0:
            bad = np.flatnonzero(~valid)
            pick = rng.choice(np.flatnonzero(valid), size=bad.size, replace=True)
            for j in u:
                orders[j][bad] = orders[j][pick]

    # ---- success-volume rank correlation between alignment units --------------------------------
    if volume_correlation is not None:
        if any(volume_correlation[i, j] != 0 for u in units for i in u for j in u if i != j):
            diag["warnings"].append("Volume correlation between jointly simulated segments is ignored — their "
                                    "dependency comes from the input dependencies")
        m = len(units)
        if m >= 2:
            target = np.eye(m)
            inconsistent = False
            for a in range(m):
                for b in range(a + 1, m):
                    vals = {float(volume_correlation[i, j]) for i in units[a] for j in units[b]}
                    target[a, b] = target[b, a] = next(iter(vals))
                    inconsistent |= len(vals) > 1
            if inconsistent:
                diag["warnings"].append("Volume correlations differ between items that are simulated jointly; "
                                        "one value per pair of groups is used")
            if not np.allclose(target, np.eye(m)):
                keys = np.column_stack([sum(items[j].outputs[key_output][orders[j]] for j in u) for u in units])
                fixed, modified = corr.nearest_correlation(target)
                if modified:
                    diag["warnings"].append("Volume correlation matrix adjusted to nearest positive-definite matrix")
                sub = corr.row_orders_for_rank_correlation(keys, fixed, rng)
                for a, u in enumerate(units):
                    for j in u:
                        orders[j] = orders[j][sub[a]]
    volumes = {o: np.column_stack([items[j].outputs[o][orders[j]] for j in range(k)]) for o in outputs}

    # ---- chance-mode minimum volume ---------------------------------------------------------------
    volume_ok = np.ones((n, k), bool)
    for j, it in enumerate(items):
        t = it.truncation
        if t.minimum is not None and t.min_mode == "chance":
            volume_ok[:, j] = it.outputs[t.output][orders[j]] >= t.minimum
            if not volume_ok[:, j].any():
                diag["warnings"].append(f"{it.name}: no trial reaches the minimum volume, so its effective chance is 0")

    # ---- chance events ------------------------------------------------------------------------------
    events = {key: rng.random(n) < p for key, p in shared.items()}
    if dependency is not None:
        fixed, modified = corr.nearest_correlation(dependency)
        if modified:
            diag["warnings"].append("Dependency matrix adjusted to nearest positive-definite matrix")
        z = rng.standard_normal((n, k)) @ np.linalg.cholesky(fixed).T
        u_ev = stats.norm.cdf(z)
    else:
        u_ev = rng.random((n, k))
    success = u_ev < np.array([it.p_residual for it in items])[None, :]
    for j, it in enumerate(items):
        for key in it.shared_events:
            success[:, j] &= events[key]
    success &= volume_ok

    groups = list(dict.fromkeys(it.group for it in items))
    return AggregationResult(items=items, success=success, volumes=volumes, volume_ok=volume_ok, groups=groups,
                             group_mefs=dict(group_mefs or {}), key_output=key_output, diagnostics=diag)
