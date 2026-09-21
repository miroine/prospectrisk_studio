"""Input probability distributions.

Percentile convention: exploration / reserves convention throughout.
**P90 is the low case** (90 % probability of exceeding the value) and **P10 the
high case**. Internally everything is driven by the quantile function
``ppf(u)`` with ``u`` the *non*-exceedance probability, so P90 = ppf(0.10).

Every distribution samples through ``ppf`` which makes Latin Hypercube
sampling and Iman-Conover rank correlation exact with respect to the marginals.
Optional truncation (``lower`` / ``upper``) re-scales the quantile range so the
truncated distribution is sampled exactly, not by rejection.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

Z90 = float(stats.norm.ppf(0.90))  # 1.2815515655446004
_U_EPS = 1e-12
_MEAN_GRID = (np.arange(40000) + 0.5) / 40000.0


class DistributionError(ValueError):
    """Raised for invalid distribution parameters."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise DistributionError(msg)


class Distribution:
    kind: str = "base"
    label: str = "Base"
    supports_truncation: bool = True

    def __init__(self, lower: float | None = None, upper: float | None = None):
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)
        if self.lower is not None and self.upper is not None:
            _check(self.lower < self.upper, "Truncation lower bound must be below the upper bound")
        if (self.lower is not None or self.upper is not None) and not self.supports_truncation:
            raise DistributionError(f"{self.label} distributions cannot be truncated")

    # ---- to be implemented by subclasses -------------------------------------
    def _ppf(self, u: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _cdf(self, x: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def params(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- public API ------------------------------------------------------------
    @property
    def is_constant(self) -> bool:
        return False

    def _mass_bounds(self) -> tuple[float, float]:
        lo = self._cdf(self.lower) if self.lower is not None else 0.0
        hi = self._cdf(self.upper) if self.upper is not None else 1.0
        _check(hi - lo > 1e-9, "Truncation bounds exclude (almost) all probability mass")
        return lo, hi

    def ppf(self, u) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), _U_EPS, 1.0 - _U_EPS)
        if self.lower is None and self.upper is None:
            return self._ppf(u)
        lo, hi = self._mass_bounds()
        x = self._ppf(np.clip(lo + u * (hi - lo), _U_EPS, 1.0 - _U_EPS))
        lb = -np.inf if self.lower is None else self.lower
        ub = np.inf if self.upper is None else self.upper
        return np.clip(x, lb, ub)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.ppf(rng.random(n))

    def exceedance(self, p_exceed_pct: float) -> float:
        """Value with ``p_exceed_pct`` % probability of being exceeded (P90 = low)."""
        return float(self.ppf(1.0 - p_exceed_pct / 100.0))

    @property
    def p90(self) -> float:
        return self.exceedance(90)

    @property
    def p50(self) -> float:
        return self.exceedance(50)

    @property
    def p10(self) -> float:
        return self.exceedance(10)

    @property
    def mean(self) -> float:
        return float(np.mean(self.ppf(_MEAN_GRID)))

    @property
    def std(self) -> float:
        return float(np.std(self.ppf(_MEAN_GRID)))

    def to_dict(self) -> dict[str, Any]:
        d = {"kind": self.kind, **self.params()}
        if self.lower is not None:
            d["lower"] = self.lower
        if self.upper is not None:
            d["upper"] = self.upper
        return d

    def describe(self, fmt: str = "{:.4g}") -> str:
        body = ", ".join(f"{k}={fmt.format(v) if isinstance(v, (int, float)) else v}"
                         for k, v in self.params().items())
        trunc = ""
        if self.lower is not None or self.upper is not None:
            trunc = f" [{'' if self.lower is None else fmt.format(self.lower)}, " \
                    f"{'' if self.upper is None else fmt.format(self.upper)}]"
        return f"{self.label}({body}){trunc}"

    def __repr__(self) -> str:
        return self.describe()


class Constant(Distribution):
    kind, label, supports_truncation = "constant", "Constant", False

    def __init__(self, value: float):
        super().__init__()
        _check(np.isfinite(value), "Constant value must be finite")
        self.value = float(value)

    @property
    def is_constant(self) -> bool:
        return True

    def _ppf(self, u):
        return np.full_like(u, self.value, dtype=float)

    def _cdf(self, x):
        return 1.0 if x >= self.value else 0.0

    def params(self):
        return {"value": self.value}


class Uniform(Distribution):
    kind, label = "uniform", "Uniform"

    def __init__(self, min: float, max: float, lower=None, upper=None):
        _check(max > min, "Uniform: max must be greater than min")
        self.min, self.max = float(min), float(max)
        super().__init__(lower, upper)

    def _ppf(self, u):
        return self.min + u * (self.max - self.min)

    def _cdf(self, x):
        return float(np.clip((x - self.min) / (self.max - self.min), 0, 1))

    def params(self):
        return {"min": self.min, "max": self.max}


class Triangular(Distribution):
    kind, label = "triangular", "Triangular"

    def __init__(self, min: float, mode: float, max: float, lower=None, upper=None):
        _check(max > min, "Triangular: max must be greater than min")
        _check(min <= mode <= max, "Triangular: mode must lie between min and max")
        self.min, self.mode, self.max = float(min), float(mode), float(max)
        c = (self.mode - self.min) / (self.max - self.min)
        self._d = stats.triang(c, loc=self.min, scale=self.max - self.min)
        super().__init__(lower, upper)

    def _ppf(self, u):
        return self._d.ppf(u)

    def _cdf(self, x):
        return float(self._d.cdf(x))

    def params(self):
        return {"min": self.min, "mode": self.mode, "max": self.max}


class PERT(Distribution):
    kind, label = "pert", "PERT"

    def __init__(self, min: float, mode: float, max: float, shape: float = 4.0, lower=None, upper=None):
        _check(max > min, "PERT: max must be greater than min")
        _check(min <= mode <= max, "PERT: mode must lie between min and max")
        _check(shape > 0, "PERT: shape must be positive")
        self.min, self.mode, self.max, self.shape = float(min), float(mode), float(max), float(shape)
        r = self.max - self.min
        a = 1.0 + self.shape * (self.mode - self.min) / r
        b = 1.0 + self.shape * (self.max - self.mode) / r
        self._d = stats.beta(a, b, loc=self.min, scale=r)
        super().__init__(lower, upper)

    def _ppf(self, u):
        return self._d.ppf(u)

    def _cdf(self, x):
        return float(self._d.cdf(x))

    def params(self):
        return {"min": self.min, "mode": self.mode, "max": self.max, "shape": self.shape}


class Normal(Distribution):
    kind, label = "normal", "Normal"

    def __init__(self, mean: float, sd: float, lower=None, upper=None):
        _check(sd > 0, "Normal: standard deviation must be positive")
        self.mu, self.sd = float(mean), float(sd)
        self._d = stats.norm(self.mu, self.sd)
        super().__init__(lower, upper)

    @classmethod
    def from_p90_p10(cls, p90: float, p10: float, **kw) -> "Normal":
        _check(p10 > p90, "P10 (high) must exceed P90 (low)")
        return cls((p90 + p10) / 2.0, (p10 - p90) / (2.0 * Z90), **kw)

    def _ppf(self, u):
        return self._d.ppf(u)

    def _cdf(self, x):
        return float(self._d.cdf(x))

    def params(self):
        return {"mean": self.mu, "sd": self.sd}


class Lognormal(Distribution):
    """Lognormal parameterised by the arithmetic mean and standard deviation."""

    kind, label = "lognormal", "Lognormal"

    def __init__(self, mean: float, sd: float, lower=None, upper=None):
        _check(mean > 0, "Lognormal: mean must be positive")
        _check(sd > 0, "Lognormal: standard deviation must be positive")
        self.m, self.s = float(mean), float(sd)
        self.sigma = float(np.sqrt(np.log1p((self.s / self.m) ** 2)))
        self.mu = float(np.log(self.m) - 0.5 * self.sigma ** 2)
        self._d = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))
        super().__init__(lower, upper)

    @classmethod
    def from_mu_sigma(cls, mu: float, sigma: float, **kw) -> "Lognormal":
        _check(sigma > 0, "Lognormal: sigma must be positive")
        mean = float(np.exp(mu + 0.5 * sigma ** 2))
        sd = float(mean * np.sqrt(np.expm1(sigma ** 2)))
        return cls(mean, sd, **kw)

    @classmethod
    def from_p90_p10(cls, p90: float, p10: float, **kw) -> "Lognormal":
        _check(p90 > 0, "Lognormal: P90 must be positive")
        _check(p10 > p90, "P10 (high) must exceed P90 (low)")
        mu = 0.5 * (np.log(p90) + np.log(p10))
        sigma = (np.log(p10) - np.log(p90)) / (2.0 * Z90)
        return cls.from_mu_sigma(mu, sigma, **kw)

    @classmethod
    def from_p90_p50_p10(cls, p90: float, p50: float, p10: float, **kw) -> tuple["Lognormal", float]:
        """Best lognormal through three percentiles.

        Returns the distribution and the relative asymmetry of the input in log
        space (0 = perfectly lognormal); values above ~0.2 mean a lognormal is a
        poor description and a percentile table should be used instead.
        """
        _check(0 < p90 < p50 < p10, "Require 0 < P90 < P50 < P10")
        lo, hi = np.log(p50) - np.log(p90), np.log(p10) - np.log(p50)
        sigma = (lo + hi) / (2.0 * Z90)
        asym = abs(hi - lo) / (hi + lo)
        return cls.from_mu_sigma(float(np.log(p50)), float(sigma), **kw), float(asym)

    def _ppf(self, u):
        return self._d.ppf(u)

    def _cdf(self, x):
        return float(self._d.cdf(x))

    def params(self):
        return {"mean": self.m, "sd": self.s}


class Beta(Distribution):
    kind, label = "beta", "Beta"

    def __init__(self, alpha: float, beta: float, min: float = 0.0, max: float = 1.0, lower=None, upper=None):
        _check(alpha > 0 and beta > 0, "Beta: alpha and beta must be positive")
        _check(max > min, "Beta: max must be greater than min")
        self.a, self.b, self.min, self.max = float(alpha), float(beta), float(min), float(max)
        self._d = stats.beta(self.a, self.b, loc=self.min, scale=self.max - self.min)
        super().__init__(lower, upper)

    def _ppf(self, u):
        return self._d.ppf(u)

    def _cdf(self, x):
        return float(self._d.cdf(x))

    def params(self):
        return {"alpha": self.a, "beta": self.b, "min": self.min, "max": self.max}


class Discrete(Distribution):
    kind, label, supports_truncation = "discrete", "Discrete", False

    def __init__(self, values: list[float], probs: list[float]):
        super().__init__()
        v = np.asarray(values, float)
        p = np.asarray(probs, float)
        _check(v.ndim == 1 and v.size >= 1 and v.size == p.size, "Discrete: values and probabilities must match")
        _check(np.all(p >= 0), "Discrete: probabilities must be non-negative")
        _check(abs(p.sum() - 1.0) < 1e-6, f"Discrete: probabilities must sum to 1 (got {p.sum():.4f})")
        order = np.argsort(v)
        self.values, self.probs = v[order], p[order]
        self._cum = np.cumsum(self.probs)
        self._cum[-1] = 1.0

    def _ppf(self, u):
        idx = np.searchsorted(self._cum, u, side="left")
        return self.values[np.minimum(idx, self.values.size - 1)]

    def _cdf(self, x):
        return float(self.probs[self.values <= x].sum())

    def params(self):
        return {"values": self.values.tolist(), "probs": self.probs.tolist()}

    def describe(self, fmt: str = "{:.4g}") -> str:
        pairs = ", ".join(f"{fmt.format(v)}:{p:.2f}" for v, p in zip(self.values, self.probs))
        return f"Discrete({pairs})"


class PercentileTable(Distribution):
    """Piecewise-linear CDF through user percentiles (exceedance convention).

    Must contain the P100 (minimum) and P0 (maximum) points, e.g.
    ``{100: 5, 90: 10, 50: 20, 10: 45, 0: 80}``.
    """

    kind, label, supports_truncation = "percentile_table", "Percentile table", False

    def __init__(self, points: dict[float, float] | list[list[float]]):
        super().__init__()
        items = list(points.items()) if isinstance(points, dict) else [tuple(p) for p in points]
        items = sorted(((float(k), float(v)) for k, v in items), key=lambda kv: -kv[0])
        exc = np.array([k for k, _ in items])
        val = np.array([v for _, v in items])
        _check(len(items) >= 2, "Percentile table needs at least two points")
        _check(exc[0] == 100.0 and exc[-1] == 0.0, "Percentile table must include P100 (min) and P0 (max)")
        _check(np.all(np.diff(exc) < 0), "Percentile table has duplicate percentiles")
        _check(np.all(np.diff(val) >= 0), "Values must increase as exceedance probability decreases")
        _check(val[-1] > val[0], "Percentile table: max must exceed min")
        self.cum = 1.0 - exc / 100.0  # non-exceedance, increasing
        self.vals = val

    def _ppf(self, u):
        return np.interp(u, self.cum, self.vals)

    def _cdf(self, x):
        return float(np.interp(x, self.vals, self.cum))

    def params(self):
        return {"points": [[float(100 * (1 - c)), float(v)] for c, v in zip(self.cum, self.vals)]}

    def describe(self, fmt: str = "{:.4g}") -> str:
        return "Table(" + ", ".join(f"P{100 * (1 - c):.0f}={fmt.format(v)}" for c, v in zip(self.cum, self.vals)) + ")"


REGISTRY: dict[str, type[Distribution]] = {
    c.kind: c for c in (Constant, Uniform, Triangular, PERT, Normal, Lognormal, Beta, Discrete, PercentileTable)
}

# Parameter schema used by the UI: (name, label)
PARAM_SCHEMA: dict[str, list[tuple[str, str]]] = {
    "constant": [("value", "Value")],
    "uniform": [("min", "Min"), ("max", "Max")],
    "triangular": [("min", "Min"), ("mode", "Most likely"), ("max", "Max")],
    "pert": [("min", "Min"), ("mode", "Most likely"), ("max", "Max"), ("shape", "Shape (λ)")],
    "normal": [("mean", "Mean"), ("sd", "Std dev")],
    "lognormal": [("mean", "Mean"), ("sd", "Std dev")],
    "beta": [("alpha", "α"), ("beta", "β"), ("min", "Min"), ("max", "Max")],
}

# Scale factor applied to each parameter when converting units (location/scale params
# scale with the unit; shape params do not).
SCALE_PARAMS = {"value", "min", "mode", "max", "mean", "sd", "lower", "upper"}


def from_dict(d: dict[str, Any]) -> Distribution:
    d = dict(d)
    kind = d.pop("kind", None)
    if kind not in REGISTRY:
        raise DistributionError(f"Unknown distribution kind '{kind}'")
    if kind == "percentile_table":
        return PercentileTable(d["points"])
    return REGISTRY[kind](**d)


def scaled(dist: Distribution, factor: float) -> Distribution:
    """Return the distribution of ``factor * X`` (factor > 0), preserving the family."""
    if factor <= 0:
        raise DistributionError("Scale factor must be positive")
    d = dist.to_dict()
    kind = d["kind"]
    if kind == "discrete":
        d["values"] = [v * factor for v in d["values"]]
    elif kind == "percentile_table":
        d["points"] = [[p, v * factor] for p, v in d["points"]]
    else:
        for k in list(d):
            if k in SCALE_PARAMS:
                d[k] = d[k] * factor
    return from_dict(d)


def summary(dist: Distribution) -> dict[str, float]:
    return {"P90": dist.p90, "P50": dist.p50, "P10": dist.p10, "Mean": dist.mean, "Std": dist.std}
