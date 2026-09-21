"""Sampling designs and rank correlation.

* Latin Hypercube Sampling (LHS) or plain Monte Carlo uniform designs.
* Iman & Conover (1982) distribution-free rank correlation: reorders the
  columns of an independent sample so their Spearman rank correlation matches
  a target matrix while keeping each marginal distribution exactly intact.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats


class CorrelationError(ValueError):
    pass


def uniform_design(n: int, k: int, rng: np.random.Generator, method: str = "lhs") -> np.ndarray:
    """n x k matrix of U(0,1) values. ``method`` is 'lhs' or 'mc'."""
    if n < 2:
        raise ValueError("Number of trials must be at least 2")
    if method == "mc":
        return rng.random((n, k))
    if method != "lhs":
        raise ValueError(f"Unknown sampling method '{method}'")
    u = np.empty((n, k))
    for j in range(k):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def nearest_correlation(c: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, bool]:
    """Nearest positive-definite correlation matrix by eigenvalue clipping.

    Returns (matrix, was_modified).
    """
    c = np.asarray(c, float)
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    if w.min() > eps:
        return c, False
    w = np.clip(w, eps, None)
    fixed = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed, True


def build_matrix(names: list[str], pairs: Iterable[tuple[str, str, float]]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    c = np.eye(len(names))
    for a, b, rho in pairs:
        if a not in idx or b not in idx:
            raise CorrelationError(f"Correlation refers to unknown or constant variable: {a} / {b}")
        if a == b:
            raise CorrelationError(f"Variable '{a}' cannot be correlated with itself")
        if not -1.0 <= rho <= 1.0:
            raise CorrelationError(f"Correlation {a}-{b} = {rho} is outside [-1, 1]")
        c[idx[a], idx[b]] = c[idx[b], idx[a]] = float(rho)
    return c


def spearman_to_normal_score(c: np.ndarray) -> np.ndarray:
    """Pearson correlation of normal scores that yields Spearman rank correlation ``c``.

    For a bivariate normal, rho_S = (6 / pi) * arcsin(r / 2), so r = 2 sin(pi rho_S / 6).
    Iman-Conover reproduces normal-score correlation, so targets are converted first.
    """
    r = 2.0 * np.sin(np.pi * np.asarray(c, float) / 6.0)
    np.fill_diagonal(r, 1.0)
    return r


def iman_conover(x: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reorder columns of ``x`` (n x k) to approximate rank correlation ``target``."""
    x = np.asarray(x, float)
    n, k = x.shape
    if k < 2 or np.allclose(target, np.eye(k)):
        return x
    target, _ = nearest_correlation(spearman_to_normal_score(target))
    scores = stats.norm.ppf(np.arange(1, n + 1) / (n + 1.0))
    m = np.column_stack([rng.permutation(scores) for _ in range(k)])
    e = np.corrcoef(m, rowvar=False)
    f = np.linalg.cholesky(nearest_correlation(e)[0])
    p = np.linalg.cholesky(target)
    t = m @ np.linalg.inv(f).T @ p.T
    out = np.empty_like(x)
    for j in range(k):
        ranks = np.argsort(np.argsort(t[:, j], kind="stable"), kind="stable")
        out[:, j] = np.sort(x[:, j], kind="stable")[ranks]
    return out


def row_orders_for_rank_correlation(keys: np.ndarray, target: np.ndarray,
                                    rng: np.random.Generator) -> list[np.ndarray]:
    """Row permutations that impose ``target`` rank correlation on ``keys`` columns.

    Used to reorder *whole trials* of several independent models (e.g. all output
    arrays of a segment) according to one key quantity per model.
    """
    n, k = keys.shape
    target, _ = nearest_correlation(spearman_to_normal_score(target))
    scores = stats.norm.ppf(np.arange(1, n + 1) / (n + 1.0))
    m = np.column_stack([rng.permutation(scores) for _ in range(k)])
    f = np.linalg.cholesky(nearest_correlation(np.corrcoef(m, rowvar=False))[0])
    t = m @ np.linalg.inv(f).T @ np.linalg.cholesky(target).T
    orders = []
    for j in range(k):
        target_rank = np.argsort(np.argsort(t[:, j], kind="stable"), kind="stable")
        by_value = np.lexsort((rng.random(n), keys[:, j]))  # random tie-break
        orders.append(by_value[target_rank])
    return orders


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).statistic)
