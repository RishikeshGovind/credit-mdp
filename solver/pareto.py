"""Pareto (non-dominated) utilities for maximisation objectives.

Everything here assumes objectives are oriented **larger-is-better** (the env and
evaluator guarantee this). Small, dependency-free, and readable.
"""

from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (>= on all, > on at least one)."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the Pareto-optimal rows of objective matrix ``F``."""
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and keep[j] and dominates(F[j], F[i]):
                keep[i] = False
                break
    return keep


def pareto_front(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows."""
    return np.where(non_dominated_mask(F))[0]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance for a set of points (larger = more isolated)."""
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(F[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = F[order[-1], k] - F[order[0], k]
        if span <= 0:
            continue
        for r in range(1, n - 1):
            dist[order[r]] += (F[order[r + 1], k] - F[order[r - 1], k]) / span
    return dist


def normalize(F: np.ndarray) -> np.ndarray:
    """Min-max each column to [0, 1] for hypervolume / plotting."""
    lo = F.min(axis=0)
    span = np.where(F.max(axis=0) - lo > 0, F.max(axis=0) - lo, 1.0)
    return (F - lo) / span


def hypervolume(F: np.ndarray, ref: np.ndarray | None = None,
                n_samples: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo hypervolume of the dominated region above ``ref`` (maximisation).

    Used as a single transparent scalar to compare how much objective space a
    policy *set* covers. Normalised inputs recommended.
    """
    F = np.atleast_2d(F)
    if ref is None:
        ref = F.min(axis=0) - 1e-9
    hi = F.max(axis=0)
    if np.any(hi <= ref):
        return 0.0
    rng = np.random.default_rng(seed)
    pts = rng.uniform(ref, hi, size=(n_samples, F.shape[1]))
    dominated = np.zeros(n_samples, dtype=bool)
    for f in F:
        dominated |= np.all(pts <= f, axis=1)
    vol_box = float(np.prod(hi - ref))
    return vol_box * dominated.mean()
