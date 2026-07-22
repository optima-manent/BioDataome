"""Reference (slow) implementations used ONLY by the faithfulness gates.

These call the ``cskl`` source of truth directly - ``fit_pca_signature``,
``cskl``, ``Pool``/``pair_pvalue_vs_pool`` - so the fast paths can be diffed
against them. They deliberately mirror the fast path's *null draws* (same pool
matrix, same per-size RNG) so any residual difference is only "gram-fit vs SVD"
and "batched vs loop", never a different random sample.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

import cskl
from cskl import PCASignature

from .pool import PoolHandle


def reference_fit(X: np.ndarray, alpha: float, feature_names, rng) -> PCASignature:
    """The reference signature fit (full SVD path)."""
    return cskl.fit_pca_signature(
        X, alpha=alpha, feature_names=feature_names, rng=rng,
        nan_policy="zero", r_compat_noise=True,
    )


def reference_observed_matrix(sigs: Dict[str, PCASignature], ids: Sequence[str]) -> np.ndarray:
    n = len(ids)
    M = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            M[i, j] = cskl.cskl(sigs[a], sigs[b])
    return M


def reference_null_bank(pool: PoolHandle, m: int, B: int) -> List[PCASignature]:
    """Null bank at size m via the reference fit, drawing the SAME bootstrap rows
    as :meth:`PoolHandle.null_bank` (identical ``_size_rng`` and ``integers`` calls)."""
    X = np.ascontiguousarray(pool.X_pool, dtype=np.float64)
    Mrows = X.shape[0]
    rng = pool._size_rng(m)
    out = []
    for _ in range(B):
        idx = rng.integers(low=0, high=Mrows, size=m, endpoint=False)
        out.append(reference_fit(X[idx, :], pool.alpha, None, rng))
    return out


def reference_profile(
    pool: PoolHandle, sig: PCASignature, grid: Sequence[int], B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """(mu, sigma) over the grid using the reference fit + the cskl loop."""
    grid = list(grid)
    mu = np.zeros(len(grid))
    sigma = np.zeros(len(grid))
    for gi, g in enumerate(grid):
        bank = reference_null_bank(pool, int(g), B)
        d = np.array([cskl.cskl(sig, r) for r in bank])
        mu[gi] = d.mean()
        sigma[gi] = d.std(ddof=0)
    return mu, sigma
