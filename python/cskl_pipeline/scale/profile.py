"""Per-dataset null profiles + interpolation + p-values (levers B + D).

A dataset's *null profile* is ``mu_D(g), sigma_D(g)`` for ``g`` in the pool grid,
where ``mu_D(g)`` and ``sigma_D(g)`` are the mean and (MLE, ddof=0) std of
``{cskl(sig_D, R_b^g) : b = 1..B}`` and ``R_b^g`` are the B transient null
signatures at size ``g`` (shared across all datasets - lever B).

At pair time the p-value is (exactly as ``cskl.pair_pvalue_vs_pool``, but from
cached profiles)::

    p = max( Phi(c_PQ; mu_P(m_Q), sigma_P(m_Q)),
             Phi(c_PQ; mu_Q(m_P), sigma_Q(m_P)) )

``mu_P(m_Q)`` / ``sigma_P(m_Q)`` are interpolated in ``m`` (``np.interp``). In
'exact-size' mode the grid already contains every distinct corpus size, so the
interpolation is exact.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from . import store as store_mod
from .fastcore import SignatureBank, batched_null_cskl_many
from .pool import PoolHandle

_SIGMA_FLOOR = 1e-12

# Cap loading-columns held in RAM per query chunk (~2 GB at 54675 features).
_DEFAULT_MAX_QUERY_COLS = 4000


# ---------------------------------------------------------------------------
# Compute (the batched sweep)
# ---------------------------------------------------------------------------

def compute_profiles(
    store: store_mod.Store,
    pool: PoolHandle,
    dataset_ids: Sequence[str],
    *,
    B: Optional[int] = None,
    max_query_cols: int = _DEFAULT_MAX_QUERY_COLS,
    verbose: bool = True,
) -> int:
    """Compute + persist null profiles for ``dataset_ids`` (those needing one).

    Generates each grid size's null bank exactly once and batches the cskl
    distances over all datasets in the chunk (the "fully batched over datasets"
    path). Returns the number of profiles written.
    """
    if B is None:
        B = pool.B
    dataset_ids = [d for d in dataset_ids if store.signature_path(d).exists()]
    if not dataset_ids:
        return 0

    grid = np.asarray(pool.grid, dtype=np.int64)
    written = 0

    # Chunk datasets so the packed query bank stays within a RAM budget. Each
    # chunk sweeps the whole grid (banks are regenerated per chunk, but a bank is
    # only ~50 s and chunks are few).
    for chunk in _chunk_by_cols(store, dataset_ids, max_query_cols):
        sigs = [store_mod.load_signature(store.signature_path(g)) for g in chunk]
        query = SignatureBank(sigs)
        n = len(chunk)
        mu = np.zeros((n, grid.shape[0]), dtype=np.float64)
        sigma = np.zeros((n, grid.shape[0]), dtype=np.float64)

        for gi, g in enumerate(grid):
            bank = pool.null_bank(int(g), B=B)
            dists = batched_null_cskl_many(query, bank)   # (n_datasets x B)
            mu[:, gi] = dists.mean(axis=1)
            sigma[:, gi] = dists.std(axis=1, ddof=0)      # matches scipy norm.fit
            del bank
        if verbose:
            print(f"[profile] computed {n} profiles over {grid.shape[0]} grid sizes "
                  f"(B={B}).")

        for di, gse in enumerate(chunk):
            store_mod.save_null_profile(
                store.null_profile_path(gse, pool.pool_version),
                grid=grid, mu=mu[di], sigma=sigma[di],
                pool_version=pool.pool_version, pool_hash=pool.pool_hash,
                feature_hash=pool.feature_hash, mode=("exact" if pool.meta.get("exact_size") else "grid"),
                B=B,
            )
            written += 1
    return written


def _chunk_by_cols(store: store_mod.Store, ids: Sequence[str], max_cols: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    cols = 0
    for gse in ids:
        try:
            c = store_mod.read_signature_meta(store.signature_path(gse))["c_components"]
        except Exception:
            c = 20
        if chunk and cols + c > max_cols:
            yield chunk
            chunk, cols = [], 0
        chunk.append(gse)
        cols += c
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Interpolation + p-values
# ---------------------------------------------------------------------------

def interp_profile(profile: store_mod.NullProfile, m: float) -> Tuple[float, float]:
    """Interpolate ``(mu, sigma)`` at sample size ``m`` (clamped to grid ends)."""
    mu = float(np.interp(m, profile.grid, profile.mu))
    sg = float(np.interp(m, profile.grid, profile.sigma))
    return mu, max(sg, _SIGMA_FLOOR)


def pair_pvalue_from_profiles(
    c_pq: float,
    prof_P: store_mod.NullProfile, m_Q: int,
    prof_Q: store_mod.NullProfile, m_P: int,
) -> float:
    """max(Phi(c; mu_P(m_Q), sig_P(m_Q)), Phi(c; mu_Q(m_P), sig_Q(m_P)))."""
    muP, sgP = interp_profile(prof_P, m_Q)
    muQ, sgQ = interp_profile(prof_Q, m_P)
    p12 = float(norm.cdf(c_pq, loc=muP, scale=sgP))
    p21 = float(norm.cdf(c_pq, loc=muQ, scale=sgQ))
    return max(p12, p21)


# --- vectorized all-pairs p-values (used by assemble) ----------------------

class ProfileTable:
    """Interpolators for a set of datasets, keyed by id, for fast all-pairs use."""

    def __init__(self, profiles: Dict[str, store_mod.NullProfile], sizes: Dict[str, int]):
        self.ids = list(profiles.keys())
        self.profiles = profiles
        self.sizes = sizes

    def mu_sigma_at(self, gse: str, m: float) -> Tuple[float, float]:
        return interp_profile(self.profiles[gse], m)

    def all_pairs_pvalues(self, ids: Sequence[str], cskl_matrix: np.ndarray) -> np.ndarray:
        """Return an ``(N, N)`` p-value matrix (diagonal set to 1.0)."""
        n = len(ids)
        sizes = np.array([self.sizes[g] for g in ids], dtype=float)
        # Precompute mu_i(m_j), sigma_i(m_j) for all i (profile) x j (partner size).
        grids = [self.profiles[g].grid for g in ids]
        mus = [self.profiles[g].mu for g in ids]
        sgs = [self.profiles[g].sigma for g in ids]

        MU = np.empty((n, n), dtype=np.float64)      # MU[i, j] = mu_i(m_j)
        SG = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            MU[i, :] = np.interp(sizes, grids[i], mus[i])
            SG[i, :] = np.maximum(np.interp(sizes, grids[i], sgs[i]), _SIGMA_FLOOR)

        # p12[i, j] uses profile i at partner j's size: Phi(c_ij; MU[i,j], SG[i,j])
        p12 = norm.cdf(cskl_matrix, loc=MU, scale=SG)
        # p21[i, j] uses profile j at partner i's size: Phi(c_ij; MU[j,i], SG[j,i])
        p21 = norm.cdf(cskl_matrix, loc=MU.T, scale=SG.T)
        P = np.maximum(p12, p21)
        np.fill_diagonal(P, 1.0)
        return P


def load_profile_table(
    store: store_mod.Store, ids: Sequence[str], pool_version: str,
) -> ProfileTable:
    profiles, sizes = {}, {}
    for gse in ids:
        profiles[gse] = store_mod.load_null_profile(store.null_profile_path(gse, pool_version))
        sizes[gse] = store_mod.read_signature_meta(store.signature_path(gse))["m_samples"]
    return ProfileTable(profiles, sizes)
