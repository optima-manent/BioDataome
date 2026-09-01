"""Fast numerical paths for c-SKL (levers B + C).

Everything here is validated to reproduce :mod:`cskl` exactly (or to Monte-Carlo
noise). The three primitives are:

* :func:`fast_fit_signature` - top-c PCA signature via ``eigh`` of the ``m x m``
  gram matrix instead of a full ``m x 54675`` SVD. Reproduces
  ``cskl.fit_pca_signature`` (same standardization, same c-selection, same
  lambda renormalization); eigenvalues match to ~2e-15, cskl to ~1e-11.

* :func:`batched_null_cskl` - ``cskl(P, R_b)`` for one query signature ``P`` and a
  whole *bank* of null signatures ``R_b`` in one matmul + segment-sums. Reproduces
  the ``cskl`` loop to ~1e-11 (diff ~0).

* :func:`observed_cskl_matrix` - all-pairs observed cskl over a set of signatures,
  computed as one blocked gram matmul + 2-D segment-sums.

The c-SKL identity used throughout (alpha, n shared; P, Q orthonormal columns)::

    S           = P^T Q                                  # (cP x cQ)
    sum_term    = sum_ij (lam^P_i + lam^Q_j) * S_ij^2
    cskl(P, Q)  = max( (2*alpha*n - sum_term) / (2*(1-alpha)), 0 )

which is exactly ``cskl.cskl``. ``sum_term`` factorizes as
``lam^P . rowsum(S^2) + colsum(S^2) . lam^Q`` - that factorization is what makes
the batched forms possible.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

import cskl
from cskl import PCASignature

# Feature threshold below which the reference variance rule flags a "zero
# variance" feature (matches cskl._standardize_features).
_ZERO_VAR_TOL = 1e-7


# ---------------------------------------------------------------------------
# Standardization (faithful fast path)
# ---------------------------------------------------------------------------

def standardize(
    X: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    *,
    r_compat_noise: bool = True,
) -> np.ndarray:
    """Return the standardized matrix ``Xs`` exactly as ``cskl._standardize_features``.

    Fast path (all-finite input, no near-zero-variance feature): plain mean/var,
    no RNG draw - numerically identical to the reference to ~1e-15. Any non-finite
    value or any zero-variance feature (which needs the R-compatible noise draw)
    is delegated to the reference implementation so behavior is byte-identical.
    """
    X = np.asarray(X, dtype=float)
    m = X.shape[0]

    if np.isfinite(X).all():
        mu = X.mean(axis=0)
        Xc = X - mu
        ss = np.einsum("ij,ij->j", Xc, Xc)          # per-feature sum of squares
        var = ss / (m - 1)
        if not np.any(var <= _ZERO_VAR_TOL):
            # No zero-variance feature -> reference injects no noise, draws no RNG.
            sd = np.sqrt(var)
            Xc /= sd                                 # sd > sqrt(1e-7) > 0 here
            return Xc

    # Edge cases (non-finite entries, or zero-variance features that require the
    # exact R-compatible recycled-noise RNG draw): use the reference verbatim.
    Xs, _, _ = cskl._standardize_features(X, rng=rng, r_compat_noise=r_compat_noise)
    return Xs


# ---------------------------------------------------------------------------
# Fast fit (lever C)
# ---------------------------------------------------------------------------

def fast_fit_signature(
    X: np.ndarray,
    alpha: float = 0.5,
    feature_names: Optional[Sequence[str]] = None,
    *,
    min_components: int = 1,
    max_components: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    nan_policy: str = "raise",
    r_compat_noise: bool = True,
) -> PCASignature:
    """Gram-trick equivalent of ``cskl.fit_pca_signature``.

    Computes the full eigenvalue spectrum from ``eigh(Xs @ Xs.T)`` (an ``m x m``
    problem) for the existing component-selection rule, then forms only the top-c
    right singular vectors ``V = Xs^T (U_c / s_c)``. Sign flips are irrelevant to
    cskl (it uses ``S^2``).
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (m_samples, n_features).")

    m, n = X.shape
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if m <= 1:
        raise ValueError("Need at least 2 samples to compute PCA signature.")

    nonfinite = ~np.isfinite(X)
    if np.any(nonfinite):
        if nan_policy == "raise":
            raise ValueError(
                f"Found {int(nonfinite.sum())} non-finite entries in X. "
                "Clean/filter the data before fitting, or pass nan_policy='zero'."
            )
        if nan_policy == "zero":
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError("nan_policy must be either 'raise' or 'zero'.")

    feature_names = list(feature_names) if feature_names is not None else None

    Xs = standardize(X, rng=rng, r_compat_noise=r_compat_noise)
    if not np.all(np.isfinite(Xs)):
        if nan_policy == "raise":
            raise ValueError("Standardized data contains non-finite values.")
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    # Gram matrix: eigenvalues are the squared singular values of Xs.
    G = Xs @ Xs.T
    w, U = np.linalg.eigh(G)          # ascending eigenvalues
    w = w[::-1]                        # descending, length m (full spectrum)
    U = U[:, ::-1]
    w = np.clip(w, 0.0, None)
    lam_raw = w / float(m - 1)

    total_var = float(np.sum(lam_raw))
    if total_var <= 0.0 or not np.isfinite(total_var):
        raise ValueError("Total PCA variance is zero or non-finite.")

    target_var = alpha * float(n)
    c = int(np.searchsorted(np.cumsum(lam_raw), target_var, side="left")) + 1
    c = max(int(min_components), c)
    if max_components is not None:
        c = min(c, int(max_components))
    c = min(c, lam_raw.shape[0])

    s_c = np.sqrt(w[:c])
    s_safe = np.where(s_c > 0.0, s_c, 1.0)
    P = (Xs.T @ U[:, :c]) / s_safe[None, :]        # (n x c)
    lam = lam_raw[:c].copy()

    target = alpha * float(n)
    s = float(np.sum(lam))
    if s > 0.0 and np.isfinite(s):
        lam = (lam / s) * target
    else:
        lam = np.full(c, target / c, dtype=float)

    return PCASignature(
        P=P,
        lam=lam,
        n_features=n,
        m_samples=m,
        alpha=float(alpha),
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

class SignatureBank:
    """A set of signatures packed for batched cskl.

    Attributes
    ----------
    P : (n_features, sum_c) float64   -- loadings concatenated column-wise
    lam : (sum_c,) float64            -- eigenvalues concatenated
    offsets : (n_sig + 1,) int        -- column block boundaries (offsets[k]:offsets[k+1])
    """

    __slots__ = ("P", "lam", "offsets", "n_features", "alpha")

    def __init__(self, signatures: Sequence[PCASignature]):
        if len(signatures) == 0:
            raise ValueError("Need at least one signature to build a bank.")
        n = signatures[0].n_features
        alpha = signatures[0].alpha
        if any(not np.isclose(signature.alpha, alpha) for signature in signatures[1:]):
            raise ValueError("All signatures in a bank must use the same alpha.")
        cs = [s.P.shape[1] for s in signatures]
        self.offsets = np.concatenate([[0], np.cumsum(cs)]).astype(np.int64)
        self.P = np.concatenate([np.ascontiguousarray(s.P, dtype=np.float64) for s in signatures], axis=1)
        self.lam = np.concatenate([np.asarray(s.lam, dtype=np.float64) for s in signatures])
        self.n_features = int(n)
        self.alpha = float(alpha)

    @property
    def n_sig(self) -> int:
        return self.offsets.shape[0] - 1


def _segment_sums(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sum ``values`` within blocks delimited by ``offsets`` (len n_block+1)."""
    return np.add.reduceat(values, offsets[:-1])


# ---------------------------------------------------------------------------
# Batched null distances (lever C)
# ---------------------------------------------------------------------------

def batched_null_cskl(sigP: PCASignature, bank: SignatureBank) -> np.ndarray:
    """``cskl(sigP, R_b)`` for every signature ``R_b`` in ``bank``.

    One matmul ``S = P^T Rcat`` then per-null segment sums over ``S^2``. Reproduces
    ``[cskl.cskl(sigP, R_b) for R_b in bank]`` to ~1e-11.
    """
    if not np.isclose(sigP.alpha, bank.alpha):
        raise ValueError("Query and bank signatures must use the same alpha.")
    alpha = float(sigP.alpha)
    n = float(sigP.n_features)
    lamP = np.asarray(sigP.lam, dtype=np.float64)

    S = np.asarray(sigP.P, dtype=np.float64).T @ bank.P     # (cP x sum_c)
    SS = S * S
    colsum = SS.sum(axis=0)                                  # per null-column
    colw = lamP @ SS                                         # lamP-weighted column sums

    term1 = _segment_sums(colw, bank.offsets)               # lam^P . rowsum_b
    term2 = _segment_sums(bank.lam * colsum, bank.offsets)  # colsum_b . lam^R_b
    sum_term = term1 + term2
    val = (2.0 * alpha * n - sum_term) / (2.0 * (1.0 - alpha))
    return np.maximum(val, 0.0)


def batched_null_cskl_many(query: SignatureBank, bank: SignatureBank) -> np.ndarray:
    """``cskl(P_a, R_b)`` for every query ``P_a`` and every null ``R_b``.

    Returns an ``(n_query, n_null)`` matrix. This is the "fully batched over
    datasets" path used by the null-profile sweep: one big matmul shared across all
    query datasets at a given grid size.
    """
    if not np.isclose(query.alpha, bank.alpha):
        raise ValueError("Query and bank signatures must use the same alpha.")
    alpha = float(query.alpha)
    n = float(query.n_features)

    S = query.P.T @ bank.P                                  # (sum_cQ x sum_cR)
    SS = S * S

    # term2 depends only on null columns: colsum_b . lam^R_b, summed per (query,null).
    col_contrib = bank.lam * SS                             # scale each null column by lam^R
    # sum over null-columns within each null block -> (sum_cQ x n_null)
    col_by_null = np.add.reduceat(col_contrib, bank.offsets[:-1], axis=1)
    # then sum over query rows within each query block -> (n_query x n_null)
    term2 = np.add.reduceat(col_by_null, query.offsets[:-1], axis=0)

    # term1: lam^P_i * SS summed over null-cols within block, then lam-weighted
    # sum over query rows within block.
    ss_by_null = np.add.reduceat(SS, bank.offsets[:-1], axis=1)       # (sum_cQ x n_null)
    row_scaled = query.lam[:, None] * ss_by_null
    term1 = np.add.reduceat(row_scaled, query.offsets[:-1], axis=0)   # (n_query x n_null)

    sum_term = term1 + term2
    val = (2.0 * alpha * n - sum_term) / (2.0 * (1.0 - alpha))
    return np.maximum(val, 0.0)


# ---------------------------------------------------------------------------
# All-pairs observed matrix
# ---------------------------------------------------------------------------

def observed_cskl_matrix(
    signatures: Sequence[PCASignature],
    *,
    row_block: int = 20_000_000,
) -> np.ndarray:
    """Symmetric ``(N, N)`` matrix of ``cskl(sig_i, sig_j)`` for all pairs.

    Equivalent to ``[[cskl.cskl(a, b) ...]]`` but computed as blocked gram matmuls
    plus 2-D segment sums. ``row_block`` caps peak memory by limiting how many
    loading-columns are processed per block (intermediate ~ ``8 * row_block`` bytes).
    """
    bank = SignatureBank(list(signatures))
    N = bank.n_sig
    alpha = bank.alpha
    n = float(bank.n_features)
    lam = bank.lam
    offs = bank.offsets

    out = np.zeros((N, N), dtype=np.float64)

    # Process query datasets in row-blocks so the intermediate (block_cols x sum_c)
    # array stays bounded.
    total_cols = bank.P.shape[1]
    start_sig = 0
    while start_sig < N:
        # grow a block of datasets until its column span exceeds a budget
        end_sig = start_sig + 1
        while end_sig < N and (offs[end_sig + 1] - offs[start_sig]) * total_cols <= row_block:
            end_sig += 1
        col_lo, col_hi = int(offs[start_sig]), int(offs[end_sig])
        Pblock = bank.P[:, col_lo:col_hi]                       # (n x block_cols)

        S = Pblock.T @ bank.P                                   # (block_cols x sum_c)
        SS = S * S
        M = (lam[col_lo:col_hi][:, None] + lam[None, :]) * SS   # (lam_i + lam_j) S^2

        # 2-D segment sum: over null-columns (all datasets), then over query rows.
        col_summed = np.add.reduceat(M, offs[:-1], axis=1)      # (block_cols x N)
        block_offs = offs[start_sig:end_sig + 1] - offs[start_sig]
        sum_term = np.add.reduceat(col_summed, block_offs[:-1], axis=0)  # (block_N x N)

        val = (2.0 * alpha * n - sum_term) / (2.0 * (1.0 - alpha))
        out[start_sig:end_sig, :] = np.maximum(val, 0.0)
        start_sig = end_sig

    # enforce exact symmetry (matmul asymmetry is ~1e-11)
    out = 0.5 * (out + out.T)
    return out


# ---------------------------------------------------------------------------
# Single-pair convenience (thin, exact wrapper around the library)
# ---------------------------------------------------------------------------

def cskl_pair(sigP: PCASignature, sigQ: PCASignature) -> float:
    """Single observed cskl - delegates to the library source of truth."""
    return cskl.cskl(sigP, sigQ)
