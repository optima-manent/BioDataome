"""Compressed symmetric KL similarity for high-dimensional biological datasets.

The module implements PCA signatures, the paper's c-SKL approximation,
semi-parametric bootstrap significance, Benjamini-Hochberg correction, network
construction, and feature-set explanations. It is self-contained apart from
NumPy and SciPy and can be used independently of the Atlas control plane.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Iterable, Dict
from dataclasses import dataclass
import scipy.stats as stats
import numpy as np


__all__ = [
    "PCASignature",
    "fit_pca_signature",
    "cskl",
    "Pool",
    "pair_pvalue_vs_pool",
    "bh_qvalues",
    "build_dataset_network",
    "explain_topk",
    "explain_set_topk",
    "pair_significant_vs_pool"
]


# -------------------------
# PCA-based dataset signature
# -------------------------

@dataclass
class PCASignature:
    """
    PCA "signature" for a dataset: top right singular vectors (feature loadings) and
    renormalized eigenvalues summing to alpha * n_features, with an isotropic residual of 1 - alpha.

    Attributes
    ----------
    P : np.ndarray of shape (n_features, c)
        Orthonormal columns = principal axes (right singular vectors).
    lam : np.ndarray of shape (c,)
        Renormalized eigenvalues (sum == alpha * n_features).
    n_features : int
        Number of features (variables).
    m_samples : int
        Number of samples (rows) used to fit this signature.
    alpha : float
        Target variance fraction captured by the low-rank component.
    feature_names : Optional[List[str]]
        Names of features (for explainers); length should equal n_features if provided.
    """
    P: np.ndarray
    lam: np.ndarray
    n_features: int
    m_samples: int
    alpha: float
    feature_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.P.ndim != 2 or self.lam.ndim != 1:
            raise ValueError("P must be 2D and lam must be 1D")
        if self.P.shape[0] != self.n_features:
            raise ValueError("P.shape[0] must equal n_features")
        if self.P.shape[1] != self.lam.shape[0]:
            raise ValueError("Number of components mismatch between P and lam")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")

        # Best-effort orthonormality check for P columns
        gram = self.P.T @ self.P
        if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-5):
            raise ValueError("Columns of P are not orthonormal; check PCA computation.")

        # ensure lam sums to alpha * n_features
        target = self.alpha * float(self.n_features)
        if not np.all(np.isfinite(self.P)):
            raise ValueError("P contains non-finite values.")
        if not np.all(np.isfinite(self.lam)):
            raise ValueError("lam contains non-finite values.")

        target = self.alpha * float(self.n_features)
        s = float(np.sum(self.lam))

        if (not np.isfinite(s)) or s <= 0.0:
            c = self.lam.shape[0]
            self.lam = np.full(c, target / c, dtype=float)
        else:
            self.lam = (self.lam / s) * target


def _standardize_features(
    X: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    *,
    r_compat_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance across rows.

    For near-zero-variance features, inject tiny uniform noise before
    standardization. With r_compat_noise=True, the same noise vector is
    added to every flagged feature, matching R's vector recycling behavior.
    """
    rng = np.random.default_rng(rng)

    # R code used variance thresholding, not standard-deviation thresholding.
    var_raw = np.nanvar(X, axis=0, ddof=1)
    zero_var_mask = (~np.isfinite(var_raw)) | (var_raw <= 1e-7)

    X_mod = X.copy()

    if np.any(zero_var_mask):
        if r_compat_noise:
            # R-compatible: d[, vars] + runif(nrow(d), 0, 1e-7)
            # recycles the same row-wise vector across all selected columns.
            noise = rng.uniform(0.0, 1e-7, size=(X.shape[0], 1))
        else:
            # More statistically natural, but not R-compatible.
            noise = rng.uniform(
                0.0,
                1e-7,
                size=(X.shape[0], int(zero_var_mask.sum())),
            )

        X_mod[:, zero_var_mask] += noise

    mu = np.nanmean(X_mod, axis=0)
    Xc = X_mod - mu

    sd = np.nanstd(Xc, axis=0, ddof=1)
    sd_safe = np.where((sd == 0.0) | ~np.isfinite(sd), 1.0, sd)

    Xs = Xc / sd_safe
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    return Xs, mu, sd_safe


def fit_pca_signature(
    X: np.ndarray,
    alpha: float = 0.5,
    feature_names: Optional[List[str]] = None,
    min_components: int = 1,
    max_components: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    nan_policy: str = "raise",
    r_compat_noise: bool = True,
) -> PCASignature:
    """
    Compute a low-rank + isotropic-noise PCA signature for dataset X.

    Parameters
    ----------
    X : array-like, shape (m_samples, n_features)
        Input data matrix. Each column is a feature/variable.
    alpha : float in (0,1)
        Target fraction of total standardized variance captured by the
        low-rank part.
    feature_names : optional list of strings
        Feature names aligned to columns of X.
    min_components : int
        Lower bound on number of kept components.
    max_components : Optional[int]
        Upper bound on number of kept components.
    rng : Optional[np.random.Generator]
        Random generator used only for zero-variance noise injection and
        SVD retry jitter.
    nan_policy : {"raise", "zero"}
        "raise" is library-safe and paper-faithful after preprocessing.
        "zero" replaces non-finite standardized values with zero.
    r_compat_noise : bool
        If True, match R-style row-wise noise recycling for constant columns.

    Returns
    -------
    PCASignature
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
    if min_components < 1:
        raise ValueError("min_components must be >= 1.")
    if max_components is not None and max_components < 1:
        raise ValueError("max_components must be >= 1 when provided.")

    if feature_names is not None:
        if len(feature_names) != n:
            raise ValueError(
                f"feature_names length must equal n_features={n}. "
                f"Got {len(feature_names)}."
            )
        feature_names = list(feature_names)

    nonfinite_mask = ~np.isfinite(X)
    if np.any(nonfinite_mask):
        n_bad = int(nonfinite_mask.sum())

        if nan_policy == "raise":
            raise ValueError(
                f"Found {n_bad} non-finite entries in X. "
                "Clean/filter the data before fitting, or call "
                "fit_pca_signature(..., nan_policy='zero') to preserve "
                "the old permissive behavior."
            )
        elif nan_policy == "zero":
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError("nan_policy must be either 'raise' or 'zero'.")

    Xs, _, _ = _standardize_features(
        X,
        rng=rng,
        r_compat_noise=r_compat_noise,
    )

    if not np.all(np.isfinite(Xs)):
        if nan_policy == "raise":
            raise ValueError("Standardized data contains non-finite values.")
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        U, S, VT = np.linalg.svd(Xs, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = 1e-8 * rng.standard_normal(size=Xs.shape)
        U, S, VT = np.linalg.svd(Xs + jitter, full_matrices=False)

    lam_raw = (S ** 2) / float(m - 1)
    total_var = float(np.sum(lam_raw))

    if total_var <= 0.0 or not np.isfinite(total_var):
        raise ValueError("Total PCA variance is zero or non-finite.")

    # Standardized data should have total variance approximately n_features.
    # Use alpha * n to match the c-SKL normalization.
    target_var = alpha * float(n)

    c = int(np.searchsorted(np.cumsum(lam_raw), target_var, side="left")) + 1
    c = max(int(min_components), c)

    if max_components is not None:
        c = min(c, int(max_components))

    c = min(c, lam_raw.shape[0])

    P = VT.T[:, :c].copy()
    lam = lam_raw[:c].copy()

    # Normalize retained eigenvalues to alpha * n_features.
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
        alpha=alpha,
        feature_names=feature_names,
    )


# -------------------------
# c-SKL (compressed symmetric KL) similarity
# -------------------------

def cskl(sigP: PCASignature, sigQ: PCASignature) -> float:
    """
    Paper c-SKL approximation (Eq. (1) in the paper).
    Smaller = more similar. ~0 when the retained subspaces coincide (up to numerics).
    """
    if sigP.n_features != sigQ.n_features:
        raise ValueError("Signatures must have the same n_features.")

    if sigP.feature_names is not None and sigQ.feature_names is not None:
        if list(sigP.feature_names) != list(sigQ.feature_names):
            raise ValueError("Signatures must use the same feature order.")

    if not np.isclose(sigP.alpha, sigQ.alpha):
        raise ValueError("Signatures must use the same alpha.")
    alpha = float(sigP.alpha)
    n = float(sigP.n_features)

    P, lamP = sigP.P, sigP.lam
    Q, lamQ = sigQ.P, sigQ.lam

    S = P.T @ Q                      # (cP x cQ)
    W = lamP[:, None] + lamQ[None, :]# (cP x cQ)

    sum_term = float(np.sum(W * (S * S)))  # Σ_{i,j} (λ_i^P + λ_j^Q) (P_i^T Q_j)^2
    val = (2.0 * alpha * n - sum_term) / (2.0 * (1.0 - alpha))

    # guard tiny negative due to floating error
    return float(max(val, 0.0))



# -------------------------
# Semi-parametric bootstrap vs pool
# -------------------------

class Pool:
    """
    Pooled background for a given platform (same features).
    Uses in-memory caching to store bootstrap PCASignatures by sample size,
    drastically accelerating pairwise p-value calculations.

    Parameters
    ----------
    X_pool : array-like, shape (M, n_features)
        Pooled samples across datasets on the same platform.
    alpha : float
        The alpha to use when fitting bootstrap signatures (should match dataset signatures).
    feature_names : Optional[List[str]]
        Feature names aligned to columns.
    """

    def __init__(
            self,
            X_pool: np.ndarray,
            alpha: float = 0.5,
            feature_names: Optional[List[str]] = None,
            nan_policy: str = "raise",
            r_compat_noise: bool = True,
    ):
        X_pool = np.asarray(X_pool, dtype=float)
        if X_pool.ndim != 2:
            raise ValueError("X_pool must be 2D")
        self.X_pool = X_pool
        self.M, self.n = X_pool.shape

        if self.M < 2:
            raise ValueError("X_pool must contain at least 2 samples.")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0,1).")

        self.alpha = float(alpha)

        if feature_names is not None:
            if len(feature_names) != self.n:
                raise ValueError(
                    f"feature_names length must equal pool n_features={self.n}. "
                    f"Got {len(feature_names)}."
                )
            self.feature_names = list(feature_names)
        else:
            self.feature_names = None

        if nan_policy not in {"raise", "zero"}:
            raise ValueError("nan_policy must be either 'raise' or 'zero'.")

        self.nan_policy = nan_policy
        self.r_compat_noise = bool(r_compat_noise)

        # NEW: Cache for null signatures based on sample size `m`.
        # Format: { m: [PCASignature_1, ..., PCASignature_B] }
        self._null_signatures_cache: Dict[int, List[PCASignature]] = {}

    def sample_signature(
            self,
            m: int,
            rng: Optional[np.random.Generator] = None,
    ) -> PCASignature:
        """
        Draw m rows with replacement, standardize within the draw, and fit a signature.
        """
        if m < 2:
            raise ValueError("Need at least 2 samples in bootstrap draw.")

        rng = np.random.default_rng(rng)
        idx = rng.integers(low=0, high=self.M, size=m, endpoint=False)
        Xb = self.X_pool[idx, :]

        return fit_pca_signature(
            Xb,
            alpha=self.alpha,
            feature_names=self.feature_names,
            rng=rng,
            nan_policy=self.nan_policy,
            r_compat_noise=self.r_compat_noise,
        )

    def get_null_signatures(self, m: int, B: int, rng: Optional[np.random.Generator] = None) -> List[PCASignature]:
        """
        Retrieve or generate B bootstrap signatures for a specific sample size m.
        Uses in-memory caching to avoid redundant PCA fits across multiple pairwise comparisons.
        """
        if m not in self._null_signatures_cache:
            # Generate B signatures ONCE for this sample size if not present
            rng = np.random.default_rng(rng)
            self._null_signatures_cache[m] = [self.sample_signature(m, rng=rng) for _ in range(B)]

        # If the requested B is larger than what is cached, generate the difference
        current_cached = len(self._null_signatures_cache[m])
        if B > current_cached:
            rng = np.random.default_rng(rng)
            additional_sigs = [self.sample_signature(m, rng=rng) for _ in range(B - current_cached)]
            self._null_signatures_cache[m].extend(additional_sigs)

        # Return exactly B signatures
        return self._null_signatures_cache[m][:B]


def pair_significant_vs_pool(
    sigP: PCASignature,
    sigQ: PCASignature,
    pool: Pool,
    B: int = 500,
    rng: Optional[np.random.Generator] = None,
    metric=cskl,
    tail_alpha: float = 0.025,
    alpha: Optional[float] = None,
) -> bool:
    """
    Legacy-style TRUE/FALSE significance call.

    Equivalent to requiring the observed c-SKL p-value to fall below
    tail_alpha. The old keyword alpha is accepted as a deprecated alias.
    """
    if alpha is not None:
        tail_alpha = alpha

    if not (0.0 < float(tail_alpha) < 1.0):
        raise ValueError("tail_alpha must be in (0,1).")

    p = pair_pvalue_vs_pool(
        sigP=sigP,
        sigQ=sigQ,
        pool=pool,
        B=B,
        rng=rng,
        metric=metric,
    )
    return bool(p <= tail_alpha)


def pair_pvalue_vs_pool(
        sigP: PCASignature,
        sigQ: PCASignature,
        pool: Pool,
        B: int = 500,
        rng: Optional[np.random.Generator] = None,
        metric=cskl,
) -> float:
    """
    Paper-style one-sided semi-parametric bootstrap p-value.

    The observed c-SKL(P, Q) is compared against two null distributions:
    c-SKL(P, J_mQ) and c-SKL(Q, J_mP), where J is the pooled background.
    The returned p-value is max(p12, p21), because the pair must be unusually
    similar relative to both null comparisons.

    This is more suitable for BH/FDR q-value calculation than the legacy R
    StatSig.R boolean threshold, which directly returned TRUE/FALSE using
    the 2.5% fitted-normal quantile.
    """
    if sigP.n_features != pool.n or sigQ.n_features != pool.n:
        raise ValueError("Signatures and pool must share the same number of features.")

    if pool.feature_names is not None:
        if sigP.feature_names is not None and list(sigP.feature_names) != list(pool.feature_names):
            raise ValueError("sigP and pool must use the same feature order.")
        if sigQ.feature_names is not None and list(sigQ.feature_names) != list(pool.feature_names):
            raise ValueError("sigQ and pool must use the same feature order.")

    if B < 2:
        raise ValueError("B must be at least 2 to fit a bootstrap null distribution.")

    if not (np.isclose(sigP.alpha, pool.alpha) and np.isclose(sigQ.alpha, pool.alpha)):
        raise ValueError("Signatures alpha must match pool alpha.")

    rng = np.random.default_rng(rng)
    c_pq = metric(sigP, sigQ)

    # 1. Retrieve exactly B pre-computed null signatures for sizes m_Q and m_P
    null_sigs_for_Q = pool.get_null_signatures(m=sigQ.m_samples, B=B, rng=rng)
    null_sigs_for_P = pool.get_null_signatures(m=sigP.m_samples, B=B, rng=rng)

    # 2. Compute the distances to the nulls (List comprehensions are faster than pre-allocating arrays here)
    # dist12: sigP vs random draws of size m_Q
    dist12 = np.array([metric(sigP, R_sizeQ) for R_sizeQ in null_sigs_for_Q])

    # dist21: sigQ vs random draws of size m_P
    dist21 = np.array([metric(sigQ, R_sizeP) for R_sizeP in null_sigs_for_P])

    # 3. Fit normal distributions (matching fitdist(..., "norm") in R)
    mu12, std12 = stats.norm.fit(dist12)
    mu21, std21 = stats.norm.fit(dist21)

    # Adding a tiny fallback for std to prevent division by zero in perfectly identical pools
    std12 = max(std12, 1e-12)
    std21 = max(std21, 1e-12)

    # 4. Calculate the CDF for the real c-SKL under both normal distributions
    p12 = stats.norm.cdf(c_pq, loc=mu12, scale=std12)
    p21 = stats.norm.cdf(c_pq, loc=mu21, scale=std21)

    # R code requires significance against BOTH null distributions.
    # Therefore, the effective p-value is the maximum (worst) of the two.
    return float(max(p12, p21))


def bh_qvalues(pvals: List[float]) -> List[float]:
    """
    Benjamini-Hochberg q-values (FDR-adjusted p-values).

    Parameters
    ----------
    pvals : list of floats

    Returns
    -------
    qvals : list of floats, same order as input
    """
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = np.array(pvals)[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q = p_sorted * (m / ranks)
    # monotone non-increasing when traversing from end
    q_rev = np.minimum.accumulate(q[::-1])[::-1]
    # clip to 1
    q_rev = np.clip(q_rev, 0.0, 1.0)
    # restore original order
    out = np.empty(m, dtype=float)
    out[order] = q_rev
    return out.tolist()


# -------------------------
# Network construction
# -------------------------

def build_dataset_network(
    signatures: Dict[str, PCASignature],
    pool: Pool,
    B: int = 500,
    fdr_alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[str, str, float, float, float]], List[Tuple[str, str, float]]]:
    """
    Build a dataset similarity network:
    - Compute c-SKL for all unordered pairs
    - Compute one-sided p-values via semi-parametric bootstrap
    - BH-correct to get q-values
    - Keep edges with q <= fdr_alpha

    Parameters
    ----------
    signatures : dict
        Mapping dataset_id -> PCASignature. All must share the same feature space and alpha.
    pool : Pool
        Pooled background on the same feature space.
    B : int
        Bootstrap iterations per pair.
    fdr_alpha : float
        FDR cutoff for keeping edges.
    rng : numpy Generator

    Returns
    -------
    all_pairs : list of (id1, id2, cskl, pval, qval)
    kept_edges : list of (id1, id2, cskl) for qval <= fdr_alpha
    """
    ids = list(signatures.keys())
    n = len(ids)
    if n < 2:
        return [], []

    rng = np.random.default_rng(rng)

    pairs = []
    pvals = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            sigA, sigB = signatures[a], signatures[b]
            c = cskl(sigA, sigB)
            p = pair_pvalue_vs_pool(sigA, sigB, pool, B=B, rng=rng)
            pairs.append((a, b, c, p))
            pvals.append(p)

    qvals = bh_qvalues(pvals)
    all_pairs = [(a, b, c, p, q) for (a, b, c, p), q in zip(pairs, qvals)]
    kept = [(a, b, c) for (a, b, c, p, q) in all_pairs if q <= fdr_alpha]
    return all_pairs, kept


# -------------------------
# Feature-level explanation B(k)
# -------------------------

def _scaled_loadings(sig: PCASignature) -> np.ndarray:
    """
    A = P * sqrt(lam)  (broadcast along columns).
    """
    return sig.P * np.sqrt(sig.lam)[None, :]


def _topk_indices(scores, k, largest=True):
    scores = np.asarray(scores)
    n = scores.shape[0]

    if k >= n:
        idx = np.arange(n)
    elif k <= 0:
        idx = np.array([], dtype=int)
    else:
        if largest:
            part = np.argpartition(-scores, k - 1)[:k]
            idx = part[np.argsort(-scores[part])]
        else:
            part = np.argpartition(scores, k - 1)[:k]
            idx = part[np.argsort(scores[part])]
    return idx

def _first_present(obj, keys):
    """Safely fetch the first non-None attribute/key from either dict or object."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
    else:
        for k in keys:
            if hasattr(obj, k) and getattr(obj, k) is not None:
                return getattr(obj, k)
    return None

def _get_sig_fields(sig):
    """
    Expect a signature-like input with at least:
      - P: (n_features, c) component/loadings matrix
      - lam: (c,) eigenvalues/weights for those components
    Optionally:
      - alpha (paper's 'a') and feature_names
    """
    P = _first_present(sig, ["P", "V", "components", "loadings"])
    lam = _first_present(sig, ["lam", "lambda", "lambda_", "eigvals", "eigs", "eigenvalues"])
    alpha = _first_present(sig, ["alpha", "a"])
    names = _first_present(sig, ["feature_names", "names", "columns", "cols"])

    if P is None or lam is None:
        raise ValueError("Signature must provide P (loadings/components) and lam (eigenvalues/weights).")

    P = np.asarray(P, dtype=float)
    lam = np.asarray(lam, dtype=float).reshape(-1)

    if P.ndim != 2:
        raise ValueError("P must be a 2D array of shape (n_features, n_components).")
    if lam.shape[0] != P.shape[1]:
        raise ValueError(f"lam must have length {P.shape[1]} to match P.shape[1]. Got {lam.shape[0]}.")

    if alpha is None:
        alpha = 0.5  # default paper-ish setting

    return P, lam, float(alpha), names

def explain_topk(sigP, sigQ, k, mode="B", *, max_iter=50, n_init=3, seed=0,
                return_scores=False, return_details=False):
    """
    Paper-faithful top-k explanation for compressed SKL feature attribution.

    Implements the paper’s objectives:

    B(k) ("Best"): minimize Eq.(2)  <=>  maximize
        f(S) = Σ_{i,j} (λ_i^P + λ_j^Q) * (P_i^T diag(S) Q_j)^2
    using the paper's biconvex relaxation Eq.(3) and alternating optimization.

    W(k) ("Worst"): maximize Eq.(2) <=> minimize f(S)

    Parameters
    ----------
    sigP, sigQ : signature objects (or dicts) containing at least P and lam
    k : int
    mode : "B" (default) or "W"
        "B" -> paper Best / most-similar explanation
        "W" -> paper Worst / most-dissimilar explanation
    max_iter : alternating steps per init
    n_init : random restarts (in addition to one deterministic init)
    seed : RNG seed for restarts
    return_scores : if True, also return per-feature linear scores for selected features
    return_details : if True, also return diagnostics (mask, objective, etc.)

    Returns
    -------
    By default:
      idx : (k,) numpy array of selected feature indices (0-based)

    If return_scores:
      (idx, scores)

    If return_details:
      (idx, scores, details)  OR (idx, details) depending on return_scores
    """
    P, lamP, alphaP, namesP = _get_sig_fields(sigP)
    Q, lamQ, alphaQ, namesQ = _get_sig_fields(sigQ)

    if not np.isclose(alphaP, alphaQ):
        raise ValueError("sigP and sigQ must use the same alpha.")

    if namesP is not None and namesQ is not None:
        if len(namesP) == P.shape[0] and len(namesQ) == Q.shape[0]:
            if list(namesP) != list(namesQ):
                raise ValueError("sigP and sigQ must use the same feature order.")

    if P.shape[0] != Q.shape[0]:
        raise ValueError(f"Signatures must have same n_features. Got {P.shape[0]} vs {Q.shape[0]}.")
    n = P.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be between 1 and n_features={n}. Got {k}.")

    mode_l = (mode or "B").strip().lower()
    want_best = mode_l in ("b", "best", "bk", "similar", "most_similar")
    want_worst = mode_l in ("w", "worst", "wk", "dissimilar", "most_dissimilar")
    if not (want_best or want_worst):
        raise ValueError("mode must be 'B'/'best' or 'W'/'worst'.")

    maximize_f = want_best

    # Paper coefficient matrix: C_ij = λ_i^P + λ_j^Q
    C = lamP[:, None] + lamQ[None, :]

    # True squared cross-term objective f(S) (the thing B(k) maximizes, W(k) minimizes)
    def f_of_mask(mask_bool):
        m = mask_bool.astype(float)
        PS = P * m[:, None]
        QS = Q * m[:, None]
        M = PS.T @ QS
        return float(np.sum(C * (M * M)))

    # Eq.(3) subproblem linear scores for selecting a new mask given the other fixed
    def linear_scores(other_mask_bool):
        m = other_mask_bool.astype(float)
        A = (P * m[:, None]).T @ (Q * m[:, None])   # a_ij(other)
        W = C * A
        scores = np.sum((P @ W) * Q, axis=1)        # score per feature
        return scores

    # Deterministic init (paper-friendly proxy):
    # w_r = (p_r^2)^T C (q_r^2)  (ignores cross-feature interaction, good start)
    tmp = (P * P) @ C
    w_init = np.sum(tmp * (Q * Q), axis=1)
    det_idx = _topk_indices(w_init, k, largest=maximize_f)
    det_mask = np.zeros(n, dtype=bool)
    det_mask[det_idx] = True

    rng = np.random.default_rng(seed)
    init_masks = [det_mask]
    for _ in range(max(0, int(n_init))):
        ridx = rng.choice(n, size=k, replace=False)
        m = np.zeros(n, dtype=bool)
        m[ridx] = True
        init_masks.append(m)

    best_mask = None
    best_f = -np.inf if maximize_f else np.inf

    for init in init_masks:
        S = init.copy()
        T = init.copy()

        local_best_mask = S.copy()
        local_best_f = f_of_mask(S)

        for _ in range(int(max_iter)):
            # Update T given S
            scT = linear_scores(S)
            T_idx = _topk_indices(scT, k, largest=maximize_f)
            T_new = np.zeros(n, dtype=bool)
            T_new[T_idx] = True

            # Update S given T
            scS = linear_scores(T_new)
            S_idx = _topk_indices(scS, k, largest=maximize_f)
            S_new = np.zeros(n, dtype=bool)
            S_new[S_idx] = True

            # Track best by true f(S)
            f_val = f_of_mask(S_new)
            if maximize_f:
                if f_val > local_best_f + 1e-12:
                    local_best_f = f_val
                    local_best_mask = S_new.copy()
            else:
                if f_val < local_best_f - 1e-12:
                    local_best_f = f_val
                    local_best_mask = S_new.copy()

            # Converged?
            if np.array_equal(S_new, S) and np.array_equal(T_new, T):
                S, T = S_new, T_new
                break
            S, T = S_new, T_new

        # Keep best across initializations
        if maximize_f:
            if local_best_f > best_f + 1e-12:
                best_f = local_best_f
                best_mask = local_best_mask.copy()
        else:
            if local_best_f < best_f - 1e-12:
                best_f = local_best_f
                best_mask = local_best_mask.copy()

    # Final ordering for reporting: use Eq.(3) linear scores at fixed point T=S=best_mask
    final_scores = linear_scores(best_mask)
    sel = np.where(best_mask)[0]
    order = np.argsort(-final_scores[sel]) if maximize_f else np.argsort(final_scores[sel])
    idx = sel[order]

    if not (return_scores or return_details):
        return idx

    out_scores = final_scores[idx]

    if not return_details:
        return idx, out_scores

    details = {
        "mode": "B" if maximize_f else "W",
        "k": int(k),
        "n_features": int(n),
        "f": float(best_f),          # the squared cross-term objective
        "mask": best_mask,
        "scores_full": final_scores,
    }
    if namesP is not None and len(namesP) == n:
        details["feature_names"] = [namesP[i] for i in idx]

    return (idx, out_scores, details) if return_scores else (idx, details)


def explain_set_topk(
    pairs: Iterable[Tuple[PCASignature, PCASignature]],
    k: int,
    iters: int = 5,
    *,
    mode: str = "B",
    max_iter: Optional[int] = None,
    n_init: int = 3,
    seed: Optional[int] = 0,
    return_scores: bool = False,
    return_details: bool = False,
    order: str = "index",
) -> object:
    """
    Explain a set of pairwise c-SKL similarities by selecting one common set
    of k features.

    This is the set-level analogue of explain_topk():

    B(k) / "Best":
        Select features that jointly maximize the c-SKL cross-term objective
        across all supplied pairs. These are the shared features that best
        explain why the pairs are similar.

    W(k) / "Worst":
        Select features that jointly minimize the same objective. These are
        features that contribute least to the shared similarity / most fail
        to align across the supplied pairs. Treat W(k) as a useful heuristic
        unless validated against exhaustive search for small n.

    Parameters
    ----------
    pairs : iterable of (sigP, sigQ)
        Signature pairs to explain. Signature-like dict/object inputs are
        accepted if they expose P and lam.
    k : int
        Number of features to select.
    iters : int, default=5
        Backward-compatible alias for max_iter. Existing calls using
        explain_set_topk(pairs, k, iters=...) still work.
    mode : {"B", "best", "W", "worst"}, default="B"
        Whether to maximize or minimize the aggregate explanation objective.
    max_iter : Optional[int]
        Alternating optimization iterations. If None, uses iters.
    n_init : int, default=3
        Number of random restarts in addition to the deterministic start.
    seed : Optional[int], default=0
        RNG seed for random restarts.
    return_scores : bool, default=False
        If True, return selected linear scores as well as indices.
    return_details : bool, default=False
        If True, return diagnostics including objective, mask, full scores.
    order : {"index", "score"}, default="index"
        "index" preserves old behavior: selected indices sorted ascending.
        "score" orders selected features by explanation score.

    Returns
    -------
    Default:
        idx : np.ndarray of shape (k,), sorted ascending by default.

    If return_scores:
        (idx, scores)

    If return_details:
        (idx, details) or (idx, scores, details)
    """
    pairs = list(pairs)
    if len(pairs) == 0:
        raise ValueError("Need at least one pair.")

    if max_iter is None:
        max_iter = iters

    max_iter = int(max_iter)
    n_init = int(n_init)

    if max_iter < 0:
        raise ValueError("max_iter/iters must be >= 0.")
    if n_init < 0:
        raise ValueError("n_init must be >= 0.")

    mode_l = (mode or "B").strip().lower()
    want_best = mode_l in ("b", "best", "bk", "similar", "most_similar")
    want_worst = mode_l in ("w", "worst", "wk", "dissimilar", "most_dissimilar")
    if not (want_best or want_worst):
        raise ValueError("mode must be 'B'/'best' or 'W'/'worst'.")

    maximize_f = want_best

    parsed_pairs = []
    n = None
    alpha0 = None
    feature_names = None

    for pair_idx, (sigP, sigQ) in enumerate(pairs):
        P, lamP, alphaP, namesP = _get_sig_fields(sigP)
        Q, lamQ, alphaQ, namesQ = _get_sig_fields(sigQ)

        if P.shape[0] != Q.shape[0]:
            raise ValueError(
                f"Pair {pair_idx} signatures must have same n_features. "
                f"Got {P.shape[0]} vs {Q.shape[0]}."
            )

        if not np.isclose(alphaP, alphaQ):
            raise ValueError(f"Pair {pair_idx} signatures must use the same alpha.")

        if n is None:
            n = P.shape[0]
            alpha0 = float(alphaP)
        else:
            if P.shape[0] != n:
                raise ValueError("All pairs must share the same feature space.")
            if not np.isclose(alphaP, alpha0):
                raise ValueError("All pairs must use the same alpha.")

        if namesP is not None and len(namesP) == P.shape[0]:
            namesP = list(namesP)
        else:
            namesP = None

        if namesQ is not None and len(namesQ) == Q.shape[0]:
            namesQ = list(namesQ)
        else:
            namesQ = None

        if namesP is not None and namesQ is not None and namesP != namesQ:
            raise ValueError(f"Pair {pair_idx} signatures must use the same feature order.")

        pair_names = namesP if namesP is not None else namesQ
        if pair_names is not None:
            if feature_names is None:
                feature_names = pair_names
            elif feature_names != pair_names:
                raise ValueError("All pairs must use the same feature order.")

        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)):
            raise ValueError(f"Pair {pair_idx} contains non-finite loadings.")
        if not np.all(np.isfinite(lamP)) or not np.all(np.isfinite(lamQ)):
            raise ValueError(f"Pair {pair_idx} contains non-finite eigenvalues.")
        if np.any(lamP < -1e-12) or np.any(lamQ < -1e-12):
            raise ValueError(f"Pair {pair_idx} contains negative eigenvalues.")

        lamP = np.maximum(lamP, 0.0)
        lamQ = np.maximum(lamQ, 0.0)

        A = P * np.sqrt(lamP)[None, :]
        B = Q * np.sqrt(lamQ)[None, :]

        parsed_pairs.append((P, Q, A, B))

    if n is None:
        raise ValueError("Need at least one valid pair.")
    if not (1 <= k <= n):
        raise ValueError(f"k must be between 1 and n_features={n}. Got {k}.")

    order_l = (order or "index").strip().lower()
    if order_l not in {"index", "score", "scores"}:
        raise ValueError("order must be 'index' or 'score'.")

    def mask_from_idx(idx_arr: np.ndarray) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        mask[idx_arr] = True
        return mask

    def objective(mask_bool: np.ndarray) -> float:
        """
        True aggregate squared cross-term objective:
            sum_pairs Σ_ij (lamP_i + lamQ_j) * (P_i^T diag(S) Q_j)^2

        Implemented as:
            ||A_S^T Q_S||_F^2 + ||P_S^T B_S||_F^2
        where A = P sqrt(lamP), B = Q sqrt(lamQ).
        """
        total = 0.0
        for P, Q, A, B in parsed_pairs:
            PS = P[mask_bool, :]
            QS = Q[mask_bool, :]
            AS = A[mask_bool, :]
            BS = B[mask_bool, :]

            M1 = AS.T @ QS
            M2 = PS.T @ BS

            total += float(np.sum(M1 * M1) + np.sum(M2 * M2))
        return total

    def linear_scores(other_mask_bool: np.ndarray) -> np.ndarray:
        """
        Linearized Eq.(3)-style feature scores for updating one selector
        while holding the other selector fixed, aggregated across all pairs.
        """
        scores = np.zeros(n, dtype=float)

        for P, Q, A, B in parsed_pairs:
            PT = P[other_mask_bool, :]
            QT = Q[other_mask_bool, :]
            AT = A[other_mask_bool, :]
            BT = B[other_mask_bool, :]

            M1 = AT.T @ QT
            M2 = PT.T @ BT

            scores += np.sum(A * (Q @ M1.T), axis=1)
            scores += np.sum(P * (B @ M2.T), axis=1)

        return scores

    def better(x: float, y: float) -> bool:
        return x > y + 1e-12 if maximize_f else x < y - 1e-12

    # Deterministic diagonal/self-feature proxy start.
    score0 = np.zeros(n, dtype=float)
    for P, Q, A, B in parsed_pairs:
        score0 += np.sum(A * A, axis=1) * np.sum(Q * Q, axis=1)
        score0 += np.sum(P * P, axis=1) * np.sum(B * B, axis=1)

    det_idx = _topk_indices(score0, k, largest=maximize_f)
    init_masks = [mask_from_idx(det_idx)]

    rng = np.random.default_rng(seed)
    for _ in range(n_init):
        ridx = rng.choice(n, size=k, replace=False)
        init_masks.append(mask_from_idx(ridx))

    best_mask = None
    best_f = -np.inf if maximize_f else np.inf
    best_converged = False
    best_n_iter = 0

    for init_mask in init_masks:
        S = init_mask.copy()
        T = init_mask.copy()

        local_best_mask = S.copy()
        local_best_f = objective(S)
        local_converged = False
        local_n_iter = 0

        for step in range(max_iter):
            local_n_iter = step + 1

            scT = linear_scores(S)
            T_new = mask_from_idx(_topk_indices(scT, k, largest=maximize_f))

            scS = linear_scores(T_new)
            S_new = mask_from_idx(_topk_indices(scS, k, largest=maximize_f))

            f_val = objective(S_new)
            if better(f_val, local_best_f):
                local_best_f = f_val
                local_best_mask = S_new.copy()

            if np.array_equal(S_new, S) and np.array_equal(T_new, T):
                local_converged = True
                S, T = S_new, T_new
                break

            S, T = S_new, T_new

        if better(local_best_f, best_f):
            best_f = local_best_f
            best_mask = local_best_mask.copy()
            best_converged = local_converged
            best_n_iter = local_n_iter

    final_scores = linear_scores(best_mask)
    idx = np.flatnonzero(best_mask)

    if order_l in {"score", "scores"}:
        if maximize_f:
            idx = idx[np.argsort(-final_scores[idx])]
        else:
            idx = idx[np.argsort(final_scores[idx])]
    else:
        idx.sort()

    if not (return_scores or return_details):
        return idx

    out_scores = final_scores[idx]

    if not return_details:
        return idx, out_scores

    details = {
        "mode": "B" if maximize_f else "W",
        "k": int(k),
        "n_features": int(n),
        "n_pairs": int(len(parsed_pairs)),
        "f": float(best_f),
        "objective": float(best_f),
        "mask": best_mask,
        "scores_full": final_scores,
        "initial_scores_full": score0,
        "max_iter": int(max_iter),
        "n_init": int(n_init),
        "converged": bool(best_converged),
        "n_iter": int(best_n_iter),
        "order": "score" if order_l in {"score", "scores"} else "index",
    }

    if feature_names is not None:
        details["feature_names"] = [feature_names[i] for i in idx]

    if return_scores:
        return idx, out_scores, details
    return idx, details


# -------------------------
# Tiny sanity demo (optional)
# -------------------------

if __name__ == "__main__":
    # Create a synthetic platform with n=200 features
    rng = np.random.default_rng(0)
    n = 200
    # Two related datasets share a 3-D subspace + noise
    m1, m2 = 60, 70
    ktrue = 3
    Utrue, _ = np.linalg.qr(rng.normal(size=(n, ktrue)))  # (n x ktrue)
    # Construct covariance with signal in Utrue, noise elsewhere
    lam_true = np.array([4.0, 2.0, 1.0])
    Sig = Utrue @ np.diag(lam_true) @ Utrue.T + np.eye(n)
    # Sample gaussian data
    X1 = rng.multivariate_normal(mean=np.zeros(n), cov=Sig, size=m1)
    X2 = rng.multivariate_normal(mean=np.zeros(n), cov=Sig, size=m2)
    Xpool = rng.multivariate_normal(mean=np.zeros(n), cov=np.eye(n), size=500)

    alpha = 0.5
    sig1 = fit_pca_signature(X1, alpha=alpha)
    sig2 = fit_pca_signature(X2, alpha=alpha)
    pool = Pool(Xpool, alpha=alpha)

    c = cskl(sig1, sig2)
    p = pair_pvalue_vs_pool(sig1, sig2, pool, B=100, rng=rng)
    print(f"c-SKL(sig1,sig2)={c:.6f}, p≈{p:.4f} (smaller is more similar)")

    # Explain with k=10 (R-style)
    idx = explain_topk(sig1, sig2, k=10, max_iter=5)
    print("Top-10 explanatory feature indices:", idx[:10])
