"""Per-dataset QC + quarantine (plan section 6).

Turns "one broken matrix kills the week-long run" into "broken matrices are logged
and skipped". Policy:

* ``< 2`` samples                         -> quarantine ("too_few_samples")
* non-finite fraction above ``max_nonfinite_fraction`` (default 1%) -> quarantine
* whole matrix (near) zero variance       -> quarantine ("zero_variance")
* otherwise -> OK; a few stray NaNs / missing probes are imputed to the feature
  mean (all-NaN columns -> 0) and the count is recorded in ``qc.json``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QCResult:
    status: str                       # "ok" | "quarantine"
    reason: str
    n_samples: int
    finite_fraction: float
    n_missing_features: int
    n_imputed_entries: int
    X: Optional[np.ndarray] = field(default=None, repr=False)  # imputed, finite (if ok)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "n_samples": int(self.n_samples),
            "finite_fraction": float(self.finite_fraction),
            "n_missing_features": int(self.n_missing_features),
            "n_imputed_entries": int(self.n_imputed_entries),
        }


def run_qc(
    X: np.ndarray,
    *,
    max_nonfinite_fraction: float = 0.01,
    min_samples: int = 2,
    zero_var_tol: float = 1e-12,
) -> QCResult:
    """Assess an aligned samples-by-features matrix and impute if salvageable."""
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    finite = np.isfinite(X)
    finite_fraction = float(finite.mean()) if X.size else 0.0
    all_nan_cols = ~finite.any(axis=0)
    n_missing_features = int(all_nan_cols.sum())

    if m < min_samples:
        return QCResult("quarantine", f"too_few_samples({m})", m, finite_fraction,
                        n_missing_features, 0)

    nonfinite_fraction = 1.0 - finite_fraction
    if nonfinite_fraction > max_nonfinite_fraction:
        return QCResult(
            "quarantine",
            f"nonfinite_fraction({nonfinite_fraction:.4f}>{max_nonfinite_fraction})",
            m, finite_fraction, n_missing_features, 0,
        )

    # Impute: stray NaNs -> per-feature finite mean; all-NaN columns -> 0.
    n_bad = int((~finite).sum())
    if n_bad:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            col_mean = np.nanmean(np.where(finite, X, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        Ximp = np.where(finite, X, col_mean[None, :])
        Ximp = np.nan_to_num(Ximp, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        Ximp = X

    # Degenerate (whole matrix ~constant) -> nothing for PCA to latch onto.
    var = Ximp.var(axis=0, ddof=1) if m > 1 else np.zeros(n)
    if not np.any(var > zero_var_tol):
        return QCResult("quarantine", "zero_variance", m, finite_fraction,
                        n_missing_features, n_bad)

    return QCResult("ok", "ok", m, finite_fraction, n_missing_features, n_bad, X=Ximp)
