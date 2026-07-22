"""Align a loaded expression matrix to the frozen platform feature space.

The feature space is frozen to ``probes.txt`` (plan section 6). Every signature is
fit on exactly those probes, in that order. A dataset that is missing a probe gets
a fully-NaN row for it (handled by QC/imputation); extra probes are dropped.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def align_to_probes(
    df: pd.DataFrame,
    probes: List[str],
) -> Tuple[np.ndarray, dict]:
    """Reindex ``df`` (features x samples) onto ``probes`` and return ``X`` and info.

    Returns
    -------
    X : (m_samples, n_probes) float64
        Samples-by-features matrix in frozen probe order. Missing probes are NaN.
    info : dict
        {n_probes, n_present, n_missing, n_extra_dropped, n_samples}
    """
    df = df.copy()
    df.index = df.index.astype(str)
    # collapse accidental duplicate probe rows (keep first) to keep reindex 1:1
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    present_mask = df.index.isin(set(probes))
    n_extra = int((~present_mask).sum())

    aligned = df.reindex(probes)                      # features x samples, NaN for missing
    n_missing = int(aligned.isna().all(axis=1).sum())

    X = aligned.to_numpy(dtype=np.float64).T          # samples x features
    info = {
        "n_probes": len(probes),
        "n_present": len(probes) - n_missing,
        "n_missing": n_missing,
        "n_extra_dropped": n_extra,
        "n_samples": int(X.shape[0]),
    }
    return X, info
