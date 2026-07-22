"""Fixed, versioned background pool (lever A) + transient null-bank generation.

``build_pool`` freezes two things for a platform release:

* ``probes.txt`` - the frozen feature order (THE feature space). Every signature is
  fit on exactly these probes, in this order.
* ``<pool_version>/`` - a frozen pool matrix + ``pool_meta.json`` (seed, grid,
  M_samples, source, feature hash, ...).

The pool makes the null model independent of the query set, which is what makes
updates cheap and gives stronger p-values. The *null bank* (B bootstrap signatures
per grid size) is NOT persisted - it is regenerated deterministically from
``(pool matrix, seed, size)`` when needed and freed (plan section 2/9).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from . import grid as grid_mod
from . import store as store_mod
from .align import align_to_probes
from .fastcore import SignatureBank, fast_fit_signature

DEFAULT_B = 100
DEFAULT_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _load_matrix(path: Path, orientation: str) -> pd.DataFrame:
    """Load a features-by-samples matrix (reusing the pipeline loader)."""
    from cskl_pipeline.io import load_expr

    return load_expr(path, orientation=orientation)


def build_pool(
    store: store_mod.Store,
    platform: str,
    *,
    pool_version: str = "pool_v1",
    from_matrix: Optional[str] = None,
    from_matrix_orientation: str = "features-by-samples",
    sample_from_corpus: bool = False,
    probes_from: Optional[str] = None,
    M_samples: int = 2000,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    B: int = DEFAULT_B,
    grid_min: Optional[int] = None,
    grid_max: Optional[int] = None,
    exact_size: bool = False,
    force: bool = False,
) -> dict:
    """Freeze ``probes.txt`` and ``<pool_version>/``. Idempotent unless ``force``.

    Exactly one pool source must be chosen:
      * ``from_matrix`` - a provided global pool matrix (features x samples).
      * ``sample_from_corpus`` - sample columns from already-ingested ``expr.tsv.gz``.
    """
    meta_path = store.pool_meta_path(platform, pool_version)
    if meta_path.exists() and not force:
        print(f"[build-pool] {platform}/{pool_version} already exists -> skipping "
              f"(use --force to rebuild).")
        return store_mod.read_json(meta_path)

    if bool(from_matrix) == bool(sample_from_corpus):
        raise ValueError("Choose exactly one pool source: --from-matrix OR --sample-from-corpus.")

    # ---- 1. Determine frozen probe order ---------------------------------
    probes: List[str]
    pool_df: Optional[pd.DataFrame] = None

    if from_matrix:
        pool_df = _load_matrix(Path(from_matrix), from_matrix_orientation)
        probes = [str(p) for p in pool_df.index]
    elif probes_from:
        ref_df = _load_matrix(Path(probes_from), "features-by-samples")
        probes = [str(p) for p in ref_df.index]
    else:
        # derive probe order from the first ingested dataset's expr matrix
        ds = store.list_datasets()
        ds = [d for d in ds if store.expr_path(d).exists()]
        if not ds:
            raise ValueError(
                "sample-from-corpus needs ingested datasets (expr.tsv.gz). "
                "Provide --probes-from a full-platform matrix, or ingest first."
            )
        ref_df = pd.read_csv(store.expr_path(ds[0]), sep="\t", index_col=0)
        probes = [str(p) for p in ref_df.index]

    # dedup, preserve order
    seen = set()
    probes = [p for p in probes if not (p in seen or seen.add(p))]
    store.write_probes(platform, probes)
    feature_hash = store_mod.feature_hash_from_probes(probes)
    n_features = len(probes)
    print(f"[build-pool] froze {n_features} probes -> {store.probes_path(platform)}")

    # ---- 2. Assemble the pool matrix (M x n_features) --------------------
    rng = np.random.default_rng(seed)
    if from_matrix:
        X_all, info = align_to_probes(pool_df, probes)  # samples x features
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        source = f"matrix:{Path(from_matrix).name}"
    else:
        cols = []
        ds = [d for d in store.list_datasets() if store.expr_path(d).exists()
              and not store.is_quarantined(d)]
        for d in ds:
            df = pd.read_csv(store.expr_path(d), sep="\t", index_col=0)
            Xd, _ = align_to_probes(df, probes)
            cols.append(np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0))
        X_all = np.vstack(cols)
        source = f"corpus:{len(ds)}datasets"

    M_total = X_all.shape[0]
    if M_total > M_samples:
        pick = rng.choice(M_total, size=M_samples, replace=False)
        pick.sort()
        X_pool = X_all[pick]
    else:
        X_pool = X_all
    X_pool = np.ascontiguousarray(X_pool, dtype=np.float32)

    store_mod.atomic_write_bytes(
        store.pool_matrix_path(platform, pool_version),
        _npy_bytes(X_pool),
    )
    matrix_sha1 = hashlib.sha1(X_pool.tobytes()).hexdigest()

    # ---- 3. Size grid ----------------------------------------------------
    # Prefer sizes from ingested signatures; fall back to expr.tsv.gz column
    # counts (so exact-size works even before any signature exists).
    sizes = _corpus_sample_sizes(store) or _expr_sample_counts(store)
    if exact_size:
        grid = grid_mod.exact_size_grid(sizes) if sizes else [X_pool.shape[0]]
    else:
        lo = grid_min if grid_min is not None else 4
        hi = grid_max if grid_max is not None else max(400, max(sizes) if sizes else 400)
        grid = grid_mod.build_size_grid(lo, hi)

    # ---- 4. Freeze meta --------------------------------------------------
    meta = {
        "platform": platform,
        "pool_version": pool_version,
        "seed": int(seed),
        "grid": [int(g) for g in grid],
        "M_samples": int(X_pool.shape[0]),
        "n_features": int(n_features),
        "alpha": float(alpha),
        "B": int(B),
        "source": source,
        "exact_size": bool(exact_size),
        "feature_hash": feature_hash,
        "matrix_sha1": matrix_sha1,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    store_mod.atomic_write_json(meta_path, meta)
    print(f"[build-pool] wrote pool {platform}/{pool_version}: "
          f"M={meta['M_samples']} n={n_features} grid={len(grid)} pts source={source}")
    return meta


def _npy_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _corpus_sample_sizes(store: store_mod.Store) -> List[int]:
    sizes = []
    for d in store.list_datasets():
        p = store.signature_path(d)
        if p.exists():
            try:
                sizes.append(store_mod.read_signature_meta(p)["m_samples"])
            except Exception:
                pass
    return sizes


def _corpus_max_size(store: store_mod.Store, default: int = 400) -> int:
    sizes = _corpus_sample_sizes(store)
    return max(sizes) if sizes else default


def _expr_sample_counts(store: store_mod.Store) -> List[int]:
    """Sample counts from expr.tsv.gz headers (cheap; works pre-signature)."""
    import gzip

    counts = []
    for d in store.list_datasets():
        p = store.expr_path(d)
        if not p.exists():
            continue
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n")
            counts.append(header.count("\t"))  # cols minus the index column
        except Exception:
            pass
    return counts


# ---------------------------------------------------------------------------
# Pool handle + transient null bank
# ---------------------------------------------------------------------------

class PoolHandle:
    """Loaded pool matrix + meta, with deterministic null-bank generation."""

    def __init__(self, store: store_mod.Store, platform: str, pool_version: str):
        self.store = store
        self.platform = platform
        self.pool_version = pool_version
        self.meta = store_mod.read_json(store.pool_meta_path(platform, pool_version))
        self.alpha = float(self.meta["alpha"])
        self.B = int(self.meta["B"])
        self.seed = int(self.meta["seed"])
        self.grid = [int(g) for g in self.meta["grid"]]
        self.feature_hash = str(self.meta["feature_hash"])
        self.pool_hash = store.pool_hash(platform, pool_version)
        self._X: Optional[np.ndarray] = None

    @property
    def X_pool(self) -> np.ndarray:
        if self._X is None:
            self._X = np.load(self.store.pool_matrix_path(self.platform, self.pool_version))
        return self._X

    def _size_rng(self, m: int) -> np.random.Generator:
        # Deterministic per (global seed, size): same machine + same seed -> identical.
        return np.random.default_rng([self.seed, int(m)])

    def null_bank(self, m: int, B: Optional[int] = None) -> SignatureBank:
        """Generate a transient bank of B null signatures at sample size ``m``.

        Deterministic from ``(pool matrix, seed, m)``. Peak memory is ~B x one
        signature; the caller is expected to build one size at a time and free it.
        """
        if B is None:
            B = self.B
        X = np.ascontiguousarray(self.X_pool, dtype=np.float64)
        M = X.shape[0]
        rng = self._size_rng(m)
        sigs = []
        for _ in range(B):
            idx = rng.integers(low=0, high=M, size=m, endpoint=False)
            Xb = X[idx, :]
            sigs.append(
                fast_fit_signature(Xb, alpha=self.alpha, feature_names=None,
                                   rng=rng, nan_policy="zero", r_compat_noise=True)
            )
        return SignatureBank(sigs)
