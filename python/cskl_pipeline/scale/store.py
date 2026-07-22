"""Artifact store: files ARE the state (no database).

Layout (see plan section 3)::

    <store>/
      platforms/<platform>/
        probes.txt                     # frozen feature order (THE feature space)
        <pool_version>/
          pool_meta.json               # {seed, grid, M_samples, source, created, ...}
          pool.npy                     # frozen pool matrix (M x n_features) float32
      datasets/<GSE>/
        signature.npz                  # P (float64), lam, n_features, m_samples, alpha, feature_hash
        null_profile__<pool_version>.npz
        sample_hashes.json
        qc.json
        expr.tsv.gz                    # normalized matrix (features x samples)
      quarantine/<GSE>/reason.txt
      runs/<run_id>/...

Every artifact is written to ``*.tmp`` then ``os.replace``d (atomic). Resume is
"skip a dataset whose signature + null profile already exist and validate".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from cskl import PCASignature


# ---------------------------------------------------------------------------
# Atomic primitives
# ---------------------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save an .npz atomically (numpy appends .npz to the tmp base name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = path.with_suffix("")  # numpy will add .npz
    tmp = Path(str(tmp_base) + ".tmp")
    np.savez(tmp, **arrays)
    produced = Path(str(tmp) + ".npz")
    os.replace(produced, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def feature_hash_from_probes(probes: List[str]) -> str:
    """Order-sensitive hash of the frozen feature list."""
    return hashlib.sha1(("\n".join(probes)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Signature IO
# ---------------------------------------------------------------------------

def save_signature(path: Path, sig: PCASignature, feature_hash: str) -> None:
    """Persist a signature. P is stored float64 to preserve <=1e-9 faithfulness.

    (The plan permits float32 for size, but float32 loadings would move cskl by
    ~1e-2 and break the faithfulness gate, so we keep float64 - ~5.7 MB/dataset.)
    """
    atomic_save_npz(
        path,
        P=np.ascontiguousarray(sig.P, dtype=np.float64),
        lam=np.asarray(sig.lam, dtype=np.float64),
        n_features=np.int64(sig.n_features),
        m_samples=np.int64(sig.m_samples),
        alpha=np.float64(sig.alpha),
        feature_hash=np.array(feature_hash),
    )


def load_signature(path: Path, feature_names: Optional[List[str]] = None) -> PCASignature:
    with np.load(path, allow_pickle=False) as z:
        P = np.asarray(z["P"], dtype=np.float64)
        lam = np.asarray(z["lam"], dtype=np.float64)
        n_features = int(z["n_features"])
        m_samples = int(z["m_samples"])
        alpha = float(z["alpha"])
    return PCASignature(
        P=P, lam=lam, n_features=n_features, m_samples=m_samples,
        alpha=alpha, feature_names=feature_names,
    )


def read_signature_feature_hash(path: Path) -> Optional[str]:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "feature_hash" in z:
                return str(z["feature_hash"])
    except Exception:
        return None
    return None


def read_signature_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {
            "n_features": int(z["n_features"]),
            "m_samples": int(z["m_samples"]),
            "alpha": float(z["alpha"]),
            "c_components": int(np.asarray(z["lam"]).shape[0]),
            "feature_hash": str(z["feature_hash"]) if "feature_hash" in z else None,
        }


# ---------------------------------------------------------------------------
# Null-profile IO
# ---------------------------------------------------------------------------

def save_null_profile(
    path: Path,
    grid: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    pool_version: str,
    pool_hash: str,
    feature_hash: str,
    mode: str,
    B: int,
) -> None:
    atomic_save_npz(
        path,
        grid=np.asarray(grid, dtype=np.int64),
        mu=np.asarray(mu, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        pool_version=np.array(pool_version),
        pool_hash=np.array(pool_hash),
        feature_hash=np.array(feature_hash),
        mode=np.array(mode),
        B=np.int64(B),
    )


@dataclass
class NullProfile:
    grid: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    pool_version: str
    pool_hash: str
    feature_hash: str
    mode: str
    B: int


def load_null_profile(path: Path) -> NullProfile:
    with np.load(path, allow_pickle=False) as z:
        return NullProfile(
            grid=np.asarray(z["grid"], dtype=np.int64),
            mu=np.asarray(z["mu"], dtype=np.float64),
            sigma=np.asarray(z["sigma"], dtype=np.float64),
            pool_version=str(z["pool_version"]),
            pool_hash=str(z["pool_hash"]),
            feature_hash=str(z["feature_hash"]),
            mode=str(z["mode"]) if "mode" in z else "grid",
            B=int(z["B"]) if "B" in z else -1,
        )


def read_null_profile_header(path: Path) -> Optional[dict]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {
                "pool_version": str(z["pool_version"]),
                "pool_hash": str(z["pool_hash"]),
                "feature_hash": str(z["feature_hash"]),
                "mode": str(z["mode"]) if "mode" in z else "grid",
                "B": int(z["B"]) if "B" in z else -1,
            }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Store object
# ---------------------------------------------------------------------------

class Store:
    """Handle to a ``<store>`` root directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    # -- directories -------------------------------------------------------
    def platform_dir(self, platform: str) -> Path:
        return self.root / "platforms" / platform

    def probes_path(self, platform: str) -> Path:
        return self.platform_dir(platform) / "probes.txt"

    def pool_dir(self, platform: str, pool_version: str) -> Path:
        return self.platform_dir(platform) / pool_version

    def pool_meta_path(self, platform: str, pool_version: str) -> Path:
        return self.pool_dir(platform, pool_version) / "pool_meta.json"

    def pool_matrix_path(self, platform: str, pool_version: str) -> Path:
        return self.pool_dir(platform, pool_version) / "pool.npy"

    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    def dataset_dir(self, gse: str) -> Path:
        return self.datasets_dir() / gse

    def signature_path(self, gse: str) -> Path:
        return self.dataset_dir(gse) / "signature.npz"

    def null_profile_path(self, gse: str, pool_version: str) -> Path:
        return self.dataset_dir(gse) / f"null_profile__{pool_version}.npz"

    def sample_hashes_path(self, gse: str) -> Path:
        return self.dataset_dir(gse) / "sample_hashes.json"

    def qc_path(self, gse: str) -> Path:
        return self.dataset_dir(gse) / "qc.json"

    def expr_path(self, gse: str) -> Path:
        return self.dataset_dir(gse) / "expr.tsv.gz"

    def quarantine_dir(self, gse: str) -> Path:
        return self.root / "quarantine" / gse

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    # -- probes / feature space -------------------------------------------
    def read_probes(self, platform: str) -> List[str]:
        path = self.probes_path(platform)
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen feature space not found: {path}. Run `build-pool` first."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh if line.strip()]

    def feature_hash(self, platform: str) -> str:
        return feature_hash_from_probes(self.read_probes(platform))

    def write_probes(self, platform: str, probes: List[str]) -> None:
        atomic_write_text(self.probes_path(platform), "\n".join(probes) + "\n")

    # -- pool --------------------------------------------------------------
    def pool_hash(self, platform: str, pool_version: str) -> str:
        """Stable identity of the pool for profile-validity checks."""
        meta = read_json(self.pool_meta_path(platform, pool_version))
        payload = {
            "seed": meta.get("seed"),
            "grid": meta.get("grid"),
            "M_samples": meta.get("M_samples"),
            "feature_hash": meta.get("feature_hash"),
            "matrix_sha1": meta.get("matrix_sha1"),
        }
        return sha1_text(json.dumps(payload, sort_keys=True))

    # -- listing -----------------------------------------------------------
    def list_datasets(self) -> List[str]:
        d = self.datasets_dir()
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def is_quarantined(self, gse: str) -> bool:
        return (self.quarantine_dir(gse) / "reason.txt").exists()

    def has_valid_signature(self, gse: str, platform: str) -> bool:
        path = self.signature_path(gse)
        if not path.exists():
            return False
        fh = read_signature_feature_hash(path)
        return fh == self.feature_hash(platform)

    def has_valid_profile(self, gse: str, platform: str, pool_version: str) -> bool:
        path = self.null_profile_path(gse, pool_version)
        if not path.exists():
            return False
        header = read_null_profile_header(path)
        if header is None:
            return False
        return (
            header["pool_version"] == pool_version
            and header["feature_hash"] == self.feature_hash(platform)
            and header["pool_hash"] == self.pool_hash(platform, pool_version)
        )
