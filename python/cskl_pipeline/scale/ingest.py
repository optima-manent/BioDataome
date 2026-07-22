"""Resumable, parallel per-dataset ingest (plan section 4.2 / 5 / 6).

Per dataset, resumable & idempotent:
  1. normalize CEL -> ``expr.tsv.gz`` (reuse ``normalize_affy_scan``; skip if present),
     or parse a preprocessed matrix once and persist it as ``expr.tsv.gz``,
  2. align to the frozen ``probes.txt`` and run QC (quarantine on failure),
  3. fast-fit the signature (lever C) -> ``signature.npz`` + ``sample_hashes.json`` + ``qc.json``.

Then a single-process profile sweep (lever B/D) fills any missing null profiles,
generating each grid size's shared null bank exactly once.

Parallelism is across datasets (this stage is R- and I/O-bound). The numerical fit
stays single-process per worker; BLAS threads are capped per worker so K workers do
not oversubscribe the machine.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cskl_pipeline.io import expand_paths, load_expr, safe_name_from_file
from cskl_pipeline.normalize import (
    ArchiveSafetyError,
    normalization_cache_matches,
    normalize_affy_scan,
    read_normalization_manifest,
    scan_log_diagnostics,
)

from . import profile as profile_mod
from . import qc as qc_mod
from . import store as store_mod
from .align import align_to_probes
from .fastcore import fast_fit_signature
from .pool import PoolHandle


# ---------------------------------------------------------------------------
# Per-dataset worker
# ---------------------------------------------------------------------------

@dataclass
class IngestItem:
    gse: str
    source: str          # path to tar or csv
    kind: str            # "tar" | "csv"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_ingest_cache_matches(
    store: store_mod.Store,
    platform: str,
    item: IngestItem,
    *,
    alpha: float,
) -> bool:
    """Bind a reusable signature to the exact normalized RAW revision."""

    if not store.has_valid_signature(item.gse, platform):
        return False
    if item.kind != "tar":
        return True
    source = Path(item.source)
    if not normalization_cache_matches(source, store.datasets_dir()):
        return False
    normalization = read_normalization_manifest(store.datasets_dir(), source)
    manifest_path = store.dataset_dir(item.gse) / "ingest-manifest.json"
    try:
        manifest = store_mod.read_json(manifest_path)
    except (OSError, ValueError, TypeError):
        return False
    signature_path = store.signature_path(item.gse)
    samples_path = store.sample_hashes_path(item.gse)
    qc_path = store.qc_path(item.gse)
    if not samples_path.is_file() or not qc_path.is_file():
        return False
    return bool(
        normalization
        and manifest.get("schema") == "cskl-scale-ingest-v2"
        and manifest.get("source_sha256") == normalization.get("source_sha256")
        and manifest.get("normalized_sha256") == normalization.get("normalized_sha256")
        and manifest.get("feature_hash") == store.feature_hash(platform)
        and manifest.get("alpha") == float(alpha)
        and manifest.get("signature_sha256") == _sha256_file(signature_path)
        and manifest.get("sample_hashes_sha256") == _sha256_file(samples_path)
        and manifest.get("qc_sha256") == _sha256_file(qc_path)
    )


def _dataset_seed(global_seed: int, gse: str) -> np.random.Generator:
    """Stable per-dataset RNG (order-independent, resume-safe)."""
    h = int.from_bytes(hashlib.sha1(gse.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng([int(global_seed), h])


def _sample_hashes(X_imputed: np.ndarray, decimals: int = 10) -> List[str]:
    """Per-sample column hash (for the shared-profile filter)."""
    Xr = np.round(X_imputed, decimals=decimals)
    hashes = []
    for i in range(Xr.shape[0]):                     # X is samples x features
        col = np.ascontiguousarray(Xr[i, :], dtype=np.float64)
        hashes.append(hashlib.sha1(col.view(np.uint8)).hexdigest())
    return hashes


def _limit_blas(n_threads: int):
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=max(1, int(n_threads)))
    except Exception:
        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Null()


def ingest_one(
    store: store_mod.Store,
    platform: str,
    item: IngestItem,
    *,
    alpha: float,
    input_orientation: str = "features-by-samples",
    rscript: str = "Rscript",
    seed: int = 0,
    max_nonfinite_fraction: float = 0.01,
    blas_threads: int = 0,
    keep_raw: bool = False,
    persist_expr: bool = True,
) -> dict:
    """Ingest one dataset. Returns a status dict. Never raises for data problems -
    it quarantines instead so a bad matrix cannot kill the run."""
    gse = item.gse
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,127}", gse) or gse in {".", ".."}:
        raise ValueError(f"unsafe dataset identifier: {gse!r}")
    ddir = store.dataset_dir(gse)
    ddir.mkdir(parents=True, exist_ok=True)
    probes = store.read_probes(platform)
    feature_hash = store.feature_hash(platform)

    # Resume: valid signature already present.
    if _raw_ingest_cache_matches(store, platform, item, alpha=alpha):
        return {"gse": gse, "status": "skip", "reason": "signature_exists"}
    if store.is_quarantined(gse):
        return {
            "gse": gse,
            "status": "quarantine",
            "reason": "already_quarantined",
            "operator_required": True,
            "recovery": "Review reason.txt, correct the source/problem, then clear quarantine.",
        }

    # 1. Get expr.tsv.gz (features x samples).
    expr_path = store.expr_path(gse)
    try:
        if item.kind == "tar":
            produced = normalize_affy_scan(Path(item.source), store.datasets_dir(), rscript)
            if Path(produced) != expr_path and not expr_path.exists():
                shutil.copy2(produced, expr_path)
            if not keep_raw:
                shutil.rmtree(ddir / "raw_extracted", ignore_errors=True)
            df = pd.read_csv(expr_path, sep="\t", index_col=0)
        else:
            df = load_expr(Path(item.source), orientation=input_orientation)
            # persist parsed matrix once so we never re-parse. gzip level 1: for the
            # big GPL570 matrices, level-9 is ~5x slower for ~20% smaller files. When
            # the source stays available (e.g. a zip), pass persist_expr=False to skip
            # this entirely - it is not needed for fitting, profiling, or assembly.
            if persist_expr and not expr_path.exists():
                tmp = expr_path.with_suffix(expr_path.suffix + ".tmp")
                df.to_csv(tmp, sep="\t",
                          compression={"method": "gzip", "compresslevel": 1})
                os.replace(tmp, expr_path)
    except ArchiveSafetyError as exc:
        detail = f"archive_rejected: {exc}"
        _quarantine(store, gse, detail)
        return {
            "gse": gse,
            "status": "quarantine",
            "reason": "archive_rejected",
            "operator_required": True,
            "detail": str(exc),
            "recovery": "Inspect or replace the RAW archive, then clear this dataset quarantine.",
        }
    except Exception as exc:  # normalization / parse failure -> quarantine
        _quarantine(store, gse, f"load_failed: {type(exc).__name__}: {exc}")
        return {
            "gse": gse,
            "status": "quarantine",
            "reason": "load_failed",
            "operator_required": True,
            "detail": f"{type(exc).__name__}: {exc}",
            "recovery": "Inspect the normalization log and source archive, then clear quarantine.",
        }

    # 2. Align + QC.
    X, info = align_to_probes(df, probes)
    qc_res = qc_mod.run_qc(X, max_nonfinite_fraction=max_nonfinite_fraction)
    qc_dict = qc_res.to_dict()
    qc_dict.update({"align": info, "platform": platform})
    if item.kind == "tar":
        qc_dict["normalization"] = scan_log_diagnostics(
            ddir / "scan_normalize_affy.log"
        )

    if qc_res.status != "ok":
        store_mod.atomic_write_json(store.qc_path(gse), qc_dict)
        _quarantine(store, gse, qc_res.reason)
        return {
            "gse": gse,
            "status": "quarantine",
            "reason": qc_res.reason,
            "operator_required": True,
            "recovery": "Review qc.json before excluding the cohort or clearing quarantine.",
        }

    # 3. Fast-fit signature (BLAS threads capped to avoid worker oversubscription).
    Ximp = qc_res.X
    try:
        with _limit_blas(blas_threads if blas_threads else os.cpu_count() or 1):
            sig = fast_fit_signature(
                Ximp, alpha=alpha, feature_names=None, rng=_dataset_seed(seed, gse),
                nan_policy="zero", r_compat_noise=True,
            )
    except Exception as exc:
        _quarantine(store, gse, f"fit_failed: {type(exc).__name__}: {exc}")
        return {
            "gse": gse,
            "status": "quarantine",
            "reason": "fit_failed",
            "operator_required": True,
            "detail": f"{type(exc).__name__}: {exc}",
            "recovery": "Inspect qc.json and the worker log, then clear quarantine to retry.",
        }

    # A refit invalidates every profile tied to the previous signature. Remove
    # those derived intermediates before atomically publishing the new signature;
    # the resumable profile sweep below recreates them from the new science state.
    for profile_path in ddir.glob("null_profile__*.npz"):
        profile_path.unlink()
    store_mod.save_signature(store.signature_path(gse), sig, feature_hash)
    store_mod.atomic_write_json(
        store.sample_hashes_path(gse),
        {"n_samples": int(Ximp.shape[0]), "hashes": _sample_hashes(Ximp)},
    )
    qc_dict.update({"c_components": int(len(sig.lam)), "m_samples": int(sig.m_samples)})
    store_mod.atomic_write_json(store.qc_path(gse), qc_dict)
    if item.kind == "tar":
        normalization = read_normalization_manifest(store.datasets_dir(), Path(item.source))
        if not normalization:
            raise RuntimeError("normalization provenance was not recorded")
        store_mod.atomic_write_json(
            ddir / "ingest-manifest.json",
            {
                "schema": "cskl-scale-ingest-v2",
                "source_sha256": normalization["source_sha256"],
                "normalized_sha256": normalization["normalized_sha256"],
                "normalization_script_sha256": normalization[
                    "normalization_script_sha256"
                ],
                "feature_hash": feature_hash,
                "alpha": float(alpha),
                "signature_sha256": _sha256_file(store.signature_path(gse)),
                "sample_hashes_sha256": _sha256_file(store.sample_hashes_path(gse)),
                "qc_sha256": _sha256_file(store.qc_path(gse)),
            },
        )
    return {"gse": gse, "status": "ok", "m_samples": int(sig.m_samples),
            "c_components": int(len(sig.lam))}


def _quarantine(store: store_mod.Store, gse: str, reason: str) -> None:
    qdir = store.quarantine_dir(gse)
    store_mod.atomic_write_text(qdir / "reason.txt", reason + "\n")


# top-level worker (picklable for ProcessPool)
def _worker(payload: Tuple) -> dict:
    (root, platform, gse, source, kind, alpha, orientation, rscript, seed,
     max_nonfinite, blas_threads, keep_raw) = payload
    try:
        store = store_mod.Store(root)
        item = IngestItem(gse=gse, source=source, kind=kind)
        return ingest_one(
            store, platform, item, alpha=alpha, input_orientation=orientation,
            rscript=rscript, seed=seed, max_nonfinite_fraction=max_nonfinite,
            blas_threads=blas_threads, keep_raw=keep_raw,
        )
    except Exception as exc:  # never let one dataset kill the whole pool
        try:
            _quarantine(store_mod.Store(root), gse, f"worker_error: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return {"gse": gse, "status": "quarantine", "reason": "worker_error"}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _collect_items(tars: Sequence[str], csvs: Sequence[str]) -> List[IngestItem]:
    items: List[IngestItem] = []
    seen = set()
    for src in expand_paths(list(tars)):
        gse = safe_name_from_file(Path(src))
        if gse not in seen:
            seen.add(gse)
            items.append(IngestItem(gse, str(Path(src).resolve()), "tar"))
    for src in expand_paths(list(csvs)):
        gse = safe_name_from_file(Path(src))
        if gse not in seen:
            seen.add(gse)
            items.append(IngestItem(gse, str(Path(src).resolve()), "csv"))
    return items


def run_ingest(
    store: store_mod.Store,
    platform: str,
    pool_version: str,
    *,
    tars: Sequence[str] = (),
    csvs: Sequence[str] = (),
    workers: int = 1,
    input_orientation: str = "features-by-samples",
    rscript: str = "Rscript",
    max_nonfinite_fraction: float = 0.01,
    keep_raw: bool = False,
    compute_profiles: bool = True,
) -> dict:
    """Ingest all inputs (skipping valid ones), then fill missing null profiles."""
    pool = PoolHandle(store, platform, pool_version)
    alpha = pool.alpha
    seed = pool.seed
    items = _collect_items(tars, csvs)

    todo = [
        it
        for it in items
        if not _raw_ingest_cache_matches(store, platform, it, alpha=alpha)
        and not store.is_quarantined(it.gse)
    ]
    already = len(items) - len(todo)
    print(f"[ingest] {len(items)} inputs: {already} already valid, {len(todo)} to (re)fit; "
          f"workers={workers}")

    per_worker_threads = max(1, (os.cpu_count() or 1) // max(1, workers))
    results = []

    if workers <= 1:
        for it in todo:
            r = ingest_one(store, platform, it, alpha=alpha,
                           input_orientation=input_orientation, rscript=rscript,
                           seed=seed, max_nonfinite_fraction=max_nonfinite_fraction,
                           blas_threads=per_worker_threads, keep_raw=keep_raw)
            results.append(r)
            print(f"  [{r['gse']}] {r['status']}"
                  + (f" ({r['reason']})" if r.get("reason") else ""))
    else:
        payloads = [
            (str(store.root), platform, it.gse, it.source, it.kind, alpha,
             input_orientation, rscript, seed, max_nonfinite_fraction,
             per_worker_threads, keep_raw)
            for it in todo
        ]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, p): p[2] for p in payloads}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(f"  [{r['gse']}] {r['status']}"
                      + (f" ({r['reason']})" if r.get("reason") else ""))

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_q = sum(1 for r in results if r["status"] == "quarantine")
    preexisting_quarantines = sorted(
        item.gse for item in items if store.is_quarantined(item.gse)
    )
    operator_actions = [
        {
            "gse": result["gse"],
            "reason": result.get("reason", "quarantined"),
            "recovery": result.get("recovery", "Review the quarantine record."),
        }
        for result in results
        if result.get("operator_required")
    ]
    known_action_ids = {item["gse"] for item in operator_actions}
    operator_actions.extend(
        {
            "gse": gse,
            "reason": "already_quarantined",
            "recovery": "Review reason.txt, correct the source/problem, then clear quarantine.",
        }
        for gse in preexisting_quarantines
        if gse not in known_action_ids
    )
    summary = {"inputs": len(items), "fit_ok": n_ok, "quarantined": n_q,
               "skipped": already, "operator_required": bool(operator_actions),
               "operator_actions": operator_actions, "results": results}

    # Profile sweep: everything with a valid signature but no valid profile.
    if compute_profiles:
        need = [g for g in store.list_datasets()
                if store.has_valid_signature(g, platform)
                and not store.is_quarantined(g)
                and not store.has_valid_profile(g, platform, pool_version)]
        if need:
            print(f"[ingest] computing null profiles for {len(need)} dataset(s)...")
            written = profile_mod.compute_profiles(store, pool, need)
            summary["profiles_written"] = written
        else:
            summary["profiles_written"] = 0
            print("[ingest] all null profiles already present.")

    return summary
