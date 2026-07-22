"""Leased, resumable scientific job runner for the Atlas control plane."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .catalog import Catalog, canonical_json, stable_id
from .incremental import DatasetSignature, iter_cross_cskl_pairs, iter_incremental_cskl_pairs

LOGGER = logging.getLogger(__name__)


class PermanentJobError(ValueError):
    """The frozen job input is invalid and retries cannot repair it."""


JobHandler = Callable[[Catalog, Mapping[str, Any], str, str], Mapping[str, Any]]


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signature_record(catalog: Catalog, version_id: str) -> dict[str, Any]:
    with catalog.reader() as connection:
        row = connection.execute(
            """SELECT v.version_id,v.feature_hash,v.config_hash,v.signature_hash,a.uri,a.checksum
               FROM dataset_versions v JOIN artifacts a
                 ON a.dataset_version_id=v.version_id AND a.kind='pca_signature'
               WHERE v.version_id=? AND a.checksum=v.signature_hash
               ORDER BY a.created_at DESC LIMIT 1""",
            (version_id,),
        ).fetchone()
    if not row:
        raise PermanentJobError(
            f"Dataset version {version_id!r} has no checksum-matching PCA signature artifact."
        )
    return dict(row)


def _load_signature(record: Mapping[str, Any]) -> DatasetSignature:
    uri = str(record["uri"])
    if "://" in uri:
        raise PermanentJobError(
            "The reference worker accepts local PCA artifact paths only; configure an object-store worker for URI artifacts."
        )
    path = Path(uri).resolve()
    if not path.is_file():
        raise PermanentJobError(f"PCA signature artifact is missing: {path}")
    if _sha256_file(path) != record["checksum"]:
        raise PermanentJobError(f"PCA signature checksum mismatch: {path}")
    from cskl_pipeline.scale.store import load_signature

    return DatasetSignature(
        dataset_id=str(record["version_id"]),
        feature_universe=str(record["feature_hash"]),
        signature=load_signature(path),
    )


def enqueue_incremental_score_job(
    catalog: Catalog,
    *,
    new_version_ids: Sequence[str],
    algorithm_hash: str,
    max_attempts: int = 5,
) -> str:
    """Freeze the exact K new and N existing version families into one job."""

    new_ids = tuple(sorted({str(value).strip() for value in new_version_ids}))
    if not new_ids or any(not value for value in new_ids):
        raise ValueError("At least one non-empty new dataset version is required.")
    if len(new_ids) != len(new_version_ids):
        raise ValueError("new_version_ids must be unique")
    if not algorithm_hash.strip():
        raise ValueError("algorithm_hash is required")
    placeholders = ",".join("?" for _ in new_ids)
    with catalog.reader() as connection:
        rows = connection.execute(
            f"""SELECT version_id,feature_hash,config_hash FROM dataset_versions
                WHERE version_id IN ({placeholders}) ORDER BY version_id""",
            new_ids,
        ).fetchall()
        if len(rows) != len(new_ids):
            raise ValueError("One or more new dataset versions are unknown.")
        strata = {(row["feature_hash"], row["config_hash"]) for row in rows}
        if len(strata) != 1:
            raise ValueError("All new dataset versions must share one feature/config stratum.")
        feature_hash, config_hash = next(iter(strata))
        existing = [
            row["version_id"]
            for row in connection.execute(
                f"""SELECT v.version_id FROM dataset_versions v
                    JOIN datasets d ON d.current_version_id=v.version_id
                    WHERE v.feature_hash=? AND v.config_hash=?
                      AND v.version_id NOT IN ({placeholders}) ORDER BY v.version_id""",
                (feature_hash, config_hash, *new_ids),
            )
        ]
    payload = {
        "algorithm_hash": algorithm_hash.strip(),
        "feature_hash": feature_hash,
        "config_hash": config_hash,
        "new_version_ids": list(new_ids),
        "existing_version_ids": existing,
    }
    fingerprint = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return catalog.enqueue_job(
        kind="score_incremental",
        job_key=stable_id("score_delta", *new_ids),
        input_fingerprint=fingerprint,
        payload=payload,
        max_attempts=max_attempts,
    )


def handle_incremental_score(
    catalog: Catalog,
    payload: Mapping[str, Any],
    job_id: str,
    worker_id: str,
) -> Mapping[str, Any]:
    """Persist exactly KxN + K(K-1)/2 raw facts with a durable N cursor."""

    algorithm_hash = str(payload.get("algorithm_hash") or "").strip()
    new_ids = payload.get("new_version_ids")
    existing_ids = payload.get("existing_version_ids")
    if (
        not algorithm_hash
        or not isinstance(new_ids, list)
        or not isinstance(existing_ids, list)
        or not new_ids
        or len(new_ids) != len(set(new_ids))
        or len(existing_ids) != len(set(existing_ids))
        or set(new_ids) & set(existing_ids)
    ):
        raise PermanentJobError("Invalid frozen score_incremental payload.")
    if new_ids != sorted(new_ids) or existing_ids != sorted(existing_ids):
        raise PermanentJobError("Frozen score families must be sorted.")

    new_signatures = tuple(_load_signature(_signature_record(catalog, value)) for value in new_ids)
    job = catalog.get_job(job_id)
    progress = (job or {}).get("progress", {})
    last_existing = progress.get("last_existing_version_id")
    if last_existing is not None and last_existing not in existing_ids:
        raise PermanentJobError("Job resume cursor is not part of the frozen existing family.")

    inserted = 0
    completed_existing = int(progress.get("existing_completed", 0))
    start = existing_ids.index(last_existing) + 1 if last_existing is not None else 0
    for index, existing_id in enumerate(existing_ids[start:], start=start):
        existing = _load_signature(_signature_record(catalog, existing_id))
        pairs = iter_cross_cskl_pairs(new_signatures, [existing])
        inserted += catalog.record_pair_scores(
            (pair.dataset_a, pair.dataset_b, algorithm_hash, pair.cskl) for pair in pairs
        )
        completed_existing = index + 1
        catalog.update_job_progress(
            job_id,
            worker_id=worker_id,
            progress={
                "existing_completed": completed_existing,
                "existing_total": len(existing_ids),
                "last_existing_version_id": existing_id,
                "new_new_complete": False,
            },
        )
        catalog.heartbeat_job(job_id, worker_id=worker_id)

    if not progress.get("new_new_complete"):
        new_pairs = iter_incremental_cskl_pairs(new_signatures, (), include_new_new=True)
        inserted += catalog.record_pair_scores(
            (pair.dataset_a, pair.dataset_b, algorithm_hash, pair.cskl) for pair in new_pairs
        )
    final_progress = {
        "existing_completed": len(existing_ids),
        "existing_total": len(existing_ids),
        "last_existing_version_id": existing_ids[-1] if existing_ids else None,
        "new_new_complete": True,
    }
    catalog.update_job_progress(job_id, worker_id=worker_id, progress=final_progress)
    return {
        "inserted_pair_facts": inserted,
        "expected_pair_facts": len(new_ids) * len(existing_ids) + len(new_ids) * (len(new_ids) - 1) // 2,
        **final_progress,
    }


def enqueue_calibration_job(
    catalog: Catalog,
    *,
    calibration_id: str,
    profile_kind: str,
    max_attempts: int = 5,
) -> str:
    if not profile_kind.startswith("null_profile:"):
        raise ValueError("profile_kind must name a versioned null_profile artifact")
    with catalog.reader() as connection:
        release = connection.execute(
            "SELECT * FROM calibration_releases WHERE calibration_id=?", (calibration_id,)
        ).fetchone()
    if not release or release["status"] != "staging":
        raise ValueError("Calibration job requires a staging release.")
    payload = {
        "calibration_id": calibration_id,
        "profile_kind": profile_kind,
        "mode": release["mode"],
        "algorithm_hash": release["algorithm_hash"],
        "family_hash": release["family_hash"],
        "expected_pair_count": release["expected_pair_count"],
        "pool_hash": release["pool_hash"],
    }
    fingerprint = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return catalog.enqueue_job(
        kind="calibrate_release",
        job_key=calibration_id,
        input_fingerprint=fingerprint,
        payload=payload,
        max_attempts=max_attempts,
    )


def _load_profile_for_version(
    catalog: Catalog,
    *,
    version_id: str,
    profile_kind: str,
    expected_pool_hash: str,
):
    with catalog.reader() as connection:
        artifact = connection.execute(
            """SELECT a.uri,a.checksum,a.manifest_json,v.feature_hash
               FROM artifacts a JOIN dataset_versions v
                 ON v.version_id=a.dataset_version_id
               WHERE a.dataset_version_id=? AND a.kind=?
               ORDER BY a.created_at DESC LIMIT 1""",
            (version_id, profile_kind),
        ).fetchone()
    if not artifact:
        raise PermanentJobError(
            f"Dataset version {version_id!r} lacks required {profile_kind!r} artifact."
        )
    manifest = json.loads(artifact["manifest_json"])
    if manifest.get("pool_hash") != expected_pool_hash:
        raise PermanentJobError(f"Null profile pool hash mismatch for {version_id!r}.")
    uri = str(artifact["uri"])
    if "://" in uri:
        raise PermanentJobError("The reference worker accepts local null-profile paths only.")
    path = Path(uri).resolve()
    if not path.is_file() or _sha256_file(path) != artifact["checksum"]:
        raise PermanentJobError(f"Null profile is missing or corrupt for {version_id!r}.")
    from cskl_pipeline.scale.store import load_null_profile

    profile = load_null_profile(path)
    if profile.pool_hash != expected_pool_hash:
        raise PermanentJobError(f"Null profile header pool hash mismatch for {version_id!r}.")
    if profile.feature_hash != artifact["feature_hash"]:
        raise PermanentJobError(f"Null profile feature hash mismatch for {version_id!r}.")
    return profile


def handle_calibrate_release(
    catalog: Catalog,
    payload: Mapping[str, Any],
    job_id: str,
    worker_id: str,
) -> Mapping[str, Any]:
    calibration_id = str(payload.get("calibration_id") or "")
    profile_kind = str(payload.get("profile_kind") or "")
    expected_pool_hash = str(payload.get("pool_hash") or "")
    with catalog.reader() as connection:
        release = connection.execute(
            "SELECT * FROM calibration_releases WHERE calibration_id=?", (calibration_id,)
        ).fetchone()
    if not release:
        raise PermanentJobError("Unknown calibration release.")
    if release["status"] in {"calibrated", "published"}:
        return {"calibration_id": calibration_id, "already_finalized": True}
    if release["status"] != "staging":
        raise PermanentJobError(f"Calibration release is {release['status']!r}, not staging.")
    frozen = {
        "mode": release["mode"],
        "algorithm_hash": release["algorithm_hash"],
        "family_hash": release["family_hash"],
        "expected_pair_count": release["expected_pair_count"],
        "pool_hash": release["pool_hash"],
    }
    if any(
        payload.get(key) != value
        for key, value in frozen.items()
        if not (key == "mode" and payload.get(key) is None)
    ):
        raise PermanentJobError("Calibration job payload differs from its frozen release contract.")
    if not profile_kind.startswith("null_profile:"):
        raise PermanentJobError("Calibration job has an invalid null-profile kind.")

    from .calibration import (
        OutOfCalibrationRange,
        pair_pvalue_from_stored_profiles,
        validated_stored_profile_grid,
    )

    profiles: dict[str, Any] = {}
    written = 0
    clamped_pairs_this_attempt = 0
    clamped_lookups_this_attempt = 0
    for batch in catalog.iter_uncalibrated_pair_scores(calibration_id, batch_size=2_000):
        pvalues: list[tuple[str, float]] = []
        for pair in batch:
            for version_id in (pair["version_a"], pair["version_b"]):
                if version_id not in profiles:
                    profiles[version_id] = _load_profile_for_version(
                        catalog,
                        version_id=version_id,
                        profile_kind=profile_kind,
                        expected_pool_hash=expected_pool_hash,
                    )
            try:
                p_value, clamped_lookups = pair_pvalue_from_stored_profiles(
                    pair["cskl"],
                    profiles[pair["version_a"]],
                    pair["samples_b"],
                    profiles[pair["version_b"]],
                    pair["samples_a"],
                    expected_pool_hash=expected_pool_hash,
                    allow_clamp=release["mode"] == "frozen",
                )
            except OutOfCalibrationRange as exc:
                raise PermanentJobError(
                    "Exact calibration encountered a dataset outside the null-profile grid: "
                    f"{exc}. Extend/rebuild the grid or explicitly stage a frozen release."
                ) from exc
            clamped_pairs_this_attempt += int(clamped_lookups > 0)
            clamped_lookups_this_attempt += clamped_lookups
            pvalues.append((pair["pair_id"], p_value))
        written += catalog.record_pvalues(calibration_id, pvalues)
        catalog.update_job_progress(
            job_id,
            worker_id=worker_id,
            progress={
                "pvalues_written_this_attempt": written,
                "expected": release["expected_pair_count"],
                "clamped_pairs_this_attempt": clamped_pairs_this_attempt,
                "clamped_profile_lookups_this_attempt": clamped_lookups_this_attempt,
            },
        )
        catalog.heartbeat_job(job_id, worker_id=worker_id)

    with catalog.reader() as connection:
        bound_members = [
            str(row["version_id"])
            for row in connection.execute(
                """SELECT version_id FROM calibration_release_members
                   WHERE calibration_id=? ORDER BY version_id""",
                (calibration_id,),
            )
        ]
        if bound_members:
            family_members = bound_members
        else:
            family_members = [
                str(row["version_id"])
                for row in connection.execute(
                    """SELECT version_id FROM (
                         SELECT p.version_a AS version_id
                         FROM calibrated_edges c JOIN pair_scores p USING(pair_id)
                         WHERE c.calibration_id=?
                         UNION
                         SELECT p.version_b AS version_id
                         FROM calibrated_edges c JOIN pair_scores p USING(pair_id)
                         WHERE c.calibration_id=?
                       ) ORDER BY version_id""",
                    (calibration_id, calibration_id),
                )
            ]
    grids: set[tuple[int, ...]] = set()
    bootstrap_counts: set[int] = set()
    profile_modes: set[str] = set()
    for version_id in family_members:
        if version_id not in profiles:
            profiles[version_id] = _load_profile_for_version(
                catalog,
                version_id=version_id,
                profile_kind=profile_kind,
                expected_pool_hash=expected_pool_hash,
            )
        profile = profiles[version_id]
        try:
            grids.add(validated_stored_profile_grid(profile))
        except ValueError as exc:
            raise PermanentJobError(f"Invalid null profile for {version_id!r}: {exc}") from exc
        bootstrap_counts.add(int(profile.B))
        profile_modes.add(str(profile.mode))
    if len(grids) != 1 or len(bootstrap_counts) != 1 or len(profile_modes) != 1:
        raise PermanentJobError(
            "Calibration members do not share one null grid, bootstrap count, and profile mode."
        )
    grid = next(iter(grids))
    bootstrap_count = next(iter(bootstrap_counts))
    profile_mode = next(iter(profile_modes))
    if bootstrap_count < 1 or not profile_mode.strip():
        raise PermanentJobError(
            "Null-profile bootstrap count must be positive and profile mode must be named."
        )
    release_manifest = json.loads(release["manifest_json"])
    if release_manifest.get("grid") not in (None, list(grid)):
        raise PermanentJobError("Frozen release grid differs from its null-profile artifacts.")
    if release_manifest.get("B") not in (None, bootstrap_count):
        raise PermanentJobError(
            "Frozen release bootstrap count differs from its null-profile artifacts."
        )
    diagnostics = catalog.calibration_boundary_diagnostics(
        calibration_id,
        grid_min=grid[0],
        grid_max=grid[-1],
    )
    diagnostics.update(
        {
            "grid": list(grid),
            "B": bootstrap_count,
            "profile_mode": profile_mode,
        }
    )
    if release["mode"] == "exact" and diagnostics["boundary_clamped_pair_count"]:
        raise PermanentJobError(
            "Exact calibration cannot finalize because one or more sample counts are outside "
            f"the null grid [{grid[0]}, {grid[-1]}]."
        )
    try:
        finalized = catalog.finalize_bh(calibration_id, diagnostics=diagnostics)
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    catalog.update_job_progress(
        job_id,
        worker_id=worker_id,
        progress={
            "pvalues_written_this_attempt": written,
            "finalized_pair_count": finalized,
            **diagnostics,
        },
    )
    return {
        "calibration_id": calibration_id,
        "pvalues_written_this_attempt": written,
        "finalized_pair_count": finalized,
        **diagnostics,
    }


DEFAULT_HANDLERS: dict[str, JobHandler] = {
    "score_incremental": handle_incremental_score,
    "calibrate_release": handle_calibrate_release,
}


def run_one_job(
    catalog: Catalog,
    *,
    worker_id: str,
    kinds: Sequence[str] | None = None,
    lease_seconds: int = 300,
    handlers: Mapping[str, JobHandler] | None = None,
) -> dict[str, Any] | None:
    handlers = handlers or DEFAULT_HANDLERS
    claimed = catalog.claim_jobs(
        worker_id=worker_id,
        kinds=kinds or tuple(handlers),
        limit=1,
        lease_seconds=lease_seconds,
    )
    if not claimed:
        return None
    job = claimed[0]
    handler = handlers.get(job["kind"])
    if handler is None:
        status = catalog.fail_job(
            job["job_id"],
            worker_id=worker_id,
            error_code="UNKNOWN_JOB_KIND",
            error_detail=f"No handler is registered for {job['kind']!r}.",
            retryable=False,
        )
        return {"job_id": job["job_id"], "status": status}
    try:
        result = dict(handler(catalog, job["payload"], job["job_id"], worker_id))
    except PermanentJobError as exc:
        status = catalog.fail_job(
            job["job_id"],
            worker_id=worker_id,
            error_code="INVALID_JOB_INPUT",
            error_detail=str(exc),
            retryable=False,
        )
        LOGGER.error("Permanent job failure", extra={"job_id": job["job_id"], "error": str(exc)})
        return {"job_id": job["job_id"], "status": status, "error": str(exc)}
    except Exception as exc:
        status = catalog.fail_job(
            job["job_id"],
            worker_id=worker_id,
            error_code=type(exc).__name__.upper(),
            error_detail=str(exc),
            retryable=True,
        )
        LOGGER.exception("Retryable job failure", extra={"job_id": job["job_id"]})
        return {"job_id": job["job_id"], "status": status, "error": str(exc)}
    catalog.complete_job(job["job_id"], worker_id=worker_id)
    return {"job_id": job["job_id"], "status": "succeeded", "result": result}


def run_worker(
    catalog: Catalog,
    *,
    worker_id: str | None = None,
    once: bool = False,
    poll_seconds: float = 5.0,
    kinds: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run leased jobs; ``once`` is convenient for schedulers and tests."""

    worker_id = worker_id or default_worker_id()
    poll_seconds = max(0.1, min(float(poll_seconds), 60.0))
    results: list[dict[str, Any]] = []
    while True:
        result = run_one_job(catalog, worker_id=worker_id, kinds=kinds)
        if result is not None:
            results.append(result)
        if once:
            return results
        if result is None:
            time.sleep(poll_seconds)
