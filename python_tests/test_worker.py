from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cskl
import numpy as np
from cskl_atlas.catalog import Catalog
from cskl_atlas.worker import (
    enqueue_calibration_job,
    enqueue_incremental_score_job,
    run_one_job,
)
from cskl_pipeline.scale.store import save_null_profile, save_signature


def _file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog_signature(
    catalog: Catalog, root, accession: str, axis: int, sample_count: int = 8
) -> str:
    directory = root / accession
    directory.mkdir()
    normalized = directory / "expr.tsv.gz"
    normalized.write_bytes(f"normalized:{accession}".encode())
    signature_path = directory / "signature.npz"
    loadings = np.zeros((3, 1), dtype=float)
    loadings[axis, 0] = 1.0
    signature = cskl.PCASignature(
        P=loadings,
        lam=np.array([1.5]),
        n_features=3,
        m_samples=sample_count,
        alpha=0.5,
        feature_names=None,
    )
    save_signature(signature_path, signature, "feature-v1")
    _, version_id = catalog.register_dataset_version(
        accession=accession,
        platform="GPL570",
        cohort="series",
        source_revision="fixture-v1",
        source_hash=_file_hash(normalized),
        normalized_hash=_file_hash(normalized),
        signature_hash=_file_hash(signature_path),
        feature_hash="feature-v1",
        config_hash="config-v1",
        sample_count=sample_count,
        metadata={"title": accession},
    )
    for kind, path in (("normalized_matrix", normalized), ("pca_signature", signature_path)):
        catalog.record_artifact(
            artifact_id=hashlib.sha256(f"{version_id}:{kind}".encode()).hexdigest(),
            kind=kind,
            uri=str(path),
            checksum=_file_hash(path),
            dependency_hash=hashlib.sha256(f"deps:{version_id}:{kind}".encode()).hexdigest(),
            manifest={},
            dataset_version_id=version_id,
        )
    catalog.promote_dataset_version(version_id)
    return version_id


def test_leased_incremental_worker_scores_only_delta_and_resumes_idempotently(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    old_a = _catalog_signature(catalog, tmp_path, "GSE1", 0)
    old_b = _catalog_signature(catalog, tmp_path, "GSE2", 1)
    new = _catalog_signature(catalog, tmp_path, "GSE3", 2)
    job_id = enqueue_incremental_score_job(
        catalog, new_version_ids=[new], algorithm_hash="cskl-core-v1"
    )

    original_heartbeat = catalog.heartbeat_job
    calls = 0

    def interrupt_after_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated process interruption")
        return original_heartbeat(*args, **kwargs)

    monkeypatch.setattr(catalog, "heartbeat_job", interrupt_after_checkpoint)
    first = run_one_job(catalog, worker_id="worker-a")
    assert first["status"] == "retry"
    interrupted = catalog.get_job(job_id)
    assert interrupted["progress"]["existing_completed"] == 1

    with catalog.transaction() as connection:
        connection.execute("UPDATE jobs SET next_retry_at=NULL WHERE job_id=?", (job_id,))
    monkeypatch.setattr(catalog, "heartbeat_job", original_heartbeat)
    second = run_one_job(catalog, worker_id="worker-b")
    assert second["status"] == "succeeded"
    assert second["result"]["expected_pair_facts"] == 2
    assert catalog.get_job(job_id)["progress"]["new_new_complete"] is True

    with catalog.reader() as connection:
        rows = connection.execute(
            "SELECT version_a,version_b,cskl FROM pair_scores ORDER BY pair_id"
        ).fetchall()
    assert len(rows) == 2
    assert all(new in {row["version_a"], row["version_b"]} for row in rows)
    assert not any({row["version_a"], row["version_b"]} == {old_a, old_b} for row in rows)


def test_worker_rejects_corrupted_signature_without_retry_loop(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    _catalog_signature(catalog, tmp_path, "GSE1", 0)
    new = _catalog_signature(catalog, tmp_path, "GSE2", 1)
    job_id = enqueue_incremental_score_job(
        catalog, new_version_ids=[new], algorithm_hash="cskl-core-v1"
    )
    with catalog.reader() as connection:
        signature_uri = connection.execute(
            "SELECT uri FROM artifacts WHERE dataset_version_id=? AND kind='pca_signature'",
            (new,),
        ).fetchone()[0]
    Path(signature_uri).write_bytes(b"corrupt")
    result = run_one_job(catalog, worker_id="worker-a")
    assert result["status"] == "dead"
    assert catalog.get_job(job_id)["error_code"] == "INVALID_JOB_INPUT"


def test_calibration_worker_uses_versioned_profiles_and_finalizes_bh(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    versions = [
        _catalog_signature(catalog, tmp_path, "GSE1", 0),
        _catalog_signature(catalog, tmp_path, "GSE2", 1),
        _catalog_signature(catalog, tmp_path, "GSE3", 2),
    ]
    catalog.record_pair_scores(
        [
            (versions[0], versions[1], "cskl-core-v1", 1.0),
            (versions[0], versions[2], "cskl-core-v1", 2.0),
            (versions[1], versions[2], "cskl-core-v1", 3.0),
        ]
    )
    pool_hash = "pool-hash-v1"
    profile_kind = "null_profile:pool_v1"
    for version_id in versions:
        with catalog.reader() as connection:
            signature_uri = connection.execute(
                "SELECT uri FROM artifacts WHERE dataset_version_id=? AND kind='pca_signature'",
                (version_id,),
            ).fetchone()[0]
        profile_path = Path(signature_uri).parent / "null_profile__pool_v1.npz"
        save_null_profile(
            profile_path,
            grid=np.array([8]),
            mu=np.array([2.0]),
            sigma=np.array([1.0]),
            pool_version="pool_v1",
            pool_hash=pool_hash,
            feature_hash="feature-v1",
            mode="exact",
            B=100,
        )
        checksum = _file_hash(profile_path)
        catalog.record_artifact(
            artifact_id=hashlib.sha256(f"{version_id}:{profile_kind}".encode()).hexdigest(),
            kind=profile_kind,
            uri=str(profile_path),
            checksum=checksum,
            dependency_hash=hashlib.sha256(f"profile:{version_id}:{pool_hash}".encode()).hexdigest(),
            manifest={"pool_hash": pool_hash, "pool_version": "pool_v1"},
            dataset_version_id=version_id,
        )
    family_hash, count = catalog.pair_family_fingerprint(algorithm_hash="cskl-core-v1")
    calibration = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash=pool_hash,
        parameter_hash="profile-parameters-v1",
        algorithm_hash="cskl-core-v1",
        family_hash=family_hash,
        expected_pair_count=count,
        manifest={"profile_kind": profile_kind},
    )
    job_id = enqueue_calibration_job(
        catalog, calibration_id=calibration, profile_kind=profile_kind
    )
    result = run_one_job(catalog, worker_id="calibration-worker")
    assert result["status"] == "succeeded"
    assert result["result"]["finalized_pair_count"] == 3
    assert catalog.get_job(job_id)["progress"]["finalized_pair_count"] == 3
    with catalog.reader() as connection:
        release = connection.execute(
            "SELECT status FROM calibration_releases WHERE calibration_id=?", (calibration,)
        ).fetchone()
        rows = connection.execute(
            "SELECT p_value,q_value,cskl_similarity_percentile FROM calibrated_edges"
        ).fetchall()
    assert release["status"] == "calibrated"
    assert len(rows) == 3
    assert all(0 <= row["p_value"] <= row["q_value"] <= 1 for row in rows)
    assert all(row["cskl_similarity_percentile"] is not None for row in rows)


def test_worker_exact_rejects_boundary_clamp_and_frozen_reports_it(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    version_a = _catalog_signature(catalog, tmp_path, "GSE1", 0, sample_count=8)
    version_b = _catalog_signature(catalog, tmp_path, "GSE2", 1, sample_count=20)
    catalog.record_pair_scores([(version_a, version_b, "cskl-core-boundary", 1.0)])
    pool_hash = "pool-boundary-v1"
    profile_kind = "null_profile:pool_boundary"
    for version_id in (version_a, version_b):
        with catalog.reader() as connection:
            signature_uri = connection.execute(
                "SELECT uri FROM artifacts WHERE dataset_version_id=? AND kind='pca_signature'",
                (version_id,),
            ).fetchone()[0]
        profile_path = Path(signature_uri).parent / "null_profile__pool_boundary.npz"
        save_null_profile(
            profile_path,
            grid=np.array([8]),
            mu=np.array([2.0]),
            sigma=np.array([1.0]),
            pool_version="pool_boundary",
            pool_hash=pool_hash,
            feature_hash="feature-v1",
            mode="exact",
            B=100,
        )
        catalog.record_artifact(
            artifact_id=hashlib.sha256(f"{version_id}:{profile_kind}".encode()).hexdigest(),
            kind=profile_kind,
            uri=str(profile_path),
            checksum=_file_hash(profile_path),
            dependency_hash=hashlib.sha256(
                f"profile:{version_id}:{pool_hash}".encode()
            ).hexdigest(),
            manifest={"pool_hash": pool_hash, "pool_version": "pool_boundary"},
            dataset_version_id=version_id,
        )

    exact = catalog.stage_current_calibration(
        stratum="GPL570:global",
        mode="exact",
        pool_hash=pool_hash,
        parameter_hash="boundary-exact",
        algorithm_hash="cskl-core-boundary",
        manifest={"profile_kind": profile_kind},
    )
    exact_job = enqueue_calibration_job(
        catalog, calibration_id=exact, profile_kind=profile_kind
    )
    exact_result = run_one_job(catalog, worker_id="exact-worker")
    assert exact_result["status"] == "dead"
    assert "outside the null-profile grid" in exact_result["error"]
    assert catalog.get_job(exact_job)["error_code"] == "INVALID_JOB_INPUT"

    frozen = catalog.stage_current_calibration(
        stratum="GPL570:global",
        mode="frozen",
        pool_hash=pool_hash,
        parameter_hash="boundary-frozen",
        algorithm_hash="cskl-core-boundary",
        manifest={"profile_kind": profile_kind},
    )
    enqueue_calibration_job(catalog, calibration_id=frozen, profile_kind=profile_kind)
    frozen_result = run_one_job(catalog, worker_id="frozen-worker")
    assert frozen_result["status"] == "succeeded"
    assert frozen_result["result"]["boundary_clamped_pair_count"] == 1
    assert frozen_result["result"]["boundary_clamped_profile_lookup_count"] == 1
    with catalog.reader() as connection:
        release = connection.execute(
            "SELECT status,manifest_json FROM calibration_releases WHERE calibration_id=?",
            (frozen,),
        ).fetchone()
    manifest = json.loads(release["manifest_json"])
    assert release["status"] == "calibrated"
    assert manifest["grid"] == [8]
    assert manifest["boundary_clamped_pair_count"] == 1
