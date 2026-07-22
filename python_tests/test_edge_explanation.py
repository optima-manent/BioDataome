from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from cskl import PCASignature
from cskl_atlas import edge_explanation as explainer
from cskl_atlas.api import app, get_catalog
from cskl_atlas.catalog import Catalog, canonical_json, pair_family_hash, stable_id
from cskl_pipeline.scale.store import save_signature
from fastapi.testclient import TestClient


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_version(catalog: Catalog, tmp_path, accession: str, offset: float) -> str:
    signature_path = tmp_path / f"{accession}.npz"
    loadings, _ = np.linalg.qr(
        np.array(
            [
                [0.80 + offset, 0.05],
                [0.20, 0.75 - offset],
                [0.30, 0.10],
                [0.10, 0.30],
            ],
            dtype=float,
        )
    )
    signature = PCASignature(
        P=loadings,
        lam=np.array([0.6, 0.4]),
        n_features=4,
        m_samples=8,
        alpha=0.5,
    )
    save_signature(signature_path, signature, _digest("features"))
    signature_checksum = _sha256(signature_path)
    _, version_id = catalog.register_dataset_version(
        accession=accession,
        platform="GPL570",
        cohort="series",
        source_revision="fixture",
        source_hash=_digest(f"source:{accession}"),
        normalized_hash=_digest(f"normalized:{accession}"),
        signature_hash=signature_checksum,
        feature_hash=_digest("features"),
        config_hash=_digest("config"),
        sample_count=8,
        metadata={},
    )
    catalog.record_artifact(
        artifact_id=stable_id("artifact", accession, "signature"),
        kind="pca_signature",
        uri=str(signature_path),
        checksum=signature_checksum,
        dependency_hash=_digest(f"dependency:{accession}"),
        manifest={},
        dataset_version_id=version_id,
    )
    catalog.record_artifact(
        artifact_id=stable_id("artifact", accession, "matrix"),
        kind="normalized_matrix",
        uri=str(tmp_path / f"{accession}.matrix"),
        checksum=_digest(f"normalized:{accession}"),
        dependency_hash=_digest(f"matrix-dependency:{accession}"),
        manifest={},
        dataset_version_id=version_id,
    )
    catalog.promote_dataset_version(version_id)
    return version_id


def _pair_fixture(catalog: Catalog, tmp_path) -> tuple[str, list[str]]:
    catalog.initialize()
    versions = [
        _register_version(catalog, tmp_path, "GSE-A", 0.00),
        _register_version(catalog, tmp_path, "GSE-B", 0.02),
        _register_version(catalog, tmp_path, "GSE-C", 0.04),
    ]
    catalog.record_pair_scores(
        [
            (versions[0], versions[1], "algo", 0.1),
            (versions[0], versions[2], "algo", 0.2),
            (versions[1], versions[2], "algo", 0.3),
        ]
    )
    with catalog.reader() as connection:
        pair_ids = [
            row["pair_id"]
            for row in connection.execute("SELECT pair_id FROM pair_scores ORDER BY cskl")
        ]
    return versions[0], pair_ids


def _dependencies(tmp_path):
    probes = tmp_path / "probes.txt"
    probes.write_text("p1\np2\np3\np4\n", encoding="utf-8")
    annotation = tmp_path / "annotation.tsv"
    annotation.write_text(
        "PROBEID\tENTREZID\tSYMBOL\tGENENAME\n"
        "p1\t1\tA\tGene A\n"
        "p2\t2\tB\tGene B\n"
        "p3\t3\tC\tGene C\n"
        "p4\t4\tD\tGene D\n",
        encoding="utf-8",
    )
    reactome = tmp_path / "reactome.sqlite"
    reactome.write_bytes(b"fixture-index")
    return probes, annotation, reactome


def _fake_explain(signature_a, signature_b, k, mode, **options):
    del signature_a, signature_b
    indices = np.arange(k) if mode == "B" else np.arange(4 - k, 4)
    scores = np.linspace(1.0, 0.5, k)
    details = {"f": 2.0 if mode == "B" else 0.25}
    if options.get("return_scores"):
        return indices, scores, details
    return indices, details


def _fake_enrichment(gene_ids, *, database_path):
    del database_path
    return {
        "reactome_release": "fixture",
        "input_gene_count": len(gene_ids),
        "results": [],
    }


def test_atomic_json_concurrent_writers_never_share_a_temp_path(tmp_path):
    destination = tmp_path / "cache.json"
    values = [{"writer": index} for index in range(32)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        checksums = list(pool.map(lambda value: explainer._atomic_json(destination, value), values))
    assert len(checksums) == len(values)
    assert json.loads(destination.read_text(encoding="utf-8")) in values
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_compute_binds_implementation_and_replay_checks_catalog_bytes(
    tmp_path, monkeypatch
):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    _, pair_ids = _pair_fixture(catalog, tmp_path)
    probes, annotation, reactome = _dependencies(tmp_path)
    monkeypatch.setattr(explainer.cskl, "explain_topk", _fake_explain)
    monkeypatch.setattr(explainer, "enrich_reactome", _fake_enrichment)

    payload = explainer.compute_edge_explanation(
        catalog,
        pair_id=pair_ids[0],
        probes_path=probes,
        annotation_path=annotation,
        reactome_database_path=reactome,
        cache_directory=tmp_path / "cache",
        k=2,
        max_iter=1,
        n_init=1,
    )
    assert payload["provenance"]["implementation_checksum"]
    assert payload["dependency_hash"] == hashlib.sha256(
        canonical_json(payload["provenance"]).encode()
    ).hexdigest()
    assert explainer.replay_edge_explanation(catalog, pair_id=pair_ids[0], k=2) == payload

    with catalog.reader() as connection:
        artifact = connection.execute(
            "SELECT uri FROM artifacts WHERE kind='edge_explanation'"
        ).fetchone()
    cache_path = Path(artifact["uri"])
    cache_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        explainer.replay_edge_explanation(catalog, pair_id=pair_ids[0], k=2)

    def write_catalog_consistent(value: dict) -> None:
        cache_path.write_text(canonical_json(value), encoding="utf-8")
        with catalog.transaction() as connection:
            connection.execute(
                "UPDATE artifacts SET checksum=? WHERE kind='edge_explanation'",
                (_sha256(cache_path),),
            )

    write_catalog_consistent({**payload, "schema": "unsupported-schema"})
    with pytest.raises(ValueError, match="schema does not match"):
        explainer.replay_edge_explanation(catalog, pair_id=pair_ids[0], k=2)

    write_catalog_consistent({**payload, "pair_id": pair_ids[1]})
    with pytest.raises(ValueError, match="pair does not match"):
        explainer.replay_edge_explanation(catalog, pair_id=pair_ids[0], k=2)

    write_catalog_consistent({**payload, "dependency_hash": "0" * 64})
    with pytest.raises(ValueError, match="dependency hash does not match"):
        explainer.replay_edge_explanation(catalog, pair_id=pair_ids[0], k=2)


def _record_fake_explanation(catalog: Catalog, directory, pair_id: str, k: int) -> dict:
    provenance = {"schema": explainer.EXPLANATION_SCHEMA, "pair_id": pair_id, "k": k}
    dependency_hash = hashlib.sha256(canonical_json(provenance).encode()).hexdigest()
    payload = {
        "schema": explainer.EXPLANATION_SCHEMA,
        "pair_id": pair_id,
        "parameters": {"k": k},
        "dependency_hash": dependency_hash,
        "provenance": provenance,
    }
    path = directory / f"{pair_id}-{dependency_hash[:16]}.json"
    checksum = explainer._atomic_json(path, payload)
    catalog.record_artifact(
        artifact_id=stable_id("artifact", "edge_explanation", dependency_hash),
        kind="edge_explanation",
        uri=str(path),
        checksum=checksum,
        dependency_hash=dependency_hash,
        manifest={"pair_id": pair_id, "schema": explainer.EXPLANATION_SCHEMA},
    )
    return payload


def test_snapshot_batch_is_bounded_checkpointed_and_resumes(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    _, pair_ids = _pair_fixture(catalog, tmp_path)
    with catalog.reader() as connection:
        versions = [
            row["version_id"]
            for row in connection.execute("SELECT version_id FROM dataset_versions ORDER BY version_id")
        ]
    calibration = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="parameters",
        algorithm_hash="algo",
        family_hash=pair_family_hash(pair_ids),
        expected_pair_count=3,
        manifest={},
    )
    catalog.record_pvalues(calibration, zip(pair_ids, (0.01, 0.02, 0.03), strict=True))
    catalog.finalize_bh(calibration)
    manifest = tmp_path / "snapshot.json"
    manifest.write_text("{}", encoding="utf-8")
    snapshot = catalog.stage_snapshot(
        calibration_id=calibration,
        stratum="human:expression:GPL570",
        policy_hash="policy",
        layout_version="layout",
        manifest_uri=str(manifest),
        manifest_checksum=_sha256(manifest),
        datasets=[(version, float(index), 0.0, "c1") for index, version in enumerate(versions)],
    )
    with catalog.transaction() as connection:
        connection.execute(
            "UPDATE graph_snapshots SET status='published',published_at='2026-01-01T00:00:00+00:00' WHERE snapshot_id=?",
            (snapshot,),
        )

    cache = tmp_path / "cache"
    _record_fake_explanation(catalog, cache, pair_ids[0], 20)
    calls: list[str] = []

    def fake_compute(target_catalog, *, pair_id, cache_directory, k, **kwargs):
        del kwargs, cache_directory
        calls.append(pair_id)
        return _record_fake_explanation(target_catalog, cache, pair_id, k)

    monkeypatch.setattr(explainer, "compute_edge_explanation", fake_compute)
    common = {
        "catalog": catalog,
        "snapshot_id": snapshot,
        "probes_path": tmp_path / "unused-probes",
        "annotation_path": tmp_path / "unused-annotation",
        "reactome_database_path": tmp_path / "unused-reactome",
        "cache_directory": cache,
        "report_path": tmp_path / "batch-report.json",
        "max_edges": 1,
        "time_budget_seconds": 60,
    }
    first = explainer.explain_snapshot_edges(**common)
    assert first["status"] == "bounded"
    assert first["cached_count"] == 1
    assert first["computed_count"] == 1
    assert first["pending_count"] == 1
    assert json.loads((tmp_path / "batch-report.json").read_text()) == first

    second = explainer.explain_snapshot_edges(**common)
    assert second["status"] == "complete"
    assert second["cached_count"] == 2
    assert second["computed_count"] == 1
    assert second["pending_count"] == 0
    assert calls == [pair_ids[1], pair_ids[2]]


def test_public_get_is_replay_only_and_post_requires_ops_token(
    tmp_path, monkeypatch
):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    compute_calls: list[str] = []

    def not_cached(*args, **kwargs):
        del args, kwargs
        raise explainer.ExplanationNotCachedError("pair")

    def fake_compute(*args, pair_id, **kwargs):
        del args, kwargs
        compute_calls.append(pair_id)
        return {"pair_id": pair_id, "schema": explainer.EXPLANATION_SCHEMA}

    monkeypatch.setattr(explainer, "replay_edge_explanation", not_cached)
    monkeypatch.setattr(explainer, "compute_edge_explanation", fake_compute)
    monkeypatch.setenv("CSKL_ATLAS_OPS_TOKEN", "secret")
    app.dependency_overrides[get_catalog] = lambda: catalog
    try:
        with TestClient(app) as client:
            response = client.get("/v1/edges/pair/explanation")
            assert response.status_code == 404
            assert compute_calls == []
            assert client.post("/v1/edges/pair/explanation").status_code == 401
            response = client.post(
                "/v1/edges/pair/explanation",
                headers={"X-Atlas-Ops-Token": "secret"},
            )
            assert response.status_code == 200
            assert compute_calls == ["pair"]
    finally:
        app.dependency_overrides.clear()
