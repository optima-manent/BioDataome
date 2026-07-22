from __future__ import annotations

import hashlib

from cskl_atlas.api import app, get_catalog
from cskl_atlas.catalog import Catalog, pair_family_hash
from fastapi.testclient import TestClient


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _version(
    catalog: Catalog, accession: str, config: str = "config"
) -> tuple[str, str]:
    signature_source = (
        f"signature:{accession}"
        if config == "config"
        else f"signature:{accession}:{config}"
    )
    dataset_uid, version_id = catalog.register_dataset_version(
        accession=accession,
        platform="GPL570",
        cohort="series",
        source_revision="2026-07-19",
        source_hash=_digest(f"source:{accession}"),
        normalized_hash=_digest(f"normalized:{accession}"),
        signature_hash=_digest(signature_source),
        feature_hash=_digest("feature"),
        config_hash=_digest(config),
        sample_count=12,
        metadata={"title": accession, "tissue": "Blood", "disease": "Control"},
    )
    with catalog.reader() as connection:
        record = connection.execute(
            "SELECT normalized_hash,signature_hash FROM dataset_versions WHERE version_id=?",
            (version_id,),
        ).fetchone()
    for kind, checksum in (
        ("normalized_matrix", record["normalized_hash"]),
        ("pca_signature", record["signature_hash"]),
    ):
        catalog.record_artifact(
            artifact_id=_digest(f"{version_id}:{kind}"),
            kind=kind,
            uri=f"objects/{version_id}/{kind}",
            checksum=checksum,
            dependency_hash=_digest(f"dependency:{version_id}:{kind}"),
            manifest={},
            dataset_version_id=version_id,
        )
    catalog.promote_dataset_version(version_id)
    return dataset_uid, version_id


def _seed(catalog: Catalog) -> tuple[str, str, str, str]:
    catalog.initialize()
    dataset_a, a = _version(catalog, "GSE1")
    _, b = _version(catalog, "GSE2")
    catalog.record_pair_scores([(a, b, "algo", 1.0)])
    with catalog.reader() as connection:
        pair_id = connection.execute("SELECT pair_id FROM pair_scores").fetchone()[0]
    calibration = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="parameters",
        algorithm_hash="algo",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={},
    )
    catalog.record_pvalues(calibration, [(pair_id, 0.01)])
    catalog.finalize_bh(calibration)
    manifest_path = catalog.path.parent / "snapshot.json"
    manifest_path.write_bytes(b"snapshot")
    manifest_checksum = _digest("snapshot")
    catalog.record_artifact(
        artifact_id=_digest("graph-manifest:snapshot"),
        kind="graph_manifest",
        uri=str(manifest_path),
        checksum=manifest_checksum,
        dependency_hash=_digest("graph-manifest-dependency:snapshot"),
        manifest={},
    )
    snapshot = catalog.stage_snapshot(
        calibration_id=calibration,
        stratum="human:expression:GPL570",
        policy_hash="policy",
        layout_version="layout",
        manifest_uri=str(manifest_path),
        manifest_checksum=manifest_checksum,
        datasets=[(a, 0.1, 0.2, "c1"), (b, 0.7, 0.8, "c1")],
    )
    catalog.publish_snapshot(snapshot)
    return snapshot, dataset_a, pair_id, calibration


def _seed_diff(catalog: Catalog) -> tuple[str, str, str]:
    catalog.initialize()
    _, a_v1 = _version(catalog, "GSE-DIFF-A")
    _, b = _version(catalog, "GSE-DIFF-B")
    _, c = _version(catalog, "GSE-DIFF-C")
    catalog.record_pair_scores(
        [(a_v1, b, "algo-diff-api", 1.0), (b, c, "algo-diff-api", 2.0)]
    )
    with catalog.reader() as connection:
        first_pairs = connection.execute(
            """SELECT pair_id,version_a,version_b FROM pair_scores
               WHERE algorithm_hash=? ORDER BY pair_id""",
            ("algo-diff-api",),
        ).fetchall()
    common_pair = next(
        row["pair_id"] for row in first_pairs if {row["version_a"], row["version_b"]} == {b, c}
    )
    removed_pair = next(row["pair_id"] for row in first_pairs if row["pair_id"] != common_pair)
    first_pair_ids = [row["pair_id"] for row in first_pairs]
    first_calibration = catalog.stage_calibration(
        stratum="human:expression:GPL570:diff-api",
        mode="exact",
        pool_hash="pool-diff-api-v1",
        parameter_hash="parameters-diff-api-v1",
        algorithm_hash="algo-diff-api",
        family_hash=pair_family_hash(first_pair_ids),
        expected_pair_count=2,
        manifest={},
    )
    catalog.record_pvalues(first_calibration, [(common_pair, 0.01), (removed_pair, 0.5)])
    catalog.finalize_bh(first_calibration)
    first_manifest = catalog.path.parent / "diff-api-first.json"
    first_manifest.write_bytes(b"diff-api-first")
    first_checksum = _digest("diff-api-first")
    first = catalog.stage_snapshot(
        calibration_id=first_calibration,
        stratum="human:expression:GPL570:diff-api",
        policy_hash="diff-api-policy-v1",
        layout_version="diff-api-layout-v1",
        manifest_uri=str(first_manifest),
        manifest_checksum=first_checksum,
        datasets=[
            (a_v1, 0.1, 0.1, "a"),
            (b, 0.2, 0.2, "b-old"),
            (c, 0.3, 0.3, "c"),
        ],
    )
    catalog.publish_snapshot(first)

    _, a_v2 = _version(catalog, "GSE-DIFF-A", "config-v2")
    _, d = _version(catalog, "GSE-DIFF-D")
    catalog.record_pair_scores([(a_v2, d, "algo-diff-api", 3.0)])
    with catalog.reader() as connection:
        added_pair = connection.execute(
            """SELECT pair_id FROM pair_scores
               WHERE algorithm_hash=? AND pair_id NOT IN (?,?)""",
            ("algo-diff-api", common_pair, removed_pair),
        ).fetchone()["pair_id"]
    second_pair_ids = [common_pair, added_pair]
    second_calibration = catalog.stage_calibration(
        stratum="human:expression:GPL570:diff-api",
        mode="exact",
        pool_hash="pool-diff-api-v2",
        parameter_hash="parameters-diff-api-v2",
        algorithm_hash="algo-diff-api",
        family_hash=pair_family_hash(second_pair_ids),
        expected_pair_count=2,
        manifest={},
    )
    catalog.record_pvalues(second_calibration, [(common_pair, 0.4), (added_pair, 0.01)])
    catalog.finalize_bh(second_calibration)
    second_manifest = catalog.path.parent / "diff-api-second.json"
    second_manifest.write_bytes(b"diff-api-second")
    second_checksum = _digest("diff-api-second")
    second = catalog.stage_snapshot(
        calibration_id=second_calibration,
        stratum="human:expression:GPL570:diff-api",
        policy_hash="diff-api-policy-v2",
        layout_version="diff-api-layout-v2",
        manifest_uri=str(second_manifest),
        manifest_checksum=second_checksum,
        datasets=[
            (a_v2, 0.15, 0.15, "a-new-version"),
            (b, 0.25, 0.25, "b-new"),
            (c, 0.35, 0.35, "c"),
            (d, 0.45, 0.45, "d"),
        ],
    )
    catalog.publish_snapshot(second)
    return first, second, common_pair


def test_snapshot_context_and_typed_query_api(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    snapshot, dataset_uid, pair_id, calibration = _seed(catalog)
    app.dependency_overrides[get_catalog] = lambda: catalog
    try:
        with TestClient(app) as client:
            graph = client.get("/v1/graph", params={"snapshot_id": snapshot})
            assert graph.status_code == 200
            assert graph.json()["snapshot"]["calibration_id"] == calibration

            overview = client.get(
                "/v1/graph/overview", params={"snapshot_id": snapshot}
            )
            assert overview.status_code == 200
            assert overview.json()["communities"][0]["dataset_count"] == 2

            center_version = graph.json()["nodes"][0]["version_id"]
            neighborhood = client.get(
                "/v1/graph/neighborhood",
                params={"snapshot_id": snapshot, "version_id": center_version},
            )
            assert neighborhood.status_code == 200
            assert len(neighborhood.json()["edges"]) == 1

            dataset = client.get(
                f"/v1/datasets/{dataset_uid}", params={"snapshot_id": snapshot}
            )
            assert dataset.status_code == 200
            assert dataset.json()["snapshot_id"] == snapshot

            edge = client.get(f"/v1/edges/{pair_id}", params={"snapshot_id": snapshot})
            assert edge.status_code == 200
            assert len(edge.json()["calibrations"]) == 1

            result = client.post(
                "/v1/query/execute",
                json={
                    "snapshot_id": snapshot,
                    "query": {"edge.q_value": {"lte": 0.05}},
                    "limit": 10,
                },
            )
            assert result.status_code == 200
            assert [item["pair_id"] for item in result.json()["edges"]] == [pair_id]

            invalid = client.post(
                "/v1/query/validate",
                json={"snapshot_id": snapshot, "query": {"raw_sql": {"eq": "SELECT 1"}}},
            )
            assert invalid.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_public_snapshot_diff_endpoint_is_bounded_and_rejects_unpublished_ids(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    first, second, common_pair = _seed_diff(catalog)
    app.dependency_overrides[get_catalog] = lambda: catalog
    try:
        with TestClient(app) as client:
            response = client.get(
                "/v1/snapshots/diff",
                params={
                    "from_snapshot_id": first,
                    "to_snapshot_id": second,
                    "detail_limit": 1,
                    "q_change_limit": 1,
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["from_snapshot_id"] == first
            assert payload["to_snapshot_id"] == second
            assert payload["datasets"]["added"]["count"] == 1
            assert payload["datasets"]["version_updated"]["count"] == 1
            assert payload["datasets"]["community_changed"]["count"] == 1
            assert payload["edges"]["added_count"] == 1
            assert payload["edges"]["removed_count"] == 1
            assert payload["edges"]["common_count"] == 1
            assert payload["edges"]["q_value_changes"]["items"][0]["pair_id"] == common_pair
            assert payload["provenance"]["limits"]["q_value_change_limit"] == 1

            missing = client.get(
                "/v1/snapshots/diff",
                params={
                    "from_snapshot_id": first,
                    "to_snapshot_id": "snapshot_not_published",
                },
            )
            assert missing.status_code == 404
            assert missing.json()["detail"].startswith("To snapshot is not an existing published")

            excessive = client.get(
                "/v1/snapshots/diff",
                params={
                    "from_snapshot_id": first,
                    "to_snapshot_id": second,
                    "q_change_limit": 1_001,
                },
            )
            assert excessive.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_operations_endpoints_require_token_and_reap_leases(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    job_id = catalog.enqueue_job(
        kind="metadata", job_key="GSE1", input_fingerprint="v1", payload={}
    )
    catalog.claim_jobs(worker_id="worker-a", lease_seconds=5)
    with catalog.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE job_id=?",
            (job_id,),
        )
    monkeypatch.setenv("CSKL_ATLAS_OPS_TOKEN", "secret-token")
    app.dependency_overrides[get_catalog] = lambda: catalog
    try:
        with TestClient(app) as client:
            assert client.post("/v1/ops/jobs/reap").status_code == 401
            response = client.post(
                "/v1/ops/jobs/reap", headers={"X-Atlas-Ops-Token": "secret-token"}
            )
            assert response.status_code == 200
            assert response.json() == {"reaped": 1, "retry": 1, "dead": 0}
    finally:
        app.dependency_overrides.clear()
