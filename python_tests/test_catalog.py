from __future__ import annotations

import hashlib
import sqlite3

import pytest
from cskl_atlas.catalog import Catalog, pair_family_hash, stable_id
from cskl_atlas.query_engine import (
    QueryContractError,
    UnsupportedQueryError,
    execute_query,
    validate_query_ast,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def graph_manifest(catalog: Catalog, directory, name: str) -> tuple[str, str]:
    path = directory / f"{name}.json"
    path.write_bytes(name.encode())
    checksum = digest(name)
    catalog.record_artifact(
        artifact_id=digest(f"graph-manifest:{name}"),
        kind="graph_manifest",
        uri=str(path),
        checksum=checksum,
        dependency_hash=digest(f"graph-manifest-dependency:{name}"),
        manifest={"fixture": True},
    )
    return str(path), checksum


def register(catalog: Catalog, accession: str, config_hash: str = "cfg-v1") -> tuple[str, str]:
    return catalog.register_dataset_version(
        accession=accession,
        platform="GPL570",
        cohort="series",
        source_revision="2026-07-19",
        source_hash=digest(f"source-{accession}"),
        normalized_hash=digest(f"normalized-{accession}"),
        signature_hash=digest(f"signature-{accession}-{config_hash}"),
        feature_hash=digest("feature-v1"),
        config_hash=digest(config_hash),
        sample_count=12,
        metadata={"title": accession, "tissue": "Blood", "disease": "Control"},
    )


def version(catalog: Catalog, accession: str, config_hash: str = "cfg-v1") -> str:
    _, version_id = register(catalog, accession, config_hash)
    with catalog.reader() as connection:
        row = connection.execute(
            "SELECT normalized_hash,signature_hash FROM dataset_versions WHERE version_id=?",
            (version_id,),
        ).fetchone()
    for kind, checksum in (
        ("normalized_matrix", row["normalized_hash"]),
        ("pca_signature", row["signature_hash"]),
    ):
        catalog.record_artifact(
            artifact_id=digest(f"{version_id}:{kind}"),
            kind=kind,
            uri=f"objects/{version_id}/{kind}",
            checksum=checksum,
            dependency_hash=digest(f"dependencies:{version_id}:{kind}"),
            manifest={"test_fixture": True},
            dataset_version_id=version_id,
        )
    catalog.promote_dataset_version(version_id)
    return version_id


def snapshot_diff_fixture(catalog: Catalog, directory) -> tuple[str, str, dict[str, str]]:
    stratum = "human:expression:GPL570:diff"
    a_v1 = version(catalog, "GSE-A")
    b = version(catalog, "GSE-B")
    c = version(catalog, "GSE-C")
    e = version(catalog, "GSE-E")
    catalog.record_pair_scores([(a_v1, b, "algo-diff", 1.0), (c, e, "algo-diff", 2.0)])
    with catalog.reader() as connection:
        first_pairs = connection.execute(
            """SELECT pair_id,version_a,version_b FROM pair_scores
               WHERE algorithm_hash=? ORDER BY pair_id""",
            ("algo-diff",),
        ).fetchall()
    common_pair_id = next(
        row["pair_id"] for row in first_pairs if {row["version_a"], row["version_b"]} == {c, e}
    )
    removed_pair_id = next(row["pair_id"] for row in first_pairs if row["pair_id"] != common_pair_id)
    first_pair_ids = [row["pair_id"] for row in first_pairs]
    first_calibration = catalog.stage_calibration(
        stratum=stratum,
        mode="exact",
        pool_hash="diff-pool-v1",
        parameter_hash="diff-parameters-v1",
        algorithm_hash="algo-diff",
        family_hash=pair_family_hash(first_pair_ids),
        expected_pair_count=len(first_pair_ids),
        manifest={},
    )
    catalog.record_pvalues(
        first_calibration,
        [(common_pair_id, 0.01), (removed_pair_id, 0.5)],
    )
    catalog.finalize_bh(first_calibration)
    first_manifest, first_checksum = graph_manifest(catalog, directory, "diff-first")
    first_snapshot = catalog.stage_snapshot(
        calibration_id=first_calibration,
        stratum=stratum,
        policy_hash="diff-policy-v1",
        layout_version="diff-layout-v1",
        manifest_uri=first_manifest,
        manifest_checksum=first_checksum,
        datasets=[
            (a_v1, 0.1, 0.1, "community-a"),
            (b, 0.2, 0.2, "community-b"),
            (c, 0.3, 0.3, "community-c-old"),
            (e, 0.4, 0.4, "community-e"),
        ],
    )
    catalog.publish_snapshot(first_snapshot)

    a_v2 = version(catalog, "GSE-A", "cfg-v2")
    d = version(catalog, "GSE-D")
    f = version(catalog, "GSE-F")
    catalog.record_pair_scores([(a_v2, d, "algo-diff", 3.0)])
    with catalog.reader() as connection:
        added_pair_id = connection.execute(
            """SELECT pair_id FROM pair_scores
               WHERE algorithm_hash=? AND pair_id NOT IN (?,?)""",
            ("algo-diff", common_pair_id, removed_pair_id),
        ).fetchone()["pair_id"]
    second_pair_ids = [common_pair_id, added_pair_id]
    second_calibration = catalog.stage_calibration(
        stratum=stratum,
        mode="exact",
        pool_hash="diff-pool-v2",
        parameter_hash="diff-parameters-v2",
        algorithm_hash="algo-diff",
        family_hash=pair_family_hash(second_pair_ids),
        expected_pair_count=len(second_pair_ids),
        manifest={},
    )
    catalog.record_pvalues(
        second_calibration,
        [(common_pair_id, 0.4), (added_pair_id, 0.01)],
    )
    catalog.finalize_bh(second_calibration)
    second_manifest, second_checksum = graph_manifest(catalog, directory, "diff-second")
    second_snapshot = catalog.stage_snapshot(
        calibration_id=second_calibration,
        stratum=stratum,
        policy_hash="diff-policy-v2",
        layout_version="diff-layout-v2",
        manifest_uri=second_manifest,
        manifest_checksum=second_checksum,
        datasets=[
            (a_v2, 0.15, 0.15, "community-a-new-version"),
            (c, 0.35, 0.35, "community-c-new"),
            (e, 0.45, 0.45, "community-e"),
            (d, 0.5, 0.5, "community-d"),
            (f, 0.6, 0.6, "community-f"),
        ],
    )
    catalog.publish_snapshot(second_snapshot)
    return first_snapshot, second_snapshot, {
        "a_v1": a_v1,
        "a_v2": a_v2,
        "common_pair_id": common_pair_id,
        "first_calibration": first_calibration,
        "second_calibration": second_calibration,
    }


def test_dependency_changes_create_a_new_dataset_version(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    first = version(catalog, "GSE1", "alpha-0.5")
    second = version(catalog, "GSE1", "alpha-0.6")
    assert first != second

    with catalog.reader() as connection:
        statuses = {
            row["version_id"]: row["status"]
            for row in connection.execute("SELECT version_id,status FROM dataset_versions")
        }
        current = connection.execute(
            "SELECT current_version_id FROM datasets WHERE accession='GSE1'"
        ).fetchone()[0]
    assert statuses[first] == "superseded"
    assert statuses[second] == "ready"
    assert current == second


def test_disk_backed_bh_and_atomic_snapshot_publish(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a, b, c = (version(catalog, accession) for accession in ("GSE1", "GSE2", "GSE3"))
    catalog.record_pair_scores(
        [(a, b, "algo-v1", 1.0), (a, c, "algo-v1", 2.0), (b, c, "algo-v1", 3.0)]
    )
    with catalog.reader() as connection:
        pairs = connection.execute("SELECT pair_id,version_a,version_b FROM pair_scores").fetchall()
    pair_ids = [row["pair_id"] for row in pairs]

    release = catalog.stage_calibration(
        stratum="human:expression:GPL570:feature-v1",
        mode="exact",
        pool_hash="pool-v2",
        parameter_hash="params-v3",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash(pair_ids),
        expected_pair_count=len(pair_ids),
        manifest={"alpha": 0.5, "B": 100},
    )
    catalog.record_pvalues(release, zip(pair_ids, [0.01, 0.04, 0.03], strict=True))
    assert catalog.finalize_bh(release) == 3

    with catalog.reader() as connection:
        values = sorted(
            (row["p_value"], row["q_value"])
            for row in connection.execute(
                "SELECT p_value,q_value FROM calibrated_edges WHERE calibration_id=?", (release,)
            )
        )
    assert values == pytest.approx([(0.01, 0.03), (0.03, 0.04), (0.04, 0.04)])

    snapshot = catalog.stage_snapshot(
        calibration_id=release,
        stratum="human:expression:GPL570:feature-v1",
        policy_hash="q05-independent",
        layout_version="layout-v1",
        manifest_uri=graph_manifest(catalog, tmp_path, "snapshot-manifest")[0],
        manifest_checksum=digest("snapshot-manifest"),
        datasets=[(a, 0.1, 0.2, "c1"), (b, 0.3, 0.4, "c1"), (c, 0.8, 0.7, "c2")],
    )
    assert catalog.current_snapshot("human:expression:GPL570:feature-v1") is None
    catalog.publish_snapshot(snapshot)
    assert catalog.current_snapshot("human:expression:GPL570:feature-v1")["snapshot_id"] == snapshot
    graph = catalog.graph_payload(snapshot_id=snapshot, q_max=0.05)
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 3


def test_graph_payload_reads_q_value_from_the_companion_independent_stratum(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    version_a, version_b = (
        version(catalog, accession) for accession in ("GSE1", "GSE2")
    )
    catalog.record_pair_scores([(version_a, version_b, "algo-v1", 1.0)])
    with catalog.reader() as connection:
        pair_id = str(connection.execute("SELECT pair_id FROM pair_scores").fetchone()[0])

    base_stratum = "human:expression:GPL570:feature-v1"
    global_release = catalog.stage_calibration(
        stratum=base_stratum,
        mode="exact",
        pool_hash="pool-v1",
        parameter_hash="params-global",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={"run_id": "release-v1", "family": "global"},
    )
    catalog.record_pvalues(global_release, [(pair_id, 0.01)])
    catalog.finalize_bh(global_release)

    independent_release = catalog.stage_calibration(
        stratum=f"{base_stratum}:independent",
        mode="exact",
        pool_hash="pool-v1",
        parameter_hash="params-independent",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={"run_id": "release-v1", "family": "independent"},
    )
    catalog.record_pvalues(independent_release, [(pair_id, 0.2)])
    catalog.finalize_bh(independent_release)

    unrelated_independent_release = catalog.stage_calibration(
        stratum=f"{base_stratum}:independent",
        mode="exact",
        pool_hash="pool-v1",
        parameter_hash="params-independent-unrelated",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={"run_id": "release-unrelated", "family": "independent"},
    )
    catalog.record_pvalues(unrelated_independent_release, [(pair_id, 0.6)])
    catalog.finalize_bh(unrelated_independent_release)

    manifest_uri, manifest_checksum = graph_manifest(catalog, tmp_path, "independent-q")
    snapshot = catalog.stage_snapshot(
        calibration_id=global_release,
        stratum=base_stratum,
        policy_hash="policy-v1",
        layout_version="layout-v1",
        manifest_uri=manifest_uri,
        manifest_checksum=manifest_checksum,
        datasets=[
            (version_a, 0.1, 0.2, "c1"),
            (version_b, 0.3, 0.4, "c1"),
        ],
    )
    assert snapshot == stable_id(
        "snapshot",
        global_release,
        base_stratum,
        "policy-v1",
        "layout-v1",
        manifest_uri,
        manifest_checksum,
        "",
        independent_release,
    )
    catalog.publish_snapshot(snapshot)

    edge = catalog.graph_payload(snapshot_id=snapshot)["edges"][0]
    assert edge["q_value"] == pytest.approx(0.01)
    assert edge["independent_q_value"] == pytest.approx(0.2)

    newer_independent_release = catalog.stage_calibration(
        stratum=f"{base_stratum}:independent",
        mode="exact",
        pool_hash="pool-v1",
        parameter_hash="params-independent-v2",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={"run_id": "release-v1", "family": "independent"},
    )
    catalog.record_pvalues(newer_independent_release, [(pair_id, 0.8)])
    catalog.finalize_bh(newer_independent_release)

    frozen = catalog.graph_payload(snapshot_id=snapshot)
    assert frozen["snapshot"]["independent_calibration_id"] == independent_release
    assert frozen["edges"][0]["independent_q_value"] == pytest.approx(0.2)
    query_result = execute_query(
        catalog,
        snapshot_id=snapshot,
        query={"edge.independent": {"eq": True}},
    )
    assert [edge["pair_id"] for edge in query_result["edges"]] == [pair_id]
    assert query_result["provenance"]["independent_calibration_id"] == independent_release

    rebound_snapshot = catalog.stage_snapshot(
        calibration_id=global_release,
        independent_calibration_id=newer_independent_release,
        stratum=base_stratum,
        policy_hash="policy-v1",
        layout_version="layout-v1",
        manifest_uri=manifest_uri,
        manifest_checksum=manifest_checksum,
        datasets=[
            (version_a, 0.1, 0.2, "c1"),
            (version_b, 0.3, 0.4, "c1"),
        ],
    )
    assert rebound_snapshot != snapshot

    # A schema-v4 catalog had no binding column. Its one-time migration must
    # recover the companion that existed when the snapshot became publishable,
    # rather than whichever release happens to be newest during migration, and
    # it must not rewrite the snapshot's legacy identity.
    legacy_snapshot_id = stable_id(
        "snapshot",
        global_release,
        base_stratum,
        "policy-v1",
        "layout-v1",
        manifest_uri,
        manifest_checksum,
        "",
    )
    assert legacy_snapshot_id != snapshot
    with sqlite3.connect(catalog.path) as connection:
        connection.execute(
            "UPDATE graph_snapshot_datasets SET snapshot_id=? WHERE snapshot_id=?",
            (legacy_snapshot_id, snapshot),
        )
        connection.execute(
            "UPDATE graph_snapshot_edges SET snapshot_id=? WHERE snapshot_id=?",
            (legacy_snapshot_id, snapshot),
        )
        connection.execute(
            "UPDATE snapshot_events SET snapshot_id=? WHERE snapshot_id=?",
            (legacy_snapshot_id, snapshot),
        )
        connection.execute(
            "UPDATE settings SET value=? WHERE value=?",
            (legacy_snapshot_id, snapshot),
        )
        connection.execute(
            """UPDATE graph_snapshots
               SET snapshot_id=?,independent_calibration_id=NULL WHERE snapshot_id=?""",
            (legacy_snapshot_id, snapshot),
        )
        connection.execute(
            "UPDATE catalog_meta SET value='4' WHERE key='schema_version'"
        )
    catalog.initialize()
    migrated = catalog.graph_payload(snapshot_id=legacy_snapshot_id)
    assert migrated["snapshot"]["snapshot_id"] == legacy_snapshot_id
    assert migrated["snapshot"]["independent_calibration_id"] == independent_release
    assert migrated["edges"][0]["independent_q_value"] == pytest.approx(0.2)


def test_snapshot_without_run_id_requires_an_explicit_independent_companion(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    version_a, version_b = (
        version(catalog, accession) for accession in ("GSE1", "GSE2")
    )
    catalog.record_pair_scores([(version_a, version_b, "algo-v1", 1.0)])
    with catalog.reader() as connection:
        pair_id = str(connection.execute("SELECT pair_id FROM pair_scores").fetchone()[0])
    stratum = "human:expression:GPL570:unlabeled"
    primary = catalog.stage_calibration(
        stratum=stratum,
        mode="exact",
        pool_hash="pool-v1",
        parameter_hash="params-global",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={},
    )
    catalog.record_pvalues(primary, [(pair_id, 0.01)])
    catalog.finalize_bh(primary)
    independent = catalog.stage_calibration(
        stratum=f"{stratum}:independent",
        mode="exact",
        pool_hash="pool-v1",
        parameter_hash="params-independent",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={},
    )
    catalog.record_pvalues(independent, [(pair_id, 0.2)])
    catalog.finalize_bh(independent)
    manifest_uri, manifest_checksum = graph_manifest(catalog, tmp_path, "unlabeled")
    snapshot_arguments = {
        "calibration_id": primary,
        "stratum": stratum,
        "policy_hash": "policy-v1",
        "layout_version": "layout-v1",
        "manifest_uri": manifest_uri,
        "manifest_checksum": manifest_checksum,
        "datasets": [
            (version_a, 0.1, 0.2, "c1"),
            (version_b, 0.3, 0.4, "c1"),
        ],
    }

    unbound_snapshot = catalog.stage_snapshot(**snapshot_arguments)
    catalog.publish_snapshot(unbound_snapshot)
    assert catalog.graph_payload(snapshot_id=unbound_snapshot)["snapshot"][
        "independent_calibration_id"
    ] is None
    with pytest.raises(UnsupportedQueryError, match="bound independent calibration"):
        execute_query(
            catalog,
            snapshot_id=unbound_snapshot,
            query={"edge.independent": {"eq": True}},
        )

    explicit_snapshot = catalog.stage_snapshot(
        **snapshot_arguments,
        independent_calibration_id=independent,
    )
    assert explicit_snapshot != unbound_snapshot
    with catalog.reader() as connection:
        binding = connection.execute(
            """SELECT independent_calibration_id FROM graph_snapshots
               WHERE snapshot_id=?""",
            (explicit_snapshot,),
        ).fetchone()[0]
    assert binding == independent


def test_snapshot_diff_is_logical_bounded_deterministic_and_provenanced(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    first, second, fixture = snapshot_diff_fixture(catalog, tmp_path)

    result = catalog.snapshot_diff(
        from_snapshot_id=first,
        to_snapshot_id=second,
        detail_limit=1,
        q_change_limit=1,
    )

    assert result == catalog.snapshot_diff(
        from_snapshot_id=first,
        to_snapshot_id=second,
        detail_limit=1,
        q_change_limit=1,
    )
    assert result["datasets"]["added"] == {
        "count": 2,
        "returned": 1,
        "truncated": True,
        "items": [
            {
                "dataset_uid": result["datasets"]["added"]["items"][0]["dataset_uid"],
                "accession": "GSE-D",
                "platform": "GPL570",
                "cohort": "series",
                "to_version_id": result["datasets"]["added"]["items"][0]["to_version_id"],
                "to_community": "community-d",
            }
        ],
    }
    assert result["datasets"]["removed"]["count"] == 1
    assert result["datasets"]["removed"]["items"][0]["accession"] == "GSE-B"
    update = result["datasets"]["version_updated"]
    assert update["count"] == 1
    assert update["items"][0]["accession"] == "GSE-A"
    assert update["items"][0]["from_version_id"] == fixture["a_v1"]
    assert update["items"][0]["to_version_id"] == fixture["a_v2"]
    community = result["datasets"]["community_changed"]
    assert community["count"] == 1
    assert community["items"][0]["accession"] == "GSE-C"
    assert community["items"][0]["from_community"] == "community-c-old"
    assert community["items"][0]["to_community"] == "community-c-new"

    assert result["edges"]["added_count"] == 1
    assert result["edges"]["removed_count"] == 1
    assert result["edges"]["common_count"] == 1
    assert result["edges"]["q_comparable_count"] == 1
    q_changes = result["edges"]["q_value_changes"]
    assert q_changes["count"] == 1
    assert q_changes["items"][0]["pair_id"] == fixture["common_pair_id"]
    assert q_changes["items"][0]["from_q_value"] == pytest.approx(0.02)
    assert q_changes["items"][0]["to_q_value"] == pytest.approx(0.4)
    assert q_changes["items"][0]["q_value_delta"] == pytest.approx(0.38)
    assert q_changes["items"][0]["absolute_q_value_delta"] == pytest.approx(0.38)
    assert result["provenance"]["from_snapshot"]["calibration_id"] == fixture[
        "first_calibration"
    ]
    assert result["provenance"]["to_snapshot"]["calibration_id"] == fixture[
        "second_calibration"
    ]
    assert result["provenance"]["limits"] == {
        "dataset_detail_limit_per_category": 1,
        "q_value_change_limit": 1,
    }

    with pytest.raises(ValueError, match="detail_limit"):
        catalog.snapshot_diff(
            from_snapshot_id=first,
            to_snapshot_id=second,
            detail_limit=0,
        )
    with pytest.raises(KeyError, match="published snapshot"):
        catalog.snapshot_diff(
            from_snapshot_id=first,
            to_snapshot_id="snapshot_not_published",
        )

    with catalog.reader() as connection:
        common_pair = connection.execute(
            "SELECT version_a,version_b FROM pair_scores WHERE pair_id=?",
            (fixture["common_pair_id"],),
        ).fetchone()
    other_calibration = catalog.stage_calibration(
        stratum="human:expression:GPL570:other",
        mode="exact",
        pool_hash="other-pool",
        parameter_hash="other-parameters",
        algorithm_hash="algo-diff",
        family_hash=pair_family_hash([fixture["common_pair_id"]]),
        expected_pair_count=1,
        manifest={},
    )
    catalog.record_pvalues(other_calibration, [(fixture["common_pair_id"], 0.1)])
    catalog.finalize_bh(other_calibration)
    other_manifest, other_checksum = graph_manifest(catalog, tmp_path, "diff-other-stratum")
    other_snapshot = catalog.stage_snapshot(
        calibration_id=other_calibration,
        stratum="human:expression:GPL570:other",
        policy_hash="other-policy",
        layout_version="other-layout",
        manifest_uri=other_manifest,
        manifest_checksum=other_checksum,
        datasets=[
            (common_pair["version_a"], 0.1, 0.1, "other"),
            (common_pair["version_b"], 0.2, 0.2, "other"),
        ],
    )
    catalog.publish_snapshot(other_snapshot)
    with pytest.raises(ValueError, match="same stratum"):
        catalog.snapshot_diff(
            from_snapshot_id=first,
            to_snapshot_id=other_snapshot,
        )


def test_job_failures_are_recoverable_and_bounded(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    job_id = catalog.enqueue_job(
        kind="geo_metadata",
        job_key="GSE1",
        input_fingerprint="source-v1",
        payload={"accession": "GSE1"},
        max_attempts=2,
    )
    claimed = catalog.claim_jobs(worker_id="worker-a")
    assert [job["job_id"] for job in claimed] == [job_id]
    assert catalog.fail_job(
        job_id, error_code="NCBI_429", error_detail="rate limited", retryable=True
    ) == "retry"

    with catalog.transaction() as connection:
        connection.execute("UPDATE jobs SET next_retry_at=NULL WHERE job_id=?", (job_id,))
    catalog.claim_jobs(worker_id="worker-b")
    assert catalog.fail_job(
        job_id, error_code="NCBI_500", error_detail="temporary failure", retryable=True
    ) == "dead"

    catalog.requeue_job(job_id)
    job = catalog.claim_jobs(worker_id="worker-c")[0]
    assert job["attempts"] == 1
    catalog.complete_job(job_id)
    assert catalog.list_jobs()[0]["status"] == "succeeded"


def test_schema_rejects_self_pairs(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a = version(catalog, "GSE1")
    with pytest.raises(ValueError, match="different dataset versions"):
        catalog.record_pair_scores([(a, a, "algo", 0.0)])


def test_dataset_candidate_requires_verified_artifacts_and_replay_does_not_rollback(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    _, candidate = register(catalog, "GSE1", "cfg-a")
    assert catalog.current_snapshot("unused") is None
    with catalog.reader() as connection:
        assert connection.execute(
            "SELECT current_version_id FROM datasets WHERE accession='GSE1'"
        ).fetchone()[0] is None
    with pytest.raises(ValueError, match="checksum-matching artifacts"):
        catalog.promote_dataset_version(candidate)

    first = version(catalog, "GSE1", "cfg-a")
    second = version(catalog, "GSE1", "cfg-b")
    assert first != second
    _, replayed = register(catalog, "GSE1", "cfg-a")
    assert replayed == first
    with catalog.reader() as connection:
        row = connection.execute(
            "SELECT current_version_id FROM datasets WHERE accession='GSE1'"
        ).fetchone()
    assert row["current_version_id"] == second


def test_finalized_calibration_is_complete_bound_and_immutable(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a, b, c = (version(catalog, accession) for accession in ("GSE1", "GSE2", "GSE3"))
    catalog.record_pair_scores([(a, b, "algo", 1.0), (a, c, "algo", 2.0)])
    with catalog.reader() as connection:
        ids = [row["pair_id"] for row in connection.execute("SELECT pair_id FROM pair_scores")]
    incomplete = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="params-incomplete",
        algorithm_hash="algo",
        family_hash=pair_family_hash(ids),
        expected_pair_count=2,
        manifest={},
    )
    catalog.record_pvalues(incomplete, [(ids[0], 0.01)])
    with pytest.raises(ValueError, match="incomplete"):
        catalog.finalize_bh(incomplete)

    release = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="params-complete",
        algorithm_hash="algo",
        family_hash=pair_family_hash(ids),
        expected_pair_count=2,
        manifest={},
    )
    catalog.record_pvalues(release, [(ids[0], 0.01), (ids[1], 0.02)])
    catalog.finalize_bh(release)
    with pytest.raises(ValueError, match="immutable"):
        catalog.record_pvalues(release, [(ids[0], 0.5)])
    with catalog.reader() as connection:
        percentiles = [
            row[0]
            for row in connection.execute(
                "SELECT cskl_similarity_percentile FROM calibrated_edges WHERE calibration_id=?",
                (release,),
            )
        ]
    assert sorted(percentiles) == pytest.approx([0.5, 1.0])


def test_current_calibration_freezes_only_current_complete_family_and_overlap(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a_v1, b, c = (version(catalog, accession) for accession in ("GSE1", "GSE2", "GSE3"))
    catalog.record_pair_scores(
        [(a_v1, b, "algo-current", 1.0), (a_v1, c, "algo-current", 2.0),
         (b, c, "algo-current", 3.0)]
    )

    _, a_v2 = catalog.register_dataset_version(
        accession="GSE1",
        platform="GPL570",
        cohort="series",
        source_revision="2026-07-20",
        source_hash=digest("source-GSE1-v2"),
        normalized_hash=digest("normalized-GSE1-v2"),
        signature_hash=digest("signature-GSE1-v2"),
        feature_hash=digest("feature-v1"),
        config_hash=digest("cfg-v1"),
        sample_count=12,
        metadata={"title": "GSE1 v2"},
    )
    with catalog.reader() as connection:
        record = connection.execute(
            "SELECT normalized_hash,signature_hash FROM dataset_versions WHERE version_id=?",
            (a_v2,),
        ).fetchone()
    for kind, checksum in (
        ("normalized_matrix", record["normalized_hash"]),
        ("pca_signature", record["signature_hash"]),
    ):
        catalog.record_artifact(
            artifact_id=digest(f"{a_v2}:{kind}"),
            kind=kind,
            uri=f"objects/{a_v2}/{kind}",
            checksum=checksum,
            dependency_hash=digest(f"dependencies:{a_v2}:{kind}"),
            manifest={"test_fixture": True},
            dataset_version_id=a_v2,
        )
    catalog.promote_dataset_version(a_v2)

    with pytest.raises(ValueError, match="Current pair family is incomplete"):
        catalog.stage_current_calibration(
            stratum="GPL570:global",
            mode="exact",
            pool_hash="pool",
            parameter_hash="params-incomplete",
            algorithm_hash="algo-current",
            manifest={},
        )

    catalog.record_pair_scores(
        [(a_v2, b, "algo-current", 1.5), (a_v2, c, "algo-current", 2.5)]
    )
    overlap_id = catalog.record_overlap(
        version_a=a_v2,
        version_b=b,
        evidence_hash="samples-current",
        shared_count=1,
        fraction_a=0.1,
        fraction_b=0.1,
        jaccard=0.05,
        overlap_coefficient=0.1,
        classification="minor",
        discovery_excluded=False,
        shared_samples=["GSM-current"],
    )
    family_hash, pair_count, member_hash, member_count = (
        catalog.current_pair_family_fingerprint(algorithm_hash="algo-current")
    )
    assert len(family_hash) == len(member_hash) == 64
    assert (pair_count, member_count) == (3, 3)

    calibration = catalog.stage_current_calibration(
        stratum="GPL570:global",
        mode="exact",
        pool_hash="pool",
        parameter_hash="params-complete",
        algorithm_hash="algo-current",
        manifest={},
    )
    batches = list(catalog.iter_uncalibrated_pair_scores(calibration, batch_size=2))
    frozen_rows = [row for batch in batches for row in batch]
    assert len(frozen_rows) == 3
    assert {a_v2, b, c} == {
        endpoint
        for row in frozen_rows
        for endpoint in (row["version_a"], row["version_b"])
    }
    assert all(a_v1 not in {row["version_a"], row["version_b"]} for row in frozen_rows)
    with catalog.reader() as connection:
        members = {
            row["version_id"]
            for row in connection.execute(
                "SELECT version_id FROM calibration_release_members WHERE calibration_id=?",
                (calibration,),
            )
        }
        bound_overlap_ids = {
            row["overlap_id"]
            for row in connection.execute(
                """SELECT overlap_id FROM calibration_release_pairs
                   WHERE calibration_id=? AND overlap_id IS NOT NULL""",
                (calibration,),
            )
        }
    assert members == {a_v2, b, c}
    assert bound_overlap_ids == {overlap_id}


def test_snapshot_binds_overlap_evidence_and_supports_audited_rollback(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a, b = (version(catalog, accession) for accession in ("GSE1", "GSE2"))
    catalog.record_pair_scores([(a, b, "algo", 1.0)])
    with catalog.reader() as connection:
        pair_id = connection.execute("SELECT pair_id FROM pair_scores").fetchone()[0]
    release = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="params",
        algorithm_hash="algo",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={},
    )
    catalog.record_pvalues(release, [(pair_id, 0.01)])
    catalog.finalize_bh(release)
    first_overlap = catalog.record_overlap(
        version_a=a,
        version_b=b,
        evidence_hash="samples-v1",
        shared_count=2,
        fraction_a=0.2,
        fraction_b=0.2,
        jaccard=0.1,
        overlap_coefficient=0.2,
        classification="minor",
        discovery_excluded=False,
        shared_samples=["GSM1", "GSM2"],
    )
    first_manifest, first_checksum = graph_manifest(catalog, tmp_path, "first")
    first = catalog.stage_snapshot(
        calibration_id=release,
        stratum="human:expression:GPL570",
        policy_hash="policy-a",
        layout_version="layout-a",
        manifest_uri=first_manifest,
        manifest_checksum=first_checksum,
        datasets=[(a, 0.1, 0.2, "c1"), (b, 0.4, 0.5, "c1")],
    )
    catalog.publish_snapshot(first, operator="release-bot", reason="first release")
    catalog.record_overlap(
        version_a=a,
        version_b=b,
        evidence_hash="samples-v2",
        shared_count=10,
        fraction_a=0.9,
        fraction_b=0.9,
        jaccard=0.8,
        overlap_coefficient=0.9,
        classification="major",
        discovery_excluded=True,
        shared_samples=[f"GSM{i}" for i in range(10)],
    )
    graph = catalog.graph_payload(snapshot_id=first, q_max=1, independent_only=False)
    assert graph["edges"][0]["overlap_id"] == first_overlap
    assert graph["edges"][0]["classification"] == "minor"

    second_manifest, second_checksum = graph_manifest(catalog, tmp_path, "second")
    second = catalog.stage_snapshot(
        calibration_id=release,
        stratum="human:expression:GPL570",
        policy_hash="policy-b",
        layout_version="layout-b",
        manifest_uri=second_manifest,
        manifest_checksum=second_checksum,
        datasets=[(a, 0.2, 0.3, "c1"), (b, 0.5, 0.6, "c1")],
    )
    catalog.publish_snapshot(second, operator="release-bot", reason="second release")
    catalog.rollback_snapshot(first, operator="curator", reason="release review failed")
    assert catalog.current_snapshot("human:expression:GPL570")["snapshot_id"] == first
    with catalog.reader() as connection:
        events = connection.execute(
            "SELECT action,operator FROM snapshot_events ORDER BY created_at"
        ).fetchall()
    assert [(row["action"], row["operator"]) for row in events] == [
        ("publish", "release-bot"),
        ("publish", "release-bot"),
        ("rollback", "curator"),
    ]


def test_text_similarity_is_versioned_finalized_and_snapshot_bound(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a, b = (version(catalog, accession) for accession in ("GSE1", "GSE2"))
    catalog.record_pair_scores([(a, b, "algo", 1.0)])
    with catalog.reader() as connection:
        pair_id = connection.execute("SELECT pair_id FROM pair_scores").fetchone()[0]
    calibration = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="params",
        algorithm_hash="algo",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={},
    )
    catalog.record_pvalues(calibration, [(pair_id, 0.01)])
    catalog.finalize_bh(calibration)
    text_release = catalog.stage_text_release(
        model_id="allenai/specter2_base",
        model_revision="pinned-revision",
        input_fields=["geo.title", "geo.summary"],
        corpus_hash=digest("corpus"),
        parameter_hash=digest("text-parameters"),
        manifest={"evidence": "computed fixture"},
    )
    catalog.record_text_pair_scores(text_release, [(a, b, 0.77)])
    assert catalog.finalize_text_release(text_release) == 1
    with pytest.raises(ValueError, match="immutable"):
        catalog.record_text_pair_scores(text_release, [(a, b, 0.1)])
    text_manifest, text_checksum = graph_manifest(catalog, tmp_path, "text-snapshot")
    snapshot = catalog.stage_snapshot(
        calibration_id=calibration,
        stratum="human:expression:GPL570",
        policy_hash="policy",
        layout_version="layout",
        manifest_uri=text_manifest,
        manifest_checksum=text_checksum,
        text_release_id=text_release,
        datasets=[(a, 0.1, 0.2, "c1"), (b, 0.4, 0.5, "c1")],
    )
    catalog.publish_snapshot(snapshot)
    edge = catalog.graph_payload(snapshot_id=snapshot)["edges"][0]
    assert edge["specter2_cosine"] == pytest.approx(0.77)
    assert edge["specter2_percentile"] == pytest.approx(1.0)
    assert edge["text_release_id"] == text_release


def test_worker_leases_recover_crashes_and_enforce_ownership(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    job_id = catalog.enqueue_job(
        kind="signature",
        job_key="GSE1",
        input_fingerprint="input-v1",
        payload={},
        max_attempts=3,
    )
    catalog.claim_jobs(worker_id="worker-a", lease_seconds=5)
    with catalog.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE job_id=?",
            (job_id,),
        )
    assert catalog.reap_expired_jobs() == {"reaped": 1, "retry": 1, "dead": 0}
    claimed = catalog.claim_jobs(worker_id="worker-b", lease_seconds=5)
    assert claimed[0]["error_code"] is None
    with pytest.raises(ValueError, match="actively leased"):
        catalog.heartbeat_job(job_id, worker_id="worker-a", lease_seconds=5)
    with pytest.raises(ValueError, match="holding"):
        catalog.complete_job(job_id, worker_id="worker-a")
    expires = catalog.heartbeat_job(job_id, worker_id="worker-b", lease_seconds=10)
    assert expires > claimed[0]["heartbeat_at"]
    catalog.complete_job(job_id, worker_id="worker-b")
    assert catalog.list_jobs()[0]["status"] == "succeeded"


def test_overlap_contract_rejects_impossible_values(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a, b = (version(catalog, accession) for accession in ("GSE1", "GSE2"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        catalog.record_overlap(
            version_a=a,
            version_b=b,
            evidence_hash="bad",
            shared_count=1,
            fraction_a=1.1,
            fraction_b=0.1,
            jaccard=0.1,
            overlap_coefficient=0.1,
            classification="minor",
            discovery_excluded=False,
            shared_samples=[],
        )


def test_initialize_refuses_a_future_schema(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    with catalog.transaction() as connection:
        connection.execute(
            "UPDATE catalog_meta SET value='999' WHERE key='schema_version'"
        )
    with pytest.raises(RuntimeError, match="newer"):
        catalog.initialize()


def test_query_engine_executes_typed_snapshot_bound_predicates(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a, b, c = (version(catalog, accession) for accession in ("GSE1", "GSE2", "GSE3"))
    for version_id, tissue in ((a, "Blood"), (b, "Blood"), (c, "Liver")):
        catalog.record_annotation_assertions(
            version_id,
            [
                {
                    "field": "tissue",
                    "value": tissue,
                    "ontology_id": None,
                    "source_kind": "human_verified",
                    "extractor_version": "query-test-v1",
                    "review_state": "accepted",
                }
            ],
        )
    catalog.record_pair_scores(
        [(a, b, "algo", 1.0), (a, c, "algo", 2.0), (b, c, "algo", 3.0)]
    )
    with catalog.reader() as connection:
        pair_rows = connection.execute(
            "SELECT pair_id,version_a,version_b FROM pair_scores ORDER BY pair_id"
        ).fetchall()
    pair_ids = [row["pair_id"] for row in pair_rows]
    excluded_pair = next(row for row in pair_rows if c in {row["version_a"], row["version_b"]})
    catalog.record_overlap(
        version_a=excluded_pair["version_a"],
        version_b=excluded_pair["version_b"],
        evidence_hash="major",
        shared_count=8,
        fraction_a=0.8,
        fraction_b=0.8,
        jaccard=0.7,
        overlap_coefficient=0.8,
        classification="major",
        discovery_excluded=True,
        shared_samples=[f"GSM{i}" for i in range(8)],
    )
    for pair in pair_rows:
        if pair["pair_id"] == excluded_pair["pair_id"]:
            continue
        catalog.record_overlap(
            version_a=pair["version_a"],
            version_b=pair["version_b"],
            evidence_hash=f"none:{pair['pair_id']}",
            shared_count=0,
            fraction_a=0,
            fraction_b=0,
            jaccard=0,
            overlap_coefficient=0,
            classification="none",
            discovery_excluded=False,
            shared_samples=[],
        )
    release = catalog.stage_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="params-query",
        algorithm_hash="algo",
        family_hash=pair_family_hash(pair_ids),
        expected_pair_count=3,
        manifest={},
    )
    catalog.record_pvalues(release, zip(pair_ids, [0.01, 0.02, 0.03], strict=True))
    catalog.finalize_bh(release)
    query_manifest, query_checksum = graph_manifest(catalog, tmp_path, "query")
    snapshot = catalog.stage_snapshot(
        calibration_id=release,
        stratum="human:expression:GPL570",
        policy_hash="query-policy",
        layout_version="query-layout",
        manifest_uri=query_manifest,
        manifest_checksum=query_checksum,
        datasets=[(a, 0.1, 0.1, "c1"), (b, 0.5, 0.5, "c1"), (c, 0.8, 0.8, "c2")],
    )
    catalog.publish_snapshot(snapshot)

    result = execute_query(
        catalog,
        snapshot_id=snapshot,
        query={
            "and": [
                {"edge.q_value": {"lte": 0.05}},
                {"edge.independent": {"eq": True}},
                {"node.tissue": {"same": True}},
            ]
        },
    )
    assert len(result["edges"]) == 1
    assert result["edges"][0]["discovery_excluded"] == 0
    assert result["provenance"]["calibration_id"] == release

    injected = execute_query(
        catalog,
        snapshot_id=snapshot,
        query={"node.tissue": {"eq": "Blood' OR 1=1 --"}},
    )
    assert injected["edges"] == []
    for comparison in ("same", "different"):
        unknown = execute_query(
            catalog,
            snapshot_id=snapshot,
            query={"node.disease": {comparison: True}},
        )
        assert unknown["edges"] == []
    with pytest.raises(UnsupportedQueryError, match="no finalized SPECTER2"):
        execute_query(
            catalog,
            snapshot_id=snapshot,
            query={"edge.specter2_percentile": {"gte": 0.9}},
        )
    with pytest.raises(QueryContractError, match="Unsupported query field"):
        validate_query_ast({"edge.q_value; DROP TABLE datasets": {"eq": 0.1}})


def test_annotation_and_ai_provenance_are_idempotent_and_reviewable(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    version_id = version(catalog, "GSE1")
    assertion_id = catalog.record_annotation_assertions(
        version_id,
        [
            {
                "field": "tissue",
                "value": "liver",
                "ontology_id": "UBERON:0002107",
                "source_kind": "llm_candidate",
                "source_field": "summary",
                "evidence_span": {"start": 4, "end": 9, "quote": "liver"},
                "extractor_version": "annotation-v1",
                "confidence": 0.88,
            }
        ],
    )[0]
    assert catalog.record_annotation_assertions(
        version_id,
        [
            {
                "field": "tissue",
                "value": "liver",
                "ontology_id": "UBERON:0002107",
                "source_kind": "llm_candidate",
                "source_field": "summary",
                "evidence_span": {"start": 4, "end": 9, "quote": "liver"},
                "extractor_version": "annotation-v1",
                "confidence": 0.88,
            }
        ],
    ) == [assertion_id]
    review_id = catalog.review_annotation(
        assertion_id, reviewer="curator@example.org", decision="accepted", note="GEO text verified"
    )
    assert review_id.startswith("review_")

    run_id = catalog.record_ai_run(
        task="geo_annotation",
        evidence_hash=digest("geo evidence"),
        prompt_hash=digest("annotation prompt"),
        provider="OpenRouter",
        model="approved/model",
        response={"assertion_ids": [assertion_id]},
        status="succeeded",
        cost_usd=0.01,
    )
    assert catalog.record_ai_run(
        task="geo_annotation",
        evidence_hash=digest("geo evidence"),
        prompt_hash=digest("annotation prompt"),
        provider="OpenRouter",
        model="approved/model",
        response={"assertion_ids": [assertion_id]},
        status="succeeded",
        cost_usd=0.01,
    ) == run_id
    with catalog.reader() as connection:
        assertion = connection.execute(
            "SELECT review_state FROM annotation_assertions WHERE assertion_id=?", (assertion_id,)
        ).fetchone()
        assert assertion["review_state"] == "accepted"
        assert connection.execute("SELECT COUNT(*) FROM annotation_reviews").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM ai_runs").fetchone()[0] == 1


def test_generated_annotation_replacement_preserves_human_review(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    version_id = version(catalog, "GSE1")
    original_rows = [
        {
            "field": "tissue",
            "value": value,
            "ontology_id": ontology_id,
            "source_kind": "llm_candidate",
            "source_field": "summary",
            "evidence_span": {"start": 0, "end": len(value), "quote": value},
            "extractor_version": "annotation-v1",
        }
        for value, ontology_id in (
            ("liver", "UBERON:0002107"),
            ("lung", "UBERON:0002048"),
        )
    ]
    liver_id, lung_id = catalog.record_annotation_assertions(version_id, original_rows)
    catalog.review_annotation(liver_id, reviewer="curator", decision="accepted")

    replacement_id = catalog.record_annotation_assertions(
        version_id,
        [
            {
                "field": "tissue",
                "value": "blood",
                "ontology_id": "UBERON:0000178",
                "source_kind": "llm_candidate",
                "source_field": "summary",
                "evidence_span": {"start": 0, "end": 5, "quote": "blood"},
                "extractor_version": "annotation-v2",
            }
        ],
        replace_generated=True,
    )[0]

    with catalog.reader() as connection:
        states = {
            row["assertion_id"]: row["review_state"]
            for row in connection.execute(
                "SELECT assertion_id,review_state FROM annotation_assertions"
            )
        }
    assert states[liver_id] == "accepted"
    assert states[lung_id] == "superseded"
    assert states[replacement_id] == "unreviewed"
