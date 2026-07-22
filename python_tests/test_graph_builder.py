from __future__ import annotations

import hashlib

from cskl_atlas.catalog import Catalog, pair_family_hash
from cskl_atlas.graph_builder import _collision_aware_layout, build_graph_snapshot


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _version(catalog: Catalog, accession: str) -> str:
    normalized_hash = _digest(f"normalized:{accession}")
    signature_hash = _digest(f"signature:{accession}")
    _, version_id = catalog.register_dataset_version(
        accession=accession,
        platform="GPL570",
        cohort="series",
        source_revision="fixture-v1",
        source_hash=normalized_hash,
        normalized_hash=normalized_hash,
        signature_hash=signature_hash,
        feature_hash=_digest("feature"),
        config_hash=_digest("config"),
        sample_count=20,
        metadata={"title": accession},
    )
    for kind, checksum in (
        ("normalized_matrix", normalized_hash),
        ("pca_signature", signature_hash),
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
    return version_id


def test_collision_aware_layout_is_deterministic_and_clears_dense_node_piles():
    coordinates = [
        (0.5 + (index % 5) * 0.00001, 0.5 + (index // 5) * 0.00001)
        for index in range(100)
    ]
    node_ids = [f"dataset-{index:04d}" for index in range(len(coordinates))]

    first, first_quality = _collision_aware_layout(coordinates, node_ids)
    second, second_quality = _collision_aware_layout(coordinates, node_ids)

    assert first == second
    assert first_quality == second_quality
    assert first_quality["algorithm"] == "igraph-fr-collision-v2"
    assert first_quality["severe_collision_pair_count"] == 0
    assert first_quality["observed_minimum_separation"] >= (
        first_quality["target_minimum_separation"] * 0.8
    )
    assert all(0.025 <= x <= 0.975 and 0.025 <= y <= 0.975 for x, y in first)


def test_leiden_builder_is_deterministic_and_separates_known_blocks(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    versions = [_version(catalog, f"GSE{index}") for index in range(1, 7)]
    left = set(versions[:3])
    rows = []
    pvalues = []
    for left_index, version_a in enumerate(versions):
        for right_index, version_b in enumerate(versions[left_index + 1 :], start=left_index + 1):
            same_block = (version_a in left) == (version_b in left)
            rows.append(
                (
                    version_a,
                    version_b,
                    "cskl-core-v1",
                    0.1 + 0.01 * right_index if same_block else 10.0 + right_index,
                )
            )
    catalog.record_pair_scores(rows)
    with catalog.reader() as connection:
        pairs = connection.execute(
            "SELECT pair_id,version_a,version_b FROM pair_scores ORDER BY pair_id"
        ).fetchall()
    for pair in pairs:
        same_block = (pair["version_a"] in left) == (pair["version_b"] in left)
        pvalues.append((pair["pair_id"], 0.001 if same_block else 0.9))
    calibration = catalog.stage_current_calibration(
        stratum="human:expression:GPL570",
        mode="exact",
        pool_hash="pool",
        parameter_hash="parameters",
        algorithm_hash="cskl-core-v1",
        manifest={},
    )
    catalog.record_pvalues(calibration, pvalues)
    catalog.finalize_bh(calibration)

    first = build_graph_snapshot(
        catalog,
        calibration_id=calibration,
        manifest_directory=tmp_path / "manifests",
        q_max=0.05,
        independent_only=True,
        top_k_per_node=2,
        resolution=1.0,
        seed=42,
        stability_runs=3,
    )
    second = build_graph_snapshot(
        catalog,
        calibration_id=calibration,
        manifest_directory=tmp_path / "manifests",
        q_max=0.05,
        independent_only=True,
        top_k_per_node=2,
        resolution=1.0,
        seed=42,
        stability_runs=3,
    )
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["manifest_checksum"] == second["manifest_checksum"]
    assert first["manifest"]["community_count"] == 2
    assert first["manifest"]["sparsified_edge_count"] == 6
    assert first["manifest"]["mean_membership_nmi"] == 1.0
    assert first["manifest"]["schema_version"] == 2
    assert first["manifest"]["layout_quality"]["algorithm"] == "igraph-fr-collision-v2"
    assert first["manifest"]["layout_quality"]["severe_collision_pair_count"] == 0
    assert first["validation"]["valid"] is True
    with catalog.reader() as connection:
        bound_edge_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM graph_snapshot_edges WHERE snapshot_id=?",
                (first["snapshot_id"],),
            ).fetchone()[0]
        )
    assert bound_edge_count == first["manifest"]["sparsified_edge_count"] == 6
    membership = {
        item["version_id"]: item["community"] for item in first["manifest"]["members"]
    }
    assert len({membership[value] for value in versions[:3]}) == 1
    assert len({membership[value] for value in versions[3:]}) == 1
    assert membership[versions[0]] != membership[versions[3]]
    catalog.publish_snapshot(first["snapshot_id"])
    overview = catalog.graph_overview(snapshot_id=first["snapshot_id"])
    assert len(overview["communities"]) == 2


def test_graph_builder_excludes_superseded_endpoints_from_legacy_release(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    a_v1 = _version(catalog, "GSE1")
    b = _version(catalog, "GSE2")
    catalog.record_pair_scores([(a_v1, b, "legacy-algorithm", 1.0)])

    _, a_v2 = catalog.register_dataset_version(
        accession="GSE1",
        platform="GPL570",
        cohort="series",
        source_revision="fixture-v2",
        source_hash=_digest("source:GSE1:v2"),
        normalized_hash=_digest("normalized:GSE1:v2"),
        signature_hash=_digest("signature:GSE1:v2"),
        feature_hash=_digest("feature"),
        config_hash=_digest("config"),
        sample_count=20,
        metadata={"title": "GSE1 v2"},
    )
    for kind, checksum in (
        ("normalized_matrix", _digest("normalized:GSE1:v2")),
        ("pca_signature", _digest("signature:GSE1:v2")),
    ):
        catalog.record_artifact(
            artifact_id=_digest(f"{a_v2}:{kind}"),
            kind=kind,
            uri=f"objects/{a_v2}/{kind}",
            checksum=checksum,
            dependency_hash=_digest(f"dependency:{a_v2}:{kind}"),
            manifest={},
            dataset_version_id=a_v2,
        )
    catalog.promote_dataset_version(a_v2)
    catalog.record_pair_scores([(a_v2, b, "legacy-algorithm", 0.5)])
    with catalog.reader() as connection:
        pair_ids = [
            row["pair_id"]
            for row in connection.execute(
                "SELECT pair_id FROM pair_scores WHERE algorithm_hash=? ORDER BY pair_id",
                ("legacy-algorithm",),
            )
        ]
    calibration = catalog.stage_calibration(
        stratum="GPL570:legacy",
        mode="exact",
        pool_hash="pool",
        parameter_hash="legacy-params",
        algorithm_hash="legacy-algorithm",
        family_hash=pair_family_hash(pair_ids),
        expected_pair_count=2,
        manifest={},
    )
    catalog.record_pvalues(calibration, [(pair_id, 0.001) for pair_id in pair_ids])
    catalog.finalize_bh(calibration)

    result = build_graph_snapshot(
        catalog,
        calibration_id=calibration,
        manifest_directory=tmp_path / "manifests",
        q_max=1.0,
        independent_only=False,
        top_k_per_node=2,
        resolution=1.0,
        seed=7,
        stability_runs=1,
    )
    member_ids = {row["version_id"] for row in result["manifest"]["members"]}
    assert member_ids == {a_v2, b}
    assert a_v1 not in member_ids
    assert result["manifest"]["eligible_edge_count"] == 1
    assert result["manifest"]["sparsified_edge_count"] == 1
    assert result["manifest"]["policy"]["endpoint_scope"] == "legacy_current_versions"
