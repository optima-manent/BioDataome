import hashlib
import json

import pytest
from cskl_atlas.catalog import Catalog, pair_family_hash
from cskl_atlas.release_audit import audit_release


def _digest(value: str | bytes) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _version(catalog: Catalog, directory, accession: str) -> str:
    normalized = directory / f"{accession}.tsv"
    signature = directory / f"{accession}.npz"
    normalized.write_bytes(f"normalized:{accession}".encode())
    signature.write_bytes(f"signature:{accession}".encode())
    _, version_id = catalog.register_dataset_version(
        accession=accession,
        platform="GPL570",
        cohort="series",
        source_revision="fixture-v1",
        source_hash=_digest(f"source:{accession}"),
        normalized_hash=_digest(normalized.read_bytes()),
        signature_hash=_digest(signature.read_bytes()),
        feature_hash=_digest("feature-v1"),
        config_hash=_digest("config-v1"),
        sample_count=12,
        metadata={"title": accession},
    )
    for kind, path in (("normalized_matrix", normalized), ("pca_signature", signature)):
        catalog.record_artifact(
            artifact_id=_digest(f"{version_id}:{kind}"),
            kind=kind,
            uri=str(path),
            checksum=_digest(path.read_bytes()),
            dependency_hash=_digest(f"dependencies:{version_id}:{kind}"),
            manifest={"fixture": True},
            dataset_version_id=version_id,
        )
    catalog.promote_dataset_version(version_id)
    return version_id


def _published_snapshot(catalog: Catalog, directory) -> tuple[str, str]:
    version_a = _version(catalog, directory, "GSE1")
    version_b = _version(catalog, directory, "GSE2")
    catalog.record_pair_scores([(version_a, version_b, "algo-v1", 1.0)])
    with catalog.reader() as connection:
        pair_id = str(connection.execute("SELECT pair_id FROM pair_scores").fetchone()[0])
    release = catalog.stage_calibration(
        stratum="GPL570:global",
        mode="exact",
        pool_hash="pool-v1",
        parameter_hash="params-v1",
        algorithm_hash="algo-v1",
        family_hash=pair_family_hash([pair_id]),
        expected_pair_count=1,
        manifest={"B": 500, "grid": [12]},
    )
    catalog.record_pvalues(release, [(pair_id, 0.01)])
    catalog.finalize_bh(release)

    graph_manifest = directory / "graph-manifest.json"
    graph_manifest.write_text('{"fixture":true}', encoding="utf-8")
    graph_checksum = _digest(graph_manifest.read_bytes())
    catalog.record_artifact(
        artifact_id=_digest("graph-manifest"),
        kind="graph_manifest",
        uri=str(graph_manifest),
        checksum=graph_checksum,
        dependency_hash=_digest("graph-manifest-dependencies"),
        manifest={"fixture": True},
    )
    snapshot = catalog.stage_snapshot(
        calibration_id=release,
        stratum="GPL570:global",
        policy_hash="policy-v1",
        layout_version="layout-v1",
        manifest_uri=str(graph_manifest),
        manifest_checksum=graph_checksum,
        datasets=[(version_a, 0.1, 0.2, "c1"), (version_b, 0.3, 0.4, "c1")],
    )
    catalog.publish_snapshot(snapshot)
    return snapshot, pair_id


def _check(result: dict, name: str) -> dict:
    return next(item for item in result["checks"] if item["name"] == name)


def test_release_audit_rejects_unknown_profile(tmp_path) -> None:
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    with pytest.raises(ValueError, match="profile"):
        audit_release(catalog, snapshot_id="missing", profile="marketing")


def test_geo_audit_checks_only_accessions_in_the_snapshot(tmp_path) -> None:
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    snapshot, _ = _published_snapshot(catalog, tmp_path)
    records = tmp_path / "geo" / "records"
    records.mkdir(parents=True)
    (records / "GSE999.json").write_text(
        json.dumps(
            {
                "sample_metadata_scope": "matrix_cohort",
                "sample_metadata_status": "complete",
            }
        ),
        encoding="utf-8",
    )

    missing_result = audit_release(
        catalog,
        snapshot_id=snapshot,
        profile="manuscript",
        metadata_directory=records.parent,
    )
    missing_check = _check(missing_result, "cohort_specific_geo_metadata")
    assert missing_check["status"] == "fail"
    assert missing_check["detail"]["records"] == 0
    assert missing_check["detail"]["missing_accessions"] == ["GSE1", "GSE2"]

    complete = {
        "sample_metadata_scope": "matrix_cohort",
        "sample_metadata_status": "complete",
    }
    for accession in ("GSE1", "GSE2"):
        (records / f"{accession}.json").write_text(json.dumps(complete), encoding="utf-8")
    (records / "GSE999.json").write_text(
        json.dumps(
            {
                "sample_metadata_scope": "series_union",
                "sample_metadata_status": "partial",
            }
        ),
        encoding="utf-8",
    )

    result = audit_release(
        catalog,
        snapshot_id=snapshot,
        profile="manuscript",
        metadata_directory=records.parent,
    )
    geo_check = _check(result, "cohort_specific_geo_metadata")
    assert geo_check["status"] == "pass"
    assert geo_check["detail"] == {
        "snapshot_accessions": 2,
        "records": 2,
        "missing": 0,
        "missing_accessions": [],
        "not_matrix_cohort": 0,
        "incomplete": 0,
    }


def test_explainer_coverage_requires_a_checksum_valid_local_artifact(tmp_path) -> None:
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    snapshot, pair_id = _published_snapshot(catalog, tmp_path)
    explainer = tmp_path / "edge-explanation.json"
    expected_payload = b'{"schema":"fixture"}'
    catalog.record_artifact(
        artifact_id=_digest("edge-explanation"),
        kind="edge_explanation",
        uri=str(explainer),
        checksum=_digest(expected_payload),
        dependency_hash=_digest("edge-explanation-dependencies"),
        manifest={"pair_id": pair_id},
    )

    missing = _check(
        audit_release(catalog, snapshot_id=snapshot, profile="manuscript"),
        "explainer_coverage",
    )
    assert missing["status"] == "fail"
    assert missing["detail"]["explained_published_edges"] == 0
    assert missing["detail"]["missing_or_checksum_invalid_artifacts"] == 1

    explainer.write_bytes(expected_payload)
    valid = _check(
        audit_release(catalog, snapshot_id=snapshot, profile="manuscript"),
        "explainer_coverage",
    )
    assert valid["status"] == "pass"
    assert valid["detail"]["explained_published_edges"] == 1
    assert valid["detail"]["coverage"] == 1

    explainer.write_bytes(b"corrupt")
    corrupt = _check(
        audit_release(catalog, snapshot_id=snapshot, profile="manuscript"),
        "explainer_coverage",
    )
    assert corrupt["status"] == "fail"
    assert corrupt["detail"]["explained_published_edges"] == 0
