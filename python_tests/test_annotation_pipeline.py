from __future__ import annotations

import httpx
from cskl_atlas.annotation_pipeline import run_annotation_pipeline
from cskl_atlas.annotations import AnnotationService, unknown_annotations
from cskl_atlas.catalog import Catalog
from cskl_atlas.openrouter import OpenRouterClient, OpenRouterSettings
from cskl_pipeline.scale.store import atomic_write_json


def _catalog_with_version(path) -> Catalog:
    catalog = Catalog(path)
    catalog.initialize()
    _, version_id = catalog.register_dataset_version(
        accession="GSE1",
        platform="GPL570",
        cohort="series",
        source_revision="test",
        source_hash="1" * 64,
        normalized_hash="1" * 64,
        signature_hash="2" * 64,
        feature_hash="3" * 40,
        config_hash="4" * 64,
        sample_count=1,
        metadata={},
    )
    for kind, checksum, dependency in (
        ("normalized_matrix", "1" * 64, "5" * 64),
        ("pca_signature", "2" * 64, "6" * 64),
    ):
        catalog.record_artifact(
            artifact_id=kind,
            kind=kind,
            uri=f"/test/{kind}",
            checksum=checksum,
            dependency_hash=dependency,
            manifest={},
            dataset_version_id=version_id,
        )
    catalog.promote_dataset_version(version_id)
    return catalog


def test_annotation_pipeline_checkpoints_geo_first_candidates(tmp_path):
    catalog = _catalog_with_version(tmp_path / "atlas.sqlite")
    records = tmp_path / "geo" / "records"
    records.mkdir(parents=True)
    atomic_write_json(
        records / "GSE1.json",
        {
            "title": "Human tissue study",
            "summary": "A public study.",
            "organisms": ["Homo sapiens"],
            "platforms": ["GPL570"],
        },
    )

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test-run",
                "choices": [{"message": {"content": unknown_annotations().model_dump_json()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = OpenRouterClient(
        OpenRouterSettings(api_key="test", allowed_models=frozenset({"test/model"})),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    report = run_annotation_pipeline(
        catalog,
        metadata_directory=tmp_path / "geo",
        output_directory=tmp_path / "candidates",
        api_key="",
        model="test/model",
        service=AnnotationService(client),
    )
    assert report["completed"] == 1
    assert report["operator_required"] is False
    assert report["ontology_resolver_version"] == "ols-exact-label-resolver-v1"
    assert report["ontology_resolution_status_counts"] == {"no_candidates": 1}
    assert (tmp_path / "candidates" / "GSE1.json").is_file()
    with catalog.reader() as connection:
        assertion = connection.execute("SELECT * FROM annotation_assertions").fetchone()
        ai_run = connection.execute("SELECT * FROM ai_runs").fetchone()
    assert assertion["field"] == "organism"
    assert assertion["value"] == "Homo sapiens"
    assert assertion["source_kind"] == "geo_structured"
    assert assertion["review_state"] == "accepted"
    assert ai_run["status"] == "succeeded"

    cached_report = run_annotation_pipeline(
        catalog,
        metadata_directory=tmp_path / "geo",
        output_directory=tmp_path / "candidates",
        api_key="",
        model="test/model",
    )
    assert cached_report["cached"] == 1
    assert cached_report["ontology_resolution_status_counts"] == {"no_candidates": 1}
    assert calls == 1

    atomic_write_json(
        records / "GSE1.json",
        {
            "title": "Human tissue study",
            "summary": "The source metadata changed.",
            "organisms": ["Homo sapiens"],
            "platforms": ["GPL570"],
        },
    )
    changed_report = run_annotation_pipeline(
        catalog,
        metadata_directory=tmp_path / "geo",
        output_directory=tmp_path / "candidates",
        api_key="",
        model="test/model",
        service=AnnotationService(client),
    )
    assert changed_report["completed"] == 1
    assert calls == 2


def test_annotation_pipeline_checkpoints_missing_secret_as_operator_action(tmp_path):
    catalog = _catalog_with_version(tmp_path / "atlas.sqlite")

    report = run_annotation_pipeline(
        catalog,
        metadata_directory=tmp_path / "geo",
        output_directory=tmp_path / "candidates",
        api_key="",
        model="test/model",
    )

    assert report == {
        "schema": "annotation-pipeline-report-v2",
        "model": "test/model",
        "workers": 1,
        "ontology_resolver_version": "ols-exact-label-resolver-v1",
        "ontology_cache_directory": str((tmp_path / "candidates" / "_ols-cache").resolve()),
        "ontology_resolution_status_counts": {},
        "requested": 1,
        "completed": 0,
        "cached": 0,
        "failed": 1,
        "operator_required": True,
        "remaining": 1,
        "failures": [
            {
                "accession": "__configuration__",
                "error": "OPENROUTER_API_KEY is not configured",
            }
        ],
    }
    assert (tmp_path / "candidates" / "last-run-report.json").is_file()
