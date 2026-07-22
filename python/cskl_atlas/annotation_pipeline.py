"""Resumable GEO-first annotation candidate production."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from cskl_pipeline.scale.store import atomic_write_json, read_json

from .annotations import (
    ANNOTATION_FIELDS,
    ANNOTATION_PROMPT_VERSION,
    AnnotationField,
    AnnotationRun,
    AnnotationService,
    DatasetAnnotations,
    GeoMetadata,
    OntologyAssertion,
    geo_evidence_span,
    unknown_annotations,
    validate_annotation_evidence,
)
from .catalog import Catalog, canonical_json
from .ontology_validation import OLS_LABEL_RESOLVER_VERSION, OLSLabelResolver
from .openrouter import OpenRouterClient, OpenRouterSettings, payload_sha256

_TAXONOMY = {
    "Homo sapiens": "NCBITaxon:9606",
    "Mus musculus": "NCBITaxon:10090",
    "Rattus norvegicus": "NCBITaxon:10116",
}
ANNOTATION_PIPELINE_REPORT_SCHEMA = "annotation-pipeline-report-v2"


def deterministic_geo_annotations(geo: GeoMetadata) -> DatasetAnnotations:
    """Lock only exact structured GEO facts with unambiguous mappings."""

    base = unknown_annotations().model_dump()
    organism_values = []
    for index, organism in enumerate(geo.organisms):
        ontology_id = _TAXONOMY.get(organism)
        if ontology_id:
            organism_values.append(
                OntologyAssertion(
                    ontology="NCBITaxon",
                    ontology_id=ontology_id,
                    label=organism,
                    evidence_spans=(
                        geo_evidence_span(
                            geo,
                            source_field=f"organisms.{index}",
                            quote=organism,
                        ),
                    ),
                    provenance="geo_structured",
                )
            )
    if organism_values:
        base["organism"] = AnnotationField(values=tuple(organism_values), unknown=False)
    return DatasetAnnotations.model_validate(base)


def _assertion_rows(run) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_name in ANNOTATION_FIELDS:
        for assertion in getattr(run.annotations, field_name).values:
            is_locked = field_name in run.locked_geo_fields
            spans = [span.model_dump(mode="json") for span in assertion.evidence_spans]
            rows.append(
                {
                    "field": field_name,
                    "value": assertion.label,
                    "ontology_id": assertion.ontology_id,
                    "source_kind": "geo_structured" if is_locked else "llm_candidate",
                    "source_field": spans[0]["source_field"] if spans else None,
                    "evidence_span": spans,
                    "extractor_version": (
                        "geo-taxonomy-v1"
                        if is_locked
                        else run.ontology_resolver_version
                        or run.provenance.prompt_template_version
                    ),
                    "review_state": "accepted" if is_locked else "unreviewed",
                    "confidence": 1.0 if is_locked else None,
                }
            )
    return rows


def run_annotation_pipeline(
    catalog: Catalog,
    *,
    metadata_directory: str | Path,
    output_directory: str | Path,
    api_key: str,
    model: str,
    force: bool = False,
    max_consecutive_failures: int = 3,
    workers: int = 1,
    service: AnnotationService | None = None,
    ontology_cache_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Create reviewable ontology candidates; never auto-accept model output."""

    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")

    records_dir = Path(metadata_directory).resolve() / "records"
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ontology_cache = Path(ontology_cache_directory or output / "_ols-cache").resolve()
    with catalog.reader() as connection:
        current = {
            row["accession"]: row["current_version_id"]
            for row in connection.execute(
                """SELECT accession,current_version_id FROM datasets
                   WHERE current_version_id IS NOT NULL ORDER BY accession"""
                )
        }

    if not model.strip():
        report = {
            "schema": ANNOTATION_PIPELINE_REPORT_SCHEMA,
            "model": model,
            "workers": workers,
            "ontology_resolver_version": OLS_LABEL_RESOLVER_VERSION,
            "ontology_cache_directory": str(ontology_cache),
            "ontology_resolution_status_counts": {},
            "requested": len(current),
            "completed": 0,
            "cached": 0,
            "failed": 1,
            "operator_required": True,
            "remaining": len(current),
            "failures": [
                {
                    "accession": "__configuration__",
                    "error": "OPENROUTER_MODEL is not configured",
                }
            ],
        }
        atomic_write_json(output / "last-run-report.json", report)
        return report

    owned_client: OpenRouterClient | None = None
    owned_resolver: OLSLabelResolver | None = None
    detach_owned_resolver = False
    completed = 0
    cached = 0
    failures: list[dict[str, str]] = []
    consecutive_failures = 0
    resolution_status_counts: dict[str, int] = {}

    def note_resolution(run: AnnotationRun) -> None:
        status = run.ontology_resolver_status
        resolution_status_counts[status] = resolution_status_counts.get(status, 0) + 1

    def load_geo(accession: str) -> GeoMetadata:
        source_path = records_dir / f"{accession}.json"
        if not source_path.is_file():
            raise FileNotFoundError("missing_geo_metadata")
        source = read_json(source_path)
        return GeoMetadata(
            accession=accession,
            title=str(source.get("title") or ""),
            summary=str(source.get("summary") or ""),
            overall_design=str(source.get("overall_design") or ""),
            assay=str(source.get("assay") or ""),
            organisms=tuple(source.get("organisms") or ()),
            platforms=tuple(source.get("platforms") or ()),
            sample_characteristics={
                str(key): tuple(str(value) for value in values)
                for key, values in (source.get("sample_characteristics") or {}).items()
                if isinstance(values, list)
            },
        )

    def infer(accession: str, version_id: str):
        try:
            assert service is not None
            geo = load_geo(accession)
            locked = deterministic_geo_annotations(geo)
            run = service.annotate_geo(model=model, geo=geo, deterministic_geo=locked)
            return accession, version_id, geo, run, None
        except Exception as exc:
            return accession, version_id, None, None, f"{type(exc).__name__}: {exc}"[:2_000]

    def record_candidate_artifact(version_id: str, destination: Path, run: AnnotationRun) -> None:
        checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
        catalog.record_artifact(
            artifact_id=hashlib.sha256(
                f"{version_id}:ai_annotations:{checksum}".encode()
            ).hexdigest(),
            kind="ai_annotation_candidates",
            uri=str(destination),
            checksum=checksum,
            dependency_hash=hashlib.sha256(
                canonical_json(
                    {
                        "dataset_version_id": version_id,
                        "model_payload_sha256": run.provenance.payload_sha256,
                        "model_response_sha256": run.provenance.response_sha256,
                        "ontology_resolver_version": run.ontology_resolver_version,
                        "ontology_resolutions": [
                            item.model_dump(mode="json")
                            for item in run.ontology_resolutions
                        ],
                    }
                ).encode()
            ).hexdigest(),
            manifest={
                "model": model,
                "prompt_template_version": run.provenance.prompt_template_version,
                "ontology_resolver_version": run.ontology_resolver_version,
                "ontology_resolver_status": run.ontology_resolver_status,
                "review_required": True,
                "zdr_enforced": True,
            },
            dataset_version_id=version_id,
        )

    def persist(
        accession: str,
        version_id: str,
        geo: GeoMetadata,
        run: AnnotationRun,
    ) -> None:
        destination = output / f"{accession}.json"
        payload = run.model_dump(mode="json")
        atomic_write_json(destination, payload)
        assertions = _assertion_rows(run)
        catalog.record_annotation_assertions(
            version_id,
            assertions,
            replace_generated=True,
        )
        response_hash = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        catalog.record_ai_run(
            task="geo_dataset_annotations",
            evidence_hash=hashlib.sha256(
                canonical_json(geo.model_dump(mode="json")).encode()
            ).hexdigest(),
            prompt_hash=run.provenance.payload_sha256,
            provider="OpenRouter",
            model=model,
            response={
                "response_hash": response_hash,
                "candidate_count": len(assertions),
                "ontology_resolver_version": run.ontology_resolver_version,
                "ontology_resolver_status": run.ontology_resolver_status,
                "ontology_resolution_count": len(run.ontology_resolutions),
                "usage": run.usage,
            },
            status="succeeded",
        )
        record_candidate_artifact(version_id, destination, run)

    def current_cached_run(accession: str) -> tuple[GeoMetadata, AnnotationRun] | None:
        destination = output / f"{accession}.json"
        if force or not destination.is_file():
            return None
        try:
            geo = load_geo(accession)
            locked = deterministic_geo_annotations(geo)
            run = AnnotationRun.model_validate(read_json(destination))
            if run.provenance.model != model:
                return None
            if run.provenance.prompt_template_version != ANNOTATION_PROMPT_VERSION:
                return None
            if (
                run.ontology_resolver_status != "no_candidates"
                and run.ontology_resolver_version != OLS_LABEL_RESOLVER_VERSION
            ):
                return None
            expected_sources = {
                "geo_metadata": payload_sha256(geo),
                "deterministic_geo_annotations": payload_sha256(locked),
            }
            if run.provenance.source_sha256 != expected_sources:
                return None
            validate_annotation_evidence(run.annotations, geo)
            return geo, run
        except (OSError, ValueError, TypeError):
            return None

    pending_items: list[tuple[str, str]] = []
    for accession, version_id in current.items():
        cached_run = current_cached_run(accession)
        if cached_run is not None:
            _, run = cached_run
            catalog.record_annotation_assertions(
                version_id,
                _assertion_rows(run),
                replace_generated=True,
            )
            record_candidate_artifact(version_id, output / f"{accession}.json", run)
            note_resolution(run)
            cached += 1
        else:
            pending_items.append((accession, version_id))

    def report_payload() -> dict[str, Any]:
        return {
            "schema": ANNOTATION_PIPELINE_REPORT_SCHEMA,
            "model": model,
            "workers": workers,
            "ontology_resolver_version": OLS_LABEL_RESOLVER_VERSION,
            "ontology_cache_directory": str(ontology_cache),
            "ontology_resolution_status_counts": dict(sorted(resolution_status_counts.items())),
            "requested": len(current),
            "completed": completed,
            "cached": cached,
            "failed": len(failures),
            "operator_required": consecutive_failures >= max_consecutive_failures,
            "remaining": len(current) - completed - cached,
            "failures": failures,
        }

    if pending_items and not api_key.strip() and service is None:
        failures.append(
            {
                "accession": "__configuration__",
                "error": "OPENROUTER_API_KEY is not configured",
            }
        )
        consecutive_failures = max_consecutive_failures
        report = report_payload()
        atomic_write_json(output / "last-run-report.json", report)
        return report

    if pending_items and service is None:
        owned_client = OpenRouterClient(
            OpenRouterSettings(
                api_key=api_key,
                allowed_models=frozenset({model}),
                app_name="C-SKL Atlas",
                app_url=os.getenv("CSKL_ATLAS_PUBLIC_ORIGIN", "").strip() or None,
                timeout_seconds=90,
                max_tokens=int(os.getenv("OPENROUTER_ANNOTATION_MAX_TOKENS", "8000")),
            )
        )
        owned_resolver = OLSLabelResolver(
            ontology_cache
        )
        service = AnnotationService(owned_client, ontology_resolver=owned_resolver)
    elif pending_items and service is not None and service.ontology_resolver is None:
        owned_resolver = OLSLabelResolver(
            ontology_cache
        )
        service.ontology_resolver = owned_resolver
        detach_owned_resolver = True

    atomic_write_json(output / "last-run-report.json", report_payload())

    executor: ThreadPoolExecutor | None = None
    try:
        if workers == 1:
            results = (infer(accession, version_id) for accession, version_id in pending_items)
            for accession, version_id, geo, run, error in results:
                if error is None:
                    persist(accession, version_id, geo, run)
                    note_resolution(run)
                    completed += 1
                    consecutive_failures = 0
                else:
                    failures.append({"accession": accession, "error": error})
                    consecutive_failures += 1
                atomic_write_json(output / "last-run-report.json", report_payload())
                if error is not None:
                    if consecutive_failures >= max_consecutive_failures:
                        break
        else:
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="geo-label")
            remaining = iter(pending_items)
            window: dict[Future, tuple[str, str]] = {}
            for _ in range(workers):
                try:
                    accession, version_id = next(remaining)
                except StopIteration:
                    break
                window[executor.submit(infer, accession, version_id)] = (accession, version_id)

            stop_requested = False
            while window:
                done, _ = wait(window, return_when=FIRST_COMPLETED)
                outcomes = sorted((future.result() for future in done), key=lambda item: item[0])
                for future in done:
                    window.pop(future)
                for accession, version_id, geo, run, error in outcomes:
                    if error is None:
                        persist(accession, version_id, geo, run)
                        note_resolution(run)
                        completed += 1
                        consecutive_failures = 0
                    else:
                        failures.append({"accession": accession, "error": error})
                        consecutive_failures += 1
                    atomic_write_json(output / "last-run-report.json", report_payload())
                    if consecutive_failures >= max_consecutive_failures:
                        stop_requested = True
                        break
                    try:
                        next_accession, next_version = next(remaining)
                    except StopIteration:
                        continue
                    window[executor.submit(infer, next_accession, next_version)] = (
                        next_accession,
                        next_version,
                    )
                if stop_requested:
                    break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if owned_client is not None:
            owned_client.close()
        if owned_resolver is not None:
            owned_resolver.close()
        if detach_owned_resolver and service is not None:
            service.ontology_resolver = None

    report = report_payload()
    atomic_write_json(output / "last-run-report.json", report)
    return report
