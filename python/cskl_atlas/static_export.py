"""Sanitized, checksum-bound graph export for the static product build."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .annotations import annotation_ontology_allowed
from .catalog import Catalog, canonical_json, stable_id
from .display_facets import (
    DISEASE_FAMILY_VERSION,
    TISSUE_SYSTEM_VERSION,
    derive_disease_family,
    derive_tissue_system,
)

STATIC_GRAPH_SCHEMA = "cskl-atlas-static-graph-v3"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selected_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = [row for row in rows if row["review_state"] == "accepted"]
    return accepted or [row for row in rows if row["review_state"] == "unreviewed"]


def _frontend_explainer(value: dict[str, Any]) -> dict[str, Any]:
    def features(section: str) -> list[dict[str, Any]]:
        output = []
        for item in value[section]["features"]:
            genes = item.get("genes") or []
            output.append(
                {
                    "feature": item["feature"],
                    "gene": genes[0]["symbol"] if genes and genes[0].get("symbol") else None,
                    "genes": genes,
                    "mappingAmbiguous": bool(item.get("mapping_ambiguous")),
                }
            )
        return output

    return {
        "provenance": "computed",
        "bSet": features("best_explaining"),
        "wSet": features("most_differentiating"),
        "trajectory": value.get("trajectory") or [],
        "bestPathways": value["best_explaining"]["reactome"]["results"],
        "worstPathways": value["most_differentiating"]["reactome"]["results"],
        "reactomeRelease": value["best_explaining"]["reactome"]["reactome_release"],
        "interpretation": value["interpretation"],
    }


def _display_facet_versions() -> dict[str, str]:
    return {
        "display_facet_version": TISSUE_SYSTEM_VERSION,
        "disease_family_version": DISEASE_FAMILY_VERSION,
    }


def _static_snapshot_provenance(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    frozen = {
        key: snapshot[key]
        for key in (
            "snapshot_id",
            "calibration_id",
            "independent_calibration_id",
            "stratum",
            "policy_hash",
            "layout_version",
            "manifest_checksum",
            "text_release_id",
            "created_at",
            "published_at",
        )
    }
    frozen["status"] = "published"
    return frozen


def _static_dependency_hash(
    *,
    snapshot_manifest_checksum: str,
    independent_calibration_id: str | None,
    metadata_hashes: list[str],
    explanation_hashes: list[str],
    annotation_hash: str,
    ontology_audit_hash: str | None,
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema": STATIC_GRAPH_SCHEMA,
                "producer_checksum": _sha256_path(Path(__file__)),
                "snapshot_manifest_checksum": snapshot_manifest_checksum,
                "independent_calibration_id": independent_calibration_id,
                "metadata_hashes": sorted(metadata_hashes),
                "explanation_hashes": sorted(explanation_hashes),
                "annotation_hash": annotation_hash,
                "ontology_audit_hash": ontology_audit_hash,
                **_display_facet_versions(),
            }
        ).encode()
    ).hexdigest()


def export_static_graph(
    catalog: Catalog,
    *,
    snapshot_id: str,
    metadata_directory: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    ontology_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export a real published snapshot without leaking host filesystem paths."""

    payload = catalog.graph_payload(
        snapshot_id=snapshot_id,
        q_max=0.05,
        independent_only=False,
        edge_limit=100_000,
    )
    snapshot = payload["snapshot"]
    with catalog.reader() as connection:
        calibration = connection.execute(
            "SELECT mode,manifest_json,expected_pair_count FROM calibration_releases WHERE calibration_id=?",
            (snapshot["calibration_id"],),
        ).fetchone()
        if not calibration:
            raise ValueError("Static export requires its bound calibration release.")
        calibration_manifest = json.loads(calibration["manifest_json"])
        grid = [int(value) for value in calibration_manifest.get("grid") or []]
        clamped_pair_count = 0
        if grid:
            clamped_pair_count = int(
                connection.execute(
                    """SELECT COUNT(*) FROM calibrated_edges c
                       JOIN pair_scores p ON p.pair_id=c.pair_id
                       JOIN dataset_versions va ON va.version_id=p.version_a
                       JOIN dataset_versions vb ON vb.version_id=p.version_b
                       WHERE c.calibration_id=? AND (
                         va.sample_count<? OR va.sample_count>? OR
                         vb.sample_count<? OR vb.sample_count>?
                       )""",
                    (
                        snapshot["calibration_id"], min(grid), max(grid), min(grid), max(grid)
                    ),
                ).fetchone()[0]
            )
    layout_quality: dict[str, Any] = {}
    snapshot_manifest_path = Path(snapshot["manifest_uri"])
    if (
        snapshot_manifest_path.is_file()
        and _sha256_path(snapshot_manifest_path) == snapshot["manifest_checksum"]
    ):
        snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        if isinstance(snapshot_manifest.get("layout_quality"), dict):
            layout_quality = snapshot_manifest["layout_quality"]
    records = Path(metadata_directory).resolve() / "records"
    with catalog.reader() as connection:
        annotation_rows = connection.execute(
            """SELECT version_id,field,value,ontology_id,source_kind,review_state,confidence,
                      extractor_version
               FROM annotation_assertions
               WHERE review_state IN ('accepted','unreviewed')
               ORDER BY version_id,field,
                        CASE review_state WHEN 'accepted' THEN 0 ELSE 1 END,
                        value,assertion_id"""
        ).fetchall()
    annotations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in annotation_rows:
        item = dict(row)
        if (
            item["review_state"] == "unreviewed"
            and item["source_kind"] == "llm_candidate"
            and not annotation_ontology_allowed(item["field"], item["ontology_id"])
        ):
            continue
        annotations.setdefault(item["version_id"], {}).setdefault(item["field"], []).append(item)
    annotation_hash = hashlib.sha256(
        canonical_json([dict(row) for row in annotation_rows]).encode()
    ).hexdigest()
    ontology_audit: dict[str, Any] | None = None
    ontology_status: dict[tuple[str, str, str, str], str] = {}
    ontology_audit_hash: str | None = None
    if ontology_audit_path:
        audit_path = Path(ontology_audit_path).resolve()
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        ontology_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        ontology_audit_hash = _sha256_path(audit_path)
        for result in ontology_audit.get("results") or []:
            ontology_status[
                (
                    str(result.get("accession") or ""),
                    str(result.get("field") or ""),
                    str(result.get("curie") or ""),
                    str(result.get("label") or ""),
                )
            ] = str(result.get("status") or "not_audited")

    metadata_hashes: list[str] = []
    metadata_count = 0
    for node in payload["nodes"]:
        node.pop("metadata_json", None)
        source = records / f"{node['accession']}.json"
        if not source.is_file():
            continue
        geo = json.loads(source.read_text(encoding="utf-8"))
        metadata_hashes.append(_sha256_path(source))
        metadata_count += 1
        fields = annotations.get(node["version_id"], {})
        tissue_rows = _selected_annotations(fields.get("tissue", []))
        disease_rows = _selected_annotations(fields.get("disease", []))
        organism_rows = _selected_annotations(fields.get("organism", []))
        tissue_labels = list(dict.fromkeys(str(row["value"]) for row in tissue_rows))
        disease_labels = list(dict.fromkeys(str(row["value"]) for row in disease_rows))
        organism_labels = list(dict.fromkeys(str(row["value"]) for row in organism_rows))
        for field_name, rows in fields.items():
            for row in rows:
                row["ontology_validation"] = (
                    "accepted"
                    if row["review_state"] == "accepted"
                    else ontology_status.get(
                        (
                            node["accession"], field_name,
                            str(row["ontology_id"] or ""), str(row["value"]),
                        ),
                        "not_audited",
                    )
                )
        validated_tissue_labels = [
            str(row["value"])
            for row in tissue_rows
            if row.get("ontology_validation") in {"accepted", "canonical_or_synonym"}
        ]
        validated_disease_labels = [
            str(row["value"])
            for row in disease_rows
            if row.get("ontology_validation") in {"accepted", "canonical_or_synonym"}
        ]
        selected_rows = [
            row
            for values in fields.values()
            for row in _selected_annotations(values)
        ]
        review_required = any(row["review_state"] == "unreviewed" for row in selected_rows)
        if review_required:
            annotation_source = "llm_candidate"
            annotation_confidence = 0.0
        elif selected_rows:
            annotation_source = str(selected_rows[0]["source_kind"])
            confidences = [
                float(row["confidence"])
                for row in selected_rows
                if row["confidence"] is not None
            ]
            annotation_confidence = min(confidences) if confidences else 1.0
        else:
            annotation_source = "unknown"
            annotation_confidence = 0.0
        node["metadata"] = {
            **node.get("metadata", {}),
            "title": geo.get("title") or node["accession"],
            "summary": geo.get("summary") or "No GEO summary was supplied.",
            "organism": (
                ", ".join(geo.get("organisms") or organism_labels) or "Unknown / not reviewed"
            ),
            "tissue": " / ".join(tissue_labels) or "Mixed / unknown",
            "tissue_system": derive_tissue_system(validated_tissue_labels),
            "tissue_system_version": TISSUE_SYSTEM_VERSION,
            "tissue_system_source": (
                "ontology_label_concordant"
                if validated_tissue_labels
                else "unvalidated_or_missing"
            ),
            "disease": "; ".join(disease_labels) or "Unknown / not reviewed",
            "disease_label_source": (
                "ontology_label_concordant"
                if validated_disease_labels
                else "unvalidated_or_missing"
            ),
            "disease_family": derive_disease_family(validated_disease_labels),
            "disease_family_version": DISEASE_FAMILY_VERSION,
            "annotation_source": annotation_source,
            "annotation_confidence": annotation_confidence,
            "annotation_state": "review_required" if review_required else "accepted_or_structured",
            "annotation_candidates": {
                field: [
                    {
                        "label": row["value"],
                        "ontology_id": row["ontology_id"],
                        "source_kind": row["source_kind"],
                        "review_state": row["review_state"],
                        "extractor_version": row["extractor_version"],
                        "ontology_validation": row.get("ontology_validation", "not_audited"),
                    }
                    for row in _selected_annotations(values)
                ]
                for field, values in sorted(fields.items())
            },
            "geo_url": geo.get("geo_url"),
            "publication_date": geo.get("publication_date"),
            "pubmed_ids": geo.get("pubmed_ids") or [],
            "assay": geo.get("assay"),
        }

    explanations: dict[str, dict[str, Any]] = {}
    explanation_checksums: dict[str, str] = {}
    with catalog.reader() as connection:
        artifacts = connection.execute(
            """SELECT uri,checksum,manifest_json FROM artifacts
               WHERE kind='edge_explanation' ORDER BY created_at"""
        ).fetchall()
    for artifact in artifacts:
        path = Path(artifact["uri"])
        if not path.is_file() or _sha256_path(path) != artifact["checksum"]:
            continue
        manifest = json.loads(artifact["manifest_json"])
        pair_id = manifest.get("pair_id")
        if pair_id:
            explanations[str(pair_id)] = _frontend_explainer(
                json.loads(path.read_text(encoding="utf-8"))
            )
            explanation_checksums[str(pair_id)] = artifact["checksum"]
    attached_explanation_hashes: list[str] = []
    for edge in payload["edges"]:
        if edge["pair_id"] in explanations:
            edge["explainer"] = explanations[edge["pair_id"]]
            attached_explanation_hashes.append(explanation_checksums[edge["pair_id"]])

    payload["snapshot"] = _static_snapshot_provenance(snapshot)
    payload["release_policy"] = {
        "primary_q_family": "global all-pairs Benjamini-Hochberg",
        "independent_q_family": "pairs with no literal shared sample",
        "independent_calibration_id": snapshot["independent_calibration_id"],
        "q_threshold": 0.05,
        "overlap_display": "exact and major overlap are dotted; all remain inspectable",
        "major_overlap_coefficient": 0.5,
        "top_k_per_node": 25,
        "top_k_policy": "union",
        "community_resolution": 0.5,
        "community_seed": 1729,
        "layout_algorithm": layout_quality.get("algorithm", snapshot["layout_version"]),
        "layout_quality": layout_quality,
        **_display_facet_versions(),
        "calibration_mode": calibration["mode"],
        "null_bootstrap_count": calibration_manifest.get("B"),
        "null_grid": grid,
        "boundary_clamped_pair_count": clamped_pair_count,
        "paper_calibration_ready": bool(
            calibration["mode"] == "exact"
            and int(calibration_manifest.get("B") or 0) >= 500
            and clamped_pair_count == 0
        ),
        "annotation_release_state": "review_required",
        "ontology_audit": {
            "release_id": (ontology_audit or {}).get("ontology_release_id"),
            "paper_gate": (ontology_audit or {}).get("paper_gate", "not_run"),
            "blocking_count": (ontology_audit or {}).get("blocking_count"),
        },
        "edge_rendering": "semantic zoom: strongest per-node backbone first, all release links by 210% zoom",
    }
    dependency_hash = _static_dependency_hash(
        snapshot_manifest_checksum=snapshot["manifest_checksum"],
        independent_calibration_id=snapshot["independent_calibration_id"],
        metadata_hashes=metadata_hashes,
        explanation_hashes=attached_explanation_hashes,
        annotation_hash=annotation_hash,
        ontology_audit_hash=ontology_audit_hash,
    )
    encoded = canonical_json(payload).encode("utf-8")
    checksum = hashlib.sha256(encoded).hexdigest()
    destination = Path(output_path).resolve()
    manifest_destination = Path(manifest_path).resolve()
    immutable_destination = (
        manifest_destination.parent
        / "releases"
        / snapshot_id
        / f"atlas-graph-{checksum}.json"
    )
    if immutable_destination.is_file():
        if _sha256_path(immutable_destination) != checksum:
            raise ValueError("Immutable static graph path contains different bytes.")
    else:
        _atomic_bytes(immutable_destination, encoded)
    _atomic_bytes(destination, encoded)
    manifest = {
        "schema": STATIC_GRAPH_SCHEMA,
        "snapshot_id": snapshot_id,
        "snapshot_manifest_checksum": snapshot["manifest_checksum"],
        "layout_version": snapshot["layout_version"],
        "layout_quality": layout_quality,
        "output_checksum": checksum,
        "dependency_hash": dependency_hash,
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
        "geo_metadata_count": metadata_count,
        "annotation_assertion_count": len(annotation_rows),
        "published_annotation_assertion_count": sum(
            len(values) for fields in annotations.values() for values in fields.values()
        ),
        "annotation_extractors": sorted(
            {str(row["extractor_version"]) for row in annotation_rows}
        ),
        **_display_facet_versions(),
        "independent_calibration_id": snapshot["independent_calibration_id"],
        "ontology_audit_hash": ontology_audit_hash,
        "ontology_paper_gate": (ontology_audit or {}).get("paper_gate", "not_run"),
        "computed_explanation_count": len(attached_explanation_hashes),
        "calibration_mode": calibration["mode"],
        "null_bootstrap_count": calibration_manifest.get("B"),
        "boundary_clamped_pair_count": clamped_pair_count,
        "artifact": Path(os.path.relpath(destination, manifest_destination.parent)).as_posix(),
        "content_addressed_filename": immutable_destination.name,
    }
    _atomic_bytes(
        manifest_destination,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    catalog.record_artifact(
        artifact_id=stable_id("artifact", "static_graph_release", dependency_hash),
        kind="static_graph_release",
        uri=str(immutable_destination),
        checksum=checksum,
        dependency_hash=dependency_hash,
        manifest=manifest,
    )
    return manifest
