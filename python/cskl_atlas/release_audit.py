"""Executable operational/manuscript release gates for C-SKL Atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .catalog import Catalog


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_release(
    catalog: Catalog,
    *,
    snapshot_id: str,
    profile: str,
    metadata_directory: str | Path | None = None,
    ontology_audit_path: str | Path | None = None,
    static_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if profile not in {"operational", "manuscript"}:
        raise ValueError("profile must be operational or manuscript")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any, *, paper_only: bool = False) -> None:
        blocking = not passed and (profile == "manuscript" or not paper_only)
        checks.append(
            {
                "name": name,
                "status": "pass" if passed else ("fail" if blocking else "warning"),
                "blocking": blocking,
                "detail": detail,
            }
        )

    snapshot_validation = catalog.validate_snapshot(snapshot_id)
    check("snapshot_publication_contract", snapshot_validation["valid"], snapshot_validation)

    with catalog.reader() as connection:
        snapshot = connection.execute(
            "SELECT * FROM graph_snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if not snapshot:
            raise KeyError(snapshot_id)
        calibration = connection.execute(
            "SELECT * FROM calibration_releases WHERE calibration_id=?",
            (snapshot["calibration_id"],),
        ).fetchone()
        manifest = json.loads(calibration["manifest_json"])
        grid = [int(value) for value in manifest.get("grid") or []]
        clamped_pairs = 0
        if grid:
            clamped_pairs = int(
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
                        calibration["calibration_id"], min(grid), max(grid),
                        min(grid), max(grid),
                    ),
                ).fetchone()[0]
            )
        unreviewed = int(
            connection.execute(
                """SELECT COUNT(*) FROM annotation_assertions a
                   JOIN graph_snapshot_datasets g ON g.version_id=a.version_id
                   WHERE g.snapshot_id=? AND a.review_state='unreviewed'
                     AND a.field IN ('tissue','disease','cell_type','assay')""",
                (snapshot_id,),
            ).fetchone()[0]
        )
        human_reviews = int(
            connection.execute(
                """SELECT COUNT(*) FROM annotation_reviews r
                   JOIN annotation_assertions a ON a.assertion_id=r.assertion_id
                   JOIN graph_snapshot_datasets g ON g.version_id=a.version_id
                   WHERE g.snapshot_id=?""",
                (snapshot_id,),
            ).fetchone()[0]
        )
        edge_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM graph_snapshot_edges WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()[0]
        )
        explained_count = int(
            connection.execute(
                """SELECT COUNT(DISTINCT se.pair_id)
                   FROM graph_snapshot_edges se
                   JOIN artifacts a ON a.kind='edge_explanation'
                    AND json_extract(a.manifest_json,'$.pair_id')=se.pair_id
                   WHERE se.snapshot_id=?""",
                (snapshot_id,),
            ).fetchone()[0]
        )
        dead_jobs = int(
            connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('dead','running','retry')"
            ).fetchone()[0]
        )

    check(
        "manuscript_null_release",
        calibration["mode"] == "exact" and int(manifest.get("B") or 0) >= 500,
        {"mode": calibration["mode"], "B": manifest.get("B")},
        paper_only=True,
    )
    check(
        "calibration_boundary_coverage",
        clamped_pairs == 0,
        {"boundary_clamped_pair_count": clamped_pairs, "grid": grid},
        paper_only=True,
    )
    check(
        "annotation_curator_signoff",
        unreviewed == 0 and human_reviews > 0,
        {"unreviewed_core_assertions": unreviewed, "human_review_count": human_reviews},
        paper_only=True,
    )
    check(
        "explainer_coverage",
        explained_count == edge_count,
        {
            "explained_published_edges": explained_count,
            "published_edges": edge_count,
            "coverage": explained_count / edge_count if edge_count else 0,
            "policy": "required only for a manuscript claiming all-edge mechanism coverage",
        },
        paper_only=True,
    )
    check("worker_backlog", dead_jobs == 0, {"dead_running_or_retry_jobs": dead_jobs})

    if ontology_audit_path:
        audit = json.loads(Path(ontology_audit_path).read_text(encoding="utf-8"))
        check(
            "ontology_candidate_compatibility",
            audit.get("paper_gate") == "pass",
            {
                "release_id": audit.get("ontology_release_id"),
                "blocking_count": audit.get("blocking_count"),
            },
            paper_only=True,
        )
    else:
        check(
            "ontology_candidate_compatibility",
            False,
            "No frozen ontology audit was supplied.",
            paper_only=True,
        )

    if metadata_directory:
        records = Path(metadata_directory).resolve() / "records"
        missing_scope = 0
        incomplete = 0
        total = 0
        for path in records.glob("GSE*.json"):
            total += 1
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("sample_metadata_scope") != "matrix_cohort":
                missing_scope += 1
            if record.get("sample_metadata_status") != "complete":
                incomplete += 1
        check(
            "cohort_specific_geo_metadata",
            total > 0 and missing_scope == 0 and incomplete == 0,
            {
                "records": total,
                "not_matrix_cohort": missing_scope,
                "incomplete": incomplete,
            },
            paper_only=True,
        )
    else:
        check(
            "cohort_specific_geo_metadata",
            False,
            "No GEO metadata directory was supplied.",
            paper_only=True,
        )

    if static_manifest_path:
        static_manifest_file = Path(static_manifest_path).resolve()
        static_manifest = json.loads(static_manifest_file.read_text(encoding="utf-8"))
        artifact_value = (
            static_manifest.get("artifact")
            or static_manifest.get("output")
            or static_manifest.get("immutable_output")
            or ""
        )
        output = Path(artifact_value)
        if not output.is_absolute():
            manifest_relative = (static_manifest_file.parent / output).resolve()
            repository_relative = output.resolve()
            output = (
                manifest_relative
                if manifest_relative.is_file() or not repository_relative.is_file()
                else repository_relative
            )
        valid = (
            output.is_file()
            and _sha256(output) == static_manifest.get("output_checksum")
            and static_manifest.get("snapshot_id") == snapshot_id
        )
        check(
            "static_product_artifact",
            valid,
            {
                "artifact": str(output),
                "expected_checksum": static_manifest.get("output_checksum"),
            },
        )
    else:
        check("static_product_artifact", False, "No static manifest was supplied.")

    blocking = [item for item in checks if item["blocking"]]
    return {
        "schema": "cskl-release-audit-v1",
        "profile": profile,
        "snapshot_id": snapshot_id,
        "ready": not blocking,
        "blocking_count": len(blocking),
        "checks": checks,
    }
