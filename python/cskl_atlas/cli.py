from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .catalog import Catalog
from .logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cskl-atlas",
        description="C-SKL Atlas control plane. Scientific computation remains available as cskl-scale.",
    )
    parser.add_argument(
        "--catalog",
        default=os.getenv("CSKL_ATLAS_CATALOG", "var/catalog/atlas.sqlite"),
        help="Path to the relational catalog.",
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="Initialize or migrate the catalog.")
    commands.add_parser("health", help="Print catalog health and backlog counts.")

    jobs = commands.add_parser("jobs", help="List recoverable worker jobs.")
    jobs.add_argument("--status")
    jobs.add_argument("--limit", type=int, default=200)

    retry = commands.add_parser("retry", help="Requeue one dead/cancelled job.")
    retry.add_argument("job_id")

    commands.add_parser("reap", help="Recover jobs whose worker lease expired.")

    validate_snapshot = commands.add_parser(
        "validate-snapshot", help="Run publication gates for an immutable graph snapshot."
    )
    validate_snapshot.add_argument("snapshot_id")

    build_snapshot = commands.add_parser(
        "build-snapshot", help="Build a versioned Leiden layout and stage it for validation."
    )
    build_snapshot.add_argument("--calibration-id", required=True)
    build_snapshot.add_argument("--manifest-directory", required=True)
    build_snapshot.add_argument("--q-max", type=float, required=True)
    build_snapshot.add_argument("--include-overlap-confounded", action="store_true")
    build_snapshot.add_argument("--top-k-per-node", type=int, required=True)
    build_snapshot.add_argument("--resolution", type=float, required=True)
    build_snapshot.add_argument("--seed", type=int, required=True)
    build_snapshot.add_argument("--stability-runs", type=int, default=5)
    build_snapshot.add_argument("--text-release-id")

    publish_snapshot = commands.add_parser(
        "publish-snapshot", help="Publish one validated staged snapshot atomically."
    )
    publish_snapshot.add_argument("snapshot_id")
    publish_snapshot.add_argument("--operator", required=True)
    publish_snapshot.add_argument("--reason", required=True)

    rollback = commands.add_parser("rollback", help="Repoint a stratum to a prior snapshot.")
    rollback.add_argument("snapshot_id")
    rollback.add_argument("--operator", required=True)
    rollback.add_argument("--reason", required=True)

    enqueue_score = commands.add_parser(
        "enqueue-score", help="Freeze and enqueue a K-by-N incremental raw-score delta."
    )
    enqueue_score.add_argument("--new-version", action="append", required=True)
    enqueue_score.add_argument("--algorithm-hash", required=True)
    enqueue_score.add_argument("--max-attempts", type=int, default=5)

    enqueue_calibration = commands.add_parser(
        "enqueue-calibration",
        help=(
            "Freeze the complete current-version pair family, stage calibration, "
            "and enqueue profile-based p-values/BH."
        ),
    )
    enqueue_calibration.add_argument("--stratum", required=True)
    enqueue_calibration.add_argument("--mode", choices=["exact", "frozen"], required=True)
    enqueue_calibration.add_argument("--pool-hash", required=True)
    enqueue_calibration.add_argument("--parameter-hash", required=True)
    enqueue_calibration.add_argument("--algorithm-hash", required=True)
    enqueue_calibration.add_argument("--profile-kind", required=True)
    enqueue_calibration.add_argument("--max-attempts", type=int, default=5)

    worker = commands.add_parser("worker", help="Run leased, resumable Atlas jobs.")
    worker.add_argument("--worker-id")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--poll-seconds", type=float, default=5.0)
    worker.add_argument("--kind", action="append")

    bridge = commands.add_parser(
        "import-scale-store",
        help="Validate and catalog completed artifacts from the preserved scalable store.",
    )
    bridge.add_argument("--store", required=True)
    bridge.add_argument("--platform", default="GPL570")
    bridge.add_argument(
        "--source-revision",
        default="",
        help="Source revision; derived from --source-archive when omitted.",
    )
    bridge.add_argument(
        "--source-archive",
        help="Preserved ZIP containing source matrices when expr.tsv.gz was intentionally evicted.",
    )
    bridge.add_argument("--dataset", action="append")
    bridge.add_argument("--pool-version")

    release_bridge = commands.add_parser(
        "import-scale-release",
        help="Validate a complete scale run and import scores, overlap, and both BH families.",
    )
    release_bridge.add_argument("--store", required=True)
    release_bridge.add_argument("--run-id", required=True)

    geo_sync = commands.add_parser(
        "sync-geo", help="Resume a policy-compliant, batched GEO metadata synchronization."
    )
    geo_sync.add_argument("--output", default="var/geo")
    geo_sync.add_argument("--accessions-file")
    geo_sync.add_argument("--email", default=os.getenv("NCBI_EMAIL", ""))
    geo_sync.add_argument("--api-key", default=os.getenv("NCBI_API_KEY", ""))
    geo_sync.add_argument("--bootstrap-without-email", action="store_true")
    geo_sync.add_argument(
        "--metadata-source",
        choices=("auto", "eutils", "geo-soft"),
        default="geo-soft",
        help="Use batched E-utilities, GEO's brief SOFT accession view, or automatic fallback.",
    )
    geo_sync.add_argument("--force", action="store_true")
    geo_sync.add_argument(
        "--workers", type=int, default=int(os.getenv("NCBI_SOFT_CONCURRENCY", "4"))
    )

    specter = commands.add_parser(
        "run-specter2", help="Build or resume the pinned all-pairs SPECTER2 text release."
    )
    specter.add_argument("--metadata", default="var/geo")
    specter.add_argument("--output", default="var/specter2/embeddings.npz")
    specter.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    specter.add_argument("--batch-size", type=int, default=0)

    annotate = commands.add_parser(
        "label-geo", help="Resume OpenRouter GEO-first ontology candidate generation."
    )
    annotate.add_argument("--metadata", default="var/geo")
    annotate.add_argument("--output", default="var/annotations/candidates")
    annotate.add_argument(
        "--ontology-cache",
        default="var/annotations/ols-resolver-cache",
        help="Frozen, resumable official OLS response cache for label resolution.",
    )
    annotate.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", ""))
    annotate.add_argument(
        "--workers", type=int, default=int(os.getenv("OPENROUTER_CONCURRENCY", "4"))
    )
    annotate.add_argument("--force", action="store_true")

    ontology_index = commands.add_parser(
        "build-ontology-index",
        help="Freeze official OLS records for every ontology candidate in a directory.",
    )
    ontology_index.add_argument("--annotations", default="var/annotations/candidates")
    ontology_index.add_argument("--output", default="var/annotations/ontology-audit")
    ontology_index.add_argument("--workers", type=int, default=4)
    ontology_index.add_argument("--force", action="store_true")

    ontology_audit = commands.add_parser(
        "audit-annotations",
        help="Check candidate labels against a frozen official OLS term index.",
    )
    ontology_audit.add_argument("--annotations", default="var/annotations/candidates")
    ontology_audit.add_argument("--ontology-index", required=True)
    ontology_audit.add_argument(
        "--output", default="var/annotations/ontology-audit/candidate-audit.json"
    )

    discover = commands.add_parser(
        "discover-geo", help="Discover a bounded window of new GEO CEL series."
    )
    discover.add_argument("--platform", default="GPL570")
    discover.add_argument("--minimum-date", required=True)
    discover.add_argument("--maximum-date", required=True)
    discover.add_argument("--output", default="var/geo/discovery/latest.json")
    discover.add_argument("--email", default=os.getenv("NCBI_EMAIL", ""))
    discover.add_argument("--api-key", default=os.getenv("NCBI_API_KEY", ""))
    discover.add_argument("--bootstrap-without-email", action="store_true")

    raw_download = commands.add_parser(
        "download-geo-raw", help="Resume and verify one discovered GEO RAW CEL archive."
    )
    raw_download.add_argument("--accession", required=True)
    raw_download.add_argument("--url", required=True)
    raw_download.add_argument("--output", default="var/geo/raw")
    raw_download.add_argument(
        "--source-revision",
        default="",
        help="Upstream revision/ETag identity supplied by discovery or the operator.",
    )
    raw_download.add_argument(
        "--expected-sha256",
        default="",
        help="Optional authoritative 64-character SHA-256 for the RAW archive.",
    )
    raw_download.add_argument(
        "--force",
        action="store_true",
        help="Revalidate and redownload instead of reusing a checksum-valid cache entry.",
    )

    reactome = commands.add_parser(
        "build-reactome-index",
        help="Build a versioned local Reactome index against the frozen GPL570 gene universe.",
    )
    reactome.add_argument("--mapping", required=True)
    reactome.add_argument(
        "--annotation", default="resources/releases/gpl570/probe-annotation.tsv"
    )
    reactome.add_argument("--database", default="var/reactome/reactome-gpl570.sqlite")
    reactome.add_argument("--manifest", default="var/reactome/manifest.json")
    reactome.add_argument("--release", required=True)

    explain_edge = commands.add_parser(
        "explain-edge", help="Compute or replay one checksum-bound C-SKL/Reactome explanation."
    )
    explain_edge.add_argument("pair_id")
    explain_edge.add_argument("--probes", default="resources/releases/gpl570/probes.txt")
    explain_edge.add_argument(
        "--annotation", default="resources/releases/gpl570/probe-annotation.tsv"
    )
    explain_edge.add_argument("--reactome", default="var/reactome/reactome-gpl570.sqlite")
    explain_edge.add_argument("--output", default="var/explanations")
    explain_edge.add_argument("--k", type=int, default=20)

    explain_snapshot = commands.add_parser(
        "explain-snapshot",
        help="Resume a bounded, checkpointed explainer batch for one published snapshot.",
    )
    explain_snapshot.add_argument("--snapshot-id", required=True)
    explain_snapshot.add_argument("--probes", default="resources/releases/gpl570/probes.txt")
    explain_snapshot.add_argument(
        "--annotation", default="resources/releases/gpl570/probe-annotation.tsv"
    )
    explain_snapshot.add_argument("--reactome", default="var/reactome/reactome-gpl570.sqlite")
    explain_snapshot.add_argument("--output", default="var/explanations")
    explain_snapshot.add_argument(
        "--report", default="var/explanations/snapshot-batch-report.json"
    )
    explain_snapshot.add_argument("--k", type=int, default=20)
    explain_snapshot.add_argument("--seed", type=int, default=1729)
    explain_snapshot.add_argument("--max-iter", type=int, default=50)
    explain_snapshot.add_argument("--n-init", type=int, default=3)
    explain_snapshot.add_argument(
        "--max-edges", type=int, default=25, help="Maximum cache misses attempted this run."
    )
    explain_snapshot.add_argument(
        "--time-budget-seconds",
        type=float,
        default=3600.0,
        help="Soft wall-clock budget checked between atomic per-edge computations.",
    )

    static_graph = commands.add_parser(
        "export-static-graph", help="Export a sanitized real snapshot for the static product."
    )
    static_graph.add_argument("--snapshot-id", required=True)
    static_graph.add_argument("--metadata", default="var/geo")
    static_graph.add_argument("--output", default="app/data/atlas-graph.json")
    static_graph.add_argument("--manifest", default="app/data/atlas-graph.manifest.json")
    static_graph.add_argument(
        "--ontology-audit",
        help="Frozen ontology-candidate audit JSON to bind into the export.",
    )

    release_audit = commands.add_parser(
        "audit-release",
        help="Run executable operational or manuscript readiness gates.",
    )
    release_audit.add_argument("--snapshot-id", required=True)
    release_audit.add_argument(
        "--profile", choices=["operational", "manuscript"], required=True
    )
    release_audit.add_argument("--metadata")
    release_audit.add_argument("--ontology-audit")
    release_audit.add_argument("--static-manifest")
    release_audit.add_argument("--output")

    serve = commands.add_parser("serve", help="Run the local/reference FastAPI service.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    os.environ["CSKL_ATLAS_CATALOG"] = str(Path(args.catalog).resolve())
    catalog = Catalog(args.catalog)

    if args.command == "init":
        catalog.initialize()
        print(json.dumps(catalog.health(), indent=2))
        return
    if args.command == "health":
        catalog.initialize()
        print(json.dumps(catalog.health(), indent=2))
        return
    if args.command == "jobs":
        catalog.initialize()
        print(json.dumps(catalog.list_jobs(status=args.status, limit=args.limit), indent=2))
        return
    if args.command == "retry":
        catalog.initialize()
        catalog.requeue_job(args.job_id)
        print(json.dumps({"job_id": args.job_id, "status": "queued"}))
        return
    if args.command == "reap":
        catalog.initialize()
        print(json.dumps(catalog.reap_expired_jobs(), indent=2))
        return
    if args.command == "validate-snapshot":
        catalog.initialize()
        report = catalog.validate_snapshot(args.snapshot_id)
        print(json.dumps(report, indent=2))
        if not report["valid"]:
            raise SystemExit(2)
        return
    if args.command == "build-snapshot":
        from .graph_builder import build_graph_snapshot

        catalog.initialize()
        result = build_graph_snapshot(
            catalog,
            calibration_id=args.calibration_id,
            manifest_directory=args.manifest_directory,
            q_max=args.q_max,
            independent_only=not args.include_overlap_confounded,
            top_k_per_node=args.top_k_per_node,
            resolution=args.resolution,
            seed=args.seed,
            stability_runs=args.stability_runs,
            text_release_id=args.text_release_id,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "publish-snapshot":
        catalog.initialize()
        catalog.publish_snapshot(
            args.snapshot_id, operator=args.operator, reason=args.reason
        )
        print(json.dumps({"snapshot_id": args.snapshot_id, "status": "published"}))
        return
    if args.command == "rollback":
        catalog.initialize()
        catalog.rollback_snapshot(
            args.snapshot_id, operator=args.operator, reason=args.reason
        )
        print(json.dumps({"snapshot_id": args.snapshot_id, "status": "published"}))
        return
    if args.command == "enqueue-score":
        from .worker import enqueue_incremental_score_job

        catalog.initialize()
        job_id = enqueue_incremental_score_job(
            catalog,
            new_version_ids=args.new_version,
            algorithm_hash=args.algorithm_hash,
            max_attempts=args.max_attempts,
        )
        print(json.dumps({"job_id": job_id, "status": "queued"}))
        return
    if args.command == "enqueue-calibration":
        from .worker import enqueue_calibration_job

        catalog.initialize()
        calibration_id = catalog.stage_current_calibration(
            stratum=args.stratum,
            mode=args.mode,
            pool_hash=args.pool_hash,
            parameter_hash=args.parameter_hash,
            algorithm_hash=args.algorithm_hash,
            manifest={"profile_kind": args.profile_kind},
        )
        job_id = enqueue_calibration_job(
            catalog,
            calibration_id=calibration_id,
            profile_kind=args.profile_kind,
            max_attempts=args.max_attempts,
        )
        print(json.dumps({"job_id": job_id, "calibration_id": calibration_id, "status": "queued"}))
        return
    if args.command == "worker":
        from .worker import run_worker

        catalog.initialize()
        results = run_worker(
            catalog,
            worker_id=args.worker_id,
            once=args.once,
            poll_seconds=args.poll_seconds,
            kinds=args.kind,
        )
        if args.once:
            print(json.dumps(results, indent=2))
        return
    if args.command == "import-scale-store":
        from .scale_bridge import import_scale_store

        catalog.initialize()
        result = import_scale_store(
            catalog,
            store_root=args.store,
            platform=args.platform,
            source_revision=args.source_revision,
            dataset_ids=args.dataset,
            pool_version=args.pool_version,
            source_archive=args.source_archive,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "import-scale-release":
        from .scale_bridge import import_scale_release

        catalog.initialize()
        result = import_scale_release(catalog, store_root=args.store, run_id=args.run_id)
        print(json.dumps(result, indent=2))
        return
    if args.command == "sync-geo":
        from .geo_acquisition import NcbiSettings, sync_geo_metadata

        catalog.initialize()
        if args.accessions_file:
            accessions = [
                line.strip() for line in Path(args.accessions_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            with catalog.reader() as connection:
                accessions = [
                    row["accession"]
                    for row in connection.execute(
                        "SELECT accession FROM datasets WHERE current_version_id IS NOT NULL ORDER BY accession"
                    )
                ]
        result = sync_geo_metadata(
            catalog,
            accessions=accessions,
            output_directory=args.output,
            settings=NcbiSettings(
                email=args.email,
                api_key=args.api_key,
                allow_missing_email=(
                    args.bootstrap_without_email or args.metadata_source == "geo-soft"
                ),
            ),
            metadata_source=args.metadata_source,
            force=args.force,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2))
        if result["operator_required"]:
            raise SystemExit(3)
        return
    if args.command == "run-specter2":
        from .specter2_release import build_specter2_release

        catalog.initialize()
        result = build_specter2_release(
            catalog,
            metadata_directory=args.metadata,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "label-geo":
        from .annotation_pipeline import run_annotation_pipeline

        catalog.initialize()
        result = run_annotation_pipeline(
            catalog,
            metadata_directory=args.metadata,
            output_directory=args.output,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model=args.model,
            force=args.force,
            workers=args.workers,
            ontology_cache_directory=args.ontology_cache,
        )
        print(json.dumps(result, indent=2))
        if result["operator_required"]:
            raise SystemExit(3)
        return
    if args.command == "build-ontology-index":
        from .ontology_validation import build_ols_candidate_index

        result = build_ols_candidate_index(
            annotation_directory=args.annotations,
            output_directory=args.output,
            workers=args.workers,
            force=args.force,
        )
        print(json.dumps(result, indent=2))
        if result["operator_required"]:
            raise SystemExit(3)
        return
    if args.command == "audit-annotations":
        from .ontology_validation import audit_annotation_candidates

        result = audit_annotation_candidates(
            annotation_directory=args.annotations,
            ontology_index=args.ontology_index,
            output_path=args.output,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
        if result["operator_required"]:
            raise SystemExit(3)
        return
    if args.command == "discover-geo":
        from .geo_acquisition import NcbiSettings, discover_geo_updates

        catalog.initialize()
        result = discover_geo_updates(
            catalog,
            output_path=args.output,
            settings=NcbiSettings(
                email=args.email,
                api_key=args.api_key,
                allow_missing_email=args.bootstrap_without_email,
            ),
            platform=args.platform,
            minimum_date=args.minimum_date,
            maximum_date=args.maximum_date,
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "records"}, indent=2
            )
        )
        return
    if args.command == "download-geo-raw":
        from .geo_acquisition import download_geo_raw_tar

        result = download_geo_raw_tar(
            accession=args.accession,
            raw_tar_url=args.url,
            output_directory=args.output,
            source_revision=args.source_revision,
            expected_sha256=args.expected_sha256,
            force=args.force,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "build-reactome-index":
        from .catalog import stable_id
        from .pathways import build_reactome_index

        catalog.initialize()
        result = build_reactome_index(
            mapping_path=args.mapping,
            annotation_path=args.annotation,
            database_path=args.database,
            manifest_path=args.manifest,
            release=args.release,
        )
        catalog.record_artifact(
            artifact_id=stable_id("artifact", "reactome_index", result["dependency_hash"]),
            kind="reactome_index",
            uri=result["database_uri"],
            checksum=result["index_checksum"],
            dependency_hash=result["dependency_hash"],
            manifest=result,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "explain-edge":
        from .edge_explanation import compute_edge_explanation

        catalog.initialize()
        result = compute_edge_explanation(
            catalog,
            pair_id=args.pair_id,
            probes_path=args.probes,
            annotation_path=args.annotation,
            reactome_database_path=args.reactome,
            cache_directory=args.output,
            k=args.k,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "explain-snapshot":
        from .edge_explanation import explain_snapshot_edges

        catalog.initialize()
        result = explain_snapshot_edges(
            catalog,
            snapshot_id=args.snapshot_id,
            probes_path=args.probes,
            annotation_path=args.annotation,
            reactome_database_path=args.reactome,
            cache_directory=args.output,
            report_path=args.report,
            k=args.k,
            seed=args.seed,
            max_iter=args.max_iter,
            n_init=args.n_init,
            max_edges=args.max_edges,
            time_budget_seconds=args.time_budget_seconds,
        )
        print(json.dumps(result, indent=2))
        if result["status"] == "operator_required":
            raise SystemExit(3)
        return
    if args.command == "export-static-graph":
        from .static_export import export_static_graph

        catalog.initialize()
        result = export_static_graph(
            catalog,
            snapshot_id=args.snapshot_id,
            metadata_directory=args.metadata,
            output_path=args.output,
            manifest_path=args.manifest,
            ontology_audit_path=args.ontology_audit,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "audit-release":
        from cskl_pipeline.scale.store import atomic_write_json

        from .release_audit import audit_release

        catalog.initialize()
        result = audit_release(
            catalog,
            snapshot_id=args.snapshot_id,
            profile=args.profile,
            metadata_directory=args.metadata,
            ontology_audit_path=args.ontology_audit,
            static_manifest_path=args.static_manifest,
        )
        if args.output:
            atomic_write_json(Path(args.output).resolve(), result)
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(4)
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run("cskl_atlas.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
