"""On-demand, checksum-bound C-SKL feature and Reactome edge explanations."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cskl
import numpy as np
from cskl_pipeline.scale.store import load_signature

from .catalog import Catalog, canonical_json, stable_id
from .pathways import enrich_reactome

EXPLANATION_SCHEMA = "cskl-edge-explanation-v2"
BATCH_REPORT_SCHEMA = "cskl-edge-explanation-batch-v1"


class ExplanationNotCachedError(LookupError):
    """Raised when a read-only replay has no matching cataloged artifact."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_probe_mapping(path: Path) -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"PROBEID", "ENTREZID", "SYMBOL", "GENENAME"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Probe annotation is missing {sorted(required)}.")
        for row in reader:
            probe = str(row["PROBEID"]).strip()
            gene_id = str(row["ENTREZID"]).strip()
            if not probe or not gene_id:
                continue
            record = {
                "gene_id": gene_id,
                "symbol": str(row["SYMBOL"]).strip(),
                "name": str(row["GENENAME"]).strip(),
            }
            if record not in mapping.setdefault(probe, []):
                mapping[probe].append(record)
    return mapping


def _atomic_json(path: Path, value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    checksum = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                # Windows can briefly deny concurrent replacement of the same
                # destination even though every producer owns a unique temp.
                time.sleep(0.005 * (2**attempt))
    finally:
        temporary.unlink(missing_ok=True)
    return checksum


def _implementation_checksum() -> str:
    """Bind outputs to every local implementation that affects their values."""

    producer_source = "\n".join(
        inspect.getsource(implementation)
        for implementation in (
            compute_edge_explanation,
            _load_probe_mapping,
            _feature_records,
        )
    )
    sources: dict[str, str] = {
        "edge_explanation": hashlib.sha256(producer_source.encode("utf-8")).hexdigest()
    }
    for label, implementation in (
        ("cskl", cskl.explain_topk),
        ("signature_loader", load_signature),
        ("reactome_enrichment", enrich_reactome),
    ):
        source = inspect.getsourcefile(implementation)
        if not source:
            raise RuntimeError(f"Cannot checksum the {label} implementation source.")
        sources[label] = _sha256_path(Path(source).resolve())
    return hashlib.sha256(canonical_json(sources).encode("utf-8")).hexdigest()


def _artifact_rows(
    catalog: Catalog,
    *,
    dependency_hash: str | None = None,
    pair_id: str | None = None,
) -> list[Any]:
    clauses = ["kind='edge_explanation'"]
    parameters: list[str] = []
    if dependency_hash is not None:
        clauses.append("dependency_hash=?")
        parameters.append(dependency_hash)
    if pair_id is not None:
        clauses.append("json_extract(manifest_json,'$.pair_id')=?")
        parameters.append(pair_id)
    with catalog.reader() as connection:
        return connection.execute(
            f"""SELECT uri,checksum,dependency_hash,manifest_json,created_at
                FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY created_at DESC""",
            parameters,
        ).fetchall()


def _validated_cached_payload(
    artifact: Any,
    *,
    pair_id: str,
    k: int,
    expected_dependency_hash: str | None = None,
) -> dict[str, Any]:
    """Validate both catalog provenance and cache bytes before any replay."""

    try:
        manifest = json.loads(artifact["manifest_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Explainer artifact has an invalid catalog manifest.") from exc
    if manifest.get("pair_id") != pair_id or manifest.get("schema") != EXPLANATION_SCHEMA:
        raise ValueError("Explainer artifact catalog identity does not match the request.")
    if manifest.get("k") is not None and manifest.get("k") != k:
        raise ValueError("Explainer artifact catalog parameters do not match the requested k.")
    dependency_hash = str(artifact["dependency_hash"])
    if expected_dependency_hash is not None and dependency_hash != expected_dependency_hash:
        raise ValueError("Explainer artifact dependency hash does not match the request.")
    path = Path(str(artifact["uri"])).resolve()
    if not path.is_file():
        raise ValueError("Cataloged explainer cache file is missing.")
    actual_checksum = _sha256_path(path)
    if actual_checksum != artifact["checksum"]:
        raise ValueError("Cataloged explainer cache checksum mismatch.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cataloged explainer cache is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Cataloged explainer cache must contain a JSON object.")
    if payload.get("schema") != EXPLANATION_SCHEMA:
        raise ValueError("Explainer cache schema does not match the supported schema.")
    if payload.get("pair_id") != pair_id:
        raise ValueError("Explainer cache pair does not match the requested relationship.")
    if payload.get("dependency_hash") != dependency_hash:
        raise ValueError("Explainer cache dependency hash does not match its catalog record.")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("k") != k:
        raise ValueError("Explainer cache parameters do not match the requested k.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("pair_id") != pair_id:
        raise ValueError("Explainer cache provenance does not match the requested relationship.")
    provenance_hash = hashlib.sha256(canonical_json(provenance).encode("utf-8")).hexdigest()
    if provenance_hash != dependency_hash:
        raise ValueError("Explainer cache provenance does not reproduce its dependency hash.")
    return payload


def _replay_from_rows(
    rows: list[Any],
    *,
    pair_id: str,
    k: int,
    dependency_hash: str | None = None,
) -> dict[str, Any]:
    candidates: list[Any] = []
    for artifact in rows:
        try:
            manifest = json.loads(artifact["manifest_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Explainer artifact has an invalid catalog manifest.") from exc
        if manifest.get("pair_id") != pair_id:
            continue
        if manifest.get("k") is not None and manifest.get("k") != k:
            continue
        if dependency_hash is not None and artifact["dependency_hash"] != dependency_hash:
            continue
        candidates.append(artifact)
    if not candidates:
        raise ExplanationNotCachedError(pair_id)
    for artifact in candidates:
        try:
            return _validated_cached_payload(
                artifact,
                pair_id=pair_id,
                k=k,
                expected_dependency_hash=dependency_hash,
            )
        except ValueError as exc:
            if "parameters do not match" in str(exc):
                continue
            raise
    raise ExplanationNotCachedError(pair_id)


def replay_edge_explanation(catalog: Catalog, *, pair_id: str, k: int = 20) -> dict[str, Any]:
    """Replay a cataloged artifact without starting scientific computation."""

    if not 1 <= k <= 500:
        raise ValueError("k must be between 1 and 500.")
    return _replay_from_rows(
        _artifact_rows(catalog, pair_id=pair_id), pair_id=pair_id, k=k
    )


def _feature_records(
    indices: np.ndarray,
    scores: np.ndarray,
    probes: list[str],
    mapping: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, score in zip(indices, scores, strict=True):
        probe = probes[int(index)]
        genes = mapping.get(probe, [])
        records.append(
            {
                "feature": probe,
                "linearized_score": float(score),
                "genes": genes,
                "mapping_ambiguous": len(genes) > 1,
            }
        )
    return records


def compute_edge_explanation(
    catalog: Catalog,
    *,
    pair_id: str,
    probes_path: str | Path,
    annotation_path: str | Path,
    reactome_database_path: str | Path,
    cache_directory: str | Path,
    k: int = 20,
    seed: int = 1729,
    max_iter: int = 50,
    n_init: int = 3,
) -> dict[str, Any]:
    """Compute or replay one paper-faithful B(k)/W(k) explanation."""

    if not 1 <= k <= 500:
        raise ValueError("k must be between 1 and 500.")
    probes_file = Path(probes_path).resolve()
    annotation_file = Path(annotation_path).resolve()
    reactome_file = Path(reactome_database_path).resolve()
    for required in (probes_file, annotation_file, reactome_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    with catalog.reader() as connection:
        pair = connection.execute(
            """SELECT p.*,da.accession AS accession_a,db.accession AS accession_b
               FROM pair_scores p
               JOIN dataset_versions va ON va.version_id=p.version_a
               JOIN dataset_versions vb ON vb.version_id=p.version_b
               JOIN datasets da ON da.dataset_uid=va.dataset_uid
               JOIN datasets db ON db.dataset_uid=vb.dataset_uid
               WHERE p.pair_id=?""",
            (pair_id,),
        ).fetchone()
        if not pair:
            raise KeyError(pair_id)
        artifacts = connection.execute(
            """SELECT dataset_version_id,uri,checksum FROM artifacts
               WHERE kind='pca_signature' AND dataset_version_id IN (?,?)""",
            (pair["version_a"], pair["version_b"]),
        ).fetchall()
    by_version = {row["dataset_version_id"]: row for row in artifacts}
    if set(by_version) != {pair["version_a"], pair["version_b"]}:
        raise ValueError("Both catalog-bound signature artifacts are required.")
    signature_paths: list[Path] = []
    signature_checksums: list[str] = []
    for version_id in (pair["version_a"], pair["version_b"]):
        artifact = by_version[version_id]
        path = Path(artifact["uri"]).resolve()
        actual = _sha256_path(path)
        if actual != artifact["checksum"]:
            raise ValueError(f"Signature checksum mismatch for {version_id}.")
        signature_paths.append(path)
        signature_checksums.append(actual)

    probes = [line.strip() for line in probes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not probes:
        raise ValueError("Frozen probe order is empty.")
    settings = {
        "schema": EXPLANATION_SCHEMA,
        "pair_id": pair_id,
        "signature_checksums": signature_checksums,
        "probe_checksum": _sha256_path(probes_file),
        "annotation_checksum": _sha256_path(annotation_file),
        "reactome_index_checksum": _sha256_path(reactome_file),
        "implementation_checksum": _implementation_checksum(),
        "k": k,
        "seed": seed,
        "max_iter": max_iter,
        "n_init": n_init,
    }
    dependency_hash = hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()
    cataloged = _artifact_rows(
        catalog, dependency_hash=dependency_hash, pair_id=pair_id
    )
    if cataloged:
        return _replay_from_rows(
            cataloged,
            pair_id=pair_id,
            k=k,
            dependency_hash=dependency_hash,
        )
    output = Path(cache_directory).resolve() / f"{pair_id}-{dependency_hash[:16]}.json"
    if output.is_file():
        # Recover the narrow crash window between the atomic file replace and
        # the catalog transaction, but never trust the orphaned bytes directly.
        checksum = _sha256_path(output)
        orphan = {
            "uri": str(output),
            "checksum": checksum,
            "dependency_hash": dependency_hash,
            "manifest_json": canonical_json(
                {"pair_id": pair_id, "schema": settings["schema"], "k": k}
            ),
        }
        payload = _validated_cached_payload(
            orphan,
            pair_id=pair_id,
            k=k,
            expected_dependency_hash=dependency_hash,
        )
        catalog.record_artifact(
            artifact_id=stable_id("artifact", "edge_explanation", dependency_hash),
            kind="edge_explanation",
            uri=str(output),
            checksum=checksum,
            dependency_hash=dependency_hash,
            manifest={
                "pair_id": pair_id,
                "schema": settings["schema"],
                "k": k,
                "annotation_checksum": settings["annotation_checksum"],
                "reactome_index_checksum": settings["reactome_index_checksum"],
                "implementation_checksum": settings["implementation_checksum"],
            },
        )
        return payload

    signature_a = load_signature(signature_paths[0], feature_names=probes)
    signature_b = load_signature(signature_paths[1], feature_names=probes)
    if signature_a.n_features != len(probes) or signature_b.n_features != len(probes):
        raise ValueError("Signature and frozen probe universe sizes differ.")
    b_indices, b_scores, b_details = cskl.explain_topk(
        signature_a,
        signature_b,
        k,
        mode="B",
        seed=seed,
        max_iter=max_iter,
        n_init=n_init,
        return_scores=True,
        return_details=True,
    )
    w_indices, w_scores, w_details = cskl.explain_topk(
        signature_a,
        signature_b,
        k,
        mode="W",
        seed=seed,
        max_iter=max_iter,
        n_init=n_init,
        return_scores=True,
        return_details=True,
    )
    mapping = _load_probe_mapping(annotation_file)
    b_features = _feature_records(b_indices, b_scores, probes, mapping)
    w_features = _feature_records(w_indices, w_scores, probes, mapping)

    trajectory: list[dict[str, float | int]] = []
    trajectory_sizes = sorted({value for value in (1, 5, 10, 20, 50, k) if value <= k})
    rng = np.random.default_rng(seed)
    coefficients = signature_a.lam[:, None] + signature_b.lam[None, :]

    def objective(indices: np.ndarray) -> float:
        mask = np.zeros(len(probes), dtype=float)
        mask[indices] = 1.0
        cross = (signature_a.P * mask[:, None]).T @ (signature_b.P * mask[:, None])
        return float(np.sum(coefficients * cross * cross))

    for size in trajectory_sizes:
        if size == k:
            best_value = float(b_details["f"])
            worst_value = float(w_details["f"])
        else:
            _, best_details = cskl.explain_topk(
                signature_a,
                signature_b,
                size,
                mode="B",
                seed=seed,
                max_iter=max_iter,
                n_init=n_init,
                return_details=True,
            )
            _, worst_details = cskl.explain_topk(
                signature_a,
                signature_b,
                size,
                mode="W",
                seed=seed,
                max_iter=max_iter,
                n_init=n_init,
                return_details=True,
            )
            best_value = float(best_details["f"])
            worst_value = float(worst_details["f"])
        random_values = [
            objective(rng.choice(len(probes), size=size, replace=False)) for _ in range(20)
        ]
        trajectory.append(
            {
                "k": size,
                "best_objective": best_value,
                "worst_objective": worst_value,
                "random_objective": float(np.mean(random_values)),
            }
        )

    def gene_ids(features: list[dict[str, Any]]) -> list[str]:
        return sorted({gene["gene_id"] for item in features for gene in item["genes"]})

    payload = {
        "schema": settings["schema"],
        "pair_id": pair_id,
        "datasets": [pair["accession_a"], pair["accession_b"]],
        "algorithm_hash": pair["algorithm_hash"],
        "raw_cskl": pair["cskl"],
        "parameters": {"k": k, "seed": seed, "max_iter": max_iter, "n_init": n_init},
        "trajectory": trajectory,
        "dependency_hash": dependency_hash,
        "best_explaining": {
            "label": "B(k): features whose joint covariance alignment best retains similarity",
            "objective": float(b_details["f"]),
            "features": b_features,
            "reactome": enrich_reactome(gene_ids(b_features), database_path=reactome_file),
        },
        "most_differentiating": {
            "label": "W(k): features with the weakest joint covariance alignment",
            "objective": float(w_details["f"]),
            "features": w_features,
            "reactome": enrich_reactome(gene_ids(w_features), database_path=reactome_file),
        },
        "interpretation": (
            "The feature sets optimize the paper's non-convex covariance-alignment objective. "
            "Scores are linearized optimizer scores, not additive effects, differential expression, "
            "causal contributions, or percentages. Pathways are over-representation hypotheses."
        ),
        "provenance": settings,
    }
    checksum = _atomic_json(output, payload)
    catalog.record_artifact(
        artifact_id=stable_id("artifact", "edge_explanation", dependency_hash),
        kind="edge_explanation",
        uri=str(output),
        checksum=checksum,
        dependency_hash=dependency_hash,
        manifest={
            "pair_id": pair_id,
            "schema": settings["schema"],
            "k": k,
            "annotation_checksum": settings["annotation_checksum"],
            "reactome_index_checksum": settings["reactome_index_checksum"],
            "implementation_checksum": settings["implementation_checksum"],
        },
    )
    return payload


def explain_snapshot_edges(
    catalog: Catalog,
    *,
    snapshot_id: str,
    probes_path: str | Path,
    annotation_path: str | Path,
    reactome_database_path: str | Path,
    cache_directory: str | Path,
    report_path: str | Path,
    k: int = 20,
    seed: int = 1729,
    max_iter: int = 50,
    n_init: int = 3,
    max_edges: int = 25,
    time_budget_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Compute a bounded snapshot slice, resuming from cataloged per-edge caches."""

    if not 1 <= k <= 500:
        raise ValueError("k must be between 1 and 500.")
    if not 1 <= max_edges <= 100_000:
        raise ValueError("max_edges must be between 1 and 100000.")
    if not 1.0 <= time_budget_seconds <= 7 * 24 * 60 * 60:
        raise ValueError("time_budget_seconds must be between 1 and 604800.")
    if max_iter < 1 or n_init < 1:
        raise ValueError("max_iter and n_init must both be positive.")

    with catalog.reader() as connection:
        snapshot = connection.execute(
            "SELECT snapshot_id,status,published_at,calibration_id FROM graph_snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if not snapshot:
            raise KeyError(snapshot_id)
        if snapshot["published_at"] is None:
            raise ValueError("Snapshot explainer batches require a published snapshot.")
        pair_rows = connection.execute(
            """SELECT se.pair_id,c.q_value,p.cskl
               FROM graph_snapshot_edges se
               JOIN graph_snapshots s ON s.snapshot_id=se.snapshot_id
               JOIN calibrated_edges c
                 ON c.calibration_id=s.calibration_id AND c.pair_id=se.pair_id
               JOIN pair_scores p ON p.pair_id=se.pair_id
               WHERE se.snapshot_id=?
               ORDER BY c.q_value ASC,p.cskl ASC,se.pair_id ASC""",
            (snapshot_id,),
        ).fetchall()

    cached_by_pair: dict[str, list[Any]] = {}
    catalog_warnings: list[str] = []
    for artifact in _artifact_rows(catalog):
        try:
            manifest = json.loads(artifact["manifest_json"])
        except (TypeError, json.JSONDecodeError):
            catalog_warnings.append("An edge_explanation artifact has invalid manifest JSON.")
            continue
        pair_id = manifest.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            cached_by_pair.setdefault(pair_id, []).append(artifact)

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    destination = Path(report_path).resolve()
    report: dict[str, Any] = {
        "schema": BATCH_REPORT_SCHEMA,
        "snapshot_id": snapshot_id,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "parameters": {
            "k": k,
            "seed": seed,
            "max_iter": max_iter,
            "n_init": n_init,
            "probes": str(Path(probes_path).resolve()),
            "annotation": str(Path(annotation_path).resolve()),
            "reactome_index": str(Path(reactome_database_path).resolve()),
            "cache_directory": str(Path(cache_directory).resolve()),
        },
        "limits": {
            "max_edges": max_edges,
            "time_budget_seconds": time_budget_seconds,
            "time_budget_enforcement": "between_edges",
        },
        "snapshot_edge_count": len(pair_rows),
        "cached_count": 0,
        "computed_count": 0,
        "failed_count": 0,
        "pending_count": 0,
        "attempted_count": 0,
        "elapsed_seconds": 0.0,
        "catalog_warnings": catalog_warnings,
        "results": [],
    }
    _atomic_json(destination, report)

    def checkpoint() -> None:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _atomic_json(destination, report)

    for pair_row in pair_rows:
        pair_id = str(pair_row["pair_id"])
        candidates = cached_by_pair.get(pair_id, [])
        if candidates:
            try:
                _replay_from_rows(candidates, pair_id=pair_id, k=k)
            except ExplanationNotCachedError:
                pass
            except (OSError, ValueError) as exc:
                report["attempted_count"] += 1
                report["failed_count"] += 1
                report["results"].append(
                    {
                        "pair_id": pair_id,
                        "status": "failed_cache_validation",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                )
                checkpoint()
                continue
            else:
                report["cached_count"] += 1
                continue

        if (
            report["attempted_count"] >= max_edges
            or time.monotonic() - started >= time_budget_seconds
        ):
            report["pending_count"] += 1
            continue
        report["attempted_count"] += 1
        edge_started = time.monotonic()
        try:
            payload = compute_edge_explanation(
                catalog,
                pair_id=pair_id,
                probes_path=probes_path,
                annotation_path=annotation_path,
                reactome_database_path=reactome_database_path,
                cache_directory=cache_directory,
                k=k,
                seed=seed,
                max_iter=max_iter,
                n_init=n_init,
            )
        except Exception as exc:  # continue the bounded batch and preserve an operator report
            report["failed_count"] += 1
            report["results"].append(
                {
                    "pair_id": pair_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "elapsed_seconds": round(time.monotonic() - edge_started, 3),
                }
            )
        else:
            report["computed_count"] += 1
            report["results"].append(
                {
                    "pair_id": pair_id,
                    "status": "computed",
                    "dependency_hash": payload["dependency_hash"],
                    "elapsed_seconds": round(time.monotonic() - edge_started, 3),
                }
            )
        checkpoint()

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    if report["failed_count"] or catalog_warnings:
        report["status"] = "operator_required"
    elif report["pending_count"]:
        report["status"] = "bounded"
    else:
        report["status"] = "complete"
    checkpoint()
    return report
