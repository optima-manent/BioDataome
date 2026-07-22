"""Validated bridge from the preserved scalable file store into Atlas."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

from cskl_pipeline.scale.store import Store, read_json, read_signature_meta
from cskl_pipeline.scale.tabio import read_table

from .catalog import Catalog, canonical_json, pair_family_hash, stable_id
from .source_archive import ArchivedSource, index_zip_sources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def import_scale_store(
    catalog: Catalog,
    *,
    store_root: str | Path,
    platform: str,
    source_revision: str,
    dataset_ids: Sequence[str] | None = None,
    pool_version: str | None = None,
    source_archive: str | Path | None = None,
) -> dict[str, Any]:
    """Catalog complete legacy artifacts without recomputing scientific work.

    Only datasets with a valid signature, normalized matrix, and consistent
    sample-hash count are promoted. Quarantined or incomplete datasets are
    reported and left untouched. Existing superseded versions are never made
    current merely because an import is replayed.
    """

    if not platform.strip():
        raise ValueError("platform is required")
    store = Store(store_root)
    feature_hash = store.feature_hash(platform)
    requested = sorted(set(dataset_ids or store.list_datasets()))
    if dataset_ids is not None and len(requested) != len(dataset_ids):
        raise ValueError("dataset_ids must be unique")
    archived_sources: dict[str, ArchivedSource] = {}
    archive_index_hash: str | None = None
    if source_archive is not None:
        eligible_for_archive = [
            accession for accession in requested if not store.is_quarantined(accession)
        ]
        archive_index_hash, archived_sources = index_zip_sources(
            source_archive,
            cache_directory=store.root / "source_manifests",
            accessions=eligible_for_archive,
        )
        if not source_revision.strip():
            source_revision = f"zip-central:{archive_index_hash}"
    if not source_revision.strip():
        raise ValueError("source_revision is required when source_archive is not supplied")

    results: list[dict[str, Any]] = []
    for accession in requested:
        if store.is_quarantined(accession):
            results.append({"accession": accession, "status": "skipped", "reason": "quarantined"})
            continue
        signature_path = store.signature_path(accession)
        normalized_path = store.expr_path(accession)
        samples_path = store.sample_hashes_path(accession)
        archive_source = archived_sources.get(accession)
        if not store.has_valid_signature(accession, platform) or (
            not normalized_path.is_file() and archive_source is None
        ):
            results.append({"accession": accession, "status": "skipped", "reason": "incomplete_artifacts"})
            continue
        signature_meta = read_signature_meta(signature_path)
        sample_hashes = read_json(samples_path) if samples_path.is_file() else None
        if (
            not isinstance(sample_hashes, dict)
            or sample_hashes.get("n_samples") != signature_meta["m_samples"]
            or not isinstance(sample_hashes.get("hashes"), list)
            or len(sample_hashes["hashes"]) != signature_meta["m_samples"]
        ):
            results.append({"accession": accession, "status": "skipped", "reason": "invalid_sample_hashes"})
            continue
        if archive_source and len(archive_source.sample_ids) != signature_meta["m_samples"]:
            results.append(
                {
                    "accession": accession,
                    "status": "skipped",
                    "reason": "source_header_sample_count_mismatch",
                }
            )
            continue
        normalized_hash = (
            _sha256(normalized_path) if normalized_path.is_file() else archive_source.checksum
        )
        signature_hash = _sha256(signature_path)
        config = {
            "bridge_contract": "scale-store-v2",
            "platform": platform,
            "feature_hash": feature_hash,
            "alpha": signature_meta["alpha"],
        }
        config_hash = hashlib.sha256(canonical_json(config).encode()).hexdigest()
        with catalog.reader() as connection:
            previous = connection.execute(
                """SELECT d.current_version_id FROM datasets d
                   WHERE d.accession=? AND d.platform=? AND d.cohort='series'""",
                (accession.upper(), platform.upper()),
            ).fetchone()
            known_versions = {
                row["version_id"]
                for row in connection.execute(
                    """SELECT v.version_id FROM dataset_versions v JOIN datasets d USING(dataset_uid)
                       WHERE d.accession=? AND d.platform=? AND v.source_revision=?
                         AND v.source_hash=? AND v.config_hash=?""",
                    (
                        accession.upper(), platform.upper(), source_revision,
                        normalized_hash, config_hash,
                    ),
                )
            }
        metadata = {
            "import_source": "preserved_source_archive" if archive_source else "legacy_scale_store",
            "signature": signature_meta,
        }
        qc_path = store.qc_path(accession)
        if qc_path.is_file():
            metadata["qc"] = read_json(qc_path)
        dataset_uid, version_id = catalog.register_dataset_version(
            accession=accession,
            platform=platform,
            cohort="series",
            source_revision=source_revision,
            source_hash=normalized_hash,
            normalized_hash=normalized_hash,
            signature_hash=signature_hash,
            feature_hash=feature_hash,
            config_hash=config_hash,
            sample_count=signature_meta["m_samples"],
            metadata=metadata,
        )
        normalized_uri = archive_source.uri if archive_source else str(normalized_path.resolve())
        normalized_manifest = {
            "bridge_contract": "scale-store-v2",
            "source_revision": source_revision,
        }
        if archive_source:
            normalized_manifest.update(
                {
                    "archive_index_hash": archive_index_hash,
                    "archive_member": archive_source.member,
                    "archive_member_size": archive_source.size,
                    "archive_member_crc32": archive_source.crc32,
                }
            )
        for kind, uri, checksum, artifact_manifest in (
            ("normalized_matrix", normalized_uri, normalized_hash, normalized_manifest),
            (
                "pca_signature",
                str(signature_path.resolve()),
                signature_hash,
                {"bridge_contract": "scale-store-v2", "source_revision": source_revision},
            ),
        ):
            catalog.record_artifact(
                artifact_id=hashlib.sha256(f"{version_id}:{kind}:{checksum}".encode()).hexdigest(),
                kind=kind,
                uri=uri,
                checksum=checksum,
                dependency_hash=hashlib.sha256(f"{version_id}:{kind}:{config_hash}".encode()).hexdigest(),
                manifest=artifact_manifest,
                dataset_version_id=version_id,
            )
        if pool_version and store.has_valid_profile(accession, platform, pool_version):
            profile_path = store.null_profile_path(accession, pool_version)
            profile_checksum = _sha256(profile_path)
            profile_kind = f"null_profile:{pool_version}"
            catalog.record_artifact(
                artifact_id=hashlib.sha256(
                    f"{version_id}:{profile_kind}:{profile_checksum}".encode()
                ).hexdigest(),
                kind=profile_kind,
                uri=str(profile_path.resolve()),
                checksum=profile_checksum,
                dependency_hash=hashlib.sha256(
                    f"{version_id}:{profile_kind}:{store.pool_hash(platform, pool_version)}".encode()
                ).hexdigest(),
                manifest={
                    "bridge_contract": "scale-store-v2",
                    "pool_version": pool_version,
                    "pool_hash": store.pool_hash(platform, pool_version),
                },
                dataset_version_id=version_id,
            )
        catalog.replace_samples(
            version_id,
            (
                {
                    "gsm": archive_source.sample_ids[position] if archive_source else None,
                    "expression_hash": value,
                }
                for position, value in enumerate(sample_hashes["hashes"])
            ),
        )
        current = previous["current_version_id"] if previous else None
        is_replay_of_superseded = version_id in known_versions and current not in {None, version_id}
        if not is_replay_of_superseded:
            catalog.promote_dataset_version(version_id)
        results.append(
            {
                "accession": accession,
                "dataset_uid": dataset_uid,
                "version_id": version_id,
                "status": "retained_superseded" if is_replay_of_superseded else "promoted",
            }
        )
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    orphan_samples_pruned = catalog.prune_orphan_samples()
    return {
        "store_root": str(store.root),
        "platform": platform,
        "source_revision": source_revision,
        "feature_hash": feature_hash,
        "pool_version": pool_version,
        "source_archive": str(Path(source_archive).resolve()) if source_archive else None,
        "archive_index_hash": archive_index_hash,
        "orphan_samples_pruned": orphan_samples_pruned,
        "counts": counts,
        "results": results,
    }


def _calibration_status(catalog: Catalog, calibration_id: str) -> str:
    with catalog.reader() as connection:
        row = connection.execute(
            "SELECT status FROM calibration_releases WHERE calibration_id=?",
            (calibration_id,),
        ).fetchone()
    if not row:
        raise KeyError(calibration_id)
    return str(row["status"])


def _import_calibration(
    catalog: Catalog,
    *,
    stratum: str,
    pool_hash: str,
    parameter_hash: str,
    algorithm_hash: str,
    pair_rows: list[tuple[str, float]],
    manifest: dict[str, Any],
) -> str:
    family_hash = pair_family_hash(pair_id for pair_id, _ in pair_rows)
    calibration_id = catalog.stage_calibration(
        stratum=stratum,
        mode="frozen",
        pool_hash=pool_hash,
        parameter_hash=parameter_hash,
        algorithm_hash=algorithm_hash,
        family_hash=family_hash,
        expected_pair_count=len(pair_rows),
        manifest=manifest,
    )
    status = _calibration_status(catalog, calibration_id)
    if status == "staging":
        catalog.record_pvalues(calibration_id, pair_rows)
        catalog.finalize_bh(calibration_id)
    elif status not in {"calibrated", "published"}:
        raise ValueError(f"Calibration {calibration_id} is not recoverable from status {status}")
    return calibration_id


def import_scale_release(
    catalog: Catalog,
    *,
    store_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Validate and import a complete scale run into the Atlas control plane.

    The bridge never trusts exported q-values: it imports raw C-SKL and p-value
    facts, recomputes BH in the transactional catalog, and verifies the result
    against both exported global and independent families.
    """

    store = Store(store_root)
    run_dir = store.run_dir(run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    platform = str(manifest.get("platform") or "").strip().upper()
    pool_version = str(manifest.get("pool_version") or "").strip()
    if not platform or not pool_version:
        raise ValueError("Run manifest is missing platform or pool_version")
    if manifest.get("feature_hash") != store.feature_hash(platform):
        raise ValueError("Run feature hash no longer matches the frozen store")
    if manifest.get("pool_hash") != store.pool_hash(platform, pool_version):
        raise ValueError("Run pool hash no longer matches the frozen store")

    edges = read_table(run_dir, "network_edges")
    required = {
        "Dataset_A", "Dataset_B", "cSKL", "p_value", "q_global", "q_independent",
        "shared_sample_count", "fraction_a", "fraction_b", "jaccard",
        "overlap_coefficient", "overlap_classification", "discovery_excluded",
    }
    missing = sorted(required - set(edges.columns))
    if missing:
        raise ValueError(f"Release edge table is missing columns: {', '.join(missing)}")
    if edges[["Dataset_A", "Dataset_B"]].duplicated().any():
        raise ValueError("Release edge table contains duplicate endpoint pairs")
    accessions = sorted(set(edges["Dataset_A"]) | set(edges["Dataset_B"]))
    expected_pairs = len(accessions) * (len(accessions) - 1) // 2
    if len(edges) != expected_pairs or int(manifest.get("n_pairs", -1)) != expected_pairs:
        raise ValueError("Release does not contain the complete all-pairs family")
    numeric = edges[["cSKL", "p_value", "q_global"]].to_numpy(dtype=float)
    if not all(math.isfinite(value) for value in numeric.ravel()):
        raise ValueError("Release contains non-finite raw/global statistical facts")

    with catalog.reader() as connection:
        version_by_accession = {
            row["accession"]: row["current_version_id"]
            for row in connection.execute(
                """SELECT accession,current_version_id FROM datasets
                   WHERE platform=? AND cohort='series' AND current_version_id IS NOT NULL""",
                (platform,),
            )
        }
    absent = sorted(set(accessions) - set(version_by_accession))
    if absent:
        raise ValueError(
            f"Import dataset versions before the release; {len(absent)} endpoints are absent"
        )

    algorithm_manifest = {
        "contract": "cskl-scale-release-v2",
        "feature_hash": manifest["feature_hash"],
        "alpha": manifest["alpha"],
        "core": "cskl.py",
    }
    algorithm_hash = hashlib.sha256(canonical_json(algorithm_manifest).encode()).hexdigest()
    score_rows: list[tuple[str, str, str, float]] = []
    pair_ids: list[str] = []
    for row in edges.itertuples(index=False):
        version_a = version_by_accession[str(row.Dataset_A)]
        version_b = version_by_accession[str(row.Dataset_B)]
        if version_a > version_b:
            version_a, version_b = version_b, version_a
        score_rows.append((version_a, version_b, algorithm_hash, float(row.cSKL)))
        pair_ids.append(stable_id("pair", version_a, version_b, algorithm_hash))
    catalog.record_pair_scores(score_rows)

    overlap_count = 0
    for row in edges.itertuples(index=False):
        if int(row.shared_sample_count) == 0:
            continue
        accession_a = str(row.Dataset_A)
        accession_b = str(row.Dataset_B)
        version_a = version_by_accession[accession_a]
        version_b = version_by_accession[accession_b]
        fraction_a = float(row.fraction_a)
        fraction_b = float(row.fraction_b)
        if version_a > version_b:
            version_a, version_b = version_b, version_a
            fraction_a, fraction_b = fraction_b, fraction_a
        hashes_a = set(read_json(store.sample_hashes_path(accession_a)).get("hashes", []))
        hashes_b = set(read_json(store.sample_hashes_path(accession_b)).get("hashes", []))
        shared_hashes = sorted(hashes_a & hashes_b)
        evidence_payload = {
            "contract": "expression-profile-overlap-v1",
            "shared_count": int(row.shared_sample_count),
            "fraction_a": fraction_a,
            "fraction_b": fraction_b,
            "jaccard": float(row.jaccard),
            "overlap_coefficient": float(row.overlap_coefficient),
            "classification": str(row.overlap_classification),
            "shared_expression_hashes": shared_hashes,
        }
        evidence_hash = hashlib.sha256(canonical_json(evidence_payload).encode()).hexdigest()
        catalog.record_overlap(
            version_a=version_a,
            version_b=version_b,
            evidence_hash=evidence_hash,
            shared_count=int(row.shared_sample_count),
            fraction_a=fraction_a,
            fraction_b=fraction_b,
            jaccard=float(row.jaccard),
            overlap_coefficient=float(row.overlap_coefficient),
            classification=str(row.overlap_classification),
            discovery_excluded=True,
            shared_samples=shared_hashes,
        )
        overlap_count += 1

    parameter_base = {
        "run_id": run_id,
        "pool_version": pool_version,
        "pool_hash": manifest["pool_hash"],
        "B": manifest["B"],
        "grid": manifest["grid"],
        "fdr_alpha": manifest["fdr_alpha"],
    }
    global_parameter_hash = hashlib.sha256(
        canonical_json({**parameter_base, "family": "global"}).encode()
    ).hexdigest()
    global_rows = [
        (pair_id, float(p_value))
        for pair_id, p_value in zip(pair_ids, edges["p_value"], strict=True)
    ]
    global_calibration_id = _import_calibration(
        catalog,
        stratum=f"{platform}:global",
        pool_hash=manifest["pool_hash"],
        parameter_hash=global_parameter_hash,
        algorithm_hash=algorithm_hash,
        pair_rows=global_rows,
        manifest={**parameter_base, "family": "global", "source": str(run_dir)},
    )

    independent_positions = [
        position for position, excluded in enumerate(edges["discovery_excluded"])
        if not bool(excluded)
    ]
    independent_parameter_hash = hashlib.sha256(
        canonical_json({**parameter_base, "family": "independent"}).encode()
    ).hexdigest()
    independent_rows = [
        (pair_ids[position], float(edges.iloc[position]["p_value"]))
        for position in independent_positions
    ]
    independent_calibration_id = _import_calibration(
        catalog,
        stratum=f"{platform}:independent",
        pool_hash=manifest["pool_hash"],
        parameter_hash=independent_parameter_hash,
        algorithm_hash=algorithm_hash,
        pair_rows=independent_rows,
        manifest={
            **parameter_base,
            "family": "independent",
            "source": str(run_dir),
            "excludes_any_shared_sample": True,
        },
    )

    expected_q = {
        global_calibration_id: dict(zip(pair_ids, edges["q_global"], strict=True)),
        independent_calibration_id: {
            pair_ids[position]: float(edges.iloc[position]["q_independent"])
            for position in independent_positions
        },
    }
    max_q_error: dict[str, float] = {}
    with catalog.reader() as connection:
        for calibration_id, expected in expected_q.items():
            rows = connection.execute(
                "SELECT pair_id,q_value FROM calibrated_edges WHERE calibration_id=?",
                (calibration_id,),
            )
            differences = [abs(float(row["q_value"]) - float(expected[row["pair_id"]])) for row in rows]
            error = max(differences, default=0.0)
            if error > 1e-12:
                raise ValueError(f"Catalog BH differs from exported release by {error:.3g}")
            max_q_error[calibration_id] = error

    return {
        "run_id": run_id,
        "platform": platform,
        "dataset_count": len(accessions),
        "pair_count": len(edges),
        "overlap_pair_count": overlap_count,
        "algorithm_hash": algorithm_hash,
        "global_calibration_id": global_calibration_id,
        "independent_calibration_id": independent_calibration_id,
        "independent_pair_count": len(independent_rows),
        "max_q_error": max_q_error,
    }
