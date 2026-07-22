"""Versioned local Reactome indexing and GPL570-background enrichment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import hypergeom

from .catalog import canonical_json


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _gene_universe(annotation_path: Path) -> set[str]:
    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "ENTREZID" not in reader.fieldnames:
            raise ValueError("Probe annotation must contain an ENTREZID column.")
        return {
            str(row["ENTREZID"]).strip()
            for row in reader
            if row.get("ENTREZID") and str(row["ENTREZID"]).strip()
        }


def build_reactome_index(
    *,
    mapping_path: str | Path,
    annotation_path: str | Path,
    database_path: str | Path,
    manifest_path: str | Path,
    release: str,
) -> dict[str, Any]:
    """Build a compact human-only index against the explicit array gene universe."""

    source = Path(mapping_path).resolve()
    annotation = Path(annotation_path).resolve()
    destination = Path(database_path).resolve()
    manifest_destination = Path(manifest_path).resolve()
    if not source.is_file() or not annotation.is_file():
        raise FileNotFoundError("Reactome mapping and probe annotation files are required.")
    if not str(release).strip():
        raise ValueError("Reactome release is required.")

    mapping_checksum = _sha256_path(source)
    annotation_checksum = _sha256_path(annotation)
    dependency = hashlib.sha256(
        canonical_json(
            {
                "schema": "reactome-gpl570-index-v1",
                "release": str(release),
                "mapping_checksum": mapping_checksum,
                "annotation_checksum": annotation_checksum,
                "species": "Homo sapiens",
                "identifier_namespace": "NCBI Entrez Gene",
            }
        ).encode()
    ).hexdigest()
    if destination.is_file() and manifest_destination.is_file():
        existing = json.loads(manifest_destination.read_text(encoding="utf-8"))
        if existing.get("dependency_hash") == dependency:
            if existing.get("index_checksum") == _sha256_path(destination):
                return existing

    universe = _gene_universe(annotation)
    if not universe:
        raise ValueError("The probe annotation produced an empty Entrez gene universe.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=FULL;
            CREATE TABLE release_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE universe(gene_id TEXT PRIMARY KEY);
            CREATE TABLE pathway_members(
                pathway_id TEXT NOT NULL,
                gene_id TEXT NOT NULL,
                pathway_name TEXT NOT NULL,
                url TEXT NOT NULL,
                evidence TEXT NOT NULL,
                PRIMARY KEY(pathway_id,gene_id)
            );
            """
        )
        connection.executemany(
            "INSERT INTO universe(gene_id) VALUES(?)", ((gene,) for gene in sorted(universe))
        )
        row_count = 0
        with source.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            batch: list[tuple[str, str, str, str, str]] = []
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 6:
                    raise ValueError(f"Invalid Reactome row {row_number}: expected 6 fields.")
                gene_id, pathway_id, url, pathway_name, evidence, species = row
                if species != "Homo sapiens" or gene_id not in universe:
                    continue
                batch.append((pathway_id, gene_id, pathway_name.strip(), url, evidence))
                if len(batch) >= 10_000:
                    before = connection.total_changes
                    connection.executemany(
                        "INSERT OR IGNORE INTO pathway_members VALUES(?,?,?,?,?)", batch
                    )
                    row_count += connection.total_changes - before
                    batch.clear()
            if batch:
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO pathway_members VALUES(?,?,?,?,?)", batch
                )
                row_count += connection.total_changes - before
        meta = {
            "schema": "reactome-gpl570-index-v1",
            "reactome_release": str(release),
            "species": "Homo sapiens",
            "identifier_namespace": "NCBI Entrez Gene",
            "mapping_checksum": mapping_checksum,
            "annotation_checksum": annotation_checksum,
            "dependency_hash": dependency,
            "background_gene_count": len(universe),
            "mapped_gene_count": int(
                connection.execute("SELECT COUNT(DISTINCT gene_id) FROM pathway_members").fetchone()[0]
            ),
            "pathway_count": int(
                connection.execute("SELECT COUNT(DISTINCT pathway_id) FROM pathway_members").fetchone()[0]
            ),
            "mapping_row_count": row_count,
        }
        connection.executemany(
            "INSERT INTO release_meta(key,value) VALUES(?,?)",
            ((key, str(value)) for key, value in meta.items()),
        )
        connection.execute("CREATE INDEX pathway_members_gene_idx ON pathway_members(gene_id)")
        connection.commit()
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()
    os.replace(temporary, destination)
    manifest = {
        **meta,
        "index_checksum": _sha256_path(destination),
        "source_uri": str(source),
        "annotation_uri": str(annotation),
        "database_uri": str(destination),
        "multiple_testing": "Benjamini-Hochberg over every size-eligible Reactome pathway",
        "test": "one-sided hypergeometric over-representation",
    }
    _atomic_json(manifest_destination, manifest)
    return manifest


def _bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def enrich_reactome(
    gene_ids: Iterable[str],
    *,
    database_path: str | Path,
    minimum_overlap: int = 2,
    minimum_pathway_size: int = 3,
    maximum_pathway_size: int = 1_500,
    limit: int = 25,
) -> dict[str, Any]:
    """Run local, release-aware over-representation with the array universe."""

    if minimum_overlap < 1 or minimum_pathway_size < 1 or limit < 1:
        raise ValueError("Enrichment thresholds and limit must be positive.")
    database = Path(database_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key,value FROM release_meta")
        }
        requested = sorted({str(value).strip() for value in gene_ids if str(value).strip()})
        if requested:
            placeholders = ",".join("?" for _ in requested)
            tested = [
                row[0]
                for row in connection.execute(
                    f"SELECT gene_id FROM universe WHERE gene_id IN ({placeholders})", requested
                )
            ]
        else:
            tested = []
        pathway_rows = connection.execute(
            """SELECT pathway_id,MIN(pathway_name) AS pathway_name,MIN(url) AS url,
                      COUNT(*) AS pathway_size
               FROM pathway_members GROUP BY pathway_id
               HAVING COUNT(*) BETWEEN ? AND ? ORDER BY pathway_id""",
            (minimum_pathway_size, maximum_pathway_size),
        ).fetchall()
        hits_by_pathway: dict[str, list[str]] = {}
        if tested:
            placeholders = ",".join("?" for _ in tested)
            for row in connection.execute(
                f"""SELECT pathway_id,gene_id FROM pathway_members
                    WHERE gene_id IN ({placeholders}) ORDER BY pathway_id,gene_id""",
                tested,
            ):
                hits_by_pathway.setdefault(row["pathway_id"], []).append(row["gene_id"])
    finally:
        connection.close()

    background_size = int(metadata.get("background_gene_count", 0))
    query_size = len(tested)
    p_values: list[float] = []
    raw: list[dict[str, Any]] = []
    for row in pathway_rows:
        hits = hits_by_pathway.get(row["pathway_id"], [])
        overlap = len(hits)
        pathway_size = int(row["pathway_size"])
        p_value = (
            float(hypergeom.sf(overlap - 1, background_size, pathway_size, query_size))
            if query_size
            else 1.0
        )
        p_values.append(p_value)
        expected = query_size * pathway_size / background_size if background_size else 0.0
        raw.append(
            {
                "pathway_id": row["pathway_id"],
                "pathway_name": row["pathway_name"],
                "url": row["url"],
                "pathway_size": pathway_size,
                "overlap_count": overlap,
                "expected_overlap": expected,
                "fold_enrichment": overlap / expected if expected else 0.0,
                "p_value": p_value,
                "gene_ids": hits,
            }
        )
    q_values = _bh_adjust(p_values)
    for item, q_value in zip(raw, q_values, strict=True):
        item["q_value"] = q_value
    results = [item for item in raw if item["overlap_count"] >= minimum_overlap]
    results.sort(
        key=lambda item: (
            item["q_value"],
            item["p_value"],
            -item["fold_enrichment"],
            item["pathway_id"],
        )
    )
    return {
        "reactome_release": metadata.get("reactome_release"),
        "database_dependency_hash": metadata.get("dependency_hash"),
        "background": "all unique Entrez genes represented by the frozen GPL570 annotation",
        "background_gene_count": background_size,
        "input_gene_count": len(requested),
        "tested_gene_count": query_size,
        "tested_pathway_count": len(pathway_rows),
        "minimum_overlap": minimum_overlap,
        "results": results[:limit],
    }
