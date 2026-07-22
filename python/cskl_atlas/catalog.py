from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: object, size: int = 24) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:size]}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ordered_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("A pair must contain two different dataset versions.")
    return (left, right) if left < right else (right, left)


def _hash_sorted_pair_ids(pair_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    previous: str | None = None
    for value in pair_ids:
        pair_id = str(value).strip()
        if not pair_id:
            raise ValueError("pair IDs must be non-empty")
        if pair_id == previous:
            raise ValueError(f"pair family contains duplicate ID: {pair_id}")
        if previous is not None and pair_id < previous:
            raise ValueError("pair IDs must be sorted for streaming fingerprinting")
        digest.update(pair_id.encode("utf-8"))
        digest.update(b"\0")
        previous = pair_id
    return digest.hexdigest()


def pair_family_hash(pair_ids: Iterable[str]) -> str:
    """Fingerprint an exact, order-independent family of calibrated pairs."""

    return _hash_sorted_pair_ids(sorted(str(value).strip() for value in pair_ids))


def _validated_sha256(value: str, *, field: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Catalog:
    """Transactional catalog for provenance, retries, serving, and releases.

    Large matrices and PCA artifacts remain in content-addressed object storage;
    this SQLite catalog is the local/reference implementation of the relational
    control plane. The schema is PostgreSQL-friendly for the server deployment.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA temp_store=FILE")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            stored = connection.execute(
                "SELECT value FROM catalog_meta WHERE key='schema_version'"
            ).fetchone()
            stored_version: int | None = None
            if stored is not None:
                try:
                    stored_version = int(stored["value"])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("Catalog schema_version is not an integer.") from exc
                if stored_version > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Catalog schema {stored_version} is newer than supported schema {SCHEMA_VERSION}."
                    )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_uid TEXT PRIMARY KEY,
                    accession TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    cohort TEXT NOT NULL,
                    current_version_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(accession, platform, cohort)
                );

                CREATE TABLE IF NOT EXISTS dataset_versions (
                    version_id TEXT PRIMARY KEY,
                    dataset_uid TEXT NOT NULL REFERENCES datasets(dataset_uid),
                    source_revision TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    normalized_hash TEXT NOT NULL,
                    signature_hash TEXT NOT NULL,
                    feature_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    sample_count INTEGER NOT NULL CHECK(sample_count >= 0),
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('ready','invalid','superseded')),
                    created_at TEXT NOT NULL,
                    UNIQUE(dataset_uid, source_revision, source_hash, config_hash)
                );

                CREATE INDEX IF NOT EXISTS dataset_versions_dataset_idx
                    ON dataset_versions(dataset_uid, created_at DESC);
                CREATE INDEX IF NOT EXISTS dataset_versions_feature_idx
                    ON dataset_versions(feature_hash, config_hash);
                CREATE INDEX IF NOT EXISTS datasets_current_version_idx
                    ON datasets(current_version_id);

                CREATE TABLE IF NOT EXISTS samples (
                    sample_uid TEXT PRIMARY KEY,
                    gsm_accession TEXT,
                    expression_hash TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(gsm_accession, expression_hash)
                );

                CREATE TABLE IF NOT EXISTS dataset_samples (
                    version_id TEXT NOT NULL REFERENCES dataset_versions(version_id) ON DELETE CASCADE,
                    sample_uid TEXT NOT NULL REFERENCES samples(sample_uid),
                    position INTEGER,
                    PRIMARY KEY(version_id, sample_uid)
                );
                CREATE INDEX IF NOT EXISTS dataset_samples_sample_idx ON dataset_samples(sample_uid);

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    dataset_version_id TEXT REFERENCES dataset_versions(version_id),
                    uri TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    dependency_hash TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(kind, dependency_hash)
                );

                CREATE TABLE IF NOT EXISTS pair_scores (
                    pair_id TEXT PRIMARY KEY,
                    version_a TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    version_b TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    algorithm_hash TEXT NOT NULL,
                    cskl REAL NOT NULL CHECK(cskl >= 0),
                    created_at TEXT NOT NULL,
                    CHECK(version_a < version_b),
                    UNIQUE(version_a, version_b, algorithm_hash)
                );
                CREATE INDEX IF NOT EXISTS pair_scores_a_idx ON pair_scores(version_a, cskl);
                CREATE INDEX IF NOT EXISTS pair_scores_b_idx ON pair_scores(version_b, cskl);
                CREATE INDEX IF NOT EXISTS pair_scores_algorithm_idx
                    ON pair_scores(algorithm_hash, pair_id);

                CREATE TABLE IF NOT EXISTS overlap_evidence (
                    overlap_id TEXT PRIMARY KEY,
                    version_a TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    version_b TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    evidence_hash TEXT NOT NULL,
                    shared_count INTEGER NOT NULL CHECK(shared_count >= 0),
                    fraction_a REAL NOT NULL,
                    fraction_b REAL NOT NULL,
                    jaccard REAL NOT NULL,
                    overlap_coefficient REAL NOT NULL,
                    classification TEXT NOT NULL,
                    discovery_excluded INTEGER NOT NULL CHECK(discovery_excluded IN (0,1)),
                    shared_samples_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    CHECK(version_a < version_b),
                    UNIQUE(version_a, version_b, evidence_hash)
                );
                CREATE INDEX IF NOT EXISTS overlap_pair_idx ON overlap_evidence(version_a, version_b);

                CREATE TABLE IF NOT EXISTS calibration_releases (
                    calibration_id TEXT PRIMARY KEY,
                    stratum TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('exact','frozen')),
                    pool_hash TEXT NOT NULL,
                    parameter_hash TEXT NOT NULL,
                    algorithm_hash TEXT,
                    family_hash TEXT,
                    expected_pair_count INTEGER CHECK(expected_pair_count >= 0),
                    manifest_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('staging','calibrated','published','failed','superseded')),
                    created_at TEXT NOT NULL,
                    finalized_at TEXT
                );
                CREATE INDEX IF NOT EXISTS calibration_stratum_idx
                    ON calibration_releases(stratum, created_at DESC);

                CREATE TABLE IF NOT EXISTS calibrated_edges (
                    calibration_id TEXT NOT NULL REFERENCES calibration_releases(calibration_id) ON DELETE CASCADE,
                    pair_id TEXT NOT NULL REFERENCES pair_scores(pair_id) ON DELETE CASCADE,
                    p_value REAL NOT NULL CHECK(p_value >= 0 AND p_value <= 1),
                    q_value REAL CHECK(q_value >= 0 AND q_value <= 1),
                    cskl_similarity_percentile REAL CHECK(
                        cskl_similarity_percentile >= 0 AND cskl_similarity_percentile <= 1
                    ),
                    PRIMARY KEY(calibration_id, pair_id)
                );
                CREATE INDEX IF NOT EXISTS calibrated_edges_q_idx
                    ON calibrated_edges(calibration_id, q_value, p_value);

                CREATE TABLE IF NOT EXISTS calibration_release_members (
                    calibration_id TEXT NOT NULL REFERENCES calibration_releases(calibration_id) ON DELETE CASCADE,
                    version_id TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    PRIMARY KEY(calibration_id, version_id)
                );

                CREATE TABLE IF NOT EXISTS calibration_release_pairs (
                    calibration_id TEXT NOT NULL REFERENCES calibration_releases(calibration_id) ON DELETE CASCADE,
                    pair_id TEXT NOT NULL REFERENCES pair_scores(pair_id),
                    overlap_id TEXT REFERENCES overlap_evidence(overlap_id),
                    PRIMARY KEY(calibration_id, pair_id)
                );
                CREATE INDEX IF NOT EXISTS calibration_release_pairs_pair_idx
                    ON calibration_release_pairs(pair_id, calibration_id);

                CREATE TABLE IF NOT EXISTS graph_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    calibration_id TEXT NOT NULL REFERENCES calibration_releases(calibration_id),
                    stratum TEXT NOT NULL,
                    policy_hash TEXT NOT NULL,
                    layout_version TEXT NOT NULL,
                    manifest_uri TEXT NOT NULL,
                    manifest_checksum TEXT,
                    text_release_id TEXT REFERENCES text_releases(text_release_id),
                    status TEXT NOT NULL CHECK(status IN ('staging','published','superseded','failed')),
                    created_at TEXT NOT NULL,
                    published_at TEXT
                );

                CREATE TABLE IF NOT EXISTS graph_snapshot_datasets (
                    snapshot_id TEXT NOT NULL REFERENCES graph_snapshots(snapshot_id) ON DELETE CASCADE,
                    version_id TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    x REAL,
                    y REAL,
                    community TEXT,
                    PRIMARY KEY(snapshot_id, version_id)
                );

                CREATE TABLE IF NOT EXISTS graph_snapshot_edges (
                    snapshot_id TEXT NOT NULL REFERENCES graph_snapshots(snapshot_id) ON DELETE CASCADE,
                    pair_id TEXT NOT NULL REFERENCES pair_scores(pair_id),
                    overlap_id TEXT REFERENCES overlap_evidence(overlap_id),
                    PRIMARY KEY(snapshot_id, pair_id)
                );
                CREATE INDEX IF NOT EXISTS graph_snapshot_edges_pair_idx
                    ON graph_snapshot_edges(pair_id, snapshot_id);

                CREATE TABLE IF NOT EXISTS text_releases (
                    text_release_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    input_fields_json TEXT NOT NULL,
                    corpus_hash TEXT NOT NULL,
                    parameter_hash TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('staging','finalized','failed','superseded')),
                    created_at TEXT NOT NULL,
                    finalized_at TEXT,
                    UNIQUE(model_id, model_revision, corpus_hash, parameter_hash)
                );

                CREATE TABLE IF NOT EXISTS text_pair_scores (
                    text_release_id TEXT NOT NULL REFERENCES text_releases(text_release_id) ON DELETE CASCADE,
                    version_a TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    version_b TEXT NOT NULL REFERENCES dataset_versions(version_id),
                    cosine_similarity REAL NOT NULL CHECK(cosine_similarity >= -1 AND cosine_similarity <= 1),
                    similarity_percentile REAL CHECK(similarity_percentile >= 0 AND similarity_percentile <= 1),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(text_release_id, version_a, version_b),
                    CHECK(version_a < version_b)
                );
                CREATE INDEX IF NOT EXISTS text_pair_scores_rank_idx
                    ON text_pair_scores(text_release_id, similarity_percentile DESC, cosine_similarity DESC);

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    job_key TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','retry','succeeded','dead','cancelled')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 5,
                    next_retry_at TEXT,
                    worker_id TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT,
                    error_detail TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(kind, job_key, input_fingerprint)
                );
                CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status, next_retry_at, created_at);

                CREATE TABLE IF NOT EXISTS annotation_assertions (
                    assertion_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL REFERENCES dataset_versions(version_id) ON DELETE CASCADE,
                    field TEXT NOT NULL,
                    value TEXT NOT NULL,
                    ontology_id TEXT,
                    source_kind TEXT NOT NULL,
                    source_field TEXT,
                    evidence_span TEXT,
                    extractor_version TEXT NOT NULL,
                    review_state TEXT NOT NULL,
                    confidence REAL,
                    supersedes TEXT REFERENCES annotation_assertions(assertion_id),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS annotation_version_field_idx
                    ON annotation_assertions(version_id, field, review_state);

                CREATE TABLE IF NOT EXISTS annotation_reviews (
                    review_id TEXT PRIMARY KEY,
                    assertion_id TEXT NOT NULL REFERENCES annotation_assertions(assertion_id),
                    reviewer TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('accepted','rejected')),
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS annotation_reviews_assertion_idx
                    ON annotation_reviews(assertion_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS snapshot_events (
                    event_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL REFERENCES graph_snapshots(snapshot_id),
                    previous_snapshot_id TEXT REFERENCES graph_snapshots(snapshot_id),
                    action TEXT NOT NULL CHECK(action IN ('publish','rollback')),
                    operator TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS snapshot_events_snapshot_idx
                    ON snapshot_events(snapshot_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS ai_runs (
                    ai_run_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    evidence_hash TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cost_usd REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task, evidence_hash, prompt_hash, model)
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            migrations: tuple[tuple[str, str, str], ...] = (
                ("jobs", "lease_expires_at", "TEXT"),
                ("jobs", "heartbeat_at", "TEXT"),
                ("jobs", "progress_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("calibration_releases", "algorithm_hash", "TEXT"),
                ("calibration_releases", "family_hash", "TEXT"),
                ("calibration_releases", "expected_pair_count", "INTEGER"),
                ("calibrated_edges", "cskl_similarity_percentile", "REAL"),
                ("graph_snapshots", "manifest_checksum", "TEXT"),
                ("graph_snapshots", "text_release_id", "TEXT REFERENCES text_releases(text_release_id)"),
            )
            for table, column, declaration in migrations:
                columns = {
                    row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
                }
                if column not in columns:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
            connection.execute(
                "INSERT INTO catalog_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def register_dataset_version(
        self,
        *,
        accession: str,
        platform: str,
        cohort: str,
        source_revision: str,
        source_hash: str,
        normalized_hash: str,
        signature_hash: str,
        feature_hash: str,
        config_hash: str,
        sample_count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Register an immutable candidate version without changing publication.

        A candidate becomes the dataset's current version only through
        :meth:`promote_dataset_version`, after its normalized matrix and PCA
        signature artifacts have been checksum-verified in the catalog.
        """
        accession = accession.strip().upper()
        platform = platform.strip().upper()
        cohort = cohort.strip() or "series"
        if not accession or not platform:
            raise ValueError("accession and platform are required")
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        dataset_uid = stable_id("ds", accession, platform, cohort)
        version_id = stable_id(
            "dsv", dataset_uid, source_revision, source_hash, normalized_hash,
            signature_hash, feature_hash, config_hash,
        )
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO datasets(dataset_uid, accession, platform, cohort, current_version_id, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(dataset_uid) DO UPDATE SET updated_at=excluded.updated_at""",
                (dataset_uid, accession, platform, cohort, None, now, now),
            )
            connection.execute(
                """INSERT INTO dataset_versions(
                       version_id,dataset_uid,source_revision,source_hash,normalized_hash,signature_hash,
                       feature_hash,config_hash,sample_count,metadata_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(version_id) DO NOTHING""",
                (
                    version_id, dataset_uid, source_revision, source_hash, normalized_hash,
                    signature_hash, feature_hash, config_hash, sample_count,
                    canonical_json(metadata or {}), "ready", now,
                ),
            )
        return dataset_uid, version_id

    def promote_dataset_version(self, version_id: str) -> None:
        """Atomically publish a candidate after its required artifacts exist."""

        with self.transaction() as connection:
            version = connection.execute(
                "SELECT * FROM dataset_versions WHERE version_id=?", (version_id,)
            ).fetchone()
            if not version or version["status"] == "invalid":
                raise ValueError("Only a valid registered dataset version can be promoted.")
            required = {
                "normalized_matrix": version["normalized_hash"],
                "pca_signature": version["signature_hash"],
            }
            artifacts = {
                row["kind"]: row["checksum"]
                for row in connection.execute(
                    """SELECT kind,checksum FROM artifacts
                       WHERE dataset_version_id=? AND kind IN ('normalized_matrix','pca_signature')""",
                    (version_id,),
                )
            }
            missing = sorted(kind for kind, checksum in required.items() if artifacts.get(kind) != checksum)
            if missing:
                raise ValueError(
                    "Dataset version cannot be promoted until checksum-matching artifacts exist: "
                    + ", ".join(missing)
                )
            dataset_uid = version["dataset_uid"]
            now = utc_now()
            connection.execute(
                """UPDATE dataset_versions SET status='superseded'
                   WHERE dataset_uid=? AND version_id<>? AND status='ready'""",
                (dataset_uid, version_id),
            )
            connection.execute(
                "UPDATE dataset_versions SET status='ready' WHERE version_id=?", (version_id,)
            )
            connection.execute(
                "UPDATE datasets SET current_version_id=?,updated_at=? WHERE dataset_uid=?",
                (version_id, now, dataset_uid),
            )

    def add_samples(self, version_id: str, samples: Iterable[Mapping[str, Any]]) -> int:
        inserted = 0
        with self.transaction() as connection:
            for position, sample in enumerate(samples):
                gsm = str(sample.get("gsm") or "").strip().upper() or None
                expression_hash = str(sample.get("expression_hash") or "").strip().lower() or None
                if not gsm and not expression_hash:
                    raise ValueError("Each sample needs a GSM accession or expression hash.")
                sample_uid = stable_id("sample", gsm or "", expression_hash or "")
                connection.execute(
                    """INSERT INTO samples(sample_uid,gsm_accession,expression_hash,metadata_json)
                       VALUES(?,?,?,?) ON CONFLICT(sample_uid) DO UPDATE SET metadata_json=excluded.metadata_json""",
                    (sample_uid, gsm, expression_hash, canonical_json(sample.get("metadata") or {})),
                )
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO dataset_samples(version_id,sample_uid,position) VALUES(?,?,?)",
                    (version_id, sample_uid, position),
                )
                inserted += cursor.rowcount
        return inserted

    def replace_samples(self, version_id: str, samples: Iterable[Mapping[str, Any]]) -> int:
        """Atomically replace a version's sample identities with richer evidence.

        This supports a safe metadata upgrade from expression-hash-only identity
        to aligned GSM+hash identity without leaving duplicate links behind.
        """

        prepared = list(samples)
        with self.transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM dataset_versions WHERE version_id=?", (version_id,)
            ).fetchone():
                raise KeyError(version_id)
            connection.execute("DELETE FROM dataset_samples WHERE version_id=?", (version_id,))
            for position, sample in enumerate(prepared):
                gsm = str(sample.get("gsm") or "").strip().upper() or None
                expression_hash = str(sample.get("expression_hash") or "").strip().lower() or None
                if not gsm and not expression_hash:
                    raise ValueError("Each sample needs a GSM accession or expression hash.")
                sample_uid = stable_id("sample", gsm or "", expression_hash or "")
                connection.execute(
                    """INSERT INTO samples(sample_uid,gsm_accession,expression_hash,metadata_json)
                       VALUES(?,?,?,?) ON CONFLICT(sample_uid) DO UPDATE SET
                       metadata_json=excluded.metadata_json""",
                    (sample_uid, gsm, expression_hash, canonical_json(sample.get("metadata") or {})),
                )
                connection.execute(
                    "INSERT INTO dataset_samples(version_id,sample_uid,position) VALUES(?,?,?)",
                    (version_id, sample_uid, position),
                )
        return len(prepared)

    def prune_orphan_samples(self) -> int:
        """Remove sample identities no longer referenced after a bulk replacement."""

        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM samples WHERE NOT EXISTS "
                "(SELECT 1 FROM dataset_samples ds WHERE ds.sample_uid=samples.sample_uid)"
            )
            return cursor.rowcount

    def record_artifact(
        self,
        *,
        artifact_id: str,
        kind: str,
        uri: str,
        checksum: str,
        dependency_hash: str,
        manifest: Mapping[str, Any],
        dataset_version_id: str | None = None,
    ) -> None:
        checksum = _validated_sha256(checksum, field="checksum")
        dependency_hash = _validated_sha256(dependency_hash, field="dependency_hash")
        if not kind.strip() or not uri.strip():
            raise ValueError("artifact kind and URI are required")
        manifest_json = canonical_json(manifest)
        proposed = {
            "artifact_id": artifact_id,
            "kind": kind,
            "dataset_version_id": dataset_version_id,
            "uri": uri,
            "checksum": checksum,
            "dependency_hash": dependency_hash,
            "manifest_json": manifest_json,
        }
        with self.transaction() as connection:
            existing = connection.execute(
                """SELECT artifact_id,kind,dataset_version_id,uri,checksum,dependency_hash,manifest_json
                   FROM artifacts WHERE artifact_id=? OR (kind=? AND dependency_hash=?)""",
                (artifact_id, kind, dependency_hash),
            ).fetchone()
            if existing:
                mismatches = [
                    key for key, value in proposed.items() if existing[key] != value
                ]
                if mismatches:
                    raise ValueError(
                        "Artifact identity already exists with different immutable fields: "
                        + ", ".join(mismatches)
                    )
                return
            connection.execute(
                """INSERT INTO artifacts(artifact_id,kind,dataset_version_id,uri,checksum,dependency_hash,manifest_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    artifact_id, kind, dataset_version_id, uri, checksum, dependency_hash,
                    manifest_json, utc_now(),
                ),
            )

    def record_annotation_assertions(
        self,
        version_id: str,
        assertions: Iterable[Mapping[str, Any]],
        *,
        replace_generated: bool = False,
    ) -> list[str]:
        """Persist evidence-grounded annotation candidates without auto-promoting them.

        ``replace_generated`` retires stale machine output atomically while
        preserving every human-reviewed LLM assertion. It is used by resumable
        annotation jobs when their metadata, model, or prompt dependency changes.
        """

        prepared: list[tuple[Any, ...]] = []
        superseded: list[str] = []
        now = utc_now()
        allowed_sources = {
            "geo_structured",
            "deterministic_ontology",
            "llm_candidate",
            "human_verified",
        }
        allowed_reviews = {"unreviewed", "accepted", "rejected", "superseded"}
        for assertion in assertions:
            field = str(assertion.get("field") or "").strip()
            value = str(assertion.get("value") or "").strip()
            source_kind = str(assertion.get("source_kind") or "").strip()
            extractor_version = str(assertion.get("extractor_version") or "").strip()
            review_state = str(assertion.get("review_state") or "unreviewed").strip()
            if not field or not value or not extractor_version:
                raise ValueError("annotation field, value, and extractor_version are required")
            if source_kind not in allowed_sources:
                raise ValueError(f"Unsupported annotation source_kind: {source_kind}")
            if review_state not in allowed_reviews:
                raise ValueError(f"Unsupported annotation review_state: {review_state}")
            confidence = assertion.get("confidence")
            if confidence is not None:
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("annotation confidence must be finite and in [0, 1]")
            ontology_id = str(assertion.get("ontology_id") or "").strip() or None
            source_field = str(assertion.get("source_field") or "").strip() or None
            evidence_span = assertion.get("evidence_span")
            evidence_json = canonical_json(evidence_span) if evidence_span is not None else None
            supersedes = str(assertion.get("supersedes") or "").strip() or None
            assertion_id = stable_id(
                "assertion", version_id, field, value, ontology_id or "", source_kind,
                source_field or "", evidence_json or "", extractor_version,
            )
            prepared.append(
                (
                    assertion_id, version_id, field, value, ontology_id, source_kind,
                    source_field, evidence_json, extractor_version, review_state, confidence,
                    supersedes, now,
                )
            )
            if supersedes:
                superseded.append(supersedes)
        if not prepared and not replace_generated:
            return []
        with self.transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM dataset_versions WHERE version_id=?", (version_id,)
            ).fetchone():
                raise KeyError(version_id)
            for assertion_id in superseded:
                prior = connection.execute(
                    "SELECT version_id,field FROM annotation_assertions WHERE assertion_id=?",
                    (assertion_id,),
                ).fetchone()
                replacement = next(row for row in prepared if row[11] == assertion_id)
                if not prior or prior["version_id"] != version_id or prior["field"] != replacement[2]:
                    raise ValueError("supersedes must reference an assertion for the same version and field")
            if prepared:
                connection.executemany(
                    """INSERT INTO annotation_assertions(
                           assertion_id,version_id,field,value,ontology_id,source_kind,source_field,
                           evidence_span,extractor_version,review_state,confidence,supersedes,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(assertion_id) DO NOTHING""",
                    prepared,
                )
            if replace_generated:
                current_ids = [row[0] for row in prepared]
                current_clause = ""
                parameters: list[Any] = [version_id]
                if current_ids:
                    placeholders = ",".join("?" for _ in current_ids)
                    current_clause = f" AND assertion_id NOT IN ({placeholders})"
                    parameters.extend(current_ids)
                connection.execute(
                    """UPDATE annotation_assertions SET review_state='superseded'
                       WHERE version_id=?
                         AND ((source_kind='llm_candidate' AND review_state='unreviewed')
                              OR (source_kind='geo_structured' AND review_state='accepted'))"""
                    + current_clause,
                    parameters,
                )
            if superseded:
                placeholders = ",".join("?" for _ in superseded)
                connection.execute(
                    f"UPDATE annotation_assertions SET review_state='superseded' WHERE assertion_id IN ({placeholders})",
                    superseded,
                )
        return [row[0] for row in prepared]

    def review_annotation(
        self,
        assertion_id: str,
        *,
        reviewer: str,
        decision: str,
        note: str = "",
    ) -> str:
        if decision not in {"accepted", "rejected"}:
            raise ValueError("decision must be accepted or rejected")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        now = utc_now()
        review_id = stable_id("review", assertion_id, reviewer.strip(), decision, note, now)
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE annotation_assertions SET review_state=?
                   WHERE assertion_id=? AND review_state<>'superseded'""",
                (decision, assertion_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Unknown or superseded annotation assertion.")
            connection.execute(
                """INSERT INTO annotation_reviews(
                       review_id,assertion_id,reviewer,decision,note,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (review_id, assertion_id, reviewer.strip(), decision, note[:4_000], now),
            )
        return review_id

    def record_ai_run(
        self,
        *,
        task: str,
        evidence_hash: str,
        prompt_hash: str,
        provider: str,
        model: str,
        response: Mapping[str, Any],
        status: str,
        cost_usd: float | None = None,
    ) -> str:
        if status not in {"succeeded", "failed", "rejected"}:
            raise ValueError("AI run status must be succeeded, failed, or rejected")
        if not all(value.strip() for value in (task, provider, model)):
            raise ValueError("AI run task, provider, and model are required")
        evidence_hash = _validated_sha256(evidence_hash, field="evidence_hash")
        prompt_hash = _validated_sha256(prompt_hash, field="prompt_hash")
        if cost_usd is not None and (not math.isfinite(cost_usd) or cost_usd < 0):
            raise ValueError("AI run cost must be finite and non-negative")
        run_id = stable_id("ai", task, evidence_hash, prompt_hash, model)
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO ai_runs(
                       ai_run_id,task,evidence_hash,prompt_hash,provider,model,response_json,
                       status,cost_usd,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(ai_run_id) DO NOTHING""",
                (
                    run_id, task.strip(), evidence_hash, prompt_hash, provider.strip(),
                    model.strip(), canonical_json(response), status, cost_usd, utc_now(),
                ),
            )
        return run_id

    def record_pair_scores(
        self,
        rows: Iterable[tuple[str, str, str, float]],
    ) -> int:
        prepared: list[tuple[str, str, str, str, float, str]] = []
        now = utc_now()
        for left, right, algorithm_hash, cskl in rows:
            if not math.isfinite(cskl) or cskl < 0:
                raise ValueError("cskl must be finite and non-negative")
            version_a, version_b = ordered_pair(left, right)
            pair_id = stable_id("pair", version_a, version_b, algorithm_hash)
            prepared.append((pair_id, version_a, version_b, algorithm_hash, float(cskl), now))
        if not prepared:
            return 0
        with self.transaction() as connection:
            before = connection.total_changes
            connection.executemany(
                """INSERT INTO pair_scores(pair_id,version_a,version_b,algorithm_hash,cskl,created_at)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(pair_id) DO NOTHING""",
                prepared,
            )
            return connection.total_changes - before

    def iter_pair_scores(
        self,
        *,
        algorithm_hash: str | None = None,
        feature_hash: str | None = None,
        batch_size: int = 10_000,
    ) -> Iterator[list[dict[str, Any]]]:
        """Stream compatible raw pair facts without a dense all-pairs matrix."""
        batch_size = max(1, min(int(batch_size), 100_000))
        clauses: list[str] = []
        parameters: list[Any] = []
        if algorithm_hash:
            clauses.append("p.algorithm_hash=?")
            parameters.append(algorithm_hash)
        if feature_hash:
            clauses.append("va.feature_hash=? AND vb.feature_hash=?")
            parameters.extend([feature_hash, feature_hash])
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self.reader() as connection:
            cursor = connection.execute(
                f"""SELECT p.*,va.sample_count AS samples_a,vb.sample_count AS samples_b,
                           va.signature_hash AS signature_a,vb.signature_hash AS signature_b,
                           va.feature_hash AS feature_a,vb.feature_hash AS feature_b
                    FROM pair_scores p
                    JOIN dataset_versions va ON va.version_id=p.version_a
                    JOIN dataset_versions vb ON vb.version_id=p.version_b
                    {where} ORDER BY p.pair_id""",
                parameters,
            )
            while rows := cursor.fetchmany(batch_size):
                yield [dict(row) for row in rows]

    def iter_uncalibrated_pair_scores(
        self,
        calibration_id: str,
        *,
        batch_size: int = 10_000,
    ) -> Iterator[list[dict[str, Any]]]:
        """Stream raw pairs missing from one staging calibration release."""

        batch_size = max(1, min(int(batch_size), 100_000))
        with self.reader() as connection:
            release = connection.execute(
                "SELECT status,algorithm_hash FROM calibration_releases WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()
            if not release:
                raise KeyError(calibration_id)
            if release["status"] != "staging":
                return
            frozen_pair_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM calibration_release_pairs WHERE calibration_id=?",
                    (calibration_id,),
                ).fetchone()[0]
            )
            if frozen_pair_count:
                cursor = connection.execute(
                    """SELECT p.*,va.sample_count AS samples_a,vb.sample_count AS samples_b
                       FROM calibration_release_pairs family
                       JOIN pair_scores p ON p.pair_id=family.pair_id
                       JOIN dataset_versions va ON va.version_id=p.version_a
                       JOIN dataset_versions vb ON vb.version_id=p.version_b
                       LEFT JOIN calibrated_edges c
                         ON c.calibration_id=family.calibration_id AND c.pair_id=p.pair_id
                       WHERE family.calibration_id=? AND c.pair_id IS NULL
                       ORDER BY p.pair_id""",
                    (calibration_id,),
                )
            else:
                # Compatibility for schema-v3/imported releases. New CLI releases always
                # bind their exact pair family in calibration_release_pairs.
                cursor = connection.execute(
                    """SELECT p.*,va.sample_count AS samples_a,vb.sample_count AS samples_b
                       FROM pair_scores p
                       JOIN dataset_versions va ON va.version_id=p.version_a
                       JOIN dataset_versions vb ON vb.version_id=p.version_b
                       LEFT JOIN calibrated_edges c
                         ON c.calibration_id=? AND c.pair_id=p.pair_id
                       WHERE p.algorithm_hash=? AND c.pair_id IS NULL
                       ORDER BY p.pair_id""",
                    (calibration_id, release["algorithm_hash"]),
                )
            while rows := cursor.fetchmany(batch_size):
                yield [dict(row) for row in rows]

    def record_overlap(
        self,
        *,
        version_a: str,
        version_b: str,
        evidence_hash: str,
        shared_count: int,
        fraction_a: float,
        fraction_b: float,
        jaccard: float,
        overlap_coefficient: float,
        classification: str,
        discovery_excluded: bool,
        shared_samples: Sequence[str],
    ) -> str:
        if isinstance(shared_count, bool) or int(shared_count) != shared_count or shared_count < 0:
            raise ValueError("shared_count must be a non-negative integer")
        metrics = {
            "fraction_a": fraction_a,
            "fraction_b": fraction_b,
            "jaccard": jaccard,
            "overlap_coefficient": overlap_coefficient,
        }
        if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in metrics.values()):
            raise ValueError("overlap fractions and coefficients must be finite values in [0, 1]")
        if classification not in {"none", "minor", "major", "exact"}:
            raise ValueError("classification must be one of: none, minor, major, exact")
        if (shared_count == 0) != (classification == "none"):
            raise ValueError("classification 'none' must correspond exactly to zero shared samples")
        if len(shared_samples) != len(set(shared_samples)):
            raise ValueError("shared_samples must not contain duplicates")
        version_a, version_b = ordered_pair(version_a, version_b)
        overlap_id = stable_id("overlap", version_a, version_b, evidence_hash)
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO overlap_evidence(
                       overlap_id,version_a,version_b,evidence_hash,shared_count,fraction_a,fraction_b,
                       jaccard,overlap_coefficient,classification,discovery_excluded,shared_samples_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(overlap_id) DO NOTHING""",
                (
                    overlap_id, version_a, version_b, evidence_hash, shared_count, fraction_a,
                    fraction_b, jaccard, overlap_coefficient, classification,
                    int(discovery_excluded), canonical_json(list(shared_samples)), utc_now(),
                ),
            )
        return overlap_id

    def stage_calibration(
        self,
        *,
        stratum: str,
        mode: str,
        pool_hash: str,
        parameter_hash: str,
        algorithm_hash: str,
        family_hash: str,
        expected_pair_count: int,
        manifest: Mapping[str, Any],
    ) -> str:
        if mode not in {"exact", "frozen"}:
            raise ValueError("mode must be 'exact' or 'frozen'")
        if expected_pair_count < 1:
            raise ValueError("expected_pair_count must be positive")
        family_hash = _validated_sha256(family_hash, field="family_hash")
        if not algorithm_hash.strip():
            raise ValueError("algorithm_hash is required")
        calibration_id = stable_id(
            "cal", stratum, mode, pool_hash, parameter_hash, algorithm_hash,
            family_hash, expected_pair_count,
        )
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO calibration_releases(
                       calibration_id,stratum,mode,pool_hash,parameter_hash,algorithm_hash,
                       family_hash,expected_pair_count,manifest_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(calibration_id) DO NOTHING""",
                (
                    calibration_id, stratum, mode, pool_hash, parameter_hash, algorithm_hash,
                    family_hash, expected_pair_count, canonical_json(manifest), "staging", utc_now(),
                ),
            )
        return calibration_id

    @staticmethod
    def _current_pair_family_contract(
        connection: sqlite3.Connection,
        *,
        algorithm_hash: str,
    ) -> dict[str, Any]:
        """Resolve and validate one complete current-version score family.

        The scoring algorithm identifies a single feature/config stratum. Every
        dataset currently published in that stratum is a member, including a new
        version which has not yet accumulated any pair rows. Requiring N choose 2
        rows therefore turns an interrupted incremental-score job into a clear
        release gate instead of silently calibrating a smaller historical corpus.
        """

        algorithm_hash = str(algorithm_hash).strip()
        if not algorithm_hash:
            raise ValueError("algorithm_hash is required")
        strata = connection.execute(
            """SELECT DISTINCT v.feature_hash,v.config_hash
               FROM dataset_versions v
               JOIN (
                 SELECT version_a AS version_id FROM pair_scores WHERE algorithm_hash=?
                 UNION
                 SELECT version_b AS version_id FROM pair_scores WHERE algorithm_hash=?
               ) scored ON scored.version_id=v.version_id
               ORDER BY v.feature_hash,v.config_hash""",
            (algorithm_hash, algorithm_hash),
        ).fetchall()
        if not strata:
            raise ValueError("No raw pair scores exist for the requested algorithm.")
        if len(strata) != 1:
            raise ValueError(
                "The scoring algorithm spans multiple feature/config strata; "
                "use a distinct algorithm hash for each comparable stratum."
            )
        feature_hash = str(strata[0]["feature_hash"])
        config_hash = str(strata[0]["config_hash"])

        moved_dataset_count = int(
            connection.execute(
                """SELECT COUNT(DISTINCT historical.dataset_uid)
                   FROM dataset_versions historical
                   JOIN datasets d ON d.dataset_uid=historical.dataset_uid
                   JOIN dataset_versions current ON current.version_id=d.current_version_id
                   WHERE historical.version_id IN (
                     SELECT version_a FROM pair_scores WHERE algorithm_hash=?
                     UNION
                     SELECT version_b FROM pair_scores WHERE algorithm_hash=?
                   )
                     AND (current.feature_hash<>? OR current.config_hash<>?)""",
                (algorithm_hash, algorithm_hash, feature_hash, config_hash),
            ).fetchone()[0]
        )
        if moved_dataset_count:
            raise ValueError(
                f"{moved_dataset_count} scored dataset(s) now have a current version in a "
                "different feature/config stratum; issue a new scoring algorithm release."
            )

        members = [
            str(row["version_id"])
            for row in connection.execute(
                """SELECT v.version_id
                   FROM dataset_versions v
                   JOIN datasets d ON d.current_version_id=v.version_id
                   WHERE v.feature_hash=? AND v.config_hash=?
                   ORDER BY v.version_id""",
                (feature_hash, config_hash),
            )
        ]
        if len(members) < 2:
            raise ValueError("A calibration family requires at least two current datasets.")
        expected_pair_count = len(members) * (len(members) - 1) // 2
        pair_query = """SELECT p.pair_id
                        FROM pair_scores p
                        JOIN dataset_versions va ON va.version_id=p.version_a
                        JOIN dataset_versions vb ON vb.version_id=p.version_b
                        JOIN datasets da ON da.current_version_id=p.version_a
                        JOIN datasets db ON db.current_version_id=p.version_b
                        WHERE p.algorithm_hash=?
                          AND va.feature_hash=? AND va.config_hash=?
                          AND vb.feature_hash=? AND vb.config_hash=?
                        ORDER BY p.pair_id"""
        pair_count = int(
            connection.execute(
                f"SELECT COUNT(*) FROM ({pair_query.removesuffix('ORDER BY p.pair_id')})",
                (algorithm_hash, feature_hash, config_hash, feature_hash, config_hash),
            ).fetchone()[0]
        )
        if pair_count != expected_pair_count:
            raise ValueError(
                "Current pair family is incomplete: "
                f"expected {expected_pair_count} scores for {len(members)} members, "
                f"found {pair_count}. Finish/recover incremental scoring before calibration."
            )
        pair_cursor = connection.execute(
            pair_query,
            (algorithm_hash, feature_hash, config_hash, feature_hash, config_hash),
        )
        family_hash = _hash_sorted_pair_ids(row["pair_id"] for row in pair_cursor)
        member_hash = hashlib.sha256(canonical_json(members).encode("utf-8")).hexdigest()
        return {
            "algorithm_hash": algorithm_hash,
            "feature_hash": feature_hash,
            "config_hash": config_hash,
            "members": members,
            "member_hash": member_hash,
            "family_hash": family_hash,
            "pair_count": pair_count,
            "pair_query": pair_query,
            "pair_parameters": (
                algorithm_hash,
                feature_hash,
                config_hash,
                feature_hash,
                config_hash,
            ),
        }

    def current_pair_family_fingerprint(
        self,
        *,
        algorithm_hash: str,
    ) -> tuple[str, int, str, int]:
        """Return pair/member fingerprints for a complete current-version family."""

        with self.reader() as connection:
            contract = self._current_pair_family_contract(
                connection, algorithm_hash=algorithm_hash
            )
        return (
            str(contract["family_hash"]),
            int(contract["pair_count"]),
            str(contract["member_hash"]),
            len(contract["members"]),
        )

    def stage_current_calibration(
        self,
        *,
        stratum: str,
        mode: str,
        pool_hash: str,
        parameter_hash: str,
        algorithm_hash: str,
        manifest: Mapping[str, Any],
    ) -> str:
        """Atomically freeze the complete current dataset and raw-pair family."""

        if mode not in {"exact", "frozen"}:
            raise ValueError("mode must be 'exact' or 'frozen'")
        if not stratum.strip() or not algorithm_hash.strip():
            raise ValueError("stratum and algorithm_hash are required")
        with self.transaction() as connection:
            contract = self._current_pair_family_contract(
                connection, algorithm_hash=algorithm_hash
            )
            family_hash = str(contract["family_hash"])
            pair_count = int(contract["pair_count"])
            calibration_id = stable_id(
                "cal",
                stratum,
                mode,
                pool_hash,
                parameter_hash,
                algorithm_hash,
                family_hash,
                pair_count,
            )
            overlap_bindings = connection.execute(
                """WITH ranked_overlap AS (
                     SELECT o.overlap_id,o.version_a,o.version_b,
                            ROW_NUMBER() OVER (
                              PARTITION BY o.version_a,o.version_b
                              ORDER BY o.created_at DESC,o.overlap_id DESC
                            ) AS rank
                     FROM overlap_evidence o
                   )
                   SELECT p.pair_id,o.overlap_id
                   FROM ranked_overlap o
                   JOIN pair_scores p
                     ON p.version_a=o.version_a AND p.version_b=o.version_b
                   JOIN datasets da ON da.current_version_id=p.version_a
                   JOIN datasets db ON db.current_version_id=p.version_b
                   WHERE o.rank=1 AND p.algorithm_hash=?""",
                (algorithm_hash,),
            ).fetchall()
            release_manifest = dict(manifest)
            release_manifest.update(
                {
                    "family_scope": "current_versions",
                    "feature_hash": contract["feature_hash"],
                    "config_hash": contract["config_hash"],
                    "member_hash": contract["member_hash"],
                    "member_count": len(contract["members"]),
                    "pair_count": pair_count,
                    "overlap_binding": "latest_available_at_staging",
                    "overlap_evidence_bound_count": len(overlap_bindings),
                }
            )
            existing = connection.execute(
                "SELECT * FROM calibration_releases WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()
            if existing and existing["status"] != "staging":
                return calibration_id
            connection.execute(
                """INSERT INTO calibration_releases(
                       calibration_id,stratum,mode,pool_hash,parameter_hash,algorithm_hash,
                       family_hash,expected_pair_count,manifest_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(calibration_id) DO UPDATE SET
                     manifest_json=excluded.manifest_json""",
                (
                    calibration_id,
                    stratum,
                    mode,
                    pool_hash,
                    parameter_hash,
                    algorithm_hash,
                    family_hash,
                    pair_count,
                    canonical_json(release_manifest),
                    "staging",
                    utc_now(),
                ),
            )
            connection.executemany(
                """INSERT INTO calibration_release_members(calibration_id,version_id)
                   VALUES(?,?) ON CONFLICT(calibration_id,version_id) DO NOTHING""",
                ((calibration_id, version_id) for version_id in contract["members"]),
            )
            connection.execute(
                f"""INSERT OR IGNORE INTO calibration_release_pairs(
                       calibration_id,pair_id,overlap_id)
                    SELECT ?,family.pair_id,NULL FROM ({contract['pair_query']}) family""",
                (calibration_id, *contract["pair_parameters"]),
            )
            connection.executemany(
                """UPDATE calibration_release_pairs SET overlap_id=?
                   WHERE calibration_id=? AND pair_id=?""",
                (
                    (row["overlap_id"], calibration_id, row["pair_id"])
                    for row in overlap_bindings
                ),
            )
            bound_members = int(
                connection.execute(
                    "SELECT COUNT(*) FROM calibration_release_members WHERE calibration_id=?",
                    (calibration_id,),
                ).fetchone()[0]
            )
            bound_pairs = int(
                connection.execute(
                    "SELECT COUNT(*) FROM calibration_release_pairs WHERE calibration_id=?",
                    (calibration_id,),
                ).fetchone()[0]
            )
            if bound_members != len(contract["members"]) or bound_pairs != pair_count:
                raise RuntimeError("Failed to persist the complete frozen calibration family.")
        return calibration_id

    def record_pvalues(self, calibration_id: str, rows: Iterable[tuple[str, float]]) -> int:
        prepared = []
        for pair_id, p_value in rows:
            if not math.isfinite(p_value) or not 0 <= p_value <= 1:
                raise ValueError("p_value must be finite and between 0 and 1")
            prepared.append((calibration_id, pair_id, float(p_value)))
        if not prepared:
            return 0
        with self.transaction() as connection:
            release = connection.execute(
                "SELECT status FROM calibration_releases WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()
            if not release:
                raise KeyError(f"Unknown calibration release: {calibration_id}")
            if release["status"] != "staging":
                raise ValueError("A finalized calibration release is immutable.")
            before = connection.total_changes
            connection.executemany(
                """INSERT INTO calibrated_edges(calibration_id,pair_id,p_value)
                   VALUES(?,?,?) ON CONFLICT(calibration_id,pair_id) DO UPDATE SET
                   p_value=excluded.p_value,q_value=NULL,cskl_similarity_percentile=NULL""",
                prepared,
            )
            return connection.total_changes - before

    def finalize_bh(
        self,
        calibration_id: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> int:
        """Apply exact Benjamini-Hochberg correction in disk-backed SQL.

        SQLite performs the sort/window operation using its temporary store, so
        the Python process never materializes all 24.5M pair values in RAM.
        """
        with self.transaction() as connection:
            release = connection.execute(
                "SELECT * FROM calibration_releases WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()
            if not release:
                raise KeyError(f"Unknown calibration release: {calibration_id}")
            if release["status"] != "staging":
                raise ValueError("Only a staging calibration can be finalized.")
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM calibrated_edges WHERE calibration_id=?",
                    (calibration_id,),
                ).fetchone()[0]
            )
            if count == 0:
                raise ValueError("Cannot finalize a calibration without p-values.")
            if release["expected_pair_count"] is None or count != release["expected_pair_count"]:
                raise ValueError(
                    f"Calibration pair family is incomplete: expected "
                    f"{release['expected_pair_count']}, found {count}."
                )
            bound_pair_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM calibration_release_pairs WHERE calibration_id=?",
                    (calibration_id,),
                ).fetchone()[0]
            )
            if bound_pair_count:
                if bound_pair_count != int(release["expected_pair_count"]):
                    raise ValueError(
                        "Frozen calibration pair family is incomplete: expected "
                        f"{release['expected_pair_count']}, found {bound_pair_count}."
                    )
                missing_from_values = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM calibration_release_pairs family
                           LEFT JOIN calibrated_edges c
                             ON c.calibration_id=family.calibration_id
                            AND c.pair_id=family.pair_id
                           WHERE family.calibration_id=? AND c.pair_id IS NULL""",
                        (calibration_id,),
                    ).fetchone()[0]
                )
                outside_frozen_family = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM calibrated_edges c
                           LEFT JOIN calibration_release_pairs family
                             ON family.calibration_id=c.calibration_id
                            AND family.pair_id=c.pair_id
                           WHERE c.calibration_id=? AND family.pair_id IS NULL""",
                        (calibration_id,),
                    ).fetchone()[0]
                )
                if missing_from_values or outside_frozen_family:
                    raise ValueError("Calibrated p-values differ from the frozen pair family.")
            bad_algorithms = int(
                connection.execute(
                    """SELECT COUNT(*) FROM calibrated_edges c
                       JOIN pair_scores p ON p.pair_id=c.pair_id
                       WHERE c.calibration_id=? AND p.algorithm_hash<>?""",
                    (calibration_id, release["algorithm_hash"]),
                ).fetchone()[0]
            )
            if bad_algorithms:
                raise ValueError("Calibration family mixes scoring algorithm versions.")
            pair_cursor = connection.execute(
                "SELECT pair_id FROM calibrated_edges WHERE calibration_id=? ORDER BY pair_id",
                (calibration_id,),
            )
            actual_family_hash = _hash_sorted_pair_ids(row["pair_id"] for row in pair_cursor)
            if actual_family_hash != release["family_hash"]:
                raise ValueError("Calibration pair family fingerprint does not match the staged release.")
            connection.execute("DROP TABLE IF EXISTS temp.bh_values")
            connection.execute(
                """CREATE TEMP TABLE bh_values AS
                   WITH ranked AS (
                     SELECT pair_id, p_value,
                            ROW_NUMBER() OVER (ORDER BY p_value ASC, pair_id ASC) AS rank,
                            COUNT(*) OVER () AS total
                     FROM calibrated_edges WHERE calibration_id=?
                   ), raw AS (
                     SELECT pair_id, rank, (p_value * total / rank) AS raw_q FROM ranked
                   )
                   SELECT pair_id,
                          MIN(raw_q) OVER (
                            ORDER BY rank DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                          ) AS q_value
                   FROM raw""",
                (calibration_id,),
            )
            connection.execute("CREATE UNIQUE INDEX temp.bh_values_pair_idx ON bh_values(pair_id)")
            connection.execute(
                """UPDATE calibrated_edges
                   SET q_value = MIN(1.0, (SELECT q_value FROM bh_values WHERE bh_values.pair_id=calibrated_edges.pair_id))
                   WHERE calibration_id=?""",
                (calibration_id,),
            )
            connection.execute("DROP TABLE IF EXISTS temp.cskl_percentiles")
            connection.execute(
                """CREATE TEMP TABLE cskl_percentiles AS
                   SELECT c.pair_id,
                          CUME_DIST() OVER (ORDER BY p.cskl DESC) AS percentile
                   FROM calibrated_edges c JOIN pair_scores p USING(pair_id)
                   WHERE c.calibration_id=?""",
                (calibration_id,),
            )
            connection.execute(
                "CREATE UNIQUE INDEX temp.cskl_percentiles_pair_idx ON cskl_percentiles(pair_id)"
            )
            connection.execute(
                """UPDATE calibrated_edges SET cskl_similarity_percentile=(
                       SELECT percentile FROM cskl_percentiles p
                       WHERE p.pair_id=calibrated_edges.pair_id)
                   WHERE calibration_id=?""",
                (calibration_id,),
            )
            connection.execute(
                "UPDATE calibration_releases SET status='calibrated', finalized_at=? WHERE calibration_id=?",
                (utc_now(), calibration_id),
            )
            if diagnostics is not None:
                release_manifest = json.loads(release["manifest_json"])
                release_diagnostics = dict(diagnostics)
                clamped_pair_count = int(
                    release_diagnostics.get("boundary_clamped_pair_count", 0)
                )
                if clamped_pair_count < 0:
                    raise ValueError("boundary_clamped_pair_count cannot be negative")
                if release["mode"] == "exact" and clamped_pair_count:
                    raise ValueError("Exact calibration cannot finalize with boundary clamps.")
                release_manifest.update(release_diagnostics)
                connection.execute(
                    "UPDATE calibration_releases SET manifest_json=? WHERE calibration_id=?",
                    (canonical_json(release_manifest), calibration_id),
                )
            connection.execute("DROP TABLE temp.bh_values")
            connection.execute("DROP TABLE temp.cskl_percentiles")
        return count

    def calibration_boundary_diagnostics(
        self,
        calibration_id: str,
        *,
        grid_min: int,
        grid_max: int,
    ) -> dict[str, int | None]:
        """Count frozen-family profile lookups which hit null-grid boundaries."""

        if grid_min > grid_max:
            raise ValueError("grid_min must not exceed grid_max")
        with self.reader() as connection:
            release = connection.execute(
                "SELECT expected_pair_count FROM calibration_releases WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()
            if not release:
                raise KeyError(calibration_id)
            bound_pair_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM calibration_release_pairs WHERE calibration_id=?",
                    (calibration_id,),
                ).fetchone()[0]
            )
            family_table = (
                "calibration_release_pairs" if bound_pair_count else "calibrated_edges"
            )
            row = connection.execute(
                f"""SELECT COUNT(*) AS pair_count,
                           COALESCE(SUM(CASE WHEN
                             va.sample_count<? OR va.sample_count>? OR
                             vb.sample_count<? OR vb.sample_count>?
                           THEN 1 ELSE 0 END),0) AS clamped_pairs,
                           COALESCE(SUM(
                             CASE WHEN va.sample_count<? OR va.sample_count>? THEN 1 ELSE 0 END +
                             CASE WHEN vb.sample_count<? OR vb.sample_count>? THEN 1 ELSE 0 END
                           ),0) AS clamped_lookups,
                           MIN(MIN(va.sample_count,vb.sample_count)) AS minimum_samples,
                           MAX(MAX(va.sample_count,vb.sample_count)) AS maximum_samples
                    FROM {family_table} family
                    JOIN pair_scores p ON p.pair_id=family.pair_id
                    JOIN dataset_versions va ON va.version_id=p.version_a
                    JOIN dataset_versions vb ON vb.version_id=p.version_b
                    WHERE family.calibration_id=?""",
                (
                    grid_min,
                    grid_max,
                    grid_min,
                    grid_max,
                    grid_min,
                    grid_max,
                    grid_min,
                    grid_max,
                    calibration_id,
                ),
            ).fetchone()
        pair_count = int(row["pair_count"])
        expected_pair_count = int(release["expected_pair_count"] or 0)
        if pair_count != expected_pair_count:
            raise ValueError(
                "Cannot compute complete calibration diagnostics: expected "
                f"{expected_pair_count} pairs, found {pair_count}."
            )
        return {
            "boundary_clamped_pair_count": int(row["clamped_pairs"]),
            "boundary_clamped_profile_lookup_count": int(row["clamped_lookups"]),
            "minimum_sample_count": (
                None if row["minimum_samples"] is None else int(row["minimum_samples"])
            ),
            "maximum_sample_count": (
                None if row["maximum_samples"] is None else int(row["maximum_samples"])
            ),
        }

    def pair_family_fingerprint(self, *, algorithm_hash: str) -> tuple[str, int]:
        """Stream the exact raw-score family fingerprint without loading IDs in Python."""

        with self.reader() as connection:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM pair_scores WHERE algorithm_hash=?", (algorithm_hash,)
                ).fetchone()[0]
            )
            cursor = connection.execute(
                "SELECT pair_id FROM pair_scores WHERE algorithm_hash=? ORDER BY pair_id",
                (algorithm_hash,),
            )
            return _hash_sorted_pair_ids(row["pair_id"] for row in cursor), count

    def fail_calibration(self, calibration_id: str, *, reason: str) -> None:
        if not reason.strip():
            raise ValueError("A calibration failure reason is required.")
        with self.transaction() as connection:
            release = connection.execute(
                "SELECT status,manifest_json FROM calibration_releases WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()
            if not release or release["status"] != "staging":
                raise ValueError("Only a staging calibration can be marked failed.")
            manifest = json.loads(release["manifest_json"])
            manifest["failure_reason"] = reason[:8_000]
            connection.execute(
                "UPDATE calibration_releases SET status='failed',manifest_json=?,finalized_at=? WHERE calibration_id=?",
                (canonical_json(manifest), utc_now(), calibration_id),
            )

    def stage_text_release(
        self,
        *,
        model_id: str,
        model_revision: str,
        input_fields: Sequence[str],
        corpus_hash: str,
        parameter_hash: str,
        manifest: Mapping[str, Any],
    ) -> str:
        """Stage a pinned text-similarity release; this never runs a model itself."""

        fields = tuple(str(value).strip() for value in input_fields)
        if not model_id.strip() or not model_revision.strip() or not fields or any(not value for value in fields):
            raise ValueError("model ID, revision, and at least one input field are required")
        if len(fields) != len(set(fields)):
            raise ValueError("text release input fields must be unique")
        corpus_hash = _validated_sha256(corpus_hash, field="corpus_hash")
        parameter_hash = _validated_sha256(parameter_hash, field="parameter_hash")
        release_id = stable_id(
            "text", model_id, model_revision, canonical_json(fields), corpus_hash, parameter_hash
        )
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO text_releases(
                       text_release_id,model_id,model_revision,input_fields_json,corpus_hash,
                       parameter_hash,manifest_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,'staging',?) ON CONFLICT(text_release_id) DO NOTHING""",
                (
                    release_id, model_id.strip(), model_revision.strip(), canonical_json(fields),
                    corpus_hash, parameter_hash, canonical_json(manifest), utc_now(),
                ),
            )
        return release_id

    def record_text_pair_scores(
        self,
        text_release_id: str,
        rows: Iterable[tuple[str, str, float]],
    ) -> int:
        prepared: list[tuple[str, str, str, float, str]] = []
        now = utc_now()
        for left, right, similarity in rows:
            similarity = float(similarity)
            if not math.isfinite(similarity) or not -1 <= similarity <= 1:
                raise ValueError("cosine similarity must be finite and in [-1, 1]")
            version_a, version_b = ordered_pair(left, right)
            prepared.append((text_release_id, version_a, version_b, similarity, now))
        if not prepared:
            return 0
        with self.transaction() as connection:
            release = connection.execute(
                "SELECT status FROM text_releases WHERE text_release_id=?", (text_release_id,)
            ).fetchone()
            if not release:
                raise KeyError(f"Unknown text release: {text_release_id}")
            if release["status"] != "staging":
                raise ValueError("A finalized text release is immutable.")
            before = connection.total_changes
            connection.executemany(
                """INSERT INTO text_pair_scores(
                       text_release_id,version_a,version_b,cosine_similarity,created_at)
                   VALUES(?,?,?,?,?) ON CONFLICT(text_release_id,version_a,version_b) DO UPDATE SET
                   cosine_similarity=excluded.cosine_similarity,similarity_percentile=NULL,
                   created_at=excluded.created_at""",
                prepared,
            )
            return connection.total_changes - before

    def finalize_text_release(self, text_release_id: str) -> int:
        """Compute within-release similarity percentiles in disk-backed SQL."""

        with self.transaction() as connection:
            release = connection.execute(
                "SELECT status FROM text_releases WHERE text_release_id=?", (text_release_id,)
            ).fetchone()
            if not release:
                raise KeyError(f"Unknown text release: {text_release_id}")
            if release["status"] != "staging":
                raise ValueError("Only a staging text release can be finalized.")
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM text_pair_scores WHERE text_release_id=?",
                    (text_release_id,),
                ).fetchone()[0]
            )
            if count == 0:
                raise ValueError("Cannot finalize a text release without pair scores.")
            connection.execute("DROP TABLE IF EXISTS temp.text_percentiles")
            connection.execute(
                """CREATE TEMP TABLE text_percentiles AS
                   SELECT version_a,version_b,
                          CUME_DIST() OVER (ORDER BY cosine_similarity ASC) AS percentile
                   FROM text_pair_scores WHERE text_release_id=?""",
                (text_release_id,),
            )
            connection.execute(
                "CREATE UNIQUE INDEX temp.text_percentile_pair_idx ON text_percentiles(version_a,version_b)"
            )
            connection.execute(
                """UPDATE text_pair_scores SET similarity_percentile=(
                       SELECT percentile FROM text_percentiles t
                       WHERE t.version_a=text_pair_scores.version_a
                         AND t.version_b=text_pair_scores.version_b)
                   WHERE text_release_id=?""",
                (text_release_id,),
            )
            connection.execute(
                "UPDATE text_releases SET status='finalized',finalized_at=? WHERE text_release_id=?",
                (utc_now(), text_release_id),
            )
            connection.execute("DROP TABLE temp.text_percentiles")
        return count

    def stage_snapshot(
        self,
        *,
        calibration_id: str,
        stratum: str,
        policy_hash: str,
        layout_version: str,
        manifest_uri: str,
        manifest_checksum: str,
        datasets: Iterable[tuple[str, float | None, float | None, str | None]],
        text_release_id: str | None = None,
        pair_ids: Iterable[str] | None = None,
    ) -> str:
        manifest_checksum = _validated_sha256(manifest_checksum, field="manifest_checksum")
        if not stratum.strip() or not policy_hash.strip() or not layout_version.strip() or not manifest_uri.strip():
            raise ValueError("snapshot stratum, policy, layout version, and manifest URI are required")
        members = tuple(datasets)
        requested_pair_ids = None if pair_ids is None else tuple(sorted(set(pair_ids)))
        if requested_pair_ids is not None and any(not value for value in requested_pair_ids):
            raise ValueError("snapshot pair IDs must be non-empty")
        version_ids = [row[0] for row in members]
        if len(version_ids) != len(set(version_ids)):
            raise ValueError("snapshot dataset versions must be unique")
        for version_id, x, y, _community in members:
            if not version_id:
                raise ValueError("snapshot version IDs must be non-empty")
            if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)):
                raise ValueError("snapshot coordinates must be finite numbers")
        snapshot_id = stable_id(
            "snapshot", calibration_id, stratum, policy_hash, layout_version, manifest_uri,
            manifest_checksum, text_release_id or "",
        )
        with self.transaction() as connection:
            release = connection.execute(
                "SELECT status,stratum FROM calibration_releases WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()
            if not release or release["status"] not in {"calibrated", "published"}:
                raise ValueError("A graph snapshot requires a finalized calibration.")
            if release["stratum"] != stratum:
                raise ValueError("Snapshot and calibration strata must match.")
            if text_release_id is not None:
                text_release = connection.execute(
                    "SELECT status FROM text_releases WHERE text_release_id=?", (text_release_id,)
                ).fetchone()
                if not text_release or text_release["status"] != "finalized":
                    raise ValueError("A snapshot can bind only a finalized text release.")
            existing = connection.execute(
                "SELECT status FROM graph_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if existing and existing["status"] != "staging":
                raise ValueError("A published graph snapshot is immutable.")
            connection.execute(
                """INSERT INTO graph_snapshots(
                       snapshot_id,calibration_id,stratum,policy_hash,layout_version,manifest_uri,
                       manifest_checksum,text_release_id,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(snapshot_id) DO NOTHING""",
                (
                    snapshot_id, calibration_id, stratum, policy_hash, layout_version,
                    manifest_uri, manifest_checksum, text_release_id, "staging", utc_now(),
                ),
            )
            connection.execute(
                "DELETE FROM graph_snapshot_datasets WHERE snapshot_id=?", (snapshot_id,)
            )
            connection.executemany(
                """INSERT INTO graph_snapshot_datasets(snapshot_id,version_id,x,y,community)
                   VALUES(?,?,?,?,?)""",
                ((snapshot_id, version_id, x, y, community) for version_id, x, y, community in members),
            )
            connection.execute("DELETE FROM graph_snapshot_edges WHERE snapshot_id=?", (snapshot_id,))
            has_frozen_pairs = bool(
                connection.execute(
                    "SELECT 1 FROM calibration_release_pairs WHERE calibration_id=? LIMIT 1",
                    (calibration_id,),
                ).fetchone()
            )
            pair_filter_join = ""
            if requested_pair_ids is not None:
                connection.execute("DROP TABLE IF EXISTS temp.requested_snapshot_pairs")
                connection.execute(
                    "CREATE TEMP TABLE requested_snapshot_pairs(pair_id TEXT PRIMARY KEY)"
                )
                connection.executemany(
                    "INSERT INTO requested_snapshot_pairs(pair_id) VALUES(?)",
                    ((pair_id,) for pair_id in requested_pair_ids),
                )
                pair_filter_join = "JOIN requested_snapshot_pairs requested USING(pair_id)"
                eligible_count = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM requested_snapshot_pairs requested
                            JOIN calibrated_edges c USING(pair_id)
                            JOIN pair_scores p USING(pair_id)
                            JOIN graph_snapshot_datasets ga
                              ON ga.snapshot_id=? AND ga.version_id=p.version_a
                            JOIN graph_snapshot_datasets gb
                              ON gb.snapshot_id=? AND gb.version_id=p.version_b
                            WHERE c.calibration_id=? AND c.q_value IS NOT NULL""",
                        (snapshot_id, snapshot_id, calibration_id),
                    ).fetchone()[0]
                )
                if eligible_count != len(requested_pair_ids):
                    raise ValueError(
                        "Every requested snapshot pair must belong to the calibration and "
                        "connect two snapshot datasets."
                    )
            if has_frozen_pairs:
                connection.execute(
                    f"""INSERT INTO graph_snapshot_edges(snapshot_id,pair_id,overlap_id)
                        SELECT ?,p.pair_id,family.overlap_id
                        FROM calibrated_edges c
                        JOIN pair_scores p ON p.pair_id=c.pair_id
                        JOIN calibration_release_pairs family
                          ON family.calibration_id=c.calibration_id
                         AND family.pair_id=c.pair_id
                        {pair_filter_join}
                        JOIN graph_snapshot_datasets ga
                          ON ga.snapshot_id=? AND ga.version_id=p.version_a
                        JOIN graph_snapshot_datasets gb
                          ON gb.snapshot_id=? AND gb.version_id=p.version_b
                        WHERE c.calibration_id=? AND c.q_value IS NOT NULL""",
                    (snapshot_id, snapshot_id, snapshot_id, calibration_id),
                )
            else:
                connection.execute(
                    f"""INSERT INTO graph_snapshot_edges(snapshot_id,pair_id,overlap_id)
                        SELECT ?,p.pair_id,(
                            SELECT o.overlap_id FROM overlap_evidence o
                            WHERE o.version_a=p.version_a AND o.version_b=p.version_b
                            ORDER BY o.created_at DESC,o.overlap_id DESC LIMIT 1)
                        FROM calibrated_edges c
                        JOIN pair_scores p ON p.pair_id=c.pair_id
                        {pair_filter_join}
                        JOIN graph_snapshot_datasets ga
                          ON ga.snapshot_id=? AND ga.version_id=p.version_a
                        JOIN graph_snapshot_datasets gb
                          ON gb.snapshot_id=? AND gb.version_id=p.version_b
                        WHERE c.calibration_id=? AND c.q_value IS NOT NULL""",
                    (snapshot_id, snapshot_id, snapshot_id, calibration_id),
                )
            if requested_pair_ids is not None:
                connection.execute("DROP TABLE temp.requested_snapshot_pairs")
        return snapshot_id

    def _snapshot_validation(
        self,
        connection: sqlite3.Connection,
        snapshot_id: str,
        *,
        require_current_versions: bool = False,
    ) -> dict[str, Any]:
        snapshot = connection.execute(
            """SELECT s.*,c.status AS calibration_status,c.stratum AS calibration_stratum,
                      c.expected_pair_count,c.family_hash,c.algorithm_hash
               FROM graph_snapshots s JOIN calibration_releases c USING(calibration_id)
               WHERE s.snapshot_id=?""",
            (snapshot_id,),
        ).fetchone()
        if not snapshot:
            raise KeyError(snapshot_id)
        errors: list[str] = []
        if snapshot["calibration_status"] not in {"calibrated", "published"}:
            errors.append("calibration is not finalized")
        if snapshot["calibration_stratum"] != snapshot["stratum"]:
            errors.append("calibration stratum differs from snapshot stratum")
        try:
            _validated_sha256(snapshot["manifest_checksum"] or "", field="manifest_checksum")
        except ValueError as exc:
            errors.append(str(exc))
        manifest_path = Path(snapshot["manifest_uri"])
        if manifest_path.is_file():
            if _sha256_path(manifest_path) != snapshot["manifest_checksum"]:
                errors.append("graph manifest file checksum does not match the snapshot")
            elif ":fr-collision-v2:" in snapshot["layout_version"]:
                try:
                    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    quality = manifest_payload["layout_quality"]
                    target = float(quality["target_minimum_separation"])
                    observed = float(quality["observed_minimum_separation"])
                    severe = int(quality["severe_collision_pair_count"])
                    if not all(math.isfinite(value) and value >= 0 for value in (target, observed)):
                        errors.append("graph layout separation diagnostics are invalid")
                    if severe:
                        errors.append(f"graph layout retains {severe} severe node collisions")
                    if target and observed < target * 0.8:
                        errors.append("graph layout minimum separation is below its release gate")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    errors.append("collision-aware graph manifest lacks valid layout diagnostics")
        else:
            registered_manifest = connection.execute(
                """SELECT 1 FROM artifacts WHERE kind='graph_manifest' AND uri=? AND checksum=?""",
                (snapshot["manifest_uri"], snapshot["manifest_checksum"]),
            ).fetchone()
            if not registered_manifest:
                errors.append("graph manifest is neither locally verifiable nor registered as an artifact")
        members = int(
            connection.execute(
                "SELECT COUNT(*) FROM graph_snapshot_datasets WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()[0]
        )
        if members == 0:
            errors.append("snapshot has no datasets")
        invalid_coordinates = int(
            connection.execute(
                """SELECT COUNT(*) FROM graph_snapshot_datasets
                   WHERE snapshot_id=? AND (
                     x IS NULL OR y IS NULL OR x!=x OR y!=y OR ABS(x)>1.0e308 OR ABS(y)>1.0e308
                   )""",
                (snapshot_id,),
            ).fetchone()[0]
        )
        if invalid_coordinates:
            errors.append(f"{invalid_coordinates} dataset coordinates are missing or non-finite")
        noncurrent = int(
            connection.execute(
                """SELECT COUNT(*) FROM graph_snapshot_datasets g
                   JOIN dataset_versions v ON v.version_id=g.version_id
                   JOIN datasets d ON d.dataset_uid=v.dataset_uid
                   WHERE g.snapshot_id=? AND d.current_version_id<>g.version_id""",
                (snapshot_id,),
            ).fetchone()[0]
        )
        if noncurrent and require_current_versions:
            errors.append(f"{noncurrent} dataset versions are not current at publication time")
        bound_edges = int(
            connection.execute(
                "SELECT COUNT(*) FROM graph_snapshot_edges WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()[0]
        )
        missing_endpoints = int(
            connection.execute(
                """SELECT COUNT(*) FROM graph_snapshot_edges se
                   JOIN pair_scores p ON p.pair_id=se.pair_id
                   LEFT JOIN graph_snapshot_datasets ga
                     ON ga.snapshot_id=se.snapshot_id AND ga.version_id=p.version_a
                   LEFT JOIN graph_snapshot_datasets gb
                     ON gb.snapshot_id=se.snapshot_id AND gb.version_id=p.version_b
                   WHERE se.snapshot_id=? AND (ga.version_id IS NULL OR gb.version_id IS NULL)""",
                (snapshot_id,),
            ).fetchone()[0]
        )
        if missing_endpoints:
            errors.append(f"{missing_endpoints} snapshot edges have missing endpoints")
        if snapshot["text_release_id"]:
            text_status = connection.execute(
                "SELECT status FROM text_releases WHERE text_release_id=?",
                (snapshot["text_release_id"],),
            ).fetchone()
            if not text_status or text_status["status"] != "finalized":
                errors.append("bound text release is not finalized")
        return {
            "snapshot_id": snapshot_id,
            "valid": not errors,
            "errors": errors,
            "dataset_count": members,
            "edge_count": bound_edges,
        }

    def validate_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.reader() as connection:
            return self._snapshot_validation(connection, snapshot_id)

    def publish_snapshot(
        self,
        snapshot_id: str,
        *,
        operator: str = "system",
        reason: str = "validated publication",
    ) -> None:
        """Validate and atomically swap the current graph pointer."""

        if not operator.strip() or not reason.strip():
            raise ValueError("snapshot publication requires an operator and reason")
        with self.transaction() as connection:
            snapshot = connection.execute(
                "SELECT * FROM graph_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if not snapshot or snapshot["status"] != "staging":
                raise ValueError("Only a staged graph snapshot can be published.")
            report = self._snapshot_validation(
                connection, snapshot_id, require_current_versions=True
            )
            if not report["valid"]:
                raise ValueError("Snapshot validation failed: " + "; ".join(report["errors"]))
            now = utc_now()
            stratum = snapshot["stratum"]
            previous = connection.execute(
                "SELECT value FROM settings WHERE key=?", (f"current_snapshot:{stratum}",)
            ).fetchone()
            connection.execute(
                "UPDATE graph_snapshots SET status='superseded' WHERE stratum=? AND status='published'",
                (stratum,),
            )
            connection.execute(
                "UPDATE graph_snapshots SET status='published', published_at=? WHERE snapshot_id=?",
                (now, snapshot_id),
            )
            connection.execute(
                "UPDATE calibration_releases SET status='published' WHERE calibration_id=?",
                (snapshot["calibration_id"],),
            )
            connection.execute(
                """INSERT INTO settings(key,value,updated_at) VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at""",
                (f"current_snapshot:{stratum}", snapshot_id, now),
            )
            connection.execute(
                """INSERT INTO snapshot_events(
                       event_id,snapshot_id,previous_snapshot_id,action,operator,reason,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    stable_id("snapshot_event", snapshot_id, now, "publish"), snapshot_id,
                    previous["value"] if previous else None, "publish", operator.strip(),
                    reason.strip(), now,
                ),
            )

    def rollback_snapshot(
        self,
        snapshot_id: str,
        *,
        operator: str,
        reason: str,
    ) -> None:
        """Atomically repoint a stratum to a previously published immutable snapshot."""

        if not operator.strip() or not reason.strip():
            raise ValueError("snapshot rollback requires an operator and reason")
        with self.transaction() as connection:
            target = connection.execute(
                "SELECT * FROM graph_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if not target or target["published_at"] is None or target["status"] not in {"published", "superseded"}:
                raise ValueError("Rollback target must be a previously published snapshot.")
            report = self._snapshot_validation(connection, snapshot_id)
            if not report["valid"]:
                raise ValueError("Rollback target validation failed: " + "; ".join(report["errors"]))
            stratum = target["stratum"]
            previous = connection.execute(
                "SELECT value FROM settings WHERE key=?", (f"current_snapshot:{stratum}",)
            ).fetchone()
            previous_id = previous["value"] if previous else None
            if previous_id == snapshot_id:
                raise ValueError("Snapshot is already current; no rollback is needed.")
            now = utc_now()
            connection.execute(
                "UPDATE graph_snapshots SET status='superseded' WHERE stratum=? AND status='published'",
                (stratum,),
            )
            connection.execute(
                "UPDATE graph_snapshots SET status='published' WHERE snapshot_id=?", (snapshot_id,)
            )
            connection.execute(
                """INSERT INTO settings(key,value,updated_at) VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at""",
                (f"current_snapshot:{stratum}", snapshot_id, now),
            )
            connection.execute(
                """INSERT INTO snapshot_events(
                       event_id,snapshot_id,previous_snapshot_id,action,operator,reason,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    stable_id("snapshot_event", snapshot_id, now, "rollback"), snapshot_id,
                    previous_id, "rollback", operator.strip(), reason.strip(), now,
                ),
            )

    def current_snapshot(self, stratum: str) -> dict[str, Any] | None:
        with self.reader() as connection:
            setting = connection.execute(
                "SELECT value FROM settings WHERE key=?", (f"current_snapshot:{stratum}",)
            ).fetchone()
            if not setting:
                return None
            row = connection.execute(
                "SELECT * FROM graph_snapshots WHERE snapshot_id=?", (setting["value"],)
            ).fetchone()
            return dict(row) if row else None

    def snapshot_diff(
        self,
        *,
        from_snapshot_id: str,
        to_snapshot_id: str,
        detail_limit: int = 500,
        q_change_limit: int = 100,
    ) -> dict[str, Any]:
        """Compare two immutable published graph snapshots in one stratum.

        Dataset membership is compared by stable logical ``dataset_uid``. A
        different version of the same logical dataset is therefore reported as
        an update, not as an addition plus a removal. Edges are intentionally
        compared by exact pair ID because a pair ID binds both dataset versions
        and the scoring algorithm. All detail arrays are bounded; their exact
        total counts remain available even when rows are truncated.
        """

        for field, value, maximum in (
            ("detail_limit", detail_limit, 5_000),
            ("q_change_limit", q_change_limit, 1_000),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{field} must be an integer between 1 and {maximum}")

        snapshot_columns = """snapshot_id,calibration_id,stratum,policy_hash,layout_version,
                              manifest_checksum,text_release_id,status,created_at,published_at"""

        def bounded_section(count: int, rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
            items = [dict(row) for row in rows]
            return {
                "count": count,
                "returned": len(items),
                "truncated": count > len(items),
                "items": items,
            }

        member_cte = """
            WITH old_members AS (
                SELECT d.dataset_uid,d.accession,d.platform,d.cohort,
                       g.version_id,g.community
                FROM graph_snapshot_datasets g
                JOIN dataset_versions v ON v.version_id=g.version_id
                JOIN datasets d ON d.dataset_uid=v.dataset_uid
                WHERE g.snapshot_id=?
            ), new_members AS (
                SELECT d.dataset_uid,d.accession,d.platform,d.cohort,
                       g.version_id,g.community
                FROM graph_snapshot_datasets g
                JOIN dataset_versions v ON v.version_id=g.version_id
                JOIN datasets d ON d.dataset_uid=v.dataset_uid
                WHERE g.snapshot_id=?
            )
        """
        edge_cte = """
            WITH old_edges AS (
                SELECT pair_id FROM graph_snapshot_edges WHERE snapshot_id=?
            ), new_edges AS (
                SELECT pair_id FROM graph_snapshot_edges WHERE snapshot_id=?
            )
        """

        with self.reader() as connection:
            old_snapshot = connection.execute(
                f"""SELECT {snapshot_columns} FROM graph_snapshots
                    WHERE snapshot_id=? AND published_at IS NOT NULL
                      AND status IN ('published','superseded')""",
                (from_snapshot_id,),
            ).fetchone()
            if not old_snapshot:
                raise KeyError(
                    f"From snapshot is not an existing published snapshot: {from_snapshot_id}"
                )
            new_snapshot = connection.execute(
                f"""SELECT {snapshot_columns} FROM graph_snapshots
                    WHERE snapshot_id=? AND published_at IS NOT NULL
                      AND status IN ('published','superseded')""",
                (to_snapshot_id,),
            ).fetchone()
            if not new_snapshot:
                raise KeyError(
                    f"To snapshot is not an existing published snapshot: {to_snapshot_id}"
                )
            if old_snapshot["stratum"] != new_snapshot["stratum"]:
                raise ValueError(
                    "Snapshot comparison requires both snapshots to have the same stratum."
                )

            dataset_counts = connection.execute(
                member_cte
                + """
                SELECT
                  (SELECT COUNT(*) FROM new_members n LEFT JOIN old_members o USING(dataset_uid)
                   WHERE o.dataset_uid IS NULL) AS added_count,
                  (SELECT COUNT(*) FROM old_members o LEFT JOIN new_members n USING(dataset_uid)
                   WHERE n.dataset_uid IS NULL) AS removed_count,
                  (SELECT COUNT(*) FROM old_members o JOIN new_members n USING(dataset_uid)
                   WHERE o.version_id<>n.version_id) AS updated_count,
                  (SELECT COUNT(*) FROM old_members o JOIN new_members n USING(dataset_uid)
                   WHERE o.version_id=n.version_id AND (
                     o.community<>n.community
                     OR (o.community IS NULL AND n.community IS NOT NULL)
                     OR (o.community IS NOT NULL AND n.community IS NULL)
                   )) AS community_changed_count
                """,
                (from_snapshot_id, to_snapshot_id),
            ).fetchone()
            added_rows = connection.execute(
                member_cte
                + """
                SELECT n.dataset_uid,n.accession,n.platform,n.cohort,
                       n.version_id AS to_version_id,n.community AS to_community
                FROM new_members n LEFT JOIN old_members o USING(dataset_uid)
                WHERE o.dataset_uid IS NULL
                ORDER BY n.accession,n.platform,n.cohort,n.dataset_uid LIMIT ?
                """,
                (from_snapshot_id, to_snapshot_id, detail_limit),
            ).fetchall()
            removed_rows = connection.execute(
                member_cte
                + """
                SELECT o.dataset_uid,o.accession,o.platform,o.cohort,
                       o.version_id AS from_version_id,o.community AS from_community
                FROM old_members o LEFT JOIN new_members n USING(dataset_uid)
                WHERE n.dataset_uid IS NULL
                ORDER BY o.accession,o.platform,o.cohort,o.dataset_uid LIMIT ?
                """,
                (from_snapshot_id, to_snapshot_id, detail_limit),
            ).fetchall()
            updated_rows = connection.execute(
                member_cte
                + """
                SELECT n.dataset_uid,n.accession,n.platform,n.cohort,
                       o.version_id AS from_version_id,n.version_id AS to_version_id,
                       o.community AS from_community,n.community AS to_community
                FROM old_members o JOIN new_members n USING(dataset_uid)
                WHERE o.version_id<>n.version_id
                ORDER BY n.accession,n.platform,n.cohort,n.dataset_uid LIMIT ?
                """,
                (from_snapshot_id, to_snapshot_id, detail_limit),
            ).fetchall()
            community_rows = connection.execute(
                member_cte
                + """
                SELECT n.dataset_uid,n.accession,n.platform,n.cohort,n.version_id,
                       o.community AS from_community,n.community AS to_community
                FROM old_members o JOIN new_members n USING(dataset_uid)
                WHERE o.version_id=n.version_id AND (
                  o.community<>n.community
                  OR (o.community IS NULL AND n.community IS NOT NULL)
                  OR (o.community IS NOT NULL AND n.community IS NULL)
                )
                ORDER BY n.accession,n.platform,n.cohort,n.dataset_uid LIMIT ?
                """,
                (from_snapshot_id, to_snapshot_id, detail_limit),
            ).fetchall()

            edge_counts = connection.execute(
                edge_cte
                + """
                SELECT
                  (SELECT COUNT(*) FROM new_edges n LEFT JOIN old_edges o USING(pair_id)
                   WHERE o.pair_id IS NULL) AS added_count,
                  (SELECT COUNT(*) FROM old_edges o LEFT JOIN new_edges n USING(pair_id)
                   WHERE n.pair_id IS NULL) AS removed_count,
                  (SELECT COUNT(*) FROM old_edges o JOIN new_edges n USING(pair_id)) AS common_count,
                  (SELECT COUNT(*) FROM old_edges o JOIN new_edges n USING(pair_id)
                   JOIN calibrated_edges oc ON oc.calibration_id=? AND oc.pair_id=o.pair_id
                   JOIN calibrated_edges nc ON nc.calibration_id=? AND nc.pair_id=n.pair_id
                   WHERE oc.q_value IS NOT NULL AND nc.q_value IS NOT NULL) AS q_comparable_count,
                  (SELECT COUNT(*) FROM old_edges o JOIN new_edges n USING(pair_id)
                   JOIN calibrated_edges oc ON oc.calibration_id=? AND oc.pair_id=o.pair_id
                   JOIN calibrated_edges nc ON nc.calibration_id=? AND nc.pair_id=n.pair_id
                   WHERE oc.q_value IS NOT NULL AND nc.q_value IS NOT NULL
                     AND oc.q_value<>nc.q_value) AS q_changed_count
                """,
                (
                    from_snapshot_id,
                    to_snapshot_id,
                    old_snapshot["calibration_id"],
                    new_snapshot["calibration_id"],
                    old_snapshot["calibration_id"],
                    new_snapshot["calibration_id"],
                ),
            ).fetchone()
            q_change_rows = connection.execute(
                edge_cte
                + """
                SELECT o.pair_id,p.version_a,p.version_b,
                       oc.q_value AS from_q_value,nc.q_value AS to_q_value,
                       (nc.q_value-oc.q_value) AS q_value_delta,
                       ABS(nc.q_value-oc.q_value) AS absolute_q_value_delta
                FROM old_edges o JOIN new_edges n USING(pair_id)
                JOIN pair_scores p ON p.pair_id=o.pair_id
                JOIN calibrated_edges oc ON oc.calibration_id=? AND oc.pair_id=o.pair_id
                JOIN calibrated_edges nc ON nc.calibration_id=? AND nc.pair_id=n.pair_id
                WHERE oc.q_value IS NOT NULL AND nc.q_value IS NOT NULL
                  AND oc.q_value<>nc.q_value
                ORDER BY absolute_q_value_delta DESC,o.pair_id ASC LIMIT ?
                """,
                (
                    from_snapshot_id,
                    to_snapshot_id,
                    old_snapshot["calibration_id"],
                    new_snapshot["calibration_id"],
                    q_change_limit,
                ),
            ).fetchall()

        return {
            "diff_id": stable_id("snapshot_diff", from_snapshot_id, to_snapshot_id),
            "from_snapshot_id": from_snapshot_id,
            "to_snapshot_id": to_snapshot_id,
            "stratum": old_snapshot["stratum"],
            "datasets": {
                "added": bounded_section(dataset_counts["added_count"], added_rows),
                "removed": bounded_section(dataset_counts["removed_count"], removed_rows),
                "version_updated": bounded_section(
                    dataset_counts["updated_count"], updated_rows
                ),
                "community_changed": bounded_section(
                    dataset_counts["community_changed_count"], community_rows
                ),
            },
            "edges": {
                "added_count": edge_counts["added_count"],
                "removed_count": edge_counts["removed_count"],
                "common_count": edge_counts["common_count"],
                "q_comparable_count": edge_counts["q_comparable_count"],
                "q_value_changes": bounded_section(
                    edge_counts["q_changed_count"], q_change_rows
                ),
            },
            "provenance": {
                "direction": "from_snapshot_id -> to_snapshot_id",
                "from_snapshot": dict(old_snapshot),
                "to_snapshot": dict(new_snapshot),
                "limits": {
                    "dataset_detail_limit_per_category": detail_limit,
                    "q_value_change_limit": q_change_limit,
                },
                "identity_rules": {
                    "dataset": "dataset_uid",
                    "edge": "pair_id",
                    "community_change": "same dataset_uid and version_id",
                    "q_value_delta": "to_q_value - from_q_value",
                },
            },
        }

    def graph_payload(
        self,
        *,
        snapshot_id: str,
        q_max: float = 0.05,
        independent_only: bool = False,
        edge_limit: int = 50_000,
    ) -> dict[str, Any]:
        if not 0 <= q_max <= 1:
            raise ValueError("q_max must be between 0 and 1")
        edge_limit = max(1, min(int(edge_limit), 100_000))
        with self.reader() as connection:
            snapshot = connection.execute(
                "SELECT * FROM graph_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if not snapshot:
                raise KeyError(snapshot_id)
            if snapshot["published_at"] is None:
                raise ValueError("Only a published snapshot can be served.")
            independent_stratum = (
                snapshot["stratum"][:-7] + ":independent"
                if snapshot["stratum"].endswith(":global")
                else snapshot["stratum"]
            )
            nodes = connection.execute(
                """SELECT g.version_id,g.x,g.y,g.community,d.dataset_uid,d.accession,d.platform,d.cohort,
                          v.sample_count,v.metadata_json,v.feature_hash,v.config_hash
                   FROM graph_snapshot_datasets g
                   JOIN dataset_versions v ON v.version_id=g.version_id
                   JOIN datasets d ON d.dataset_uid=v.dataset_uid
                   WHERE g.snapshot_id=? ORDER BY d.accession""",
                (snapshot_id,),
            ).fetchall()
            independence_clause = "AND COALESCE(o.discovery_excluded,0)=0" if independent_only else ""
            edges = connection.execute(
                f"""SELECT p.pair_id,p.version_a,p.version_b,p.algorithm_hash,p.cskl,
                            c.p_value,c.q_value,c.cskl_similarity_percentile,
                            (SELECT independent.q_value
                             FROM calibrated_edges independent
                             JOIN calibration_releases release
                               ON release.calibration_id=independent.calibration_id
                             WHERE independent.pair_id=p.pair_id AND release.stratum=?
                               AND release.status IN ('calibrated','published')
                             ORDER BY release.created_at DESC LIMIT 1) AS independent_q_value,
                            o.shared_count,o.fraction_a,o.fraction_b,o.jaccard,o.overlap_coefficient,
                            o.classification,o.discovery_excluded,se.overlap_id,
                            t.cosine_similarity AS specter2_cosine,
                            t.similarity_percentile AS specter2_percentile,
                            ? AS text_release_id
                     FROM graph_snapshot_edges se
                     JOIN calibrated_edges c
                       ON c.calibration_id=? AND c.pair_id=se.pair_id
                     JOIN pair_scores p ON p.pair_id=c.pair_id
                     LEFT JOIN overlap_evidence o ON o.overlap_id=se.overlap_id
                     LEFT JOIN text_pair_scores t
                       ON t.text_release_id=? AND t.version_a=p.version_a AND t.version_b=p.version_b
                     JOIN graph_snapshot_datasets ga
                       ON ga.snapshot_id=? AND ga.version_id=p.version_a
                    JOIN graph_snapshot_datasets gb
                      ON gb.snapshot_id=? AND gb.version_id=p.version_b
                     WHERE se.snapshot_id=? AND c.q_value<=? {independence_clause}
                     ORDER BY c.q_value ASC,p.cskl ASC LIMIT ?""",
                (
                    independent_stratum, snapshot["text_release_id"], snapshot["calibration_id"],
                    snapshot["text_release_id"], snapshot_id, snapshot_id, snapshot_id,
                    q_max, edge_limit,
                ),
            ).fetchall()
            return {
                "snapshot": dict(snapshot),
                "nodes": [
                    {**dict(row), "metadata": json.loads(row["metadata_json"])} for row in nodes
                ],
                "edges": [dict(row) for row in edges],
            }

    def graph_overview(
        self,
        *,
        snapshot_id: str,
        q_max: float = 0.05,
        independent_only: bool = True,
    ) -> dict[str, Any]:
        """Return community supernodes instead of thousands of browser nodes."""

        if not 0 <= q_max <= 1:
            raise ValueError("q_max must be between 0 and 1")
        with self.reader() as connection:
            snapshot = connection.execute(
                "SELECT * FROM graph_snapshots WHERE snapshot_id=? AND published_at IS NOT NULL",
                (snapshot_id,),
            ).fetchone()
            if not snapshot:
                raise KeyError(snapshot_id)
            communities = connection.execute(
                """SELECT COALESCE(community,'unassigned') AS community,
                          COUNT(*) AS dataset_count,AVG(x) AS x,AVG(y) AS y,
                          SUM(v.sample_count) AS sample_count
                   FROM graph_snapshot_datasets g
                   JOIN dataset_versions v ON v.version_id=g.version_id
                   WHERE g.snapshot_id=? GROUP BY COALESCE(community,'unassigned')
                   ORDER BY dataset_count DESC,community""",
                (snapshot_id,),
            ).fetchall()
            independence = "AND COALESCE(o.discovery_excluded,0)=0" if independent_only else ""
            relationships = connection.execute(
                f"""WITH grouped AS (
                       SELECT CASE WHEN COALESCE(ga.community,'unassigned')<=COALESCE(gb.community,'unassigned')
                                   THEN COALESCE(ga.community,'unassigned') ELSE COALESCE(gb.community,'unassigned') END AS source,
                              CASE WHEN COALESCE(ga.community,'unassigned')<=COALESCE(gb.community,'unassigned')
                                   THEN COALESCE(gb.community,'unassigned') ELSE COALESCE(ga.community,'unassigned') END AS target,
                              c.q_value,p.cskl
                       FROM graph_snapshot_edges se
                       JOIN calibrated_edges c
                         ON c.calibration_id=? AND c.pair_id=se.pair_id
                       JOIN pair_scores p ON p.pair_id=se.pair_id
                       JOIN graph_snapshot_datasets ga
                         ON ga.snapshot_id=se.snapshot_id AND ga.version_id=p.version_a
                       JOIN graph_snapshot_datasets gb
                         ON gb.snapshot_id=se.snapshot_id AND gb.version_id=p.version_b
                       LEFT JOIN overlap_evidence o ON o.overlap_id=se.overlap_id
                       WHERE se.snapshot_id=? AND c.q_value<=? {independence}
                     )
                     SELECT source,target,COUNT(*) AS edge_count,MIN(q_value) AS min_q_value,
                            AVG(cskl) AS mean_cskl
                     FROM grouped GROUP BY source,target ORDER BY edge_count DESC,source,target""",
                (snapshot["calibration_id"], snapshot_id, q_max),
            ).fetchall()
        return {
            "snapshot": dict(snapshot),
            "communities": [dict(row) for row in communities],
            "relationships": [dict(row) for row in relationships],
            "filters": {"q_max": q_max, "independent_only": independent_only},
        }

    def graph_neighborhood(
        self,
        *,
        snapshot_id: str,
        version_id: str,
        q_max: float = 0.05,
        independent_only: bool = True,
        limit: int = 250,
    ) -> dict[str, Any]:
        """Return a bounded, ranked one-hop expansion for progressive graph loading."""

        if not 0 <= q_max <= 1:
            raise ValueError("q_max must be between 0 and 1")
        limit = max(1, min(int(limit), 5_000))
        with self.reader() as connection:
            snapshot = connection.execute(
                "SELECT * FROM graph_snapshots WHERE snapshot_id=? AND published_at IS NOT NULL",
                (snapshot_id,),
            ).fetchone()
            if not snapshot:
                raise KeyError(snapshot_id)
            member = connection.execute(
                """SELECT 1 FROM graph_snapshot_datasets
                   WHERE snapshot_id=? AND version_id=?""",
                (snapshot_id, version_id),
            ).fetchone()
            if not member:
                raise KeyError(version_id)
            independence = "AND COALESCE(o.discovery_excluded,0)=0" if independent_only else ""
            edges = connection.execute(
                f"""SELECT p.pair_id,p.version_a,p.version_b,p.algorithm_hash,p.cskl,
                            c.p_value,c.q_value,c.cskl_similarity_percentile,
                            o.shared_count,o.fraction_a,o.fraction_b,o.jaccard,o.overlap_coefficient,
                            o.classification,o.discovery_excluded,se.overlap_id,
                            t.cosine_similarity AS specter2_cosine,
                            t.similarity_percentile AS specter2_percentile,
                            ? AS text_release_id
                     FROM graph_snapshot_edges se
                     JOIN calibrated_edges c
                       ON c.calibration_id=? AND c.pair_id=se.pair_id
                     JOIN pair_scores p ON p.pair_id=se.pair_id
                     LEFT JOIN overlap_evidence o ON o.overlap_id=se.overlap_id
                     LEFT JOIN text_pair_scores t
                       ON t.text_release_id=? AND t.version_a=p.version_a AND t.version_b=p.version_b
                     WHERE se.snapshot_id=? AND c.q_value<=?
                       AND (p.version_a=? OR p.version_b=?) {independence}
                     ORDER BY c.q_value,p.cskl,p.pair_id LIMIT ?""",
                (
                    snapshot["text_release_id"], snapshot["calibration_id"],
                    snapshot["text_release_id"], snapshot_id, q_max, version_id, version_id, limit,
                ),
            ).fetchall()
            endpoint_ids = {version_id}
            for edge in edges:
                endpoint_ids.update((edge["version_a"], edge["version_b"]))
            placeholders = ",".join("?" for _ in endpoint_ids)
            nodes = connection.execute(
                f"""SELECT g.version_id,g.x,g.y,g.community,d.dataset_uid,d.accession,d.platform,d.cohort,
                            v.sample_count,v.metadata_json,v.feature_hash,v.config_hash
                     FROM graph_snapshot_datasets g
                     JOIN dataset_versions v ON v.version_id=g.version_id
                     JOIN datasets d ON d.dataset_uid=v.dataset_uid
                     WHERE g.snapshot_id=? AND g.version_id IN ({placeholders}) ORDER BY d.accession""",
                (snapshot_id, *sorted(endpoint_ids)),
            ).fetchall()
        return {
            "snapshot": dict(snapshot),
            "center_version_id": version_id,
            "nodes": [{**dict(row), "metadata": json.loads(row["metadata_json"])} for row in nodes],
            "edges": [dict(row) for row in edges],
            "filters": {"q_max": q_max, "independent_only": independent_only, "limit": limit},
        }

    def enqueue_job(
        self,
        *,
        kind: str,
        job_key: str,
        input_fingerprint: str,
        payload: Mapping[str, Any],
        max_attempts: int = 5,
    ) -> str:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        job_id = stable_id("job", kind, job_key, input_fingerprint)
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO jobs(
                       job_id,kind,job_key,input_fingerprint,payload_json,status,attempts,max_attempts,
                       created_at,updated_at)
                   VALUES(?,?,?,?,?,'queued',0,?,?,?) ON CONFLICT(job_id) DO NOTHING""",
                (
                    job_id, kind, job_key, input_fingerprint, canonical_json(payload),
                    max_attempts, now, now,
                ),
            )
        return job_id

    def claim_jobs(
        self,
        *,
        worker_id: str,
        kinds: Sequence[str] | None = None,
        limit: int = 1,
        lease_seconds: int = 300,
    ) -> list[dict[str, Any]]:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id is required")
        lease_seconds = int(lease_seconds)
        if not 5 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be between 5 and 86400")
        limit = max(1, min(int(limit), 100))
        now_value = datetime.now(timezone.utc)
        now = now_value.isoformat()
        lease_expires = (now_value + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction() as connection:
            self._reap_expired_jobs(connection, now=now)
            params: list[Any] = [now]
            kind_clause = ""
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                kind_clause = f" AND kind IN ({placeholders})"
                params.extend(kinds)
            params.append(limit)
            rows = connection.execute(
                f"""SELECT * FROM jobs
                    WHERE status IN ('queued','retry')
                      AND (next_retry_at IS NULL OR next_retry_at<=?) {kind_clause}
                    ORDER BY created_at ASC LIMIT ?""",
                params,
            ).fetchall()
            ids = [row["job_id"] for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    f"""UPDATE jobs SET status='running',worker_id=?,attempts=attempts+1,
                                       updated_at=?,heartbeat_at=?,lease_expires_at=?,
                                       error_code=NULL,error_detail=NULL
                        WHERE job_id IN ({placeholders})""",
                    [worker_id, now, now, lease_expires, *ids],
                )
                rows = connection.execute(
                    f"SELECT * FROM jobs WHERE job_id IN ({placeholders}) ORDER BY created_at", ids
                ).fetchall()
            return [
                {
                    **dict(row),
                    "payload": json.loads(row["payload_json"]),
                    "progress": json.loads(row["progress_json"] or "{}"),
                }
                for row in rows
            ]

    @staticmethod
    def _reap_expired_jobs(connection: sqlite3.Connection, *, now: str) -> dict[str, int]:
        expired = connection.execute(
            """SELECT job_id,attempts,max_attempts FROM jobs
               WHERE status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at<=?""",
            (now,),
        ).fetchall()
        retried = 0
        dead = 0
        for row in expired:
            can_retry = row["attempts"] < row["max_attempts"]
            status = "retry" if can_retry else "dead"
            retried += int(can_retry)
            dead += int(not can_retry)
            connection.execute(
                """UPDATE jobs SET status=?,next_retry_at=?,worker_id=NULL,heartbeat_at=NULL,
                                   lease_expires_at=NULL,error_code='LEASE_EXPIRED',
                                   error_detail='Worker lease expired before completion.',updated_at=?
                   WHERE job_id=? AND status='running'""",
                (status, now if can_retry else None, now, row["job_id"]),
            )
        return {"reaped": len(expired), "retry": retried, "dead": dead}

    def reap_expired_jobs(self) -> dict[str, int]:
        with self.transaction() as connection:
            return self._reap_expired_jobs(connection, now=utc_now())

    def heartbeat_job(self, job_id: str, *, worker_id: str, lease_seconds: int = 300) -> str:
        worker_id = worker_id.strip()
        lease_seconds = int(lease_seconds)
        if not worker_id or not 5 <= lease_seconds <= 86_400:
            raise ValueError("valid worker_id and lease_seconds in [5, 86400] are required")
        now_value = datetime.now(timezone.utc)
        now = now_value.isoformat()
        expires = (now_value + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET heartbeat_at=?,lease_expires_at=?,updated_at=?
                   WHERE job_id=? AND status='running' AND worker_id=?
                     AND lease_expires_at>?""",
                (now, expires, now, job_id, worker_id, now),
            )
            if cursor.rowcount != 1:
                raise ValueError("Job is not actively leased by this worker.")
        return expires

    def update_job_progress(
        self,
        job_id: str,
        *,
        worker_id: str,
        progress: Mapping[str, Any],
    ) -> None:
        """Persist an idempotent resume cursor while the caller owns the lease."""

        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET progress_json=?,updated_at=?
                   WHERE job_id=? AND status='running' AND worker_id=?
                     AND lease_expires_at>?""",
                (canonical_json(progress), now, job_id, worker_id, now),
            )
            if cursor.rowcount != 1:
                raise ValueError("Job progress can be updated only by its active lease owner.")

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.reader() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload_json"])
        result["progress"] = json.loads(result["progress_json"] or "{}")
        return result

    def complete_job(self, job_id: str, *, worker_id: str | None = None) -> None:
        with self.transaction() as connection:
            ownership = " AND worker_id=?" if worker_id is not None else ""
            parameters: list[Any] = [utc_now(), job_id]
            if worker_id is not None:
                parameters.append(worker_id)
            cursor = connection.execute(
                f"""UPDATE jobs SET status='succeeded',worker_id=NULL,next_retry_at=NULL,
                                    heartbeat_at=NULL,lease_expires_at=NULL,updated_at=?
                    WHERE job_id=? AND status='running'{ownership}""",
                parameters,
            )
            if cursor.rowcount != 1:
                raise ValueError("Only the worker holding a running job lease can complete it.")

    def fail_job(
        self,
        job_id: str,
        *,
        error_code: str,
        error_detail: str,
        retryable: bool,
        worker_id: str | None = None,
    ) -> str:
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["status"] != "running":
                raise ValueError("Only a running job can fail.")
            if worker_id is not None and row["worker_id"] != worker_id:
                raise ValueError("Only the worker holding a running job lease can fail it.")
            can_retry = retryable and row["attempts"] < row["max_attempts"]
            status = "retry" if can_retry else "dead"
            next_retry = None
            if can_retry:
                base_seconds = min(86_400, 30 * (2 ** max(row["attempts"] - 1, 0)))
                jitter = int(hashlib.sha256(job_id.encode()).hexdigest()[:4], 16) % max(base_seconds // 4, 1)
                next_retry = (
                    datetime.now(timezone.utc) + timedelta(seconds=base_seconds + jitter)
                ).isoformat()
            connection.execute(
                """UPDATE jobs SET status=?,worker_id=NULL,next_retry_at=?,heartbeat_at=NULL,
                                   lease_expires_at=NULL,error_code=?,error_detail=?,updated_at=?
                   WHERE job_id=?""",
                (
                    status, next_retry, error_code, error_detail[:8_000], utc_now(), job_id,
                ),
            )
            return status

    def requeue_job(self, job_id: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET status='queued',attempts=0,next_retry_at=NULL,worker_id=NULL,
                                  heartbeat_at=NULL,lease_expires_at=NULL,error_code=NULL,
                                  error_detail=NULL,updated_at=?
                   WHERE job_id=? AND status IN ('dead','cancelled')""",
                (utc_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only a dead or cancelled job can be requeued.")

    def list_jobs(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1_000))
        with self.reader() as connection:
            if status:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [
                {
                    **dict(row),
                    "payload": json.loads(row["payload_json"]),
                    "progress": json.loads(row["progress_json"] or "{}"),
                }
                for row in rows
            ]

    def health(self) -> dict[str, Any]:
        with self.reader() as connection:
            schema = connection.execute(
                "SELECT value FROM catalog_meta WHERE key='schema_version'"
            ).fetchone()
            counts = {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("datasets", "dataset_versions", "pair_scores", "graph_snapshots", "jobs")
            }
            backlog = {
                row["status"]: int(row["count"])
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM jobs GROUP BY status"
                ).fetchall()
            }
            return {
                "status": "ok",
                "schema_version": int(schema["value"]) if schema else None,
                "database": str(self.path),
                "counts": counts,
                "jobs": backlog,
                "time": utc_now(),
            }
