"""Frozen OLS term snapshots and audits for generated ontology candidates.

LLM output is never treated as an ontology resolver.  This module snapshots
the official EMBL-EBI OLS records for every proposed CURIE, then checks that a
candidate surface label is an exact canonical label or synonym.  Discordant
items remain reviewable candidates but fail the manuscript release gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx
from cskl_pipeline.scale.store import atomic_write_json, read_json

from .catalog import canonical_json, stable_id

OLS_API = "https://www.ebi.ac.uk/ols4/api"
ONTOLOGY_AUDIT_VERSION = "ols-candidate-audit-v1"
OLS_LABEL_RESOLVER_VERSION = "ols-exact-label-resolver-v1"

_SLUGS = {
    "NCBITaxon": "ncbitaxon",
    "UBERON": "uberon",
    "MONDO": "mondo",
    "CL": "cl",
    "OBI": "obi",
    "EFO": "efo",
    "CHEBI": "chebi",
}


def _atomic_cache_json(path: Path, value: Any) -> None:
    """Publish one cache record safely across threads and worker processes."""

    encoded = canonical_json(value).encode("utf-8")
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
                time.sleep(0.005 * (2**attempt))
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class OLSLabelResolution:
    """One fail-closed resolution decision against frozen official OLS responses."""

    status: str
    surface_label: str
    allowed_ontologies: tuple[str, ...]
    resolver_version: str
    source: str
    ontology: str | None = None
    curie: str | None = None
    canonical_label: str | None = None
    match_kind: str | None = None
    candidate_count: int = 0
    query_response_sha256: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "surface_label": self.surface_label,
            "allowed_ontologies": list(self.allowed_ontologies),
            "resolver_version": self.resolver_version,
            "source": self.source,
            "ontology": self.ontology,
            "curie": self.curie,
            "canonical_label": self.canonical_label,
            "match_kind": self.match_kind,
            "candidate_count": self.candidate_count,
            "query_response_sha256": dict(self.query_response_sha256),
        }


class OLSLabelResolver:
    """Resolve evidence-grounded labels without trusting model-produced CURIEs.

    Every ontology query is frozen in a content-addressed response file. A small
    deterministic query record points at that response, so interrupted runs can
    resume without repeating completed OLS requests. Only one non-obsolete term
    whose canonical label or synonym exactly matches after Unicode normalization
    is accepted; zero or multiple terms fail to an explicit unknown state.
    """

    version = OLS_LABEL_RESOLVER_VERSION
    source = OLS_API

    def __init__(
        self,
        cache_directory: str | Path,
        *,
        http_client: httpx.Client | None = None,
        force_refresh: bool = False,
        max_concurrent_requests: int = 4,
    ) -> None:
        if not 1 <= max_concurrent_requests <= 8:
            raise ValueError("max_concurrent_requests must be between 1 and 8")
        self.cache_directory = Path(cache_directory).resolve()
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self._owned_client = http_client is None
        self._client = http_client or httpx.Client(
            follow_redirects=True,
            headers={"User-Agent": "C-SKL-Atlas/0.5 ontology-label-resolver"},
        )
        self.force_refresh = force_refresh
        self._query_locks: dict[str, threading.Lock] = {}
        self._query_locks_guard = threading.Lock()
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)

    def close(self) -> None:
        if self._owned_client:
            self._client.close()

    def __enter__(self) -> OLSLabelResolver:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _query_payload(ontology: str, label: str) -> dict[str, str]:
        return {
            "resolver_version": OLS_LABEL_RESOLVER_VERSION,
            "source": OLS_API,
            "ontology": ontology,
            "normalized_label": normalized_term(label),
        }

    def _query_paths(self, ontology: str, label: str) -> tuple[Path, str]:
        query_hash = hashlib.sha256(
            canonical_json(self._query_payload(ontology, label)).encode()
        ).hexdigest()
        return (
            self.cache_directory / "queries" / query_hash[:2] / f"{query_hash}.json",
            query_hash,
        )

    def _search_one(self, ontology: str, label: str) -> tuple[list[dict[str, Any]], str]:
        if ontology not in _SLUGS:
            raise ValueError(f"Unsupported ontology namespace: {ontology}")
        query_path, query_hash = self._query_paths(ontology, label)
        with self._query_locks_guard:
            query_lock = self._query_locks.setdefault(query_hash, threading.Lock())
        with query_lock:
            return self._search_one_locked(
                ontology=ontology,
                label=label,
                query_path=query_path,
                query_hash=query_hash,
            )

    def _search_one_locked(
        self,
        *,
        ontology: str,
        label: str,
        query_path: Path,
        query_hash: str,
    ) -> tuple[list[dict[str, Any]], str]:
        if query_path.is_file() and not self.force_refresh:
            record = read_json(query_path)
            if (
                record.get("schema") != OLS_LABEL_RESOLVER_VERSION
                or record.get("query_hash") != query_hash
                or record.get("query") != self._query_payload(ontology, label)
            ):
                raise ValueError(f"Invalid ontology resolver cache record: {query_path}")
            response_hash = str(record.get("response_sha256") or "")
            response_path = (
                self.cache_directory
                / "responses"
                / response_hash[:2]
                / f"{response_hash}.json"
            )
            if not response_hash or not response_path.is_file():
                raise ValueError(f"Missing ontology resolver response: {response_path}")
            payload = read_json(response_path)
            if hashlib.sha256(canonical_json(payload).encode()).hexdigest() != response_hash:
                raise ValueError(f"Ontology resolver response checksum mismatch: {response_path}")
            return self._matching_documents(payload, ontology, label), response_hash

        with self._request_slots:
            payload = _request_json(
                self._client,
                f"{OLS_API}/search",
                params={
                    "q": label,
                    "ontology": _SLUGS[ontology],
                    "exact": "true",
                    "rows": "1000",
                },
            )
        response_hash = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        response_path = (
            self.cache_directory
            / "responses"
            / response_hash[:2]
            / f"{response_hash}.json"
        )
        if not response_path.is_file():
            _atomic_cache_json(response_path, payload)
        _atomic_cache_json(
            query_path,
            {
                "schema": OLS_LABEL_RESOLVER_VERSION,
                "query_hash": query_hash,
                "query": self._query_payload(ontology, label),
                "response_sha256": response_hash,
            },
        )
        return self._matching_documents(payload, ontology, label), response_hash

    @staticmethod
    def _matching_documents(
        payload: Mapping[str, Any], ontology: str, label: str
    ) -> list[dict[str, Any]]:
        target = normalized_term(label)
        response = payload.get("response")
        documents = response.get("docs") if isinstance(response, Mapping) else []
        if isinstance(response, Mapping):
            reported_count = int(response.get("numFound") or len(documents or []))
            if reported_count > len(documents or []):
                raise ValueError(
                    "OLS label response was truncated; refusing an ambiguity decision"
                )
        matches: dict[str, dict[str, Any]] = {}
        for document in documents or []:
            if not isinstance(document, Mapping):
                continue
            namespace = str(
                document.get("ontology_prefix")
                or document.get("ontology_name")
                or ""
            ).casefold()
            if namespace and namespace not in {ontology.casefold(), _SLUGS[ontology]}:
                continue
            obsolete = document.get("is_obsolete", False)
            if obsolete is True or str(obsolete).casefold() == "true":
                continue
            curie = str(document.get("obo_id") or "").strip()
            if not curie:
                short_form = str(document.get("short_form") or "").strip()
                prefix = f"{ontology}_"
                if short_form.startswith(prefix):
                    curie = f"{ontology}:{short_form[len(prefix):]}"
            if not curie.startswith(f"{ontology}:"):
                continue
            canonical_label = str(document.get("label") or "").strip()
            synonyms_value = document.get("synonym") or document.get("synonyms") or []
            if isinstance(synonyms_value, str):
                synonyms: Sequence[object] = (synonyms_value,)
            elif isinstance(synonyms_value, Sequence):
                synonyms = synonyms_value
            else:
                synonyms = ()
            canonical_match = normalized_term(canonical_label) == target
            synonym_match = any(normalized_term(str(value)) == target for value in synonyms)
            if not canonical_match and not synonym_match:
                continue
            match_kind = "canonical" if canonical_match else "synonym"
            existing = matches.get(curie)
            if existing is None or match_kind == "canonical":
                matches[curie] = {
                    "ontology": ontology,
                    "curie": curie,
                    "canonical_label": canonical_label or None,
                    "match_kind": match_kind,
                }
        return [matches[curie] for curie in sorted(matches)]

    def resolve(
        self,
        *,
        label: str,
        allowed_ontologies: Iterable[str],
    ) -> OLSLabelResolution:
        ontologies = tuple(sorted(set(allowed_ontologies)))
        if not normalized_term(label):
            return OLSLabelResolution(
                status="unresolved",
                surface_label=label,
                allowed_ontologies=ontologies,
                resolver_version=self.version,
                source=self.source,
            )
        matches: list[dict[str, Any]] = []
        response_hashes: list[tuple[str, str]] = []
        for ontology in ontologies:
            ontology_matches, response_hash = self._search_one(ontology, label)
            matches.extend(ontology_matches)
            response_hashes.append((ontology, response_hash))
        unique = {(item["ontology"], item["curie"]): item for item in matches}
        if len(unique) != 1:
            return OLSLabelResolution(
                status="ambiguous" if unique else "unresolved",
                surface_label=label,
                allowed_ontologies=ontologies,
                resolver_version=self.version,
                source=self.source,
                candidate_count=len(unique),
                query_response_sha256=tuple(response_hashes),
            )
        match = next(iter(unique.values()))
        return OLSLabelResolution(
            status="resolved",
            surface_label=label,
            allowed_ontologies=ontologies,
            resolver_version=self.version,
            source=self.source,
            ontology=match["ontology"],
            curie=match["curie"],
            canonical_label=match["canonical_label"],
            match_kind=match["match_kind"],
            candidate_count=1,
            query_response_sha256=tuple(response_hashes),
        )


def normalized_term(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", str(value)).casefold()
        if character.isalnum()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("GSE*.json") if path.is_file())


def iter_candidate_assertions(directory: str | Path) -> Iterable[dict[str, str]]:
    for path in _candidate_files(Path(directory).resolve()):
        payload = read_json(path)
        annotations = payload.get("annotations") or {}
        for field, section in annotations.items():
            if not isinstance(section, Mapping):
                continue
            for assertion in section.get("values") or []:
                if not isinstance(assertion, Mapping):
                    continue
                ontology = str(assertion.get("ontology") or "").strip()
                curie = str(assertion.get("ontology_id") or "").strip()
                label = str(assertion.get("label") or "").strip()
                if ontology and curie and label:
                    yield {
                        "accession": path.stem,
                        "field": str(field),
                        "ontology": ontology,
                        "curie": curie,
                        "label": label,
                    }


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    attempts: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, params=params, timeout=45)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < attempts:
                    retry_after = response.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else attempt)
                    continue
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("OLS response was not a JSON object")
            return value
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt)
    assert last_error is not None
    raise last_error


def _term_record(payload: Mapping[str, Any], ontology: str, curie: str) -> dict[str, Any]:
    terms = ((payload.get("_embedded") or {}).get("terms") or [])
    exact = [term for term in terms if str(term.get("obo_id") or "") == curie]
    if not exact:
        return {
            "ontology": ontology,
            "curie": curie,
            "status": "missing",
            "canonical_label": None,
            "synonyms": [],
            "obsolete": False,
            "replacement_ids": [],
            "iri": None,
        }
    term = exact[0]
    annotations = term.get("annotation") if isinstance(term.get("annotation"), Mapping) else {}
    replacements = annotations.get("term replaced by") or annotations.get("replaced_by") or []
    if isinstance(replacements, str):
        replacements = [replacements]
    return {
        "ontology": ontology,
        "curie": curie,
        "status": "resolved",
        "canonical_label": str(term.get("label") or "").strip() or None,
        "synonyms": sorted({str(value).strip() for value in term.get("synonyms") or [] if value}),
        "obsolete": bool(term.get("is_obsolete")),
        "replacement_ids": sorted(str(value) for value in replacements),
        "iri": str(term.get("iri") or "").strip() or None,
    }


def build_ols_candidate_index(
    *,
    annotation_directory: str | Path,
    output_directory: str | Path,
    workers: int = 4,
    force: bool = False,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch proposed CURIEs once and publish a content-addressed SQLite index."""

    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    assertions = list(iter_candidate_assertions(annotation_directory))
    requested = sorted({(item["ontology"], item["curie"]) for item in assertions})
    unsupported = sorted({ontology for ontology, _ in requested if ontology not in _SLUGS})
    if unsupported:
        raise ValueError(f"Unsupported ontology namespaces: {unsupported}")
    output = Path(output_directory).resolve()
    cache = output / "response-cache"
    cache.mkdir(parents=True, exist_ok=True)
    owned = http_client is None
    client = http_client or httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": "C-SKL-Atlas/0.5 ontology-audit"},
    )

    def fetch(item: tuple[str, str]) -> dict[str, Any]:
        ontology, curie = item
        cache_path = cache / _SLUGS[ontology] / f"{curie.replace(':', '_')}.json"
        if cache_path.is_file() and not force:
            return read_json(cache_path)
        try:
            payload = _request_json(
                client,
                f"{OLS_API}/ontologies/{_SLUGS[ontology]}/terms",
                params={"obo_id": curie},
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            payload = {}
        record = _term_record(payload, ontology, curie)
        atomic_write_json(cache_path, record)
        return record

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ols-term") as executor:
            futures = {executor.submit(fetch, item): item for item in requested}
            for future in as_completed(futures):
                ontology, curie = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:  # report and fail closed after all requests settle
                    failures.append(
                        {
                            "ontology": ontology,
                            "curie": curie,
                            "error": f"{type(exc).__name__}: {exc}"[:1000],
                        }
                    )
    finally:
        if owned:
            client.close()
    if failures:
        report = {
            "schema": ONTOLOGY_AUDIT_VERSION,
            "requested_term_count": len(requested),
            "failed": failures,
            "operator_required": True,
        }
        atomic_write_json(output / "last-build-report.json", report)
        return report

    records.sort(key=lambda item: (item["ontology"], item["curie"]))
    release_payload = {
        "schema": ONTOLOGY_AUDIT_VERSION,
        "source": OLS_API,
        "terms": records,
    }
    dependency_hash = hashlib.sha256(canonical_json(release_payload).encode()).hexdigest()
    release_id = stable_id("ontology_release", dependency_hash)
    destination = output / f"{release_id}.sqlite"
    if not destination.is_file():
        temporary = destination.with_suffix(".sqlite.tmp")
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE release_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE terms(
                  ontology TEXT NOT NULL,
                  curie TEXT NOT NULL,
                  status TEXT NOT NULL,
                  canonical_label TEXT,
                  synonyms_json TEXT NOT NULL,
                  obsolete INTEGER NOT NULL,
                  replacement_ids_json TEXT NOT NULL,
                  iri TEXT,
                  PRIMARY KEY(ontology,curie)
                );
                """
            )
            connection.executemany(
                """INSERT INTO terms(
                     ontology,curie,status,canonical_label,synonyms_json,obsolete,
                     replacement_ids_json,iri) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    (
                        item["ontology"], item["curie"], item["status"],
                        item["canonical_label"], canonical_json(item["synonyms"]),
                        int(item["obsolete"]), canonical_json(item["replacement_ids"]),
                        item["iri"],
                    )
                    for item in records
                ),
            )
            connection.executemany(
                "INSERT INTO release_meta(key,value) VALUES(?,?)",
                (
                    ("schema", ONTOLOGY_AUDIT_VERSION),
                    ("release_id", release_id),
                    ("dependency_hash", dependency_hash),
                    ("source", OLS_API),
                ),
            )
            connection.commit()
            connection.execute("VACUUM")
        finally:
            connection.close()
        os.replace(temporary, destination)
    manifest = {
        "schema": ONTOLOGY_AUDIT_VERSION,
        "release_id": release_id,
        "dependency_hash": dependency_hash,
        "source": OLS_API,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "requested_term_count": len(requested),
        "resolved_term_count": sum(item["status"] == "resolved" for item in records),
        "missing_term_count": sum(item["status"] == "missing" for item in records),
        "obsolete_term_count": sum(bool(item["obsolete"]) for item in records),
        "index_path": str(destination),
        "index_checksum": _sha256(destination),
        "operator_required": False,
    }
    atomic_write_json(output / f"{release_id}.manifest.json", manifest)
    atomic_write_json(output / "last-build-report.json", manifest)
    return manifest


def audit_annotation_candidates(
    *,
    annotation_directory: str | Path,
    ontology_index: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    assertions = list(iter_candidate_assertions(annotation_directory))
    connection = sqlite3.connect(Path(ontology_index).resolve())
    connection.row_factory = sqlite3.Row
    try:
        release = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key,value FROM release_meta")
        }
        results: list[dict[str, Any]] = []
        counts: dict[str, dict[str, int]] = {}
        for assertion in assertions:
            term = connection.execute(
                "SELECT * FROM terms WHERE ontology=? AND curie=?",
                (assertion["ontology"], assertion["curie"]),
            ).fetchone()
            status = "not_in_release"
            canonical = None
            if term:
                canonical = term["canonical_label"]
                if term["status"] != "resolved":
                    status = "missing"
                elif term["obsolete"]:
                    status = "obsolete"
                else:
                    accepted = {
                        normalized_term(term["canonical_label"] or ""),
                        *(normalized_term(value) for value in json.loads(term["synonyms_json"])),
                    }
                    status = (
                        "canonical_or_synonym"
                        if normalized_term(assertion["label"]) in accepted
                        else "label_mismatch"
                    )
            counts.setdefault(assertion["field"], {}).setdefault(status, 0)
            counts[assertion["field"]][status] += 1
            results.append({**assertion, "status": status, "canonical_label": canonical})
    finally:
        connection.close()
    blocking = [
        item for item in results if item["status"] != "canonical_or_synonym"
    ]
    report = {
        "schema": ONTOLOGY_AUDIT_VERSION,
        "ontology_release_id": release.get("release_id"),
        "ontology_dependency_hash": release.get("dependency_hash"),
        "candidate_assertion_count": len(results),
        "counts_by_field": counts,
        "blocking_count": len(blocking),
        "operator_required": bool(blocking),
        "paper_gate": "fail" if blocking else "pass",
        "results": results,
    }
    atomic_write_json(Path(output_path).resolve(), report)
    return report
