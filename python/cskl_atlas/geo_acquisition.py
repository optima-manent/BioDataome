"""Revision-aware GEO metadata acquisition through documented NCBI interfaces."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx
from cskl_pipeline.normalize import (
    DEFAULT_ARCHIVE_LIMITS,
    ArchiveSafetyError,
    ArchiveSafetyLimits,
    inspect_tar_archive,
)
from cskl_pipeline.scale.store import atomic_write_json, read_json

from .catalog import Catalog, canonical_json

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_ACCESSION_VIEWER = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_validator(state: Mapping[str, Any]) -> str:
    etag = str(state.get("etag") or "")
    if etag and not etag.startswith("W/"):
        return etag
    return str(state.get("last_modified") or "")


def _archive_inspection_within_limits(
    value: Any,
    limits: ArchiveSafetyLimits,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        return (
            0 <= int(value["member_count"]) <= limits.max_members
            and 0 <= int(value["largest_member_bytes"]) <= limits.max_member_bytes
            and 0 <= int(value["expanded_bytes"]) <= limits.max_expanded_bytes
            and int(value["cel_member_count"]) > 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _response_chunks(response: httpx.Response, chunk_size: int) -> Iterable[bytes]:
    """Yield wire bytes, with a compatibility path for pre-buffered test transports."""

    if response.is_stream_consumed:
        content = response.content
        for start in range(0, len(content), chunk_size):
            yield content[start : start + chunk_size]
        return
    yield from response.iter_raw(chunk_size)


class NcbiAcquisitionError(RuntimeError):
    """A retried NCBI request could not be completed safely."""

    def __init__(self, message: str, *, operator_required: bool = False) -> None:
        super().__init__(message)
        self.operator_required = operator_required


@dataclass(frozen=True, slots=True)
class NcbiSettings:
    email: str = ""
    api_key: str = ""
    tool: str = "BioDataome"
    timeout_seconds: float = 60.0
    max_attempts: int = 5
    allow_missing_email: bool = False
    max_soft_response_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.tool.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.tool):
            raise ValueError("NCBI tool must be a simple non-empty identifier")
        if not self.email.strip() and not self.allow_missing_email:
            raise ValueError(
                "NCBI_EMAIL is required for scheduled automation; use allow_missing_email "
                "only for a small, user-initiated bootstrap."
            )
        if self.max_attempts < 1 or self.timeout_seconds <= 0:
            raise ValueError("NCBI retry and timeout settings must be positive")
        if self.max_soft_response_bytes < 1024:
            raise ValueError("max_soft_response_bytes must be at least 1024")

    @property
    def requests_per_second(self) -> float:
        return 9.0 if self.api_key.strip() else 2.5


class NcbiEutilsClient:
    """Globally rate-limited, retrying E-utilities client."""

    def __init__(
        self,
        settings: NcbiSettings,
        *,
        http_client: httpx.Client | None = None,
        sleeper=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings
        self._client = http_client or httpx.Client(timeout=settings.timeout_seconds)
        self._owns_client = http_client is None
        self._sleeper = sleeper
        self._clock = clock
        self._lock = threading.Lock()
        self._last_request = 0.0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "NcbiEutilsClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _throttle(self) -> None:
        interval = 1.0 / self.settings.requests_per_second
        with self._lock:
            wait = interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleeper(wait)
            self._last_request = self._clock()

    def request_json(self, endpoint: str, params: Mapping[str, Any]) -> dict[str, Any]:
        request_params = {**params, "tool": self.settings.tool}
        if self.settings.email.strip():
            request_params["email"] = self.settings.email.strip()
        if self.settings.api_key.strip():
            request_params["api_key"] = self.settings.api_key.strip()
        last_error = "unknown failure"
        for attempt in range(1, self.settings.max_attempts + 1):
            self._throttle()
            try:
                response = self._client.get(
                    f"{EUTILS}/{endpoint}.fcgi",
                    params=request_params,
                    timeout=self.settings.timeout_seconds,
                )
                if response.status_code in {403, 429} or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < self.settings.max_attempts:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                        self._sleeper(delay + random.random() * 0.25)
                        continue
                    raise NcbiAcquisitionError(
                        f"NCBI {endpoint} remained unavailable after {attempt} attempts ({last_error})",
                        operator_required=response.status_code in {403, 429},
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("response is not a JSON object")
                return payload
            except NcbiAcquisitionError:
                raise
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.settings.max_attempts:
                    self._sleeper(2 ** attempt + random.random() * 0.25)
                    continue
        raise NcbiAcquisitionError(
            f"NCBI {endpoint} failed after {self.settings.max_attempts} attempts: {last_error}"
        )

    def series_summaries(self, accessions: Iterable[str], *, batch_size: int = 25) -> dict[str, dict[str, Any]]:
        requested = sorted({str(value).strip().upper() for value in accessions if str(value).strip()})
        if any(not re.fullmatch(r"GSE\d+", value) for value in requested):
            raise ValueError("Every GEO series accession must match GSE<digits>")
        results: dict[str, dict[str, Any]] = {}
        for start in range(0, len(requested), batch_size):
            batch = requested[start : start + batch_size]
            terms = " OR ".join(f"{accession}[ACCN]" for accession in batch)
            search = self.request_json(
                "esearch",
                {"db": "gds", "term": f"({terms}) AND gse[ETYP]", "retmode": "json", "retmax": len(batch)},
            )
            uids = search.get("esearchresult", {}).get("idlist", [])
            if not uids:
                continue
            summary = self.request_json(
                "esummary",
                {"db": "gds", "id": ",".join(uids), "retmode": "json", "version": "2.0"},
            )
            body = summary.get("result", {})
            for uid in body.get("uids", []):
                record = body.get(str(uid))
                if not isinstance(record, dict):
                    continue
                accession = str(record.get("accession") or "").upper()
                if accession in batch:
                    results[accession] = record
        return results

    def discover_series(
        self,
        *,
        platform: str,
        minimum_date: str,
        maximum_date: str,
        require_cel: bool = True,
        maximum_results: int = 10_000,
    ) -> dict[str, dict[str, Any]]:
        """Discover released GEO series for a bounded revision window."""

        platform = platform.strip().upper()
        if not re.fullmatch(r"GPL\d+", platform):
            raise ValueError("platform must match GPL<digits>")
        term = f"{platform}[ACCN] AND gse[ETYP]"
        if require_cel:
            term += " AND cel[suppFile]"
        search = self.request_json(
            "esearch",
            {
                "db": "gds",
                "term": term,
                "retmode": "json",
                "retmax": maximum_results,
                "datetype": "pdat",
                "mindate": minimum_date,
                "maxdate": maximum_date,
            },
        )
        result = search.get("esearchresult", {})
        count = int(result.get("count", 0))
        if count > maximum_results:
            raise NcbiAcquisitionError(
                f"Discovery returned {count} records, above the {maximum_results} safety cap",
                operator_required=True,
            )
        uids = [str(value) for value in result.get("idlist", [])]
        records: dict[str, dict[str, Any]] = {}
        for start in range(0, len(uids), 200):
            summary = self.request_json(
                "esummary",
                {
                    "db": "gds",
                    "id": ",".join(uids[start : start + 200]),
                    "retmode": "json",
                    "version": "2.0",
                },
            )
            body = summary.get("result", {})
            for uid in body.get("uids", []):
                record = body.get(str(uid))
                if isinstance(record, dict):
                    accession = str(record.get("accession") or "").upper()
                    if re.fullmatch(r"GSE\d+", accession):
                        records[accession] = record
        return records


class GeoAccessionSoftClient:
    """Retrying client for GEO's documented brief SOFT accession view.

    This is an identity-free fallback for accession metadata, not a discovery
    interface. Discovery remains on E-utilities because the accession viewer
    cannot perform bounded platform/date queries.
    """

    def __init__(
        self,
        settings: NcbiSettings,
        *,
        http_client: httpx.Client | None = None,
        sleeper=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings
        self._client = http_client or httpx.Client(timeout=settings.timeout_seconds)
        self._owns_client = http_client is None
        self._sleeper = sleeper
        self._clock = clock
        self._last_request = 0.0
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "GeoAccessionSoftClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _throttle(self) -> None:
        interval = 1.0 / min(self.settings.requests_per_second, 2.5)
        with self._lock:
            wait = interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleeper(wait)
            self._last_request = self._clock()

    def series_summary(
        self,
        accession: str,
        *,
        included_samples: set[str] | None = None,
    ) -> dict[str, Any]:
        accession = accession.strip().upper()
        if not re.fullmatch(r"GSE\d+", accession):
            raise ValueError("GEO series accession must match GSE<digits>")
        series_text = self._request_view(accession, target="self")
        record = _normalise_soft_series(accession, series_text)
        try:
            sample_text = self._request_view(accession, target="gsm")
            sample_metadata = _normalise_soft_samples(sample_text)
        except (NcbiAcquisitionError, ValueError) as exc:
            record["sample_metadata_status"] = "unavailable"
            record["sample_metadata_error"] = f"{type(exc).__name__}: {exc}"[:2_000]
            return record
        record["schema"] = "ncbi-geo-soft-series-samples-v1"
        record["source"] = "NCBI GEO accession viewer brief Series and Sample SOFT"
        available = {
            str(sample["accession"]): sample
            for sample in sample_metadata["samples"]
        }
        requested = {value.upper() for value in included_samples or set()}
        missing = sorted(requested - set(available))
        excluded = sorted(set(available) - requested) if requested else []
        selected = (
            [available[value] for value in sorted(requested & set(available))]
            if requested
            else list(available.values())
        )
        summarized = _summarize_soft_sample_records(selected)
        record["sample_metadata_status"] = "incomplete" if missing else "complete"
        record["sample_metadata_scope"] = "matrix_cohort" if requested else "series_union"
        record["sample_metadata_coverage"] = {
            "matrix_sample_count": len(requested) if requested else None,
            "matched_matrix_sample_count": len(selected),
            "missing_matrix_samples": missing,
            "excluded_series_only_sample_count": len(excluded),
        }
        record["sample_records"] = selected
        record["organisms"] = sorted(
            set(record["organisms"]) | set(summarized["organisms"])
        )
        record["sample_characteristics"] = summarized["sample_characteristics"]
        return record

    def _request_view(self, accession: str, *, target: str) -> str:
        last_error = "unknown failure"
        for attempt in range(1, self.settings.max_attempts + 1):
            self._throttle()
            try:
                with self._client.stream(
                    "GET",
                    GEO_ACCESSION_VIEWER,
                    params={
                        "acc": accession,
                        "targ": target,
                        "view": "brief",
                        "form": "text",
                    },
                    headers={"User-Agent": f"{self.settings.tool}/0.4 GEO-metadata"},
                    timeout=self.settings.timeout_seconds,
                ) as response:
                    if response.status_code in {403, 429} or response.status_code >= 500:
                        last_error = f"HTTP {response.status_code}"
                        if attempt < self.settings.max_attempts:
                            retry_after = response.headers.get("Retry-After")
                            delay = (
                                float(retry_after)
                                if retry_after and retry_after.isdigit()
                                else 2**attempt
                            )
                            self._sleeper(delay + random.random() * 0.25)
                            continue
                        raise NcbiAcquisitionError(
                            f"GEO accession view remained unavailable after {attempt} attempts "
                            f"({last_error})",
                            operator_required=response.status_code in {403, 429},
                        )
                    response.raise_for_status()
                    advertised = response.headers.get("Content-Length")
                    if advertised and advertised.isdigit() and int(advertised) > self.settings.max_soft_response_bytes:
                        raise NcbiAcquisitionError(
                            f"GEO {target} view exceeds the {self.settings.max_soft_response_bytes} byte safety cap"
                        )
                    payload = bytearray()
                    for chunk in response.iter_bytes():
                        payload.extend(chunk)
                        if len(payload) > self.settings.max_soft_response_bytes:
                            raise NcbiAcquisitionError(
                                f"GEO {target} view exceeds the {self.settings.max_soft_response_bytes} byte safety cap"
                            )
                    return bytes(payload).decode(response.encoding or "utf-8")
            except NcbiAcquisitionError:
                raise
            except (httpx.HTTPError, UnicodeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.settings.max_attempts:
                    self._sleeper(2**attempt + random.random() * 0.25)
                    continue
        raise NcbiAcquisitionError(
            f"GEO accession {target} view failed after {self.settings.max_attempts} "
            f"attempts: {last_error}"
        )


def _series_bucket(accession: str) -> str:
    digits = accession[3:]
    return f"GSE{digits[:-3]}nnn" if len(digits) > 3 else "GSEnnn"


def _normalise_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    accession = str(record.get("accession") or "").upper()
    samples = record.get("samples") if isinstance(record.get("samples"), list) else []
    sample_accessions = sorted(
        {
            str(sample.get("accession") or "").upper()
            for sample in samples
            if isinstance(sample, Mapping) and re.fullmatch(r"GSM\d+", str(sample.get("accession") or ""), re.I)
        }
    )
    platforms = sorted(
        f"GPL{value}" for value in re.findall(r"\d+", str(record.get("gpl") or ""))
    )
    bucket = _series_bucket(accession)
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{accession}"
    return {
        "schema": "ncbi-geo-esummary-v1",
        "source": "NCBI GEO DataSets ESummary",
        "accession": accession,
        "title": str(record.get("title") or ""),
        "summary": str(record.get("summary") or ""),
        "organisms": sorted({value.strip() for value in str(record.get("taxon") or "").split(";") if value.strip()}),
        "platforms": platforms,
        "assay": str(record.get("gdstype") or ""),
        "publication_date": str(record.get("pdat") or ""),
        "supplementary_file_types": sorted(
            {value.strip() for value in str(record.get("suppfile") or "").split(";") if value.strip()}
        ),
        "pubmed_ids": sorted({str(value) for value in record.get("pubmedids", [])}),
        "sample_accessions": sample_accessions,
        "sample_count": len(sample_accessions),
        "geo_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        "family_soft_url": f"{base}/soft/{accession}_family.soft.gz",
        "raw_tar_url": f"{base}/suppl/{accession}_RAW.tar",
    }


def _normalise_soft_series(accession: str, text: str) -> dict[str, Any]:
    """Parse one documented ``targ=self&view=brief&form=text`` response."""

    attributes: dict[str, list[str]] = {}
    entity_accession = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("^SERIES ="):
            entity_accession = line.split("=", 1)[1].strip().upper()
            continue
        if not line.startswith("!Series_") or "=" not in line:
            continue
        key, value = line[1:].split("=", 1)
        attributes.setdefault(key.strip(), []).append(value.strip())
    if entity_accession != accession:
        raise ValueError(
            f"GEO accession response identified {entity_accession or 'no series'} instead of "
            f"{accession}"
        )

    def values(name: str) -> list[str]:
        return [value for value in attributes.get(name, []) if value]

    platforms = sorted(set(value.upper() for value in values("Series_platform_id")))
    samples = sorted(set(value.upper() for value in values("Series_sample_id")))
    organisms = sorted(
        set(
            value
            for key, items in attributes.items()
            if key.startswith("Series_organism")
            for value in items
            if value
        )
    )
    supplementary_urls = values("Series_supplementary_file")
    supplementary_types = sorted(
        {
            "RAW" if value.upper().endswith("_RAW.TAR") else Path(value).suffix.lstrip(".").upper()
            for value in supplementary_urls
            if Path(value).suffix
        }
    )
    status = " ".join(values("Series_status"))
    publication_date = status.removeprefix("Public on ") if status else ""
    bucket = _series_bucket(accession)
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{accession}"
    raw_url = next(
        (
            value.replace("ftp://", "https://")
            for value in supplementary_urls
            if value.upper().endswith("_RAW.TAR")
        ),
        f"{base}/suppl/{accession}_RAW.tar",
    )
    return {
        "schema": "ncbi-geo-soft-brief-v1",
        "source": "NCBI GEO accession viewer brief SOFT",
        "accession": accession,
        "title": " ".join(values("Series_title")),
        "summary": "\n".join(values("Series_summary")),
        "overall_design": "\n".join(values("Series_overall_design")),
        "organisms": organisms,
        "platforms": platforms,
        "assay": "; ".join(values("Series_type")),
        "publication_date": publication_date,
        "supplementary_file_types": supplementary_types,
        "pubmed_ids": sorted(set(values("Series_pubmed_id"))),
        "sample_accessions": samples,
        "sample_count": len(samples),
        "geo_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        "family_soft_url": f"{base}/soft/{accession}_family.soft.gz",
        "raw_tar_url": raw_url,
    }


def _normalise_soft_samples(text: str) -> dict[str, Any]:
    """Retain biological labeling fields keyed by their GEO sample accession."""

    key_map = {
        "Sample_source_name_ch1": "source_name",
        "Sample_characteristics_ch1": "characteristics",
        "Sample_treatment_protocol_ch1": "treatment_protocol",
    }
    samples: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("^SAMPLE ="):
            accession = line.split("=", 1)[1].strip().upper()
            if not re.fullmatch(r"GSM\d+", accession):
                raise ValueError(f"Invalid GEO Sample accession: {accession!r}")
            current = {
                "accession": accession,
                "organisms": [],
                "source_name": [],
                "characteristics": [],
                "treatment_protocol": [],
                "platforms": [],
            }
            samples[accession] = current
            continue
        if current is None or not line.startswith("!Sample_") or "=" not in line:
            continue
        key, value = line[1:].split("=", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        if key.startswith("Sample_organism_ch"):
            current["organisms"].append(value)
        if key == "Sample_platform_id":
            current["platforms"].append(value.upper())
        mapped = key_map.get(key)
        if mapped:
            current[mapped].append(value)
    if not samples:
        raise ValueError("GEO Sample response contained no SAMPLE entities")
    records = []
    for sample in samples.values():
        records.append(
            {
                key: sorted(set(values)) if isinstance(values, list) else values
                for key, values in sample.items()
            }
        )
    return {"samples": records, **_summarize_soft_sample_records(records)}


def _summarize_soft_sample_records(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    organisms: set[str] = set()
    characteristics: dict[str, set[str]] = {
        "source_name": set(),
        "characteristics": set(),
        "treatment_protocol": set(),
    }
    for sample in samples:
        organisms.update(str(value) for value in sample.get("organisms") or [] if value)
        for key in characteristics:
            characteristics[key].update(
                str(value) for value in sample.get(key) or [] if value
            )
    return {
        "organisms": sorted(organisms),
        "sample_characteristics": {
            key: sorted(values) for key, values in characteristics.items() if values
        },
    }


def sync_geo_metadata(
    catalog: Catalog,
    *,
    accessions: Iterable[str],
    output_directory: str | Path,
    settings: NcbiSettings,
    metadata_source: str = "auto",
    force: bool = False,
    workers: int = 4,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch, normalize, checkpoint, and catalog GEO summaries."""

    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    output = Path(output_directory).resolve()
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    requested = sorted({str(value).strip().upper() for value in accessions})
    cached = [accession for accession in requested if (records_dir / f"{accession}.json").is_file()]
    pending = requested if force else sorted(set(requested) - set(cached))
    if metadata_source not in {"auto", "eutils", "geo-soft"}:
        raise ValueError("metadata_source must be auto, eutils, or geo-soft")
    failures: list[dict[str, Any]] = []
    fetched: dict[str, dict[str, Any]] = {}
    acquisition_errors: dict[str, tuple[str, bool]] = {}

    def checkpoint_record(accession: str, record: dict[str, Any]) -> None:
        content = dict(record)
        content.pop("retrieved_at", None)
        content.pop("content_sha256", None)
        record["content_sha256"] = hashlib.sha256(canonical_json(content).encode()).hexdigest()
        record["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        immutable = output / "releases" / accession / f"{record['content_sha256']}.json"
        if not immutable.is_file():
            atomic_write_json(immutable, record)
        current = records_dir / f"{accession}.json"
        if current.is_file():
            try:
                if read_json(current).get("content_sha256") == record["content_sha256"]:
                    return
            except (OSError, ValueError, TypeError):
                pass
        atomic_write_json(current, record)

    with catalog.reader() as connection:
        cohort_samples: dict[str, set[str]] = {}
        for row in connection.execute(
            """SELECT d.accession,s.gsm_accession
               FROM datasets d
               JOIN dataset_samples ds ON ds.version_id=d.current_version_id
               JOIN samples s ON s.sample_uid=ds.sample_uid
               WHERE d.current_version_id IS NOT NULL AND s.gsm_accession IS NOT NULL"""
        ):
            cohort_samples.setdefault(row["accession"], set()).add(row["gsm_accession"])

    if pending and metadata_source in {"auto", "eutils"}:
        try:
            with NcbiEutilsClient(settings, http_client=http_client) as client:
                raw = client.series_summaries(pending)
            fetched = {accession: _normalise_summary(record) for accession, record in raw.items()}
        except NcbiAcquisitionError as exc:
            acquisition_errors.update(
                {accession: (str(exc), exc.operator_required) for accession in pending}
            )
        for accession in sorted(set(pending) - set(fetched)):
            acquisition_errors.setdefault(accession, ("not_found_in_gds", False))

    soft_pending = (
        sorted(set(pending) - set(fetched))
        if metadata_source == "auto"
        else (pending if metadata_source == "geo-soft" else [])
    )
    if soft_pending:
        with GeoAccessionSoftClient(settings, http_client=http_client) as client:
            def fetch_soft(accession: str) -> tuple[str, dict[str, Any] | None, Exception | None]:
                try:
                    return accession, client.series_summary(
                        accession,
                        included_samples=cohort_samples.get(accession),
                    ), None
                except (NcbiAcquisitionError, ValueError) as exc:
                    return accession, None, exc

            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="geo-soft",
            ) as executor:
                futures = [executor.submit(fetch_soft, accession) for accession in soft_pending]
                for future in as_completed(futures):
                    accession, record, error = future.result()
                    if record is not None:
                        checkpoint_record(accession, record)
                        fetched[accession] = record
                        acquisition_errors.pop(accession, None)
                        continue
                    assert error is not None
                    operator_required = (
                        error.operator_required
                        if isinstance(error, NcbiAcquisitionError)
                        else False
                    )
                    acquisition_errors[accession] = (str(error), operator_required)
    for accession in sorted(set(pending) - set(fetched)):
        error, operator_required = acquisition_errors.get(
            accession, ("not_found_in_geo", False)
        )
        failures.append(
            {
                "accession": accession,
                "error": error,
                "operator_required": operator_required,
            }
        )

    for accession, record in fetched.items():
        if not (records_dir / f"{accession}.json").is_file():
            checkpoint_record(accession, record)

    with catalog.reader() as connection:
        versions = {
            row["accession"]: row["current_version_id"]
            for row in connection.execute(
                """SELECT accession,current_version_id FROM datasets
                   WHERE current_version_id IS NOT NULL"""
            )
        }
    cataloged = 0
    for accession in requested:
        path = records_dir / f"{accession}.json"
        version_id = versions.get(accession)
        if not path.is_file() or not version_id:
            continue
        record = read_json(path)
        immutable_path = (
            output / "releases" / accession / f"{record['content_sha256']}.json"
        )
        if not immutable_path.is_file():
            atomic_write_json(immutable_path, record)
        checksum = hashlib.sha256(immutable_path.read_bytes()).hexdigest()
        record_schema = str(record.get("schema") or "unknown-geo-schema")
        dependency_hash = hashlib.sha256(
            f"{version_id}\0{record['content_sha256']}\0{record_schema}\0geo-metadata-artifact-v2".encode()
        ).hexdigest()
        catalog.record_artifact(
            artifact_id=hashlib.sha256(
                f"{version_id}:geo_metadata:v2:{record['content_sha256']}".encode()
            ).hexdigest(),
            kind="geo_metadata",
            uri=str(immutable_path),
            checksum=checksum,
            dependency_hash=dependency_hash,
            manifest={
                "source": record.get("source", "NCBI GEO"),
                "schema": record_schema,
                "retrieved_at": record.get("retrieved_at"),
                "content_sha256": record.get("content_sha256"),
            },
            dataset_version_id=version_id,
        )
        cataloged += 1

    incomplete = []
    for accession in requested:
        path = records_dir / f"{accession}.json"
        if not path.is_file():
            continue
        record = read_json(path)
        if record.get("sample_metadata_status") == "incomplete":
            incomplete.append(
                {
                    "accession": accession,
                    "missing_matrix_samples": record.get("sample_metadata_coverage", {}).get(
                        "missing_matrix_samples", []
                    ),
                }
            )
    report = {
        "schema": "geo-sync-report-v1",
        "requested": len(requested),
        "cached": len(cached) if not force else 0,
        "fetched": len(fetched),
        "cataloged": cataloged,
        "failed": len(failures),
        "incomplete_sample_metadata": incomplete,
        "operator_required": bool(incomplete) or any(item["operator_required"] for item in failures),
        "failures": failures,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output / "last-sync-report.json", report)
    return report


def discover_geo_updates(
    catalog: Catalog,
    *,
    output_path: str | Path,
    settings: NcbiSettings,
    platform: str,
    minimum_date: str,
    maximum_date: str,
    require_cel: bool = True,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Persist a discovery checkpoint without mutating the published corpus."""

    with NcbiEutilsClient(settings, http_client=http_client) as client:
        raw = client.discover_series(
            platform=platform,
            minimum_date=minimum_date,
            maximum_date=maximum_date,
            require_cel=require_cel,
        )
    records = {accession: _normalise_summary(value) for accession, value in raw.items()}
    with catalog.reader() as connection:
        existing = {
            row["accession"]
            for row in connection.execute(
                "SELECT accession FROM datasets WHERE platform=? AND current_version_id IS NOT NULL",
                (platform.upper(),),
            )
        }
    payload = {
        "schema": "geo-discovery-v1",
        "platform": platform.upper(),
        "minimum_date": minimum_date,
        "maximum_date": maximum_date,
        "require_cel": require_cel,
        "discovered_count": len(records),
        "new_count": len(set(records) - existing),
        "existing_count": len(set(records) & existing),
        "new_accessions": sorted(set(records) - existing),
        "records": records,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(Path(output_path), payload)
    return payload


def download_geo_raw_tar(
    *,
    accession: str,
    raw_tar_url: str,
    output_directory: str | Path,
    timeout_seconds: float = 120.0,
    max_attempts: int = 5,
    reserve_bytes: int = 5 * 1024**3,
    max_download_bytes: int = 200 * 1024**3,
    source_revision: str = "",
    expected_sha256: str = "",
    force: bool = False,
    archive_limits: ArchiveSafetyLimits = DEFAULT_ARCHIVE_LIMITS,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Resume, validate, checksum, and atomically publish one GEO RAW tar."""

    accession = accession.strip().upper()
    if not re.fullmatch(r"GSE\d+", accession):
        raise ValueError("accession must match GSE<digits>")
    if not raw_tar_url.startswith("https://"):
        raise ValueError("raw_tar_url must use HTTPS")
    if max_attempts < 1 or timeout_seconds <= 0:
        raise ValueError("download retries and timeout must be positive")
    if reserve_bytes < 0 or max_download_bytes < 1:
        raise ValueError("download size settings are invalid")
    expected_sha256 = expected_sha256.strip().lower()
    if expected_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    root = Path(output_directory).resolve() / accession
    root_is_junction = getattr(root, "is_junction", lambda: False)
    if root.is_symlink() or root_is_junction():
        raise NcbiAcquisitionError(
            "GEO RAW output directory cannot be a link or junction",
            operator_required=True,
        )
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"{accession}_RAW.tar"
    partial = root / f"{accession}_RAW.tar.part"
    partial_state_path = root / "raw-partial.json"
    manifest_path = root / "raw-manifest.json"
    for managed_path in (final, partial, partial_state_path, manifest_path):
        managed_is_junction = getattr(managed_path, "is_junction", lambda: False)
        if managed_path.is_symlink() or managed_is_junction():
            raise NcbiAcquisitionError(
                f"managed RAW path cannot be a link or junction: {managed_path.name}",
                operator_required=True,
            )
    if final.is_file() and manifest_path.is_file() and not force:
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, TypeError):
            manifest = {}
        digest = _file_sha256(final)
        revision_matches = not source_revision or manifest.get("source_revision") == source_revision
        if (
            manifest.get("accession") == accession
            and manifest.get("source_url") == raw_tar_url
            and manifest.get("size") == final.stat().st_size
            and digest == manifest.get("sha256")
            and (not expected_sha256 or digest == expected_sha256)
            and revision_matches
        ):
            archived_inspection = manifest.get("archive")
            inspection_within_limits = _archive_inspection_within_limits(
                archived_inspection, archive_limits
            )
            if not inspection_within_limits:
                try:
                    inspection = inspect_tar_archive(
                        final,
                        limits=archive_limits,
                        verify_payloads=True,
                    )
                except ArchiveSafetyError as exc:
                    raise NcbiAcquisitionError(
                        f"Cached RAW archive failed safety validation: {exc}",
                        operator_required=True,
                    ) from exc
                manifest = {
                    **manifest,
                    "schema": "geo-raw-artifact-v2",
                    "archive": inspection,
                    "archive_limits": {
                        "max_members": archive_limits.max_members,
                        "max_member_bytes": archive_limits.max_member_bytes,
                        "max_expanded_bytes": archive_limits.max_expanded_bytes,
                    },
                }
                atomic_write_json(manifest_path, manifest)
            return {**manifest, "status": "cached", "path": str(final)}

    partial_state: dict[str, Any] = {}
    if partial.is_file():
        try:
            partial_state = read_json(partial_state_path)
        except (OSError, ValueError, TypeError):
            partial_state = {}
        if (
            partial_state.get("source_url") != raw_tar_url
            or partial_state.get("source_revision", "") != source_revision
            or not _resume_validator(partial_state)
        ):
            partial.unlink(missing_ok=True)
            partial_state_path.unlink(missing_ok=True)
            partial_state = {}

    client = http_client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
    owns_client = http_client is None
    try:
        last_error = "unknown failure"
        remote_etag = ""
        remote_last_modified = ""
        for attempt in range(1, max_attempts + 1):
            if partial.is_file() and not _resume_validator(partial_state):
                partial.unlink(missing_ok=True)
                partial_state_path.unlink(missing_ok=True)
                partial_state = {}
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            validator = _resume_validator(partial_state)
            if offset and validator:
                headers["If-Range"] = str(validator)
            try:
                with client.stream(
                    "GET", raw_tar_url, headers=headers, timeout=timeout_seconds
                ) as response:
                    if response.status_code == 416 and offset:
                        partial.unlink(missing_ok=True)
                        partial_state_path.unlink(missing_ok=True)
                        partial_state = {}
                        last_error = "server rejected the saved resume offset"
                        if attempt == max_attempts:
                            raise NcbiAcquisitionError(
                                f"GEO RAW resume failed: {last_error}",
                                operator_required=True,
                            )
                        continue
                    if response.status_code in {403, 429} or response.status_code >= 500:
                        last_error = f"HTTP {response.status_code}"
                        if attempt == max_attempts:
                            raise NcbiAcquisitionError(
                                f"GEO RAW download exhausted retries ({last_error})",
                                operator_required=response.status_code in {403, 429},
                            )
                        time.sleep(2**attempt + random.random() * 0.25)
                        continue
                    if 400 <= response.status_code < 500:
                        raise NcbiAcquisitionError(
                            f"GEO RAW download returned HTTP {response.status_code}",
                            operator_required=True,
                        )
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    range_total: int | None = None
                    if append:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                        range_start = int(match.group(1)) if match else -1
                        range_end = int(match.group(2)) if match else -1
                        range_total = (
                            int(match.group(3))
                            if match and match.group(3) != "*"
                            else None
                        )
                        prior_etag = str(partial_state.get("etag") or "")
                        if prior_etag.startswith("W/"):
                            prior_etag = ""
                        prior_modified = str(partial_state.get("last_modified") or "")
                        returned_etag = response.headers.get("ETag", "")
                        returned_modified = response.headers.get("Last-Modified", "")
                        validator_matches = (
                            (prior_etag and returned_etag == prior_etag)
                            or (prior_modified and returned_modified == prior_modified)
                        )
                        if (
                            not match
                            or range_start != offset
                            or range_end < range_start
                            or range_total is None
                            or range_end >= range_total
                            or not validator_matches
                        ):
                            partial.unlink(missing_ok=True)
                            partial_state_path.unlink(missing_ok=True)
                            partial_state = {}
                            last_error = "server returned an invalid Content-Range"
                            if attempt == max_attempts:
                                raise NcbiAcquisitionError(
                                    f"GEO RAW resume failed: {last_error}",
                                    operator_required=True,
                                )
                            continue
                    elif offset:
                        offset = 0
                        partial_state = {}
                    length_header = response.headers.get("Content-Length")
                    try:
                        remaining = int(length_header) if length_header is not None else None
                    except ValueError as exc:
                        raise NcbiAcquisitionError(
                            "GEO RAW response has an invalid Content-Length",
                            operator_required=True,
                        ) from exc
                    if remaining is not None and remaining < 0:
                        raise NcbiAcquisitionError(
                            "GEO RAW response has a negative Content-Length",
                            operator_required=True,
                        )
                    if append and remaining is not None and range_end - offset + 1 != remaining:
                        raise NcbiAcquisitionError(
                            "GEO RAW range length disagrees with Content-Length",
                            operator_required=True,
                        )
                    expected_total = range_total
                    if expected_total is None and remaining is not None:
                        expected_total = offset + remaining
                    if expected_total is not None and expected_total > max_download_bytes:
                        raise NcbiAcquisitionError(
                            f"GEO RAW archive exceeds the {max_download_bytes:,}-byte cap",
                            operator_required=True,
                        )
                    free_bytes = shutil.disk_usage(root).free
                    required = (remaining or 0) + reserve_bytes
                    if free_bytes < required:
                        raise NcbiAcquisitionError(
                            f"Insufficient disk space for {accession}; need {required} free bytes",
                            operator_required=True,
                        )
                    writable_bytes = free_bytes - reserve_bytes
                    remote_etag = response.headers.get("ETag", "")
                    if remote_etag.startswith("W/"):
                        remote_etag = ""
                    remote_last_modified = response.headers.get("Last-Modified", "")
                    partial_state = {
                        "schema": "geo-raw-partial-v1",
                        "accession": accession,
                        "source_url": raw_tar_url,
                        "source_revision": source_revision,
                        "etag": remote_etag,
                        "last_modified": remote_last_modified,
                        "expected_total_size": expected_total,
                    }
                    atomic_write_json(partial_state_path, partial_state)
                    received = 0
                    with partial.open("ab" if append else "wb") as handle:
                        for chunk in _response_chunks(response, 8 * 1024 * 1024):
                            received += len(chunk)
                            if offset + received > max_download_bytes:
                                raise NcbiAcquisitionError(
                                    f"GEO RAW archive exceeded the {max_download_bytes:,}-byte cap",
                                    operator_required=True,
                                )
                            if received > writable_bytes:
                                raise NcbiAcquisitionError(
                                    f"GEO RAW download would consume the {reserve_bytes:,}-byte disk reserve",
                                    operator_required=True,
                                )
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if remaining is not None and received != remaining:
                        raise NcbiAcquisitionError(
                            "GEO RAW response ended before its declared Content-Length",
                            operator_required=True,
                        )
                    if expected_total is not None and partial.stat().st_size != expected_total:
                        raise NcbiAcquisitionError(
                            "GEO RAW response did not complete the declared remote object",
                            operator_required=True,
                        )
                break
            except NcbiAcquisitionError:
                raise
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == max_attempts:
                    raise NcbiAcquisitionError(
                        f"GEO RAW download failed after {attempt} attempts: {last_error}"
                    ) from exc
                time.sleep(2**attempt + random.random() * 0.25)
        else:  # pragma: no cover
            raise NcbiAcquisitionError(last_error)

        try:
            inspection = inspect_tar_archive(
                partial,
                limits=archive_limits,
                verify_payloads=True,
            )
        except ArchiveSafetyError as exc:
            raise NcbiAcquisitionError(
                f"Downloaded RAW archive is invalid or unsafe: {exc}",
                operator_required=True,
            ) from exc
        if inspection["cel_member_count"] == 0:
            raise NcbiAcquisitionError(
                "Downloaded RAW archive contains no CEL files", operator_required=True
            )
        digest = _file_sha256(partial)
        if expected_sha256 and digest != expected_sha256:
            raise NcbiAcquisitionError(
                "Downloaded RAW archive checksum does not match the expected digest",
                operator_required=True,
            )
        os.replace(partial, final)
        partial_state_path.unlink(missing_ok=True)
        manifest = {
            "schema": "geo-raw-artifact-v2",
            "status": "verified",
            "accession": accession,
            "source_url": raw_tar_url,
            "source_revision": source_revision,
            "remote_etag": remote_etag,
            "remote_last_modified": remote_last_modified,
            "size": final.stat().st_size,
            "sha256": digest,
            "cel_member_count": inspection["cel_member_count"],
            "archive": inspection,
            "archive_limits": {
                "max_members": archive_limits.max_members,
                "max_member_bytes": archive_limits.max_member_bytes,
                "max_expanded_bytes": archive_limits.max_expanded_bytes,
            },
            "path": str(final),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(manifest_path, manifest)
        return manifest
    finally:
        if owns_client:
            client.close()
