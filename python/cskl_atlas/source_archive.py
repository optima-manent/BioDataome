"""Resumable, checksum-verified access to preserved ZIP source matrices."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from cskl_pipeline.scale.store import atomic_write_json, read_json

_ACCESSION = re.compile(r"^(GSE\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ArchivedSource:
    accession: str
    member: str
    checksum: str
    size: int
    crc32: str
    uri: str
    sample_ids: tuple[str | None, ...]


def _central_directory_hash(archive: Path, infos: Iterable[zipfile.ZipInfo]) -> str:
    """Fingerprint the archive index without rereading every large member."""

    digest = hashlib.sha256()
    stat = archive.stat()
    digest.update(f"{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    for info in infos:
        digest.update(info.filename.encode("utf-8"))
        digest.update(
            f"\0{info.file_size}\0{info.compress_size}\0{info.CRC}\0{info.compress_type}\0".encode()
        )
    return digest.hexdigest()


def _accession_for_member(member: str) -> str | None:
    stem = Path(member).stem.upper()
    return stem if _ACCESSION.fullmatch(stem) else None


def index_zip_sources(
    archive_path: str | Path,
    *,
    cache_directory: str | Path,
    accessions: Iterable[str] | None = None,
) -> tuple[str, dict[str, ArchivedSource]]:
    """Hash requested source members with an atomic per-member resume cache.

    The ZIP central-directory fingerprint invalidates stale caches. A crash can
    lose at most the member currently being hashed; completed multi-gigabyte
    members are never read again on replay.
    """

    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    cache_root = Path(cache_directory).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zipped:
        infos = [info for info in zipped.infolist() if not info.is_dir()]
        central_hash = _central_directory_hash(archive, infos)
        by_accession: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            accession = _accession_for_member(info.filename)
            if accession is None:
                continue
            if accession in by_accession:
                raise ValueError(f"Archive contains duplicate source members for {accession}")
            by_accession[accession] = info

        requested = sorted(
            by_accession if accessions is None else {str(value).strip().upper() for value in accessions}
        )
        missing = sorted(set(requested) - set(by_accession))
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"Source archive is missing {len(missing)} requested datasets: {preview}")

        cache_path = cache_root / f"zip-members-{central_hash}.json"
        cache: dict[str, Any]
        if cache_path.is_file():
            loaded = read_json(cache_path)
            cache = loaded if loaded.get("central_directory_hash") == central_hash else {}
        else:
            cache = {}
        cache.setdefault("schema", "cskl-source-archive-v1")
        cache.setdefault("central_directory_hash", central_hash)
        cache.setdefault("archive", str(archive))
        cache.setdefault("members", {})

        results: dict[str, ArchivedSource] = {}
        archive_uri = archive.as_uri()
        for position, accession in enumerate(requested, start=1):
            info = by_accession[accession]
            cached = cache["members"].get(accession)
            expected_crc = f"{info.CRC:08x}"
            if (
                isinstance(cached, dict)
                and cached.get("member") == info.filename
                and cached.get("size") == info.file_size
                and cached.get("crc32") == expected_crc
                and isinstance(cached.get("sha256"), str)
                and len(cached["sha256"]) == 64
            ):
                checksum = cached["sha256"]
            else:
                digest = hashlib.sha256()
                with zipped.open(info, "r") as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
                checksum = digest.hexdigest()
                cache["members"][accession] = {
                    "member": info.filename,
                    "size": info.file_size,
                    "crc32": expected_crc,
                    "sha256": checksum,
                }
                cache["completed"] = len(cache["members"])
                cache["requested"] = len(requested)
                atomic_write_json(cache_path, cache)
                print(
                    f"[source-archive] verified {accession} "
                    f"({position}/{len(requested)}, {info.file_size / (1024 ** 2):.1f} MiB)"
                )
            cached = cache["members"][accession]
            sample_ids = cached.get("sample_ids")
            if not isinstance(sample_ids, list) or cached.get("sample_id_parser") != "gsm-prefix-v1":
                with zipped.open(info, "r") as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                    header = next(csv.reader(text))
                sample_ids = []
                for value in header[1:]:
                    match = re.match(r"^(GSM\d+)(?:\b|_)", value.strip(), re.I)
                    sample_ids.append(match.group(1).upper() if match else None)
                cached["sample_ids"] = sample_ids
                cached["sample_id_parser"] = "gsm-prefix-v1"
                atomic_write_json(cache_path, cache)
            results[accession] = ArchivedSource(
                accession=accession,
                member=info.filename,
                checksum=checksum,
                size=info.file_size,
                crc32=expected_crc,
                uri=f"zip+{archive_uri}!/{quote(info.filename)}",
                sample_ids=tuple(sample_ids),
            )
    return central_hash, results
