from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .io import safe_name_from_file


@dataclass(frozen=True, slots=True)
class ArchiveSafetyLimits:
    """Resource limits applied before any archive member reaches the filesystem."""

    max_members: int = 25_000
    max_member_bytes: int = 16 * 1024**3
    max_expanded_bytes: int = 250 * 1024**3
    minimum_free_bytes_after_extract: int = 2 * 1024**3

    def __post_init__(self) -> None:
        if min(
            self.max_members,
            self.max_member_bytes,
            self.max_expanded_bytes,
        ) < 1:
            raise ValueError("archive safety limits must be positive")
        if self.minimum_free_bytes_after_extract < 0:
            raise ValueError("minimum_free_bytes_after_extract cannot be negative")


DEFAULT_ARCHIVE_LIMITS = ArchiveSafetyLimits()


class ArchiveSafetyError(RuntimeError):
    """An archive needs operator inspection instead of automatic extraction."""

    operator_required = True


_WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{value}" for value in range(1, 10)),
    *(f"LPT{value}" for value in range(1, 10)),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_path(name: str) -> PurePosixPath:
    """Return a portable relative member path or reject ambiguous filesystem names."""

    if not name or "\x00" in name or "\\" in name:
        raise ArchiveSafetyError(f"unsafe archive member path: {name!r}")
    if re.match(r"^[A-Za-z]:", name):
        raise ArchiveSafetyError(f"drive-qualified archive member path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ArchiveSafetyError(f"non-relative archive member path: {name!r}")
    for part in relative.parts:
        if ":" in part or part.endswith((" ", ".")):
            raise ArchiveSafetyError(f"non-portable archive member path: {name!r}")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_DEVICE_NAMES:
            raise ArchiveSafetyError(f"reserved archive member path: {name!r}")
    return relative


def inspect_tar_archive(
    tar_path: Path,
    *,
    limits: ArchiveSafetyLimits = DEFAULT_ARCHIVE_LIMITS,
    verify_payloads: bool = False,
) -> dict[str, int]:
    """Validate paths, types and declared expansion before extraction.

    Only regular files and directories are accepted. Links, devices and FIFOs are
    unnecessary for GEO CEL bundles and are rejected rather than interpreted.
    """

    member_count = 0
    file_count = 0
    cel_member_count = 0
    expanded_bytes = 0
    largest_member_bytes = 0
    seen: set[str] = set()
    try:
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                member_count += 1
                if member_count > limits.max_members:
                    raise ArchiveSafetyError(
                        f"archive has more than {limits.max_members:,} members"
                    )
                relative = _safe_archive_path(member.name)
                identity = relative.as_posix().casefold()
                if identity in seen:
                    raise ArchiveSafetyError(
                        f"archive contains a duplicate/case-colliding path: {member.name!r}"
                    )
                seen.add(identity)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ArchiveSafetyError(
                        f"archive member is not a regular file or directory: {member.name!r}"
                    )
                if member.size < 0 or member.size > limits.max_member_bytes:
                    raise ArchiveSafetyError(
                        f"archive member size is outside the safety limit: {member.name!r}"
                    )
                expanded_bytes += member.size
                largest_member_bytes = max(largest_member_bytes, member.size)
                if expanded_bytes > limits.max_expanded_bytes:
                    raise ArchiveSafetyError(
                        "archive declared expansion exceeds "
                        f"{limits.max_expanded_bytes:,} bytes"
                    )
                file_count += 1
                lower_name = relative.name.casefold()
                if lower_name.endswith((".cel", ".cel.gz")):
                    cel_member_count += 1
                if verify_payloads:
                    source = archive.extractfile(member)
                    if source is None:
                        raise ArchiveSafetyError(
                            f"archive member payload is unreadable: {member.name!r}"
                        )
                    consumed = 0
                    while chunk := source.read(8 * 1024 * 1024):
                        consumed += len(chunk)
                    if consumed != member.size:
                        raise ArchiveSafetyError(
                            f"archive member payload is truncated: {member.name!r}"
                        )
    except ArchiveSafetyError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise ArchiveSafetyError(
            f"archive cannot be read safely: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "member_count": member_count,
        "file_count": file_count,
        "cel_member_count": cel_member_count,
        "expanded_bytes": expanded_bytes,
        "largest_member_bytes": largest_member_bytes,
    }


R_NORMALIZE_SCRIPT = r"""
suppressPackageStartupMessages({
  library(SCAN.UPC)
  library(Biobase)
})

message("R runtime: ", R.version.string)
message("SCAN.UPC version: ", as.character(utils::packageVersion("SCAN.UPC")))
message("Biobase version: ", as.character(utils::packageVersion("Biobase")))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript scan_normalize_affy.R <cel_root_dir> <out_tsv_gz>")
}

raw_root <- normalizePath(args[1], winslash="/", mustWork=TRUE)
out_path <- args[2]

cel_files <- list.files(
  raw_root,
  pattern="\\.cel(\\.gz)?$",
  full.names=TRUE,
  recursive=TRUE,
  ignore.case=TRUE
)

if (length(cel_files) == 0) {
  stop(paste("No CEL/CEL.gz files found under:", raw_root))
}

cel_stems <- sub("\\.cel(\\.gz)?$", "", basename(cel_files), ignore.case=TRUE)
normalized_stems <- tolower(cel_stems)
if (anyDuplicated(normalized_stems)) {
  duplicates <- unique(basename(cel_files)[duplicated(normalized_stems)])
  stop(paste("Duplicate CEL sample names cannot be flattened safely:", paste(duplicates, collapse=", ")))
}
cel_is_gzip <- grepl("\\.gz$", basename(cel_files), ignore.case=TRUE)
staged_names <- paste0(cel_stems, ".CEL", ifelse(cel_is_gzip, ".gz", ""))

flat_dir <- file.path(tempdir(), paste0("scan_upc_cels_", as.integer(Sys.time())))
dir.create(flat_dir, recursive=TRUE, showWarnings=FALSE)
flat_dir <- normalizePath(flat_dir, winslash="/", mustWork=TRUE)

ok <- file.copy(cel_files, file.path(flat_dir, staged_names), overwrite=FALSE)
if (!all(ok)) {
  stop(paste("Failed to stage CEL files:", paste(basename(cel_files[!ok]), collapse=", ")))
}

pattern <- paste0(flat_dir, "/*.CEL*")

message("SCAN.UPC::SCAN('", pattern, "') with ", length(cel_files), " files")
eset <- SCAN.UPC::SCAN(pattern, outFilePath=NA, verbose=TRUE)
mat <- Biobase::exprs(eset)
if (ncol(mat) != length(cel_files)) {
  stop(paste("SCAN.UPC returned", ncol(mat), "samples for", length(cel_files), "CEL inputs"))
}

con <- gzfile(out_path, "wt")
write.table(mat, file=con, sep="\t", quote=FALSE, col.names=NA)
close(con)

cat("Wrote:", out_path, "\n")
"""


def run(cmd: list[str], *, log_path: Path | None = None, **kwargs) -> None:
    print("+", " ".join(map(str, cmd)))
    if log_path is None:
        subprocess.check_call(cmd, **kwargs)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **kwargs,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def scan_log_diagnostics(log_path: Path) -> dict[str, Any]:
    """Return bounded, machine-readable diagnostics from a SCAN.UPC run log.

    SCAN can emit convergence warnings while still returning a complete expression
    matrix. Those warnings are useful provenance, but are not by themselves grounds
    for quarantining a dataset whose numerical QC succeeds.
    """
    result: dict[str, Any] = {
        "engine": "SCAN.UPC",
        "log_path": log_path.name,
        "log_present": log_path.is_file(),
        "status": "not_recorded",
        "convergence_warning_count": 0,
        "warning_lines": [],
    }
    if not log_path.is_file():
        return result

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    convergence_lines = [
        line.strip()
        for line in lines
        if "converg" in line.casefold()
        and any(
            marker in line.casefold()
            for marker in ("limit", "maximum", "max iter", "failed", "not converg")
        )
    ]
    generic_warnings = [
        line.strip()
        for line in lines
        if "warning" in line.casefold() and line.strip() not in convergence_lines
    ]
    warning_lines = convergence_lines + generic_warnings
    result.update(
        {
            "status": "warning" if warning_lines else "ok",
            "convergence_warning_count": len(convergence_lines),
            "warning_lines": warning_lines[:20],
            "warning_lines_truncated": len(warning_lines) > 20,
        }
    )
    return result


def extract_tar(
    tar_path: Path,
    dest_dir: Path,
    *,
    limits: ArchiveSafetyLimits = DEFAULT_ARCHIVE_LIMITS,
) -> dict[str, int]:
    """Safely extract an archive through a fresh sibling staging directory.

    This intentionally does not rely on ``tarfile.extractall(filter=...)`` so the
    same protections apply on every supported Python version. Existing extracted
    data is replaced only after the complete staged extraction succeeds.
    """

    tar_path = Path(tar_path).resolve(strict=True)
    requested_destination = Path(dest_dir).absolute()
    dest_dir = requested_destination.parent.resolve() / requested_destination.name
    is_junction = getattr(dest_dir, "is_junction", lambda: False)
    if dest_dir.is_symlink() or is_junction():
        raise ArchiveSafetyError("extraction destination cannot be a link or junction")
    if dest_dir.exists() and not dest_dir.is_dir():
        raise ArchiveSafetyError("extraction destination exists but is not a directory")
    inspection = inspect_tar_archive(tar_path, limits=limits)
    required = inspection["expanded_bytes"] + limits.minimum_free_bytes_after_extract
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dest_dir.parent).free < required:
        raise ArchiveSafetyError(
            f"insufficient disk for safe extraction; need {required:,} free bytes"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{dest_dir.name}.extract-", dir=dest_dir.parent)
    ).resolve()
    previous: Path | None = None
    try:
        extracted_members = 0
        extracted_files = 0
        extracted_bytes = 0
        extracted_seen: set[str] = set()
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                extracted_members += 1
                if extracted_members > limits.max_members:
                    raise ArchiveSafetyError(
                        f"archive has more than {limits.max_members:,} members"
                    )
                relative = _safe_archive_path(member.name)
                identity = relative.as_posix().casefold()
                if identity in extracted_seen:
                    raise ArchiveSafetyError(
                        f"archive contains a duplicate/case-colliding path: {member.name!r}"
                    )
                extracted_seen.add(identity)
                target = (staging / Path(*relative.parts)).resolve()
                if not target.is_relative_to(staging):  # defence in depth
                    raise ArchiveSafetyError(
                        f"archive member escapes extraction root: {member.name!r}"
                    )
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ArchiveSafetyError(
                        f"archive member is not a regular file: {member.name!r}"
                    )
                if member.size < 0 or member.size > limits.max_member_bytes:
                    raise ArchiveSafetyError(
                        f"archive member size is outside the safety limit: {member.name!r}"
                    )
                extracted_files += 1
                extracted_bytes += member.size
                if extracted_bytes > limits.max_expanded_bytes:
                    raise ArchiveSafetyError(
                        "archive declared expansion exceeds "
                        f"{limits.max_expanded_bytes:,} bytes"
                    )
                if shutil.disk_usage(staging).free < (
                    member.size + limits.minimum_free_bytes_after_extract
                ):
                    raise ArchiveSafetyError(
                        f"insufficient disk while extracting {member.name!r}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ArchiveSafetyError(
                        f"archive member payload is unreadable: {member.name!r}"
                    )
                copied = 0
                with target.open("xb") as output:
                    while chunk := source.read(8 * 1024 * 1024):
                        copied += len(chunk)
                        if copied > member.size:
                            raise ArchiveSafetyError(
                                f"archive member exceeds its declared size: {member.name!r}"
                            )
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if copied != member.size:
                    raise ArchiveSafetyError(
                        f"archive member payload is truncated: {member.name!r}"
                    )

        if (
            extracted_members != inspection["member_count"]
            or extracted_files != inspection["file_count"]
            or extracted_bytes != inspection["expanded_bytes"]
        ):
            raise ArchiveSafetyError("archive changed during safety validation")

        if dest_dir.exists():
            previous = dest_dir.parent / f".{dest_dir.name}.previous-{uuid.uuid4().hex}"
            os.replace(dest_dir, previous)
        os.replace(staging, dest_dir)
        if previous is not None:
            shutil.rmtree(previous)
        return inspection
    except Exception:
        if previous is not None and previous.exists() and not dest_dir.exists():
            os.replace(previous, dest_dir)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_r_script(path: Path) -> None:
    path.write_text(R_NORMALIZE_SCRIPT, encoding="utf-8")


def _normalization_manifest_path(work_root: Path, tar_path: Path) -> Path:
    return work_root / _validated_dataset_name(tar_path) / "normalization-manifest.json"


def _validated_dataset_name(tar_path: Path) -> str:
    name = safe_name_from_file(tar_path)
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,127}", name) or name in {".", ".."}:
        raise ValueError(f"unsafe dataset name derived from archive: {name!r}")
    return name


def read_normalization_manifest(work_root: Path, tar_path: Path) -> dict[str, Any] | None:
    path = _normalization_manifest_path(work_root, tar_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def normalization_cache_matches(
    tar_path: Path,
    work_root: Path,
    *,
    verify_hashes: bool = True,
) -> bool:
    """Whether cached normalization is provenance-bound to this RAW archive."""

    tar_path = Path(tar_path).resolve()
    manifest = read_normalization_manifest(work_root, tar_path)
    if not manifest or manifest.get("schema") != "cskl-scan-normalization-v2":
        return False
    output = Path(work_root) / _validated_dataset_name(tar_path) / "expr.tsv.gz"
    if not tar_path.is_file() or not output.is_file():
        return False
    source_stat = tar_path.stat()
    output_stat = output.stat()
    if manifest.get("source_size") != source_stat.st_size:
        return False
    if manifest.get("normalization_script_sha256") != hashlib.sha256(
        R_NORMALIZE_SCRIPT.encode("utf-8")
    ).hexdigest():
        return False
    if manifest.get("normalized_size") != output_stat.st_size:
        return False
    if not verify_hashes:
        return (
            manifest.get("source_mtime_ns") == source_stat.st_mtime_ns
            and manifest.get("normalized_mtime_ns") == output_stat.st_mtime_ns
        )
    return (
        manifest.get("source_sha256") == _sha256_file(tar_path)
        and manifest.get("normalized_sha256") == _sha256_file(output)
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_affy_scan(
    tar_path: Path,
    work_root: Path,
    rscript: str,
    *,
    archive_limits: ArchiveSafetyLimits = DEFAULT_ARCHIVE_LIMITS,
) -> Path:
    tar_path = Path(tar_path).resolve(strict=True)
    work_root = Path(work_root).resolve()
    gse = _validated_dataset_name(tar_path)
    gse_dir = work_root / gse
    raw_dir = gse_dir / "raw_extracted"
    out_expr = gse_dir / "expr.tsv.gz"
    gse_dir.mkdir(parents=True, exist_ok=True)

    if normalization_cache_matches(tar_path, work_root):
        print(f"[{gse}] Using existing normalized matrix: {out_expr}")
        return out_expr

    source_stat_before = tar_path.stat()
    source_sha256_before = _sha256_file(tar_path)
    print(f"[{gse}] Extracting {tar_path} -> {raw_dir}")
    inspection = extract_tar(tar_path, raw_dir, limits=archive_limits)

    rfile = gse_dir / "scan_normalize_affy.R"
    write_r_script(rfile)

    print(f"[{gse}] Normalizing with SCAN.UPC -> {out_expr}")
    pending = gse_dir / f".expr.{uuid.uuid4().hex}.tsv.gz"
    try:
        run(
            [rscript, str(rfile), str(raw_dir), str(pending)],
            log_path=gse_dir / "scan_normalize_affy.log",
        )
        if not pending.is_file() or pending.stat().st_size == 0:
            raise RuntimeError("SCAN.UPC completed without a non-empty expression matrix")
        source_stat = tar_path.stat()
        source_sha256_after = _sha256_file(tar_path)
        if (
            source_stat.st_size != source_stat_before.st_size
            or source_stat.st_mtime_ns != source_stat_before.st_mtime_ns
            or source_sha256_after != source_sha256_before
        ):
            raise ArchiveSafetyError(
                "RAW archive changed while normalization was in progress; retry a stable revision"
            )
        normalized_sha256 = _sha256_file(pending)
        os.replace(pending, out_expr)
        output_stat = out_expr.stat()
        manifest = {
            "schema": "cskl-scan-normalization-v2",
            "engine": "SCAN.UPC",
            "source_path": str(tar_path),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_sha256": source_sha256_after,
            "normalized_path": str(out_expr),
            "normalized_size": output_stat.st_size,
            "normalized_mtime_ns": output_stat.st_mtime_ns,
            "normalized_sha256": normalized_sha256,
            "normalization_script_sha256": hashlib.sha256(
                R_NORMALIZE_SCRIPT.encode("utf-8")
            ).hexdigest(),
            "archive": inspection,
            "archive_limits": asdict(archive_limits),
        }
        _atomic_write_json(_normalization_manifest_path(work_root, tar_path), manifest)
    finally:
        pending.unlink(missing_ok=True)
    return out_expr
