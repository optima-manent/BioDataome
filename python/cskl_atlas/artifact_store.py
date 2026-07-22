"""Immutable, content-addressed artifact bundles with checksum validation.

An artifact is published only after every payload and its manifest have been
written under a private staging directory. Publication is one same-filesystem
directory rename, so readers observe either no artifact or a complete bundle.
Existing content addresses are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .fingerprints import (
    DEFAULT_CHUNK_SIZE,
    canonical_json_bytes,
    canonical_json_value,
    sha256_file,
    validate_sha256,
)
from .models import (
    AnalysisDependencies,
    ArtifactFile,
    ArtifactManifest,
    ContractError,
    normalize_artifact_relative_path,
    validate_artifact_type,
)

MANIFEST_FILENAME = "manifest.json"
PAYLOAD_DIRECTORY = "files"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class ArtifactStoreError(RuntimeError):
    """Base error for artifact-store operations."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when a requested content address is absent."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when a bundle fails manifest or payload validation."""


class BundleStateError(ArtifactStoreError):
    """Raised when a bundle writer is used after commit or abort."""


@dataclass(frozen=True)
class ValidationResult:
    artifact_id: str
    valid: bool
    errors: tuple[str, ...]
    manifest: ArtifactManifest | None = None

    def require_valid(self) -> ArtifactManifest:
        if not self.valid or self.manifest is None:
            detail = "; ".join(self.errors) if self.errors else "unknown integrity failure"
            raise ArtifactIntegrityError(f"artifact {self.artifact_id} is invalid: {detail}")
        return self.manifest


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    path: Path
    manifest: ArtifactManifest


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate key in manifest JSON: {key!r}")
        value[key] = item
    return value


def _read_manifest(path: Path) -> ArtifactManifest:
    if path.is_symlink():
        raise ArtifactIntegrityError("manifest must not be a symbolic link")
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise ArtifactNotFoundError(f"missing manifest: {path}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise ArtifactIntegrityError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=_json_without_duplicate_keys)
        return ArtifactManifest.from_dict(raw)
    except (ArtifactStoreError, ContractError):
        raise
    except Exception as exc:
        raise ArtifactIntegrityError(f"cannot parse manifest: {type(exc).__name__}: {exc}") from exc


def _write_all_and_fsync(stream: BinaryIO, data: bytes) -> None:
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            _write_all_and_fsync(stream, data)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync (opening directories is unsupported on Windows)."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_link_like(path: Path) -> bool:
    """True for symlinks and Windows directory junctions/reparse aliases."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


class ArtifactStore:
    """A local immutable object store rooted at ``root``.

    Layout::

        root/
          objects/sha256/ab/<artifact_id>/manifest.json
          objects/sha256/ab/<artifact_id>/files/...
          .staging/.bundle-<random>/...
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if self.root.exists() and not self.root.is_dir():
            raise ArtifactStoreError(f"artifact-store root is not a directory: {self.root}")
        self.objects_root = self.root / "objects" / "sha256"
        self.staging_root = self.root / ".staging"
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def artifact_dir(self, artifact_id: str) -> Path:
        digest = validate_sha256(artifact_id, field="artifact_id")
        candidate = self.objects_root / digest[:2] / digest
        resolved_parent = self.objects_root.resolve()
        resolved_candidate = candidate.resolve(strict=False)
        if not _is_relative_to(resolved_candidate, resolved_parent):
            raise ContractError("artifact path escapes the object store")
        return candidate

    def bundle(
        self,
        *,
        artifact_type: str,
        dependencies: AnalysisDependencies,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactBundleWriter":
        return ArtifactBundleWriter(
            self,
            artifact_type=artifact_type,
            dependencies=dependencies,
            metadata=metadata,
        )

    def put_bytes(
        self,
        *,
        artifact_type: str,
        dependencies: AnalysisDependencies,
        files: Mapping[str, bytes | bytearray | memoryview],
        metadata: Mapping[str, Any] | None = None,
        media_types: Mapping[str, str] | None = None,
    ) -> ArtifactRef:
        with self.bundle(
            artifact_type=artifact_type,
            dependencies=dependencies,
            metadata=metadata,
        ) as writer:
            for relative_path, data in files.items():
                media_type = media_types.get(relative_path) if media_types else None
                writer.add_bytes(relative_path, data, media_type=media_type)
            return writer.commit()

    def put_files(
        self,
        *,
        artifact_type: str,
        dependencies: AnalysisDependencies,
        files: Mapping[str, str | Path],
        metadata: Mapping[str, Any] | None = None,
        media_types: Mapping[str, str] | None = None,
    ) -> ArtifactRef:
        with self.bundle(
            artifact_type=artifact_type,
            dependencies=dependencies,
            metadata=metadata,
        ) as writer:
            for relative_path, source in files.items():
                media_type = media_types.get(relative_path) if media_types else None
                writer.add_file(relative_path, source, media_type=media_type)
            return writer.commit()

    def validate(
        self,
        artifact_id: str,
        *,
        expected_dependencies: AnalysisDependencies | None = None,
        expected_artifact_type: str | None = None,
    ) -> ValidationResult:
        digest = validate_sha256(artifact_id, field="artifact_id")
        artifact_dir = self.artifact_dir(digest)
        errors: list[str] = []
        manifest: ArtifactManifest | None = None

        if not artifact_dir.exists():
            return ValidationResult(digest, False, ("artifact directory does not exist",), None)
        if _is_link_like(artifact_dir) or not artifact_dir.is_dir():
            return ValidationResult(digest, False, ("artifact path is not a real directory",), None)

        allowed_root_entries = {MANIFEST_FILENAME, PAYLOAD_DIRECTORY}
        try:
            for child in artifact_dir.iterdir():
                if child.name not in allowed_root_entries:
                    errors.append(f"unexpected bundle entry: {child.name}")
                if _is_link_like(child):
                    errors.append(f"symbolic links are forbidden in bundles: {child.name}")
        except OSError as exc:
            errors.append(f"cannot enumerate artifact directory: {exc}")

        try:
            manifest = _read_manifest(artifact_dir / MANIFEST_FILENAME)
        except (ArtifactStoreError, ContractError) as exc:
            errors.append(str(exc))
            return ValidationResult(digest, False, tuple(errors), None)

        if manifest.artifact_id != digest:
            errors.append(
                f"manifest artifact_id {manifest.artifact_id} does not match requested address {digest}"
            )
        if expected_artifact_type is not None:
            expected_type = validate_artifact_type(expected_artifact_type)
            if manifest.artifact_type != expected_type:
                errors.append(
                    f"artifact type mismatch: expected {expected_type}, found {manifest.artifact_type}"
                )
        if expected_dependencies is not None and manifest.dependencies.digest != expected_dependencies.digest:
            errors.append(
                "dependency mismatch: "
                f"expected {expected_dependencies.digest}, found {manifest.dependencies.digest}"
            )

        payload_root = artifact_dir / PAYLOAD_DIRECTORY
        if _is_link_like(payload_root) or not payload_root.is_dir():
            errors.append("payload directory is missing or is a symbolic link")
            return ValidationResult(digest, False, tuple(errors), manifest)

        expected_paths = {record.path for record in manifest.files}
        expected_directories: set[str] = set()
        for expected_path in expected_paths:
            parts = expected_path.split("/")[:-1]
            for end in range(1, len(parts) + 1):
                expected_directories.add("/".join(parts[:end]))
        actual_paths: set[str] = set()
        actual_directories: set[str] = set()
        payload_root_resolved = payload_root.resolve()
        try:
            for entry in payload_root.rglob("*"):
                relative = entry.relative_to(payload_root).as_posix()
                if _is_link_like(entry):
                    errors.append(f"symbolic links are forbidden in payloads: {relative}")
                    continue
                if entry.is_file():
                    try:
                        normalized = normalize_artifact_relative_path(relative)
                    except ContractError as exc:
                        errors.append(f"unsafe on-disk payload path {relative!r}: {exc}")
                        continue
                    if not _is_relative_to(entry.resolve(), payload_root_resolved):
                        errors.append(f"payload escapes bundle: {relative}")
                        continue
                    actual_paths.add(normalized)
                elif entry.is_dir():
                    try:
                        actual_directories.add(normalize_artifact_relative_path(relative))
                    except ContractError as exc:
                        errors.append(f"unsafe on-disk payload directory {relative!r}: {exc}")
                else:
                    errors.append(f"unsupported payload entry: {relative}")
        except OSError as exc:
            errors.append(f"cannot enumerate payload directory: {exc}")

        for unexpected in sorted(actual_paths - expected_paths):
            errors.append(f"unmanifested payload file: {unexpected}")
        for unexpected in sorted(actual_directories - expected_directories):
            errors.append(f"unmanifested payload directory: {unexpected}")
        for missing in sorted(expected_paths - actual_paths):
            errors.append(f"manifested payload file is missing: {missing}")

        for record in manifest.files:
            if record.path not in actual_paths:
                continue
            payload = payload_root.joinpath(*record.path.split("/"))
            try:
                stat = payload.stat()
                if stat.st_size != record.size_bytes:
                    errors.append(
                        f"size mismatch for {record.path}: expected {record.size_bytes}, found {stat.st_size}"
                    )
                    continue
                checksum, size = sha256_file(payload)
                if size != record.size_bytes or checksum != record.sha256:
                    errors.append(
                        f"checksum mismatch for {record.path}: expected {record.sha256}, found {checksum}"
                    )
            except OSError as exc:
                errors.append(f"cannot validate {record.path}: {exc}")

        return ValidationResult(digest, not errors, tuple(errors), manifest)

    def get(
        self,
        artifact_id: str,
        *,
        expected_dependencies: AnalysisDependencies | None = None,
        expected_artifact_type: str | None = None,
    ) -> ArtifactRef:
        result = self.validate(
            artifact_id,
            expected_dependencies=expected_dependencies,
            expected_artifact_type=expected_artifact_type,
        )
        manifest = result.require_valid()
        return ArtifactRef(result.artifact_id, self.artifact_dir(result.artifact_id), manifest)

    def payload_path(
        self,
        artifact_id: str,
        relative_path: str,
        *,
        expected_dependencies: AnalysisDependencies | None = None,
        expected_artifact_type: str | None = None,
    ) -> Path:
        relative_path = normalize_artifact_relative_path(relative_path)
        ref = self.get(
            artifact_id,
            expected_dependencies=expected_dependencies,
            expected_artifact_type=expected_artifact_type,
        )
        manifested = {item.path for item in ref.manifest.files}
        if relative_path not in manifested:
            raise ArtifactNotFoundError(
                f"artifact {artifact_id} has no manifested payload {relative_path!r}"
            )
        return ref.path.joinpath(PAYLOAD_DIRECTORY, *relative_path.split("/"))

    def read_bytes(
        self,
        artifact_id: str,
        relative_path: str,
        *,
        expected_dependencies: AnalysisDependencies | None = None,
        expected_artifact_type: str | None = None,
    ) -> bytes:
        return self.payload_path(
            artifact_id,
            relative_path,
            expected_dependencies=expected_dependencies,
            expected_artifact_type=expected_artifact_type,
        ).read_bytes()


class ArtifactBundleWriter:
    """Build one artifact privately and publish it with an atomic directory rename."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        artifact_type: str,
        dependencies: AnalysisDependencies,
        metadata: Mapping[str, Any] | None,
    ):
        self.store = store
        self.artifact_type = validate_artifact_type(artifact_type)
        self.dependencies = dependencies
        normalized_metadata = canonical_json_value({} if metadata is None else metadata)
        if not isinstance(normalized_metadata, dict):
            raise ContractError("artifact metadata must be a JSON object")
        self.metadata = normalized_metadata
        self._stage = Path(tempfile.mkdtemp(prefix=".bundle-", dir=self.store.staging_root))
        self._payload_root = self._stage / PAYLOAD_DIRECTORY
        self._payload_root.mkdir()
        self._files: dict[str, ArtifactFile] = {}
        self._closed = False
        self._committed = False

    @property
    def staging_path(self) -> Path:
        return self._stage

    def _ensure_open(self) -> None:
        if self._closed:
            raise BundleStateError("artifact bundle is already committed or aborted")

    def _destination(self, relative_path: str) -> tuple[str, Path]:
        self._ensure_open()
        normalized = normalize_artifact_relative_path(relative_path)
        for existing in self._files:
            if (
                existing == normalized
                or existing.startswith(normalized + "/")
                or normalized.startswith(existing + "/")
            ):
                raise ContractError(f"duplicate or colliding artifact payload path: {relative_path!r}")
        destination = self._payload_root.joinpath(*normalized.split("/"))
        if not _is_relative_to(destination.resolve(strict=False), self._payload_root.resolve()):
            raise ContractError("artifact payload path escapes its staging bundle")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return normalized, destination

    def add_bytes(
        self,
        relative_path: str,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str | None = None,
    ) -> ArtifactFile:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        normalized, destination = self._destination(relative_path)
        payload = bytes(data)
        try:
            with destination.open("xb") as stream:
                _write_all_and_fsync(stream, payload)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        record = ArtifactFile(
            path=normalized,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            media_type=media_type,
        )
        self._files[normalized] = record
        return record

    def add_json(
        self,
        relative_path: str,
        value: Any,
        *,
        media_type: str = "application/json",
    ) -> ArtifactFile:
        return self.add_bytes(relative_path, canonical_json_bytes(value), media_type=media_type)

    def add_file(
        self,
        relative_path: str,
        source: str | Path,
        *,
        media_type: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> ArtifactFile:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"not a regular source file: {source_path}")
        normalized, destination = self._destination(relative_path)
        digest = hashlib.sha256()
        size = 0
        try:
            with source_path.open("rb") as source_stream, destination.open("xb") as destination_stream:
                while True:
                    chunk = source_stream.read(chunk_size)
                    if not chunk:
                        break
                    destination_stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        record = ArtifactFile(normalized, digest.hexdigest(), size, media_type)
        self._files[normalized] = record
        return record

    def commit(self) -> ArtifactRef:
        self._ensure_open()
        try:
            manifest = ArtifactManifest.create(
                artifact_type=self.artifact_type,
                dependencies=self.dependencies,
                files=tuple(self._files.values()),
                metadata=self.metadata,
            )
            manifest_bytes = canonical_json_bytes(manifest.to_dict()) + b"\n"
            if len(manifest_bytes) > MAX_MANIFEST_BYTES:
                raise ArtifactStoreError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
            _atomic_write_bytes(self._stage / MANIFEST_FILENAME, manifest_bytes)
            _fsync_directory(self._payload_root)
            _fsync_directory(self._stage)

            final = self.store.artifact_dir(manifest.artifact_id)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                existing = self.store.validate(
                    manifest.artifact_id,
                    expected_dependencies=self.dependencies,
                    expected_artifact_type=self.artifact_type,
                )
                existing_manifest = existing.require_valid()
                self._cleanup_stage()
                self._closed = True
                self._committed = True
                return ArtifactRef(manifest.artifact_id, final, existing_manifest)

            try:
                os.rename(self._stage, final)
            except OSError:
                # Another writer may have atomically published the same content.
                if final.exists():
                    existing = self.store.validate(
                        manifest.artifact_id,
                        expected_dependencies=self.dependencies,
                        expected_artifact_type=self.artifact_type,
                    )
                    existing_manifest = existing.require_valid()
                    self._cleanup_stage()
                    self._closed = True
                    self._committed = True
                    return ArtifactRef(manifest.artifact_id, final, existing_manifest)
                raise

            self._closed = True
            self._committed = True
            _fsync_directory(final.parent)
            published = self.store.validate(
                manifest.artifact_id,
                expected_dependencies=self.dependencies,
                expected_artifact_type=self.artifact_type,
            )
            published_manifest = published.require_valid()
            return ArtifactRef(manifest.artifact_id, final, published_manifest)
        except Exception:
            if not self._committed:
                self._cleanup_stage()
                self._closed = True
            raise

    def _cleanup_stage(self) -> None:
        if not self._stage.exists():
            return
        stage_root = self.store.staging_root.resolve()
        stage = self._stage.resolve()
        if not _is_relative_to(stage, stage_root) or not self._stage.name.startswith(".bundle-"):
            raise ArtifactStoreError(f"refusing to remove unsafe staging path: {self._stage}")
        shutil.rmtree(self._stage)

    def abort(self) -> None:
        if self._closed:
            return
        self._cleanup_stage()
        self._closed = True

    def __enter__(self) -> "ArtifactBundleWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._closed:
            self.abort()
        return False
