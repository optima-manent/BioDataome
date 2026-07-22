"""Immutable artifact and dependency contracts for C-SKL Atlas."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any

from .fingerprints import (
    canonical_json_value,
    fingerprint_json,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)

MANIFEST_SCHEMA_VERSION = 1
DEPENDENCY_SCHEMA_VERSION = 1
_ARTIFACT_TYPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_WINDOWS_FORBIDDEN = set('<>:"|?*')


class ContractError(ValueError):
    """Raised when an artifact contract is malformed or internally inconsistent."""


def _require_exact_keys(data: Mapping[str, Any], expected: set[str], context: str) -> None:
    if not isinstance(data, Mapping):
        raise ContractError(f"{context} must be a mapping")
    actual = set(data)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise ContractError(f"invalid {context} keys: {', '.join(details)}")


def _nonempty_text(value: str, field_name: str, *, max_length: int = 512) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field_name} must be a string")
    value = unicodedata.normalize("NFC", value)
    if not value or len(value) > max_length or "\x00" in value:
        raise ContractError(f"{field_name} must be non-empty, NUL-free, and <= {max_length} characters")
    return value


def validate_artifact_type(value: str) -> str:
    value = _nonempty_text(value, "artifact_type", max_length=128)
    if _ARTIFACT_TYPE_RE.fullmatch(value) is None or ".." in value:
        raise ContractError(
            "artifact_type must contain only letters, numbers, '.', '_', and '-' without '..'"
        )
    return value


def normalize_artifact_relative_path(value: str) -> str:
    """Return a portable safe payload path, rejecting traversal and Windows aliases."""

    raw = _nonempty_text(value, "artifact payload path", max_length=2048)
    if PureWindowsPath(raw).drive or PureWindowsPath(raw).root:
        raise ContractError(f"artifact payload path must be relative: {value!r}")
    raw = raw.replace("\\", "/")
    if raw.startswith("/") or raw.endswith("/"):
        raise ContractError(f"artifact payload path must name a file: {value!r}")
    raw_parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ContractError(f"artifact payload path contains an unsafe segment: {value!r}")

    parts: list[str] = []
    for raw_part in raw_parts:
        part = unicodedata.normalize("NFC", raw_part)
        if len(part) > 255 or part.endswith((" ", ".")):
            raise ContractError(f"artifact payload path has a non-portable segment: {raw_part!r}")
        if any(ord(char) < 32 or char in _WINDOWS_FORBIDDEN for char in part):
            raise ContractError(f"artifact payload path has a forbidden character: {raw_part!r}")
        if part.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise ContractError(f"artifact payload path uses a reserved Windows name: {raw_part!r}")
        parts.append(part)
    return "/".join(parts)


def _freeze_json(value: Any) -> Any:
    value = canonical_json_value(value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class SourceFingerprint:
    """Identity of one immutable source asset."""

    content_sha256: str
    size_bytes: int
    source_id: str
    revision: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_sha256", validate_sha256(self.content_sha256, field="content_sha256"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ContractError("size_bytes must be a non-negative integer")
        object.__setattr__(self, "source_id", _nonempty_text(self.source_id, "source_id"))
        if self.revision is not None:
            object.__setattr__(self, "revision", _nonempty_text(self.revision, "revision"))
        if self.media_type is not None:
            object.__setattr__(self, "media_type", _nonempty_text(self.media_type, "media_type", max_length=255))

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        source_id: str,
        revision: str | None = None,
        media_type: str | None = None,
    ) -> "SourceFingerprint":
        return cls(sha256_bytes(data), memoryview(data).nbytes, source_id, revision, media_type)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        source_id: str | None = None,
        revision: str | None = None,
        media_type: str | None = None,
    ) -> "SourceFingerprint":
        source = Path(path)
        digest, size = sha256_file(source)
        return cls(digest, size, source_id or source.name, revision, media_type)

    @property
    def digest(self) -> str:
        return fingerprint_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "media_type": self.media_type,
            "revision": self.revision,
            "size_bytes": self.size_bytes,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceFingerprint":
        _require_exact_keys(
            data,
            {"content_sha256", "media_type", "revision", "size_bytes", "source_id"},
            "source fingerprint",
        )
        return cls(
            content_sha256=data["content_sha256"],
            size_bytes=data["size_bytes"],
            source_id=data["source_id"],
            revision=data["revision"],
            media_type=data["media_type"],
        )


@dataclass(frozen=True)
class ComponentFingerprint:
    """Versioned code plus canonical configuration for a pipeline component."""

    name: str
    version: str
    code_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "component name", max_length=128))
        object.__setattr__(self, "version", _nonempty_text(self.version, "component version", max_length=128))
        object.__setattr__(self, "code_sha256", validate_sha256(self.code_sha256, field="code_sha256"))
        object.__setattr__(self, "config_sha256", validate_sha256(self.config_sha256, field="config_sha256"))

    @classmethod
    def from_config(
        cls,
        *,
        name: str,
        version: str,
        code_sha256: str,
        config: Mapping[str, Any],
    ) -> "ComponentFingerprint":
        if not isinstance(config, Mapping):
            raise ContractError("component config must be a JSON object")
        return cls(name, version, code_sha256, fingerprint_json(config))

    @property
    def digest(self) -> str:
        return fingerprint_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "name": self.name,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComponentFingerprint":
        _require_exact_keys(data, {"code_sha256", "config_sha256", "name", "version"}, "component fingerprint")
        return cls(
            name=data["name"],
            version=data["version"],
            code_sha256=data["code_sha256"],
            config_sha256=data["config_sha256"],
        )


@dataclass(frozen=True, order=True)
class NamedFingerprint:
    """An additional named dependency, such as a pool, grid, or annotation release."""

    name: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "fingerprint name", max_length=128))
        object.__setattr__(self, "digest", validate_sha256(self.digest, field=f"{self.name} digest"))

    def to_dict(self) -> dict[str, str]:
        return {"digest": self.digest, "name": self.name}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NamedFingerprint":
        _require_exact_keys(data, {"digest", "name"}, "named fingerprint")
        return cls(name=data["name"], digest=data["digest"])


@dataclass(frozen=True)
class AnalysisDependencies:
    """Complete dependency identity for a derived numerical artifact.

    The required fields intentionally include every invalidation dimension that
    the legacy file-presence cache omitted: source content, normalization code and
    settings, core code and settings, alpha, seed, ordered feature space, and QC
    code/policy. ``extra`` binds artifact-specific inputs such as pool/grid/B.
    """

    source: SourceFingerprint
    normalization: ComponentFingerprint
    core: ComponentFingerprint
    alpha: float
    seed: int
    feature_space_sha256: str
    qc: ComponentFingerprint
    extra: tuple[NamedFingerprint, ...] = ()
    schema_version: int = DEPENDENCY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != DEPENDENCY_SCHEMA_VERSION:
            raise ContractError(f"unsupported dependency schema_version: {self.schema_version}")
        if not isinstance(self.source, SourceFingerprint):
            raise ContractError("source must be a SourceFingerprint")
        if not isinstance(self.normalization, ComponentFingerprint):
            raise ContractError("normalization must be a ComponentFingerprint")
        if not isinstance(self.core, ComponentFingerprint):
            raise ContractError("core must be a ComponentFingerprint")
        if not isinstance(self.qc, ComponentFingerprint):
            raise ContractError("qc must be a ComponentFingerprint")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ContractError("alpha must be a finite number in (0, 1)")
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ContractError("alpha must be a finite number in (0, 1)")
        object.__setattr__(self, "alpha", alpha)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise ContractError("seed must be an integer in [0, 2**64)")
        object.__setattr__(
            self,
            "feature_space_sha256",
            validate_sha256(self.feature_space_sha256, field="feature_space_sha256"),
        )
        if not isinstance(self.extra, tuple) or not all(isinstance(item, NamedFingerprint) for item in self.extra):
            raise ContractError("extra must be a tuple of NamedFingerprint values")
        extras = tuple(sorted(self.extra, key=lambda item: item.name))
        if len({item.name for item in extras}) != len(extras):
            raise ContractError("extra dependency fingerprint names must be unique")
        object.__setattr__(self, "extra", extras)

    @property
    def digest(self) -> str:
        return fingerprint_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "core": self.core.to_dict(),
            "extra": [item.to_dict() for item in self.extra],
            "feature_space_sha256": self.feature_space_sha256,
            "normalization": self.normalization.to_dict(),
            "qc": self.qc.to_dict(),
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisDependencies":
        _require_exact_keys(
            data,
            {
                "alpha",
                "core",
                "extra",
                "feature_space_sha256",
                "normalization",
                "qc",
                "schema_version",
                "seed",
                "source",
            },
            "analysis dependencies",
        )
        extra = data["extra"]
        if not isinstance(extra, Sequence) or isinstance(extra, (str, bytes, bytearray)):
            raise ContractError("analysis dependencies extra must be an array")
        return cls(
            source=SourceFingerprint.from_dict(data["source"]),
            normalization=ComponentFingerprint.from_dict(data["normalization"]),
            core=ComponentFingerprint.from_dict(data["core"]),
            alpha=data["alpha"],
            seed=data["seed"],
            feature_space_sha256=data["feature_space_sha256"],
            qc=ComponentFingerprint.from_dict(data["qc"]),
            extra=tuple(NamedFingerprint.from_dict(item) for item in extra),
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True, order=True)
class ArtifactFile:
    path: str
    sha256: str
    size_bytes: int
    media_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_artifact_relative_path(self.path))
        object.__setattr__(self, "sha256", validate_sha256(self.sha256, field=f"sha256 for {self.path}"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ContractError("artifact file size_bytes must be a non-negative integer")
        if self.media_type is not None:
            object.__setattr__(self, "media_type", _nonempty_text(self.media_type, "media_type", max_length=255))

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactFile":
        _require_exact_keys(data, {"media_type", "path", "sha256", "size_bytes"}, "artifact file")
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            media_type=data["media_type"],
        )


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    artifact_type: str
    dependency_id: str
    dependencies: AnalysisDependencies
    files: tuple[ArtifactFile, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ContractError(f"unsupported manifest schema_version: {self.schema_version}")
        if not isinstance(self.dependencies, AnalysisDependencies):
            raise ContractError("dependencies must be an AnalysisDependencies contract")
        object.__setattr__(self, "artifact_id", validate_sha256(self.artifact_id, field="artifact_id"))
        object.__setattr__(self, "artifact_type", validate_artifact_type(self.artifact_type))
        object.__setattr__(self, "dependency_id", validate_sha256(self.dependency_id, field="dependency_id"))
        if self.dependency_id != self.dependencies.digest:
            raise ContractError("manifest dependency_id does not match its dependency contract")
        if not isinstance(self.files, tuple) or not all(isinstance(item, ArtifactFile) for item in self.files):
            raise ContractError("files must be a tuple of ArtifactFile records")
        files = tuple(sorted(self.files, key=lambda item: item.path))
        if not files:
            raise ContractError("an artifact bundle must contain at least one payload file")
        if len({item.path for item in files}) != len(files):
            raise ContractError("artifact payload paths must be unique")
        object.__setattr__(self, "files", files)
        if not isinstance(self.metadata, Mapping):
            raise ContractError("artifact metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        expected = self.compute_artifact_id(
            artifact_type=self.artifact_type,
            dependency_id=self.dependency_id,
            files=files,
            metadata=self.metadata,
            schema_version=self.schema_version,
        )
        if self.artifact_id != expected:
            raise ContractError("manifest artifact_id does not match its content")

    @staticmethod
    def compute_artifact_id(
        *,
        artifact_type: str,
        dependency_id: str,
        files: Sequence[ArtifactFile],
        metadata: Mapping[str, Any],
        schema_version: int = MANIFEST_SCHEMA_VERSION,
    ) -> str:
        return fingerprint_json(
            {
                "artifact_type": validate_artifact_type(artifact_type),
                "dependency_id": validate_sha256(dependency_id, field="dependency_id"),
                "files": [item.to_dict() for item in sorted(files, key=lambda item: item.path)],
                "metadata": _thaw_json(_freeze_json(metadata)),
                "schema_version": schema_version,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        artifact_type: str,
        dependencies: AnalysisDependencies,
        files: Sequence[ArtifactFile],
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactManifest":
        metadata = {} if metadata is None else metadata
        dependency_id = dependencies.digest
        artifact_id = cls.compute_artifact_id(
            artifact_type=artifact_type,
            dependency_id=dependency_id,
            files=files,
            metadata=metadata,
        )
        return cls(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            dependency_id=dependency_id,
            dependencies=dependencies,
            files=tuple(files),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "dependencies": self.dependencies.to_dict(),
            "dependency_id": self.dependency_id,
            "files": [item.to_dict() for item in self.files],
            "metadata": _thaw_json(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactManifest":
        _require_exact_keys(
            data,
            {
                "artifact_id",
                "artifact_type",
                "dependencies",
                "dependency_id",
                "files",
                "metadata",
                "schema_version",
            },
            "artifact manifest",
        )
        files = data["files"]
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)):
            raise ContractError("manifest files must be an array")
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            dependency_id=data["dependency_id"],
            dependencies=AnalysisDependencies.from_dict(data["dependencies"]),
            files=tuple(ArtifactFile.from_dict(item) for item in files),
            metadata=data["metadata"],
            schema_version=data["schema_version"],
        )
