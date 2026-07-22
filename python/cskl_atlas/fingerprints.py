"""Deterministic SHA-256 fingerprints for C-SKL Atlas artifacts.

The helpers in this module deliberately accept only JSON-compatible values.
Rejecting implicit conversions (sets, paths, arbitrary objects, NaN, and
infinity) keeps dependency identities stable across processes and machines.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

SHA256_HEX_LENGTH = 64
DEFAULT_CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class FingerprintError(ValueError):
    """Raised when a value cannot be fingerprinted unambiguously."""


def validate_sha256(value: str, *, field: str = "sha256") -> str:
    """Return a normalized SHA-256 digest or raise :class:`FingerprintError`."""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FingerprintError(f"{field} must be exactly 64 hexadecimal characters")
    return value.lower()


def _normalize_text(value: str) -> str:
    if "\x00" in value:
        raise FingerprintError("JSON strings must not contain NUL characters")
    return unicodedata.normalize("NFC", value)


def canonical_json_value(value: Any) -> Any:
    """Return a detached, canonical JSON-compatible representation of ``value``.

    Mapping keys must be strings. Strings are normalized to Unicode NFC, and
    duplicate keys created by normalization are rejected. Tuples are represented
    as JSON arrays. Sets and arbitrary iterables are intentionally unsupported
    because their ordering is not an artifact contract.
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FingerprintError("JSON numbers must be finite")
        # Avoid two fingerprints for the semantically identical 0.0 and -0.0.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise FingerprintError("JSON object keys must be strings")
            key = _normalize_text(raw_key)
            if key in normalized:
                raise FingerprintError(f"duplicate JSON key after Unicode normalization: {key!r}")
            normalized[key] = canonical_json_value(raw_value)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [canonical_json_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return canonical_json_value(value.to_dict())
    raise FingerprintError(
        f"unsupported value for canonical JSON: {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize ``value`` to deterministic UTF-8 JSON bytes."""

    normalized = canonical_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """SHA-256 of an in-memory byte string."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    return hashlib.sha256(data).hexdigest()


def sha256_stream(stream: BinaryIO, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[str, int]:
    """Consume a binary stream and return ``(sha256, size_bytes)``."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("binary stream returned non-bytes data")
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[str, int]:
    """Return the SHA-256 and size of a regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"not a regular file: {source}")
    with source.open("rb") as stream:
        return sha256_stream(stream, chunk_size=chunk_size)


def fingerprint_json(value: Any) -> str:
    """SHA-256 fingerprint of canonical JSON data."""

    return sha256_bytes(canonical_json_bytes(value))


def fingerprint_feature_space(feature_ids: Sequence[str]) -> str:
    """Order-sensitive fingerprint of a frozen feature-space identifier list."""

    if isinstance(feature_ids, (str, bytes, bytearray)):
        raise FingerprintError("feature_ids must be a sequence of strings")
    normalized: list[str] = []
    for index, feature_id in enumerate(feature_ids):
        if not isinstance(feature_id, str) or not feature_id:
            raise FingerprintError(f"feature_ids[{index}] must be a non-empty string")
        normalized.append(_normalize_text(feature_id))
    if not normalized:
        raise FingerprintError("feature_ids must not be empty")
    return fingerprint_json({"feature_ids": normalized, "schema": "ordered-feature-space/v1"})


def combine_fingerprints(namespace: str, parts: Mapping[str, str]) -> str:
    """Combine named SHA-256 values into a domain-separated fingerprint."""

    if not isinstance(namespace, str) or not namespace.strip():
        raise FingerprintError("namespace must be a non-empty string")
    normalized_parts = {
        str(name): validate_sha256(digest, field=f"parts[{name!r}]")
        for name, digest in parts.items()
    }
    return fingerprint_json(
        {
            "namespace": _normalize_text(namespace),
            "parts": normalized_parts,
            "schema": "combined-fingerprint/v1",
        }
    )
