from __future__ import annotations

from dataclasses import replace

import pytest
from cskl_atlas.fingerprints import (
    FingerprintError,
    canonical_json_bytes,
    fingerprint_feature_space,
    fingerprint_json,
    sha256_bytes,
)
from cskl_atlas.models import (
    AnalysisDependencies,
    ArtifactFile,
    ArtifactManifest,
    ComponentFingerprint,
    ContractError,
    SourceFingerprint,
    normalize_artifact_relative_path,
)


def make_dependencies() -> AnalysisDependencies:
    return AnalysisDependencies(
        source=SourceFingerprint.from_bytes(
            b"source-matrix-v1",
            source_id="GSE100-GPL570",
            revision="geo-2026-07-19",
            media_type="text/tab-separated-values",
        ),
        normalization=ComponentFingerprint.from_config(
            name="scan-upc",
            version="2.46.0",
            code_sha256=sha256_bytes(b"normalization R source"),
            config={"package": "SCAN.UPC", "r": "4.5.1"},
        ),
        core=ComponentFingerprint.from_config(
            name="cskl-core",
            version="0.2.0",
            code_sha256=sha256_bytes(b"cskl.py source"),
            config={"r_compat_noise": True},
        ),
        alpha=0.5,
        seed=17,
        feature_space_sha256=fingerprint_feature_space(["1007_s_at", "1053_at"]),
        qc=ComponentFingerprint.from_config(
            name="expression-qc",
            version="1",
            code_sha256=sha256_bytes(b"qc.py source"),
            config={"max_nonfinite_fraction": 0.01, "min_samples": 2},
        ),
    )


def test_canonical_json_is_order_stable_and_rejects_ambiguous_values() -> None:
    left = {"b": [2, 1], "a": {"x": True}}
    right = {"a": {"x": True}, "b": (2, 1)}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert fingerprint_json(left) == fingerprint_json(right)

    with pytest.raises(FingerprintError):
        fingerprint_json({"bad": float("nan")})
    with pytest.raises(FingerprintError):
        fingerprint_json({"unordered": {1, 2}})
    with pytest.raises(FingerprintError):
        fingerprint_json({1: "non-string key"})


def test_feature_space_fingerprint_is_order_sensitive() -> None:
    first = fingerprint_feature_space(["probe-a", "probe-b"])
    second = fingerprint_feature_space(["probe-b", "probe-a"])
    assert first != second


def test_dependency_roundtrip_and_every_required_input_invalidates() -> None:
    base = make_dependencies()
    assert AnalysisDependencies.from_dict(base.to_dict()) == base

    variants = [
        replace(
            base,
            source=SourceFingerprint.from_bytes(
                b"source-matrix-v2", source_id="GSE100-GPL570", revision="geo-2026-07-19"
            ),
        ),
        replace(
            base,
            normalization=replace(
                base.normalization,
                config_sha256=sha256_bytes(b"different normalization parameters"),
            ),
        ),
        replace(base, core=replace(base.core, code_sha256=sha256_bytes(b"different cskl core"))),
        replace(base, alpha=0.6),
        replace(base, seed=18),
        replace(base, feature_space_sha256=fingerprint_feature_space(["probe-a", "probe-c"])),
        replace(base, qc=replace(base.qc, config_sha256=sha256_bytes(b"different QC policy"))),
    ]
    assert len({base.digest, *(variant.digest for variant in variants)}) == len(variants) + 1


def test_manifest_id_binds_dependencies_payloads_paths_type_and_metadata() -> None:
    dependencies = make_dependencies()
    file_record = ArtifactFile(
        path="signature/signature.npz",
        sha256=sha256_bytes(b"payload"),
        size_bytes=7,
        media_type="application/octet-stream",
    )
    metadata = {"dataset": "GSE100", "labels": ["baseline"]}
    manifest = ArtifactManifest.create(
        artifact_type="dataset-signature",
        dependencies=dependencies,
        files=[file_record],
        metadata=metadata,
    )
    assert ArtifactManifest.from_dict(manifest.to_dict()) == manifest

    metadata["labels"].append("mutated-after-construction")
    assert manifest.to_dict()["metadata"]["labels"] == ["baseline"]

    changed_dependency = ArtifactManifest.create(
        artifact_type="dataset-signature",
        dependencies=replace(dependencies, seed=dependencies.seed + 1),
        files=[file_record],
        metadata={"dataset": "GSE100", "labels": ["baseline"]},
    )
    changed_path = ArtifactManifest.create(
        artifact_type="dataset-signature",
        dependencies=dependencies,
        files=[replace(file_record, path="other/signature.npz")],
        metadata={"dataset": "GSE100", "labels": ["baseline"]},
    )
    assert len({manifest.artifact_id, changed_dependency.artifact_id, changed_path.artifact_id}) == 3


@pytest.mark.parametrize(
    "unsafe",
    [
        "",
        ".",
        "../escape",
        "nested/../../escape",
        "/absolute/file",
        r"C:\absolute\file",
        r"\\server\share\file",
        "nested//file",
        "NUL.txt",
        "trailing-dot.",
        "alternate:stream",
    ],
)
def test_artifact_relative_paths_reject_traversal_and_nonportable_names(unsafe: str) -> None:
    with pytest.raises(ContractError):
        normalize_artifact_relative_path(unsafe)


def test_artifact_relative_path_has_one_canonical_separator() -> None:
    assert normalize_artifact_relative_path(r"nested\signature.npz") == "nested/signature.npz"
