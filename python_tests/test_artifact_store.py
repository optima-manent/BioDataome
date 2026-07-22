from __future__ import annotations

import json
from dataclasses import replace

import pytest
from cskl_atlas.artifact_store import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStore,
)
from cskl_atlas.fingerprints import fingerprint_feature_space, sha256_bytes
from cskl_atlas.models import (
    AnalysisDependencies,
    ComponentFingerprint,
    ContractError,
    SourceFingerprint,
)


def make_dependencies() -> AnalysisDependencies:
    return AnalysisDependencies(
        source=SourceFingerprint.from_bytes(b"matrix", source_id="GSE200-GPL570"),
        normalization=ComponentFingerprint.from_config(
            name="scan-upc",
            version="1",
            code_sha256=sha256_bytes(b"normalize"),
            config={"mode": "single-sample"},
        ),
        core=ComponentFingerprint.from_config(
            name="cskl-core",
            version="1",
            code_sha256=sha256_bytes(b"core"),
            config={"r_compat_noise": True},
        ),
        alpha=0.5,
        seed=0,
        feature_space_sha256=fingerprint_feature_space(["p1", "p2", "p3"]),
        qc=ComponentFingerprint.from_config(
            name="qc",
            version="1",
            code_sha256=sha256_bytes(b"qc"),
            config={"max_nonfinite_fraction": 0.01},
        ),
    )


def test_bundle_is_content_addressed_atomic_and_order_independent(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    dependencies = make_dependencies()
    metadata = {"dataset": "GSE200", "component_count": 7}

    first = store.put_bytes(
        artifact_type="dataset-signature",
        dependencies=dependencies,
        files={"signature.npz": b"npz-data", "qc/report.json": b"{}"},
        metadata=metadata,
    )
    second = store.put_bytes(
        artifact_type="dataset-signature",
        dependencies=dependencies,
        files={"qc/report.json": b"{}", "signature.npz": b"npz-data"},
        metadata={"component_count": 7, "dataset": "GSE200"},
    )

    assert first.artifact_id == second.artifact_id
    assert first.path == second.path
    assert store.validate(
        first.artifact_id,
        expected_dependencies=dependencies,
        expected_artifact_type="dataset-signature",
    ).valid
    assert store.read_bytes(first.artifact_id, "signature.npz") == b"npz-data"
    assert not any(store.staging_root.iterdir())


def test_dependency_expectation_is_checked_not_just_directory_presence(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    dependencies = make_dependencies()
    ref = store.put_bytes(
        artifact_type="null-profile",
        dependencies=dependencies,
        files={"profile.npz": b"profile"},
    )

    wrong = replace(dependencies, seed=1)
    result = store.validate(ref.artifact_id, expected_dependencies=wrong)
    assert not result.valid
    assert any("dependency mismatch" in error for error in result.errors)

    fake_id = "f" * 64
    fake_dir = store.artifact_dir(fake_id)
    (fake_dir / "files").mkdir(parents=True)
    (fake_dir / "files" / "present-but-unmanifested.bin").write_bytes(b"data")
    presence_only = store.validate(fake_id)
    assert not presence_only.valid
    assert any("missing manifest" in error for error in presence_only.errors)


def test_payload_corruption_is_detected_and_never_overwritten(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    dependencies = make_dependencies()
    ref = store.put_bytes(
        artifact_type="dataset-signature",
        dependencies=dependencies,
        files={"signature.npz": b"original"},
    )
    payload = ref.path / "files" / "signature.npz"
    payload.write_bytes(b"tampered")

    result = store.validate(ref.artifact_id)
    assert not result.valid
    assert any("size mismatch" in error or "checksum mismatch" in error for error in result.errors)

    with pytest.raises(ArtifactIntegrityError):
        store.put_bytes(
            artifact_type="dataset-signature",
            dependencies=dependencies,
            files={"signature.npz": b"original"},
        )
    assert payload.read_bytes() == b"tampered"


def test_manifest_tampering_and_unmanifested_files_are_detected(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(
        artifact_type="dataset-signature",
        dependencies=make_dependencies(),
        files={"signature.npz": b"signature"},
    )

    extra = ref.path / "files" / "extra.bin"
    extra.write_bytes(b"not in manifest")
    result = store.validate(ref.artifact_id)
    assert not result.valid
    assert any("unmanifested payload" in error for error in result.errors)
    extra.unlink()

    extra_directory = ref.path / "files" / "empty-unmanifested-directory"
    extra_directory.mkdir()
    result = store.validate(ref.artifact_id)
    assert not result.valid
    assert any("unmanifested payload directory" in error for error in result.errors)
    extra_directory.rmdir()

    manifest_path = ref.path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["dependency_id"] = "0" * 64
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    result = store.validate(ref.artifact_id)
    assert not result.valid
    assert any("dependency_id" in error for error in result.errors)


def test_aborted_bundle_never_becomes_visible_and_staging_is_cleaned(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(RuntimeError):
        with store.bundle(
            artifact_type="dataset-signature",
            dependencies=make_dependencies(),
        ) as writer:
            writer.add_bytes("signature.npz", b"partial")
            raise RuntimeError("simulated crash before commit")

    assert not list(store.objects_root.rglob("manifest.json"))
    assert not any(store.staging_root.iterdir())


def test_put_files_and_manifested_payload_lookup(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"streamed source")
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_files(
        artifact_type="normalized-expression",
        dependencies=make_dependencies(),
        files={"matrix/expr.tsv.gz": source},
        media_types={"matrix/expr.tsv.gz": "application/gzip"},
    )

    assert store.payload_path(ref.artifact_id, r"matrix\expr.tsv.gz").read_bytes() == b"streamed source"
    with pytest.raises(ArtifactNotFoundError):
        store.payload_path(ref.artifact_id, "matrix/not-manifested.tsv.gz")


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", r"C:\escape", "NUL"])
def test_writer_rejects_unsafe_payload_paths(tmp_path, unsafe: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    with store.bundle(
        artifact_type="dataset-signature",
        dependencies=make_dependencies(),
    ) as writer:
        with pytest.raises(ContractError):
            writer.add_bytes(unsafe, b"data")


def test_writer_rejects_file_directory_collisions(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    with store.bundle(
        artifact_type="dataset-signature",
        dependencies=make_dependencies(),
    ) as writer:
        writer.add_bytes("nested", b"file")
        with pytest.raises(ContractError):
            writer.add_bytes("nested/child", b"collision")
