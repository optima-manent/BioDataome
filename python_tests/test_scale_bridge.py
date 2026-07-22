from __future__ import annotations

import zipfile

import cskl
import numpy as np
from cskl_atlas.catalog import Catalog
from cskl_atlas.scale_bridge import import_scale_store
from cskl_pipeline.scale.store import Store, atomic_write_json, save_signature


def test_scale_store_bridge_imports_complete_artifacts_idempotently(tmp_path):
    store = Store(tmp_path / "scale-store")
    store.write_probes("GPL570", ["p1", "p2", "p3"])
    feature_hash = store.feature_hash("GPL570")
    dataset_dir = store.dataset_dir("GSE1")
    dataset_dir.mkdir(parents=True)
    store.expr_path("GSE1").write_bytes(b"probe\ts1\ts2\np1\t1\t2\n")
    signature = cskl.PCASignature(
        P=np.array([[1.0], [0.0], [0.0]]),
        lam=np.array([1.5]),
        n_features=3,
        m_samples=2,
        alpha=0.5,
        feature_names=None,
    )
    save_signature(store.signature_path("GSE1"), signature, feature_hash)
    atomic_write_json(
        store.sample_hashes_path("GSE1"),
        {"n_samples": 2, "hashes": ["a" * 40, "b" * 40]},
    )
    atomic_write_json(store.qc_path("GSE1"), {"status": "ok"})

    # An incomplete directory is reported, never silently promoted.
    store.dataset_dir("GSE2").mkdir(parents=True)

    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    first = import_scale_store(
        catalog,
        store_root=store.root,
        platform="GPL570",
        source_revision="local-store-v1",
    )
    assert first["counts"] == {"promoted": 1, "skipped": 1}
    assert first["feature_hash"] == feature_hash
    second = import_scale_store(
        catalog,
        store_root=store.root,
        platform="GPL570",
        source_revision="local-store-v1",
    )
    assert second["results"][0]["version_id"] == first["results"][0]["version_id"]
    with catalog.reader() as connection:
        assert connection.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM dataset_samples").fetchone()[0] == 2


def test_scale_store_bridge_uses_verified_preserved_archive_when_matrix_is_evicted(tmp_path):
    store = Store(tmp_path / "scale-store")
    store.write_probes("GPL570", ["p1", "p2", "p3"])
    feature_hash = store.feature_hash("GPL570")
    store.dataset_dir("GSE9").mkdir(parents=True)
    signature = cskl.PCASignature(
        P=np.array([[1.0], [0.0], [0.0]]),
        lam=np.array([1.5]),
        n_features=3,
        m_samples=2,
        alpha=0.5,
        feature_names=None,
    )
    save_signature(store.signature_path("GSE9"), signature, feature_hash)
    atomic_write_json(
        store.sample_hashes_path("GSE9"),
        {"n_samples": 2, "hashes": ["c" * 40, "d" * 40]},
    )
    archive = tmp_path / "preserved.zip"
    source_bytes = b"probe,s1,s2\np1,1,2\n"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
        zipped.writestr("Public/Dataome/GPL570/GSE9.csv", source_bytes)

    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    result = import_scale_store(
        catalog,
        store_root=store.root,
        platform="GPL570",
        source_revision="",
        source_archive=archive,
    )
    assert result["counts"] == {"promoted": 1}
    assert result["source_revision"].startswith("zip-central:")
    with catalog.reader() as connection:
        version = connection.execute("SELECT * FROM dataset_versions").fetchone()
        artifact = connection.execute(
            "SELECT * FROM artifacts WHERE kind='normalized_matrix'"
        ).fetchone()
    import hashlib

    assert version["source_hash"] == hashlib.sha256(source_bytes).hexdigest()
    assert artifact["uri"].startswith("zip+file:")
    assert "GSE9.csv" in artifact["uri"]
