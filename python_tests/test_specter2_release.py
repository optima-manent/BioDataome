from __future__ import annotations

import hashlib

import numpy as np
from cskl_atlas.catalog import Catalog
from cskl_atlas.specter2_release import build_specter2_release
from cskl_pipeline.scale.store import atomic_write_json


def _add_version(catalog: Catalog, accession: str, marker: str) -> str:
    _, version_id = catalog.register_dataset_version(
        accession=accession,
        platform="GPL570",
        cohort="series",
        source_revision="test",
        source_hash=marker * 64,
        normalized_hash=marker * 64,
        signature_hash=(str((int(marker) + 1) % 10)) * 64,
        feature_hash="f" * 40,
        config_hash="e" * 64,
        sample_count=2,
        metadata={},
    )
    for kind, checksum, dependency in (
        ("normalized_matrix", marker * 64, hashlib.sha256(f"{accession}:n".encode()).hexdigest()),
        (
            "pca_signature",
            (str((int(marker) + 1) % 10)) * 64,
            hashlib.sha256(f"{accession}:s".encode()).hexdigest(),
        ),
    ):
        catalog.record_artifact(
            artifact_id=f"{accession}:{kind}",
            kind=kind,
            uri=f"/test/{accession}/{kind}",
            checksum=checksum,
            dependency_hash=dependency,
            manifest={},
            dataset_version_id=version_id,
        )
    catalog.promote_dataset_version(version_id)
    return version_id


def test_specter2_release_is_complete_pinned_and_replay_safe(tmp_path):
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    versions = [_add_version(catalog, f"GSE{index}", str(index)) for index in range(1, 4)]
    records = tmp_path / "geo" / "records"
    records.mkdir(parents=True)
    for index in range(1, 4):
        atomic_write_json(
            records / f"GSE{index}.json",
            {
                "title": f"Title {index}",
                "summary": f"Summary {index}",
                "content_sha256": str(index) * 64,
            },
        )

    calls = 0

    def embedder(titles, summaries, device, batch_size):
        nonlocal calls
        calls += 1
        assert titles == ["Title 1", "Title 2", "Title 3"]
        assert summaries == ["Summary 1", "Summary 2", "Summary 3"]
        assert device == "auto"
        assert batch_size == 0
        return np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32), {"device": "test"}

    first = build_specter2_release(
        catalog,
        metadata_directory=tmp_path / "geo",
        output_path=tmp_path / "embeddings.npz",
        embedder=embedder,
    )
    assert first["pair_count"] == 3
    assert first["reused"] is False
    with catalog.reader() as connection:
        rows = connection.execute(
            """SELECT version_a,version_b,cosine_similarity,similarity_percentile
               FROM text_pair_scores ORDER BY cosine_similarity DESC"""
        ).fetchall()
        release = connection.execute("SELECT * FROM text_releases").fetchone()
    assert len(rows) == 3
    assert {rows[0]["version_a"], rows[0]["version_b"]} == {versions[0], versions[1]}
    assert abs(rows[0]["cosine_similarity"] - 0.8) < 1e-6
    assert all(row["similarity_percentile"] is not None for row in rows)
    assert release["status"] == "finalized"

    second = build_specter2_release(
        catalog,
        metadata_directory=tmp_path / "geo",
        output_path=tmp_path / "embeddings.npz",
        embedder=embedder,
    )
    assert second["reused"] is True
    assert calls == 1
