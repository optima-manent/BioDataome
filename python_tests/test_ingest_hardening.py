from pathlib import Path

import pytest
from cskl_pipeline.normalize import ArchiveSafetyError
from cskl_pipeline.scale import ingest
from cskl_pipeline.scale.store import Store


def _ingested_csv(tmp_path: Path) -> tuple[Store, ingest.IngestItem]:
    store = Store(tmp_path / "store")
    store.write_probes("GPL570", ["probe-1", "probe-2", "probe-3"])
    source = tmp_path / "GSE42.csv"
    source.write_text(
        "probe,s1,s2,s3,s4\n"
        "probe-1,1,2,4,8\n"
        "probe-2,2,5,3,9\n"
        "probe-3,7,1,6,2\n",
        encoding="utf-8",
    )
    item = ingest.IngestItem(gse="GSE42", source=str(source), kind="csv")
    result = ingest.ingest_one(
        store,
        "GPL570",
        item,
        alpha=0.5,
        seed=17,
        input_orientation="features-by-samples",
        blas_threads=1,
    )
    assert result["status"] == "ok"
    return store, item


def test_raw_archive_rejection_is_quarantined_with_operator_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "store")
    store.write_probes("GPL570", ["probe-1"])
    source = tmp_path / "GSE9_RAW.tar"
    source.write_bytes(b"not a safe archive")

    def reject(*_args, **_kwargs):
        raise ArchiveSafetyError("archive contains a symbolic link")

    monkeypatch.setattr(ingest, "normalize_affy_scan", reject)
    result = ingest.ingest_one(
        store,
        "GPL570",
        ingest.IngestItem(gse="GSE9", source=str(source), kind="tar"),
        alpha=0.5,
    )

    assert result["status"] == "quarantine"
    assert result["reason"] == "archive_rejected"
    assert result["operator_required"] is True
    assert "clear this dataset quarantine" in result["recovery"]
    reason = (store.quarantine_dir("GSE9") / "reason.txt").read_text(encoding="utf-8")
    assert "symbolic link" in reason


def test_unchanged_csv_resume_uses_verified_manifest(tmp_path: Path) -> None:
    store, item = _ingested_csv(tmp_path)

    assert ingest._raw_ingest_cache_matches(
        store,
        "GPL570",
        item,
        alpha=0.5,
        seed=17,
        input_orientation="features-by-samples",
    )
    assert ingest.ingest_one(
        store,
        "GPL570",
        item,
        alpha=0.5,
        seed=17,
        input_orientation="features-by-samples",
        blas_threads=1,
    )["status"] == "skip"


@pytest.mark.parametrize("persist_expr", [False, True])
def test_failed_csv_refit_preserves_published_expression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_expr: bool,
) -> None:
    store, item = _ingested_csv(tmp_path)
    expr_path = store.expr_path(item.gse)
    published = expr_path.read_bytes()
    source_path = Path(item.source)
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace("probe-1,1,2,4,8", "probe-1,1,2,4,10"),
        encoding="utf-8",
    )

    def fail_fit(*_args, **_kwargs):
        raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(ingest, "fast_fit_signature", fail_fit)
    result = ingest.ingest_one(
        store,
        "GPL570",
        item,
        alpha=0.5,
        seed=17,
        persist_expr=persist_expr,
        blas_threads=1,
    )

    assert result["status"] == "quarantine"
    assert result["reason"] == "fit_failed"
    assert expr_path.read_bytes() == published
    assert not expr_path.with_suffix(expr_path.suffix + ".pending").exists()


@pytest.mark.parametrize("persist_expr", [False, True])
def test_csv_expression_change_waits_for_companion_artifact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_expr: bool,
) -> None:
    store, item = _ingested_csv(tmp_path)
    expr_path = store.expr_path(item.gse)
    published = expr_path.read_bytes()
    source_path = Path(item.source)
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace("probe-1,1,2,4,8", "probe-1,1,2,4,10"),
        encoding="utf-8",
    )
    original_write_json = ingest.store_mod.atomic_write_json

    def fail_manifest(path, payload):
        if Path(path).name == "ingest-manifest.json":
            raise OSError("simulated manifest publication failure")
        return original_write_json(path, payload)

    monkeypatch.setattr(ingest.store_mod, "atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="manifest publication failure"):
        ingest.ingest_one(
            store,
            "GPL570",
            item,
            alpha=0.5,
            seed=17,
            persist_expr=persist_expr,
            blas_threads=1,
        )

    assert expr_path.read_bytes() == published
    assert not expr_path.with_suffix(expr_path.suffix + ".pending").exists()


def test_csv_expression_gzip_is_deterministic_across_refits(tmp_path: Path) -> None:
    store, item = _ingested_csv(tmp_path)
    expr_path = store.expr_path(item.gse)
    first = expr_path.read_bytes()

    result = ingest.ingest_one(
        store,
        "GPL570",
        item,
        alpha=0.6,
        seed=17,
        blas_threads=1,
    )

    assert result["status"] == "ok"
    assert first[:2] == b"\x1f\x8b"
    assert int.from_bytes(first[4:8], "little") == 0
    assert expr_path.read_bytes() == first


@pytest.mark.parametrize(
    "mismatch",
    [
        "source_content",
        "source_identity",
        "alpha",
        "seed",
        "orientation",
        "qc_policy",
        "normalized_expr",
        "sample_hashes",
        "qc",
    ],
)
def test_csv_resume_rejects_stale_inputs_or_incomplete_artifacts(
    tmp_path: Path, mismatch: str
) -> None:
    store, item = _ingested_csv(tmp_path)
    alpha = 0.5
    seed = 17
    orientation = "features-by-samples"
    max_nonfinite_fraction = 0.01

    if mismatch == "source_content":
        Path(item.source).write_text(
            Path(item.source).read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif mismatch == "source_identity":
        moved = tmp_path / "same-content.csv"
        moved.write_bytes(Path(item.source).read_bytes())
        item = ingest.IngestItem(gse=item.gse, source=str(moved), kind="csv")
    elif mismatch == "alpha":
        alpha = 0.6
    elif mismatch == "seed":
        seed = 18
    elif mismatch == "orientation":
        orientation = "samples-by-features"
    elif mismatch == "qc_policy":
        max_nonfinite_fraction = 0.005
    elif mismatch == "normalized_expr":
        store.expr_path(item.gse).write_bytes(b"corrupt")
    elif mismatch == "sample_hashes":
        store.sample_hashes_path(item.gse).unlink()
    elif mismatch == "qc":
        store.qc_path(item.gse).unlink()

    assert not ingest._raw_ingest_cache_matches(
        store,
        "GPL570",
        item,
        alpha=alpha,
        seed=seed,
        input_orientation=orientation,
        max_nonfinite_fraction=max_nonfinite_fraction,
    )


def test_csv_resume_creates_normalized_matrix_when_persistence_is_requested(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    store.write_probes("GPL570", ["probe-1", "probe-2", "probe-3"])
    source = tmp_path / "GSE43.csv"
    source.write_text(
        "probe,s1,s2,s3,s4\n"
        "probe-1,1,2,4,8\n"
        "probe-2,2,5,3,9\n"
        "probe-3,7,1,6,2\n",
        encoding="utf-8",
    )
    item = ingest.IngestItem(gse="GSE43", source=str(source), kind="csv")

    first = ingest.ingest_one(
        store,
        "GPL570",
        item,
        alpha=0.5,
        seed=17,
        persist_expr=False,
        blas_threads=1,
    )
    assert first["status"] == "ok"
    assert not store.expr_path(item.gse).exists()

    second = ingest.ingest_one(
        store,
        "GPL570",
        item,
        alpha=0.5,
        seed=17,
        persist_expr=True,
        blas_threads=1,
    )
    assert second["status"] == "ok"
    assert store.expr_path(item.gse).is_file()
