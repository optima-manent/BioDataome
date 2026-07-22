from pathlib import Path

import pytest
from cskl_pipeline.normalize import ArchiveSafetyError
from cskl_pipeline.scale import ingest
from cskl_pipeline.scale.store import Store


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
