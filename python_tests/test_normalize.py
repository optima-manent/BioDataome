import io
import json
import tarfile
from pathlib import Path

import pytest
from cskl_pipeline import normalize
from cskl_pipeline.normalize import (
    ArchiveSafetyError,
    ArchiveSafetyLimits,
    extract_tar,
    normalization_cache_matches,
    scan_log_diagnostics,
)


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for info, payload in members:
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _regular(name: str, payload: bytes = b"CEL") -> tuple[tarfile.TarInfo, bytes]:
    return tarfile.TarInfo(name), payload


def test_scan_log_diagnostics_records_convergence_warning(tmp_path: Path) -> None:
    log_path = tmp_path / "scan_normalize_affy.log"
    log_path.write_text(
        "Processing GSM1.CEL\n"
        "Warning: convergence limit reached after maximum iterations\n"
        "Wrote: expr.tsv.gz\n",
        encoding="utf-8",
    )

    diagnostics = scan_log_diagnostics(log_path)

    assert diagnostics["status"] == "warning"
    assert diagnostics["convergence_warning_count"] == 1
    assert diagnostics["log_present"] is True
    assert diagnostics["warning_lines"] == [
        "Warning: convergence limit reached after maximum iterations"
    ]


def test_scan_log_diagnostics_reports_missing_log(tmp_path: Path) -> None:
    diagnostics = scan_log_diagnostics(tmp_path / "missing.log")

    assert diagnostics["status"] == "not_recorded"
    assert diagnostics["log_present"] is False
    assert diagnostics["convergence_warning_count"] == 0


def test_extract_tar_rejects_traversal_and_never_writes_outside_destination(tmp_path: Path) -> None:
    archive = tmp_path / "GSE1_RAW.tar"
    _write_tar(archive, [_regular("../escaped.CEL")])

    with pytest.raises(ArchiveSafetyError, match="non-relative"):
        extract_tar(archive, tmp_path / "extracted")

    assert not (tmp_path / "escaped.CEL").exists()
    assert not (tmp_path / "extracted").exists()


def test_extract_tar_rejects_links_devices_and_expansion_over_limit(tmp_path: Path) -> None:
    linked = tmp_path / "GSE2_RAW.tar"
    info = tarfile.TarInfo("linked.CEL")
    info.type = tarfile.SYMTYPE
    info.linkname = "outside"
    _write_tar(linked, [(info, b"")])
    with pytest.raises(ArchiveSafetyError, match="not a regular file"):
        extract_tar(linked, tmp_path / "linked")

    oversized = tmp_path / "GSE3_RAW.tar"
    _write_tar(oversized, [_regular("GSM1.CEL", b"12345")])
    limits = ArchiveSafetyLimits(
        max_members=5,
        max_member_bytes=4,
        max_expanded_bytes=10,
        minimum_free_bytes_after_extract=0,
    )
    with pytest.raises(ArchiveSafetyError, match="size is outside"):
        extract_tar(oversized, tmp_path / "oversized", limits=limits)


def test_safe_extract_replaces_destination_only_after_complete_validation(tmp_path: Path) -> None:
    archive = tmp_path / "GSE4_RAW.tar"
    _write_tar(archive, [_regular("nested/GSM1.CEL.gz", b"safe")])
    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "stale.txt").write_text("old", encoding="utf-8")

    inspection = extract_tar(
        archive,
        destination,
        limits=ArchiveSafetyLimits(minimum_free_bytes_after_extract=0),
    )

    assert inspection["cel_member_count"] == 1
    assert (destination / "nested" / "GSM1.CEL.gz").read_bytes() == b"safe"
    assert not (destination / "stale.txt").exists()


def test_normalization_cache_is_bound_to_raw_bytes_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "GSE5_RAW.tar"
    _write_tar(archive, [_regular("GSM1.CEL", b"first")])
    calls = 0

    def fake_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        Path(command[-1]).write_bytes(f"normalized-{calls}".encode())

    monkeypatch.setattr(normalize, "run", fake_run)
    output = normalize.normalize_affy_scan(
        archive,
        tmp_path / "work",
        "Rscript",
        archive_limits=ArchiveSafetyLimits(minimum_free_bytes_after_extract=0),
    )
    assert output.read_bytes() == b"normalized-1"
    assert normalization_cache_matches(archive, tmp_path / "work") is True
    normalize.normalize_affy_scan(archive, tmp_path / "work", "Rscript")
    assert calls == 1

    _write_tar(archive, [_regular("GSM1.CEL", b"revised")])
    normalize.normalize_affy_scan(
        archive,
        tmp_path / "work",
        "Rscript",
        archive_limits=ArchiveSafetyLimits(minimum_free_bytes_after_extract=0),
    )
    assert calls == 2
    assert output.read_bytes() == b"normalized-2"
    manifest = json.loads(
        (tmp_path / "work" / "GSE5" / "normalization-manifest.json").read_text()
    )
    assert manifest["schema"] == "cskl-scan-normalization-v2"
    assert manifest["archive"]["cel_member_count"] == 1

    previous = output.read_bytes()
    _write_tar(archive, [_regular("GSM1.CEL", b"third")])
    monkeypatch.setattr(
        normalize,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("R failed")),
    )
    with pytest.raises(RuntimeError, match="R failed"):
        normalize.normalize_affy_scan(
            archive,
            tmp_path / "work",
            "Rscript",
            archive_limits=ArchiveSafetyLimits(minimum_free_bytes_after_extract=0),
        )
    assert output.read_bytes() == previous
    assert normalization_cache_matches(archive, tmp_path / "work") is False
