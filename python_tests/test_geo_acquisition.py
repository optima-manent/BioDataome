from __future__ import annotations

import io
import json
import tarfile

import httpx
from cskl_atlas.catalog import Catalog
from cskl_atlas.cli import build_parser
from cskl_atlas.geo_acquisition import (
    GeoAccessionSoftClient,
    NcbiEutilsClient,
    NcbiSettings,
    download_geo_raw_tar,
    sync_geo_metadata,
)


def _current_version(catalog: Catalog) -> str:
    dataset_uid, version_id = catalog.register_dataset_version(
        accession="GSE9",
        platform="GPL570",
        cohort="series",
        source_revision="test",
        source_hash="1" * 64,
        normalized_hash="2" * 64,
        signature_hash="3" * 64,
        feature_hash="4" * 40,
        config_hash="5" * 64,
        sample_count=1,
        metadata={},
    )
    assert dataset_uid
    for kind, checksum in (("normalized_matrix", "2" * 64), ("pca_signature", "3" * 64)):
        catalog.record_artifact(
            artifact_id=f"{kind}-test",
            kind=kind,
            uri=f"/test/{kind}",
            checksum=checksum,
            dependency_hash=("6" if kind == "normalized_matrix" else "7") * 64,
            manifest={},
            dataset_version_id=version_id,
        )
    catalog.promote_dataset_version(version_id)
    return version_id


def test_geo_sync_batches_caches_and_catalogs_official_summary(tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint = request.url.path.rsplit("/", 1)[-1]
        calls.append(endpoint)
        if endpoint == "esearch.fcgi":
            return httpx.Response(200, json={"esearchresult": {"idlist": ["200000009"]}})
        return httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["200000009"],
                    "200000009": {
                        "accession": "GSE9",
                        "title": "A title",
                        "summary": "A summary",
                        "taxon": "Homo sapiens",
                        "gpl": "570",
                        "gdstype": "Expression profiling by array",
                        "pdat": "2020/01/01",
                        "suppfile": "CEL",
                        "samples": [{"accession": "GSM1", "title": "sample"}],
                        "pubmedids": ["123"],
                    },
                }
            },
        )

    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    version_id = _current_version(catalog)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = NcbiSettings(allow_missing_email=True)
    first = sync_geo_metadata(
        catalog,
        accessions=["GSE9"],
        output_directory=tmp_path / "geo",
        settings=settings,
        http_client=client,
    )
    assert first["fetched"] == 1
    assert first["failed"] == 0
    assert calls == ["esearch.fcgi", "esummary.fcgi"]
    record = json.loads((tmp_path / "geo" / "records" / "GSE9.json").read_text())
    assert record["sample_accessions"] == ["GSM1"]
    assert record["family_soft_url"].endswith("/GSEnnn/GSE9/soft/GSE9_family.soft.gz")
    with catalog.reader() as connection:
        artifact = connection.execute(
            "SELECT * FROM artifacts WHERE kind='geo_metadata'"
        ).fetchone()
    assert artifact["dataset_version_id"] == version_id

    second = sync_geo_metadata(
        catalog,
        accessions=["GSE9"],
        output_directory=tmp_path / "geo",
        settings=settings,
        http_client=client,
    )
    assert second["cached"] == 1
    assert second["fetched"] == 0
    assert calls == ["esearch.fcgi", "esummary.fcgi"]


def test_scheduled_ncbi_client_requires_contact_email():
    try:
        NcbiSettings()
    except ValueError as exc:
        assert "NCBI_EMAIL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("scheduled NCBI settings accepted a missing contact")


def test_geo_sync_cli_defaults_to_identity_free_soft_metadata():
    args = build_parser().parse_args(["sync-geo"])
    assert args.metadata_source == "geo-soft"


def test_raw_download_cli_exposes_revision_and_checksum_recovery_controls():
    digest = "a" * 64
    args = build_parser().parse_args(
        [
            "download-geo-raw",
            "--accession",
            "GSE40082",
            "--url",
            "https://example.invalid/GSE40082_RAW.tar",
            "--source-revision",
            "etag:revision-2",
            "--expected-sha256",
            digest,
            "--force",
        ]
    )

    assert args.source_revision == "etag:revision-2"
    assert args.expected_sha256 == digest
    assert args.force is True


def test_geo_sync_can_use_documented_identity_free_soft_accession_view(tmp_path):
    soft = """^SERIES = GSE9
!Series_title = A title
!Series_geo_accession = GSE9
!Series_status = Public on Jan 01 2020
!Series_pubmed_id = 123
!Series_summary = A summary
!Series_type = Expression profiling by array
!Series_sample_id = GSM1
!Series_sample_id = GSM2
!Series_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE9/suppl/GSE9_RAW.tar
!Series_platform_id = GPL570
"""
    samples = """^SAMPLE = GSM1
!Sample_source_name_ch1 = Peripheral blood
!Sample_organism_ch1 = Homo sapiens
!Sample_characteristics_ch1 = disease: asthma
^SAMPLE = GSM2
!Sample_source_name_ch1 = Peripheral blood
!Sample_organism_ch1 = Homo sapiens
!Sample_treatment_protocol_ch1 = vehicle control
"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path.endswith("/geo/query/acc.cgi")
        assert request.url.params["form"] == "text"
        assert request.url.params["view"] == "brief"
        return httpx.Response(
            200,
            request=request,
            text=soft if request.url.params["targ"] == "self" else samples,
        )

    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    _current_version(catalog)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = sync_geo_metadata(
        catalog,
        accessions=["GSE9"],
        output_directory=tmp_path / "geo",
        settings=NcbiSettings(allow_missing_email=True),
        metadata_source="geo-soft",
        http_client=client,
    )

    assert report["fetched"] == 1
    assert report["failed"] == 0
    assert calls == 2
    record = json.loads((tmp_path / "geo" / "records" / "GSE9.json").read_text())
    assert record["schema"] == "ncbi-geo-soft-series-samples-v1"
    assert record["platforms"] == ["GPL570"]
    assert record["sample_accessions"] == ["GSM1", "GSM2"]
    assert record["organisms"] == ["Homo sapiens"]
    assert record["sample_characteristics"]["source_name"] == ["Peripheral blood"]
    assert [sample["accession"] for sample in record["sample_records"]] == ["GSM1", "GSM2"]
    assert record["sample_metadata_scope"] == "series_union"
    assert record["raw_tar_url"].startswith("https://")


def test_geo_soft_metadata_is_filtered_to_matrix_cohort(tmp_path):
    soft = """^SERIES = GSE9
!Series_title = Cohort-filter test
!Series_sample_id = GSM1
!Series_sample_id = GSM2
!Series_sample_id = GSM3
!Series_platform_id = GPL570
"""
    samples = """^SAMPLE = GSM1
!Sample_source_name_ch1 = Blood
!Sample_characteristics_ch1 = disease: asthma
^SAMPLE = GSM2
!Sample_source_name_ch1 = Lung
!Sample_characteristics_ch1 = disease: COPD
^SAMPLE = GSM3
!Sample_source_name_ch1 = Breast
!Sample_characteristics_ch1 = disease: breast cancer
"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            text=soft if request.url.params["targ"] == "self" else samples,
        )

    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    version_id = _current_version(catalog)
    catalog.replace_samples(
        version_id,
        [
            {"gsm": "GSM1", "expression_hash": "a" * 64},
            {"gsm": "GSM2", "expression_hash": "b" * 64},
        ],
    )
    report = sync_geo_metadata(
        catalog,
        accessions=["GSE9"],
        output_directory=tmp_path / "geo",
        settings=NcbiSettings(allow_missing_email=True),
        metadata_source="geo-soft",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert report["operator_required"] is False
    record = json.loads((tmp_path / "geo" / "records" / "GSE9.json").read_text())
    assert record["sample_metadata_scope"] == "matrix_cohort"
    assert [sample["accession"] for sample in record["sample_records"]] == ["GSM1", "GSM2"]
    assert record["sample_metadata_coverage"]["excluded_series_only_sample_count"] == 1
    assert "Breast" not in record["sample_characteristics"]["source_name"]


def test_geo_soft_client_rejects_wrong_series_identity():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request, text="^SERIES = GSE10\n")
        )
    )
    with GeoAccessionSoftClient(
        NcbiSettings(allow_missing_email=True, max_attempts=1),
        http_client=client,
        sleeper=lambda _: None,
        clock=lambda: 1.0,
    ) as geo:
        try:
            geo.series_summary("GSE9")
        except Exception as exc:
            assert "instead of GSE9" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("mismatched GEO response was accepted")


def test_geo_soft_client_keeps_series_when_sample_view_exceeds_safety_cap():
    series = """^SERIES = GSE9
!Series_title = Safe series
!Series_geo_accession = GSE9
!Series_sample_id = GSM1
!Series_platform_id = GPL570
"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["targ"] == "self":
            return httpx.Response(200, request=request, text=series)
        return httpx.Response(
            200,
            request=request,
            content=b"oversized",
            headers={"Content-Length": "2048"},
        )

    with GeoAccessionSoftClient(
        NcbiSettings(allow_missing_email=True, max_soft_response_bytes=1024),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _: None,
        clock=lambda: 1.0,
    ) as geo:
        record = geo.series_summary("GSE9")

    assert record["title"] == "Safe series"
    assert record["sample_metadata_status"] == "unavailable"
    assert "safety cap" in record["sample_metadata_error"]


def test_ncbi_client_retries_rate_limit_then_marks_operator_gate():
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "0"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = NcbiSettings(allow_missing_email=True, max_attempts=3)
    with NcbiEutilsClient(
        settings,
        http_client=client,
        sleeper=lambda _: None,
        clock=lambda: 1.0,
    ) as ncbi:
        try:
            ncbi.request_json("esearch", {"db": "gds"})
        except Exception as exc:
            assert getattr(exc, "operator_required", False) is True
        else:  # pragma: no cover
            raise AssertionError("rate-limit exhaustion did not fail")
    assert calls == 3


def test_geo_raw_download_resumes_validates_and_replays_without_network(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        payload = b"CEL test payload"
        info = tarfile.TarInfo("nested/GSM1.CEL.gz")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    raw_tar = buffer.getvalue()
    root = tmp_path / "raw" / "GSE1"
    root.mkdir(parents=True)
    split = len(raw_tar) // 3
    (root / "GSE1_RAW.tar.part").write_bytes(raw_tar[:split])
    (root / "raw-partial.json").write_text(
        json.dumps(
            {
                "schema": "geo-raw-partial-v1",
                "accession": "GSE1",
                "source_url": "https://ftp.ncbi.nlm.nih.gov/test/GSE1_RAW.tar",
                "source_revision": "",
                "etag": '"revision-1"',
                "last_modified": "",
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Range"] == f"bytes={split}-"
        assert request.headers["If-Range"] == '"revision-1"'
        return httpx.Response(
            206,
            request=request,
            content=raw_tar[split:],
            headers={
                "Content-Length": str(len(raw_tar) - split),
                "Content-Range": f"bytes {split}-{len(raw_tar) - 1}/{len(raw_tar)}",
                "ETag": '"revision-1"',
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = download_geo_raw_tar(
        accession="GSE1",
        raw_tar_url="https://ftp.ncbi.nlm.nih.gov/test/GSE1_RAW.tar",
        output_directory=tmp_path / "raw",
        reserve_bytes=0,
        http_client=client,
    )
    assert first["status"] == "verified"
    assert first["cel_member_count"] == 1
    assert (root / "GSE1_RAW.tar").read_bytes() == raw_tar
    second = download_geo_raw_tar(
        accession="GSE1",
        raw_tar_url="https://ftp.ncbi.nlm.nih.gov/test/GSE1_RAW.tar",
        output_directory=tmp_path / "raw",
        reserve_bytes=0,
        http_client=client,
    )
    assert second["status"] == "cached"
    assert calls == 1


def test_geo_raw_download_rejects_unsafe_archive_as_operator_gate(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../GSM1.CEL")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    raw_tar = buffer.getvalue()

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request, content=raw_tar)
        )
    )
    try:
        download_geo_raw_tar(
            accession="GSE2",
            raw_tar_url="https://ftp.ncbi.nlm.nih.gov/test/GSE2_RAW.tar",
            output_directory=tmp_path / "raw",
            reserve_bytes=0,
            http_client=client,
        )
    except Exception as exc:
        assert getattr(exc, "operator_required", False) is True
        assert "unsafe" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsafe RAW archive was accepted")
    assert not (tmp_path / "raw" / "GSE2" / "GSE2_RAW.tar").exists()


def test_geo_raw_cache_requires_matching_source_revision(tmp_path):
    def tar_bytes(payload: bytes) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo("GSM1.CEL.gz")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return buffer.getvalue()

    revisions = [tar_bytes(b"one"), tar_bytes(b"two")]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        payload = revisions[calls]
        calls += 1
        return httpx.Response(200, request=request, content=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = download_geo_raw_tar(
        accession="GSE3",
        raw_tar_url="https://ftp.ncbi.nlm.nih.gov/test/GSE3_RAW.tar",
        source_revision="geo-revision-1",
        output_directory=tmp_path / "raw",
        reserve_bytes=0,
        http_client=client,
    )
    cached = download_geo_raw_tar(
        accession="GSE3",
        raw_tar_url="https://ftp.ncbi.nlm.nih.gov/test/GSE3_RAW.tar",
        source_revision="geo-revision-1",
        output_directory=tmp_path / "raw",
        reserve_bytes=0,
        http_client=client,
    )
    second = download_geo_raw_tar(
        accession="GSE3",
        raw_tar_url="https://ftp.ncbi.nlm.nih.gov/test/GSE3_RAW.tar",
        source_revision="geo-revision-2",
        output_directory=tmp_path / "raw",
        reserve_bytes=0,
        http_client=client,
    )

    assert first["sha256"] != second["sha256"]
    assert cached["status"] == "cached"
    assert second["source_revision"] == "geo-revision-2"
    assert calls == 2


def test_geo_raw_unknown_partial_is_restarted_instead_of_mixed(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        payload = b"current"
        info = tarfile.TarInfo("GSM1.CEL")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    raw_tar = buffer.getvalue()
    root = tmp_path / "raw" / "GSE4"
    root.mkdir(parents=True)
    (root / "GSE4_RAW.tar.part").write_bytes(b"unbound-old-revision")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "Range" not in request.headers
        return httpx.Response(200, request=request, content=raw_tar)

    result = download_geo_raw_tar(
        accession="GSE4",
        raw_tar_url="https://ftp.ncbi.nlm.nih.gov/test/GSE4_RAW.tar",
        output_directory=tmp_path / "raw",
        reserve_bytes=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result["status"] == "verified"
    assert (root / "GSE4_RAW.tar").read_bytes() == raw_tar
