from __future__ import annotations

from cskl_pipeline.scale.assemble import pair_overlap_evidence
from cskl_pipeline.scale.store import Store, atomic_write_json


def _write_hashes(store: Store, accession: str, values: list[str]) -> None:
    store.dataset_dir(accession).mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        store.sample_hashes_path(accession),
        {"n_samples": len(values), "hashes": values},
    )


def test_pair_overlap_retains_endpoints_and_classifies_multiset_containment(tmp_path):
    store = Store(tmp_path)
    _write_hashes(store, "GSE1", ["a", "a", "b", "c"])
    _write_hashes(store, "GSE2", ["a", "a", "b"])
    _write_hashes(store, "GSE3", ["z"])

    evidence = pair_overlap_evidence(store, ["GSE1", "GSE2", "GSE3"])

    assert set(evidence) == {("GSE1", "GSE2")}
    overlap = evidence[("GSE1", "GSE2")]
    assert overlap["shared_sample_count"] == 3
    assert overlap["overlap_coefficient"] == 1
    assert overlap["overlap_classification"] == "major"
    assert overlap["discovery_excluded"] is True
    assert overlap["edge_style"] == "dotted"
