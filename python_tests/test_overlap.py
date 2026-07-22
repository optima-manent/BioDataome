from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from cskl_atlas.overlap import (  # noqa: E402
    DatasetSampleIndex,
    OverlapClass,
    OverlapPolicy,
    compute_overlap,
    hash_expression_profiles,
)


def _index(dataset_id: str, gsms, hashes=None) -> DatasetSampleIndex:
    return DatasetSampleIndex.from_columns(
        dataset_id,
        gsm_ids=gsms,
        expression_hashes=hashes,
    )


def test_exact_overlap_deduplicates_gsm_and_hash_evidence() -> None:
    left = _index("GSE-A", ["gsm1", "GSM2"], ["aa", "bb"])
    right = _index("GSE-B", ["GSM2", "GSM1"], ["bb", "aa"])

    evidence = compute_overlap(left, right)

    assert evidence.shared_sample_count == 2
    assert evidence.shared_gsm_ids == ("GSM1", "GSM2")
    assert evidence.shared_expression_hashes == ("aa", "bb")
    assert evidence.matched_by_both_count == 2
    assert evidence.left_fraction == pytest.approx(1.0)
    assert evidence.right_fraction == pytest.approx(1.0)
    assert evidence.jaccard == pytest.approx(1.0)
    assert evidence.overlap_coefficient == pytest.approx(1.0)
    assert evidence.classification is OverlapClass.EXACT
    assert evidence.exclude_from_discovery is True


def test_containment_is_major_using_the_smaller_endpoint() -> None:
    left = _index("small", ["GSM1", "GSM2"])
    right = _index("large", ["GSM0", "GSM1", "GSM2", "GSM3"])

    evidence = compute_overlap(left, right)

    assert evidence.shared_sample_count == 2
    assert evidence.left_fraction == pytest.approx(1.0)
    assert evidence.right_fraction == pytest.approx(0.5)
    assert evidence.jaccard == pytest.approx(0.5)
    assert evidence.overlap_coefficient == pytest.approx(1.0)
    assert evidence.classification is OverlapClass.MAJOR
    assert evidence.exclude_from_discovery is True


def test_minor_and_none_pairs_are_retained_for_discovery_by_default() -> None:
    left = _index("left", ["GSM1", "GSM2", "GSM3", "GSM4"])
    minor = _index("minor", ["GSM1", "GSM5", "GSM6", "GSM7"])
    none = _index("none", ["GSM8", "GSM9"])

    minor_evidence = compute_overlap(left, minor)
    none_evidence = compute_overlap(left, none)

    assert minor_evidence.classification is OverlapClass.MINOR
    assert minor_evidence.overlap_coefficient == pytest.approx(0.25)
    assert minor_evidence.exclude_from_discovery is False
    assert none_evidence.classification is OverlapClass.NONE
    assert none_evidence.shared_sample_count == 0
    assert none_evidence.exclude_from_discovery is False


def test_expression_hash_finds_overlap_when_gsm_metadata_differs() -> None:
    left = _index("left", ["GSM1", None], ["hash-a", "hash-b"])
    right = _index("right", ["GSM9", None], ["hash-a", "hash-z"])

    evidence = compute_overlap(
        left,
        right,
        policy=OverlapPolicy(major_overlap_coefficient=0.75),
    )

    assert evidence.shared_gsm_count == 0
    assert evidence.shared_expression_hashes == ("hash-a",)
    assert evidence.matched_by_expression_hash_count == 1
    assert evidence.shared_sample_count == 1
    assert evidence.classification is OverlapClass.MINOR


def test_maximum_matching_does_not_undercount_ambiguous_identifiers() -> None:
    # A greedy L1->R1 choice would strand L2.  The augmenting path must rematch
    # L1 to R2 and preserve both physical sample matches.
    left = _index("left", ["GSM-A", "GSM-A"], ["hash-x", "hash-y"])
    right = _index("right", ["GSM-A", "GSM-B"], ["hash-y", "hash-x"])

    evidence = compute_overlap(left, right)

    assert evidence.shared_sample_count == 2
    assert evidence.classification is OverlapClass.EXACT


def test_profile_hashes_match_legacy_rounding_and_validate_shape() -> None:
    matrix = np.array([[1.0, 2.0], [1.0 + 1e-12, 2.0 - 1e-12]])
    hashes = hash_expression_profiles(matrix, decimals=10)

    assert hashes[0] == hashes[1]
    with pytest.raises(ValueError, match="samples, features"):
        hash_expression_profiles(np.array([1.0, 2.0]))


def test_identifier_columns_must_be_aligned() -> None:
    with pytest.raises(ValueError, match="sample-aligned"):
        DatasetSampleIndex.from_columns(
            "broken",
            gsm_ids=["GSM1", "GSM2"],
            expression_hashes=["one"],
        )
