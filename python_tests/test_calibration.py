from __future__ import annotations

import numpy as np
import pytest
from cskl_atlas.calibration import (
    NullProfileArtifact,
    OutOfCalibrationRange,
    calibrate_pairs,
    pair_pvalue,
)


def profile(version: str, signature: str = "sig") -> NullProfileArtifact:
    return NullProfileArtifact(
        version_id=version,
        signature_hash=signature,
        pool_hash="pool-v1",
        feature_hash="GPL570-probes-v1",
        alpha=0.5,
        bootstrap_count=100,
        grid=np.array([4, 10, 20]),
        mu=np.array([10.0, 12.0, 14.0]),
        sigma=np.array([1.0, 1.5, 2.0]),
    )


def test_exact_release_rejects_out_of_grid_updates():
    with pytest.raises(OutOfCalibrationRange):
        pair_pvalue(
            5.0, profile("a"), 30, profile("b"), 10, expected_pool_hash="pool-v1"
        )
    value = pair_pvalue(
        5.0,
        profile("a"),
        30,
        profile("b"),
        10,
        expected_pool_hash="pool-v1",
        allow_clamp=True,
    )
    assert 0 <= value <= 1


def test_profile_is_bound_to_signature_and_pool_hash():
    rows = [
        {
            "pair_id": "pair-1",
            "version_a": "a",
            "version_b": "b",
            "cskl": 8.0,
            "samples_a": 10,
            "samples_b": 20,
            "signature_a": "sig-a-new",
            "signature_b": "sig-b",
        }
    ]
    profiles = {"a": profile("a", "sig-a-old"), "b": profile("b", "sig-b")}
    with pytest.raises(ValueError, match="Stale null profile"):
        list(
            calibrate_pairs(
                rows, profiles.__getitem__, expected_pool_hash="pool-v1"
            )
        )


def test_profile_validation_rejects_mixed_feature_spaces():
    other = NullProfileArtifact(
        **{**profile("b").__dict__, "feature_hash": "another-platform"}
    )
    with pytest.raises(ValueError, match="feature universes"):
        pair_pvalue(5.0, profile("a"), 10, other, 10, expected_pool_hash="pool-v1")
