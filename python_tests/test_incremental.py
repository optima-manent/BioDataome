from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_ROOT))

import cskl  # noqa: E402
from cskl_atlas.incremental import (  # noqa: E402
    DatasetSignature,
    DuplicateDatasetError,
    IncompatibleSignatureError,
    PairKind,
    iter_cross_cskl_pairs,
    iter_incremental_cskl_pairs,
    stream_raw_cskl_updates,
)


def _signature(
    dataset_id: str,
    axis: int,
    *,
    universe: str = "GPL570:probe-order-v1",
    alpha: float = 0.5,
    feature_names=("p1", "p2", "p3"),
) -> DatasetSignature:
    loadings = np.zeros((3, 1), dtype=float)
    loadings[axis, 0] = 1.0
    signature = cskl.PCASignature(
        P=loadings,
        lam=np.array([alpha * 3.0]),
        n_features=3,
        m_samples=8,
        alpha=alpha,
        feature_names=list(feature_names),
    )
    return DatasetSignature(dataset_id, universe, signature)


def test_cross_iterator_is_exactly_k_by_n_and_never_scores_old_old() -> None:
    new = [_signature("new-1", 0), _signature("new-2", 1)]
    existing = [_signature("old-1", 0), _signature("old-2", 1), _signature("old-3", 2)]
    calls: list[tuple[int, int]] = []

    def tracked(left, right):
        calls.append((id(left), id(right)))
        return cskl.cskl(left, right)

    pairs = list(iter_cross_cskl_pairs(new, existing, metric=tracked))

    assert len(pairs) == 2 * 3
    assert len(calls) == 2 * 3
    assert {(pair.dataset_a, pair.dataset_b) for pair in pairs} == {
        (new_item.dataset_id, old_item.dataset_id)
        for new_item in new
        for old_item in existing
    }
    assert all(pair.kind is PairKind.NEW_EXISTING for pair in pairs)
    assert not any(
        pair.dataset_a.startswith("old-") and pair.dataset_b.startswith("old-")
        for pair in pairs
    )


def test_complete_delta_adds_new_new_pairs_but_no_old_old_pairs() -> None:
    new = [_signature("new-1", 0), _signature("new-2", 1), _signature("new-3", 2)]
    existing = [_signature("old-1", 0), _signature("old-2", 1)]

    pairs = list(iter_incremental_cskl_pairs(new, existing))

    assert len(pairs) == 3 * 2 + 3
    assert sum(pair.kind is PairKind.NEW_EXISTING for pair in pairs) == 6
    assert sum(pair.kind is PairKind.NEW_NEW for pair in pairs) == 3
    assert not any(
        pair.dataset_a.startswith("old-") and pair.dataset_b.startswith("old-")
        for pair in pairs
    )


def test_streaming_batches_bound_the_materialized_pair_results() -> None:
    new = [_signature("new-1", 0), _signature("new-2", 1)]
    existing = [_signature(f"old-{index}", index % 3) for index in range(5)]

    batches = list(
        stream_raw_cskl_updates(
            new,
            existing,
            include_new_new=False,
            batch_size=3,
        )
    )

    assert [len(batch) for batch in batches] == [3, 3, 3, 1]
    assert sum(map(len, batches)) == 10
    assert all(not isinstance(batch, np.ndarray) for batch in batches)


def test_one_shot_existing_iterable_is_consumed_one_signature_at_a_time() -> None:
    new = [_signature("new-1", 0), _signature("new-2", 1)]
    consumed: list[str] = []

    def existing_signatures():
        for index in range(3):
            dataset_id = f"old-{index}"
            consumed.append(dataset_id)
            yield _signature(dataset_id, index % 3)

    pairs = iter_cross_cskl_pairs(new, existing_signatures())

    first = next(pairs)
    assert consumed == ["old-0"]
    assert (first.dataset_a, first.dataset_b) == ("new-1", "old-0")

    second = next(pairs)
    assert consumed == ["old-0"]
    assert (second.dataset_a, second.dataset_b) == ("new-2", "old-0")

    third = next(pairs)
    assert consumed == ["old-0", "old-1"]
    assert (third.dataset_a, third.dataset_b) == ("new-1", "old-1")


def test_one_shot_existing_iterable_documents_partial_emission_on_late_error() -> None:
    new = [_signature("new-1", 0), _signature("new-2", 1)]
    metric_calls = 0

    def tracked(left, right):
        nonlocal metric_calls
        metric_calls += 1
        return cskl.cskl(left, right)

    def existing_signatures():
        yield _signature("old-ok", 2)
        yield _signature("old-incompatible", 0, universe="GPL96:v1")

    pairs = iter_cross_cskl_pairs(new, existing_signatures(), metric=tracked)
    assert [next(pairs).dataset_b, next(pairs).dataset_b] == ["old-ok", "old-ok"]
    with pytest.raises(IncompatibleSignatureError, match="feature universe"):
        next(pairs)
    assert metric_calls == 2


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (_signature("other-platform", 0, universe="GPL96:v1"), "feature universe"),
        (_signature("other-alpha", 0, alpha=0.6), "alpha differs"),
        (
            _signature("other-order", 0, feature_names=("p2", "p1", "p3")),
            "feature order differs",
        ),
    ],
)
def test_incompatible_inputs_fail_before_any_score(candidate, message) -> None:
    calls = 0

    def tracked(left, right):
        nonlocal calls
        calls += 1
        return cskl.cskl(left, right)

    iterator = iter_cross_cskl_pairs(
        [_signature("new", 0)],
        [_signature("old", 1), candidate],
        metric=tracked,
    )
    with pytest.raises(IncompatibleSignatureError, match=message):
        list(iterator)
    assert calls == 0


def test_reused_dataset_id_is_rejected_to_prevent_self_recomputation() -> None:
    with pytest.raises(DuplicateDatasetError, match="already present"):
        list(
            iter_cross_cskl_pairs(
                [_signature("same", 0)],
                [_signature("same", 1)],
            )
        )


def test_pair_values_delegate_to_vendored_cskl() -> None:
    new = _signature("new", 0)
    old = _signature("old", 1)

    pair = next(iter_cross_cskl_pairs([new], [old]))

    assert pair.value == pytest.approx(cskl.cskl(new.signature, old.signature))
    assert pair.feature_universe == new.feature_universe
    assert pair.alpha == pytest.approx(0.5)
