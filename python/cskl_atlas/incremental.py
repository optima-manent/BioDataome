"""Streaming raw C-SKL computation for incremental graph updates.

The iterators in this module never construct an ``N x N`` distance matrix.
Callers explicitly provide the K new signatures and N existing signatures, so
existing-existing pairs cannot be recomputed accidentally.  The vendored
``python/cskl.py`` implementation remains the numerical source of truth.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Iterator, Optional

import cskl as _cskl

__all__ = [
    "DatasetSignature",
    "DuplicateDatasetError",
    "IncompatibleSignatureError",
    "PairKind",
    "RawCSKLPair",
    "iter_cross_cskl_pairs",
    "iter_incremental_cskl_pairs",
    "iter_pair_batches",
    "stream_raw_cskl_updates",
]


class IncompatibleSignatureError(ValueError):
    """Raised before scoring when signatures do not share a C-SKL stratum."""


class DuplicateDatasetError(ValueError):
    """Raised when incremental and existing ID sets are not disjoint and unique."""


@dataclass(frozen=True, slots=True)
class DatasetSignature:
    """A PCA signature bound to a dataset and frozen feature universe."""

    dataset_id: str
    feature_universe: str
    signature: _cskl.PCASignature

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        feature_universe = str(self.feature_universe).strip()
        if not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if not feature_universe:
            raise ValueError("feature_universe must be a non-empty ID or hash")
        if not isinstance(self.signature, _cskl.PCASignature):
            raise TypeError("signature must be a cskl.PCASignature")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "feature_universe", feature_universe)

    @property
    def alpha(self) -> float:
        return float(self.signature.alpha)

    @property
    def n_features(self) -> int:
        return int(self.signature.n_features)


class PairKind(str, Enum):
    NEW_EXISTING = "new-existing"
    NEW_NEW = "new-new"


@dataclass(frozen=True, slots=True)
class RawCSKLPair:
    """One raw, unthresholded C-SKL update suitable for persistent streaming."""

    dataset_a: str
    dataset_b: str
    value: float
    feature_universe: str
    alpha: float
    kind: PairKind

    @property
    def cskl(self) -> float:
        return self.value


Metric = Callable[[_cskl.PCASignature, _cskl.PCASignature], float]


def _materialise_unique(
    records: Iterable[DatasetSignature],
    *,
    label: str,
) -> tuple[DatasetSignature, ...]:
    items = tuple(records)
    if not all(isinstance(item, DatasetSignature) for item in items):
        raise TypeError(f"{label} must contain DatasetSignature values")
    ids = [item.dataset_id for item in items]
    duplicate_ids = sorted(
        dataset_id
        for dataset_id, count in Counter(ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise DuplicateDatasetError(
            f"{label} contains duplicate dataset IDs: {', '.join(duplicate_ids)}"
        )
    return items


def _compatibility_problem(
    reference: DatasetSignature,
    candidate: DatasetSignature,
) -> Optional[str]:
    if candidate.feature_universe != reference.feature_universe:
        return (
            "feature universe differs "
            f"({reference.feature_universe!r} != {candidate.feature_universe!r})"
        )
    if candidate.n_features != reference.n_features:
        return f"n_features differs ({reference.n_features} != {candidate.n_features})"
    if not math.isclose(candidate.alpha, reference.alpha, rel_tol=1e-12, abs_tol=1e-12):
        return f"alpha differs ({reference.alpha} != {candidate.alpha})"

    reference_names = reference.signature.feature_names
    candidate_names = candidate.signature.feature_names
    if reference_names is not None and candidate_names is not None:
        if tuple(reference_names) != tuple(candidate_names):
            return "feature order differs despite a matching feature-universe ID"
    return None


def _prepare_new(
    new_signatures: Iterable[DatasetSignature],
) -> tuple[DatasetSignature, ...]:
    """Materialize and validate only the K newly admitted signatures."""

    new_items = _materialise_unique(new_signatures, label="new_signatures")
    if new_items:
        reference = new_items[0]
        for candidate in new_items[1:]:
            problem = _compatibility_problem(reference, candidate)
            if problem is not None:
                raise IncompatibleSignatureError(
                    f"{reference.dataset_id!r} and {candidate.dataset_id!r} are "
                    f"not C-SKL compatible: {problem}"
                )
    return new_items


def _validate_existing_candidate(
    candidate: object,
    *,
    reference: DatasetSignature | None,
    new_ids: set[str],
    seen_existing_ids: set[str],
) -> DatasetSignature:
    if not isinstance(candidate, DatasetSignature):
        raise TypeError("existing_signatures must contain DatasetSignature values")
    if candidate.dataset_id in new_ids:
        raise DuplicateDatasetError(
            "new and existing dataset IDs must be disjoint; already present: "
            + candidate.dataset_id
        )
    if candidate.dataset_id in seen_existing_ids:
        raise DuplicateDatasetError(
            "existing_signatures contains duplicate dataset IDs: "
            + candidate.dataset_id
        )
    if reference is not None:
        problem = _compatibility_problem(reference, candidate)
        if problem is not None:
            raise IncompatibleSignatureError(
                f"{reference.dataset_id!r} and {candidate.dataset_id!r} are "
                f"not C-SKL compatible: {problem}"
            )
    seen_existing_ids.add(candidate.dataset_id)
    return candidate


def _iter_validated_existing(
    existing_signatures: Iterable[DatasetSignature],
    new_items: tuple[DatasetSignature, ...],
) -> Iterator[DatasetSignature]:
    """Validate existing signatures without retaining their PCA payloads.

    A replayable ``Sequence`` is fully preflighted before its first item is
    yielded, preserving the fail-before-score behavior used by callers that can
    afford an in-memory index. Other iterables are validated and yielded one at
    a time. The latter path stores only dataset IDs, so an incompatibility or
    duplicate discovered later may be raised after earlier pair rows have
    already been emitted.
    """

    new_ids = {item.dataset_id for item in new_items}
    seen_existing_ids: set[str] = set()
    reference: DatasetSignature | None = new_items[0] if new_items else None

    if isinstance(existing_signatures, SequenceABC):
        for candidate in existing_signatures:
            validated = _validate_existing_candidate(
                candidate,
                reference=reference,
                new_ids=new_ids,
                seen_existing_ids=seen_existing_ids,
            )
            if reference is None:
                reference = validated
        yield from existing_signatures
        return

    for candidate in existing_signatures:
        validated = _validate_existing_candidate(
            candidate,
            reference=reference,
            new_ids=new_ids,
            seen_existing_ids=seen_existing_ids,
        )
        if reference is None:
            reference = validated
        yield validated


def _score_pair(
    left: DatasetSignature,
    right: DatasetSignature,
    *,
    kind: PairKind,
    metric: Metric,
) -> RawCSKLPair:
    value = float(metric(left.signature, right.signature))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"raw C-SKL for {left.dataset_id!r}/{right.dataset_id!r} "
            f"must be finite and non-negative; got {value!r}"
        )
    return RawCSKLPair(
        dataset_a=left.dataset_id,
        dataset_b=right.dataset_id,
        value=value,
        feature_universe=left.feature_universe,
        alpha=left.alpha,
        kind=kind,
    )


def iter_cross_cskl_pairs(
    new_signatures: Iterable[DatasetSignature],
    existing_signatures: Iterable[DatasetSignature],
    *,
    metric: Optional[Metric] = None,
) -> Iterator[RawCSKLPair]:
    """Yield exactly K x N new-existing scores, one pair at a time.

    The K new signatures are materialized. If ``existing_signatures`` is a
    replayable sequence, compatibility is validated across the complete input
    before the first score is emitted. A one-shot iterable is instead consumed
    and validated one signature at a time, so a later invalid item can raise
    after earlier pair rows were emitted. No existing-existing or new-new pair
    is evaluated.
    """

    new_items = _prepare_new(new_signatures)
    metric_fn = metric or _cskl.cskl
    for existing_item in _iter_validated_existing(existing_signatures, new_items):
        for new_item in new_items:
            yield _score_pair(
                new_item,
                existing_item,
                kind=PairKind.NEW_EXISTING,
                metric=metric_fn,
            )


def iter_incremental_cskl_pairs(
    new_signatures: Iterable[DatasetSignature],
    existing_signatures: Iterable[DatasetSignature],
    *,
    include_new_new: bool = True,
    metric: Optional[Metric] = None,
) -> Iterator[RawCSKLPair]:
    """Yield the complete delta for a batch without any old-old work.

    The delta is K x N new-existing pairs and, by default, the K(K-1)/2
    undirected pairs inside the new batch.  Set ``include_new_new=False`` for
    the strict K x N operation exposed by :func:`iter_cross_cskl_pairs`.

    Replayable existing sequences are fully preflighted. For a one-shot
    existing iterable, a compatibility or duplicate-ID error discovered later
    is raised after any earlier new-existing rows have been emitted; new-new
    rows are emitted only after that iterable is exhausted successfully.
    """

    new_items = _prepare_new(new_signatures)
    metric_fn = metric or _cskl.cskl

    for existing_item in _iter_validated_existing(existing_signatures, new_items):
        for new_item in new_items:
            yield _score_pair(
                new_item,
                existing_item,
                kind=PairKind.NEW_EXISTING,
                metric=metric_fn,
            )

    if include_new_new:
        for left_index, left in enumerate(new_items):
            for right in new_items[left_index + 1 :]:
                yield _score_pair(
                    left,
                    right,
                    kind=PairKind.NEW_NEW,
                    metric=metric_fn,
                )


def iter_pair_batches(
    pairs: Iterable[RawCSKLPair],
    *,
    batch_size: int = 1_000,
) -> Iterator[tuple[RawCSKLPair, ...]]:
    """Bound the number of pair results held before a persistence flush."""

    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batch: list[RawCSKLPair] = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def stream_raw_cskl_updates(
    new_signatures: Iterable[DatasetSignature],
    existing_signatures: Iterable[DatasetSignature],
    *,
    include_new_new: bool = True,
    batch_size: int = 1_000,
    metric: Optional[Metric] = None,
) -> Iterator[tuple[RawCSKLPair, ...]]:
    """Yield bounded result batches while retaining streamed-input semantics.

    With a one-shot existing iterable, completed batches may already have been
    yielded when a later existing signature fails validation. Persistence
    callers should therefore use idempotent pair keys when retrying the update.
    """

    return iter_pair_batches(
        iter_incremental_cskl_pairs(
            new_signatures,
            existing_signatures,
            include_new_new=include_new_new,
            metric=metric,
        ),
        batch_size=batch_size,
    )
