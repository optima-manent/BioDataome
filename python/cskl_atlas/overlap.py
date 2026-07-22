"""Pair-level shared-sample evidence for C-SKL graph edges.

The published C-SKL workflow removed every dataset that shared a profile with
another dataset.  Atlas retains every dataset and records overlap on the one
edge it can confound.  This module deliberately contains no dataset-filtering
function.

Sample identities may be established independently by a GEO sample accession
(GSM) or by a hash of the aligned expression profile.  When both identify the
same sample pair, maximum bipartite matching counts it once.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "DatasetSampleIndex",
    "OverlapClass",
    "OverlapEvidence",
    "OverlapPolicy",
    "SampleIdentity",
    "compute_overlap",
    "hash_expression_profile",
    "hash_expression_profiles",
]


def _normalise_gsm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalised = str(value).strip().upper()
    return normalised or None


def _normalise_expression_hash(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalised = str(value).strip().lower()
    return normalised or None


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    """Identifiers for one physical sample.

    Either identifier may be absent.  GSM identifiers are normalised to upper
    case and expression hashes to lower case so metadata formatting does not
    hide genuine matches.
    """

    gsm: Optional[str] = None
    expression_hash: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gsm", _normalise_gsm(self.gsm))
        object.__setattr__(
            self,
            "expression_hash",
            _normalise_expression_hash(self.expression_hash),
        )


@dataclass(frozen=True, slots=True)
class DatasetSampleIndex:
    """The ordered sample identities for one retained dataset."""

    dataset_id: str
    samples: tuple[SampleIdentity, ...]

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        if not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "samples", tuple(self.samples))
        if not all(isinstance(sample, SampleIdentity) for sample in self.samples):
            raise TypeError("samples must contain SampleIdentity values")

    @classmethod
    def from_columns(
        cls,
        dataset_id: str,
        *,
        gsm_ids: Optional[Sequence[Optional[str]]] = None,
        expression_hashes: Optional[Sequence[Optional[str]]] = None,
        sample_count: Optional[int] = None,
    ) -> "DatasetSampleIndex":
        """Build an index from sample-aligned identifier columns.

        Every supplied column must have exactly ``sample_count`` entries.  If
        ``sample_count`` is omitted it is inferred from the supplied columns.
        Requiring aligned columns avoids silently double-counting a sample that
        is known by both GSM and expression hash.
        """

        gsms = None if gsm_ids is None else tuple(gsm_ids)
        hashes = None if expression_hashes is None else tuple(expression_hashes)
        lengths = [len(values) for values in (gsms, hashes) if values is not None]

        if sample_count is None:
            if lengths and any(length != lengths[0] for length in lengths[1:]):
                raise ValueError("identifier columns must be sample-aligned")
            count = lengths[0] if lengths else 0
        else:
            count = int(sample_count)
            if count < 0:
                raise ValueError("sample_count must be >= 0")
            if any(length != count for length in lengths):
                raise ValueError(
                    "every supplied identifier column must have sample_count entries"
                )

        if gsms is None:
            gsms = (None,) * count
        if hashes is None:
            hashes = (None,) * count

        return cls(
            dataset_id=dataset_id,
            samples=tuple(
                SampleIdentity(gsm=gsm, expression_hash=expression_hash)
                for gsm, expression_hash in zip(gsms, hashes, strict=True)
            ),
        )

    @property
    def sample_count(self) -> int:
        return len(self.samples)


class OverlapClass(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    EXACT = "exact"


@dataclass(frozen=True, slots=True)
class OverlapPolicy:
    """Classification and discovery policy for a pair of retained datasets.

    ``major_overlap_coefficient`` is measured against the smaller endpoint.
    This correctly classifies containment (for example, 20/20 samples from one
    dataset appearing inside a 200-sample dataset) as major overlap.
    """

    major_overlap_coefficient: float = 0.5
    exclude_exact: bool = True
    exclude_major: bool = True
    exclude_minor: bool = False

    def __post_init__(self) -> None:
        threshold = float(self.major_overlap_coefficient)
        if not (0.0 < threshold <= 1.0):
            raise ValueError("major_overlap_coefficient must be in (0, 1]")
        object.__setattr__(self, "major_overlap_coefficient", threshold)

    def excludes(self, classification: OverlapClass) -> bool:
        return {
            OverlapClass.NONE: False,
            OverlapClass.MINOR: self.exclude_minor,
            OverlapClass.MAJOR: self.exclude_major,
            OverlapClass.EXACT: self.exclude_exact,
        }[classification]


@dataclass(frozen=True, slots=True)
class OverlapEvidence:
    """Serializable overlap evidence for one undirected dataset pair."""

    left_dataset_id: str
    right_dataset_id: str
    left_sample_count: int
    right_sample_count: int
    shared_sample_count: int
    shared_gsm_ids: tuple[str, ...]
    shared_expression_hashes: tuple[str, ...]
    matched_by_gsm_count: int
    matched_by_expression_hash_count: int
    matched_by_both_count: int
    left_fraction: float
    right_fraction: float
    jaccard: float
    overlap_coefficient: float
    classification: OverlapClass
    exclude_from_discovery: bool

    @property
    def has_overlap(self) -> bool:
        return self.shared_sample_count > 0

    @property
    def shared_gsm_count(self) -> int:
        return len(self.shared_gsm_ids)

    @property
    def shared_expression_hash_count(self) -> int:
        return len(self.shared_expression_hashes)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classification"] = self.classification.value
        payload["shared_gsm_count"] = self.shared_gsm_count
        payload["shared_expression_hash_count"] = self.shared_expression_hash_count
        payload["has_overlap"] = self.has_overlap
        return payload


def hash_expression_profile(
    values: Iterable[float],
    *,
    decimals: Optional[int] = 10,
) -> str:
    """Return the legacy-compatible SHA-1 identity of one expression profile.

    Values must already be aligned to the frozen feature universe.  Rounding to
    ten decimals matches the existing pipeline's shared-profile detector.
    """

    profile = np.asarray(tuple(values), dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError("an expression profile must be one-dimensional")
    if not np.all(np.isfinite(profile)):
        raise ValueError("expression profiles must contain only finite values")
    if decimals is not None:
        if int(decimals) < 0:
            raise ValueError("decimals must be >= 0 or None")
        profile = np.round(profile, decimals=int(decimals))
    contiguous = np.ascontiguousarray(profile, dtype=np.float64)
    return hashlib.sha1(contiguous.view(np.uint8)).hexdigest()


def hash_expression_profiles(
    matrix: np.ndarray,
    *,
    decimals: Optional[int] = 10,
) -> tuple[str, ...]:
    """Hash every sample row in a samples-by-features matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must have shape (samples, features)")
    return tuple(
        hash_expression_profile(row, decimals=decimals)
        for row in values
    )


def _candidate_adjacency(
    left: Sequence[SampleIdentity],
    right: Sequence[SampleIdentity],
) -> list[tuple[int, ...]]:
    right_by_gsm: dict[str, list[int]] = {}
    right_by_hash: dict[str, list[int]] = {}
    for index, sample in enumerate(right):
        if sample.gsm is not None:
            right_by_gsm.setdefault(sample.gsm, []).append(index)
        if sample.expression_hash is not None:
            right_by_hash.setdefault(sample.expression_hash, []).append(index)

    adjacency: list[tuple[int, ...]] = []
    for sample in left:
        candidates: set[int] = set()
        if sample.gsm is not None:
            candidates.update(right_by_gsm.get(sample.gsm, ()))
        if sample.expression_hash is not None:
            candidates.update(right_by_hash.get(sample.expression_hash, ()))
        adjacency.append(tuple(sorted(candidates)))
    return adjacency


def _maximum_matching(adjacency: Sequence[Sequence[int]], n_right: int) -> list[tuple[int, int]]:
    """Maximum cardinality matching without recursion or external graph state."""

    left_to_right = [-1] * len(adjacency)
    right_to_left = [-1] * n_right

    # Low-degree samples first makes the deterministic augmenting-path search
    # efficient for the duplicate-heavy overlap clusters seen in GEO.
    order = sorted(range(len(adjacency)), key=lambda i: (len(adjacency[i]), i))
    for start_left in order:
        if not adjacency[start_left]:
            continue

        stack = [start_left]
        seen_left = {start_left}
        seen_right: set[int] = set()
        parent_for_right: dict[int, int] = {}
        free_right = -1

        while stack and free_right < 0:
            left_index = stack.pop()
            for right_index in adjacency[left_index]:
                if right_index in seen_right:
                    continue
                seen_right.add(right_index)
                parent_for_right[right_index] = left_index
                previous_left = right_to_left[right_index]
                if previous_left < 0:
                    free_right = right_index
                    break
                if previous_left not in seen_left:
                    seen_left.add(previous_left)
                    stack.append(previous_left)

        if free_right < 0:
            continue

        # Flip every edge along the discovered augmenting path.
        right_index = free_right
        while right_index >= 0:
            left_index = parent_for_right[right_index]
            previous_right = left_to_right[left_index]
            left_to_right[left_index] = right_index
            right_to_left[right_index] = left_index
            right_index = previous_right

    return [
        (left_index, right_index)
        for left_index, right_index in enumerate(left_to_right)
        if right_index >= 0
    ]


def compute_overlap(
    left: DatasetSampleIndex,
    right: DatasetSampleIndex,
    *,
    policy: Optional[OverlapPolicy] = None,
) -> OverlapEvidence:
    """Compute pair evidence while retaining both endpoint datasets."""

    if not isinstance(left, DatasetSampleIndex) or not isinstance(right, DatasetSampleIndex):
        raise TypeError("left and right must be DatasetSampleIndex values")
    if left.dataset_id == right.dataset_id:
        raise ValueError("overlap requires two distinct dataset IDs")
    policy = policy or OverlapPolicy()

    adjacency = _candidate_adjacency(left.samples, right.samples)
    matching = _maximum_matching(adjacency, right.sample_count)
    shared_count = len(matching)

    left_gsms = {sample.gsm for sample in left.samples if sample.gsm is not None}
    right_gsms = {sample.gsm for sample in right.samples if sample.gsm is not None}
    left_hashes = {
        sample.expression_hash
        for sample in left.samples
        if sample.expression_hash is not None
    }
    right_hashes = {
        sample.expression_hash
        for sample in right.samples
        if sample.expression_hash is not None
    }

    matched_by_gsm = 0
    matched_by_hash = 0
    matched_by_both = 0
    for left_index, right_index in matching:
        sample_left = left.samples[left_index]
        sample_right = right.samples[right_index]
        gsm_match = sample_left.gsm is not None and sample_left.gsm == sample_right.gsm
        hash_match = (
            sample_left.expression_hash is not None
            and sample_left.expression_hash == sample_right.expression_hash
        )
        matched_by_gsm += int(gsm_match)
        matched_by_hash += int(hash_match)
        matched_by_both += int(gsm_match and hash_match)

    left_fraction = shared_count / left.sample_count if left.sample_count else 0.0
    right_fraction = shared_count / right.sample_count if right.sample_count else 0.0
    union_count = left.sample_count + right.sample_count - shared_count
    jaccard = shared_count / union_count if union_count else 0.0
    smaller_count = min(left.sample_count, right.sample_count)
    coefficient = shared_count / smaller_count if smaller_count else 0.0

    if shared_count and shared_count == left.sample_count == right.sample_count:
        classification = OverlapClass.EXACT
    elif shared_count and coefficient >= policy.major_overlap_coefficient:
        classification = OverlapClass.MAJOR
    elif shared_count:
        classification = OverlapClass.MINOR
    else:
        classification = OverlapClass.NONE

    return OverlapEvidence(
        left_dataset_id=left.dataset_id,
        right_dataset_id=right.dataset_id,
        left_sample_count=left.sample_count,
        right_sample_count=right.sample_count,
        shared_sample_count=shared_count,
        shared_gsm_ids=tuple(sorted(left_gsms & right_gsms)),
        shared_expression_hashes=tuple(sorted(left_hashes & right_hashes)),
        matched_by_gsm_count=matched_by_gsm,
        matched_by_expression_hash_count=matched_by_hash,
        matched_by_both_count=matched_by_both,
        left_fraction=float(left_fraction),
        right_fraction=float(right_fraction),
        jaccard=float(jaccard),
        overlap_coefficient=float(coefficient),
        classification=classification,
        exclude_from_discovery=policy.excludes(classification),
    )
