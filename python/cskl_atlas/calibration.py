from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .catalog import Catalog


class OutOfCalibrationRange(ValueError):
    """Raised when a new sample size falls outside a frozen null-profile grid."""


@dataclass(frozen=True)
class NullProfileArtifact:
    version_id: str
    signature_hash: str
    pool_hash: str
    feature_hash: str
    alpha: float
    bootstrap_count: int
    grid: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def validate(self) -> None:
        grid = np.asarray(self.grid)
        mu = np.asarray(self.mu)
        sigma = np.asarray(self.sigma)
        if grid.ndim != 1 or len(grid) < 1:
            raise ValueError("A null profile needs a one-dimensional grid.")
        if len(grid) != len(mu) or len(grid) != len(sigma):
            raise ValueError("grid, mu, and sigma lengths differ.")
        if not np.all(np.diff(grid) > 0):
            raise ValueError("Null-profile grid must be strictly increasing.")
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
            raise ValueError("Null-profile values must be finite.")
        if np.any(sigma < 0):
            raise ValueError("Null-profile sigma cannot be negative.")
        if not 0 < self.alpha < 1 or self.bootstrap_count < 1:
            raise ValueError("Invalid null-profile alpha or bootstrap count.")

    def at(self, sample_count: int, *, allow_clamp: bool = False) -> tuple[float, float]:
        self.validate()
        low, high = int(self.grid[0]), int(self.grid[-1])
        if not allow_clamp and not low <= sample_count <= high:
            raise OutOfCalibrationRange(
                f"sample_count={sample_count} is outside calibrated grid [{low}, {high}]"
            )
        mu = float(np.interp(sample_count, self.grid, self.mu))
        sigma = max(float(np.interp(sample_count, self.grid, self.sigma)), 1e-12)
        return mu, sigma


def normal_cdf(value: float, mean: float, sigma: float) -> float:
    z = (value - mean) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def validated_stored_profile_grid(profile: object) -> tuple[int, ...]:
    """Validate the compact null-profile format used by the scalable store."""

    grid = np.asarray(getattr(profile, "grid", None))
    mu = np.asarray(getattr(profile, "mu", None))
    sigma = np.asarray(getattr(profile, "sigma", None))
    if grid.ndim != 1 or len(grid) < 1:
        raise ValueError("A stored null profile needs a one-dimensional grid.")
    if len(grid) != len(mu) or len(grid) != len(sigma):
        raise ValueError("Stored null-profile grid, mu, and sigma lengths differ.")
    if not np.all(np.isfinite(grid)) or not np.all(np.equal(grid, np.floor(grid))):
        raise ValueError("Stored null-profile grid values must be finite integers.")
    if np.any(grid < 0) or not np.all(np.diff(grid) > 0):
        raise ValueError("Stored null-profile grid must be non-negative and strictly increasing.")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("Stored null-profile values must be finite.")
    if np.any(sigma < 0):
        raise ValueError("Stored null-profile sigma cannot be negative.")
    return tuple(int(value) for value in grid)


def pair_pvalue_from_stored_profiles(
    cskl: float,
    profile_a: object,
    samples_b: int,
    profile_b: object,
    samples_a: int,
    *,
    expected_pool_hash: str,
    allow_clamp: bool,
) -> tuple[float, int]:
    """Calculate one p-value and report how many profile lookups clamped.

    Unlike the preserved scale implementation, this release-path helper does
    not silently clamp. Exact releases raise :class:`OutOfCalibrationRange`;
    frozen operational releases opt into clamping and receive diagnostics.
    """

    if not math.isfinite(float(cskl)) or float(cskl) < 0:
        raise ValueError("cskl must be finite and non-negative")
    if getattr(profile_a, "pool_hash", None) != expected_pool_hash or getattr(
        profile_b, "pool_hash", None
    ) != expected_pool_hash:
        raise ValueError("A stored null profile belongs to a different pool release.")
    if getattr(profile_a, "feature_hash", None) != getattr(profile_b, "feature_hash", None):
        raise ValueError("Stored profiles from different feature universes are not comparable.")
    grid_a = validated_stored_profile_grid(profile_a)
    grid_b = validated_stored_profile_grid(profile_b)

    def at(profile: object, grid: tuple[int, ...], sample_count: int) -> tuple[float, float, bool]:
        if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        clamped = not grid[0] <= int(sample_count) <= grid[-1]
        if clamped and not allow_clamp:
            raise OutOfCalibrationRange(
                f"sample_count={sample_count} is outside calibrated grid [{grid[0]}, {grid[-1]}]"
            )
        mu = float(np.interp(sample_count, grid, np.asarray(profile.mu)))
        sigma = max(
            float(np.interp(sample_count, grid, np.asarray(profile.sigma))),
            1e-12,
        )
        return mu, sigma, clamped

    mu_a, sigma_a, clamped_a = at(profile_a, grid_a, samples_b)
    mu_b, sigma_b, clamped_b = at(profile_b, grid_b, samples_a)
    p_value = max(
        normal_cdf(float(cskl), mu_a, sigma_a),
        normal_cdf(float(cskl), mu_b, sigma_b),
    )
    return p_value, int(clamped_a) + int(clamped_b)


def pair_pvalue(
    cskl: float,
    profile_a: NullProfileArtifact,
    samples_b: int,
    profile_b: NullProfileArtifact,
    samples_a: int,
    *,
    expected_pool_hash: str,
    allow_clamp: bool = False,
) -> float:
    if profile_a.pool_hash != expected_pool_hash or profile_b.pool_hash != expected_pool_hash:
        raise ValueError("A null profile belongs to a different pool release.")
    if profile_a.feature_hash != profile_b.feature_hash:
        raise ValueError("Profiles from different feature universes are not comparable.")
    if not math.isclose(profile_a.alpha, profile_b.alpha, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Profiles with different alpha values are not comparable.")
    mu_a, sigma_a = profile_a.at(samples_b, allow_clamp=allow_clamp)
    mu_b, sigma_b = profile_b.at(samples_a, allow_clamp=allow_clamp)
    return max(normal_cdf(cskl, mu_a, sigma_a), normal_cdf(cskl, mu_b, sigma_b))


def calibrate_pairs(
    pair_rows: Iterable[Mapping[str, object]],
    profile_loader: Callable[[str], NullProfileArtifact],
    *,
    expected_pool_hash: str,
    allow_clamp: bool = False,
) -> Iterator[tuple[str, float]]:
    """Stream p-values while binding every profile to its signature and pool."""

    @lru_cache(maxsize=512)
    def load(version_id: str) -> NullProfileArtifact:
        profile = profile_loader(version_id)
        if profile.version_id != version_id:
            raise ValueError("Profile loader returned the wrong dataset version.")
        profile.validate()
        return profile

    for row in pair_rows:
        version_a = str(row["version_a"])
        version_b = str(row["version_b"])
        profile_a = load(version_a)
        profile_b = load(version_b)
        expected_signature_a = str(row.get("signature_a") or "")
        expected_signature_b = str(row.get("signature_b") or "")
        if expected_signature_a and profile_a.signature_hash != expected_signature_a:
            raise ValueError(f"Stale null profile for {version_a}: signature hash changed.")
        if expected_signature_b and profile_b.signature_hash != expected_signature_b:
            raise ValueError(f"Stale null profile for {version_b}: signature hash changed.")
        yield str(row["pair_id"]), pair_pvalue(
            float(row["cskl"]),
            profile_a,
            int(row["samples_b"]),
            profile_b,
            int(row["samples_a"]),
            expected_pool_hash=expected_pool_hash,
            allow_clamp=allow_clamp,
        )


def run_calibration(
    catalog: Catalog,
    *,
    calibration_id: str,
    pair_batches: Iterable[Sequence[Mapping[str, object]]],
    profile_loader: Callable[[str], NullProfileArtifact],
    expected_pool_hash: str,
    mode: str,
) -> int:
    """Compute streamed p-values, then exact global BH for one release.

    ``mode='exact'`` rejects out-of-grid sample sizes. ``mode='frozen'`` may
    clamp to the frozen operational grid, but the release mode makes that
    methodological variant explicit in provenance and UI.
    """
    if mode not in {"exact", "frozen"}:
        raise ValueError("mode must be exact or frozen")
    total = 0
    for batch in pair_batches:
        values = list(
            calibrate_pairs(
                batch,
                profile_loader,
                expected_pool_hash=expected_pool_hash,
                allow_clamp=mode == "frozen",
            )
        )
        catalog.record_pvalues(calibration_id, values)
        total += len(values)
    catalog.finalize_bh(calibration_id)
    return total
