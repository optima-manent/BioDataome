"""Scalable, updateable, crash-safe run pipeline for the c-SKL method.

This subpackage adds *faster paths* and a *resumable pipeline shape* around the
polished numerical core in :mod:`cskl`. It never changes the c-SKL math; every
fast path is validated to reproduce :mod:`cskl` (see :mod:`cskl_pipeline.scale.reference`
and the tests under ``tests/test_scale_*``).

Design (see ``SCALABLE_PIPELINE_PLAN.md``):
  * lever A - one frozen, versioned background pool per platform.
  * lever B - memoize the bootstrap null per ``(dataset, size)``.
  * lever C - fast gram-trick fit + batched null distances (numerically exact).
  * lever D - size-grid + interpolation for the per-dataset null profile.

Files are the state model: no database. Every artifact is written atomically and
resume is "skip datasets whose signature + null profile already exist and validate".
"""

from __future__ import annotations

__all__ = [
    "fastcore",
    "store",
    "grid",
    "pool",
    "qc",
    "align",
    "profile",
    "ingest",
    "assemble",
    "reference",
    "cli",
]
