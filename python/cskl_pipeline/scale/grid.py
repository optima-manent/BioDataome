"""Size grid for the null profile (lever D).

A pair ``(P, Q)`` needs ``P``'s null behaviour at the *partner's* sample size
``m_Q``, which can be any size in the corpus. Rather than computing every dataset
at every distinct partner size (``O(N^2)`` again), we compute each dataset's null
profile on a fixed grid ``G`` and interpolate. The grid is denser at small ``m``
(where mu, sigma move fastest) and spans the corpus range.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

# Base anchor points (plan section 2, lever D). Extended to cover the corpus range.
_BASE_GRID = [4, 6, 8, 10, 14, 20, 28, 40, 56, 80, 110, 150, 200, 280, 400]


def build_size_grid(min_m: int, max_m: int, base: Iterable[int] = _BASE_GRID) -> List[int]:
    """Return a sorted grid of sample sizes covering ``[min_m, max_m]``.

    Always includes ``min_m`` and ``max_m`` as endpoints so interpolation never
    extrapolates past the corpus. Beyond 400 the grid continues geometrically
    (~1.4x) so very large datasets are still bracketed.
    """
    min_m = max(2, int(min_m))
    max_m = max(min_m, int(max_m))

    pts = {min_m, max_m}
    for g in base:
        if min_m <= g <= max_m:
            pts.add(int(g))

    # geometric extension above the largest base anchor, if the corpus needs it
    g = base[-1] if base else 400
    while g < max_m:
        g = int(round(g * 1.4))
        if min_m <= g <= max_m:
            pts.add(g)

    return sorted(pts)


def exact_size_grid(sample_sizes: Iterable[int]) -> List[int]:
    """The 'exact-size' grid: every distinct corpus sample size.

    In this mode interpolation is exact because every queried partner size is a
    grid node. Used for small corpora and the faithfulness gates.
    """
    return sorted({int(m) for m in sample_sizes if int(m) >= 2})
