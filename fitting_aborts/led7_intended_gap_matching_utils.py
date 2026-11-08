"""Whole-row matching of ordered timing pairs, with an exact KS constraint.

For gap <= intended, F_gap(t) - F_intended(t) is the fraction of selected
intervals [gap, intended) covering t. Matching therefore reduces to selecting
as many intervals as possible with maximum overlap / selected count <= 1/q.
No outcomes, abort times or LED flags enter these pure functions.
"""

import heapq
import numpy as np


def paired_ks(intended, gap):
    """Unbinned empirical KS distance; no independent-samples p-value."""
    x, y = np.sort(np.asarray(intended)), np.sort(np.asarray(gap))
    assert len(x) == len(y)
    if not len(x):
        return np.nan
    points = np.unique(np.r_[x, y])
    return float(np.abs(np.searchsorted(y, points, side="right")
                        - np.searchsorted(x, points, side="right")).max() / len(x))


def match_intended_gap(intended, gap, ks_inverse=5):
    """Maximize retained rows at KS <= 1/ks_inverse, with no replacement.

Input row order breaks exact timing ties; the caller supplies seeded ordering.
At fixed overlap capacity k, the earliest-start sweep discards the furthest
ending active interval on overflow. This maximizes retained interval count.

Feasible capacities form a prefix: from a feasible set, retain q*k rows
including all zero-length intervals when possible, color the nonzero interval
graph into k nonoverlapping classes, and remove a smallest class (at most q
rows) to obtain feasibility at k-1. Thus binary search is valid. Any KS-feasible
subset has overlap k <= floor(N/q), and cannot exceed the sweep's count at k.
The final k and infeasible k+1 boundary therefore certify maximal cardinality.
"""
    x, y = np.asarray(intended, dtype=float), np.asarray(gap, dtype=float)
    assert x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 0
    assert np.isfinite(x).all() and np.isfinite(y).all() and (x >= y).all()
    assert isinstance(ks_inverse, int) and ks_inverse >= 1
    n = len(x)
    zero = x == y
    order = np.lexsort((np.arange(n), x, y))
    order = order[~zero[order]]

    def pack(capacity):
        keep = zero.copy()
        if capacity == 0:
            return keep
        alive = np.zeros(n, dtype=bool)
        earliest, latest, active = [], [], 0
        for i in order:
            # Half-open intervals: an interval ending at this start no longer overlaps.
            while earliest and earliest[0][0] <= y[i]:
                _, j = heapq.heappop(earliest)
                if alive[j]:
                    alive[j] = False
                    active -= 1
            alive[i] = keep[i] = True
            active += 1
            heapq.heappush(earliest, (x[i], i))
            heapq.heappush(latest, (-x[i], i))
            if active > capacity:
                while not alive[latest[0][1]]:
                    heapq.heappop(latest)
                _, j = heapq.heappop(latest)
                alive[j] = keep[j] = False
                active -= 1
        return keep

    low, high = 0, n // ks_inverse
    while low < high:
        middle = (low + high + 1) // 2
        if int(pack(middle).sum()) >= ks_inverse * middle:
            low = middle
        else:
            high = middle - 1
    keep = pack(low)
    retained = int(keep.sum())
    assert retained >= ks_inverse * low and keep[zero].all()
    next_retained = int(pack(low + 1).sum()) if low + 1 <= n // ks_inverse else None
    assert next_retained is None or next_retained < ks_inverse * (low + 1)
    distance = paired_ks(x[keep], y[keep])
    if retained:
        assert distance <= 1 / ks_inverse + 1e-12
    return keep, {
        "original_count": n, "retained_count": retained, "removed_count": n - retained,
        "ks_before": paired_ks(x, y), "ks_after": distance, "ks_limit": 1 / ks_inverse,
        "zero_length_count": int(zero.sum()), "capacity": low,
        "next_capacity": low + 1 if next_retained is not None else None,
        "next_capacity_max_retained": next_retained,
        "next_capacity_required_count": ks_inverse * (low + 1) if next_retained is not None else None,
        "capacity_boundary_verified": True, "maximum_retention": True,
    }
