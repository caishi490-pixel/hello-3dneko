# SPDX-License-Identifier: Apache-2.0
"""Conservative triangle self-intersection with stable baseline triangle indices."""

import numpy as np


class PairBudgetExceeded(ValueError):
    """The broad phase reached its configured resource limit."""


def _sweep(lower: np.ndarray, upper: np.ndarray, active: np.ndarray, eps: float, limit: int) -> np.ndarray:
    # Sweep the axis with the greatest spread in box centers.
    axis = int(np.argmax(np.ptp(lower[active] + upper[active], axis=0)))
    order = active[np.argsort(lower[active, axis], kind="stable")]
    starts = lower[order, axis]
    ends = np.searchsorted(starts, upper[order, axis] + eps, side="right")
    chunks = []
    count = 0
    for offset, left in enumerate(order):
        right = order[offset + 1:ends[offset]]
        hit = np.all((lower[right] <= upper[left] + eps) & (upper[right] >= lower[left] - eps), axis=1)
        right = right[hit]
        count += len(right)
        if count > limit:
            raise PairBudgetExceeded(f"AABB pair limit exceeded ({limit})")
        if len(right):
            chunks.append(np.column_stack((np.full(len(right), left), right)))
    return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.intp)


def _cut_interval(points: np.ndarray, distances: np.ndarray, direction: np.ndarray,
                  eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = np.einsum("nij,nj->ni", points, direction)
    ending = np.roll(distances, -1, axis=1)
    opposite = ((distances > eps) & (ending < -eps)) | ((distances < -eps) & (ending > eps))
    fraction = np.divide(distances, distances - ending, out=np.zeros_like(distances), where=opposite)
    cuts = projected + fraction * (np.roll(projected, -1, axis=1) - projected)
    values = np.concatenate((projected, cuts), axis=1)
    valid = np.concatenate((np.abs(distances) <= eps, opposite), axis=1)
    return np.min(np.where(valid, values, np.inf), axis=1), np.max(np.where(valid, values, -np.inf), axis=1)


def _coplanar(a: np.ndarray, b: np.ndarray, normal: np.ndarray, shared: np.ndarray, eps: float) -> np.ndarray:
    ea, eb = (np.roll(points, -1, axis=1) - points for points in (a, b))
    axes = np.cross(np.concatenate((ea, eb), axis=1), normal[:, None, :])
    axes /= np.linalg.norm(axes, axis=2, keepdims=True)
    pa, pb = (np.einsum("nij,nkj->nki", points, axes) for points in (a, b))
    gap = np.maximum(pa.min(axis=2), pb.min(axis=2)) - np.minimum(pa.max(axis=2), pb.max(axis=2))
    contact = np.all(gap <= eps, axis=1)
    hit = contact & ((shared == 0) | np.all(gap < -eps, axis=1))
    # Incident faces sharing one vertex may still overlap along a collinear edge segment.
    indices = np.flatnonzero(contact & (shared == 1) & ~hit)
    if len(indices):
        start_a, start_b = a[indices], b[indices]
        end_b = np.roll(start_b, -1, axis=1)
        edge_a = ea[indices]
        lengths = np.linalg.norm(edge_a, axis=2)
        tangent = edge_a / lengths[:, :, None]
        from_start = start_b[:, None, :, :] - start_a[:, :, None, :]
        from_end = end_b[:, None, :, :] - start_a[:, :, None, :]
        axis = tangent[:, :, None, :]
        on_line = (np.linalg.norm(np.cross(from_start, axis), axis=3) <= eps) & (
            np.linalg.norm(np.cross(from_end, axis), axis=3) <= eps)
        lo, hi = (np.sum(offset * axis, axis=3) for offset in (from_start, from_end))
        overlap = np.minimum(np.maximum(lo, hi), lengths[:, :, None]) - np.maximum(np.minimum(lo, hi), 0)
        hit[indices] = np.any(on_line & (overlap > eps), axis=(1, 2))
    return hit


def _intersect(a: np.ndarray, b: np.ndarray, shared: np.ndarray, eps: float) -> np.ndarray:
    a, b = a - a[:, :1], b - a[:, :1]
    na, nb = (np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]) for points in (a, b))
    na /= np.linalg.norm(na, axis=1, keepdims=True)
    nb /= np.linalg.norm(nb, axis=1, keepdims=True)
    db = np.einsum("nij,nj->ni", b, na)
    da = np.einsum("nij,nj->ni", a - b[:, :1], nb)
    crosses = np.cross(na, nb)
    lengths = np.linalg.norm(crosses, axis=1)
    hit = np.zeros(len(a), dtype=bool)
    planar = (lengths <= 1e-6) & (np.max(np.abs(da), axis=1) <= eps) & (np.max(np.abs(db), axis=1) <= eps)
    if np.any(planar):
        hit[planar] = _coplanar(a[planar], b[planar], na[planar], shared[planar], eps)
    transverse = ~planar & (lengths > 0) & (shared < 2)
    if np.any(transverse):
        sin_angle = lengths[transverse, None]
        line = crosses[transverse] / sin_angle
        a_cut, b_cut = a[transverse], b[transverse]
        # Distance to the other plane is sin(angle) times distance to the cut line.
        # Using the full spatial epsilon here expands almost-parallel cut intervals.
        scale = np.maximum(np.abs(a_cut).max(axis=(1, 2)), np.abs(b_cut).max(axis=(1, 2)))
        plane_eps = np.maximum(eps * sin_angle, (16 * np.finfo(float).eps * scale)[:, None])
        lo_a, hi_a = _cut_interval(a_cut, da[transverse], line, plane_eps)
        lo_b, hi_b = _cut_interval(b_cut, db[transverse], line, plane_eps)
        width = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
        hit[transverse] = np.where(shared[transverse] == 0, width >= -eps, width > eps)
    return hit


def intersection_pairs(V: np.ndarray, triangles: np.ndarray, eligible: np.ndarray,
                       eps: float, pair_limit: int) -> set[tuple[int, int]]:
    """Count non-adjacent contacts and overlaps beyond a shared vertex or edge."""
    active = np.flatnonzero(eligible)
    if len(active) < 2:
        return set()
    points = V[triangles]
    pairs = _sweep(points.min(axis=1), points.max(axis=1), active, eps, pair_limit)
    result = set()
    for offset in range(0, len(pairs), 8192):
        indices = pairs[offset:offset + 8192]
        left, right = indices.T
        shared = np.sum(triangles[left, :, None] == triangles[right, None, :], axis=(1, 2))
        hits = _intersect(points[left], points[right], shared, eps)
        result.update(tuple(sorted(pair)) for pair in indices[hits].tolist())
    return result
