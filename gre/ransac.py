# SPDX-License-Identifier: Apache-2.0
"""Deterministic, area-weighted plane detection with connected face support."""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from gre.geometry import face_edges
from gre.plane import DegeneratePlane, fit_plane


@dataclass(frozen=True)
class PlaneCandidate:
    normal: np.ndarray
    d: float
    inliers: np.ndarray
    coverage: float
    rms_m: float
    minimum_normal_cosine: float
    iterations: int
    estimated_iterations: int


def iteration_count(probability: float, fraction: float, sample_size: int, cap: int) -> int:
    if fraction <= 0:
        return cap
    if fraction >= 1:
        return 1
    success = fraction**sample_size
    if not success:
        return cap
    if success >= 1:
        return 1
    estimate = math.log1p(-probability) / math.log1p(-success)
    return cap if estimate >= cap else max(1, math.ceil(estimate))


def face_neighbors(faces: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    by_edge = defaultdict(list)
    for index, face in enumerate(faces):
        for edge in face_edges((face,)):
            by_edge[edge].append(index)
    neighbors = [set() for _ in faces]
    for indices in by_edge.values():
        if len(indices) == 2:
            left, right = indices
            neighbors[left].add(right)
            neighbors[right].add(left)
    return tuple(tuple(sorted(row)) for row in neighbors)


def _largest_connected(mask: np.ndarray, neighbors: tuple[tuple[int, ...], ...], areas: np.ndarray) -> np.ndarray:
    remaining = set(np.flatnonzero(mask).tolist())
    largest = np.empty(0, dtype=np.intp)
    covered = 0.0
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        pending, group = [first], [first]
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    group.append(neighbor)
        group.sort()
        area = float(areas[group].sum())
        if area > covered:
            largest, covered = np.array(group, dtype=np.intp), area
    return largest


def detect_plane(points: np.ndarray, normals: np.ndarray, areas: np.ndarray,
                 neighbors: tuple[tuple[int, ...], ...], *, seed: int, distance_m: float,
                 minimum_coverage: float = 0.60, normal_tolerance_deg: float = 12.0,
                 confidence: float = 0.999, cap: int = 256) -> PlaneCandidate | None:
    """Detect one connected plane within an approved region, without writing coordinates."""
    if (len(points) != len(normals) or len(points) != len(areas) or len(points) != len(neighbors)
            or not np.isfinite(points).all() or not np.isfinite(normals).all() or not np.isfinite(areas).all()
            or np.any(areas < 0) or not 0 < minimum_coverage <= 1 or not 0 < confidence < 1
            or not 0 < distance_m or not 0 < normal_tolerance_deg < 90 or cap < 1):
        raise ValueError("Invalid plane RANSAC inputs or thresholds")
    if np.count_nonzero(areas) < 3:
        return None
    rng = np.random.default_rng(seed)
    total = float(areas.sum())
    probabilities = areas / total
    cosine = float(np.cos(np.deg2rad(normal_tolerance_deg)))
    area_floor = float(np.max(np.ptp(points, axis=0))**2 * 1e-12)

    def support(n: np.ndarray, d: float) -> np.ndarray:
        inliers = (np.abs(points @ n + d) <= distance_m) & (normals @ n >= cosine) & (areas > 0)
        return _largest_connected(inliers, neighbors, areas)

    best = None
    bound = cap
    iterations = 0
    while iterations < bound:
        iterations += 1
        a, b, c = rng.choice(len(points), 3, replace=False, p=probabilities)
        cross = np.cross(points[b] - points[a], points[c] - points[a])
        magnitude = np.linalg.norm(cross)
        if magnitude <= area_floor:
            continue
        n = cross / magnitude
        if n @ normals[[a, b, c]].sum(axis=0) < 0:
            n = -n
        ids = support(n, -float(points[a] @ n))
        if len(ids) < 3:
            continue
        try:
            for _ in range(4):
                refit, d, rms, _ = fit_plane(points[ids], areas[ids])
                sign = 1 if n @ refit >= 0 else -1
                n, d = refit * sign, d * sign
                refined = support(n, d)
                if np.array_equal(ids, refined):
                    break
                ids = refined
            else:
                continue
        except DegeneratePlane:
            continue
        coverage = float(areas[ids].sum() / total)
        if coverage < minimum_coverage:
            continue
        if best is None or (coverage, -rms) > (best[3], -best[4]):
            normal_cosine = float(np.clip(np.min(normals[ids] @ n), -1, 1))
            sign = 1 if n[np.argmax(np.abs(n))] >= 0 else -1
            best = (n * sign, d * sign, ids, coverage, rms, normal_cosine)
            bound = iteration_count(confidence, coverage, 3, cap)
    return None if best is None else PlaneCandidate(*best, iterations, bound)
