# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic wave field and geodesic transition weights (NumPy only)."""

import heapq
from collections import deque

import numpy as np


def distances(graph: dict[int, list[tuple[int, float]]], sources: set[int]) -> dict[int, float]:
    result = dict.fromkeys(sources, 0.0)
    queue = [(0.0, vertex) for vertex in sources]
    heapq.heapify(queue)
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance > result[vertex]:
            continue
        for neighbor, length in graph[vertex]:
            candidate = distance + length
            if candidate < result.get(neighbor, np.inf):
                result[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return result


def transition_weights(
    graph: dict[int, list[tuple[int, float]]], boundary: set[int], rings: int,
) -> dict[int, float]:
    weights = dict.fromkeys(graph, 1.0)
    weights.update(dict.fromkeys(boundary, 0.0))
    if not boundary or not rings:
        return weights

    levels = dict.fromkeys(boundary, 0)
    queue = deque(sorted(boundary))
    while queue:
        vertex = queue.popleft()
        level = levels[vertex]
        if level == rings:
            continue
        for neighbor, _ in graph[vertex]:
            if neighbor not in levels:
                levels[neighbor] = level + 1
                queue.append(neighbor)

    inner = graph.keys() - levels.keys()
    to_boundary = distances(graph, boundary)
    to_interior = distances(graph, inner)
    for vertex in levels.keys() - boundary:
        distance = to_boundary[vertex]
        fraction = distance / (distance + to_interior.get(vertex, np.inf))
        weights[vertex] = fraction * fraction * (3 - 2 * fraction)
    return weights


def normal_offsets(
    points_m: np.ndarray, weights: np.ndarray, peak_m: float, wavelength_m: float, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(5, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    phases = rng.uniform(0, 2 * np.pi, size=len(directions))
    wave = np.sin((2 * np.pi / wavelength_m) * (points_m @ directions.T) + phases).mean(axis=1)
    tapered = wave * weights
    peak = np.max(np.abs(tapered))
    if not np.isfinite(peak) or peak <= np.finfo(np.float64).eps:
        raise ValueError("Selected region has no usable wave support outside the locked boundary")
    return tapered * (peak_m / peak)
