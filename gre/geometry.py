# SPDX-License-Identifier: Apache-2.0
"""Geometry analysis in object-local coordinates, without Blender dependencies."""

import heapq
from collections import Counter, deque

import numpy as np

Graph = dict[int, list[tuple[int, float]]]


def face_geometry(V: np.ndarray, triangles: np.ndarray, face_index: np.ndarray,
                  face_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return polygon normals, polygon areas and barycentric per-vertex areas."""
    points = V[triangles]
    crosses = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    triangle_areas = np.linalg.norm(crosses, axis=1) * 0.5
    polygon_areas = np.bincount(face_index, weights=triangle_areas, minlength=face_count)
    normals = np.zeros((face_count, 3))
    np.add.at(normals, face_index, crosses)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, lengths, where=lengths > 0, out=normals)
    vertex_areas = np.zeros(len(V))
    np.add.at(vertex_areas, triangles.ravel(), np.repeat(triangle_areas / 3, 3))
    return normals, polygon_areas, vertex_areas


def face_edges(faces: tuple[tuple[int, ...], ...]) -> set[tuple[int, int]]:
    return {(min(left, right), max(left, right)) for face in faces
            for left, right in zip(face, (*face[1:], face[0]))}


def closed_locked_rim(faces: tuple[tuple[int, ...], ...], locked: set[int]) -> np.ndarray | None:
    """Return a single closed region perimeter when every perimeter vertex is hard-locked."""
    uses = Counter(edge for face in faces for edge in face_edges((face,)))
    if any(count > 2 for count in uses.values()):
        return None
    rim: dict[int, set[int]] = {}
    for (left, right), count in uses.items():
        if count == 1:
            rim.setdefault(left, set()).add(right)
            rim.setdefault(right, set()).add(left)
    if len(rim) < 3 or rim.keys() != locked or any(len(neighbors) != 2 for neighbors in rim.values()):
        return None
    pending = [min(rim)]
    seen = set()
    while pending:
        vertex = pending.pop()
        if vertex not in seen:
            seen.add(vertex)
            pending.extend(rim[vertex] - seen)
    if len(seen) != len(rim):
        return None
    return np.array(sorted(seen), dtype=np.intp)


def adjacency(V: np.ndarray, edges: set[tuple[int, int]]) -> tuple[Graph, float]:
    """Build the weighted adjacency list and return the local median edge length."""
    if not edges:
        raise ValueError("Region has no edges")
    graph: Graph = {}
    lengths = []
    for left, right in sorted(edges):
        length = float(np.linalg.norm(V[left] - V[right]))
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Region contains a zero-length or nonfinite edge")
        graph.setdefault(left, []).append((right, length))
        graph.setdefault(right, []).append((left, length))
        lengths.append(length)
    return graph, float(np.median(lengths))


def bbox_diagonal(V: np.ndarray) -> float:
    return float(np.linalg.norm(np.ptp(V, axis=0)))


def geodesic_distances(graph: Graph, sources: set[int]) -> dict[int, float]:
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


def partition_region(graph: Graph, boundary: set[int], rings: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return core, blend vertices, and smoothstep weights from the locked boundary to the core."""
    levels = dict.fromkeys(boundary, 0)
    queue = deque(sorted(boundary))
    while queue:
        vertex = queue.popleft()
        if levels[vertex] == rings:
            continue
        for neighbor, _ in graph[vertex]:
            if neighbor not in levels:
                levels[neighbor] = levels[vertex] + 1
                queue.append(neighbor)

    core = graph.keys() - levels.keys()
    blend = sorted(levels.keys() - boundary)
    if not blend:
        return np.array(sorted(core), dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0)
    from_core = geodesic_distances(graph, core)
    from_boundary = geodesic_distances(graph, boundary)
    blend = [vertex for vertex in blend if vertex in from_core]
    # Locally, ring_width is the sum of distances from the vertex to its two interfaces.
    t = np.array([from_boundary[vertex] / (from_core[vertex] + from_boundary[vertex])
                  for vertex in blend])
    return np.array(sorted(core), dtype=np.intp), np.array(blend, dtype=np.intp), t * t * (3 - 2 * t)


def distance_statistics(vectors: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(vectors, axis=1)
    return dict(rms=float(np.sqrt(np.mean(distances**2))), p95=float(np.percentile(distances, 95)),
                max=float(np.max(distances)))
