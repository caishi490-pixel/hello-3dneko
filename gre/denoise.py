# SPDX-License-Identifier: Apache-2.0
"""Masked, budgeted position filters for fixed-topology meshes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterResult:
    V: np.ndarray
    iterations: int
    stopped_by_budget: bool
    proposed_peak_m: float


def pack_neighbors(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(1, max(map(len, rows), default=0))
    neighbors = np.broadcast_to(np.arange(len(rows))[:, None], (len(rows), width)).copy()
    active = np.zeros(neighbors.shape, dtype=bool)
    for i, row in enumerate(rows):
        neighbors[i, :len(row)] = row
        active[i, :len(row)] = True
    return neighbors, active


def _run(V, mask, iters, budget, update) -> FilterResult:
    if (mask.shape != (len(V),) or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1))
            or iters < 0 or np.isnan(budget) or budget < 0):
        raise ValueError('Invalid filter mask, iteration count or budget')
    editable = mask > 0
    current = V.copy()
    peak = 0.
    for iteration in range(iters):
        trial = update(current)
        trial[~editable] = V[~editable]
        peak = float(np.linalg.norm(trial - V, axis=1).max(initial=0))
        if not np.isfinite(peak):
            raise ValueError('Non-finite filter update')
        if peak > budget:
            return FilterResult(current, iteration, True, peak)
        current = trial
    return FilterResult(current, iters, False, peak)


def _laplacian(V, adj):
    indices, active = adj
    differences = V[indices] - V[:, None, :]
    return np.sum(differences * active[..., None], axis=1) / np.maximum(1, active.sum(axis=1))[:, None]


def laplacian(V, adj, mask, lam, iters, *, budget=np.inf):
    if not 0 < lam <= 1:
        raise ValueError('Laplacian step must be in (0, 1]')
    return _run(V, mask, iters, budget, lambda X: X + (lam * mask[:, None]) * _laplacian(X, adj))


def taubin(V, adj, mask, lam, mu, iters, *, budget=np.inf):
    if not 0 < lam < 1 or not -1 < mu < 0:
        raise ValueError('Invalid Taubin coefficients')

    def update(X):
        first = X + lam * mask[:, None] * _laplacian(X, adj)
        return first + mu * mask[:, None] * _laplacian(first, adj)

    return _run(V, mask, iters, budget, update)


def _unit(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


def robust_normals(V: np.ndarray, triangles: np.ndarray, adj: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Area-weighted normals with one angularly robust neighborhood pass."""
    points = V[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    field, area = np.zeros_like(V), np.zeros(len(V))
    np.add.at(field, triangles.ravel(), np.repeat(cross, 3, axis=0))
    np.add.at(area, triangles.ravel(), np.repeat(np.linalg.norm(cross, axis=1), 3))
    base = _unit(field)
    indices, active = adj
    near = base[indices]
    cosine = np.einsum('nki,ni->nk', near, base).clip(-1,1)
    weights = active * (cosine >= .60) * np.exp(-.5*((1-cosine)/.12)**2) * area[indices]
    return _unit(field + np.sum(near * weights[..., None], axis=1))


def bilateral_vertex(V, normals, adj_ring, sigma_s, sigma_r, iters, mask, *, budget=np.inf):
    """Run the B2 spatial/normal-height filter; distances and sigmas share units."""
    if not np.isfinite([sigma_s, sigma_r]).all() or min(sigma_s,sigma_r) <= 0:
        raise ValueError('Bilateral sigmas must be positive finite distances')
    indices, active = adj_ring

    def update(X):
        n = normals(X) if callable(normals) else normals
        delta = X[indices] - X[:, None, :]
        height = np.einsum('nki,ni->nk', delta, n)
        space2 = np.sum(delta**2, axis=-1)
        weights = active * np.exp(-.5*(space2/sigma_s**2 + (height/sigma_r)**2))
        total = weights.sum(axis=1)
        offsets = np.divide(np.sum(weights * height, axis=1), total, out=np.zeros(len(X)), where=total>0)
        return X + (mask * offsets)[:, None] * n

    return _run(V, mask, iters, budget, update)


def sharp_vertices(faces: tuple[tuple[int, ...], ...], normals: np.ndarray, angle_deg: float) -> set[int]:
    """Freeze dihedral features and open or non-manifold edge endpoints."""
    owners = {}
    for index, face in enumerate(faces):
        for a,b in zip(face, (*face[1:],face[0])):
            owners.setdefault(tuple(sorted((a,b))), []).append(index)
    cosine = np.cos(np.radians(angle_deg))
    return {i for edge, pair in owners.items() if len(pair)!=2 or normals[pair[0]]@normals[pair[1]]<cosine for i in edge}
