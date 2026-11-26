# SPDX-License-Identifier: Apache-2.0
"""A small NumPy LM solve for explicitly approved plane relations and junctions."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


class Infeasible(ValueError):
    """The declared relations or junction constraints cannot be met."""


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    d: float


@dataclass(frozen=True)
class Observation:
    plane: Plane
    points: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class Relation:
    kind: str
    a: str
    b: str
    distance_m: float = 0.0


def relation_error(planes: dict[str, Plane], relation: Relation) -> dict:
    left, right = planes[relation.a], planes[relation.b]
    dot = float(np.clip(left.normal @ right.normal, -1, 1))
    cross = float(np.linalg.norm(np.cross(left.normal, right.normal)))
    angle = float(np.degrees(np.arctan2(cross, abs(dot))))
    values = {"type": relation.kind, "a": relation.a, "b": relation.b,
              "deviation_deg": abs(90 - angle) if relation.kind == "perpendicular" else angle}
    if relation.kind == "offset":
        separation = left.d - (right.d if dot >= 0 else -right.d)
        values.update(distance_m=float(separation), expected_m=relation.distance_m,
                      offset_error_m=abs(float(separation) - relation.distance_m))
    return values


def solve_planes(observations: dict[str, Observation], relations: tuple[Relation, ...], *,
                 max_steps: int = 60) -> tuple[dict[str, Plane], int]:
    """Fit normals on unit tangent charts with LM and validate exact relation tolerances."""
    names = tuple(observations)
    ids = {name: i for i, name in enumerate(names)}
    if not names or max_steps < 1 or any(relation.kind not in {"parallel", "perpendicular", "offset"}
                                       or relation.a not in ids or relation.b not in ids or relation.a == relation.b
                                       or not np.isfinite(relation.distance_m) for relation in relations):
        raise ValueError("Invalid plane relation")
    values = list(observations.values())
    initial = np.array([item.plane.normal for item in values])
    basis = np.eye(3)[np.argmin(np.abs(initial), axis=1)]
    tangent_u = np.cross(initial, basis)
    tangent_u /= np.linalg.norm(tangent_u, axis=1, keepdims=True)
    tangent_v = np.cross(initial, tangent_u)
    centers = np.array([np.average(item.points, axis=0, weights=item.weights) for item in values])
    factors = []
    for item, center in zip(values, centers):
        positions = item.points - center
        _, singular, axes = np.linalg.svd(positions * np.sqrt(item.weights[:, None] / item.weights.sum()),
                                         full_matrices=False)
        factors.append(singular[:, None] * axes)
    factors = np.array(factors)
    start = np.zeros((len(names), 3))
    start[:, 2] = np.sum(initial * centers, axis=1) + np.array([item.plane.d for item in values])
    links = [(relation, ids[relation.a], ids[relation.b]) for relation in relations]
    # Remove the global translation from the signed offset residual when normals are nearly parallel.
    origin = centers.mean(axis=0)

    def decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = x.reshape(-1, 3)
        n = initial + p[:, :1] * tangent_u + p[:, 1:2] * tangent_v
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        offsets = p[:, 2] - np.sum(n * centers, axis=1)
        return n, offsets

    def residual(x: np.ndarray) -> np.ndarray:
        n, d = decode(x)
        fit = np.column_stack((x.reshape(-1, 3)[:, 2], np.einsum("nij,nj->ni", factors, n))).ravel()
        terms = [fit]
        for relation, a, b in links:
            if relation.kind == "perpendicular":
                terms.append(np.array([10 * (n[a] @ n[b])]))
            else:
                terms.append(10 * np.cross(n[a], n[b]))
                if relation.kind == "offset":
                    sign = 1 if initial[a] @ initial[b] >= 0 else -1
                    difference = d[a] - sign * d[b] + (n[a] - sign * n[b]) @ origin
                    terms.append(np.array([40 * (difference - relation.distance_m)]))
        return np.concatenate(terms)

    x = start.ravel()
    damping = 1e-4
    steps = 0
    perturbations = np.eye(len(x)) * 1e-6
    r = residual(x)
    for steps in range(1, max_steps + 1):
        J = np.column_stack([(residual(x + delta) - residual(x - delta)) / 2e-6 for delta in perturbations])
        gradient = J.T @ r
        if np.linalg.norm(gradient, ord=np.inf) < 1e-10:
            break
        system = J.T @ J
        step = np.linalg.solve(system + damping * np.diag(np.maximum(np.diag(system), 1e-8)), -gradient)
        if np.linalg.norm(step) < 1e-11:
            break
        trial = residual(x + step)
        if trial @ trial < r @ r:
            x += step
            r = trial
            damping = max(damping / 3, 1e-10)
        else:
            damping = min(damping * 10, 1e12)
    n, d = decode(x)
    planes = {name: Plane(n[i], float(d[i])) for name, i in ids.items()}
    for relation in relations:
        error = relation_error(planes, relation)
        if not np.isfinite(error["deviation_deg"]) or error["deviation_deg"] > 0.045 or error.get("offset_error_m", 0) > 0.0001:
            raise Infeasible(f"{relation.kind}({relation.a}, {relation.b}) exceeds relation tolerance")
    return planes, steps


def project_junctions(V: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                      memberships: dict[int, set[str]], planes: dict[str, Plane],
                      approved_junctions: set[frozenset[str]]) -> tuple[np.ndarray, dict]:
    """Nearest correction for one plane, a two-plane line, or a shared multi-plane point."""
    result = V.copy()
    groups = defaultdict(list)
    weights_by_index = dict(zip(indices.tolist(), weights.tolist()))
    for index in indices.tolist():
        groups[tuple(sorted(memberships[index]))].append(index)
    summary = {"line_core_vertices": 0, "corner_core_vertices": 0, "core_max_residual_m": 0.0}
    for names, group in groups.items():
        if len(names) > 1 and frozenset(names) not in approved_junctions:
            raise Infeasible(f"Unapproved junction {' / '.join(names)}")
        N = np.array([planes[name].normal for name in names])
        d = np.array([planes[name].d for name in names])
        goal = -(V[group] @ N.T + d)
        corrections, _, rank, _ = np.linalg.lstsq(N, goal.T, rcond=1e-9)
        actual = corrections.T @ N.T - goal
        if rank < min(3, len(names)) or np.max(np.abs(actual), initial=0) > 1e-7:
            raise Infeasible(f"Incompatible junction {' / '.join(names)}")
        weights = np.array([weights_by_index[index] for index in group])
        core = weights == 1
        result[group] += corrections.T * weights[:, None]
        if len(names) > 1:
            key = "line_core_vertices" if len(names) == 2 else "corner_core_vertices"
            summary[key] += int(core.sum())
            summary["core_max_residual_m"] = max(summary["core_max_residual_m"], float(np.max(np.abs(actual[core]), initial=0)))
    return result, summary
