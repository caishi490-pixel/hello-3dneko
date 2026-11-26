# SPDX-License-Identifier: Apache-2.0
"""Independent P2 gates for a frozen snapshot and a stable-ID displacement patch."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from gre.collision import PairBudgetExceeded, intersection_pairs
from gre.geometry import distance_statistics
from gre.snapshot import MeshSnapshot


INVARIANTS = ("topology_hash", "uv_hash", "material_hash", "object_relationship_hash", "protected_attribute_hash")
CHECKS = ("integrity", "locked", "displacement", "degenerate", "flip", "self_intersection", "uv_stretch", "area_volume", "target_error")


@dataclass(frozen=True)
class CandidatePatch:
    """Sparse targets encode each proposed displacement using exact stored destination coordinates."""

    vertex_ids: np.ndarray
    positions: np.ndarray
    hashes: dict[str, object]

    @classmethod
    def between(cls, baseline: MeshSnapshot, observed: MeshSnapshot) -> "CandidatePatch":
        if not np.array_equal(baseline.vid, observed.vid):
            return cls(np.empty(0, dtype=np.int32), np.empty((0, 3)), observed.hashes)
        changed = np.any(baseline.V.view(np.uint64) != observed.V.view(np.uint64), axis=1)
        return cls(observed.vid[changed], observed.V[changed], observed.hashes)


@dataclass(frozen=True)
class QAProfile:
    editable_vids: frozenset[int]
    max_displacement_m: float
    min_angle_deg: float = 0.1
    uv_p95: float = 0.02
    uv_max: float = 0.05
    area_limit: float = 0.01
    volume_limit: float = 0.005
    pair_limit: int = 500_000
    target_vids: tuple[int, ...] = ()
    target_plane: tuple[np.ndarray, float] | None = None
    clean_positions: np.ndarray | None = None


def _item(status: str, reason: str | None = None, **values) -> dict:
    return {"status": status, "reason": reason, **values}


def _triangle_stats(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.roll(points, -1, axis=1) - points
    cross = np.cross(edges[:, 0], -edges[:, 2])
    area2 = np.linalg.norm(cross, axis=1)
    dots = -np.sum(edges * np.roll(edges, 1, axis=1), axis=2)
    minimum_angle = np.degrees(np.min(np.arctan2(area2[:, None], dots), axis=1))
    return cross, area2, minimum_angle


def continuous_orientation(before: np.ndarray, after: np.ndarray, area_epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Minimum projected signed determinant ratio over the linear motion of each triangle."""
    e1, e2 = before[:, 1] - before[:, 0], before[:, 2] - before[:, 0]
    p1, p2 = after[:, 1] - after[:, 0] - e1, after[:, 2] - after[:, 0] - e2
    initial = np.cross(e1, e2)
    area = np.linalg.norm(initial, axis=1)
    valid = area > area_epsilon
    # Using the original oriented normal is equivalent to determinants in its local tangent frame.
    frame = np.divide(initial, area[:, None]**2, out=np.zeros_like(initial), where=valid[:, None])
    a = np.einsum("ij,ij->i", np.cross(p1, p2), frame)
    b = np.einsum("ij,ij->i", np.cross(p1, e2) + np.cross(e1, p2), frame)
    t = np.divide(-b, 2 * a, out=np.zeros_like(b), where=a > 0).clip(0, 1)
    minimum = np.minimum(1 + a + b, 1 + t * (a * t + b))
    return minimum, valid


def weighted_p95(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    totals = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(totals, 0.95 * totals[-1])]])


def _uv_stretch(base: MeshSnapshot, before: np.ndarray, after: np.ndarray,
                area: np.ndarray, area_epsilon: float, profile: QAProfile) -> dict:
    if not base.uv_layers:
        return _item("n/a", "no_uv_layers")
    edge = before[:, 1] - before[:, 0]
    length = np.linalg.norm(edge, axis=1)
    geometric = area > area_epsilon
    tangent = np.divide(edge, length[:, None], out=np.zeros_like(edge), where=geometric[:, None])
    parallel = np.sum((before[:, 2] - before[:, 0]) * tangent, axis=1)
    height = np.divide(area, length, out=np.zeros_like(area), where=geometric)
    f1 = np.divide(after[:, 1] - after[:, 0], length[:, None], out=np.zeros_like(edge), where=geometric[:, None])
    f2 = np.divide(after[:, 2] - after[:, 0] - f1 * parallel[:, None], height[:, None],
                   out=np.zeros_like(edge), where=geometric[:, None])
    singular = np.linalg.svd(np.stack((f1, f2), axis=2), compute_uv=False)
    usable = geometric & (singular[:, 1] > 1e-15)
    log_stretch = np.zeros(len(before))
    log_stretch[usable] = np.max(np.abs(np.log(singular[usable])), axis=1)
    rows = {}
    for name, corners in base.uv_layers.items():
        uv = corners[base.triangle_loops].astype(np.float64)
        u, v = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        determinant = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        limit = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) * 1e-12
        valid_uv = np.isfinite(determinant) & (determinant > limit)
        valid = usable & valid_uv
        p95 = weighted_p95(log_stretch[valid], area[valid]) if valid.any() else None
        maximum = float(np.max(log_stretch[valid])) if valid.any() else None
        invalid_uv = int(np.count_nonzero(~valid_uv))
        invalid_geometry = int(np.count_nonzero(valid_uv & ~usable))
        over = not valid.any() or invalid_uv > 0 or invalid_geometry > 0 or p95 > profile.uv_p95 or maximum > profile.uv_max
        rows[name] = _item("fail" if over else "pass", "invalid_uv" if invalid_uv else
                           "singular_geometry" if invalid_geometry else "stretch_limit" if over else None,
                           p95_log=p95, max_log=maximum, invalid_uv=invalid_uv, invalid_geometry=invalid_geometry)
    fail = any(value["status"] == "fail" for value in rows.values())
    return _item("fail" if fail else "pass", "invalid_or_stretched_uv" if fail else None,
                 layers=rows, limits_log={"p95": profile.uv_p95, "max": profile.uv_max})


def _components(triangles: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    parents = list(range(len(triangles)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edges = defaultdict(list)
    for index, (a, b, c) in enumerate(triangles.tolist()):
        for left, right in ((a, b), (b, c), (c, a)):
            key = (min(left, right), max(left, right))
            links = edges[key]
            if links:
                parents[find(index)] = find(links[0][0])
            links.append((index, left < right))
    groups = defaultdict(list)
    for index in range(len(triangles)):
        groups[find(index)].append(index)
    open_components = {find(links[0][0]) for links in edges.values()
                       if len(links) != 2 or links[0][1] == links[1][1]}
    vertex_components = {}
    for key, indices in groups.items():
        for vertex in np.unique(triangles[indices]).tolist():
            previous = vertex_components.setdefault(vertex, key)
            if previous != key:
                open_components.update((key, previous))
    return [(np.array(indices, dtype=np.intp), key not in open_components) for key, indices in groups.items()]


def _volume(points: np.ndarray) -> float:
    shifted = points - points[0, 0]
    return float(np.sum(np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2]))) / 6)


def _area_volume(triangles: np.ndarray, before: np.ndarray, after: np.ndarray, a0: np.ndarray, a1: np.ndarray,
                 invalid: set[int], unknown_collisions: bool, profile: QAProfile, eps: float) -> dict:
    rows = []
    fail = False
    for indices, closed in _components(triangles):
        area0, area1 = float(a0[indices].sum() / 2), float(a1[indices].sum() / 2)
        area_change = abs(area1 / area0 - 1) if area0 > 0 else None
        volume = _item("n/a", "open_nonmanifold_or_inconsistent_winding")
        if closed:
            volume = _item("n/a", "surface_not_verified_valid")
            if not unknown_collisions and not invalid.intersection(indices.tolist()):
                v0, v1 = _volume(before[indices]), _volume(after[indices])
                if abs(v0) > eps**3 and abs(v1) > eps**3:
                    change = abs(v1 / v0 - 1)
                    volume = _item("fail" if change > profile.volume_limit else "pass",
                                   "volume_limit" if change > profile.volume_limit else None,
                                   before_signed_m3=v0, after_signed_m3=v1, relative_change=change)
        area_failed = area_change is not None and area_change > profile.area_limit
        fail |= area_failed or volume["status"] == "fail"
        rows.append({"component": len(rows), "triangles": len(indices), "area_before_m2": area0, "area_after_m2": area1,
                     "area_relative_change": area_change, "area_status": "fail" if area_failed else
                     "pass" if area_change is not None else "n/a", "volume": volume})
    return _item("fail" if fail else "pass", "component_area_or_volume_limit" if fail else None,
                 components=rows, limits={"area": profile.area_limit, "volume": profile.volume_limit})


def _target(before: np.ndarray, after: np.ndarray, indices: list[int], profile: QAProfile) -> dict:
    if not indices:
        return _item("n/a", "no_target_vertices")
    values = {}
    if profile.target_plane is not None:
        normal, d = profile.target_plane
        values["approved_plane_m"] = {name: distance_statistics((points[indices] @ normal + d)[:, None])
                                      for name, points in (("before", before), ("after", after))}
    if profile.clean_positions is not None:
        values["clean_reference_m"] = {name: distance_statistics(points[indices] - profile.clean_positions[indices])
                                       for name, points in (("before", before), ("after", after))}
    worsened = any(row["after"]["rms"] > row["before"]["rms"] + 1e-9 for row in values.values())
    if not values:
        return _item("n/a", "no_approved_plane_or_clean_reference")
    return _item("fail" if worsened else "pass", "target_rms_increased" if worsened else None, **values)


def evaluate(base: MeshSnapshot, patch: CandidatePatch, profile: QAProfile) -> dict:
    checks = {name: _item("n/a", "blocked_by_integrity") for name in CHECKS}
    result = {"status": "rejected", "checks": checks}
    invariant_equal = {key: key in base.hashes and base.hashes[key] == patch.hashes.get(key) for key in INVARIANTS}
    invariant_equal.update({key: base.hashes[key] == patch.hashes.get(key)
                            for key in ("unit_scale", "unit_system", "matrix_world") if key in base.hashes})
    index = {int(vid): i for i, vid in enumerate(base.vid)}
    ids = patch.vertex_ids.tolist()
    plausible = (len(set(ids)) == len(ids) and all(type(vid) is int and vid in index for vid in ids)
                 and patch.positions.shape == (len(ids), 3) and bool(np.isfinite(patch.positions).all()))
    consistent = all(invariant_equal.values()) and plausible
    checks["integrity"] = _item("pass" if consistent else "fail", None if consistent else "invariant_or_patch_mismatch",
                                invariant_equal=invariant_equal, valid_patch=plausible)
    if not consistent:
        result.update(hard_failures=["integrity"], review_reasons=[])
        return result

    raw_after = base.V.copy()
    raw_after[[index[vid] for vid in ids]] = patch.positions
    if "position_hash" in patch.hashes:
        actual = hashlib.sha256(raw_after.astype("<f4").tobytes()).hexdigest()
        if actual != patch.hashes["position_hash"]:
            checks["integrity"] = _item("fail", "patch_does_not_match_observed_positions", invariant_equal=invariant_equal)
            result.update(hard_failures=["integrity"], review_reasons=[])
            return result
    locked = np.array([vid not in profile.editable_vids for vid in index])
    same = np.all(base.V.view(np.uint64) == raw_after.view(np.uint64), axis=1)
    moved_locked = int(np.count_nonzero(locked & ~same))
    V0, V1 = base.V * base.unit_scale, raw_after * base.unit_scale
    displacement = np.linalg.norm(V1 - V0, axis=1)
    peak = float(displacement.max(initial=0))
    checks["locked"] = _item("fail" if moved_locked else "pass", "locked_coordinates_changed" if moved_locked else None,
                             changed=moved_locked, locked_count=int(locked.sum()), maximum_m=float(displacement[locked].max(initial=0)))
    over = peak > profile.max_displacement_m
    checks["displacement"] = _item("fail" if over else "pass", "displacement_budget" if over else None,
                                   maximum_m=peak, budget_m=profile.max_displacement_m)
    eps = max(float(np.linalg.norm(np.ptp(V0, axis=0))), np.finfo(float).eps) * 1e-10
    area_epsilon = eps**2
    before, after = V0[base.triangles], V1[base.triangles]
    _, a0, angle0 = _triangle_stats(before)
    _, a1, angle1 = _triangle_stats(after)
    zero0, zero1 = a0 <= area_epsilon, a1 <= area_epsilon
    small0, small1 = angle0 < profile.min_angle_deg, angle1 < profile.min_angle_deg
    new_zero, new_small = zero1 & ~zero0, small1 & ~small0
    bad_geometry = bool(np.any(new_zero | new_small))
    checks["degenerate"] = _item("fail" if bad_geometry else "pass", "new_degenerate_or_sliver" if bad_geometry else None,
                                 baseline_zero=int(zero0.sum()), new_zero=int(new_zero.sum()), baseline_small=int(small0.sum()),
                                 new_small=int(new_small.sum()), angle_limit_deg=profile.min_angle_deg)
    det, valid = continuous_orientation(before, after, area_epsilon)
    flip = valid & (det < -1e-10)
    uncertain = valid & ~flip & (det <= 1e-10) & ~zero1
    checks["flip"] = _item("fail" if flip.any() or uncertain.any() else "pass",
                           "continuous_orientation_reversed" if flip.any() else "orientation_tolerance" if uncertain.any() else None,
                           count=int(flip.sum()), uncertain=int(uncertain.sum()), min_determinant_ratio=float(det[valid].min()) if valid.any() else None)

    invalid = set(np.flatnonzero(zero0 | zero1 | flip | uncertain).tolist())
    collision_unknown = False
    try:
        old_pairs = intersection_pairs(V0, base.triangles, ~zero0, eps, profile.pair_limit)
        new_pairs = intersection_pairs(V1, base.triangles, ~zero1, eps, profile.pair_limit)
        added = sorted(new_pairs - old_pairs)
        invalid.update(vertex for pair in old_pairs | new_pairs for vertex in pair)
        checks["self_intersection"] = _item("fail" if added else "pass", "new_self_intersection" if added else None,
                                             baseline_pairs=len(old_pairs), candidate_pairs=len(new_pairs), new_pairs=len(added),
                                             new_examples=[{"triangle_indices": pair,
                                                            "face_ids": [int(base.triangle_fid[i]) for i in pair]} for pair in added[:20]])
    except PairBudgetExceeded as error:
        collision_unknown = True
        checks["self_intersection"] = _item("n/a", "pair_budget", detail=str(error))

    checks["uv_stretch"] = _uv_stretch(base, before, after, a0, area_epsilon, profile)
    checks["area_volume"] = _area_volume(base.triangles, before, after, a0, a1, invalid, collision_unknown, profile, eps)
    checks["target_error"] = _target(V0, V1, [index[vid] for vid in profile.target_vids], profile)
    hard = [name for name in CHECKS[:6] if checks[name]["status"] == "fail"]
    if checks["flip"]["reason"] == "orientation_tolerance":
        hard.remove("flip")
    review = [name for name in CHECKS[6:] if checks[name]["status"] == "fail"]
    review += ["flip"] if uncertain.any() else []
    review += ["self_intersection"] if collision_unknown else []
    result.update(status="rejected" if hard else "needs_review" if review else "accepted", hard_failures=hard, review_reasons=review)
    return result
