"""World-space welded shell identity, gap probes and local surface coverage."""

from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
from itertools import accumulate
import math
import random

import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _bounds(points):
    return [[op(p[i] for p in points) for i in range(3)] for op in (min, max)]


def _diagonal(bounds):
    return (Vector(bounds[1]) - Vector(bounds[0])).length


@contextmanager
def welded_surface(objects, weld_dist=1e-4):
    from topology_inspector import _gather_world_bmesh

    bm = _gather_world_bmesh(objects)
    try:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=weld_dist)
        bm.verts.index_update()
        bm.faces.index_update()
        components = []
        seen = set()
        for face in bm.faces:
            if face.index in seen:
                continue
            shell = [face]
            seen.add(face.index)
            for current in shell:
                for vert in current.verts:
                    for adjacent in vert.link_faces:
                        if adjacent.index not in seen:
                            shell.append(adjacent)
                            seen.add(adjacent.index)
            components.append(shell)
        yield bm, components
    finally:
        bm.free()


def sample_shells(objects, weld_dist=1e-4):
    shells = []
    for obj in sorted(objects, key=lambda ob: ob.name):
        with welded_surface([obj], weld_dist) as (_, components):
            items = []
            for faces in components:
                bounds = _bounds([v.co for v in {v for f in faces for v in f.verts}])
                areas = list(accumulate(f.calc_area() for f in faces))
                if areas[-1] <= 0:
                    continue
                rng = random.Random(0)
                size = max(32, min(128, len(faces)))
                samples = []
                for i in range(size):
                    f = faces[bisect_left(areas, areas[-1] * (i + 0.5) / size)]
                    a, b, c = (v.co for v in f.verts)
                    radius, blend = math.sqrt(rng.random()), rng.random()
                    samples.append(list(a * (1 - radius) + b * (radius * (1 - blend)) + c * (radius * blend)))
                items.append({"bbox": bounds, "samples": samples, "surface_area": areas[-1]})
            for i, item in enumerate(sorted(items, key=lambda row: (row["bbox"], -row["surface_area"]))):
                shells.append({"shell_id": f"{obj.name}:{i}", **item})
    return shells


def check_shell_contract(objects, shells, contract=None, *, phase="final", max_p95_pct=2.2):
    contract = contract or {}
    groups = contract.get("separation_groups", [])
    rays = contract.get("gap_rays", [])
    envelope = contract.get("single_shell_envelope", False)
    requires_gaps = bool(groups or rays or contract.get("require_gap_preservation")
                         or "require_gap_preservation" in contract.get("asset_tags", []))
    issues = []
    coverage = []
    missing = []
    assignments = defaultdict(set)
    blocked = []
    if (requires_gaps and (envelope or len(groups) < 2 or not rays)) or not shells:
        issues.append("invalid_shell_contract")

    group_of = {}
    known_shells = {shell["shell_id"] for shell in shells}
    for group in groups:
        group_id = group["id"]
        members = group["shell_ids"]
        if (not members or group_id in assignments
                or any(shell not in known_shells or shell in group_of for shell in members)):
            issues.append("invalid_separation_groups")
        assignments[group_id]
        group_of.update((shell, group_id) for shell in members)

    limit = max(1e-6, _diagonal(_bounds([p for s in shells for p in s["bbox"]])) * max_p95_pct / 100) if shells else 1e-6
    with welded_surface(objects) as (bm, components):
        tree = BVHTree.FromBMesh(bm) if components else None
        face_to_shell = {f.index: i for i, comp in enumerate(components) for f in comp}
        if envelope and len(components) != 1:
            issues.append("expected_single_shell_envelope")
        for shell in shells:
            sid = shell["shell_id"]
            if envelope or (phase == "preprocess" and sid not in group_of):
                continue
            threshold = max(1e-6, min(limit, _diagonal(shell["bbox"]) * 0.08))
            candidates = [tree.find_nearest(Vector(point), threshold) for point in shell["samples"]] if tree else []
            hits = [(index, dist) for _, _, index, dist in candidates if index is not None]
            distances = sorted(dist for _, dist in hits)
            total = len(shell["samples"])
            p95_index = math.ceil(0.95 * total) - 1
            p95 = distances[p95_index] if p95_index < len(distances) else None
            ratio = len(hits) / total
            coverage.append({"shell_id": sid, "bbox": shell["bbox"], "p95_distance": p95,
                             "hit_rate": ratio, "threshold": threshold})
            if phase != "preprocess" and (p95 is None or ratio < 0.95):
                missing.append(sid)
                issues.append(f"shell_coverage_missing:{sid}")
            if sid in group_of:
                assignments[group_of[sid]].update(face_to_shell[index] for index, _ in hits)
                if ratio < 0.95:
                    issues.append(f"separation_group_missing:{sid}")

        for i, ray in enumerate(rays):
            start, end = Vector(ray["start"]), Vector(ray["end"])
            direction = end - start
            if not all(math.isfinite(v) for v in (*start, *end)) or direction.length <= 1e-7:
                issues.append(f"invalid_gap_ray:{i}")
            elif tree and tree.ray_cast(start, direction.normalized(), direction.length)[0] is not None:
                blocked.append(ray.get("id", i))

        assigned = [index for indices in assignments.values() for index in indices]
        if blocked or len(assigned) != len(set(assigned)):
            issues.append("gap_bridged_or_merged")
        if any(not indices for indices in assignments.values()):
            issues.append("separation_group_missing")
        return {"passed": not issues, "phase": phase, "issues": issues, "blocked_gap_rays": blocked,
                "shells_out": len(components), "local_coverage": coverage, "shell_coverage_missing": missing,
                "separation_groups_out": {key: sorted(ids) for key, ids in assignments.items()},
                "single_shell_envelope": envelope}
