# SPDX-License-Identifier: Apache-2.0
"""Detect and refine an approved plane family, then run independent P2 QA."""

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import mkstemp

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gre import blender_io as io
from gre.geometry import adjacency, bbox_diagonal, distance_statistics, face_edges, face_geometry, partition_region
from gre.plane import DegeneratePlane, fit_plane
from gre.qa import CandidatePatch, INVARIANTS, QAProfile, evaluate
from gre.ransac import PlaneCandidate, detect_plane, face_neighbors
from gre.relations import Infeasible, Observation, Plane, Relation, project_junctions, relation_error, solve_planes
from gre.report import write_report
from gre.snapshot import file_hash, read_json, write_json


def parse_annotations(plan: dict, mesh) -> tuple[dict[str, np.ndarray], tuple[Relation, ...], set[frozenset[str]]]:
    if (plan.get("schema_version") != 1 or plan.get("unit_scale") != mesh.unit_scale
            or plan.get("matrix_world") != mesh.matrix_world.tolist()):
        raise ValueError("Annotation schema, units or transform mismatch")
    cap = plan.get("max_displacement_m")
    if cap is not None and (type(cap) not in (int, float) or not math.isfinite(cap) or cap < 0):
        raise ValueError("Invalid annotated displacement cap")
    index = {fid: index for index, fid in enumerate(mesh.fid.tolist())}
    seen, names = set(), set()
    regions = {}
    for row in plan["regions"]:
        name, role, ids = row["name"], row["role"], row["face_ids"]
        if (not isinstance(name, str) or not name or name in names or not isinstance(ids, list) or not ids
                or any(type(fid) is not int or fid not in index for fid in ids) or len(set(ids)) != len(ids)
                or seen.intersection(ids) or role not in {"editable", "protected", "unknown"}):
            raise ValueError(f"Invalid or overlapping face annotation in {name}")
        names.add(name)
        seen.update(ids)
        if role == "editable" and row.get("semantic") == "plane":
            regions[name] = np.array(sorted(index[fid] for fid in ids), dtype=np.intp)
    if not regions:
        raise ValueError("No approved editable plane region")
    links, declared = [], set()
    for row in plan["relations"]:
        kind = row["type"]
        keys = {"type", "a", "b", "distance_m"} if kind == "offset" else {"type", "a", "b"}
        if set(row) != keys or kind not in {"parallel", "perpendicular", "offset"}:
            raise ValueError("Unknown relation type or fields")
        a, b = row["a"], row["b"]
        distance = row.get("distance_m", 0.0)
        key = (kind, frozenset((a, b)))
        if (a not in regions or b not in regions or a == b or key in declared
                or type(distance) not in (int, float) or not math.isfinite(distance)):
            raise ValueError("Invalid or duplicate approved relation")
        declared.add(key)
        links.append(Relation(kind, a, b, distance))
    junctions = set()
    for group in plan.get("junctions", []):
        names = frozenset(group)
        if len(names) != len(group) or len(names) < 2 or not names <= regions.keys() or names in junctions:
            raise ValueError("Invalid or duplicate approved junction")
        junctions.add(names)
    return regions, tuple(links), junctions


def detect_regions(V: np.ndarray, mesh, regions: dict, args, seed: int) -> tuple[dict, dict]:
    normals, areas, _ = face_geometry(V, mesh.triangles, mesh.triangle_faces, len(mesh.F))
    centers = np.array([V[list(face)].mean(axis=0) for face in mesh.F])
    found, observations = {}, {}
    for name, faces in regions.items():
        graph = face_neighbors(tuple(mesh.F[i] for i in faces))
        candidate = detect_plane(centers[faces], normals[faces], areas[faces], graph, seed=seed,
                                 distance_m=args.ransac_distance, minimum_coverage=args.min_coverage)
        if candidate is not None:
            found[name] = candidate
            ids = faces[candidate.inliers]
            observations[name] = Observation(Plane(candidate.normal, candidate.d), centers[ids], areas[ids])
    return found, observations


def detect_summary(found: dict[str, PlaneCandidate], regions: dict, fid: np.ndarray) -> dict:
    return {name: {"normal": candidate.normal.tolist(), "d_m": candidate.d,
                   "coverage": candidate.coverage, "fit_rms_m": candidate.rms_m,
                   "minimum_normal_cosine": candidate.minimum_normal_cosine, "iterations": candidate.iterations,
                   "estimated_iterations": candidate.estimated_iterations, "inlier_face_ids": fid[regions[name][candidate.inliers]].tolist()}
            for name, candidate in found.items()}


def footprint(mesh, V: np.ndarray, regions: dict[str, np.ndarray]) -> tuple[dict, dict, set, float]:
    selected = {int(i) for ids in regions.values() for i in ids}
    edges = face_edges(tuple(mesh.F[index] for index in sorted(selected)))
    graph, h = adjacency(V, edges)
    outside = {vertex for index, face in enumerate(mesh.F) if index not in selected for vertex in face}
    outside.update(vertex for edge in mesh.edges if tuple(sorted(edge)) not in edges for vertex in edge)
    memberships = defaultdict(set)
    for name, face_indices in regions.items():
        for face_index in face_indices:
            for vertex in mesh.F[face_index]:
                memberships[vertex].add(name)
    return graph, memberships, graph.keys() & outside, h


def with_locked_support(observations: dict, membership: dict, boundary: set[int], V: np.ndarray,
                        tolerance_m: float) -> dict:
    result = {}
    for name, observation in observations.items():
        indices = [index for index in sorted(boundary) if name in membership[index]]
        anchors = V[indices]
        if len(anchors):
            consistent = np.abs(anchors @ observation.plane.normal + observation.plane.d) <= tolerance_m
            anchors = anchors[consistent]
        if len(anchors):
            # Boundary observations guide the fit and remain hard-locked in the position solve.
            weight = 48 * observation.weights.sum() / len(anchors)
            result[name] = Observation(observation.plane, np.vstack((observation.points, anchors)),
                                       np.concatenate((observation.weights, np.full(len(anchors), weight))))
        else:
            result[name] = observation
    return result


def measure(V: np.ndarray, vertices: dict[str, np.ndarray], links: tuple[Relation, ...], targets: dict) -> dict:
    planes, rows = {}, {}
    for name, indices in vertices.items():
        points = V[indices]
        normal, d, rms, _ = fit_plane(points, np.ones(len(indices)))
        planes[name] = Plane(normal, d)
        row = {"vertices": len(indices), "normal": normal.tolist(), "d_m": d, "measured_fit_rms_m": rms}
        if name in targets:
            target = targets[name]
            direction = np.asarray(target["normal"], dtype=np.float64)
            scale = float(np.linalg.norm(direction))
            offset = target["d_m"]
            if direction.shape != (3,) or not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
                raise ValueError(f"Invalid independent approved target for {name}")
            row["approved_plane_error_m"] = distance_statistics(((points @ direction + offset) / scale)[:, None])
        rows[name] = row
    return {"planes": rows, "relations": [relation_error(planes, link) for link in links]}


def stable_detection(args, source, V, regions, first: dict) -> dict:
    rows = {seed: detect_regions(V, source, regions, args, seed)[0]
            if seed != args.seed else first for seed in args.stability_seeds}
    peak, support_equal = {}, {}
    for name in regions:
        detected = [candidates[name] for candidates in rows.values() if name in candidates]
        if len(detected) != len(rows):
            peak[name] = None
            support_equal[name] = False
            continue
        normals = [candidate.normal for candidate in detected]
        support_equal[name] = all(np.array_equal(candidate.inliers, detected[0].inliers) for candidate in detected[1:])
        peak[name] = max((float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), abs(a @ b))))
                          for a, b in itertools.combinations(normals, 2)), default=0.0)
    return {"seeds": args.stability_seeds, "region_max_pairwise_normal_deg": peak, "region_identical_support": support_equal,
            "normals": {seed: {name: candidate.normal.tolist() for name, candidate in row.items()} for seed, row in rows.items()}}


def run(args, folder: Path, metrics: dict) -> None:
    source_file = args.source.resolve(strict=True)
    snapshot_hash = file_hash(source_file)
    annotation_hash = file_hash(args.annotations)
    plan = read_json(args.annotations)
    name = plan["object_name"]
    io.open_blend(source_file)
    obj = io.mesh_object(name)
    source = io.read_mesh(obj)
    baseline = io.qa_snapshot(obj, source)
    V = source.V * source.unit_scale
    regions, links, junctions = parse_annotations(plan, source)
    found, observations = detect_regions(V, source, regions, args, args.seed)
    metrics.update(asset_id=plan["asset_id"], input_file_sha256=snapshot_hash, annotations_sha256=annotation_hash,
                   source_hash=source.hashes["source_hash"], unit_scale=source.unit_scale, matrix_world=source.matrix_world.tolist(),
                   seed=args.seed, ransac={"distance_m": args.ransac_distance, "minimum_coverage": args.min_coverage,
                                         "normal_tolerance_deg": 12, "confidence": 0.999, "cap": 256},
                   detected=detect_summary(found, regions, source.fid))
    if args.stability_seeds:
        metrics["stability"] = stable_detection(args, source, V, regions, found)
    if found.keys() != regions.keys():
        metrics.update(status="needs_review", reason="insufficient_connected_plane_support", missing=sorted(regions.keys() - found.keys()))
        return
    if args.detect_only:
        metrics.update(status="candidate", reason="detection_only")
        return

    reference_file = (args.reference or args.annotations.with_name("cabinet_clean.blend")).resolve(strict=True)
    out = args.out.resolve()
    if out in {source_file, reference_file} or os.path.lexists(out):
        raise ValueError("Output must be a new file distinct from the source and reference")
    approved, approved_members, boundary, h = footprint(source, V, regions)
    active = {name: ids[found[name].inliers] for name, ids in regions.items()}
    graph, memberships, active_boundary, _ = footprint(source, V, active)
    rings = plan["transition_rings"]
    if type(rings) is not int or rings < 0:
        raise ValueError("Invalid transition_rings")
    core, blend, taper = partition_region(graph, active_boundary, rings)
    if not len(core):
        metrics.update(status="needs_review", reason="no_editable_core")
        return
    targets = plan.get("approved_targets_for_qa", {})
    core_sets = {name: np.array([index for index in core if name in memberships[index]], dtype=np.intp) for name in regions}
    all_sets = {name: np.array(sorted(index for index in approved if name in approved_members[index]), dtype=np.intp)
                for name in regions}

    try:
        metrics["relations_before"] = measure(V, all_sets, links, targets)
        observations = with_locked_support(observations, memberships, active_boundary, V, args.ransac_distance)
        plane_family, steps = solve_planes(observations, links)
        metrics["joint_fit"] = {"lm_steps": steps, "relations": [relation_error(plane_family, link) for link in links]}
        moving = np.concatenate((core, blend))
        positions, junction_metrics = project_junctions(
            V, moving, np.concatenate((np.ones(len(core)), 1 - (1 - taper)**8)), memberships,
            plane_family, junctions,
        )
        metrics["junctions"] = junction_metrics
    except (Infeasible, DegeneratePlane) as error:
        metrics.update(status="rejected", reason="infeasible", detail=str(error))
        return

    stored = source.V.copy()
    stored[moving] = (positions[moving] / source.unit_scale).astype(np.float32)
    changed = np.any(stored != source.V, axis=1)
    if not changed.any():
        metrics.update(status="no_action", reason="no_stored_coordinate_change")
        return
    reference_hash = file_hash(reference_file)
    io.open_blend(reference_file)
    reference_obj = io.mesh_object(name)
    reference_data = io.read_mesh(reference_obj)
    reference = io.qa_snapshot(reference_obj, reference_data)
    if (plan["source_hash"] != reference_data.hashes["source_hash"]
            or any(reference.hashes[key] != baseline.hashes[key] for key in INVARIANTS)):
        raise ValueError("Reference does not match annotation or baseline structure")
    if any(file_hash(path) != digest for path, digest in ((source_file, snapshot_hash), (args.annotations, annotation_hash),
                                                         (reference_file, reference_hash))):
        raise ValueError("P3 input changed during evaluation")
    io.open_blend(source_file)
    obj = io.mesh_object(name)
    io.write_positions(obj, dict(zip(source.vid[changed].tolist(), stored[changed])), source.hashes["source_hash"])
    after = io.read_mesh(obj)
    observed = io.qa_snapshot(obj, after)
    result = after.V * after.unit_scale
    p2 = evaluate(baseline, CandidatePatch.between(baseline, observed), QAProfile(
        frozenset(int(source.vid[index]) for index in approved.keys() - boundary),
        min(args.budget_m, plan["max_displacement_m"]) if plan.get("max_displacement_m") is not None else args.budget_m,
        target_vids=tuple(int(source.vid[i]) for i in sorted(approved)), clean_positions=reference.V * reference.unit_scale,
    ))
    metrics.update(reference_file_sha256=reference_hash, requested_budget_m=args.budget_m, local_median_edge_m=h,
                   bbox_diagonal_m=bbox_diagonal(V), core_vertices=len(core), transition_vertices=len(blend), qa=p2)
    metrics["protected"] = {}
    choices = {fid: index for index, fid in enumerate(source.fid.tolist())}
    for row in plan["regions"]:
        if row["role"] == "protected":
            ids = sorted({index for fid in row["face_ids"] for index in source.F[choices[fid]]})
            metrics["protected"][row["name"]] = {"vertices": len(ids),
                "maximum_m": float(np.linalg.norm(result[ids] - V[ids], axis=1).max(initial=0)),
                "coordinates_exact": after.V[ids].tobytes() == source.V[ids].tobytes()}
    try:
        metrics["relations_after_core"] = measure(result, core_sets, links, targets)
        metrics["relations_after_full"] = measure(result, all_sets, links, targets)
    except DegeneratePlane as error:
        metrics.update(status="needs_review" if p2["status"] != "rejected" else "rejected",
                       reason="insufficient_core_for_verification", detail=str(error))
        return
    if p2["status"] != "accepted":
        metrics.update(status=p2["status"], reason="p2_gate")
        return
    measured = metrics["relations_after_full"]["relations"] + metrics["relations_after_core"]["relations"]
    if any(row["deviation_deg"] >= 0.1 or row.get("offset_error_m", 0) >= 0.001 for row in measured):
        metrics.update(status="needs_review", reason="measured_relation_out_of_tolerance")
        return
    if args.stability_seeds and any(value is None or value >= 0.05 for value in metrics["stability"]["region_max_pairwise_normal_deg"].values()):
        metrics.update(status="needs_review", reason="seed_stability")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, path = mkstemp(prefix=".gre-p3-", suffix=".blend", dir=out.parent)
    os.close(descriptor)
    staged = Path(path)
    try:
        io.save_copy(staged, source_file, out)
        io.open_blend(staged)
        obj = io.mesh_object(name)
        saved = io.read_mesh(obj)
        reopened = io.qa_snapshot(obj, saved)
        if reopened.hashes != observed.hashes or reopened.V.tobytes() != observed.V.tobytes():
            raise RuntimeError("Staged cabinet failed the Blender disk round trip")
        if not args.no_images:
            qa_folder = folder / "p2"
            qa_folder.mkdir()
            io.render_previews(obj, source, saved, reference_data, qa_folder, qa=True, fit_frame=True)
        if file_hash(source_file) != snapshot_hash:
            raise ValueError("Source changed before publishing")
        sha = file_hash(staged)
        os.link(staged, out)
        metrics.update(status="accepted", output=out.name, output_file_sha256=sha)
    finally:
        staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-seeds", nargs="+", type=int, default=[])
    parser.add_argument("--ransac-distance", type=float, default=0.004)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--budget-m", type=float, default=0.006)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if (not args.detect_only and (args.out is None or args.out.suffix.lower() != ".blend")
            or not math.isfinite(args.budget_m) or args.budget_m < 0 or not math.isfinite(args.ransac_distance)
            or args.ransac_distance <= 0 or not 0 < args.min_coverage <= 1
            or any(not 0 <= seed < 2**64 for seed in [args.seed, *args.stability_seeds])
            or len(set(args.stability_seeds)) != len(args.stability_seeds)):
        parser.error("Invalid output, tolerances or seeds")
    folder = args.report.resolve()
    if args.out and (args.out.resolve() == folder or args.out.resolve() in folder.parents):
        parser.error("Report folder may not be inside the output .blend path")
    folder.mkdir(parents=True, exist_ok=False)
    metrics = {"stage": "P3"}
    try:
        run(args, folder, metrics)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        metrics.update(status="unsupported" if isinstance(error, io.Unsupported) else "solver_failed", reason=str(error))
    write_json(folder / "metrics.json", metrics)
    if "qa" in metrics:
        qa_folder = folder / "p2"
        qa_folder.mkdir(exist_ok=True)
        write_report(qa_folder, metrics["qa"])
    print(json.dumps({key: metrics[key] for key in ("stage", "status", "reason", "output") if key in metrics},
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
