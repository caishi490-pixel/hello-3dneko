# SPDX-License-Identifier: Apache-2.0
"""Blender entry point for independent P2 QA, commit and exact rollback."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from tempfile import mkstemp

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gre import blender_io as io
from gre.geometry import adjacency, bbox_diagonal, face_edges
from gre.qa import CandidatePatch, INVARIANTS, QAProfile, evaluate
from gre.report import write_report
from gre.snapshot import commit, file_hash, read_json, rollback, save_snapshot


def _open(path: Path, object_name: str):
    io.open_blend(path)
    obj = io.mesh_object(object_name)
    data = io.read_mesh(obj)
    return data, io.qa_snapshot(obj, data)


def load_profile(args, plan: dict, baseline, reference) -> QAProfile:
    if plan.get("schema_version") != 1 or plan.get("unit_scale") != baseline.unit_scale:
        raise ValueError("Annotation schema or unit scale mismatch")
    if "matrix_world" in plan and plan["matrix_world"] != baseline.matrix_world.tolist():
        raise ValueError("Annotation transform mismatch")
    if plan.get("source_hash") != reference.hashes["source_hash"]:
        raise ValueError("Annotation source hash does not match the approved reference")
    if any(reference.hashes[key] != baseline.hashes[key] for key in io.INVARIANTS):
        raise ValueError("Reference and baseline structures differ")
    vertex_by_id = {int(vid): i for i, vid in enumerate(reference.vid)}
    aligned = reference.V[[vertex_by_id[int(vid)] for vid in baseline.vid]] * baseline.unit_scale
    known = set(baseline.fid.tolist())
    names = set()
    seen = set()
    selected = set()
    for region in plan["regions"]:
        ids = region["face_ids"]
        name, role = region["name"], region["role"]
        if (not isinstance(ids, list) or not ids or any(type(fid) is not int or fid not in known for fid in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids) or name in names
                or role not in {"editable", "protected", "unknown"}):
            raise ValueError(f"Invalid, duplicate or overlapping annotation: {name}")
        seen.update(ids)
        names.add(name)
        if name == args.region and role == "editable":
            selected = set(ids)
    if not selected:
        raise ValueError("P2 requires an approved editable region")
    faces = tuple(face for fid, face in zip(baseline.fid.tolist(), baseline.F) if fid in selected)
    approved_edges = face_edges(faces)
    graph, h = adjacency(baseline.V * baseline.unit_scale, approved_edges)
    region_vertices = set(graph)
    outside = {vertex for fid, face in zip(baseline.fid.tolist(), baseline.F) if fid not in selected for vertex in face}
    outside.update(vertex for edge in baseline.edges if tuple(sorted(edge)) not in approved_edges for vertex in edge)
    approved = frozenset(int(baseline.vid[i]) for i in region_vertices - outside)
    D = bbox_diagonal(baseline.V * baseline.unit_scale)
    config = read_json(args.profile) if args.profile else {}
    keys = {"max_displacement_m", "min_angle_deg", "uv_p95", "uv_max", "area_limit", "volume_limit", "pair_limit", "approved_plane"}
    if config.keys() - keys:
        raise ValueError(f"Unknown QA profile fields: {sorted(config.keys() - keys)}")
    table = plan.get("asset_id") == "gre_table_v1"
    budget = config.pop("max_displacement_m", min(h, 0.011 * D) if table else min(0.1 * h, 0.002 * D))
    cap = plan.get("max_displacement_m")
    if cap is not None:
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap < 0:
            raise ValueError("Invalid annotated displacement cap")
        budget = min(budget, cap) if type(budget) in (int, float) and math.isfinite(budget) else budget
    target = config.pop("approved_plane", None)
    plane = None
    if target is not None:
        if set(target) != {"normal", "d_m"} or type(target["d_m"]) not in (int, float):
            raise ValueError("Invalid approved_plane")
        normal = np.asarray(target["normal"], dtype=np.float64)
        length = np.linalg.norm(normal)
        if normal.shape != (3,) or not np.isfinite(length) or length <= 0 or not math.isfinite(target["d_m"]):
            raise ValueError("Invalid approved_plane")
        plane = (normal / length, target["d_m"] / length)
    limits = {"max_displacement_m": budget, **config}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in limits.values()):
        raise ValueError("QA thresholds must be finite nonnegative numbers")
    if "pair_limit" in limits and (type(limits["pair_limit"]) is not int or not limits["pair_limit"]):
        raise ValueError("pair_limit must be a positive integer")
    if limits.get("min_angle_deg", 0) > 60:
        raise ValueError("min_angle_deg must not exceed 60")
    return QAProfile(approved, target_vids=tuple(int(baseline.vid[i]) for i in sorted(region_vertices)),
                     target_plane=plane, clean_positions=aligned, **limits)


def _commit(args, source, source_hash, baseline, patch, observed, metrics):
    if metrics["status"] == "rejected" or (metrics["status"] == "needs_review" and not args.review_note.strip()):
        metrics["transaction"] = {"status": "not_committed", "reason": "qa_rejected_or_review_note_required"}
        return
    if not len(patch.vertex_ids):
        metrics["transaction"] = {"status": "not_committed", "reason": "no_changed_coordinates"}
        return
    if metrics["checks"]["self_intersection"]["status"] == "n/a":
        metrics["transaction"] = {"status": "not_committed", "reason": "collision_check_incomplete"}
        return
    working = args.workspace / "working" / args.working_file
    if (Path(args.working_file).name != args.working_file or working.suffix != ".blend"
            or working.parent.resolve() != working.parent or os.path.lexists(working)):
        raise ValueError("Commit requires a new plain .blend filename under workspace/working")
    working.parent.mkdir(parents=True, exist_ok=True)
    descriptor, path = mkstemp(prefix=".gre-candidate-", suffix=".blend", dir=working.parent)
    os.close(descriptor)
    staged = Path(path)
    try:
        if file_hash(source) != source_hash:
            raise ValueError("Source changed after QA")
        io.open_blend(source)
        obj = io.mesh_object(metrics["object"])
        io.write_positions(obj, dict(zip(patch.vertex_ids.tolist(), patch.positions)), baseline.hashes["source_hash"])
        io.save_copy(staged, source, working)
        _, saved = _open(staged, metrics["object"])
        if (any(saved.hashes[name] != baseline.hashes[name] for name in INVARIANTS)
                or saved.V.tobytes() != observed.V.tobytes() or file_hash(source) != source_hash):
            raise ValueError("Staged .blend failed the independent round-trip check")
        receipt = commit(args.workspace, metrics["asset_id"], source_hash, staged, args.working_file,
                         metrics["status"], args.review_note)
        metrics["transaction"] = {"status": "committed", **receipt}
    finally:
        staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("check", "commit"):
        entry = actions.add_parser(name)
        entry.add_argument("--source", type=Path, required=True)
        entry.add_argument("--candidate", type=Path, required=True)
        entry.add_argument("--annotations", type=Path, required=True)
        entry.add_argument("--region", required=True)
        entry.add_argument("--reference", type=Path)
        entry.add_argument("--profile", type=Path)
        entry.add_argument("--report", type=Path, required=True)
        entry.add_argument("--no-images", action="store_true")
        if name == "commit":
            entry.add_argument("--working-file", required=True)
            entry.add_argument("--review-note", default="")
    restore = actions.add_parser("rollback")
    restore.add_argument("--audit-id", required=True)
    restore.add_argument("--expected-current-hash", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    args.workspace = args.workspace.resolve()
    if args.action == "rollback":
        print(json.dumps({"status": "rolled_back", **rollback(args.workspace, args.audit_id, args.expected_current_hash)}))
        return
    annotation_hash = file_hash(args.annotations)
    profile_hash = file_hash(args.profile) if args.profile else None
    plan = read_json(args.annotations)
    object_name = plan.get("object_name", "Table")
    source = args.source.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    reference_path = (args.reference or args.annotations.with_name("table_clean.blend")).resolve(strict=True)
    source_hash, candidate_hash, reference_hash = map(file_hash, (source, candidate, reference_path))
    before, baseline = _open(source, object_name)
    after, observed = _open(candidate, object_name)
    reference = before if reference_path == source else _open(reference_path, object_name)[0]
    profile = load_profile(args, plan, before, reference)
    files = {source: source_hash, candidate: candidate_hash, reference_path: reference_hash,
             args.annotations: annotation_hash}
    if args.profile:
        files[args.profile] = profile_hash
    if any(file_hash(path) != expected for path, expected in files.items()):
        raise ValueError("Input changed while reading P2 data")
    folder = args.report.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    meta = save_snapshot(args.workspace, plan["asset_id"], baseline, source, source_hash)
    patch = CandidatePatch.between(baseline, observed)
    metrics = evaluate(baseline, patch, profile)
    plane = profile.target_plane
    metrics.update(stage="P2", asset_id=meta["asset_id"], object=object_name, region=args.region,
                   source_file_sha256=source_hash, candidate_file_sha256=candidate_hash,
                   reference_file_sha256=reference_hash, annotations_sha256=annotation_hash,
                   profile_sha256=profile_hash,
                   approved_plane=None if plane is None else {"normal": plane[0].tolist(), "d_m": plane[1]},
                   baseline_hashes=baseline.hashes, candidate_hashes=observed.hashes,
                   snapshot_npz_sha256=meta["snapshot_npz_sha256"],
                   review_thresholds={"displacement_m": profile.max_displacement_m, "min_angle_deg": profile.min_angle_deg,
                                      "uv_p95": profile.uv_p95, "uv_max": profile.uv_max, "area": profile.area_limit,
                                      "volume": profile.volume_limit, "pair_limit": profile.pair_limit})
    if not args.no_images and metrics["status"] != "rejected":
        if file_hash(candidate) != candidate_hash:
            raise ValueError("Candidate changed before rendering P2 previews")
        io.open_blend(candidate)
        io.render_previews(io.mesh_object(object_name), before, after, reference, folder, qa=True)
    if any(file_hash(path) != expected for path, expected in files.items()):
        raise ValueError("Input changed during P2 evaluation")
    if args.action == "commit":
        _commit(args, source, source_hash, baseline, patch, observed, metrics)
    write_report(folder, metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
