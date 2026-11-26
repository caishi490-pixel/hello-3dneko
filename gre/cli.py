# SPDX-License-Identifier: Apache-2.0
"""Blender entry point for GRE P1's native, positions-only plane refinement."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gre import blender_io as io
from gre.geometry import (adjacency, bbox_diagonal, closed_locked_rim, distance_statistics,
                          face_edges, face_geometry, partition_region)
from gre.plane import DegeneratePlane, fit_locked_rim, fit_plane, project_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clean", type=Path, help="Default: table_clean.blend beside the annotations")
    parser.add_argument("--max-planarity-ratio", type=float, default=0.005)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not math.isfinite(args.max_planarity_ratio) or not 0 < args.max_planarity_ratio < 1:
        parser.error("--max-planarity-ratio must be between zero and one")
    if args.out.suffix.lower() != ".blend":
        parser.error("--out must have the .blend suffix")
    return args


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_faces(plan: dict, name: str, data: io.MeshData) -> np.ndarray:
    if plan.get("schema_version") != 1 or plan.get("unit_scale") != data.unit_scale:
        raise io.Unsupported("Unsupported annotation schema or mismatched unit scale")
    if type(plan.get("transition_rings")) is not int or plan["transition_rings"] < 0:
        raise io.Unsupported("transition_rings must be a nonnegative integer")
    if "matrix_world" in plan and plan["matrix_world"] != data.hashes["matrix_world"]:
        raise io.Unsupported("Annotation transform does not match the source")
    if "max_displacement_m" in plan and plan["max_displacement_m"] is not None:
        cap = plan["max_displacement_m"]
        if type(cap) not in (float, int) or not math.isfinite(cap) or cap < 0:
            raise io.Unsupported("max_displacement_m must be null or nonnegative metres")
    face_index = {fid: index for index, fid in enumerate(data.fid.tolist())}
    seen = set()
    names = set()
    selected = None
    for region in plan["regions"]:
        ids = region["face_ids"]
        role, label = region["role"], region["name"]
        if (label in names or role not in {"editable", "protected", "unknown"} or not ids
                or any(type(fid) is not int or fid not in face_index for fid in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise io.Unsupported(f"Invalid or overlapping face annotation in {label}")
        names.add(label)
        seen.update(ids)
        if label == name and role == "editable" and region.get("semantic") == "plane":
            selected = np.array(sorted(face_index[fid] for fid in ids), dtype=np.intp)
    if selected is None:
        raise io.Unsupported("Select an editable region with plane semantics")
    return selected


def run(args: argparse.Namespace, folder: Path, staged_blend: Path, out: Path, metrics: dict) -> bool:
    source = io.input_path()
    reference_path = (args.clean or args.annotations.with_name("table_clean.blend")).resolve()
    source_hash = file_hash(source)
    if out in {source, reference_path} or os.path.lexists(out):
        raise io.Unsupported("Output must be a new .blend distinct from the source and reference")
    if not reference_path.is_file():
        raise io.Unsupported("A clean .blend reference is required")
    plan = json.loads(args.annotations.read_text(encoding="utf-8"))
    name = plan.get("object_name", "Table")
    source_data = io.read_mesh(io.mesh_object(name))
    face_indices = select_faces(plan, args.region, source_data)
    metrics.update(asset_id=plan.get("asset_id"), object=name, source=source.name, clean=reference_path.name,
                   source_file_sha256=source_hash, input_hashes=source_data.hashes,
                   unit_scale=source_data.unit_scale, matrix_world=source_data.matrix_world.tolist(),
                   transition_rings=plan["transition_rings"], max_planarity_ratio=args.max_planarity_ratio)
    io.open_blend(reference_path)
    reference = io.read_mesh(io.mesh_object(name))
    if (plan.get("source_hash") != reference.hashes["source_hash"]
            or any(source_data.hashes[key] != reference.hashes[key] for key in io.INVARIANTS)
            or source_data.unit_scale != reference.unit_scale):
        raise io.Unsupported("Reference, source and annotations do not describe the same fixture")
    metrics["clean_source_hash"] = reference.hashes["source_hash"]
    reference_idx = {vid: index for index, vid in enumerate(reference.vid.tolist())}
    ref = reference.V[[reference_idx[vid] for vid in source_data.vid.tolist()]] * source_data.unit_scale

    V = source_data.V * source_data.unit_scale
    selection = set(face_indices.tolist())
    faces = tuple(source_data.F[index] for index in face_indices)
    edge_set = face_edges(faces)
    graph, h = adjacency(V, edge_set)
    region = np.array(sorted(graph), dtype=np.intp)
    outside = {vertex for index, face in enumerate(source_data.F) if index not in selection for vertex in face}
    outside.update(vertex for edge in source_data.edges if tuple(sorted(edge)) not in edge_set for vertex in edge)
    boundary = graph.keys() & outside
    core, ring, falloff = partition_region(graph, boundary, plan["transition_rings"])
    locked = np.ones(len(V), dtype=bool)
    locked[core] = False
    locked[ring] = False
    D = bbox_diagonal(V)
    before = distance_statistics(V[region] - ref[region])
    metrics.update(reference_before_m=before, h_m=h, D_m=D, region_faces=len(selection), region_vertices=len(region),
                   core_vertices=len(core), ring_vertices=len(ring), locked_vertices=int(locked.sum()))
    if not len(core) or np.array_equal(V[region], ref[region]):
        metrics.update(status="no_action", reason="already_at_reference" if not before["max"] else "no_editable_core")
        return False

    triangles = np.isin(source_data.triangle_faces, face_indices)
    _, _, weights = face_geometry(V, source_data.triangles[triangles], source_data.triangle_faces[triangles], len(source_data.F))
    try:
        area_fit = fit_plane(V[region], weights[region])
    except DegeneratePlane as error:
        metrics.update(status="rejected", reason="not_planar", detail=str(error))
        return False
    normal, offset, rms, ratio = area_fit
    metrics.update(fit_plane={"normal": normal.tolist(), "d_m": offset, "fit_rms_m": rms, "eigenvalue_ratio": ratio})
    if ratio > args.max_planarity_ratio:
        metrics.update(status="rejected", reason="not_planar")
        return False

    rim = closed_locked_rim(faces, boundary)
    rim_fit = None if rim is None else fit_locked_rim(V[region], weights[region], V[rim], weights[rim], area_fit, h)
    target_source = "area_fit"
    passes = 8
    if rim_fit is not None:
        normal, offset, rim_rms, _ = rim_fit
        target_source = "locked_rim"
        passes = 16
        metrics["locked_rim_anchor"] = {"vertices": len(rim), "fit_rms_m": rim_rms}
    metrics["target_plane"] = {"source": target_source, "normal": normal.tolist(), "d_m": offset}
    projected = project_region(V, core, ring, normal, offset, falloff, passes=passes) / source_data.unit_scale
    metrics["projection_passes"] = passes
    moving = np.concatenate((core, ring))
    stored = projected[moving].astype(np.float32).astype(np.float64)
    changed = np.any(stored != source_data.V[moving], axis=1)
    if not np.any(changed):
        metrics.update(status="no_action", reason="no_stored_coordinate_change")
        return False
    patch = {int(source_data.vid[i]): projected[i] for i in moving[changed]}
    io.open_blend(source)
    obj = io.mesh_object(name)
    io.write_positions(obj, patch, source_data.hashes["source_hash"])
    candidate = io.read_mesh(obj)
    after = distance_statistics(candidate.V[region] * candidate.unit_scale - ref[region])
    gain = (before["rms"] - after["rms"]) / before["rms"]
    metrics.update(reference_after_m=after, rms_reduction_fraction=gain)
    if gain < 0.50:
        metrics.update(status="rejected", reason="reference_rms_reduction_below_50_percent")
        return False

    io.save_copy(staged_blend, source, out)
    io.open_blend(staged_blend)
    obj = io.mesh_object(name)
    saved = io.read_mesh(obj)
    same = {key: source_data.hashes[key] == saved.hashes[key] for key in io.INVARIANTS}
    frozen = saved.V[locked].tobytes() == source_data.V[locked].tobytes()
    delta = (saved.V - source_data.V) * source_data.unit_scale
    peak = float(np.max(np.linalg.norm(delta, axis=1)))
    locked_peak = float(np.max(np.linalg.norm(delta[locked], axis=1), initial=0))
    metrics.update(output_hashes=saved.hashes, invariant_equal=same,
                   locked_coordinates_exact=frozen, max_displacement_m=peak, locked_max_displacement_m=locked_peak,
                   output_file_sha256=file_hash(staged_blend), moved_vertices=int(np.count_nonzero(np.any(delta, axis=1))),
                   advisory_displacement={"0.1h_m": 0.1 * h, "0.002D_m": 0.002 * D,
                                          "max_over_h": peak / h, "max_over_D": peak / D,
                                          "within_0.1h": peak <= 0.1 * h, "within_0.002D": peak <= 0.002 * D,
                                          "annotation_cap_m": plan.get("max_displacement_m")})
    if (not all(same.values()) or not frozen or source_hash != file_hash(source)
            or saved.V.tobytes() != candidate.V.tobytes()):
        metrics.update(status="rejected", reason="positions_only_or_roundtrip_check_failed")
        return False

    # The render function works on copies, after the output file has already been serialized.
    io.render_previews(obj, source_data, saved, reference, folder)
    rim_faces = [index for index in face_indices
                 if any(left in boundary and right in boundary for left, right in face_edges((source_data.F[index],)))]
    if rim_faces:
        rim_normals = []
        for data in (reference, source_data, saved):
            mask = np.isin(data.triangle_faces, rim_faces)
            normals, _, _ = face_geometry(data.V, data.triangles[mask], data.triangle_faces[mask], len(data.F))
            rim_normals.append(normals[rim_faces])
        metrics["rim_face_normal_deviation_deg"] = {}
        for label, normals in zip(("before", "after"), rim_normals[1:]):
            cosine = np.einsum("ij,ij->i", normals, rim_normals[0])
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            metrics["rim_face_normal_deviation_deg"][label] = {
                "p95": float(np.percentile(angle, 95)), "max": float(np.max(angle)),
            }
    metrics.update(status="accepted", output=out.name, source_file_unchanged=True,
                   images={"before": "before.png", "after": "after.png", "heatmap": "displacement.png",
                           "heatmap_peak_m": peak})
    return True


def main() -> None:
    args = parse_args()
    out, report = args.out.resolve(), args.report.resolve()
    if report == out or report in out.parents or os.path.lexists(report):
        raise SystemExit("--report requires a new directory separate from --out")
    report.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"stage": "P1", "region": args.region}
    with TemporaryDirectory(prefix=".gre-report-", dir=report.parent) as report_stage:
        folder = Path(report_stage) / "report"
        folder.mkdir()
        descriptor, staged_path = mkstemp(prefix=".gre-p1-", suffix=".blend", dir=out.parent)
        os.close(descriptor)
        staged_blend = Path(staged_path)
        published = False
        try:
            if run(args, folder, staged_blend, out, metrics):
                os.link(staged_blend, out)
                published = True
        except io.Unsupported as error:
            metrics.update(status="unsupported", reason=str(error))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, np.linalg.LinAlgError) as error:
            metrics.pop("output", None)
            metrics.update(status="solver_failed", reason=str(error))
        finally:
            staged_blend.unlink(missing_ok=True)
        try:
            with (folder / "metrics.json").open("x", encoding="utf-8") as stream:
                json.dump(metrics, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            os.rename(folder, report)
        except (OSError, ValueError):
            if published:
                out.unlink(missing_ok=True)
            raise
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
