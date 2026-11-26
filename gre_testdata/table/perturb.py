# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a new .blend with seeded normal offsets on approved stable face IDs."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_asset import coordinates, fingerprint, mesh_object, save_new_blend, script_args, stable_ids
from noise_field import normal_offsets, transition_weights


INVARIANTS = ("topology_hash", "uv_hash", "material_hash", "object_relationship_hash", "unit_scale", "matrix_world")


def approved_faces(annotations: dict, face_ids: list[int] | None, known: set[int]) -> set[int]:
    seen = set()
    editable = set()
    if annotations["schema_version"] != 1:
        raise ValueError("Expected annotation schema_version 1")
    for region in annotations["regions"]:
        ids = set(region["face_ids"])
        if (not ids or len(ids) != len(region["face_ids"]) or not ids <= known or ids & seen
                or any(type(face_id) is not int for face_id in ids)):
            raise ValueError(f"Invalid, unknown, or overlapping face IDs in {region['name']}")
        seen.update(ids)
        if region["role"] == "editable":
            editable.update(ids)
        elif region["role"] not in {"protected", "unknown"}:
            raise ValueError(f"Unknown region role: {region['role']}")
    selected = set(face_ids) if face_ids is not None else editable
    if not selected or not selected <= editable:
        raise ValueError("Select at least one face and use only editable gre_fid values")
    return selected


def make_offsets(
    mesh: bpy.types.Mesh, fids: np.ndarray, selected: set[int], unit_scale: float, rings: int,
    amplitude: float, wavelength: float, seed: int,
) -> tuple[dict[int, float], float]:
    positions_m = coordinates(mesh).astype(np.float64) * unit_scale
    edges = set()
    outside = set()
    for face_id, face in zip(fids.tolist(), mesh.polygons):
        if face_id not in selected:
            outside.update(face.vertices)
        else:
            edges.update(tuple(sorted(edge)) for edge in face.edge_keys)

    graph: dict[int, list[tuple[int, float]]] = {}
    lengths = []
    for left, right in sorted(edges):
        length = float(np.linalg.norm(positions_m[left] - positions_m[right]))
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Selected region contains an invalid or zero-length edge")
        graph.setdefault(left, []).append((right, length))
        graph.setdefault(right, []).append((left, length))
        lengths.append(length)
    h = float(np.median(lengths))
    weights = transition_weights(graph, set(graph) & outside, rings)
    vids = stable_ids(mesh, "gre_vid", "POINT")
    moving = sorted((index for index, weight in weights.items() if weight > 0), key=lambda index: vids[index])
    if not moving:
        raise ValueError("No eligible vertices after locking the boundary and transition")
    peak = amplitude * h
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("Requested peak is outside the supported numeric range")
    offsets = normal_offsets(positions_m[moving], np.array([weights[index] for index in moving]), peak, wavelength, seed)
    return {index: float(offset) for index, offset in zip(moving, offsets) if offset != 0}, h


def check_supported(obj: bpy.types.Object) -> None:
    if obj.data.users != 1 or obj.data.shape_keys or obj.modifiers or obj.find_armature():
        raise ValueError("unsupported: shared mesh, shape keys, unapplied modifiers or armature binding")
    parent = obj.parent
    while parent is not None:
        if parent.type == "ARMATURE":
            raise ValueError("unsupported: armature ancestor")
        parent = parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--amplitude", required=True, type=float, help="Peak absolute offset as a multiple of h")
    parser.add_argument("--wavelength", required=True, type=float, help="Nominal wavelength in local-space metres")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--object")
    parser.add_argument("--face-ids", nargs="+", type=int, help="Optional editable gre_fid subset")
    args = parser.parse_args(script_args())
    if not all(math.isfinite(value) and value > 0 for value in (args.amplitude, args.wavelength)):
        parser.error("amplitude and wavelength must be finite and greater than zero")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be in the unsigned 64-bit range")
    if not bpy.data.filepath:
        parser.error("load the clean .blend before running this script")

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    obj = mesh_object(args.object or annotations.get("object_name", "Table"))
    check_supported(obj)
    baseline = fingerprint(obj)
    if annotations["source_hash"] != baseline["source_hash"]:
        raise ValueError("Annotations source_hash does not match the loaded clean mesh and scene")
    unit_scale = annotations["unit_scale"]
    rings = annotations["transition_rings"]
    if not math.isfinite(unit_scale) or unit_scale <= 0 or unit_scale != baseline["unit_scale"]:
        raise ValueError("Invalid or mismatched unit_scale")
    if type(rings) is not int or rings < 0:
        raise ValueError("transition_rings must be a nonnegative integer")
    mesh = obj.data
    fids = stable_ids(mesh, "gre_fid", "FACE")
    selected = approved_faces(annotations, args.face_ids, set(fids.tolist()))
    offsets, h = make_offsets(mesh, fids, selected, unit_scale, rings, args.amplitude, args.wavelength, args.seed)
    cap = annotations["max_displacement_m"]
    if cap is not None and (not math.isfinite(cap) or cap < max(map(abs, offsets.values()))):
        raise ValueError("Requested peak exceeds max_displacement_m")

    amp = str(args.amplitude).removesuffix(".0")
    output = args.output_dir.resolve() / f"table_noisy_s{args.seed}_a{amp}.blend"
    if os.path.lexists(output) or output == Path(bpy.data.filepath).resolve():
        raise FileExistsError(output)
    normals = {index: mesh.vertices[index].normal.copy() for index in offsets}
    if any(normal.length == 0 for normal in normals.values()):
        raise ValueError("An eligible vertex has an undefined normal")
    for index, offset in offsets.items():
        mesh.vertices[index].co += normals[index] * (offset / unit_scale)
    mesh.update()
    if not np.isfinite(coordinates(mesh)).all():
        raise ValueError("Disturbance produced nonfinite Blender coordinates")
    after = fingerprint(obj)
    if after["position_hash"] == baseline["position_hash"]:
        raise ValueError("Offsets did not change the stored positions")
    if any(after[key] != baseline[key] for key in INVARIANTS):
        raise RuntimeError("A positions-only invariant failed")

    save_new_blend(output)
    print(json.dumps({"output": output.name, "seed": args.seed, "amplitude_h": args.amplitude,
                      "wavelength_m": args.wavelength, "median_edge_m": h, "moved_vertices": len(offsets),
                      "position_hash": after["position_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
