# SPDX-License-Identifier: GPL-3.0-or-later
"""Reopen the table files and measure P0 acceptance against the clean coordinates."""

import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_asset import coordinates, fingerprint, mesh_object, put_json, script_args, stable_ids


def reopen(path: Path) -> bpy.types.Object:
    if bpy.ops.wm.open_mainfile(filepath=str(path.resolve()), load_ui=False, use_scripts=False) != {"FINISHED"}:
        raise RuntimeError(f"Cannot open {path}")
    return mesh_object("Table")


def verify_clean_mesh(mesh: bpy.types.Mesh) -> None:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        assert all(edge.is_manifold and edge.is_contiguous for edge in bm.edges), "Invalid boundary or winding"
        assert bm.calc_volume(signed=True) > 0, "Invalid signed volume"
    finally:
        bm.free()
    assert len(mesh.uv_layers) >= 1
    mesh.calc_loop_triangles()
    triangles = np.array([list(triangle.loops) for triangle in mesh.loop_triangles])
    for layer in mesh.uv_layers:
        flat = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        layer.uv.foreach_get("vector", flat)
        uv = flat.reshape(-1, 2)[triangles].astype(np.float64)
        a = uv[:, 1] - uv[:, 0]
        b = uv[:, 2] - uv[:, 0]
        assert np.isfinite(uv).all() and np.all(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 0), layer.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    args = parser.parse_args(script_args())
    labels = json.loads((args.data_dir / "table_annotations.json").read_text(encoding="utf-8"))
    obj = reopen(args.data_dir / "table_clean.blend")
    mesh = obj.data
    clean = fingerprint(obj)
    clean_positions = coordinates(mesh)
    verify_clean_mesh(mesh)
    fids = stable_ids(mesh, "gre_fid", "FACE").tolist()
    editable = {value for region in labels["regions"] if region["role"] == "editable" for value in region["face_ids"]}
    protected = {value for region in labels["regions"] if region["role"] == "protected" for value in region["face_ids"]}
    assert not editable & protected and editable | protected == set(fids)
    assert sum(len(region["face_ids"]) for region in labels["regions"]) == len(fids)
    assert labels["schema_version"] == 1 and labels["source_hash"] == clean["source_hash"]
    assert labels["unit_scale"] == clean["unit_scale"] == 1 and labels["matrix_world"] == clean["matrix_world"]
    assert labels["transition_rings"] == 2
    tabletop = set()
    locked = set()
    edges = set()
    for face_id, face in zip(fids, mesh.polygons):
        if face_id in editable:
            tabletop.update(face.vertices)
            edges.update(tuple(sorted(edge)) for edge in face.edge_keys)
            assert len(face.vertices) == 4 and face.normal.z > 0.99999
        else:
            locked.update(face.vertices)
    table_coords = clean_positions[sorted(tabletop)]
    grid = [len(np.unique(table_coords[:, axis])) - 1 for axis in (0, 1)]
    assert grid == labels["tabletop_grid"] == [64, 64] and len(editable) == grid[0] * grid[1]
    assert np.ptp(table_coords[:, 2]) == 0
    measured = np.ptp(clean_positions, axis=0)
    assert np.allclose(measured, [labels["dimensions_m"][name] for name in ("width", "depth", "height")], atol=1e-7, rtol=0)
    left, right = np.array(sorted(edges)).T
    h = float(np.median(np.linalg.norm(clean_positions[left].astype(np.float64) - clean_positions[right], axis=1)))
    locked_indices = sorted(set(range(len(clean_positions))) - (tabletop - locked))
    rows = []
    for ratio in ("0.5", "1", "3"):
        name = f"table_noisy_s42_a{ratio}.blend"
        noisy_obj = reopen(args.data_dir / name)
        noisy = fingerprint(noisy_obj)
        positions = coordinates(noisy_obj.data)
        repeated = coordinates(reopen(args.repeat_dir / name).data)
        assert positions.tobytes() == repeated.tobytes(), f"Repeat positions differ: {name}"
        assert positions[locked_indices].tobytes() == clean_positions[locked_indices].tobytes(), name
        assert positions[:, :2].tobytes() == clean_positions[:, :2].tobytes(), name
        assert all(noisy[key] == value for key, value in clean.items() if key not in {"position_hash", "source_hash"}), name
        delta = positions[sorted(tabletop), 2].astype(np.float64) - table_coords[:, 2]
        peak = float(np.max(np.abs(delta)))
        assert abs(peak - h * float(ratio)) < 1e-7
        assert np.min(positions[sorted(tabletop), 2]) > float(labels["dimensions_m"]["height"] - labels["dimensions_m"]["tabletop_thickness"])
        rows.append({"file": name, "seed": 42, "amplitude_h": float(ratio), "wavelength_m": 0.32,
                     "peak_m": peak, "rms_m": float(np.sqrt(np.mean(delta**2))),
                     "p95_abs_m": float(np.percentile(np.abs(delta), 95)), "moved_vertices": int(np.count_nonzero(delta)),
                     "locked_peak_m": 0.0, "repeat_positions_exact": True, "positions_only": True,
                     "position_hash": noisy["position_hash"]})
    result = {"clean": clean, "grid": grid, "dimensions_m": labels["dimensions_m"], "median_edge_m": h,
              "editable_faces": len(editable), "protected_faces": len(protected), "face_overlap": len(editable & protected),
              "locked_vertices": len(locked_indices), "clean_manifold": True, "clean_uv_complete": True, "variants": rows}
    put_json(args.data_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print("P0 ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
