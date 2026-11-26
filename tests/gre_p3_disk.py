# SPDX-License-Identifier: GPL-3.0-or-later
"""Independently verify the regenerated assets and P3 outputs from disk."""

import argparse
import json
import sys
from pathlib import Path

import bmesh
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gre import blender_io as io
from gre.qa import INVARIANTS


def read(path: Path):
    io.open_blend(path.resolve())
    obj = io.mesh_object("Cabinet")
    mesh = io.read_mesh(obj)
    return obj, mesh, io.qa_snapshot(obj, mesh)


def verify_closed(obj) -> None:
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        assert all(edge.is_manifold and edge.is_contiguous for edge in bm.edges)
        assert bm.calc_volume(signed=True) > 0
    finally:
        bm.free()


def plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    n = axes[-1]
    n = n if n[np.argmax(np.abs(n))] >= 0 else -n
    return n, -float(points.mean(axis=0) @ n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "repeat", "results"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    plan = json.loads((args.data / "cabinet_annotations.json").read_text(encoding="utf-8"))
    assert plan == json.loads((args.repeat / "cabinet_annotations.json").read_text(encoding="utf-8"))
    base = None
    for mode in ("clean", "tilted", "noisy"):
        filename = f"cabinet_{mode}.blend"
        obj, mesh, snapshot = read(args.data / filename)
        if mode == "clean":
            base = snapshot
            verify_closed(obj)
            assert snapshot.hashes["source_hash"] == plan["source_hash"]
            dimensions = plan["dimensions"]
            np.testing.assert_allclose(np.ptp(mesh.V, axis=0), [dimensions[name] for name in ("width_m", "depth_m", "height_m")],
                                       atol=1e-7, rtol=0)
            uv = snapshot.uv_layers["CabinetUV"][snapshot.triangle_loops].astype(np.float64)
            a, b = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
            assert np.isfinite(uv).all() and np.all(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])
        assert all(snapshot.hashes[key] == base.hashes[key] for key in INVARIANTS)
        _, _, replay = read(args.repeat / filename)
        assert snapshot.hashes == replay.hashes and snapshot.V.tobytes() == replay.V.tobytes()
        print(f"REGENERATED {mode}: coordinates and fingerprints identical", flush=True)

    for mode in ("tilted", "noisy"):
        _, source, baseline = read(args.data / f"cabinet_{mode}.blend")
        _, actual, candidate = read(args.results / mode / "cabinet_repaired.blend")
        assert np.array_equal(source.vid, actual.vid) and np.array_equal(source.fid, actual.fid)
        assert all(baseline.hashes[key] == candidate.hashes[key] for key in INVARIANTS)
        indices = {fid: i for i, fid in enumerate(actual.fid.tolist())}
        region_verts = {region["name"]: sorted({vertex for fid in region["face_ids"] for vertex in actual.F[indices[fid]]})
                        for region in plan["regions"]}
        locked = sorted({vertex for region in plan["regions"] if region["role"] == "protected"
                         for vertex in region_verts[region["name"]]})
        assert source.V[locked].tobytes() == actual.V[locked].tobytes()
        clean_error_old = np.sqrt(np.mean(np.sum((source.V - base.V)**2, axis=1)))
        clean_error_new = np.sqrt(np.mean(np.sum((actual.V - base.V)**2, axis=1)))
        assert clean_error_new < clean_error_old
        peaks = {"parallel": 0.0, "perpendicular": 0.0, "offset_mm": 0.0}
        planes = {region["name"]: plane(actual.V[region_verts[region["name"]]] * actual.unit_scale)
                  for region in plan["regions"] if region["role"] == "editable"}
        for relation in plan["relations"]:
            n, d = planes[relation["a"]]
            m, e = planes[relation["b"]]
            dot = float(n @ m)
            degrees = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n, m)), abs(dot))))
            if relation["type"] == "perpendicular":
                peaks["perpendicular"] = max(peaks["perpendicular"], abs(degrees - 90))
            else:
                peaks["parallel"] = max(peaks["parallel"], degrees)
            if relation["type"] == "offset":
                error_mm = 1000 * abs(d - (e if dot >= 0 else -e) - relation["distance_m"])
                peaks["offset_mm"] = max(peaks["offset_mm"], error_mm)
        assert peaks["parallel"] < 0.1 and peaks["perpendicular"] < 0.1 and peaks["offset_mm"] < 1
        print(f"DISK {mode}: {json.dumps(peaks)} / {len(locked)} protected vertices exact", flush=True)


if __name__ == "__main__":
    main()
