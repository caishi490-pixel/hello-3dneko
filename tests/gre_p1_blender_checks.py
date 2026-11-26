# SPDX-License-Identifier: GPL-3.0-or-later
"""Reopen completed P1 outputs and independently check them using P0's raw-data reader."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def reopen(path: Path) -> None:
    assert bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False) == {"FINISHED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.data.resolve()))
    from gre import blender_io as io
    from hash_asset import coordinates, fingerprint, mesh_object, stable_ids

    plan = json.loads((args.data / "table_annotations.json").read_text(encoding="utf-8"))
    face_ids = {fid for region in plan["regions"] if region["name"] == "tabletop" for fid in region["face_ids"]}
    reopen(args.data / "table_clean.blend")
    clean = mesh_object("Table").data
    fid = stable_ids(clean, "gre_fid", "FACE").tolist()
    chosen = {vertex for face_id, face in zip(fid, clean.polygons) if face_id in face_ids for vertex in face.vertices}
    fixed = {vertex for face_id, face in zip(fid, clean.polygons) if face_id not in face_ids for vertex in face.vertices}
    locked = sorted(set(range(len(clean.vertices))) - (chosen - fixed))
    top = sorted(chosen)
    reference = coordinates(clean).astype(np.float64)

    for ratio in ("0.5", "1"):
        reopen(args.data / f"table_noisy_s42_a{ratio}.blend")
        original = mesh_object("Table")
        before, baseline = coordinates(original.data), fingerprint(original)
        reopen(args.results / f"a{ratio}" / "table_repaired.blend")
        result = mesh_object("Table")
        after, repaired = coordinates(result.data), fingerprint(result)
        assert before[locked].tobytes() == after[locked].tobytes()
        assert all(repaired[key] == value for key, value in baseline.items() if key not in {"position_hash", "source_hash"})
        peak_mm = 1000 * np.max(np.linalg.norm(after.astype(np.float64) - before, axis=1))
        before_mm = 1000 * np.sqrt(np.mean(np.sum((before[top] - reference[top])**2, axis=1)))
        after_mm = 1000 * np.sqrt(np.mean(np.sum((after[top] - reference[top])**2, axis=1)))
        assert after_mm <= before_mm / 2
        assert len(bpy.context.scene.objects) == 1 and not result.data.color_attributes
        print(f"Disk recheck a{ratio}: RMS {before_mm:.6f} -> {after_mm:.6f} mm; peak {peak_mm:.6f} mm; hashes and {len(locked)} locked coordinates exact")

    for scenario in ("shared", "shape_keys", "modifier", "armature_parent", "stale_hash", "second_uv_layer"):
        reopen(args.data / "table_clean.blend")
        obj = mesh_object("Table")
        if scenario == "shared":
            extra = bpy.data.objects.new("Extra user", obj.data)
            bpy.context.scene.collection.objects.link(extra)
        elif scenario == "shape_keys":
            obj.shape_key_add(name="Basis")
        elif scenario == "modifier":
            obj.modifiers.new("Unapplied bevel", "BEVEL")
        elif scenario == "armature_parent":
            parent = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
            bpy.context.scene.collection.objects.link(parent)
            obj.parent = parent
        elif scenario == "stale_hash":
            baseline = coordinates(obj.data)
            vid = int(stable_ids(obj.data, "gre_vid", "POINT")[0])
            try:
                io.write_positions(obj, {vid: np.ones(3)}, "stale")
            except ValueError as error:
                assert "Stale source hash" in str(error)
                assert baseline.tobytes() == coordinates(obj.data).tobytes()
                print("PASS stale_hash: no coordinate was written")
                continue
            raise AssertionError("Stale source hash was accepted")
        else:
            layer = obj.data.uv_layers.new(name="ExtraUV", do_init=True)
            before = io.read_mesh(obj).hashes
            layer.uv[0].vector.x += 0.01
            after = io.read_mesh(obj).hashes
            assert before["uv_hash"] != after["uv_hash"] and before["topology_hash"] == after["topology_hash"]
            assert after["uv_hash"] == fingerprint(obj)["uv_hash"]
            print("PASS second_uv_layer: a single corner edit changes the UV fingerprint")
            continue
        try:
            io.read_mesh(obj)
        except io.Unsupported as error:
            print(f"PASS {scenario}: {error}")
        else:
            raise AssertionError(f"Unsupported input passed: {scenario}")


if __name__ == "__main__":
    main()
