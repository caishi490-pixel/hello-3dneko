"""Blender-side exchange formats for geometry and face-corner UVs."""

import argparse
import json
from pathlib import Path
import sys

import bmesh
import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_runtime import junk_objects
from gap_coverage_validator import sample_shells
from topology_inspector import load_scene_file


def _meshes():
    objects = bpy.context.scene.objects
    junk = set(junk_objects(objects))
    return [obj for obj in objects if obj.type == "MESH" and obj not in junk]


def welded_mesh_copy(mesh, weld_dist=1e-4):
    result = mesh.copy()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=weld_dist)
        bm.to_mesh(result)
    finally:
        bm.free()
    return result


def export_welded_geometry_obj(src, out_obj, weld_dist=1e-4):
    out = Path(out_obj)
    if out.suffix.lower() != ".obj":
        raise ValueError("Welded geometry output requires an .obj path")
    if weld_dist < 0:
        raise ValueError("weld_dist must be nonnegative")
    load_scene_file(src)
    meshes = _meshes()
    if not meshes:
        raise ValueError("Input contains no mesh surfaces")
    samples = sample_shells(meshes, weld_dist)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.data = welded_mesh_copy(obj.data, weld_dist)
        obj.select_set(True)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.obj_export(filepath=str(out), export_selected_objects=True, export_uv=False,
                          export_materials=False, apply_modifiers=False)
    out.with_name("weld_report.json").write_text(
        json.dumps({"weld_dist": weld_dist, "shell_samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_uv_preserving_mesh(src, out_mesh):
    out = Path(out_mesh)
    if out.suffix.lower() != ".blend":
        raise ValueError("The lossless face-corner exchange format requires a .blend path")
    load_scene_file(src)
    if not _meshes() or any(not obj.data.uv_layers.active for obj in _meshes()):
        raise ValueError("The UV-preserving exchange requires an active UV layer on every mesh")
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("welded", "uv"))
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--weld-dist", type=float, default=1e-4)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.mode == "welded":
        export_welded_geometry_obj(args.source, args.output, args.weld_dist)
    else:
        export_uv_preserving_mesh(args.source, args.output)
