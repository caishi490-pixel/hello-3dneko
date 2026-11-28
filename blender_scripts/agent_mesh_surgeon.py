"""Agent Mesh Surgeon — Stateful Interactive Edit-Mode Workbench for Multimodal Agents (Blender 5.1+).

Gives any Vision Agent a human-grade iterative mesh surgery workflow inside headless Blender:
  - Stateful sessions with automatic snapshot checkpoints and instant `undo`.
  - Triptych inspection renders on every step:
      [1] 10x10 A0..J9 Grid + Red Selection Highlight
      [2] Clay MatCap (immediately exposes flipped black normals & messy inner double-walls)
      [3] Dark-Matte Body + 0.85px Orange Wireframe (exposes shredded topology & floating rings)
  - Camera zoom (`--zoom 1.0..4.0`) & `--focus-selected` close-ups.
  - Human Edit-Mode operators:
      open | look | select-box | select-linked | select-backface | grow | shrink |
      delete | bisect | clean | undo | export
"""

import argparse
import json
import math
import re
import shutil
import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from mathutils.bvhtree import BVHTree


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_runtime import configure_cycles, remove_export_junk

GRID_COLS = "ABCDEFGHIJ"


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Stateful Edit-Mode mesh surgery workbench for AI Agents.")
    p.add_argument("--session", required=True, help="Session workspace directory storing state, snapshots, and renders.")
    p.add_argument(
        "--action",
        required=True,
        choices=(
            "open",
            "look",
            "select-box",
            "select-linked",
            "select-backface",
            "grow",
            "shrink",
            "delete",
            "bisect",
            "clean",
            "undo",
            "export",
        ),
    )
    p.add_argument("--input", default="", help="Input .blend or .glb (required for --action open).")
    p.add_argument("--target-mesh", default="", help="Optional exact mesh object name.")
    p.add_argument("--view", default="front_skirt", help="Camera preset: front|front_skirt|bottom_up|back|left|right|34_front")
    p.add_argument("--zoom", type=float, default=1.0, help="Camera zoom factor (1.0 = default, 2.0 = 2x close-up).")
    p.add_argument("--focus-selected", action="store_true", help="Center camera target on current selected faces.")
    p.add_argument("--res", type=int, default=448, help="Per-panel resolution for triptych renders.")
    p.add_argument("--id-res", type=int, default=256, help="Raycast Face-ID buffer resolution.")
    p.add_argument("--box", default="", help="2D selection box: 'x0,y0,x1,y1' or 'D4:F6' (top-left origin).")
    p.add_argument("--point", default="", help="2D seed point 'x,y' (or cell 'E5') for select-linked.")
    p.add_argument("--op", choices=("set", "add", "sub"), default="set", help="Selection boolean operation.")
    p.add_argument("--xray", action="store_true", help="Penetrating X-Ray + sub-pixel centroid sweep for select-box.")
    p.add_argument("--depth-m", type=float, default=0.16, help="Penetration/cut depth band in meters.")
    p.add_argument("--angle-deg", type=float, default=38.0, help="Dihedral angle limit for select-linked.")
    p.add_argument("--steps", type=int, default=1, help="Ring steps for grow/shrink.")
    p.add_argument("--orphan-max-pct", type=float, default=3.0, help="Auto-prune cut-adjacent 3D-welded orphan islands <= pct.")
    p.add_argument(
        "--line",
        action="append",
        default=[],
        help="Surgical knife line in current view: 'x0,y0->x1,y1:drop=left|right|above|below'.",
    )
    p.add_argument("--quadify", action="store_true", help="Convert planar triangles to quads during --action clean.")
    p.add_argument(
        "--recalc-normals",
        action="store_true",
        help="Recalculate face normals during --action clean (use only on watertight/welded meshes, not UV-split GLBs).",
    )
    p.add_argument("--out-blend", default="", help="Output .blend path for --action export.")
    p.add_argument("--out-glb", default="", help="Output .glb path for --action export.")
    return p.parse_args(argv)


def _load_scene(filepath: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fp = str(filepath)
    if fp.lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=fp)
    else:
        bpy.ops.import_scene.gltf(filepath=fp)


def _get_target_mesh(name_hint: str = ""):
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons) > 0]
    if not meshes:
        raise RuntimeError("No non-empty mesh object found in scene.")
    if name_hint:
        for o in meshes:
            if o.name == name_hint:
                return o
        raise RuntimeError(f"Target mesh '{name_hint}' not found; available: {[o.name for o in meshes]}")
    for o in meshes:
        if "garment" in o.name.lower():
            return o
    return max(meshes, key=lambda o: len(o.data.polygons))


def _mesh_bounds(target_obj):
    mw = target_obj.matrix_world
    pts = [mw @ v.co for v in target_obj.data.vertices]
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    span = max((max_p - min_p).x, (max_p - min_p).y, (max_p - min_p).z, 0.1)
    return min_p, max_p, center, span


def _compute_camera_spec(target_obj, view_name: str, zoom: float = 1.0, focus_faces: set[int] | None = None) -> dict:
    min_p, max_p, center, span = _mesh_bounds(target_obj)
    lower_tgt = Vector((center.x, center.y, min_p.z + 0.34 * (max_p.z - min_p.z)))
    presets = {
        "front": (center + Vector((0.0, -1.85, 0.05)) * span, center, 48.0),
        "front_skirt": (lower_tgt + Vector((0.0, -1.10, -0.04)) * span, lower_tgt, 48.0),
        "bottom_up": (lower_tgt + Vector((0.0, -0.66, -0.90)) * span, lower_tgt + Vector((0.0, 0.0, 0.08 * span)), 42.0),
        "back": (center + Vector((0.0, 1.85, 0.05)) * span, center, 48.0),
        "left": (center + Vector((1.85, 0.0, 0.05)) * span, center, 48.0),
        "right": (center + Vector((-1.85, 0.0, 0.05)) * span, center, 48.0),
        "34_front": (center + Vector((-1.35, -1.35, 0.20)) * span, center, 48.0),
    }
    if view_name not in presets:
        raise ValueError(f"Unknown view '{view_name}'; choose from {sorted(presets.keys())}")
    loc, tgt, lens = presets[view_name]
    if focus_faces:
        mw = target_obj.matrix_world
        pts = [mw @ target_obj.data.polygons[fi].center for fi in focus_faces if 0 <= fi < len(target_obj.data.polygons)]
        if pts:
            f_center = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
            offset = loc - tgt
            tgt = f_center
            loc = tgt + offset
    z = max(0.5, min(6.0, float(zoom)))
    lens = lens * z
    return {
        "name": view_name,
        "zoom": z,
        "location": [loc.x, loc.y, loc.z],
        "target": [tgt.x, tgt.y, tgt.z],
        "lens": lens,
    }


def _setup_cycles(res: int):
    scene = bpy.context.scene
    configure_cycles(scene)
    scene.cycles.samples = 20
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.view_settings.view_transform = "Standard"


def _create_studio_rig(target_obj, cam_spec: dict, res: int):
    _setup_cycles(res)
    scene = bpy.context.scene
    _, _, center, span = _mesh_bounds(target_obj)
    w = bpy.data.worlds.new("SurgeonWorld")
    scene.world = w
    if w.node_tree is None:
        w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.85, 0.86, 0.89, 1.0)
    bg.inputs[1].default_value = 0.65

    created = []
    for lname, off, eng in [
        ("SG_Key", Vector((-1.2, -1.6, 1.0)), 110),
        ("SG_Fill", Vector((1.4, -1.2, 0.4)), 80),
        ("SG_Bottom", Vector((0.0, -0.4, -1.6)), 120),
        ("SG_Back", Vector((0.0, 1.8, 0.8)), 95),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng * max(1.0, span * span)
        ld.size = span * 1.4
        lo = bpy.data.objects.new(lname, ld)
        scene.collection.objects.link(lo)
        lo.location = center + off * span
        lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()
        created.append(lo)

    cam_d = bpy.data.cameras.new("SGCam")
    cam = bpy.data.objects.new("SGCam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    loc = Vector(cam_spec["location"])
    tgt = Vector(cam_spec["target"])
    cam.data.lens = float(cam_spec["lens"])
    cam.location = loc
    cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    created.append(cam)
    return cam, created, bg


def _draw_10x10_grid(img_arr: np.ndarray) -> np.ndarray:
    out = img_arr.copy()
    h, w, _ = out.shape
    for k in range(1, 10):
        px = int(round(k * w / 10.0))
        py_top = int(round(k * h / 10.0))
        py_bl = h - 1 - py_top
        thickness = 2 if k == 5 else 1
        color = np.array([0.08, 0.85, 0.85], dtype=np.float32) if k == 5 else np.array([0.12, 0.16, 0.22], dtype=np.float32)
        alpha = 0.72 if k == 5 else 0.42
        x0, x1 = max(0, px - thickness // 2), min(w, px + (thickness + 1) // 2)
        y0, y1 = max(0, py_bl - thickness // 2), min(h, py_bl + (thickness + 1) // 2)
        out[:, x0:x1, :3] = out[:, x0:x1, :3] * (1.0 - alpha) + color * alpha
        out[y0:y1, :, :3] = out[y0:y1, :, :3] * (1.0 - alpha) + color * alpha

    glyphs = {
        "A": ["010", "101", "111", "101", "101"],
        "B": ["110", "101", "110", "101", "110"],
        "C": ["011", "100", "100", "100", "011"],
        "D": ["110", "101", "101", "101", "110"],
        "E": ["111", "100", "110", "100", "111"],
        "F": ["111", "100", "110", "100", "100"],
        "G": ["011", "100", "101", "101", "011"],
        "H": ["101", "101", "111", "101", "101"],
        "I": ["111", "010", "010", "010", "111"],
        "J": ["001", "001", "001", "101", "010"],
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"],
        "7": ["111", "001", "001", "010", "010"],
        "8": ["111", "101", "111", "101", "111"],
        "9": ["111", "101", "111", "001", "111"],
    }

    def _stamp(ch: str, cx_top: int, cy_top: int, scale: int = 2):
        g = glyphs[ch]
        gh, gw = len(g) * scale, len(g[0]) * scale
        bx0, bx1 = max(0, cx_top - gw // 2 - 2), min(w, cx_top + gw // 2 + 3)
        by0, by1 = max(0, cy_top - gh // 2 - 2), min(h, cy_top + gh // 2 + 3)
        out[h - by1 : h - by0, bx0:bx1, :3] = 0.08
        for r_i, row_s in enumerate(g):
            for c_i, bit in enumerate(row_s):
                if bit == "1":
                    px0 = cx_top - gw // 2 + c_i * scale
                    py0 = cy_top - gh // 2 + r_i * scale
                    out[h - (py0 + scale) : h - py0, px0 : px0 + scale, :3] = 0.98

    for k in range(10):
        _stamp(GRID_COLS[k], int((k + 0.5) * w / 10.0), 11, scale=2)
        _stamp(str(k), 11, int((k + 0.5) * h / 10.0), scale=2)
    return out


def _save_rgba(arr: np.ndarray, dst: Path, name: str):
    h, w, _ = arr.shape
    img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    img.pixels.foreach_set(arr.ravel())
    img.filepath_raw = str(dst)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def _render_single_array(tmp_path: Path, res: int) -> np.ndarray:
    bpy.context.scene.render.filepath = str(tmp_path)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(tmp_path))
    arr = np.array(img.pixels[:], dtype=np.float32).reshape((res, res, 4))
    bpy.data.images.remove(img)
    tmp_path.unlink(missing_ok=True)
    return arr


def _build_bvh(obj) -> BVHTree:
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys)


def _raycast_face_id(cam, target_obj, occluders: list, id_res: int) -> np.ndarray:
    t_bvh = _build_bvh(target_obj)
    o_bvhs = [_build_bvh(o) for o in occluders]
    scene = bpy.context.scene
    frame = [cam.matrix_world @ v for v in cam.data.view_frame(scene=scene)]
    tr, br, bl, tl = frame[0], frame[1], frame[2], frame[3]
    origin = cam.matrix_world.translation.copy()
    vec_r, vec_d = tr - tl, bl - tl
    id_map = np.full((id_res, id_res), -1, dtype=np.int32)
    for row in range(id_res):
        row_base = tl + vec_d * ((row + 0.5) / id_res)
        for col in range(id_res):
            rdir = (row_base + vec_r * ((col + 0.5) / id_res) - origin).normalized()
            loc, _, f_idx, dist = t_bvh.ray_cast(origin, rdir, 50.0)
            if loc is None or f_idx is None:
                continue
            if any((ob.ray_cast(origin, rdir, dist - 1e-4)[0] is not None) for ob in o_bvhs):
                continue
            id_map[row, col] = int(f_idx)
    return id_map


def _render_triptych_and_ids(target_obj, occluders: list, cam_spec: dict, renders_dir: Path, res: int, id_res: int, selected_faces: set[int]) -> dict:
    """Render [Grid+Selection | Clay MatCap | Dark-Matte Body + 0.85px Orange Wireframe] + save face_id.npy."""
    renders_dir.mkdir(parents=True, exist_ok=True)
    cam, created, bg = _create_studio_rig(target_obj, cam_spec, res)

    # Export exact Face-ID buffer for current camera & zoom
    id_map = _raycast_face_id(cam, target_obj, occluders, id_res)
    id_path = renders_dir / "face_id.npy"
    np.save(str(id_path), id_map)

    orig_mats = list(target_obj.data.materials)
    orig_mids = [p.material_index for p in target_obj.data.polygons]
    occ_backup = [(o, list(o.data.materials), [p.material_index for p in o.data.polygons], o.hide_render) for o in occluders]

    # Panel 1: Color + Red Selection Highlight + 10x10 Grid
    if selected_faces:
        red_mat = bpy.data.materials.new("SG_RedSel")
        red_mat.use_nodes = True
        bsdf = next(n for n in red_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (0.95, 0.08, 0.08, 1.0)
        bsdf.inputs["Emission Color"].default_value = (0.95, 0.10, 0.10, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.55
        r_idx = len(target_obj.data.materials)
        target_obj.data.materials.append(red_mat)
        for p in target_obj.data.polygons:
            if p.index in selected_faces:
                p.material_index = r_idx
    arr_color = _render_single_array(renders_dir / "_tmp_c.png", res)
    arr_grid = _draw_10x10_grid(arr_color)

    # Panel 2: Clay MatCap (body hidden so messy inner walls & black flipped normals are 100% exposed!)
    bg.inputs[0].default_value = (0.18, 0.19, 0.21, 1.0)
    bg.inputs[1].default_value = 0.75
    for o in occluders:
        o.hide_render = True
    clay_mat = bpy.data.materials.new("SG_Clay")
    clay_mat.use_nodes = True
    nt_c = clay_mat.node_tree
    bsdf_c = next(n for n in nt_c.nodes if n.type == "BSDF_PRINCIPLED")
    # Highlight backfaces (inner walls / flipped normals) in dark crimson-brown so Agent spots them immediately!
    geom_n = nt_c.nodes.new("ShaderNodeNewGeometry")
    mix_col = nt_c.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.inputs["A"].default_value = (0.76, 0.77, 0.79, 1.0)  # Clean front-facing clay
    mix_col.inputs["B"].default_value = (0.22, 0.06, 0.06, 1.0)  # Backfacing / inner wall warning tint
    nt_c.links.new(geom_n.outputs["Backfacing"], mix_col.inputs["Factor"])
    nt_c.links.new(mix_col.outputs["Result"], bsdf_c.inputs["Base Color"])
    bsdf_c.inputs["Roughness"].default_value = 0.55
    target_obj.data.materials.clear()
    target_obj.data.materials.append(clay_mat)
    for p in target_obj.data.polygons:
        p.material_index = 0
    arr_clay = _render_single_array(renders_dir / "_tmp_m.png", res)

    # Panel 3: Dark-Matte Body + 0.85px Orange Garment Wireframe
    bg.inputs[0].default_value = (0.14, 0.15, 0.17, 1.0)
    bg.inputs[1].default_value = 0.9
    dark_body = bpy.data.materials.new("SG_DarkBody")
    dark_body.use_nodes = True
    bsdf_b = next(n for n in dark_body.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_b.inputs["Base Color"].default_value = (0.09, 0.095, 0.11, 1.0)
    bsdf_b.inputs["Roughness"].default_value = 0.95
    for o in occluders:
        o.hide_render = False
        o.data.materials.clear()
        o.data.materials.append(dark_body)
        for p in o.data.polygons:
            p.material_index = 0

    wf_mat = bpy.data.materials.new("SG_Wire")
    wf_mat.use_nodes = True
    nt_w = wf_mat.node_tree
    nt_w.nodes.clear()
    out_n = nt_w.nodes.new("ShaderNodeOutputMaterial")
    mix_out = nt_w.nodes.new("ShaderNodeMixShader")
    mix_fill = nt_w.nodes.new("ShaderNodeMixShader")
    transp = nt_w.nodes.new("ShaderNodeBsdfTransparent")
    diff = nt_w.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (0.18, 0.19, 0.22, 1.0)
    mix_fill.inputs["Fac"].default_value = 0.86
    nt_w.links.new(diff.outputs["BSDF"], mix_fill.inputs[1])
    nt_w.links.new(transp.outputs["BSDF"], mix_fill.inputs[2])
    wire_n = nt_w.nodes.new("ShaderNodeWireframe")
    wire_n.use_pixel_size = True
    wire_n.inputs["Size"].default_value = 0.85
    emit_n = nt_w.nodes.new("ShaderNodeEmission")
    emit_n.inputs["Color"].default_value = (1.0, 0.34, 0.04, 1.0)
    emit_n.inputs["Strength"].default_value = 1.45
    nt_w.links.new(wire_n.outputs["Fac"], mix_out.inputs["Fac"])
    nt_w.links.new(mix_fill.outputs["Shader"], mix_out.inputs[1])
    nt_w.links.new(emit_n.outputs["Emission"], mix_out.inputs[2])
    nt_w.links.new(mix_out.outputs["Shader"], out_n.inputs["Surface"])

    target_obj.data.materials.clear()
    target_obj.data.materials.append(wf_mat)
    for p in target_obj.data.polygons:
        p.material_index = 0
    arr_wire = _render_single_array(renders_dir / "_tmp_w.png", res)

    # Restore all scene materials & visibility
    target_obj.data.materials.clear()
    for m in orig_mats:
        target_obj.data.materials.append(m)
    for p, mid in zip(target_obj.data.polygons, orig_mids):
        p.material_index = mid
    for o, mats_b, mids_b, h_b in occ_backup:
        o.hide_render = h_b
        o.data.materials.clear()
        for m in mats_b:
            o.data.materials.append(m)
        for p, mid in zip(o.data.polygons, mids_b):
            p.material_index = mid
    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)

    # Save individual panels + combined 1x3 triptych
    grid_p = renders_dir / "inspect_grid.png"
    clay_p = renders_dir / "inspect_matcap.png"
    wire_p = renders_dir / "inspect_wireframe.png"
    trip_p = renders_dir / "inspect_triptych.png"
    _save_rgba(arr_grid, grid_p, "SG_Grid")
    _save_rgba(arr_clay, clay_p, "SG_Clay")
    _save_rgba(arr_wire, wire_p, "SG_Wire")
    triptych = np.concatenate([arr_grid, arr_clay, arr_wire], axis=1)
    _save_rgba(triptych, trip_p, "SG_Triptych")
    return {
        "triptych": str(trip_p),
        "grid": str(grid_p),
        "matcap": str(clay_p),
        "wireframe": str(wire_p),
        "face_id_npy": str(id_path),
    }


def _parse_box(spec: str) -> tuple[float, float, float, float]:
    s = spec.strip()
    m = re.fullmatch(r"([A-Ja-j])([0-9])(?::([A-Ja-j])([0-9]))?", s)
    if m:
        c0, r0 = GRID_COLS.index(m.group(1).upper()), int(m.group(2))
        c1 = GRID_COLS.index(m.group(3).upper()) if m.group(3) else c0
        r1 = int(m.group(4)) if m.group(4) else r0
        return min(c0, c1) / 10.0, min(r0, r1) / 10.0, (max(c0, c1) + 1) / 10.0, (max(r0, r1) + 1) / 10.0
    parts = [float(v.strip()) for v in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid --box '{spec}': expected 'x0,y0,x1,y1' or 'D4:F6'")
    x0, y0, x1, y1 = parts
    return max(0.0, min(x0, x1)), max(0.0, min(y0, y1)), min(1.0, max(x0, x1)), min(1.0, max(y0, y1))


def _parse_point(spec: str) -> tuple[float, float]:
    s = spec.strip()
    m = re.fullmatch(r"([A-Ja-j])([0-9])", s)
    if m:
        return (GRID_COLS.index(m.group(1).upper()) + 0.5) / 10.0, (int(m.group(2)) + 0.5) / 10.0
    parts = [float(v.strip()) for v in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid --point '{spec}': expected 'x,y' or 'E5'")
    return max(0.0, min(1.0, parts[0])), max(0.0, min(1.0, parts[1]))


def _select_box_faces(target_obj, occluders: list, cam_spec: dict, renders_dir: Path, box_spec: str, xray: bool, depth_m: float, id_res: int) -> set[int]:
    x0, y0, x1, y1 = _parse_box(box_spec)
    id_map = np.load(str(renders_dir / "face_id.npy"))
    h, w = id_map.shape
    c0, c1 = max(0, min(w - 1, int(math.floor(x0 * w)))), max(1, min(w, int(math.ceil(x1 * w))))
    r0, r1 = max(0, min(h - 1, int(math.floor(y0 * h)))), max(1, min(h, int(math.ceil(y1 * h))))
    patch = id_map[r0:r1, c0:c1]
    picked = {int(v) for v in np.unique(patch[patch >= 0])}
    if not xray:
        return picked

    cam, created, _ = _create_studio_rig(target_obj, cam_spec, max(64, id_res))
    scene = bpy.context.scene
    t_bvh = _build_bvh(target_obj)
    o_bvhs = [_build_bvh(o) for o in occluders]
    frame = [cam.matrix_world @ v for v in cam.data.view_frame(scene=scene)]
    tr, br, bl, tl = frame[0], frame[1], frame[2], frame[3]
    origin = cam.matrix_world.translation.copy()
    vec_r, vec_d = tr - tl, bl - tl

    anchor_map = np.full((id_res, id_res), np.inf, dtype=np.float32)
    xc0, xc1 = max(0, min(id_res - 1, int(math.floor(x0 * id_res)))), max(1, min(id_res, int(math.ceil(x1 * id_res))))
    xr0, xr1 = max(0, min(id_res - 1, int(math.floor(y0 * id_res)))), max(1, min(id_res, int(math.ceil(y1 * id_res))))
    for row in range(xr0, xr1):
        row_base = tl + vec_d * ((row + 0.5) / id_res)
        for col in range(xc0, xc1):
            rdir = (row_base + vec_r * ((col + 0.5) / id_res) - origin).normalized()
            anchor = None
            gl, _, _, gd = t_bvh.ray_cast(origin, rdir, 50.0)
            if gl is not None:
                anchor = gd
            for ob in o_bvhs:
                ol, _, _, od = ob.ray_cast(origin, rdir, 50.0)
                if ol is not None and (anchor is None or od < anchor):
                    anchor = od
            if anchor is None:
                continue
            anchor_map[row, col] = float(anchor)
            cur_d = 0.0
            d_lim = anchor + max(0.02, float(depth_m))
            for _ in range(8):
                rem = d_lim - cur_d
                if rem <= 1e-4:
                    break
                hl, _, hi, hd = t_bvh.ray_cast(origin + rdir * cur_d, rdir, rem)
                if hl is None or hi is None:
                    break
                picked.add(int(hi))
                cur_d += hd + 2e-4

    mw = target_obj.matrix_world
    for p in target_obj.data.polygons:
        if p.index in picked:
            continue
        wc = mw @ p.center
        ndc = world_to_camera_view(scene, cam, wc)
        u_p, v_top = float(ndc.x), 1.0 - float(ndc.y)
        if not (x0 <= u_p <= x1 and y0 <= v_top <= y1 and ndc.z > 0.0):
            continue
        a_d = float(anchor_map[max(xr0, min(xr1 - 1, int(v_top * id_res))), max(xc0, min(xc1 - 1, int(u_p * id_res)))])
        if math.isfinite(a_d) and (a_d - 0.02) <= (wc - origin).length <= (a_d + max(0.02, float(depth_m))):
            picked.add(p.index)

    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)
    return picked


def _welded_pos_map(bm):
    pk = lambda v: (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
    pf: dict[tuple[float, float, float], list[int]] = {}
    for f in bm.faces:
        for v in f.verts:
            pf.setdefault(pk(v), []).append(f.index)
    return pk, pf


def _select_linked(target_obj, seed_faces: set[int], angle_deg: float) -> set[int]:
    if not seed_faces:
        return set()
    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    pk, pf = _welded_pos_map(bm)
    min_dot = math.cos(math.radians(angle_deg))
    sel = {fi for fi in seed_faces if 0 <= fi < len(bm.faces)}
    q = deque(sel)
    while q:
        fi = q.popleft()
        fn = bm.faces[fi].normal
        nbrs = set()
        for v in bm.faces[fi].verts:
            nbrs.update(pf.get(pk(v), ()))
        for nfi in nbrs:
            if nfi not in sel and fn.dot(bm.faces[nfi].normal) >= min_dot:
                sel.add(nfi)
                q.append(nfi)
    bm.free()
    return sel


def _grow_or_shrink(target_obj, current: set[int], steps: int, grow: bool) -> set[int]:
    if not current or steps <= 0:
        return set(current)
    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bm.faces.ensure_lookup_table()
    pk, pf = _welded_pos_map(bm)
    sel = {fi for fi in current if 0 <= fi < len(bm.faces)}
    for _ in range(steps):
        if grow:
            nxt = set(sel)
            for fi in sel:
                for v in bm.faces[fi].verts:
                    nxt.update(pf.get(pk(v), ()))
            sel = nxt
        else:
            drop = set()
            for fi in sel:
                for v in bm.faces[fi].verts:
                    if any(nfi not in sel for nfi in pf.get(pk(v), ())):
                        drop.add(fi)
                        break
            sel -= drop
    bm.free()
    return sel


def _select_backfacing_in_box(target_obj, cam_spec: dict, box_spec: str) -> set[int]:
    """Select faces inside 2D box whose world normals face AWAY from camera (inner double-walls / flipped lining)."""
    x0, y0, x1, y1 = _parse_box(box_spec) if box_spec else (0.0, 0.0, 1.0, 1.0)
    cam, created, _ = _create_studio_rig(target_obj, cam_spec, 128)
    scene = bpy.context.scene
    origin = cam.matrix_world.translation.copy()
    mw = target_obj.matrix_world
    nmat = mw.to_3x3().inverted().transposed()
    picked = set()
    for p in target_obj.data.polygons:
        wc = mw @ p.center
        ndc = world_to_camera_view(scene, cam, wc)
        u_p, v_top = float(ndc.x), 1.0 - float(ndc.y)
        if not (x0 <= u_p <= x1 and y0 <= v_top <= y1 and ndc.z > 0.0):
            continue
        wn = (nmat @ p.normal).normalized()
        to_cam = (origin - wc).normalized()
        if wn.dot(to_cam) < -0.05:
            picked.add(p.index)
    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)
    return picked


def _delete_selected_with_orphan_cleanup(target_obj, selected: set[int], orphan_max_pct: float = 3.0) -> dict:
    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bm.faces.ensure_lookup_table()
    pre_faces = len(bm.faces)
    to_del = [bm.faces[i] for i in sorted(selected) if 0 <= i < len(bm.faces)]
    pk = lambda v: (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
    cut_keys = {pk(v) for f in to_del for v in f.verts}
    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context="FACES")

    pruned_faces = 0
    pruned_islands = 0
    cap = max(4, int(pre_faces * max(0.0, float(orphan_max_pct)) / 100.0))
    if cut_keys and cap > 0:
        bm.faces.ensure_lookup_table()
        _, pf = _welded_pos_map(bm)
        vis = set()
        orphans = []
        for fi in range(len(bm.faces)):
            if fi in vis:
                continue
            comp = []
            touches = False
            q = deque([fi])
            vis.add(fi)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for v in bm.faces[cur].verts:
                    k = pk(v)
                    if k in cut_keys:
                        touches = True
                    for nfi in pf.get(k, ()):
                        if nfi not in vis:
                            vis.add(nfi)
                            q.append(nfi)
            if touches and len(comp) <= cap:
                pruned_islands += 1
                orphans.extend(bm.faces[i] for i in comp)
        if orphans:
            pruned_faces = len(orphans)
            bmesh.ops.delete(bm, geom=orphans, context="FACES")

    for e in [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]:
        bm.edges.remove(e)
    for v in [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]:
        bm.verts.remove(v)

    post_faces = len(bm.faces)
    bm.to_mesh(target_obj.data)
    bm.free()
    target_obj.data.update()
    return {
        "pre_faces": pre_faces,
        "deleted_faces": len(to_del),
        "pruned_orphan_faces": pruned_faces,
        "pruned_orphan_islands": pruned_islands,
        "post_faces": post_faces,
    }


def _bisect_current_view(target_obj, occluders: list, cam_spec: dict, line_exprs: list[str], depth_m: float, orphan_max_pct: float) -> dict:
    if not line_exprs:
        raise ValueError("--action bisect requires at least one --line 'x0,y0->x1,y1:drop=left|right|above|below'.")
    cam, created, _ = _create_studio_rig(target_obj, cam_spec, 128)
    scene = bpy.context.scene
    frame = [cam.matrix_world @ v for v in cam.data.view_frame(scene=scene)]
    tr, br, bl, tl = frame[0], frame[1], frame[2], frame[3]
    origin = cam.matrix_world.translation.copy()
    cam_fwd = (Vector(cam_spec["target"]) - origin).normalized()

    def _ray(u: float, v_top: float) -> Vector:
        return (tl + (tr - tl) * u + (bl - tl) * v_top - origin).normalized()

    world_planes: list[tuple[Vector, Vector]] = []
    for expr in line_exprs:
        parts = [p.strip() for p in expr.split(":")]
        if len(parts) != 2 or "->" not in parts[0] or not parts[1].startswith("drop="):
            raise ValueError(f"Invalid --line '{expr}': expected 'x0,y0->x1,y1:drop=left|right|above|below'")
        p0_s, p1_s = [s.strip() for s in parts[0].split("->", 1)]
        x0, y0 = [float(v.strip()) for v in p0_s.split(",")]
        x1, y1 = [float(v.strip()) for v in p1_s.split(",")]
        drop = parts[1].split("=", 1)[1].strip().lower()
        pno = _ray(x0, y0).cross(_ray(x1, y1)).normalized()
        du, dv = {"left": (-0.06, 0.0), "right": (0.06, 0.0), "above": (0.0, -0.06), "below": (0.0, 0.06)}[drop]
        if _ray(0.5 * (x0 + x1) + du, 0.5 * (y0 + y1) + dv).dot(pno) < 0.0:
            pno = -pno
        world_planes.append((origin.copy(), pno))

    if depth_m > 0.0:
        mw_t = target_obj.matrix_world
        ds = [(mw_t @ p.center - origin).dot(cam_fwd) for p in target_obj.data.polygons if all((mw_t @ p.center - pco).dot(pno) >= -0.005 for pco, pno in world_planes)]
        if ds:
            front_d = min(ds)
            world_planes.append((origin + cam_fwd * (front_d + float(depth_m)), -cam_fwd))

    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)

    mw = target_obj.matrix_world
    mw_inv = mw.inverted()
    mw_inv_t = mw_inv.transposed().to_3x3()
    local_planes = [(mw_inv @ pco, (mw_inv_t @ pno).normalized()) for pco, pno in world_planes]

    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bm.faces.ensure_lookup_table()
    pre_faces = len(bm.faces)
    sliced_edges = 0
    for lco, lno in local_planes:
        cands = [f for f in bm.faces if f.is_valid and any(all((v.co - pc).dot(pn) >= -0.02 for pc, pn in local_planes) for v in f.verts)]
        if not cands:
            continue
        res = bmesh.ops.bisect_plane(
            bm,
            geom=cands + list({e for f in cands for e in f.edges}) + list({v for f in cands for v in f.verts}),
            dist=1e-5,
            plane_co=lco,
            plane_no=lno,
        )
        sliced_edges += sum(1 for g in res.get("geom_cut", []) if isinstance(g, bmesh.types.BMEdge))

    bm.faces.ensure_lookup_table()
    del_indices = {f.index for f in bm.faces if f.is_valid and all((f.calc_center_median() - lco).dot(lno) > 1e-5 for lco, lno in local_planes)}
    bm.to_mesh(target_obj.data)
    bm.free()
    stats = _delete_selected_with_orphan_cleanup(target_obj, del_indices, orphan_max_pct=orphan_max_pct)
    stats["pre_faces"] = pre_faces
    stats["sliced_edges"] = sliced_edges
    return stats


def _clean_mesh(target_obj, quadify: bool = False, recalc_normals: bool = False) -> dict:
    """Remove degenerate zero-area faces, optionally convert tris to quads, and optionally recalculate normals."""
    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    pre_faces = len(bm.faces)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=list(bm.edges))
    if quadify:
        bmesh.ops.join_triangles(
            bm,
            faces=list(bm.faces),
            angle_face_threshold=math.radians(38.0),
            angle_shape_threshold=math.radians(38.0),
            cmp_uvs=True,
            cmp_materials=True,
        )
    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    for e in [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]:
        bm.edges.remove(e)
    for v in [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]:
        bm.verts.remove(v)
    post_faces = len(bm.faces)
    quad_faces = sum(1 for f in bm.faces if len(f.verts) == 4)
    bm.to_mesh(target_obj.data)
    bm.free()
    for p in target_obj.data.polygons:
        p.use_smooth = True
    target_obj.data.update()
    return {
        "pre_faces": pre_faces,
        "post_faces": post_faces,
        "quad_ratio": round(quad_faces / max(1, post_faces), 4),
        "normals_recalculated": bool(recalc_normals),
    }


def _save_snapshot(session_dir: Path, step: int) -> Path:
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dst = snap_dir / f"step_{step:03d}.blend"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    return dst


def main():
    args = _parse_args()
    sdir = Path(args.session)
    sdir.mkdir(parents=True, exist_ok=True)
    state_p = sdir / "session_state.json"
    renders_dir = sdir / "renders"

    if args.action == "open":
        if not args.input:
            raise ValueError("--action open requires --input <file.blend|file.glb>.")
        _load_scene(args.input)
        target_obj = _get_target_mesh(args.target_mesh)
        occluders = [o for o in bpy.context.scene.objects if o.type == "MESH" and o != target_obj and len(o.data.polygons) > 0]
        snap0 = _save_snapshot(sdir, 0)
        cam_spec = _compute_camera_spec(target_obj, args.view, args.zoom)
        renders = _render_triptych_and_ids(target_obj, occluders, cam_spec, renders_dir, args.res, args.id_res, set())
        state = {
            "step": 0,
            "input": str(args.input),
            "target_mesh": target_obj.name,
            "current_blend": str(snap0),
            "camera": cam_spec,
            "selected_faces": [],
            "faces": len(target_obj.data.polygons),
            "history": [{"step": 0, "action": "open", "blend": str(snap0), "faces": len(target_obj.data.polygons)}],
            "renders": renders,
        }
        state_p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if not state_p.is_file():
        raise FileNotFoundError(f"No active session in {sdir}. Run --action open first.")
    state = json.loads(state_p.read_text(encoding="utf-8"))

    if args.action == "undo":
        if len(state["history"]) <= 1:
            print(json.dumps({"status": "already_at_step_0", "step": 0}, ensure_ascii=False, indent=2))
            return
        state["history"].pop()
        prev = state["history"][-1]
        state["step"] = prev["step"]
        state["current_blend"] = prev["blend"]
        state["selected_faces"] = []
        _load_scene(state["current_blend"])
        target_obj = _get_target_mesh(state["target_mesh"])
        occluders = [o for o in bpy.context.scene.objects if o.type == "MESH" and o != target_obj and len(o.data.polygons) > 0]
        renders = _render_triptych_and_ids(target_obj, occluders, state["camera"], renders_dir, args.res, args.id_res, set())
        state["faces"] = len(target_obj.data.polygons)
        state["renders"] = renders
        state_p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "undone", "step": state["step"], "faces": state["faces"], "renders": renders}, ensure_ascii=False, indent=2))
        return

    _load_scene(state["current_blend"])
    target_obj = _get_target_mesh(state["target_mesh"])
    occluders = [o for o in bpy.context.scene.objects if o.type == "MESH" and o != target_obj and len(o.data.polygons) > 0]
    selected = set(state.get("selected_faces", []))
    cam_spec = state["camera"]
    extra_stats = {}

    if args.action == "look":
        cam_spec = _compute_camera_spec(target_obj, args.view, args.zoom, focus_faces=selected if args.focus_selected else None)
        state["camera"] = cam_spec
    elif args.action == "select-box":
        if not args.box:
            raise ValueError("--action select-box requires --box 'x0,y0,x1,y1' or 'D4:F6'.")
        hits = _select_box_faces(target_obj, occluders, cam_spec, renders_dir, args.box, args.xray, args.depth_m, args.id_res)
        selected = hits if args.op == "set" else (selected | hits if args.op == "add" else selected - hits)
    elif args.action == "select-linked":
        seeds = set(selected)
        if args.point:
            px, py = _parse_point(args.point)
            id_map = np.load(str(renders_dir / "face_id.npy"))
            h, w = id_map.shape
            r0 = max(0, min(h - 1, int(py * h)))
            c0 = max(0, min(w - 1, int(px * w)))
            patch = id_map[max(0, r0 - 2) : min(h, r0 + 3), max(0, c0 - 2) : min(w, c0 + 3)]
            seeds = {int(v) for v in np.unique(patch[patch >= 0])}
        hits = _select_linked(target_obj, seeds, args.angle_deg)
        selected = hits if args.op == "set" else (selected | hits if args.op == "add" else selected - hits)
    elif args.action == "select-backface":
        hits = _select_backfacing_in_box(target_obj, cam_spec, args.box)
        selected = hits if args.op == "set" else (selected | hits if args.op == "add" else selected - hits)
    elif args.action == "grow":
        selected = _grow_or_shrink(target_obj, selected, args.steps, grow=True)
    elif args.action == "shrink":
        selected = _grow_or_shrink(target_obj, selected, args.steps, grow=False)
    elif args.action in ("delete", "bisect", "clean"):
        if args.action == "delete":
            extra_stats = _delete_selected_with_orphan_cleanup(target_obj, selected, orphan_max_pct=args.orphan_max_pct)
            selected = set()
        elif args.action == "bisect":
            extra_stats = _bisect_current_view(target_obj, occluders, cam_spec, args.line, args.depth_m, args.orphan_max_pct)
            selected = set()
        elif args.action == "clean":
            extra_stats = _clean_mesh(target_obj, quadify=args.quadify, recalc_normals=args.recalc_normals)
            selected = set()
        new_step = int(state["step"]) + 1
        snap = _save_snapshot(sdir, new_step)
        state["step"] = new_step
        state["current_blend"] = str(snap)
        state["faces"] = len(target_obj.data.polygons)
        state["history"].append({"step": new_step, "action": args.action, "blend": str(snap), "faces": state["faces"], "stats": extra_stats})
    elif args.action == "export":
        out_b = Path(args.out_blend) if args.out_blend else (sdir / "surgeon_output.blend")
        out_g = Path(args.out_glb) if args.out_glb else (sdir / "surgeon_output.glb")
        out_b.parent.mkdir(parents=True, exist_ok=True)
        out_g.parent.mkdir(parents=True, exist_ok=True)
        remove_export_junk()
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out_b))
        bpy.ops.export_scene.gltf(filepath=str(out_g), export_format="GLB", export_animations=True, export_skins=True)
        rep = {"status": "exported", "step": state["step"], "faces": len(target_obj.data.polygons), "out_blend": str(out_b), "out_glb": str(out_g)}
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return

    renders = _render_triptych_and_ids(target_obj, occluders, cam_spec, renders_dir, args.res, args.id_res, selected)
    state["selected_faces"] = sorted(selected)
    state["faces"] = len(target_obj.data.polygons)
    state["renders"] = renders
    state_p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    out_summary = {
        "action": args.action,
        "step": state["step"],
        "faces": state["faces"],
        "selected_count": len(selected),
        "stats": extra_stats,
        "renders": renders,
    }
    print(json.dumps(out_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
