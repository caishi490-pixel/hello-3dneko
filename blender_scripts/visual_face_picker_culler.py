"""Universal Visual Face-ID Picker & Surgical Culler for 3D Meshes (Blender 5.1+).

Three-step closed loop allowing any multimodal Vision Agent to pick and delete unwanted 3D faces
directly from rendered images without hardcoded geometric coordinates:
  1. capture: Render multi-view images with crisp 10x10 A0..J9 coordinate grids AND export exact
              per-pixel 3D face-index buffers (`face_id_<view>.npy`) via camera raycasting
              (zero anti-aliasing ID corruption, occlusion-aware across multi-object scenes).
  2. preview: Convert Agent-supplied 2D boxes/grid cells (`front_skirt:D5:F7` or `0.35,0.50,0.65,0.72`)
              into 3D seed faces, optionally flood-fill across smooth dihedral angles, and render
              high-contrast red highlight previews for visual verification.
  3. apply:   Delete confirmed faces with UV-seam safety (zero unintended UV-island pruning or cracks)
              and export cleaned .blend/.glb + Before/After comparison sheets.
"""

import argparse
import json
import math
import re
import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view


GRID_COLS = "ABCDEFGHIJ"


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Universal Face-ID visual picker & surgical face culler.")
    p.add_argument("--input", required=True, help="Input .blend or .glb scene.")
    p.add_argument("--out-dir", required=True, help="Workspace directory for capture/preview/apply outputs.")
    p.add_argument("--mode", choices=["capture", "preview", "apply", "bisect"], required=True)
    p.add_argument("--target-mesh", default="", help="Optional exact mesh object name (defaults to garment or largest mesh).")
    p.add_argument("--views", default="front,front_skirt,bottom_up,back", help="Comma-separated camera views to capture.")
    p.add_argument("--res", type=int, default=512, help="Render resolution for grid inspection views.")
    p.add_argument("--id-res", type=int, default=256, help="Exact raycast Face-ID buffer resolution.")
    p.add_argument(
        "--pick",
        action="append",
        default=[],
        help="2D region to pick: 'view:x0,y0,x1,y1' (normalized 0..1, top-left origin) OR 'view:C4:F7' (A..J cols, 0..9 rows).",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Optional 2D protection region to exclude from selection (same syntax as --pick).",
    )
    p.add_argument(
        "--bisect",
        action="append",
        default=[],
        help="Surgical knife cut line: 'view:x0,y0->x1,y1:drop=left|right|above|below' (multi-plane intersection creates razor-straight UV/weight-safe cuts).",
    )
    p.add_argument(
        "--bisect-depth-m",
        type=float,
        default=0.14,
        help="Depth band (meters past front surface) limiting surgical bisect cuts so back layers stay untouched.",
    )
    p.add_argument("--grow-angle-deg", type=float, default=38.0, help="Max dihedral angle for smooth flood-fill from picked seeds.")
    p.add_argument("--grow-hops", type=int, default=2, help="Max topological face hops to grow from picked visible seeds.")
    p.add_argument(
        "--xray-garment",
        action="store_true",
        help="Penetrating raycast through occluders and garment walls inside --pick boxes (catches back walls & inner tubes).",
    )
    p.add_argument(
        "--xray-depth-m",
        type=float,
        default=0.22,
        help="Max penetration depth (meters past first hit) when --xray-garment is enabled.",
    )
    p.add_argument("--mask-json", default="", help="Path to confirmed selected_faces.json for apply mode.")
    p.add_argument("--prune-cut-slivers", type=int, default=4, help="Prune tiny isolated slivers (<=N faces) touching cut edges.")
    p.add_argument(
        "--orphan-max-pct",
        type=float,
        default=3.0,
        help="Auto-prune 3D-welded floating orphan islands/rings touching the cut boundary whose size is <= this percent of pre-cull faces.",
    )
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


def _scene_bounds(target_obj):
    mw = target_obj.matrix_world
    pts = [mw @ v.co for v in target_obj.data.vertices]
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    span = max((max_p - min_p).x, (max_p - min_p).y, (max_p - min_p).z, 0.1)
    return min_p, max_p, center, span


def _build_view_specs(target_obj, view_names: list[str]) -> list[dict]:
    min_p, max_p, center, span = _scene_bounds(target_obj)
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
    specs = []
    for vn in view_names:
        vn = vn.strip()
        if vn not in presets:
            raise ValueError(f"Unknown view '{vn}'; choose from {sorted(presets.keys())}")
        loc, tgt, lens = presets[vn]
        specs.append({"name": vn, "location": [loc.x, loc.y, loc.z], "target": [tgt.x, tgt.y, tgt.z], "lens": lens})
    return specs


def _setup_studio_scene(target_obj, res: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.refresh_devices()
        if any(d.type != "CPU" for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type != "CPU"
            scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.view_settings.view_transform = "Standard"

    _, _, center, span = _scene_bounds(target_obj)
    w = bpy.data.worlds.new("VisualPickerWorld")
    scene.world = w
    if w.node_tree is None:
        w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.86, 0.87, 0.90, 1.0)
    bg.inputs[1].default_value = 0.65

    created_objs = []
    for lname, off, eng in [
        ("VP_Key", Vector((-1.2, -1.6, 1.0)), 110),
        ("VP_Fill", Vector((1.4, -1.2, 0.4)), 80),
        ("VP_Bottom", Vector((0.0, -0.4, -1.6)), 120),
        ("VP_Back", Vector((0.0, 1.8, 0.8)), 95),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng * max(1.0, span * span)
        ld.size = span * 1.4
        lo = bpy.data.objects.new(lname, ld)
        scene.collection.objects.link(lo)
        lo.location = center + off * span
        lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()
        created_objs.append(lo)

    cam_d = bpy.data.cameras.new("VPCam")
    cam = bpy.data.objects.new("VPCam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    created_objs.append(cam)
    return cam, created_objs


def _position_camera(cam, spec: dict):
    loc = Vector(spec["location"])
    tgt = Vector(spec["target"])
    cam.data.lens = float(spec["lens"])
    cam.location = loc
    cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def _build_eval_bvh_with_poly_map(obj):
    """Build world-space BVH preserving exact mapping from BVH face index -> original polygon index."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    bvh = BVHTree.FromPolygons(verts, polys)
    return bvh


def _raycast_face_id_buffer(cam, target_obj, occluder_objs, id_res: int) -> np.ndarray:
    """Compute exact (id_res, id_res) int32 Face-ID map (row 0 = top, col 0 = left) with occlusion awareness."""
    target_bvh = _build_eval_bvh_with_poly_map(target_obj)
    occ_bvhs = [_build_eval_bvh_with_poly_map(o) for o in occluder_objs]

    # Exact camera view frame corners in world space (Blender returns: top-right, bottom-right, bottom-left, top-left)
    scene = bpy.context.scene
    frame = [cam.matrix_world @ v for v in cam.data.view_frame(scene=scene)]
    tr, br, bl, tl = frame[0], frame[1], frame[2], frame[3]
    origin = cam.matrix_world.translation.copy()

    id_map = np.full((id_res, id_res), -1, dtype=np.int32)
    vec_right = tr - tl
    vec_down = bl - tl
    max_dist = 50.0

    for row in range(id_res):
        v_f = (row + 0.5) / id_res
        row_base = tl + vec_down * v_f
        for col in range(id_res):
            u_f = (col + 0.5) / id_res
            ray_dir = (row_base + vec_right * u_f - origin).normalized()
            loc, _, f_idx, dist = target_bvh.ray_cast(origin, ray_dir, max_dist)
            if loc is None or f_idx is None:
                continue
            occluded = False
            for obvh in occ_bvhs:
                o_loc, _, _, o_dist = obvh.ray_cast(origin, ray_dir, dist - 1e-4)
                if o_loc is not None and o_dist < dist - 1e-4:
                    occluded = True
                    break
            if not occluded:
                id_map[row, col] = int(f_idx)
    return id_map


def _draw_10x10_grid_overlay(img_arr: np.ndarray) -> np.ndarray:
    """Draw clean 10x10 coordinate grid lines & crisp corner markers directly in numpy (row 0 = bottom in Blender)."""
    out = img_arr.copy()
    h, w, _ = out.shape
    # Draw subtle dark-teal grid lines at 10% intervals (top-left origin corresponds to row_top = 0..h-1 -> y_bl = h-1-row_top)
    for k in range(1, 10):
        px = int(round(k * w / 10.0))
        py_top = int(round(k * h / 10.0))
        py_bl = h - 1 - py_top
        thickness = 2 if k == 5 else 1
        color = np.array([0.08, 0.85, 0.85, 1.0], dtype=np.float32) if k == 5 else np.array([0.12, 0.16, 0.22, 1.0], dtype=np.float32)
        alpha = 0.72 if k == 5 else 0.42
        x0, x1 = max(0, px - thickness // 2), min(w, px + (thickness + 1) // 2)
        y0, y1 = max(0, py_bl - thickness // 2), min(h, py_bl + (thickness + 1) // 2)
        out[:, x0:x1, :3] = out[:, x0:x1, :3] * (1.0 - alpha) + color[:3] * alpha
        out[y0:y1, :, :3] = out[y0:y1, :, :3] * (1.0 - alpha) + color[:3] * alpha

    # Draw minimal 3x5 bitmap labels (A..J along top edge, 0..9 along left edge) for instant human/VLM reading
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

    def _stamp(ch: str, cx_top: int, cy_top: int, scale: int = 3):
        g = glyphs[ch]
        gh, gw = len(g) * scale, len(g[0]) * scale
        bx0 = max(0, cx_top - gw // 2 - 2)
        bx1 = min(w, cx_top + gw // 2 + 3)
        by_top0 = max(0, cy_top - gh // 2 - 2)
        by_top1 = min(h, cy_top + gh // 2 + 3)
        out[h - by_top1 : h - by_top0, bx0:bx1, :3] = 0.08
        for r_i, row_s in enumerate(g):
            for c_i, bit in enumerate(row_s):
                if bit == "1":
                    px0 = cx_top - gw // 2 + c_i * scale
                    py_t0 = cy_top - gh // 2 + r_i * scale
                    out[h - (py_t0 + scale) : h - py_t0, px0 : px0 + scale, :3] = 0.98

    for k in range(10):
        _stamp(GRID_COLS[k], int((k + 0.5) * w / 10.0), 14, scale=3)
        _stamp(str(k), 14, int((k + 0.5) * h / 10.0), scale=3)
    return out


def _save_rgba_png(arr: np.ndarray, dst: Path, name: str):
    h, w, _ = arr.shape
    img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    img.pixels.foreach_set(arr.ravel())
    img.filepath_raw = str(dst)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def _parse_pick_expr(expr: str) -> tuple[str, float, float, float, float]:
    """Parse 'view:x0,y0,x1,y1' or 'view:C4:F7' into (view_name, x0, y0, x1, y1) in [0, 1] top-left coords."""
    if ":" not in expr:
        raise ValueError(f"Invalid pick expression '{expr}': expected 'view:coords'")
    view_name, spec = expr.split(":", 1)
    view_name = view_name.strip()
    spec = spec.strip()

    m_grid = re.fullmatch(r"([A-Ja-j])([0-9])(?::([A-Ja-j])([0-9]))?", spec)
    if m_grid:
        c0 = GRID_COLS.index(m_grid.group(1).upper())
        r0 = int(m_grid.group(2))
        c1 = GRID_COLS.index(m_grid.group(3).upper()) if m_grid.group(3) else c0
        r1 = int(m_grid.group(4)) if m_grid.group(4) else r0
        x0, x1 = min(c0, c1) / 10.0, (max(c0, c1) + 1) / 10.0
        y0, y1 = min(r0, r1) / 10.0, (max(r0, r1) + 1) / 10.0
        return view_name, x0, y0, x1, y1

    parts = [float(p.strip()) for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid box '{expr}': expected 4 floats x0,y0,x1,y1 or grid cells like D5:F7")
    x0, y0, x1, y1 = parts
    return view_name, max(0.0, min(x0, x1)), max(0.0, min(y0, y1)), min(1.0, max(x0, x1)), min(1.0, max(y0, y1))


def _faces_from_pick_boxes(out_dir: Path, pick_exprs: list[str]) -> set[int]:
    picked: set[int] = set()
    for expr in pick_exprs:
        vname, x0, y0, x1, y1 = _parse_pick_expr(expr)
        npy_path = out_dir / f"face_id_{vname}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"Missing Face-ID buffer for view '{vname}': {npy_path}. Run --mode capture first.")
        id_map = np.load(str(npy_path))
        h, w = id_map.shape
        c0 = max(0, min(w - 1, int(math.floor(x0 * w))))
        c1 = max(c0 + 1, min(w, int(math.ceil(x1 * w))))
        r0 = max(0, min(h - 1, int(math.floor(y0 * h))))
        r1 = max(r0 + 1, min(h, int(math.ceil(y1 * h))))
        patch = id_map[r0:r1, c0:c1]
        valid = patch[patch >= 0]
        picked.update(int(v) for v in np.unique(valid))
    return picked


def _xray_faces_from_pick_boxes(
    target_obj,
    occluder_objs,
    view_specs: list[dict],
    pick_exprs: list[str],
    id_res: int,
    xray_depth_m: float = 0.22,
    max_layers: int = 8,
) -> set[int]:
    """Penetrating multi-hit raycast inside 2D pick boxes to capture front walls, inner walls, and back walls behind limbs."""
    if not pick_exprs:
        return set()
    spec_by_name = {sp["name"]: sp for sp in view_specs}
    target_bvh = _build_eval_bvh_with_poly_map(target_obj)
    occ_bvhs = [_build_eval_bvh_with_poly_map(o) for o in occluder_objs]
    cam, created = _setup_studio_scene(target_obj, max(64, id_res))
    scene = bpy.context.scene
    picked: set[int] = set()
    max_dist = 50.0

    for expr in pick_exprs:
        vname, x0, y0, x1, y1 = _parse_pick_expr(expr)
        if vname not in spec_by_name:
            raise ValueError(f"Unknown view '{vname}' in X-Ray pick '{expr}'.")
        _position_camera(cam, spec_by_name[vname])
        frame = [cam.matrix_world @ v for v in cam.data.view_frame(scene=scene)]
        tr, br, bl, tl = frame[0], frame[1], frame[2], frame[3]
        origin = cam.matrix_world.translation.copy()
        vec_right = tr - tl
        vec_down = bl - tl

        c0 = max(0, min(id_res - 1, int(math.floor(x0 * id_res))))
        c1 = max(c0 + 1, min(id_res, int(math.ceil(x1 * id_res))))
        r0 = max(0, min(id_res - 1, int(math.floor(y0 * id_res))))
        r1 = max(r0 + 1, min(id_res, int(math.ceil(y1 * id_res))))

        anchor_map = np.full((id_res, id_res), np.inf, dtype=np.float32)
        for row in range(r0, r1):
            v_f = (row + 0.5) / id_res
            row_base = tl + vec_down * v_f
            for col in range(c0, c1):
                u_f = (col + 0.5) / id_res
                ray_dir = (row_base + vec_right * u_f - origin).normalized()
                # Anchor depth band at first hit on garment or occluder (e.g. leg skin)
                anchor_dist = None
                g_loc, _, g_idx, g_dist = target_bvh.ray_cast(origin, ray_dir, max_dist)
                if g_loc is not None:
                    anchor_dist = g_dist
                for obvh in occ_bvhs:
                    o_loc, _, _, o_dist = obvh.ray_cast(origin, ray_dir, max_dist)
                    if o_loc is not None and (anchor_dist is None or o_dist < anchor_dist):
                        anchor_dist = o_dist
                if anchor_dist is None:
                    continue
                anchor_map[row, col] = float(anchor_dist)

                depth_limit = anchor_dist + max(0.02, float(xray_depth_m))
                cur_dist = 0.0
                for _ in range(max_layers):
                    cur_origin = origin + ray_dir * cur_dist
                    rem = depth_limit - cur_dist
                    if rem <= 1e-4:
                        break
                    h_loc, _, h_idx, h_dist = target_bvh.ray_cast(cur_origin, ray_dir, rem)
                    if h_loc is None or h_idx is None:
                        break
                    picked.add(int(h_idx))
                    cur_dist += h_dist + 2e-4

        # Sub-pixel face-centroid projection sweep: catches razor-thin 1mm ribbons that slip between raycast scanlines
        mw = target_obj.matrix_world
        for poly in target_obj.data.polygons:
            fi = poly.index
            if fi in picked:
                continue
            w_co = mw @ poly.center
            ndc = world_to_camera_view(scene, cam, w_co)
            u_p = float(ndc.x)
            v_top = 1.0 - float(ndc.y)
            if not (x0 <= u_p <= x1 and y0 <= v_top <= y1 and ndc.z > 0.0):
                continue
            p_col = max(c0, min(c1 - 1, int(u_p * id_res)))
            p_row = max(r0, min(r1 - 1, int(v_top * id_res)))
            a_dist = float(anchor_map[p_row, p_col])
            if not math.isfinite(a_dist):
                continue
            ray_d = (w_co - origin).length
            if (a_dist - 0.02) <= ray_d <= (a_dist + max(0.02, float(xray_depth_m))):
                picked.add(fi)

    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)
    return picked


def _grow_faces_smooth(obj, seed_faces: set[int], exclude_faces: set[int], angle_deg: float, max_hops: int) -> set[int]:
    if not seed_faces or max_hops <= 0:
        return set(seed_faces) - exclude_faces
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    min_dot = math.cos(math.radians(angle_deg))

    # Build welded edge adjacency across quantized positions so UV seams don't block smooth flood-fill
    pos_faces: dict[tuple[float, float, float], list[int]] = {}
    for f in bm.faces:
        for v in f.verts:
            key = (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
            pos_faces.setdefault(key, []).append(f.index)

    selected = set(seed_faces) - exclude_faces
    q = deque((fi, 0) for fi in selected if 0 <= fi < len(bm.faces))
    while q:
        fi, depth = q.popleft()
        if depth >= max_hops:
            continue
        fn = bm.faces[fi].normal
        nbr_faces = set()
        for v in bm.faces[fi].verts:
            key = (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
            nbr_faces.update(pos_faces.get(key, ()))
        for nfi in nbr_faces:
            if nfi in selected or nfi in exclude_faces:
                continue
            if fn.dot(bm.faces[nfi].normal) >= min_dot:
                selected.add(nfi)
                q.append((nfi, depth + 1))
    bm.free()
    return selected


def _render_views_sheet(obj, view_specs: list[dict], out_dir: Path, out_name: str, res: int, highlight_faces: set[int] | None = None, draw_grid: bool = False):
    cam, created = _setup_studio_scene(obj, res)
    orig_mats = list(obj.data.materials)
    orig_mids = [p.material_index for p in obj.data.polygons]

    if highlight_faces is not None:
        red_mat = bpy.data.materials.new("VP_Red_Highlight")
        if red_mat.node_tree is None:
            red_mat.use_nodes = True
        bsdf = next(n for n in red_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (0.95, 0.08, 0.08, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Emission Color"].default_value = (0.95, 0.10, 0.10, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.45
        r_idx = len(obj.data.materials)
        obj.data.materials.append(red_mat)
        for p in obj.data.polygons:
            if p.index in highlight_faces:
                p.material_index = r_idx

    saved_individual = {}
    tiles = []
    for sp in view_specs:
        _position_camera(cam, sp)
        tmp_p = out_dir / f"_vp_tmp_{sp['name']}.png"
        bpy.context.scene.render.filepath = str(tmp_p)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(tmp_p))
        arr = np.array(img.pixels[:], dtype=np.float32).reshape((res, res, 4))
        bpy.data.images.remove(img)
        tmp_p.unlink(missing_ok=True)
        if draw_grid:
            arr_g = _draw_10x10_grid_overlay(arr)
            indiv_p = out_dir / f"grid_{sp['name']}.png"
            _save_rgba_png(arr_g, indiv_p, f"Grid_{sp['name']}")
            saved_individual[sp["name"]] = str(indiv_p)
            tiles.append(arr_g)
        else:
            tiles.append(arr)

    # Restore materials & remove studio rig
    if highlight_faces is not None:
        for p, mid in zip(obj.data.polygons, orig_mids):
            p.material_index = mid
        obj.data.materials.clear()
        for m in orig_mats:
            obj.data.materials.append(m)
    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)

    # Assemble 2x2 sheet (or 1xN if fewer than 3 views)
    n = len(tiles)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    sheet = np.full((rows * res, cols * res, 4), 0.88, dtype=np.float32)
    sheet[..., 3] = 1.0
    for idx, t in enumerate(tiles):
        r_top = idx // cols
        c = idx % cols
        r_bl = rows - 1 - r_top
        sheet[r_bl * res : (r_bl + 1) * res, c * res : (c + 1) * res] = t
    sheet_p = out_dir / out_name
    _save_rgba_png(sheet, sheet_p, f"VPSheet_{out_name}")
    return str(sheet_p), saved_individual


def _render_wireframe_xray_sheet(obj, occluders: list, view_specs: list[dict], out_dir: Path, out_name: str, res: int) -> str:
    """Render high-contrast Blender-style dark-matte body + crisp thin orange garment wireframe audit."""
    cam, created = _setup_studio_scene(obj, res)
    scene = bpy.context.scene
    if scene.world and scene.world.node_tree:
        bg = next((n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"), None)
        if bg is not None:
            bg.inputs[0].default_value = (0.14, 0.15, 0.17, 1.0)
            bg.inputs[1].default_value = 0.9

    # Replace occluder (body) materials with dark matte charcoal so any stray wire across the legs pops out instantly
    occ_backup = []
    dark_body_mat = bpy.data.materials.new("VP_DarkMatteBody")
    if dark_body_mat.node_tree is None:
        dark_body_mat.use_nodes = True
    bsdf_b = next(n for n in dark_body_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_b.inputs["Base Color"].default_value = (0.09, 0.095, 0.11, 1.0)
    bsdf_b.inputs["Roughness"].default_value = 0.95
    for o in occluders:
        occ_backup.append((o, list(o.data.materials), [p.material_index for p in o.data.polygons]))
        o.data.materials.clear()
        o.data.materials.append(dark_body_mat)
        for p in o.data.polygons:
            p.material_index = 0

    orig_mats = list(obj.data.materials)
    orig_mids = [p.material_index for p in obj.data.polygons]

    wf_mat = bpy.data.materials.new("VP_Wireframe_XRay")
    if wf_mat.node_tree is None:
        wf_mat.use_nodes = True
    nt = wf_mat.node_tree
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    mix_outer = nt.nodes.new("ShaderNodeMixShader")
    mix_fill = nt.nodes.new("ShaderNodeMixShader")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    dark_bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    dark_bsdf.inputs["Color"].default_value = (0.18, 0.19, 0.22, 1.0)
    mix_fill.inputs["Fac"].default_value = 0.86  # 86% transparent interior so wires stay razor crisp
    nt.links.new(dark_bsdf.outputs["BSDF"], mix_fill.inputs[1])
    nt.links.new(transp.outputs["BSDF"], mix_fill.inputs[2])

    wire_n = nt.nodes.new("ShaderNodeWireframe")
    wire_n.use_pixel_size = True
    wire_n.inputs["Size"].default_value = 0.85
    emit_n = nt.nodes.new("ShaderNodeEmission")
    emit_n.inputs["Color"].default_value = (1.0, 0.34, 0.04, 1.0)
    emit_n.inputs["Strength"].default_value = 1.45

    nt.links.new(wire_n.outputs["Fac"], mix_outer.inputs["Fac"])
    nt.links.new(mix_fill.outputs["Shader"], mix_outer.inputs[1])
    nt.links.new(emit_n.outputs["Emission"], mix_outer.inputs[2])
    nt.links.new(mix_outer.outputs["Shader"], out_n.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(wf_mat)
    for p in obj.data.polygons:
        p.material_index = 0

    tiles = []
    for sp in view_specs:
        _position_camera(cam, sp)
        tmp_p = out_dir / f"_vp_wf_{sp['name']}.png"
        scene.render.filepath = str(tmp_p)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(tmp_p))
        arr = np.array(img.pixels[:], dtype=np.float32).reshape((res, res, 4))
        bpy.data.images.remove(img)
        tmp_p.unlink(missing_ok=True)
        tiles.append(arr)

    # Restore scene state
    for o, mats_b, mids_b in occ_backup:
        o.data.materials.clear()
        for m in mats_b:
            o.data.materials.append(m)
        for p, mid in zip(o.data.polygons, mids_b):
            p.material_index = mid
    obj.data.materials.clear()
    for m in orig_mats:
        obj.data.materials.append(m)
    for p, mid in zip(obj.data.polygons, orig_mids):
        p.material_index = mid
    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)

    n = len(tiles)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    sheet = np.full((rows * res, cols * res, 4), 0.06, dtype=np.float32)
    sheet[..., 3] = 1.0
    for idx, t in enumerate(tiles):
        r_top = idx // cols
        c = idx % cols
        r_bl = rows - 1 - r_top
        sheet[r_bl * res : (r_bl + 1) * res, c * res : (c + 1) * res] = t
    sheet_p = out_dir / out_name
    _save_rgba_png(sheet, sheet_p, f"VPWireXRay_{out_name}")
    return str(sheet_p)


def _apply_uv_safe_delete(obj, face_indices: list[int], prune_slivers: int = 4, orphan_max_pct: float = 3.0) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    pre_faces = len(bm.faces)
    to_del = [bm.faces[i] for i in sorted(set(face_indices)) if 0 <= i < len(bm.faces)]

    def _pos_key(v):
        return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))

    cut_pos_keys = {_pos_key(v) for f in to_del for v in f.verts}
    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context="FACES")

    # 3D-welded cut-adjacent orphan ring/sliver cleanup (preserves UV seams on kept mesh, never touches unrelated islands)
    pruned_orphan_faces = 0
    pruned_orphan_islands = 0
    orphan_cap = max(int(prune_slivers), int(pre_faces * max(0.0, float(orphan_max_pct)) / 100.0))
    if orphan_cap > 0 and cut_pos_keys:
        bm.faces.ensure_lookup_table()
        pos_to_faces: dict[tuple[float, float, float], list[int]] = {}
        for f in bm.faces:
            for v in f.verts:
                pos_to_faces.setdefault(_pos_key(v), []).append(f.index)

        visited = set()
        orphan_faces = []
        for fi in range(len(bm.faces)):
            if fi in visited:
                continue
            comp = []
            touches_cut = False
            q = deque([fi])
            visited.add(fi)
            while q:
                cur = q.popleft()
                comp.append(cur)
                f_obj = bm.faces[cur]
                for v in f_obj.verts:
                    pk = _pos_key(v)
                    if pk in cut_pos_keys:
                        touches_cut = True
                    for nfi in pos_to_faces.get(pk, ()):
                        if nfi not in visited:
                            visited.add(nfi)
                            q.append(nfi)
            if touches_cut and len(comp) <= orphan_cap:
                pruned_orphan_islands += 1
                orphan_faces.extend(bm.faces[i] for i in comp)

        if orphan_faces:
            pruned_orphan_faces = len(orphan_faces)
            bmesh.ops.delete(bm, geom=orphan_faces, context="FACES")

    for e in [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]:
        bm.edges.remove(e)
    for v in [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]:
        bm.verts.remove(v)

    post_faces = len(bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.update()
    return {
        "pre_faces": pre_faces,
        "deleted_faces": len(to_del),
        "pruned_slivers": pruned_orphan_faces,
        "pruned_orphan_islands": pruned_orphan_islands,
        "orphan_cap_faces": orphan_cap,
        "post_faces": post_faces,
    }


def _parse_bisect_expr(expr: str) -> tuple[str, float, float, float, float, str]:
    """Parse 'view:x0,y0->x1,y1:drop=left|right|above|below'."""
    parts = [p.strip() for p in expr.split(":")]
    if len(parts) != 3 or "->" not in parts[1] or not parts[2].startswith("drop="):
        raise ValueError(
            f"Invalid --bisect '{expr}': expected 'view:x0,y0->x1,y1:drop=left|right|above|below'"
        )
    vname = parts[0]
    p0_s, p1_s = [s.strip() for s in parts[1].split("->", 1)]
    x0, y0 = [float(v.strip()) for v in p0_s.split(",")]
    x1, y1 = [float(v.strip()) for v in p1_s.split(",")]
    drop_side = parts[2].split("=", 1)[1].strip().lower()
    if drop_side not in {"left", "right", "above", "below"}:
        raise ValueError(f"Invalid drop side '{drop_side}' in '{expr}': use left|right|above|below")
    return vname, x0, y0, x1, y1, drop_side


def _screen_pt_to_world_ray(cam, scene, u: float, v_top: float) -> tuple[Vector, Vector]:
    frame = [cam.matrix_world @ v for v in cam.data.view_frame(scene=scene)]
    tr, br, bl, tl = frame[0], frame[1], frame[2], frame[3]
    origin = cam.matrix_world.translation.copy()
    pt_frame = tl + (tr - tl) * u + (bl - tl) * v_top
    return origin, (pt_frame - origin).normalized()


def _apply_surgical_bisect(
    obj,
    occluders: list,
    view_specs: list[dict],
    bisect_exprs: list[str],
    exclude_faces: set[int],
    depth_m: float = 0.14,
    orphan_max_pct: float = 3.0,
) -> dict:
    """Slice mesh cleanly along screen-projected surgical planes and drop the intersection half-space."""
    if not bisect_exprs:
        raise ValueError("--mode bisect requires at least one --bisect expression.")

    spec_by_name = {sp["name"]: sp for sp in view_specs}
    cam, created = _setup_studio_scene(obj, 256)
    scene = bpy.context.scene

    # Build world-space cutting planes (plane_co, plane_no) where plane_no points toward the DROP side
    world_planes: list[tuple[Vector, Vector]] = []
    primary_view = None
    mid_uvs = []
    for expr in bisect_exprs:
        vname, x0, y0, x1, y1, drop_side = _parse_bisect_expr(expr)
        if vname not in spec_by_name:
            for o in created:
                bpy.data.objects.remove(o, do_unlink=True)
            raise ValueError(f"Unknown view '{vname}' in --bisect '{expr}'.")
        if primary_view is None:
            primary_view = vname
        _position_camera(cam, spec_by_name[vname])
        origin, ray0 = _screen_pt_to_world_ray(cam, scene, x0, y0)
        _, ray1 = _screen_pt_to_world_ray(cam, scene, x1, y1)
        plane_no = ray0.cross(ray1)
        if plane_no.length < 1e-8:
            for o in created:
                bpy.data.objects.remove(o, do_unlink=True)
            raise ValueError(f"Degenerate --bisect line '{expr}': endpoints coincide.")
        plane_no.normalize()

        # Orient plane_no strictly toward drop_side using an offset test ray from line midpoint
        mu, mv = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        mid_uvs.append((vname, mu, mv))
        du, dv = {"left": (-0.06, 0.0), "right": (0.06, 0.0), "above": (0.0, -0.06), "below": (0.0, 0.06)}[drop_side]
        _, ray_drop = _screen_pt_to_world_ray(cam, scene, mu + du, mv + dv)
        if ray_drop.dot(plane_no) < 0.0:
            plane_no = -plane_no
        world_planes.append((origin.copy(), plane_no))

    # Optional depth-guard plane along primary camera axis so back skirt layers are never cut
    if depth_m > 0.0 and primary_view is not None:
        _position_camera(cam, spec_by_name[primary_view])
        cam_origin = cam.matrix_world.translation.copy()
        cam_fwd = (Vector(spec_by_name[primary_view]["target"]) - cam_origin).normalized()
        target_bvh = _build_eval_bvh_with_poly_map(obj)
        occ_bvhs = [_build_eval_bvh_with_poly_map(o) for o in occluders]
        # Sample front-most depth inside the drop intersection using target vertices
        mw_tmp = obj.matrix_world
        front_depths = []
        for p in obj.data.polygons:
            if p.index in exclude_faces:
                continue
            wc = mw_tmp @ p.center
            if all((wc - pco).dot(pno) >= -0.005 for pco, pno in world_planes):
                front_depths.append((wc - cam_origin).dot(cam_fwd))
        if front_depths:
            front_d = min(front_depths)
            # Also check occluder leg front so depth band starts cleanly at front of leg/pouch
            for obvh in occ_bvhs:
                o_loc, _, _, o_dist = obvh.ray_cast(cam_origin, cam_fwd, 50.0)
                if o_loc is not None:
                    front_d = min(front_d, (o_loc - cam_origin).dot(cam_fwd))
            back_guard_pt = cam_origin + cam_fwd * (front_d + float(depth_m))
            # Drop side is IN FRONT of back_guard_pt -> plane_no points toward -cam_fwd
            world_planes.append((back_guard_pt, -cam_fwd))

    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)

    # Convert planes to object local space
    mw = obj.matrix_world
    mw_inv = mw.inverted()
    mw_inv_t = mw_inv.transposed().to_3x3()
    local_planes: list[tuple[Vector, Vector]] = []
    for pco, pno in world_planes:
        lco = mw_inv @ pco
        lno = (mw_inv_t @ pno).normalized()
        local_planes.append((lco, lno))

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    pre_faces = len(bm.faces)

    # Tag excluded faces via a custom int layer so exclusions survive bisect subdivision cleanly
    excl_layer = bm.faces.layers.int.new("vp_excl")
    for f in bm.faces:
        f[excl_layer] = 1 if f.index in exclude_faces else 0

    def _in_drop_margin(pt: Vector, margin: float = 0.015) -> bool:
        return all((pt - lco).dot(lno) >= -margin for lco, lno in local_planes)

    # Slice along each knife plane ONLY on faces touching the drop region (preserves UV & vertex weights!)
    sliced_edges_total = 0
    for lco, lno in local_planes:
        cand_faces = [
            f for f in bm.faces
            if f.is_valid and f[excl_layer] == 0 and any(_in_drop_margin(v.co, margin=0.02) for v in f.verts)
        ]
        if not cand_faces:
            continue
        cand_edges = list({e for f in cand_faces for e in f.edges})
        cand_verts = list({v for f in cand_faces for v in f.verts})
        res_bi = bmesh.ops.bisect_plane(
            bm,
            geom=cand_faces + cand_edges + cand_verts,
            dist=1e-5,
            plane_co=lco,
            plane_no=lno,
            clear_inner=False,
            clear_outer=False,
        )
        sliced_edges_total += sum(1 for g in res_bi.get("geom_cut", []) if isinstance(g, bmesh.types.BMEdge))

    bm.faces.ensure_lookup_table()
    to_del = [
        f for f in bm.faces
        if f.is_valid
        and f[excl_layer] == 0
        and all((f.calc_center_median() - lco).dot(lno) > 1e-5 for lco, lno in local_planes)
    ]

    def _pos_key(v):
        return (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))

    cut_pos_keys = {_pos_key(v) for f in to_del for v in f.verts}
    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context="FACES")

    # 3D-welded cut-adjacent orphan ring cleanup
    pruned_orphan_faces = 0
    pruned_orphan_islands = 0
    orphan_cap = max(4, int(pre_faces * max(0.0, float(orphan_max_pct)) / 100.0))
    if orphan_cap > 0 and cut_pos_keys:
        bm.faces.ensure_lookup_table()
        pos_to_faces: dict[tuple[float, float, float], list[int]] = {}
        for f in bm.faces:
            for v in f.verts:
                pos_to_faces.setdefault(_pos_key(v), []).append(f.index)
        visited = set()
        orphan_faces = []
        for fi in range(len(bm.faces)):
            if fi in visited:
                continue
            comp = []
            touches_cut = False
            q = deque([fi])
            visited.add(fi)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for v in bm.faces[cur].verts:
                    pk = _pos_key(v)
                    if pk in cut_pos_keys:
                        touches_cut = True
                    for nfi in pos_to_faces.get(pk, ()):
                        if nfi not in visited:
                            visited.add(nfi)
                            q.append(nfi)
            if touches_cut and len(comp) <= orphan_cap:
                pruned_orphan_islands += 1
                orphan_faces.extend(bm.faces[i] for i in comp)
        if orphan_faces:
            pruned_orphan_faces = len(orphan_faces)
            bmesh.ops.delete(bm, geom=orphan_faces, context="FACES")

    for e in [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]:
        bm.edges.remove(e)
    for v in [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]:
        bm.verts.remove(v)

    bm.faces.layers.int.remove(excl_layer)
    post_faces = len(bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.update()
    return {
        "pre_faces": pre_faces,
        "bisect_planes": len(local_planes),
        "sliced_edges": sliced_edges_total,
        "deleted_faces": len(to_del),
        "pruned_slivers": pruned_orphan_faces,
        "pruned_orphan_islands": pruned_orphan_islands,
        "post_faces": post_faces,
    }


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _load_scene(args.input)
    target_obj = _get_target_mesh(args.target_mesh)
    occluders = [o for o in bpy.context.scene.objects if o.type == "MESH" and o != target_obj and len(o.data.polygons) > 0]
    meta_path = out_dir / "capture_meta.json"

    if args.mode == "capture":
        vnames = [v.strip() for v in args.views.split(",") if v.strip()]
        view_specs = _build_view_specs(target_obj, vnames)
        sheet_p, indiv_grids = _render_views_sheet(target_obj, view_specs, out_dir, "01_grid_capture_sheet.png", args.res, draw_grid=True)

        # Compute exact occlusion-aware Face-ID buffers per view
        cam, created = _setup_studio_scene(target_obj, args.res)
        id_files = {}
        for sp in view_specs:
            _position_camera(cam, sp)
            id_map = _raycast_face_id_buffer(cam, target_obj, occluders, args.id_res)
            npy_p = out_dir / f"face_id_{sp['name']}.npy"
            np.save(str(npy_p), id_map)
            id_files[sp["name"]] = str(npy_p)
        for o in created:
            bpy.data.objects.remove(o, do_unlink=True)

        meta = {
            "input": str(args.input),
            "target_mesh": target_obj.name,
            "total_faces": len(target_obj.data.polygons),
            "id_res": args.id_res,
            "views": view_specs,
            "grid_sheet": sheet_p,
            "grid_images": indiv_grids,
            "face_id_buffers": id_files,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing capture_meta.json in {out_dir}. Run --mode capture first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    view_specs = meta["views"]

    if args.mode == "preview":
        if not args.pick:
            raise ValueError("Preview mode requires at least one --pick 'view:D5:F7' or 'view:x0,y0,x1,y1'.")
        seed_faces = _faces_from_pick_boxes(out_dir, args.pick)
        if args.xray_garment:
            xray_seeds = _xray_faces_from_pick_boxes(
                target_obj,
                occluders,
                view_specs,
                args.pick,
                id_res=int(meta.get("id_res", args.id_res)),
                xray_depth_m=args.xray_depth_m,
            )
            seed_faces.update(xray_seeds)
        excl_faces = _faces_from_pick_boxes(out_dir, args.exclude) if args.exclude else set()
        selected = _grow_faces_smooth(target_obj, seed_faces, excl_faces, args.grow_angle_deg, args.grow_hops)
        mask_path = out_dir / "selected_faces.json"
        mask_payload = {
            "target_mesh": target_obj.name,
            "picks": args.pick,
            "excludes": args.exclude,
            "xray_garment": bool(args.xray_garment),
            "xray_depth_m": float(args.xray_depth_m),
            "seed_count": len(seed_faces),
            "selected_count": len(selected),
            "selected_pct": round(100.0 * len(selected) / max(1, len(target_obj.data.polygons)), 2),
            "selected_faces": sorted(selected),
        }
        mask_path.write_text(json.dumps(mask_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        preview_p, _ = _render_views_sheet(target_obj, view_specs, out_dir, "02_preview_red_mask.png", args.res, highlight_faces=selected, draw_grid=False)
        res = {"mode": "preview", **mask_payload, "preview_sheet": preview_p, "mask_json": str(mask_path)}
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.mode == "bisect":
        excl_faces = _faces_from_pick_boxes(out_dir, args.exclude) if args.exclude else set()
        before_p, _ = _render_views_sheet(target_obj, view_specs, out_dir, "00_before_bisect.png", args.res)
        bi_stats = _apply_surgical_bisect(
            target_obj,
            occluders,
            view_specs,
            args.bisect,
            exclude_faces=excl_faces,
            depth_m=args.bisect_depth_m,
            orphan_max_pct=args.orphan_max_pct,
        )
        stem = Path(args.input).stem
        out_blend = out_dir / f"{stem}_surgical_bisected.blend"
        out_glb = out_dir / f"{stem}_surgical_bisected.glb"
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
        bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB", export_animations=True, export_skins=True)
        after_p, _ = _render_views_sheet(target_obj, view_specs, out_dir, "03_after_bisect.png", args.res)
        wire_xray_p = _render_wireframe_xray_sheet(target_obj, occluders, view_specs, out_dir, "04_after_bisect_wireframe.png", args.res)
        res = {
            "mode": "bisect",
            "target_mesh": target_obj.name,
            "bisect_exprs": args.bisect,
            "excludes": args.exclude,
            "bisect_depth_m": float(args.bisect_depth_m),
            "bisect_stats": bi_stats,
            "before_sheet": before_p,
            "after_sheet": after_p,
            "wireframe_xray_sheet": wire_xray_p,
            "output_blend": str(out_blend),
            "output_glb": str(out_glb),
        }
        (out_dir / "surgical_bisect_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # mode == "apply"
    mask_p = Path(args.mask_json) if args.mask_json else (out_dir / "selected_faces.json")
    if not mask_p.is_file():
        raise FileNotFoundError(f"Missing mask JSON for apply mode: {mask_p}")
    mask_data = json.loads(mask_p.read_text(encoding="utf-8"))
    selected_list = list(mask_data["selected_faces"])

    before_p, _ = _render_views_sheet(target_obj, view_specs, out_dir, "00_before_cull.png", args.res)
    del_stats = _apply_uv_safe_delete(
        target_obj,
        selected_list,
        prune_slivers=args.prune_cut_slivers,
        orphan_max_pct=args.orphan_max_pct,
    )
    stem = Path(args.input).stem
    out_blend = out_dir / f"{stem}_visual_culled.blend"
    out_glb = out_dir / f"{stem}_visual_culled.glb"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB", export_animations=True, export_skins=True)
    after_p, _ = _render_views_sheet(target_obj, view_specs, out_dir, "03_after_cull.png", args.res)
    wire_xray_p = _render_wireframe_xray_sheet(target_obj, occluders, view_specs, out_dir, "04_after_wireframe_xray.png", args.res)
    res = {
        "mode": "apply",
        "target_mesh": target_obj.name,
        "deletion": del_stats,
        "before_sheet": before_p,
        "after_sheet": after_p,
        "wireframe_xray_sheet": wire_xray_p,
        "output_blend": str(out_blend),
        "output_glb": str(out_glb),
    }
    (out_dir / "visual_cull_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
