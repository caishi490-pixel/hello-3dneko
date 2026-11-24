"""Visual-Guided & Occlusion-Aware Smart Face Culler for AI-generated 3D Garments (Blender 5.1+).

Detects and removes false internal caps, occluded leg pouches, and unwanted inner shells
while preserving outer garment silhouettes. Supports a mandatory red-mask preview step
before destructive face deletion.
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Smart Face Culler (preview red mask -> apply delete).")
    p.add_argument("--input", required=True, help="Input .blend or .glb garment asset.")
    p.add_argument("--out-dir", required=True, help="Output directory for previews, mask JSON, and culled mesh.")
    p.add_argument("--mode", choices=["preview", "apply"], default="preview")
    p.add_argument("--target-mode", choices=["skirt_cavity", "fitted_leg_pouch"], default="skirt_cavity")
    p.add_argument("--mask-json", default="", help="Optional pre-computed selected_faces.json for apply mode.")
    p.add_argument("--z-min-rel", type=float, default=0.14, help="Lower normalized Z bound of skirt pouch zone.")
    p.add_argument("--z-max-rel", type=float, default=0.52, help="Upper normalized Z bound of skirt pouch zone.")
    p.add_argument("--inner-x-ratio", type=float, default=0.25, help="Half-width X ratio relative to mesh width for inner leg cavity.")
    p.add_argument("--inner-y-ratio", type=float, default=0.34, help="Half-depth Y ratio relative to mesh depth for inner leg cavity.")
    p.add_argument("--outer-ray-hits", type=int, default=2, help="Min horizontal outward rays hitting outer skirt shell to confirm inner cavity.")
    p.add_argument("--grow-angle-deg", type=float, default=42.0, help="Max dihedral angle for flood-fill growth inside cavity.")
    p.add_argument("--leg-z-min", type=float, default=0.52, help="World Z min for fitted leg pouch detection (meters).")
    p.add_argument("--leg-z-max", type=float, default=0.80, help="World Z max for fitted leg pouch detection (meters).")
    p.add_argument("--leg-max-dist", type=float, default=0.042, help="Max distance from body thigh surface to classify as clinging leg pouch (meters).")
    p.add_argument("--min-shell-faces", type=int, default=32, help="Prune tiny disconnected debris shells left after culling.")
    return p.parse_args(argv)


def _load_scene(filepath: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fp = str(filepath)
    if fp.lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=fp)
    else:
        bpy.ops.import_scene.gltf(filepath=fp)


def _get_target_mesh():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons) > 0]
    if not meshes:
        raise RuntimeError("No non-empty mesh found in scene.")
    # Prefer Fitted_Garment or largest mesh
    for o in meshes:
        if "garment" in o.name.lower():
            return o
    return max(meshes, key=lambda o: len(o.data.polygons))


def _detect_skirt_pouch_faces(
    obj,
    z_min_rel: float,
    z_max_rel: float,
    inner_x_ratio: float,
    inner_y_ratio: float,
    outer_ray_hits: int,
    grow_angle_deg: float,
) -> tuple[list[int], dict]:
    """Identify false internal skirt-bottom leg pouches/caps using cavity ray-enclosure + bottom-facing normals."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    mw = obj.matrix_world
    bmesh.ops.transform(bm, matrix=mw, verts=bm.verts)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    pts = [v.co for v in bm.verts]
    min_x = min(p.x for p in pts)
    max_x = max(p.x for p in pts)
    min_y = min(p.y for p in pts)
    max_y = max(p.y for p in pts)
    min_z = min(p.z for p in pts)
    max_z = max(p.z for p in pts)

    span_x = max(1e-4, max_x - min_x)
    span_y = max(1e-4, max_y - min_y)
    span_z = max(1e-4, max_z - min_z)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    z_lo = min_z + z_min_rel * span_z
    z_hi = min_z + z_max_rel * span_z
    rx_lim = inner_x_ratio * span_x
    ry_lim = inner_y_ratio * span_y
    bvh = BVHTree.FromBMesh(bm)

    # 8 horizontal compass rays + upward ray to test whether a face sits INSIDE the outer skirt bell
    horiz_dirs = [
        Vector((math.cos(k * math.pi / 4.0), math.sin(k * math.pi / 4.0), 0.0)).normalized()
        for k in range(8)
    ]
    max_r = max(span_x, span_y) * 0.85
    eps = max(span_x, span_y, span_z) * 0.004

    seed_faces = set()
    pouch_zone_faces = set()
    for f in bm.faces:
        fc = f.calc_center_median()
        if not (z_lo <= fc.z <= z_hi):
            continue
        dx = abs(fc.x - cx)
        dy = abs(fc.y - cy)
        if dx > rx_lim or dy > ry_lim:
            continue
        # Elliptical inner leg-tunnel check
        if (dx / max(1e-4, rx_lim)) ** 2 + (dy / max(1e-4, ry_lim)) ** 2 > 1.05:
            continue
        pouch_zone_faces.add(f.index)

        # Count how many horizontal rays from this face hit an outer skirt shell (meaning it's enclosed inside the skirt)
        enclosed_hits = 0
        for d in horiz_dirs:
            loc, _, hit_idx, dist = bvh.ray_cast(fc + d * eps, d, max_r)
            if loc is not None and hit_idx != f.index and dist > eps * 1.5:
                enclosed_hits += 1
        # Also check if there is outer skirt shell above (or back/sides)
        up_loc, _, up_idx, _ = bvh.ray_cast(fc + Vector((0, 0, eps)), Vector((0, 0, 1)), span_z * 0.55)
        has_roof = up_loc is not None and up_idx != f.index

        # Seed condition: enclosed inside outer skirt bell (front open arch still has >=3 side/back/diagonal hits!)
        # OR downward-pointing false cap inside the cavity
        is_bottom_cap = f.normal.z < -0.25 and enclosed_hits >= max(1, outer_ray_hits - 1)
        is_inner_pouch_wall = enclosed_hits >= outer_ray_hits and (has_roof or fc.z < min_z + (z_max_rel - 0.05) * span_z)
        if is_bottom_cap or is_inner_pouch_wall:
            seed_faces.add(f.index)

    # Flood-fill smoothly inside the cavity zone across edges with gentle dihedral angle
    # so we cleanly capture the whole inner pouch while stopping at outer skirt hem/creases
    max_cos = math.cos(math.radians(grow_angle_deg))
    selected = set(seed_faces)
    q = deque(seed_faces)
    # Slightly expanded geometric fence for flood-fill (never crosses outside inner leg tunnel)
    rx_fence = rx_lim * 1.12
    ry_fence = ry_lim * 1.12
    z_lo_fence = min_z + max(0.05, z_min_rel - 0.03) * span_z
    z_hi_fence = min_z + min(0.62, z_max_rel + 0.04) * span_z

    while q:
        fi = q.popleft()
        f = bm.faces[fi]
        fn = f.normal
        for e in f.edges:
            for nf in e.link_faces:
                nfi = nf.index
                if nfi in selected:
                    continue
                nfc = nf.calc_center_median()
                if not (z_lo_fence <= nfc.z <= z_hi_fence):
                    continue
                if abs(nfc.x - cx) > rx_fence or abs(nfc.y - cy) > ry_fence:
                    continue
                if fn.dot(nf.normal) >= max_cos:
                    # Verify neighbor is still interior (at least 2 horizontal rays hit surrounding skirt)
                    nhits = 0
                    for d in horiz_dirs[::2]:
                        loc, _, hidx, dist = bvh.ray_cast(nfc + d * eps, d, max_r)
                        if loc is not None and hidx != nfi and dist > eps * 1.5:
                            nhits += 1
                    if nhits >= 2 or nf.normal.z < -0.35:
                        selected.add(nfi)
                        q.append(nfi)

    stats = {
        "mesh": obj.name,
        "total_faces": len(bm.faces),
        "zone_faces": len(pouch_zone_faces),
        "seed_faces": len(seed_faces),
        "selected_faces": len(selected),
        "selected_pct": round(100.0 * len(selected) / max(1, len(bm.faces)), 2),
        "bounds_world": {
            "x": [round(min_x, 4), round(max_x, 4)],
            "y": [round(min_y, 4), round(max_y, 4)],
            "z": [round(min_z, 4), round(max_z, 4)],
        },
        "cavity_box": {
            "z_range": [round(z_lo, 4), round(z_hi, 4)],
            "rx_lim": round(rx_lim, 4),
            "ry_lim": round(ry_lim, 4),
        },
    }
    bm.free()
    return sorted(selected), stats


def _render_four_views(obj, out_dir: Path, out_name: str, highlight_faces: set[int] | None = None, res: int = 512):
    """Render 2x2 inspection sheet: Front, 3/4 Front, Bottom-Up Skirt Cavity, Back."""
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

    # Backup materials & polygon material indices if highlighting
    orig_mats = list(obj.data.materials)
    orig_mids = [p.material_index for p in obj.data.polygons]

    if highlight_faces is not None:
        red_mat = bpy.data.materials.new("Cull_Red_Highlight")
        if red_mat.node_tree is None:
            red_mat.use_nodes = True
        bsdf = next(n for n in red_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (0.95, 0.08, 0.08, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Emission Color"].default_value = (0.95, 0.10, 0.10, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.45
        red_idx = len(obj.data.materials)
        obj.data.materials.append(red_mat)
        for p in obj.data.polygons:
            if p.index in highlight_faces:
                p.material_index = red_idx

    mw = obj.matrix_world
    pts = [mw @ v.co for v in obj.data.vertices]
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    span = max((max_p - min_p).x, (max_p - min_p).y, (max_p - min_p).z, 0.1)

    w = bpy.data.worlds.new("CullWorld")
    scene.world = w
    if w.node_tree is None:
        w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.86, 0.87, 0.90, 1.0)
    bg.inputs[1].default_value = 0.65

    lights = []
    for lname, off, eng in [
        ("CL_Key", Vector((-1.2, -1.6, 1.0)), 110),
        ("CL_Fill", Vector((1.4, -1.2, 0.4)), 80),
        ("CL_Bottom", Vector((0.0, -0.4, -1.6)), 120),
        ("CL_Back", Vector((0.0, 1.8, 0.8)), 95),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng * max(1.0, span * span)
        ld.size = span * 1.4
        lo = bpy.data.objects.new(lname, ld)
        scene.collection.objects.link(lo)
        lo.location = center + off * span
        lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()
        lights.append(lo)

    cam_d = bpy.data.cameras.new("CullCam")
    cam = bpy.data.objects.new("CullCam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam

    skirt_tgt = Vector((center.x, center.y, min_p.z + 0.34 * (max_p.z - min_p.z)))
    views = [
        ("front", center + Vector((0.0, -1.85, 0.05)) * span, center, 48),
        ("front_skirt", skirt_tgt + Vector((0.0, -1.15, -0.05)) * span, skirt_tgt, 48),
        ("bottom_up", skirt_tgt + Vector((0.0, -0.68, -0.92)) * span, skirt_tgt + Vector((0.0, 0.0, 0.08 * span)), 42),
        ("back", center + Vector((0.0, 1.85, 0.05)) * span, center, 48),
    ]

    tiles = []
    for vname, loc, tgt, lens in views:
        cam.data.lens = lens
        cam.location = loc
        cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
        tmp_p = out_dir / f"_tmp_{vname}.png"
        scene.render.filepath = str(tmp_p)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(tmp_p))
        arr = np.array(img.pixels[:], dtype=np.float32).reshape((res, res, 4))
        tiles.append(arr)
        bpy.data.images.remove(img)
        tmp_p.unlink(missing_ok=True)

    # Stitch 2x2 contact sheet (row 0 = bottom in Blender)
    sheet = np.zeros((res * 2, res * 2, 4), dtype=np.float32)
    sheet[res : res * 2, 0:res] = tiles[0]          # top-left: front
    sheet[res : res * 2, res : res * 2] = tiles[1]  # top-right: front_skirt closeup
    sheet[0:res, 0:res] = tiles[2]                  # bottom-left: bottom_up cavity view
    sheet[0:res, res : res * 2] = tiles[3]          # bottom-right: back
    out_path = out_dir / out_name
    img_s = bpy.data.images.new(f"Sheet_{out_name}", width=res * 2, height=res * 2, alpha=True)
    img_s.pixels.foreach_set(sheet.ravel())
    img_s.filepath_raw = str(out_path)
    img_s.file_format = "PNG"
    img_s.save()

    # Restore original materials & cleanup render rig so saved .blend stays spotless
    if highlight_faces is not None:
        for p, mid in zip(obj.data.polygons, orig_mids):
            p.material_index = mid
        obj.data.materials.clear()
        for m in orig_mats:
            obj.data.materials.append(m)
    for lo in lights + [cam]:
        bpy.data.objects.remove(lo, do_unlink=True)
    return str(out_path)


def _apply_face_deletion(obj, face_indices: list[int], min_shell_faces: int = 0) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    pre_faces = len(bm.faces)
    del_set = {i for i in face_indices if 0 <= i < len(bm.faces)}
    to_del = [bm.faces[i] for i in sorted(del_set)]

    # Remember spatial neighborhood of deleted faces so we only prune tiny dangling triangles right at the cut
    cut_verts = {v for f in to_del for v in f.verts}
    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context="FACES")

    # Prune ONLY tiny 1..3-face isolated slivers touching the cut (never touch unwelded UV islands elsewhere!)
    pruned_faces = 0
    bm.faces.ensure_lookup_table()
    if min_shell_faces > 0:
        cut_v_set = {v for v in cut_verts if v.is_valid}
        visited = set()
        debris_f = []
        for v in cut_v_set:
            for f in v.link_faces:
                if not f.is_valid or f.index in visited:
                    continue
                comp = []
                q = deque([f])
                visited.add(f.index)
                while q:
                    cur = q.popleft()
                    comp.append(cur)
                    for e in cur.edges:
                        for nf in e.link_faces:
                            if nf.is_valid and nf.index not in visited:
                                visited.add(nf.index)
                                q.append(nf)
                if len(comp) <= min(6, min_shell_faces):
                    debris_f.extend(comp)
        if debris_f:
            pruned_faces = len(debris_f)
            bmesh.ops.delete(bm, geom=debris_f, context="FACES")

    loose_e = [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]
    if loose_e:
        bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
    loose_v = [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context="VERTS")

    post_faces = len(bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.update()
    return {
        "pre_faces": pre_faces,
        "deleted_selected_faces": len(to_del),
        "pruned_debris_faces": pruned_faces,
        "post_faces": post_faces,
    }


def _detect_fitted_leg_pouch_faces(
    garment_obj,
    leg_z_min: float = 0.52,
    leg_z_max: float = 0.80,
    leg_max_dist: float = 0.042,
) -> tuple[list[int], dict]:
    """Detect cloth faces inflated/clinging tightly around the body's thighs/knees inside an open skirt arch."""
    body_objs = [
        o for o in bpy.context.scene.objects
        if o.type == "MESH" and o != garment_obj and len(o.data.polygons) > 0
        and ("body" in o.name.lower() or "skin" in o.name.lower())
    ]
    if not body_objs:
        raise RuntimeError("No body/skin mesh found in scene for fitted_leg_pouch detection.")

    # Build BVH of body thigh/knee skin in [leg_z_min - 0.05, leg_z_max + 0.08]
    bm_body = bmesh.new()
    deps = bpy.context.evaluated_depsgraph_get()
    for bo in body_objs:
        ev = bo.evaluated_get(deps)
        me = ev.to_mesh()
        bm_tmp = bmesh.new()
        bm_tmp.from_mesh(me)
        bmesh.ops.transform(bm_tmp, matrix=bo.matrix_world, verts=bm_tmp.verts)
        vmap = {}
        for v in bm_tmp.verts:
            vmap[v.index] = bm_body.verts.new(v.co)
        for f in bm_tmp.faces:
            fc = f.calc_center_median()
            if leg_z_min - 0.06 <= fc.z <= leg_z_max + 0.08:
                try:
                    bm_body.faces.new([vmap[v.index] for v in f.verts])
                except ValueError:
                    pass
        bm_tmp.free()
        ev.to_mesh_clear()
    bm_body.verts.ensure_lookup_table()
    bm_body.faces.ensure_lookup_table()
    body_bvh = BVHTree.FromBMesh(bm_body)

    # Evaluate garment in rest pose
    bm_g = bmesh.new()
    bm_g.from_mesh(garment_obj.data)
    bmesh.ops.transform(bm_g, matrix=garment_obj.matrix_world, verts=bm_g.verts)
    bm_g.normal_update()
    bm_g.faces.ensure_lookup_table()
    g_bvh = BVHTree.FromBMesh(bm_g)

    selected = set()
    for f in bm_g.faces:
        fc = f.calc_center_median()
        if not (leg_z_min <= fc.z <= leg_z_max):
            continue
        # Only inner leg tunnel X/Y (never touch outer flowing lateral or trailing back ruffles!)
        if abs(fc.x) > 0.175 or fc.y > 0.135 or fc.y < -0.145:
            continue
        loc, bn, _, dist = body_bvh.find_nearest(fc)
        if loc is None or dist > leg_max_dist:
            continue
        # Protect front-center upper apron / starlight panel (sits further out in -Y or above leg_z_max)
        # Check if an outer skirt layer exists behind (+Y) or to the side (+X/-X) so we never delete outer back skirt
        back_hit, _, bh_idx, bh_d = g_bvh.ray_cast(fc + Vector((0, 0.004, 0)), Vector((0, 1, 0)), 0.35)
        has_outer_back = back_hit is not None and bh_idx != f.index and bh_d > 0.020
        # Also if face is strictly on the front/inner/bottom of the thigh sleeve (fc.y < 0.075), it's definitely the pouch!
        if fc.y < 0.075 or has_outer_back or f.normal.z < -0.25:
            selected.add(f.index)

    # 1-ring clean expansion ONLY for immediate bridging faces strictly inside [leg_z_min-0.02, leg_z_max+0.02] & dist < leg_max_dist*1.3
    extra = set()
    for fi in list(selected):
        for e in bm_g.faces[fi].edges:
            for nf in e.link_faces:
                if nf.index in selected:
                    continue
                nfc = nf.calc_center_median()
                if not (leg_z_min - 0.02 <= nfc.z <= leg_z_max + 0.015):
                    continue
                if abs(nfc.x) > 0.175 or nfc.y > 0.095 or nfc.y < -0.145:
                    continue
                loc, _, _, dist = body_bvh.find_nearest(nfc)
                if loc is not None and dist <= leg_max_dist * 1.25:
                    extra.add(nf.index)
    selected |= extra

    stats = {
        "mesh": garment_obj.name,
        "total_faces": len(bm_g.faces),
        "selected_faces": len(selected),
        "selected_pct": round(100.0 * len(selected) / max(1, len(bm_g.faces)), 2),
        "leg_z_range": [leg_z_min, leg_z_max],
        "leg_max_dist": leg_max_dist,
    }
    bm_g.free()
    bm_body.free()
    return sorted(selected), stats


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _load_scene(args.input)
    obj = _get_target_mesh()

    if args.mask_json and Path(args.mask_json).is_file():
        mask_data = json.loads(Path(args.mask_json).read_text(encoding="utf-8"))
        selected_faces = list(mask_data["selected_faces"])
        stats = mask_data.get("stats", {})
    elif args.target_mode == "fitted_leg_pouch":
        selected_faces, stats = _detect_fitted_leg_pouch_faces(
            obj,
            leg_z_min=args.leg_z_min,
            leg_z_max=args.leg_z_max,
            leg_max_dist=args.leg_max_dist,
        )
        mask_path = out_dir / "selected_faces.json"
        mask_path.write_text(
            json.dumps({"stats": stats, "selected_faces": selected_faces}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        selected_faces, stats = _detect_skirt_pouch_faces(
            obj,
            z_min_rel=args.z_min_rel,
            z_max_rel=args.z_max_rel,
            inner_x_ratio=args.inner_x_ratio,
            inner_y_ratio=args.inner_y_ratio,
            outer_ray_hits=args.outer_ray_hits,
            grow_angle_deg=args.grow_angle_deg,
        )
        mask_path = out_dir / "selected_faces.json"
        mask_path.write_text(
            json.dumps({"stats": stats, "selected_faces": selected_faces}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.mode == "preview":
        preview_img = _render_four_views(obj, out_dir, "01_preview_red_mask.png", highlight_faces=set(selected_faces))
        res = {"mode": "preview", "stats": stats, "preview_image": preview_img, "mask_json": str(out_dir / "selected_faces.json")}
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    before_img = _render_four_views(obj, out_dir, "00_before_cull.png", highlight_faces=None)
    del_stats = _apply_face_deletion(obj, selected_faces, min_shell_faces=args.min_shell_faces)
    stem = Path(args.input).stem
    out_blend = out_dir / f"{stem}_culled.blend"
    out_glb = out_dir / f"{stem}_culled.glb"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB", export_animations=True, export_skins=True)
    after_img = _render_four_views(obj, out_dir, "02_after_cull.png", highlight_faces=None)
    res = {
        "mode": "apply",
        "stats": stats,
        "deletion": del_stats,
        "before_image": before_img,
        "after_image": after_img,
        "output_blend": str(out_blend),
        "output_glb": str(out_glb),
    }
    (out_dir / "cull_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
