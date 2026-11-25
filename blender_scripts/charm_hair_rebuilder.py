"""Headless Blender script: rebuild control-point hair strand JSON (e.g. CHARM output)
into 100% pure-quad, arched, razor-tapered stylized anime hair with vertical strip UVs,
recessed inner scalp cap, Smooth Cage normal transfer, and matte Toon gradient atlas.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Rebuild hair strand control-point JSON into pure-quad anime hair.")
    p.add_argument("--json", required=True, help="Input strand control-point JSON ([{shape, seq: [[x,y,z,w,t], ...]}])")
    p.add_argument("--out-dir", required=True, help="Output directory for .blend, .glb, and 4-view QC render")
    p.add_argument("--meta", default="", help="Optional normalization meta JSON containing {center, scale}")
    p.add_argument("--head", default="", help="Optional reference head .glb/.blend for exact skull fitting")
    p.add_argument("--atlas", default="", help="Optional vertical strip hair texture PNG (auto-generated if omitted)")
    p.add_argument("--root-hex", default="#4A5BD4", help="Root gradient hex color when auto-generating atlas")
    p.add_argument("--mid-hex", default="#8C6CE8", help="Mid gradient hex color when auto-generating atlas")
    p.add_argument("--tip-hex", default="#F2A4D0", help="Tip gradient hex color when auto-generating atlas")
    p.add_argument("--jump-threshold", type=float, default=0.085, help="Split strand when step jump exceeds threshold")
    p.add_argument("--samples", type=int, default=22, help="Resampled longitudinal rings per strand")
    p.add_argument("--res", type=int, default=512, help="Per-tile resolution for 4-view QC render")
    p.add_argument("--no-render", action="store_true", help="Skip Cycles 4-view QC render")
    return p.parse_args(argv)


def _hex_to_rgb(hx: str) -> tuple[float, float, float]:
    h = hx.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _generate_gradient_atlas(out_path: Path, root_rgb, mid_rgb, tip_rgb, w: int = 512, h: int = 1024) -> Path:
    arr = np.zeros((h, w, 4), dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # 0=bottom(tip), 1=top(root)
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    # Smooth 3-stop vertical gradient (tip -> mid -> root)
    r_col = np.array(root_rgb, dtype=np.float32)
    m_col = np.array(mid_rgb, dtype=np.float32)
    t_col = np.array(tip_rgb, dtype=np.float32)

    low_t = np.clip(ys / 0.55, 0.0, 1.0)[..., None]
    hi_t = np.clip((ys - 0.55) / 0.45, 0.0, 1.0)[..., None]
    base = np.where(ys[..., None] < 0.55, t_col * (1.0 - low_t) + m_col * low_t, m_col * (1.0 - hi_t) + r_col * hi_t)

    # Subtle vertical strand grooves + stylized anime angel-ring highlight band near crown
    stripes = 0.94 + 0.06 * np.sin(xs * math.pi * 64.0)[..., None]
    halo = np.exp(-((ys - 0.78) ** 2) / (2.0 * 0.032 ** 2))[..., None] * 0.42
    rgb = np.clip(base * stripes + halo, 0.0, 1.0)
    arr[..., :3] = rgb
    arr[..., 3] = 1.0

    img = bpy.data.images.new("AnimeHairGradientAtlas", width=w, height=h, alpha=True)
    img.pixels.foreach_set(arr.ravel())
    img.filepath_raw = str(out_path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return out_path


def _gltf_to_blender(pt: np.ndarray) -> np.ndarray:
    return np.array([pt[0], -pt[2], pt[1]], dtype=np.float64)


def _clean_and_split_strands(raw_data: list, center_gltf: np.ndarray, scale: float, jump_thr: float):
    cleaned = []
    for blk in raw_data:
        arr = np.asarray(blk.get("seq", []), dtype=np.float64)
        if len(arr) < 4:
            continue
        pts_norm = arr[:, :3]
        ws_norm = arr[:, 3]
        ths_norm = arr[:, 4]

        steps = np.linalg.norm(np.diff(pts_norm, axis=0), axis=1)
        split_idx = [0] + (np.where(steps > jump_thr)[0] + 1).tolist() + [len(pts_norm)]

        for s, e in zip(split_idx[:-1], split_idx[1:]):
            seg_pts = pts_norm[s:e]
            seg_w = ws_norm[s:e]
            seg_t = ths_norm[s:e]
            if len(seg_pts) < 5:
                continue

            dz = abs(seg_pts[-1, 1] - seg_pts[0, 1])
            dxy = math.hypot(seg_pts[-1, 0] - seg_pts[0, 0], seg_pts[-1, 2] - seg_pts[0, 2])
            if dz < 0.04 and dxy > 0.14:
                continue

            arc = float(np.sum(np.linalg.norm(np.diff(seg_pts, axis=0), axis=1)))
            if arc < 0.08:
                continue

            if seg_pts[0, 1] < seg_pts[-1, 1]:
                seg_pts = seg_pts[::-1]
                seg_w = seg_w[::-1]
                seg_t = seg_t[::-1]

            pts_world_gltf = seg_pts / scale + center_gltf
            pts_bl = np.array([_gltf_to_blender(p) for p in pts_world_gltf], dtype=np.float64)
            cleaned.append((pts_bl, seg_w / scale, seg_t / scale))
    return cleaned


def _smooth_and_resample(pts: np.ndarray, ws: np.ndarray, ts: np.ndarray, n_samples: int):
    p = pts.copy()
    for _ in range(4):
        p_new = p.copy()
        p_new[1:-1] = 0.25 * p[:-2] + 0.50 * p[1:-1] + 0.25 * p[2:]
        p = p_new

    diffs = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total < 1e-5:
        return None
    u_orig = cum / total
    u_new = np.linspace(0.0, 1.0, n_samples)

    out_p = np.column_stack([np.interp(u_new, u_orig, p[:, k]) for k in range(3)])
    out_w = np.interp(u_new, u_orig, ws)
    out_t = np.interp(u_new, u_orig, ts)
    return out_p, out_w, out_t, total


def _add_anime_arch_strand(
    bm: bmesh.types.BMesh,
    uv_layer,
    pts: np.ndarray,
    ws: np.ndarray,
    head_center: Vector,
    u_lo: float,
    u_hi: float,
    g_zmin: float,
    g_zmax: float,
) -> None:
    n = len(pts)
    mean_y = float(np.mean(pts[:, 1]))
    is_front = mean_y < (head_center.y - 0.03)
    arc_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    # 6-point arched anime ribbon cross-section (binormal, outward_normal)
    profile = [(-0.50, -0.05), (-0.25, 0.42), (0.00, 0.62), (0.25, 0.42), (0.50, -0.05), (0.00, -0.35)]
    u_prof = [0.0, 0.22, 0.50, 0.78, 1.0, 0.50]
    n_prof = len(profile)

    rings = []
    prev_b = None
    g_zspan = max(g_zmax - g_zmin, 1e-4)

    for i in range(n):
        t = i / max(1, n - 1)
        p = Vector(pts[i])
        if i == 0:
            tang = Vector(pts[1] - pts[0])
        elif i == n - 1:
            tang = Vector(pts[-1] - pts[-2])
        else:
            tang = Vector(pts[i + 1] - pts[i - 1])
        if tang.length < 1e-6:
            tang = Vector((0.0, 0.0, -1.0))
        tang.normalize()

        out_vec = Vector((p.x - head_center.x, p.y - head_center.y, (p.z - head_center.z) * 0.35))
        if out_vec.length < 1e-4:
            out_vec = Vector((0.0, 1.0, 0.0))
        out_vec.normalize()

        b_vec = tang.cross(out_vec)
        if b_vec.length < 1e-4:
            b_vec = Vector((1.0, 0.0, 0.0))
        b_vec.normalize()
        if prev_b is not None and b_vec.dot(prev_b) < 0:
            b_vec = -b_vec
        prev_b = b_vec

        n_vec = b_vec.cross(tang)
        if n_vec.length < 1e-4:
            n_vec = out_vec
        n_vec.normalize()
        if n_vec.dot(out_vec) < 0:
            n_vec = -n_vec

        base_w = float(ws[i])
        if is_front:
            w = max(0.024, min(0.048, base_w * 2.4 + 0.016))
        else:
            w = max(0.045, min(0.075, base_w * 3.4 + 0.030))
            if arc_len > 0.35:
                w *= 1.12

        root_env = 0.62 + 0.38 * min(1.0, t / 0.14)
        if t > 0.72:
            tip_t = (t - 0.72) / 0.28
            tip_env = max(0.015, (1.0 - tip_t) ** 1.65)
        else:
            tip_env = 1.0 + 0.12 * math.sin(math.pi * t / 0.72)

        w_cur = w * root_env * tip_env
        th_cur = max(0.002, w_cur * (0.34 if is_front else 0.30))

        ring_verts = [bm.verts.new(p + b_vec * (px * w_cur) + n_vec * (py * th_cur)) for (px, py) in profile]
        rings.append((ring_verts, p.z))

    for i in range(n - 1):
        r0, z0 = rings[i]
        r1, z1 = rings[i + 1]
        v0 = max(0.01, min(0.99, (z0 - g_zmin) / g_zspan))
        v1 = max(0.01, min(0.99, (z1 - g_zmin) / g_zspan))
        for j in range(n_prof):
            jn = (j + 1) % n_prof
            try:
                f = bm.faces.new([r0[j], r0[jn], r1[jn], r1[j]])
            except ValueError:
                continue
            ua = u_lo + (u_hi - u_lo) * u_prof[j]
            ub = u_lo + (u_hi - u_lo) * u_prof[jn]
            for loop, uv in zip(f.loops, [(ua, v0), (ub, v0), (ub, v1), (ua, v1)]):
                loop[uv_layer].uv = uv


def _build_recessed_scalp_cap(head_center: Vector, head_r: float, g_zmin: float, g_zmax: float):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()
    g_zspan = max(g_zmax - g_zmin, 1e-4)

    n_u, n_v = 32, 12
    dome_rings = []
    for iv in range(n_v):
        t = iv / (n_v - 1)
        phi = 0.05 + t * (math.pi * 0.50)
        ring = []
        for iu in range(n_u):
            ang = 2.0 * math.pi * iu / n_u
            rad_x = head_r * 0.80
            rad_y = head_r * (0.86 if math.sin(ang) > 0 else 0.72)
            rad_z = head_r * 0.84
            x = head_center.x + rad_x * math.sin(phi) * math.cos(ang)
            y = head_center.y + rad_y * math.sin(phi) * math.sin(ang) + 0.006
            z = head_center.z + rad_z * math.cos(phi) + 0.012
            ring.append(bm.verts.new(Vector((x, y, z))))
        dome_rings.append(ring)

    for iv in range(n_v - 1):
        for iu in range(n_u):
            iun = (iu + 1) % n_u
            f = bm.faces.new([dome_rings[iv][iu], dome_rings[iv][iun], dome_rings[iv + 1][iun], dome_rings[iv + 1][iu]])
            for loop in f.loops:
                vz = max(0.01, min(0.99, (loop.vert.co.z - g_zmin) / g_zspan))
                loop[uv_layer].uv = (0.02 + 0.96 * (iu / n_u), vz)

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("AnimeHair_InnerScalpCap")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("AnimeHair_InnerScalpCap", me)
    bpy.context.scene.collection.objects.link(obj)
    for p in me.polygons:
        p.use_smooth = True
    return obj


def _apply_smooth_cage_normals(hair_objs: list) -> None:
    pts = [o.matrix_world @ v.co for o in hair_objs for v in o.data.vertices]
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    rad = 0.5 * (max_p - min_p)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20, radius=1.0, location=center)
    cage = bpy.context.active_object
    cage.name = "SmoothCage"
    cage.scale = (max(rad.x * 1.18, 0.05), max(rad.y * 1.18, 0.05), max(rad.z * 1.12, 0.05))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for p in cage.data.polygons:
        p.use_smooth = True

    for obj in hair_objs:
        mod = obj.modifiers.new(name="SmoothCageNormals", type="DATA_TRANSFER")
        mod.object = cage
        mod.use_loop_data = True
        mod.data_types_loops = {"CUSTOM_NORMAL"}
        mod.loop_mapping = "POLYINTERP_NEAREST"
        mod.mix_factor = 0.80
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.data.objects.remove(cage, do_unlink=True)


def _apply_matte_toon_material(hair_objs: list, atlas_path: Path) -> None:
    mat = bpy.data.materials.new("AnimeHair_Rebuilt_Toon")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(atlas_path))
    tex.extension = "EXTEND"
    bsdf.inputs["Roughness"].default_value = 0.95
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.02
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    for obj in hair_objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def _render_4view(target_objs: list, out_png: Path, res: int = 512) -> str:
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

    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.view_settings.view_transform = "Standard"

    pts = [o.matrix_world @ v.co for o in target_objs for v in o.data.vertices]
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    span = max((max_p - min_p).x, (max_p - min_p).y, (max_p - min_p).z, 0.25)

    w = bpy.data.worlds.new("HairRebuildWorld")
    scene.world = w
    w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.90, 0.91, 0.94, 1.0)
    bg.inputs[1].default_value = 0.95

    created = []
    for lname, off, eng in [
        ("Key", Vector((-1.2, -1.6, 1.0)), 110),
        ("Fill", Vector((1.4, -1.2, 0.4)), 85),
        ("Back", Vector((0.0, 1.8, 0.8)), 95),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng * max(0.5, span * span)
        ld.size = span * 1.4
        lo = bpy.data.objects.new(lname, ld)
        scene.collection.objects.link(lo)
        lo.location = center + off * span
        lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()
        created.append(lo)

    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 50.0
    cam = bpy.data.objects.new("Cam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    created.append(cam)

    views = [
        ("front", center + Vector((0.0, -1.85, 0.08)) * span),
        ("34", center + Vector((-1.35, -1.35, 0.12)) * span),
        ("side", center + Vector((1.85, 0.0, 0.08)) * span),
        ("back", center + Vector((0.0, 1.85, 0.08)) * span),
    ]
    tiles = []
    for vn, loc in views:
        cam.location = loc
        cam.rotation_euler = (center - loc).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        tmp = out_png.parent / f"_tile_{vn}.png"
        scene.render.filepath = str(tmp)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(tmp))
        arr = np.array(img.pixels[:], dtype=np.float32).reshape((res, res, 4))
        bpy.data.images.remove(img)
        tmp.unlink(missing_ok=True)
        tiles.append(arr)

    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)

    sheet = np.full((2 * res, 2 * res, 4), 0.90, dtype=np.float32)
    sheet[..., 3] = 1.0
    for idx, t in enumerate(tiles):
        r_bl = 1 - (idx // 2)
        c = idx % 2
        sheet[r_bl * res : (r_bl + 1) * res, c * res : (c + 1) * res] = t

    out_img = bpy.data.images.new("RebuiltHairSheet", width=2 * res, height=2 * res, alpha=True)
    out_img.pixels.foreach_set(sheet.ravel())
    out_img.filepath_raw = str(out_png)
    out_img.file_format = "PNG"
    out_img.save()
    bpy.data.images.remove(out_img)
    return str(out_png)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    raw_data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    center_gltf = np.zeros(3, dtype=np.float64)
    scale = 1.0
    if args.meta and Path(args.meta).is_file():
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        center_gltf = np.asarray(meta.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        scale = float(meta.get("scale", 1.0))

    head_objs = []
    if args.head and Path(args.head).is_file():
        hp = Path(args.head)
        if hp.suffix.lower() == ".blend":
            bpy.ops.wm.open_mainfile(filepath=str(hp))
        else:
            bpy.ops.import_scene.gltf(filepath=str(hp))
        for o in list(bpy.context.scene.objects):
            if o.type == "MESH":
                pts_o = [o.matrix_world @ v.co for v in o.data.vertices]
                span_o = max(
                    (max(p.x for p in pts_o) - min(p.x for p in pts_o)) if pts_o else 0.0,
                    (max(p.y for p in pts_o) - min(p.y for p in pts_o)) if pts_o else 0.0,
                )
                if span_o < 0.6:
                    head_objs.append(o)
                else:
                    bpy.data.objects.remove(o, do_unlink=True)

    strands = _clean_and_split_strands(raw_data, center_gltf, scale, args.jump_threshold)
    if not strands:
        raise RuntimeError("No valid hair strands remained after cleaning.")

    all_pts = np.concatenate([s[0] for s in strands], axis=0)
    g_zmin, g_zmax = float(all_pts[:, 2].min()), float(all_pts[:, 2].max())

    if head_objs:
        h_pts = [o.matrix_world @ v.co for o in head_objs for v in o.data.vertices]
        z_hi = max(p.z for p in h_pts)
        h_pts = [p for p in h_pts if p.z > z_hi - 0.30]
        h_min = Vector((min(p.x for p in h_pts), min(p.y for p in h_pts), min(p.z for p in h_pts)))
        h_max = Vector((max(p.x for p in h_pts), max(p.y for p in h_pts), max(p.z for p in h_pts)))
        head_center = Vector((0.5 * (h_min.x + h_max.x), 0.5 * (h_min.y + h_max.y) + 0.02, 0.5 * (h_min.z + h_max.z) + 0.02))
        head_r = min(0.118, max(0.085, max(h_max.x - h_min.x, h_max.y - h_min.y) * 0.52))
    else:
        top_pts = all_pts[all_pts[:, 2] >= np.percentile(all_pts[:, 2], 70)]
        head_center = Vector((float(top_pts[:, 0].mean()), float(top_pts[:, 1].mean()), float(np.percentile(all_pts[:, 2], 82))))
        head_r = max(0.085, min(0.13, float(np.percentile(np.abs(top_pts[:, 0] - head_center.x), 85))))

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()
    cols = 8
    built_count = 0
    for idx, (pts, ws, ts) in enumerate(strands):
        res = _smooth_and_resample(pts, ws, ts, n_samples=args.samples)
        if res is None:
            continue
        p_s, w_s, _, _ = res
        col = idx % cols
        _add_anime_arch_strand(bm, uv_layer, p_s, w_s, head_center, col / cols + 0.008, (col + 1) / cols - 0.008, g_zmin, g_zmax)
        built_count += 1

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("AnimeHair_RebuiltStrands")
    bm.to_mesh(me)
    bm.free()
    strands_obj = bpy.data.objects.new("AnimeHair_RebuiltStrands", me)
    bpy.context.scene.collection.objects.link(strands_obj)
    for p in me.polygons:
        p.use_smooth = True

    scalp_obj = _build_recessed_scalp_cap(head_center, head_r, g_zmin, g_zmax)
    hair_objs = [strands_obj, scalp_obj]

    _apply_smooth_cage_normals(hair_objs)
    atlas_p = (
        Path(args.atlas)
        if args.atlas and Path(args.atlas).is_file()
        else _generate_gradient_atlas(
            out_dir / "hair_vertical_atlas_512x1024.png",
            _hex_to_rgb(args.root_hex),
            _hex_to_rgb(args.mid_hex),
            _hex_to_rgb(args.tip_hex),
        )
    )
    _apply_matte_toon_material(hair_objs, atlas_p)

    total_polys = sum(len(o.data.polygons) for o in hair_objs)
    quad_polys = sum(sum(1 for p in o.data.polygons if len(p.vertices) == 4) for o in hair_objs)
    quad_ratio = round(quad_polys / max(1, total_polys), 4)

    out_blend = out_dir / "anime_hair_rebuilt.blend"
    out_glb = out_dir / "anime_hair_rebuilt.glb"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    bpy.ops.object.select_all(action="DESELECT")
    for o in hair_objs:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB", use_selection=True)

    qc_png = ""
    if not args.no_render:
        qc_png = _render_4view(hair_objs + head_objs, out_dir / "anime_hair_rebuilt_4view.png", res=args.res)

    rep = {
        "status": "PASS" if quad_ratio >= 0.99 else "WARN",
        "raw_strands": len(raw_data),
        "cleaned_strands": built_count,
        "total_polys": total_polys,
        "quad_ratio": quad_ratio,
        "out_blend": str(out_blend),
        "out_glb": str(out_glb),
        "atlas": str(atlas_p),
        "qc_sheet": qc_png,
    }
    (out_dir / "hair_rebuild_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
