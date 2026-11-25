"""Headless Blender script: rebuild control-point hair strand JSON (e.g. CHARM output)
into 100% pure-quad, 3-tier layered, 8-point crescent-arch stylized anime hair with
crown root smoothing, back waterfall relaxation, recessed scalp+back veil, Smooth Cage
normal transfer, 1024x2048 painted dual-halo gradient atlas, and luminous MToon shader.
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
    p.add_argument("--jump-threshold", type=float, default=0.080, help="Split strand when step jump exceeds threshold")
    p.add_argument("--samples", type=int, default=26, help="Resampled longitudinal rings per strand")
    p.add_argument("--res", type=int, default=640, help="Per-tile resolution for 4-view QC render")
    p.add_argument("--no-render", action="store_true", help="Skip Cycles 4-view QC render")
    return p.parse_args(argv)


def _hex_to_rgb(hx: str) -> tuple[float, float, float]:
    h = hx.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _generate_hd_anime_hair_atlas(
    out_path: Path,
    root_rgb: tuple[float, float, float],
    mid_rgb: tuple[float, float, float],
    tip_rgb: tuple[float, float, float],
    w: int = 1024,
    h: int = 2048,
) -> Path:
    rng = np.random.default_rng(2026)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    r_c = np.array(root_rgb, dtype=np.float32)
    m_c = np.array(mid_rgb, dtype=np.float32)
    t_c = np.array(tip_rgb, dtype=np.float32)
    stops_y = np.array([0.0, 0.30, 0.62, 0.85, 1.0], dtype=np.float32)
    stops_c = np.array(
        [
            np.clip(t_c * 1.06, 0.0, 1.0),
            t_c * 0.94 + m_c * 0.06,
            m_c,
            r_c * 0.72 + m_c * 0.28,
            r_c * 0.82,
        ],
        dtype=np.float32,
    )
    base = np.zeros((h, 1, 3), dtype=np.float32)
    for c in range(3):
        base[:, 0, c] = np.interp(ys[:, 0], stops_y, stops_c[:, c])

    col_phase = (xs * 8.0) % 1.0
    edge_ao = 0.89 + 0.11 * np.sin(col_phase * math.pi) ** 0.55

    fine_1d = np.zeros(w, dtype=np.float32)
    for freq, amp in [(24, 0.025), (64, 0.032), (140, 0.028), (280, 0.020)]:
        phase = rng.uniform(0, 2 * math.pi)
        fine_1d += amp * np.sin(xs[0] * freq * 2 * math.pi + phase)
    for _ in range(90):
        cx = rng.uniform(0.0, 1.0)
        sig = rng.uniform(0.0012, 0.0035)
        amp = rng.uniform(-0.045, 0.055)
        fine_1d += amp * np.exp(-((xs[0] - cx) ** 2) / (2 * sig * sig))

    fibers = (1.0 + fine_1d[None, :]) * edge_ao

    zigzag = 0.024 * np.sin(xs * 16.0 * math.pi) + 0.012 * np.sin(xs * 42.0 * math.pi + 1.1)
    y_h1 = 0.74 + zigzag
    halo_main = np.exp(-((ys - y_h1) ** 2) / (2.0 * 0.018 ** 2))
    halo_cel = np.clip((halo_main - 0.35) / 0.65, 0.0, 1.0) ** 0.85

    y_h2 = 0.66 + zigzag * 0.7
    halo_sub = np.exp(-((ys - y_h2) ** 2) / (2.0 * 0.026 ** 2)) * 0.22
    root_depth = 1.0 - 0.10 * np.clip((ys - 0.90) / 0.10, 0.0, 1.0)

    rgb = base * fibers[..., None] * root_depth[..., None]
    halo_color = np.array([0.96, 0.94, 1.00], dtype=np.float32)
    rgb = rgb * (1.0 - 0.40 * halo_cel[..., None]) + halo_color * (0.40 * halo_cel[..., None])
    rgb = np.clip(rgb + halo_sub[..., None] * np.array([0.92, 0.84, 0.98], dtype=np.float32), 0.0, 1.0)

    arr = np.ones((h, w, 4), dtype=np.float32)
    arr[..., :3] = rgb

    img = bpy.data.images.new("HDAnimeHairAtlas", width=w, height=h, alpha=True)
    img.pixels.foreach_set(arr.ravel())
    img.filepath_raw = str(out_path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return out_path


def _gltf_to_blender(pt: np.ndarray) -> np.ndarray:
    return np.array([pt[0], -pt[2], pt[1]], dtype=np.float64)


def _clean_and_split_strands(
    raw_data: list, center_gltf: np.ndarray, scale: float, jump_thr: float, head_center: Vector, head_r: float
):
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
            if dz < 0.045 and dxy > 0.12:
                continue

            arc = float(np.sum(np.linalg.norm(np.diff(seg_pts, axis=0), axis=1)))
            if arc < 0.09:
                continue

            if seg_pts[0, 1] < seg_pts[-1, 1]:
                seg_pts = seg_pts[::-1]
                seg_w = seg_w[::-1]
                seg_t = seg_t[::-1]

            pts_world_gltf = seg_pts / scale + center_gltf
            pts_bl = np.array([_gltf_to_blender(p) for p in pts_world_gltf], dtype=np.float64)
            if pts_bl[0, 2] > head_center.z + head_r * 0.70 and (pts_bl[0, 2] - pts_bl[-1, 2]) < 0.06:
                continue
            cleaned.append((pts_bl, seg_w / scale, seg_t / scale))
    return cleaned


def _sculpt_strand_curve(
    pts: np.ndarray,
    ws: np.ndarray,
    head_center: Vector,
    head_r: float,
    n_samples: int = 26,
    layer_offset: float = 0.0,
    is_back_long: bool = False,
):
    p = pts.copy()
    for _ in range(8):
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

    hc = np.array([head_center.x, head_center.y, head_center.z], dtype=np.float64)
    for i in range(n_samples):
        t = i / max(1, n_samples - 1)
        rel = out_p[i] - hc

        if t < 0.20 and rel[2] > 0.0:
            dist = np.linalg.norm(rel)
            if dist > 1e-4:
                target_r = head_r * (0.96 + 0.06 * (t / 0.20)) + layer_offset
                blend = 0.45 * (1.0 - t / 0.20)
                if dist > target_r:
                    out_p[i] = hc + rel * ((1.0 - blend) + blend * (target_r / dist))

        if is_back_long and t > 0.35:
            wt = (t - 0.35) / 0.65
            out_p[i, 0] = hc[0] + (out_p[i, 0] - hc[0]) * (1.0 + 0.16 * (wt ** 1.15))
            out_p[i, 1] += 0.012 * (wt ** 1.2)

        if abs(layer_offset) > 1e-6:
            horiz = np.array([out_p[i, 0] - hc[0], out_p[i, 1] - hc[1], 0.0], dtype=np.float64)
            h_len = np.linalg.norm(horiz)
            if h_len > 1e-4:
                out_p[i] += (horiz / h_len) * layer_offset * min(1.0, t / 0.18)

    return out_p, out_w, total


def _add_crescent_anime_strand(
    bm: bmesh.types.BMesh,
    uv_layer,
    pts: np.ndarray,
    ws: np.ndarray,
    head_center: Vector,
    u_lo: float,
    u_hi: float,
    g_zmin: float,
    g_zmax: float,
    tier: int,
    is_front: bool,
) -> None:
    n = len(pts)
    profile = [
        (-0.50, -0.08),
        (-0.32, 0.26),
        (-0.14, 0.50),
        (0.00, 0.58),
        (0.14, 0.50),
        (0.32, 0.26),
        (0.50, -0.08),
        (0.00, -0.18),
    ]
    u_prof = [0.00, 0.18, 0.36, 0.50, 0.64, 0.82, 1.00, 0.50]
    n_prof = len(profile)

    rings = []
    prev_b = None
    g_zspan = max(g_zmax - g_zmin, 1e-4)
    s_zmin, s_zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    s_zspan = max(s_zmax - s_zmin, 1e-4)

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

        out_vec = Vector((p.x - head_center.x, p.y - head_center.y, (p.z - head_center.z) * 0.28))
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
            if tier == 0:
                w = max(0.032, min(0.054, base_w * 2.7 + 0.020))
            elif tier == 1:
                w = max(0.022, min(0.038, base_w * 2.2 + 0.015))
            else:
                w = max(0.013, min(0.024, base_w * 1.5 + 0.010))
        else:
            if tier == 0:
                w = max(0.062, min(0.095, base_w * 4.2 + 0.042))
            elif tier == 1:
                w = max(0.038, min(0.060, base_w * 2.8 + 0.024))
            else:
                w = max(0.016, min(0.028, base_w * 1.6 + 0.011))

        root_env = 0.58 + 0.42 * min(1.0, t / 0.14)
        tip_start = 0.78 if (not is_front and tier == 0) else 0.70
        if t > tip_start:
            tip_t = (t - tip_start) / (1.0 - tip_start)
            tip_env = max(0.012, (1.0 - tip_t) ** 1.55)
        else:
            tip_env = 1.0 + 0.10 * math.sin(math.pi * t / tip_start)

        w_cur = w * root_env * tip_env
        th_cur = max(0.0018, w_cur * (0.28 if is_front else 0.24))

        ring_verts = [bm.verts.new(p + b_vec * (px * w_cur) + n_vec * (py * th_cur)) for (px, py) in profile]
        v_glob = (p.z - g_zmin) / g_zspan
        v_loc = (p.z - s_zmin) / s_zspan
        v_mix = 0.65 * v_glob + 0.35 * v_loc if is_front else 0.85 * v_glob + 0.15 * v_loc
        rings.append((ring_verts, max(0.01, min(0.99, v_mix))))

    for i in range(n - 1):
        r0, v0 = rings[i]
        r1, v1 = rings[i + 1]
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


def _build_recessed_scalp_and_veil(head_center: Vector, head_r: float, back_bottom_z: float, g_zmin: float, g_zmax: float):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()
    g_zspan = max(g_zmax - g_zmin, 1e-4)

    n_u, n_v = 32, 12
    dome_rings = []
    for iv in range(n_v):
        t = iv / (n_v - 1)
        phi = 0.04 + t * (math.pi * 0.52)
        ring = []
        for iu in range(n_u):
            ang = 2.0 * math.pi * iu / n_u
            rx = head_r * 0.84
            ry = head_r * (0.90 if math.sin(ang) > 0 else 0.74)
            rz = head_r * 0.88
            x = head_center.x + rx * math.sin(phi) * math.cos(ang)
            y = head_center.y + ry * math.sin(phi) * math.sin(ang) + 0.008
            z = head_center.z + rz * math.cos(phi) + 0.012
            ring.append(bm.verts.new(Vector((x, y, z))))
        dome_rings.append(ring)

    for iv in range(n_v - 1):
        for iu in range(n_u):
            iun = (iu + 1) % n_u
            f = bm.faces.new([dome_rings[iv][iu], dome_rings[iv][iun], dome_rings[iv + 1][iun], dome_rings[iv + 1][iu]])
            for loop in f.loops:
                vz = max(0.01, min(0.99, (loop.vert.co.z - g_zmin) / g_zspan))
                loop[uv_layer].uv = (0.02 + 0.96 * (iu / n_u), vz)

    c_u, c_v = 28, 16
    ang_s, ang_e = math.radians(22), math.radians(158)
    top_z = head_center.z + head_r * 0.25
    veil_rings = []
    for iv in range(c_v):
        t = iv / (c_v - 1)
        ring = []
        for iu in range(c_u):
            s = iu / (c_u - 1)
            ang = ang_s + (ang_e - ang_s) * s
            scallop = 0.016 * abs(math.sin(s * math.pi * 6.0)) * (t ** 2)
            z = top_z * (1.0 - t) + (back_bottom_z + 0.045 + scallop) * t
            rx = head_r * (0.94 + 0.20 * (t ** 1.15))
            ry = head_r * (0.90 + 0.16 * (t ** 1.15))
            x = head_center.x + rx * math.cos(ang)
            y = head_center.y + ry * math.sin(ang) + 0.034 + 0.036 * t
            ring.append(bm.verts.new(Vector((x, y, z))))
        veil_rings.append(ring)

    for iv in range(c_v - 1):
        for iu in range(c_u - 1):
            f = bm.faces.new([veil_rings[iv][iu], veil_rings[iv][iu + 1], veil_rings[iv + 1][iu + 1], veil_rings[iv + 1][iu]])
            for loop in f.loops:
                vz = max(0.01, min(0.99, (loop.vert.co.z - g_zmin) / g_zspan))
                loop[uv_layer].uv = (0.02 + 0.96 * (iu / max(1, c_u - 1)), vz)

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("AnimeHair_ScalpAndVeil")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("AnimeHair_ScalpAndVeil", me)
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

    bpy.ops.mesh.primitive_uv_sphere_add(segments=36, ring_count=24, radius=1.0, location=center)
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
        mod.mix_factor = 0.78
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.data.objects.remove(cage, do_unlink=True)


def _apply_mtoon_luminous_material(hair_objs: list, atlas_path: Path) -> None:
    mat = bpy.data.materials.new("AnimeHair_Rebuilt_MToon")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(atlas_path))
    tex.extension = "EXTEND"

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.94
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.03

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 0.84

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.36

    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emit.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    for obj in hair_objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def _render_4view(target_objs: list, out_png: Path, res: int = 640) -> str:
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

    scene.cycles.samples = 40
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
    bg.inputs[0].default_value = (0.91, 0.92, 0.95, 1.0)
    bg.inputs[1].default_value = 0.92

    created = []
    for lname, off, eng in [
        ("Key", Vector((-1.1, -1.7, 1.0)), 95),
        ("Fill", Vector((1.4, -1.2, 0.4)), 78),
        ("Back", Vector((0.0, 1.8, 0.8)), 85),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng * max(0.5, span * span)
        ld.size = span * 1.5
        lo = bpy.data.objects.new(lname, ld)
        scene.collection.objects.link(lo)
        lo.location = center + off * span
        lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()
        created.append(lo)

    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 52.0
    cam = bpy.data.objects.new("Cam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    created.append(cam)

    views = [
        ("front", center + Vector((0.0, -1.85, 0.06)) * span),
        ("34", center + Vector((-1.32, -1.32, 0.10)) * span),
        ("side", center + Vector((1.85, 0.0, 0.06)) * span),
        ("back", center + Vector((0.0, 1.85, 0.06)) * span),
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

    sheet = np.full((2 * res, 2 * res, 4), 0.91, dtype=np.float32)
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

    if head_objs:
        h_pts = [o.matrix_world @ v.co for o in head_objs for v in o.data.vertices]
        z_hi = max(p.z for p in h_pts)
        h_pts = [p for p in h_pts if p.z > z_hi - 0.30]
        h_min = Vector((min(p.x for p in h_pts), min(p.y for p in h_pts), min(p.z for p in h_pts)))
        h_max = Vector((max(p.x for p in h_pts), max(p.y for p in h_pts), max(p.z for p in h_pts)))
        head_center = Vector((0.5 * (h_min.x + h_max.x), 0.5 * (h_min.y + h_max.y) + 0.02, 0.5 * (h_min.z + h_max.z) + 0.02))
        head_r = min(0.118, max(0.085, max(h_max.x - h_min.x, h_max.y - h_min.y) * 0.52))
    else:
        head_center = Vector((0.0, 0.0, 0.25))
        head_r = 0.105

    strands = _clean_and_split_strands(raw_data, center_gltf, scale, args.jump_threshold, head_center, head_r)
    if not strands:
        raise RuntimeError("No valid hair strands remained after cleaning.")

    all_pts = np.concatenate([s[0] for s in strands], axis=0)
    g_zmin, g_zmax = float(all_pts[:, 2].min()), float(all_pts[:, 2].max())
    if not head_objs:
        top_pts = all_pts[all_pts[:, 2] >= np.percentile(all_pts[:, 2], 70)]
        head_center = Vector((float(top_pts[:, 0].mean()), float(top_pts[:, 1].mean()), float(np.percentile(all_pts[:, 2], 82))))
        head_r = max(0.085, min(0.13, float(np.percentile(np.abs(top_pts[:, 0] - head_center.x), 85))))

    scored = []
    for idx, (pts, ws, ts) in enumerate(strands):
        arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        scored.append((float(np.mean(ws)) * 0.6 + arc * 0.4, idx, pts, ws, ts, arc))
    scored.sort(key=lambda x: x[0], reverse=True)
    n_s = len(scored)

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()
    cols = 8
    built_count = 0
    for rank, (_, idx, pts, ws, ts, arc) in enumerate(scored):
        if rank < int(n_s * 0.40):
            tier, layer_off = 0, 0.000
        elif rank < int(n_s * 0.80):
            tier, layer_off = 1, 0.0035
        else:
            tier, layer_off = 2, 0.0068
        mean_y = float(np.mean(pts[:, 1]))
        is_front = mean_y < (head_center.y - 0.025)
        is_back_long = (not is_front) and (arc > 0.30)
        res = _sculpt_strand_curve(pts, ws, head_center, head_r, args.samples, layer_off, is_back_long)
        if res is None:
            continue
        p_s, w_s, _ = res
        col = idx % cols
        _add_crescent_anime_strand(
            bm, uv_layer, p_s, w_s, head_center, col / cols + 0.008, (col + 1) / cols - 0.008, g_zmin, g_zmax, tier, is_front
        )
        built_count += 1

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("AnimeHair_RebuiltStrands")
    bm.to_mesh(me)
    bm.free()
    strands_obj = bpy.data.objects.new("AnimeHair_RebuiltStrands", me)
    bpy.context.scene.collection.objects.link(strands_obj)
    for p in me.polygons:
        p.use_smooth = True

    scalp_veil = _build_recessed_scalp_and_veil(head_center, head_r, g_zmin, g_zmin, g_zmax)
    hair_objs = [strands_obj, scalp_veil]

    _apply_smooth_cage_normals(hair_objs)
    atlas_p = (
        Path(args.atlas)
        if args.atlas and Path(args.atlas).is_file()
        else _generate_hd_anime_hair_atlas(
            out_dir / "hd_anime_hair_atlas_1024x2048.png",
            _hex_to_rgb(args.root_hex),
            _hex_to_rgb(args.mid_hex),
            _hex_to_rgb(args.tip_hex),
        )
    )
    _apply_mtoon_luminous_material(hair_objs, atlas_p)

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
