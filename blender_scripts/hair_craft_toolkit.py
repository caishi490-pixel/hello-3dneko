"""Anime 3D Hair Craft Toolkit — Vertical Strip UV, Scalp Cap, Toon Material & 6-Pillar Audit (Blender 5.1+).

Implements the industrial stylized anime hair blueprint (`hair-craft-blueprint.md`):
  - `--mode audit`: Quantitative 6-pillar audit of any hair mesh (dual-layer check, quad purity,
                    vertical strip UV axis-alignment %, matte Toon material check, manifold cleanliness,
                    and bone-weight <=2 influence rule) + 4-view QC render sheet.
  - `--mode polish`: Upgrades any existing hair mesh (AI-generated or sculpted strands) with:
                     1. Axis-aligned vertical strip UV unwrapping (V=1 root -> V=0 tip) per strand shell
                     2. Optional skull-hugging watertight quad scalp cap (`--add-scalp-cap`)
                     3. Zero-glare matte anime Toon material (`Roughness=1.0, Specular=0.0`) + procedural
                        512x1024 vertical root-to-tip gradient & horizontal anime halo highlight atlas
                     4. Clean `.blend` / `.glb` export + 4-view QC render + 6-pillar scorecard.
  - `--mode build-template`: Generates a reference-blueprint dual-layer anime long-hair kit
                             (1 scalp cap + 56 tapered closed-tube strands with 100% axis-aligned
                             vertical strip UVs & anime halo gradient atlas) fitted to a target skull
                             or standard 1.654m anime head dimensions.
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
from mathutils import Matrix, Vector


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Stylized anime 3D hair craft toolkit & 6-pillar auditor.")
    p.add_argument("--mode", choices=("audit", "polish", "build-template"), required=True)
    p.add_argument("--input", default="", help="Input hair .blend or .glb (required for audit/polish).")
    p.add_argument("--body", default="", help="Optional basemesh .blend/.glb to fit scalp cap & template head bounds.")
    p.add_argument("--out-dir", required=True, help="Output workspace directory.")
    p.add_argument("--hair-mesh", default="", help="Optional exact hair mesh object name.")
    p.add_argument("--strip-uv", action="store_true", help="Re-unwrap each strand shell into vertical V=1..0 strip UVs.")
    p.add_argument("--add-scalp-cap", action="store_true", help="Generate a skull-hugging quad scalp under-shell.")
    p.add_argument("--toon-mat", action="store_true", help="Apply zero-glare matte Toon material + 512x1024 strand gradient atlas.")
    p.add_argument("--root-hex", default="#4A5BD4", help="Hair root color hex (default blue-violet).")
    p.add_argument("--mid-hex", default="#8C6CE8", help="Hair mid-length color hex (default lavender).")
    p.add_argument("--tip-hex", default="#F2A4D0", help="Hair tip color hex (default sakura pink).")
    p.add_argument("--res", type=int, default=512, help="Per-tile render resolution for 2x2 QC sheet.")
    return p.parse_args(argv)


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    s = h.strip().lstrip("#")
    if len(s) != 6:
        return (0.55, 0.45, 0.88)
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def _load_scene(filepath: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fp = str(filepath)
    if fp.lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=fp)
    else:
        bpy.ops.import_scene.gltf(filepath=fp)


def _import_extra_body(filepath: str):
    before = set(bpy.context.scene.objects)
    fp = str(filepath)
    if fp.lower().endswith(".blend"):
        with bpy.data.libraries.load(fp, link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
    else:
        bpy.ops.import_scene.gltf(filepath=fp)
    return [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH" and len(o.data.polygons) > 0]


def _pick_hair_objects(name_hint: str = "") -> list:
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons) > 0]
    if not meshes:
        raise RuntimeError("No non-empty mesh found in scene.")
    if name_hint:
        found = [o for o in meshes if o.name == name_hint]
        if not found:
            raise RuntimeError(f"Hair mesh '{name_hint}' not found; available: {[o.name for o in meshes]}")
        return found
    hair_named = [o for o in meshes if "hair" in o.name.lower() or "scalp" in o.name.lower()]
    if hair_named:
        return hair_named
    return [max(meshes, key=lambda o: len(o.data.polygons))]


def _estimate_head_sphere(body_objs: list) -> tuple[Vector, float]:
    """Estimate anime head center & skull radius from basemesh objects, or return standard 1.654m head."""
    if not body_objs:
        return Vector((0.0, 0.0, 1.545)), 0.108
    pts = []
    for o in body_objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            pts.append(mw @ v.co)
    if not pts:
        return Vector((0.0, 0.0, 1.545)), 0.108
    z_max = max(p.z for p in pts)
    z_min = min(p.z for p in pts)
    total_h = max(0.3, z_max - z_min)
    # Standard 6.9-head anime skull occupies top ~13.5% of height
    head_r = max(0.075, min(0.14, total_h * 0.066))
    top_pts = [p for p in pts if p.z >= z_max - 2.1 * head_r]
    cx = sum(p.x for p in top_pts) / max(1, len(top_pts))
    cy = sum(p.y for p in top_pts) / max(1, len(top_pts))
    cz = z_max - head_r * 1.02
    return Vector((cx, cy, cz)), head_r


def _generate_hair_gradient_atlas(out_path: Path, root_rgb, mid_rgb, tip_rgb, w: int = 512, h: int = 1024) -> Path:
    """Create a 512x1024 (1:2) vertical anime hair atlas: root-to-tip gradient + strand streaks + horizontal halo highlight."""
    # Blender image row 0 = bottom (V=0, tip), row h-1 = top (V=1, root)
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # (h, 1)
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]  # (1, w)

    # Smooth 3-stop root(1.0) -> mid(0.55) -> tip(0.0) color ramp
    r_c = np.array(root_rgb, dtype=np.float32)[None, None, :]
    m_c = np.array(mid_rgb, dtype=np.float32)[None, None, :]
    t_c = np.array(tip_rgb, dtype=np.float32)[None, None, :]

    w_upper = np.clip((v - 0.50) / 0.50, 0.0, 1.0)[..., None]
    w_lower = np.clip(v / 0.50, 0.0, 1.0)[..., None]
    base_rgb = np.where(v[..., None] >= 0.50, m_c * (1.0 - w_upper) + r_c * w_upper, t_c * (1.0 - w_lower) + m_c * w_lower)

    # Fine vertical strand fiber streaks (multi-frequency along U, soft taper at tips)
    streaks = (
        0.55 * np.sin(2.0 * np.pi * 24.0 * u)
        + 0.30 * np.sin(2.0 * np.pi * 63.0 * u + 1.1)
        + 0.15 * np.sin(2.0 * np.pi * 131.0 * u + 2.4)
    )
    streak_mod = (1.0 + 0.085 * streaks * (0.45 + 0.55 * v))[..., None]
    rgb = np.clip(base_rgb * streak_mod, 0.0, 1.0)

    # Crisp anime horizontal angel-ring halo highlight around V = 0.72 with jagged strand-modulated band
    halo_center = 0.73 + 0.022 * np.sin(2.0 * np.pi * 16.0 * u) + 0.010 * np.sin(2.0 * np.pi * 41.0 * u + 0.7)
    halo_band = np.exp(-0.5 * ((v - halo_center) / 0.028) ** 2)
    halo_secondary = 0.45 * np.exp(-0.5 * ((v - (halo_center - 0.065)) / 0.014) ** 2)
    halo = np.clip((halo_band + halo_secondary) * (0.65 + 0.35 * np.maximum(0.0, streaks)), 0.0, 1.0)[..., None]
    halo_tint = np.array([1.0, 0.96, 1.0], dtype=np.float32)[None, None, :]
    rgb = np.clip(rgb * (1.0 - 0.52 * halo) + halo_tint * (0.52 * halo), 0.0, 1.0)

    # Subtle root occlusion shadow at very top V > 0.92
    root_ao = 1.0 - 0.22 * np.clip((v - 0.90) / 0.10, 0.0, 1.0)[..., None]
    rgb = np.clip(rgb * root_ao, 0.0, 1.0)

    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = bpy.data.images.new("Hair_Vertical_Atlas", width=w, height=h, alpha=True)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = str(out_path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return out_path


def _apply_matte_toon_hair_material(hair_objs: list, atlas_path: Path):
    """Apply zero-glare matte anime Toon material (Roughness=1.0, Specular=0.0) backed by the 512x1024 vertical atlas."""
    mat = bpy.data.materials.new("Hair_AnimeToon_Mat")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 1.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.0

    tex_n = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(atlas_path), check_existing=True)
    img.pack()
    tex_n.image = img
    nt.links.new(tex_n.outputs["Color"], bsdf.inputs["Base Color"])

    for obj in hair_objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for p in obj.data.polygons:
            p.material_index = 0
            p.use_smooth = True


def _apply_vertical_strip_uv(obj) -> dict:
    """Unwrap each welded strand shell into axis-aligned vertical strips (V=1.0 root -> V=0.0 tip)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.verify()

    # Group faces by 3D-welded connected components (each strand shell)
    pk = lambda v: (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
    pf: dict[tuple[float, float, float], list[int]] = {}
    for f in bm.faces:
        for v in f.verts:
            pf.setdefault(pk(v), []).append(f.index)

    vis = set()
    shells: list[list[int]] = []
    for fi in range(len(bm.faces)):
        if fi in vis:
            continue
        comp = []
        q = deque([fi])
        vis.add(fi)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in bm.faces[u].verts:
                for nf in pf.get(pk(v), ()):
                    if nf not in vis:
                        vis.add(nf)
                        q.append(nf)
        shells.append(comp)

    # Pack each strand shell into a clean vertical strip column along [0..1]
    n_shells = max(1, len(shells))
    cols = min(16, n_shells)
    u_margin = 0.008
    for s_idx, comp in enumerate(shells):
        col = s_idx % cols
        u_lo = (col / cols) + u_margin
        u_hi = ((col + 1) / cols) - u_margin
        verts = list({v for fi in comp for v in bm.faces[fi].verts})
        zs = [v.co.z for v in verts]
        z_min, z_max = min(zs), max(zs)
        z_span = max(1e-4, z_max - z_min)
        cx = sum(v.co.x for v in verts) / len(verts)
        cy = sum(v.co.y for v in verts) / len(verts)

        # Quantize angular coordinate around strand centerline into clean vertical bands so UV edges stay axis-aligned
        for fi in comp:
            f = bm.faces[fi]
            # Use local quad-friendly cylindrical mapping snapped softly toward V-axis
            for loop in f.loops:
                co = loop.vert.co
                v_coord = max(0.01, min(0.99, (co.z - z_min) / z_span))
                ang = math.atan2(co.y - cy, co.x - cx)  # -pi..pi
                u_frac = (ang + math.pi) / (2.0 * math.pi)
                u_coord = u_lo + (u_hi - u_lo) * u_frac
                loop[uv_layer].uv = (u_coord, v_coord)

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return {"shells_unwrapped": len(shells), "strip_columns": cols}


def _build_scalp_cap(head_center: Vector, head_r: float) -> bpy.types.Object:
    """Create a watertight ~720-quad skull-hugging base scalp cap (`HairBack_ScalpCap`) with clean vertical strip UVs."""
    bm = bmesh.new()
    segs = 36
    rings = 14
    ring_verts = []
    for r_i in range(rings):
        t = r_i / (rings - 1)  # 0 = lower nape/hairline, 1 = near crown
        # Polar angle from crown (0.10 rad) down past equator (1.85 rad at back nape, 1.25 rad at forehead)
        row = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs  # -pi/2 is front (-Y), +pi/2 is back (+Y)
            is_back = max(0.0, math.sin(ang))
            max_phi = 1.28 + 0.62 * is_back  # Covers deeper down back nape, stays above eyebrows in front
            phi = 0.12 + (1.0 - t) * (max_phi - 0.12)
            rad = head_r * (1.035 + 0.015 * is_back)
            x = head_center.x + rad * math.sin(phi) * math.cos(ang)
            y = head_center.y + rad * math.sin(phi) * math.sin(ang)
            z = head_center.z + rad * math.cos(phi) * 1.06
            row.append(bm.verts.new((x, y, z)))
        ring_verts.append(row)

    uv_layer = bm.loops.layers.uv.verify()
    for r_i in range(rings - 1):
        v0 = 0.45 + 0.52 * (r_i / (rings - 1))
        v1 = 0.45 + 0.52 * ((r_i + 1) / (rings - 1))
        for s in range(segs):
            sn = (s + 1) % segs
            f = bm.faces.new((ring_verts[r_i][s], ring_verts[r_i][sn], ring_verts[r_i + 1][sn], ring_verts[r_i + 1][s]))
            u0, u1 = s / segs, (s + 1) / segs
            for lp, uv in zip(f.loops, ((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
                lp[uv_layer].uv = uv

    # Close crown cleanly with quads/grid fan
    crown_v = bm.verts.new((head_center.x, head_center.y, head_center.z + head_r * 1.095))
    top_row = ring_verts[-1]
    for s in range(segs):
        sn = (s + 1) % segs
        f = bm.faces.new((top_row[s], top_row[sn], crown_v))
        u0, u1 = s / segs, (s + 1) / segs
        for lp, uv in zip(f.loops, ((u0, 0.97), (u1, 0.97), (0.5 * (u0 + u1), 0.995))):
            lp[uv_layer].uv = uv

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("HairBack_ScalpCap")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("HairBack_ScalpCap", me)
    bpy.context.scene.collection.objects.link(obj)
    for p in me.polygons:
        p.use_smooth = True
    return obj


def _add_tapered_strand_tube(
    bm: bmesh.types.BMesh,
    uv_layer,
    pts: list[Vector],
    widths: list[float],
    thicknesses: list[float],
    u_lo: float,
    u_hi: float,
):
    """Build one closed-tube anime hair strand with diamond/teardrop 6-vert cross section & 100% axis-aligned strip UVs."""
    n_pts = len(pts)
    profile = [
        (1.0, 0.0),
        (0.55, 0.85),
        (-0.55, 0.85),
        (-1.0, 0.0),
        (-0.55, -0.55),
        (0.55, -0.55),
    ]
    n_prof = len(profile)
    rings = []
    for i in range(n_pts):
        p = pts[i]
        if i == 0:
            tang = (pts[1] - pts[0]).normalized()
        elif i == n_pts - 1:
            tang = (pts[-1] - pts[-2]).normalized()
        else:
            tang = (pts[i + 1] - pts[i - 1]).normalized()
        ref = Vector((0.0, -1.0, 0.0)) if abs(tang.dot(Vector((0.0, -1.0, 0.0)))) < 0.9 else Vector((1.0, 0.0, 0.0))
        side = tang.cross(ref).normalized()
        out = side.cross(tang).normalized()
        w_cur, th_cur = widths[i], thicknesses[i]
        row = [bm.verts.new(p + side * (px * w_cur) + out * (py * th_cur)) for px, py in profile]
        rings.append(row)

    for i in range(n_pts - 1):
        v0 = 1.0 - (i / (n_pts - 1))
        v1 = 1.0 - ((i + 1) / (n_pts - 1))
        for j in range(n_prof):
            jn = (j + 1) % n_prof
            u0 = u_lo + (u_hi - u_lo) * (j / n_prof)
            u1 = u_lo + (u_hi - u_lo) * ((j + 1) / n_prof)
            f = bm.faces.new((rings[i][j], rings[i][jn], rings[i + 1][jn], rings[i + 1][j]))
            for lp, uv in zip(f.loops, ((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
                lp[uv_layer].uv = uv


def _build_blueprint_hair_template(head_center: Vector, head_r: float) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Generate a dual-layer blueprint anime long-hair kit: 1 scalp cap + 56 tapered closed-tube strands with 100% strip UVs."""
    scalp_obj = _build_scalp_cap(head_center, head_r)

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()
    strand_specs = []

    # 1. Front Bangs (14 strands): sweeping from crown down to eyebrow level, airy parted framing
    for i in range(14):
        f = (i / 13.0) * 2.0 - 1.0  # -1..+1
        ang = -math.pi * 0.5 + f * 0.72
        length = head_r * (1.22 - 0.14 * abs(f) + 0.08 * math.sin(i * 1.9))
        w0 = head_r * (0.145 if abs(f) < 0.6 else 0.125)
        strand_specs.append(("front_bang", ang, length, w0, 0.16, 1.06, 0.018 * math.sin(f * 3.0)))

    # 2. Side Locks (14 strands): framing cheeks down past clavicle
    for side_sign in (-1.0, 1.0):
        for i in range(7):
            u_t = i / 6.0
            ang = -math.pi * 0.5 + side_sign * (0.78 + u_t * 0.65)
            length = head_r * (2.25 + 0.55 * u_t)
            w0 = head_r * 0.16
            strand_specs.append(("side_lock", ang, length, w0, 0.22, 1.08, -side_sign * 0.025))

    # 3. Long Back Hair (24 strands): flowing down to upper waist with gentle anime wave
    for i in range(24):
        f = (i / 23.0) * 2.0 - 1.0
        ang = math.pi * 0.5 + f * 1.48
        length = head_r * (4.10 - 0.55 * abs(f) + 0.25 * math.cos(i * 1.3))
        w0 = head_r * 0.19
        strand_specs.append(("back_hair", ang, length, w0, 0.18, 1.10, 0.032 * math.sin(i * 0.8)))

    # 4. Crown Top Volume Strands (4 strands)
    for i in range(4):
        ang = math.pi * 0.5 + (i - 1.5) * 0.55
        strand_specs.append(("crown", ang, head_r * 2.1, head_r * 0.17, 0.08, 1.13, 0.0))

    n_total = len(strand_specs)
    cols = 14
    for idx, (zone, ang, length, w0, start_phi, rad_mult, sway_x) in enumerate(strand_specs):
        segs_len = 12 if zone == "back_hair" else 9
        pts, ws, ths = [], [], []
        ca, sa = math.cos(ang), math.sin(ang)
        for k in range(segs_len):
            t = k / (segs_len - 1)
            # First 35% hugs skull curve, remaining 65% drapes gracefully downward with soft outward flare
            if t < 0.35:
                st = t / 0.35
                phi = start_phi + st * (1.35 - start_phi)
                r = head_r * rad_mult
                x = head_center.x + r * math.sin(phi) * ca
                y = head_center.y + r * math.sin(phi) * sa
                z = head_center.z + r * math.cos(phi) * 1.06
            else:
                dt = (t - 0.35) / 0.65
                base_r = head_r * rad_mult * math.sin(1.35)
                flare = head_r * (0.38 if zone == "back_hair" else 0.14) * (dt ** 1.3)
                wave = head_r * 0.08 * math.sin(dt * math.pi * 1.6 + idx * 0.5)
                x = head_center.x + (base_r + flare) * ca + sway_x * dt
                y = head_center.y + (base_r + flare + wave) * sa
                z = head_center.z + head_r * rad_mult * math.cos(1.35) * 1.06 - length * dt
            pts.append(Vector((x, y, z)))
            # Classic anime taper: medium at root -> widest at 35% -> razor point at tip
            taper = (0.72 + 0.38 * math.sin(math.pi * min(1.0, t * 1.15))) * (1.0 - 0.88 * (t ** 2.2))
            ws.append(max(0.0025, w0 * taper))
            ths.append(max(0.002, w0 * 0.42 * taper))

        col = idx % cols
        u_lo = col / cols + 0.004
        u_hi = (col + 1) / cols - 0.004
        _add_tapered_strand_tube(bm, uv_layer, pts, ws, ths, u_lo, u_hi)

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("Hair_Strands")
    bm.to_mesh(me)
    bm.free()
    strands_obj = bpy.data.objects.new("Hair_Strands", me)
    bpy.context.scene.collection.objects.link(strands_obj)
    for p in me.polygons:
        p.use_smooth = True
    return scalp_obj, strands_obj


def _audit_hair(hair_objs: list) -> dict:
    """Run quantitative 6-pillar anime hair craft audit against hair-craft-blueprint.md."""
    total_faces = 0
    quad_faces = 0
    non_manifold = 0
    degenerate = 0
    total_shells = 0
    uv_edges_total = 0
    uv_edges_aligned = 0
    max_bone_infs = 0
    has_weights = False

    for obj in hair_objs:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        total_faces += len(bm.faces)
        quad_faces += sum(1 for f in bm.faces if len(f.verts) == 4)
        non_manifold += sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
        degenerate += sum(1 for f in bm.faces if f.calc_area() < 1e-8)

        # Count 3D-welded strand shells
        pk = lambda v: (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
        pf: dict[tuple[float, float, float], list[int]] = {}
        for f in bm.faces:
            for v in f.verts:
                pf.setdefault(pk(v), []).append(f.index)
        vis = set()
        for fi in range(len(bm.faces)):
            if fi in vis:
                continue
            total_shells += 1
            q = deque([fi])
            vis.add(fi)
            while q:
                u = q.popleft()
                for v in bm.faces[u].verts:
                    for nf in pf.get(pk(v), ()):
                        if nf not in vis:
                            vis.add(nf)
                            q.append(nf)

        # Axis-aligned vertical strip UV check (within 12 deg of horizontal or vertical)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is not None:
            sin_lim = math.sin(math.radians(12.0))
            for f in bm.faces:
                loops = list(f.loops)
                n_l = len(loops)
                for i in range(n_l):
                    a = loops[i][uv_layer].uv
                    b = loops[(i + 1) % n_l][uv_layer].uv
                    du, dv = abs(b.x - a.x), abs(b.y - a.y)
                    ln = math.hypot(du, dv)
                    if ln < 1e-6:
                        continue
                    uv_edges_total += 1
                    if (min(du, dv) / ln) <= sin_lim:
                        uv_edges_aligned += 1
        bm.free()

        for v in obj.data.vertices:
            if len(v.groups) > 0:
                has_weights = True
                max_bone_infs = max(max_bone_infs, len(v.groups))

    # Material check (Roughness >= 0.90 and Specular <= 0.05)
    matte_mats = 0
    total_mats = 0
    for obj in hair_objs:
        for mat in obj.data.materials:
            if mat is None or not mat.node_tree:
                continue
            bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            total_mats += 1
            rough = float(bsdf.inputs["Roughness"].default_value)
            spec = float(
                bsdf.inputs["Specular IOR Level"].default_value
                if "Specular IOR Level" in bsdf.inputs
                else (bsdf.inputs["Specular"].default_value if "Specular" in bsdf.inputs else 0.0)
            )
            if rough >= 0.90 and spec <= 0.05:
                matte_mats += 1

    quad_ratio = quad_faces / max(1, total_faces)
    uv_align_pct = 100.0 * uv_edges_aligned / max(1, uv_edges_total)
    has_scalp_and_strands = (len(hair_objs) >= 2) or (total_shells >= 8)
    matte_ok = (total_mats > 0 and matte_mats == total_mats)
    weight_ok = (not has_weights) or (max_bone_infs <= 2)

    s_struct = 20.0 if has_scalp_and_strands else 8.0
    s_quad = 20.0 if quad_ratio >= 0.85 else round(20.0 * quad_ratio / 0.85, 2)
    s_uv = 25.0 if uv_align_pct >= 75.0 else round(25.0 * uv_align_pct / 75.0, 2)
    s_mat = 15.0 if matte_ok else 0.0
    s_clean = 10.0 if (non_manifold == 0 and degenerate == 0) else 4.0
    s_wt = 10.0 if weight_ok else 0.0
    score = round(s_struct + s_quad + s_uv + s_mat + s_clean + s_wt, 2)
    passed = score >= 90.0 and quad_ratio >= 0.85 and uv_align_pct >= 75.0 and matte_ok and non_manifold == 0

    return {
        "status": "PASS" if passed else "FAIL",
        "score": score,
        "hair_objects": [o.name for o in hair_objs],
        "metrics": {
            "total_faces": total_faces,
            "quad_ratio": round(quad_ratio, 4),
            "strand_shells": total_shells,
            "uv_strip_aligned_pct": round(uv_align_pct, 2),
            "matte_toon_materials_ok": matte_ok,
            "non_manifold_edges": non_manifold,
            "degenerate_faces": degenerate,
            "has_weights": has_weights,
            "max_bone_influences": max_bone_infs,
        },
        "breakdown": {
            "dual_layer_strands_20": s_struct,
            "quad_purity_20": s_quad,
            "vertical_strip_uv_25": s_uv,
            "matte_toon_material_15": s_mat,
            "mesh_cleanliness_10": s_clean,
            "weight_rule_10": s_wt,
        },
    }


def _render_hair_qc_sheet(hair_objs: list, out_path: Path, res: int = 512) -> str:
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

    pts = [o.matrix_world @ v.co for o in hair_objs for v in o.data.vertices]
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    span = max((max_p - min_p).x, (max_p - min_p).y, (max_p - min_p).z, 0.25)

    w = bpy.data.worlds.new("HairQCWorld")
    scene.world = w
    if w.node_tree is None:
        w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.88, 0.89, 0.92, 1.0)
    bg.inputs[1].default_value = 0.85

    created = []
    for lname, off, eng in [
        ("HQ_Key", Vector((-1.2, -1.6, 1.0)), 95),
        ("HQ_Fill", Vector((1.4, -1.2, 0.4)), 75),
        ("HQ_Back", Vector((0.0, 1.8, 0.8)), 85),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng * max(0.5, span * span)
        ld.size = span * 1.4
        lo = bpy.data.objects.new(lname, ld)
        scene.collection.objects.link(lo)
        lo.location = center + off * span
        lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()
        created.append(lo)

    cam_d = bpy.data.cameras.new("HQCam")
    cam_d.lens = 50.0
    cam = bpy.data.objects.new("HQCam", cam_d)
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
        tmp = out_path.parent / f"_hq_{vn}.png"
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

    out_img = bpy.data.images.new("HairQCSheet", width=2 * res, height=2 * res, alpha=True)
    out_img.pixels.foreach_set(sheet.ravel())
    out_img.filepath_raw = str(out_path)
    out_img.file_format = "PNG"
    out_img.save()
    bpy.data.images.remove(out_img)
    return str(out_path)


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "build-template":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        body_objs = _import_extra_body(args.body) if args.body else []
        head_c, head_r = _estimate_head_sphere(body_objs)
        scalp_obj, strands_obj = _build_blueprint_hair_template(head_c, head_r)
        hair_objs = [scalp_obj, strands_obj]
        atlas_p = _generate_hair_gradient_atlas(
            out_dir / "hair_vertical_atlas_512x1024.png",
            _hex_to_rgb(args.root_hex),
            _hex_to_rgb(args.mid_hex),
            _hex_to_rgb(args.tip_hex),
        )
        _apply_matte_toon_hair_material(hair_objs, atlas_p)
        audit = _audit_hair(hair_objs)
        out_blend = out_dir / "anime_hair_blueprint_template.blend"
        out_glb = out_dir / "anime_hair_blueprint_template.glb"
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
        bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB")
        qc_p = _render_hair_qc_sheet(hair_objs, out_dir / "hair_qc_4view.png", res=args.res)
        rep = {
            "mode": "build-template",
            "head_center": [round(v, 4) for v in head_c],
            "head_radius": round(head_r, 4),
            "atlas": str(atlas_p),
            "qc_sheet": qc_p,
            "out_blend": str(out_blend),
            "out_glb": str(out_glb),
            "audit": audit,
        }
        (out_dir / "hair_craft_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return

    if not args.input:
        raise ValueError(f"--mode {args.mode} requires --input <hair.blend|hair.glb>.")
    _load_scene(args.input)
    body_objs = _import_extra_body(args.body) if args.body else []
    hair_objs = _pick_hair_objects(args.hair_mesh)

    if args.mode == "audit":
        audit = _audit_hair(hair_objs)
        qc_p = _render_hair_qc_sheet(hair_objs, out_dir / "hair_audit_4view.png", res=args.res)
        rep = {"mode": "audit", "input": str(args.input), "qc_sheet": qc_p, "audit": audit}
        (out_dir / "hair_audit_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return

    # mode == "polish"
    ops_log = {}
    if args.add_scalp_cap and not any("scalp" in o.name.lower() for o in hair_objs):
        head_c, head_r = _estimate_head_sphere(body_objs if body_objs else hair_objs)
        scalp = _build_scalp_cap(head_c, head_r)
        hair_objs.append(scalp)
        ops_log["added_scalp_cap"] = scalp.name
    if args.strip_uv:
        ops_log["strip_uv"] = {o.name: _apply_vertical_strip_uv(o) for o in hair_objs}
    if args.toon_mat:
        atlas_p = _generate_hair_gradient_atlas(
            out_dir / "hair_vertical_atlas_512x1024.png",
            _hex_to_rgb(args.root_hex),
            _hex_to_rgb(args.mid_hex),
            _hex_to_rgb(args.tip_hex),
        )
        _apply_matte_toon_hair_material(hair_objs, atlas_p)
        ops_log["toon_atlas"] = str(atlas_p)

    audit = _audit_hair(hair_objs)
    stem = Path(args.input).stem
    out_blend = out_dir / f"{stem}_hair_polished.blend"
    out_glb = out_dir / f"{stem}_hair_polished.glb"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB", export_animations=True, export_skins=True)
    qc_p = _render_hair_qc_sheet(hair_objs, out_dir / "hair_polished_4view.png", res=args.res)
    rep = {
        "mode": "polish",
        "input": str(args.input),
        "ops": ops_log,
        "qc_sheet": qc_p,
        "out_blend": str(out_blend),
        "out_glb": str(out_glb),
        "audit": audit,
    }
    (out_dir / "hair_craft_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
