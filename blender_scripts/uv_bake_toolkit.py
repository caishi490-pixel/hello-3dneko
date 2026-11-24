"""Autonomous UV Seam Unwrapping, Layout Export, and Multi-Map PBR+AO Cycles Baking Toolkit."""

import argparse
import heapq
import json
import math
import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from topology_toolkit import (  # noqa: E402
    _backup_and_wire_metallic_emit,
    _combine_orm_image,
    _dilate_diffuse_rgb_keep_alpha,
    _get_primary_meshes,
    _import_temp_reference,
    _load_scene,
    _prepare_ref_color_baking_shader,
    _restore_materials_from_emit,
    _sanitize_tangent_normal_image,
    _set_active_object,
)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    return p.parse_args(argv)


def _find_shortest_edge_path(v_start, v_end, prefer_back_y=None):
    """Dijkstra shortest edge sequence connecting two vertices, biased toward back +Y."""
    dist = {v_start.index: 0.0}
    prev = {}
    pq = [(0.0, v_start.index)]
    vert_map = {v_start.index: v_start}

    while pq:
        cur_d, cur_idx = heapq.heappop(pq)
        if cur_idx == v_end.index:
            break
        if cur_d > dist.get(cur_idx, 1e18):
            continue
        cur_v = vert_map[cur_idx]
        for e in cur_v.link_edges:
            ov = e.other_vert(cur_v)
            vert_map[ov.index] = ov
            elen = max(1e-6, e.calc_length())
            weight = elen
            if prefer_back_y is not None:
                mid_y = 0.5 * (cur_v.co.y + ov.co.y)
                if mid_y < prefer_back_y:
                    weight *= 3.5
            nd = cur_d + weight
            if nd < dist.get(ov.index, 1e18):
                dist[ov.index] = nd
                prev[ov.index] = (cur_idx, e)
                heapq.heappush(pq, (nd, ov.index))

    path_edges = []
    cur = v_end.index
    while cur in prev:
        p_idx, edge = prev[cur]
        path_edges.append(edge)
        cur = p_idx
    return path_edges


def _mark_garment_seams_bm(bm, seam_angle_deg=42.0, cut_closed_shells=True):
    """Mark dihedral crease seams plus back-plane & meridian cuts on closed/toroidal shells."""
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for e in bm.edges:
        e.seam = False

    angle_thresh = math.radians(float(seam_angle_deg))
    crease_count = 0
    for e in bm.edges:
        if e.is_boundary:
            continue
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) >= angle_thresh:
            e.seam = True
            crease_count += 1

    # Detect connected face islands bounded only by seam edges; cut closed seamless tubes/rings
    cut_edges = 0
    if cut_closed_shells and bm.faces:
        visited = set()
        for start_f in bm.faces:
            if start_f.index in visited:
                continue
            shell = []
            q = deque([start_f])
            visited.add(start_f.index)
            has_seam_or_boundary = False
            while q:
                cf = q.popleft()
                shell.append(cf)
                for e in cf.edges:
                    if e.seam or e.is_boundary:
                        has_seam_or_boundary = True
                        continue
                    for nf in e.link_faces:
                        if nf.index not in visited:
                            visited.add(nf.index)
                            q.append(nf)

            if len(shell) >= 6:
                shell_verts = {v for f in shell for v in f.verts}
                pts = [v.co for v in shell_verts]
                cx = sum(p.x for p in pts) / len(pts)
                cy = sum(p.y for p in pts) / len(pts)
                spans = (
                    max(p.x for p in pts) - min(p.x for p in pts),
                    max(p.y for p in pts) - min(p.y for p in pts),
                    max(p.z for p in pts) - min(p.z for p in pts),
                )
                axis = 2 if spans[2] >= spans[0] and spans[2] >= spans[1] else (0 if spans[0] >= spans[1] else 1)

                # Back symmetry seam cut (x ~= cx on +Y back side) slices waist belts/tubes flat cleanly
                shell_edges = {e for f in shell for e in f.edges}
                for e in shell_edges:
                    if e.seam or e.is_boundary:
                        continue
                    v0, v1 = e.verts
                    if min(v0.co.y, v1.co.y) >= cy:
                        if (v0.co.x - cx) * (v1.co.x - cx) <= 0.0 and abs(v0.co.x - v1.co.x) < 0.15:
                            e.seam = True
                            cut_edges += 1

                if not has_seam_or_boundary:
                    # Also connect extreme poles along back meridian so even spheres/closed pods unfold
                    ordered = sorted(shell_verts, key=lambda v: v.co[axis])
                    v_min, v_max = ordered[0], ordered[-1]
                    for edge in _find_shortest_edge_path(v_min, v_max, prefer_back_y=cy):
                        if not edge.seam:
                            edge.seam = True
                            cut_edges += 1

    total_seams = sum(1 for e in bm.edges if e.seam)
    return {"crease_seams": crease_count, "meridian_cut_seams": cut_edges, "total_seams": total_seams}


def _export_uv_layout_pil(obj, out_png, res=1024):
    """Render a crisp 2D UV wireframe inspection PNG directly via NumPy + Blender Images."""
    me = obj.data
    uv_layer = me.uv_layers.active
    if not uv_layer:
        return None

    out_p = Path(out_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros((res, res, 4), dtype=np.float32)
    canvas[..., 0] = 0.07
    canvas[..., 1] = 0.08
    canvas[..., 2] = 0.11
    canvas[..., 3] = 1.0

    # Subtle 4x4 grid
    for gi in range(1, 4):
        pos = int(gi * res / 4)
        canvas[:, pos, :3] = (0.14, 0.16, 0.21)
        canvas[pos, :, :3] = (0.14, 0.16, 0.21)

    uv_data = uv_layer.data
    seam_vert_pairs = {(e.vertices[0], e.vertices[1]) for e in me.edges if e.use_seam}
    seam_vert_pairs |= {(b, a) for a, b in seam_vert_pairs}

    def _draw_segment(x0, y0, x1, y1, color, thick=False):
        dx = x1 - x0
        dy = y1 - y0
        steps = max(2, int(max(abs(dx), abs(dy)) * (res - 1)) + 1)
        ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        xs = np.clip(((x0 + dx * ts) * (res - 1)).astype(np.int32), 0, res - 1)
        ys = np.clip(((y0 + dy * ts) * (res - 1)).astype(np.int32), 0, res - 1)
        canvas[ys, xs, :3] = color
        if thick:
            canvas[np.clip(ys + 1, 0, res - 1), xs, :3] = color
            canvas[ys, np.clip(xs + 1, 0, res - 1), :3] = color

    # Mark UV triangle interior bounding shading lightly on centroids + crisp wireframe edges
    for poly in me.polygons:
        lidxs = list(poly.loop_indices)
        n = len(lidxs)
        for i in range(n):
            li0 = lidxs[i]
            li1 = lidxs[(i + 1) % n]
            u0, v0 = uv_data[li0].uv.x, uv_data[li0].uv.y
            u1, v1 = uv_data[li1].uv.x, uv_data[li1].uv.y
            vi0 = me.loops[li0].vertex_index
            vi1 = me.loops[li1].vertex_index
            if (vi0, vi1) in seam_vert_pairs:
                _draw_segment(u0, v0, u1, v1, (0.98, 0.68, 0.24), thick=True)
            else:
                _draw_segment(u0, v0, u1, v1, (0.45, 0.78, 0.98), thick=False)

    img = bpy.data.images.new("__UVLayoutTmp", width=res, height=res, alpha=False)
    img.colorspace_settings.name = "sRGB"
    img.pixels[:] = canvas.reshape(-1).tolist()
    img.filepath_raw = str(out_p)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return str(out_p)


def _rescue_collapsed_seam_faces(obj, island_margin=0.004):
    """Detect closed-ring faces crushed to zero/extreme stretch by unwrap and smart-project only them."""
    me = obj.data
    uv_data = me.uv_layers.active.data
    ratios = []
    poly_stats = []
    signed_sum = 0.0
    for p in me.polygons:
        lidxs = list(p.loop_indices)
        pts = [(uv_data[li].uv.x, uv_data[li].uv.y) for li in lidxs]
        s_area = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            s_area += x1 * y2 - x2 * y1
        s_area *= 0.5
        signed_sum += s_area
        uarea = abs(s_area)
        a3 = max(1e-12, p.area)
        r = uarea / a3
        poly_stats.append((p.index, s_area, uarea, a3, r))
        if uarea > 1e-10 and p.area > 1e-9:
            ratios.append(r)

    if not ratios:
        return 0
    dom_sign = 1.0 if signed_sum >= 0.0 else -1.0
    med = float(np.median(np.array(ratios, dtype=np.float64)))
    bad_idxs = {
        idx
        for idx, sa, ua, a3, r in poly_stats
        if (ua <= 1e-10 and a3 > 1e-9) or (sa * dom_sign <= 0.0) or r < med * 0.14 or r > med * 9.0
    }
    if not bad_idxs:
        return 0

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for idx in bad_idxs:
        me.polygons[idx].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(82.0), island_margin=island_margin)
    bpy.ops.object.mode_set(mode="OBJECT")
    return len(bad_idxs)


def action_prepare_seams_and_unwrap(params, uv_layout_dir):
    """Unwrap low-poly mesh via explicit seam cutting, conformal/angle unwrap, or smart project."""
    mode = str(params.get("mode", "seam_unwrap"))
    unwrap_method = str(params.get("unwrap_method", "ANGLE_BASED"))
    seam_angle_deg = float(params.get("seam_angle_deg", 42.0))
    smart_angle_deg = float(params.get("smart_angle_deg", 84.0))
    island_margin = float(params.get("island_margin", 0.003))
    average_scale = bool(params.get("average_scale", True))
    cut_closed = bool(params.get("cut_closed_shells", True))

    out = []
    for obj in _get_primary_meshes():
        _set_active_object(obj)
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")

        seam_stats = {"crease_seams": 0, "meridian_cut_seams": 0, "total_seams": 0, "rescued_faces": 0}
        if mode == "seam_unwrap":
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            seam_stats.update(
                _mark_garment_seams_bm(bm, seam_angle_deg=seam_angle_deg, cut_closed_shells=cut_closed)
            )
            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.unwrap(method=unwrap_method, margin=island_margin)
            bpy.ops.object.mode_set(mode="OBJECT")

            rescued = _rescue_collapsed_seam_faces(obj, island_margin=island_margin)
            seam_stats["rescued_faces"] = rescued

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            if average_scale:
                bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(margin=island_margin, rotate=True, scale=True)
            bpy.ops.object.mode_set(mode="OBJECT")
        elif mode == "hybrid_conformal":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(
                angle_limit=math.radians(smart_angle_deg), island_margin=island_margin
            )
            bpy.ops.uv.seams_from_islands()
            bpy.ops.uv.unwrap(method="CONFORMAL", margin=island_margin)
            bpy.ops.object.mode_set(mode="OBJECT")

            rescued = _rescue_collapsed_seam_faces(obj, island_margin=island_margin)
            seam_stats["rescued_faces"] = rescued
            seam_stats["total_seams"] = sum(1 for e in obj.data.edges if e.use_seam)

            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            if average_scale:
                bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(margin=island_margin, rotate=True, scale=True)
            bpy.ops.object.mode_set(mode="OBJECT")
        elif mode == "smart_project":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(
                angle_limit=math.radians(smart_angle_deg), island_margin=island_margin
            )
            if average_scale:
                bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(margin=island_margin, rotate=True, scale=True)
            bpy.ops.object.mode_set(mode="OBJECT")
        elif mode == "repack_only":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            if average_scale:
                bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(margin=island_margin, rotate=True, scale=True)
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            raise ValueError(f"Unknown UV unwrap mode: {mode}")

        layout_path = Path(uv_layout_dir) / f"{obj.name}_uv_layout.png"
        saved_layout = _export_uv_layout_pil(obj, layout_path, res=1024)
        out.append(
            {
                "mesh": obj.name,
                "mode": mode,
                "unwrap_method": unwrap_method if mode == "seam_unwrap" else mode,
                **seam_stats,
                "uv_layout_png": saved_layout,
            }
        )
    return {"unwrapped": out}


def _create_displaced_cage(obj, offset):
    """Create a smooth pushed cage copy from low-poly mesh for clean SELECTED_TO_ACTIVE baking."""
    cage_mesh = obj.data.copy()
    cage_obj = bpy.data.objects.new(f"__BakeCage_{obj.name}", cage_mesh)
    bpy.context.scene.collection.objects.link(cage_obj)
    cage_obj.matrix_world = obj.matrix_world.copy()
    mod = cage_obj.modifiers.new("CagePush", "DISPLACE")
    mod.direction = "NORMAL"
    mod.strength = float(offset)
    mod.mid_level = 0.0
    _set_active_object(cage_obj)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return cage_obj


def _reference_has_color_signal(ref_obj):
    """Detect whether high-poly reference carries non-uniform Base Color textures or vertex colors."""
    for mat in ref_obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                return True
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf and bsdf.inputs["Base Color"].is_linked:
            return True
    return bool(ref_obj.data.color_attributes)


def _combine_full_orm_image(name, ao_img, rough_img, met_img, out_path):
    """Pack AO (Red), Roughness (Green), and Metallic (Blue) into glTF-standard ORM texture."""
    w, h = rough_img.size
    a_px = np.array(ao_img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    r_px = np.array(rough_img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    m_px = np.array(met_img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    orm = np.ones((h, w, 4), dtype=np.float32)
    orm[..., 0] = a_px[..., 0]
    orm[..., 1] = r_px[..., 1]
    orm[..., 2] = m_px[..., 0]
    orm[..., 3] = 1.0
    orm_img = bpy.data.images.new(name, width=w, height=h, alpha=False)
    orm_img.colorspace_settings.name = "Non-Color"
    orm_img.pixels[:] = orm.reshape(-1).tolist()
    orm_img.filepath_raw = str(out_path)
    orm_img.file_format = "PNG"
    orm_img.save()
    return orm_img


def _backup_and_wire_ao_emit(ref_obj, ao_distance=0.08, ao_samples=16):
    """Wire a distance-clamped ShaderNodeAmbientOcclusion via Emission so enclosed cloth interiors do not bake solid black."""
    backups = []
    for mat in ref_obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        out_node = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        if not out_node:
            out_node = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if not out_node:
            continue
        orig_surf = out_node.inputs["Surface"].links[0].from_socket if out_node.inputs["Surface"].is_linked else None
        ao_n = nodes.new("ShaderNodeAmbientOcclusion")
        ao_n.samples = int(ao_samples)
        ao_n.inputs["Distance"].default_value = float(ao_distance)
        em_n = nodes.new("ShaderNodeEmission")
        links.new(ao_n.outputs["AO"], em_n.inputs["Color"])
        links.new(em_n.outputs["Emission"], out_node.inputs["Surface"])
        backups.append((mat, out_node, orig_surf, ao_n, em_n))
    return backups


def _restore_materials_from_ao_emit(backups):
    for mat, out_node, orig_surf, ao_n, em_n in backups:
        if orig_surf is not None:
            mat.node_tree.links.new(orig_surf, out_node.inputs["Surface"])
        mat.node_tree.nodes.remove(em_n)
        mat.node_tree.nodes.remove(ao_n)


def action_bake_pbr_maps(params, reference_glb, textures_out_dir):
    """Cycles-bake Normal, AO, Base Color (if present), and Roughness/Metallic ORM cleanly."""
    if not reference_glb or not Path(reference_glb).is_file():
        raise FileNotFoundError(f"Missing high-poly reference for bake: {reference_glb}")

    tex_size = int(params.get("texture_size", 2048))
    cage_extrusion = float(params.get("cage_extrusion", 0.003))
    max_ray_distance = float(params.get("max_ray_distance", 0.018))
    ao_distance = float(params.get("ao_distance", 0.08))
    use_cage_mesh = bool(params.get("use_cage", False))
    normal_strength = float(params.get("normal_strength", 0.55))
    min_normal_b = float(params.get("min_normal_b", 0.68))
    samples = int(params.get("bake_samples", 16))
    margin_px = int(params.get("margin", 12))

    out_dir = Path(textures_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_meshes = _get_primary_meshes()
    ref_obj, imported = _import_temp_reference(reference_glb)
    sc = bpy.context.scene
    prev_engine = sc.render.engine
    created_cages = []

    try:
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.refresh_devices()
            if any(d.type != "CPU" for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                sc.cycles.device = "GPU"
        except Exception:
            sc.cycles.device = "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_denoising = False

        has_ref_color = _reference_has_color_signal(ref_obj)
        baked_info = {"has_reference_color": has_ref_color, "meshes": {}}

        def _bake_single_map(obj, cage_obj, mat, role, btype, pfilter, cspace, dst):
            img = bpy.data.images.new(f"{obj.name}_baked_{role}", width=tex_size, height=tex_size, alpha=True)
            img.colorspace_settings.name = cspace
            tnode = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tnode.image = img
            mat.node_tree.nodes.active = tnode
            tnode.select = True

            bpy.ops.object.select_all(action="DESELECT")
            ref_obj.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            sc.render.bake.use_pass_direct = False
            sc.render.bake.use_pass_indirect = False
            sc.render.bake.use_pass_color = True
            bake_kwargs = {
                "type": btype,
                "use_selected_to_active": True,
                "use_cage": cage_obj is not None,
                "cage_extrusion": cage_extrusion,
                "max_ray_distance": max_ray_distance,
                "margin": margin_px,
                "margin_type": "EXTEND",
                "use_clear": True,
                "normal_space": "TANGENT",
            }
            if cage_obj is not None:
                bake_kwargs["cage_object"] = cage_obj.name
            if pfilter:
                bake_kwargs["pass_filter"] = pfilter
            bpy.ops.object.bake(**bake_kwargs)
            img.filepath_raw = str(dst)
            img.file_format = "PNG"
            img.save()
            return tnode, img

        for obj in cand_meshes:
            _set_active_object(obj)
            if not obj.data.uv_layers:
                raise RuntimeError(f"Mesh '{obj.name}' has no active UV map before bake.")

            cage_obj = None
            if use_cage_mesh:
                cage_obj = _create_displaced_cage(obj, max(0.001, cage_extrusion))
                created_cages.append(cage_obj)
                _set_active_object(obj)

            obj.data.materials.clear()
            mat = bpy.data.materials.new(f"{obj.name}_UVBakeMat")
            mat.use_nodes = True
            bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Roughness"].default_value = 0.65
            obj.data.materials.append(mat)

            # 1. Metallic via Emission before touching reference shader
            met_backups = _backup_and_wire_metallic_emit(ref_obj)
            t_met, img_met = _bake_single_map(
                obj, cage_obj, mat, "metallic", "EMIT", set(), "Non-Color", out_dir / f"{obj.name}_baked_metallic.png"
            )
            _restore_materials_from_emit(met_backups)

            # 2. Roughness, AO, and Tangent Normal
            t_rough, img_rough = _bake_single_map(
                obj, cage_obj, mat, "roughness", "ROUGHNESS", set(), "Non-Color", out_dir / f"{obj.name}_baked_roughness.png"
            )
            ao_dst = out_dir / f"{obj.name}_baked_ao.png"
            ao_backups = _backup_and_wire_ao_emit(ref_obj, ao_distance=ao_distance, ao_samples=samples)
            t_ao, img_ao = _bake_single_map(obj, cage_obj, mat, "ao", "EMIT", set(), "Non-Color", ao_dst)
            _restore_materials_from_ao_emit(ao_backups)
            norm_dst = out_dir / f"{obj.name}_baked_normal.png"
            t_norm, img_norm = _bake_single_map(
                obj, cage_obj, mat, "normal", "NORMAL", set(), "Non-Color", norm_dst
            )
            _sanitize_tangent_normal_image(img_norm, min_b=min_normal_b)

            # 3. Base Color (with metallic=0 bypass so gold/metal surfaces preserve crisp albedo)
            diff_dst = out_dir / f"{obj.name}_baked_diffuse.png"
            _prepare_ref_color_baking_shader(ref_obj)
            t_diff, img_diff = _bake_single_map(
                obj, cage_obj, mat, "diffuse", "DIFFUSE", {"COLOR"}, "sRGB", diff_dst
            )
            _dilate_diffuse_rgb_keep_alpha(img_diff, passes=6)

            # Pack AO(R) + Roughness(G) + Metallic(B) into glTF-compliant ORM texture
            orm_dst = out_dir / f"{obj.name}_baked_orm.png"
            orm_img = _combine_full_orm_image(f"{obj.name}_baked_orm", img_ao, img_rough, img_met, orm_dst)

            mat.node_tree.nodes.remove(t_met)
            mat.node_tree.nodes.remove(t_rough)
            mat.node_tree.nodes.remove(t_ao)

            t_orm = mat.node_tree.nodes.new("ShaderNodeTexImage")
            t_orm.image = orm_img
            sep = mat.node_tree.nodes.new("ShaderNodeSeparateColor")

            links = mat.node_tree.links
            links.new(t_diff.outputs["Color"], bsdf.inputs["Base Color"])
            links.new(t_orm.outputs["Color"], sep.inputs["Color"])
            links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
            links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

            nmap = mat.node_tree.nodes.new("ShaderNodeNormalMap")
            nmap.inputs["Strength"].default_value = normal_strength
            links.new(t_norm.outputs["Color"], nmap.inputs["Color"])
            links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

            baked_info["meshes"][obj.name] = {
                "diffuse": str(diff_dst),
                "normal": str(norm_dst),
                "ao": str(ao_dst),
                "orm": str(orm_dst),
            }

        return {"baked_pbr": baked_info}
    finally:
        sc.render.engine = prev_engine
        for c in created_cages:
            if c.name in bpy.data.objects:
                mesh_ref = c.data
                bpy.data.objects.remove(c, do_unlink=True)
                if mesh_ref and mesh_ref.users == 0:
                    bpy.data.meshes.remove(mesh_ref)
        for o in imported:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)


ACTION_DISPATCH = {
    "prepare_seams_and_unwrap": lambda p, ctx: action_prepare_seams_and_unwrap(p, ctx["uv_layout_dir"]),
    "bake_pbr_maps": lambda p, ctx: action_bake_pbr_maps(p, ctx["reference_glb"], ctx["textures_dir"]),
}


def main():
    args = _parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
    _load_scene(plan["low_poly_input"])

    ctx = {
        "reference_glb": plan["high_poly_reference"],
        "textures_dir": plan["textures_dir"],
        "uv_layout_dir": plan["uv_layout_dir"],
    }
    step_logs = []
    for step in plan["actions"]:
        name = step["action"]
        params = step.get("params") or {}
        if name not in ACTION_DISPATCH:
            raise ValueError(f"Unknown uv-bake action: {name}")
        res = ACTION_DISPATCH[name](params, ctx)
        step_logs.append({"action": name, "params": params, "result": res})

    bpy.ops.file.pack_all()
    out_blend = Path(plan["output_blend"])
    out_glb = Path(plan["output_glb"])
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        export_animations=True,
        export_skins=True,
    )

    report = {
        "status": "executed",
        "output_blend": str(out_blend),
        "output_glb": str(out_glb),
        "steps": step_logs,
    }
    Path(plan["execution_log"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("UV & Bake toolkit finished executing plan ->", out_glb)


if __name__ == "__main__":
    main()
