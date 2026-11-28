"""Autonomous UV Seam Unwrapping, Layout Export, and Multi-Map PBR+AO Cycles Baking Toolkit."""

import argparse
import heapq
import json
import math
import sys
from collections import deque
from contextlib import contextmanager
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from topology_toolkit import (  # noqa: E402
    _dilate_diffuse_rgb_keep_alpha,
    _get_primary_meshes,
    _import_temp_reference,
    _load_scene,
    _set_active_object,
)
from blender_runtime import configure_cycles, junk_objects, remove_export_junk
from normal_bake_diagnostics import diagnose_normal_bake, trace_normal_probes
from pipeline.asset_ids import mesh_part_id


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
    for index, obj in enumerate(sorted(_get_primary_meshes(), key=lambda ob: ob.name)):
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

        layout_path = Path(uv_layout_dir) / f"part_{index:03d}_uv_layout.png"
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
    for mat in dict.fromkeys(ref_obj.data.materials):
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                return True
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf and bsdf.inputs["Base Color"].is_linked:
            return True
    values = {tuple(node.inputs["Base Color"].default_value)
              for mat in ref_obj.data.materials if mat and mat.use_nodes
              for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED"}
    return len(values) > 1 or bool(ref_obj.data.color_attributes)


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
    for mat in dict.fromkeys(ref_obj.data.materials):
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


def _pack_shared_atlas(meshes, tex_size, margin):
    """Keep each part's island layout, fitting it inside a disjoint padded atlas cell."""
    columns = math.ceil(math.sqrt(len(meshes)))
    rows = math.ceil(len(meshes) / columns)
    pitch = 1.0 / max(columns, rows)
    padding = (margin + 1) / tex_size
    if pitch <= 2 * padding:
        raise ValueError("Atlas resolution is too small for the part count and bake margin")
    for i, mesh in enumerate(meshes):
        uv = mesh.data.uv_layers.active.uv
        coordinates = np.empty(len(uv) * 2, dtype=np.float32)
        uv.foreach_get("vector", coordinates)
        points = coordinates.reshape((-1, 2))
        minimum = points.min(axis=0)
        span = np.ptp(points, axis=0)
        if max(span) <= 1e-9:
            raise ValueError(f"Degenerate UV bounds for {mesh.name}")
        points[:] = (points - minimum - span / 2) * ((pitch - 2 * padding) / max(span))
        points += ((i % columns + 0.5) * pitch, (i // columns + 0.5) * pitch)
        uv.foreach_set("vector", coordinates)


def _temporary_atlas_mesh(meshes):
    copies = []
    for index, obj in enumerate(meshes):
        clone = obj.copy()
        clone.data = obj.data.copy()
        clone.data.uv_layers.active.name = "__SharedBakeUV"
        part = clone.data.attributes.new("__atlas_part", "INT", "FACE")
        part.data.foreach_set("value", np.full(len(part.data), index, dtype=np.int32))
        bpy.context.scene.collection.objects.link(clone)
        copies.append(clone)
    _set_active_object(copies[0])
    for clone in copies:
        clone.select_set(True)
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    mapping = {}
    indices = {}
    for index, item in enumerate(joined.data.attributes["__atlas_part"].data):
        indices.setdefault(item.value, []).append(index)
    for index, name in enumerate(obj.name for obj in meshes):
        faces = indices.get(index, [])
        if not faces or faces[-1] - faces[0] + 1 != len(faces):
            raise RuntimeError(f"Invalid shared-atlas face range for {name}")
        mapping[mesh_part_id(name)] = [faces[0], faces[-1] + 1]
    return joined, mapping


@contextmanager
def _bake_emission(ref_obj, channel):
    backups = []
    try:
        for mat in dict.fromkeys(ref_obj.data.materials):
            if not mat or not mat.use_nodes:
                continue
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
            if not out or not bsdf:
                continue
            previous = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
            emit = nodes.new("ShaderNodeEmission")
            socket = bsdf.inputs[channel]
            if socket.is_linked:
                links.new(socket.links[0].from_socket, emit.inputs["Color"])
            elif channel == "Metallic":
                emit.inputs["Color"].default_value = (socket.default_value,) * 3 + (1.0,)
            else:
                emit.inputs["Color"].default_value = socket.default_value
            links.new(emit.outputs["Emission"], out.inputs["Surface"])
            backups.append((nodes, links, out, previous, emit))
        yield
    finally:
        for nodes, links, out, previous, emit in backups:
            nodes.remove(emit)
            if previous is not None:
                links.new(previous, out.inputs["Surface"])


def action_bake_pbr_maps(params, reference_glb, textures_out_dir):
    """Bake each channel once into a shared UV atlas and attach it to the original parts."""
    if not reference_glb or not Path(reference_glb).is_file():
        raise FileNotFoundError("Missing high-poly reference for bake")
    started = time.monotonic()
    size = int(params.get("texture_size", 2048))
    extrusion = float(params.get("cage_extrusion", 0.003))
    ray_distance = float(params.get("max_ray_distance", 0.0))
    samples = int(params.get("bake_samples", 16))
    margin = int(params.get("margin", 12))
    output = Path(textures_out_dir)
    output.mkdir(parents=True, exist_ok=True)

    meshes = sorted(_get_primary_meshes(), key=lambda obj: obj.name)
    if not meshes or any(not obj.data.uv_layers.active or not obj.data.polygons for obj in meshes):
        raise ValueError("Every part needs a nonempty mesh and active UV map before baking")
    for obj in meshes:
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    target = meshes[0]
    face_ranges = {mesh_part_id(target.name): [0, len(target.data.polygons)]}
    if len(meshes) > 1:
        _pack_shared_atlas(meshes, size, margin)
        target, face_ranges = _temporary_atlas_mesh(meshes)
    reference, imported = _import_temp_reference(reference_glb)
    scene = bpy.context.scene
    previous_engine = scene.render.engine
    cage = None
    paths = {}
    try:
        device = configure_cycles(scene)
        scene.cycles.samples = samples
        scene.cycles.use_denoising = False
        has_color = _reference_has_color_signal(reference)
        if params.get("use_cage"):
            cage = _create_displaced_cage(target, max(1e-6, extrusion))

        material = bpy.data.materials.new("SharedBakedPBR")
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
        target.data.materials.clear()
        target.data.materials.append(material)
        for face in target.data.polygons:
            face.material_index = 0

        def bake(role, bake_type, colorspace="Non-Color"):
            image = bpy.data.images.new(f"atlas_baked_{role}", width=size, height=size, alpha=True)
            image.colorspace_settings.name = colorspace
            node = nodes.new("ShaderNodeTexImage")
            node.image = image
            nodes.active = node
            _set_active_object(target)
            reference.select_set(True)
            result = bpy.ops.object.bake(
                type=bake_type, use_selected_to_active=True, use_clear=True,
                cage_extrusion=extrusion, max_ray_distance=ray_distance, margin=margin,
                margin_type="EXTEND", use_cage=cage is not None, cage_object=cage.name if cage else "",
                normal_space="TANGENT",
            )
            if "FINISHED" not in result:
                raise RuntimeError(f"Cycles {role} bake did not finish")
            path = output / f"atlas_baked_{role}.png"
            paths[role] = str(path)
            image.filepath_raw = str(path)
            image.file_format = "PNG"
            return node, image

        with _bake_emission(reference, "Base Color"):
            diffuse, diffuse_image = bake("diffuse", "EMIT", "sRGB")
        _dilate_diffuse_rgb_keep_alpha(diffuse_image, passes=min(6, margin))
        diffuse_image.save()

        normal, normal_image = bake("normal", "NORMAL")
        probes, check_mode = trace_normal_probes(target, reference, cage_extrusion=extrusion,
                                                 max_ray_distance=ray_distance, cage=cage)
        diagnostics = diagnose_normal_bake(
            normal_image, probes, check_mode=check_mode, confirmed=params.get("confirmed_tangent_sign", False),
            expected_tangent_normal=params.get("expected_tangent_normal"),
        )
        normal_image.save()

        rough, rough_image = bake("roughness", "ROUGHNESS")
        with _bake_emission(reference, "Metallic"):
            metal, metal_image = bake("metallic", "EMIT")
        ao_backups = _backup_and_wire_ao_emit(reference, ao_distance=float(params.get("ao_distance", 0.08)),
                                             ao_samples=samples)
        try:
            ao, ao_image = bake("ao", "EMIT")
        finally:
            _restore_materials_from_ao_emit(ao_backups)
        ao_image.save()
        paths["orm"] = str(output / "atlas_baked_orm.png")
        orm_image = _combine_full_orm_image("atlas_baked_orm", ao_image, rough_image, metal_image, paths["orm"])
        for node in (rough, metal, ao):
            nodes.remove(node)
        for image in (rough_image, metal_image, ao_image):
            bpy.data.images.remove(image)
        del paths["roughness"], paths["metallic"]

        orm = nodes.new("ShaderNodeTexImage")
        orm.image = orm_image
        separate = nodes.new("ShaderNodeSeparateColor")
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = float(params.get("normal_strength", 0.55))
        links.new(diffuse.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(orm.outputs["Color"], separate.inputs["Color"])
        links.new(separate.outputs["Green"], bsdf.inputs["Roughness"])
        links.new(separate.outputs["Blue"], bsdf.inputs["Metallic"])
        gltf_settings = bpy.data.node_groups.get("glTF Material Output")
        if gltf_settings is None or not any(item.name == "Occlusion" for item in gltf_settings.interface.items_tree):
            gltf_settings = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
            gltf_settings.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
        export_settings = nodes.new("ShaderNodeGroup")
        export_settings.node_tree = gltf_settings
        links.new(separate.outputs["Red"], export_settings.inputs["Occlusion"])
        links.new(normal.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        for obj in meshes:
            obj.data.materials.clear()
            obj.data.materials.append(material)
            for face in obj.data.polygons:
                face.material_index = 0
        (output / "normal_diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
        return {"baked_pbr": {"has_reference_color": has_color, "meshes": {obj.name: paths for obj in meshes},
                              "face_ranges": face_ranges, "cycles_device": device, "bake_calls": 5,
                              "normal_diagnostics": diagnostics, "elapsed_seconds": time.monotonic() - started}}
    finally:
        scene.render.engine = previous_engine
        temporary = [*imported]
        if cage is not None:
            temporary.append(cage)
        if len(meshes) > 1:
            temporary.append(target)
        for obj in temporary:
            bpy.data.objects.remove(obj, do_unlink=True)


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

    remove_export_junk()
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
        export_tangents=True,
    )

    report = {
        "status": "executed",
        "output_blend": str(out_blend),
        "output_glb": str(out_glb),
        "steps": step_logs,
        "junk_objects": [o.name for o in junk_objects(bpy.context.scene.objects)],
    }
    Path(plan["execution_log"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("UV & Bake toolkit finished executing plan ->", out_glb)


if __name__ == "__main__":
    main()
