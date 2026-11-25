# SPDX-License-Identifier: GPL-3.0-or-later
"""UV generation, source material sampling and atlas rasterization."""

from collections import defaultdict
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from uv_bake_toolkit import _export_uv_layout_pil, _mark_garment_seams_bm
from uv_bake_validator import inspect_uv_geometry


HD_UV = "HD_Retexture"


def pixels(image):
    width, height = image.size
    result = np.empty(height * width * 4, dtype=np.float32)
    image.pixels.foreach_get(result)
    return result.reshape(height, width, 4)


def sample(image, uv, wrap=False):
    height, width = image.shape[:2]
    coords = uv * [width, height] - 0.5
    lo = np.floor(coords).astype(np.int32)
    hi = lo + 1
    weight = (coords - lo).astype(np.float32)
    if wrap:
        lo %= [width, height]
        hi %= [width, height]
    else:
        lo = np.clip(lo, 0, [width - 1, height - 1])
        hi = np.clip(hi, 0, [width - 1, height - 1])
    x, y = lo.T
    xx, yy = hi.T
    u, v = weight.T[:, :, None]
    return (image[y, x] * (1 - u) + image[y, xx] * u) * (1 - v) + (
        image[yy, x] * (1 - u) + image[yy, xx] * u
    ) * v


def material_maps(obj):
    """Read the Principled and packed glTF ORM graphs used by the asset inputs."""
    cache = {}
    result = []
    for material in obj.data.materials:
        shader = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        if shader is None:
            raise ValueError(f"Material {material.name} requires a Principled shader")
        maps = []
        for name in ("Base Color", "Roughness", "Metallic"):
            socket = shader.inputs[name]
            channel = None
            node = socket.links[0].from_node if socket.is_linked else None
            if node and node.type in {"SEPARATE_COLOR", "SEPRGB"}:
                channel = list(node.outputs).index(socket.links[0].from_socket)
                node = node.inputs[0].links[0].from_node if node.inputs[0].is_linked else None
            if node and node.type == "TEX_IMAGE" and node.image:
                if node.image.name not in cache:
                    cache[node.image.name] = pixels(node.image)
                image = cache[node.image.name]
                maps.append((image, channel, None))
            elif socket.is_linked:
                raise ValueError(f"Unsupported graph for {material.name} / {name}")
            else:
                value = socket.default_value
                default = np.array(list(value)[:3] if name == "Base Color" else [value], dtype=np.float32)
                if name == "Base Color":
                    default = np.where(default > 0.0031308, 1.055 * default ** (1 / 2.4) - 0.055, 12.92 * default)
                maps.append((None, None, default))
        result.append(maps)
    if not result:
        raise ValueError(f"Mesh {obj.name} has no material")
    return result


def sample_material(maps, uv):
    values = []
    for image, channel, default in maps:
        if image is None:
            values.append(np.broadcast_to(default, (len(uv), len(default))))
        else:
            rgba = sample(image, uv, wrap=True)
            values.append(rgba[:, :3] if channel is None else rgba[:, channel:channel + 1])
    base, rough, metal = values
    ao = np.ones_like(rough[:, :1])
    return np.concatenate((base, ao, rough[:, :1], metal[:, :1]), axis=1)


def tri_area(coords):
    return np.linalg.norm(np.cross(coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0]), axis=-1) / 2


def mesh_arrays(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    triangles = mesh.loop_triangles
    vertex_ids = np.empty(len(triangles) * 3, dtype=np.int32)
    loop_ids = np.empty_like(vertex_ids)
    materials = np.empty(len(triangles), dtype=np.int32)
    triangles.foreach_get("vertices", vertex_ids)
    triangles.foreach_get("loops", loop_ids)
    triangles.foreach_get("material_index", materials)
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    world = np.asarray(obj.matrix_world, dtype=np.float32)
    coords = coords.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    return coords, vertex_ids.reshape(-1, 3), loop_ids.reshape(-1, 3), materials


def uv_coords(layer):
    result = np.empty(len(layer.uv) * 2, dtype=np.float32)
    layer.uv.foreach_get("vector", result)
    return result.reshape(-1, 2)


def semantic_regions(obj, front, corset):
    coords, vertex_ids, loops, mat_ids = mesh_arrays(obj)
    triangles = coords[vertex_ids]
    centers = triangles.mean(axis=1)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    outward = np.asarray(front) / np.linalg.norm(front)
    labels = np.full(len(triangles), "other", dtype="<U24")
    labels[normal @ outward > 0.45] = "front_panel"
    if corset:
        uv = uv_coords(obj.data.uv_layers.active)[loops].mean(axis=1)
        colors = np.empty((len(uv), 3), dtype=np.float32)
        for material, maps in enumerate(material_maps(obj)):
            mask = mat_ids == material
            colors[mask] = sample_material(maps, uv[mask])[:, :3]
        gold = colors[:, 0] - colors[:, 2] > 0.045
        height = (centers[:, 2] - coords[:, 2].min()) / np.ptp(coords[:, 2])
        across = np.cross(outward, [0, 0, 1])
        widths = coords @ across
        position = ((centers @ across) - widths.min()) / np.ptp(widths)
        front_facing = normal @ outward > 0.28
        forward = coords @ outward
        outer_front = centers @ outward > (forward.min() + forward.max()) / 2
        labels[(height > 0.60) & (height < 0.93) & (abs(position - 0.51) < 0.13)
               & front_facing & outer_front] = "lacing"
        labels[(height > 0.24) & (height < 0.76) & (position < 0.47)] = "bow_and_ribbon"
        labels[height <= 0.24] = "tassel"
        labels[gold & (height >= 0.59)] = "gold_filigree"
        labels[gold & (height < 0.59)] = "bell_and_beads"
    return labels, tri_area(triangles), loops


def island_weights(obj, labels, loops, areas):
    mesh = obj.data
    coords = uv_coords(mesh.uv_layers.active)
    parents = np.arange(len(coords))

    def root(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    corners = {}
    for loop in mesh.loops:
        i = loop.index
        key = (loop.vertex_index, round(float(coords[i, 0]), 6), round(float(coords[i, 1]), 6))
        if key in corners:
            parents[root(i)] = root(corners[key])
        else:
            corners[key] = i
    for polygon in mesh.polygons:
        first = polygon.loop_start
        for i in range(first + 1, first + polygon.loop_total):
            parents[root(i)] = root(first)
    islands = defaultdict(list)
    for i in range(len(coords)):
        islands[root(i)].append(i)
    weights = {"front_panel": 1.18, "lacing": 1.3, "gold_filigree": 1.25,
               "bow_and_ribbon": 1.27, "bell_and_beads": 1.32, "tassel": 1.3}
    surface = defaultdict(float)
    importance = defaultdict(float)
    for corners, label, area in zip(loops, labels, areas):
        index = root(corners[0])
        surface[index] += area
        importance[index] += area * weights.get(label, 1)
    for index, corners in islands.items():
        if surface[index] > 0:
            points = coords[corners]
            center = points.mean(axis=0)
            coords[corners] = center + (points - center) * (importance[index] / surface[index])
    mesh.uv_layers.active.uv.foreach_set("vector", coords.ravel())


def pack_uv(margin, size):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.pack_islands(shape_method="CONCAVE", margin_method="FRACTION", margin=(margin + 0.5) / size)
    bpy.ops.object.mode_set(mode="OBJECT")


def separate_overlap_faces(obj, conflicts):
    """Preserve each larger chart where possible, isolating the smaller overlapping faces."""
    mesh = obj.data
    coords = uv_coords(mesh.uv_layers.active)
    detached = set()
    for first, second in sorted(conflicts):
        if first not in detached and second not in detached:
            detached.add(min((first, second), key=lambda i: (mesh.polygons[i].area, i)))
    for index, face_id in enumerate(sorted(detached)):
        face = mesh.polygons[face_id]
        loops = list(face.loop_indices)
        xyz = np.array([tuple(obj.matrix_world @ mesh.vertices[mesh.loops[i].vertex_index].co) for i in loops])
        axis = xyz[1] - xyz[0]
        axis /= max(np.linalg.norm(axis), 1e-12)
        across = np.cross(np.array(tuple(face.normal)), axis)
        across /= max(np.linalg.norm(across), 1e-12)
        chart = (xyz - xyz[0]) @ np.stack((axis, across)).T
        a, b = chart[1:-1] - chart[0], chart[2:] - chart[0]
        signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if not ((signed > 1e-10).all() or (signed < -1e-10).all()):
            angles = np.arange(len(loops)) * 2 * np.pi / len(loops)
            chart = np.stack((np.cos(angles), np.sin(angles)), axis=1)
            chart *= math.sqrt(face.area / max(0.5 * len(loops) * math.sin(2 * np.pi / len(loops)), 1e-12))
        coords[loops] = chart * 0.2 + [2 + index % 32, 2 + index // 32]
    mesh.uv_layers.active.uv.foreach_set("vector", coords.ravel())
    return len(detached)


def prepare_uv(obj, root, front, size, margin, corset):
    mesh = obj.data
    if HD_UV in mesh.uv_layers:
        raise ValueError(f"Input already contains {HD_UV}")
    source = mesh.uv_layers.active.name
    labels, areas, loops = semantic_regions(obj, front, corset)
    mesh.uv_layers.active = mesh.uv_layers.new(name=HD_UV)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if np.allclose(front, [0, -1, 0]):
        _mark_garment_seams_bm(bm, seam_angle_deg=52)
    else:
        old_uv = bm.loops.layers.uv.get(source)
        for edge in bm.edges:
            links = list(edge.link_loops)
            edge.seam = bool(len(links) == 2 and any(
                (a[old_uv].uv - b[old_uv].uv).length > 1e-5
                for a, b in ((links[0], links[1].link_loop_next), (links[1], links[0].link_loop_next))
                if a.vert == b.vert
            ))
    bm.to_mesh(mesh)
    bm.free()
    candidates = []
    for mode in ("side_and_crease", "smart_75", "smart_60"):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if mode == "side_and_crease":
            bpy.ops.uv.unwrap(method="ANGLE_BASED")
        else:
            bpy.ops.uv.smart_project(angle_limit=math.radians(int(mode[-2:])), correct_aspect=False)
        bpy.ops.uv.average_islands_scale()
        bpy.ops.object.mode_set(mode="OBJECT")
        island_weights(obj, labels, loops, areas)
        pack_uv(margin, size)
        conflicts = set()
        metrics = inspect_uv_geometry(obj, overlap_faces=conflicts)
        if mode.startswith("smart") and metrics["uv_overlap_area_pct"] < 2.0 and conflicts:
            repaired = 0
            initial = metrics["uv_overlap_area_pct"]
            for _ in range(3):
                repaired += separate_overlap_faces(obj, conflicts)
                pack_uv(margin, size)
                conflicts.clear()
                metrics = inspect_uv_geometry(obj, overlap_faces=conflicts)
                if not conflicts:
                    break
            metrics["detached_overlap_faces"] = repaired
            metrics["overlap_before_repair_pct"] = initial
        valid = (metrics["uv_overlap_area_pct"] < 0.5 and not metrics["uv_oob_loops"]
                 and metrics["uv_zero_area_tris"] <= 5)
        print("UV candidate", mode, metrics, flush=True)
        candidates.append((mode, metrics, valid, uv_coords(mesh.uv_layers.active)))
        if valid and metrics["uv_p95_stretch"] <= 2.4 and metrics["uv_overlap_area_pct"] <= 0.05:
            break
    valid = [candidate for candidate in candidates if candidate[2]]
    if not valid:
        raise RuntimeError("UV overlap/degeneracy gate failed for every unwrap")
    mode, metrics, _, coords = min(valid, key=lambda c: (
        c[1]["uv_overlap_area_pct"] > 0.05,
        c[1]["uv_p95_stretch"] + (c[0] != "side_and_crease") * 0.25,
        -c[1]["uv_packing_coverage_pct"],
    ))
    mesh.uv_layers.active.uv.foreach_set("vector", coords.ravel())
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.seams_from_islands()
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh.uv_layers[source].active_render = True
    uv_tri = coords[loops]
    a = uv_tri[:, 1] - uv_tri[:, 0]
    b = uv_tri[:, 2] - uv_tri[:, 0]
    uv_areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    regions = {}
    for name in sorted(set(labels)):
        mask = labels == name
        regions[name] = {
            "atlas_pixels_estimate": round(float(uv_areas[mask].sum() * size ** 2)),
            "occupied_uv_pct": round(float(100 * uv_areas[mask].sum() / uv_areas.sum()), 3),
            "surface_area_pct": round(float(100 * areas[mask].sum() / areas.sum()), 3),
        }
    _export_uv_layout_pil(obj, root / "uv_layout.png", res=2048)
    return {"source_uv": source, "target_uv": HD_UV, "size": size, "bleed_px": margin,
            "route": mode, "metrics": metrics, "regions": regions,
            "region_method": "source albedo and normalized geometry; estimated semantic areas" if corset else "normal orientation",
            "candidates": {name: report for name, report, _, _ in candidates}}


def raster_triangle(uv, size):
    coords = uv.astype(np.float64) * size - 0.5
    a, b, c = coords
    low = np.maximum(np.ceil(coords.min(axis=0)).astype(int), 0)
    high = np.minimum(np.floor(coords.max(axis=0)).astype(int), size - 1)
    if np.any(high < low):
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty((0, 3), dtype=np.float32)
    yy, xx = np.mgrid[low[1]:high[1] + 1, low[0]:high[0] + 1]
    ab, ac = b - a, c - a
    det = ab[0] * ac[1] - ab[1] * ac[0]
    if abs(det) < 1e-12:
        return xx.ravel()[:0], yy.ravel()[:0], np.empty((0, 3), dtype=np.float32)
    u = ((xx - a[0]) * ac[1] - (yy - a[1]) * ac[0]) / det
    v = (ab[0] * (yy - a[1]) - ab[1] * (xx - a[0])) / det
    w = 1 - u - v
    keep = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7)
    return xx[keep], yy[keep], np.stack((w[keep], u[keep], v[keep]), axis=1).astype(np.float32)
