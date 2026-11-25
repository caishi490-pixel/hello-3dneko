# SPDX-License-Identifier: GPL-3.0-or-later
"""Occlusion-aware camera reprojection and PBR atlas assembly."""

import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from uv_hd_retex_uv import material_maps, mesh_arrays, pixels, raster_triangle, sample, sample_material, uv_coords


def load_pixels(path, data=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    if data:
        image.colorspace_settings.name = "Non-Color"
    result = pixels(image)
    bpy.data.images.remove(image)
    return result


def box_blur(rgb, mask, radius):
    values = np.concatenate((rgb * mask, mask), axis=2)
    for axis in (0, 1):
        p = [(0, 0)] * 3
        p[axis] = (radius, radius)
        padded = np.pad(values, p, mode="edge")
        total = np.cumsum(padded, axis=axis, dtype=np.float64)
        p[axis] = (1, 0)
        total = np.pad(total, p)
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(2 * radius + 1, None), slice(None, -(2 * radius + 1))
        values = ((total[tuple(a)] - total[tuple(b)]) / (2 * radius + 1)).astype(np.float32)
    return values[:, :, :3] / np.maximum(values[:, :, 3:], 0.0001)


def projection_views(root, names, cameras):
    result = []
    for name in names:
        camera = cameras["views"][name]
        if camera["type"] != "ORTHO":
            raise ValueError("Only calibrated orthographic cameras are supported")
        aligned = load_pixels(root / "aligned" / f"{name}.png")
        confidence = load_pixels(root / "aligned" / f"{name}_confidence.png", data=True)[:, :, :1]
        baseline = load_pixels(root / "before" / f"{name}.png")
        height, width = aligned.shape[:2]
        y, x = np.mgrid[:height, :width]
        uv = (np.stack([x.ravel(), y.ravel()], axis=1) + 0.5) / [width, height]
        baseline = sample(baseline, uv).reshape(height, width, 4)
        mask = (np.minimum(aligned[:, :, 3:], baseline[:, :, 3:]) > 0.85).astype(np.float32)
        ai_rgb, original_rgb = aligned[:, :, :3], baseline[:, :, :3]
        radius = max(2, round(width / 150))
        delta = ai_rgb - original_rgb
        large = box_blur(delta, mask, radius)
        fine = delta - large
        # Transfer local color detail while suppressing the lighting already present in the image.
        detail = np.clip(large * 0.55 + fine * 1.25, -0.21, 0.21)
        payload = np.concatenate((detail, confidence * mask), axis=2)
        direction = np.asarray(camera["camera_world"], dtype=np.float32)[:3, 2]
        result.append((name, np.asarray(camera["world_to_clip"], dtype=np.float32), direction, payload))
    return result


def high_reference(tree, coords, tris, uvs, materials, maps, points, normal, distance, fallback):
    hits = [tree.find_nearest(point, distance) for point in points.tolist()]
    normal = Vector(normal)
    accepted = np.array([hit[2] is not None and hit[1].dot(normal) > 0.25 for hit in hits], dtype=bool)
    selected = np.flatnonzero(accepted)
    if not len(selected):
        return fallback, 0
    ids = np.array([hits[i][2] for i in selected])
    nearest = np.array([tuple(hits[i][0]) for i in selected], dtype=np.float32)
    triangles = coords[tris[ids]]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    gap = nearest - triangles[:, 0]
    aa, bb, ab = np.sum(a * a, axis=1), np.sum(b * b, axis=1), np.sum(a * b, axis=1)
    pa, pb = np.sum(gap * a, axis=1), np.sum(gap * b, axis=1)
    det = np.maximum(aa * bb - ab ** 2, 1e-22)
    u = (bb * pa - ab * pb) / det
    v = (aa * pb - ab * pa) / det
    weights = np.clip(np.stack((1 - u - v, u, v), axis=1), 0, 1)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    uv = np.einsum("ij,ijk->ik", weights, uvs[ids])
    high = np.empty((len(selected), 6), dtype=np.float32)
    for material, channel_maps in enumerate(maps):
        mask = materials[ids] == material
        high[mask] = sample_material(channel_maps, uv[mask])
    original = fallback[selected]
    low_gold = original[:, 0] - original[:, 2]
    high_gold = high[:, 0] - high[:, 2]
    color_error = np.max(np.abs(high[:, :3] - original[:, :3]), axis=1)
    agreement = np.clip((0.31 - color_error) / 0.18, 0, 1)
    agreement[(low_gold > 0.09) != (high_gold > 0.09)] *= 0.15
    confidence = (agreement * 0.90)[:, None]
    fallback[selected] = original + (high - original) * confidence
    return fallback, int((confidence > 0.1).sum())


def atlas_bleed(atlas, occupied, radius):
    size = len(occupied)
    owner = np.where(occupied, np.arange(size * size, dtype=np.int32).reshape(size, size), -1)
    pad = np.pad(occupied, 1)
    inner = np.logical_and.reduce([pad[y:y + size, x:x + size] for y in range(3) for x in range(3)])
    y, x = np.where(occupied & ~inner)
    offsets = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1))
    filled = 0
    for _ in range(radius):
        sources = owner[y, x]
        frontier = []
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            inside = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
            yy, xx, origin = yy[inside], xx[inside], sources[inside]
            empty = owner[yy, xx] == -1
            yy, xx, origin = yy[empty], xx[empty], origin[empty]
            owner[yy, xx] = origin
            atlas[yy, xx] = atlas[origin // size, origin % size]
            frontier.append((yy, xx))
        y = np.concatenate([item[0] for item in frontier])
        x = np.concatenate([item[1] for item in frontier])
        filled += len(x)
    return filled


def save_texture(path, rgb, colorspace):
    size = rgb.shape[0]
    image = bpy.data.images.new(path.stem, width=size, height=size, alpha=False)
    image.colorspace_settings.name = colorspace
    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    image.pack()
    return image


def bind_material(obj, base, orm, source_uv, target_uv):
    material = bpy.data.materials.new("HD garment PBR")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = target_uv
    uv.location = (-720, 170)
    base_tex = nodes.new("ShaderNodeTexImage")
    base_tex.image = base
    base_tex.location = (-500, 320)
    orm_tex = nodes.new("ShaderNodeTexImage")
    orm_tex.image = orm
    orm_tex.location = (-500, -40)
    for tex in (base_tex, orm_tex):
        links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(base_tex.outputs["Color"], shader.inputs["Base Color"])
    split = nodes.new("ShaderNodeSeparateColor")
    split.location = (-240, -40)
    links.new(orm_tex.outputs["Color"], split.inputs[0])
    links.new(split.outputs["Green"], shader.inputs["Roughness"])
    links.new(split.outputs["Blue"], shader.inputs["Metallic"])
    shader.inputs["Specular IOR Level"].default_value = 0.4
    # Retain the supplied tangent-space normal with its original UV basis.
    old_materials = list(obj.data.materials)
    if len(old_materials) == 1:
        old_nodes = old_materials[0].node_tree.nodes
        old_shader = next(node for node in old_nodes if node.type == "BSDF_PRINCIPLED")
        normal_input = old_shader.inputs["Normal"]
        old_normal = normal_input.links[0].from_node if normal_input.is_linked else None
        if old_normal and old_normal.type == "NORMAL_MAP" and old_normal.inputs["Color"].is_linked:
            old_texture = old_normal.inputs["Color"].links[0].from_node
            if old_texture.type == "TEX_IMAGE" and old_texture.image:
                normal_uv = nodes.new("ShaderNodeUVMap")
                normal_uv.uv_map = source_uv
                normal_uv.location = (-720, -400)
                texture = nodes.new("ShaderNodeTexImage")
                texture.image = old_texture.image
                texture.location = (-500, -400)
                normal = nodes.new("ShaderNodeNormalMap")
                normal.uv_map = source_uv
                normal.inputs["Strength"].default_value = min(0.55, old_normal.inputs["Strength"].default_value)
                normal.location = (-240, -400)
                links.new(normal_uv.outputs["UV"], texture.inputs["Vector"])
                links.new(texture.outputs["Color"], normal.inputs["Color"])
                links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj.data.uv_layers[target_uv].active_render = True
    obj.data.uv_layers.active = obj.data.uv_layers[target_uv]


def bake_atlas(obj, reference, root, panels, cameras, audit, deliver):
    size = audit["size"]
    coords, indices, loops, materials = mesh_arrays(obj)
    source_uv = uv_coords(obj.data.uv_layers[audit["source_uv"]])[loops]
    target_uv = uv_coords(obj.data.uv_layers[audit["target_uv"]])[loops]
    maps = material_maps(obj)
    low_tree = BVHTree.FromPolygons(coords.tolist(), indices.tolist(), all_triangles=True)
    hc, ht, hl, hm = mesh_arrays(reference)
    high_tree = BVHTree.FromPolygons(hc.tolist(), ht.tolist(), all_triangles=True)
    high_uv = uv_coords(reference.data.uv_layers.active)[hl]
    high_maps = material_maps(reference)
    asset_size = float(np.ptp(coords, axis=0).max())
    if np.linalg.norm(hc.mean(axis=0) - coords.mean(axis=0)) > asset_size * 0.4:
        raise ValueError("High-poly reference and target are not in the same frame")
    atlas = np.empty((size, size, 6), dtype=np.float32)
    atlas[:] = [0.22, 0.16, 0.32, 1, 0.75, 0]
    occupied = np.zeros((size, size), dtype=bool)
    views = projection_views(root, panels, cameras)
    corset = "gold_filigree" in audit["regions"]
    count = {name: 0 for name, *_ in views}
    projected = high_count = occluded = 0
    vertices = coords[indices]
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    offset = asset_size * 0.000025
    directions = [Vector(direction) for _, _, direction, _ in views]
    for i, (tri, normal) in enumerate(zip(vertices, normals)):
        x, y, bary = raster_triangle(target_uv[i], size)
        fresh = ~occupied[y, x]
        x, y, bary = x[fresh], y[fresh], bary[fresh]
        if not len(x):
            continue
        world = bary @ tri
        old = sample_material(maps[materials[i]], bary @ source_uv[i])
        signal = np.zeros((len(x), 3), dtype=np.float32)
        weight = np.zeros((len(x), 1), dtype=np.float32)
        for (name, transform, direction, payload), vector in zip(views, directions):
            angle = float(normal @ direction)
            if angle <= 0.25:
                continue
            screen = (world @ transform[:2, :3].T + transform[:2, 3]) * 0.5 + 0.5
            indices_in_view = np.flatnonzero(((screen > 0) & (screen < 1)).all(axis=1))
            sampled = sample(payload, screen[indices_in_view])
            confident = sampled[:, 3] > 0.015
            indices_in_view, sampled = indices_in_view[confident], sampled[confident]
            if not len(indices_in_view):
                continue
            origins = (world[indices_in_view] + direction * offset).tolist()
            visible = np.fromiter((low_tree.ray_cast(p, vector, asset_size * 8)[2] is None for p in origins),
                                  dtype=bool, count=len(origins))
            occluded += int((~visible).sum())
            selected, sampled = indices_in_view[visible], sampled[visible]
            strength = sampled[:, 3:] * min(1, (angle - 0.25) / 0.20) * angle ** 4
            signal[selected] += sampled[:, :3] * strength
            weight[selected] += strength
            count[name] += len(selected)
        blend = np.clip(weight[:, 0] / 0.33, 0, 1)
        blend = blend ** 2 * (3 - 2 * blend)
        missing = blend < 0.99
        if missing.any():
            merged, hits = high_reference(high_tree, hc, ht, high_uv, hm, high_maps, world[missing],
                                           normal, asset_size * 0.009,
                                           old[missing].copy())
            old[missing] += (merged - old[missing]) * (1 - blend[missing, None])
            high_count += hits
        old[:, :3] = np.clip(old[:, :3] + signal * blend[:, None] / np.maximum(weight, 1e-6), 0, 1)
        if corset:
            chroma = old[:, 0] - old[:, 2]
            gold = np.clip((chroma - 0.035) / 0.125, 0, 1)
            gold = gold ** 2 * (3 - 2 * gold)
            old[:, 4] = (1 - gold) * np.clip(old[:, 4], 0.72, 0.88) + gold * np.clip(old[:, 4], 0.31, 0.46)
            old[:, 5] = (1 - gold) * np.clip(old[:, 5], 0, 0.08) + gold * np.clip(old[:, 5], 0.56, 0.83)
        atlas[y, x] = old
        occupied[y, x] = True
        projected += int((blend > 0.05).sum())
        if i % 4000 == 0:
            print(f"Reprojecting triangle {i}/{len(vertices)}", flush=True)
    total = int(occupied.sum())
    if not total or not projected:
        raise RuntimeError("Projection produced no covered texels")
    padded = atlas_bleed(atlas, occupied, audit["bleed_px"])
    base = save_texture(deliver / "base_color.png", atlas[:, :, :3], "sRGB")
    orm = save_texture(deliver / "orm.png", atlas[:, :, 3:], "Non-Color")
    bind_material(obj, base, orm, audit["source_uv"], audit["target_uv"])
    return {"atlas_size": size, "occupied_texels": total, "projected_texels": projected,
            "projected_pct": round(100 * projected / total, 2),
            "high_reference_blended_texels": high_count, "occlusion_rejections": occluded,
            "view_contributions": count, "bleed_px": audit["bleed_px"], "bleed_texels": padded,
            "color_mode": "view-weighted de-lit detail residual over original base color",
            "unseen_texels": total - projected, "filled_texels": total}
