"""Spatial coverage and tangent-space checks for selected-to-active normal bakes."""

from math import degrees

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def trace_normal_probes(low, high, *, cage_extrusion, max_ray_distance, cage=None):
    """Trace deterministic face-center probes with explicit source layer IDs when present."""
    low_mesh, high_mesh = low.data, high.data
    high_tree = BVHTree.FromPolygons(
        [high.matrix_world @ vertex.co for vertex in high_mesh.vertices],
        [tuple(face.vertices) for face in high_mesh.polygons],
    )
    low_layers = low_mesh.attributes.get("bake_layer")
    high_layers = high_mesh.attributes.get("bake_layer")
    explicit_layers = low_layers is not None and high_layers is not None
    check_mode = "explicit_bake_layer" if explicit_layers else "normal_dot_heuristic"
    normal_matrix = low.matrix_world.to_3x3().inverted_safe().transposed()
    uv_layer = low_mesh.uv_layers.active.uv
    probes = []
    faces = low_mesh.polygons
    step = max(1, (len(faces) + 4095) // 4096)
    for index in range(0, len(faces), step):
        face = faces[index]
        center = low.matrix_world @ face.center
        normal = (normal_matrix @ face.normal).normalized()
        if cage is not None:
            origin = cage.matrix_world @ cage.data.polygons[index].center
            direction = (center - origin).normalized()
        else:
            origin = center + normal * max(1e-6, cage_extrusion)
            direction = -normal
        _, hit_normal, hit_index, _ = high_tree.ray_cast(
            origin, direction, max_ray_distance if max_ray_distance > 0 else 1e30,
        )
        wrong = False
        if hit_index is not None:
            if explicit_layers:
                wrong = low_layers.data[index].value != high_layers.data[hit_index].value
            else:
                wrong = normal.dot(hit_normal) < -0.25
        uv = sum((uv_layer[i].vector for i in face.loop_indices), Vector((0.0, 0.0))) / face.loop_total
        probes.append((tuple(uv), hit_index is not None, wrong))
    return probes, check_mode


def resolve_tangent_sign(image, *, confirmed, cross_layer_mis_hit_pct):
    """Invert the full encoded vector only after route and independent hit checks agree."""
    if not confirmed or cross_layer_mis_hit_pct > 0:
        return False
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    rgba = pixels.reshape((-1, 4))
    mask = (rgba[:, 3] > 0.5) & (rgba[:, 2] < 0.5)
    if np.any(mask):
        rgba[mask, :3] = 1.0 - rgba[mask, :3]
        image.pixels.foreach_set(pixels)
        image.update()
        return True
    return False


def diagnose_normal_bake(image, probes, *, check_mode, confirmed=False, expected_tangent_normal=None):
    width, height = image.size
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    rgba = pixels.reshape((height, width, 4))
    hits, valid_layer_hits, flipped, angles = 0, 0, 0, []
    wrong = sum(wrong_layer for _, _, wrong_layer in probes)
    total = max(1, len(probes))
    wrong_pct = 100.0 * wrong / total
    can_resolve = confirmed and wrong == 0
    expected = Vector(expected_tangent_normal).normalized() if expected_tangent_normal is not None else None
    for (u, v), hit, wrong_layer in probes:
        if not hit:
            continue
        sample = rgba[max(0, min(height - 1, int(v * height))), max(0, min(width - 1, int(u * width)))]
        if sample[3] <= 0.5:
            continue
        hits += 1
        if wrong_layer:
            continue
        valid_layer_hits += 1
        decoded = Vector(sample[:3] * 2.0 - 1.0)
        if decoded.z < 0:
            flipped += 1
            if can_resolve:
                decoded.negate()
        if expected is not None and decoded.length_squared > 1e-12:
            angles.append(degrees(decoded.angle(expected)))
    changed = resolve_tangent_sign(image, confirmed=confirmed, cross_layer_mis_hit_pct=wrong_pct)
    return {
        "probe_count": len(probes),
        "ray_valid_hit_pct": 100.0 * hits / total,
        "cross_layer_check_mode": check_mode,
        "cross_layer_mis_hit_pct": wrong_pct,
        "slope_decode_angle_error_deg": max(angles) if angles and not wrong else None,
        "tangent_sign_flipped_pct": 100.0 * flipped / max(1, valid_layer_hits),
        "tangent_sign_resolved": changed,
    }
