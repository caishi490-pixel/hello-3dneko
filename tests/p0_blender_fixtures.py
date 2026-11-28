# SPDX-License-Identifier: GPL-3.0-or-later
"""Programmatically generated meshes and headless P0 contract assertions."""

import json
from math import cos, radians, sin, tau
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_scripts"))

from blender_runtime import configure_cycles, remove_export_junk
from gap_coverage_validator import check_shell_contract, sample_shells
from normal_bake_diagnostics import diagnose_normal_bake, trace_normal_probes
from topology_inspector import compute_symmetric_fidelity, inspect_mesh_object, load_scene_file
from uv_bake_toolkit import action_bake_pbr_maps
from uv_bake_validator import evaluate_uv_and_bake
from topology_toolkit import _autoremesh_part, _set_active_object, action_autoremesher_retopo
from pipeline.retopo_backends import AutoRemesherError
from pipeline.sandbox import ShortPathWorkspace


def mesh(name, vertices, faces):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def boxes(name, bounds, omitted_face=None):
    vertices, faces = [], []
    for index, ((x, y, z), (a, b, c)) in enumerate(bounds):
        offset = len(vertices)
        vertices.extend([(x, y, z), (a, y, z), (a, b, z), (x, b, z),
                         (x, y, c), (a, y, c), (a, b, c), (x, b, c)])
        sides = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        faces.extend(tuple(offset + i for i in face) for j, face in enumerate(sides)
                     if omitted_face is None or j != omitted_face[index])
    return mesh(name, vertices, faces)


def geometry(root):
    inventory = [((i * 0.11, 0, 0), (i * 0.11 + 0.10, 1, 0.04)) for i in range(51)]
    source = boxes("Strands", inventory)
    merged = boxes("Merged", [(inventory[0][0], inventory[-1][1])])
    shells = sample_shells([source])
    assert len(shells) == 51
    contract = {
        "separation_groups": [{"id": str(i), "shell_ids": [shell["shell_id"]]} for i, shell in enumerate(shells)],
        "gap_rays": [{"start": (i * 0.11 + 0.105, 0.5, -0.1), "end": (i * 0.11 + 0.105, 0.5, 0.1)}
                     for i in range(50)],
    }
    assert inspect_mesh_object(merged)["welded_non_manifold_edges"] == 0
    assert compute_symmetric_fidelity([merged], [source])["sym_max_dev_pct"] < 3.0
    assert check_shell_contract([source], shells, contract)["passed"]
    for phase in ("preprocess", "final"):
        result = check_shell_contract([merged], shells, contract, phase=phase)
        assert result["shells_out"] == 1 and not result["passed"], result
        assert "gap_bridged_or_merged" in result["issues"] and len(result["blocked_gap_rays"]) == 50
    assert check_shell_contract([merged], shells, {"single_shell_envelope": True})["passed"]
    outer = ((-1, -1, -1), (1, 1, 1))
    reference = boxes("Housing", [outer, ((-0.03, -0.03, -0.03), (0.03, 0.03, 0.03))])
    split = boxes("Split", [((-1, -1, -1), (-0.002, 1, 1)), ((0.002, -1, -1), (1, 1, 1))], (3, 5))
    before, after = inspect_mesh_object(reference), inspect_mesh_object(split)
    assert before["welded_shells_count"] == after["welded_shells_count"] == 2
    assert before["world_bbox"] == after["world_bbox"]
    result = check_shell_contract([split], sample_shells([reference]))
    assert result["issues"] == ["shell_coverage_missing:Housing:1"], result
    assert result["local_coverage"][1]["hit_rate"] == 0.0
    (root / "geometry.json").write_text(json.dumps(result), encoding="utf-8")


def autoremesher_shell_rollback(root):
    outer = ((-1, -1, -1), (1, 1, 1))
    reference = boxes("Housing", [outer, ((-0.03, -0.03, -0.03), (0.03, 0.03, 0.03))])
    split = boxes("Split", [((-1, -1, -1), (-0.002, 1, 1)), ((0.002, -1, -1), (1, 1, 1))], (3, 5))
    assert inspect_mesh_object(reference)["world_bbox"] == inspect_mesh_object(split)["world_bbox"]
    assert len(sample_shells([reference])) == len(sample_shells([split])) == 2
    original = reference.data
    original_objects = set(bpy.context.scene.objects)

    class DroppedGasketAdapter:
        def remesh(self, source, output, *, target_quads, timeout):
            obj_text = source.read_text(encoding="utf-8")
            assert "\nvt " not in obj_text and "\nvn " not in obj_text
            assert timeout <= 150
            _set_active_object(split)
            bpy.ops.wm.obj_export(filepath=str(output), forward_axis="Y", up_axis="Z",
                                  export_selected_objects=True, export_uv=False, export_materials=False)
            return 0.0

    with ShortPathWorkspace(root) as workspace:
        try:
            _autoremesh_part(reference, 200, DroppedGasketAdapter(), workspace, time.monotonic() + 150, 2.2)
        except AutoRemesherError as error:
            assert error.code == "autoremesher_shell_coverage" and "shell_coverage_missing:Housing:1" in str(error)
        else:
            raise AssertionError("AutoRemesher accepted a candidate with a missing gasket")
    assert reference.data == original and set(bpy.context.scene.objects) == original_objects
    assert len(reference.data.polygons) == 12

    replacement = root / "split.obj"
    _set_active_object(split)
    bpy.ops.wm.obj_export(filepath=str(replacement), forward_axis="Y", up_axis="Z",
                          export_selected_objects=True, export_uv=False, export_materials=False)
    bpy.data.objects.remove(split, do_unlink=True)
    plate = boxes("Plate", [((3, 0, 0), (5, 1, 0.6))])
    original_plate = plate.data.as_pointer()

    class MixedAdapter:
        def __init__(self):
            self.first = True

        def remesh(self, source, output, *, target_quads, timeout):
            shutil.copyfile(replacement if self.first else source, output)
            self.first = False
            return 0.0

    with patch("topology_toolkit.AutoRemesherAdapter.discover", return_value=MixedAdapter()):
        result = action_autoremesher_retopo({
            "target_faces": 400, "per_mesh_targets": {"Housing": 200, "Plate": 200},
            "auto_parts": ["Housing", "Plate"], "max_p95_dev_pct": 2.2,
        })
    assert [item["mesh"] for item in result["autoremesher"]] == ["Plate"], result
    assert result["local_rollbacks"][0]["reason"] == "autoremesher_shell_coverage", result
    assert [item["mesh"] for item in result["quadriflow"]] == ["Housing"], result
    assert plate.data.as_pointer() != original_plate and len(sample_shells([reference])) == 2


def triage(root):
    boxes("Armor_Plate", [((0, 0, 0), (3.5, 1, 0.53))])
    ribbon, _ = plane("Ribbon_Strip", x=5, width=4.0, height=0.28, z=0)
    ribbon.data.update()
    tassel = boxes("Dirty_Tassel", [((10 + i * 0.5, 0, 0), (10.2 + i * 0.5, 0.2, 1)) for i in range(3)])
    bm = bmesh.new()
    bm.from_mesh(tassel.data)
    for edge in list(bm.edges)[:18]:
        v0, v1 = edge.verts
        bm.faces.new((v0, v1, bm.verts.new((v0.co + v1.co) * 0.5 + Vector((0.02, 0.03, 0.04)))))
    bm.to_mesh(tassel.data)
    bm.free()
    save(root / "mixed.blend")


def save(path):
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def multipart_small_clasp(root):
    for name, segments, sides, radius, tube, center in (
        ("Main", 50, 50, 1.0, 0.32, 0.0),
        ("Clasp", 25, 10, 0.10, 0.042, 1.65),
    ):
        vertices, faces = [], []
        for i in range(segments):
            a = tau * i / segments
            for j in range(sides):
                b = tau * j / sides
                vertices.append((center + (radius + tube * cos(b)) * cos(a),
                                 (radius + tube * cos(b)) * sin(a), tube * sin(b)))
                v0 = i * sides + j
                v1 = (i + 1) % segments * sides + j
                v2 = (i + 1) % segments * sides + (j + 1) % sides
                v3 = i * sides + (j + 1) % sides
                faces.extend(((v0, v1, v2), (v0, v2, v3)))
        mesh(name, vertices, faces)
    save(root / "multipart.blend")


def pymeshlab_seam_source(root):
    steps = 12
    vertices = [(i / steps - 1, j / steps, 0.0)
                for i in range(2 * steps + 1) for j in range(steps + 1)]
    faces = []
    for i in range(2 * steps):
        for j in range(steps):
            a = i * (steps + 1) + j
            b = a + steps + 1
            faces.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    obj = mesh("SeamedFabric", vertices, faces)
    data = obj.data
    for name in ("Left", "Right"):
        data.materials.append(bpy.data.materials.new(name))
    primary = data.uv_layers.new(name="Primary")
    detail = data.uv_layers.new(name="Detail")
    for face in data.polygons:
        right = face.center.x > 0
        face.material_index = int(right)
        for loop_index, vertex in zip(face.loop_indices, face.vertices):
            x, y, _ = vertices[vertex]
            u = (0.55 if right else 0.45) + 0.4 * x
            v = 0.1 + 0.8 * y
            primary.uv[loop_index].vector = (u, v)
            detail.uv[loop_index].vector = (v, u)
    for edge in data.edges:
        edge.use_seam = all(vertices[index][0] == 0 for index in edge.vertices)
    save(root / "seam.blend")


def pymeshlab_seam_verify(root):
    def current_mesh():
        return next(item.data for item in bpy.context.scene.objects if item.type == "MESH")

    def corner_uv(data, tolerance=0.0):
        assert len(data.uv_layers) == 2
        assert [mat.name for mat in data.materials] == ["Left", "Right"]
        result = {}
        sides = [set(), set()]
        for face in data.polygons:
            slot = face.material_index
            for loop_index, vertex in zip(face.loop_indices, face.vertices):
                point = tuple(round(value, 6) for value in data.vertices[vertex].co)
                assert point[0] >= 0 if slot else point[0] <= 0
                if point[0] == 0:
                    sides[slot].add(point[1])
                key = (point, slot)
                coords = tuple(value for layer in data.uv_layers for value in layer.uv[loop_index].vector)
                if key in result:
                    assert max(abs(a - b) for a, b in zip(coords, result[key])) <= tolerance
                result[key] = coords
        assert len(sides[0]) == 13 and sides[0] == sides[1]
        return result

    load_scene_file(root / "seam.blend")
    original = corner_uv(current_mesh())
    report = {}
    for fmt, tolerance in (("blend", 0.0), ("glb", 1e-6)):
        load_scene_file(root / "retopo" / "deliverables" / f"seam_retopo.{fmt}")
        data = current_mesh()
        assert len(data.polygons) == 360
        if fmt == "blend":
            assert sum(edge.use_seam for edge in data.edges) == 12
        result = corner_uv(data, tolerance)
        error = max(abs(a - b) for key, values in result.items() for a, b in zip(values, original[key]))
        assert error <= tolerance, (fmt, error)
        report[fmt] = {"max_corner_uv_error": error, "seam_vertices_each_side": 13}
    (root / "seam_verification.json").write_text(json.dumps(report), encoding="utf-8")


def pymeshlab_nonmanifold_source(root):
    mesh("Nonmanifold", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)],
         [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    save(root / "nonmanifold.blend")


def exchange_source(root):
    obj = mesh("FaceCornerSeam", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 1, 2), (0, 2, 3)])
    uv = obj.data.uv_layers.new().uv
    for corner, coordinates in zip(uv, [(0, 0), (0.4, 0), (0.4, 0.4), (0.6, 0.6), (1, 1), (0.6, 1)]):
        corner.vector = coordinates
    for edge in obj.data.edges:
        edge.use_seam = set(edge.vertices) == {0, 2}
    for index in range(2):
        obj.data.materials.append(bpy.data.materials.new(f"Slot_{index}"))
        obj.data.polygons[index].material_index = index
    save(root / "seam.blend")


def exchange_verify(root):
    load_scene_file(root / "seam.blend")
    obj = bpy.context.scene.objects["FaceCornerSeam"]
    assert len(obj.data.vertices) == 4
    uv = [tuple(loop.vector) for loop in obj.data.uv_layers.active.uv]
    assert np.allclose(uv, [(0, 0), (0.4, 0), (0.4, 0.4), (0.6, 0.6), (1, 1), (0.6, 1)])
    assert [set(edge.vertices) for edge in obj.data.edges if edge.use_seam] == [{0, 2}]
    assert [slot.name for slot in obj.data.materials] == ["Slot_0", "Slot_1"]
    assert [poly.material_index for poly in obj.data.polygons] == [0, 1]


COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
          (0, 1, 1), (1, 0.35, 0), (0.3, 0.1, 0.75), (0.1, 0.6, 0.35), (0.8, 0.8, 0.8)]


def plane(name, x=0, width=1, height=1, z=0, *, tris=False, reverse=False, slope=0):
    points = [(x, 0, z), (x + width, 0, z + slope), (x + width, height, z + slope), (x, height, z)]
    faces = [(0, 1, 2), (0, 2, 3)] if tris else [(0, 1, 2, 3)]
    obj = mesh(name, points, [tuple(reversed(face)) for face in faces] if reverse else faces)
    uv = obj.data.uv_layers.new().uv
    for loop in obj.data.loops:
        uv[loop.index].vector = [(0, 0), (1, 0), (1, 1), (0, 1)][loop.vertex_index]
    return obj, uv


def paint(obj, color):
    mat = bpy.data.materials.new(obj.name)
    mat.use_nodes = True
    shader = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    obj.data.materials.append(mat)


def atlas_source(root):
    for i, color in enumerate(COLORS):
        obj, _ = plane(f"Part_{i:02d}", x=2 * i, z=0.001)
        paint(obj, color)
    save(root / "high.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for i in range(10):
        plane(f"Part_{i:02d}", x=2 * i, tris=True)
    save(root / "low.blend")


def atlas_verify(root):
    load_scene_file(root / "c.glb")
    objects = sorted((obj for obj in bpy.context.scene.objects if obj.type == "MESH"), key=lambda obj: obj.name)
    assert len(objects) == 10, [obj.name for obj in objects]
    images = set()
    for color, obj in zip(COLORS, objects, strict=True):
        assert len(obj.data.polygons) == 2 and all(poly.area > 0.4 for poly in obj.data.polygons)
        tree = obj.data.materials[0].node_tree
        shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
        image = shader.inputs["Base Color"].links[0].from_node.image
        images.add(image)
        width, height = image.size
        uv = obj.data.uv_layers.active.uv
        texels = []
        for poly in obj.data.polygons:
            u, v = sum(uv[i].vector.x for i in poly.loop_indices) / poly.loop_total, sum(
                uv[i].vector.y for i in poly.loop_indices
            ) / poly.loop_total
            i = 4 * (int(u * width) + int(v * height) * width)
            texels.append(tuple(image.pixels[i:i + 3]))
        expected = np.asarray([12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in color])
        for texel in texels:
            assert np.linalg.norm(np.asarray(texel) - expected) < 0.065, (obj.name, color, texel)
    assert len(images) == 1


def normal(root):
    target, _ = plane("Low")
    paint(target, (0, 0, 1))
    base, _ = plane("SourceLayer", z=0.0)
    paint(base, (0, 1, 0))
    back, _ = plane("WrongLayer", z=0.002, reverse=True)
    paint(back, (1, 0, 0))
    base.data.attributes.new("bake_layer", "INT", "FACE").data[0].value = 0
    back.data.attributes.new("bake_layer", "INT", "FACE").data[0].value = 1
    bpy.data.objects.remove(target, do_unlink=True)
    save(root / "double.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    target, _ = plane("Low")
    target.data.attributes.new("bake_layer", "INT", "FACE").data[0].value = 0
    settings = {"texture_size": 32, "margin": 2, "bake_samples": 1, "cage_extrusion": 0.005,
                "confirmed_tangent_sign": True}
    result = action_bake_pbr_maps(settings, root / "double.blend", root / "double")["baked_pbr"]["normal_diagnostics"]
    assert result["cross_layer_mis_hit_pct"] == 100.0 and result["ray_valid_hit_pct"] == 100.0, result
    assert result["cross_layer_check_mode"] == "explicit_bake_layer"
    assert not result["tangent_sign_resolved"] and result["tangent_sign_flipped_pct"] == 0
    diffuse = bpy.data.images.load(str(root / "double" / "atlas_baked_diffuse.png"))
    center_pixel = (16 * 32 + 16) * 4
    assert diffuse.pixels[center_pixel] > 0.9 and diffuse.pixels[center_pixel + 1] < 0.1
    bpy.ops.wm.read_factory_settings(use_empty=True)
    high, _ = plane("Slope", z=0.001, slope=0.5)
    paint(high, (0, 0, 1))
    save(root / "slope.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    low, _ = plane("Low")
    angle = radians(26.565051177)
    expected = (-sin(angle), 0.0, cos(angle))
    result = action_bake_pbr_maps(
        {**settings, "cage_extrusion": 0.6, "confirmed_tangent_sign": False, "expected_tangent_normal": expected},
        root / "slope.blend", root / "slope",
    )["baked_pbr"]["normal_diagnostics"]
    assert result["slope_decode_angle_error_deg"] < 2.0, result
    image = bpy.data.images["atlas_baked_normal"]
    payload = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(payload)
    payload.reshape((-1, 4))[:, :3] = 1.0 - payload.reshape((-1, 4))[:, :3]
    image.pixels.foreach_set(payload)
    high, _ = plane("SlopeProbe", z=0.001, slope=0.5)
    probes, check_mode = trace_normal_probes(low, high, cage_extrusion=0.6, max_ray_distance=0)
    result = diagnose_normal_bake(image, probes, check_mode=check_mode, confirmed=True, expected_tangent_normal=expected)
    assert result["slope_decode_angle_error_deg"] < 2.0 and result["tangent_sign_flipped_pct"] == 100.0, result
    assert result["tangent_sign_resolved"]

    for mode, rate, limit, blocked in (
        ("explicit_bake_layer", 0.0, 3.5, False),
        ("explicit_bake_layer", 0.01, 3.5, True),
        ("normal_dot_heuristic", 0.48, 3.5, False),
        ("normal_dot_heuristic", 3.50, 3.5, False),
        ("normal_dot_heuristic", 3.51, 3.5, True),
        ("normal_dot_heuristic", 0.48, 0.4, True),
    ):
        diagnostics = {**result, "cross_layer_check_mode": mode, "cross_layer_mis_hit_pct": rate}
        verdict = evaluate_uv_and_bake({}, {"normal_diagnostics": diagnostics}, {},
                                       {"min_bake_coverage": 0.96, "max_heuristic_cross_layer_pct": limit})
        assert any(issue.startswith("BakeCrossLayer:") for issue in verdict["blocking_failures"]) == blocked, verdict


def unconfirmed_tangent_sign(root):
    image = bpy.data.images.new("NegativeNormalSamples", width=5, height=1, alpha=True, float_buffer=True)
    pixels = np.array([
        (0.5, 0.5, 0.25, 1.0), (0.5, 0.5, 0.75, 1.0),
        (0.5, 0.5, 0.25, 1.0), (0.5, 0.5, 0.25, 1.0), (0.5, 0.5, 0.25, 0.0),
    ], dtype=np.float32).ravel()
    image.pixels.foreach_set(pixels)
    probes = [(((i + 0.5) / 5, 0.5), i != 3, i == 2) for i in range(5)]
    for confirmed in (False, True):
        result = diagnose_normal_bake(image, probes, check_mode="explicit_bake_layer", confirmed=confirmed)
        assert result["tangent_sign_flipped_pct"] == 50.0 and not result["tangent_sign_resolved"], result
        unchanged = np.empty_like(pixels)
        image.pixels.foreach_get(unchanged)
        np.testing.assert_array_equal(unchanged, pixels)


def cpu(root):
    from types import SimpleNamespace

    class Preferences:
        compute_device_type = "NONE"
        supported = "CUDA"

        def refresh_devices(self):
            devices = [SimpleNamespace(type="CPU", use=True)]
            if self.compute_device_type == self.supported:
                devices.append(SimpleNamespace(type=self.supported, use=False))
            self.devices = devices

    prefs = Preferences()
    scene = bpy.context.scene
    assert configure_cycles(scene, prefs) == "CUDA" and scene.cycles.device == "GPU"
    prefs.supported = ""
    assert configure_cycles(scene, prefs) == "CPU" and scene.cycles.device == "CPU"
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2)
    assert remove_export_junk() == ["Icosphere"]
    assert not scene.objects
    bpy.ops.object.camera_add(location=(0, 0, 4))
    scene.camera = bpy.context.object
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.cycles.samples = 1
    assert "FINISHED" in bpy.ops.render.render()


if __name__ == "__main__":
    mode, output = sys.argv[sys.argv.index("--") + 1 :]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    globals()[mode](Path(output))
