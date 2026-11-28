# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender worker for fixed-camera garment photography and UV reprojection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))


VIEWS = ("front", "back", "left", "right", "detail")
OBLIQUES = ("front_high", "back_high", "left_low", "right_low")
LINING = ("front_rim", "back_rim", "left_hem", "right_hem")


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def matrix_rows(matrix):
    return [list(row) for row in matrix]


def import_meshes(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]


def open_asset(path):
    if path.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    elif path.suffix.lower() == ".glb":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        meshes = import_meshes(path)
    else:
        raise ValueError("Expected a .blend or .glb asset")
    if not meshes:
        raise ValueError("The asset has no renderable mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        if obj.modifiers:
            raise ValueError("Apply mesh modifiers before UV reprojection")
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    if not obj.data.uv_layers.active:
        raise ValueError("Source material needs a UV map")
    return obj


def studio(obj, resolution):
    scene = bpy.context.scene
    for other in scene.objects:
        if other != obj:
            other.hide_render = True
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 24
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new("Retexture studio")
    scene.world.use_nodes = True
    world = scene.world.node_tree
    world.nodes.clear()
    background = world.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.65, 0.65, 0.65, 1)
    background.inputs["Strength"].default_value = 0.45
    world.links.new(background.outputs[0], world.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    if scene.compositing_node_group:
        scene.compositing_node_group = None
    bounds = np.array([tuple(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box])
    center = Vector((bounds.min(axis=0) + bounds.max(axis=0)) / 2)
    size = float(np.ptp(bounds, axis=0).max())
    for axis in ((1, 1, 0.65), (-1, 1, 0.65), (1, -1, 0.65), (-1, -1, 0.65), (0, 0, 1)):
        light = bpy.data.lights.new("Retexture softbox", "AREA")
        light.energy = 200 * size ** 2
        light.specular_factor = 0.26
        light.use_shadow = False
        light.shape = "DISK"
        light.size = size * 1.8
        node = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(node)
        node.location = center + Vector(axis) * (1.7 * size)
        node.rotation_euler = (center - node.location).to_track_quat("-Z", "Y").to_euler()
    return scene, center, size


def render_views(obj, config, root, previous=None):
    folder = root / ("before" if previous is None else "after")
    folder.mkdir(exist_ok=True)
    scene, center, size = studio(obj, config["resolution"])
    lens = bpy.data.cameras.new("Retexture camera")
    camera = bpy.data.objects.new(lens.name, lens)
    scene.collection.objects.link(camera)
    scene.camera = camera
    direction = Vector(config.get("front", (0, -1, 0))).normalized()
    left = direction.cross(Vector((0, 0, 1)))
    if left.length < 1e-6:
        raise ValueError("Front direction must have a horizontal component")
    left.normalize()
    up = Vector((0, 0, 1))
    lens.type = "ORTHO"
    lens.clip_start = size * 0.005
    lens.clip_end = size * 20
    views = {}
    detail_offset = config.get("detail_offset", (0, 0))
    orientations = dict(zip(VIEWS, (direction, -direction, left, -left, direction)))
    if config.get("obliques"):
        orientations.update(zip(OBLIQUES, (
            (direction + left * 0.68 + up * 0.78).normalized(),
            (-direction - left * 0.68 + up * 0.78).normalized(),
            (left - direction * 0.62 - up * 0.7).normalized(),
            (-left + direction * 0.62 - up * 0.7).normalized(),
        )))
    if config.get("lining"):
        orientations.update(zip(LINING, (
            (direction - left * 0.65 + up * 1.5).normalized(),
            (-direction + left * 0.65 + up * 1.5).normalized(),
            (left + direction * 0.62 - up * 1.3).normalized(),
            (-left - direction * 0.62 - up * 1.3).normalized(),
        )))
    for name in (previous["views"] if previous else orientations):
        if previous is None:
            outward = orientations[name]
            target = center.copy()
            lens.ortho_scale = size * 1.13
            if name == "detail":
                target += (left * detail_offset[0] + Vector((0, 0, detail_offset[1]))) * size
                lens.ortho_scale *= config["detail_scale"]
            camera.location = target + outward * (3 * size)
            camera.rotation_euler = (-outward).to_track_quat("-Z", "Y").to_euler()
        else:
            metadata = previous["views"][name]
            camera.matrix_world = Matrix(metadata["camera_world"])
            lens.ortho_scale = metadata["ortho_scale"]
            lens.lens = metadata["lens_mm"]
            lens.clip_start, lens.clip_end = metadata["clip"]
        bpy.context.view_layer.update()
        projection = camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(), x=scene.render.resolution_x, y=scene.render.resolution_y,
        )
        width = scene.render.resolution_x
        views[name] = {
            "camera_world": matrix_rows(camera.matrix_world),
            "world_to_camera": matrix_rows(camera.matrix_world.inverted()),
            "projection": matrix_rows(projection),
            "world_to_clip": matrix_rows(projection @ camera.matrix_world.inverted()),
            "ortho_scale": lens.ortho_scale,
            "lens_mm": lens.lens,
            "clip": [lens.clip_start, lens.clip_end],
            "resolution": [width, scene.render.resolution_y],
            "type": lens.type,
        }
        scene.render.filepath = str(folder / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    if previous is None:
        write_json(root / "cameras.json", {
            "schema": 1,
            "coordinates": "Blender world, row-major matrices, PNG origin top left",
            "object": obj.name,
            "object_world": matrix_rows(obj.matrix_world),
            "views": views,
        })
    else:
        for name, view in views.items():
            if not np.allclose(view["world_to_clip"], previous["views"][name]["world_to_clip"], atol=1e-6):
                raise ValueError(f"Camera reconstruction did not match {name}")
        front = previous["views"]["front"]
        camera.matrix_world = Matrix(front["camera_world"])
        lens.ortho_scale = front["ortho_scale"]
        lens.lens = front["lens_mm"]
        lens.clip_start, lens.clip_end = front["clip"]
        scene.render.filepath = "//preview.png"


def bake(obj, root, config):
    from uv_hd_retex_projection import bake_atlas

    cameras = json.loads((root / "cameras.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "uv_audit.json").read_text(encoding="utf-8"))
    if obj.name != cameras["object"] or not np.allclose(matrix_rows(obj.matrix_world), cameras["object_world"]):
        raise ValueError("UV object does not match the recorded camera calibration")
    source = Path(config["high"])
    if source.suffix.lower() != ".glb":
        raise ValueError("High reference must be a .glb")
    high = import_meshes(source)
    if len(high) != 1:
        raise ValueError("High reference must contain one mesh")
    bpy.context.view_layer.update()
    deliver = root / "deliverables"
    deliver.mkdir(exist_ok=True)
    result = bake_atlas(obj, high[0], root, config["views"], cameras, audit, deliver)
    bpy.data.objects.remove(high[0], do_unlink=True)
    render_views(obj, {"resolution": cameras["views"]["front"]["resolution"][0]}, root, cameras)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(deliver / "hd_retextured.blend"))
    bpy.ops.export_scene.gltf(filepath=str(deliver / "hd_retextured.glb"), export_format="GLB",
                             use_selection=True, export_cameras=False, export_lights=False)
    write_json(root / "projection_report.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["out"])
    action = config["action"]
    obj = open_asset(Path(config["input"]))
    if action == "photo":
        render_views(obj, config, root)
    elif action == "uv":
        from uv_hd_retex_uv import prepare_uv

        template = "corset.json" if config.get("corset") else "generic.json"
        profile = config.get("region_profile") or json.loads(
            (Path(__file__).resolve().parent.parent / "templates" / "uv_regions" / template).read_text(encoding="utf-8")
        )
        report = prepare_uv(obj, root, config["front"], config["size"], config["bleed"], profile)
        write_json(root / "uv_audit.json", report)
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(root / "work" / "uv.blend"))
    elif action == "bake":
        bake(obj, root, config)
    else:
        raise ValueError(f"Unknown action {action}")


if __name__ == "__main__":
    main()
