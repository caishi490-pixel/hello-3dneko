# SPDX-License-Identifier: GPL-3.0-or-later
"""Native Blender reads, a coordinate-only write boundary, and isolated previews."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


INVARIANTS = ("topology_hash", "uv_hash", "material_hash", "object_relationship_hash")


class Unsupported(ValueError):
    """The mesh cannot enter the P1 native write path."""


@dataclass
class MeshData:
    V: np.ndarray
    F: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    loops: tuple[int, ...]
    uv_layers: dict[str, np.ndarray]
    vid: np.ndarray
    fid: np.ndarray
    triangles: np.ndarray
    triangle_faces: np.ndarray
    matrix_world: np.ndarray
    unit_scale: float
    hashes: dict


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _float_hash(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def _get_ids(mesh: bpy.types.Mesh, name: str, domain: str) -> np.ndarray:
    attribute = mesh.attributes.get(name)
    if attribute is None or (attribute.data_type, attribute.domain) != ("INT", domain):
        raise Unsupported(f"Missing integer {domain} attribute {name}")
    values = np.empty(len(attribute.data), dtype=np.int32)
    attribute.data.foreach_get("value", values)
    if len(values) == 0 or np.any(values <= 0) or len(np.unique(values)) != len(values):
        raise Unsupported(f"{name} must contain unique positive IDs")
    return values


def _get_floats(collection: object, key: str, width: int) -> np.ndarray:
    data = np.empty(len(collection) * width, dtype=np.float32)
    collection.foreach_get(key, data)
    return data.reshape(-1, width)


def _check_supported(obj: bpy.types.Object) -> None:
    mesh = obj.data
    if (obj.library or mesh.library or mesh.users - int(mesh.use_fake_user) != 1 or mesh.shape_keys
            or obj.modifiers or obj.find_armature() or obj.mode != "OBJECT"):
        raise Unsupported("Linked/shared mesh, shape keys, modifiers, armature binding or non-object mode")
    parent = obj.parent
    while parent is not None:
        if parent.type == "ARMATURE":
            raise Unsupported("Armature ancestor")
        parent = parent.parent


def mesh_object(name: str) -> bpy.types.Object:
    obj = bpy.context.scene.objects.get(name)
    if obj is None or obj.type != "MESH":
        raise Unsupported(f"Mesh object not found in the active scene: {name}")
    return obj


def read_mesh(obj: bpy.types.Object) -> MeshData:
    """Read raw mesh data and Blender's analysis triangulation without editing topology."""
    _check_supported(obj)
    mesh = obj.data
    V = _get_floats(mesh.vertices, "co", 3).astype(np.float64)
    vid = _get_ids(mesh, "gre_vid", "POINT")
    fid = _get_ids(mesh, "gre_fid", "FACE")
    faces = tuple(tuple(face.vertices) for face in mesh.polygons)
    edges = tuple(tuple(edge.vertices) for edge in mesh.edges)
    loops = tuple(loop.vertex_index for loop in mesh.loops)
    uv = {layer.name: _get_floats(layer.uv, "vector", 2) for layer in mesh.uv_layers}
    scene = bpy.context.scene
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    scale = float(scene.unit_settings.scale_length)
    if not np.isfinite(V).all() or not np.isfinite(matrix).all() or not np.isfinite(scale) or scale <= 0:
        raise Unsupported("Nonfinite positions, transform or unit scale")

    vertex_ids = vid.tolist()
    topology = {
        "vertices": vertex_ids,
        "edges": [[vertex_ids[index] for index in edge] for edge in edges],
        "faces": [[face_id, [vertex_ids[index] for index in face]] for face_id, face in zip(fid.tolist(), faces)],
        "loops": [vertex_ids[index] for index in loops],
    }
    materials = {
        "slots": [(slot.link, slot.material.name if slot.material else None) for slot in obj.material_slots],
        "faces": [face.material_index for face in mesh.polygons],
    }
    relations = [
        {"name": item.name, "type": item.type, "parent": item.parent.name if item.parent else None,
         "parent_type": item.parent_type, "parent_bone": item.parent_bone,
         "collections": sorted(collection.name for collection in item.users_collection),
         "matrix_world": [list(row) for row in item.matrix_world]}
        for item in sorted(scene.objects, key=lambda item: item.name)
    ]
    layers = [(layer.name, layer.active_render, _float_hash(uv[layer.name])) for layer in mesh.uv_layers]
    # This serialization matches P0's annotation source_hash and recorded fixture fingerprints.
    hashes = {
        "object": obj.name, "vertex_count": len(vid), "edge_count": len(edges),
        "face_count": len(fid), "loop_count": len(loops), "uv_layers": list(uv),
        "unit_system": scene.unit_settings.system, "unit_scale": scale, "matrix_world": matrix.tolist(),
        "topology_hash": _digest(topology),
        "uv_hash": _digest({"active_index": mesh.uv_layers.active_index, "layers": layers}),
        "material_hash": _digest(materials), "object_relationship_hash": _digest(relations),
        "position_hash": _float_hash(V),
    }
    hashes["source_hash"] = _digest(hashes)
    mesh.calc_loop_triangles()
    triangles = np.array([tuple(triangle.vertices) for triangle in mesh.loop_triangles], dtype=np.intp).reshape(-1, 3)
    triangle_faces = np.array([triangle.polygon_index for triangle in mesh.loop_triangles], dtype=np.intp)
    return MeshData(V, faces, edges, loops, uv, vid, fid, triangles, triangle_faces, matrix, scale, hashes)


def write_positions(obj: bpy.types.Object, vid_to_pos: dict[int, np.ndarray], expected_source_hash: str) -> None:
    before = read_mesh(obj)
    if before.hashes["source_hash"] != expected_source_hash:
        raise ValueError("Stale source hash before applying gre_vid patch")
    index = {vid: offset for offset, vid in enumerate(before.vid.tolist())}
    patch = []
    for vid, position in vid_to_pos.items():
        stored = np.asarray(position, dtype=np.float32)
        if type(vid) is not int or vid not in index or stored.shape != (3,) or not np.isfinite(stored).all():
            raise ValueError(f"Invalid coordinate patch for gre_vid {vid}")
        patch.append((index[vid], stored))
    mesh = obj.data
    for offset, stored in patch:
        mesh.vertices[offset].co = stored
    mesh.update()
    bpy.context.view_layer.update()


def input_path() -> Path:
    if not bpy.data.filepath:
        raise Unsupported("Open the input .blend when launching Blender")
    return Path(bpy.data.filepath).resolve()


def open_blend(path: Path) -> None:
    if bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False) != {"FINISHED"}:
        raise OSError(f"Could not open {path}")


def save_copy(path: Path, source: Path, destination: Path) -> None:
    if source.parent != destination.parent and any(
        item.startswith("//") for item in bpy.utils.blend_paths(absolute=False, packed=False)
    ):
        raise Unsupported("Output in another directory would change relative external resource resolution")
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    status = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False, compress=True,
                                         relative_remap=False, copy=True)
    if status != {"FINISHED"}:
        raise OSError(f"Could not save Blender candidate {path}")


def qa_snapshot(obj: bpy.types.Object, data: MeshData):
    """Read protected mesh attributes and analysis corner indices for P2."""
    from gre.snapshot import MeshSnapshot

    mesh = obj.data
    # Blender exposes custom split normals as evaluated values. Their storage cannot be
    # verified through this adapter, so exclude such assets from this write path.
    if mesh.has_custom_normals:
        raise Unsupported("P2 cannot verify raw custom split normals")
    protected = []
    for attribute in sorted(mesh.attributes, key=lambda value: value.name):
        if attribute.name == "position":
            continue
        values = []
        if len(attribute.data):
            properties = [(prop.identifier, getattr(prop, "is_array", False)) for prop in attribute.data[0].bl_rna.properties
                          if prop.identifier != "rna_type"]
            if not properties:
                raise Unsupported(f"Cannot read protected mesh attribute {attribute.name}")
            for item in attribute.data:
                values.append([list(getattr(item, name)) if array else getattr(item, name)
                               for name, array in properties])
        protected.append((attribute.name, attribute.domain, attribute.data_type, values))
    extra = {
        "mesh_attributes": protected,
        "vertex_groups": [(group.name, group.lock_weight) for group in obj.vertex_groups],
        "vertex_weights": [[(group.group, group.weight) for group in vertex.groups] for vertex in mesh.vertices],
        "uv_aux": [(layer.name, [item.value for item in layer.pin]) for layer in mesh.uv_layers],
        "active_color": mesh.color_attributes.active_color_name,
        "render_color_index": mesh.color_attributes.render_color_index,
    }
    hashes = {**data.hashes, "protected_attribute_hash": _digest(extra)}
    loops = np.array([tuple(triangle.loops) for triangle in mesh.loop_triangles], dtype=np.intp).reshape(-1, 3)
    return MeshSnapshot(data.V, data.vid, data.triangles, loops, data.fid[data.triangle_faces],
                        data.uv_layers, hashes, data.unit_scale)


def render_previews(obj: bpy.types.Object, before: MeshData, after: MeshData,
                    reference: MeshData, folder: Path, *, qa: bool = False, fit_frame: bool = False) -> None:
    """Render only copies of the inspection object, at identical views for all images."""
    scene = bpy.data.scenes.new("GRE preview")
    mesh = obj.data.copy()
    preview = bpy.data.objects.new("Geometry", mesh)
    scene.collection.objects.link(preview)
    preview.matrix_world = obj.matrix_world
    camera_data = bpy.data.cameras.new("GRE fixed camera")
    camera = bpy.data.objects.new("GRE fixed camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    xyz = reference.V @ reference.matrix_world[:3, :3].T + reference.matrix_world[:3, 3]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = Vector((lo + hi) * 0.5)
    span = float(np.linalg.norm(hi - lo))
    camera.location = center + Vector((1.6, -2.6, 1.55)) * span
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span * 1.10
    camera_data.clip_end = max(100.0, span * 10)
    camera_data.clip_start = max(1e-6, span / 10000)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = 1120, 860
    if fit_frame:
        aspect = scene.render.resolution_x / scene.render.resolution_y
        frame = (xyz - np.array(center)) @ np.array(camera.rotation_euler.to_matrix())
        extent = np.max(np.abs(frame[:, :2]), axis=0)
        camera_data.ortho_scale = max(camera_data.ortho_scale, 2.24 * max(extent[0], aspect * extent[1]))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    shading = scene.display.shading
    shading.type = "SOLID"
    shading.light = "MATCAP"
    shading.studio_light = "basic_side.exr"
    shading.color_type = "SINGLE"
    shading.background_type = "WORLD"
    world = bpy.data.worlds.new("GRE preview background")
    world.color = (0.039, 0.047, 0.061)
    scene.world = world

    def render(name: str) -> None:
        scene.render.filepath = str(folder / f"{name}.png")
        if bpy.ops.render.render(scene=scene.name, write_still=True) != {"FINISHED"}:
            raise OSError(f"Preview render failed: {name}")

    for name, data in (("before", before), ("after", after)):
        mesh.vertices.foreach_set("co", data.V.astype(np.float32).ravel())
        mesh.update()
        render(name)

    magnitude = np.linalg.norm((after.V - before.V) * before.unit_scale, axis=1)
    ratio = magnitude / max(float(np.max(magnitude)), np.finfo(float).tiny)
    keys = np.array([[0.035, 0.058, 0.11], [0.025, 0.54, 0.82], [0.98, 0.73, 0.06], [0.95, 0.045, 0.015]])
    color = np.ones((len(magnitude), 4), dtype=np.float32)
    for channel in range(3):
        color[:, channel] = np.interp(ratio, (0, 0.30, 0.65, 1), keys[:, channel])
    attribute = mesh.color_attributes.new(name="GRE move magnitude", type="FLOAT_COLOR", domain="POINT")
    attribute.data.foreach_set("color", color.ravel())
    mesh.color_attributes.active_color = attribute
    shading.light = "FLAT"
    shading.color_type = "VERTEX"
    render("displacement")
    if not qa:
        return

    shading.color_type = "OBJECT"
    preview.color = (0.52, 0.57, 0.62, 1)
    lines = bpy.data.objects.new("Wire inspection", mesh.copy())
    lines.matrix_world = preview.matrix_world
    scene.collection.objects.link(lines)
    lines.color = (0.023, 0.035, 0.055, 1)
    wire = lines.modifiers.new("Native polygon edges", "WIREFRAME")
    wire.thickness = span * 0.00045
    wire.offset = 0.5
    wire.use_boundary = True
    render("wireframe")
    lines.hide_render = True

    if mesh.uv_layers:
        checker = bpy.data.materials.new("GRE UV checker")
        if checker.node_tree is None:
            checker.use_nodes = True
        tree = checker.node_tree
        tree.nodes.clear()
        uv = tree.nodes.new("ShaderNodeUVMap")
        uv.uv_map = mesh.uv_layers.active.name
        pattern = tree.nodes.new("ShaderNodeTexChecker")
        pattern.inputs["Scale"].default_value = 32
        pattern.inputs["Color1"].default_value = (0.72, 0.76, 0.82, 1)
        pattern.inputs["Color2"].default_value = (0.028, 0.13, 0.22, 1)
        light = tree.nodes.new("ShaderNodeEmission")
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(uv.outputs["UV"], pattern.inputs["Vector"])
        tree.links.new(pattern.outputs["Color"], light.inputs["Color"])
        tree.links.new(light.outputs["Emission"], output.inputs["Surface"])
        mesh.materials.clear()
        mesh.materials.append(checker)
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 4
        render("uv_checker")
