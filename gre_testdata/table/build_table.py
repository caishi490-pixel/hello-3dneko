# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the original, watertight GRE table fixture and v1 annotations."""

import argparse
import os
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_asset import fingerprint, put_json, save_new_blend, script_args


GRID = 64
WIDTH, DEPTH, HEIGHT = 1.6, 0.85, 0.75
THICKNESS, BEVEL = 0.065, 0.003
ASSET_ID = "gre_table_v1"


def perimeter(x0: int, x1: int, y0: int, y1: int) -> list[tuple[int, int]]:
    return ([(x, y0) for x in range(x0, x1)] + [(x1, y) for y in range(y0, y1)]
            + [(x, y1) for x in range(x1, x0, -1)] + [(x0, y) for y in range(y1, y0, -1)])


def make_table() -> tuple[bpy.types.Object, dict[str, list[int]]]:
    vertices = []
    faces = []
    by_position = {}
    region_ids = {name: [] for name in ("tabletop", "bevel", "tabletop_body", "legs")}
    regions = []

    def vertex(x: int, y: int, height: float, inset: float = BEVEL) -> int:
        point = ((x / GRID - 0.5) * (WIDTH - 2 * inset),
                 (y / GRID - 0.5) * (DEPTH - 2 * inset), height)
        if point not in by_position:
            by_position[point] = len(vertices)
            vertices.append(point)
        return by_position[point]

    def face(region: str, corners: tuple[int, ...]) -> None:
        faces.append(corners)
        regions.append(region)
        region_ids[region].append(len(faces))

    def bridge(upper: list[int], lower: list[int], region: str) -> None:
        for index in range(len(upper)):
            next_index = (index + 1) % len(upper)
            face(region, (upper[index], lower[index], lower[next_index], upper[next_index]))

    underside = HEIGHT - THICKNESS
    legs = [(x, y) for x in (4, 57) for y in (6, 52)]
    leg_cells = {(x, y) for x0, y0 in legs for x in range(x0, x0 + 3) for y in range(y0, y0 + 6)}
    for y in range(GRID):
        for x in range(GRID):
            cell = ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1))
            face("tabletop", tuple(vertex(i, j, HEIGHT) for i, j in cell))
            if (x, y) not in leg_cells:
                face("tabletop_body", tuple(vertex(i, j, underside) for i, j in reversed(cell)))

    outline = perimeter(0, GRID, 0, GRID)
    rings = [[vertex(x, y, z, inset) for x, y in outline] for z, inset in (
        (HEIGHT, BEVEL), (HEIGHT - BEVEL, 0.0), (underside + BEVEL, 0.0), (underside, BEVEL),
    )]
    for index, region in enumerate(("bevel", "tabletop_body", "bevel")):
        bridge(rings[index], rings[index + 1], region)

    for x, y in legs:
        outline = perimeter(x, x + 3, y, y + 6)
        upper = [vertex(i, j, underside) for i, j in outline]
        lower = [vertex(i, j, 0.0) for i, j in outline]
        bridge(upper, lower, "legs")
        for j in range(y, y + 6):
            for i in range(x, x + 3):
                face("legs", (vertex(i, j, 0.0), vertex(i, j + 1, 0.0),
                              vertex(i + 1, j + 1, 0.0), vertex(i + 1, j, 0.0)))

    mesh = bpy.data.meshes.new("TableMesh")
    mesh.from_pydata(vertices, [], faces)
    if mesh.validate(verbose=True):
        raise RuntimeError("Generated mesh required repair")
    obj = bpy.data.objects.new("Table", mesh)
    bpy.context.scene.collection.objects.link(obj)
    for name, domain, count in (("gre_vid", "POINT", len(vertices)), ("gre_fid", "FACE", len(faces))):
        mesh.attributes.new(name=name, type="INT", domain=domain).data.foreach_set("value", list(range(1, count + 1)))

    for name, color in (("Oak", (0.42, 0.23, 0.105, 1.0)), ("Oak end grain", (0.31, 0.148, 0.06, 1.0))):
        material = bpy.data.materials.new(name)
        material.diffuse_color = color
        if material.node_tree is None:
            material.use_nodes = True
        shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = 0.46
        texture = material.node_tree.nodes.new("ShaderNodeTexNoise")
        texture.inputs["Scale"].default_value = 6.0
        texture.inputs["Detail"].default_value = 2.0
        coords = material.node_tree.nodes.new("ShaderNodeTexCoord")
        stretch = material.node_tree.nodes.new("ShaderNodeVectorMath")
        stretch.operation = "MULTIPLY"
        stretch.inputs[1].default_value = (1.0, 30.0, 6.0) if name == "Oak" else (18.0, 20.0, 1.0)
        ramp = material.node_tree.nodes.new("ShaderNodeValToRGB")
        for element, factor in zip(ramp.color_ramp.elements, (0.62, 1.22)):
            element.color = (*[channel * factor for channel in color[:3]], 1.0)
        links = material.node_tree.links
        links.new(coords.outputs["Object"], stretch.inputs[0])
        links.new(stretch.outputs["Vector"], texture.inputs["Vector"])
        links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
        mesh.materials.append(material)
    for polygon, region in zip(mesh.polygons, regions):
        polygon.material_index = int(region == "legs")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.012)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.uv_layers.active.name = "TableUV"
    bpy.context.view_layer.update()
    return obj, region_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(script_args())
    output = args.output_dir.resolve()
    clean = output / "table_clean.blend"
    labels = output / "table_annotations.json"
    for path in (clean, labels):
        if os.path.lexists(path):
            raise FileExistsError(path)
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    obj, face_groups = make_table()
    dimensions = {
        "width": WIDTH, "depth": DEPTH, "height": HEIGHT,
        "tabletop_thickness": THICKNESS, "bevel": BEVEL,
        "leg_width": 3 * (WIDTH - 2 * BEVEL) / GRID,
        "leg_depth": 6 * (DEPTH - 2 * BEVEL) / GRID,
    }
    obj["gre_asset_id"] = ASSET_ID
    obj["gre_dimensions_m"] = dimensions
    asset = fingerprint(obj)
    regions = []
    for name, face_ids in face_groups.items():
        region = {"name": name, "role": "protected", "face_ids": face_ids}
        if name == "tabletop":
            region.update(role="editable", semantic="plane")
        regions.append(region)
    annotations = {
        "schema_version": 1, "asset_id": ASSET_ID, "source_hash": asset["source_hash"],
        "unit_scale": asset["unit_scale"], "regions": regions, "transition_rings": 2,
        "max_displacement_m": None, "annotator": "GRE P0 procedural fixture",
        "notes": "Original programmatic reference geometry. All face references use gre_fid.",
        "object_name": obj.name, "matrix_world": asset["matrix_world"],
        "dimensions_m": dimensions, "tabletop_grid": [GRID, GRID],
    }
    save_new_blend(clean)
    put_json(labels, annotations)
    print(f"ASSET: {ASSET_ID}; topology counts: {asset['vertex_count']} vertices, {asset['face_count']} faces")
    print(f"SOURCE_HASH: {asset['source_hash']}")


if __name__ == "__main__":
    main()
