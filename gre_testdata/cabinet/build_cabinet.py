# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the original P3 cabinet, stable annotation and seeded disturbances."""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gre import blender_io as io
from gre.geometry import adjacency, face_edges, partition_region
from gre.snapshot import write_json


DIMENSIONS = {"width_m": 1.10, "depth_m": 0.512, "height_m": 1.405, "door_gap_m": 0.006, "bevel_m": 0.003,
              "outside_side_spacing_m": 1.10, "shelf_to_right_top_m": 0.540, "intentional_slope_deg": 12.0}


class Cabinet:
    def __init__(self) -> None:
        self.vertices: list[tuple] = []
        self.faces: list[tuple] = []
        self.groups: dict[str, list[int]] = defaultdict(list)
        self.materials: list[int] = []

    def face(self, corners, region: str, material: int) -> None:
        self.faces.append(tuple(corners))
        self.groups[region].append(len(self.faces))
        self.materials.append(material)

    def board(self, bounds, counts, regions=None, material=0) -> None:
        """Make a closed gridded box with identical vertices at sharp junctions."""
        regions = regions or {}
        local = {}

        def vertex(index: tuple) -> int:
            if index not in local:
                local[index] = len(self.vertices)
                self.vertices.append(tuple(lo + (hi - lo) * i / n for i, n, (lo, hi) in zip(index, counts, bounds)))
            return local[index]

        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for sign in (-1, 1):
                for i in range(counts[u]):
                    for j in range(counts[v]):
                        corners = []
                        for x, y in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                            index = [0, 0, 0]
                            index[axis] = counts[axis] if sign > 0 else 0
                            index[u], index[v] = x, y
                            corners.append(vertex(tuple(index)))
                        self.face(corners if sign > 0 else corners[::-1], regions.get((axis, sign), "body"), material)

    def slab(self, xspan, yspan, depth, counts, region, *, facing="up", seam=None, material=0, tilt=0.0) -> None:
        """Gridded broad faces with bevels and an optional protected seam-facing edge."""
        nx, ny = counts
        bevel = DIMENSIONS["bevel_m"]
        (x0, x1), (y0, y1), (low, high) = xspan, yspan, depth
        outline = ([(i, 0) for i in range(nx)] + [(nx, j) for j in range(ny)]
                   + [(i, ny) for i in range(nx, 0, -1)] + [(0, j) for j in range(ny, 0, -1)])
        local = {}
        rotation = np.radians(tilt)
        sine, cosine = np.sin(rotation), np.cos(rotation)
        cy, cz = -(low + high) / 2, (y0 + y1) / 2

        def vertex(i, j, layer) -> int:
            key = (i, j, layer)
            if key not in local:
                z, inset = ((high, bevel), (high - bevel, 0), (low + bevel, 0), (low, bevel))[layer]
                x = x0 + inset + (x1 - x0 - 2 * inset) * i / nx
                y = y0 + inset + (y1 - y0 - 2 * inset) * j / ny
                point = (x, y, z) if facing == "up" else (x, -z, y)
                if tilt:
                    px, py, pz = point
                    point = (px, cy + cosine * (py - cy) - sine * (pz - cz),
                             cz + sine * (py - cy) + cosine * (pz - cz))
                local[key] = len(self.vertices)
                self.vertices.append(point)
            return local[key]

        for layer in (0, 3):
            for i in range(nx):
                for j in range(ny):
                    corners = [vertex(x, y, layer) for x, y in ((i, j), (i+1, j), (i+1, j+1), (i, j+1))]
                    self.face(corners if layer == 0 else corners[::-1], region if layer == 0 else "body", material)
        for layer in range(3):
            for index, (i, j) in enumerate(outline):
                a, b = outline[(index + 1) % len(outline)]
                group = "bevel" if layer != 1 else "door_gap" if i == a == seam else "body"
                self.face((vertex(i, j, layer), vertex(i, j, layer+1), vertex(a, b, layer+1), vertex(a, b, layer)),
                          group, material)

    def knob(self, x: float, z: float) -> None:
        perimeter = [(np.cos(angle), np.sin(angle)) for angle in np.linspace(0, 2*np.pi, 24, endpoint=False)]
        rings = []
        for depth, radius in ((0.2475, 0.0065), (0.265, 0.011), (0.271, 0.009)):
            rings.append([len(self.vertices) + i for i in range(len(perimeter))])
            self.vertices.extend((x + radius * a, -depth, z + radius * b) for a, b in perimeter)
        self.face(rings[0][::-1], "hardware", 2)
        self.face(rings[-1], "hardware", 2)
        for lower, upper in zip(rings, rings[1:]):
            for i in range(len(perimeter)):
                j = (i + 1) % len(perimeter)
                self.face((upper[i], lower[i], lower[j], upper[j]), "hardware", 2)


def material(name: str, color: tuple, metallic: float = 0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    if mat.node_tree is None:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = mat.diffuse_color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = 0.44 if not metallic else 0.27
    if name == "Natural oak":
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 5
        uv = nodes.new("ShaderNodeTexCoord")
        stretch = nodes.new("ShaderNodeVectorMath")
        stretch.operation = "MULTIPLY"
        stretch.inputs[1].default_value = (38, 14, 2.2)
        ramp = nodes.new("ShaderNodeValToRGB")
        for element, factor in zip(ramp.color_ramp.elements, (0.65, 1.22)):
            element.color = (*[factor * value for value in color], 1)
        links = mat.node_tree.links
        links.new(uv.outputs["Object"], stretch.inputs[0])
        links.new(stretch.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    return mat


def build() -> tuple[bpy.types.Object, Cabinet]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1
    cabinet = Cabinet()
    cabinet.board(((-0.55, -0.51), (-0.21, 0.225), (0.045, 1.405)), (6, 18, 36), {(0, -1): "side_left"})
    cabinet.board(((0.51, 0.55), (-0.21, 0.225), (0.045, 1.405)), (6, 18, 36),
                  {(0, 1): "side_right", (1, -1): "right_end", (2, 1): "right_top"})
    cabinet.board(((-0.508, 0.508), (-0.207, 0.222), (1.365, 1.402)), (12, 6, 1))
    cabinet.board(((-0.508, 0.508), (-0.207, 0.222), (0.046, 0.092)), (12, 6, 1))
    cabinet.board(((-0.518, 0.518), (-0.182, 0.219), (0, 0.043)), (10, 6, 1), material=1)
    cabinet.board(((-0.548, 0.548), (0.226, 0.241), (0.049, 1.400)), (10, 1, 10), material=1)
    cabinet.slab((-0.508, -0.003), (0.10, 0.79), (0.215, 0.246), (18, 24), "front_left",
                 facing="front", seam=18, material=1)
    cabinet.slab((0.003, 0.508), (0.10, 0.79), (0.215, 0.246), (18, 24), "front_right",
                 facing="front", seam=0, material=1)
    cabinet.slab((-0.508, 0.508), (-0.206, 0.221), (0.833, 0.865), (24, 14), "shelf")
    cabinet.slab((-0.405, 0.405), (0.951, 1.24), (0.154, 0.181), (18, 14), "angled_panel",
                 facing="front", material=1, tilt=DIMENSIONS["intentional_slope_deg"])
    cabinet.knob(-0.061, 0.68)
    cabinet.knob(0.061, 0.68)
    mesh = bpy.data.meshes.new("CabinetMesh")
    mesh.from_pydata(cabinet.vertices, [], cabinet.faces)
    if mesh.validate(verbose=True):
        raise RuntimeError("Generated cabinet required topology repair")
    obj = bpy.data.objects.new("Cabinet", mesh)
    scene.collection.objects.link(obj)
    for name, domain, count in (("gre_vid", "POINT", len(mesh.vertices)), ("gre_fid", "FACE", len(mesh.polygons))):
        mesh.attributes.new(name=name, type="INT", domain=domain).data.foreach_set("value", list(range(1, count+1)))
    for mat in (material("Natural oak", (0.42, 0.225, 0.094)), material("Sage finish", (0.095, 0.19, 0.153)),
                material("Brushed brass", (0.66, 0.416, 0.142), 0.82)):
        mesh.materials.append(mat)
    for polygon, index in zip(mesh.polygons, cabinet.materials):
        polygon.material_index = index
    obj["gre_asset_id"] = "gre_cabinet_v1"
    obj["gre_dimensions_m"] = DIMENSIONS
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.011)
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh.uv_layers.active.name = "CabinetUV"
    bpy.context.view_layer.update()
    return obj, cabinet


def directions() -> dict[str, np.ndarray]:
    t = np.radians(DIMENSIONS["intentional_slope_deg"])
    return {"side_left": np.array([1, 0, 0]), "side_right": np.array([1, 0, 0]),
            "right_end": np.array([0, 1, 0]), "right_top": np.array([0, 0, 1]),
            "front_left": np.array([0, 1, 0]), "front_right": np.array([0, 1, 0]),
            "shelf": np.array([0, 0, 1]), "angled_panel": np.array([0, np.cos(t), np.sin(t)])}


def annotations(data, fixture: Cabinet) -> dict:
    normal = directions()
    pairs = [("parallel", "front_left", "front_right"), ("parallel", "front_right", "right_end"),
             ("parallel", "side_left", "side_right"), ("parallel", "shelf", "right_top"),
             ("perpendicular", "side_right", "right_end"), ("perpendicular", "side_right", "right_top"),
             ("perpendicular", "right_end", "right_top")]
    relations = [{"type": kind, "a": a, "b": b} for kind, a, b in pairs]
    relations.extend(({"type": "offset", "a": "side_left", "b": "side_right", "distance_m": 1.10},
                      {"type": "offset", "a": "shelf", "b": "right_top", "distance_m": 0.540}))
    corner = ["side_right", "right_end", "right_top"]
    targets = {}
    regions = []
    for name, ids in fixture.groups.items():
        row = {"name": name, "role": "editable" if name in normal else "protected", "face_ids": ids}
        if name in normal:
            row["semantic"] = "plane"
            vertex = data.F[ids[0]-1][0]
            targets[name] = {"normal": normal[name].tolist(), "d_m": -float(data.V[vertex] @ normal[name])}
        regions.append(row)
    return {"schema_version": 1, "asset_id": "gre_cabinet_v1", "object_name": "Cabinet",
            "source_hash": data.hashes["source_hash"], "unit_scale": 1, "matrix_world": data.matrix_world.tolist(),
            "regions": regions, "relations": relations, "junctions": [corner[:2], corner[1:], [corner[0], corner[2]], corner],
            "transition_rings": 2, "max_displacement_m": 0.006, "dimensions": DIMENSIONS,
            "approved_targets_for_qa": targets, "annotator": "GRE P3 original cabinet fixture"}


def apply_noise(obj, base, plan: dict, mode: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    normal = directions()
    editable = {region["name"]: region["face_ids"] for region in plan["regions"] if region["role"] == "editable"}
    selected = set().union(*(set(ids) for ids in editable.values()))
    faces = tuple(base.F[index-1] for index in sorted(selected))
    graph, _ = adjacency(base.V, face_edges(faces))
    exterior = {vertex for i, face in enumerate(base.F, 1) if i not in selected for vertex in face}
    core, blend, weights = partition_region(graph, graph.keys() & exterior, plan["transition_rings"])
    field = defaultdict(list)
    for name, face_ids in editable.items():
        if name == "angled_panel":
            continue
        indices = sorted({vertex for fid in face_ids for vertex in base.F[fid-1]})
        n = normal[name]
        tangent = rng.normal(size=3)
        tangent -= (tangent @ n) * n
        tangent /= np.linalg.norm(tangent)
        points = base.V[indices]
        phase = rng.uniform(0, 2*np.pi)
        if name == "side_right":
            # A displacement along X is tangential to the shared 6.7 mm end/top grid.
            # Keep that component small so the native fixture represents panel tilt,
            # instead of an artificial compression of the adjacent edge cell.
            tangent = np.array([0, 1, 0])
        if mode == "tilted":
            # A pronounced core tilt, smoothly tapering to locks at the board perimeter.
            angle = rng.uniform(0.31, 0.45)
            if name == "side_right":
                angle *= 0.10
            field_m = np.tan(np.deg2rad(angle)) * ((points - points.mean(axis=0)) @ tangent)
        else:
            amplitude = 0.00010 if name == "side_right" else 0.00060
            field_m = amplitude * np.sin((2*np.pi/0.24) * (points @ tangent) + phase)
        for index, value in zip(indices, field_m):
            field[index].append((n, float(value)))
    patch = {}
    for index, weight in zip(np.concatenate((core, blend)), np.concatenate((np.ones(len(core)), weights))):
        if index not in field:
            continue
        rows, values = zip(*field[index])
        offset = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)[0]
        patch[int(base.vid[index])] = base.V[index] + weight * offset
    io.write_positions(obj, patch, base.hashes["source_hash"])


def save(path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)
    with TemporaryDirectory(prefix=".gre-cabinet-", dir=path.parent) as directory:
        stage = Path(directory) / path.name
        io.save_copy(stage, path, path)
        os.link(stage, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("The fixture output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    obj, fixture = build()
    clean = output / "cabinet_clean.blend"
    base = io.read_mesh(obj)
    plan = annotations(base, fixture)
    save(clean)
    write_json(output / "cabinet_annotations.json", plan)
    wrong = {**plan, "relations": [*plan["relations"], {"type": "perpendicular", "a": "angled_panel", "b": "shelf"}]}
    write_json(output / "cabinet_wrong_relation.json", wrong)
    if not args.clean_only:
        for mode in ("tilted", "noisy"):
            io.open_blend(clean)
            obj = io.mesh_object("Cabinet")
            source = io.read_mesh(obj)
            apply_noise(obj, source, plan, mode, args.seed)
            save(output / f"cabinet_{mode}.blend")
    print(f"CABINET {len(base.V)} vertices, {len(base.F)} faces, seed={args.seed}, hash={base.hashes['source_hash']}", flush=True)


if __name__ == "__main__":
    main()
