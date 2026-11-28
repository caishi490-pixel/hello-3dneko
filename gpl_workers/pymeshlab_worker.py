# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated PyMeshLab triangle and face-corner UV worker."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pymeshlab as pml


def _mesh_set(vertices, faces):
    params = {"vertex_matrix": np.ascontiguousarray(vertices, dtype=np.float64),
              "face_matrix": np.ascontiguousarray(faces, dtype=np.int32)}
    scene = pml.MeshSet()
    scene.add_mesh(pml.Mesh(**params))
    return scene


def _textured_scene(folder, vertices, faces, uv):
    # The array constructor leaves wedge texture IDs unset. OBJ assigns an ID
    # with a scratch material while retaining a separate UV for every corner.
    (folder / "chart.ppm").write_text("P3\n1 1\n255\n255 255 255\n", encoding="ascii")
    (folder / "chart.mtl").write_text("newmtl Chart\nmap_Kd chart.ppm\n", encoding="ascii")
    source = folder / "chart.obj"
    with source.open("w", encoding="ascii") as handle:
        handle.write("mtllib chart.mtl\n")
        np.savetxt(handle, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(handle, uv.reshape(-1, 2), fmt="vt %.17g %.17g")
        handle.write("usemtl Chart\n")
        indices = np.stack((faces + 1, np.arange(1, faces.size + 1).reshape(-1, 3)), axis=-1)
        np.savetxt(handle, indices.reshape(-1, 6), fmt="f %d/%d %d/%d %d/%d")
    scene = pml.MeshSet()
    scene.load_new_mesh(str(source))
    return scene


def _topology(scene, *, require_manifold=True):
    props = scene.get_topological_measures()
    result = {key: int(props[key]) for key in
              ("non_two_manifold_edges", "non_two_manifold_vertices", "boundary_edges")}
    result["connected_components"] = int(props["connected_components_number"])
    if require_manifold and not bool(props["is_mesh_two_manifold"]):
        raise ValueError(f"pymeshlab_nonmanifold: {result}")
    return result


def _edges(face):
    for i, j in ((0, 1), (1, 2), (2, 0)):
        yield (i, j) if face[i] < face[j] else (j, i)


def _charts(faces, uv, material, seams):
    edge_data = defaultdict(list)
    for index, face in enumerate(faces):
        for a, b in _edges(face):
            edge = (face[a], face[b])
            if edge not in seams:
                edge_data[edge].append((index, a, b))
    neighbors = [[] for _ in faces]
    for rows in edge_data.values():
        if len(rows) == 2:
            (i, a, b), (j, c, d) = rows
            if material[i] == material[j] and np.array_equal(uv[:, i, [a, b]], uv[:, j, [c, d]]):
                neighbors[i].append(j)
                neighbors[j].append(i)
    seen = set()
    for i in range(len(faces)):
        if i in seen:
            continue
        group = [i]
        seen.add(i)
        for index in group:
            for neighbor in neighbors[index]:
                if neighbor not in seen:
                    group.append(neighbor)
                    seen.add(neighbor)
        yield group


def _boundary(faces):
    counts = Counter((face[a], face[b]) for face in faces for a, b in _edges(face))
    return {edge for edge, count in counts.items() if count == 1}


def _double_area(points):
    return np.linalg.norm(np.cross(points[..., 1, :] - points[..., 0, :],
                                   points[..., 2, :] - points[..., 0, :]), axis=-1)


def _repair_collinear_faces(vertices, faces):
    # Endpoint-preserving decimation can leave a flat triangle on an interior
    # edge. Flip that edge across the flat triangle's middle vertex.
    flat = np.flatnonzero(_double_area(vertices[faces]) <= 2e-12)
    if not len(flat):
        return
    incident = defaultdict(set)
    for index, face in enumerate(faces):
        for vertex in face:
            incident[vertex].add(index)
    for index in flat:
        tri = faces[index].copy()
        if _double_area(vertices[tri]) > 2e-12:
            continue
        side = int(np.argmax(np.sum((vertices[tri] - vertices[np.roll(tri, -1)]) ** 2, axis=1)))
        a, b, middle = np.roll(tri, -side)
        neighbors = (incident[a] & incident[b]) - {index}
        if len(neighbors) != 1:
            raise ValueError("pymeshlab_uv_contract: flat face cannot be retriangulated inside its chart")
        other = neighbors.pop()
        old = faces[other].copy()
        opposite = next(vertex for vertex in old if vertex not in (a, b))
        replacements = np.array([(middle, a, opposite), (b, middle, opposite)])
        if incident[middle] & incident[opposite] or np.any(_double_area(vertices[replacements]) <= 2e-12):
            raise ValueError("pymeshlab_uv_contract: flat face has no safe interior edge flip")
        for row, original, replacement in ((index, tri, replacements[0]), (other, old, replacements[1])):
            for vertex in original:
                incident[vertex].remove(row)
            for vertex in replacement:
                incident[vertex].add(row)
            faces[row] = replacement


def _reduce_chart(folder, vertices, faces, uv, target):
    if len(faces) <= target:
        return faces, uv
    ids, local = np.unique(faces, return_inverse=True)
    scene = _textured_scene(folder, vertices[ids], local.reshape(-1, 3), uv[0])
    coordinates = {tuple(vertices[index]): index for index in ids}
    if len(coordinates) != len(ids):
        raise ValueError("pymeshlab_uv_contract: coincident indexed chart vertices are ambiguous")
    uv_by_vertex = {}
    for index, coords in zip(faces.ravel(), uv.reshape(len(uv), -1, 2).transpose(1, 0, 2)):
        if index in uv_by_vertex and not np.array_equal(uv_by_vertex[index], coords):
            raise ValueError("pymeshlab_uv_contract: discontinuous chart corner")
        uv_by_vertex[index] = coords
    scene.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=target, preserveboundary=True, preservenormal=True,
        optimalplacement=False, extratcoordw=1.0,
    )
    _topology(scene)
    output = scene.current_mesh()
    try:
        mapped = np.array([coordinates[tuple(point)] for point in output.vertex_matrix()], dtype=np.int32)
    except KeyError as error:
        raise ValueError("pymeshlab_uv_contract: decimation moved an indexed vertex") from error
    reduced = mapped[output.face_matrix()]
    _repair_collinear_faces(vertices, reduced)
    if not len(reduced) or _boundary(reduced) != _boundary(faces):
        raise ValueError("pymeshlab_uv_contract: chart boundary changed")
    retained = np.array([uv_by_vertex[index] for index in reduced.ravel()])
    return reduced, retained.reshape(-1, 3, len(uv), 2).transpose(2, 0, 1, 3)


def _qecd(folder, vertices, faces, uv, materials, seams, target):
    if not len(uv) or target < 1:
        raise ValueError("pymeshlab_uv_contract: original UV layers and positive budget are required")
    if len(faces) <= target:
        return faces, uv, materials, 0
    chunks, kept_uv, slots = [], [], []
    chart_count = 0
    for group in _charts(faces, uv, materials, set(map(tuple, seams))):
        quota = max(1, len(group) * target // len(faces))
        reduced, coords = _reduce_chart(folder, vertices, faces[group], uv[:, group], quota)
        chunks.append(reduced)
        kept_uv.append(coords)
        slots.extend([materials[group[0]]] * len(reduced))
        chart_count += 1
    result = np.concatenate(chunks)
    if len(result) > target:
        raise ValueError(f"pymeshlab_budget_exceeded: chart boundary locks leave {len(result)} faces (quota {target})")
    return result, np.concatenate(kept_uv, axis=1), np.array(slots), chart_count


def _perform(job):
    mode = job["operation"]
    with np.load(job["input"], allow_pickle=False) as data:
        vertices, faces = data["vertices"], data["faces"]
        uv, materials, seams = data["uv"], data["materials"], data["seams"]
    if (not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
            or not np.isfinite(vertices).all() or not np.isfinite(uv).all()):
        raise ValueError("pymeshlab_invalid_mesh: empty mesh or non-finite coordinates")
    scene = _mesh_set(vertices, faces)
    before = _topology(scene, require_manifold=mode != "alpha-wrap")
    initial_count = len(faces)
    if mode == "qecd-with-texture":
        faces, uv, materials, charts = _qecd(Path(job["input"]).parent, vertices, faces, uv, materials,
                                             seams, job["target_faces"])
        ids, renumbered = np.unique(faces, return_inverse=True)
        vertices, faces = vertices[ids], renumbered.reshape(-1, 3)
        retain = dict(zip(ids.tolist(), range(len(ids))))
        seams = np.asarray([(retain[a], retain[b]) for a, b in seams if a in retain and b in retain],
                           dtype=np.int32).reshape(-1, 2)
        scene = _mesh_set(vertices, faces)
    elif mode == "close-holes":
        for _ in range(3):
            if not _topology(scene)["boundary_edges"]:
                break
            result = scene.meshing_close_holes(maxholesize=30, selfintersection=True, refinehole=True)
            if not result.get("closed_holes", 1):
                break
        charts = 0
    elif mode == "alpha-wrap":
        if job.get("single_shell_envelope") is not True:
            raise ValueError("pymeshlab_envelope_requires_opt_in")
        scene.generate_alpha_wrap(alpha=pml.PercentageValue(1.5), offset=pml.PercentageValue(0.3))
        charts = 0
    else:
        raise ValueError(f"Unknown PyMeshLab operation: {mode}")
    after = _topology(scene)
    if mode == "close-holes" and before["boundary_edges"] and after["boundary_edges"] >= before["boundary_edges"]:
        raise ValueError("pymeshlab_close_holes_no_change: no supported boundary loop could be closed")
    if mode == "alpha-wrap" and (after["connected_components"] != 1 or after["boundary_edges"] != 0):
        raise ValueError(f"pymeshlab_envelope_not_watertight: {after}")
    if mode != "qecd-with-texture":
        output = scene.current_mesh()
        vertices, faces = output.vertex_matrix(), output.face_matrix()
        uv, materials, seams = np.empty((0, len(faces), 3, 2)), np.zeros(len(faces), dtype=np.int32), np.empty((0, 2))
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("pymeshlab_invalid_mesh: empty or non-finite result")
    np.savez(job["output"], vertices=vertices, faces=faces, uv=uv, materials=materials, seams=seams)
    return {"operation": mode, "faces_in": initial_count, "faces_out": len(faces),
            "uv_charts": charts, "topology_in": before, "topology_out": after}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.probe:
        required = ("meshing_decimation_quadric_edge_collapse_with_texture", "meshing_close_holes",
                    "generate_alpha_wrap", "get_topological_measures")
        if not all(hasattr(pml.MeshSet, name) for name in required):
            return 1
        print(json.dumps({"backend": "pymeshlab", "version": version("pymeshlab")}))
        return 0
    job = json.loads(args.job.read_text(encoding="utf-8"))
    try:
        print(json.dumps(_perform(job)))
    except (RuntimeError, ValueError, pml.PyMeshLabException) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
