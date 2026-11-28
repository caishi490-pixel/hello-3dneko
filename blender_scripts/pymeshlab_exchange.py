# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender triangle and face-corner exchange for the isolated repair worker."""

from pathlib import Path

import bpy
import numpy as np

from gap_coverage_validator import check_shell_contract, sample_shells
from mesh_exchange import welded_mesh_copy
from pipeline.retopo_backends import PyMeshLabAdapter, permits_alpha_wrap
from pipeline.sandbox import ShortPathWorkspace


def _mesh_arrays(obj, preserve_uv, *, world=False):
    mesh = obj.data if preserve_uv else welded_mesh_copy(obj.data)
    try:
        mesh.calc_loop_triangles()
        vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", vertices.ravel())
        if world:
            matrix = np.asarray(obj.matrix_world)
            vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        faces = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", faces.ravel())
        materials = np.zeros(len(faces), dtype=np.int32)
        uv = np.empty((0, len(faces), 3, 2))
        seams = np.empty((0, 2), dtype=np.int32)
        if preserve_uv:
            mesh.loop_triangles.foreach_get("material_index", materials)
            loops = np.empty(faces.shape, dtype=np.int32)
            mesh.loop_triangles.foreach_get("loops", loops.ravel())
            coords = []
            for layer in mesh.uv_layers:
                values = np.empty((len(mesh.loops), 2), dtype=np.float32)
                layer.uv.foreach_get("vector", values.ravel())
                coords.append(values[loops])
            uv = np.stack(coords)
            seams = np.asarray([sorted(edge.vertices) for edge in mesh.edges if edge.use_seam],
                               dtype=np.int32).reshape(-1, 2)
        return vertices, faces, uv, materials, seams
    finally:
        if not preserve_uv:
            bpy.data.meshes.remove(mesh)


def _combined_geometry(objects):
    points, triangles = [], []
    offset = 0
    for obj in objects:
        vertices, faces, _, _, _ = _mesh_arrays(obj, False, world=True)
        points.append(vertices)
        triangles.append(faces + offset)
        offset += len(vertices)
    faces = np.concatenate(triangles)
    return (np.concatenate(points), faces, np.empty((0, len(faces), 3, 2)),
            np.zeros(len(faces), dtype=np.int32), np.empty((0, 2), dtype=np.int32))


def _output_object(path, source):
    with np.load(path, allow_pickle=False) as data:
        vertices, faces = data["vertices"], data["faces"]
        uv, materials = data["uv"], data["materials"]
        seams = {tuple(sorted(pair)) for pair in data["seams"]}
    mesh = source.data.copy() if len(uv) else bpy.data.meshes.new("PyMeshLab")
    if len(uv):
        uv_layers = [(layer.name, layer.active_render) for layer in source.data.uv_layers]
        active = int(source.data.uv_layers.active_index)
        mesh.clear_geometry()
        for layer in list(mesh.uv_layers):
            mesh.uv_layers.remove(layer)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    if mesh.validate(verbose=False, clean_customdata=False):
        bpy.data.meshes.remove(mesh)
        raise RuntimeError("pymeshlab_invalid_mesh: Blender rejected imported topology")
    if len(uv):
        mesh.polygons.foreach_set("material_index", materials.astype(np.int32))
        for (name, active_render), coords in zip(uv_layers, uv):
            layer = mesh.uv_layers.new(name=name, do_init=False)
            layer.uv.foreach_set("vector", coords.astype(np.float32).ravel())
            layer.active_render = active_render
        mesh.uv_layers.active_index = active
        for edge in mesh.edges:
            edge.use_seam = tuple(sorted(edge.vertices)) in seams
    mesh.update()
    result = bpy.data.objects.new("PyMeshLabCandidate", mesh)
    bpy.context.scene.collection.objects.link(result)
    if source is not None:
        result.matrix_world = source.matrix_world
    return result


def run_pymeshlab_exchange(objects, params, contract, python):
    if not objects:
        raise RuntimeError("pymeshlab_invalid_mesh: no mesh surfaces")
    operation = params["operation"]
    envelope = operation == "alpha-wrap"
    preserve_uv = operation == "qecd-with-texture"
    if envelope and not permits_alpha_wrap(contract):
        raise RuntimeError("pymeshlab_envelope_requires_opt_in: conflicting or absent separation intent")
    targets = params.get("per_mesh_targets", {})
    adapter = PyMeshLabAdapter(python)
    groups = [objects] if envelope else [[obj] for obj in objects]
    reports = []
    with ShortPathWorkspace(Path.cwd()) as workspace:
        for group in groups:
            source = None if envelope else group[0]
            if preserve_uv and (not source.data.uv_layers or source.data.shape_keys or source.vertex_groups):
                raise RuntimeError("pymeshlab_uv_contract: requires UVs and a static mesh without weights or shape keys")
            arrays = _combined_geometry(group) if envelope else _mesh_arrays(source, preserve_uv)
            np.savez(workspace.path("in.npz"), **dict(zip(("vertices", "faces", "uv", "materials", "seams"), arrays)))
            report = adapter.process(workspace, operation, target_faces=targets.get(source.name) if source else None,
                                     single_shell_envelope=envelope)
            candidate = _output_object(workspace.path("out.npz"), source)
            data = candidate.data
            keep_candidate = False
            try:
                bpy.context.view_layer.update()
                samples = sample_shells(group)
                coverage = check_shell_contract([candidate], samples, contract if envelope else {},
                                                max_p95_pct=params["max_p95_dev_pct"])
                if not coverage["passed"]:
                    raise RuntimeError(f"pymeshlab_shell_coverage: {coverage['issues']}")
                report.update(mesh="Envelope" if envelope else source.name, shells_in=len(samples),
                              shells_out=coverage["shells_out"], shell_coverage_missing=coverage["shell_coverage_missing"])
                if envelope:
                    candidate.name = "Envelope"
                    for obj in group:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    keep_candidate = True
                else:
                    old = source.data
                    source.data = data
                    if old.users == 0:
                        bpy.data.meshes.remove(old)
                reports.append(report)
            finally:
                if not keep_candidate:
                    bpy.data.objects.remove(candidate, do_unlink=True)
                    if data.users == 0:
                        bpy.data.meshes.remove(data)
    return {"pymeshlab": reports}
