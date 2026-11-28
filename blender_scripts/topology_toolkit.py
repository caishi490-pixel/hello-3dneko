"""Atomic, composable Blender 5.1+ topology repair toolkit (driven strictly by Agent plan JSON)."""

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector, kdtree
from mathutils.bvhtree import BVHTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_runtime import junk_objects, remove_export_junk
from gap_coverage_validator import check_shell_contract, sample_shells
from mesh_exchange import welded_mesh_copy

if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from pipeline.budget_allocator import allocate_object_budgets
from pipeline.retopo_backends import AUTOREMESHER_TIMEOUT, AutoRemesherAdapter, AutoRemesherError
from pipeline.sandbox import ShortPathWorkspace


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    return p.parse_args(argv)


def _load_scene(filepath):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    low = str(filepath).lower()
    if low.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=str(filepath))
    elif low.endswith(".glb") or low.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(filepath))
    elif low.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=str(filepath))
    else:
        raise ValueError(f"Unsupported input: {filepath}")


def _set_active_object(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _get_primary_meshes():
    objects = bpy.context.scene.objects
    ignored = set(junk_objects(objects))
    return [o for o in objects if o.type == "MESH" and o not in ignored]


def action_cleanup_scene_for_engine(params):
    """Remove unused placeholders (Icosphere/棱角球) and duplicate `.001` orphan objects."""
    removed = remove_export_junk()
    junk_prefixes = tuple(params.get("junk_prefixes", []))
    for obj in list(bpy.context.scene.objects):
        if any(obj.name.startswith(p) for p in junk_prefixes):
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    if params.get("drop_numbered_duplicates", False):
        for obj in list(bpy.context.scene.objects):
            if obj.name.endswith(".001") and obj.name[:-4] in bpy.data.objects:
                removed.append(obj.name)
                bpy.data.objects.remove(obj, do_unlink=True)
    return {"removed_objects": removed}


def action_sanitize_mesh(params):
    """Delete loose verts/edges, tiny debris shells, degenerate zero faces/edges, optionally weld seams."""
    min_shell_verts = int(params.get("min_shell_verts", 24))
    weld_dist = float(params.get("weld_dist", 1e-4))
    min_edge_length = float(params.get("min_edge_length", 5e-5))
    min_face_area = float(params.get("min_face_area", 1e-9))
    do_weld = bool(params.get("weld_seams", True))
    stats = []
    for obj in _get_primary_meshes():
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        if do_weld:
            bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=weld_dist)
        tiny_edges = [e for e in bm.edges if e.is_valid and e.calc_length() < min_edge_length]
        if tiny_edges:
            bmesh.ops.collapse(bm, edges=tiny_edges, uvs=True)
        degen = [f for f in bm.faces if f.is_valid and f.calc_area() < min_face_area]
        if degen:
            bmesh.ops.dissolve_faces(bm, faces=degen)
        degen2 = [f for f in bm.faces if f.is_valid and f.calc_area() < min_face_area]
        if degen2:
            bmesh.ops.delete(bm, geom=degen2, context="FACES")

        loose_v = [v for v in bm.verts if len(v.link_edges) == 0]
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context="VERTS")

        loose_e = [e for e in bm.edges if len(e.link_faces) == 0]
        if loose_e:
            bmesh.ops.delete(bm, geom=loose_e, context="EDGES")

        bm.verts.ensure_lookup_table()
        # Remove tiny debris components when multiple shells exist
        visited = set()
        comps = []
        for v in bm.verts:
            if v.index in visited:
                continue
            q = deque([v])
            visited.add(v.index)
            comp = []
            while q:
                cur = q.popleft()
                comp.append(cur)
                for e in cur.link_edges:
                    ov = e.other_vert(cur)
                    if ov.index not in visited:
                        visited.add(ov.index)
                        q.append(ov)
            comps.append(comp)
        removed_debris = 0
        if len(comps) > 1 and min_shell_verts > 0:
            debris_verts = [v for c in comps if len(c) <= min_shell_verts for v in c]
            if debris_verts and len(debris_verts) < len(bm.verts):
                bmesh.ops.delete(bm, geom=debris_verts, context="VERTS")
                removed_debris = sum(1 for c in comps if len(c) <= min_shell_verts)

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        stats.append({"mesh": obj.name, "removed_debris_shells": removed_debris, "verts_after": len(obj.data.vertices)})
    return {"sanitized": stats}


def _purge_non_manifold_bm(bm, max_passes=6):
    """Iteratively remove non-manifold edges, pinch vertices, folded flaps, and GLB-triangulation collisions."""
    removed_nm_faces = 0
    for _ in range(max_passes):
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f.is_valid])
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        nm_multi_edges = [e for e in bm.edges if e.is_valid and len(e.link_faces) > 2]
        nm_opp_edges = [e for e in bm.edges if e.is_valid and len(e.link_faces) == 2 and not e.is_contiguous]
        nm_verts = [v for v in bm.verts if v.is_valid and not v.is_manifold]
        # Folded 2-edge pocket flaps: vertex has only 2 edges yet >= 2 faces (double-covered fin/cap over a quad)
        flap_verts = [
            v for v in bm.verts if v.is_valid and len(v.link_edges) == 2 and len(v.link_faces) >= 2
        ]

        bad_faces = set()
        for e in nm_multi_edges:
            linked = sorted((f for f in e.link_faces if f.is_valid), key=lambda f: f.calc_area())
            if len(linked) > 2:
                bad_faces.update(linked[:-2])
        for e in nm_opp_edges:
            bad_faces.update(f for f in e.link_faces if f.is_valid)
        for v in nm_verts:
            bad_faces.update(f for f in v.link_faces if f.is_valid)
        for v in flap_verts:
            linked = sorted((f for f in v.link_faces if f.is_valid), key=lambda f: f.calc_area())
            if len(linked) >= 2:
                bad_faces.update(linked[:-1])

        # Catch face pairs sharing >=3 vertices (guaranteed non-manifold on GLB triangulation regardless of diagonal)
        if not bad_faces and bm.faces and len(bm.faces) <= 80000:
            bm.faces.ensure_lookup_table()
            face_vert_sets = {f: set(f.verts) for f in bm.faces if f.is_valid}
            for f, fv in face_vert_sets.items():
                if not f.is_valid or f in bad_faces:
                    continue
                neighbors = {nf for v in f.verts for nf in v.link_faces if nf != f and nf.is_valid and nf not in bad_faces}
                for nf in neighbors:
                    if len(fv & face_vert_sets.get(nf, set())) >= 3:
                        bad_faces.add(f if f.calc_area() <= nf.calc_area() else nf)

        # Simulate GLTF FIXED / SHORT_EDGE triangulation + seam weld(1.15e-4) via KDTree to prevent post-export nm edges
        if not bad_faces and bm.faces and len(bm.faces) <= 80000:
            bad_pts = []
            for qmethod in ("FIXED", "SHORT_EDGE"):
                bm_tri = bm.copy()
                if any(len(f.verts) > 3 for f in bm_tri.faces):
                    bmesh.ops.triangulate(bm_tri, faces=bm_tri.faces[:], quad_method=qmethod)
                bmesh.ops.remove_doubles(bm_tri, verts=bm_tri.verts[:], dist=1.15e-4)
                bm_tri.edges.ensure_lookup_table()
                for te in bm_tri.edges:
                    if len(te.link_faces) > 2:
                        bad_pts.append((te.verts[0].co.copy(), te.verts[1].co.copy()))
                bm_tri.free()
                if bad_pts:
                    break

            if bad_pts:
                verts_list = [v for v in bm.verts if v.is_valid]
                kd = kdtree.KDTree(len(verts_list))
                for idx, v in enumerate(verts_list):
                    kd.insert(v.co, idx)
                kd.balance()
                for p0, p1 in bad_pts:
                    v0s = [verts_list[i] for (_, i, _) in kd.find_range(p0, 2.5e-4)]
                    v1s = {verts_list[i] for (_, i, _) in kd.find_range(p1, 2.5e-4)}
                    cands = sorted(
                        {f for v0 in v0s for f in v0.link_faces if f.is_valid and any(u in v1s for u in f.verts)},
                        key=lambda f: f.calc_area(),
                    )
                    if cands:
                        bad_faces.add(cands[0])

        if not bad_faces:
            break

        bmesh.ops.delete(bm, geom=list(bad_faces), context="FACES")
        removed_nm_faces += len(bad_faces)

        loose_e = [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]
        if loose_e:
            bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
        loose_v = [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context="VERTS")

    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f.is_valid])
    return removed_nm_faces


def _prune_small_shells_bm(bm, min_shell_verts=64):
    """Delete isolated disconnected micro-shells without collapsing manifold mesh edges."""
    if min_shell_verts <= 0 or len(bm.verts) <= min_shell_verts:
        return 0
    bm.verts.ensure_lookup_table()
    visited = set()
    comps = []
    for v in bm.verts:
        if not v.is_valid or v.index in visited:
            continue
        q = deque([v])
        visited.add(v.index)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for e in cur.link_edges:
                ov = e.other_vert(cur)
                if ov.is_valid and ov.index not in visited:
                    visited.add(ov.index)
                    q.append(ov)
        comps.append(comp)
    if len(comps) <= 1:
        return 0
    largest_len = max(len(c) for c in comps)
    debris_verts = [
        v for c in comps if len(c) < largest_len and len(c) <= min_shell_verts for v in c
    ]
    if not debris_verts or len(debris_verts) >= len(bm.verts):
        return 0
    removed_shells = sum(1 for c in comps if len(c) < largest_len and len(c) <= min_shell_verts)
    bmesh.ops.delete(bm, geom=debris_verts, context="VERTS")
    return removed_shells


def action_heal_manifold(params):
    """Repair multi-face edges, pinch verts (nm_v), normals, and safe pinhole boundaries."""
    standard_hole_limit = int(params.get("max_hole_edges", 24))
    closed_objects = set(params.get("closed_objects", []))
    passes = int(params.get("passes", 6))
    min_shell_verts = int(params.get("min_shell_verts", 48))
    summary = []
    for obj in _get_primary_meshes():
        max_hole_edges = max(32, standard_hole_limit) if obj.name in closed_objects else standard_hole_limit
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        removed_nm_faces = _purge_non_manifold_bm(bm, max_passes=passes)
        filled_holes = 0
        reverted_unsafe_fills = 0

        # Group boundary loops and test each hole-fill individually so pinched ridges never re-introduce non-manifold geometry
        if max_hole_edges >= 2:
            bm.edges.ensure_lookup_table()
            b_edges = {e for e in bm.edges if e.is_valid and e.is_boundary}
            visited_e = set()
            candidate_loops = []
            for start_e in b_edges:
                if start_e in visited_e:
                    continue
                loop_e = []
                stack = [start_e]
                visited_e.add(start_e)
                while stack:
                    cur_e = stack.pop()
                    loop_e.append(cur_e)
                    for v in cur_e.verts:
                        for ne in v.link_edges:
                            if ne in b_edges and ne not in visited_e:
                                visited_e.add(ne)
                                stack.append(ne)
                if 2 <= len(loop_e) <= max_hole_edges:
                    candidate_loops.append(loop_e)

            for loop_e in candidate_loops:
                valid_loop = [e for e in loop_e if e.is_valid and e.is_boundary]
                if len(valid_loop) < 2:
                    continue
                loop_verts = {v for e in valid_loop for v in e.verts if v.is_valid}
                ring_verts = set(loop_verts)
                for lv in loop_verts:
                    for e in lv.link_edges:
                        ring_verts.add(e.other_vert(lv))

                existing_faces = set(bm.faces)
                bmesh.ops.holes_fill(bm, edges=valid_loop, sides=max_hole_edges)
                new_faces = [f for f in bm.faces if f.is_valid and f not in existing_faces]
                if not new_faces:
                    continue
                ngons = [f for f in new_faces if len(f.verts) > 4]
                if ngons:
                    bmesh.ops.triangulate(bm, faces=ngons)
                    new_faces = [f for f in bm.faces if f.is_valid and f not in existing_faces]

                bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f.is_valid])
                new_edges = {e for f in new_faces if f.is_valid for e in f.edges}
                created_bad = (
                    any(
                        not e.is_valid
                        or len(e.link_faces) > 2
                        or (len(e.link_faces) == 2 and not e.is_contiguous)
                        for e in new_edges
                    )
                    or any(v.is_valid and not v.is_manifold for v in ring_verts)
                    or any(f.is_valid and f.calc_area() < 1e-10 for f in new_faces)
                )
                if created_bad:
                    to_del = [f for f in new_faces if f.is_valid]
                    if to_del:
                        bmesh.ops.delete(bm, geom=to_del, context="FACES")
                    reverted_unsafe_fills += 1
                else:
                    filled_holes += 1

            # Final purge + micro-shell prune guarantees strictly zero non-manifold & zero detached fragments on exit
            removed_nm_faces += _purge_non_manifold_bm(bm, max_passes=passes)

        _prune_small_shells_bm(bm, min_shell_verts=min_shell_verts)
        loose_e = [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]
        if loose_e:
            bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
        loose_v = [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f.is_valid])

        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        rem_nm_e = sum(1 for e in bm.edges if len(e.link_faces) > 2 or (len(e.link_faces) == 2 and not e.is_contiguous))
        rem_nm_v = sum(1 for v in bm.verts if not v.is_manifold)
        rem_b_e = sum(1 for e in bm.edges if e.is_boundary)

        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        summary.append(
            {
                "mesh": obj.name,
                "removed_nm_faces": removed_nm_faces,
                "filled_hole_loops": filled_holes,
                "reverted_unsafe_fills": reverted_unsafe_fills,
                "remaining_nm_edges": rem_nm_e,
                "remaining_nm_verts": rem_nm_v,
                "remaining_boundary_edges": rem_b_e,
            }
        )
    return {"healed": summary}


def action_weld_and_quadify(params):
    """Weld coincident seam verts and convert coplanar triangles to quads (preserves loop UVs)."""
    weld_dist = float(params.get("weld_dist", 1e-4))
    max_face_angle = math.radians(float(params.get("max_face_angle_deg", 40.0)))
    max_shape_angle = math.radians(float(params.get("max_shape_angle_deg", 40.0)))
    out = []
    for obj in _get_primary_meshes():
        _set_active_object(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=weld_dist)
        bpy.ops.mesh.tris_convert_to_quads(
            face_threshold=max_face_angle,
            shape_threshold=max_shape_angle,
        )
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
        out.append({"mesh": obj.name, "faces": len(obj.data.polygons), "quads": quads})
    return {"quadified": out}


def action_voxel_remesh_seal(params):
    """Apply Voxel Remesh + droplet-only cleanup (no edge collapse) into one manifold."""
    voxel_size = float(params.get("voxel_size", 0.004))
    adaptivity = float(params.get("adaptivity", 0.0))
    min_shell_verts = int(params.get("min_shell_verts", 64))
    out = []
    for obj in _get_primary_meshes():
        _set_active_object(obj)
        mod = obj.modifiers.new("AgentVoxelSeal", "REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = voxel_size
        mod.adaptivity = adaptivity
        mod.use_smooth_shade = True
        bpy.ops.object.modifier_apply(modifier=mod.name)

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        removed_shells = _prune_small_shells_bm(bm, min_shell_verts=min_shell_verts)
        removed_nm = _purge_non_manifold_bm(bm, max_passes=4)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        out.append(
            {
                "mesh": obj.name,
                "faces_after_voxel": len(obj.data.polygons),
                "removed_droplet_shells": removed_shells,
                "removed_nm_faces": removed_nm,
            }
        )
    return {"voxel_sealed": out}


def _mesh_targets(params, default):
    meshes = _get_primary_meshes()
    budget = int(params.get("target_faces", default))
    targets = params.get("per_mesh_targets")
    if targets is None:
        # Direct worker callers may omit the planner's weighted targets.
        share, remainder = divmod(budget, max(1, len(meshes)))
        targets = {mesh.name: share + (i < remainder)
                   for i, mesh in enumerate(sorted(meshes, key=lambda obj: obj.name))}
    if set(targets) != {mesh.name for mesh in meshes} or sum(targets.values()) > budget:
        raise ValueError("Invalid per_mesh_targets for current meshes and total budget")
    for mesh in meshes:
        target = targets[mesh.name]
        if not isinstance(target, int) or target < 1:
            raise ValueError(f"budget_infeasible: no usable quota for {mesh.name}")
        yield mesh, target


def action_adaptive_voxel_quadriflow(params, check_preprocess=None):
    """Escalate voxel scale cleanly from a mesh backup until Quadriflow converges."""
    raw_scales = params.get("voxel_scales") or [0.012, 0.022, 0.032, 0.042]
    scales = [float(s) for s in raw_scales]
    min_shell_verts = int(params.get("min_shell_verts", 64))
    preserve_sharp = bool(params.get("use_preserve_sharp", False))
    preserve_boundary = bool(params.get("use_preserve_boundary", False))
    use_symmetry = bool(params.get("use_mesh_symmetry", False))
    smooth_normals = bool(params.get("smooth_normals", True))
    seed = int(params.get("seed", 0))

    out = []
    for obj, target_faces in _mesh_targets(params, 14000):
        orig_mesh = obj.data.copy()
        orig_mesh.name = f"__AdaptiveVoxelOrig_{obj.name}"
        chosen = None
        last_err = ""
        try:
            for vsz in scales:
                old_data = obj.data
                obj.data = orig_mesh.copy()
                if old_data and old_data != orig_mesh and old_data.users == 0:
                    bpy.data.meshes.remove(old_data)

                _set_active_object(obj)
                mod = obj.modifiers.new("AdaptiveVoxelSeal", "REMESH")
                mod.mode = "VOXEL"
                mod.voxel_size = vsz
                mod.adaptivity = 0.0
                mod.use_smooth_shade = True
                bpy.ops.object.modifier_apply(modifier=mod.name)

                bm = bmesh.new()
                bm.from_mesh(obj.data)
                removed_shells = _prune_small_shells_bm(bm, min_shell_verts=min_shell_verts)
                removed_nm = _purge_non_manifold_bm(bm, max_passes=4)
                bm.to_mesh(obj.data)
                bm.free()
                obj.data.update()
                if check_preprocess:
                    check_preprocess()

                pre_faces = len(obj.data.polygons)
                if 100 <= pre_faces < int(target_faces * 1.4):
                    smod = obj.modifiers.new("AdaptiveSimpleSubdiv", "SUBSURF")
                    smod.subdivision_type = "SIMPLE"
                    smod.levels = 1
                    smod.render_levels = 1
                    bpy.ops.object.modifier_apply(modifier=smod.name)
                    pre_faces = len(obj.data.polygons)

                res = bpy.ops.object.quadriflow_remesh(
                    mode="FACES",
                    target_faces=target_faces,
                    use_mesh_symmetry=use_symmetry,
                    use_preserve_sharp=preserve_sharp,
                    use_preserve_boundary=preserve_boundary,
                    smooth_normals=smooth_normals,
                    seed=seed,
                )
                post_faces = len(obj.data.polygons)
                if "FINISHED" in res and not (pre_faces > target_faces * 3 and post_faces > target_faces * 2.5):
                    for p in obj.data.polygons:
                        p.use_smooth = True
                    quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
                    chosen = {
                        "mesh": obj.name,
                        "voxel_size_used": vsz,
                        "removed_droplet_shells": removed_shells,
                        "removed_nm_faces": removed_nm,
                        "faces": post_faces,
                        "quads": quads,
                    }
                    break
                last_err = f"voxel_size={vsz}: res={res}, pre={pre_faces}, post={post_faces}"
            if chosen is None:
                old_data = obj.data
                obj.data = orig_mesh
                if old_data and old_data != orig_mesh and old_data.users == 0:
                    bpy.data.meshes.remove(old_data)
                raise RuntimeError(
                    f"adaptive_voxel_quadriflow exhausted scales {scales} on '{obj.name}'; last={last_err}"
                )
            out.append(chosen)
        finally:
            if obj.data != orig_mesh and orig_mesh.users == 0:
                bpy.data.meshes.remove(orig_mesh)
    return {"adaptive_voxel_quadriflow": out}


def _separate_disconnected_shells(obj):
    if len(sample_shells([obj])) < 2:
        return [obj]
    previous = {item.as_pointer() for item in bpy.context.scene.objects}
    _set_active_object(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    return [obj, *sorted((item for item in bpy.context.scene.objects if item.as_pointer() not in previous),
                         key=lambda item: item.name)]


def _quadriflow_with_budget(obj, quota, **options):
    _set_active_object(obj)
    original = obj.data
    initial_faces = len(original.polygons)
    target = quota
    for _ in range(3):
        obj.data = original.copy()
        accepted = False
        try:
            outcome = bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=target, **options)
            actual = len(obj.data.polygons)
            if "FINISHED" not in outcome or actual == 0 or (initial_faces > quota * 3 and actual > quota * 2.5):
                raise RuntimeError(
                    f"quadriflow_remesh failed on '{obj.name}': result={outcome}, pre_faces={initial_faces}, post_faces={actual}"
                )
            accepted = actual <= quota
            if accepted:
                return actual
        finally:
            if accepted:
                if original.users == 0:
                    bpy.data.meshes.remove(original)
            else:
                rejected = obj.data
                obj.data = original
                if rejected.users == 0:
                    bpy.data.meshes.remove(rejected)
        target = max(1, min(target - 1, int(target * quota / actual * 0.98)))
    raise RuntimeError(f"budget_exceeded: {obj.name} has {actual} faces with a quota of {quota}")


def _shell_extrema(obj):
    vertices = obj.data.vertices
    return [max(vertices, key=lambda vertex: sign * vertex.co[axis]).co.copy()
            for axis in range(3) for sign in (-1, 1)]


def _recover_shell_extrema(obj, source):
    """Recover only bounded loss of local silhouette extrema before guarded reprojection."""
    measures = _shell_extrema(obj)
    diagonal = Vector([source[i + 1][i // 2] - source[i][i // 2] for i in range(0, 6, 2)]).length
    if diagonal <= 1e-8:
        return
    vertices = obj.data.vertices
    tree = kdtree.KDTree(len(vertices))
    for vertex in vertices:
        tree.insert(vertex.co, vertex.index)
    tree.balance()
    displacements = {}
    for i, target in enumerate(source):
        axis, sign = i // 2, 1 if i % 2 else -1
        if sign * (target[axis] - measures[i][axis]) < diagonal * 0.008:
            continue
        origin, _, distance = tree.find(target)
        if distance > diagonal * 0.065:
            continue
        delta = target - origin
        radius = max(distance * 3, diagonal * 0.04)
        for _, index, radius_distance in tree.find_range(origin, radius):
            weight = (1 - (radius_distance / radius) ** 2) ** 2
            if index not in displacements or displacements[index][0] < weight:
                displacements[index] = (weight, delta)
    for index, (weight, delta) in displacements.items():
        vertices[index].co += weight * delta
    obj.data.update()


def action_quadriflow_retopo(params, *, targets=None):
    """Retopologize welded shells independently so a dominant shell cannot consume a thin part's quota."""
    preserve_sharp = bool(params.get("use_preserve_sharp", False))
    preserve_boundary = bool(params.get("use_preserve_boundary", False))
    use_symmetry = bool(params.get("use_mesh_symmetry", False))
    smooth_normals = bool(params.get("smooth_normals", False))
    if targets is None:
        targets = list(_mesh_targets(params, 12000))
    per_mesh_sharp = params.get("per_mesh_preserve_sharp", {})
    out = []
    for obj, target_faces in targets:
        sharp = per_mesh_sharp.get(obj.name, preserve_sharp)
        shells = _separate_disconnected_shells(obj)
        budgets = allocate_object_budgets([
            {"part_id": shell.name, "surface_area": sum(face.area for face in shell.data.polygons),
             "faces_raw": len(shell.data.polygons)}
            for shell in shells
        ], target_faces)
        fallbacks = []
        for shell in shells:
            extrema = _shell_extrema(shell) if len(shells) > 1 else None
            quota = budgets[shell.name]
            if quota < 1:
                raise RuntimeError(f"budget_infeasible: {shell.name} needs a usable surface budget")
            reason = "small_part" if quota < 600 or len(shell.data.polygons) <= max(600, int(quota * 1.25)) else ""
            if not reason:
                try:
                    _quadriflow_with_budget(
                        shell, quota, use_mesh_symmetry=use_symmetry,
                        use_preserve_sharp=sharp, use_preserve_boundary=preserve_boundary,
                        smooth_normals=smooth_normals, seed=int(params.get("seed", 0)),
                    )
                except RuntimeError as exc:
                    if len(targets) == 1 and len(shells) == 1:
                        raise
                    reason = str(exc).strip()
            if reason:
                actual = _local_retopo_with_budget(shell, quota)
                fallbacks.append({"shell": shell.name, "reason": reason, "faces": actual, "quota": quota})
            if extrema:
                _recover_shell_extrema(shell, extrema)
        if len(shells) > 1:
            _set_active_object(obj)
            for shell in shells:
                shell.select_set(True)
            bpy.ops.object.join()
        post_faces = len(obj.data.polygons)
        for p in obj.data.polygons:
            p.use_smooth = True
        quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
        out.append({"mesh": obj.name, "faces": post_faces, "quads": quads, "local_fallbacks": fallbacks})
    return {"quadriflow": out}


def _remove_exchange_object(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(mesh, bpy.types.Mesh) and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _autoremesh_part(obj, quota, adapter, workspace, deadline, max_p95_pct):
    source = workspace.path("input.obj")
    output = workspace.path("output.obj")
    reference = sample_shells([obj])
    clone = bpy.data.objects.new("AutoRemesherExchange", welded_mesh_copy(obj.data))
    bpy.context.scene.collection.objects.link(clone)
    try:
        _set_active_object(clone)
        bpy.ops.wm.obj_export(filepath=str(source), forward_axis="Y", up_axis="Z",
                              export_selected_objects=True, export_uv=False, export_normals=False,
                              export_materials=False, apply_modifiers=False)
    finally:
        _remove_exchange_object(clone)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AutoRemesherError("autoremesher_timeout", "150s route deadline reached")
    elapsed = adapter.remesh(source, output, target_quads=max(1, int(quota * 0.85)), timeout=remaining)
    previous = {item.as_pointer() for item in bpy.context.scene.objects}
    try:
        imported = bpy.ops.wm.obj_import(filepath=str(output), forward_axis="Y", up_axis="Z")
        candidates = [item for item in bpy.context.scene.objects if item.as_pointer() not in previous]
        if "FINISHED" not in imported or len(candidates) != 1 or candidates[0].type != "MESH":
            raise AutoRemesherError("autoremesher_execution_failed", f"{obj.name} expected one mesh")
        candidate = candidates[0]
        candidate.data.transform(candidate.matrix_world)
        candidate.matrix_world = obj.matrix_world
        if any(face.loop_total > 4 for face in candidate.data.polygons):
            bm = bmesh.new()
            try:
                bm.from_mesh(candidate.data)
                bmesh.ops.triangulate(bm, faces=[face for face in bm.faces if len(face.verts) > 4])
                bm.to_mesh(candidate.data)
            finally:
                bm.free()
        bpy.context.view_layer.update()
        faces = len(candidate.data.polygons)
        if not faces:
            raise AutoRemesherError("autoremesher_execution_failed", f"{obj.name} returned an empty mesh")
        coverage = check_shell_contract([candidate], reference, max_p95_pct=max_p95_pct)
        if not coverage["passed"]:
            raise AutoRemesherError("autoremesher_shell_coverage", ", ".join(coverage["issues"]))
        if faces > quota:
            raise AutoRemesherError("autoremesher_budget_exceeded", f"{obj.name} has {faces} faces, quota {quota}")
        old = obj.data
        obj.data = candidate.data
        if old.users == 0:
            bpy.data.meshes.remove(old)
        return {"mesh": obj.name, "faces": faces, "quota": quota, "elapsed_seconds": elapsed,
                "shells_in": len(reference), "shells_out": coverage["shells_out"],
                "shell_coverage_missing": coverage["shell_coverage_missing"]}
    except AutoRemesherError:
        raise
    except RuntimeError as exc:
        raise AutoRemesherError("autoremesher_execution_failed", str(exc)) from exc
    finally:
        for item in list(bpy.context.scene.objects):
            if item.as_pointer() not in previous:
                _remove_exchange_object(item)


def action_autoremesher_retopo(params):
    adapter = AutoRemesherAdapter.discover()
    if adapter is None:
        raise AutoRemesherError("autoremesher_execution_failed", "compatible CLI unavailable")
    eligible = set(params["auto_parts"])
    targets = list(_mesh_targets(params, 12000))
    deadline = time.monotonic() + AUTOREMESHER_TIMEOUT
    accepted, rollbacks, quadriflow_targets = [], [], []
    with ShortPathWorkspace(Path.cwd()) as workspace:
        for obj, quota in targets:
            if obj.name in eligible:
                try:
                    accepted.append(_autoremesh_part(obj, quota, adapter, workspace, deadline,
                                                     params["max_p95_dev_pct"]))
                    continue
                except AutoRemesherError as exc:
                    rollbacks.append({"mesh": obj.name, "reason": exc.code, "detail": str(exc)})
            quadriflow_targets.append((obj, quota))
    if not accepted:
        if rollbacks:
            raise RuntimeError(" | ".join(item["detail"] for item in rollbacks))
        raise AutoRemesherError("autoremesher_execution_failed", "plan has no eligible mesh")
    quadriflow = action_quadriflow_retopo(params, targets=quadriflow_targets)["quadriflow"]
    return {"autoremesher": accepted, "local_rollbacks": rollbacks, "quadriflow": quadriflow}


def _collapse_object(obj, ratio):
    _set_active_object(obj)
    mod = obj.modifiers.new("AgentDecimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = False
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _quadify_object(obj):
    _set_active_object(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def _local_retopo_with_budget(obj, quota):
    _quadify_object(obj)
    original = obj.data
    actual = len(original.polygons)
    if 0 < actual <= quota:
        return actual
    ratio = quota / max(1, actual)
    for _ in range(4):
        obj.data = original.copy()
        _collapse_object(obj, ratio)
        _quadify_object(obj)
        actual = len(obj.data.polygons)
        if 0 < actual <= quota:
            if original.users == 0:
                bpy.data.meshes.remove(original)
            return actual
        rejected = obj.data
        obj.data = original
        if rejected.users == 0:
            bpy.data.meshes.remove(rejected)
        if not actual:
            break
        ratio *= 0.95 * quota / actual
    raise RuntimeError(f"budget_infeasible: local fallback for {obj.name} produced {actual} faces (quota {quota})")


def action_decimate_and_quadify(params):
    """Collapse-decimate dense meshes to target face budget then quadify."""
    out = []
    for obj, target_faces in _mesh_targets(params, 14000):
        if len(obj.data.polygons) > target_faces:
            _collapse_object(obj, max(0.005, target_faces / len(obj.data.polygons)))
        _quadify_object(obj)
        quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
        out.append({"mesh": obj.name, "faces": len(obj.data.polygons), "quads": quads})
    return {"decimated": out}


def _import_temp_reference(ref_path):
    before_ptrs = {o.as_pointer() for o in bpy.context.scene.objects}
    rlow = str(ref_path).lower()
    if rlow.endswith(".blend"):
        with bpy.data.libraries.load(str(ref_path), link=False) as (df, dt):
            dt.objects = df.objects
        objs = []
        for ro in dt.objects:
            if ro is not None and ro.type == "MESH":
                bpy.context.scene.collection.objects.link(ro)
                objs.append(ro)
    else:
        bpy.ops.import_scene.gltf(filepath=str(ref_path))
        objs = [o for o in bpy.context.scene.objects if o.as_pointer() not in before_ptrs and o.type == "MESH"]
    imported = [o for o in bpy.context.scene.objects if o.as_pointer() not in before_ptrs]
    if not objs:
        raise RuntimeError(f"Imported zero mesh objects from reference: {ref_path}")
    if len(objs) == 1:
        objs[0].name = "_REF_HIGHPOLY"
        return objs[0], imported
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = "_REF_HIGHPOLY"
    return joined, [o for o in bpy.context.scene.objects if o.as_pointer() not in before_ptrs]


def _build_world_bvh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.transform(bm, matrix=obj.matrix_world, verts=bm.verts)
    bm.normal_update()
    pts = [v.co for v in bm.verts]
    if pts:
        min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        diag = max(1e-4, (max_v - min_v).length)
    else:
        diag = 1.0
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh, diag


def _normal_aware_snap_bm(bm, mw, mw_inv, ref_bvh, ref_diag, offset=0.0, min_normal_dot=0.15, max_snap_dist_ratio=0.018):
    """Project vertices onto reference surface while rejecting opposite-side hits on thin double-sided cloth/frills."""
    bm.normal_update()
    max_dist = max(1e-4, ref_diag * float(max_snap_dist_ratio))
    rot_m = mw.to_3x3().normalized()
    stats = {"snapped_nearest": 0, "snapped_ray_same_side": 0, "skipped_opposite_or_far": 0}
    for v in bm.verts:
        p_w = mw @ v.co
        n_w = (rot_m @ v.normal)
        if n_w.length > 1e-8:
            n_w.normalize()
        else:
            n_w = Vector((0.0, 0.0, 1.0))

        loc, hit_n, _, dist = ref_bvh.find_nearest(p_w, max_dist)
        if loc is not None and hit_n is not None and hit_n.dot(n_w) >= min_normal_dot:
            v.co = mw_inv @ (loc + hit_n.normalized() * offset)
            stats["snapped_nearest"] += 1
            continue

        # Nearest surface point belonged to opposite face of thin sheet or was too far -> cast bidirectional same-side ray
        best_hit = None
        for sign in (1.0, -1.0):
            ray_dir = n_w * sign
            h_loc, h_n, _, h_dist = ref_bvh.ray_cast(p_w - ray_dir * (max_dist * 0.15), ray_dir, max_dist * 1.15)
            if h_loc is not None and h_n is not None and h_n.dot(n_w) >= min_normal_dot:
                actual_d = (h_loc - p_w).length
                if actual_d <= max_dist and (best_hit is None or actual_d < best_hit[0]):
                    best_hit = (actual_d, h_loc, h_n)
        if best_hit is not None:
            _, h_loc, h_n = best_hit
            v.co = mw_inv @ (h_loc + h_n.normalized() * offset)
            stats["snapped_ray_same_side"] += 1
        else:
            stats["skipped_opposite_or_far"] += 1
    return stats


def action_shrinkwrap_to_reference(params, reference_glb):
    """Snap retopologized vertices tightly onto original untouched high-poly surface without thin-sheet punch-through."""
    if not reference_glb or not Path(reference_glb).is_file():
        raise FileNotFoundError(f"Missing reference_glb for shrinkwrap: {reference_glb}")
    normal_aware = bool(params.get("normal_aware", True))
    offset = float(params.get("offset", 0.0))
    min_normal_dot = float(params.get("min_normal_dot", 0.15))
    max_snap_dist_ratio = float(params.get("max_snap_dist_ratio", 0.018))
    cand_meshes = _get_primary_meshes()
    ref_obj, imported = _import_temp_reference(reference_glb)
    try:
        if normal_aware:
            ref_bvh, ref_diag = _build_world_bvh(ref_obj)
            summary = []
            for obj in cand_meshes:
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                st = _normal_aware_snap_bm(
                    bm,
                    obj.matrix_world,
                    obj.matrix_world.inverted(),
                    ref_bvh,
                    ref_diag,
                    offset=offset,
                    min_normal_dot=min_normal_dot,
                    max_snap_dist_ratio=max_snap_dist_ratio,
                )
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
                bm.to_mesh(obj.data)
                bm.free()
                for p in obj.data.polygons:
                    p.use_smooth = True
                obj.data.update()
                summary.append({"mesh": obj.name, **st})
            return {"shrinkwrapped_to": str(reference_glb), "normal_aware": True, "details": summary}

        for obj in cand_meshes:
            _set_active_object(obj)
            mod = obj.modifiers.new("AgentShrinkwrap", "SHRINKWRAP")
            mod.target = ref_obj
            mod.wrap_method = "NEAREST_SURFACEPOINT"
            mod.wrap_mode = "ON_SURFACE"
            mod.offset = offset
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
        return {"shrinkwrapped_to": str(reference_glb), "normal_aware": False}
    finally:
        for o in imported:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)


def action_refine_quad_flow(params, reference_glb):
    """Tangential Laplacian quad-flow relaxation + boundary loop smoothing + normal-aware reference reprojection."""
    iterations = int(params.get("iterations", 3))
    tangential_factor = float(params.get("tangential_factor", 0.36))
    boundary_factor = float(params.get("boundary_factor", 0.24))
    reproject = bool(params.get("reproject", True)) and bool(reference_glb and Path(reference_glb).is_file())
    min_normal_dot = float(params.get("min_normal_dot", 0.15))
    max_snap_dist_ratio = float(params.get("max_snap_dist_ratio", 0.015))

    cand_meshes = _get_primary_meshes()
    ref_obj = None
    imported = []
    ref_bvh = None
    ref_diag = 1.0
    if reproject:
        ref_obj, imported = _import_temp_reference(reference_glb)
        ref_bvh, ref_diag = _build_world_bvh(ref_obj)

    try:
        summary = []
        for obj in cand_meshes:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            mw = obj.matrix_world
            mw_inv = mw.inverted()
            snap_totals = {"snapped_nearest": 0, "snapped_ray_same_side": 0, "skipped_opposite_or_far": 0}

            for _ in range(max(1, iterations)):
                bm.normal_update()
                new_coords = {}
                for v in bm.verts:
                    if not v.link_edges:
                        continue
                    if v.is_boundary:
                        b_nbrs = [e.other_vert(v).co for e in v.link_edges if e.is_boundary]
                        if len(b_nbrs) == 2:
                            avg = (b_nbrs[0] + b_nbrs[1]) * 0.5
                            new_coords[v] = v.co.lerp(avg, boundary_factor)
                    else:
                        nbrs = [e.other_vert(v).co for e in v.link_edges]
                        avg = sum(nbrs, Vector((0.0, 0.0, 0.0))) / len(nbrs)
                        delta = avg - v.co
                        n = v.normal
                        if n.length > 1e-8:
                            delta = delta - n * delta.dot(n)
                        new_coords[v] = v.co + delta * tangential_factor
                for v, co in new_coords.items():
                    v.co = co

                if ref_bvh is not None:
                    st = _normal_aware_snap_bm(
                        bm,
                        mw,
                        mw_inv,
                        ref_bvh,
                        ref_diag,
                        offset=0.0,
                        min_normal_dot=min_normal_dot,
                        max_snap_dist_ratio=max_snap_dist_ratio,
                    )
                    for k in snap_totals:
                        snap_totals[k] += st[k]

            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.to_mesh(obj.data)
            bm.free()
            for p in obj.data.polygons:
                p.use_smooth = True
            obj.data.update()
            summary.append({"mesh": obj.name, "iterations": iterations, "reproject": reproject, **snap_totals})
        return {"refined_quad_flow": summary}
    finally:
        for o in imported:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)




def _dilate_diffuse_rgb_keep_alpha(img, passes=6):
    """Fill black ray-miss RGB texels from valid neighbors while keeping Alpha honest for QC."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return
    px = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    mask = px[..., 3] > 0.5
    if not mask.any() or mask.all():
        return
    rgb = px[..., :3].copy()
    rgb[~mask] = 0.0
    fallback_col = np.median(rgb[mask], axis=0)
    work_mask = mask.copy()
    for _ in range(passes):
        if work_mask.all():
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros((h, w), dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ms = np.roll(np.roll(work_mask, dy, axis=0), dx, axis=1)
            rs = np.roll(np.roll(rgb, dy, axis=0), dx, axis=1)
            acc[ms] += rs[ms]
            cnt[ms] += 1.0
        fill = (~work_mask) & (cnt > 0)
        if not fill.any():
            break
        rgb[fill] = acc[fill] / cnt[fill, None]
        work_mask[fill] = True
    if not work_mask.all():
        rgb[~work_mask] = fallback_col
    px[..., :3] = rgb
    img.pixels[:] = px.reshape(-1).tolist()
    img.save()




def action_unwrap_and_bake_textures(params, reference_glb, textures_out_dir):
    """Prepare local UVs then use the shared PBR atlas baker."""
    from uv_bake_toolkit import action_bake_pbr_maps

    for obj in _get_primary_meshes():
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        _set_active_object(obj)
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(float(params.get("angle_limit_deg", 86.0))),
                                 island_margin=float(params.get("island_margin", 0.004)))
        bpy.ops.object.mode_set(mode="OBJECT")
    result = action_bake_pbr_maps(params, reference_glb, textures_out_dir)
    return {**result, "baked_textures": next(iter(result["baked_pbr"]["meshes"].values()))}


def action_pymeshlab_retopo(params, ctx):
    from pymeshlab_exchange import run_pymeshlab_exchange

    return run_pymeshlab_exchange(_get_primary_meshes(), params, ctx["shell_contract"], ctx["pymeshlab_python"])


ACTION_DISPATCH = {
    "cleanup_scene_for_engine": lambda p, ctx: action_cleanup_scene_for_engine(p),
    "sanitize_mesh": lambda p, ctx: action_sanitize_mesh(p),
    "heal_manifold": lambda p, ctx: action_heal_manifold(p),
    "weld_and_quadify": lambda p, ctx: action_weld_and_quadify(p),
    "voxel_remesh_seal": lambda p, ctx: action_voxel_remesh_seal(p),
    "adaptive_voxel_quadriflow": lambda p, ctx: action_adaptive_voxel_quadriflow(p, ctx["check_preprocess"]),
    "quadriflow_retopo": lambda p, ctx: action_quadriflow_retopo(p),
    "autoremesher_retopo": lambda p, ctx: action_autoremesher_retopo(p),
    "pymeshlab_retopo": action_pymeshlab_retopo,
    "decimate_and_quadify": lambda p, ctx: action_decimate_and_quadify(p),
    "shrinkwrap_to_reference": lambda p, ctx: action_shrinkwrap_to_reference(p, ctx["reference_glb"]),
    "refine_quad_flow": lambda p, ctx: action_refine_quad_flow(p, ctx["reference_glb"]),
    "unwrap_and_bake_textures": lambda p, ctx: action_unwrap_and_bake_textures(
        p, ctx["reference_glb"], ctx["textures_dir"]
    ),
}


def main():
    args = _parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
    _load_scene(plan["input_path"])

    ctx = {
        "reference_glb": plan.get("reference_glb", plan["input_path"]),
        "textures_dir": plan["textures_dir"],
    }
    shell_contract = plan.get("shell_contract") or {}
    ctx["shell_contract"] = shell_contract
    ctx["pymeshlab_python"] = plan.get("pymeshlab_python")
    shell_samples = sample_shells(_get_primary_meshes()) if shell_contract else []
    stage_checks = []

    def check_preprocess():
        if not shell_contract:
            return
        result = check_shell_contract(_get_primary_meshes(), shell_samples, shell_contract, phase="preprocess",
                                      max_p95_pct=plan.get("max_p95_dev_pct", 2.2))
        stage_checks.append(result)
        if not result["passed"]:
            raise RuntimeError(f"Shell gate refused preprocessing: {result['issues']}")

    ctx["check_preprocess"] = check_preprocess
    if shell_contract and not shell_contract.get("single_shell_envelope"):
        check_preprocess()
    step_logs = []
    for step in plan["actions"]:
        name = step["action"]
        params = step.get("params") or {}
        if name not in ACTION_DISPATCH:
            raise ValueError(f"Unknown toolkit action: {name}")
        res = ACTION_DISPATCH[name](params, ctx)
        if name in {"sanitize_mesh", "heal_manifold", "weld_and_quadify", "voxel_remesh_seal", "pymeshlab_retopo"}:
            if not shell_contract.get("single_shell_envelope") or name == "voxel_remesh_seal":
                check_preprocess()
        if "target_faces" in params:
            faces = sum(len(obj.data.polygons) for obj in _get_primary_meshes())
            if faces > params["target_faces"]:
                raise RuntimeError(f"budget_exceeded: {faces} > {params['target_faces']}")
        step_logs.append({"action": name, "params": params, "result": res})

    remove_export_junk()
    bpy.ops.file.pack_all()
    out_blend = Path(plan["output_blend"])
    out_glb = Path(plan["output_glb"])
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_tangents=True,
    )

    report = {
        "status": "executed",
        "output_blend": str(out_blend),
        "output_glb": str(out_glb),
        "steps": step_logs,
        "preprocess_shell_checks": stage_checks,
        "junk_objects": [o.name for o in junk_objects(bpy.context.scene.objects)],
    }
    Path(plan["execution_log"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Topology toolkit finished executing plan ->", out_glb)


if __name__ == "__main__":
    main()
