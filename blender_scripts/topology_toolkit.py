"""Atomic, composable Blender 5.1+ topology repair toolkit (driven strictly by Agent plan JSON)."""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector, kdtree


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
    else:
        raise ValueError(f"Unsupported input: {filepath}")


def _set_active_object(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _get_primary_meshes():
    junk = {"icosphere", "棱角球"}
    return [
        o
        for o in bpy.context.scene.objects
        if o.type == "MESH" and o.name.split(".")[0].lower() not in junk
    ]


def action_cleanup_scene_for_engine(params):
    """Remove unused placeholders (Icosphere/棱角球) and duplicate `.001` orphan objects."""
    removed = []
    junk_prefixes = tuple(params.get("junk_prefixes", ["Icosphere", "棱角球"]))
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
    max_hole_edges = int(params.get("max_hole_edges", 24))
    passes = int(params.get("passes", 6))
    summary = []
    for obj in _get_primary_meshes():
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

        _prune_small_shells_bm(bm, min_shell_verts=48)
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


def action_adaptive_voxel_quadriflow(params):
    """Escalate voxel scale cleanly from a mesh backup until Quadriflow converges."""
    raw_scales = params.get("voxel_scales") or [0.012, 0.022, 0.032, 0.042]
    scales = [float(s) for s in raw_scales]
    target_faces = int(params.get("target_faces", 14000))
    min_shell_verts = int(params.get("min_shell_verts", 64))
    preserve_sharp = bool(params.get("use_preserve_sharp", False))
    preserve_boundary = bool(params.get("use_preserve_boundary", False))
    use_symmetry = bool(params.get("use_mesh_symmetry", False))
    smooth_normals = bool(params.get("smooth_normals", True))
    seed = int(params.get("seed", 0))

    out = []
    for obj in _get_primary_meshes():
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


def action_quadriflow_retopo(params):
    """Retopologize meshes into clean edge-loop pure quads using Quadriflow."""
    target_faces = int(params.get("target_faces", 12000))
    preserve_sharp = bool(params.get("use_preserve_sharp", False))
    preserve_boundary = bool(params.get("use_preserve_boundary", False))
    use_symmetry = bool(params.get("use_mesh_symmetry", False))
    smooth_normals = bool(params.get("smooth_normals", False))
    out = []
    for obj in _get_primary_meshes():
        pre_faces = len(obj.data.polygons)
        _set_active_object(obj)
        res = bpy.ops.object.quadriflow_remesh(
            mode="FACES",
            target_faces=target_faces,
            use_mesh_symmetry=use_symmetry,
            use_preserve_sharp=preserve_sharp,
            use_preserve_boundary=preserve_boundary,
            smooth_normals=smooth_normals,
            seed=int(params.get("seed", 0)),
        )
        post_faces = len(obj.data.polygons)
        if "FINISHED" not in res or (pre_faces > target_faces * 3 and post_faces > target_faces * 2.5):
            raise RuntimeError(
                f"quadriflow_remesh failed on '{obj.name}': result={res}, pre_faces={pre_faces}, post_faces={post_faces}"
            )
        for p in obj.data.polygons:
            p.use_smooth = True
        quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
        out.append({"mesh": obj.name, "faces": post_faces, "quads": quads})
    return {"quadriflow": out}


def action_decimate_and_quadify(params):
    """Collapse-decimate dense meshes to target face budget then quadify."""
    target_faces = int(params.get("target_faces", 14000))
    out = []
    for obj in _get_primary_meshes():
        cur_f = max(1, len(obj.data.polygons))
        if cur_f > target_faces:
            ratio = max(0.005, min(1.0, target_faces / cur_f))
            _set_active_object(obj)
            mod = obj.modifiers.new("AgentDecimate", "DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = ratio
            mod.use_collapse_triangulate = False
            bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=1e-4)
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
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


def action_shrinkwrap_to_reference(params, reference_glb):
    """Snap retopologized vertices tightly onto original untouched high-poly surface."""
    if not reference_glb or not Path(reference_glb).is_file():
        raise FileNotFoundError(f"Missing reference_glb for shrinkwrap: {reference_glb}")
    cand_meshes = _get_primary_meshes()
    ref_obj, imported = _import_temp_reference(reference_glb)
    try:
        for obj in cand_meshes:
            _set_active_object(obj)
            mod = obj.modifiers.new("AgentShrinkwrap", "SHRINKWRAP")
            mod.target = ref_obj
            mod.wrap_method = "NEAREST_SURFACEPOINT"
            mod.wrap_mode = "ON_SURFACE"
            mod.offset = float(params.get("offset", 0.0))
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
        return {"shrinkwrapped_to": str(reference_glb)}
    finally:
        for o in imported:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)


def _backup_and_wire_metallic_emit(ref_obj):
    """Switch ref_obj materials temporarily to output Metallic channel cleanly via Emission."""
    backups = []
    for mat in ref_obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        out_node = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        if not out_node:
            out_node = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not out_node or not bsdf:
            continue
        orig_surf = out_node.inputs["Surface"].links[0].from_socket if out_node.inputs["Surface"].is_linked else None
        em = nodes.new("ShaderNodeEmission")
        met_in = bsdf.inputs["Metallic"]
        if met_in.is_linked:
            links.new(met_in.links[0].from_socket, em.inputs["Color"])
        else:
            mval = float(met_in.default_value)
            em.inputs["Color"].default_value = (mval, mval, mval, 1.0)
        links.new(em.outputs["Emission"], out_node.inputs["Surface"])
        backups.append((mat, out_node, orig_surf, em))
    return backups


def _restore_materials_from_emit(backups):
    for mat, out_node, orig_surf, em in backups:
        if orig_surf is not None:
            mat.node_tree.links.new(orig_surf, out_node.inputs["Surface"])
        mat.node_tree.nodes.remove(em)


def _prepare_ref_color_baking_shader(ref_obj):
    """Make metallic surfaces emit pure albedo during bake so gold/silver doesn't bake black."""
    for mat in ref_obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf and not bsdf.inputs["Metallic"].is_linked:
            bsdf.inputs["Metallic"].default_value = 0.0
        elif bsdf and bsdf.inputs["Metallic"].is_linked:
            for lk in list(mat.node_tree.links):
                if lk.to_socket == bsdf.inputs["Metallic"]:
                    mat.node_tree.links.remove(lk)
            bsdf.inputs["Metallic"].default_value = 0.0


def _combine_orm_image(name, rough_img, met_img, out_path):
    """Pack Roughness (Green) & Metallic (Blue) cleanly into one glTF-standard ORM texture."""
    w, h = rough_img.size
    r_px = np.array(rough_img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    m_px = np.array(met_img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    orm = np.ones((h, w, 4), dtype=np.float32)
    orm[..., 0] = 1.0
    orm[..., 1] = r_px[..., 1]
    orm[..., 2] = m_px[..., 0]
    orm[..., 3] = 1.0
    orm_img = bpy.data.images.new(name, width=w, height=h, alpha=False)
    orm_img.colorspace_settings.name = "Non-Color"
    orm_img.pixels[:] = orm.reshape(-1).tolist()
    orm_img.filepath_raw = str(out_path)
    orm_img.file_format = "PNG"
    orm_img.save()
    return orm_img


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


def _sanitize_tangent_normal_image(img, min_b=0.68):
    """Clamp backface-hit tangent normals (Z <= 0 / strong inversion) to flat tangent normal."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return
    px = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    bad = px[..., 2] < float(min_b)
    if bad.any():
        px[bad, 0] = 0.5
        px[bad, 1] = 0.5
        px[bad, 2] = 1.0
        img.pixels[:] = px.reshape(-1).tolist()
        img.save()


def action_unwrap_and_bake_textures(params, reference_glb, textures_out_dir):
    """Smart UV project cleaned low-poly and Cycles-bake Diffuse/Normal/Roughness+Metallic from reference."""
    if not reference_glb or not Path(reference_glb).is_file():
        raise FileNotFoundError(f"Missing reference_glb for texture bake: {reference_glb}")

    tex_size = int(params.get("texture_size", 2048))
    angle_limit_deg = float(params.get("angle_limit_deg", 86.0))
    island_margin = float(params.get("island_margin", 0.004))
    cage_extrusion = float(params.get("cage_extrusion", 0.003))
    max_ray_distance = float(params.get("max_ray_distance", 0.018))
    normal_strength = float(params.get("normal_strength", 0.55))
    samples = int(params.get("bake_samples", 12))
    out_dir = Path(textures_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_meshes = _get_primary_meshes()
    ref_obj, imported = _import_temp_reference(reference_glb)
    sc = bpy.context.scene
    prev_engine = sc.render.engine
    try:
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.refresh_devices()
            if any(d.type != "CPU" for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                sc.cycles.device = "GPU"
        except Exception:
            sc.cycles.device = "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_denoising = False

        baked_paths = {}

        def _bake_single_map(obj, mat, role, btype, pfilter, cspace, dst):
            img = bpy.data.images.new(f"{obj.name}_baked_{role}", width=tex_size, height=tex_size, alpha=True)
            img.colorspace_settings.name = cspace
            tnode = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tnode.image = img
            mat.node_tree.nodes.active = tnode
            tnode.select = True

            bpy.ops.object.select_all(action="DESELECT")
            ref_obj.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            sc.render.bake.use_pass_direct = False
            sc.render.bake.use_pass_indirect = False
            sc.render.bake.use_pass_color = True
            bake_kwargs = {
                "type": btype,
                "use_selected_to_active": True,
                "cage_extrusion": cage_extrusion,
                "max_ray_distance": max_ray_distance,
                "margin": 12,
                "margin_type": "EXTEND",
                "use_clear": True,
                "normal_space": "TANGENT",
            }
            if pfilter:
                bake_kwargs["pass_filter"] = pfilter
            bpy.ops.object.bake(**bake_kwargs)
            img.filepath_raw = str(dst)
            img.file_format = "PNG"
            img.save()
            return tnode, img

        for obj in cand_meshes:
            _set_active_object(obj)
            if not obj.data.uv_layers:
                obj.data.uv_layers.new(name="UVMap")
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=math.radians(angle_limit_deg), island_margin=island_margin)
            bpy.ops.object.mode_set(mode="OBJECT")

            obj.data.materials.clear()
            mat = bpy.data.materials.new(f"{obj.name}_RetopoMat")
            mat.use_nodes = True
            bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Roughness"].default_value = 0.65
            obj.data.materials.append(mat)

            # 1. Bake Metallic via Emit while source materials are untouched
            met_backups = _backup_and_wire_metallic_emit(ref_obj)
            t_met, img_met = _bake_single_map(
                obj, mat, "metallic", "EMIT", set(), "Non-Color", out_dir / f"{obj.name}_baked_metallic.png"
            )
            _restore_materials_from_emit(met_backups)

            # 2. Bake Roughness & Tangent Normal from untouched source materials
            t_rough, img_rough = _bake_single_map(
                obj, mat, "roughness", "ROUGHNESS", set(), "Non-Color", out_dir / f"{obj.name}_baked_roughness.png"
            )
            t_norm, img_norm = _bake_single_map(
                obj, mat, "normal", "NORMAL", set(), "Non-Color", out_dir / f"{obj.name}_baked_normal.png"
            )
            _sanitize_tangent_normal_image(img_norm, min_b=0.68)

            # 3. Zero Metallic on reference so Diffuse albedo bakes crisp on gold/silver surfaces
            _prepare_ref_color_baking_shader(ref_obj)
            t_diff, img_diff = _bake_single_map(
                obj, mat, "diffuse", "DIFFUSE", {"COLOR"}, "sRGB", out_dir / f"{obj.name}_baked_diffuse.png"
            )
            _dilate_diffuse_rgb_keep_alpha(img_diff, passes=6)

            # Pack ORM (G=Roughness, B=Metallic) & remove scratch single-channel nodes cleanly
            orm_dst = out_dir / f"{obj.name}_baked_orm.png"
            orm_img = _combine_orm_image(f"{obj.name}_baked_orm", img_rough, img_met, orm_dst)
            mat.node_tree.nodes.remove(t_met)
            mat.node_tree.nodes.remove(t_rough)
            t_orm = mat.node_tree.nodes.new("ShaderNodeTexImage")
            t_orm.image = orm_img
            sep = mat.node_tree.nodes.new("ShaderNodeSeparateColor")

            links = mat.node_tree.links
            links.new(t_diff.outputs["Color"], bsdf.inputs["Base Color"])
            links.new(t_orm.outputs["Color"], sep.inputs["Color"])
            links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
            links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

            nmap = mat.node_tree.nodes.new("ShaderNodeNormalMap")
            nmap.inputs["Strength"].default_value = normal_strength
            links.new(t_norm.outputs["Color"], nmap.inputs["Color"])
            links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

            baked_paths[f"{obj.name}_diffuse"] = str(out_dir / f"{obj.name}_baked_diffuse.png")
            baked_paths[f"{obj.name}_normal"] = str(out_dir / f"{obj.name}_baked_normal.png")
            baked_paths[f"{obj.name}_orm"] = str(orm_dst)

        return {"baked_textures": baked_paths}
    finally:
        sc.render.engine = prev_engine
        for o in imported:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)


ACTION_DISPATCH = {
    "cleanup_scene_for_engine": lambda p, ctx: action_cleanup_scene_for_engine(p),
    "sanitize_mesh": lambda p, ctx: action_sanitize_mesh(p),
    "heal_manifold": lambda p, ctx: action_heal_manifold(p),
    "weld_and_quadify": lambda p, ctx: action_weld_and_quadify(p),
    "voxel_remesh_seal": lambda p, ctx: action_voxel_remesh_seal(p),
    "adaptive_voxel_quadriflow": lambda p, ctx: action_adaptive_voxel_quadriflow(p),
    "quadriflow_retopo": lambda p, ctx: action_quadriflow_retopo(p),
    "decimate_and_quadify": lambda p, ctx: action_decimate_and_quadify(p),
    "shrinkwrap_to_reference": lambda p, ctx: action_shrinkwrap_to_reference(p, ctx["reference_glb"]),
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
    step_logs = []
    for step in plan["actions"]:
        name = step["action"]
        params = step.get("params") or {}
        if name not in ACTION_DISPATCH:
            raise ValueError(f"Unknown toolkit action: {name}")
        res = ACTION_DISPATCH[name](params, ctx)
        step_logs.append({"action": name, "params": params, "result": res})

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
    )

    report = {
        "status": "executed",
        "output_blend": str(out_blend),
        "output_glb": str(out_glb),
        "steps": step_logs,
    }
    Path(plan["execution_log"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Topology toolkit finished executing plan ->", out_glb)


if __name__ == "__main__":
    main()
