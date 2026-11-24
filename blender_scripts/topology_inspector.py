"""Read-only 6-pillar Geometry, Topology, Rig, UV & Fidelity Inspector for Blender 5.1+."""

import argparse
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Candidate .glb or .blend path")
    p.add_argument("--reference", default="", help="Optional original reference .glb/.blend")
    p.add_argument("--out-json", required=True)
    p.add_argument("--debris-max-verts", type=int, default=24)
    p.add_argument("--sample-count", type=int, default=6000)
    return p.parse_args(argv)


def load_scene_file(filepath):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fp = str(filepath)
    low = fp.lower()
    if low.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=fp)
    elif low.endswith(".glb") or low.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=fp)
    elif low.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=fp)
    elif low.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=fp)
    else:
        raise ValueError(f"Unsupported 3D asset format: {fp}")


def _gather_world_bmesh(mesh_objs):
    """Merge visible non-junk mesh objects into one world-space BMesh for global surface fidelity checks."""
    bm_out = bmesh.new()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in mesh_objs:
        eval_obj = obj.evaluated_get(depsgraph)
        me = eval_obj.to_mesh()
        bm_tmp = bmesh.new()
        bm_tmp.from_mesh(me)
        bmesh.ops.transform(bm_tmp, matrix=obj.matrix_world, verts=bm_tmp.verts)
        bmesh.ops.triangulate(bm_tmp, faces=bm_tmp.faces[:])
        v_map = {}
        for v in bm_tmp.verts:
            v_map[v.index] = bm_out.verts.new(v.co)
        for f in bm_tmp.faces:
            try:
                bm_out.faces.new([v_map[v.index] for v in f.verts])
            except ValueError:
                pass
        bm_tmp.free()
        eval_obj.to_mesh_clear()
    bm_out.verts.ensure_lookup_table()
    bm_out.faces.ensure_lookup_table()
    return bm_out


def _sample_surface_points(bm, target_count=6000):
    if len(bm.faces) == 0:
        return []
    pts = [v.co.copy() for v in bm.verts]
    pts.extend(f.calc_center_median() for f in bm.faces)
    if len(pts) <= target_count:
        return pts
    step = max(1, len(pts) // target_count)
    return pts[::step][:target_count]


def compute_symmetric_fidelity(cand_meshes, ref_meshes, sample_count=6000):
    bm_cand = _gather_world_bmesh(cand_meshes)
    bm_ref = _gather_world_bmesh(ref_meshes)
    try:
        if len(bm_cand.faces) == 0 or len(bm_ref.faces) == 0:
            return {"valid": False, "error": "Empty faces in candidate or reference"}

        ref_pts = [v.co for v in bm_ref.verts]
        min_v = Vector((min(p.x for p in ref_pts), min(p.y for p in ref_pts), min(p.z for p in ref_pts)))
        max_v = Vector((max(p.x for p in ref_pts), max(p.y for p in ref_pts), max(p.z for p in ref_pts)))
        diag = max(1e-5, (max_v - min_v).length)

        bvh_cand = BVHTree.FromBMesh(bm_cand)
        bvh_ref = BVHTree.FromBMesh(bm_ref)

        cand_samples = _sample_surface_points(bm_cand, sample_count)
        ref_samples = _sample_surface_points(bm_ref, sample_count)

        d_c2r = []
        for pt in cand_samples:
            _, _, _, d = bvh_ref.find_nearest(pt)
            if d is not None:
                d_c2r.append(d / diag)

        d_r2c = []
        for pt in ref_samples:
            _, _, _, d = bvh_cand.find_nearest(pt)
            if d is not None:
                d_r2c.append(d / diag)

        all_d = sorted(d_c2r + d_r2c)
        mean_pct = round(100.0 * (sum(all_d) / max(1, len(all_d))), 4)
        p95_pct = round(100.0 * all_d[min(len(all_d) - 1, int(len(all_d) * 0.95))], 4)
        max_pct = round(100.0 * all_d[-1], 4)

        cand_pts = [v.co for v in bm_cand.verts]
        c_min = Vector((min(p.x for p in cand_pts), min(p.y for p in cand_pts), min(p.z for p in cand_pts)))
        c_max = Vector((max(p.x for p in cand_pts), max(p.y for p in cand_pts), max(p.z for p in cand_pts)))
        ref_dim = max_v - min_v
        cand_dim = c_max - c_min
        dim_drift_pct = round(100.0 * (cand_dim - ref_dim).length / diag, 4)

        return {
            "valid": True,
            "ref_bbox_diag": round(diag, 5),
            "sym_mean_dev_pct": mean_pct,
            "sym_p95_dev_pct": p95_pct,
            "sym_max_dev_pct": max_pct,
            "c2r_p95_pct": round(100.0 * sorted(d_c2r)[min(len(d_c2r) - 1, int(len(d_c2r) * 0.95))], 4) if d_c2r else 999.0,
            "r2c_p95_pct": round(100.0 * sorted(d_r2c)[min(len(d_r2c) - 1, int(len(d_r2c) * 0.95))], 4) if d_r2c else 999.0,
            "bbox_dim_drift_pct": dim_drift_pct,
        }
    finally:
        bm_cand.free()
        bm_ref.free()


def _check_recoverable_quads(mesh_obj):
    """Estimate underlying quad regularity even when loaded from triangulated glTF."""
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)
    bm.faces.ensure_lookup_table()
    # Use temporary scene mesh for bpy.ops.mesh.tris_convert_to_quads cleanly
    tmp_me = bpy.data.meshes.new("_tmp_quad_eval")
    bm.to_mesh(tmp_me)
    bm.free()
    tmp_obj = bpy.data.objects.new("_tmp_quad_eval", tmp_me)
    bpy.context.scene.collection.objects.link(tmp_obj)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        tmp_obj.select_set(True)
        bpy.context.view_layer.objects.active = tmp_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.mode_set(mode="OBJECT")

        bm2 = bmesh.new()
        bm2.from_mesh(tmp_obj.data)
        total_f = max(1, len(bm2.faces))
        total_v = max(1, len(bm2.verts))
        quads = sum(1 for f in bm2.faces if len(f.verts) == 4)
        tris = sum(1 for f in bm2.faces if len(f.verts) == 3)
        ngons = sum(1 for f in bm2.faces if len(f.verts) > 4)
        val4 = sum(1 for v in bm2.verts if len(v.link_edges) == 4)
        bm2.free()
        return {
            "recovered_faces": len(tmp_obj.data.polygons),
            "recovered_quad_ratio": round(quads / total_f, 4),
            "recovered_tris": tris,
            "recovered_ngons": ngons,
            "recovered_valence4_ratio": round(val4 / total_v, 4),
        }
    finally:
        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        bpy.data.meshes.remove(tmp_me, do_unlink=True)


def _measure_diffuse_coverage(mesh_obj):
    if not mesh_obj.data.uv_layers.active:
        return {"has_uv": False, "uv_oob_loops": 0, "textures": [], "bake_alpha_coverage": None}
    uv_data = mesh_obj.data.uv_layers.active.data
    oob = 0
    for lp in uv_data:
        if not (-1e-4 <= lp.uv.x <= 1.0001 and -1e-4 <= lp.uv.y <= 1.0001):
            oob += 1

    textures = []
    alpha_cov = None
    for mat in mesh_obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        base_color_img = None
        if bsdf and bsdf.inputs["Base Color"].is_linked:
            src_node = bsdf.inputs["Base Color"].links[0].from_node
            if src_node.type == "TEX_IMAGE" and src_node.image:
                base_color_img = src_node.image
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and node.image.size[0] > 0:
                img = node.image
                textures.append({"material": mat.name, "image": img.name, "size": list(img.size)})
                is_diffuse = (img == base_color_img) or (
                    "diffuse" in img.name.lower() or "base" in img.name.lower() or "baked" in img.name.lower()
                )
                if alpha_cov is None and is_diffuse:
                    w, h = img.size
                    if w > 0 and h > 0:
                        px = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
                        hits = 0
                        total = 0
                        polys = mesh_obj.data.polygons
                        step = max(1, len(polys) // 4000)
                        for pi in range(0, len(polys), step):
                            p = polys[pi]
                            cu = sum(uv_data[li].uv.x for li in p.loop_indices) / len(p.loop_indices)
                            cv = sum(uv_data[li].uv.y for li in p.loop_indices) / len(p.loop_indices)
                            ix = max(0, min(w - 1, int(cu * w)))
                            iy = max(0, min(h - 1, int(cv * h)))
                            total += 1
                            if px[iy, ix, 3] > 0.5:
                                hits += 1
                        if total > 0:
                            alpha_cov = round(hits / total, 4)

    return {
        "has_uv": True,
        "uv_oob_loops": oob,
        "textures": textures,
        "bake_alpha_coverage": alpha_cov,
    }


def inspect_mesh_object(mesh_obj, debris_max_verts=24):
    me = mesh_obj.data
    mw = mesh_obj.matrix_world

    # 1. Raw counts
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    raw_f = len(bm.faces)
    raw_tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    raw_quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    raw_ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
    degen_faces = sum(1 for f in bm.faces if f.calc_area() < 1e-10)
    zero_edges = sum(1 for e in bm.edges if e.calc_length() < 1e-7)
    loose_verts = sum(1 for v in bm.verts if len(v.link_edges) == 0)

    # Check high-aspect-ratio stretched polygons (> 12:1 edge ratio)
    bad_aspect = 0
    for f in bm.faces:
        lens = [e.calc_length() for e in f.edges]
        if lens and min(lens) > 1e-9 and (max(lens) / min(lens)) > 12.0:
            bad_aspect += 1
    bm.free()

    # 2. Welded view (dist=1e-4) distinguishing UV-seam splits from true holes/non-manifold
    bm_w = bmesh.new()
    bm_w.from_mesh(me)
    bmesh.ops.remove_doubles(bm_w, verts=bm_w.verts[:], dist=1e-4)
    bm_w.verts.ensure_lookup_table()
    bm_w.edges.ensure_lookup_table()
    bm_w.faces.ensure_lookup_table()

    welded_non_manifold = sum(1 for e in bm_w.edges if not e.is_manifold and not e.is_boundary)
    welded_boundary = sum(1 for e in bm_w.edges if e.is_boundary)
    val_hist = Counter(len(v.link_edges) for v in bm_w.verts)
    welded_v = max(1, len(bm_w.verts))
    val4_ratio = round(val_hist.get(4, 0) / welded_v, 4)

    # Normal consistency check: recalc normals on copy and count flips
    pre_normals = [f.normal.copy() for f in bm_w.faces]
    bmesh.ops.recalc_face_normals(bm_w, faces=bm_w.faces[:])
    flipped_normals = sum(
        1 for i, f in enumerate(bm_w.faces) if f.normal.dot(pre_normals[i]) < -0.1
    )

    visited = set()
    shells = []
    for v in bm_w.verts:
        if v.index in visited:
            continue
        q = deque([v])
        visited.add(v.index)
        cnt = 0
        while q:
            cur = q.popleft()
            cnt += 1
            for e in cur.link_edges:
                ov = e.other_vert(cur)
                if ov.index not in visited:
                    visited.add(ov.index)
                    q.append(ov)
        shells.append(cnt)
    shells.sort(reverse=True)
    debris_shells = sum(1 for s in shells if s <= debris_max_verts)
    bm_w.free()

    # 3. Rig weight hygiene check
    unnormalized_weights = 0
    zero_weight_verts = 0
    if mesh_obj.vertex_groups:
        for v in me.vertices:
            wsum = sum(g.weight for g in v.groups)
            if wsum < 1e-4:
                zero_weight_verts += 1
            elif abs(wsum - 1.0) > 0.02:
                unnormalized_weights += 1

    rec_quads = _check_recoverable_quads(mesh_obj)
    uv_mat = _measure_diffuse_coverage(mesh_obj)

    pts = [mw @ v.co for v in me.vertices]
    dims = [0.0, 0.0, 0.0]
    if pts:
        dims = [
            round(max(p.x for p in pts) - min(p.x for p in pts), 4),
            round(max(p.y for p in pts) - min(p.y for p in pts), 4),
            round(max(p.z for p in pts) - min(p.z for p in pts), 4),
        ]

    return {
        "name": mesh_obj.name,
        "verts_raw": len(me.vertices),
        "faces_raw": raw_f,
        "tris_raw": raw_tris,
        "quads_raw": raw_quads,
        "ngons_raw": raw_ngons,
        "raw_quad_ratio": round(raw_quads / max(1, raw_f), 4),
        "welded_verts": sum(shells),
        "welded_shells_count": len(shells),
        "debris_shells_count": debris_shells,
        "welded_non_manifold_edges": welded_non_manifold,
        "welded_boundary_edges": welded_boundary,
        "degenerate_faces": degen_faces,
        "zero_length_edges": zero_edges,
        "loose_verts": loose_verts,
        "bad_aspect_faces": bad_aspect,
        "flipped_normals_estimate": flipped_normals,
        "welded_valence4_ratio": val4_ratio,
        "unnormalized_weight_verts": unnormalized_weights,
        "zero_weight_verts": zero_weight_verts,
        "dims": dims,
        **rec_quads,
        **uv_mat,
    }


def inspect_current_scene(reference_path="", debris_max_verts=24, sample_count=6000):
    sc = bpy.context.scene
    all_objs = list(sc.objects)
    junk_names = {"icosphere", "棱角球", "cube", "立方体"}
    junk_objs = [o.name for o in all_objs if o.name.split(".")[0].lower() in junk_names and len(getattr(o.data, "vertices", [])) <= 128]
    meshes = [o for o in all_objs if o.type == "MESH" and o.name not in junk_objs]
    arms = [o for o in all_objs if o.type == "ARMATURE"]

    mesh_reports = [inspect_mesh_object(m, debris_max_verts=debris_max_verts) for m in meshes]
    total_faces_raw = sum(m["faces_raw"] for m in mesh_reports)
    total_faces_rec = sum(m["recovered_faces"] for m in mesh_reports)

    summary = {
        "objects_total": len(all_objs),
        "mesh_count": len(meshes),
        "armature_count": len(arms),
        "junk_objects": junk_objs,
        "total_faces_raw": total_faces_raw,
        "total_faces_recovered": total_faces_rec,
        "meshes": mesh_reports,
    }

    if reference_path and Path(reference_path).is_file() and meshes:
        cand_objs = list(meshes)
        before_ptrs = {o.as_pointer() for o in sc.objects}
        rlow = str(reference_path).lower()
        if rlow.endswith(".blend"):
            with bpy.data.libraries.load(str(reference_path), link=False) as (data_from, data_to):
                data_to.objects = data_from.objects
            ref_objs = []
            for ro in data_to.objects:
                if ro is not None and ro.type == "MESH":
                    sc.collection.objects.link(ro)
                    ref_objs.append(ro)
        else:
            bpy.ops.import_scene.gltf(filepath=str(reference_path))
            ref_objs = [o for o in sc.objects if o.as_pointer() not in before_ptrs and o.type == "MESH"]
        summary["fidelity"] = compute_symmetric_fidelity(cand_objs, ref_objs, sample_count=sample_count)
        for ro in [o for o in sc.objects if o.as_pointer() not in before_ptrs]:
            bpy.data.objects.remove(ro, do_unlink=True)
    return summary


def main():
    args = _parse_args()
    load_scene_file(args.input)
    report = inspect_current_scene(
        reference_path=args.reference,
        debris_max_verts=args.debris_max_verts,
        sample_count=args.sample_count,
    )
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Inspected {args.input} -> {out_p}")


if __name__ == "__main__":
    main()
