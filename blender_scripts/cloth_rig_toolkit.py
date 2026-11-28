"""Autonomous Garment Fitting, Skeleton Binding, Weight Transfer, and Self-Repair Toolkit."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cloth_rig_inspector import (  # noqa: E402
    analyze_garment_and_propose_fit,
    extract_body_profile,
    import_asset_additive,
    map_humanoid_skeleton,
    strip_bone_shape_junk,
)
from topology_inspector import load_scene_file  # noqa: E402
from blender_runtime import remove_export_junk


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--body", required=True)
    p.add_argument("--garment", required=True)
    p.add_argument("--actions-json", required=True)
    p.add_argument("--out-blend", required=True)
    p.add_argument("--out-glb", required=True)
    p.add_argument("--report-json", required=True)
    return p.parse_args(argv)


def _dist_point_to_seg(pt: Vector, a: Vector, b: Vector) -> float:
    ab = b - a
    denom = ab.length_squared
    if denom <= 1e-10:
        return (pt - a).length
    t = max(0.0, min(1.0, (pt - a).dot(ab) / denom))
    proj = a + ab * t
    return (pt - proj).length


def _build_welded_groups(mesh):
    """Group coincident UV-split vertices sharing the same spatial coordinates."""
    key_to_vs = defaultdict(list)
    v_to_key = {}
    for v in mesh.vertices:
        k = (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
        key_to_vs[k].append(v.index)
        v_to_key[v.index] = k
    return key_to_vs, v_to_key


def _build_mesh_neighbors(mesh):
    """Build adjacency bridging UV seams across spatially coincident vertices."""
    key_to_vs, v_to_key = _build_welded_groups(mesh)
    k_nbrs = defaultdict(set)
    for e in mesh.edges:
        ka = v_to_key[e.vertices[0]]
        kb = v_to_key[e.vertices[1]]
        if ka != kb:
            k_nbrs[ka].add(kb)
            k_nbrs[kb].add(ka)
    nbrs = {}
    for i, k in v_to_key.items():
        adj_vs = set()
        for nk in k_nbrs.get(k, ()):
            adj_vs.update(key_to_vs[nk])
        adj_vs.discard(i)
        nbrs[i] = list(adj_vs)
    return nbrs


def _smooth_vector_field(disp: list[Vector], nbrs: dict[int, list[int]], passes: int = 4, alpha: float = 0.6, mesh=None) -> list[Vector]:
    cur = [d.copy() for d in disp]
    welded = _build_welded_groups(mesh)[0] if mesh is not None else None
    for _ in range(passes):
        nxt = [None] * len(cur)
        for i, v in enumerate(cur):
            ns = nbrs.get(i)
            if not ns:
                nxt[i] = v.copy()
                continue
            avg = Vector((0.0, 0.0, 0.0))
            for ni in ns:
                avg += cur[ni]
            avg /= len(ns)
            nxt[i] = v * (1.0 - alpha) + avg * alpha
        if welded is not None:
            for vs in welded.values():
                if len(vs) > 1:
                    m = sum((nxt[vi] for vi in vs), Vector((0.0, 0.0, 0.0))) / len(vs)
                    for vi in vs:
                        nxt[vi] = m.copy()
        cur = nxt
    return cur


def _pose_and_bake_armature_rest(arm_obj, body_objs, skel_map, target_arm_deg: float):
    """Pose basemesh upper arms smoothly from T-Pose to target A-Pose angle and bake as clean Rest Pose."""
    slots = skel_map["slots"]
    u_l_name = slots.get("upper_arm_l")
    u_r_name = slots.get("upper_arm_r")
    l_l_name = slots.get("lower_arm_l")
    l_r_name = slots.get("lower_arm_r")
    if not u_l_name or not u_r_name or not l_l_name or not l_r_name:
        return 0.0

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm_obj.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()

    # Measure current left upper arm angle from vertical (-Z)
    pb_ul = arm_obj.pose.bones[u_l_name]
    pb_ll = arm_obj.pose.bones[l_l_name]
    pb_ur = arm_obj.pose.bones[u_r_name]
    v_cur = ((arm_obj.matrix_world @ pb_ll.head) - (arm_obj.matrix_world @ pb_ul.head)).normalized()
    cur_deg = math.degrees(math.acos(max(-1.0, min(1.0, v_cur.dot(Vector((0, 0, -1)))))))
    drop_deg = cur_deg - float(target_arm_deg)
    if abs(drop_deg) < 1.5:
        bpy.ops.object.mode_set(mode="OBJECT")
        return 0.0

    drop_rad = math.radians(drop_deg)
    # Left upper arm (+X): rotating +drop_rad around world +Y swings +X down toward -Z
    for pb, sign in ((pb_ul, 1.0), (pb_ur, -1.0)):
        h_w = arm_obj.matrix_world @ pb.head
        rot_w = Matrix.Rotation(sign * drop_rad, 4, Vector((0, 1, 0)))
        m_w = arm_obj.matrix_world @ pb.matrix
        m_new = Matrix.Translation(h_w) @ rot_w @ Matrix.Translation(-h_w) @ m_w
        pb.matrix = arm_obj.matrix_world.inverted() @ m_new
    bpy.context.view_layer.update()

    # Bake posed body vertex coordinates onto mesh data, then apply pose as Armature Rest Pose
    deps = bpy.context.evaluated_depsgraph_get()
    for obj in body_objs:
        ev = obj.evaluated_get(deps)
        tmp_me = bpy.data.meshes.new_from_object(ev)
        inv_mw = obj.matrix_world.inverted()
        for i, v in enumerate(tmp_me.vertices):
            obj.data.vertices[i].co = inv_mw @ (ev.matrix_world @ v.co)
        obj.data.update()
        bpy.data.meshes.remove(tmp_me)

    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return round(drop_deg, 2)


def _nearest_axis_dir(pw: Vector, bone_segs: list[tuple[Vector, Vector]]) -> Vector:
    best_pt = pw
    best_d = 1e9
    for a, b in bone_segs:
        ab = b - a
        denom = ab.length_squared
        t = 0.0 if denom <= 1e-10 else max(0.0, min(1.0, (pw - a).dot(ab) / denom))
        cand = a + ab * t
        d = (pw - cand).length
        if d < best_d:
            best_d = d
            best_pt = cand
    v = pw - best_pt
    return v.normalized() if v.length > 1e-6 else Vector((0.0, -1.0, 0.0))


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _build_joint_segments(arm_obj, skel_map, slot_pairs):
    """Build true joint-to-child-joint anatomical centerlines (VRM bone tails can be short stubs)."""
    segs = []
    if arm_obj is None or skel_map is None:
        return segs
    mw_a = arm_obj.matrix_world
    slots = skel_map.get("slots", {})
    for sa, sb in slot_pairs:
        ba = arm_obj.data.bones.get(slots.get(sa, ""))
        if not ba:
            continue
        pa = mw_a @ ba.head_local
        bb = arm_obj.data.bones.get(slots.get(sb, "")) if sb else None
        pb = (mw_a @ bb.head_local) if bb else (mw_a @ ba.tail_local)
        segs.append((pa, pb))
    return segs


def _outward_clearance_wrap(
    body_objs,
    garment_obj,
    margin_m=0.012,
    smooth_passes=5,
    inward_pull=0.0,
    arm_obj=None,
    skel_map=None,
    max_wrap_z=1.42,
    skirt_margin_m=None,
    max_passes=5,
    max_step=0.075,
    only_verts=None,
    center_y=0.0,
):
    """Push any garment vertex sunk inside or too close to skin cleanly outward over progressive passes."""
    deps = bpy.context.evaluated_depsgraph_get()
    bvhs = [BVHTree.FromObject(o, deps) for o in body_objs]
    mw_g = garment_obj.matrix_world
    inv_g = mw_g.inverted()
    nbrs = _build_mesh_neighbors(garment_obj.data)

    bone_segs = _build_joint_segments(
        arm_obj,
        skel_map,
        (
            ("hips", "spine"),
            ("spine", "chest"),
            ("chest", "upper_chest"),
            ("upper_chest", "neck"),
            ("upper_arm_l", "lower_arm_l"),
            ("lower_arm_l", "hand_l"),
            ("upper_arm_r", "lower_arm_r"),
            ("lower_arm_r", "hand_r"),
            ("upper_leg_l", "lower_leg_l"),
            ("lower_leg_l", "foot_l"),
            ("upper_leg_r", "lower_leg_r"),
            ("lower_leg_r", "foot_r"),
        ),
    )
    arm_axes = _build_joint_segments(
        arm_obj,
        skel_map,
        (
            ("upper_arm_l", "lower_arm_l"),
            ("lower_arm_l", "hand_l"),
            ("upper_arm_r", "lower_arm_r"),
            ("lower_arm_r", "hand_r"),
        ),
    )
    leg_segs = _build_joint_segments(
        arm_obj,
        skel_map,
        (
            ("upper_leg_l", "lower_leg_l"),
            ("lower_leg_l", "foot_l"),
            ("upper_leg_r", "lower_leg_r"),
            ("lower_leg_r", "foot_r"),
        ),
    )

    total_pushed = 0
    for it_pass in range(max_passes):
        raw_disp = [Vector((0.0, 0.0, 0.0)) for _ in garment_obj.data.vertices]
        pass_pushed = 0

        for i, v in enumerate(garment_obj.data.vertices):
            if only_verts is not None and i not in only_verts:
                continue
            pw = mw_g @ v.co
            if pw.z > max_wrap_z:
                continue
            best = None
            for bvh in bvhs:
                loc, norm, _, dist = bvh.find_nearest(pw, 0.22)
                if loc is not None and loc.z <= max_wrap_z and (best is None or dist < best[2]):
                    best = (loc, norm, dist)
            if best is None:
                continue
            loc, norm, _ = best
            if norm.length < 1e-6:
                continue
            norm = norm.normalized()
            signed_d = (pw - loc).dot(norm)
            d_arm = min((_dist_point_to_seg(pw, a, b) for a, b in arm_axes), default=9.9)
            d_leg = min((_dist_point_to_seg(pw, a, b) for a, b in leg_segs), default=9.9)
            is_arm_area = d_arm < 0.105 and abs(pw.x) > 0.145 and pw.z > 1.05
            is_leg_skirt_area = skirt_margin_m is not None and pw.z < 0.88 and d_leg < 0.145
            if is_arm_area:
                local_margin = max(margin_m, 0.026)
            elif is_leg_skirt_area:
                crotch_out = _smoothstep(0.032, 0.072, max(abs(pw.x) * 0.85, abs(pw.y - center_y) * 1.1))
                back_room = float(skirt_margin_m) + 0.008 * _smoothstep(0.02, 0.08, pw.y) * (1.0 - _smoothstep(0.55, 0.72, pw.z))
                local_margin = margin_m + (back_room - margin_m) * crotch_out
            else:
                local_margin = margin_m
            if signed_d < local_margin:
                if is_arm_area and arm_axes and signed_d < 0.008:
                    out_dir = _nearest_axis_dir(pw, arm_axes)
                elif signed_d < -0.004 and leg_segs and pw.z < 0.94:
                    if skirt_margin_m is not None and abs(pw.x) < 0.075:
                        dy = pw.y - center_y
                        out_dir = Vector((pw.x * 0.35, dy if abs(dy) > 0.005 else (1.0 if pw.y >= 0.0 else -1.0), 0.0)).normalized()
                    else:
                        radial = _nearest_axis_dir(pw, leg_segs)
                        horiz = Vector((radial.x, radial.y, 0.0))
                        out_dir = horiz.normalized() if horiz.length > 0.15 else radial
                elif signed_d < -0.005 and bone_segs:
                    out_dir = _nearest_axis_dir(pw, bone_segs)
                else:
                    out_dir = norm
                if out_dir.dot(norm) < -0.25 and bone_segs:
                    if skirt_margin_m is not None and pw.z < 0.94 and abs(pw.x) < 0.075:
                        dy = pw.y - center_y
                        out_dir = Vector((0.0, 1.0 if dy >= 0.0 else -1.0, 0.0))
                    else:
                        out_dir = _nearest_axis_dir(pw, bone_segs)
                if pw.z > max_wrap_z - 0.045 and out_dir.z > 0.35:
                    out_dir = Vector((out_dir.x, out_dir.y, 0.0)).normalized() if (abs(out_dir.x) + abs(out_dir.y)) > 1e-4 else norm
                need = local_margin - signed_d
                raw_disp[i] = out_dir * min(max_step, need * 1.12)
                pass_pushed += 1
            elif it_pass == 0 and inward_pull > 0.0 and 0.036 < signed_d < 0.085 and pw.z > 1.04:
                pull = (signed_d - 0.026) * inward_pull
                raw_disp[i] = -norm * min(0.022, pull)

        if pass_pushed == 0 and it_pass > 0:
            break
        total_pushed = max(total_pushed, pass_pushed)
        smooth_d = _smooth_vector_field(raw_disp, nbrs, passes=smooth_passes, alpha=0.55, mesh=garment_obj.data) if smooth_passes > 0 else raw_disp
        for i, d_base in enumerate(raw_disp):
            if only_verts is not None and i not in only_verts:
                continue
            d = smooth_d[i]
            if d_base.length > 1e-6:
                d_dir = d_base.normalized()
                proj = d.dot(d_dir)
                if proj < d_base.length:
                    d = d + d_dir * (d_base.length - proj)
            if d.length > 1e-6:
                pw_new = (mw_g @ garment_obj.data.vertices[i].co) + d
                if pw_new.z <= max_wrap_z + 0.01:
                    garment_obj.data.vertices[i].co = inv_g @ pw_new
        garment_obj.data.update()
    return total_pushed


def _subdivide_post_wrap_chords(skin_objs, mesh_obj, max_edge_len: float = 0.028, near_skin_m: float = 0.11, max_passes: int = 3) -> set[int]:
    """Subdivide long bridging edges near skin with quad poke cleanup and return indices of newly created vertices."""
    mw = mesh_obj.matrix_world
    deps = bpy.context.evaluated_depsgraph_get()
    bvhs = [BVHTree.FromObject(o, deps) for o in skin_objs]
    n_before = len(mesh_obj.data.vertices)
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    for _ in range(max_passes):
        if len(bm.verts) > 65000:
            break
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        long_edges = []
        for e in bm.edges:
            pa, pb = mw @ e.verts[0].co, mw @ e.verts[1].co
            if (pa - pb).length > max_edge_len and any(
                any(bvh.find_nearest(pa.lerp(pb, t), near_skin_m)[0] is not None for bvh in bvhs)
                for t in (0.15, 0.35, 0.5, 0.65, 0.85)
            ):
                long_edges.append(e)
        if not long_edges:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True, use_single_edge=True)
        ngons = [f for f in bm.faces if len(f.verts) > 4]
        if ngons:
            bmesh.ops.poke(bm, faces=ngons)
    bm.to_mesh(mesh_obj.data)
    mesh_obj.data.update()
    bm.free()
    return set(range(n_before, len(mesh_obj.data.vertices)))


def _pop_subdivided_verts(
    body_objs,
    mesh_obj,
    arm_obj,
    skel_map,
    center_y: float,
    max_wrap_z: float,
    new_verts: set[int],
    min_clear: float = 0.014,
    target_clear: float = 0.020,
    max_disp: float = 0.060,
) -> int:
    """Push newly subdivided chord midpoints out around limbs using neighbor-coherent and span-perpendicular wrap directions."""
    if not new_verts:
        return 0
    deps = bpy.context.evaluated_depsgraph_get()
    bvhs = [BVHTree.FromObject(o, deps) for o in body_objs]
    mw_g = mesh_obj.matrix_world
    inv_g = mw_g.inverted()
    nbrs = _build_mesh_neighbors(mesh_obj.data)
    leg_segs = _build_joint_segments(
        arm_obj, skel_map,
        (("upper_leg_l", "lower_leg_l"), ("lower_leg_l", "foot_l"), ("upper_leg_r", "lower_leg_r"), ("lower_leg_r", "foot_r")),
    )
    arm_axes = _build_joint_segments(
        arm_obj, skel_map,
        (("upper_arm_l", "lower_arm_l"), ("lower_arm_l", "hand_l"), ("upper_arm_r", "lower_arm_r"), ("lower_arm_r", "hand_r")),
    )
    moved = 0
    for _ in range(3):
        all_pw = [mw_g @ v.co for v in mesh_obj.data.vertices]
        disp = {}
        for i in new_verts:
            if i >= len(all_pw):
                continue
            pw = all_pw[i]
            if pw.z > max_wrap_z:
                continue
            best = None
            for bvh in bvhs:
                loc, norm, _, dist = bvh.find_nearest(pw, 0.20)
                if loc is not None and loc.z <= max_wrap_z and (best is None or dist < best[2]):
                    best = (loc, norm, dist)
            if best is None:
                continue
            loc, norm, _ = best
            if norm.length < 1e-6:
                continue
            norm = norm.normalized()
            signed_d = (pw - loc).dot(norm)
            crotch = _smoothstep(0.022, 0.048, abs(pw.x)) if (0.68 <= pw.z <= 0.92 and abs(pw.y - center_y) < 0.065) else 1.0
            m_min = 0.008 + max(0.0, min_clear - 0.008) * crotch if min_clear > 0.008 else min_clear
            m_tgt = 0.010 + max(0.0, target_clear - 0.010) * crotch if target_clear > 0.010 else target_clear
            if signed_d >= m_min:
                continue

            d_arm = min((_dist_point_to_seg(pw, a, b) for a, b in arm_axes), default=9.9)
            coh = Vector((0.0, 0.0, 0.0))
            if signed_d < -0.002 and leg_segs and pw.z < 0.94:
                for nb in nbrs.get(i, ()):
                    if nb in new_verts:
                        continue
                    rad_nb = _nearest_axis_dir(all_pw[nb], leg_segs)
                    hz_nb = Vector((rad_nb.x, rad_nb.y, 0.0))
                    if hz_nb.length > 0.15:
                        coh += hz_nb.normalized()

            if d_arm < 0.105 and abs(pw.x) > 0.145 and pw.z > 1.05 and signed_d < 0.006:
                od = _nearest_axis_dir(pw, arm_axes)
            elif coh.length > 0.2:
                od = coh.normalized()
            elif signed_d < -0.002 and leg_segs and pw.z < 0.94:
                if pw.z >= 0.70 and abs(pw.x) < 0.042:
                    dy = pw.y - center_y
                    od = Vector((0.0, 1.0 if dy >= 0.0 else -1.0, 0.0))
                else:
                    rad = _nearest_axis_dir(pw, leg_segs)
                    hz = Vector((rad.x, rad.y, 0.0))
                    od = hz.normalized() if hz.length > 0.15 else rad
            else:
                od = norm
            if od.dot(norm) < -0.25 and coh.length <= 0.2:
                od = norm

            step = min(max_disp, max(0.0, m_tgt - signed_d))
            if signed_d < 0.004 and leg_segs and pw.z < 0.94:
                nb_pts = [all_pw[nb] for nb in nbrs.get(i, ())]
                best_span = 0.0
                span_v = None
                for ai_n in range(len(nb_pts)):
                    for bi_n in range(ai_n + 1, len(nb_pts)):
                        dv_xy = Vector((nb_pts[bi_n].x - nb_pts[ai_n].x, nb_pts[bi_n].y - nb_pts[ai_n].y, 0.0))
                        if dv_xy.length > best_span:
                            best_span = dv_xy.length
                            span_v = dv_xy
                if best_span > 0.055 and span_v is not None:
                    perp = Vector((-span_v.y, span_v.x, 0.0)).normalized()
                    rad_l = _nearest_axis_dir(pw, leg_segs)
                    pref = coh if coh.length > 0.2 else Vector((rad_l.x, rad_l.y, 0.0))
                    if perp.dot(pref) < 0.0:
                        perp = -perp
                    od = perp
                    step = max(step, min(max_disp, 0.032))

            cand = pw + od * step
            loc2, n2, _, _ = bvhs[0].find_nearest(cand, 0.20)
            if loc2 is not None and n2.length > 1e-6 and (cand - loc2).dot(n2.normalized()) < signed_d:
                cand = pw + norm * min(max_disp, max(0.0, m_tgt - signed_d))
            disp[i] = cand

        if not disp:
            break
        for i, cand in disp.items():
            mesh_obj.data.vertices[i].co = inv_g @ cand
        moved += len(disp)
        mesh_obj.data.update()
    return moved


def _action_match_pose_and_fit(arm_obj, skin_objs, body_all_objs, garment_obj, inspection, params):
    g_info = inspection["garment_analysis"]
    fit = g_info["proposed_fit"]
    target_arm_deg = float(params.get("target_arm_deg", fit["target_arm_abduction_deg"]))
    baked_deg = _pose_and_bake_armature_rest(arm_obj, body_all_objs, inspection["skeleton_map"], target_arm_deg)

    # Apply fit transform onto garment and bake into vertex coordinates cleanly
    scale = [s * float(params.get("scale_mul", 1.0)) for s in fit["scale"]]
    trans = Vector(fit["translation"]) + Vector((0.0, 0.0, float(params.get("z_offset", 0.0))))

    garment_obj.scale = tuple(scale)
    garment_obj.location = trans
    bpy.ops.object.select_all(action="DESELECT")
    garment_obj.select_set(True)
    bpy.context.view_layer.objects.active = garment_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    has_pendulum = bool(g_info.get("has_offcenter_lower_pendulum"))
    margin_m = float(params.get("clearance_margin_m", 0.012))
    if not has_pendulum:
        margin_m = max(0.012, margin_m)
    inward_pull = float(params.get("inward_pull", 0.28 if g_info["garment_type"] == "waist_corset_belt" else 0.0))
    max_wrap_z = inspection["body_profile"]["landmarks_z"]["neck"] + 0.038
    center_y = float(inspection["body_profile"].get("hip_slice", {}).get("center_y", 0.0))
    skirt_margin_m = None if has_pendulum else float(params.get("skirt_margin_m", 0.046))

    pushed = _outward_clearance_wrap(
        skin_objs,
        garment_obj,
        margin_m=margin_m,
        smooth_passes=int(params.get("wrap_smooth_passes", 5)),
        inward_pull=inward_pull,
        arm_obj=arm_obj,
        skel_map=inspection["skeleton_map"],
        max_wrap_z=max_wrap_z,
        skirt_margin_m=skirt_margin_m,
        max_passes=5,
        max_step=0.075 if has_pendulum else 0.045,
        center_y=center_y,
    )

    split_chords = 0
    if not has_pendulum:
        nv1 = _subdivide_post_wrap_chords(skin_objs, garment_obj, max_edge_len=0.025, near_skin_m=0.11, max_passes=4)
        split_chords += len(nv1)
        pushed += _outward_clearance_wrap(
            skin_objs,
            garment_obj,
            margin_m=margin_m,
            smooth_passes=0,
            inward_pull=0.0,
            arm_obj=arm_obj,
            skel_map=inspection["skeleton_map"],
            max_wrap_z=max_wrap_z,
            skirt_margin_m=max(0.044, float(skirt_margin_m) - 0.002),
            max_passes=2,
            max_step=0.032,
            only_verts=nv1,
            center_y=center_y,
        )
        for edge_len, max_p, min_c, tgt_c, m_disp in (
            (0.028, 3, 0.016, 0.024, 0.060),
            (0.028, 3, 0.014, 0.018, 0.060),
            (0.028, 3, 0.012, 0.015, 0.060),
            (0.030, 4, 0.011, 0.016, 0.060),
            (0.030, 4, 0.010, 0.014, 0.060),
            (0.032, 4, 0.008, 0.012, 0.060),
            (0.034, 3, 0.006, 0.010, 0.018),
        ):
            nv_stage = _subdivide_post_wrap_chords(skin_objs, garment_obj, max_edge_len=edge_len, near_skin_m=0.11, max_passes=max_p)
            split_chords += len(nv_stage)
            _pop_subdivided_verts(
                skin_objs,
                garment_obj,
                arm_obj,
                inspection["skeleton_map"],
                center_y,
                max_wrap_z,
                nv_stage,
                min_clear=min_c,
                target_clear=tgt_c,
                max_disp=m_disp,
            )
        nv_final = _subdivide_post_wrap_chords(skin_objs, garment_obj, max_edge_len=0.035, near_skin_m=0.10, max_passes=2)
        split_chords += len(nv_final)

    return {
        "baked_arm_pose_delta_deg": baked_deg,
        "applied_scale": scale,
        "applied_translation": [round(x, 4) for x in trans],
        "split_long_chord_edges": split_chords,
        "clearance_pushed_verts": pushed,
    }


def _create_smooth_body_proxy(skin_objs, inflate_m=0.016):
    """Build a continuous inflated body proxy hull (reduces groin/armpit weight-transfer discontinuities)."""
    src = skin_objs[0]
    tmp_me = src.data.copy()
    proxy = bpy.data.objects.new("__ClothProxyHull", tmp_me)
    bpy.context.scene.collection.objects.link(proxy)
    proxy.matrix_world = src.matrix_world.copy()
    for vg in src.vertex_groups:
        proxy.vertex_groups.new(name=vg.name)
    for v in src.data.vertices:
        for g in v.groups:
            proxy.vertex_groups[g.group].add([v.index], g.weight, "REPLACE")

    disp = proxy.modifiers.new("Inflate", "DISPLACE")
    disp.strength = inflate_m
    disp.mid_level = 0.0
    sm = proxy.modifiers.new("Smooth", "SMOOTH")
    sm.factor = 0.55
    sm.iterations = 4
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.object.modifier_apply(modifier=disp.name)
    bpy.ops.object.modifier_apply(modifier=sm.name)
    return proxy


def _action_bind_and_transfer_weights(arm_obj, skin_objs, garment_obj, params):
    method = params.get("method", "POLYINTERP_NEAREST")
    preserve_vol = bool(params.get("preserve_volume", True))
    use_proxy = bool(params.get("use_proxy_hull", False))

    # Remove stale armature modifiers / vertex groups before clean re-transfer
    for m in list(garment_obj.modifiers):
        if m.type == "ARMATURE":
            garment_obj.modifiers.remove(m)
    garment_obj.vertex_groups.clear()

    src_obj = _create_smooth_body_proxy(skin_objs, inflate_m=float(params.get("proxy_inflate_m", 0.016))) if use_proxy else skin_objs[0]
    try:
        bpy.ops.object.select_all(action="DESELECT")
        garment_obj.select_set(True)
        src_obj.select_set(True)
        bpy.context.view_layer.objects.active = src_obj

        vert_mapping = "NEAREST" if method == "NEAREST" else (
            "POLYINTERP_VNORPROJ" if method == "POLYINTERP_VNORPROJ" else "POLYINTERP_NEAREST"
        )
        bpy.ops.object.data_transfer(
            use_reverse_transfer=False,
            data_type="VGROUP_WEIGHTS",
            vert_mapping=vert_mapping,
            layers_select_src="ALL",
            layers_select_dst="NAME",
            mix_mode="REPLACE",
        )
    finally:
        if use_proxy:
            me = src_obj.data
            bpy.data.objects.remove(src_obj, do_unlink=True)
            bpy.data.meshes.remove(me, do_unlink=True)

    # Ensure parent & Armature modifier cleanly bound; remove any stray duplicate .001 vertex groups created by DataTransfer
    valid_bones = {b.name for b in arm_obj.data.bones}
    for vg in list(garment_obj.vertex_groups):
        if vg.name not in valid_bones:
            garment_obj.vertex_groups.remove(vg)
    garment_obj.parent = arm_obj
    garment_obj.matrix_parent_inverse = arm_obj.matrix_world.inverted()
    arm_mod = garment_obj.modifiers.new("Armature", "ARMATURE")
    arm_mod.object = arm_obj
    arm_mod.use_deform_preserve_volume = preserve_vol

    return {
        "transfer_method": method,
        "used_proxy_hull": use_proxy,
        "preserve_volume": preserve_vol,
        "transferred_groups_count": len(garment_obj.vertex_groups),
    }


def _normalize_vertex_weights(mesh_obj, vertex_indices, weights_by_group: dict[int, float]):
    """Replace all bone weights on one vertex cleanly keeping max 4 normalized influences (always clears old groups!)."""
    non_zero = [(gid, float(w)) for gid, w in weights_by_group.items() if w > 1e-6]
    sig = [kv for kv in non_zero if kv[1] > 0.004]
    items = sig if sig else non_zero
    existing_gids = [g.group for g in mesh_obj.data.vertices[vertex_indices].groups]
    for gid in existing_gids:
        mesh_obj.vertex_groups[gid].remove([vertex_indices])
    if not items:
        return
    items.sort(key=lambda kv: -kv[1])
    items = items[:4]
    tot = sum(w for _, w in items)
    if tot <= 1e-8:
        return
    for gid, w in items:
        mesh_obj.vertex_groups[gid].add([vertex_indices], w / tot, "REPLACE")


def _sync_welded_weights(mesh_obj):
    """Ensure coincident UV-seam vertices share identical skin weights."""
    welded, _ = _build_welded_groups(mesh_obj.data)
    for vs in welded.values():
        if len(vs) <= 1:
            continue
        avg = defaultdict(float)
        for vi in vs:
            for g in mesh_obj.data.vertices[vi].groups:
                avg[g.group] += g.weight / len(vs)
        for vi in vs:
            _normalize_vertex_weights(mesh_obj, vi, dict(avg))


def _relax_high_gradient_edges(mesh_obj, allowed_verts: set[int], nbrs: dict[int, list[int]], max_grad_per_m: float = 3.2, passes: int = 240):
    """Fast pure-Python Gauss-Seidel weight relaxation eliminating dense-edge skinning spikes."""
    if not allowed_verts or passes <= 0:
        return 0
    mw = mesh_obj.matrix_world
    edge_pairs = []
    for e in mesh_obj.data.edges:
        a, b = e.vertices[0], e.vertices[1]
        if a in allowed_verts and b in allowed_verts:
            l0 = ((mw @ mesh_obj.data.vertices[a].co) - (mw @ mesh_obj.data.vertices[b].co)).length
            if l0 >= 0.0018:
                edge_pairs.append((a, b, l0))
    if not edge_pairs:
        return 0

    welded, _ = _build_welded_groups(mesh_obj.data)
    welded_groups = [vs for vs in welded.values() if len(vs) > 1 and all(vi in allowed_verts for vi in vs)]

    w_cache = {
        vi: {g.group: g.weight for g in mesh_obj.data.vertices[vi].groups if g.weight > 1e-4}
        for vi in allowed_verts
    }

    def _norm4(wdict: dict[int, float]) -> dict[int, float]:
        items = sorted(((k, v) for k, v in wdict.items() if v > 0.003), key=lambda kv: -kv[1])[:4]
        tot = sum(v for _, v in items)
        return {k: v / tot for k, v in items} if tot > 1e-8 else wdict

    modified_verts = set()
    for it_rel in range(passes):
        changed_pass = False
        for a, b, l0 in edge_pairs:
            wa = w_cache[a]
            wb = w_cache[b]
            keys = set(wa.keys()) | set(wb.keys())
            l1 = sum(abs(wa.get(k, 0.0) - wb.get(k, 0.0)) for k in keys)
            limit = max_grad_per_m * l0
            if l1 > limit:
                # Pull both endpoints toward midpoint cleanly so ||wa' - wb'|| <= limit (or reduced 65% per step)
                alpha = min(0.45, 0.5 * (1.0 - (limit / max(1e-8, l1))) * 0.85)
                na = {}
                nb = {}
                for k in keys:
                    va = wa.get(k, 0.0)
                    vb = wb.get(k, 0.0)
                    d = vb - va
                    na[k] = va + alpha * d
                    nb[k] = vb - alpha * d
                w_cache[a] = _norm4(na)
                w_cache[b] = _norm4(nb)
                modified_verts.add(a)
                modified_verts.add(b)
                changed_pass = True
        if welded_groups and (changed_pass or it_rel == 0):
            for vs in welded_groups:
                if any(vi in modified_verts for vi in vs):
                    avg = defaultdict(float)
                    for vi in vs:
                        for k, w in w_cache[vi].items():
                            avg[k] += w / len(vs)
                    nm = _norm4(dict(avg))
                    for vi in vs:
                        w_cache[vi] = nm.copy()
        if not changed_pass:
            break

    for vi in modified_verts:
        _normalize_vertex_weights(mesh_obj, vi, w_cache[vi])
    return len(modified_verts)


def _solve_scalar_harmonic(edge_w: dict[int, list[tuple[int, float]]], welded_groups: list[list[int]], pin_dict: dict[int, float], init_vals: list[float], passes: int = 160) -> list[float]:
    """Solve inverse-edge-length weighted Laplace harmonic field with welded seam synchronization."""
    val = [pin_dict.get(i, init_vals[i]) for i in range(len(init_vals))]
    for _ in range(passes):
        for vi in range(len(val)):
            if vi in pin_dict:
                continue
            ns = edge_w.get(vi)
            if ns:
                val[vi] = sum(wt * val[ni] for ni, wt in ns) / sum(wt for _, wt in ns)
        for vs in welded_groups:
            fvs = [v for v in vs if v not in pin_dict]
            if fvs:
                pvs = [pin_dict[v] for v in vs if v in pin_dict]
                vmean = (sum(pvs) / len(pvs)) if pvs else (sum(val[v] for v in vs) / len(vs))
                for v in fvs:
                    val[v] = vmean
    return val


def _relax_scalar_lipschitz(
    vals: list[float],
    edge_limits: list[tuple[int, int, float, bool, bool]],
    welded_groups: list[list[int]],
    locked_pins: set[int],
    passes: int = 90,
    both_gain: float = 0.48,
    one_gain: float = 1.5,
) -> list[float]:
    """Enforce edge-wise Lipschitz slope limits on a [0, 1] scalar field while preserving locked boundary pins."""
    for _ in range(passes):
        changed = False
        for a, b, lim, af, bf in edge_limits:
            d = vals[b] - vals[a]
            if abs(d) <= lim:
                continue
            corr = both_gain * (abs(d) - lim) * (1.0 if d > 0.0 else -1.0)
            if af and bf:
                vals[a] = max(0.0, min(1.0, vals[a] + corr))
                vals[b] = max(0.0, min(1.0, vals[b] - corr))
                changed = True
            elif af:
                vals[a] = max(0.0, min(1.0, vals[a] + corr * one_gain))
                changed = True
            elif bf:
                vals[b] = max(0.0, min(1.0, vals[b] - corr * one_gain))
                changed = True
        for vs in welded_groups:
            fvs = [v for v in vs if v not in locked_pins]
            if fvs:
                vmean = sum(vals[v] for v in vs) / len(vs)
                for v in fvs:
                    vals[v] = max(0.0, min(1.0, vmean))
        if not changed:
            break
    return vals


def _smooth_weights_region(mesh_obj, vert_set: set[int], nbrs: dict[int, list[int]], passes: int = 2, blend: float = 0.45):
    if not vert_set or passes <= 0:
        return
    for _ in range(passes):
        updates = {}
        for vi in vert_set:
            ns = [ni for ni in nbrs.get(vi, ()) if ni in vert_set]
            if not ns:
                continue
            cur = {g.group: g.weight for g in mesh_obj.data.vertices[vi].groups if g.weight > 1e-4}
            nbr_avg = defaultdict(float)
            for ni in ns:
                for g in mesh_obj.data.vertices[ni].groups:
                    nbr_avg[g.group] += g.weight / len(ns)
            merged = defaultdict(float)
            for k, w in cur.items():
                merged[k] += w * (1.0 - blend)
            for k, w in nbr_avg.items():
                merged[k] += w * blend
            updates[vi] = dict(merged)
        for vi, wmap in updates.items():
            _normalize_vertex_weights(mesh_obj, vi, wmap)


def _repair_weights_harmonic(
    arm_obj,
    garment_obj,
    inspection,
    params,
    *,
    nbrs,
    arm_segs,
    arm_ids,
    leg_l_ids,
    leg_r_ids,
    hips_id,
    id_uleg_l,
    id_lleg_l,
    id_uleg_r,
    id_lleg_r,
    id_uarm_l,
    id_larm_l,
    id_hand_l,
    id_uarm_r,
    id_larm_r,
    id_hand_r,
    uarm_l_h,
    larm_l_h,
    hand_l_h,
    uarm_r_h,
    larm_r_h,
    hand_r_h,
    smooth_torso_fallback,
):
    """Topology-native harmonic Laplace + edge-Lipschitz weight partition for full sleeved/unsleeved dresses."""
    bp = inspection["body_profile"]
    bz = bp["landmarks_z"]
    hips_z = bz["hips_bone"]
    waist_z = bz["waist_narrowest"]
    chest_z = bz["bust_peak"]
    shoulder_z = bz.get("shoulder", bz["neck"] - 0.036)
    knee_z = bz["knee"]
    has_sleeves = bool(inspection["garment_analysis"].get("has_sleeves", False))

    waist_width_x = float(bp.get("waist_slice", {}).get("width_x", 0.17))
    bust_width_x = float(bp.get("bust_slice", {}).get("width_x", 0.24))
    hip_cy = float(bp.get("hip_slice", {}).get("center_y", 0.0))
    torso_core_half_x = max(0.058, 0.34 * waist_width_x)
    arm_joint_half_x = 0.5 * abs(uarm_l_h.x - uarm_r_h.x) if (uarm_l_h and uarm_r_h) else 0.5 * float(bp.get("shoulder_joint_span", 0.175))
    arm_pin_x = max(0.160, arm_joint_half_x + 0.072)
    arm_root_z = max(chest_z + 0.030, shoulder_z - 0.084)
    rib_front_z = chest_z + 0.048
    collar_z = shoulder_z - 0.019

    bones_w = bp.get("bones_world", {})
    ul_x = abs(float(bones_w.get("upper_leg_l", {}).get("head", [0.0717, 0.0, 0.0])[0]))
    ur_x = abs(float(bones_w.get("upper_leg_r", {}).get("head", [-0.0717, 0.0, 0.0])[0]))
    leg_sx = max(0.75, min(1.35, 0.5 * (ul_x + ur_x) / 0.0717)) if (ul_x > 0.01 and ur_x > 0.01) else 1.0

    mw_g = garment_obj.matrix_world
    verts_w = [(mw_g @ v.co).copy() for v in garment_obj.data.vertices]
    raw_weights = {v.index: {g.group: g.weight for g in v.groups if g.weight > 1e-6} for v in garment_obj.data.vertices}
    welded, _ = _build_welded_groups(garment_obj.data)
    welded_groups = [vs for vs in welded.values() if len(vs) > 1]
    edge_w = defaultdict(list)
    for vi, nlist in nbrs.items():
        for ni in nlist:
            edge_w[vi].append((ni, 1.0 / max(0.002, (verts_w[vi] - verts_w[ni]).length)))

    # 1. Harmonic arm-to-torso membership field (bell sleeve BFS + Lipschitz armhole relaxation)
    if has_sleeves and params.get("spatial_sleeve_field", True) and arm_segs:
        bfs_sleeve = set()
        for sign in (1.0, -1.0):
            seeds = [
                vi for vi, pw in enumerate(verts_w)
                if pw.x * sign >= 0.185 and pw.z > hips_z - 0.10 and pw.z <= waist_z + 0.02
                and min((_dist_point_to_seg(pw, a, b) for a, b in arm_segs), default=9.9) < 0.13
            ]
            vis = set(seeds)
            q = list(seeds)
            while q:
                u = q.pop(0)
                for nb in nbrs.get(u, ()):
                    if nb not in vis:
                        pnb = verts_w[nb]
                        if pnb.x * sign >= 0.135 and pnb.z <= waist_z + 0.02:
                            vis.add(nb)
                            q.append(nb)
            bfs_sleeve |= vis

        pin_arm = {}
        for vi, pw in enumerate(verts_w):
            d_arm = min((_dist_point_to_seg(pw, a, b) for a, b in arm_segs), default=9.9)
            in_bell = vi in bfs_sleeve
            if (abs(pw.x) >= arm_pin_x and d_arm < 0.082 and pw.z >= arm_root_z) or (in_bell and abs(pw.x) >= 0.215 and d_arm < 0.155):
                pin_arm[vi] = 1.0
            elif (
                abs(pw.x) <= torso_core_half_x
                or (pw.y < hip_cy - 0.058 and abs(pw.x) <= 0.110 and pw.z < rib_front_z and not in_bell)
                or (pw.z <= hips_z + 0.025 and not in_bell)
                or (d_arm >= 0.162 and pw.z < waist_z + 0.035 and not in_bell)
            ):
                pin_arm[vi] = 0.0

        alpha_arm = _solve_scalar_harmonic(
            edge_w, welded_groups, pin_arm, [_smoothstep(torso_core_half_x, 0.195, abs(pw.x)) for pw in verts_w], passes=180
        )
        locked_arm = set(pin_arm.keys())
        arm_edge_limits = []
        for e in garment_obj.data.edges:
            a, b = e.vertices[0], e.vertices[1]
            if a not in locked_arm or b not in locked_arm:
                l0 = (verts_w[a] - verts_w[b]).length
                if l0 >= 0.0015:
                    arm_edge_limits.append((a, b, 4.6 * l0, a not in locked_arm, b not in locked_arm))
        _relax_scalar_lipschitz(alpha_arm, arm_edge_limits, welded_groups, locked_arm, passes=90, both_gain=0.48, one_gain=1.5)
    else:
        alpha_arm = [0.0] * len(verts_w)

    # 2. Harmonic L/R skirt separation field (crotch-aware Dirichlet boundary + stride-safe Lipschitz slope)
    has_skirt_legs = (id_uleg_l is not None and id_uleg_r is not None and hips_id is not None)
    if has_skirt_legs:
        pin_sk = {}
        fy_pin = hip_cy + 0.017
        by_pin = hip_cy - 0.013
        for vi, pw in enumerate(verts_w):
            if alpha_arm[vi] >= 0.15 or pw.z > hips_z + 0.06:
                continue
            if pw.x >= 0.168 * leg_sx or (pw.x >= 0.055 * leg_sx and pw.y <= fy_pin) or (pw.z >= 0.80 and pw.x >= 0.065 * leg_sx) or (pw.z >= 0.56 and pw.x >= 0.118 * leg_sx) or (pw.z >= 0.48 and pw.x >= 0.140 * leg_sx):
                pin_sk[vi] = 1.0
            elif pw.x <= -0.162 * leg_sx or (pw.x <= -0.055 * leg_sx and pw.y >= by_pin) or (pw.z >= 0.80 and pw.x <= -0.065 * leg_sx) or (pw.z >= 0.56 and pw.x <= -0.118 * leg_sx) or (pw.z >= 0.48 and pw.x <= -0.136 * leg_sx):
                pin_sk[vi] = 0.0

        s_left_harm = _solve_scalar_harmonic(
            edge_w, welded_groups, pin_sk, [max(0.0, min(1.0, (pw.x + 0.06) / 0.18)) for pw in verts_w], passes=280
        )
        pin_sk_hard = {
            vi for vi in pin_sk
            if (verts_w[vi].z >= 0.56 and (
                (verts_w[vi].x >= 0.065 * leg_sx and verts_w[vi].y <= hip_cy + 0.012)
                or (verts_w[vi].x <= -0.065 * leg_sx and verts_w[vi].y >= hip_cy - 0.008)
                or verts_w[vi].x >= 0.118 * leg_sx
                or verts_w[vi].x <= -0.118 * leg_sx
            ))
            or verts_w[vi].x >= 0.195 * leg_sx
            or verts_w[vi].x <= -0.195 * leg_sx
        }
        sk_edge_limits = []
        for e in garment_obj.data.edges:
            a, b = e.vertices[0], e.vertices[1]
            if alpha_arm[a] < 0.15 and alpha_arm[b] < 0.15 and verts_w[a].z <= hips_z + 0.06 and verts_w[b].z <= hips_z + 0.06:
                l0 = max(0.0015, (verts_w[a] - verts_w[b]).length)
                zm = 0.5 * (verts_w[a].z + verts_w[b].z)
                lim = (2.85 + 3.0 * _smoothstep(0.55, 0.85, zm)) * l0
                sk_edge_limits.append((a, b, lim, a not in pin_sk_hard, b not in pin_sk_hard))
        _relax_scalar_lipschitz(s_left_harm, sk_edge_limits, welded_groups, pin_sk_hard, passes=520, both_gain=0.49, one_gain=1.6)
    else:
        s_left_harm = [0.5] * len(verts_w)

    # 3. Compose per-vertex skin weights cleanly inside 4-bone budget
    purged_arm = 0
    sleeve_projected = 0
    skirt_analytical = 0
    torso_relax_verts = set()
    arm_set_verts = set()
    skirt_set_verts = set()

    for vi, pw in enumerate(verts_w):
        a_arm = alpha_arm[vi]
        wmap = dict(raw_weights[vi])
        if a_arm < 0.03:
            if any(gid in arm_ids for gid in wmap):
                rem = {gid: w for gid, w in wmap.items() if gid not in arm_ids}
                wmap = rem if sum(rem.values()) > 0.25 else smooth_torso_fallback(pw.z)
                purged_arm += 1
            if pw.z >= hips_z + 0.025 and a_arm < 0.01:
                torso_relax_verts.add(vi)
            elif pw.z >= collar_z and a_arm < 0.03:
                torso_relax_verts.add(vi)
        else:
            arm_set_verts.add(vi)
            if pw.z >= collar_z and a_arm < 0.12:
                torso_relax_verts.add(vi)
            is_left = pw.x >= 0.0
            sh_p, el_p, ha_p = (uarm_l_h, larm_l_h, hand_l_h) if is_left else (uarm_r_h, larm_r_h, hand_r_h)
            id_ua, id_la, id_ha = (id_uarm_l, id_larm_l, id_hand_l) if is_left else (id_uarm_r, id_larm_r, id_hand_r)
            if sh_p and el_p and ha_p and id_ua is not None and id_la is not None:
                ax = ha_p - sh_p
                u_arm = max(0.0, min(1.0, (pw - sh_p).dot(ax) / max(1e-6, ax.length_squared)))
                u_elbow = max(0.2, min(0.8, (el_p - sh_p).dot(ax) / max(1e-6, ax.length_squared)))
                eb = _smoothstep(u_elbow - 0.12, u_elbow + 0.12, u_arm)
                hb = _smoothstep(0.88, 1.0, u_arm) * 0.12 if id_ha is not None else 0.0
                sw = {}
                if (1.0 - eb) > 0.005:
                    sw[id_ua] = 1.0 - eb
                if eb * (1.0 - hb) > 0.005:
                    sw[id_la] = eb * (1.0 - hb)
                if hb > 0.005 and id_ha is not None:
                    sw[id_ha] = hb
                if a_arm < 0.28 and len(sw) > 1:
                    sw = {max(sw, key=sw.get): 1.0}
                tc = {gid: w for gid, w in wmap.items() if gid not in leg_l_ids and gid not in leg_r_ids and gid not in arm_ids}
                if sum(tc.values()) < 0.25:
                    tc = smooth_torso_fallback(pw.z)
                max_tc = max(1, 4 - len(sw))
                tc = dict(sorted(tc.items(), key=lambda kv: -kv[1])[:max_tc])
                tsum = sum(tc.values()) or 1.0
                merged = defaultdict(float)
                for gid, w in tc.items():
                    merged[gid] += (w / tsum) * (1.0 - a_arm)
                for gid, w in sw.items():
                    merged[gid] += w * a_arm
                wmap = dict(merged)
                sleeve_projected += 1

        if has_skirt_legs and a_arm < 0.15 and pw.z <= hips_z + 0.045:
            skirt_set_verts.add(vi)
            t_drop = max(0.0, min(1.0, (hips_z + 0.025 - pw.z) / 0.215))
            hips_w = 0.92 * (1.0 - _smoothstep(0.0, 1.0, t_drop))
            if hips_w < 0.005:
                hips_w = 0.0
            s_left = s_left_harm[vi]
            s_right = 1.0 - s_left
            u_knee = 0.98 * (1.0 - _smoothstep(knee_z - 0.075, knee_z + 0.075, pw.z)) if (id_lleg_l is not None and id_lleg_r is not None) else 0.0
            if u_knee < 0.005:
                u_knee = 0.0
            leg_tot = max(0.0, 1.0 - hips_w)
            fw = {}
            if hips_w > 0.0:
                fw[hips_id] = hips_w
            for gid, val in (
                (id_uleg_l, leg_tot * s_left * (1.0 - u_knee)),
                (id_lleg_l, leg_tot * s_left * u_knee),
                (id_uleg_r, leg_tot * s_right * (1.0 - u_knee)),
                (id_lleg_r, leg_tot * s_right * u_knee),
            ):
                if gid is not None and val > 0.001:
                    fw[gid] = val
            bi = (1.0 - _smoothstep(hips_z - 0.005, hips_z + 0.045, pw.z)) * (1.0 - a_arm / 0.15)
            wsum = sum(wmap.values()) or 1.0
            ms = defaultdict(float)
            for gid, w in wmap.items():
                ms[gid] += (w / wsum) * (1.0 - bi)
            for gid, w in fw.items():
                ms[gid] += w * bi
            wmap = dict(ms)
            skirt_analytical += 1

        _normalize_vertex_weights(garment_obj, vi, wmap)

    relaxed_count = _relax_high_gradient_edges(garment_obj, torso_relax_verts, nbrs, max_grad_per_m=8.0, passes=35)
    _sync_welded_weights(garment_obj)
    for v in garment_obj.data.vertices:
        _normalize_vertex_weights(garment_obj, v.index, {g.group: g.weight for g in v.groups})

    return {
        "purged_arm_skirt_verts": purged_arm,
        "sleeve_projected_verts": sleeve_projected,
        "skirt_analytical_verts": skirt_analytical,
        "bridged_leg_verts": skirt_analytical,
        "stiffened_pendulum_verts": 0,
        "smoothed_skirt_verts": len(skirt_set_verts),
        "gradient_relaxed_verts": relaxed_count,
    }


def _action_repair_weight_anomalies(arm_obj, skin_objs, garment_obj, inspection, params):
    skel = inspection["skeleton_map"]
    slots = skel["slots"]
    bones_w = inspection["body_profile"]["bones_world"]
    bz = inspection["body_profile"]["landmarks_z"]
    arm_bones = set(skel["arm_bones_all"])
    leg_l_bones = set(skel["leg_l_bones"])
    leg_r_bones = set(skel["leg_r_bones"])
    vg_by_name = {vg.name: vg.index for vg in garment_obj.vertex_groups}
    vg_by_id = {vg.index: vg.name for vg in garment_obj.vertex_groups}
    arm_ids = {vg_by_name[n] for n in arm_bones if n in vg_by_name}
    leg_l_ids = {vg_by_name[n] for n in leg_l_bones if n in vg_by_name}
    leg_r_ids = {vg_by_name[n] for n in leg_r_bones if n in vg_by_name}
    hips_id = vg_by_name.get(slots.get("hips", ""))
    spine_id = vg_by_name.get(slots.get("spine", ""))

    # Build deformed rest bone segments using joint-to-child-joint spans (VRM tails are often short stubs!)
    mw_arm = arm_obj.matrix_world
    def _bseg(slot_a, slot_b=None):
        ba = arm_obj.data.bones.get(slots.get(slot_a, ""))
        if not ba:
            return None
        pa = mw_arm @ ba.head_local
        if slot_b and slots.get(slot_b) in arm_obj.data.bones:
            pb = mw_arm @ arm_obj.data.bones[slots[slot_b]].head_local
        else:
            pb = mw_arm @ ba.tail_local
        return (pa, pb)

    arm_segs = [s for s in (_bseg("upper_arm_l", "lower_arm_l"), _bseg("lower_arm_l", "hand_l"), _bseg("upper_arm_r", "lower_arm_r"), _bseg("lower_arm_r", "hand_r")) if s]
    torso_leg_segs = [
        s for s in (
            _bseg("hips", "chest"),
            _bseg("chest", "neck"),
            _bseg("upper_leg_l", "lower_leg_l"),
            _bseg("lower_leg_l", "foot_l"),
            _bseg("upper_leg_r", "lower_leg_r"),
            _bseg("lower_leg_r", "foot_r"),
        ) if s
    ]

    mw_g = garment_obj.matrix_world
    hips_z = bz["hips_bone"]
    waist_z = bz["waist_narrowest"]
    knee_z = bz["knee"]
    ankle_z = bz["ankle"]
    has_sleeves = bool(inspection["garment_analysis"].get("has_sleeves", False))
    armhole_half_x = max(0.140, 0.5 * inspection["body_profile"]["waist_slice"]["width_x"] + 0.055)
    purged_arm = 0
    bridged_leg = 0
    sleeve_projected = 0
    skirt_analytical = 0
    skirt_verts = set()
    corset_waist_verts = set()

    def _bpt(slot_k):
        bname = slots.get(slot_k, "")
        return (mw_arm @ arm_obj.data.bones[bname].head_local) if bname in arm_obj.data.bones else None

    uarm_l_h, larm_l_h, hand_l_h = _bpt("upper_arm_l"), _bpt("lower_arm_l"), _bpt("hand_l")
    uarm_r_h, larm_r_h, hand_r_h = _bpt("upper_arm_r"), _bpt("lower_arm_r"), _bpt("hand_r")
    id_uarm_l = vg_by_name.get(slots.get("upper_arm_l", ""))
    id_larm_l = vg_by_name.get(slots.get("lower_arm_l", ""))
    id_hand_l = vg_by_name.get(slots.get("hand_l", ""))
    id_uarm_r = vg_by_name.get(slots.get("upper_arm_r", ""))
    id_larm_r = vg_by_name.get(slots.get("lower_arm_r", ""))
    id_hand_r = vg_by_name.get(slots.get("hand_r", ""))
    id_uleg_l = vg_by_name.get(slots.get("upper_leg_l", ""))
    id_lleg_l = vg_by_name.get(slots.get("lower_leg_l", ""))
    id_uleg_r = vg_by_name.get(slots.get("upper_leg_r", ""))
    id_lleg_r = vg_by_name.get(slots.get("lower_leg_r", ""))

    id_chest = vg_by_name.get(slots.get("chest", ""))
    id_uchest = vg_by_name.get(slots.get("upper_chest", ""))
    chest_z = bz["bust_peak"]

    def _smooth_torso_fallback(pz: float) -> dict[int, float]:
        """Smooth vertical torso bone gradient (Hips -> Spine -> Chest -> UpperChest) to avoid cliff jumps."""
        anchors = []
        if hips_id is not None:
            anchors.append((hips_z, hips_id))
        if spine_id is not None:
            anchors.append((waist_z - 0.05, spine_id))
        if id_chest is not None:
            anchors.append((waist_z + 0.05, id_chest))
        if id_uchest is not None:
            anchors.append((chest_z + 0.04, id_uchest))
        if not anchors:
            return {hips_id: 1.0} if hips_id is not None else {}
        if pz <= anchors[0][0]:
            return {anchors[0][1]: 1.0}
        if pz >= anchors[-1][0]:
            return {anchors[-1][1]: 1.0}
        for k in range(len(anchors) - 1):
            z0, b0 = anchors[k]
            z1, b1 = anchors[k + 1]
            if z0 <= pz <= z1:
                t = (pz - z0) / max(1e-4, z1 - z0)
                return {b0: 1.0 - t, b1: t} if b0 != b1 else {b0: 1.0}
        return {anchors[0][1]: 1.0}

    nbrs = _build_mesh_neighbors(garment_obj.data)
    if not inspection["garment_analysis"].get("has_offcenter_lower_pendulum"):
        return _repair_weights_harmonic(
            arm_obj,
            garment_obj,
            inspection,
            params,
            nbrs=nbrs,
            arm_segs=arm_segs,
            arm_ids=arm_ids,
            leg_l_ids=leg_l_ids,
            leg_r_ids=leg_r_ids,
            hips_id=hips_id,
            id_uleg_l=id_uleg_l,
            id_lleg_l=id_lleg_l,
            id_uleg_r=id_uleg_r,
            id_lleg_r=id_lleg_r,
            id_uarm_l=id_uarm_l,
            id_larm_l=id_larm_l,
            id_hand_l=id_hand_l,
            id_uarm_r=id_uarm_r,
            id_larm_r=id_larm_r,
            id_hand_r=id_hand_r,
            uarm_l_h=uarm_l_h,
            larm_l_h=larm_l_h,
            hand_l_h=hand_l_h,
            uarm_r_h=uarm_r_h,
            larm_r_h=larm_r_h,
            hand_r_h=hand_r_h,
            smooth_torso_fallback=_smooth_torso_fallback,
        )

    bfs_sleeve_verts = set()
    if has_sleeves and params.get("spatial_sleeve_field", True):
        # Flood wide bell-sleeve shells topologically below armpit/waist so hanging cuffs never split at Euclidean radius
        for side_sign in (1.0, -1.0):
            seeds = []
            for v in garment_obj.data.vertices:
                pw = mw_g @ v.co
                if pw.x * side_sign <= 0.0:
                    continue
                d_arm = min((_dist_point_to_seg(pw, a, b) for a, b in arm_segs), default=9.9)
                if d_arm < 0.12 and pw.z > hips_z - 0.09 and pw.z <= waist_z - 0.008 and abs(pw.x) > armhole_half_x + 0.03:
                    seeds.append(v.index)
            vis = set(seeds)
            q = list(seeds)
            while q:
                u = q.pop(0)
                for nb in nbrs.get(u, ()):
                    if nb not in vis:
                        pnb = mw_g @ garment_obj.data.vertices[nb].co
                        if pnb.x * side_sign > 0.0 and pnb.z <= waist_z - 0.008 and abs(pnb.x) > armhole_half_x - 0.025:
                            vis.add(nb)
                            q.append(nb)
            bfs_sleeve_verts |= vis

    sleeve_verts = set()
    for v in garment_obj.data.vertices:
        pw = mw_g @ v.co
        d_arm = min((_dist_point_to_seg(pw, a, b) for a, b in arm_segs), default=9.9)
        d_core = min((_dist_point_to_seg(pw, a, b) for a, b in torso_leg_segs), default=9.9)
        in_bfs_sleeve = v.index in bfs_sleeve_verts
        is_sleeve_zone = has_sleeves and (
            (in_bfs_sleeve and (pw.z <= waist_z - 0.030 or abs(pw.x) >= armhole_half_x + 0.015))
            or (pw.z > hips_z - 0.078 and d_arm < 0.154 and abs(pw.x) > armhole_half_x + 0.015 and (d_arm < d_core + 0.022))
        )
        is_sleeve_blend = (
            has_sleeves
            and (not is_sleeve_zone)
            and (
                (in_bfs_sleeve and abs(pw.x) < armhole_half_x + 0.015)
                or (
                    pw.z > waist_z - 0.030
                    and d_arm < 0.158
                    and (d_arm < d_core + 0.045)
                    and 0.095 <= abs(pw.x) <= armhole_half_x + 0.022
                )
            )
        )

        wmap = {g.group: g.weight for g in v.groups if g.weight > 1e-6}
        changed = False

        # 1A. Global Arm Purge on SLEEVELESS garments (prevents elbow/wrist spikes & noisy fallback jumps!)
        if not has_sleeves and params.get("purge_arm_skirt_crosstalk", True):
            if any(gid in arm_ids for gid in wmap):
                rem = {gid: w for gid, w in wmap.items() if gid not in arm_ids}
                if sum(rem.values()) < 0.35:
                    wmap = _smooth_torso_fallback(pw.z)
                else:
                    wmap = rem
                purged_arm += 1
                changed = True
        # 1B. Sleek Arm-Capsule Projection on SLEEVE vertices (covers full Hanfu bell sleeve cuff cleanly!)
        elif (is_sleeve_zone or is_sleeve_blend) and params.get("spatial_sleeve_field", True):
            sleeve_verts.add(v.index)
            is_left = pw.x >= 0.0
            sh_p, el_p, ha_p = (uarm_l_h, larm_l_h, hand_l_h) if is_left else (uarm_r_h, larm_r_h, hand_r_h)
            id_ua, id_la, id_ha = (id_uarm_l, id_larm_l, id_hand_l) if is_left else (id_uarm_r, id_larm_r, id_hand_r)
            if sh_p and el_p and ha_p and id_ua is not None and id_la is not None:
                ax = ha_p - sh_p
                ax_len2 = max(1e-6, ax.length_squared)
                u_arm = max(0.0, min(1.0, (pw - sh_p).dot(ax) / ax_len2))
                u_elbow = max(0.2, min(0.8, (el_p - sh_p).dot(ax) / ax_len2))
                eb = _smoothstep(u_elbow - 0.12, u_elbow + 0.12, u_arm)
                hb = _smoothstep(0.88, 1.0, u_arm) * 0.12 if id_ha is not None else 0.0
                sleeve_w = {}
                if (1.0 - eb) > 0.005:
                    sleeve_w[id_ua] = 1.0 - eb
                if eb * (1.0 - hb) > 0.005:
                    sleeve_w[id_la] = eb * (1.0 - hb)
                if hb > 0.005 and id_ha is not None:
                    sleeve_w[id_ha] = hb

                alpha = 1.0 if is_sleeve_zone else _smoothstep(0.095, armhole_half_x + 0.020, abs(pw.x))
                torso_clean = {gid: w for gid, w in wmap.items() if gid not in leg_l_ids and gid not in leg_r_ids and gid != hips_id and gid not in arm_ids}
                if not torso_clean:
                    torso_clean = _smooth_torso_fallback(pw.z) if alpha < 0.99 else sleeve_w
                tc_sum = sum(torso_clean.values()) or 1.0
                merged_sl = defaultdict(float)
                for gid, w in torso_clean.items():
                    merged_sl[gid] += (w / tc_sum) * (1.0 - alpha)
                for gid, w in sleeve_w.items():
                    merged_sl[gid] += w * alpha
                wmap = dict(merged_sl)
                sleeve_projected += 1
                changed = True
        elif params.get("purge_arm_skirt_crosstalk", True) and not (is_sleeve_zone or is_sleeve_blend):
            if any(gid in arm_ids for gid in wmap):
                rem = {gid: w for gid, w in wmap.items() if gid not in arm_ids}
                wmap = _smooth_torso_fallback(pw.z) if sum(rem.values()) < 0.35 else rem
                purged_arm += 1
                changed = True

        # 2. Smooth Skirt Weight Assignment (Analytical Multi-Layer Field OR Cross-Leg Bridge)
        is_skirt_domain = (not is_sleeve_zone) and (not is_sleeve_blend) and (pw.z <= hips_z + 0.055)
        if is_skirt_domain:
            skirt_verts.add(v.index)
            if params.get("skirt_analytical_field", False) and not inspection["garment_analysis"].get("has_offcenter_lower_pendulum"):
                if id_uleg_l is not None and id_uleg_r is not None and hips_id is not None:
                    u_drop = max(0.0, min(1.0, (hips_z + 0.025 - pw.z) / 0.16))
                    s_hip = _smoothstep(0.0, 1.0, u_drop)
                    hips_w = 0.85 * (1.0 - s_hip) + 0.02
                    s_left = max(0.0, min(1.0, (pw.x + 0.080) / 0.160))
                    s_right = 1.0 - s_left
                    u_knee = (1.0 - _smoothstep(knee_z - 0.14, knee_z + 0.14, pw.z)) if (id_lleg_l is not None and id_lleg_r is not None) else 0.0
                    # Strict <= 4 bones per vertex everywhere (never activate both LowerLeg_L and LowerLeg_R together)
                    lo_scale_l = _smoothstep(0.50, 0.72, s_left)
                    lo_scale_r = _smoothstep(0.50, 0.72, s_right)
                    lo_frac_l = 0.90 * u_knee * lo_scale_l
                    lo_frac_r = 0.90 * u_knee * lo_scale_r
                    leg_tot = max(0.0, 1.0 - hips_w)
                    field_w = {hips_id: hips_w}
                    if leg_tot * s_left * (1.0 - lo_frac_l) > 0.004:
                        field_w[id_uleg_l] = leg_tot * s_left * (1.0 - lo_frac_l)
                    if leg_tot * s_left * lo_frac_l > 0.004 and id_lleg_l is not None:
                        field_w[id_lleg_l] = leg_tot * s_left * lo_frac_l
                    if leg_tot * s_right * (1.0 - lo_frac_r) > 0.004:
                        field_w[id_uleg_r] = leg_tot * s_right * (1.0 - lo_frac_r)
                    if leg_tot * s_right * lo_frac_r > 0.004 and id_lleg_r is not None:
                        field_w[id_lleg_r] = leg_tot * s_right * lo_frac_r
                    blend_in = 1.0 - _smoothstep(hips_z - 0.005, hips_z + 0.055, pw.z)
                    w_sum = sum(wmap.values()) or 1.0
                    merged_sk = defaultdict(float)
                    for gid, w in wmap.items():
                        merged_sk[gid] += (w / w_sum) * (1.0 - blend_in)
                    for gid, w in field_w.items():
                        merged_sk[gid] += w * blend_in
                    wmap = dict(merged_sk)
                    skirt_analytical += 1
                    changed = True
            elif params.get("blend_skirt_leg_bridge", True):
                hip_floor = float(params.get("skirt_hips_floor", 0.30))
                center_band = float(params.get("cross_leg_blend_band", 0.075))
                if not inspection["garment_analysis"].get("has_offcenter_lower_pendulum"):
                    t_z = _smoothstep(hips_z - 0.16, hips_z + 0.01, pw.z)
                    loc_hip_floor = 0.04 + (hip_floor - 0.04) * t_z
                    # Purge opposite-leg weights smoothly from x=0 out to |x|=0.075, redirecting onto same-side UpperLeg!
                    purge_opp = _smoothstep(0.015, center_band, abs(pw.x))
                    if pw.x >= 0.0 and purge_opp > 0.02:
                        bad_r = sum(wmap.get(gid, 0.0) for gid in leg_r_ids)
                        if bad_r > 0.01:
                            drop_r = bad_r * purge_opp
                            scale_r = (bad_r - drop_r) / bad_r
                            for gid in leg_r_ids:
                                if gid in wmap:
                                    wmap[gid] *= scale_r
                            target_l = id_uleg_l if id_uleg_l is not None else hips_id
                            if target_l is not None:
                                wmap[target_l] = wmap.get(target_l, 0.0) + drop_r
                            bridged_leg += 1
                            changed = True
                    elif pw.x < 0.0 and purge_opp > 0.02:
                        bad_l = sum(wmap.get(gid, 0.0) for gid in leg_l_ids)
                        if bad_l > 0.01:
                            drop_l = bad_l * purge_opp
                            scale_l = (bad_l - drop_l) / bad_l
                            for gid in leg_l_ids:
                                if gid in wmap:
                                    wmap[gid] *= scale_l
                            target_r = id_uleg_r if id_uleg_r is not None else hips_id
                            if target_r is not None:
                                wmap[target_r] = wmap.get(target_r, 0.0) + drop_l
                            bridged_leg += 1
                            changed = True
                    if hips_id is not None and wmap.get(hips_id, 0.0) > loc_hip_floor and pw.z < hips_z - 0.10:
                        # Release excessive hips lag below thigh root so thighs/knees track deep_squat & stride cleanly
                        excess = (wmap[hips_id] - loc_hip_floor) * 0.70
                        wmap[hips_id] -= excess
                        target_side = (id_uleg_l if pw.x >= 0.0 else id_uleg_r) or hips_id
                        wmap[target_side] = wmap.get(target_side, 0.0) + excess
                        changed = True
                elif abs(pw.x) < center_band and hips_id is not None:
                    new_map = {}
                    for gid, w in wmap.items():
                        new_map[gid] = w * (1.0 - max(0.0, hip_floor - wmap.get(hips_id, 0.0)))
                    new_map[hips_id] = max(new_map.get(hips_id, 0.0), hip_floor)
                    wmap = new_map
                    bridged_leg += 1
                    changed = True
                elif pw.x > center_band:
                    bad_r = sum(wmap.get(gid, 0.0) for gid in leg_r_ids)
                    if bad_r > 0.04:
                        for gid in leg_r_ids:
                            wmap.pop(gid, None)
                        if hips_id is not None:
                            wmap[hips_id] = wmap.get(hips_id, 0.0) + bad_r
                        bridged_leg += 1
                        changed = True
                elif pw.x < -center_band:
                    bad_l = sum(wmap.get(gid, 0.0) for gid in leg_l_ids)
                    if bad_l > 0.04:
                        for gid in leg_l_ids:
                            wmap.pop(gid, None)
                        if hips_id is not None:
                            wmap[hips_id] = wmap.get(hips_id, 0.0) + bad_l
                        bridged_leg += 1
                        changed = True

        if pw.z >= hips_z - 0.03 and not is_sleeve_zone:
            corset_waist_verts.add(v.index)

        if changed:
            _normalize_vertex_weights(garment_obj, v.index, wmap)

    # 3. Stiffen lower hanging accessories / pendulums (corset ribbon + bell + tassel) with smooth 4.5cm attach hinge
    stiffened_pendants = 0
    pendulum_verts_set = set()
    if params.get("stiffen_pendulum", True) and inspection["garment_analysis"].get("has_offcenter_lower_pendulum"):
        fit = inspection["garment_analysis"]["proposed_fit"]
        cuff_z_world = inspection["garment_analysis"]["enclosing_band_z"][0] * fit["scale"][2] + fit["translation"][2]
        cutoff_z = min(hips_z - 0.025, cuff_z_world + 0.025)
        hinge_top = cutoff_z + 0.045
        rigid_w = {}
        if hips_id is not None:
            rigid_w[hips_id] = 0.88
        if spine_id is not None:
            rigid_w[spine_id] = 0.12
        if rigid_w:
            for v in garment_obj.data.vertices:
                pz = (mw_g @ v.co).z
                if pz < cutoff_z:
                    _normalize_vertex_weights(garment_obj, v.index, rigid_w)
                    pendulum_verts_set.add(v.index)
                    stiffened_pendants += 1
                elif pz < hinge_top:
                    # Linearly blend transferred weight -> rigid_w over 4.5cm hinge band (zero cliff seam!)
                    alpha = (hinge_top - pz) / max(1e-4, hinge_top - cutoff_z)
                    cur_w = {g.group: g.weight for g in v.groups if g.weight > 1e-4}
                    blended = defaultdict(float)
                    for gid, w in cur_w.items():
                        blended[gid] += w * (1.0 - alpha)
                    for gid, w in rigid_w.items():
                        blended[gid] += w * alpha
                    _normalize_vertex_weights(garment_obj, v.index, dict(blended))
                    pendulum_verts_set.add(v.index)

    # 4. Smooth weights across waist cuff and (only if non-analytical) skirt; preserve analytical C1 continuity
    smooth_passes = int(params.get("weight_smooth_passes", 3))
    skirt_verts -= pendulum_verts_set
    corset_waist_verts -= pendulum_verts_set
    corset_waist_verts -= sleeve_verts
    if skirt_verts and smooth_passes > 0 and not (params.get("skirt_analytical_field", False) and not inspection["garment_analysis"].get("has_offcenter_lower_pendulum")):
        _smooth_weights_region(garment_obj, skirt_verts, nbrs, passes=smooth_passes, blend=0.42)
    if corset_waist_verts and smooth_passes > 0:
        _smooth_weights_region(garment_obj, corset_waist_verts, nbrs, passes=max(3, smooth_passes + 1), blend=0.45)

    # 4B. Fast Gauss-Seidel Edge-Lipschitz gradient relaxation across skirt & upper armhole partitions
    relaxed_edges_count = 0
    if not inspection["garment_analysis"].get("has_offcenter_lower_pendulum"):
        core_set = {v.index for v in garment_obj.data.vertices} - sleeve_verts - pendulum_verts_set
        relaxed_edges_count += _relax_high_gradient_edges(garment_obj, core_set, nbrs, max_grad_per_m=3.1, passes=260)
        if sleeve_verts:
            # Include immediate upper-chest armhole seam band (d_arm < 0.165, z >= waist_z) so shoulder transition clamps cleanly!
            upper_armhole_band = {
                v.index
                for v in garment_obj.data.vertices
                if (mw_g @ v.co).z >= waist_z - 0.015
                and min((_dist_point_to_seg(mw_g @ v.co, a, b) for a, b in arm_segs), default=9.9) < 0.158
            }
            relaxed_edges_count += _relax_high_gradient_edges(garment_obj, sleeve_verts | upper_armhole_band, nbrs, max_grad_per_m=3.8, passes=180)

    # 5. Synchronize coincident UV-seam twins & enforce glTF contract (<= 4 bones/vert, normalized = 1.0)
    _sync_welded_weights(garment_obj)
    for v in garment_obj.data.vertices:
        wmap = {g.group: g.weight for g in v.groups}
        _normalize_vertex_weights(garment_obj, v.index, wmap)

    return {
        "purged_arm_skirt_verts": purged_arm,
        "sleeve_projected_verts": sleeve_projected,
        "skirt_analytical_verts": skirt_analytical,
        "bridged_leg_verts": bridged_leg,
        "stiffened_pendulum_verts": stiffened_pendants,
        "smoothed_skirt_verts": len(skirt_verts),
        "gradient_relaxed_verts": relaxed_edges_count,
    }


def _export_rigged_bundle(arm_obj, skin_objs, face_objs, garment_obj, out_blend: Path, out_glb: Path):
    keep_names = [obj.name for obj in [arm_obj, *skin_objs, *face_objs, garment_obj]]
    remove_export_junk()
    bpy.ops.object.select_all(action="DESELECT")
    keep = [bpy.context.scene.objects[name] for name in keep_names if name in bpy.context.scene.objects]
    for o in keep:
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    out_blend.parent.mkdir(parents=True, exist_ok=True)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_skins=True,
    )


def main():
    args = _parse_args()
    actions = json.loads(Path(args.actions_json).read_text(encoding="utf-8-sig"))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    load_scene_file(Path(args.body))
    arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    body_meshes = strip_bone_shape_junk([o for o in bpy.data.objects if o.type == "MESH"])
    skin_objs = [o for o in body_meshes if "skin" in o.name.lower() or "body" in o.name.lower()] or body_meshes
    face_objs = [o for o in body_meshes if "face" in o.name.lower() and o not in skin_objs]

    before = {o.as_pointer() for o in bpy.data.objects}
    import_asset_additive(Path(args.garment))
    garment_meshes = [o for o in bpy.data.objects if o.as_pointer() not in before and o.type == "MESH"]
    garment_obj = max(garment_meshes, key=lambda o: len(o.data.vertices))
    garment_obj.name = "Fitted_Garment"

    skel_map = map_humanoid_skeleton(arm_obj)
    body_profile = extract_body_profile(arm_obj, skin_objs, skel_map)
    garment_analysis = analyze_garment_and_propose_fit(garment_obj, body_profile)
    inspection = {
        "skeleton_map": skel_map,
        "body_profile": body_profile,
        "garment_analysis": garment_analysis,
    }

    action_logs = []
    for step in actions:
        act = step.get("action")
        params = step.get("params", {})
        if act == "match_pose_and_fit":
            res = _action_match_pose_and_fit(arm_obj, skin_objs, body_meshes, garment_obj, inspection, params)
            # Refresh profile after arm rest pose bake!
            inspection["body_profile"] = extract_body_profile(arm_obj, skin_objs, skel_map)
        elif act == "bind_and_transfer_weights":
            res = _action_bind_and_transfer_weights(arm_obj, skin_objs, garment_obj, params)
        elif act == "repair_weight_anomalies":
            res = _action_repair_weight_anomalies(arm_obj, skin_objs, garment_obj, inspection, params)
        elif act == "extra_clearance_wrap":
            max_wrap_z = inspection["body_profile"]["landmarks_z"]["neck"] + 0.038
            skirt_margin_m = None if inspection["garment_analysis"].get("has_offcenter_lower_pendulum") else float(params.get("skirt_margin_m", 0.032))
            center_y = float(inspection["body_profile"].get("hip_slice", {}).get("center_y", 0.0))
            pushed = _outward_clearance_wrap(
                skin_objs,
                garment_obj,
                margin_m=float(params.get("margin_m", 0.015)),
                smooth_passes=int(params.get("smooth_passes", 6)),
                inward_pull=float(params.get("inward_pull", 0.0)),
                arm_obj=arm_obj,
                skel_map=inspection["skeleton_map"],
                max_wrap_z=max_wrap_z,
                skirt_margin_m=skirt_margin_m,
                center_y=center_y,
            )
            res = {"extra_clearance_pushed_verts": pushed}
        else:
            res = {"skipped_unknown_action": act}
        action_logs.append({"action": act, "params": params, "result": res})

    _export_rigged_bundle(arm_obj, skin_objs, face_objs, garment_obj, Path(args.out_blend), Path(args.out_glb))

    report = {
        "inspection": inspection,
        "action_logs": action_logs,
        "outputs": {"blend": args.out_blend, "glb": args.out_glb},
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Cloth-rig toolkit completed ->", args.out_glb)


if __name__ == "__main__":
    main()
