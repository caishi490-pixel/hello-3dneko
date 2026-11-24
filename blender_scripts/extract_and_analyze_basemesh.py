"""Disassemble a VRM/GLB character master into clean anatomical basemesh (素模), separated garment/hair layers, exact macro proportions, and topology/rigging craft metrics."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip-renders", action="store_true")
    return p.parse_args(argv)


def _mesh_topology_stats(obj, quadify_copy=True):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)
    raw_tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    raw_quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    nm = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    be = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()

    rec_quads = raw_quads
    rec_tris = raw_tris
    rec_faces = len(me.polygons)
    val4_ratio = 0.0
    pole3_cnt = 0
    pole5p_cnt = 0

    if quadify_copy:
        tmp_me = me.copy()
        tmp_obj = bpy.data.objects.new("__TopoStatTmp", tmp_me)
        bpy.context.scene.collection.objects.link(tmp_obj)
        try:
            bpy.ops.object.select_all(action="DESELECT")
            tmp_obj.select_set(True)
            bpy.context.view_layer.objects.active = tmp_obj
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles(threshold=1e-4)
            bpy.ops.mesh.tris_convert_to_quads(
                face_threshold=math.radians(40.0), shape_threshold=math.radians(40.0)
            )
            bpy.ops.object.mode_set(mode="OBJECT")
            bm2 = bmesh.new()
            bm2.from_mesh(tmp_obj.data)
            rec_faces = max(1, len(bm2.faces))
            rec_quads = sum(1 for f in bm2.faces if len(f.verts) == 4)
            rec_tris = sum(1 for f in bm2.faces if len(f.verts) == 3)
            val_hist = Counter(len(v.link_edges) for v in bm2.verts)
            tot_v = max(1, len(bm2.verts))
            val4_ratio = round(val_hist.get(4, 0) / tot_v, 4)
            pole3_cnt = val_hist.get(3, 0)
            pole5p_cnt = sum(c for k, c in val_hist.items() if k >= 5)
            bm2.free()
        finally:
            bpy.data.objects.remove(tmp_obj, do_unlink=True)
            bpy.data.meshes.remove(tmp_me, do_unlink=True)

    return {
        "verts_raw": len(me.vertices),
        "faces_raw": len(me.polygons),
        "recovered_faces": rec_faces,
        "recovered_quads": rec_quads,
        "recovered_tris": rec_tris,
        "recovered_quad_ratio": round(rec_quads / max(1, rec_faces), 4),
        "recovered_valence4_ratio": val4_ratio,
        "pole_3_count": pole3_cnt,
        "pole_5plus_count": pole5p_cnt,
        "welded_non_manifold": nm,
        "welded_boundary_edges": be,
    }


def _slice_metrics_at_z(mesh_objs, z_target, band=0.015, x_filter=None):
    pts = []
    for o in mesh_objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            wp = mw @ v.co
            if abs(wp.z - z_target) <= band:
                if x_filter is None or x_filter(wp.x):
                    pts.append(wp)
    if len(pts) < 4:
        return None
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return {
        "z": round(z_target, 4),
        "width_x": round(max(xs) - min(xs), 4),
        "depth_y": round(max(ys) - min(ys), 4),
        "min_y": round(min(ys), 4),
        "max_y": round(max(ys), 4),
        "center_y": round(0.5 * (max(ys) + min(ys)), 4),
        "sample_count": len(pts),
    }


def _scan_torso_extrema(skin_objs, z_lo, z_hi, torso_x_limit, step=0.005):
    """Scan dense Z slices across torso (excluding arms) for narrowest waist, widest hips, and chest depth."""
    pts = []
    for o in skin_objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            wp = mw @ v.co
            if z_lo <= wp.z <= z_hi and abs(wp.x) <= torso_x_limit:
                pts.append(wp)
    rows = []
    z = z_lo
    while z <= z_hi + 1e-6:
        band_pts = [p for p in pts if abs(p.z - z) <= 0.012]
        if len(band_pts) >= 6:
            xs = [p.x for p in band_pts]
            ys = [p.y for p in band_pts]
            rows.append({
                "z": round(z, 4),
                "width_x": round(max(xs) - min(xs), 4),
                "depth_y": round(max(ys) - min(ys), 4),
                "front_y": round(min(ys), 4),
                "back_y": round(max(ys), 4),
            })
        z += step
    return rows


def _extract_skeleton_proportions(arm_obj, skin_objs, face_objs, hair_objs):
    mw = arm_obj.matrix_world
    bones_world = {}
    for b in arm_obj.data.bones:
        h = mw @ b.head_local
        t = mw @ b.tail_local
        bones_world[b.name] = {
            "head": [round(x, 4) for x in h],
            "tail": [round(x, 4) for x in t],
            "length": round((t - h).length, 4),
        }

    def _pt(name, which="head"):
        return Vector(bones_world[name][which]) if name in bones_world else None

    all_anat_pts = []
    for o in skin_objs + face_objs:
        omw = o.matrix_world
        for v in o.data.vertices:
            all_anat_pts.append(omw @ v.co)
    sole_z = min(p.z for p in all_anat_pts)
    skin_crown_z = max(p.z for p in all_anat_pts)

    hair_pts = [o.matrix_world @ v.co for o in hair_objs for v in o.data.vertices]
    full_top_z = max([skin_crown_z] + [p.z for p in hair_pts]) if hair_pts else skin_crown_z

    face_pts = [o.matrix_world @ v.co for o in face_objs for v in o.data.vertices]
    chin_z = min(p.z for p in face_pts) if face_pts else bones_world["J_Bip_C_Head"]["head"][2]
    face_top_z = max(p.z for p in face_pts) if face_pts else skin_crown_z
    head_height_chin_to_crown = max(1e-4, skin_crown_z - chin_z)
    body_height_total = max(1e-4, skin_crown_z - sole_z)
    head_to_body_ratio = round(body_height_total / head_height_chin_to_crown, 2)

    l_uarm = _pt("J_Bip_L_UpperArm", "head")
    l_larm = _pt("J_Bip_L_LowerArm", "head")
    arm_angle_deg = None
    if l_uarm and l_larm:
        v_arm = (l_larm - l_uarm).normalized()
        arm_angle_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, v_arm.dot(Vector((0, 0, -1))))))), 2)

    l_uleg = _pt("J_Bip_L_UpperLeg", "head")
    l_foot = _pt("J_Bip_L_Foot", "head")
    leg_splay_deg = None
    if l_uleg and l_foot:
        v_leg = (l_foot - l_uleg).normalized()
        leg_splay_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, v_leg.dot(Vector((0, 0, -1))))))), 2)

    def _pct(z):
        return round(100.0 * (z - sole_z) / body_height_total, 2)

    hips_z = bones_world["J_Bip_C_Hips"]["head"][2]
    waist_z = bones_world["J_Bip_C_Spine"]["head"][2]
    chest_z = bones_world["J_Bip_C_Chest"]["head"][2]
    upper_chest_z = bones_world["J_Bip_C_UpperChest"]["head"][2]
    shoulder_z = bones_world["J_Bip_L_UpperArm"]["head"][2]
    neck_z = bones_world["J_Bip_C_Neck"]["head"][2]
    head_bone_z = bones_world["J_Bip_C_Head"]["head"][2]
    knee_z = bones_world["J_Bip_L_LowerLeg"]["head"][2]
    ankle_z = bones_world["J_Bip_L_Foot"]["head"][2]
    shoulder_joint_x = abs(bones_world["J_Bip_L_UpperArm"]["head"][0] - bones_world["J_Bip_R_UpperArm"]["head"][0])
    hip_joint_x = abs(bones_world["J_Bip_L_UpperLeg"]["head"][0] - bones_world["J_Bip_R_UpperLeg"]["head"][0])
    fingertip_span_x = round(
        max(p.x for p in all_anat_pts) - min(p.x for p in all_anat_pts), 4
    )

    # Dense scan for exact anatomical torso & hip extrema (excluding arms outside deltoid root)
    torso_scan = _scan_torso_extrema(skin_objs, hips_z - 0.14, upper_chest_z + 0.04, torso_x_limit=0.175)
    waist_candidates = [r for r in torso_scan if hips_z + 0.03 <= r["z"] <= chest_z + 0.05]
    waist_exact = min(waist_candidates, key=lambda r: r["width_x"]) if waist_candidates else None
    hip_candidates = [r for r in torso_scan if hips_z - 0.12 <= r["z"] <= hips_z + 0.03]
    hip_widest = max(hip_candidates, key=lambda r: r["width_x"]) if hip_candidates else None
    bust_candidates = [r for r in torso_scan if chest_z - 0.02 <= r["z"] <= shoulder_z - 0.04]
    bust_peak = min(bust_candidates, key=lambda r: r["front_y"]) if bust_candidates else None

    # Torso breadth below deltoids / axillary line (|x| <= 0.155 near armpit height)
    axilla_x_lim = abs(bones_world["J_Bip_L_UpperArm"]["head"][0]) + 0.065

    slices = {
        "head_face_level": _slice_metrics_at_z(face_objs, 0.5 * (chin_z + face_top_z), band=0.035),
        "neck_mid": _slice_metrics_at_z(skin_objs, 0.5 * (neck_z + head_bone_z), band=0.015),
        "torso_shoulder_deltoid_root": _slice_metrics_at_z(
            skin_objs, shoulder_z - 0.035, band=0.018, x_filter=lambda x: abs(x) <= axilla_x_lim + 0.02
        ),
        "bust_upper_chest": bust_peak or _slice_metrics_at_z(skin_objs, upper_chest_z, band=0.018, x_filter=lambda x: abs(x) <= 0.16),
        "waist_narrowest": waist_exact or _slice_metrics_at_z(skin_objs, waist_z + 0.05, band=0.015, x_filter=lambda x: abs(x) <= 0.15),
        "pelvis_hips_widest": hip_widest or _slice_metrics_at_z(skin_objs, hips_z - 0.06, band=0.018, x_filter=lambda x: abs(x) <= 0.20),
        "single_thigh_upper": _slice_metrics_at_z(skin_objs, hips_z - 0.14, band=0.015, x_filter=lambda x: x > 0.002),
        "single_knee": _slice_metrics_at_z(skin_objs, knee_z, band=0.015, x_filter=lambda x: x > 0.002),
        "single_calf_mid": _slice_metrics_at_z(skin_objs, 0.5 * (knee_z + ankle_z), band=0.015, x_filter=lambda x: x > 0.002),
        "single_ankle": _slice_metrics_at_z(skin_objs, ankle_z + 0.015, band=0.012, x_filter=lambda x: x > 0.002),
    }

    shoulder_torso_w = (slices["torso_shoulder_deltoid_root"] or {}).get("width_x") or shoulder_joint_x
    waist_w = (slices["waist_narrowest"] or {}).get("width_x")
    hip_w = (slices["pelvis_hips_widest"] or {}).get("width_x")
    key_ratios = {
        "head_to_body_ratio": head_to_body_ratio,
        "head_width_to_shoulder_torso_ratio": round(
            slices["head_face_level"]["width_x"] / max(1e-4, shoulder_torso_w), 2
        ) if slices["head_face_level"] else None,
        "waist_to_hip_width_ratio": round(waist_w / max(1e-4, hip_w), 3) if waist_w and hip_w else None,
        "waist_to_shoulder_torso_ratio": round(waist_w / max(1e-4, shoulder_torso_w), 3) if waist_w else None,
        "shoulder_torso_to_hip_width_ratio": round(shoulder_torso_w / max(1e-4, hip_w), 3) if hip_w else None,
        "upper_to_lower_arm_ratio": round(
            bones_world["J_Bip_L_UpperArm"]["length"] / max(1e-4, bones_world["J_Bip_L_LowerArm"]["length"]), 3
        ),
        "thigh_to_calf_bone_ratio": round(
            abs(hips_z - knee_z) / max(1e-4, abs(knee_z - ankle_z)), 3
        ),
        "leg_crotch_to_height_pct": _pct(
            bones_world["J_Bip_L_UpperLeg"]["head"][2] - 0.055
        ),
    }

    return {
        "height_metrics": {
            "sole_z": round(sole_z, 4),
            "chin_z": round(chin_z, 4),
            "skin_crown_z": round(skin_crown_z, 4),
            "full_with_hair_top_z": round(full_top_z, 4),
            "anatomical_height_m": round(body_height_total, 4),
            "fingertip_armspan_m": fingertip_span_x,
            "armspan_to_height_ratio": round(fingertip_span_x / body_height_total, 3),
            "head_height_chin_to_crown_m": round(head_height_chin_to_crown, 4),
            "head_to_body_ratio": head_to_body_ratio,
        },
        "pose_angles": {
            "upper_arm_abduction_from_vertical_deg": arm_angle_deg,
            "leg_splay_from_vertical_deg": leg_splay_deg,
            "shoulder_joint_span_m": round(shoulder_joint_x, 4),
            "hip_joint_span_m": round(hip_joint_x, 4),
            "shoulder_to_hip_joint_ratio": round(shoulder_joint_x / max(1e-4, hip_joint_x), 3),
        },
        "key_proportion_ratios": key_ratios,
        "landmark_z_percentages": {
            "sole": 0.0,
            "ankle": _pct(ankle_z),
            "knee": _pct(knee_z),
            "hip_widest": _pct(slices["pelvis_hips_widest"]["z"]) if slices["pelvis_hips_widest"] else _pct(hips_z),
            "hips_pelvis_bone": _pct(hips_z),
            "waist_narrowest": _pct(slices["waist_narrowest"]["z"]) if slices["waist_narrowest"] else _pct(waist_z),
            "bust_peak": _pct(slices["bust_upper_chest"]["z"]) if slices["bust_upper_chest"] else _pct(upper_chest_z),
            "upper_chest": _pct(upper_chest_z),
            "shoulder_joint": _pct(shoulder_z),
            "neck_base": _pct(neck_z),
            "chin": _pct(chin_z),
            "head_bone": _pct(head_bone_z),
            "crown": 100.0,
        },
        "cross_sections": slices,
        "key_bones_world": {
            k: bones_world[k]
            for k in (
                "J_Bip_C_Hips",
                "J_Bip_C_Spine",
                "J_Bip_C_Chest",
                "J_Bip_C_UpperChest",
                "J_Bip_C_Neck",
                "J_Bip_C_Head",
                "J_Bip_L_Shoulder",
                "J_Bip_L_UpperArm",
                "J_Bip_L_LowerArm",
                "J_Bip_L_Hand",
                "J_Bip_L_UpperLeg",
                "J_Bip_L_LowerLeg",
                "J_Bip_L_Foot",
                "J_Bip_L_ToeBase",
            )
            if k in bones_world
        },
    }


def _analyze_craft_secrets(arm_obj, skin_obj, face_obj, proportions):
    """Distill manual artist craft: regional poly budget, limb loop rings, Face 8-layer stack, 136-bone taxonomy, weights, and boundary seal audit."""
    kb = proportions["key_bones_world"]
    hand_x = abs(kb["J_Bip_L_Hand"]["head"][0]) - 0.012
    uarm_x = abs(kb["J_Bip_L_UpperArm"]["head"][0]) + 0.055
    hips_z = kb["J_Bip_C_Hips"]["head"][2] - 0.08
    neck_z = kb["J_Bip_C_Neck"]["head"][2]
    ankle_z = kb["J_Bip_L_Foot"]["head"][2] + 0.02

    # 1. Body_SKIN regional polygon distribution & boundary edge locations
    bm = bmesh.new()
    bm.from_mesh(skin_obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)
    mw = skin_obj.matrix_world
    region_counts = Counter()
    for f in bm.faces:
        c = mw @ f.calc_center_median()
        if abs(c.x) >= hand_x:
            region_counts["hands_and_fingers"] += 1
        elif abs(c.x) >= uarm_x:
            region_counts["arms_excl_hands"] += 1
        elif c.z >= neck_z:
            region_counts["neck_upper"] += 1
        elif c.z >= hips_z:
            region_counts["torso_and_bust"] += 1
        elif c.z >= ankle_z:
            region_counts["legs"] += 1
        else:
            region_counts["feet"] += 1

    boundary_clusters = Counter()
    for e in bm.edges:
        if e.is_boundary:
            mp = mw @ (0.5 * (e.verts[0].co + e.verts[1].co))
            if mp.z >= neck_z - 0.02:
                boundary_clusters["neck_top_rim_to_head"] += 1
            elif mp.z <= ankle_z and mp.z < 0.03:
                boundary_clusters["feet_sole_or_toe"] += 1
            elif abs(mp.x) >= hand_x:
                boundary_clusters["hand_finger_detail"] += 1
            else:
                boundary_clusters[f"other_z_{round(mp.z, 2)}"] += 1

    visited = set()
    shell_sizes = []
    for f in bm.faces:
        if f.index in visited:
            continue
        q = [f]
        visited.add(f.index)
        cnt = 0
        while q:
            cur = q.pop()
            cnt += 1
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf.index not in visited:
                        visited.add(nf.index)
                        q.append(nf)
        shell_sizes.append(cnt)
    shell_sizes.sort(reverse=True)
    bm.free()

    # 2. Circumferential loop ring counts measured on recovered QUAD topology (once)
    tmp_me = skin_obj.data.copy()
    tmp_obj = bpy.data.objects.new("__QuadRingTmp", tmp_me)
    bpy.context.scene.collection.objects.link(tmp_obj)
    loop_rings = {}
    try:
        bpy.ops.object.select_all(action="DESELECT")
        tmp_obj.select_set(True)
        bpy.context.view_layer.objects.active = tmp_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=1e-4)
        bpy.ops.mesh.tris_convert_to_quads(
            face_threshold=math.radians(40.0), shape_threshold=math.radians(40.0)
        )
        bpy.ops.object.mode_set(mode="OBJECT")

        qbm = bmesh.new()
        qbm.from_mesh(tmp_obj.data)

        def _quad_ring_edges(axis, val, x_filter=None, z_filter=None):
            hits = 0
            for e in qbm.edges:
                p0 = mw @ e.verts[0].co
                p1 = mw @ e.verts[1].co
                mid = 0.5 * (p0 + p1)
                if x_filter and not x_filter(mid.x):
                    continue
                if z_filter and not z_filter(mid.z):
                    continue
                c0 = p0[axis] - val
                c1 = p1[axis] - val
                if (c0 <= 0 < c1) or (c1 <= 0 < c0):
                    hits += 1
            return hits

        elbow_x = abs(kb["J_Bip_L_LowerArm"]["head"][0])
        wrist_x = abs(kb["J_Bip_L_Hand"]["head"][0])
        upperarm_mid_x = 0.5 * (abs(kb["J_Bip_L_UpperArm"]["head"][0]) + elbow_x)
        forearm_mid_x = 0.5 * (elbow_x + wrist_x)
        hips_bone_z = kb["J_Bip_C_Hips"]["head"][2]
        thigh_mid_z = 0.5 * (kb["J_Bip_L_UpperLeg"]["head"][2] + kb["J_Bip_L_LowerLeg"]["head"][2])
        calf_mid_z = 0.5 * (kb["J_Bip_L_LowerLeg"]["head"][2] + kb["J_Bip_L_Foot"]["head"][2])
        waist_z_exact = proportions["cross_sections"]["waist_narrowest"]["z"]

        loop_rings = {
            "neck_mid_quad_loop_edges": _quad_ring_edges(2, 1.41),
            "waist_quad_loop_edges": _quad_ring_edges(2, waist_z_exact, x_filter=lambda x: abs(x) <= 0.16),
            "hips_quad_loop_edges": _quad_ring_edges(2, hips_bone_z, x_filter=lambda x: abs(x) <= 0.18),
            "left_upper_arm_quad_loop_edges": _quad_ring_edges(0, upperarm_mid_x, z_filter=lambda z: z > 1.25),
            "left_forearm_quad_loop_edges": _quad_ring_edges(0, forearm_mid_x, z_filter=lambda z: z > 1.25),
            "left_thigh_quad_loop_edges": _quad_ring_edges(2, thigh_mid_z, x_filter=lambda x: x > 0.002),
            "left_calf_quad_loop_edges": _quad_ring_edges(2, calf_mid_z, x_filter=lambda x: x > 0.002),
        }
        qbm.free()
    finally:
        bpy.data.objects.remove(tmp_obj, do_unlink=True)
        bpy.data.meshes.remove(tmp_me, do_unlink=True)

    # 3. Face 8-material stack breakdown
    face_slots = {}
    for idx, slot in enumerate(face_obj.material_slots):
        mname = slot.material.name.replace(" (Instance)", "").strip() if slot.material else f"Slot_{idx}"
        face_slots[mname] = {"slot_index": idx, "polys": 0, "verts": 0}
    slot_verts = {i: set() for i in range(len(face_obj.material_slots))}
    for p in face_obj.data.polygons:
        idx = p.material_index
        if idx < len(face_obj.material_slots):
            mname = face_obj.material_slots[idx].material.name.replace(" (Instance)", "").strip()
            face_slots[mname]["polys"] += 1
            slot_verts[idx].update(p.vertices)
    for idx, slot in enumerate(face_obj.material_slots):
        mname = slot.material.name.replace(" (Instance)", "").strip() if slot.material else f"Slot_{idx}"
        face_slots[mname]["verts"] = len(slot_verts.get(idx, ()))

    # 4. Armature 136-bone taxonomy & skinning influence audit
    bone_names = [b.name for b in arm_obj.data.bones]
    sec_categories = Counter()
    for n in bone_names:
        if n.startswith("J_Bip_C_"):
            sec_categories["core_spine_head"] += 1
        elif n.startswith("J_Bip_L_") or n.startswith("J_Bip_R_"):
            if any(k in n for k in ("Thumb", "Index", "Middle", "Ring", "Little")):
                sec_categories["biped_fingers"] += 1
            else:
                sec_categories["biped_limbs"] += 1
        elif n.startswith("J_Sec_"):
            if "Bust" in n:
                sec_categories["secondary_bust_physics"] += 1
            elif "Hair" in n:
                sec_categories["secondary_hair_physics"] += 1
            elif "Skirt" in n:
                sec_categories["secondary_skirt_physics"] += 1
            else:
                sec_categories["secondary_other"] += 1
        elif n.startswith("J_Adj_"):
            sec_categories["corrective_adj_bones"] += 1
        else:
            sec_categories["other"] += 1

    def _skinning_audit(mesh_obj):
        max_inf = 0
        over4 = 0
        total_w_err = 0
        vg_names = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
        active_vgs = Counter()
        for v in mesh_obj.data.vertices:
            nz = [g for g in v.groups if g.weight > 1e-4]
            max_inf = max(max_inf, len(nz))
            if len(nz) > 4:
                over4 += 1
            if nz and abs(sum(g.weight for g in nz) - 1.0) > 0.02:
                total_w_err += 1
            for g in nz:
                if g.group in vg_names:
                    active_vgs[vg_names[g.group]] += 1
        return {
            "max_influences_per_vertex": max_inf,
            "verts_exceeding_4_influences": over4,
            "unnormalized_verts": total_w_err,
            "used_bone_groups_count": len(active_vgs),
            "bust_sec_weighted_verts": sum(c for k, c in active_vgs.items() if "Bust" in k),
        }

    total_anat_polys = len(skin_obj.data.polygons) + len(face_obj.data.polygons)
    return {
        "basemesh_poly_budget": {
            "total_basemesh_raw_tris_polys": total_anat_polys,
            "face_head_share_pct": round(100.0 * len(face_obj.data.polygons) / max(1, total_anat_polys), 2),
            "hands_share_of_body_pct": round(
                100.0 * region_counts["hands_and_fingers"] / max(1, len(skin_obj.data.polygons)), 2
            ),
            "hands_plus_face_share_of_basemesh_pct": round(
                100.0 * (len(face_obj.data.polygons) + region_counts["hands_and_fingers"]) / max(1, total_anat_polys), 2
            ),
            "body_skin_regions_raw_polys": dict(region_counts),
        },
        "circumferential_quad_loop_rings": loop_rings,
        "body_skin_shells": {
            "total_shells": len(shell_sizes),
            "main_body_raw_tris": shell_sizes[0] if shell_sizes else 0,
            "tiny_fingernail_micro_shells_count": max(0, len(shell_sizes) - 1),
        },
        "body_skin_boundary_edges_audit": {
            "note": "Check whether hidden torso/hip polygons under dress were culled or sealed",
            "clusters": dict(boundary_clusters),
        },
        "face_8_material_stack": face_slots,
        "armature_136_bones_taxonomy": {
            "total_bones": len(bone_names),
            "breakdown": dict(sec_categories),
        },
        "skinning_and_weights": {
            "body_skin": _skinning_audit(skin_obj),
            "face_head": _skinning_audit(face_obj),
        },
    }


def _split_mesh_by_material(src_obj, target_collection):
    """Duplicate src_obj and split cleanly by material slots into standalone named objects."""
    dup_me = src_obj.data.copy()
    dup_obj = bpy.data.objects.new(f"__Split_{src_obj.name}", dup_me)
    target_collection.objects.link(dup_obj)
    if src_obj.parent is not None:
        dup_obj.parent = src_obj.parent
        dup_obj.matrix_parent_inverse = src_obj.matrix_parent_inverse.copy()
    dup_obj.matrix_world = src_obj.matrix_world.copy()
    for mod in src_obj.modifiers:
        if mod.type == "ARMATURE":
            nm = dup_obj.modifiers.new(mod.name, "ARMATURE")
            nm.object = mod.object
            nm.use_deform_preserve_volume = True

    before = {o.as_pointer() for o in bpy.context.scene.objects}
    bpy.ops.object.select_all(action="DESELECT")
    dup_obj.select_set(True)
    bpy.context.view_layer.objects.active = dup_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts = [dup_obj] + [o for o in bpy.context.scene.objects if o.as_pointer() not in before and o.type == "MESH"]
    # Rename each part cleanly from its active face material
    out = []
    for p in parts:
        if len(p.data.polygons) == 0:
            bpy.data.objects.remove(p, do_unlink=True)
            continue
        if src_obj.parent is not None and p.parent is None:
            p.parent = src_obj.parent
            p.matrix_parent_inverse = src_obj.matrix_parent_inverse.copy()
        mat_idx = p.data.polygons[0].material_index
        mat_name = (
            p.material_slots[mat_idx].material.name
            if mat_idx < len(p.material_slots) and p.material_slots[mat_idx].material
            else "UnknownMat"
        )
        # Drop unused empty slots so each part carries only its own material
        used_mat = p.material_slots[mat_idx].material if mat_idx < len(p.material_slots) else None
        p.data.materials.clear()
        if used_mat:
            p.data.materials.append(used_mat)
        for poly in p.data.polygons:
            poly.material_index = 0
        clean_tag = mat_name.replace(" (Instance)", "").strip()
        p.name = f"{src_obj.name}__{clean_tag}"
        out.append(p)
    return out


def _export_selected_glb(objs, arm_obj, dst_path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    if arm_obj is not None:
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
    elif objs:
        bpy.context.view_layer.objects.active = objs[0]
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(dst_path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_skins=True,
    )


def _render_study_views(objs, out_dir, prefix, use_clay=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = bpy.context.scene

    # Isolate only requested objects visible in render
    prev_hide = {}
    target_ptrs = {o.as_pointer() for o in objs}
    for o in sc.objects:
        if o.type == "MESH":
            prev_hide[o.as_pointer()] = (o.hide_render, o.hide_viewport)
            show = o.as_pointer() in target_ptrs
            o.hide_render = not show
            o.hide_viewport = not show

    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    span = max((max_p - min_p).x, (max_p - min_p).y, (max_p - min_p).z, 0.1)
    dist = span * 1.95

    created = []
    orig_mats = {}
    orig_mat_indices = {}
    orig_wire_mods = []
    try:
        w = bpy.data.worlds.new("StudyWorld")
        sc.world = w
        w.use_nodes = True
        bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
        bg.inputs[0].default_value = (0.86, 0.88, 0.91, 1.0)
        bg.inputs[1].default_value = 0.42
        created.append(w)

        for lname, off, eng in [
            ("Study_Key", Vector((-1.4, -1.8, 1.0)), 48),
            ("Study_Fill", Vector((1.6, -1.5, 0.5)), 26),
            ("Study_Rim", Vector((0.0, 2.0, 1.1)), 34),
        ]:
            ld = bpy.data.lights.new(lname, type="AREA")
            ld.energy = eng * max(1.0, span * span)
            ld.size = span * 1.6
            lo = bpy.data.objects.new(lname, ld)
            sc.collection.objects.link(lo)
            lo.location = center + off * span
            lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()
            created.append(lo)

        cam_d = bpy.data.cameras.new("StudyCam")
        cam = bpy.data.objects.new("StudyCam", cam_d)
        sc.collection.objects.link(cam)
        sc.camera = cam
        created.append(cam)

        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.refresh_devices()
            for d in prefs.devices:
                d.use = d.type != "CPU"
            sc.cycles.device = "GPU"
        except Exception:
            sc.cycles.device = "CPU"
        sc.cycles.samples = 24
        sc.cycles.use_denoising = True
        sc.render.resolution_x = 768
        sc.render.resolution_y = 768
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.exposure = 0.0 if use_clay else -0.55

        if use_clay:
            clay = bpy.data.materials.new("Study_Clay")
            clay.use_nodes = True
            cb = next(n for n in clay.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            cb.inputs["Base Color"].default_value = (0.82, 0.80, 0.84, 1.0)
            cb.inputs["Roughness"].default_value = 0.55

            wire = bpy.data.materials.new("Study_Wire")
            wire.use_nodes = True
            wb = next(n for n in wire.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            wb.inputs["Base Color"].default_value = (0.08, 0.11, 0.18, 1.0)
            wb.inputs["Roughness"].default_value = 0.85

            for o in objs:
                orig_mats[o.as_pointer()] = list(o.data.materials)
                orig_mat_indices[o.as_pointer()] = [p.material_index for p in o.data.polygons]
                o.data.materials.clear()
                o.data.materials.append(clay)
                o.data.materials.append(wire)
                for p in o.data.polygons:
                    p.material_index = 0
                wf = o.modifiers.new("StudyWire", "WIREFRAME")
                wf.thickness = span * 0.0012
                wf.use_even_offset = False
                wf.use_replace = False
                wf.material_offset = 1
                orig_wire_mods.append((o, wf.name))

        def _snap(vec, fname, ortho=True):
            cam.location = center + vec.normalized() * dist
            cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
            if ortho:
                cam.data.type = "ORTHO"
                cam.data.ortho_scale = span * 1.18
            else:
                cam.data.type = "PERSP"
                cam.data.lens = 55
            dst = out_dir / fname
            sc.render.filepath = str(dst)
            bpy.ops.render.render(write_still=True)
            return str(dst)

        views = {
            "front": _snap(Vector((0.0, -1.0, 0.0)), f"{prefix}_front.png", ortho=True),
            "side": _snap(Vector((1.0, 0.0, 0.0)), f"{prefix}_side.png", ortho=True),
            "back": _snap(Vector((0.0, 1.0, 0.0)), f"{prefix}_back.png", ortho=True),
            "34": _snap(Vector((-0.65, -0.85, 0.15)), f"{prefix}_34.png", ortho=False),
        }
        return views
    finally:
        for o, mname in orig_wire_mods:
            if mname in o.modifiers:
                o.modifiers.remove(o.modifiers[mname])
        for o in objs:
            if o.as_pointer() in orig_mats:
                o.data.materials.clear()
                for m in orig_mats[o.as_pointer()]:
                    o.data.materials.append(m)
                for p, midx in zip(o.data.polygons, orig_mat_indices.get(o.as_pointer(), [])):
                    p.material_index = midx
        for o in sc.objects:
            if o.as_pointer() in prev_hide:
                o.hide_render, o.hide_viewport = prev_hide[o.as_pointer()]
        for item in created:
            if isinstance(item, bpy.types.World):
                sc.world = None
                bpy.data.worlds.remove(item)
            elif isinstance(item, bpy.types.Object):
                bpy.data.objects.remove(item, do_unlink=True)


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dis_dir = out_dir / "disassembled"
    ren_dir = out_dir / "renders"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input))

    # 1. Strip duplicate .001 ghost scene & Icosphere clutter
    removed = []
    for o in list(bpy.data.objects):
        base = o.name.split(".")[0].lower()
        if o.name.endswith(".001") or ".001" in o.name or base == "icosphere" or (
            o.type == "EMPTY" and (o.parent is None or o.parent.name.endswith(".001"))
        ):
            removed.append(o.name)
            bpy.data.objects.remove(o, do_unlink=True)

    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    body_obj = next((o for o in bpy.data.objects if o.type == "MESH" and o.name == "Body"), None)
    face_obj = next((o for o in bpy.data.objects if o.type == "MESH" and o.name == "Face"), None)
    hair_obj = next((o for o in bpy.data.objects if o.type == "MESH" and o.name == "Hair"), None)

    if not body_obj or not face_obj:
        raise RuntimeError("Could not find primary Body and Face meshes after cleanup.")

    # Render full cleaned dressed original first
    if args.skip_renders:
        dressed_renders = {
            k: str(ren_dir / f"01_full_dressed_clean_{k}.png") for k in ("front", "side", "back", "34")
        }
    else:
        dressed_renders = _render_study_views(
            [o for o in (body_obj, face_obj, hair_obj) if o is not None],
            ren_dir,
            prefix="01_full_dressed_clean",
            use_clay=False,
        )

    # 2. Disassemble Body by material into naked Skin Basemesh vs BackHair vs Cloth layers
    split_parts = _split_mesh_by_material(body_obj, bpy.context.scene.collection)
    skin_part = next((p for p in split_parts if "Body_00_SKIN" in p.name or "SKIN" in p.name), None)
    cloth_parts = [p for p in split_parts if "CLOTH" in p.name]
    hair_back_parts = [p for p in split_parts if "HAIR" in p.name]

    # Collect per-part topology telemetry and export individual clean GLBs
    part_stats = {}
    if skin_part:
        part_stats["body_skin_basemesh"] = _mesh_topology_stats(skin_part, quadify_copy=True)
        _export_selected_glb([skin_part], arm, dis_dir / "01_body_skin_basemesh.glb")
    part_stats["face_head"] = _mesh_topology_stats(face_obj, quadify_copy=True)
    _export_selected_glb([face_obj], arm, dis_dir / "02_face_head.glb")
    if hair_obj:
        part_stats["hair_merged"] = _mesh_topology_stats(hair_obj, quadify_copy=False)
        _export_selected_glb([hair_obj], arm, dis_dir / "03_hair_merged.glb")
    for idx, cp in enumerate(cloth_parts, start=1):
        key = cp.name.split("__")[-1]
        part_stats[f"garment_{key}"] = _mesh_topology_stats(cp, quadify_copy=True)
        _export_selected_glb([cp], arm, dis_dir / f"04_{idx:02d}_{key}.glb")
    for idx, hp in enumerate(hair_back_parts, start=1):
        key = hp.name.split("__")[-1]
        part_stats[f"hair_back_{key}"] = _mesh_topology_stats(hp, quadify_copy=True)
        _export_selected_glb([hp], arm, dis_dir / f"05_{idx:02d}_{key}.glb")

    # 3. Extract pure anatomical basemesh (Body_SKIN + Face): export & textured renders BEFORE clay override
    out_stem = f"{Path(args.input).stem}_clean_basemesh_rigged"
    basemesh_objs = [o for o in (skin_part, face_obj) if o is not None]
    _export_selected_glb(basemesh_objs, arm, out_dir / f"{out_stem}.glb")
    if args.skip_renders:
        basemesh_tex_renders = {
            k: str(ren_dir / f"03_basemesh_textured_{k}.png") for k in ("front", "side", "back", "34")
        }
        basemesh_clay_wire_renders = {
            k: str(ren_dir / f"02_basemesh_clay_wire_{k}.png") for k in ("front", "side", "back", "34")
        }
    else:
        basemesh_tex_renders = _render_study_views(
            basemesh_objs, ren_dir, prefix="03_basemesh_textured", use_clay=False
        )
        basemesh_clay_wire_renders = _render_study_views(
            basemesh_objs, ren_dir, prefix="02_basemesh_clay_wire", use_clay=True
        )

    # 4. Proportions + manual craft audit
    proportions = _extract_skeleton_proportions(
        arm,
        [skin_part] if skin_part else [body_obj],
        [face_obj],
        [hair_obj] if hair_obj else [],
    )
    craft_secrets = _analyze_craft_secrets(arm, skin_part, face_obj, proportions)

    # Save clean scene state keeping Armature + basemesh_objs (+ cloth/hair hidden cleanly in backup collection)
    garment_col = bpy.data.collections.new("Disassembled_Garments_And_Hair")
    bpy.context.scene.collection.children.link(garment_col)
    for o in [body_obj, hair_obj] + cloth_parts + hair_back_parts:
        if o and o.name in bpy.data.objects:
            for col in list(o.users_collection):
                col.objects.unlink(o)
            garment_col.objects.link(o)
            o.hide_viewport = True
            o.hide_render = True
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / f"{out_stem}.blend"))

    report = {
        "source_glb": Path(args.input).name,
        "removed_duplicates_and_clutter_count": len(removed),
        "disassembled_parts_topology": part_stats,
        "proportions_and_landmarks": proportions,
        "manual_craft_secrets": craft_secrets,
        "renders": {
            "full_dressed": dressed_renders,
            "basemesh_clay_wire": basemesh_clay_wire_renders,
            "basemesh_textured": basemesh_tex_renders,
        },
        "deliverables": {
            "clean_basemesh_glb": str(out_dir / f"{out_stem}.glb"),
            "clean_basemesh_blend": str(out_dir / f"{out_stem}.blend"),
            "disassembled_dir": str(dis_dir),
        },
    }
    (out_dir / "craft_analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Clean basemesh extraction & analysis complete ->", out_dir / "craft_analysis_report.json")


if __name__ == "__main__":
    main()
