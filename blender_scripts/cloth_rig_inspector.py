"""Inspect humanoid basemesh skeleton + standalone garment geometry to solve pose-aware fitting & region mapping."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from topology_inspector import load_scene_file  # noqa: E402


def import_asset_additive(filepath: Path):
    """Import .glb/.fbx/.obj/.blend into the current scene without wiping existing objects."""
    fp = str(filepath)
    low = fp.lower()
    if low.endswith(".glb") or low.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=fp)
    elif low.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=fp)
    elif low.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=fp)
    elif low.endswith(".blend"):
        with bpy.data.libraries.load(fp, link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for o in data_to.objects:
            if o is not None and o.name not in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.link(o)
    else:
        raise ValueError(f"Unsupported 3D asset format: {fp}")


BONE_SLOT_ALIASES = {
    "hips": ("j_bip_c_hips", "mixamorig:hips", "hips", "pelvis", "root_hips"),
    "spine": ("j_bip_c_spine", "mixamorig:spine", "spine", "spine_01", "spine.001"),
    "chest": ("j_bip_c_chest", "mixamorig:spine1", "chest", "spine_02", "spine.002"),
    "upper_chest": ("j_bip_c_upperchest", "mixamorig:spine2", "upperchest", "upper_chest", "spine_03"),
    "neck": ("j_bip_c_neck", "mixamorig:neck", "neck"),
    "head": ("j_bip_c_head", "mixamorig:head", "head"),
    "shoulder_l": ("j_bip_l_shoulder", "mixamorig:leftshoulder", "shoulder.l", "clavicle_l", "l_shoulder"),
    "shoulder_r": ("j_bip_r_shoulder", "mixamorig:rightshoulder", "shoulder.r", "clavicle_r", "r_shoulder"),
    "upper_arm_l": ("j_bip_l_upperarm", "mixamorig:leftarm", "upperarm.l", "upper_arm.l", "upperarm_l"),
    "upper_arm_r": ("j_bip_r_upperarm", "mixamorig:rightarm", "upperarm.r", "upper_arm.r", "upperarm_r"),
    "lower_arm_l": ("j_bip_l_lowerarm", "mixamorig:leftforearm", "lowerarm.l", "forearm.l", "lowerarm_l"),
    "lower_arm_r": ("j_bip_r_lowerarm", "mixamorig:rightforearm", "lowerarm.r", "forearm.r", "lowerarm_r"),
    "hand_l": ("j_bip_l_hand", "mixamorig:lefthand", "hand.l", "hand_l"),
    "hand_r": ("j_bip_r_hand", "mixamorig:righthand", "hand.r", "hand_r"),
    "upper_leg_l": ("j_bip_l_upperleg", "mixamorig:leftupleg", "thigh.l", "upperleg.l", "thigh_l"),
    "upper_leg_r": ("j_bip_r_upperleg", "mixamorig:rightupleg", "thigh.r", "upperleg.r", "thigh_r"),
    "lower_leg_l": ("j_bip_l_lowerleg", "mixamorig:leftleg", "shin.l", "calf.l", "lowerleg.l", "calf_l"),
    "lower_leg_r": ("j_bip_r_lowerleg", "mixamorig:rightleg", "shin.r", "calf.r", "lowerleg.r", "calf_r"),
    "foot_l": ("j_bip_l_foot", "mixamorig:leftfoot", "foot.l", "foot_l"),
    "foot_r": ("j_bip_r_foot", "mixamorig:rightfoot", "foot.r", "foot_r"),
}


def strip_bone_shape_junk(objs):
    """Remove exporter bone-shape placeholder meshes (e.g. Icosphere) and return valid anatomical/garment meshes."""
    clean = []
    for o in list(objs):
        lname = o.name.lower()
        if "icosphere" in lname or lname.startswith("wgt_") or lname.startswith("cs_"):
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        clean.append(o)
    return clean


def map_humanoid_skeleton(arm_obj):
    """Auto-map armature bone names (VRM, Mixamo, Rigify, generic) to canonical skeleton slots."""
    if arm_obj is None:
        return {"slots": {}, "secondary": {}, "bone_count": 0}
    lower_lookup = {b.name.lower(): b.name for b in arm_obj.data.bones}
    slots = {}
    for slot, aliases in BONE_SLOT_ALIASES.items():
        for cand in aliases:
            if cand in lower_lookup:
                slots[slot] = lower_lookup[cand]
                break
        if slot not in slots:
            # Substring fallback excluding J_Sec secondary chains
            tok = slot.replace("_l", "").replace("_r", "").replace("_", "")
            side = "_l_" if slot.endswith("_l") else ("_r_" if slot.endswith("_r") else None)
            for bname_low, bname in lower_lookup.items():
                if bname_low.startswith("j_sec_") or bname_low.startswith("j_adj_"):
                    continue
                if side and side not in bname_low and not bname_low.endswith(side[1]):
                    continue
                if tok in bname_low.replace("_", ""):
                    slots[slot] = bname
                    break

    secondary = {
        "bust": [b.name for b in arm_obj.data.bones if "j_sec_" in b.name.lower() and "bust" in b.name.lower()],
        "skirt": [b.name for b in arm_obj.data.bones if "j_sec_" in b.name.lower() and "skirt" in b.name.lower()],
        "hair": [b.name for b in arm_obj.data.bones if "j_sec_" in b.name.lower() and "hair" in b.name.lower()],
    }
    return {
        "armature_name": arm_obj.name,
        "bone_count": len(arm_obj.data.bones),
        "slots": slots,
        "secondary": secondary,
        "arm_bones_all": [
            b.name
            for b in arm_obj.data.bones
            if any(k in b.name.lower() for k in ("shoulder", "upperarm", "lowerarm", "forearm", "hand", "thumb", "index", "middle", "ring", "little", "pinky"))
            and "j_sec_" not in b.name.lower()
        ],
        "leg_l_bones": [
            b.name
            for b in arm_obj.data.bones
            if ("l_" in b.name.lower() or ".l" in b.name.lower() or "left" in b.name.lower())
            and any(k in b.name.lower() for k in ("leg", "thigh", "shin", "calf", "foot", "toe"))
            and "j_sec_" not in b.name.lower()
        ],
        "leg_r_bones": [
            b.name
            for b in arm_obj.data.bones
            if ("r_" in b.name.lower() or ".r" in b.name.lower() or "right" in b.name.lower())
            and any(k in b.name.lower() for k in ("leg", "thigh", "shin", "calf", "foot", "toe"))
            and "j_sec_" not in b.name.lower()
        ],
    }


def extract_body_profile(arm_obj, skin_objs, skel_map):
    """Measure key vertical Z landmarks, torso cross-sections, and current arm abduction angle on the basemesh."""
    slots = skel_map["slots"]
    mw = arm_obj.matrix_world
    bones_w = {}
    for slot, bname in slots.items():
        b = arm_obj.data.bones[bname]
        bones_w[slot] = {
            "name": bname,
            "head": [round(x, 4) for x in (mw @ b.head_local)],
            "tail": [round(x, 4) for x in (mw @ b.tail_local)],
        }

    pts = [o.matrix_world @ v.co for o in skin_objs for v in o.data.vertices]
    sole_z = min(p.z for p in pts)
    crown_z = max(p.z for p in pts)

    u_l = Vector(bones_w["upper_arm_l"]["head"])
    l_l = Vector(bones_w["lower_arm_l"]["head"])
    v_arm = (l_l - u_l).normalized()
    arm_angle_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, v_arm.dot(Vector((0, 0, -1))))))), 2)

    hips_z = bones_w["hips"]["head"][2]
    upper_chest_z = bones_w.get("upper_chest", bones_w.get("chest", bones_w["hips"]))["head"][2]
    neck_z = bones_w["neck"]["head"][2]
    shoulder_z = bones_w["upper_arm_l"]["head"][2]
    knee_z = bones_w["lower_leg_l"]["head"][2]
    ankle_z = bones_w["foot_l"]["head"][2]
    shoulder_x_span = abs(bones_w["upper_arm_l"]["head"][0] - bones_w["upper_arm_r"]["head"][0])

    # Narrowest waist scan excluding arms
    torso_pts = [p for p in pts if abs(p.x) <= 0.175 and hips_z - 0.14 <= p.z <= upper_chest_z + 0.04]
    waist_rows = []
    hip_rows = []
    bust_rows = []
    z = hips_z - 0.14
    while z <= upper_chest_z + 0.04:
        band = [p for p in torso_pts if abs(p.z - z) <= 0.014]
        if len(band) >= 6:
            xs = [p.x for p in band]
            ys = [p.y for p in band]
            row = {
                "z": round(z, 4),
                "width_x": round(max(xs) - min(xs), 4),
                "depth_y": round(max(ys) - min(ys), 4),
                "front_y": round(min(ys), 4),
                "center_y": round(0.5 * (max(ys) + min(ys)), 4),
            }
            if hips_z + 0.03 <= z <= bones_w["chest"]["head"][2] + 0.06:
                waist_rows.append(row)
            if hips_z - 0.13 <= z <= hips_z + 0.03:
                hip_rows.append(row)
            if bones_w["chest"]["head"][2] - 0.02 <= z <= shoulder_z - 0.04:
                bust_rows.append(row)
        z += 0.008

    waist_slice = min(waist_rows, key=lambda r: r["width_x"]) if waist_rows else {"z": bones_w["spine"]["head"][2], "width_x": 0.17, "depth_y": 0.13, "center_y": -0.02}
    hip_slice = max(hip_rows, key=lambda r: r["width_x"]) if hip_rows else {"z": hips_z - 0.08, "width_x": 0.31, "depth_y": 0.16, "center_y": 0.0}
    bust_slice = min(bust_rows, key=lambda r: r["front_y"]) if bust_rows else {"z": upper_chest_z, "width_x": 0.24, "depth_y": 0.20, "center_y": -0.04}

    return {
        "sole_z": round(sole_z, 4),
        "crown_z": round(crown_z, 4),
        "height_m": round(crown_z - sole_z, 4),
        "arm_abduction_deg": arm_angle_deg,
        "shoulder_joint_span": round(shoulder_x_span, 4),
        "landmarks_z": {
            "ankle": round(ankle_z, 4),
            "knee": round(knee_z, 4),
            "hip_widest": hip_slice["z"],
            "hips_bone": round(hips_z, 4),
            "waist_narrowest": waist_slice["z"],
            "bust_peak": bust_slice["z"],
            "shoulder": round(shoulder_z, 4),
            "neck": round(neck_z, 4),
        },
        "waist_slice": waist_slice,
        "hip_slice": hip_slice,
        "bust_slice": bust_slice,
        "bones_world": bones_w,
    }


def _slice_x_span(pts, z_val, band=0.025):
    b = [p for p in pts if abs(p.z - z_val) <= band]
    if len(b) < 6:
        return None
    xs = [p.x for p in b]
    ys = [p.y for p in b]
    return {
        "z": round(z_val, 4),
        "min_x": round(min(xs), 4),
        "max_x": round(max(xs), 4),
        "width_x": round(max(xs) - min(xs), 4),
        "depth_y": round(max(ys) - min(ys), 4),
        "center_x": round(0.5 * (max(xs) + min(xs)), 4),
        "center_y": round(0.5 * (max(ys) + min(ys)), 4),
        "crosses_center_x": (min(xs) < -0.04 and max(xs) > 0.04),
    }


def analyze_garment_and_propose_fit(garment_obj, body_profile):
    """Classify garment structure (dress/sleeves vs corset/belt), detect sleeve angle & cuff bounds, and compute fit transform."""
    mw = garment_obj.matrix_world
    pts = [mw @ v.co for v in garment_obj.data.vertices]
    z0 = min(p.z for p in pts)
    z1 = max(p.z for p in pts)
    h_raw = max(1e-4, z1 - z0)

    slices = []
    for i in range(25):
        frac = 0.02 + 0.96 * (i / 24.0)
        sl = _slice_x_span(pts, z0 + frac * h_raw, band=h_raw * 0.022)
        if sl:
            sl["frac"] = round(frac, 3)
            slices.append(sl)

    # Identify core torso-enclosing band (slices crossing both sides of X=0 with significant depth)
    enclosing = [s for s in slices if s["crosses_center_x"] and s["width_x"] > 0.12]
    enc_z_min = min(s["z"] for s in enclosing) if enclosing else z0
    enc_z_max = max(s["z"] for s in enclosing) if enclosing else z1
    enc_frac_span = (enc_z_max - enc_z_min) / h_raw

    # Check if top/bottom differ structurally (e.g., corset cuff upper half + dangling pendant off-center lower half)
    non_enclosing_lower = [s for s in slices if s["z"] < enc_z_min and not s["crosses_center_x"]]
    # Measure sleeve wings: in upper half of enclosing region, look for wide bilateral protrusion above waist
    upper_enc = [s for s in enclosing if s["z"] >= enc_z_min + 0.45 * (enc_z_max - enc_z_min)]
    max_upper_w = max((s["width_x"] for s in upper_enc), default=0.0)
    # Central waist depth minimum inside enclosing upper half
    narrow_slices = sorted(enclosing, key=lambda s: s["depth_y"])

    # Sleeve detection: compare width at upper 72-88% against narrowest upper torso depth/width
    # Specifically, look for lateral vertices beyond |x| > 0.24 * h_raw that are disconnected or branched sideways above waist
    sleeve_angle_deg = None
    has_sleeves = False

    if enc_frac_span < 0.58 and len(non_enclosing_lower) >= 3:
        garment_type = "waist_corset_belt"
    elif enc_z_max - enc_z_min > 0.55 * h_raw and max_upper_w > 0.40 * h_raw:
        # Check bilateral sleeve wings sloping downward from shoulder to cuff
        wing_anchor_z = z0 + 0.78 * h_raw
        upper_core = [p for p in pts if abs(p.z - wing_anchor_z) < 0.03]
        cuff_band = [p for p in pts if abs(p.x) > 0.30 * h_raw * 0.55 and p.z > z0 + 0.55 * h_raw]
        if upper_core and len(cuff_band) > 40:
            has_sleeves = True
            # Measure shoulder root (|x| ~ 0.14..0.19) to outer wing centroid (x > max_x - 0.07)
            r_wing = [p for p in pts if p.x > 0.12 and p.z > z0 + 0.55 * h_raw]
            if r_wing:
                max_rx = max(p.x for p in r_wing)
                inner_pts = [p for p in r_wing if 0.13 <= p.x <= 0.185 and p.z > z0 + 0.72 * h_raw]
                outer_pts = [p for p in r_wing if p.x >= max_rx - 0.07]
                if inner_pts and outer_pts:
                    pin = Vector((sum(p.x for p in inner_pts) / len(inner_pts), 0.0, sum(p.z for p in inner_pts) / len(inner_pts)))
                    pout = Vector((sum(p.x for p in outer_pts) / len(outer_pts), 0.0, sum(p.z for p in outer_pts) / len(outer_pts)))
                    vec = (pout - pin).normalized()
                    sleeve_angle_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, vec.dot(Vector((0, 0, -1))))))), 1)
        garment_type = "full_sleeved_dress" if has_sleeves else "full_dress"
    else:
        garment_type = "skirt_or_top"

    bz = body_profile["landmarks_z"]
    # Solve Fit Transform (Scale + Translation + Target Arm Pose) mapping garment onto body
    if garment_type == "waist_corset_belt":
        cuff_pts = [p for p in pts if p.z >= enc_z_min - 0.02]
        cuff_min_z = min(p.z for p in cuff_pts)
        cuff_max_z = max(p.z for p in cuff_pts)
        cuff_h = max(1e-4, cuff_max_z - cuff_min_z)
        target_cuff_lo = bz["hips_bone"] - 0.045
        target_cuff_hi = bz["bust_peak"] - 0.045
        scale_z = round((target_cuff_hi - target_cuff_lo) / cuff_h, 4)
        waist_enc = min(enclosing, key=lambda s: s["depth_y"])
        hip_cuff_enc = max(
            [s for s in enclosing if s["z"] <= cuff_min_z + 0.35 * cuff_h] or enclosing,
            key=lambda s: s["width_x"],
        )
        # Ensure BOTH waist AND wider bottom hip cuff clear underlying anime anatomy (waist -> 31.8cm hip flare!)
        need_sx_waist = (body_profile["waist_slice"]["width_x"] + 0.032) / max(1e-4, waist_enc["width_x"])
        need_sy_waist = (body_profile["waist_slice"]["depth_y"] + 0.032) / max(1e-4, waist_enc["depth_y"])
        need_sx_hip = (body_profile["hip_slice"]["width_x"] + 0.042) / max(1e-4, hip_cuff_enc["width_x"])
        need_sy_hip = (body_profile["hip_slice"]["depth_y"] + 0.042) / max(1e-4, hip_cuff_enc["depth_y"])
        raw_sx = max(need_sx_waist, need_sx_hip)
        raw_sy = max(need_sy_waist, need_sy_hip)
        sx = round(max(scale_z * 0.85, min(scale_z * 1.35, raw_sx)), 4)
        sy = round(max(scale_z * 0.82, min(scale_z * 1.35, raw_sy)), 4)
        cuff_cx = sum(s["center_x"] for s in enclosing) / max(1, len(enclosing))
        cuff_cy = sum(s["center_y"] for s in enclosing) / max(1, len(enclosing))
        target_cz = 0.5 * (target_cuff_lo + target_cuff_hi)
        cuff_cz = 0.5 * (cuff_min_z + cuff_max_z)
        offset = [
            round(0.0 - cuff_cx * sx, 4),
            round(body_profile["hip_slice"]["center_y"] - cuff_cy * sy, 4),
            round(target_cz - cuff_cz * scale_z, 4),
        ]
        scale_vec = [sx, sy, scale_z]
        target_arm_deg = 45.0
    else:
        collar_raw_z = z1
        torso_slices = [s for s in enclosing if 0.58 <= s["frac"] <= 0.82]
        waist_sl = min(torso_slices, key=lambda s: s["depth_y"]) if torso_slices else enclosing[len(enclosing) // 2]
        waist_raw_z = waist_sl["z"]
        neck_to_waist_raw = max(0.15, collar_raw_z - waist_raw_z)
        neck_to_waist_body = max(0.15, bz["neck"] + 0.035 - bz["waist_narrowest"])
        scale_z = (neck_to_waist_body / neck_to_waist_raw) * 1.08
        # Compute sleeve horizontal span fit alongside torso depth clearance
        wings = [p for p in pts if abs(p.x) > 0.14 and p.z > z0 + 0.55 * h_raw]
        wing_span_raw = (max(p.x for p in wings) - min(p.x for p in wings)) if wings else 0.78
        sh_span = body_profile["shoulder_joint_span"]
        uarm_len = abs(body_profile["bones_world"]["upper_arm_l"]["tail"][0] - body_profile["bones_world"]["upper_arm_l"]["head"][0])
        larm_len = abs(body_profile["bones_world"]["lower_arm_l"]["tail"][0] - body_profile["bones_world"]["lower_arm_l"]["head"][0])
        arm_ang = math.radians(sleeve_angle_deg or 40.0)
        body_cuff_span = sh_span + 2.0 * (uarm_len + larm_len * 0.88) * math.sin(arm_ang)
        scale_arm_x = body_cuff_span / max(0.2, wing_span_raw)
        scale_bust_y = (body_profile["bust_slice"]["depth_y"] + 0.032) / max(0.05, waist_sl["depth_y"])
        scale_uniform = round(max(0.56, min(0.70, 0.55 * scale_z + 0.45 * scale_arm_x)), 4)
        sy = round(max(scale_uniform, min(scale_uniform * 1.24, scale_bust_y * 0.72)), 4)
        scale_vec = [scale_uniform, sy, scale_uniform]
        arm_y = body_profile["bones_world"]["upper_arm_l"]["head"][1]
        torso_center_y = 0.55 * body_profile["waist_slice"]["center_y"] + 0.45 * arm_y
        offset = [
            round(0.0 - waist_sl["center_x"] * scale_vec[0], 4),
            round(torso_center_y - waist_sl["center_y"] * scale_vec[1], 4),
            round(bz["waist_narrowest"] - waist_raw_z * scale_vec[2], 4),
        ]
        target_arm_deg = sleeve_angle_deg if sleeve_angle_deg else 42.0

    return {
        "garment_name": garment_obj.name,
        "garment_type": garment_type,
        "has_sleeves": has_sleeves,
        "detected_sleeve_angle_deg": sleeve_angle_deg,
        "raw_z_range": [round(z0, 4), round(z1, 4)],
        "enclosing_band_z": [round(enc_z_min, 4), round(enc_z_max, 4)],
        "has_offcenter_lower_pendulum": len(non_enclosing_lower) >= 3,
        "proposed_fit": {
            "scale": scale_vec,
            "translation": offset,
            "target_arm_abduction_deg": target_arm_deg,
            "arm_pose_delta_deg": round(target_arm_deg - body_profile["arm_abduction_deg"], 2),
        },
    }


def inspect_pair(body_path: Path, garment_path: Path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    load_scene_file(body_path)
    arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    body_meshes = strip_bone_shape_junk([o for o in bpy.data.objects if o.type == "MESH"])
    skin_objs = [o for o in body_meshes if "skin" in o.name.lower() or "body" in o.name.lower()] or body_meshes

    skel_map = map_humanoid_skeleton(arm_obj)
    body_profile = extract_body_profile(arm_obj, skin_objs, skel_map)

    before = {o.as_pointer() for o in bpy.data.objects}
    import_asset_additive(garment_path)
    garment_meshes = [o for o in bpy.data.objects if o.as_pointer() not in before and o.type == "MESH"]
    garment_obj = max(garment_meshes, key=lambda o: len(o.data.vertices))
    garment_analysis = analyze_garment_and_propose_fit(garment_obj, body_profile)

    return {
        "body_path": str(body_path),
        "garment_path": str(garment_path),
        "skeleton_map": skel_map,
        "body_profile": body_profile,
        "garment_analysis": garment_analysis,
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--body", required=True)
    p.add_argument("--garment", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args(argv)

    rep = inspect_pair(Path(args.body), Path(args.garment))
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Cloth-rig matching inspection written ->", out)


if __name__ == "__main__":
    main()
