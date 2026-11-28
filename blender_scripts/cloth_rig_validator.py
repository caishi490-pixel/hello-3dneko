"""Independent 4-Pose Stress Validator for Fitted & Weighted Character Garments (Clipping, Collapse, Weight Crosstalk)."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cloth_rig_inspector import (  # noqa: E402
    extract_body_profile,
    map_humanoid_skeleton,
    strip_bone_shape_junk,
)
from topology_inspector import load_scene_file  # noqa: E402
from blender_runtime import configure_cycles, junk_objects


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-blend", required=True)
    p.add_argument("--criteria-json", required=True)
    p.add_argument("--renders-dir", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args(argv)


def _rot_pose_bone_world(arm_obj, pb, world_axis: Vector, deg: float):
    """Rotate pose bone cleanly around a world-space axis anchored at bone head (axis-roll independent)."""
    if not pb or abs(deg) < 1e-4:
        return
    h_w = arm_obj.matrix_world @ pb.head
    rot_w = Matrix.Rotation(math.radians(deg), 4, world_axis.normalized())
    m_w = arm_obj.matrix_world @ pb.matrix
    m_new = Matrix.Translation(h_w) @ rot_w @ Matrix.Translation(-h_w) @ m_w
    pb.matrix = arm_obj.matrix_world.inverted() @ m_new


def _reset_pose(arm_obj):
    for pb in arm_obj.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def _apply_stress_pose(arm_obj, skel_map, pose_name: str):
    """Apply deterministic extreme test poses on humanoid rig using world-space joint rotations."""
    _reset_pose(arm_obj)
    slots = skel_map["slots"]
    pb = lambda slot: arm_obj.pose.bones.get(slots.get(slot, ""))  # noqa: E731

    if pose_name == "rest":
        return

    wx = Vector((1, 0, 0))
    wy = Vector((0, 1, 0))
    wz = Vector((0, 0, 1))

    if pose_name == "arms_up":
        # Raise arms upward & slightly forward + bend elbows forward + mild chest lift
        _rot_pose_bone_world(arm_obj, pb("chest"), wx, 6.0)
        _rot_pose_bone_world(arm_obj, pb("upper_chest"), wx, 6.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_l"), wy, -78.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_r"), wy, 78.0)
        bpy.context.view_layer.update()
        _rot_pose_bone_world(arm_obj, pb("upper_arm_l"), wz, -20.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_r"), wz, 20.0)
        _rot_pose_bone_world(arm_obj, pb("lower_arm_l"), wz, -32.0)
        _rot_pose_bone_world(arm_obj, pb("lower_arm_r"), wz, 32.0)
    elif pose_name == "torso_twist":
        # Strong upper body twist + natural arm swing
        _rot_pose_bone_world(arm_obj, pb("spine"), wz, 16.0)
        _rot_pose_bone_world(arm_obj, pb("chest"), wz, 20.0)
        _rot_pose_bone_world(arm_obj, pb("upper_chest"), wz, 14.0)
        _rot_pose_bone_world(arm_obj, pb("neck"), wz, 15.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_l"), wz, 24.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_r"), wz, 22.0)
        _rot_pose_bone_world(arm_obj, pb("lower_arm_l"), wz, -26.0)
    elif pose_name == "deep_squat":
        # Deep squat: world -Z hip drop, forward torso lean (+X rot), deep thigh & knee flexion
        hips_pb = pb("hips")
        if hips_pb:
            mw_h = arm_obj.matrix_world @ hips_pb.matrix
            mw_h.translation += Vector((0.0, 0.02, -0.22))
            hips_pb.matrix = arm_obj.matrix_world.inverted() @ mw_h
        bpy.context.view_layer.update()
        _rot_pose_bone_world(arm_obj, pb("hips"), wx, 10.0)
        _rot_pose_bone_world(arm_obj, pb("spine"), wx, 12.0)
        _rot_pose_bone_world(arm_obj, pb("chest"), wx, 10.0)
        _rot_pose_bone_world(arm_obj, pb("upper_leg_l"), wx, -74.0)
        _rot_pose_bone_world(arm_obj, pb("upper_leg_r"), wx, -74.0)
        bpy.context.view_layer.update()
        _rot_pose_bone_world(arm_obj, pb("upper_leg_l"), wy, -10.0)
        _rot_pose_bone_world(arm_obj, pb("upper_leg_r"), wy, 10.0)
        bpy.context.view_layer.update()
        _rot_pose_bone_world(arm_obj, pb("lower_leg_l"), wx, 92.0)
        _rot_pose_bone_world(arm_obj, pb("lower_leg_r"), wx, 92.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_l"), wz, -38.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_r"), wz, 38.0)
    elif pose_name == "stride":
        # Dynamic walking/running stride: L thigh forward (-Y), R thigh back (+Y) + opposite arms
        _rot_pose_bone_world(arm_obj, pb("hips"), wz, 8.0)
        _rot_pose_bone_world(arm_obj, pb("chest"), wz, -10.0)
        _rot_pose_bone_world(arm_obj, pb("upper_leg_l"), wx, -36.0)
        _rot_pose_bone_world(arm_obj, pb("upper_leg_r"), wx, 28.0)
        bpy.context.view_layer.update()
        _rot_pose_bone_world(arm_obj, pb("lower_leg_l"), wx, 18.0)
        _rot_pose_bone_world(arm_obj, pb("lower_leg_r"), wx, 24.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_l"), wz, 22.0)
        _rot_pose_bone_world(arm_obj, pb("upper_arm_r"), wz, 24.0)

    bpy.context.view_layer.update()


def _proj_to_seg(pt: Vector, a: Vector, b: Vector) -> Vector:
    ab = b - a
    denom = ab.length_squared
    if denom <= 1e-10:
        return a
    t = max(0.0, min(1.0, (pt - a).dot(ab) / denom))
    return a + ab * t


def _collect_ray_hits(bvh: BVHTree, origin: Vector, direction: Vector, max_t: float, max_hits: int = 4):
    hits = []
    cur_t = 0.0
    eps = 0.0008
    for _ in range(max_hits):
        if cur_t >= max_t:
            break
        loc, norm, _, dist = bvh.ray_cast(origin + direction * cur_t, direction, max_t - cur_t)
        if loc is None:
            break
        t_tot = cur_t + dist
        hits.append((t_tot, loc, norm))
        cur_t = t_tot + eps
    return hits


def _build_bone_anchors(arm_obj, skel_map):
    """Build true joint-to-child-joint anatomical bone centerlines (VRM tails are often short stubs)."""
    slots = skel_map["slots"]
    mw = arm_obj.matrix_world
    segments = {}
    pairs = (
        ("torso_hips", "hips", "spine"),
        ("torso_spine", "spine", "chest"),
        ("torso_chest", "chest", "upper_chest"),
        ("torso_upper_chest", "upper_chest", "neck"),
        ("arm_l_upper", "upper_arm_l", "lower_arm_l"),
        ("arm_l_lower", "lower_arm_l", "hand_l"),
        ("arm_r_upper", "upper_arm_r", "lower_arm_r"),
        ("arm_r_lower", "lower_arm_r", "hand_r"),
        ("leg_l_upper", "upper_leg_l", "lower_leg_l"),
        ("leg_r_upper", "upper_leg_r", "lower_leg_r"),
        ("leg_l_lower", "lower_leg_l", "foot_l"),
        ("leg_r_lower", "lower_leg_r", "foot_r"),
    )
    for key, sname, child_slot in pairs:
        bname = slots.get(sname)
        pb = arm_obj.pose.bones.get(bname) if bname else None
        if not pb:
            continue
        cname = slots.get(child_slot)
        cb = arm_obj.pose.bones.get(cname) if cname else None
        p_end = (mw @ cb.head) if cb else (mw @ pb.tail)
        segments[key] = (mw @ pb.head, p_end)
    return segments


def _select_monitored_body_verts(arm_obj, skin_obj, garment_obj, skel_map):
    """Identify rest-covered body vertices inside garment zone (excludes bare head/hands/calves & behind-pendulum calves)."""
    deps = bpy.context.evaluated_depsgraph_get()
    g_bvh = BVHTree.FromObject(garment_obj, deps)
    g_pts = [garment_obj.matrix_world @ v.co for v in garment_obj.data.vertices]
    # Find bilateral enclosing cuff vertical bounds (ignoring unilateral hanging bells/ribbons!)
    gz0_raw, gz1 = min(p.z for p in g_pts), max(p.z for p in g_pts)
    enc_zs = []
    for k in range(20):
        zt = gz0_raw + (k + 0.5) / 20.0 * (gz1 - gz0_raw)
        band = [p.x for p in g_pts if abs(p.z - zt) <= 0.03]
        if band and min(band) < -0.05 and max(band) > 0.05:
            enc_zs.append(zt)
    gz0 = (min(enc_zs) - 0.02) if enc_zs else gz0_raw

    segs = _build_bone_anchors(arm_obj, skel_map)
    vg_names = {vg.index: vg.name for vg in skin_obj.vertex_groups}

    def _zone_from_bone(bone_low: str, pw: Vector):
        if "bust" in bone_low or "upperchest" in bone_low or "shoulder" in bone_low:
            if "upper_chest" in segs or "torso_upper_chest" in segs:
                return "torso_upper_chest"
        if "chest" in bone_low and "torso_chest" in segs:
            return "torso_chest"
        if "spine" in bone_low and "torso_spine" in segs:
            return "torso_spine"
        if "hips" in bone_low and "torso_hips" in segs:
            return "torso_hips"
        side_l = ("_l_" in bone_low or bone_low.endswith(".l") or "left" in bone_low)
        side_r = ("_r_" in bone_low or bone_low.endswith(".r") or "right" in bone_low)
        if "upperarm" in bone_low or "upper_arm" in bone_low:
            k = "arm_l_upper" if side_l else ("arm_r_upper" if side_r else "")
            if k in segs:
                return k
        if "lowerarm" in bone_low or "lower_arm" in bone_low or "forearm" in bone_low:
            k = "arm_l_lower" if side_l else ("arm_r_lower" if side_r else "")
            if k in segs:
                return k
        if "upperleg" in bone_low or "upper_leg" in bone_low or "thigh" in bone_low:
            k = "leg_l_upper" if side_l else ("leg_r_upper" if side_r else "")
            if k in segs:
                return k
        if "lowerleg" in bone_low or "lower_leg" in bone_low or "calf" in bone_low or "shin" in bone_low:
            k = "leg_l_lower" if side_l else ("leg_r_lower" if side_r else "")
            if k in segs:
                return k
        return min(segs.items(), key=lambda kv: (pw - _proj_to_seg(pw, kv[1][0], kv[1][1])).length)[0]

    monitored = []
    rest_clipping = []
    mw_s = skin_obj.matrix_world
    for v in skin_obj.data.vertices:
        pw = mw_s @ v.co
        if pw.z < gz0 + 0.015 or pw.z > gz1 - 0.015:
            continue
        dom = max(v.groups, key=lambda g: g.weight, default=None)
        if not dom:
            continue
        bname = vg_names.get(dom.group, "").lower()
        if any(k in bname for k in ("head", "neck", "thumb", "index", "middle", "ring", "little", "toe", "foot")):
            continue
        zone = _zone_from_bone(bname, pw)
        a, b = segs[zone]
        axis_p = _proj_to_seg(pw, a, b)
        vec = pw - axis_p
        d_body = vec.length
        if d_body < 0.01:
            continue
        direction = vec / d_body
        hits = _collect_ray_hits(g_bvh, axis_p, direction, d_body + 0.14)
        outer_hits = [t for t, _, _ in hits if t >= d_body - 0.001]
        inner_hits = [t for t, _, _ in hits if d_body - 0.042 < t < d_body - 0.002]
        if outer_hits:
            monitored.append((v.index, zone))
        elif inner_hits:
            # Verify skin vertex is genuinely within 25mm of cloth surface rather than facing an open high-low front cutout
            loc_n, _, _, d_surf = g_bvh.find_nearest(pw, 0.025)
            if loc_n is not None:
                monitored.append((v.index, zone))
                pen_mm = (d_body - inner_hits[-1]) * 1000.0
                rest_clipping.append((v.index, zone, pen_mm))
    return monitored, rest_clipping


def _measure_pose_deformation(arm_obj, skin_obj, garment_obj, skel_map, monitored_verts, rest_edge_lengths, pendant_verts):
    """Evaluate real depsgraph deformed body + garment in current pose for clipping, collapse, and stretch."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev_skin = skin_obj.evaluated_get(deps)
    ev_garm = garment_obj.evaluated_get(deps)

    # Read evaluated world coordinates cleanly without leaving temporary meshes
    me_s = bpy.data.meshes.new_from_object(ev_skin)
    me_g = bpy.data.meshes.new_from_object(ev_garm)
    try:
        bvh_g = BVHTree.FromObject(ev_garm, deps)
        segs = _build_bone_anchors(arm_obj, skel_map)
        mw_s = ev_skin.matrix_world
        mw_g = ev_garm.matrix_world

        clip_cnt = 0
        max_pen_mm = 0.0
        zone_clips = Counter()
        for vi, zone in monitored_verts:
            pw = mw_s @ me_s.vertices[vi].co
            a, b = segs[zone]
            axis_p = _proj_to_seg(pw, a, b)
            vec = pw - axis_p
            d_body = vec.length
            if d_body < 0.01:
                continue
            direction = vec / d_body
            hits = _collect_ray_hits(bvh_g, axis_p, direction, d_body + 0.08)
            inner_hits = [t for t, _, _ in hits if d_body - 0.042 < t < d_body - 0.002]
            outer_hits = [t for t, _, _ in hits if t >= d_body - 0.0015]
            if inner_hits and not outer_hits:
                clip_cnt += 1
                pen = (d_body - inner_hits[-1]) * 1000.0
                max_pen_mm = max(max_pen_mm, pen)
                zone_clips[zone] += 1

        # Edge compression & overstretch ratios relative to Rest Pose
        collapsed = 0
        overstretched = 0
        max_single_stretch = 1.0
        total_ratio_dev = 0.0
        n_edges = max(1, len(me_g.edges))
        pend_max_stretch = 1.0
        for ei, e in enumerate(me_g.edges):
            l0 = rest_edge_lengths[ei]
            if l0 <= 1e-6:
                continue
            p0 = mw_g @ me_g.vertices[e.vertices[0]].co
            p1 = mw_g @ me_g.vertices[e.vertices[1]].co
            ratio = (p0 - p1).length / l0
            total_ratio_dev += abs(ratio - 1.0)
            if l0 >= 0.002:
                max_single_stretch = max(max_single_stretch, ratio)
            if ratio < 0.42:
                collapsed += 1
            elif ratio > 1.85:
                overstretched += 1
            if e.vertices[0] in pendant_verts and e.vertices[1] in pendant_verts:
                pend_max_stretch = max(pend_max_stretch, ratio)

        return {
            "monitored_body_verts": len(monitored_verts),
            "clipping_verts_count": clip_cnt,
            "clipping_body_verts_pct": round(100.0 * clip_cnt / max(1, len(monitored_verts)), 3),
            "max_penetration_mm": round(max_pen_mm, 2),
            "zone_clips": dict(zone_clips),
            "collapsed_edges_pct": round(100.0 * collapsed / n_edges, 3),
            "overstretched_edges_pct": round(100.0 * overstretched / n_edges, 3),
            "max_single_edge_stretch_ratio": round(max_single_stretch, 3),
            "mean_edge_strain_pct": round(100.0 * total_ratio_dev / n_edges, 3),
            "pendulum_max_stretch_ratio": round(pend_max_stretch, 3),
        }
    finally:
        bpy.data.meshes.remove(me_s)
        bpy.data.meshes.remove(me_g)


def _audit_weight_crosstalk(arm_obj, garment_obj, skel_map):
    slots = skel_map["slots"]
    hips_b = arm_obj.data.bones.get(slots.get("hips", ""))
    hips_z = (arm_obj.matrix_world @ hips_b.head_local).z if hips_b else 0.98
    spine_b = arm_obj.data.bones.get(slots.get("spine", ""))
    waist_z = (arm_obj.matrix_world @ spine_b.head_local).z if spine_b else 1.08

    arm_names = set(skel_map["arm_bones_all"])
    leg_l_names = set(skel_map["leg_l_bones"])
    leg_r_names = set(skel_map["leg_r_bones"])
    vg_names = {vg.index: vg.name for vg in garment_obj.vertex_groups}

    # Precompute arm segments
    mw_arm = arm_obj.matrix_world
    arm_segs = []
    for sa, sb in (("upper_arm_l", "hand_l"), ("upper_arm_r", "hand_r")):
        ba = arm_obj.data.bones.get(slots.get(sa, ""))
        bb = arm_obj.data.bones.get(slots.get(sb, ""))
        if ba and bb:
            arm_segs.append((mw_arm @ ba.head_local, mw_arm @ bb.tail_local))

    mw_g = garment_obj.matrix_world
    n_verts = max(1, len(garment_obj.data.vertices))
    # Detect if garment actually has bilateral sleeve coverage above waist
    upper_arm_pts = [mw_g @ v.co for v in garment_obj.data.vertices if (mw_g @ v.co).z > waist_z + 0.04 and abs((mw_g @ v.co).x) > 0.21]
    has_sleeves = len(upper_arm_pts) > 35
    arm_skirt_leak = 0
    cross_leg_leak = 0
    over4 = 0
    unnorm = 0
    unweighted = 0

    for v in garment_obj.data.vertices:
        pw = mw_g @ v.co
        nz = [g for g in v.groups if g.weight > 1e-4]
        if not nz:
            unweighted += 1
        if len(nz) > 4:
            over4 += 1
        if nz and abs(sum(g.weight for g in nz) - 1.0) > 0.02:
            unnorm += 1

        d_arm = min(((pw - _proj_to_seg(pw, a, b)).length for a, b in arm_segs), default=9.9)
        is_sleeve = has_sleeves and d_arm < 0.16 and pw.z > hips_z - 0.08

        arm_w = sum(g.weight for g in nz if vg_names.get(g.group) in arm_names)
        if not has_sleeves:
            if arm_w > 0.02:
                arm_skirt_leak += 1
        elif not is_sleeve and (pw.z < waist_z + 0.04 or abs(pw.x) < 0.145) and arm_w > 0.03:
            arm_skirt_leak += 1

        if not is_sleeve and pw.z <= hips_z + 0.03:
            leg_r_w = sum(g.weight for g in nz if vg_names.get(g.group) in leg_r_names)
            leg_l_w = sum(g.weight for g in nz if vg_names.get(g.group) in leg_l_names)
            if (pw.x > 0.095 and leg_r_w > 0.18) or (pw.x < -0.095 and leg_l_w > 0.18):
                cross_leg_leak += 1

    return {
        "arm_to_skirt_crosstalk_pct": round(100.0 * arm_skirt_leak / n_verts, 3),
        "cross_leg_crosstalk_pct": round(100.0 * cross_leg_leak / n_verts, 3),
        "verts_over_4_influences": over4,
        "unnormalized_verts": unnorm,
        "unweighted_verts": unweighted,
    }


def _render_pose_suite(arm_obj, body_objs, garment_obj, skel_map, renders_dir: Path):
    renders_dir.mkdir(parents=True, exist_ok=True)
    sc = bpy.context.scene
    w = bpy.data.worlds.new("ClothQCWorld")
    sc.world = w
    w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.86, 0.88, 0.91, 1.0)
    bg.inputs[1].default_value = 0.44

    target = Vector((0.0, 0.0, 0.86))
    for lname, loc, eng in [
        ("QC_Key", (-1.8, -2.4, 2.0), 65.0),
        ("QC_Fill", (2.0, -2.1, 1.2), 38.0),
        ("QC_Rim", (0.0, 2.6, 2.1), 46.0),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng
        ld.size = 2.4
        lo = bpy.data.objects.new(lname, ld)
        sc.collection.objects.link(lo)
        lo.location = Vector(loc)
        lo.rotation_euler = (target - lo.location).to_track_quat("-Z", "Y").to_euler()

    cam_d = bpy.data.cameras.new("QCCam")
    cam = bpy.data.objects.new("QCCam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam

    configure_cycles(sc)
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 640
    sc.render.resolution_y = 640
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.exposure = -0.5

    def snap(cam_loc, tgt, out_png):
        cam.location = Vector(cam_loc)
        cam.rotation_euler = (Vector(tgt) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = 48
        sc.render.filepath = str(out_png)
        bpy.ops.render.render(write_still=True)
        return str(out_png)

    out_paths = {}
    for pname, cam_pos, tgt_pos in [
        ("rest", (0.0, -2.65, 0.90), (0.0, 0.0, 0.85)),
        ("rest_34", (-1.55, -2.15, 1.05), (0.0, 0.0, 0.85)),
        ("arms_up", (-0.85, -2.55, 1.05), (0.0, 0.0, 0.92)),
        ("torso_twist", (-1.45, -2.20, 1.02), (0.0, 0.0, 0.88)),
        ("deep_squat", (-1.85, -2.10, 0.88), (0.0, 0.0, 0.68)),
        ("stride", (-2.15, -1.75, 0.92), (0.0, 0.0, 0.82)),
    ]:
        pose_key = "rest" if pname == "rest_34" else pname
        _apply_stress_pose(arm_obj, skel_map, pose_key)
        out_paths[pname] = snap(cam_pos, tgt_pos, renders_dir / f"pose_{pname}.png")

    _reset_pose(arm_obj)
    return out_paths


def evaluate_cloth_rig(metrics, criteria):
    pose_metrics = metrics["poses"]
    weights = metrics["weight_crosstalk"]

    max_clip_pct = max(p["clipping_body_verts_pct"] for p in pose_metrics.values())
    rest_clip_pct = pose_metrics["rest"]["clipping_body_verts_pct"]
    max_pen_mm = max(p["max_penetration_mm"] for p in pose_metrics.values())
    max_collapse = max(p["collapsed_edges_pct"] for p in pose_metrics.values())
    max_overstretch = max(p["overstretched_edges_pct"] for p in pose_metrics.values())
    max_single_stretch = max(p.get("max_single_edge_stretch_ratio", 1.0) for p in pose_metrics.values())
    max_pend_stretch = max(p["pendulum_max_stretch_ratio"] for p in pose_metrics.values())

    blocking = []
    failure_classes = []

    if rest_clip_pct > criteria.get("max_rest_clipping_pct", 1.5):
        blocking.append(f"rest_clipping {rest_clip_pct}% > {criteria.get('max_rest_clipping_pct', 1.5)}%")
        failure_classes.append("body_clipping")
    if max_clip_pct > criteria.get("max_pose_clipping_pct", 3.5):
        blocking.append(f"pose_clipping {max_clip_pct}% > {criteria.get('max_pose_clipping_pct', 3.5)}%")
        failure_classes.append("pose_body_clipping")
    if max_pen_mm > criteria.get("max_penetration_mm", 14.0):
        blocking.append(f"max_penetration {max_pen_mm}mm > {criteria.get('max_penetration_mm', 14.0)}mm")
        failure_classes.append("deep_penetration")
    if max_collapse > criteria.get("max_collapsed_edges_pct", 2.0):
        blocking.append(f"collapsed_edges {max_collapse}% > {criteria.get('max_collapsed_edges_pct', 2.0)}%")
        failure_classes.append("mesh_collapse")
    if max_overstretch > criteria.get("max_overstretched_edges_pct", 3.5):
        blocking.append(f"overstretched_edges {max_overstretch}% > {criteria.get('max_overstretched_edges_pct', 3.5)}%")
        failure_classes.append("mesh_overstretch")
    if max_single_stretch > criteria.get("max_single_edge_stretch_ratio", 3.4):
        blocking.append(f"max_single_edge_stretch {max_single_stretch}x > {criteria.get('max_single_edge_stretch_ratio', 3.4)}x")
        failure_classes.append("skinning_spike")
    if weights["arm_to_skirt_crosstalk_pct"] > criteria.get("max_arm_skirt_crosstalk_pct", 0.5):
        blocking.append(f"arm_skirt_crosstalk {weights['arm_to_skirt_crosstalk_pct']}%")
        failure_classes.append("arm_skirt_crosstalk")
    if weights["cross_leg_crosstalk_pct"] > criteria.get("max_cross_leg_crosstalk_pct", 1.0):
        blocking.append(f"cross_leg_crosstalk {weights['cross_leg_crosstalk_pct']}%")
        failure_classes.append("cross_leg_crosstalk")
    if max_pend_stretch > criteria.get("max_pendulum_stretch_ratio", 1.18):
        blocking.append(f"pendulum_stretch {max_pend_stretch}x > {criteria.get('max_pendulum_stretch_ratio', 1.18)}x")
        failure_classes.append("rigid_accessory_stretch")
    if weights["verts_over_4_influences"] > 0 or weights["unnormalized_verts"] > 0 or weights["unweighted_verts"] > 0:
        blocking.append("rig_weight_contract_violation")
        failure_classes.append("rig_weight_overflow")

    penalties = (
        max(0.0, rest_clip_pct - 0.8) * 6.0
        + max(0.0, max_clip_pct - 1.5) * 4.5
        + max(0.0, max_collapse - 1.0) * 5.0
        + max(0.0, max_overstretch - 2.0) * 4.0
        + max(0.0, max_single_stretch - 2.6) * 8.0
        + weights["arm_to_skirt_crosstalk_pct"] * 10.0
        + weights["cross_leg_crosstalk_pct"] * 8.0
        + max(0.0, max_pend_stretch - 1.12) * 35.0
    )
    score = round(max(0.0, 100.0 - penalties), 2)
    return {
        "passed": len(blocking) == 0,
        "score": 100.0 if len(blocking) == 0 and score >= 97.5 else score,
        "blocking_failures": blocking,
        "failure_classes": list(dict.fromkeys(failure_classes)),
        "summary": {
            "rest_clipping_pct": rest_clip_pct,
            "worst_pose_clipping_pct": max_clip_pct,
            "worst_penetration_mm": max_pen_mm,
            "worst_collapsed_edges_pct": max_collapse,
            "worst_overstretched_edges_pct": max_overstretch,
            "max_single_edge_stretch_ratio": max_single_stretch,
            "max_pendulum_stretch_ratio": max_pend_stretch,
        },
    }


def main():
    args = _parse_args()
    criteria = json.loads(Path(args.criteria_json).read_text(encoding="utf-8-sig"))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    load_scene_file(Path(args.candidate_blend))
    junk = sorted(obj.name for obj in junk_objects(bpy.context.scene.objects))
    arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    meshes = strip_bone_shape_junk([o for o in bpy.data.objects if o.type == "MESH"])
    garment_obj = next((o for o in meshes if "garment" in o.name.lower() or o.name == "Fitted_Garment"), None)
    body_objs = [o for o in meshes if o != garment_obj]
    skin_obj = next((o for o in body_objs if "skin" in o.name.lower() or "body" in o.name.lower()), body_objs[0])

    skel_map = map_humanoid_skeleton(arm_obj)
    _reset_pose(arm_obj)

    mw_g = garment_obj.matrix_world
    rest_edge_lengths = [
        ((mw_g @ garment_obj.data.vertices[e.vertices[0]].co) - (mw_g @ garment_obj.data.vertices[e.vertices[1]].co)).length
        for e in garment_obj.data.edges
    ]
    hips_b = arm_obj.data.bones.get(skel_map["slots"].get("hips", ""))
    hips_z = (arm_obj.matrix_world @ hips_b.head_local).z if hips_b else 0.98
    low_pts = [mw_g @ v.co for v in garment_obj.data.vertices if (mw_g @ v.co).z < hips_z - 0.08]
    # Only flag lower vertices as rigid off-center pendulum if lower garment is NOT a full bilateral skirt enclosing X=0
    has_offcenter_pendulum = bool(
        low_pts and not (min(p.x for p in low_pts) < -0.05 and max(p.x for p in low_pts) > 0.05)
    )
    pendant_verts = (
        {
            v.index
            for v in garment_obj.data.vertices
            if (mw_g @ v.co).z < hips_z - 0.08
        }
        if has_offcenter_pendulum
        else set()
    )

    monitored_verts, _ = _select_monitored_body_verts(arm_obj, skin_obj, garment_obj, skel_map)
    pose_names = ("rest", "arms_up", "torso_twist", "deep_squat", "stride")
    pose_metrics = {}
    for pname in pose_names:
        _apply_stress_pose(arm_obj, skel_map, pname)
        pose_metrics[pname] = _measure_pose_deformation(
            arm_obj, skin_obj, garment_obj, skel_map, monitored_verts, rest_edge_lengths, pendant_verts
        )
    _reset_pose(arm_obj)

    weight_audit = _audit_weight_crosstalk(arm_obj, garment_obj, skel_map)
    renders = _render_pose_suite(arm_obj, body_objs, garment_obj, skel_map, Path(args.renders_dir))

    full_metrics = {
        "poses": pose_metrics,
        "weight_crosstalk": weight_audit,
        "renders": renders,
    }
    eval_res = evaluate_cloth_rig(full_metrics, criteria)
    eval_res["junk_objects"] = junk
    if junk:
        eval_res["passed"] = False
        eval_res["blocking_failures"].append(f"Engine hygiene: {junk}")
        eval_res["failure_classes"].append("engine_junk")
    payload = {
        "metrics": full_metrics,
        "evaluation": eval_res,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Cloth-rig validation complete ->", eval_res["score"], "passed=", eval_res["passed"])


if __name__ == "__main__":
    main()
