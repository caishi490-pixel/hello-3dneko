"""Headless Blender Stage 3: clean Quad/GLB meshes, fix shell bleed, extract textures & render views."""

import argparse
import json
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def configure_cycles(scene, device_type="OPTIX", samples=48, res=896):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    used_device = "CPU"
    try:
        prefs.compute_device_type = device_type
        prefs.refresh_devices()
        gpu_ok = False
        for dev in prefs.devices:
            dev.use = dev.type != "CPU"
            if dev.use:
                gpu_ok = True
        if gpu_ok:
            scene.cycles.device = "GPU"
            used_device = f"GPU:{device_type}"
        else:
            scene.cycles.device = "CPU"
    except Exception:
        scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.view_settings.view_transform = "Standard"
    return used_device


def setup_studio(scene, target_height):
    world = bpy.data.worlds.new("Studio")
    scene.world = world
    world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg:
        bg.inputs[0].default_value = (0.94, 0.94, 0.96, 1.0)
        bg.inputs[1].default_value = 0.55

    def add_area(name, loc, z_tgt, energy, size):
        ldata = bpy.data.lights.new(name, type="AREA")
        ldata.energy = energy
        ldata.size = size
        lobj = bpy.data.objects.new(name, ldata)
        scene.collection.objects.link(lobj)
        lobj.location = Vector(loc)
        direction = Vector((0.0, 0.0, z_tgt)) - Vector(loc)
        lobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    h_mid = target_height * 0.58
    add_area("KeyLight", (-2.0, -2.5, target_height * 1.4), h_mid, 95.0, 2.4)
    add_area("FillLight", (2.2, -2.2, target_height * 0.95), h_mid, 65.0, 2.8)
    add_area("RimLight", (0.0, 2.5, target_height * 1.28), h_mid, 88.0, 2.0)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def snap(scene, cam, pos, tgt, lens, out_png):
    bpy.context.view_layer.update()
    cam.data.lens = lens
    cam.location = Vector(pos)
    cam.rotation_euler = (Vector(tgt) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out_png)
    bpy.ops.render.render(write_still=True)


def normalize_scene_scale(scene, mesh_objs, arm_obj, target_height):
    root_obj = arm_obj if arm_obj is not None else mesh_objs[0]
    pts = []
    for mobj in mesh_objs:
        mw = mobj.matrix_world
        step = max(1, len(mobj.data.vertices) // 5000)
        pts.extend(mw @ mobj.data.vertices[i].co for i in range(0, len(mobj.data.vertices), step))
    min_z = min(p.z for p in pts)
    max_z = max(p.z for p in pts)
    scale_f = target_height / max(1e-4, max_z - min_z)
    root_obj.scale *= scale_f
    bpy.context.view_layer.update()

    pts2 = []
    for mobj in mesh_objs:
        mw = mobj.matrix_world
        step = max(1, len(mobj.data.vertices) // 5000)
        pts2.extend(mw @ mobj.data.vertices[i].co for i in range(0, len(mobj.data.vertices), step))
    cx = 0.5 * (min(p.x for p in pts2) + max(p.x for p in pts2))
    cy = 0.5 * (min(p.y for p in pts2) + max(p.y for p in pts2))
    mz = min(p.z for p in pts2)
    root_obj.location -= Vector((cx, cy, mz))
    bpy.context.view_layer.update()


def clean_quad_topology(mesh_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh_obj.data.update()
    quad_count = sum(1 for p in mesh_obj.data.polygons if len(p.vertices) == 4)
    return {"quads": quad_count, "polys": len(mesh_obj.data.polygons)}


def strip_shell_weight_bleed(mesh_obj, target_height):
    vg_names = [vg.name for vg in mesh_obj.vertex_groups]
    vg_map = {vg.name: vg.index for vg in mesh_obj.vertex_groups}
    arm_ids = {
        i
        for i, name in enumerate(vg_names)
        if any(tok in name for tok in ("ForeArm", "Hand", "Thumb", "Index", "Middle", "Ring", "Pinky"))
    }
    if not arm_ids:
        return 0
    scale_h = target_height / 1.56
    mw = mesh_obj.matrix_world
    cleaned = 0
    for vert in mesh_obj.data.vertices:
        wp = mw @ vert.co
        in_skirt_core = (abs(wp.x) < 0.245 * scale_h and wp.z < 0.90 * scale_h) or (
            abs(wp.x) < 0.295 * scale_h and wp.z < 0.74 * scale_h
        )
        if not in_skirt_core:
            continue
        bad = [g for g in vert.groups if g.group in arm_ids and g.weight > 0.0]
        if not bad:
            continue
        cleaned += 1
        for g in bad:
            mesh_obj.vertex_groups[g.group].remove([vert.index])
        rem = [(g.group, g.weight) for g in vert.groups if g.group not in arm_ids]
        rem_sum = sum(w for _, w in rem)
        if rem_sum > 1e-4:
            factor = 1.0 / rem_sum
            for gid, w in rem:
                mesh_obj.vertex_groups[gid].add([vert.index], w * factor, "REPLACE")
        elif "Hips" in vg_map:
            mesh_obj.vertex_groups["Hips"].add([vert.index], 0.75, "REPLACE")
            if "Spine" in vg_map:
                mesh_obj.vertex_groups["Spine"].add([vert.index], 0.25, "REPLACE")
    arm_mod = next((m for m in mesh_obj.modifiers if m.type == "ARMATURE"), None)
    if arm_mod:
        arm_mod.use_deform_preserve_volume = True
    return cleaned


def extract_textures(mesh_obj, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}
    for mat in mesh_obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not bsdf:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            lname = node.image.name.lower()
            if "diffuse" in lname or "base" in lname or "color" in lname:
                dst = out_dir / "diffuse_extracted.png"
                node.image.filepath_raw = str(dst)
                node.image.file_format = "PNG"
                node.image.save()
                extracted["diffuse"] = str(dst)
                extracted["diffuse_size"] = list(node.image.size)
            elif "rough" in lname or "metal" in lname:
                dst = out_dir / "roughness_extracted.png"
                node.image.filepath_raw = str(dst)
                node.image.file_format = "PNG"
                node.image.save()
                extracted["roughness"] = str(dst)
    return extracted


def probe_eye_uv_regions(mesh_obj, diffuse_path, target_height):
    if not diffuse_path or not Path(diffuse_path).is_file() or not mesh_obj.data.uv_layers.active:
        return {}
    img = next((im for im in bpy.data.images if "diffuse" in im.name.lower()), None)
    if not img:
        return {}
    w_px, h_px = img.size
    if w_px <= 0 or h_px <= 0:
        return {}
    px_arr = np.array(img.pixels[:], dtype=np.float32).reshape((h_px, w_px, 4))
    uv_layer = mesh_obj.data.uv_layers.active.data
    mw = mesh_obj.matrix_world
    scale_h = target_height / 1.56
    z_lo, z_hi = 1.35 * scale_h, 1.42 * scale_h

    def scan_side(x_lo, x_hi):
        hits = []
        for poly in mesh_obj.data.polygons:
            center = mw @ poly.center
            if z_lo <= center.z <= z_hi and center.y < -0.15 * scale_h and x_lo <= center.x <= x_hi:
                uvs = [uv_layer[li].uv for li in poly.loop_indices]
                cu = sum(u.x for u in uvs) / len(uvs)
                cv = sum(u.y for u in uvs) / len(uvs)
                px_x = max(0, min(w_px - 1, int(cu * w_px)))
                pil_y = max(0, min(h_px - 1, int((1.0 - cv) * h_px)))
                by = max(0, min(h_px - 1, int(cv * h_px)))
                r, g, b = px_arr[by, px_x, :3] * 255.0
                if not (r > 215 and g > 195 and b > 180):
                    hits.append((px_x, pil_y))
        if not hits:
            return None
        xs = [p[0] for p in hits]
        ys = [p[1] for p in hits]
        mx, my = int(np.median(xs)), int(np.median(ys))
        tight = [(x, y) for x, y in hits if abs(x - mx) < 120 and abs(y - my) < 120]
        if not tight:
            return None
        return {
            "pil_center": [mx, my],
            "pil_box": [
                max(0, min(x for x, _ in tight) - 4),
                max(0, min(y for _, y in tight) - 4),
                min(w_px, max(x for x, _ in tight) + 5),
                min(h_px, max(y for _, y in tight) + 5),
            ],
            "hit_polys": len(tight),
        }

    regions = {}
    right_eye = scan_side(-0.065 * scale_h, -0.015 * scale_h)
    left_eye = scan_side(0.015 * scale_h, 0.065 * scale_h)
    if right_eye:
        regions["right_eye"] = right_eye
    if left_eye:
        regions["left_eye"] = left_eye
    return regions


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=cfg["input_glb"])
    scene = bpy.context.scene

    for obj in list(scene.objects):
        if obj.name in ("Icosphere", "棱角球"):
            bpy.data.objects.remove(obj, do_unlink=True)

    arm = next((o for o in scene.objects if o.type == "ARMATURE"), None)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects found in {cfg['input_glb']}")

    saved_action = arm.animation_data.action if arm and arm.animation_data else None
    if arm and arm.animation_data:
        arm.animation_data.action = None
    if arm:
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()

    target_height = float(cfg.get("target_height", 1.56))
    normalize_scene_scale(scene, meshes, arm, target_height)
    primary = max(meshes, key=lambda o: len(o.data.vertices))
    primary.name = cfg.get("primary_mesh_name", "Character_Body")

    report = {"mode": cfg.get("mode", "rig_glb"), "primary_mesh": primary.name}
    if cfg.get("mode") == "quad":
        report["topology"] = clean_quad_topology(primary)
    else:
        report["shell_bleed_vertices_cleaned"] = strip_shell_weight_bleed(primary, target_height)

    textures = extract_textures(primary, cfg["textures_dir"])
    report["extracted_textures"] = textures
    report["eye_regions"] = probe_eye_uv_regions(primary, textures.get("diffuse"), target_height)

    if saved_action and arm:
        arm.animation_data.action = saved_action

    cam = setup_studio(scene, target_height)
    report["cycles_device"] = configure_cycles(
        scene,
        device_type=cfg.get("cycles_device_type", "OPTIX"),
        samples=int(cfg.get("render_samples", 48)),
        res=int(cfg.get("render_resolution", 896)),
    )

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=cfg["output_blend"])
    bpy.ops.export_scene.gltf(
        filepath=cfg["output_glb"],
        export_format="GLB",
        export_animations=True,
        export_skins=True,
    )

    if arm and arm.animation_data:
        arm.animation_data.action = None
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.scale = (1, 1, 1)
        bpy.context.view_layer.update()

    renders_dir = Path(cfg["renders_dir"])
    renders_dir.mkdir(parents=True, exist_ok=True)
    h = target_height
    views = [
        ("v1_front.png", (0.0, -2.55 * h / 1.56, 0.54 * h), (0.0, 0.0, 0.51 * h), 46),
        ("v2_side.png", (-2.55 * h / 1.56, 0.0, 0.54 * h), (0.0, 0.0, 0.51 * h), 46),
        ("v3_back.png", (0.0, 2.55 * h / 1.56, 0.54 * h), (0.0, 0.0, 0.51 * h), 46),
        ("v4_34.png", (-1.65 * h / 1.56, -1.85 * h / 1.56, 0.62 * h), (0.0, 0.0, 0.52 * h), 48),
        ("v5_face.png", (-0.28 * h / 1.56, -0.72 * h / 1.56, 0.91 * h), (0.0, -0.08 * h / 1.56, 0.885 * h), 55),
    ]
    rendered = {}
    for fname, pos, tgt, lens in views:
        dst = renders_dir / fname
        snap(scene, cam, pos, tgt, lens, dst)
        rendered[fname] = str(dst)

    if arm and saved_action and len(bpy.data.actions) > 0:
        first_action = bpy.data.actions[0]
        arm.animation_data.action = first_action
        scene.frame_set(min(22, int(first_action.frame_range[1])))
        pose_dst = renders_dir / "v6_action_rom.png"
        snap(scene, cam, (-0.85 * h / 1.56, -2.35 * h / 1.56, 0.58 * h), (0.0, 0.0, 0.51 * h), 46, pose_dst)
        rendered["v6_action_rom.png"] = str(pose_dst)

    report["renders"] = rendered
    Path(cfg["report_json"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Stage 3 clean_rig_and_render completed successfully.")


if __name__ == "__main__":
    main()
