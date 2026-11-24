"""Headless Blender Stage 5: apply Codex-optimized UV textures, eye slots, Toon matte fix & export."""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def _find_principled(mat):
    if not mat or not mat.use_nodes:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _replace_role_image(mat, role_keywords, new_path, colorspace):
    if not new_path or not Path(new_path).is_file():
        return False
    img = bpy.data.images.load(str(new_path), check_existing=True)
    img.colorspace_settings.name = colorspace
    replaced = False
    for node in mat.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            lname = node.image.name.lower()
            if any(kw in lname for kw in role_keywords):
                node.image = img
                replaced = True
    return replaced


def _make_eye_material(name, tex_path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _find_principled(mat)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    tex_node = nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(tex_path), check_existing=True)
    img.colorspace_settings.name = "sRGB"
    tex_node.image = img
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.38
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    return mat


def _split_and_project_eye(mesh_obj, x_min, x_max, z_lo, z_hi, y_max, mat_index):
    me = mesh_obj.data
    mw = mesh_obj.matrix_world
    uv_layer = me.uv_layers.active.data if me.uv_layers.active else me.uv_layers.new(name="UVMap").data
    selected_polys = []
    for poly in me.polygons:
        c = mw @ poly.center
        if z_lo <= c.z <= z_hi and c.y < y_max and x_min <= c.x <= x_max:
            selected_polys.append(poly)
    if not selected_polys:
        return 0

    verts_world = [mw @ me.vertices[vi].co for p in selected_polys for vi in p.vertices]
    min_x = min(v.x for v in verts_world)
    max_x = max(v.x for v in verts_world)
    min_z = min(v.z for v in verts_world)
    max_z = max(v.z for v in verts_world)
    span_x = max(1e-5, max_x - min_x)
    span_z = max(1e-5, max_z - min_z)

    for poly in selected_polys:
        poly.material_index = mat_index
        for li in poly.loop_indices:
            wp = mw @ me.vertices[me.loops[li].vertex_index].co
            uv_layer[li].uv = ((wp.x - min_x) / span_x, (wp.z - min_z) / span_z)
    return len(selected_polys)


def _calibrate_matte_toon(mat, roughness_floor=0.80, normal_strength=0.35):
    bsdf = _find_principled(mat)
    if not bsdf:
        return
    if not bsdf.inputs["Roughness"].is_linked:
        bsdf.inputs["Roughness"].default_value = max(
            bsdf.inputs["Roughness"].default_value, roughness_floor
        )
    if not bsdf.inputs["Metallic"].is_linked:
        bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = min(
            bsdf.inputs["Specular IOR Level"].default_value, 0.25
        )
    for node in mat.node_tree.nodes:
        if node.type == "NORMAL_MAP" and "Strength" in node.inputs:
            node.inputs["Strength"].default_value = float(normal_strength)


def snap(scene, cam, pos, tgt, lens, out_png):
    bpy.context.view_layer.update()
    cam.data.lens = lens
    cam.location = Vector(pos)
    cam.rotation_euler = (Vector(tgt) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out_png)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    apply_spec = json.loads(Path(cfg["apply_spec_json"]).read_text(encoding="utf-8-sig"))

    bpy.ops.wm.open_mainfile(filepath=cfg["input_blend"])
    scene = bpy.context.scene
    arm = next((o for o in scene.objects if o.type == "ARMATURE"), None)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    primary = max(meshes, key=lambda o: len(o.data.vertices))

    if arm and arm.animation_data:
        saved_action = arm.animation_data.action
        arm.animation_data.action = None
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.scale = (1, 1, 1)
    else:
        saved_action = None
    bpy.context.view_layer.update()

    report = {"replaced_diffuse": False, "replaced_roughness": False, "split_eyes": {}}
    base_mat = primary.data.materials[0] if primary.data.materials else None
    if base_mat:
        report["replaced_diffuse"] = _replace_role_image(
            base_mat, ("diffuse", "base", "color"), apply_spec.get("diffuse_override"), "sRGB"
        )
        report["replaced_roughness"] = _replace_role_image(
            base_mat, ("rough", "metal"), apply_spec.get("roughness_override"), "Non-Color"
        )
        if apply_spec.get("toon_shader_blend", True):
            _calibrate_matte_toon(
                base_mat,
                roughness_floor=float(apply_spec.get("skin_roughness_floor", 0.80)),
                normal_strength=float(apply_spec.get("normal_strength", 0.35)),
            )

    target_height = float(cfg.get("target_height", 1.56))
    scale_h = target_height / 1.56
    if apply_spec.get("split_eye_materials"):
        eye_l_path = apply_spec.get("eye_left_texture")
        eye_r_path = apply_spec.get("eye_right_texture")
        if eye_r_path and Path(eye_r_path).is_file():
            mat_r = _make_eye_material("Eye_Right_HiRes", eye_r_path)
            primary.data.materials.append(mat_r)
            report["split_eyes"]["right_polys"] = _split_and_project_eye(
                primary,
                -0.065 * scale_h,
                -0.015 * scale_h,
                1.35 * scale_h,
                1.41 * scale_h,
                -0.18 * scale_h,
                len(primary.data.materials) - 1,
            )
        if eye_l_path and Path(eye_l_path).is_file():
            mat_l = _make_eye_material("Eye_Left_HiRes", eye_l_path)
            primary.data.materials.append(mat_l)
            report["split_eyes"]["left_polys"] = _split_and_project_eye(
                primary,
                0.015 * scale_h,
                0.065 * scale_h,
                1.35 * scale_h,
                1.41 * scale_h,
                -0.18 * scale_h,
                len(primary.data.materials) - 1,
            )

    if arm and saved_action:
        arm.animation_data.action = saved_action

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

    cam = scene.camera or next((o for o in scene.objects if o.type == "CAMERA"), None)
    renders_dir = Path(cfg["renders_dir"])
    renders_dir.mkdir(parents=True, exist_ok=True)
    rendered = {}
    if cam:
        h = target_height
        for fname, pos, tgt, lens in [
            ("qc_front.png", (0.0, -2.55 * h / 1.56, 0.54 * h), (0.0, 0.0, 0.51 * h), 46),
            ("qc_34.png", (-1.65 * h / 1.56, -1.85 * h / 1.56, 0.62 * h), (0.0, 0.0, 0.52 * h), 48),
            ("qc_face.png", (-0.28 * h / 1.56, -0.72 * h / 1.56, 0.91 * h), (0.0, -0.08 * h / 1.56, 0.885 * h), 55),
        ]:
            dst = renders_dir / fname
            snap(scene, cam, pos, tgt, lens, dst)
            rendered[fname] = str(dst)

    report["renders"] = rendered
    Path(cfg["report_json"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Stage 5 apply_uv_toon_bake completed successfully.")


if __name__ == "__main__":
    main()
