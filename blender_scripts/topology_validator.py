"""Independent 6-Pillar Quality Validator & Wireframe/Shaded QC Renderer."""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from topology_inspector import inspect_current_scene, load_scene_file  # noqa: E402


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-glb", required=True)
    p.add_argument("--candidate-blend", default="")
    p.add_argument("--reference-glb", required=True)
    p.add_argument("--criteria-json", required=True)
    p.add_argument("--renders-dir", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args(argv)


def _render_qc_views(out_dir, prefix="cand", samples=28, res=768, mode="both"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = bpy.context.scene
    meshes = [o for o in sc.objects if o.type == "MESH"]
    if not meshes:
        return {}

    pts = []
    for o in meshes:
        step = max(1, len(o.data.vertices) // 2000)
        for i in range(0, len(o.data.vertices), step):
            pts.append(o.matrix_world @ o.data.vertices[i].co)
    min_p = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_p = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = 0.5 * (min_p + max_p)
    span = max((max_p - min_p).x, (max_p - min_p).y, (max_p - min_p).z, 0.1)
    dist = span * 2.1

    w = bpy.data.worlds.new("QCWorld")
    sc.world = w
    if w.node_tree is None:
        w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.94, 0.94, 0.96, 1.0)
    bg.inputs[1].default_value = 0.55

    for lname, off, eng in [
        ("QC_Key", Vector((-1.4, -1.6, 1.2)), 120),
        ("QC_Fill", Vector((1.6, -1.4, 0.6)), 75),
        ("QC_Rim", Vector((0.0, 1.8, 1.2)), 95),
    ]:
        ld = bpy.data.lights.new(lname, type="AREA")
        ld.energy = eng * max(1.0, span * span)
        ld.size = span * 1.5
        lo = bpy.data.objects.new(lname, ld)
        sc.collection.objects.link(lo)
        lo.location = center + off * span
        lo.rotation_euler = (center - lo.location).to_track_quat("-Z", "Y").to_euler()

    cam_d = bpy.data.cameras.new("QCCam")
    cam = bpy.data.objects.new("QCCam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam

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
    sc.cycles.use_denoising = True
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    sc.view_settings.view_transform = "Standard"

    def snap(vec, fname):
        cam.location = center + vec.normalized() * dist
        cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = 50
        dst = out_dir / fname
        sc.render.filepath = str(dst)
        bpy.ops.render.render(write_still=True)
        return str(dst)

    renders = {}
    if mode in ("both", "shaded"):
        renders["shaded_front"] = snap(Vector((0.0, -1.0, 0.12)), f"{prefix}_shaded_front.png")
        renders["shaded_34"] = snap(Vector((-0.75, -0.75, 0.25)), f"{prefix}_shaded_34.png")

    if mode not in ("both", "wire"):
        return renders

    clay = bpy.data.materials.new("QC_Clay")
    if clay.node_tree is None:
        clay.use_nodes = True
    c_bsdf = next(n for n in clay.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    c_bsdf.inputs["Base Color"].default_value = (0.76, 0.78, 0.84, 1.0)
    c_bsdf.inputs["Roughness"].default_value = 0.65

    total_polys = sum(len(o.data.polygons) for o in meshes)
    if total_polys <= 60000:
        wire = bpy.data.materials.new("QC_Wire")
        if wire.node_tree is None:
            wire.use_nodes = True
        w_bsdf = next(n for n in wire.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        w_bsdf.inputs["Base Color"].default_value = (0.04, 0.05, 0.09, 1.0)
        w_bsdf.inputs["Roughness"].default_value = 0.8
        for o in meshes:
            o.data.materials.clear()
            o.data.materials.append(clay)
            o.data.materials.append(wire)
            wf = o.modifiers.new("QCWireMod", "WIREFRAME")
            wf.thickness = span * 0.0018
            wf.use_even_offset = False
            wf.use_replace = False
            wf.material_offset = 1
            for p in o.data.polygons:
                p.use_smooth = False
    else:
        wnode = clay.node_tree.nodes.new("ShaderNodeWireframe")
        wnode.use_pixel_size = True
        wnode.inputs["Size"].default_value = 1.2
        mix = clay.node_tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs[6].default_value = (0.76, 0.78, 0.84, 1.0)
        mix.inputs[7].default_value = (0.05, 0.06, 0.12, 1.0)
        clay.node_tree.links.new(wnode.outputs["Fac"], mix.inputs[0])
        clay.node_tree.links.new(mix.outputs[2], c_bsdf.inputs["Base Color"])
        for o in meshes:
            o.data.materials.clear()
            o.data.materials.append(clay)

    renders["wire_front"] = snap(Vector((0.0, -1.0, 0.12)), f"{prefix}_wire_front.png")
    renders["wire_34"] = snap(Vector((-0.75, -0.75, 0.25)), f"{prefix}_wire_34.png")
    return renders


def evaluate_six_pillars(glb_report, blend_report, criteria):
    meshes = glb_report.get("meshes", [])
    if not meshes:
        return {
            "passed": False,
            "score": 0.0,
            "blocking_failures": ["No meshes found in candidate GLB"],
            "warnings": [],
            "pillars": {},
        }

    # Aggregate metrics across meshes (prefer raw .blend topology when available)
    blend_meshes = (blend_report or {}).get("meshes", [])
    faces_rec = sum(m["recovered_faces"] for m in meshes)
    total_faces_raw_blend = sum(m["faces_raw"] for m in blend_meshes) if blend_meshes else faces_rec
    blend_quad_ratio = (
        round(sum(m["quads_raw"] for m in blend_meshes) / max(1, total_faces_raw_blend), 4)
        if blend_meshes
        else round(
            sum(m["recovered_quad_ratio"] * m["recovered_faces"] for m in meshes) / max(1, faces_rec), 4
        )
    )
    glb_rec_quad_ratio = round(
        sum(m["recovered_quad_ratio"] * m["recovered_faces"] for m in meshes) / max(1, faces_rec), 4
    )
    eff_quad_ratio = max(blend_quad_ratio, glb_rec_quad_ratio)
    glb_val4_ratio = round(
        sum(m["recovered_valence4_ratio"] * m["recovered_faces"] for m in meshes) / max(1, faces_rec), 4
    )
    blend_val4_ratio = (
        round(
            sum(m.get("welded_valence4_ratio", 0.0) * m["faces_raw"] for m in blend_meshes)
            / max(1, total_faces_raw_blend),
            4,
        )
        if blend_meshes
        else glb_val4_ratio
    )
    val4_ratio = max(blend_val4_ratio, glb_val4_ratio)
    ortho_src = blend_meshes if blend_meshes else meshes
    quad_ortho = round(
        sum(m.get("quad_orthogonality_mean", 0.0) * m["faces_raw"] for m in ortho_src)
        / max(1, sum(m["faces_raw"] for m in ortho_src)),
        4,
    )
    pole_cluster_pct = round(
        sum(m.get("pole_cluster_pct", 0.0) * m["faces_raw"] for m in ortho_src)
        / max(1, sum(m["faces_raw"] for m in ortho_src)),
        2,
    )
    ngons = sum(m["ngons_raw"] for m in blend_meshes) if blend_meshes else sum(m["recovered_ngons"] for m in meshes)
    bad_aspect_pct = round(
        100.0 * sum(m["bad_aspect_faces"] for m in meshes) / max(1, sum(m["faces_raw"] for m in meshes)), 3
    )

    welded_non_man = sum(m["welded_non_manifold_edges"] for m in meshes)
    degen = sum(m["degenerate_faces"] for m in meshes)
    zero_e = sum(m["zero_length_edges"] for m in meshes)
    loose_v = sum(m["loose_verts"] for m in meshes)
    debris = sum(m["debris_shells_count"] for m in meshes)
    flipped_pct = round(
        100.0 * sum(m["flipped_normals_estimate"] for m in meshes) / max(1, sum(m["faces_raw"] for m in meshes)), 3
    )

    fidelity = glb_report.get("fidelity") or {"valid": False}
    has_uv = all(m["has_uv"] for m in meshes)
    uv_oob = sum(m["uv_oob_loops"] for m in meshes)
    tex_count = sum(len(m["textures"]) for m in meshes)
    blend_covs = [m["bake_alpha_coverage"] for m in blend_meshes if m.get("bake_alpha_coverage") is not None]
    glb_covs = [m["bake_alpha_coverage"] for m in meshes if m.get("bake_alpha_coverage") is not None]
    cov_list = blend_covs if blend_covs else glb_covs
    min_cov = min(cov_list) if cov_list else None

    unnorm_w = sum(m["unnormalized_weight_verts"] for m in meshes)
    zero_w = sum(m["zero_weight_verts"] for m in meshes)
    junk_count = len(glb_report.get("junk_objects", []))

    blocking = []
    warnings = []

    # 1. Geometric fidelity
    p1_ok = bool(fidelity.get("valid"))
    if p1_ok:
        if fidelity["sym_p95_dev_pct"] > criteria["max_p95_dev_pct"]:
            p1_ok = False
            blocking.append(
                f"Pillar1 Geometric Fidelity: sym_p95_dev={fidelity['sym_p95_dev_pct']}% > {criteria['max_p95_dev_pct']}%"
            )
        if fidelity["sym_mean_dev_pct"] > criteria["max_mean_dev_pct"]:
            p1_ok = False
            blocking.append(
                f"Pillar1 Geometric Fidelity: sym_mean_dev={fidelity['sym_mean_dev_pct']}% > {criteria['max_mean_dev_pct']}%"
            )
        if fidelity["bbox_dim_drift_pct"] > criteria["max_bbox_drift_pct"]:
            p1_ok = False
            blocking.append(
                f"Pillar1 Bounding Drift: {fidelity['bbox_dim_drift_pct']}% > {criteria['max_bbox_drift_pct']}%"
            )
        max_dev_lim = criteria.get("max_sym_max_dev_pct")
        if max_dev_lim is not None and fidelity.get("sym_max_dev_pct", 0.0) > float(max_dev_lim):
            p1_ok = False
            blocking.append(
                f"Pillar1 Thin-Detail Drop: sym_max_dev={fidelity['sym_max_dev_pct']}% > {max_dev_lim}%"
            )
    else:
        blocking.append("Pillar1 Geometric Fidelity: failed to evaluate BVH surface distance.")

    # 2. Topology quality
    p2_ok = True
    if eff_quad_ratio < criteria["min_quad_ratio"]:
        p2_ok = False
        blocking.append(f"Pillar2 Topology: quad_ratio={eff_quad_ratio} < {criteria['min_quad_ratio']}")
    if val4_ratio < criteria["min_valence4_ratio"]:
        p2_ok = False
        blocking.append(f"Pillar2 Topology: valence4_ratio={val4_ratio} < {criteria['min_valence4_ratio']}")
    if ngons > criteria["max_ngons"]:
        p2_ok = False
        blocking.append(f"Pillar2 Topology: ngons={ngons} > {criteria['max_ngons']}")
    if bad_aspect_pct > criteria["max_bad_aspect_pct"]:
        warnings.append(f"Pillar2 Aspect Ratio: {bad_aspect_pct}% stretched faces")

    # 3. Manifold & degenerate cleanliness
    p3_ok = True
    if welded_non_man > criteria["max_welded_non_manifold"]:
        p3_ok = False
        blocking.append(f"Pillar3 Manifold: welded_non_manifold_edges={welded_non_man}")
    if degen > 0 or zero_e > 0 or loose_v > 0:
        p3_ok = False
        blocking.append(f"Pillar3 Cleanliness: degen={degen}, zero_edges={zero_e}, loose_verts={loose_v}")
    if debris > criteria["max_debris_shells"]:
        p3_ok = False
        blocking.append(f"Pillar3 Debris: debris_shells={debris} > {criteria['max_debris_shells']}")
    if flipped_pct > criteria["max_flipped_normal_pct"]:
        p3_ok = False
        blocking.append(f"Pillar3 Normals: flipped_estimate={flipped_pct}%")

    # 4. Polycount budget
    eval_faces = total_faces_raw_blend if blend_meshes else faces_rec
    p4_ok = criteria["min_faces"] <= eval_faces <= criteria["max_faces"]
    if not p4_ok:
        blocking.append(
            f"Pillar4 Polycount Budget: faces={eval_faces} outside [{criteria['min_faces']}, {criteria['max_faces']}]"
        )

    # 5. UV & materials
    p5_ok = True
    if not has_uv or uv_oob > 0:
        p5_ok = False
        blocking.append(f"Pillar5 UV: has_uv={has_uv}, uv_oob_loops={uv_oob}")
    if tex_count < 1:
        p5_ok = False
        blocking.append("Pillar5 Materials: zero textures attached to candidate materials.")
    if min_cov is not None and min_cov < criteria["min_bake_coverage"]:
        p5_ok = False
        blocking.append(f"Pillar5 Bake Coverage: {min_cov} < {criteria['min_bake_coverage']}")

    # 6. Rigging / engine readiness
    p6_ok = True
    if junk_count > 0:
        p6_ok = False
        blocking.append(f"Pillar6 Engine Hygiene: found junk objects {glb_report.get('junk_objects')}")
    if unnorm_w > 0 or zero_w > 0:
        p6_ok = False
        blocking.append(f"Pillar6 Rig Weights: unnormalized={unnorm_w}, zero_weight={zero_w}")
    if not p2_ok or not p3_ok:
        p6_ok = False

    pillar_flags = [p1_ok, p2_ok, p3_ok, p4_ok, p5_ok, p6_ok]
    score = round(100.0 * sum(1 for x in pillar_flags if x) / 6.0, 2)
    passed = all(pillar_flags)

    return {
        "passed": passed,
        "score": score,
        "blocking_failures": blocking,
        "warnings": warnings,
        "pillars": {
            "1_geometric_fidelity": {"passed": p1_ok, **fidelity},
            "2_topology_quality": {
                "passed": p2_ok,
                "blend_quad_ratio": blend_quad_ratio,
                "glb_recovered_quad_ratio": glb_rec_quad_ratio,
                "effective_quad_ratio": eff_quad_ratio,
                "valence4_ratio": val4_ratio,
                "blend_valence4_ratio": blend_val4_ratio,
                "quad_orthogonality_mean": quad_ortho,
                "pole_cluster_pct": pole_cluster_pct,
                "ngons": ngons,
                "bad_aspect_pct": bad_aspect_pct,
            },
            "3_manifold_cleanliness": {
                "passed": p3_ok,
                "welded_non_manifold_edges": welded_non_man,
                "degenerate_faces": degen,
                "zero_length_edges": zero_e,
                "loose_verts": loose_v,
                "debris_shells": debris,
                "flipped_normals_pct": flipped_pct,
            },
            "4_polycount_budget": {
                "passed": p4_ok,
                "eval_faces": eval_faces,
                "glb_tris_raw": sum(m["tris_raw"] for m in meshes),
                "target_min": criteria["min_faces"],
                "target_max": criteria["max_faces"],
            },
            "5_uv_and_materials": {
                "passed": p5_ok,
                "has_uv": has_uv,
                "uv_oob_loops": uv_oob,
                "texture_count": tex_count,
                "min_bake_coverage": min_cov,
            },
            "6_rig_engine_ready": {
                "passed": p6_ok,
                "junk_objects": glb_report.get("junk_objects", []),
                "unnormalized_weight_verts": unnorm_w,
                "zero_weight_verts": zero_w,
                "glb_roundtrip_ok": True,
            },
        },
    }


def main():
    args = _parse_args()
    criteria = json.loads(Path(args.criteria_json).read_text(encoding="utf-8-sig"))

    blend_report = None
    renders = {}
    prefix = Path(args.candidate_glb).stem
    if args.candidate_blend and Path(args.candidate_blend).is_file():
        load_scene_file(args.candidate_blend)
        blend_report = inspect_current_scene(reference_path="")
        # Render true pure-quad wireframe directly from .blend before GLB triangulation
        renders.update(_render_qc_views(args.renders_dir, prefix=prefix, mode="wire"))

    # Always validate the round-tripped GLB independently as the true engine artifact
    load_scene_file(args.candidate_glb)
    glb_report = inspect_current_scene(reference_path=args.reference_glb)
    renders.update(_render_qc_views(args.renders_dir, prefix=prefix, mode="shaded" if renders else "both"))

    verdict = evaluate_six_pillars(glb_report, blend_report, criteria)
    verdict["renders"] = renders
    verdict["candidate_glb"] = str(args.candidate_glb)
    verdict["reference_glb"] = str(args.reference_glb)

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation finished: passed={verdict['passed']} score={verdict['score']}")


if __name__ == "__main__":
    main()
