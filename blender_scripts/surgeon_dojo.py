"""Agent Surgeon Dojo — Procedural Practice Asset & Automated Grader (Blender 5.1+).

Gives any AI Agent (or developer) a self-contained, zero-copyright practice arena to train and
verify its mastery of `run_pipeline.py surgeon`:
  - `--action setup`: Procedurally generates a realistic stylized dress + leg-pair scene containing:
      1. Clean outer layered skirt & back panel (MUST BE PRESERVED >= 95%)
      2. Fused thigh/knee inner leg pouches + a sneaky 2mm knee ring (MUST BE REMOVED >= 98%)
      3. Ground-truth bounding zones saved to `dojo_truth.json` for objective grading.
  - `--action grade`: Evaluates the Agent's exported `.blend`/`.glb` against the ground-truth
      scoring rubric (100 points total, PASS >= 90) and writes `dojo_scorecard.json`.
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Procedural 3D Agent practice dojo & automated grader.")
    p.add_argument("--dojo-dir", required=True, help="Workspace folder for dojo asset, truth, and scorecard.")
    p.add_argument(
        "--exam",
        choices=("surgery", "retopo_500k"),
        default="surgery",
        help="Practice exam type: 'surgery' (leg pouch/ring removal) or 'retopo_500k' (~500k high-poly quad retopo).",
    )
    p.add_argument("--action", choices=("setup", "grade"), required=True)
    p.add_argument("--candidate", default="", help="Agent-exported .blend or .glb to grade (required for --action grade).")
    p.add_argument("--session", default="", help="Optional surgeon session directory to verify triptych audit artifacts.")
    p.add_argument(
        "--target-tris",
        type=int,
        default=500000,
        help="Target triangle count for --exam retopo_500k setup (default 500,000; lower for fast CI tests).",
    )
    return p.parse_args(argv)


def _make_mat(name: str, rgba: tuple[float, float, float, float], rough: float = 0.65):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def _build_dojo_scene(dojo_dir: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 1. Build clean basemesh legs (occluder body)
    bm_body = bmesh.new()
    for sx in (-0.085, 0.085):
        bmesh.ops.create_cone(
            bm_body,
            cap_ends=True,
            cap_tris=False,
            segments=24,
            radius1=0.042,
            radius2=0.062,
            depth=0.78,
            matrix=Matrix.Translation((sx, 0.0, 0.42)),
        )
    # Torso & pelvis
    bmesh.ops.create_cone(
        bm_body,
        cap_ends=True,
        cap_tris=False,
        segments=24,
        radius1=0.145,
        radius2=0.115,
        depth=0.45,
        matrix=Matrix.Translation((0.0, 0.0, 1.00)),
    )
    body_me = bpy.data.meshes.new("Dojo_Body")
    bm_body.to_mesh(body_me)
    bm_body.free()
    body_obj = bpy.data.objects.new("Dojo_Body", body_me)
    scene.collection.objects.link(body_obj)
    body_obj.data.materials.append(_make_mat("Dojo_SkinStocking", (0.94, 0.92, 0.95, 1.0), rough=0.85))
    for p in body_me.polygons:
        p.use_smooth = True

    # 2. Build stylized layered skirt + intentional AI-artifact defects inside ONE garment mesh
    bm_g = bmesh.new()

    # 2a. Outer ruffled skirt shell (open front archway above knees, long flowing back & sides — MUST KEEP!)
    rings_z = [0.22, 0.34, 0.46, 0.58, 0.70, 0.82, 0.94, 1.08]
    segs = 48
    ring_verts = []
    for rz_i, z in enumerate(rings_z):
        t = rz_i / (len(rings_z) - 1)  # 0 = hem, 1 = waist
        base_r = 0.34 * (1.0 - t) + 0.14 * t
        row = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            # Flowing ruffle waves around outer skirt
            wave = 0.025 * (1.0 - t) * math.sin(8.0 * ang)
            rx = (base_r + wave) * math.cos(ang)
            ry = (base_r * 0.92 + wave) * math.sin(ang)
            # Raise front center hem cleanly above thighs so legs show in front archway (y < 0 is front)
            z_cur = z
            if math.sin(ang) < -0.35 and z < 0.72:
                front_lift = (-math.sin(ang) - 0.35) / 0.65
                z_cur = z + 0.44 * front_lift * (1.0 - t)
            row.append(bm_g.verts.new((rx, ry, z_cur)))
        ring_verts.append(row)

    for rz_i in range(len(ring_verts) - 1):
        r0, r1 = ring_verts[rz_i], ring_verts[rz_i + 1]
        for s in range(segs):
            sn = (s + 1) % segs
            bm_g.faces.new((r0[s], r0[sn], r1[sn], r1[s]))

    outer_keep_faces = len(bm_g.faces)

    # 2b. DEFECT 1: Two fused thigh/knee inner pouches wrapping tightly around both legs (MUST REMOVE!)
    for sx in (-0.085, 0.085):
        bmesh.ops.create_cone(
            bm_g,
            cap_ends=False,
            cap_tris=False,
            segments=24,
            radius1=0.068,
            radius2=0.078,
            depth=0.22,
            matrix=Matrix.Translation((sx, 0.0, 0.50)),
        )
        # DEFECT 2: Sneaky 3mm thin ring right at the knee that catches careless single-ray cullers!
        bmesh.ops.create_cone(
            bm_g,
            cap_ends=False,
            cap_tris=False,
            segments=24,
            radius1=0.064,
            radius2=0.064,
            depth=0.008,
            matrix=Matrix.Translation((sx, 0.0, 0.365)),
        )

    # 2c. DEFECT 3: Inner webbing bridge between the two thighs (MUST REMOVE!)
    bmesh.ops.create_grid(
        bm_g,
        x_segments=8,
        y_segments=8,
        size=0.08,
        matrix=Matrix.Translation((0.0, -0.02, 0.50)) @ Matrix.Rotation(math.radians(90.0), 4, "X"),
    )

    bmesh.ops.recalc_face_normals(bm_g, faces=list(bm_g.faces))
    g_me = bpy.data.meshes.new("Dojo_Garment")
    bm_g.to_mesh(g_me)
    bm_g.free()
    g_obj = bpy.data.objects.new("Dojo_Garment", g_me)
    scene.collection.objects.link(g_obj)
    g_obj.data.materials.append(_make_mat("Dojo_LavenderDress", (0.72, 0.62, 0.96, 1.0), rough=0.72))
    for p in g_me.polygons:
        p.use_smooth = True

    # Compute ground-truth defect & keep counts using spatial zones
    def _is_defect_center(c: Vector) -> bool:
        return abs(c.x) <= 0.165 and -0.10 <= c.y <= 0.10 and 0.34 <= c.z <= 0.625

    def _is_back_or_outer_keep(c: Vector) -> bool:
        return (c.y >= 0.14) or (abs(c.x) >= 0.20) or (c.z >= 0.72)

    total_faces = len(g_me.polygons)
    defect_faces = sum(1 for p in g_me.polygons if _is_defect_center(p.center))
    keep_faces = sum(1 for p in g_me.polygons if _is_back_or_outer_keep(p.center))

    dojo_dir.mkdir(parents=True, exist_ok=True)
    blend_p = dojo_dir / "dojo_practice_dress.blend"
    glb_p = dojo_dir / "dojo_practice_dress.glb"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_p))
    bpy.ops.export_scene.gltf(filepath=str(glb_p), export_format="GLB")

    truth = {
        "dojo_blend": str(blend_p),
        "dojo_glb": str(glb_p),
        "target_mesh": "Dojo_Garment",
        "initial_total_faces": total_faces,
        "initial_outer_shell_faces": outer_keep_faces,
        "initial_defect_faces": defect_faces,
        "initial_keep_zone_faces": keep_faces,
        "defect_zone": {"abs_x_max": 0.165, "y_min": -0.10, "y_max": 0.10, "z_min": 0.34, "z_max": 0.625},
        "keep_zone": {"back_y_min": 0.14, "outer_abs_x_min": 0.20, "upper_z_min": 0.72},
        "mission": (
            "Use `run_pipeline.py surgeon` to inspect the triptych, remove ALL fused leg pouches, "
            "the inter-thigh webbing, and the sneaky knee rings inside the leg channel while preserving "
            "the outer ruffled skirt, upper front bodice, and full back skirt, then export and run `--action grade`."
        ),
    }
    (dojo_dir / "dojo_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def _grade_candidate(dojo_dir: Path, candidate_path: str, session_dir: str = "") -> dict:
    truth_p = dojo_dir / "dojo_truth.json"
    if not truth_p.is_file():
        raise FileNotFoundError(f"Missing dojo_truth.json in {dojo_dir}. Run --action setup first.")
    truth = json.loads(truth_p.read_text(encoding="utf-8"))

    cand = Path(candidate_path)
    if not cand.is_file():
        raise FileNotFoundError(f"Candidate mesh file not found: {cand}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if str(cand).lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=str(cand))
    else:
        bpy.ops.import_scene.gltf(filepath=str(cand))

    g_obj = None
    for o in bpy.context.scene.objects:
        if o.type == "MESH" and ("garment" in o.name.lower() or o.name == truth["target_mesh"]):
            g_obj = o
            break
    if g_obj is None:
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons) > 0]
        g_obj = max(meshes, key=lambda o: len(o.data.polygons))

    dz = truth["defect_zone"]
    kz = truth["keep_zone"]
    mw = g_obj.matrix_world

    remaining_defects = 0
    remaining_keep = 0
    degenerate_faces = 0
    for p in g_obj.data.polygons:
        wc = mw @ p.center
        if p.area < 1e-8:
            degenerate_faces += 1
        if abs(wc.x) <= dz["abs_x_max"] and dz["y_min"] <= wc.y <= dz["y_max"] and dz["z_min"] <= wc.z <= dz["z_max"]:
            remaining_defects += 1
        if (wc.y >= kz["back_y_min"]) or (abs(wc.x) >= kz["outer_abs_x_min"]) or (wc.z >= kz["upper_z_min"]):
            remaining_keep += 1

    # Check 3D-welded connected components (should be 1 clean garment shell, zero floating rings!)
    bm = bmesh.new()
    bm.from_mesh(g_obj.data)
    bm.faces.ensure_lookup_table()
    pk = lambda v: (round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4))
    pf: dict[tuple[float, float, float], list[int]] = {}
    for f in bm.faces:
        for v in f.verts:
            pf.setdefault(pk(v), []).append(f.index)
    vis = set()
    comp_sizes = []
    for fi in range(len(bm.faces)):
        if fi in vis:
            continue
        c = 0
        q = deque([fi])
        vis.add(fi)
        while q:
            u = q.popleft()
            c += 1
            for v in bm.faces[u].verts:
                for nf in pf.get(pk(v), ()):
                    if nf not in vis:
                        vis.add(nf)
                        q.append(nf)
        comp_sizes.append(c)
    bm.free()
    comp_sizes.sort(reverse=True)
    floating_islands = max(0, len(comp_sizes) - 1)

    init_def = max(1, int(truth["initial_defect_faces"]))
    init_keep = max(1, int(truth["initial_keep_zone_faces"]))
    defect_cleared_pct = round(100.0 * max(0, init_def - remaining_defects) / init_def, 2)
    keep_preserved_pct = round(100.0 * min(init_keep, remaining_keep) / init_keep, 2)

    # Scoring rubric (100 points total)
    score_defect = 40.0 if defect_cleared_pct >= 98.0 else round(40.0 * (defect_cleared_pct / 98.0), 2)
    score_keep = 25.0 if keep_preserved_pct >= 95.0 else round(25.0 * max(0.0, keep_preserved_pct / 95.0), 2)
    score_islands = 15.0 if floating_islands == 0 else max(0.0, 15.0 - 5.0 * floating_islands)
    score_clean = 10.0 if degenerate_faces == 0 else max(0.0, 10.0 - 2.0 * degenerate_faces)

    triptych_verified = False
    if session_dir:
        triptych_verified = (Path(session_dir) / "renders" / "inspect_triptych.png").is_file()
    else:
        triptych_verified = True
    score_workflow = 10.0 if triptych_verified else 0.0

    total_score = round(score_defect + score_keep + score_islands + score_clean + score_workflow, 2)
    passed = (
        total_score >= 90.0
        and defect_cleared_pct >= 98.0
        and keep_preserved_pct >= 95.0
        and floating_islands == 0
    )

    scorecard = {
        "status": "PASS" if passed else "FAIL",
        "total_score": total_score,
        "candidate": str(cand),
        "metrics": {
            "post_faces": len(g_obj.data.polygons),
            "remaining_defect_faces": remaining_defects,
            "defect_cleared_pct": defect_cleared_pct,
            "keep_preserved_pct": keep_preserved_pct,
            "floating_orphan_islands": floating_islands,
            "degenerate_faces": degenerate_faces,
            "triptych_verified": triptych_verified,
        },
        "breakdown": {
            "defect_removal_40": score_defect,
            "outer_preservation_25": score_keep,
            "zero_floating_islands_15": score_islands,
            "mesh_cleanliness_10": score_clean,
            "workflow_audit_10": score_workflow,
        },
    }
    (dojo_dir / "dojo_scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")
    return scorecard


def _build_retopo_500k_exam(dojo_dir: Path, target_tris: int = 500000) -> dict:
    """Procedurally generate a ~500k-tri raw AI-style sculpted corset + asymmetric slender ribbon high-poly exam."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bm = bmesh.new()
    # 1. Sculpted ruffled corset & peplum shell (closed thick shell so QuadriFlow has clean volume)
    rings_z = [0.68, 0.75, 0.83, 0.91, 0.99, 1.06]
    segs = 32
    outer_rings = []
    inner_rings = []
    for rz_i, z in enumerate(rings_z):
        t = rz_i / (len(rings_z) - 1)
        # Hourglass waist cinch at t=0.45, flared ruffled peplum at t=0.0
        r_base = 0.24 * (1.0 - t) ** 1.6 + 0.125 + 0.045 * t * t
        o_row, i_row = [], []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            pleat = 0.018 * (1.0 - t) * math.sin(10.0 * ang) + 0.006 * math.cos(4.0 * ang)
            rx = (r_base + pleat) * math.cos(ang)
            ry = (r_base * 0.84 + pleat) * math.sin(ang)
            o_row.append(bm.verts.new((rx, ry, z)))
            i_row.append(bm.verts.new((rx * 0.90, ry * 0.90, z)))
        outer_rings.append(o_row)
        inner_rings.append(i_row)

    for rz_i in range(len(rings_z) - 1):
        for s in range(segs):
            sn = (s + 1) % segs
            bm.faces.new((outer_rings[rz_i][s], outer_rings[rz_i][sn], outer_rings[rz_i + 1][sn], outer_rings[rz_i + 1][s]))
            bm.faces.new((inner_rings[rz_i][sn], inner_rings[rz_i][s], inner_rings[rz_i + 1][s], inner_rings[rz_i + 1][sn]))
    # Cap top & bottom rims to keep shell watertight
    for s in range(segs):
        sn = (s + 1) % segs
        bm.faces.new((inner_rings[0][s], inner_rings[0][sn], outer_rings[0][sn], outer_rings[0][s]))
        bm.faces.new((outer_rings[-1][s], outer_rings[-1][sn], inner_rings[-1][sn], inner_rings[-1][s]))

    # 2. Asymmetric slender hanging side ribbon (physically welded onto the corset rim via Boolean Union, like Rodin raw meshes)
    bm_rib = bmesh.new()
    bmesh.ops.create_cone(
        bm_rib,
        cap_ends=True,
        cap_tris=False,
        segments=16,
        radius1=0.028,
        radius2=0.022,
        depth=0.38,
        matrix=Matrix.Translation((0.20, -0.02, 0.58)) @ Matrix.Rotation(math.radians(6.0), 4, "Y"),
    )
    bmesh.ops.recalc_face_normals(bm_rib, faces=list(bm_rib.faces))
    rib_me = bpy.data.meshes.new("Dojo_Ribbon_Tmp")
    bm_rib.to_mesh(rib_me)
    bm_rib.free()
    rib_obj = bpy.data.objects.new("Dojo_Ribbon_Tmp", rib_me)
    scene.collection.objects.link(rib_obj)

    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    me = bpy.data.meshes.new("Dojo_HighPoly_Corset")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("Dojo_HighPoly_Corset", me)
    scene.collection.objects.link(obj)
    obj.data.materials.append(_make_mat("Dojo_CorsetGold", (0.58, 0.50, 0.90, 1.0), rough=0.68))

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bool_m = obj.modifiers.new("UnionRibbon", type="BOOLEAN")
    bool_m.operation = "UNION"
    bool_m.object = rib_obj
    bpy.ops.object.modifier_apply(modifier=bool_m.name)
    bpy.data.objects.remove(rib_obj, do_unlink=True)

    # Subdivide smoothly until triangle count approaches target_tris (fast C++ modifier!)
    base_tris = sum(max(1, len(p.vertices) - 2) for p in obj.data.polygons)
    levels = 0
    est = base_tris
    while est * 4 <= max(base_tris * 4, int(target_tris * 1.35)) and levels < 6:
        est *= 4
        levels += 1
    if levels > 0:
        sub = obj.modifiers.new("Subsurf", type="SUBSURF")
        sub.levels = levels
        sub.render_levels = levels
    tri = obj.modifiers.new("Triangulate", type="TRIANGULATE")
    tri.quad_method = "BEAUTY"

    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)

    for p in obj.data.polygons:
        p.use_smooth = True

    # Inject a typical AI-export junk camera/empty so cleanup is also tested
    junk = bpy.data.objects.new("AI_Export_Junk_Empty", None)
    scene.collection.objects.link(junk)

    tri_count = len(obj.data.polygons)
    dojo_dir.mkdir(parents=True, exist_ok=True)
    blend_p = dojo_dir / "dojo_500k_highpoly.blend"
    glb_p = dojo_dir / "dojo_500k_highpoly.glb"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_p))
    bpy.ops.export_scene.gltf(filepath=str(glb_p), export_format="GLB")

    truth = {
        "exam": "retopo_500k",
        "dojo_blend": str(blend_p),
        "dojo_glb": str(glb_p),
        "target_mesh": obj.name,
        "initial_tris": tri_count,
        "target_face_range": [2500, 22000],
        "mission": (
            "Run `python run_pipeline.py retopo --input <dojo_500k_highpoly.glb> --out <dir>` "
            "to reduce this ~500k-tri messy high-poly garment into a clean 2.5k-22k quad mesh "
            "while preserving the asymmetric slender side ribbon, then grade with `dojo --exam retopo_500k --action grade`."
        ),
    }
    (dojo_dir / "dojo_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return truth


def _grade_retopo_500k_candidate(dojo_dir: Path, candidate_path: str) -> dict:
    truth_p = dojo_dir / "dojo_truth.json"
    if not truth_p.is_file():
        raise FileNotFoundError(f"Missing dojo_truth.json in {dojo_dir}. Run --exam retopo_500k --action setup first.")
    truth = json.loads(truth_p.read_text(encoding="utf-8"))
    ref_path = Path(truth["dojo_blend"])
    cand_path = Path(candidate_path)
    if not cand_path.is_file():
        raise FileNotFoundError(f"Candidate mesh not found: {cand_path}")

    # Load reference high-poly into BVH
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=str(ref_path))
    ref_obj = max([o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons) > 0], key=lambda o: len(o.data.polygons))
    mw_r = ref_obj.matrix_world
    ref_verts = [mw_r @ v.co for v in ref_obj.data.vertices]
    ref_polys = [tuple(p.vertices) for p in ref_obj.data.polygons]
    ref_bvh = BVHTree.FromPolygons(ref_verts, ref_polys)
    r_min = Vector((min(v.x for v in ref_verts), min(v.y for v in ref_verts), min(v.z for v in ref_verts)))
    r_max = Vector((max(v.x for v in ref_verts), max(v.y for v in ref_verts), max(v.z for v in ref_verts)))
    ref_diag = max((r_max - r_min).length, 1e-4)
    # Sample reference points evenly (up to 4000) for fast reverse-direction anti-truncation check
    step_r = max(1, len(ref_verts) // 4000)
    ref_samples = ref_verts[::step_r]

    # Load candidate retopo mesh
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if str(cand_path).lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=str(cand_path))
    else:
        bpy.ops.import_scene.gltf(filepath=str(cand_path))
    c_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons) > 0]
    if not c_meshes:
        raise RuntimeError("No non-empty mesh found in candidate file.")
    c_obj = max(c_meshes, key=lambda o: len(o.data.polygons))
    mw_c = c_obj.matrix_world
    c_verts = [mw_c @ v.co for v in c_obj.data.vertices]
    c_polys = [tuple(p.vertices) for p in c_obj.data.polygons]
    c_bvh = BVHTree.FromPolygons(c_verts, c_polys)

    # Topology metrics via bmesh
    bm = bmesh.new()
    bm.from_mesh(c_obj.data)
    post_faces = len(bm.faces)
    quad_faces = sum(1 for f in bm.faces if len(f.verts) == 4)
    quad_ratio = quad_faces / max(1, post_faces)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-8)

    interior_verts = [v for v in bm.verts if not v.is_boundary and len(v.link_edges) > 0]
    val4 = sum(1 for v in interior_verts if len(v.link_edges) == 4)
    valence4_ratio = val4 / max(1, len(interior_verts))
    poles = {v.index for v in interior_verts if len(v.link_edges) != 4}
    pole_clusters = sum(
        1 for v in interior_verts
        if v.index in poles and any(e.other_vert(v).index in poles for e in v.link_edges)
    )
    pole_cluster_pct = 100.0 * pole_clusters / max(1, len(interior_verts))

    # Mean quad orthogonality (1.0 = perfect 90-degree quads)
    ortho_vals = []
    for f in bm.faces:
        if len(f.verts) != 4:
            continue
        cs = [v.co for v in f.verts]
        ang_err = 0.0
        for i in range(4):
            e1 = (cs[(i - 1) % 4] - cs[i]).normalized()
            e2 = (cs[(i + 1) % 4] - cs[i]).normalized()
            ang_err += abs(e1.dot(e2))
        ortho_vals.append(max(0.0, 1.0 - ang_err / 4.0))
    quad_ortho = sum(ortho_vals) / max(1, len(ortho_vals))
    bm.free()

    # Bidirectional silhouette deviation (catches ribbon truncation!)
    fwd_dists = []
    step_c = max(1, len(c_verts) // 4000)
    for pt in c_verts[::step_c]:
        _, _, _, d = ref_bvh.find_nearest(pt)
        if d is not None:
            fwd_dists.append(d)
    rev_dists = []
    for pt in ref_samples:
        _, _, _, d = c_bvh.find_nearest(pt)
        if d is not None:
            rev_dists.append(d)
    max_dev_pct = 100.0 * max(max(fwd_dists or [0.0]), max(rev_dists or [0.0])) / ref_diag
    mean_dev_pct = 100.0 * (sum(fwd_dists) / max(1, len(fwd_dists)) + sum(rev_dists) / max(1, len(rev_dists))) / (2.0 * ref_diag)

    # 100-point 6-pillar scoring rubric
    fmin, fmax = truth["target_face_range"]
    score_budget = 20.0 if (fmin <= post_faces <= fmax) else max(0.0, 20.0 * min(post_faces / fmin, fmax / max(1, post_faces)))
    score_quad = 20.0 if quad_ratio >= 0.90 else round(20.0 * (quad_ratio / 0.90), 2)
    score_val4 = 20.0 if (valence4_ratio >= 0.72 and pole_cluster_pct <= 15.0) else round(20.0 * min(1.0, valence4_ratio / 0.72), 2)
    score_sil = 25.0 if max_dev_pct <= 5.0 else round(max(0.0, 25.0 * (1.0 - (max_dev_pct - 5.0) / 15.0)), 2)
    score_clean = 15.0 if (non_manifold == 0 and degenerate == 0 and quad_ortho >= 0.75) else round(
        max(0.0, 15.0 - (5.0 if non_manifold > 0 else 0.0) - (5.0 if degenerate > 0 else 0.0) - (5.0 if quad_ortho < 0.75 else 0.0)),
        2,
    )
    total_score = round(score_budget + score_quad + score_val4 + score_sil + score_clean, 2)
    passed = total_score >= 90.0 and (fmin <= post_faces <= fmax) and quad_ratio >= 0.85 and max_dev_pct <= 5.0 and non_manifold == 0

    scorecard = {
        "exam": "retopo_500k",
        "status": "PASS" if passed else "FAIL",
        "total_score": total_score,
        "candidate": str(cand_path),
        "metrics": {
            "initial_tris": truth["initial_tris"],
            "post_faces": post_faces,
            "compression_ratio": round(truth["initial_tris"] / max(1, post_faces), 1),
            "quad_ratio": round(quad_ratio, 4),
            "valence4_ratio": round(valence4_ratio, 4),
            "pole_cluster_pct": round(pole_cluster_pct, 2),
            "quad_orthogonality_mean": round(quad_ortho, 4),
            "max_silhouette_dev_pct": round(max_dev_pct, 3),
            "mean_silhouette_dev_pct": round(mean_dev_pct, 3),
            "non_manifold_edges": non_manifold,
            "degenerate_faces": degenerate,
        },
        "breakdown": {
            "poly_budget_20": round(score_budget, 2),
            "quad_purity_20": round(score_quad, 2),
            "valence4_flow_20": round(score_val4, 2),
            "silhouette_and_ribbon_25": round(score_sil, 2),
            "manifold_and_ortho_15": round(score_clean, 2),
        },
    }
    (dojo_dir / "dojo_scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")
    return scorecard


def main():
    args = _parse_args()
    dojo_dir = Path(args.dojo_dir)
    if args.exam == "retopo_500k":
        if args.action == "setup":
            res = _build_retopo_500k_exam(dojo_dir, target_tris=args.target_tris)
        else:
            if not args.candidate:
                raise ValueError("--action grade requires --candidate <retopo.blend|retopo.glb>.")
            res = _grade_retopo_500k_candidate(dojo_dir, args.candidate)
    else:
        if args.action == "setup":
            res = _build_dojo_scene(dojo_dir)
        else:
            if not args.candidate:
                raise ValueError("--action grade requires --candidate <exported.blend|exported.glb>.")
            res = _grade_candidate(dojo_dir, args.candidate, session_dir=args.session)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
