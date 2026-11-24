"""Independent Validator for Autonomous UV Unwrapping, Overlap/Stretch Audit, and PBR+AO Baking."""

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from topology_inspector import inspect_current_scene, load_scene_file  # noqa: E402
from topology_validator import _render_qc_views  # noqa: E402


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-glb", required=True)
    p.add_argument("--candidate-blend", default="")
    p.add_argument("--high-reference", required=True)
    p.add_argument("--textures-dir", default="")
    p.add_argument("--criteria-json", required=True)
    p.add_argument("--renders-dir", required=True)
    p.add_argument("--uv-report-json", required=True)
    p.add_argument("--bake-report-json", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args(argv)


def _poly_signed_area(pts):
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _clip_tris_overlap_area(tri_a, tri_b):
    """Exact 2D Sutherland-Hodgman polygon intersection area between two triangles."""
    if _poly_signed_area(tri_a) < 0:
        tri_a = [tri_a[0], tri_a[2], tri_a[1]]
    if _poly_signed_area(tri_b) < 0:
        tri_b = [tri_b[0], tri_b[2], tri_b[1]]

    subject = list(tri_a)
    for i in range(3):
        cx1, cy1 = tri_b[i]
        cx2, cy2 = tri_b[(i + 1) % 3]
        if not subject:
            return 0.0
        inp = subject
        subject = []
        s_pt = inp[-1]
        for e_pt in inp:
            e_in = (cx2 - cx1) * (e_pt[1] - cy1) - (cy2 - cy1) * (e_pt[0] - cx1) >= -1e-11
            s_in = (cx2 - cx1) * (s_pt[1] - cy1) - (cy2 - cy1) * (s_pt[0] - cx1) >= -1e-11
            if e_in:
                if not s_in:
                    dcx, dcy = cx1 - cx2, cy1 - cy2
                    dpx, dpy = s_pt[0] - e_pt[0], s_pt[1] - e_pt[1]
                    n1 = cx1 * cy2 - cy1 * cx2
                    n2 = s_pt[0] * e_pt[1] - s_pt[1] * e_pt[0]
                    denom = dcx * dpy - dcy * dpx
                    if abs(denom) > 1e-15:
                        subject.append(((n1 * dpx - n2 * dcx) / denom, (n1 * dpy - n2 * dcy) / denom))
                subject.append(e_pt)
            elif s_in:
                dcx, dcy = cx1 - cx2, cy1 - cy2
                dpx, dpy = s_pt[0] - e_pt[0], s_pt[1] - e_pt[1]
                n1 = cx1 * cy2 - cy1 * cx2
                n2 = s_pt[0] * e_pt[1] - s_pt[1] * e_pt[0]
                denom = dcx * dpy - dcy * dpx
                if abs(denom) > 1e-15:
                    subject.append(((n1 * dpx - n2 * dcx) / denom, (n1 * dpy - n2 * dcy) / denom))
            s_pt = e_pt
    return abs(_poly_signed_area(subject))


def inspect_uv_geometry(mesh_obj, overlap_faces=None):
    """Audit UV bounds, island count, exact 2D triangle overlap, area stretch, and packing utilization."""
    me = mesh_obj.data
    if not me.uv_layers.active:
        return {
            "has_uv": False,
            "uv_oob_loops": 0,
            "uv_islands_count": 0,
            "uv_triangles": 0,
            "uv_zero_area_tris": 0,
            "uv_flipped_tris": 0,
            "uv_overlapping_tri_pairs": 0,
            "uv_overlap_area_pct": 100.0,
            "uv_packing_coverage_pct": 0.0,
            "uv_mean_stretch": 999.0,
            "uv_p95_stretch": 999.0,
        }

    uv_data = me.uv_layers.active.data
    oob_loops = 0
    for lp in uv_data:
        if not (-1e-4 <= lp.uv.x <= 1.0001 and -1e-4 <= lp.uv.y <= 1.0001):
            oob_loops += 1

    tris_uv = []
    face_ids = []
    tris_uv_keys = []
    tris_vids = []
    uv_areas = []
    area3d_list = []
    signed_areas = []
    mw = mesh_obj.matrix_world

    for poly in me.polygons:
        lidxs = list(poly.loop_indices)
        if len(lidxs) < 3:
            continue
        p_pts = [mw @ me.vertices[me.loops[li].vertex_index].co for li in lidxs]
        u_pts = [(float(uv_data[li].uv.x), float(uv_data[li].uv.y)) for li in lidxs]
        u_keys = [(round(u, 5), round(v, 5)) for u, v in u_pts]
        v_ids = [me.loops[li].vertex_index for li in lidxs]
        for i in range(1, len(lidxs) - 1):
            tu = [u_pts[0], u_pts[i], u_pts[i + 1]]
            tk = {u_keys[0], u_keys[i], u_keys[i + 1]}
            tv = {v_ids[0], v_ids[i], v_ids[i + 1]}
            sa = _poly_signed_area(tu)
            ua = abs(sa)
            a3 = 0.5 * (p_pts[i] - p_pts[0]).cross(p_pts[i + 1] - p_pts[0]).length
            tris_uv.append(tu)
            face_ids.append(poly.index)
            tris_uv_keys.append(tk)
            tris_vids.append(tv)
            signed_areas.append(sa)
            uv_areas.append(ua)
            area3d_list.append(a3)

    n_tris = len(tris_uv)
    total_uv_area = sum(uv_areas)
    total_3d_area = max(1e-12, sum(area3d_list))
    zero_area_tris = sum(1 for ua, a3 in zip(uv_areas, area3d_list) if ua <= 1e-10 and a3 > 1e-9)

    # Stretch metrics (area-normalized ratio between 3D triangle and UV triangle)
    stretches = []
    weights = []
    if total_uv_area > 1e-10:
        for ua, a3 in zip(uv_areas, area3d_list):
            if a3 <= 1e-10:
                continue
            a_norm = a3 / total_3d_area
            u_norm = ua / total_uv_area
            if u_norm <= 1e-12:
                st = 50.0
            else:
                st = max(u_norm / a_norm, a_norm / u_norm)
            stretches.append(min(50.0, st))
            weights.append(a3)
    if stretches:
        w_arr = np.array(weights, dtype=np.float64)
        s_arr = np.array(stretches, dtype=np.float64)
        mean_stretch = float(np.sum(s_arr * w_arr) / max(1e-12, np.sum(w_arr)))
        order = np.argsort(s_arr)
        cum_w = np.cumsum(w_arr[order]) / max(1e-12, np.sum(w_arr))
        p95_idx = int(np.searchsorted(cum_w, 0.95))
        p95_stretch = float(s_arr[order[min(len(s_arr) - 1, p95_idx)]])
    else:
        mean_stretch = 999.0
        p95_stretch = 999.0

    # UV Island count via shared (vertex_id + quantized UV corner) adjacency
    corner_to_tris = defaultdict(list)
    for ti, (tv, tk) in enumerate(zip(tris_vids, tris_uv_keys)):
        for k in tk:
            corner_to_tris[k].append(ti)
    vis = [False] * n_tris
    islands = 0
    for i in range(n_tris):
        if vis[i]:
            continue
        islands += 1
        q = deque([i])
        vis[i] = True
        while q:
            cur = q.popleft()
            for k in tris_uv_keys[cur]:
                for nb in corner_to_tris[k]:
                    if not vis[nb]:
                        vis[nb] = True
                        q.append(nb)

    # Exact 2D UV Triangle Overlap via 48x48 spatial hash grid + Sutherland-Hodgman clip
    grid_res = 48
    grid = defaultdict(list)
    bboxes = []
    for ti, tu in enumerate(tris_uv):
        xs = (tu[0][0], tu[1][0], tu[2][0])
        ys = (tu[0][1], tu[1][1], tu[2][1])
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        bboxes.append((x0, y0, x1, y1))
        if uv_areas[ti] <= 1e-10:
            continue
        gx0 = max(0, min(grid_res - 1, int(x0 * grid_res)))
        gx1 = max(0, min(grid_res - 1, int(x1 * grid_res)))
        gy0 = max(0, min(grid_res - 1, int(y0 * grid_res)))
        gy1 = max(0, min(grid_res - 1, int(y1 * grid_res)))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid[(gx, gy)].append(ti)

    checked_pairs = set()
    overlap_pairs = 0
    overlap_area = 0.0
    for cell_tris in grid.values():
        if len(cell_tris) < 2:
            continue
        for idx_a in range(len(cell_tris)):
            ta = cell_tris[idx_a]
            ax0, ay0, ax1, ay1 = bboxes[ta]
            for idx_b in range(idx_a + 1, len(cell_tris)):
                tb = cell_tris[idx_b]
                pair = (ta, tb) if ta < tb else (tb, ta)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                bx0, by0, bx1, by1 = bboxes[tb]
                if ax1 <= bx0 + 1e-7 or bx1 <= ax0 + 1e-7 or ay1 <= by0 + 1e-7 or by1 <= ay0 + 1e-7:
                    continue
                # Adjacent UV triangles along an unfolded edge share >= 2 UV corners; only check non-adjacent or single-corner
                if overlap_faces is None and len(tris_uv_keys[ta] & tris_uv_keys[tb]) >= 2:
                    continue
                inter = _clip_tris_overlap_area(tris_uv[ta], tris_uv[tb])
                if inter > 2e-8:
                    overlap_pairs += 1
                    overlap_area += inter
                    if overlap_faces is not None:
                        overlap_faces.add(tuple(sorted((face_ids[ta], face_ids[tb]))))

    overlap_pct = round(100.0 * overlap_area / max(1e-9, total_uv_area), 3) if total_uv_area > 1e-9 else 100.0
    eff_pack_cov = round(max(0.0, min(100.0, 100.0 * max(0.0, total_uv_area - overlap_area))), 2)

    return {
        "has_uv": True,
        "uv_oob_loops": oob_loops,
        "uv_islands_count": islands,
        "uv_triangles": n_tris,
        "uv_zero_area_tris": zero_area_tris,
        "uv_overlapping_tri_pairs": overlap_pairs,
        "uv_overlap_area_pct": overlap_pct,
        "uv_packing_coverage_pct": eff_pack_cov,
        "uv_mean_stretch": round(mean_stretch, 3),
        "uv_p95_stretch": round(p95_stretch, 3),
    }


def _sample_face_centroids(mesh_obj, max_samples=5000):
    me = mesh_obj.data
    uv_data = me.uv_layers.active.data
    polys = me.polygons
    step = max(1, len(polys) // max_samples)
    uvs = []
    for pi in range(0, len(polys), step):
        p = polys[pi]
        cu = sum(uv_data[li].uv.x for li in p.loop_indices) / len(p.loop_indices)
        cv = sum(uv_data[li].uv.y for li in p.loop_indices) / len(p.loop_indices)
        uvs.append((max(0.0, min(1.0, cu)), max(0.0, min(1.0, cv))))
    return uvs


def _load_image_pixels(path_or_img):
    if isinstance(path_or_img, (str, Path)):
        p = Path(path_or_img)
        if not p.is_file():
            return None
        img = bpy.data.images.load(str(p), check_existing=True)
    else:
        img = path_or_img
    if not img or img.size[0] <= 0 or img.size[1] <= 0:
        return None
    w, h = img.size
    return np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))


def inspect_baked_maps(mesh_obj, textures_dir):
    """Verify Diffuse/Normal/AO/ORM quality directly against candidate UV face centroids."""
    uvs = _sample_face_centroids(mesh_obj, max_samples=5000)
    if not uvs:
        return {"valid": False, "reason": "No UV sample points."}

    tex_dir = Path(textures_dir) if textures_dir else None
    found_imgs = {}

    # Check explicit disk PNGs first (highest fidelity including preserved Alpha channel)
    if tex_dir and tex_dir.is_dir():
        for role in ("diffuse", "normal", "ao", "orm"):
            cands = sorted(tex_dir.glob(f"*_baked_{role}.png"))
            if cands:
                found_imgs[role] = _load_image_pixels(cands[0])

    # Fallback to embedded GLB textures if no disk folder provided
    if not found_imgs:
        for mat in mesh_obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    nm = node.image.name.lower()
                    for role in ("diffuse", "normal", "ao", "orm"):
                        if role in nm and role not in found_imgs:
                            found_imgs[role] = _load_image_pixels(node.image)

    def _sample_px(arr):
        if arr is None:
            return None
        h, w, _ = arr.shape
        xs = np.array([max(0, min(w - 1, int(u * (w - 1)))) for u, _ in uvs], dtype=np.int32)
        ys = np.array([max(0, min(h - 1, int(v * (h - 1)))) for _, v in uvs], dtype=np.int32)
        return arr[ys, xs]

    report = {"valid": True, "maps_present": sorted(found_imgs.keys())}

    diff_s = _sample_px(found_imgs.get("diffuse"))
    if diff_s is not None:
        cov = float(np.mean(diff_s[:, 3] > 0.5))
        black_leak = float(np.mean(np.max(diff_s[:, :3], axis=1) < 0.02))
        report["diffuse"] = {
            "present": True,
            "uv_alpha_coverage": round(cov, 4),
            "black_leak_pct": round(100.0 * black_leak, 3),
        }
    else:
        report["diffuse"] = {"present": False, "uv_alpha_coverage": 0.0, "black_leak_pct": 100.0}

    norm_s = _sample_px(found_imgs.get("normal"))
    if norm_s is not None:
        cov = float(np.mean(norm_s[:, 3] > 0.5))
        inv_pct = float(100.0 * np.mean(norm_s[:, 2] <= 0.50))
        tilt_pct = float(100.0 * np.mean(norm_s[:, 2] < 0.65))
        report["normal"] = {
            "present": True,
            "uv_alpha_coverage": round(cov, 4),
            "inverted_z_pct": round(inv_pct, 3),
            "extreme_tilt_pct": round(tilt_pct, 3),
        }
    else:
        report["normal"] = {"present": False, "uv_alpha_coverage": 0.0, "inverted_z_pct": 100.0}

    ao_s = _sample_px(found_imgs.get("ao"))
    if ao_s is not None:
        cov = float(np.mean(ao_s[:, 3] > 0.5))
        luma = ao_s[:, 0]
        report["ao"] = {
            "present": True,
            "uv_alpha_coverage": round(cov, 4),
            "ao_mean": round(float(np.mean(luma)), 4),
            "ao_std": round(float(np.std(luma)), 4),
            "ao_blackout_pct": round(float(100.0 * np.mean(luma < 0.04)), 3),
        }
    else:
        report["ao"] = {"present": False, "uv_alpha_coverage": 0.0, "ao_std": 0.0, "ao_blackout_pct": 100.0}

    orm_s = _sample_px(found_imgs.get("orm"))
    report["orm"] = {"present": orm_s is not None}
    return report


def evaluate_uv_and_bake(uv_rep, bake_rep, fidelity, criteria):
    blocking = []
    warnings = []

    fid_ok = bool(fidelity.get("valid"))
    if not fid_ok:
        blocking.append("SpatialCorrespondence: BVHTree correspondence evaluation failed.")
    else:
        if fidelity["sym_mean_dev_pct"] > criteria["max_mean_dev_pct"]:
            fid_ok = False
            blocking.append(
                f"SpatialCorrespondence: mean_dev={fidelity['sym_mean_dev_pct']}% > {criteria['max_mean_dev_pct']}%"
            )
        if fidelity["bbox_dim_drift_pct"] > criteria["max_bbox_drift_pct"]:
            fid_ok = False
            blocking.append(
                f"SpatialCorrespondence: bbox_drift={fidelity['bbox_dim_drift_pct']}% > {criteria['max_bbox_drift_pct']}%"
            )

    uv_ok = bool(uv_rep.get("has_uv"))
    if not uv_ok:
        blocking.append("UVAudit: candidate mesh has no active UV map.")
    else:
        if uv_rep["uv_oob_loops"] > 0:
            uv_ok = False
            blocking.append(f"UVBounds: uv_oob_loops={uv_rep['uv_oob_loops']} > 0")
        if uv_rep["uv_zero_area_tris"] > criteria.get("max_zero_area_tris", 4):
            uv_ok = False
            blocking.append(f"UVZeroArea: zero_area_tris={uv_rep['uv_zero_area_tris']}")
        if uv_rep["uv_overlap_area_pct"] > criteria["max_overlap_pct"]:
            uv_ok = False
            blocking.append(
                f"UVOverlap: overlap_area={uv_rep['uv_overlap_area_pct']}% > {criteria['max_overlap_pct']}%"
            )
        if uv_rep["uv_p95_stretch"] > criteria["max_p95_stretch"]:
            uv_ok = False
            blocking.append(
                f"UVStretch: p95_stretch={uv_rep['uv_p95_stretch']} > {criteria['max_p95_stretch']}"
            )
        if uv_rep["uv_mean_stretch"] > criteria["max_mean_stretch"]:
            uv_ok = False
            blocking.append(
                f"UVStretch: mean_stretch={uv_rep['uv_mean_stretch']} > {criteria['max_mean_stretch']}"
            )
        if uv_rep["uv_packing_coverage_pct"] < criteria["min_packing_coverage_pct"]:
            uv_ok = False
            blocking.append(
                f"UVPacking: effective_coverage={uv_rep['uv_packing_coverage_pct']}% < {criteria['min_packing_coverage_pct']}%"
            )

    bake_ok = True
    diff = bake_rep.get("diffuse", {})
    norm = bake_rep.get("normal", {})
    ao = bake_rep.get("ao", {})

    for rname, rinfo in (("diffuse", diff), ("normal", norm), ("ao", ao)):
        if not rinfo.get("present"):
            bake_ok = False
            blocking.append(f"BakeMissing: missing baked map '{rname}'.")
        elif rinfo.get("uv_alpha_coverage", 0.0) < criteria["min_bake_coverage"]:
            bake_ok = False
            blocking.append(
                f"BakeCoverage: {rname} coverage={rinfo.get('uv_alpha_coverage')} < {criteria['min_bake_coverage']}"
            )

    if norm.get("present") and norm.get("inverted_z_pct", 0.0) > criteria["max_inverted_normal_pct"]:
        bake_ok = False
        blocking.append(
            f"BakeNormalFlip: inverted_z={norm['inverted_z_pct']}% > {criteria['max_inverted_normal_pct']}%"
        )
    if ao.get("present"):
        if ao.get("ao_std", 0.0) < criteria["min_ao_std"]:
            bake_ok = False
            blocking.append(f"BakeAOFlat: ao_std={ao['ao_std']} < {criteria['min_ao_std']}")
        if ao.get("ao_blackout_pct", 0.0) > criteria["max_ao_blackout_pct"]:
            bake_ok = False
            blocking.append(
                f"BakeAOBlackout: ao_blackout={ao['ao_blackout_pct']}% > {criteria['max_ao_blackout_pct']}%"
            )
    if diff.get("present") and diff.get("black_leak_pct", 0.0) > criteria["max_diffuse_black_leak_pct"]:
        bake_ok = False
        blocking.append(
            f"BakeDiffuseLeak: black_leak={diff['black_leak_pct']}% > {criteria['max_diffuse_black_leak_pct']}%"
        )

    checks = [fid_ok, uv_ok, bake_ok]
    score = round(100.0 * sum(1 for c in checks if c) / len(checks), 2)
    return {
        "passed": all(checks),
        "score": score,
        "blocking_failures": blocking,
        "warnings": warnings,
        "gates": {
            "spatial_correspondence_passed": fid_ok,
            "uv_quality_passed": uv_ok,
            "bake_quality_passed": bake_ok,
        },
        "fidelity": fidelity,
        "uv_metrics": uv_rep,
        "bake_metrics": bake_rep,
    }


def main():
    args = _parse_args()
    criteria = json.loads(Path(args.criteria_json).read_text(encoding="utf-8-sig"))

    load_scene_file(args.candidate_glb)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects inside candidate GLB: {args.candidate_glb}")
    primary = meshes[0]

    uv_rep = inspect_uv_geometry(primary)
    bake_rep = inspect_baked_maps(primary, args.textures_dir)
    scene_rep = inspect_current_scene(reference_path=args.high_reference)
    fidelity = scene_rep.get("fidelity") or {"valid": False}
    renders = _render_qc_views(args.renders_dir, prefix=Path(args.candidate_glb).stem)

    verdict = evaluate_uv_and_bake(uv_rep, bake_rep, fidelity, criteria)
    verdict["renders"] = renders
    verdict["candidate_glb"] = str(args.candidate_glb)
    verdict["high_reference"] = str(args.high_reference)

    Path(args.uv_report_json).write_text(json.dumps(uv_rep, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.bake_report_json).write_text(json.dumps(bake_rep, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"UV & Bake validation finished: passed={verdict['passed']} score={verdict['score']}")


if __name__ == "__main__":
    main()
