"""Per-part P0 routing from live Blender inspection."""

import json
from pathlib import Path

from .asset_ids import mesh_part_id
from .config import DEFAULT_BLENDER_EXE
from .retopo_backends import AUTOREMESHER_ROUTE, AutoRemesherAdapter, PyMeshLabAdapter
from .sandbox import ShortPathWorkspace


def _classify(mesh, backends):
    nonmanifold = mesh["welded_non_manifold_edges"]
    minor = 1 <= nonmanifold <= 15
    dirty = nonmanifold > 15
    multi = mesh["welded_shells_count"] > 1
    thin = mesh["thin_sheet_score"] >= 0.48
    slender = mesh["slender_aspect_ratio"] >= 4.0
    tags, forbidden = [], []
    if nonmanifold:
        tags.append("minor_nonmanifold" if minor else "dirty_nonmanifold")
        forbidden.append("raw_quadriflow")
    if multi and not minor:
        tags.extend(("multi_strand" if dirty else "multi_shell", "require_gap_preservation"))
        forbidden.extend(("voxel_seal_quadriflow_bake", "single_shell_alpha_wrap"))
    if thin:
        tags.append("thin_sheet")
        forbidden.extend(("instant_meshes", "remi_double_shell_im", "voxel_seal_quadriflow_bake", AUTOREMESHER_ROUTE))
    if slender:
        tags.append("slender_ribbon" if thin else "slender_part")
    hard = not nonmanifold and not (slender and thin) and mesh["planar_dominance_score"] >= 0.58
    if hard:
        tags.extend(("hard_surface", "clean_part"))
    elif not nonmanifold and not thin:
        tags.append("clean_part")

    if dirty and multi:
        route = "per_shell_strand_repair_or_rebuild" if backends["blender_mesh_tools"] else "manual_review"
    elif hard and not thin and backends.get("autoremesher_cli"):
        route = AUTOREMESHER_ROUTE
    elif not backends["blender_quadriflow"]:
        route = "decimate_quadify_bake" if backends["blender_mesh_tools"] and not nonmanifold else "manual_review"
    elif nonmanifold:
        route = "quadriflow_healed_post_clean_bake"
    elif hard:
        route = "quadriflow_sharp_flow_refine_bake"
    else:
        route = "quadriflow_smooth_flow_refine_bake"
    return tags, route, list(dict.fromkeys(forbidden))


def route_parts(meshes, backends):
    asset_kind = "multi_part" if len(meshes) > 1 else "single_prop"
    parts = []
    for mesh in sorted(meshes, key=lambda item: item["name"]):
        name = mesh["name"]
        tags, route, forbidden = _classify(mesh, backends)
        parts.append({
            "part_id": mesh_part_id(name), "object_name": name,
            "shells_welded": mesh["welded_shells_count"], "faces_raw": mesh["faces_raw"],
            "surface_area": mesh["surface_area"],
            "welded_non_manifold_edges": mesh["welded_non_manifold_edges"],
            "thin_sheet_score": mesh["thin_sheet_score"], "slender_aspect_ratio": mesh["slender_aspect_ratio"],
            "planar_dominance_score": mesh["planar_dominance_score"], "has_existing_uv": mesh["has_uv"],
            "asset_tags": [asset_kind, *tags], "recommended_retopo_route": route, "forbidden_routes": forbidden,
        })
    return parts


def triage_asset(input_asset, output_dir, *, blender_exe=DEFAULT_BLENDER_EXE):
    from .topology_agent_loop import INSPECTOR_SCRIPT, _run_blender

    source = Path(input_asset).resolve(strict=True)
    output = Path(output_dir).resolve()
    with ShortPathWorkspace(output) as workspace:
        report_path = workspace.path("profile.json")
        ok, log = _run_blender(blender_exe, INSPECTOR_SCRIPT, [
            "--input", str(source), "--out-json", str(report_path), "--triage",
        ], workspace.path("triage.log"), timeout=300)
        if not ok:
            raise RuntimeError(f"Asset inspection failed:\n{log[-1200:]}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        backends = {
            **report["available_backends"],
            "autoremesher_cli": AutoRemesherAdapter.discover() is not None,
            "instant_meshes_cli": False, "xatlas_py": False,
            "pymeshlab_worker": PyMeshLabAdapter.discover() is not None, "curiomesh_cli": False,
        }
        meshes = report["meshes"]
        if not meshes:
            raise ValueError("Asset contains no inspectable mesh parts")
        parts = route_parts(meshes, backends)
        profile = {
            "schema_version": "2.1", "source_path": str(source), "available_backends": backends,
            "scene_summary": {
                "mesh_count": len(parts),
                "shells_welded": sum(part["shells_welded"] for part in parts),
                "total_faces_raw": report["total_faces_raw"],
                "welded_non_manifold_edges": sum(part["welded_non_manifold_edges"] for part in parts),
                "welded_non_manifold_verts": sum(mesh["welded_non_manifold_verts"] for mesh in meshes),
                "has_existing_uv": any(part["has_existing_uv"] for part in parts),
                "bbox_diag_m": report["bbox_diag_m"],
            },
            "parts": parts,
        }
        report_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        workspace.publish("profile.json", output / "asset_profile.json")
    return profile
