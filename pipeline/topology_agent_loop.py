"""Closed-loop autonomous topology repair & validation engine.

Implements:
Raw Asset -> Geometry Detector -> Agent Decision -> Topology Repair Toolkit
-> Independent Quality Validator -> Pass?
    - Yes -> Deliver Asset
    - No -> Rollback & Error Analysis -> Agent Decision
    - Budget Exceeded -> Record Failure
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .asset_triage import route_parts
from .budget_allocator import allocate_object_budgets
from .config import DEFAULT_BLENDER_EXE, REPO_ROOT, now_iso, require_blender
from .experiment_ledger import record_experiment
from .knowledge_recall import apply_priors_to_candidates, load_strategy_priors
from .retopo_backends import (
    AUTOREMESHER_ROUTE, PYMESHLAB_ROUTES, AutoRemesherAdapter, PyMeshLabAdapter, permits_alpha_wrap,
)
from .sandbox import ShortPathWorkspace, run_staged_blender_job


INSPECTOR_SCRIPT = REPO_ROOT / "blender_scripts" / "topology_inspector.py"
TOOLKIT_SCRIPT = REPO_ROOT / "blender_scripts" / "topology_toolkit.py"
VALIDATOR_SCRIPT = REPO_ROOT / "blender_scripts" / "topology_validator.py"
DEFAULT_KNOWLEDGE_DIR = REPO_ROOT / "knowledge"

STRATEGY_ROUTES: dict[str, str] = {
    AUTOREMESHER_ROUTE: AUTOREMESHER_ROUTE,
    **{route: "pymeshlab_route" for route in PYMESHLAB_ROUTES.values()},
    "clean_weld_quadify": "quadify_only_route",
    "quadriflow_sharp_flow_refine_bake": "quadriflow_route",
    "quadriflow_smooth_flow_refine_bake": "quadriflow_route",
    "quadriflow_healed_post_clean_bake": "quadriflow_route",
    "quadriflow_healed_sharp_shrink_bake": "quadriflow_route",
    "quadriflow_healed_smooth_shrink_bake": "quadriflow_route",
    "quadriflow_direct_shrink_bake": "quadriflow_route",
    "quadriflow_smooth_post_clean_bake": "quadriflow_route",
    "quadriflow_sharp_no_shrink_high": "quadriflow_route",
    "voxel_seal_quadriflow_bake": "voxel_route",
    "decimate_quadify_bake": "decimate_route",
}


def get_strategy_route(strategy_id: str) -> str:
    return STRATEGY_ROUTES.get(strategy_id, "custom_route")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class RetopoCriteria:
    """6-pillar acceptance thresholds anchored on standardized rigged anime basemesh targets."""

    min_faces: int = 2500
    max_faces: int = 22000
    min_quad_ratio: float = 0.80
    min_valence4_ratio: float = 0.68
    max_ngons: int = 0
    max_bad_aspect_pct: float = 12.0
    max_welded_non_manifold: int = 0
    max_debris_shells: int = 0
    max_flipped_normal_pct: float = 2.0
    max_mean_dev_pct: float = 0.85
    max_p95_dev_pct: float = 2.2
    max_sym_max_dev_pct: float = 5.0
    max_bbox_drift_pct: float = 2.5
    min_bake_coverage: float = 0.94

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_blender(
    blender_exe: str,
    script: Path,
    args: list[str],
    log_path: Path,
    timeout: int = 600,
) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [require_blender(blender_exe), "-b", "--python-exit-code", "1", "--python", str(script), "--"]

    def invoke(worker_args):
        proc = subprocess.run(command + worker_args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, check=False)
        return proc.returncode == 0, proc.stdout + ("\n---STDERR---\n" + proc.stderr if proc.stderr else "")

    if script.name in {"topology_toolkit.py", "uv_bake_toolkit.py", "topology_validator.py", "uv_bake_validator.py"}:
        with ShortPathWorkspace(log_path.parent) as workspace:
            ok, combined = run_staged_blender_job(args, workspace, invoke)
    else:
        ok, combined = invoke(args)
    log_path.write_text(combined, encoding="utf-8")
    return ok, combined


def _bbox_diag_from_report(report: dict[str, Any]) -> float:
    meshes = report.get("meshes", [])
    if not meshes:
        return 1.0
    d = meshes[0].get("dims", [1.0, 1.0, 1.0])
    return max(0.05, (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5)


def agent_decide_repair_plan(
    baseline_metrics: dict[str, Any],
    attempt_history: list[dict[str, Any]],
    banned_strategies: set[str],
    criteria: RetopoCriteria,
    banned_routes: set[str] | None = None,
    knowledge_dir: Path | None = None,
) -> tuple[str, list[dict[str, Any]], str]:
    """Select next parameterized repair recipe using Geometry Detector + Rollback Error Analysis."""
    diag = _bbox_diag_from_report(baseline_metrics)
    last_err = attempt_history[-1]["error_analysis"] if attempt_history else {}
    last_classes = set(last_err.get("failure_classes", []))

    has_junk = len(baseline_metrics.get("junk_objects", [])) > 0
    faces = int(baseline_metrics.get("total_faces_raw", 0))
    meshes = baseline_metrics.get("meshes", [])
    avg_quad = (
        sum(m["raw_quad_ratio"] * m["faces_raw"] for m in meshes) / max(1, faces)
        if meshes
        else 0.0
    )
    non_man = sum(m.get("welded_non_manifold_edges", 0) for m in meshes)
    boundary = sum(m.get("welded_boundary_edges", 0) for m in meshes)

    prefix_actions: list[dict[str, Any]] = []
    if has_junk:
        prefix_actions.append({"action": "cleanup_scene_for_engine", "params": {"drop_numbered_duplicates": False}})

    operation = baseline_metrics.get("pymeshlab_operation")
    if operation:
        route = PYMESHLAB_ROUTES[operation]
        if route in banned_strategies or get_strategy_route(route) in (banned_routes or set()):
            raise RuntimeError(f"Requested PyMeshLab route rejected: {route}")
        params = {"operation": operation, "max_p95_dev_pct": criteria.max_p95_dev_pct}
        if operation == "qecd-with-texture":
            params["target_faces"] = criteria.max_faces
        actions = prefix_actions + [{"action": "pymeshlab_retopo", "params": params}]
        if operation == "close-holes":
            actions.append({"action": "unwrap_and_bake_textures", "params": {
                "texture_size": 2048, "cage_extrusion": 0.003, "max_ray_distance": 0.018,
            }})
        return route, actions, f"Run isolated PyMeshLab {operation} with its purpose-specific acceptance criteria."

    # Candidate ordered strategy portfolio (adapted dynamically by last failure analysis)
    target_mid = max(criteria.min_faces + 1500, min(14000, criteria.max_faces - 4000))
    target_high = min(criteria.max_faces - 1500, max(16000, int(criteria.max_faces * 0.85)))

    candidates: list[tuple[str, list[dict[str, Any]], str]] = []
    parts = baseline_metrics.get("parts_routing", [])
    tags = set(baseline_metrics.get("asset_tags", []))
    auto_parts = [
        part["object_name"] for part in parts
        if part["recommended_retopo_route"] == AUTOREMESHER_ROUTE
    ]

    if faces <= criteria.max_faces and avg_quad >= criteria.min_quad_ratio and non_man == 0:
        candidates.append(
            (
                "clean_weld_quadify",
                prefix_actions
                + [
                    {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 24}},
                    {"action": "weld_and_quadify", "params": {"weld_dist": 1e-4}},
                ],
                "Input already near quad target; run non-destructive seam weld, debris sanitize, and quadify.",
            )
        )

    if auto_parts:
        candidates.append((
            AUTOREMESHER_ROUTE,
            prefix_actions + [
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 0}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 24, "min_shell_verts": 0}},
                {"action": "autoremesher_retopo", "params": {
                    "target_faces": target_high, "auto_parts": auto_parts,
                    "use_preserve_sharp": True, "max_p95_dev_pct": criteria.max_p95_dev_pct,
                }},
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 0}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 12, "min_shell_verts": 0}},
                {"action": "unwrap_and_bake_textures", "params": {
                    "texture_size": 2048, "angle_limit_deg": 66.0, "island_margin": 0.004,
                    "cage_extrusion": 0.003, "max_ray_distance": 0.018,
                }},
            ],
            "Use anisotropic AutoRemesher for eligible hard parts with per-shell coverage; "
            "use the local QuadriFlow route for other parts before shared baking.",
        ))

    # Primary flagship strategy: pre-heal + sharp Quadriflow + normal-aware shrinkwrap + tangential quad-flow relaxation + ORM bake
    candidates.append(
        (
            "quadriflow_sharp_flow_refine_bake",
            prefix_actions
            + [
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 24}},
                {
                    "action": "quadriflow_retopo",
                    "params": {
                        "target_faces": target_high,
                        "use_preserve_sharp": True,
                        "smooth_normals": False,
                        "use_mesh_symmetry": False,
                    },
                },
                {
                    "action": "shrinkwrap_to_reference",
                    "params": {"offset": 0.0, "normal_aware": True, "min_normal_dot": 0.15, "max_snap_dist_ratio": 0.018},
                },
                {
                    "action": "refine_quad_flow",
                    "params": {"iterations": 3, "tangential_factor": 0.36, "boundary_factor": 0.24, "reproject": True},
                },
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                {
                    "action": "unwrap_and_bake_textures",
                    "params": {
                        "texture_size": 2048,
                        "angle_limit_deg": 86.0,
                        "island_margin": 0.004,
                        "cage_extrusion": 0.003,
                        "max_ray_distance": 0.018,
                    },
                },
            ],
            "Flagship healed sharp Quadriflow + normal-aware thin-sheet shrinkwrap + tangential quad-flow relaxation + ORM bake.",
        )
    )

    # Secondary flagship for slender ribbons/tassels: smooth-normal Quadriflow preserves delicate straps without sharp-edge pinching
    candidates.append(
        (
            "quadriflow_smooth_flow_refine_bake",
            prefix_actions
            + [
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 24}},
                {
                    "action": "quadriflow_retopo",
                    "params": {
                        "target_faces": target_high,
                        "use_preserve_sharp": False,
                        "smooth_normals": True,
                        "use_mesh_symmetry": False,
                    },
                },
                {
                    "action": "shrinkwrap_to_reference",
                    "params": {"offset": 0.0, "normal_aware": True, "min_normal_dot": 0.15, "max_snap_dist_ratio": 0.018},
                },
                {
                    "action": "refine_quad_flow",
                    "params": {"iterations": 3, "tangential_factor": 0.34, "boundary_factor": 0.22, "reproject": True},
                },
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                {
                    "action": "unwrap_and_bake_textures",
                    "params": {
                        "texture_size": 2048,
                        "angle_limit_deg": 86.0,
                        "island_margin": 0.004,
                        "cage_extrusion": 0.003,
                        "max_ray_distance": 0.018,
                    },
                },
            ],
            "Smooth-normal Quadriflow + normal-aware shrinkwrap + tangential quad-flow relaxation (preserves slender ribbons/tassels intact).",
        )
    )

    # Pre-heal Strategy when Geometry Detector found raw non-manifold/pinholes before Quadriflow
    if non_man > 0 or boundary > 0:
        candidates.append(
            (
                "quadriflow_healed_post_clean_bake",
                prefix_actions
                + [
                    {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                    {"action": "heal_manifold", "params": {"max_hole_edges": 24}},
                    {
                        "action": "quadriflow_retopo",
                        "params": {
                            "target_faces": target_high,
                            "use_preserve_sharp": False,
                            "smooth_normals": True,
                            "use_mesh_symmetry": False,
                        },
                    },
                    {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
                    {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                    {
                        "action": "unwrap_and_bake_textures",
                        "params": {"texture_size": 2048, "angle_limit_deg": 86.0, "island_margin": 0.004, "cage_extrusion": 0.003, "max_ray_distance": 0.018},
                    },
                ],
                "Raw input has welded non-manifold/boundary defects -> heal manifold pinholes first, then Quadriflow + post-clean + tight bake.",
            )
        )

    healed_sharp_shrink_entry = (
        "quadriflow_healed_sharp_shrink_bake",
        prefix_actions
        + [
            {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
            {"action": "heal_manifold", "params": {"max_hole_edges": 24}},
            {
                "action": "quadriflow_retopo",
                "params": {
                    "target_faces": target_high,
                    "use_preserve_sharp": True,
                    "smooth_normals": False,
                    "use_mesh_symmetry": False,
                },
            },
            {"action": "shrinkwrap_to_reference", "params": {"offset": 0.0}},
            {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
            {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
            {
                "action": "unwrap_and_bake_textures",
                "params": {
                    "texture_size": 2048,
                    "angle_limit_deg": 86.0,
                    "island_margin": 0.004,
                    "cage_extrusion": 0.003,
                    "max_ray_distance": 0.018,
                },
            },
        ],
        "Healed sharp high-budget Quadriflow + reference Shrinkwrap (preserves thin collar/skirt hems <=1.6% drift & valence4 >=0.72) + GLB-verified heal + ORM bake.",
    )

    healed_shrink_entry = (
        "quadriflow_healed_smooth_shrink_bake",
        prefix_actions
        + [
            {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
            {"action": "heal_manifold", "params": {"max_hole_edges": 24}},
            {
                "action": "quadriflow_retopo",
                "params": {
                    "target_faces": target_high,
                    "use_preserve_sharp": False,
                    "smooth_normals": True,
                    "use_mesh_symmetry": False,
                },
            },
            {"action": "shrinkwrap_to_reference", "params": {"offset": 0.0}},
            {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
            {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
            {
                "action": "unwrap_and_bake_textures",
                "params": {
                    "texture_size": 2048,
                    "angle_limit_deg": 86.0,
                    "island_margin": 0.004,
                    "cage_extrusion": 0.003,
                    "max_ray_distance": 0.018,
                },
            },
        ],
        "Healed smooth high-budget Quadriflow + reference Shrinkwrap (eliminates thin hem bbox drift) + GLB-verified manifold heal + ORM bake.",
    )

    # Strategy 1 (escalation on bounding/silhouette drift): Healed sharp high-budget Quadriflow + Reference Shrinkwrap
    if "high_surface_deviation" in last_classes and "quadriflow_execution_failed" not in last_classes:
        candidates.append(healed_sharp_shrink_entry)
        candidates.append(healed_shrink_entry)

    # Strategy 2: Quadriflow smooth WITHOUT shrinkwrap + Post-Sanitize (skip if previous attempt already failed deviation without shrinkwrap)
    if "high_surface_deviation" not in last_classes:
        candidates.append(
            (
                "quadriflow_smooth_post_clean_bake",
                prefix_actions
                + [
                    {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                    {"action": "heal_manifold", "params": {"max_hole_edges": 16}},
                    {
                        "action": "quadriflow_retopo",
                        "params": {
                            "target_faces": target_high,
                            "use_preserve_sharp": False,
                            "smooth_normals": True,
                            "use_mesh_symmetry": False,
                        },
                    },
                    {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
                    {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                    {
                        "action": "unwrap_and_bake_textures",
                        "params": {"texture_size": 2048, "angle_limit_deg": 86.0, "island_margin": 0.004, "cage_extrusion": 0.003, "max_ray_distance": 0.018},
                    },
                ],
                "Direct Quadriflow without Shrinkwrap + post-retopo sanitize/heal (eliminates zero-area/edge artifacts) + tight-cage ORM bake.",
            )
        )

    if "high_surface_deviation" not in last_classes and "quadriflow_execution_failed" not in last_classes:
        candidates.append(healed_sharp_shrink_entry)
        candidates.append(healed_shrink_entry)

    # Strategy 3: Direct sanitize + Quadriflow (preserve_sharp=True) + Shrinkwrap + Heal + Post-Sanitize + Bake
    if "quadriflow_execution_failed" not in last_classes and "non_manifold_blocking" not in last_classes:
        candidates.append(
            (
                "quadriflow_direct_shrink_bake",
                prefix_actions
                + [
                    {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                    {"action": "heal_manifold", "params": {"max_hole_edges": 16}},
                    {
                        "action": "quadriflow_retopo",
                        "params": {
                            "target_faces": target_mid,
                            "use_preserve_sharp": True,
                            "use_preserve_boundary": False,
                            "use_mesh_symmetry": False,
                        },
                    },
                    {"action": "shrinkwrap_to_reference", "params": {"offset": 0.0}},
                    {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
                    {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                    {
                        "action": "unwrap_and_bake_textures",
                        "params": {"texture_size": 2048, "angle_limit_deg": 86.0, "island_margin": 0.004, "cage_extrusion": 0.003, "max_ray_distance": 0.018},
                    },
                ],
                "Dense raw mesh -> sanitize + safe heal + Quadriflow sharp loops + Shrinkwrap + manifold heal + Cycles ORM bake.",
            )
        )

    # Strategy 4: Adaptive Voxel seal (droplet prune without edge collapse) -> Quadriflow -> post-clean -> Bake
    voxel_sz = round(max(0.004, min(0.015, diag * 0.0032)), 5)
    voxel_scales = sorted(
        {
            voxel_sz,
            round(max(voxel_sz * 1.6, 0.022), 4),
            round(max(voxel_sz * 2.2, 0.032), 4),
            0.042,
        }
    )
    candidates.append(
        (
            "voxel_seal_quadriflow_bake",
            prefix_actions
            + [
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                {
                    "action": "adaptive_voxel_quadriflow",
                    "params": {
                        "voxel_scales": voxel_scales,
                        "target_faces": target_high,
                        "min_shell_verts": 64,
                        "smooth_normals": True,
                        "use_preserve_sharp": False,
                        "use_preserve_boundary": False,
                        "use_mesh_symmetry": False,
                    },
                },
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 64}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                {
                    "action": "unwrap_and_bake_textures",
                    "params": {"texture_size": 2048, "cage_extrusion": 0.015, "max_ray_distance": 0.06},
                },
            ],
            "Seal manifold via adaptive multi-scale Voxel Remesh (no post-voxel edge collapse) + Quadriflow + post-clean + Cycles ORM bake.",
        )
    )

    # Strategy 4: Structure-preserving Collapse Decimate + Quadify + Post-clean + Heal + Re-bake
    candidates.append(
        (
            "decimate_quadify_bake",
            prefix_actions
            + [
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                {"action": "decimate_and_quadify", "params": {"target_faces": target_high}},
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                {
                    "action": "unwrap_and_bake_textures",
                    "params": {"texture_size": 2048, "cage_extrusion": 0.045, "max_ray_distance": 0.15},
                },
            ],
            "Preserve separate thin garment laces/tassels via Collapse Decimate -> Quadify -> post-clean -> Heal -> Cycles Re-bake.",
        )
    )

    # Strategy 5: Higher-budget pure Quadriflow sharp without shrinkwrap fallback
    candidates.append(
        (
            "quadriflow_sharp_no_shrink_high",
            prefix_actions
            + [
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "min_shell_verts": 48}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 16}},
                {
                    "action": "quadriflow_retopo",
                    "params": {
                        "target_faces": target_high,
                        "use_preserve_sharp": True,
                        "smooth_normals": True,
                        "use_mesh_symmetry": False,
                    },
                },
                {"action": "sanitize_mesh", "params": {"weld_seams": True, "weld_dist": 2e-4, "min_edge_length": 1e-4, "min_shell_verts": 48}},
                {"action": "heal_manifold", "params": {"max_hole_edges": 12}},
                {
                    "action": "unwrap_and_bake_textures",
                    "params": {"texture_size": 2048, "cage_extrusion": 0.045, "max_ray_distance": 0.16},
                },
            ],
            "High-budget sharp Quadriflow without shrinkwrap + post-retopo sanitize/heal + bake.",
        )
    )

    if knowledge_dir is not None and not attempt_history:
        priors = load_strategy_priors(knowledge_dir, "retopo", set(baseline_metrics.get("asset_tags", [])))
        candidates = apply_priors_to_candidates(candidates, priors)

    forbidden = {route for part in parts for route in part["forbidden_routes"]}
    if auto_parts:
        # Thin or dirty members of a mixed assembly are excluded individually.
        forbidden.discard(AUTOREMESHER_ROUTE)
    if "multi_part" in tags and "single_shell_envelope" not in tags:
        forbidden.add("voxel_seal_quadriflow_bake")
    if len(parts) == 1 and candidates[0][0] != "clean_weld_quadify":
        preferred = parts[0]["recommended_retopo_route"]
        if {"thin_sheet", "minor_nonmanifold", "hard_surface"} & tags:
            candidates.sort(key=lambda entry: entry[0] != preferred)
    elif "multi_strand" in tags:
        candidates.sort(key=lambda entry: entry[0] != "quadriflow_healed_post_clean_bake")
    elif auto_parts:
        candidates.sort(key=lambda entry: entry[0] not in {"clean_weld_quadify", AUTOREMESHER_ROUTE})
    banned_r = banned_routes or set()
    for strat_id, actions, rationale in candidates:
        if strat_id not in banned_strategies | forbidden and get_strategy_route(strat_id) not in banned_r:
            return strat_id, actions, rationale

    raise RuntimeError("No unbanned repair strategies or routes left within Agent Planner portfolio.")


def analyze_failure_and_rollback(
    pre_checkpoint: Path,
    active_baseline: Path,
    strategy_id: str,
    toolkit_ok: bool,
    toolkit_log: str,
    verdict: dict[str, Any] | None,
) -> dict[str, Any]:
    """Restore clean pre-attempt checkpoint and classify root causes for next iteration."""
    shutil.copy2(pre_checkpoint, active_baseline)
    failure_classes: list[str] = []
    details: list[str] = []

    if not toolkit_ok:
        err_tail = "\n".join(toolkit_log.splitlines()[-6:]).lower()
        for reason in ("autoremesher_timeout", "autoremesher_shell_coverage", "autoremesher_execution_failed",
                       "autoremesher_budget_exceeded", "pymeshlab_nonmanifold", "pymeshlab_timeout",
                       "pymeshlab_shell_coverage", "pymeshlab_uv_contract", "pymeshlab_budget_exceeded",
                       "pymeshlab_close_holes_no_change"):
            if reason in err_tail:
                failure_classes.append(reason)
        if "quadriflow_remesh" in toolkit_log.lower() and "failed" in err_tail:
            failure_classes.append("quadriflow_execution_failed")
        if "non-manifold" in err_tail or "manifold" in err_tail:
            failure_classes.append("non_manifold_blocking")
        if "bake" in err_tail:
            failure_classes.append("bake_execution_failed")
        if not failure_classes:
            failure_classes.append("toolkit_runtime_error")
        details.append(toolkit_log[-500:])
    elif verdict is not None:
        pillars = verdict.get("pillars", {})
        if not pillars.get("1_geometric_fidelity", {}).get("passed", True):
            failure_classes.append("high_surface_deviation")
        if not pillars.get("2_topology_quality", {}).get("passed", True):
            failure_classes.append("topology_quality_below_target")
        if not pillars.get("3_manifold_cleanliness", {}).get("passed", True):
            failure_classes.append("non_manifold_blocking")
        if not pillars.get("4_polycount_budget", {}).get("passed", True):
            failure_classes.append("polycount_out_of_budget")
        if not pillars.get("5_uv_and_materials", {}).get("passed", True):
            failure_classes.append("uv_or_bake_coverage_low")
        if not pillars.get("6_rig_engine_ready", {}).get("passed", True):
            failure_classes.append("engine_or_rig_not_ready")
        details.extend(verdict.get("blocking_failures", []))

    return {
        "rolled_back_to": str(pre_checkpoint),
        "banned_strategy": strategy_id,
        "route_family": get_strategy_route(strategy_id),
        "failure_classes": failure_classes,
        "details": details,
    }


def _decompose_partial_capabilities(
    actions: list[dict[str, Any]],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    """Separate validated reusable capabilities from unresolved blockers on partial success."""
    pillars = verdict.get("pillars", {})
    passed_pillars = [k for k, v in pillars.items() if v.get("passed")]
    blocked_pillars = [k for k, v in pillars.items() if not v.get("passed")]
    reusable_steps: list[str] = []
    if pillars.get("2_topology_quality", {}).get("passed") and pillars.get("4_polycount_budget", {}).get("passed"):
        reusable_steps.extend(
            step["action"]
            for step in actions
            if step["action"] in ("quadriflow_retopo", "autoremesher_retopo", "weld_and_quadify", "decimate_and_quadify")
        )
    if pillars.get("5_uv_and_materials", {}).get("passed"):
        reusable_steps.extend(step["action"] for step in actions if step["action"] == "unwrap_and_bake_textures")
    if pillars.get("3_manifold_cleanliness", {}).get("passed"):
        reusable_steps.extend(step["action"] for step in actions if step["action"] == "sanitize_mesh")
    return {
        "passed_pillars": passed_pillars,
        "blocked_pillars": blocked_pillars,
        "reusable_actions": sorted(set(reusable_steps)),
        "blocking_failures": verdict.get("blocking_failures", []),
    }


def triage_and_persist_outcome(
    outcome_kind: str,
    asset_name: str,
    work_dir: Path,
    knowledge_dir: Path,
    baseline_metrics: dict[str, Any],
    attempt_history: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    winning_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply 4-outcome evolution rule: Success->Skill, Failure->Log, Partial->Decompose, TotalFail->SwitchRoute."""
    work_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"time": now_iso(), "outcome_kind": outcome_kind, "asset": asset_name}

    if outcome_kind == "SUCCESS" and best_attempt:
        recipe = {
            **record,
            "strategy_id": best_attempt["strategy_id"],
            "route_family": get_strategy_route(best_attempt["strategy_id"]),
            "score": best_attempt["score"],
            "actions": winning_actions or [],
            "baseline_faces": baseline_metrics.get("total_faces_raw"),
            "asset_tags": baseline_metrics.get("asset_tags", []),
            "final_pillars": best_attempt["verdict"].get("pillars", {}),
        }
        _append_jsonl(knowledge_dir / "learned_recipes.jsonl", recipe)
        md_path = work_dir / "skill_recipe.md"
        md_path.write_text(
            "\n".join(
                [
                    f"# Skill Recipe Candidate: {best_attempt['strategy_id']}",
                    "",
                    f"- Asset: `{asset_name}` ({baseline_metrics.get('total_faces_raw')} raw faces)",
                    f"- Route Family: `{recipe['route_family']}` | Score: `{best_attempt['score']}`",
                    "- Action Sequence:",
                    "```json",
                    json.dumps(winning_actions or [], ensure_ascii=False, indent=2),
                    "```",
                ]
            ),
            encoding="utf-8",
        )
        record["skill_recipe_md"] = str(md_path)
        return record

    if outcome_kind == "PARTIAL_SUCCESS" and best_attempt:
        decomp = _decompose_partial_capabilities(
            best_attempt.get("actions", []), best_attempt["verdict"]
        )
        best_dir = work_dir / "best_partial"
        best_dir.mkdir(parents=True, exist_ok=True)
        for src_key in ("cand_blend", "cand_glb"):
            src_p = best_attempt.get(src_key)
            if src_p and Path(src_p).is_file():
                shutil.copy2(src_p, best_dir / Path(src_p).name)
        partial_payload = {
            **record,
            "best_strategy_id": best_attempt["strategy_id"],
            "best_score": best_attempt["score"],
            "decomposition": decomp,
            "partial_assets_dir": str(best_dir),
        }
        (work_dir / "partial_decomposition.json").write_text(
            json.dumps(partial_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _append_jsonl(knowledge_dir / "reusable_capabilities.jsonl", partial_payload)
        return partial_payload

    # TOTAL_FAILURE: escalate beyond parameter tweaks -> switch algorithm / route
    tried_routes = sorted(
        {get_strategy_route(a["strategy_id"]) for a in attempt_history}
    )
    pivot_payload = {
        **record,
        "tried_routes": tried_routes,
        "best_score": best_attempt["score"] if best_attempt else 0.0,
        "recommended_route_pivots": [
            "split_by_connected_shells_and_retopo_per_part",
            "manual_quad_draw_cage_with_datatable_uv_transfer",
            "parametric_blender_rebuild_for_thin_ribbons_or_cords",
        ],
        "attempt_summary": [
            {
                "strategy_id": a["strategy_id"],
                "route": get_strategy_route(a["strategy_id"]),
                "score": a.get("score", 0.0),
                "failure_classes": a.get("error_analysis", {}).get("failure_classes", []),
            }
            for a in attempt_history
        ],
    }
    (work_dir / "route_pivot_recommendation.json").write_text(
        json.dumps(pivot_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _append_jsonl(knowledge_dir / "route_pivots.jsonl", pivot_payload)
    return pivot_payload


def run_closed_loop_retopo(
    input_asset: Path,
    work_dir: Path,
    criteria: RetopoCriteria | None = None,
    max_iterations: int = 5,
    blender_exe: str = DEFAULT_BLENDER_EXE,
    knowledge_dir: Path | None = None,
    dry_run: bool = False,
    shell_contract: dict[str, Any] | None = None,
    pymeshlab_operation: str | None = None,
) -> dict[str, Any]:
    """Execute the full closed-loop retopology & independent validation workflow."""
    t_start = time.monotonic()
    input_asset = Path(input_asset).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    criteria = criteria or RetopoCriteria()
    kdir = Path(knowledge_dir).resolve() if knowledge_dir else DEFAULT_KNOWLEDGE_DIR
    envelope = (shell_contract or {}).get("single_shell_envelope", False)
    if (pymeshlab_operation == "alpha-wrap" or envelope) and not permits_alpha_wrap(shell_contract):
        return {"status": "blocked", "reason": "pymeshlab_envelope_requires_opt_in",
                "detail": "A single-shell envelope requires explicit intent and cannot include separation groups or gap rays."}
    if envelope and pymeshlab_operation not in (None, "alpha-wrap"):
        return {"status": "blocked", "reason": "pymeshlab_conflicting_operation"}

    checkpoints_dir = work_dir / "checkpoints"
    attempts_dir = work_dir / "attempts"
    deliverables_dir = work_dir / "deliverables"
    checkpoints_dir.mkdir(exist_ok=True)
    attempts_dir.mkdir(exist_ok=True)
    deliverables_dir.mkdir(exist_ok=True)

    criteria_path = work_dir / "acceptance_criteria.json"
    criteria_data = {**criteria.to_dict(), "shell_contract": shell_contract or {}}
    criteria_path.write_text(json.dumps(criteria_data, indent=2), encoding="utf-8")

    reference_copy = checkpoints_dir / f"00_raw_reference{input_asset.suffix.lower()}"
    if input_asset.is_file():
        ShortPathWorkspace.publish_file(input_asset, reference_copy)

    if dry_run:
        return {
            "status": "dry_run",
            "input_asset": str(input_asset),
            "work_dir": str(work_dir),
            "max_iterations": max_iterations,
            "criteria": criteria.to_dict(),
            "pymeshlab_operation": pymeshlab_operation,
        }

    if not input_asset.is_file():
        raise FileNotFoundError(f"Missing input asset: {input_asset}")

    pymeshlab = PyMeshLabAdapter.discover()
    if pymeshlab_operation and not pymeshlab:
        return {"status": "blocked", "reason": "pymeshlab_worker_unavailable"}
    if envelope and pymeshlab:
        pymeshlab_operation = "alpha-wrap"
    if pymeshlab_operation:
        criteria_data.update(validation_mode=pymeshlab_operation, min_quad_ratio=0.0, min_valence4_ratio=0.0)
    reproducible_command = (
        f'python run_pipeline.py retopo --input "{input_asset}" --out "{work_dir}" --max-iters {max_iterations}'
        f" --min-faces {criteria.min_faces} --max-faces {criteria.max_faces}"
        f" --min-quad-ratio {criteria.min_quad_ratio} --min-valence4-ratio {criteria.min_valence4_ratio}"
    )
    if shell_contract:
        contract_path = work_dir / "shell_contract.json"
        contract_path.write_text(json.dumps(shell_contract, ensure_ascii=False, indent=2), encoding="utf-8")
        reproducible_command += f' --shell-contract "{contract_path}"'
    if pymeshlab_operation:
        reproducible_command += f" --pymeshlab-operation {pymeshlab_operation}"
    experiment_config = {"task_type": "retopo", "asset": input_asset.name, "criteria": criteria_data}

    # Step 1: Geometry Detector on raw model
    baseline_json = work_dir / "01_baseline_detector.json"
    ok_det, det_log = _run_blender(
        blender_exe,
        INSPECTOR_SCRIPT,
        ["--input", str(reference_copy), "--out-json", str(baseline_json), "--triage", "--shell-samples"],
        work_dir / "01_baseline_detector.log",
    )
    if not ok_det:
        raise RuntimeError(f"Initial Geometry Detector failed:\n{det_log[-1000:]}")
    baseline_metrics = json.loads(baseline_json.read_text(encoding="utf-8-sig"))
    baseline_metrics["available_backends"]["autoremesher_cli"] = AutoRemesherAdapter.discover() is not None
    baseline_metrics["available_backends"]["pymeshlab_worker"] = pymeshlab is not None
    if pymeshlab_operation:
        baseline_metrics["pymeshlab_operation"] = pymeshlab_operation
    criteria_path.write_text(json.dumps({
        **criteria_data, "reference_shell_samples": baseline_metrics.pop("shell_samples"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    parts = route_parts(baseline_metrics["meshes"], baseline_metrics["available_backends"])
    tags = {"multi_part" if baseline_metrics["mesh_count"] > 1 else "single_prop"}
    tags.update(tag for part in parts for tag in part["asset_tags"])
    if shell_contract:
        tags.update(shell_contract.get("asset_tags", []))
        if shell_contract.get("single_shell_envelope"):
            tags.add("single_shell_envelope")
        if shell_contract.get("separation_groups") or shell_contract.get("require_gap_preservation"):
            tags.add("require_gap_preservation")
    baseline_metrics["asset_tags"] = sorted(tags)
    baseline_metrics["parts_routing"] = parts
    baseline_json.write_text(json.dumps(baseline_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    preserve_shells = (
        ("multi_part" in tags or "require_gap_preservation" in tags)
        and "single_shell_envelope" not in tags
    )

    active_baseline = checkpoints_dir / f"active_baseline{input_asset.suffix.lower()}"
    shutil.copy2(reference_copy, active_baseline)

    banned: set[str] = set()
    banned_routes: set[str] = set()
    attempt_history: list[dict[str, Any]] = []
    best_attempt: dict[str, Any] | None = None

    for it in range(1, max_iterations + 1):
        try:
            strat_id, actions, rationale = agent_decide_repair_plan(
                baseline_metrics,
                attempt_history,
                banned,
                criteria,
                banned_routes=banned_routes,
                knowledge_dir=kdir,
            )
        except RuntimeError as exc:
            best_score = best_attempt["score"] if best_attempt else 0.0
            outcome_kind = "PARTIAL_SUCCESS" if best_score >= 50.0 else "TOTAL_FAILURE"
            triage = triage_and_persist_outcome(
                outcome_kind,
                input_asset.name,
                work_dir,
                kdir,
                baseline_metrics,
                attempt_history,
                best_attempt,
            )
            fail_report = {
                "status": "partial_success" if outcome_kind == "PARTIAL_SUCCESS" else "total_failure_route_pivot",
                "outcome_kind": outcome_kind,
                "reason": str(exc),
                "iterations_used": it - 1,
                "max_iterations": max_iterations,
                "triage": triage,
                "baseline_metrics": baseline_metrics,
                "attempt_history": attempt_history,
                "finished_at": now_iso(),
            }
            (work_dir / "failure_report.json").write_text(
                json.dumps(fail_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            record_experiment(
                kdir,
                task_type="retopo",
                asset=input_asset.name,
                work_dir=work_dir,
                autonomous_completed=False,
                human_interventions=0,
                iterations_used=max(1, it - 1),
                elapsed_seconds=time.monotonic() - t_start,
                final_score=best_score,
                passed=False,
                winning_strategy_id=best_attempt.get("strategy_id", "") if best_attempt else "",
                failure_reasons=[bf for a in attempt_history for bf in (a.get("verdict") or {}).get("blocking_failures", [])] or [str(exc)],
                seed=0,
                config_payload=experiment_config,
                reproducible_command=reproducible_command,
            )
            return fail_report

        iter_dir = attempts_dir / f"iter_{it:02d}_{strat_id}"
        closed_objects = [
            mesh["name"] for mesh in baseline_metrics["meshes"]
            if mesh["welded_boundary_edges"] == 0 and mesh["welded_non_manifold_edges"] == 0
        ]
        for step in actions:
            params = step.get("params") or {}
            if preserve_shells and step["action"] in {"sanitize_mesh", "heal_manifold"}:
                params["min_shell_verts"] = 0
            if step["action"] == "heal_manifold" and closed_objects:
                params["closed_objects"] = closed_objects
            if step["action"] in {"quadriflow_retopo", "autoremesher_retopo"} and len(parts) > 1:
                params["per_mesh_preserve_sharp"] = {
                    part["object_name"]: "hard_surface" in part["asset_tags"]
                    for part in parts
                    if {"hard_surface", "thin_sheet", "minor_nonmanifold"} & set(part["asset_tags"])
                }
            if "target_faces" in params:
                total = min(params["target_faces"], criteria.max_faces)
                meshes = baseline_metrics["meshes"]
                params["target_faces"] = total
                params["per_mesh_targets"] = allocate_object_budgets(
                    [{**mesh, "part_id": mesh["name"]} for mesh in meshes], total
                )
        iter_dir.mkdir(parents=True, exist_ok=True)
        pre_ckpt = checkpoints_dir / f"ckpt_pre_iter_{it:02d}{active_baseline.suffix.lower()}"
        shutil.copy2(active_baseline, pre_ckpt)

        cand_blend = iter_dir / f"{input_asset.stem}_candidate.blend"
        cand_glb = iter_dir / f"{input_asset.stem}_candidate.glb"
        plan_json = iter_dir / "repair_plan.json"
        exec_log = iter_dir / "toolkit_exec.json"

        plan_payload = {
            "iteration": it,
            "strategy_id": strat_id,
            "rationale": rationale,
            "input_path": str(active_baseline),
            "reference_glb": str(reference_copy),
            "textures_dir": str(iter_dir / "baked_textures"),
            "output_blend": str(cand_blend),
            "output_glb": str(cand_glb),
            "execution_log": str(exec_log),
            "actions": actions,
            "shell_contract": shell_contract or {},
            "max_p95_dev_pct": criteria.max_p95_dev_pct,
        }
        if pymeshlab_operation:
            plan_payload["pymeshlab_python"] = pymeshlab.python
        plan_json.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        tk_ok, tk_log = _run_blender(
            blender_exe,
            TOOLKIT_SCRIPT,
            ["--plan", str(plan_json)],
            iter_dir / "toolkit.log",
            timeout=600,
        )

        verdict = None
        val_json = iter_dir / "validation_verdict.json"
        if tk_ok and cand_glb.is_file():
            val_ok, _ = _run_blender(
                blender_exe,
                VALIDATOR_SCRIPT,
                [
                    "--candidate-glb",
                    str(cand_glb),
                    "--candidate-blend",
                    str(cand_blend),
                    "--reference-glb",
                    str(reference_copy),
                    "--criteria-json",
                    str(criteria_path),
                    "--renders-dir",
                    str(iter_dir / "qc_renders"),
                    "--out-json",
                    str(val_json),
                ],
                iter_dir / "validator.log",
                timeout=300,
            )
            if val_ok and val_json.is_file():
                verdict = json.loads(val_json.read_text(encoding="utf-8-sig"))

        cur_score = float(verdict.get("score", 0.0)) if verdict else 0.0
        if verdict and (best_attempt is None or cur_score > best_attempt["score"]):
            best_attempt = {
                "iteration": it,
                "strategy_id": strat_id,
                "score": cur_score,
                "verdict": verdict,
                "actions": actions,
                "cand_blend": str(cand_blend),
                "cand_glb": str(cand_glb),
            }

        if verdict and verdict.get("passed"):
            final_blend = deliverables_dir / f"{input_asset.stem}_retopo.blend"
            final_glb = deliverables_dir / f"{input_asset.stem}_retopo.glb"
            shutil.copy2(cand_blend, final_blend)
            shutil.copy2(cand_glb, final_glb)
            qc_dst = deliverables_dir / "qc_renders"
            if (iter_dir / "qc_renders").is_dir():
                if qc_dst.exists():
                    shutil.rmtree(qc_dst)
                shutil.copytree(iter_dir / "qc_renders", qc_dst)
            if (iter_dir / "baked_textures").is_dir():
                shutil.copytree(iter_dir / "baked_textures", deliverables_dir / "textures", dirs_exist_ok=True)

            triage = triage_and_persist_outcome(
                "SUCCESS",
                input_asset.name,
                work_dir,
                kdir,
                baseline_metrics,
                attempt_history,
                best_attempt,
                winning_actions=actions,
            )
            result = {
                "status": "delivered",
                "outcome_kind": "SUCCESS",
                "strategy_id": strat_id,
                "iterations_used": it,
                "max_iterations": max_iterations,
                "deliverables": {
                    "blend": str(final_blend),
                    "glb": str(final_glb),
                    "qc_renders_dir": str(qc_dst),
                },
                "final_verdict": verdict,
                "triage": triage,
                "baseline_metrics": baseline_metrics,
                "attempt_history": attempt_history
                + [{"iteration": it, "strategy_id": strat_id, "passed": True, "score": cur_score}],
                "finished_at": now_iso(),
            }
            (work_dir / "delivery_scorecard.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            record_experiment(
                kdir,
                task_type="retopo",
                asset=input_asset.name,
                work_dir=work_dir,
                autonomous_completed=True,
                human_interventions=0,
                iterations_used=it,
                elapsed_seconds=time.monotonic() - t_start,
                final_score=cur_score,
                passed=True,
                winning_strategy_id=strat_id,
                failure_reasons=[bf for a in attempt_history for bf in (a.get("verdict") or {}).get("blocking_failures", [])],
                seed=0,
                config_payload=experiment_config,
                reproducible_command=reproducible_command,
            )
            return result

        # Independent Validator rejected -> Rollback, record Failure Lesson, escalate route if < 50
        err_analysis = analyze_failure_and_rollback(
            pre_ckpt, active_baseline, strat_id, tk_ok, tk_log, verdict
        )
        route = get_strategy_route(strat_id)
        banned.add(strat_id)
        if cur_score < 50.0:
            banned_routes.add(route)

        _append_jsonl(
            kdir / "failure_lessons.jsonl",
            {
                "time": now_iso(),
                "asset": input_asset.name,
                "iteration": it,
                "strategy_id": strat_id,
                "route_family": route,
                "score": cur_score,
                "route_banned": cur_score < 50.0,
                "asset_tags": baseline_metrics["asset_tags"],
                **err_analysis,
            },
        )
        attempt_history.append(
            {
                "iteration": it,
                "strategy_id": strat_id,
                "route_family": route,
                "rationale": rationale,
                "passed": False,
                "score": cur_score,
                "error_analysis": err_analysis,
                "verdict": verdict,
            }
        )

    best_score = best_attempt["score"] if best_attempt else 0.0
    outcome_kind = "PARTIAL_SUCCESS" if best_score >= 50.0 else "TOTAL_FAILURE"
    triage = triage_and_persist_outcome(
        outcome_kind,
        input_asset.name,
        work_dir,
        kdir,
        baseline_metrics,
        attempt_history,
        best_attempt,
    )
    fail_report = {
        "status": "partial_success" if outcome_kind == "PARTIAL_SUCCESS" else "total_failure_route_pivot",
        "outcome_kind": outcome_kind,
        "iterations_used": max_iterations,
        "max_iterations": max_iterations,
        "triage": triage,
        "baseline_metrics": baseline_metrics,
        "attempt_history": attempt_history,
        "finished_at": now_iso(),
    }
    (work_dir / "failure_report.json").write_text(
        json.dumps(fail_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record_experiment(
        kdir,
        task_type="retopo",
        asset=input_asset.name,
        work_dir=work_dir,
        autonomous_completed=False,
        human_interventions=0,
        iterations_used=max_iterations,
        elapsed_seconds=time.monotonic() - t_start,
        final_score=best_score,
        passed=False,
        winning_strategy_id=best_attempt.get("strategy_id", "") if best_attempt else "",
        failure_reasons=[bf for a in attempt_history for bf in (a.get("verdict") or {}).get("blocking_failures", [])],
        seed=0,
        config_payload=experiment_config,
        reproducible_command=reproducible_command,
    )
    return fail_report
