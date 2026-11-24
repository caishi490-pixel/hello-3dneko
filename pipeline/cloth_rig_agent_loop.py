"""Closed-loop Autonomous Clothing Fitting, Skeleton Weighting & 4-Pose Stress Test Engine."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipeline.config import DEFAULT_BLENDER_EXE, REPO_ROOT
from pipeline.experiment_ledger import record_experiment
from pipeline.image_tools import make_contact_sheet
from pipeline.knowledge_recall import apply_priors_to_candidates, load_strategy_priors
from pipeline.topology_agent_loop import (
    DEFAULT_KNOWLEDGE_DIR,
    _append_jsonl,
    _run_blender,
    now_iso,
)

CLOTH_INSPECTOR_SCRIPT = REPO_ROOT / "blender_scripts" / "cloth_rig_inspector.py"
CLOTH_TOOLKIT_SCRIPT = REPO_ROOT / "blender_scripts" / "cloth_rig_toolkit.py"
CLOTH_VALIDATOR_SCRIPT = REPO_ROOT / "blender_scripts" / "cloth_rig_validator.py"

CLOTH_STRATEGY_ROUTES: dict[str, str] = {
    "surface_polyinterp_clean_fit": "surface_transfer_clean_route",
    "inflated_proxy_capsule_transfer": "proxy_hull_skirt_route",
    "wide_margin_wrap_nearest_smooth": "relaxed_margin_nearest_route",
    "double_wrap_high_clearance_proxy": "cage_boost_high_clearance_route",
}


@dataclass
class ClothRigCriteria:
    max_rest_clipping_pct: float = 1.5
    max_pose_clipping_pct: float = 3.5
    max_penetration_mm: float = 14.0
    max_collapsed_edges_pct: float = 2.0
    max_overstretched_edges_pct: float = 3.5
    max_single_edge_stretch_ratio: float = 3.4
    max_arm_skirt_crosstalk_pct: float = 0.5
    max_cross_leg_crosstalk_pct: float = 1.0
    max_pendulum_stretch_ratio: float = 1.18


def get_cloth_strategy_route(strategy_id: str) -> str:
    return CLOTH_STRATEGY_ROUTES.get(strategy_id, "unknown_cloth_route")


def agent_decide_cloth_rig_plan(
    inspection: dict[str, Any],
    attempt_history: list[dict[str, Any]],
    banned_strategies: set[str],
    criteria: ClothRigCriteria,
    *,
    banned_routes: set[str] | None = None,
    knowledge_dir: Path | None = None,
) -> tuple[str, list[dict[str, Any]], str]:
    """Select next body-garment fitting + weight-transfer + anomaly-repair strategy based on prior failures."""
    last_classes: set[str] = set()
    if attempt_history:
        last_classes = set(attempt_history[-1].get("error_analysis", {}).get("failure_classes", []))

    garment_type = inspection.get("garment_analysis", {}).get("garment_type", "full_dress")
    is_corset = garment_type == "waist_corset_belt"

    clip_issue = bool({"body_clipping", "pose_body_clipping", "deep_penetration"} & last_classes)
    stretch_or_leg_issue = bool({"mesh_overstretch", "mesh_collapse", "cross_leg_crosstalk"} & last_classes)

    margin_1 = 0.015 if clip_issue else 0.011
    inward_1 = 0.24 if is_corset else 0.0

    candidates: list[tuple[str, list[dict[str, Any]], str]] = [
        (
            "surface_polyinterp_clean_fit",
            [
                {
                    "action": "match_pose_and_fit",
                    "params": {
                        "clearance_margin_m": margin_1,
                        "wrap_smooth_passes": 5,
                        "inward_pull": inward_1,
                    },
                },
                {
                    "action": "bind_and_transfer_weights",
                    "params": {
                        "method": "POLYINTERP_NEAREST",
                        "preserve_volume": True,
                        "use_proxy_hull": False,
                    },
                },
                {
                    "action": "repair_weight_anomalies",
                    "params": {
                        "purge_arm_skirt_crosstalk": True,
                        "spatial_sleeve_field": True,
                        "skirt_analytical_field": False,
                        "blend_skirt_leg_bridge": True,
                        "skirt_hips_floor": 0.30,
                        "cross_leg_blend_band": 0.065,
                        "stiffen_pendulum": True,
                        "weight_smooth_passes": 3,
                    },
                },
            ],
            "Pose-matched A-Pose rest bake + outward clearance wrap -> barycentric POLYINTERP_NEAREST weight transfer -> arm/skirt & sleeveless purge + sleeve projection + pendulum rigidification.",
        ),
        (
            "inflated_proxy_capsule_transfer",
            [
                {
                    "action": "match_pose_and_fit",
                    "params": {
                        "clearance_margin_m": 0.017 if clip_issue else 0.014,
                        "wrap_smooth_passes": 6,
                        "inward_pull": 0.20 if is_corset else 0.0,
                    },
                },
                {
                    "action": "bind_and_transfer_weights",
                    "params": {
                        "method": "POLYINTERP_NEAREST",
                        "preserve_volume": True,
                        "use_proxy_hull": True,
                        "proxy_inflate_m": 0.018,
                    },
                },
                {
                    "action": "repair_weight_anomalies",
                    "params": {
                        "purge_arm_skirt_crosstalk": True,
                        "spatial_sleeve_field": True,
                        "skirt_analytical_field": True,
                        "blend_skirt_leg_bridge": True,
                        "skirt_hips_floor": 0.22,
                        "cross_leg_blend_band": 0.095,
                        "stiffen_pendulum": True,
                        "weight_smooth_passes": 3,
                    },
                },
            ],
            "Inflated proxy hull + continuous spatial sleeve & analytical multi-layer skirt weight field to cure squat/stride leg clipping and ruffle tearing.",
        ),
        (
            "wide_margin_wrap_nearest_smooth",
            [
                {
                    "action": "match_pose_and_fit",
                    "params": {
                        "clearance_margin_m": 0.018,
                        "wrap_smooth_passes": 7,
                        "inward_pull": 0.16 if is_corset else 0.0,
                    },
                },
                {
                    "action": "bind_and_transfer_weights",
                    "params": {
                        "method": "NEAREST",
                        "preserve_volume": True,
                        "use_proxy_hull": True,
                        "proxy_inflate_m": 0.020,
                    },
                },
                {
                    "action": "repair_weight_anomalies",
                    "params": {
                        "purge_arm_skirt_crosstalk": True,
                        "spatial_sleeve_field": True,
                        "skirt_analytical_field": True,
                        "blend_skirt_leg_bridge": True,
                        "skirt_hips_floor": 0.22,
                        "cross_leg_blend_band": 0.10,
                        "stiffen_pendulum": True,
                        "weight_smooth_passes": 4,
                    },
                },
            ],
            "Fallback route switching to NEAREST projection from inflated hull + 18mm wrap margin + analytical skirt field + 4-pass weight smoothing.",
        ),
        (
            "double_wrap_high_clearance_proxy",
            [
                {
                    "action": "match_pose_and_fit",
                    "params": {
                        "clearance_margin_m": 0.020,
                        "wrap_smooth_passes": 7,
                        "inward_pull": 0.12 if is_corset else 0.0,
                    },
                },
                {
                    "action": "extra_clearance_wrap",
                    "params": {
                        "margin_m": 0.021,
                        "smooth_passes": 7,
                        "inward_pull": 0.0,
                    },
                },
                {
                    "action": "bind_and_transfer_weights",
                    "params": {
                        "method": "POLYINTERP_NEAREST",
                        "preserve_volume": True,
                        "use_proxy_hull": True,
                        "proxy_inflate_m": 0.022,
                    },
                },
                {
                    "action": "repair_weight_anomalies",
                    "params": {
                        "purge_arm_skirt_crosstalk": True,
                        "spatial_sleeve_field": True,
                        "skirt_analytical_field": True,
                        "blend_skirt_leg_bridge": True,
                        "skirt_hips_floor": 0.24,
                        "cross_leg_blend_band": 0.105,
                        "stiffen_pendulum": True,
                        "weight_smooth_passes": 5,
                    },
                },
            ],
            "High-clearance dual wrap + 22mm proxy transfer + analytical multi-layer skirt/sleeve fields.",
        ),
    ]

    if knowledge_dir is not None and not attempt_history:
        priors = load_strategy_priors(knowledge_dir, "cloth_rig", asset_tags={garment_type})
        candidates = apply_priors_to_candidates(candidates, priors)

    b_routes = banned_routes or set()
    # If previous failure involved pose clipping, skirt stretch, or spikes, promote proxy_hull_skirt_route directly!
    if clip_issue or stretch_or_leg_issue or ("skinning_spike" in last_classes):
        candidates.sort(key=lambda c: 0 if get_cloth_strategy_route(c[0]) == "proxy_hull_skirt_route" else 1)

    for sid, actions, rationale in candidates:
        route = get_cloth_strategy_route(sid)
        if sid not in banned_strategies and route not in b_routes:
            return sid, actions, rationale
    for sid, actions, rationale in candidates:
        if sid not in banned_strategies:
            return sid, actions, rationale
    return candidates[-1]


def run_closed_loop_cloth_rig(
    body_asset: Path,
    garment_asset: Path,
    work_dir: Path,
    *,
    criteria: ClothRigCriteria | None = None,
    max_iterations: int = 4,
    blender_exe: str = DEFAULT_BLENDER_EXE,
    knowledge_dir: Path | None = None,
    seed: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run closed-loop Body+Garment Matching -> Rig/Weight Transfer -> 4-Pose Stress Validator -> Rollback/Repair."""
    t0 = time.monotonic()
    crit = criteria or ClothRigCriteria()
    kdir = knowledge_dir or DEFAULT_KNOWLEDGE_DIR
    kdir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    snap_dir = work_dir / "checkpoints"
    snap_dir.mkdir(parents=True, exist_ok=True)
    raw_body_copy = snap_dir / f"00_raw_body{body_asset.suffix}"
    raw_garm_copy = snap_dir / f"00_raw_garment{garment_asset.suffix}"
    if body_asset.is_file() and not raw_body_copy.exists():
        shutil.copy2(body_asset, raw_body_copy)
    if garment_asset.is_file() and not raw_garm_copy.exists():
        shutil.copy2(garment_asset, raw_garm_copy)

    crit_path = work_dir / "cloth_rig_criteria.json"
    crit_path.write_text(json.dumps(asdict(crit), ensure_ascii=False, indent=2), encoding="utf-8")

    inspect_json = work_dir / "01_matching_inspection.json"
    if dry_run:
        dummy_insp = {
            "body_path": str(body_asset),
            "garment_path": str(garment_asset),
            "garment_analysis": {"garment_type": "full_sleeved_dress", "has_sleeves": True},
        }
        inspect_json.write_text(json.dumps(dummy_insp, indent=2), encoding="utf-8")
        sid, actions, rationale = agent_decide_cloth_rig_plan(dummy_insp, [], set(), crit)
        plan = {
            "status": "dry_run",
            "strategy_id": sid,
            "route_family": get_cloth_strategy_route(sid),
            "rationale": rationale,
            "actions": actions,
        }
        (work_dir / "dry_run_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return plan

    ok, out = _run_blender(
        blender_exe,
        CLOTH_INSPECTOR_SCRIPT,
        ["--body", str(body_asset), "--garment", str(garment_asset), "--out-json", str(inspect_json)],
        work_dir / "01_inspector.log",
    )
    if not ok or not inspect_json.is_file():
        raise RuntimeError(f"Cloth-rig inspector failed:\n{out[-1200:]}")
    inspection = json.loads(inspect_json.read_text(encoding="utf-8"))

    attempt_history: list[dict[str, Any]] = []
    banned_strategies: set[str] = set()
    banned_routes: set[str] = set()
    route_fail_counts: dict[str, int] = {}
    delivered: dict[str, Any] | None = None

    for it in range(1, max_iterations + 1):
        sid, actions, rationale = agent_decide_cloth_rig_plan(
            inspection,
            attempt_history,
            banned_strategies,
            crit,
            banned_routes=banned_routes,
            knowledge_dir=kdir,
        )
        route_family = get_cloth_strategy_route(sid)
        it_dir = work_dir / "attempts" / f"iter_{it:02d}_{sid}"
        it_dir.mkdir(parents=True, exist_ok=True)
        act_path = it_dir / "actions.json"
        act_path.write_text(json.dumps(actions, ensure_ascii=False, indent=2), encoding="utf-8")

        cand_blend = it_dir / f"{garment_asset.stem}_fitted_rigged.blend"
        cand_glb = it_dir / f"{garment_asset.stem}_fitted_rigged.glb"
        toolkit_json = it_dir / "toolkit_report.json"

        ok_tk, out_tk = _run_blender(
            blender_exe,
            CLOTH_TOOLKIT_SCRIPT,
            [
                "--body", str(body_asset),
                "--garment", str(garment_asset),
                "--actions-json", str(act_path),
                "--out-blend", str(cand_blend),
                "--out-glb", str(cand_glb),
                "--report-json", str(toolkit_json),
            ],
            it_dir / "toolkit.log",
        )
        if not ok_tk or not cand_blend.is_file():
            banned_strategies.add(sid)
            attempt_history.append({
                "iteration": it,
                "strategy_id": sid,
                "route_family": route_family,
                "passed": False,
                "score": 0.0,
                "error_analysis": {"failure_classes": ["toolkit_crash"], "details": out_tk[-600:]},
            })
            continue

        val_json = it_dir / "validation_report.json"
        renders_dir = it_dir / "qc_renders"
        ok_val, out_val = _run_blender(
            blender_exe,
            CLOTH_VALIDATOR_SCRIPT,
            [
                "--candidate-blend", str(cand_blend),
                "--criteria-json", str(crit_path),
                "--renders-dir", str(renders_dir),
                "--out-json", str(val_json),
            ],
            it_dir / "validator.log",
        )
        if not ok_val or not val_json.is_file():
            banned_strategies.add(sid)
            attempt_history.append({
                "iteration": it,
                "strategy_id": sid,
                "route_family": route_family,
                "passed": False,
                "score": 0.0,
                "error_analysis": {"failure_classes": ["validator_crash"], "details": out_val[-600:]},
            })
            continue

        val_rep = json.loads(val_json.read_text(encoding="utf-8"))
        ev = val_rep["evaluation"]
        pose_imgs = [Path(p) for p in val_rep["metrics"].get("renders", {}).values() if Path(p).is_file()]
        sheet_path = it_dir / "pose_stress_contact_sheet.png"
        if pose_imgs:
            make_contact_sheet(pose_imgs, sheet_path, cols=3, cell_size=512)

        attempt_row = {
            "iteration": it,
            "strategy_id": sid,
            "route_family": route_family,
            "rationale": rationale,
            "passed": bool(ev["passed"]),
            "score": float(ev["score"]),
            "summary": ev["summary"],
            "error_analysis": {
                "blocking_failures": ev["blocking_failures"],
                "failure_classes": ev["failure_classes"],
            },
            "pose_stress_contact_sheet": str(sheet_path),
        }
        attempt_history.append(attempt_row)

        if ev["passed"]:
            deliv_dir = work_dir / "deliverables"
            deliv_dir.mkdir(parents=True, exist_ok=True)
            final_blend = deliv_dir / cand_blend.name
            final_glb = deliv_dir / cand_glb.name
            shutil.copy2(cand_blend, final_blend)
            shutil.copy2(cand_glb, final_glb)
            shutil.copy2(val_json, deliv_dir / "validation_report.json")
            if sheet_path.is_file():
                shutil.copy2(sheet_path, deliv_dir / "pose_stress_contact_sheet.png")
            delivered = {
                "status": "delivered",
                "iterations_used": it,
                "winning_strategy_id": sid,
                "route_family": route_family,
                "final_score": ev["score"],
                "summary": ev["summary"],
                "deliverables": {
                    "blend": str(final_blend),
                    "glb": str(final_glb),
                    "validation_report": str(deliv_dir / "validation_report.json"),
                    "pose_stress_contact_sheet": str(deliv_dir / "pose_stress_contact_sheet.png"),
                    "qc_renders": val_rep["metrics"].get("renders", {}),
                },
            }
            break

        # Rollback mindset: do not keep failed asset as deliverable; ban failing strategy & switch route
        banned_strategies.add(sid)
        route_fail_counts[route_family] = route_fail_counts.get(route_family, 0) + 1
        if route_fail_counts[route_family] >= 1:
            banned_routes.add(route_family)
        _append_jsonl(
            kdir / "failure_lessons.jsonl",
            {
                "time": now_iso(),
                "task_type": "cloth_rig",
                "asset": garment_asset.name,
                "strategy_id": sid,
                "route_family": route_family,
                "failure_classes": ev["failure_classes"],
                "blocking_failures": ev["blocking_failures"],
            },
        )

    elapsed = round(time.monotonic() - t0, 2)
    passed = delivered is not None
    final_score = delivered["final_score"] if delivered else max((a["score"] for a in attempt_history), default=0.0)
    win_sid = delivered["winning_strategy_id"] if delivered else ""
    fail_reasons = [] if passed else attempt_history[-1]["error_analysis"].get("blocking_failures", [])

    record_experiment(
        kdir,
        task_type="cloth_rig",
        asset=garment_asset.name,
        work_dir=work_dir,
        autonomous_completed=passed,
        human_interventions=0,
        iterations_used=len(attempt_history),
        elapsed_seconds=elapsed,
        final_score=final_score,
        passed=passed,
        winning_strategy_id=win_sid,
        failure_reasons=fail_reasons,
        seed=seed,
        config_payload={"body": str(body_asset), "garment": str(garment_asset), "criteria": asdict(crit)},
        reproducible_command=(
            f'python run_pipeline.py cloth-rig --body "{body_asset}" --garment "{garment_asset}" --out "{work_dir}" --seed {seed}'
        ),
    )

    if passed:
        _append_jsonl(
            kdir / "learned_recipes.jsonl",
            {
                "time": now_iso(),
                "task_type": "cloth_rig",
                "asset": garment_asset.name,
                "strategy_id": win_sid,
                "route_family": delivered["route_family"],
                "score": final_score,
                "summary": delivered["summary"],
            },
        )

    final_report = {
        "status": "delivered" if passed else "failed_budget_exhausted",
        "elapsed_seconds": elapsed,
        "iterations_used": len(attempt_history),
        "attempts": attempt_history,
        "result": delivered,
    }
    (work_dir / "cloth_rig_scorecard.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_report
