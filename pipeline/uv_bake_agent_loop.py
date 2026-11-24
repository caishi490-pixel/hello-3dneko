"""Closed-loop Autonomous UV & Baking v0.1 Engine with Self-Repair, Rollback, and Experiment Ledger."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipeline.config import DEFAULT_BLENDER_EXE, REPO_ROOT
from pipeline.experiment_ledger import record_experiment
from pipeline.knowledge_recall import apply_priors_to_candidates, load_strategy_priors
from pipeline.topology_agent_loop import (
    INSPECTOR_SCRIPT,
    _append_jsonl,
    _run_blender,
    now_iso,
)

UV_TOOLKIT_SCRIPT = REPO_ROOT / "blender_scripts" / "uv_bake_toolkit.py"
UV_VALIDATOR_SCRIPT = REPO_ROOT / "blender_scripts" / "uv_bake_validator.py"

UV_STRATEGY_ROUTES: dict[str, str] = {
    "seam_angle_conformal_tight_bake": "seam_unwrap_route",
    "seam_conformal_smooth_cage_bake": "seam_unwrap_route",
    "hybrid_seams_conformal_tight_bake": "hybrid_unwrap_route",
    "smart_project_packed_tight_bake": "smart_project_route",
    "smart_project_safe_margin_cage_bake": "smart_project_route",
}


@dataclass
class UVBakeCriteria:
    max_mean_dev_pct: float = 0.8
    max_bbox_drift_pct: float = 2.5
    max_overlap_pct: float = 0.8
    max_p95_stretch: float = 4.5
    max_mean_stretch: float = 2.2
    max_zero_area_tris: int = 4
    min_packing_coverage_pct: float = 30.0
    min_bake_coverage: float = 0.96
    max_inverted_normal_pct: float = 0.5
    min_ao_std: float = 0.005
    max_ao_blackout_pct: float = 10.0
    max_diffuse_black_leak_pct: float = 12.0


def get_uv_strategy_route(strategy_id: str) -> str:
    return UV_STRATEGY_ROUTES.get(strategy_id, "unknown_uv_route")


def agent_decide_uv_bake_plan(
    baseline_low_metrics: dict[str, Any],
    attempt_history: list[dict[str, Any]],
    banned_strategies: set[str],
    criteria: UVBakeCriteria,
    *,
    banned_routes: set[str] | None = None,
    texture_size: int = 2048,
    knowledge_dir: Path | None = None,
) -> tuple[str, list[dict[str, Any]], str]:
    """Select next seam/UV + PBR/AO bake repair plan dynamically from previous failure diagnostics."""
    meshes = baseline_low_metrics.get("meshes", [])
    dims = meshes[0].get("dims", [1.0, 1.0, 1.0]) if meshes else [1.0, 1.0, 1.0]
    diag = (dims[0] ** 2 + dims[1] ** 2 + dims[2] ** 2) ** 0.5 or 1.0

    last_classes: set[str] = set()
    if attempt_history:
        last_classes = set(attempt_history[-1].get("error_analysis", {}).get("failure_classes", []))

    tight_cage = round(max(0.002, min(0.005, diag * 0.0018)), 5)
    tight_ray = round(max(0.012, min(0.025, diag * 0.010)), 5)
    wide_cage = round(max(0.004, min(0.010, diag * 0.0035)), 5)
    wide_ray = round(max(0.025, min(0.055, diag * 0.024)), 5)

    use_cage = bool({"bake_blackout", "bake_normal_flip", "bake_low_coverage"} & last_classes)
    cur_cage = wide_cage if "bake_low_coverage" in last_classes else tight_cage
    cur_ray = wide_ray if "bake_low_coverage" in last_classes else tight_ray
    ao_dist = round(max(0.045, min(0.10, diag * 0.025)), 3)

    candidates: list[tuple[str, list[dict[str, Any]], str]] = []

    # 1. Seam-based unwrap (ANGLE_BASED, 42° crease + meridian seam cut on closed shells + collapsed-island rescue)
    if "uv_overlap" not in last_classes and "uv_stretch_high" not in last_classes:
        candidates.append(
            (
                "seam_angle_conformal_tight_bake",
                [
                    {
                        "action": "prepare_seams_and_unwrap",
                        "params": {
                            "mode": "seam_unwrap",
                            "unwrap_method": "ANGLE_BASED",
                            "seam_angle_deg": 42.0,
                            "island_margin": 0.0025,
                            "average_scale": True,
                            "cut_closed_shells": True,
                        },
                    },
                    {
                        "action": "bake_pbr_maps",
                        "params": {
                            "texture_size": texture_size,
                            "cage_extrusion": cur_cage,
                            "max_ray_distance": cur_ray,
                            "ao_distance": ao_dist,
                            "use_cage": use_cage,
                            "bake_samples": 16,
                            "normal_strength": 0.55,
                        },
                    },
                ],
                "Generate automatic crease + closed-shell meridian seams -> ANGLE_BASED unwrap (with collapsed-island rescue) + compact 0.0025 pack -> Cycles local-distance AO/Normal/BaseColor/ORM bake.",
            )
        )

    # 2. Hybrid Smart-Seams + CONFORMAL Unwrap + Problem-Face Rescue (crushes layered pleats overlap/stretch cleanly!)
    candidates.append(
        (
            "hybrid_seams_conformal_tight_bake",
            [
                {
                    "action": "prepare_seams_and_unwrap",
                    "params": {
                        "mode": "hybrid_conformal",
                        "smart_angle_deg": 64.0,
                        "island_margin": 0.0014,
                        "average_scale": True,
                    },
                },
                {
                    "action": "bake_pbr_maps",
                    "params": {
                        "texture_size": texture_size,
                        "cage_extrusion": cur_cage,
                        "max_ray_distance": cur_ray,
                        "ao_distance": ao_dist,
                        "use_cage": use_cage,
                        "bake_samples": 16,
                        "normal_strength": 0.55,
                    },
                },
            ],
            "Hybrid smart-island seams (64 deg) -> CONFORMAL unwrap + collapsed/flipped face rescue + compact 0.0014 pack -> distance-clamped AO+PBR bake.",
        )
    )

    # 3. Non-overlapping Smart UV Project (85 deg large sheets, compact 0.002 margin) + Calibrated Tight/Cage Bake
    candidates.append(
        (
            "smart_project_packed_tight_bake",
            [
                {
                    "action": "prepare_seams_and_unwrap",
                    "params": {
                        "mode": "smart_project",
                        "smart_angle_deg": 85.0,
                        "island_margin": 0.002,
                        "average_scale": True,
                    },
                },
                {
                    "action": "bake_pbr_maps",
                    "params": {
                        "texture_size": texture_size,
                        "cage_extrusion": tight_cage,
                        "max_ray_distance": cur_ray,
                        "ao_distance": ao_dist,
                        "use_cage": use_cage,
                        "bake_samples": 16,
                        "normal_strength": 0.55,
                    },
                },
            ],
            "Zero-overlap Smart UV Project (85 deg large islands + compact 0.002 packing) + tight calibrated PBR+AO Cycles bake.",
        )
    )

    # 4. Refined Seams (32 deg CONFORMAL) + Displaced Smooth Bake Cage
    candidates.append(
        (
            "seam_conformal_smooth_cage_bake",
            [
                {
                    "action": "prepare_seams_and_unwrap",
                    "params": {
                        "mode": "seam_unwrap",
                        "unwrap_method": "CONFORMAL",
                        "seam_angle_deg": 32.0,
                        "island_margin": 0.003,
                        "average_scale": True,
                        "cut_closed_shells": True,
                    },
                },
                {
                    "action": "bake_pbr_maps",
                    "params": {
                        "texture_size": texture_size,
                        "cage_extrusion": tight_cage,
                        "max_ray_distance": max(cur_ray, tight_ray * 1.4),
                        "ao_distance": ao_dist,
                        "use_cage": True,
                        "bake_samples": 16,
                        "normal_strength": 0.50,
                    },
                },
            ],
            "Increase seam cuts (32 deg CONFORMAL + problem-face rescue) + tight 0.003 pack -> switch to displaced smooth BakeCage.",
        )
    )

    # 5. Conservative Smart Project (62 deg, 0.002 margin + wider displaced cage fallback)
    candidates.append(
        (
            "smart_project_safe_margin_cage_bake",
            [
                {
                    "action": "prepare_seams_and_unwrap",
                    "params": {
                        "mode": "smart_project",
                        "smart_angle_deg": 62.0,
                        "island_margin": 0.002,
                        "average_scale": True,
                    },
                },
                {
                    "action": "bake_pbr_maps",
                    "params": {
                        "texture_size": texture_size,
                        "cage_extrusion": wide_cage,
                        "max_ray_distance": wide_ray,
                        "ao_distance": ao_dist,
                        "use_cage": True,
                        "bake_samples": 16,
                        "normal_strength": 0.50,
                    },
                },
            ],
            "Safe narrower-angle Smart Project + displaced cage and ray range recovery fallback.",
        )
    )

    if knowledge_dir is not None and not attempt_history:
        priors = load_strategy_priors(knowledge_dir, "uv_bake")
        candidates = apply_priors_to_candidates(candidates, priors)

    banned_r = banned_routes or set()
    for strat_id, actions, rationale in candidates:
        if strat_id not in banned_strategies and get_uv_strategy_route(strat_id) not in banned_r:
            return strat_id, actions, rationale

    raise RuntimeError("No unbanned UV/Baking strategies remaining within autonomous budget.")


def analyze_uv_failure_and_rollback(
    pre_checkpoint: Path,
    active_low: Path,
    strategy_id: str,
    toolkit_ok: bool,
    toolkit_log: str,
    verdict: dict[str, Any] | None,
    val_log: str = "",
) -> dict[str, Any]:
    """Restore clean pre-attempt low-poly checkpoint and classify exact UV/Bake blockers."""
    shutil.copy2(pre_checkpoint, active_low)
    failure_classes: list[str] = []
    details: list[str] = []

    if not toolkit_ok:
        failure_classes.append("toolkit_runtime_error")
        details.append(toolkit_log[-500:])
    elif verdict is None:
        failure_classes.append("validator_runtime_error")
        details.append(val_log[-500:] if val_log else "Validator produced no verdict JSON.")
    else:
        for bf in verdict.get("blocking_failures", []):
            details.append(bf)
            low_bf = bf.lower()
            if "uvoverlap" in low_bf or "overlap" in low_bf:
                failure_classes.append("uv_overlap")
            elif "uvstretch" in low_bf or "zeroarea" in low_bf or "uvbounds" in low_bf:
                failure_classes.append("uv_stretch_high")
            elif "uvpacking" in low_bf:
                failure_classes.append("uv_packing_low")
            elif "bakecoverage" in low_bf or "bakemissing" in low_bf:
                failure_classes.append("bake_low_coverage")
            elif "normalflip" in low_bf:
                failure_classes.append("bake_normal_flip")
            elif "blackout" in low_bf or "black_leak" in low_bf or "flat" in low_bf:
                failure_classes.append("bake_blackout")
            elif "spatialcorrespondence" in low_bf:
                failure_classes.append("spatial_mismatch")
        if not failure_classes:
            failure_classes.append("uv_bake_validation_rejected")

    return {
        "rolled_back_to": str(pre_checkpoint),
        "banned_strategy": strategy_id,
        "route_family": get_uv_strategy_route(strategy_id),
        "failure_classes": sorted(set(failure_classes)),
        "details": details,
    }


def _build_auto_repair_log(attempt_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    log = []
    for a in attempt_history:
        log.append(
            {
                "iteration": a.get("iteration"),
                "strategy_id": a.get("strategy_id"),
                "route_family": a.get("route_family"),
                "rationale": a.get("rationale"),
                "passed": a.get("passed"),
                "score": a.get("score"),
                "failure_classes": a.get("error_analysis", {}).get("failure_classes", []),
                "blocking_failures": (a.get("verdict") or {}).get("blocking_failures", []),
                "rolled_back_to": a.get("error_analysis", {}).get("rolled_back_to"),
            }
        )
    return log


def run_closed_loop_uv_bake(
    *,
    high_poly: Path,
    low_poly: Path,
    work_dir: Path,
    criteria: UVBakeCriteria | None = None,
    max_iterations: int = 4,
    texture_size: int = 2048,
    seed: int = 42,
    knowledge_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Autonomous UV & Baking v0.1 without mutating original source assets."""
    t_start = time.monotonic()
    crit = criteria or UVBakeCriteria()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    kdir = (knowledge_dir or (REPO_ROOT / "knowledge")).resolve()
    kdir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = work_dir / "checkpoints"
    attempts_dir = work_dir / "attempts"
    deliverables_dir = work_dir / "deliverables"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    deliverables_dir.mkdir(parents=True, exist_ok=True)

    criteria_path = work_dir / "uv_bake_criteria.json"
    criteria_path.write_text(json.dumps(asdict(crit), ensure_ascii=False, indent=2), encoding="utf-8")

    # Never mutate originals: duplicate both inputs into checkpoint vault
    high_ref = checkpoints_dir / f"00_high_reference{high_poly.suffix.lower() or '.glb'}"
    low_ref = checkpoints_dir / f"00_low_reference{low_poly.suffix.lower() or '.glb'}"
    if high_poly.is_file():
        shutil.copy2(high_poly, high_ref)
    if low_poly.is_file():
        shutil.copy2(low_poly, low_ref)

    if dry_run:
        sample_plan = {
            "mode": "dry_run",
            "high_poly": str(high_poly),
            "low_poly": str(low_poly),
            "work_dir": str(work_dir),
            "criteria": asdict(crit),
            "deliverables_planned": [
                "final .blend",
                "final .glb",
                "uv_layout.png",
                "baked_normal.png",
                "baked_ao.png",
                "baked_diffuse.png",
                "baked_orm.png",
                "uv_inspection_report.json",
                "bake_quality_report.json",
                "auto_repair_log.json",
            ],
        }
        return {"status": "dry_run", **sample_plan}

    if not high_poly.is_file():
        raise FileNotFoundError(f"Missing high-poly model: {high_poly}")
    if not low_poly.is_file():
        raise FileNotFoundError(f"Missing low-poly retopo model: {low_poly}")

    blender_exe = DEFAULT_BLENDER_EXE

    # Step 1: Check integrity of both inputs
    high_det_json = work_dir / "01_high_integrity.json"
    low_det_json = work_dir / "01_low_integrity.json"
    ok_h, log_h = _run_blender(
        blender_exe,
        INSPECTOR_SCRIPT,
        ["--input", str(high_ref), "--out-json", str(high_det_json)],
        work_dir / "01_high_integrity.log",
    )
    if not ok_h:
        raise RuntimeError(f"High-poly integrity inspection failed:\n{log_h[-800:]}")
    ok_l, log_l = _run_blender(
        blender_exe,
        INSPECTOR_SCRIPT,
        ["--input", str(low_ref), "--out-json", str(low_det_json), "--reference", str(high_ref)],
        work_dir / "01_low_integrity.log",
    )
    if not ok_l:
        raise RuntimeError(f"Low-poly integrity inspection failed:\n{log_l[-800:]}")

    baseline_low = json.loads(low_det_json.read_text(encoding="utf-8-sig"))
    active_low = checkpoints_dir / f"active_low{low_poly.suffix.lower()}"
    shutil.copy2(low_ref, active_low)

    banned: set[str] = set()
    banned_routes: set[str] = set()
    attempt_history: list[dict[str, Any]] = []
    repro_cmd = (
        f"python run_pipeline.py uv-bake --high \"{high_poly}\" --low \"{low_poly}\" "
        f"--out \"{work_dir}\" --max-iters {max_iterations} --texture-size {texture_size} --seed {seed}"
    )
    cfg_payload = {
        "task_type": "uv_bake",
        "high_poly": high_poly.name,
        "low_poly": low_poly.name,
        "texture_size": texture_size,
        "seed": seed,
        "criteria": asdict(crit),
    }

    for it in range(1, max_iterations + 1):
        try:
            strat_id, actions, rationale = agent_decide_uv_bake_plan(
                baseline_low,
                attempt_history,
                banned,
                crit,
                banned_routes=banned_routes,
                texture_size=texture_size,
                knowledge_dir=kdir,
            )
        except RuntimeError as exc:
            elapsed = time.monotonic() - t_start
            repair_log = _build_auto_repair_log(attempt_history)
            (deliverables_dir / "auto_repair_log.json").write_text(
                json.dumps(repair_log, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            reasons = sorted(
                {bf for a in attempt_history for bf in (a.get("verdict") or {}).get("blocking_failures", [])}
            ) or [str(exc)]
            exp = record_experiment(
                kdir,
                task_type="uv_bake",
                asset=low_poly.name,
                work_dir=work_dir,
                autonomous_completed=False,
                human_interventions=0,
                iterations_used=max(1, len(attempt_history)),
                elapsed_seconds=elapsed,
                final_score=max((float(a.get("score", 0.0)) for a in attempt_history), default=0.0),
                passed=False,
                winning_strategy_id="",
                failure_reasons=reasons,
                seed=seed,
                config_payload=cfg_payload,
                reproducible_command=repro_cmd,
            )
            return {
                "status": "failed",
                "reason": str(exc),
                "iterations_used": len(attempt_history),
                "attempt_history": attempt_history,
                "experiment": exp,
            }

        iter_dir = attempts_dir / f"iter_{it:02d}_{strat_id}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        pre_ckpt = checkpoints_dir / f"ckpt_pre_iter_{it:02d}{active_low.suffix.lower()}"
        shutil.copy2(active_low, pre_ckpt)

        cand_blend = iter_dir / f"{low_poly.stem}_uv_baked.blend"
        cand_glb = iter_dir / f"{low_poly.stem}_uv_baked.glb"
        tex_dir = iter_dir / "textures"
        layout_dir = iter_dir / "uv_layouts"
        plan_json = iter_dir / "uv_bake_plan.json"
        exec_log = iter_dir / "toolkit_exec.json"

        plan_payload = {
            "iteration": it,
            "strategy_id": strat_id,
            "rationale": rationale,
            "low_poly_input": str(active_low),
            "high_poly_reference": str(high_ref),
            "textures_dir": str(tex_dir),
            "uv_layout_dir": str(layout_dir),
            "output_blend": str(cand_blend),
            "output_glb": str(cand_glb),
            "execution_log": str(exec_log),
            "actions": actions,
        }
        plan_json.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        tk_ok, tk_log = _run_blender(
            blender_exe,
            UV_TOOLKIT_SCRIPT,
            ["--plan", str(plan_json)],
            iter_dir / "toolkit.log",
            timeout=600,
        )

        verdict = None
        uv_rep = None
        bake_rep = None
        val_log = ""
        uv_rep_p = iter_dir / "uv_inspection_report.json"
        bake_rep_p = iter_dir / "bake_quality_report.json"
        verdict_p = iter_dir / "validation_verdict.json"

        if tk_ok and cand_glb.is_file():
            val_ok, val_log = _run_blender(
                blender_exe,
                UV_VALIDATOR_SCRIPT,
                [
                    "--candidate-glb",
                    str(cand_glb),
                    "--candidate-blend",
                    str(cand_blend),
                    "--high-reference",
                    str(high_ref),
                    "--textures-dir",
                    str(tex_dir),
                    "--criteria-json",
                    str(criteria_path),
                    "--renders-dir",
                    str(iter_dir / "qc_renders"),
                    "--uv-report-json",
                    str(uv_rep_p),
                    "--bake-report-json",
                    str(bake_rep_p),
                    "--out-json",
                    str(verdict_p),
                ],
                iter_dir / "validator.log",
                timeout=300,
            )
            if val_ok and verdict_p.is_file():
                verdict = json.loads(verdict_p.read_text(encoding="utf-8-sig"))
                if uv_rep_p.is_file():
                    uv_rep = json.loads(uv_rep_p.read_text(encoding="utf-8-sig"))
                if bake_rep_p.is_file():
                    bake_rep = json.loads(bake_rep_p.read_text(encoding="utf-8-sig"))

        cur_score = float(verdict.get("score", 0.0)) if verdict else 0.0
        if verdict and verdict.get("passed"):
            final_blend = deliverables_dir / f"{low_poly.stem}_uv_baked.blend"
            final_glb = deliverables_dir / f"{low_poly.stem}_uv_baked.glb"
            shutil.copy2(cand_blend, final_blend)
            shutil.copy2(cand_glb, final_glb)

            deliv_tex = deliverables_dir / "textures"
            if deliv_tex.exists():
                shutil.rmtree(deliv_tex)
            if tex_dir.is_dir():
                shutil.copytree(tex_dir, deliv_tex)

            layout_pngs = sorted(layout_dir.glob("*_uv_layout.png"))
            deliv_layout = deliverables_dir / "uv_layout.png"
            if layout_pngs:
                shutil.copy2(layout_pngs[0], deliv_layout)

            qc_dst = deliverables_dir / "qc_renders"
            if qc_dst.exists():
                shutil.rmtree(qc_dst)
            if (iter_dir / "qc_renders").is_dir():
                shutil.copytree(iter_dir / "qc_renders", qc_dst)

            shutil.copy2(uv_rep_p, deliverables_dir / "uv_inspection_report.json")
            shutil.copy2(bake_rep_p, deliverables_dir / "bake_quality_report.json")

            attempt_history.append(
                {
                    "iteration": it,
                    "strategy_id": strat_id,
                    "route_family": get_uv_strategy_route(strat_id),
                    "rationale": rationale,
                    "passed": True,
                    "score": cur_score,
                    "verdict": verdict,
                }
            )
            repair_log = _build_auto_repair_log(attempt_history)
            repair_log_p = deliverables_dir / "auto_repair_log.json"
            repair_log_p.write_text(json.dumps(repair_log, ensure_ascii=False, indent=2), encoding="utf-8")

            _append_jsonl(
                kdir / "learned_recipes.jsonl",
                {
                    "time": now_iso(),
                    "outcome_kind": "SUCCESS",
                    "task_type": "uv_bake",
                    "asset": low_poly.name,
                    "strategy_id": strat_id,
                    "route_family": get_uv_strategy_route(strat_id),
                    "score": cur_score,
                    "actions": actions,
                    "uv_metrics": uv_rep,
                    "bake_metrics": bake_rep,
                },
            )

            elapsed = time.monotonic() - t_start
            reworked_reasons = sorted(
                {
                    bf
                    for a in attempt_history[:-1]
                    for bf in (a.get("verdict") or {}).get("blocking_failures", [])
                }
            )
            exp = record_experiment(
                kdir,
                task_type="uv_bake",
                asset=low_poly.name,
                work_dir=work_dir,
                autonomous_completed=True,
                human_interventions=0,
                iterations_used=it,
                elapsed_seconds=elapsed,
                final_score=cur_score,
                passed=True,
                winning_strategy_id=strat_id,
                failure_reasons=reworked_reasons,
                seed=seed,
                config_payload=cfg_payload,
                reproducible_command=repro_cmd,
            )

            result = {
                "status": "delivered",
                "task_type": "uv_bake",
                "strategy_id": strat_id,
                "iterations_used": it,
                "max_iterations": max_iterations,
                "elapsed_seconds": round(elapsed, 2),
                "deliverables": {
                    "blend": str(final_blend),
                    "glb": str(final_glb),
                    "uv_layout": str(deliv_layout) if deliv_layout.is_file() else "",
                    "textures_dir": str(deliv_tex),
                    "uv_inspection_report": str(deliverables_dir / "uv_inspection_report.json"),
                    "bake_quality_report": str(deliverables_dir / "bake_quality_report.json"),
                    "auto_repair_log": str(repair_log_p),
                    "qc_renders_dir": str(qc_dst),
                },
                "final_verdict": verdict,
                "attempt_history": attempt_history,
                "experiment": exp,
                "finished_at": now_iso(),
            }
            (work_dir / "uv_bake_scorecard.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result

        err = analyze_uv_failure_and_rollback(pre_ckpt, active_low, strat_id, tk_ok, tk_log, verdict, val_log=val_log)
        banned.add(strat_id)
        if not tk_ok:
            banned_routes.add(get_uv_strategy_route(strat_id))

        _append_jsonl(
            kdir / "failure_lessons.jsonl",
            {
                "time": now_iso(),
                "task_type": "uv_bake",
                "asset": low_poly.name,
                "iteration": it,
                "strategy_id": strat_id,
                "score": cur_score,
                **err,
            },
        )
        attempt_history.append(
            {
                "iteration": it,
                "strategy_id": strat_id,
                "route_family": get_uv_strategy_route(strat_id),
                "rationale": rationale,
                "passed": False,
                "score": cur_score,
                "error_analysis": err,
                "verdict": verdict,
            }
        )

    # Budget exhausted
    elapsed = time.monotonic() - t_start
    repair_log = _build_auto_repair_log(attempt_history)
    (deliverables_dir / "auto_repair_log.json").write_text(
        json.dumps(repair_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reasons = sorted(
        {bf for a in attempt_history for bf in (a.get("verdict") or {}).get("blocking_failures", [])}
    )
    best_sc = max((float(a.get("score", 0.0)) for a in attempt_history), default=0.0)
    exp = record_experiment(
        kdir,
        task_type="uv_bake",
        asset=low_poly.name,
        work_dir=work_dir,
        autonomous_completed=False,
        human_interventions=0,
        iterations_used=max_iterations,
        elapsed_seconds=elapsed,
        final_score=best_sc,
        passed=False,
        winning_strategy_id="",
        failure_reasons=reasons,
        seed=seed,
        config_payload=cfg_payload,
        reproducible_command=repro_cmd,
    )
    out = {
        "status": "budget_exhausted",
        "task_type": "uv_bake",
        "iterations_used": max_iterations,
        "elapsed_seconds": round(elapsed, 2),
        "attempt_history": attempt_history,
        "experiment": exp,
        "finished_at": now_iso(),
    }
    (work_dir / "uv_bake_failure_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out
