"""Unified 6-stage pipeline orchestrating Antigravity, Codex (image/UV worker), Rodin, and Blender."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .blender_runner import run_blender_script
from .codex_worker import CodexWorkOrder, render_template, run_codex_work_order
from .config import (
    REPO_ROOT,
    ManifestStore,
    ProjectConfig,
    ProjectLayout,
    append_ledger,
)
from .guards import CircuitBreaker
from .image_tools import (
    audit_uv_atlas,
    make_contact_sheet,
    slice_breakdown_components,
    slice_turnaround_views,
)
from .rodin_client import RodinJobSpec, run_rodin_batch


BLENDER_CLEAN_SCRIPT = REPO_ROOT / "blender_scripts" / "clean_rig_and_render.py"
BLENDER_APPLY_SCRIPT = REPO_ROOT / "blender_scripts" / "apply_uv_toon_bake.py"
STAGE1_TEMPLATE = REPO_ROOT / "templates" / "codex_stage1_2d_prompt.md"
STAGE4_TEMPLATE = REPO_ROOT / "templates" / "codex_stage4_uv_polish_prompt.md"


@dataclass(frozen=True)
class StageSpec:
    index: int
    name: str
    description: str


STAGE_ORDER: tuple[StageSpec, ...] = (
    StageSpec(0, "init", "Validate project config and manifest workspace"),
    StageSpec(1, "design_2d", "Codex generates 40-deg A-pose turnaround and modular breakdown sheets, then slices 1024px inputs"),
    StageSpec(2, "rodin_3d", "Hyper3D Rodin 3D generation (whole/quad/parts/BANG/rigging) with persisted generation_ids"),
    StageSpec(3, "clean_rig_render", "Headless Blender cleans topology/bleed, extracts atlases, probes eye bounds, and renders baseline views"),
    StageSpec(4, "codex_uv_polish", "UV pixel audit + Codex generates high-res eyes/polished renders and optimizes UV/roughness maps"),
    StageSpec(5, "bake_export_qc", "Headless Blender splits eye slots, applies calibrated textures/Toon, exports blend/glb, and renders QC"),
)

STAGE_BY_NAME = {s.name: s for s in STAGE_ORDER}
STAGE_BY_INDEX = {s.index: s for s in STAGE_ORDER}


def resolve_stage_index(value: str | int) -> int:
    if isinstance(value, int):
        if value not in STAGE_BY_INDEX:
            raise ValueError(f"Unknown stage index: {value}")
        return value
    s = str(value).strip()
    if s.isdigit() and int(s) in STAGE_BY_INDEX:
        return int(s)
    if s in STAGE_BY_NAME:
        return STAGE_BY_NAME[s].index
    raise ValueError(f"Unknown stage '{value}'. Valid names: {[x.name for x in STAGE_ORDER]}")


def _stage0_init(
    layout: ProjectLayout,
    cfg: ProjectConfig,
    store: ManifestStore,
    breaker: CircuitBreaker,
    dry_run: bool = False,
) -> dict[str, Any]:
    del breaker
    layout.ensure_dirs()
    result = {
        "status": "dry_run" if dry_run else "completed",
        "project_root": str(layout.root),
        "character_id": cfg.character_id,
    }
    if not dry_run:
        store.update_stage("init", result)
        append_ledger(layout.ledger_path, {"stage": "init", "status": "completed"})
    return result


def _stage1_design_2d(
    layout: ProjectLayout,
    cfg: ProjectConfig,
    store: ManifestStore,
    breaker: CircuitBreaker,
    dry_run: bool = False,
) -> dict[str, Any]:
    turnaround = layout.dir_2d / f"{cfg.character_id}_turnaround_apose.png"
    sheet_garments = layout.dir_2d / f"{cfg.character_id}_sheet_garments.png"
    sheet_props = layout.dir_2d / f"{cfg.character_id}_sheet_props.png"

    prompt = render_template(
        STAGE1_TEMPLATE,
        {
            "CHARACTER_ID": cfg.character_id,
            "CHARACTER_NAME": cfg.character_name,
            "DESIGN_BRIEF": cfg.design_brief,
            "STYLE_NOTES": cfg.style_notes,
            "OUT_DIR": str(layout.dir_2d),
            "TURNAROUND_OUT": str(turnaround),
            "SHEET_GARMENTS_OUT": str(sheet_garments),
            "SHEET_PROPS_OUT": str(sheet_props),
            "PART_NAMES": ", ".join(cfg.part_names),
        },
    )
    order = CodexWorkOrder(
        task_id="stage1_2d_design",
        stage="design_2d",
        prompt_text=prompt,
        work_dir=REPO_ROOT,
        input_images=[Path(p) for p in cfg.reference_images],
        expected_outputs=[turnaround, sheet_garments, sheet_props],
        mode=cfg.codex.mode,
        sandbox=cfg.codex.sandbox,
        model=cfg.codex.model,
        codex_cmd=cfg.tools.codex_cmd,
        timeout_seconds=cfg.codex.timeout_seconds,
    )
    codex_res = run_codex_work_order(order, layout.dir_codex_orders, breaker, dry_run=dry_run)
    if dry_run:
        return {
            "status": "dry_run",
            "codex": codex_res,
            "planned_slices_dir": str(layout.dir_2d_slices),
        }

    if codex_res.get("missing_expected_outputs"):
        raise RuntimeError(
            f"Stage 1 Codex missing deliverables: {codex_res['missing_expected_outputs']}"
        )

    view_paths = slice_turnaround_views(turnaround, layout.dir_2d_slices / "views")
    garment_slices = slice_breakdown_components(
        sheet_garments,
        layout.dir_2d_slices / "garments",
        prefix="garment",
        labels_hint=cfg.part_names,
        bridge_px=6,
        min_area=2000,
    )
    prop_slices = slice_breakdown_components(
        sheet_props,
        layout.dir_2d_slices / "props",
        prefix="prop",
        bridge_px=3,
        min_area=1200,
    )
    overview_items = view_paths + [Path(x["path"]) for x in garment_slices + prop_slices]
    overview = make_contact_sheet(
        overview_items,
        layout.dir_2d_slices / "00_slices_overview.png",
        cols=4,
    )
    out = {
        "status": "completed",
        "turnaround": str(turnaround),
        "views": [str(p) for p in view_paths],
        "garment_slices": garment_slices,
        "prop_slices": prop_slices,
        "overview_sheet": str(overview),
        "codex_response": codex_res.get("response_path"),
    }
    store.update_stage("design_2d", out)
    append_ledger(layout.ledger_path, {"stage": "design_2d", "status": "completed"})
    return out


def _stage2_rodin_3d(
    layout: ProjectLayout,
    cfg: ProjectConfig,
    store: ManifestStore,
    breaker: CircuitBreaker,
    dry_run: bool = False,
) -> dict[str, Any]:
    views_dir = layout.dir_2d_slices / "views"
    view_imgs = [views_dir / "01_front.png", views_dir / "02_side.png", views_dir / "03_back.png"]

    whole_name = f"{cfg.character_id}_whole"
    specs = [
        RodinJobSpec(
            name=whole_name,
            action="generate",
            images=view_imgs,
            tier="Gen-2.5-High",
            mesh_mode="Raw",
            prompt=f"High-detail anime 3D character in symmetrical 40-degree A-pose: {cfg.design_brief}",
        ),
        RodinJobSpec(
            name=f"{cfg.character_id}_bang",
            action="bang",
            target_job_name=whole_name,
            bang_strength=12,
            bang_instruction=", ".join(cfg.part_names),
            poll_timeout=600,
        ),
        RodinJobSpec(
            name=f"{cfg.character_id}_rig",
            action="rigging",
            target_job_name=whole_name,
            poll_timeout=600,
        ),
    ]
    res = run_rodin_batch(
        specs,
        layout.dir_3d_rodin,
        store,
        breaker,
        hyper3d_cmd=cfg.tools.hyper3d_cmd,
        dry_run=dry_run,
    )
    if not dry_run:
        store.update_stage("rodin_3d", res)
        append_ledger(layout.ledger_path, {"stage": "rodin_3d", "status": "completed", "jobs": len(specs)})
    return res


def _resolve_stage3_input_glb(layout: ProjectLayout, cfg: ProjectConfig, store: ManifestStore) -> Path:
    manifest = store.load()
    rig_entry = manifest.get("rodin_jobs", {}).get(f"{cfg.character_id}_rig", {})
    for info in rig_entry.get("downloaded", {}).values():
        if str(info["path"]).lower().endswith(".glb"):
            return Path(info["path"])
    candidate = layout.dir_3d_rodin / f"{cfg.character_id}_rig" / "base_basic_pbr.glb"
    return candidate


def _stage3_clean_rig(
    layout: ProjectLayout,
    cfg: ProjectConfig,
    store: ManifestStore,
    breaker: CircuitBreaker,
    dry_run: bool = False,
) -> dict[str, Any]:
    del breaker
    input_glb = _resolve_stage3_input_glb(layout, cfg, store)
    if not dry_run and not input_glb.is_file():
        raise FileNotFoundError(f"Stage 3 requires rigged GLB from Stage 2: {input_glb}")

    textures_dir = layout.dir_rig_clean / "extracted_textures"
    output_blend = layout.dir_rig_clean / f"{cfg.character_id}_cleaned.blend"
    output_glb = layout.dir_rig_clean / f"{cfg.character_id}_cleaned.glb"
    report_json = layout.dir_rig_clean / "blender_stage3_report.json"

    job_cfg = {
        "input_glb": str(input_glb),
        "mode": "rig_glb",
        "primary_mesh_name": f"{cfg.character_id}_Body",
        "target_height": cfg.blender.target_height,
        "cycles_device_type": cfg.blender.cycles_device_type,
        "render_samples": cfg.blender.render_samples,
        "render_resolution": cfg.blender.render_resolution,
        "textures_dir": str(textures_dir),
        "renders_dir": str(layout.dir_rig_renders),
        "output_blend": str(output_blend),
        "output_glb": str(output_glb),
        "report_json": str(report_json),
    }
    res = run_blender_script(
        cfg.tools.blender_exe,
        BLENDER_CLEAN_SCRIPT,
        job_cfg,
        layout.dir_rig_clean / "stage3_blender_job.json",
        layout.dir_rig_clean / "stage3_blender.log",
        dry_run=dry_run,
    )
    if dry_run:
        return {"status": "dry_run", "blender": res, "planned_outputs": [str(output_blend), str(output_glb)]}

    report = json.loads(report_json.read_text(encoding="utf-8"))
    out = {"status": "completed", "blender": res, "report_json": str(report_json), "report": report}
    store.update_stage("clean_rig_render", out)
    append_ledger(layout.ledger_path, {"stage": "clean_rig_render", "status": "completed"})
    return out


def _stage4_codex_polish(
    layout: ProjectLayout,
    cfg: ProjectConfig,
    store: ManifestStore,
    breaker: CircuitBreaker,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = store.load()
    s3_report = manifest.get("stages", {}).get("clean_rig_render", {}).get("report", {})
    textures_dir = layout.dir_rig_clean / "extracted_textures"
    diffuse_in = Path(s3_report.get("extracted_textures", {}).get("diffuse") or textures_dir / "diffuse_extracted.png")
    roughness_in = Path(s3_report.get("extracted_textures", {}).get("roughness") or textures_dir / "roughness_extracted.png")

    eye_regions = s3_report.get("eye_regions") or {
        "right_eye": {"pil_center": [159, 945], "radius": 24},
        "left_eye": {"pil_center": [494, 96], "radius": 24},
    }
    uv_audit_dir = layout.dir_codex_polish / "uv_audit"
    uv_audit_json = uv_audit_dir / "uv_audit.json"
    if not dry_run:
        if not diffuse_in.is_file():
            raise FileNotFoundError(f"Missing extracted diffuse atlas for Stage 4: {diffuse_in}")
        audit_uv_atlas(diffuse_in, eye_regions, uv_audit_dir)

    eye_left_out = layout.dir_codex_polish / f"{cfg.character_id}_eye_left_hires.png"
    eye_right_out = layout.dir_codex_polish / f"{cfg.character_id}_eye_right_hires.png"
    polished_face_out = layout.dir_codex_polish / f"{cfg.character_id}_polished_face.png"
    diffuse_opt_out = layout.dir_codex_polish / f"{cfg.character_id}_diffuse_opt.png"
    roughness_opt_out = layout.dir_codex_polish / f"{cfg.character_id}_roughness_opt.png"
    apply_spec_out = layout.dir_codex_polish / "stage4_apply.json"

    turnaround_ref = Path(
        manifest.get("stages", {}).get("design_2d", {}).get("turnaround")
        or (layout.dir_2d / f"{cfg.character_id}_turnaround_apose.png")
    )
    attach_images: list[Path] = []
    for candidate in (
        layout.dir_rig_renders / "v5_face.png",
        layout.dir_rig_renders / "v1_front.png",
        turnaround_ref,
        diffuse_in,
    ):
        if dry_run or candidate.is_file():
            attach_images.append(candidate)

    prompt = render_template(
        STAGE4_TEMPLATE,
        {
            "CHARACTER_ID": cfg.character_id,
            "CHARACTER_NAME": cfg.character_name,
            "OUT_DIR": str(layout.dir_codex_polish),
            "UV_AUDIT_JSON": str(uv_audit_json),
            "DIFFUSE_IN": str(diffuse_in),
            "ROUGHNESS_IN": str(roughness_in),
            "RENDERS_DIR": str(layout.dir_rig_renders),
            "TURNAROUND_REF": str(turnaround_ref),
            "EYE_LEFT_OUT": str(eye_left_out),
            "EYE_RIGHT_OUT": str(eye_right_out),
            "POLISHED_FACE_OUT": str(polished_face_out),
            "DIFFUSE_OPT_OUT": str(diffuse_opt_out),
            "ROUGHNESS_OPT_OUT": str(roughness_opt_out),
            "APPLY_SPEC_OUT": str(apply_spec_out),
        },
    )
    order = CodexWorkOrder(
        task_id="stage4_uv_polish",
        stage="codex_uv_polish",
        prompt_text=prompt,
        work_dir=REPO_ROOT,
        input_images=attach_images,
        expected_outputs=[diffuse_opt_out, apply_spec_out],
        mode=cfg.codex.mode,
        sandbox=cfg.codex.sandbox,
        model=cfg.codex.model,
        codex_cmd=cfg.tools.codex_cmd,
        timeout_seconds=cfg.codex.timeout_seconds,
    )
    codex_res = run_codex_work_order(order, layout.dir_codex_orders, breaker, dry_run=dry_run)
    if dry_run:
        return {"status": "dry_run", "codex": codex_res, "planned_apply_spec": str(apply_spec_out)}

    if codex_res.get("missing_expected_outputs"):
        raise RuntimeError(
            f"Stage 4 Codex missing required optimization outputs: {codex_res['missing_expected_outputs']}"
        )
    out = {
        "status": "completed",
        "uv_audit_json": str(uv_audit_json),
        "apply_spec_json": str(apply_spec_out),
        "codex_response": codex_res.get("response_path"),
    }
    store.update_stage("codex_uv_polish", out)
    append_ledger(layout.ledger_path, {"stage": "codex_uv_polish", "status": "completed"})
    return out


def _stage5_bake_export(
    layout: ProjectLayout,
    cfg: ProjectConfig,
    store: ManifestStore,
    breaker: CircuitBreaker,
    dry_run: bool = False,
) -> dict[str, Any]:
    del breaker
    input_blend = layout.dir_rig_clean / f"{cfg.character_id}_cleaned.blend"
    apply_spec_json = layout.dir_codex_polish / "stage4_apply.json"
    if not dry_run:
        for req in (input_blend, apply_spec_json):
            if not req.is_file():
                raise FileNotFoundError(f"Stage 5 missing required artifact: {req}")

    output_blend = layout.dir_final_bake / f"{cfg.character_id}_final_polished.blend"
    output_glb = layout.dir_final_bake / f"{cfg.character_id}_final_polished.glb"
    report_json = layout.dir_final_bake / "blender_stage5_report.json"

    job_cfg = {
        "input_blend": str(input_blend),
        "apply_spec_json": str(apply_spec_json),
        "target_height": cfg.blender.target_height,
        "output_blend": str(output_blend),
        "output_glb": str(output_glb),
        "renders_dir": str(layout.dir_final_renders),
        "report_json": str(report_json),
    }
    res = run_blender_script(
        cfg.tools.blender_exe,
        BLENDER_APPLY_SCRIPT,
        job_cfg,
        layout.dir_final_bake / "stage5_blender_job.json",
        layout.dir_final_bake / "stage5_blender.log",
        dry_run=dry_run,
    )
    if dry_run:
        return {"status": "dry_run", "blender": res, "planned_outputs": [str(output_blend), str(output_glb)]}

    qc_images = [
        layout.dir_rig_renders / "v5_face.png",
        layout.dir_final_renders / "qc_face.png",
        layout.dir_rig_renders / "v1_front.png",
        layout.dir_final_renders / "qc_front.png",
        layout.dir_final_renders / "qc_34.png",
    ]
    existing_qc = [p for p in qc_images if p.is_file()]
    qc_sheet = None
    if existing_qc:
        qc_sheet = str(
            make_contact_sheet(
                existing_qc,
                layout.dir_final_renders / "00_before_after_qc_sheet.png",
                cols=3,
            )
        )
    report = json.loads(report_json.read_text(encoding="utf-8"))
    out = {
        "status": "completed",
        "blender": res,
        "final_blend": str(output_blend),
        "final_glb": str(output_glb),
        "qc_sheet": qc_sheet,
        "report": report,
    }
    store.update_stage("bake_export_qc", out)
    append_ledger(layout.ledger_path, {"stage": "bake_export_qc", "status": "completed"})
    return out


STAGE_RUNNERS: dict[
    str, Callable[[ProjectLayout, ProjectConfig, ManifestStore, CircuitBreaker, bool], dict[str, Any]]
] = {
    "init": _stage0_init,
    "design_2d": _stage1_design_2d,
    "rodin_3d": _stage2_rodin_3d,
    "clean_rig_render": _stage3_clean_rig,
    "codex_uv_polish": _stage4_codex_polish,
    "bake_export_qc": _stage5_bake_export,
}


def run_stage_range(
    layout: ProjectLayout,
    cfg: ProjectConfig,
    store: ManifestStore,
    from_stage: str | int = 0,
    to_stage: str | int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute consecutive stages [from_stage..to_stage] deterministically."""
    start_idx = resolve_stage_index(from_stage)
    end_idx = resolve_stage_index(to_stage)
    if start_idx > end_idx:
        raise ValueError(f"from_stage ({start_idx}) must be <= to_stage ({end_idx})")

    breaker = CircuitBreaker(layout.breaker_path)
    results: dict[str, Any] = {}
    for idx in range(start_idx, end_idx + 1):
        spec = STAGE_BY_INDEX[idx]
        results[spec.name] = STAGE_RUNNERS[spec.name](layout, cfg, store, breaker, dry_run)
    return {
        "character_id": cfg.character_id,
        "dry_run": dry_run,
        "from_stage": STAGE_BY_INDEX[start_idx].name,
        "to_stage": STAGE_BY_INDEX[end_idx].name,
        "stages": results,
    }
