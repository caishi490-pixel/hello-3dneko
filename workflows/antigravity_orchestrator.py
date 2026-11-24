"""Hello 3DNeko `run_workflow` multi-agent orchestrator for the 3D production chain.

Coordinates:
- Orchestrator Stage Executors & Vision QA Gates (`phase`, `agent`)
- Codex Image Worker for Stage 1 (2D image gen) and Stage 4 (AI visual polish & UV optimization)
- Hyper3D Rodin (Stage 2) and headless Blender subprocess runner (Stage 3 & Stage 5)
"""

from pathlib import Path


PIPELINE_ROOT = (
    str(Path(__file__).resolve().parent.parent)
    if "__file__" in globals()
    else str(Path(WORKSPACES[0]) if globals().get("WORKSPACES") else Path.cwd())
)
PROJECT_DIR = args["project_dir"]
FROM_STAGE = int(args.get("from_stage", 1))
TO_STAGE = int(args.get("to_stage", 5))
CODEX_MODE = str(args.get("codex_mode", "cli"))
DRY_RUN = bool(args.get("dry_run", False))
DRY_FLAG = "--dry-run" if DRY_RUN else ""

GATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["passed", "summary", "defects", "artifacts"],
    "properties": {
        "passed": {"type": "boolean"},
        "summary": {"type": "string"},
        "defects": {"type": "array", "items": {"type": "string"}},
        "artifacts": {"type": "array", "items": {"type": "string"}},
    },
}


async def run_stage_gate(stage_idx: int, role: str, qa_instructions: str):
    prompt = (
        f"Execute stage {stage_idx} in workspace '{PIPELINE_ROOT}' for project '{PROJECT_DIR}'.\n"
        f"Run command:\n"
        f"python run_pipeline.py run --project \"{PROJECT_DIR}\" --from-stage {stage_idx} "
        f"--to-stage {stage_idx} --codex-mode {CODEX_MODE} {DRY_FLAG}\n\n"
        f"Then perform QA verification:\n{qa_instructions}\n"
        f"Call `finish` matching the schema (`passed`, `summary`, `defects`, `artifacts`)."
    )
    return await agent(prompt, schema=GATE_SCHEMA, role=role)


phase("Stage 0: Preflight")
preflight = await agent(
    f"Inspect '{PROJECT_DIR}/project.json' and run `python run_pipeline.py status --project \"{PROJECT_DIR}\"` "
    f"inside '{PIPELINE_ROOT}'. Confirm project config, codex mode '{CODEX_MODE}', and dry_run={DRY_RUN}.",
    schema=GATE_SCHEMA,
    role="Pipeline Preflight Auditor",
)
log("Preflight:", preflight)
if not preflight["passed"]:
    raise RuntimeError(f"Preflight failed: {preflight['defects']}")

if FROM_STAGE <= 1 <= TO_STAGE:
    phase("Stage 1: Codex 2D Image Gen & L/R QA")
    s1 = await run_stage_gate(
        1,
        "Stage 1 Codex 2D + QA Gate",
        (
            "If dry_run is False, visually inspect the generated turnaround PNG and `00_slices_overview.png` "
            "via `read_file`. Verify: (1) 40-degree open A-pose, (2) strict Front/Back Left-Right consistency "
            "for asymmetric traits, (3) pure #FFFFFF background with cleanly separated modular slices. "
            "Do NOT pass if L/R symmetry is contradictory!"
        ),
    )
    log("Stage 1:", s1)
    if not s1["passed"]:
        raise RuntimeError(f"Stage 1 QA rejected before 3D spend: {s1['defects']}")

if FROM_STAGE <= 2 <= TO_STAGE:
    phase("Stage 2: Hyper3D Rodin 3D Generation")
    s2 = await run_stage_gate(
        2,
        "Stage 2 Rodin 3D Executor",
        "Confirm `manifest.json` persisted all generation_ids cleanly and output GLBs exist (or dry-run plan succeeded).",
    )
    log("Stage 2:", s2)
    if not s2["passed"]:
        raise RuntimeError(f"Stage 2 failed: {s2['defects']}")

if FROM_STAGE <= 3 <= TO_STAGE:
    phase("Stage 3: Headless Blender Clean & Baseline Renders")
    s3 = await run_stage_gate(
        3,
        "Stage 3 Blender Clean & Baseline QA",
        (
            "If dry_run is False, inspect `03_rig_clean/renders/v1_front.png`, `v5_face.png`, and `v6_action_rom.png` "
            "(if present) using `read_file`, plus `blender_stage3_report.json`. Note remaining blurry eyes or stains for Stage 4."
        ),
    )
    log("Stage 3:", s3)
    if not s3["passed"]:
        raise RuntimeError(f"Stage 3 failed: {s3['defects']}")

if FROM_STAGE <= 4 <= TO_STAGE:
    phase("Stage 4: Codex Visual Polish & UV/Material Optimization")
    s4 = await run_stage_gate(
        4,
        "Stage 4 Codex Polish & UV Audit",
        (
            "If dry_run is False, verify `uv_audit/uv_audit.json`, visually inspect Codex's generated high-res eye maps, "
            "`polished_face`, `diffuse_opt`, `roughness_opt` via `read_file`, and validate `stage4_apply.json`."
        ),
    )
    log("Stage 4:", s4)
    if not s4["passed"]:
        raise RuntimeError(f"Stage 4 Codex optimization gate failed: {s4['defects']}")

if FROM_STAGE <= 5 <= TO_STAGE:
    phase("Stage 5: Headless Blender Bake, Export & Final Visual QC")
    s5 = await run_stage_gate(
        5,
        "Stage 5 Final Bake & Visual QC",
        (
            "If dry_run is False, inspect `05_final_bake/renders/00_before_after_qc_sheet.png` and `qc_face.png` via `read_file`. "
            "Confirm crisp eyes, matte-balanced skin/cloth without PBR washout, and exported `.blend` + `.glb`."
        ),
    )
    log("Stage 5:", s5)
