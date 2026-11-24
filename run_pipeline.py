"""CLI entrypoint for Hello 3DNeko — AI-driven end-to-end 3D asset & garment production workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import (
    REPO_ROOT,
    CodexSettings,
    ProjectConfig,
    init_project,
    load_project,
)
from pipeline.image_tools import audit_uv_atlas
from pipeline.experiment_ledger import compute_ledger_summary, load_experiments, rebuild_scorecard_markdown
from pipeline.stages import STAGE_ORDER, run_stage_range
from pipeline.topology_agent_loop import RetopoCriteria, run_closed_loop_retopo
from pipeline.uv_bake_agent_loop import UVBakeCriteria, run_closed_loop_uv_bake
from pipeline.cloth_rig_agent_loop import ClothRigCriteria, run_closed_loop_cloth_rig


def cmd_init(args: argparse.Namespace) -> int:
    projects_dir = Path(args.projects_dir) if args.projects_dir else REPO_ROOT / "projects"
    cfg = ProjectConfig(
        character_id=args.id,
        character_name=args.name,
        design_brief=args.brief,
        reference_images=[str(Path(p).resolve()) for p in (args.ref or [])],
        codex=CodexSettings(mode=args.codex_mode),
    )
    layout = init_project(projects_dir, cfg, overwrite=args.force)
    print(json.dumps({"status": "initialized", "project_root": str(layout.root)}, ensure_ascii=False, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    layout, cfg, store = load_project(Path(args.project))
    manifest = store.load()
    print(
        json.dumps(
            {
                "character_id": cfg.character_id,
                "character_name": cfg.character_name,
                "project_root": str(layout.root),
                "codex_mode": cfg.codex.mode,
                "stages": manifest.get("stages", {}),
                "rodin_jobs": {k: {"action": v.get("action"), "status": v.get("status")} for k, v in manifest.get("rodin_jobs", {}).items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    layout, cfg, store = load_project(Path(args.project))
    if args.codex_mode:
        cfg.codex.mode = args.codex_mode
    result = run_stage_range(
        layout,
        cfg,
        store,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_inspect_uv(args: argparse.Namespace) -> int:
    regions = json.loads(Path(args.regions_json).read_text(encoding="utf-8-sig"))
    res = audit_uv_atlas(Path(args.texture), regions, Path(args.out), starved_threshold_px=args.threshold)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_retopo(args: argparse.Namespace) -> int:
    criteria = RetopoCriteria(
        min_faces=args.min_faces,
        max_faces=args.max_faces,
        min_quad_ratio=args.min_quad_ratio,
        min_valence4_ratio=args.min_valence4_ratio,
    )
    res = run_closed_loop_retopo(
        input_asset=Path(args.input),
        work_dir=Path(args.out),
        criteria=criteria,
        max_iterations=args.max_iters,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("status") in ("delivered", "dry_run") else 1


def cmd_uv_bake(args: argparse.Namespace) -> int:
    criteria = UVBakeCriteria(
        max_overlap_pct=args.max_overlap_pct,
        max_p95_stretch=args.max_p95_stretch,
        min_packing_coverage_pct=args.min_pack_pct,
        min_bake_coverage=args.min_bake_cov,
    )
    res = run_closed_loop_uv_bake(
        high_poly=Path(args.high),
        low_poly=Path(args.low),
        work_dir=Path(args.out),
        criteria=criteria,
        max_iterations=args.max_iters,
        texture_size=args.texture_size,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("status") in ("delivered", "dry_run") else 1


def cmd_cloth_rig(args: argparse.Namespace) -> int:
    criteria = ClothRigCriteria(
        max_rest_clipping_pct=args.max_rest_clip_pct,
        max_pose_clipping_pct=args.max_pose_clip_pct,
        max_penetration_mm=args.max_pen_mm,
        max_collapsed_edges_pct=args.max_collapse_pct,
        max_overstretched_edges_pct=args.max_stretch_pct,
    )
    res = run_closed_loop_cloth_rig(
        body_asset=Path(args.body),
        garment_asset=Path(args.garment),
        work_dir=Path(args.out),
        criteria=criteria,
        max_iterations=args.max_iters,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("status") in ("delivered", "dry_run") else 1


def cmd_experiments(args: argparse.Namespace) -> int:
    kdir = Path(args.knowledge_dir) if args.knowledge_dir else REPO_ROOT / "knowledge"
    rows = load_experiments(kdir)
    md = rebuild_scorecard_markdown(kdir)
    summary = compute_ledger_summary(rows)
    print(
        json.dumps(
            {
                "summary": summary,
                "scorecard_md": str(md),
                "runs": rows[-args.limit :] if args.limit > 0 else rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hello 3DNeko — Multi-Agent 3D Character, Topology, UV/Bake & Garment Rigging Pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize a new character project workspace")
    p_init.add_argument("--id", required=True, help="Slug ID (e.g., neko_demo_v1)")
    p_init.add_argument("--name", required=True, help="Character display name")
    p_init.add_argument("--brief", required=True, help="Visual & outfit design brief")
    p_init.add_argument("--ref", action="append", help="Optional reference image path (repeatable)")
    p_init.add_argument("--projects-dir", default="", help="Base projects folder override")
    p_init.add_argument("--codex-mode", default="cli", choices=("cli", "desktop"))
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="Show stage & Rodin manifest status for a project")
    p_status.add_argument("--project", required=True)
    p_status.set_defaults(func=cmd_status)

    stage_help = ", ".join(f"{s.index}={s.name}" for s in STAGE_ORDER)
    p_run = sub.add_parser("run", help=f"Run pipeline stages ({stage_help})")
    p_run.add_argument("--project", required=True)
    p_run.add_argument("--from-stage", default="0")
    p_run.add_argument("--to-stage", default="5")
    p_run.add_argument("--codex-mode", default="", choices=("", "cli", "desktop"))
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_uv = sub.add_parser("inspect-uv", help="Run UV pixel density audit and zoomed crop generator")
    p_uv.add_argument("--texture", required=True)
    p_uv.add_argument("--regions-json", required=True)
    p_uv.add_argument("--out", required=True)
    p_uv.add_argument("--threshold", type=int, default=48)
    p_uv.set_defaults(func=cmd_inspect_uv)

    p_ret = sub.add_parser("retopo", help="Run closed-loop autonomous topology repair and 6-pillar validation")
    p_ret.add_argument("--input", required=True, help="Raw .glb or .blend asset to retopologize")
    p_ret.add_argument("--out", required=True, help="Output closed-loop working directory")
    p_ret.add_argument("--max-iters", type=int, default=5)
    p_ret.add_argument("--min-faces", type=int, default=2500)
    p_ret.add_argument("--max-faces", type=int, default=22000)
    p_ret.add_argument("--min-quad-ratio", type=float, default=0.80)
    p_ret.add_argument("--min-valence4-ratio", type=float, default=0.68)
    p_ret.add_argument("--dry-run", action="store_true")
    p_ret.set_defaults(func=cmd_retopo)

    p_uvb = sub.add_parser("uv-bake", help="Autonomous UV & Baking v0.1 (Seam unwrap, Overlap/Stretch QA, Normal/AO/BaseColor/ORM bake)")
    p_uvb.add_argument("--high", required=True, help="High-density reference .glb/.blend (~500k tris)")
    p_uvb.add_argument("--low", required=True, help="Retopologized low-poly .glb/.blend mesh")
    p_uvb.add_argument("--out", required=True, help="Output UV & Baking working directory")
    p_uvb.add_argument("--max-iters", type=int, default=4)
    p_uvb.add_argument("--texture-size", type=int, default=2048)
    p_uvb.add_argument("--max-overlap-pct", type=float, default=0.8)
    p_uvb.add_argument("--max-p95-stretch", type=float, default=4.5)
    p_uvb.add_argument("--min-pack-pct", type=float, default=30.0)
    p_uvb.add_argument("--min-bake-cov", type=float, default=0.96)
    p_uvb.add_argument("--seed", type=int, default=42)
    p_uvb.add_argument("--dry-run", action="store_true")
    p_uvb.set_defaults(func=cmd_uv_bake)

    p_exp = sub.add_parser("experiments", help="Show unified 7-metric experiment ledger and regenerate EXPERIMENT_SCORECARD.md")
    p_exp.add_argument("--knowledge-dir", default="")
    p_exp.add_argument("--limit", type=int, default=20)
    p_exp.set_defaults(func=cmd_experiments)

    p_cloth = sub.add_parser("cloth-rig", help="Run closed-loop garment fitting, skeleton weighting, and 4-pose stress test")
    p_cloth.add_argument("--body", required=True, help="Rigged basemesh .glb/.blend")
    p_cloth.add_argument("--garment", required=True, help="Retopologized/baked garment .glb/.blend")
    p_cloth.add_argument("--out", required=True)
    p_cloth.add_argument("--max-iters", type=int, default=4)
    p_cloth.add_argument("--max-rest-clip-pct", type=float, default=1.5)
    p_cloth.add_argument("--max-pose-clip-pct", type=float, default=3.5)
    p_cloth.add_argument("--max-pen-mm", type=float, default=14.0)
    p_cloth.add_argument("--max-collapse-pct", type=float, default=2.0)
    p_cloth.add_argument("--max-stretch-pct", type=float, default=3.5)
    p_cloth.add_argument("--seed", type=int, default=0)
    p_cloth.add_argument("--dry-run", action="store_true")
    p_cloth.set_defaults(func=cmd_cloth_rig)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
