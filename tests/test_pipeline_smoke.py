"""Offline smoke & contract tests for anime3d_agent_pipeline (zero paid API calls)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.codex_worker import CodexWorkOrder, build_codex_exec_command  # noqa: E402
from pipeline.config import CodexSettings, ProjectConfig, init_project, load_project  # noqa: E402
from pipeline.guards import CircuitBreaker, PaidCallGuardError, redact_secrets  # noqa: E402
from pipeline.image_tools import (  # noqa: E402
    audit_uv_atlas,
    make_contact_sheet,
    slice_breakdown_components,
    slice_turnaround_views,
)
from pipeline.stages import run_stage_range  # noqa: E402


class PipelineSmokeTests(unittest.TestCase):
    def test_secret_redaction_and_circuit_breaker(self) -> None:
        raw = "token comfyui-AbCdEf1234567890_xyz url https://cdn.example.com/a.glb?sig=secret123"
        clean = redact_secrets(raw)
        self.assertNotIn("comfyui-AbCdEf", clean)
        self.assertNotIn("sig=secret123", clean)

        with tempfile.TemporaryDirectory() as td:
            breaker = CircuitBreaker(Path(td) / "cb.json", max_consecutive_failures=2)
            breaker.record_attempt(False)
            breaker.check_before_call("second")
            breaker.record_attempt(False)
            with self.assertRaises(PaidCallGuardError):
                breaker.check_before_call("third")

    def test_turnaround_breakdown_and_uv_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Build synthetic 3-view sheet on #FFFFFF
            turnaround = root / "turnaround.png"
            im = Image.new("RGB", (900, 300), (255, 255, 255))
            draw = ImageDraw.Draw(im)
            draw.rectangle((40, 20, 220, 270), fill=(120, 90, 200))
            draw.rectangle((360, 20, 520, 270), fill=(140, 100, 210))
            draw.rectangle((660, 20, 840, 270), fill=(160, 110, 220))
            im.save(turnaround, "PNG")

            views = slice_turnaround_views(turnaround, root / "views", size=256)
            self.assertEqual(len(views), 3)
            for vp in views:
                with Image.open(vp) as out_im:
                    self.assertEqual(out_im.size, (256, 256))

            # Build synthetic 2-part modular breakdown sheet
            sheet = root / "sheet.png"
            sim = Image.new("RGB", (600, 300), (255, 255, 255))
            sdraw = ImageDraw.Draw(sim)
            sdraw.ellipse((40, 40, 200, 240), fill=(80, 60, 180))
            sdraw.ellipse((350, 50, 540, 250), fill=(210, 130, 190))
            sim.save(sheet, "PNG")

            parts = slice_breakdown_components(sheet, root / "parts", labels_hint=["coat", "dress"], size=256)
            self.assertEqual(len(parts), 2)
            self.assertTrue((root / "parts" / "01_coat.png").is_file())
            self.assertTrue((root / "parts" / "02_dress.png").is_file())

            # Audit synthetic diffuse atlas where iris region is tiny (20x20 < 48px threshold)
            atlas = root / "atlas.png"
            aim = Image.new("RGB", (512, 512), (240, 220, 210))
            adraw = ImageDraw.Draw(aim)
            adraw.ellipse((90, 90, 110, 110), fill=(240, 180, 40))
            aim.save(atlas, "PNG")

            audit = audit_uv_atlas(
                atlas,
                {"left_eye": {"pil_center": [100, 100], "radius": 10}},
                root / "uv_audit",
                starved_threshold_px=48,
            )
            self.assertTrue(audit["regions"]["left_eye"]["starved"])
            self.assertEqual(audit["regions"]["left_eye"]["pixel_size"], [20, 20])
            self.assertTrue(Path(audit["regions"]["left_eye"]["zoom_crop"]).is_file())

            grid = make_contact_sheet(views, root / "grid.png", cols=3, cell_size=128)
            self.assertTrue(grid.is_file())

    def test_project_init_codex_cmd_and_full_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            projects_dir = Path(td) / "projects"
            cfg = ProjectConfig(
                character_id="demo_catgirl",
                character_name="Demo Catgirl",
                design_brief="Periwinkle-to-pink gradient catgirl in 40-deg A-pose",
                codex=CodexSettings(mode="cli"),
            )
            layout = init_project(projects_dir, cfg)
            _, loaded_cfg, store = load_project(layout.root)
            self.assertEqual(loaded_cfg.character_id, "demo_catgirl")

            summary = run_stage_range(layout, loaded_cfg, store, from_stage=0, to_stage=5, dry_run=True)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(
                list(summary["stages"].keys()),
                ["init", "design_2d", "rodin_3d", "clean_rig_render", "codex_uv_polish", "bake_export_qc"],
            )
            self.assertTrue((layout.dir_codex_orders / "stage1_2d_design_dry_run.json").is_file())
            self.assertTrue((layout.dir_codex_orders / "stage4_uv_polish_dry_run.json").is_file())

            # Verify CLI schema & command flags
            order = CodexWorkOrder(
                task_id="t",
                stage="design_2d",
                prompt_text="hi",
                work_dir=REPO_ROOT,
                input_images=[layout.config_path],
            )
            cmd = build_codex_exec_command(order, Path(td) / "resp.json")
            self.assertIn("exec", cmd)
            self.assertIn("--output-schema", cmd)
            self.assertIn("--skip-git-repo-check", cmd)

    def test_cli_init_and_dry_run_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td) / "projects"
            init_proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "init",
                    "--id",
                    "cli_test",
                    "--name",
                    "CLI Test",
                    "--brief",
                    "40-deg A-pose test character",
                    "--projects-dir",
                    str(pdir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(init_proc.returncode, 0, msg=init_proc.stderr)
            run_proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "run",
                    "--project",
                    str(pdir / "cli_test"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run_proc.returncode, 0, msg=run_proc.stderr)
            payload = json.loads(run_proc.stdout)
            self.assertEqual(payload["to_stage"], "bake_export_qc")

    def test_blender_scripts_compile_cleanly(self) -> None:
        for script_name in (
            "clean_rig_and_render.py",
            "apply_uv_toon_bake.py",
            "topology_inspector.py",
            "topology_toolkit.py",
            "topology_validator.py",
        ):
            path = REPO_ROOT / "blender_scripts" / script_name
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_closed_loop_retopo_planner_rollback_and_cli_dry_run(self) -> None:
        from pipeline.topology_agent_loop import (
            RetopoCriteria,
            agent_decide_repair_plan,
            analyze_failure_and_rollback,
            run_closed_loop_retopo,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_glb = root / "prop.glb"
            fake_glb.write_bytes(b"glTF")
            res = run_closed_loop_retopo(
                fake_glb,
                root / "retopo_out",
                criteria=RetopoCriteria(),
                max_iterations=3,
                dry_run=True,
            )
            self.assertEqual(res["status"], "dry_run")
            self.assertTrue((root / "retopo_out" / "acceptance_criteria.json").is_file())

            # Test Rollback + Error Analysis -> Agent Decision escalation
            pre = root / "pre.glb"
            active = root / "active.glb"
            pre.write_bytes(b"CLEAN_PRE")
            active.write_bytes(b"CORRUPTED_CANDIDATE")
            diag = analyze_failure_and_rollback(
                pre,
                active,
                "quadriflow_direct_shrink_bake",
                toolkit_ok=True,
                toolkit_log="",
                verdict={
                    "passed": False,
                    "score": 66.67,
                    "blocking_failures": ["Pillar3 Manifold: welded_non_manifold_edges=1"],
                    "pillars": {"3_manifold_cleanliness": {"passed": False}},
                },
            )
            self.assertEqual(active.read_bytes(), b"CLEAN_PRE")
            self.assertIn("non_manifold_blocking", diag["failure_classes"])

            baseline = {
                "junk_objects": [],
                "total_faces_raw": 500000,
                "meshes": [{"faces_raw": 500000, "raw_quad_ratio": 0.0, "dims": [1.0, 1.0, 1.0]}],
            }
            strat, actions, _ = agent_decide_repair_plan(
                baseline,
                [{"error_analysis": diag}],
                {"quadriflow_direct_shrink_bake"},
                RetopoCriteria(),
            )
            self.assertEqual(strat, "quadriflow_smooth_post_clean_bake")
            self.assertTrue(any(a["action"] == "unwrap_and_bake_textures" for a in actions))

            cli_proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "retopo",
                    "--input",
                    str(fake_glb),
                    "--out",
                    str(root / "cli_retopo"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(cli_proc.returncode, 0, msg=cli_proc.stderr)
            self.assertEqual(json.loads(cli_proc.stdout)["status"], "dry_run")

    def test_four_outcome_evolution_triage_and_route_switching(self) -> None:
        from pipeline.topology_agent_loop import (
            RetopoCriteria,
            agent_decide_repair_plan,
            get_strategy_route,
            triage_and_persist_outcome,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kdir = root / "knowledge"
            baseline = {
                "junk_objects": [],
                "total_faces_raw": 500000,
                "meshes": [{"faces_raw": 500000, "raw_quad_ratio": 0.0, "dims": [1.0, 1.0, 1.0]}],
            }

            # 1b. high_surface_deviation escalates directly to quadriflow_healed_sharp_shrink_bake
            strat_dev, dev_actions, _ = agent_decide_repair_plan(
                baseline,
                [{"error_analysis": {"failure_classes": ["high_surface_deviation"]}}],
                {"quadriflow_healed_post_clean_bake"},
                RetopoCriteria(),
            )
            self.assertEqual(strat_dev, "quadriflow_healed_sharp_shrink_bake")
            self.assertEqual(get_strategy_route(strat_dev), "quadriflow_route")
            self.assertTrue(any(a["action"] == "shrinkwrap_to_reference" for a in dev_actions))

            # 1. TOTAL_FAILURE on a route forces switching to a distinct algorithm route
            strat_switched, voxel_actions, _ = agent_decide_repair_plan(
                baseline,
                [],
                set(),
                RetopoCriteria(),
                banned_routes={"quadriflow_route"},
            )
            self.assertNotEqual(get_strategy_route(strat_switched), "quadriflow_route")
            self.assertEqual(get_strategy_route(strat_switched), "voxel_route")
            self.assertTrue(any(a["action"] == "adaptive_voxel_quadriflow" for a in voxel_actions))

            # 2. SUCCESS -> crystallize Skill recipe
            succ = triage_and_persist_outcome(
                "SUCCESS",
                "corset.glb",
                root / "w_succ",
                kdir,
                baseline,
                [],
                {
                    "strategy_id": "quadriflow_smooth_post_clean_bake",
                    "score": 100.0,
                    "verdict": {"pillars": {"2_topology_quality": {"passed": True}}},
                },
                winning_actions=[{"action": "quadriflow_retopo", "params": {"target_faces": 16000}}],
            )
            self.assertEqual(succ["outcome_kind"], "SUCCESS")
            self.assertTrue((root / "w_succ" / "skill_recipe.md").is_file())
            self.assertTrue((kdir / "learned_recipes.jsonl").is_file())

            # 3. PARTIAL_SUCCESS -> decompose reusable capabilities
            fake_cand = root / "partial.glb"
            fake_cand.write_bytes(b"glTF")
            part = triage_and_persist_outcome(
                "PARTIAL_SUCCESS",
                "tricky.glb",
                root / "w_part",
                kdir,
                baseline,
                [],
                {
                    "strategy_id": "decimate_quadify_bake",
                    "score": 66.67,
                    "cand_glb": str(fake_cand),
                    "actions": [
                        {"action": "sanitize_mesh"},
                        {"action": "decimate_and_quadify"},
                        {"action": "unwrap_and_bake_textures"},
                    ],
                    "verdict": {
                        "blocking_failures": ["Pillar2 Topology: quad_ratio=0.17 < 0.8"],
                        "pillars": {
                            "1_geometric_fidelity": {"passed": True},
                            "2_topology_quality": {"passed": False},
                            "3_manifold_cleanliness": {"passed": True},
                            "4_polycount_budget": {"passed": True},
                            "5_uv_and_materials": {"passed": True},
                            "6_rig_engine_ready": {"passed": False},
                        },
                    },
                },
            )
            self.assertEqual(part["outcome_kind"], "PARTIAL_SUCCESS")
            self.assertIn("sanitize_mesh", part["decomposition"]["reusable_actions"])
            self.assertIn("unwrap_and_bake_textures", part["decomposition"]["reusable_actions"])
            self.assertTrue((root / "w_part" / "partial_decomposition.json").is_file())

            # 4. TOTAL_FAILURE -> switch algorithm / route recommendation
            tot = triage_and_persist_outcome(
                "TOTAL_FAILURE",
                "broken.glb",
                root / "w_tot",
                kdir,
                baseline,
                [{"strategy_id": "voxel_seal_quadriflow_bake", "score": 33.33, "error_analysis": {"failure_classes": ["polycount_out_of_budget"]}}],
                {"strategy_id": "voxel_seal_quadriflow_bake", "score": 33.33},
            )
            self.assertEqual(tot["outcome_kind"], "TOTAL_FAILURE")
            self.assertTrue((root / "w_tot" / "route_pivot_recommendation.json").is_file())
            self.assertTrue((kdir / "route_pivots.jsonl").is_file())

    def test_retopo_experiment_ledger_and_cli(self) -> None:
        from pipeline.experiment_ledger import compute_ledger_summary, load_experiments, record_experiment

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kdir = root / "knowledge"
            record_experiment(
                kdir,
                task_type="retopo",
                asset="corset.glb",
                work_dir=root / "w1",
                autonomous_completed=True,
                human_interventions=0,
                iterations_used=3,
                elapsed_seconds=106.2,
                final_score=100.0,
                passed=True,
                winning_strategy_id="quadriflow_smooth_post_clean_bake",
                seed=42,
            )
            rows = load_experiments(kdir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["auto_rework_count"], 2)
            summary = compute_ledger_summary(rows)
            self.assertEqual(summary["autonomous_completion_rate_pct"], 100.0)
            self.assertTrue((kdir / "EXPERIMENT_SCORECARD.md").is_file())

            exp_proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "experiments",
                    "--knowledge-dir",
                    str(kdir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(exp_proc.returncode, 0, msg=exp_proc.stderr)
            self.assertEqual(json.loads(exp_proc.stdout)["summary"]["total_runs"], 1)

    def test_repo_hygiene_no_private_paths_secrets_or_gpl_boundary_leak(self) -> None:
        """Gate against personal OS paths, credential file pointers, non-original refs, and bpy leakage."""
        forbidden_tokens = [
            "C:\\" + "Users\\" + "Gx" + "Caishi",
            "Gx" + "Caishi",
            "重要密钥" + "与VPS",
            "comfyui" + "密钥",
            "codex-" + "bridge-mcp",
            "gemini-" + "desktop-mcp",
            "collab-" + "env",
            "F:\\" + "Blender",
            "F:\\" + "tis",
            "tis " + "2026",
            "未" + "命名" + ".glb",
            "未" + "命名" + "_clean_basemesh",
            "纯" + "悦",
            "一号" + "机",
            "软" + "行",
            "彩" + "石",
        ]
        scan_files = (
            [
                REPO_ROOT / "run_pipeline.py",
                REPO_ROOT / "README.md",
                REPO_ROOT / "AGENTS.md",
                REPO_ROOT / "SECURITY.md",
                REPO_ROOT / "pyproject.toml",
                REPO_ROOT / ".env.example",
                REPO_ROOT / ".gitignore",
                REPO_ROOT / "knowledge" / "README.md",
                REPO_ROOT / "blender_scripts" / "LICENSE-GPL-3.0-NOTICE.md",
            ]
            + sorted((REPO_ROOT / "pipeline").glob("*.py"))
            + sorted((REPO_ROOT / "blender_scripts").glob("*.py"))
            + sorted((REPO_ROOT / "workflows").glob("*"))
            + sorted((REPO_ROOT / "templates").glob("*"))
            + sorted((REPO_ROOT / "tests").glob("*.py"))
            + sorted((REPO_ROOT / "skills").rglob("*.md"))
        )
        for fp in scan_files:
            if not fp.is_file():
                continue
            text = fp.read_text(encoding="utf-8-sig")
            for tok in forbidden_tokens:
                self.assertNotIn(tok, text, msg=f"Sensitive/private token {tok!r} found in {fp}")

        # Verify Apache-2.0 external layer never directly imports GPL Blender modules (`bpy` / `bmesh`)
        external_py = [REPO_ROOT / "run_pipeline.py"] + sorted((REPO_ROOT / "pipeline").glob("*.py")) + sorted((REPO_ROOT / "workflows").glob("*.py"))
        for fp in external_py:
            lines = fp.read_text(encoding="utf-8-sig").splitlines()
            for idx, ln in enumerate(lines, start=1):
                stripped = ln.strip()
                if stripped.startswith(("import ", "from ")):
                    self.assertNotRegex(
                        stripped,
                        r"^(import|from)\s+(bpy|bmesh|mathutils)\b",
                        msg=f"Cross-process GPL boundary violation in {fp}:{idx}: {stripped}",
                    )

        # Verify .gitignore covers local private data & assets
        gi_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for req_rule in ("projects/", "*.glb", "*.blend", ".env", "knowledge/*.jsonl"):
            self.assertIn(req_rule, gi_text)

        # Verify withheld proprietary post-retopo modules are not bundled in this open-source cut
        withheld_relpaths = (
            "pipeline/uv_bake_agent_loop.py",
            "blender_scripts/uv_bake_toolkit.py",
            "blender_scripts/uv_bake_validator.py",
            "pipeline/cloth_rig_agent_loop.py",
            "blender_scripts/cloth_rig_inspector.py",
            "blender_scripts/cloth_rig_toolkit.py",
            "blender_scripts/cloth_rig_validator.py",
        )
        for rel in withheld_relpaths:
            self.assertFalse((REPO_ROOT / rel).exists(), msg=f"Withheld proprietary module present in OSS cut: {rel}")

        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for req_heading in ("## 如何使用", "知识产权", "暂不开源"):
            self.assertIn(req_heading, readme_text)


if __name__ == "__main__":
    unittest.main()
