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
            self.assertEqual(strat, "quadriflow_sharp_flow_refine_bake")
            self.assertTrue(any(a["action"] == "refine_quad_flow" for a in actions))
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

            # 1b. high_surface_deviation escalates from sharp-flow to smooth-flow when sharp-flow is banned
            strat_dev, dev_actions, _ = agent_decide_repair_plan(
                baseline,
                [{"error_analysis": {"failure_classes": ["high_surface_deviation"]}}],
                {"quadriflow_sharp_flow_refine_bake", "quadriflow_healed_post_clean_bake"},
                RetopoCriteria(),
            )
            self.assertEqual(strat_dev, "quadriflow_smooth_flow_refine_bake")
            self.assertEqual(get_strategy_route(strat_dev), "quadriflow_route")
            self.assertTrue(any(a["action"] == "shrinkwrap_to_reference" for a in dev_actions))
            self.assertTrue(any(a["action"] == "refine_quad_flow" for a in dev_actions))

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

    def test_uv_bake_loop_rollback_cli_and_experiment_ledger(self) -> None:
        from pipeline.experiment_ledger import compute_ledger_summary, load_experiments, record_experiment
        from pipeline.uv_bake_agent_loop import (
            UVBakeCriteria,
            agent_decide_uv_bake_plan,
            analyze_uv_failure_and_rollback,
            get_uv_strategy_route,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pre = root / "pre.blend"
            active = root / "active.blend"
            pre.write_bytes(b"CLEAN_UV_PRE")
            active.write_bytes(b"BAD_UV_OVP")

            diag = analyze_uv_failure_and_rollback(
                pre,
                active,
                "seam_angle_conformal_tight_bake",
                toolkit_ok=True,
                toolkit_log="",
                verdict={
                    "passed": False,
                    "score": 33.33,
                    "blocking_failures": [
                        "UVOverlap: overlap_area=15.5% > 0.8%",
                        "BakeAOBlackout: ao_blackout=31.5% > 10.0%",
                    ],
                },
            )
            self.assertEqual(active.read_bytes(), b"CLEAN_UV_PRE")
            self.assertIn("uv_overlap", diag["failure_classes"])
            self.assertIn("bake_blackout", diag["failure_classes"])

            baseline = {"meshes": [{"dims": [1.0, 0.8, 1.8]}]}
            strat, actions, _ = agent_decide_uv_bake_plan(
                baseline,
                [{"error_analysis": diag}],
                {"seam_angle_conformal_tight_bake"},
                UVBakeCriteria(),
            )
            self.assertEqual(strat, "hybrid_seams_conformal_tight_bake")
            self.assertEqual(get_uv_strategy_route(strat), "hybrid_unwrap_route")
            self.assertTrue(any(a["action"] == "bake_pbr_maps" and a["params"].get("use_cage") for a in actions))

            kdir = root / "knowledge"
            record_experiment(
                kdir,
                task_type="uv_bake",
                asset="corset.blend",
                work_dir=root / "w1",
                autonomous_completed=True,
                human_interventions=0,
                iterations_used=3,
                elapsed_seconds=106.2,
                final_score=100.0,
                passed=True,
                winning_strategy_id="smart_project_packed_tight_bake",
                seed=42,
            )
            rows = load_experiments(kdir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["auto_rework_count"], 2)
            summary = compute_ledger_summary(rows)
            self.assertEqual(summary["autonomous_completion_rate_pct"], 100.0)
            self.assertTrue((kdir / "EXPERIMENT_SCORECARD.md").is_file())

            hi = root / "high.glb"
            lo = root / "low.glb"
            hi.write_bytes(b"glTF")
            lo.write_bytes(b"glTF")
            uv_proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "uv-bake",
                    "--high",
                    str(hi),
                    "--low",
                    str(lo),
                    "--out",
                    str(root / "cli_uvb"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(uv_proc.returncode, 0, msg=uv_proc.stderr)
            self.assertEqual(json.loads(uv_proc.stdout)["status"], "dry_run")

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

    def test_cloth_rig_planner_rollback_and_cli_dry_run(self) -> None:
        from pipeline.cloth_rig_agent_loop import (  # noqa: E402
            ClothRigCriteria,
            agent_decide_cloth_rig_plan,
            get_cloth_strategy_route,
        )

        # Corset clean path -> surface_polyinterp_clean_fit
        insp_corset = {
            "garment_analysis": {
                "garment_type": "waist_corset_belt",
                "has_sleeves": False,
                "has_offcenter_lower_pendulum": True,
            }
        }
        s1, a1, _ = agent_decide_cloth_rig_plan(insp_corset, [], set(), ClothRigCriteria())
        self.assertEqual(s1, "surface_polyinterp_clean_fit")
        self.assertEqual(get_cloth_strategy_route(s1), "surface_transfer_clean_route")
        self.assertTrue(any(act["action"] == "match_pose_and_fit" for act in a1))

        # Blocking clipping/overstretch promotes proxy hull analytical route when surface route is banned
        insp_dress = {
            "garment_analysis": {
                "garment_type": "full_dress",
                "has_sleeves": True,
                "has_offcenter_lower_pendulum": False,
            }
        }
        fail_hist = [
            {
                "error_analysis": {
                    "failure_classes": ["pose_body_clipping", "skinning_spike"],
                }
            }
        ]
        s2, a2, _ = agent_decide_cloth_rig_plan(
            insp_dress,
            fail_hist,
            {"surface_polyinterp_clean_fit"},
            ClothRigCriteria(),
            banned_routes={"surface_transfer_clean_route"},
        )
        self.assertEqual(s2, "inflated_proxy_capsule_transfer")
        self.assertEqual(get_cloth_strategy_route(s2), "proxy_hull_skirt_route")
        self.assertTrue(
            any(act["action"] == "repair_weight_anomalies" and act["params"].get("skirt_analytical_field") for act in a2)
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            body = root / "body.glb"
            garm = root / "dress.glb"
            body.write_bytes(b"glTF")
            garm.write_bytes(b"glTF")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "cloth-rig",
                    "--body",
                    str(body),
                    "--garment",
                    str(garm),
                    "--out",
                    str(root / "cloth_dry"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(payload["strategy_id"], "surface_polyinterp_clean_fit")
            self.assertEqual(payload["route_family"], "surface_transfer_clean_route")

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

    def test_knowledge_recall_actually_reorders_planners_from_jsonl(self) -> None:
        from pipeline.cloth_rig_agent_loop import ClothRigCriteria, agent_decide_cloth_rig_plan
        from pipeline.topology_agent_loop import RetopoCriteria, agent_decide_repair_plan
        from pipeline.uv_bake_agent_loop import UVBakeCriteria, agent_decide_uv_bake_plan

        with tempfile.TemporaryDirectory() as td:
            kdir = Path(td) / "knowledge"
            kdir.mkdir(parents=True, exist_ok=True)
            (kdir / "learned_recipes.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"task_type": "retopo", "strategy_id": "quadriflow_smooth_flow_refine_bake", "score": 100.0}),
                        json.dumps({"task_type": "uv_bake", "strategy_id": "smart_project_packed_tight_bake", "score": 100.0}),
                        json.dumps({"task_type": "cloth_rig", "strategy_id": "inflated_proxy_capsule_transfer", "score": 98.0, "garment_type": "full_dress"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (kdir / "failure_lessons.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"task_type": "retopo", "banned_strategy": "quadriflow_sharp_flow_refine_bake"}),
                        json.dumps({"task_type": "retopo", "banned_strategy": "quadriflow_sharp_flow_refine_bake"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            retopo_base = {"junk_objects": [], "total_faces_raw": 500000, "meshes": [{"faces_raw": 500000, "raw_quad_ratio": 0.0, "dims": [1.0, 1.0, 1.0]}]}
            r_sid, _, r_rat = agent_decide_repair_plan(retopo_base, [], set(), RetopoCriteria(), knowledge_dir=kdir)
            self.assertEqual(r_sid, "quadriflow_smooth_flow_refine_bake")
            self.assertIn("Knowledge Recall", r_rat)

            uv_base = {"meshes": [{"dims": [1.0, 1.0, 1.0]}]}
            u_sid, _, u_rat = agent_decide_uv_bake_plan(uv_base, [], set(), UVBakeCriteria(), knowledge_dir=kdir)
            self.assertEqual(u_sid, "smart_project_packed_tight_bake")
            self.assertIn("Knowledge Recall", u_rat)

            cloth_insp = {"garment_analysis": {"garment_type": "full_dress"}}
            c_sid, _, c_rat = agent_decide_cloth_rig_plan(cloth_insp, [], set(), ClothRigCriteria(), knowledge_dir=kdir)
            self.assertEqual(c_sid, "inflated_proxy_capsule_transfer")
            self.assertIn("Knowledge Recall", c_rat)

    def test_live_blender_geometry_flow_and_culler_regression(self) -> None:
        """Real headless Blender regression test on a procedural synthetic 3D mesh (zero external private assets)."""
        import shutil
        from pipeline.config import DEFAULT_BLENDER_EXE

        blender_bin = DEFAULT_BLENDER_EXE if Path(DEFAULT_BLENDER_EXE).is_file() else (shutil.which(DEFAULT_BLENDER_EXE) or "")
        if not blender_bin:
            self.skipTest("Blender executable not installed in this environment.")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gen_script = root / "make_synthetic_mesh.py"
            raw_glb = root / "synthetic_garment.glb"
            gen_script.write_text(
                "\n".join(
                    [
                        "import bpy, bmesh, sys",
                        "out_glb = sys.argv[sys.argv.index('--') + 1]",
                        "bpy.ops.wm.read_factory_settings(use_empty=True)",
                        "bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.45)",
                        "obj = bpy.context.active_object",
                        "obj.name = 'SyntheticGarment'",
                        "bpy.ops.export_scene.gltf(filepath=out_glb, export_format='GLB')",
                    ]
                ),
                encoding="utf-8",
            )
            p_gen = subprocess.run([blender_bin, "-b", "--python-exit-code", "1", "--python", str(gen_script), "--", str(raw_glb)], capture_output=True, text=True, check=False)
            self.assertEqual(p_gen.returncode, 0, msg=p_gen.stderr)
            self.assertTrue(raw_glb.is_file())

            # 1. Run topology_inspector.py on the real GLB and verify quad_orthogonality_mean & pole_cluster_pct
            insp_json = root / "inspect.json"
            p_insp = subprocess.run(
                [blender_bin, "-b", "--python-exit-code", "1", "--python", str(REPO_ROOT / "blender_scripts" / "topology_inspector.py"), "--", "--input", str(raw_glb), "--out-json", str(insp_json)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_insp.returncode, 0, msg=p_insp.stderr)
            insp_data = json.loads(insp_json.read_text(encoding="utf-8"))
            self.assertEqual(len(insp_data["meshes"]), 1)
            self.assertIn("quad_orthogonality_mean", insp_data["meshes"][0])
            self.assertIn("pole_cluster_pct", insp_data["meshes"][0])

            # 2. Run topology_toolkit.py with normal-aware shrinkwrap + refine_quad_flow
            plan_path = root / "plan.json"
            out_blend = root / "refined.blend"
            out_glb = root / "refined.glb"
            exec_log = root / "exec_log.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "input_path": str(raw_glb),
                        "reference_glb": str(raw_glb),
                        "textures_dir": str(root / "tex"),
                        "output_blend": str(out_blend),
                        "output_glb": str(out_glb),
                        "execution_log": str(exec_log),
                        "actions": [
                            {"action": "weld_and_quadify", "params": {"weld_dist": 1e-4}},
                            {"action": "shrinkwrap_to_reference", "params": {"offset": 0.0, "normal_aware": True}},
                            {"action": "refine_quad_flow", "params": {"iterations": 2, "tangential_factor": 0.3, "reproject": True}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            p_tk = subprocess.run(
                [blender_bin, "-b", "--python-exit-code", "1", "--python", str(REPO_ROOT / "blender_scripts" / "topology_toolkit.py"), "--", "--plan", str(plan_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_tk.returncode, 0, msg=p_tk.stderr)
            self.assertTrue(out_blend.is_file())
            self.assertTrue(out_glb.is_file())
            tk_rep = json.loads(exec_log.read_text(encoding="utf-8"))
            self.assertEqual(tk_rep["status"], "executed")
            self.assertTrue(tk_rep["steps"][1]["result"]["normal_aware"])
            self.assertEqual(tk_rep["steps"][2]["action"], "refine_quad_flow")

            # 3. Run visual_face_picker_culler.py end-to-end (capture -> preview -> apply)
            vp_dir = root / "visual_pick_demo"
            vp_script = REPO_ROOT / "blender_scripts" / "visual_face_picker_culler.py"
            p_cap = subprocess.run(
                [
                    blender_bin,
                    "-b",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(vp_script),
                    "--",
                    "--input",
                    str(raw_glb),
                    "--out-dir",
                    str(vp_dir),
                    "--mode",
                    "capture",
                    "--views",
                    "front,back",
                    "--res",
                    "128",
                    "--id-res",
                    "64",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_cap.returncode, 0, msg=p_cap.stderr)
            self.assertTrue((vp_dir / "capture_meta.json").is_file())
            self.assertTrue((vp_dir / "01_grid_capture_sheet.png").is_file())
            self.assertTrue((vp_dir / "face_id_front.npy").is_file())

            p_prev = subprocess.run(
                [
                    blender_bin,
                    "-b",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(vp_script),
                    "--",
                    "--input",
                    str(raw_glb),
                    "--out-dir",
                    str(vp_dir),
                    "--mode",
                    "preview",
                    "--xray-garment",
                    "--xray-depth-m",
                    "0.25",
                    "--views",
                    "front,back",
                    "--res",
                    "128",
                    "--pick",
                    "front:D3:G6",
                    "--exclude",
                    "back:0.0,0.0,1.0,1.0",
                    "--grow-hops",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_prev.returncode, 0, msg=p_prev.stderr)
            mask_json = vp_dir / "selected_faces.json"
            self.assertTrue(mask_json.is_file())
            self.assertTrue((vp_dir / "02_preview_red_mask.png").is_file())
            mask_data = json.loads(mask_json.read_text(encoding="utf-8"))
            self.assertGreater(mask_data["selected_count"], 0)
            self.assertTrue(mask_data["xray_garment"])

            p_app = subprocess.run(
                [
                    blender_bin,
                    "-b",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(vp_script),
                    "--",
                    "--input",
                    str(raw_glb),
                    "--out-dir",
                    str(vp_dir),
                    "--mode",
                    "apply",
                    "--views",
                    "front,back",
                    "--res",
                    "128",
                    "--mask-json",
                    str(mask_json),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_app.returncode, 0, msg=p_app.stderr)
            self.assertTrue((vp_dir / "03_after_cull.png").is_file())
            self.assertTrue((vp_dir / "04_after_wireframe_xray.png").is_file())
            self.assertTrue((vp_dir / "synthetic_garment_visual_culled.glb").is_file())
            app_rep = json.loads((vp_dir / "visual_cull_report.json").read_text(encoding="utf-8"))
            self.assertLess(app_rep["deletion"]["post_faces"], app_rep["deletion"]["pre_faces"])

            # 4. Run surgical bisect mode on the synthetic mesh and verify clean plane slicing
            p_bi = subprocess.run(
                [
                    blender_bin,
                    "-b",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(vp_script),
                    "--",
                    "--input",
                    str(raw_glb),
                    "--out-dir",
                    str(vp_dir),
                    "--mode",
                    "bisect",
                    "--views",
                    "front,back",
                    "--res",
                    "128",
                    "--bisect",
                    "front:0.40,0.20->0.40,0.80:drop=right",
                    "--bisect",
                    "front:0.60,0.20->0.60,0.80:drop=left",
                    "--exclude",
                    "back:0.0,0.0,1.0,1.0",
                    "--bisect-depth-m",
                    "0.35",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_bi.returncode, 0, msg=p_bi.stderr)
            self.assertTrue((vp_dir / "04_after_bisect_wireframe.png").is_file())
            self.assertTrue((vp_dir / "synthetic_garment_surgical_bisected.glb").is_file())
            bi_rep = json.loads((vp_dir / "surgical_bisect_report.json").read_text(encoding="utf-8"))
            self.assertGreater(bi_rep["bisect_stats"]["sliced_edges"], 0)
            self.assertGreater(bi_rep["bisect_stats"]["deleted_faces"], 0)

            # 5. Verify run_pipeline.py visual-cull CLI wrapper dry-run
            p_cli = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "visual-cull",
                    "--input",
                    str(raw_glb),
                    "--out",
                    str(vp_dir),
                    "--mode",
                    "bisect",
                    "--bisect",
                    "front:0.40,0.20->0.40,0.80:drop=right",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_cli.returncode, 0, msg=p_cli.stderr)
            cli_payload = json.loads(p_cli.stdout)
            self.assertEqual(cli_payload["status"], "dry_run")
            self.assertEqual(cli_payload["command"], "visual-cull")
            self.assertEqual(cli_payload["mode"], "bisect")

            # 6. Verify stateful agent_mesh_surgeon (open -> select-box -> grow -> delete -> undo -> clean -> export)
            sg_dir = root / "surgeon_sess"
            sg_script = REPO_ROOT / "blender_scripts" / "agent_mesh_surgeon.py"
            for sg_args in [
                ["--action", "open", "--input", str(raw_glb), "--view", "front", "--res", "128", "--id-res", "64"],
                ["--action", "select-box", "--box", "D3:F6", "--res", "128", "--id-res", "64"],
                ["--action", "grow", "--steps", "1", "--res", "128", "--id-res", "64"],
                ["--action", "delete", "--res", "128", "--id-res", "64"],
                ["--action", "undo", "--res", "128", "--id-res", "64"],
                ["--action", "clean", "--quadify", "--res", "128", "--id-res", "64"],
                ["--action", "export", "--out-blend", str(sg_dir / "out.blend"), "--out-glb", str(sg_dir / "out.glb")],
            ]:
                p_sg = subprocess.run(
                    [blender_bin, "-b", "--python-exit-code", "1", "--python", str(sg_script), "--", "--session", str(sg_dir), *sg_args],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(p_sg.returncode, 0, msg=f"surgeon {sg_args} failed: {p_sg.stderr}")
            self.assertTrue((sg_dir / "renders" / "inspect_triptych.png").is_file())
            self.assertTrue((sg_dir / "out.glb").is_file())

            p_sg_cli = subprocess.run(
                [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "surgeon", "--session", str(sg_dir), "--action", "undo", "--dry-run"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_sg_cli.returncode, 0, msg=p_sg_cli.stderr)
            self.assertEqual(json.loads(p_sg_cli.stdout)["command"], "surgeon")

            # 7. Verify procedural Agent Practice Dojo (setup + baseline grade FAIL + CLI dry-run)
            dj_dir = root / "dojo_test"
            dj_script = REPO_ROOT / "blender_scripts" / "surgeon_dojo.py"
            p_dj_setup = subprocess.run(
                [blender_bin, "-b", "--python-exit-code", "1", "--python", str(dj_script), "--", "--dojo-dir", str(dj_dir), "--action", "setup"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_dj_setup.returncode, 0, msg=p_dj_setup.stderr)
            self.assertTrue((dj_dir / "dojo_practice_dress.blend").is_file())
            self.assertTrue((dj_dir / "dojo_truth.json").is_file())

            p_dj_grade = subprocess.run(
                [
                    blender_bin,
                    "-b",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(dj_script),
                    "--",
                    "--dojo-dir",
                    str(dj_dir),
                    "--action",
                    "grade",
                    "--candidate",
                    str(dj_dir / "dojo_practice_dress.blend"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_dj_grade.returncode, 0, msg=p_dj_grade.stderr)
            sc = json.loads((dj_dir / "dojo_scorecard.json").read_text(encoding="utf-8"))
            self.assertEqual(sc["status"], "FAIL")

            p_dj_cli = subprocess.run(
                [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "dojo", "--dojo-dir", str(dj_dir), "--action", "setup", "--dry-run"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_dj_cli.returncode, 0, msg=p_dj_cli.stderr)
            self.assertEqual(json.loads(p_dj_cli.stdout)["command"], "dojo")

            # 8. Verify retopo_500k exam setup & grader (fast low-tris CI variant)
            dj_ret = root / "dojo_retopo_test"
            p_ret_setup = subprocess.run(
                [
                    blender_bin,
                    "-b",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(dj_script),
                    "--",
                    "--dojo-dir",
                    str(dj_ret),
                    "--exam",
                    "retopo_500k",
                    "--action",
                    "setup",
                    "--target-tris",
                    "4000",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_ret_setup.returncode, 0, msg=p_ret_setup.stderr)
            self.assertTrue((dj_ret / "dojo_500k_highpoly.glb").is_file())
            p_ret_grade = subprocess.run(
                [
                    blender_bin,
                    "-b",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(dj_script),
                    "--",
                    "--dojo-dir",
                    str(dj_ret),
                    "--exam",
                    "retopo_500k",
                    "--action",
                    "grade",
                    "--candidate",
                    str(dj_ret / "dojo_500k_highpoly.blend"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p_ret_grade.returncode, 0, msg=p_ret_grade.stderr)
            ret_sc = json.loads((dj_ret / "dojo_scorecard.json").read_text(encoding="utf-8"))
            self.assertEqual(ret_sc["exam"], "retopo_500k")
            self.assertEqual(ret_sc["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
