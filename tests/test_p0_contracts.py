"""P0 regression contracts on generated geometry and real local Blender."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from pipeline.asset_triage import _classify
from pipeline.budget_allocator import BudgetInfeasible, allocate_object_budgets
from pipeline.asset_ids import mesh_part_id
from pipeline.config import DEFAULT_BLENDER_EXE, require_blender
from pipeline.glb_postprocess import _read_glb, repair_zero_tangents
from pipeline.knowledge_recall import load_strategy_priors
from pipeline.retopo_backends import AUTOREMESHER_ROUTE, PyMeshLabAdapter
from pipeline.sandbox import export_uv_preserving_mesh, export_welded_geometry_obj
from pipeline.topology_agent_loop import (
    TOOLKIT_SCRIPT, RetopoCriteria, _run_blender, analyze_failure_and_rollback, run_closed_loop_retopo,
)


ROOT = Path(__file__).resolve().parents[1]


class P0ControlTests(unittest.TestCase):
    def test_triage_calibrates_thin_curved_and_nonmanifold_boundaries(self):
        backends = {"blender_quadriflow": True, "blender_mesh_tools": True}
        curved = {"welded_non_manifold_edges": 0, "welded_shells_count": 1,
                  "thin_sheet_score": 0.47, "slender_aspect_ratio": 2.0, "planar_dominance_score": 0.484}
        self.assertNotIn("hard_surface", _classify(curved, backends)[0])
        for score in (0.48, 0.51):
            tags, route, forbidden = _classify({**curved, "thin_sheet_score": score}, backends)
            self.assertIn("thin_sheet", tags)
            self.assertEqual(route, "quadriflow_smooth_flow_refine_bake")
            self.assertIn("instant_meshes", forbidden)
        for count in (1, 15):
            tags, route, forbidden = _classify(
                {**curved, "welded_non_manifold_edges": count, "welded_shells_count": 2}, backends,
            )
            self.assertEqual(tags, ["minor_nonmanifold"])
            self.assertEqual(route, "quadriflow_healed_post_clean_bake")
            self.assertEqual(forbidden, ["raw_quadriflow"])
        tags, route, forbidden = _classify(
            {**curved, "welded_non_manifold_edges": 16, "welded_shells_count": 2}, backends,
        )
        self.assertTrue({"dirty_nonmanifold", "multi_strand", "require_gap_preservation"} <= set(tags))
        self.assertEqual(route, "per_shell_strand_repair_or_rebuild")
        self.assertEqual(set(forbidden), {"raw_quadriflow", "voxel_seal_quadriflow_bake", "single_shell_alpha_wrap"})

    def test_budget_conserved_and_infeasible_hard_floor(self):
        parts = [{"part_id": str(i), "surface_area": i + 1, "faces_raw": 3 * i + 1} for i in range(60)]
        budgets = allocate_object_budgets(parts, 16000)
        self.assertEqual(sum(budgets.values()), 16000)
        self.assertGreater(budgets["59"], budgets["0"])
        self.assertEqual(budgets, allocate_object_budgets(list(reversed(parts)), 16000))
        self.assertEqual(sum(allocate_object_budgets(parts, 7).values()), 7)
        with self.assertRaises(BudgetInfeasible) as ctx:
            allocate_object_budgets(parts, 16000, strict_min_floor=True)
        self.assertEqual((ctx.exception.code, ctx.exception.required, ctx.exception.available),
                         ("budget_infeasible", 18000, 16000))

    def test_tags_prevent_cross_category_recipes_and_failure_prior(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            records = [
                {"strategy_id": "single", "asset_tags": ["single_prop", "hard_surface"], "score": 100},
                {"strategy_id": "envelope", "asset_tags": ["multi_part", "single_shell_envelope"], "score": 100},
                {"strategy_id": "strands", "asset_tags": ["multi_part", "require_gap_preservation"], "score": 94},
            ]
            (folder / "learned_recipes.jsonl").write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            failures = [{"banned_strategy": "strands", "asset_tags": ["single_prop"]}] * 5
            (folder / "failure_lessons.jsonl").write_text("\n".join(map(json.dumps, failures)), encoding="utf-8")
            priors = load_strategy_priors(folder, "retopo", {"multi_part", "require_gap_preservation", "hard_surface"})
            self.assertEqual(priors["preferred_order"], ["strands"])
            self.assertEqual(set(priors["scores"]), {"strands"})
            self.assertEqual(priors["failures_read"], 0)

    def test_pack_glb_repairs_interleaved_zero_tangent(self):
        data = struct.pack("<14f", 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, 1)
        doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(data)}],
               "bufferViews": [{"buffer": 0, "byteLength": len(data), "byteStride": 28}],
               "accessors": [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"},
                             {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 2, "type": "VEC4"}],
               "meshes": [{"primitives": [{"attributes": {"NORMAL": 0, "TANGENT": 1}}]}]}
        metadata = json.dumps(doc).encode()
        metadata += b" " * (-len(metadata) % 4)
        payload = struct.pack("<I4s", len(metadata), b"JSON") + metadata + struct.pack("<I4s", len(data), b"BIN\0") + data
        with tempfile.TemporaryDirectory() as temp:
            source, target = (Path(temp) / name for name in ("in.glb", "out.glb"))
            source.write_bytes(struct.pack("<4sII", b"glTF", 2, len(payload) + 12) + payload)
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_pipeline.py"), "pack-glb", "--input", str(source), "--out", str(target)],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout)["zero_tangents_repaired"], 1)
            fixed = _read_glb(target)[1][1]
            self.assertEqual(struct.unpack_from("<4f", fixed, 12), (0, 1, 0, -1))
            self.assertEqual(fixed[:12] + fixed[28:], data[:12] + data[28:])
            self.assertEqual(repair_zero_tangents(target)["zero_tangents_repaired"], 0)


class P0BlenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blender = require_blender(DEFAULT_BLENDER_EXE)

    def fixture(self, mode, out):
        proc = subprocess.run(
            [self.blender, "-b", "--python-exit-code", "1", "--python", str(ROOT / "tests" / "p0_blender_fixtures.py"),
             "--", mode, str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-1600:] + proc.stderr[-2400:])

    def test_case_02b_gap_merge_and_small_gasket_loss_trigger_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture("geometry", folder)
            gate = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
            checkpoint, active = folder / "checkpoint", folder / "active"
            checkpoint.write_bytes(b"reference checkpoint")
            active.write_bytes(b"candidate state")
            verdict = {"passed": False, "blocking_failures": gate["issues"], "score": 83.33}
            analyze_failure_and_rollback(checkpoint, active, "clean_weld_quadify", True, "", verdict)
            self.assertEqual(active.read_bytes(), checkpoint.read_bytes())

    def test_autoremesher_rejects_missing_gasket_with_unchanged_shell_count_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture("autoremesher_shell_rollback", temp)

    def test_uv_seam_and_geometry_exchange_with_deep_output(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture("exchange_source", folder)
            dest = folder.joinpath(*(["深层输出目录" * 8] * 5), "seam.blend")
            export_uv_preserving_mesh(folder / "seam.blend", dest, blender_exe=self.blender)
            self.assertGreater(len(str(dest)), 260)
            published = folder / "published"
            published.mkdir()
            (published / "seam.blend").write_bytes(dest.read_bytes())
            self.fixture("exchange_verify", published)
            obj = export_welded_geometry_obj(folder / "seam.blend", folder / "welded.obj", blender_exe=self.blender)
            self.assertNotIn("\nvt ", obj.read_text(encoding="utf-8"))
            self.assertEqual(len(json.loads((folder / "weld_report.json").read_bytes())["shell_samples"]), 1)

    def test_ten_color_atlas_glb_roundtrip_and_bake_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture("atlas_source", folder)
            output = folder.joinpath(*(["多件烘焙" * 11] * 6))
            plan = {
                "input_path": str(folder / "low.blend"), "reference_glb": str(folder / "high.blend"),
                "textures_dir": str(output / "textures"), "output_blend": str(output / "c.blend"),
                "output_glb": str(output / "c.glb"), "execution_log": str(output / "exec.json"),
                "actions": [
                    {"action": "decimate_and_quadify", "params": {"target_faces": 20,
                                                                  "per_mesh_targets": {f"Part_{i:02d}": 2 for i in range(10)}}},
                    {"action": "unwrap_and_bake_textures", "params": {"texture_size": 256, "margin": 4, "bake_samples": 1}},
                ],
            }
            plan_path = folder / "plan.json"
            plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
            ok, log = _run_blender(self.blender, TOOLKIT_SCRIPT, ["--plan", str(plan_path)], folder / "atlas.log")
            self.assertTrue(ok, log[-2800:])
            report = json.loads((output / "exec.json").read_bytes())
            self.assertEqual(report["junk_objects"], [])
            faces = report["steps"][0]["result"]["decimated"]
            self.assertEqual(sum(item["faces"] for item in faces), 10)
            result = report["steps"][1]["result"]["baked_pbr"]
            self.assertEqual(result["bake_calls"], 5)
            self.assertEqual(set(result["face_ranges"]), {mesh_part_id(f"Part_{i:02d}") for i in range(10)})
            self.assertLess(result["elapsed_seconds"], 120.0)
            self.assertEqual(len(list((output / "textures").glob("*_baked_diffuse.png"))), 1)
            self.assertTrue((output / "c.blend").stat().st_size > 0)
            (folder / "c.glb").write_bytes((output / "c.glb").read_bytes())
            gltf = json.loads(_read_glb(folder / "c.glb")[0][1])
            material = gltf["materials"][0]
            ao = material["occlusionTexture"]["index"]
            orm = material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
            self.assertEqual(gltf["textures"][ao]["source"], gltf["textures"][orm]["source"])
            self.fixture("atlas_verify", folder)

    def test_double_layer_and_known_slope_normal_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture("normal", temp)

    def test_unconfirmed_tangent_sign_counts_only_valid_correct_layer_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture("unconfirmed_tangent_sign", temp)

    def test_retopo_closed_loop_preserves_500_face_clasp_with_local_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture("multipart_small_clasp", folder)
            output = folder / "retopo"
            result = run_closed_loop_retopo(
                folder / "multipart.blend", output, RetopoCriteria(min_faces=800, max_faces=3200),
                max_iterations=2, blender_exe=self.blender, knowledge_dir=folder / "knowledge",
            )
            self.assertEqual(result["status"], "delivered", result.get("attempt_history"))
            baseline = result["baseline_metrics"]
            self.assertEqual({p["object_name"]: p["faces_raw"] for p in baseline["parts_routing"]},
                             {"Main": 5000, "Clasp": 500})
            self.assertIn("thin_sheet", baseline["asset_tags"])
            verdict = result["final_verdict"]
            coverage = verdict["pillars"]["1_geometric_fidelity"]["shell_validation"]
            self.assertEqual(coverage["shell_coverage_missing"], [])
            self.assertEqual(coverage["shells_out"], 2)
            self.assertIn("Clasp:0", {row["shell_id"] for row in coverage["local_coverage"]})
            for path in (output / "attempts").glob("*/repair_plan.json"):
                plan = json.loads(path.read_bytes())
                self.assertFalse(any("voxel" in step["action"] for step in plan["actions"]))
            exec_log = next((output / "attempts").glob("*/toolkit_exec.json"))
            steps = json.loads(exec_log.read_bytes())["steps"]
            meshes = next(step["result"]["quadriflow"] for step in steps if step["action"] == "quadriflow_retopo")
            clasp = next(item for item in meshes if item["mesh"] == "Clasp")
            self.assertEqual(clasp["local_fallbacks"][0]["reason"], "small_part")
            recipes = (folder / "knowledge" / "learned_recipes.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertIn("thin_sheet", json.loads(recipes[-1])["asset_tags"])

    def test_no_optix_cycles_fallback_and_export_junk(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture("cpu", temp)

    def test_triage_cli_routes_mixed_asset_per_part_and_detects_backends(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture("triage", folder)
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_pipeline.py"), "triage", "--input", str(folder / "mixed.blend"),
                 "--out", str(folder / "triage")],
                capture_output=True, encoding="utf-8", errors="replace", timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            profile = json.loads((folder / "triage" / "asset_profile.json").read_bytes())
            self.assertEqual(profile["schema_version"], "2.1")
            self.assertIn("welded_non_manifold_edges", profile["scene_summary"])
            self.assertNotIn("welded_nm_edges", profile["scene_summary"])
            self.assertTrue(profile["available_backends"]["blender_quadriflow"])
            self.assertEqual(profile["available_backends"]["pymeshlab_worker"], PyMeshLabAdapter.discover() is not None)
            self.assertIn("CPU", profile["available_backends"]["cycles_devices_detected"])
            self.assertEqual(len({part["part_id"] for part in profile["parts"]}), 3)
            armor, tassel, ribbon = profile["parts"]
            self.assertIn("hard_surface", armor["asset_tags"])
            expected = (AUTOREMESHER_ROUTE if profile["available_backends"]["autoremesher_cli"]
                        else "quadriflow_sharp_flow_refine_bake")
            self.assertEqual(armor["recommended_retopo_route"], expected)
            self.assertEqual(armor["forbidden_routes"], [])
            self.assertIn("slender_ribbon", ribbon["asset_tags"])
            self.assertEqual(ribbon["recommended_retopo_route"], "quadriflow_smooth_flow_refine_bake")
            self.assertIn("instant_meshes", ribbon["forbidden_routes"])
            self.assertIn("require_gap_preservation", tassel["asset_tags"])
            self.assertEqual(tassel["recommended_retopo_route"], "per_shell_strand_repair_or_rebuild")
            self.assertIn("single_shell_alpha_wrap", tassel["forbidden_routes"])

    def test_pymeshlab_qecd_round_trips_two_uv_layers_materials_and_seam(self):
        if PyMeshLabAdapter.discover() is None:
            self.skipTest("Configure the optional PyMeshLab Python runtime")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture("pymeshlab_seam_source", folder)
            output = folder / "retopo"
            result = run_closed_loop_retopo(
                folder / "seam.blend", output, RetopoCriteria(min_faces=1, max_faces=360),
                max_iterations=1, blender_exe=self.blender, knowledge_dir=folder / "knowledge",
                pymeshlab_operation="qecd-with-texture",
            )
            self.assertEqual(result["status"], "delivered", result.get("attempt_history"))
            self.assertEqual(result["strategy_id"], "pymeshlab_qecd_with_texture")
            self.assertEqual(result["final_verdict"]["pillars"]["4_polycount_budget"]["eval_faces"], 360)
            self.fixture("pymeshlab_seam_verify", folder)

    def test_pymeshlab_nonmanifold_guard_rolls_back_before_hole_filter(self):
        if PyMeshLabAdapter.discover() is None:
            self.skipTest("Configure the optional PyMeshLab Python runtime")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.fixture("pymeshlab_nonmanifold_source", folder)
            source = folder / "nonmanifold.blend"
            output = folder / "retopo"
            result = run_closed_loop_retopo(
                source, output, max_iterations=1, blender_exe=self.blender,
                knowledge_dir=folder / "knowledge", pymeshlab_operation="close-holes",
            )
            failures = result["attempt_history"][0]["error_analysis"]["failure_classes"]
            self.assertIn("pymeshlab_nonmanifold", failures)
            self.assertEqual(source.read_bytes(), (output / "checkpoints" / "active_baseline.blend").read_bytes())
            self.assertEqual(list(output.glob("attempts/*/*candidate*")), [])
