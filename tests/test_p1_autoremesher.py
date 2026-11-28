"""AutoRemesher control-plane contracts without an installed optional binary."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.asset_triage import _classify
from pipeline.retopo_backends import AUTOREMESHER_ROUTE, AutoRemesherAdapter, AutoRemesherError
from pipeline.topology_agent_loop import RetopoCriteria, agent_decide_repair_plan, analyze_failure_and_rollback


class AutoRemesherTests(unittest.TestCase):
    def test_timeout_caps_process_and_discards_partial_output(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.obj"
            source.write_bytes(b"source")
            output = Path(tmp) / "output.obj"
            checkpoint, active = Path(tmp) / "checkpoint", Path(tmp) / "active"
            checkpoint.write_bytes(b"checkpoint")
            active.write_bytes(b"changed")

            def stall(args, **options):
                self.assertEqual(options["timeout"], 150)
                self.assertEqual(args[-2:], ["--anisotropy", "1.0"])
                output.write_bytes(b"partial obj")
                raise subprocess.TimeoutExpired(args, options["timeout"])

            with patch("pipeline.retopo_backends.subprocess.run", side_effect=stall):
                with self.assertRaises(AutoRemesherError) as caught:
                    AutoRemesherAdapter("autoremesher").remesh(source, output, target_quads=100, timeout=500)
            self.assertEqual(caught.exception.code, "autoremesher_timeout")
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), b"source")
            result = analyze_failure_and_rollback(checkpoint, active, AUTOREMESHER_ROUTE,
                                                  False, str(caught.exception), None)
            self.assertIn("autoremesher_timeout", result["failure_classes"])
            self.assertEqual(active.read_bytes(), checkpoint.read_bytes())

    def test_installed_backend_routes_only_eligible_parts_and_can_be_banned(self):
        hard = {"welded_non_manifold_edges": 0, "welded_shells_count": 1, "thin_sheet_score": 0.30,
                "slender_aspect_ratio": 2.0, "planar_dominance_score": 0.62}
        backends = {"blender_quadriflow": True, "blender_mesh_tools": True, "autoremesher_cli": True}
        routes = []
        for name, mesh in (("plate", hard), ("ribbon", {**hard, "thin_sheet_score": 0.72})):
            tags, route, forbidden = _classify(mesh, backends)
            routes.append({"object_name": name, "asset_tags": tags, "recommended_retopo_route": route,
                           "forbidden_routes": forbidden})
        self.assertEqual([p["recommended_retopo_route"] == AUTOREMESHER_ROUTE for p in routes], [True, False])
        self.assertEqual(_classify(hard, {**backends, "autoremesher_cli": False})[1],
                         "quadriflow_sharp_flow_refine_bake")
        baseline = {"total_faces_raw": 24000, "parts_routing": routes,
                    "asset_tags": ["multi_part", "hard_surface", "thin_sheet"]}
        strategy, actions, _ = agent_decide_repair_plan(baseline, [], set(), RetopoCriteria())
        self.assertEqual(strategy, AUTOREMESHER_ROUTE)
        action = next(step for step in actions if step["action"] == "autoremesher_retopo")
        self.assertEqual(action["params"]["auto_parts"], ["plate"])
        strategy, _, _ = agent_decide_repair_plan(baseline, [], set(), RetopoCriteria(),
                                                  banned_routes={AUTOREMESHER_ROUTE})
        self.assertEqual(strategy, "quadriflow_sharp_flow_refine_bake")
