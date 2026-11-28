"""PyMeshLab route policy in the control process."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.retopo_backends import permits_alpha_wrap
from pipeline.topology_agent_loop import run_closed_loop_retopo


class PyMeshLabPolicyTests(unittest.TestCase):
    def test_alpha_wrap_requires_explicit_envelope_and_blocks_separation_before_worker(self):
        envelope = {"single_shell_envelope": True}
        self.assertTrue(permits_alpha_wrap(envelope))
        with TemporaryDirectory() as temp, patch("pipeline.topology_agent_loop.PyMeshLabAdapter.discover") as probe:
            output = Path(temp) / "blocked"
            for contract in (
                None,
                {**envelope, "separation_groups": [{"id": "hair", "shell_ids": ["Hair:0"]}]},
                {**envelope, "gap_rays": [{"start": [0, 0, 0], "end": [0, 0, 1]}]},
                {**envelope, "asset_tags": ["require_gap_preservation"]},
            ):
                result = run_closed_loop_retopo(Path(temp) / "unused.blend", output,
                                                shell_contract=contract, pymeshlab_operation="alpha-wrap")
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["reason"], "pymeshlab_envelope_requires_opt_in")
            probe.assert_not_called()
            self.assertFalse((output / "attempts").exists())
