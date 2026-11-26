# SPDX-License-Identifier: Apache-2.0
"""Pure NumPy tests for P2 geometry, all hard gates, and snapshot transactions."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Run GRE QA tests with a Python interpreter with NumPy") from error

from gre.collision import PairBudgetExceeded, intersection_pairs
from gre.qa import CandidatePatch, INVARIANTS, QAProfile, continuous_orientation, evaluate, weighted_p95
from gre.snapshot import MeshSnapshot, commit, file_hash, load_snapshot, rollback, save_snapshot


def fixture(vertices=None, triangles=None, uv=True) -> MeshSnapshot:
    if vertices is None:
        vertices = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [3, 0, 1], [4, 0, 1], [3, 1, 1]]
    if triangles is None:
        triangles = [[0, 1, 2], [3, 4, 5]]
    V = np.array(vertices, dtype=np.float64)
    tri = np.array(triangles, dtype=np.intp)
    loops = np.arange(3 * len(tri), dtype=np.intp).reshape(-1, 3)
    uv_values = np.tile([[0, 0], [1, 0], [0, 1]], (len(tri), 1)).astype(np.float32)
    return MeshSnapshot(V, np.arange(1, len(V) + 1), tri, loops, np.arange(1, len(tri) + 1),
                        {"UV": uv_values} if uv else {}, dict.fromkeys(INVARIANTS, "unchanged"))


def bad_cases() -> dict[str, tuple[MeshSnapshot, CandidatePatch, QAProfile, str]]:
    base = fixture()
    profile = QAProfile(frozenset(base.vid.tolist()), 10)
    changed_uv = replace(base, hashes={**base.hashes, "uv_hash": "changed"})
    locked = base.V.copy()
    locked[0, 2] += 1e-6
    collapse = base.V.copy()
    collapse[2] = (1, 0, 0)
    flipped = base.V.copy()
    flipped[2] = (0, -1, 0)
    intersected = base.V.copy()
    intersected[3:] = [[0.5, 0.5, -0.4], [1.6, 0.5, 0.6], [0.5, 1.1, 0.6]]
    over = base.V.copy()
    over[3:, 2] += 0.2
    return {
        "a_locked": (base, CandidatePatch.between(base, replace(base, V=locked)),
                     replace(profile, editable_vids=profile.editable_vids - {1}), "locked"),
        "b_uv": (base, CandidatePatch.between(base, changed_uv), profile, "integrity"),
        "c_degenerate": (base, CandidatePatch.between(base, replace(base, V=collapse)), profile, "degenerate"),
        "d_flip": (base, CandidatePatch.between(base, replace(base, V=flipped)), profile, "flip"),
        "e_intersection": (base, CandidatePatch.between(base, replace(base, V=intersected)), profile, "self_intersection"),
        "f_displacement": (base, CandidatePatch.between(base, replace(base, V=over)),
                           replace(profile, max_displacement_m=0.1), "displacement"),
    }


class QATest(unittest.TestCase):
    def test_six_bad_candidates_are_rejected(self) -> None:
        for name, (base, patch, profile, failed) in bad_cases().items():
            with self.subTest(name=name):
                result = evaluate(base, patch, profile)
                self.assertEqual(result["status"], "rejected")
                self.assertIn(failed, result["hard_failures"])
                print(f"BAD CASE {name}: {failed} -> rejected", flush=True)

    def test_projected_determinant_detects_mid_path_flip(self) -> None:
        before = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        after = np.array([[[0, 0, 0], [-2, 0, 0], [0, -0.5, 0]]], dtype=float)
        ratio, valid = continuous_orientation(before, after, 1e-14)
        self.assertTrue(valid[0])
        self.assertLess(ratio[0], -0.1)
        self.assertGreater(np.cross(after[0, 1], after[0, 2])[2], 0)

    def test_existing_intersection_is_counted_separately(self) -> None:
        base, patch, profile, _ = bad_cases()["e_intersection"]
        V = base.V.copy()
        V[patch.vertex_ids - 1] = patch.positions
        baseline = replace(base, V=V)
        result = evaluate(baseline, CandidatePatch.between(baseline, baseline), profile)
        row = result["checks"]["self_intersection"]
        self.assertEqual((row["baseline_pairs"], row["candidate_pairs"], row["new_pairs"]), (1, 1, 0))
        self.assertEqual(row["status"], "pass")

    def test_adjacent_touch_vs_coplanar_overlap_and_budget(self) -> None:
        V = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, -2, 0], [0.5, 0, 0], [0.5, 0.5, 0]], dtype=float)
        triangles = np.array([[0, 1, 2], [1, 0, 3], [0, 4, 5]])
        hits = intersection_pairs(V, triangles, np.ones(3, dtype=bool), 1e-10, 10)
        self.assertEqual(hits, {(0, 2), (1, 2)})
        with self.assertRaises(PairBudgetExceeded):
            intersection_pairs(V, triangles, np.ones(3, dtype=bool), 1e-10, 1)
        almost_parallel = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                                    [0.25, 0.25, -1e-11], [1.25, 0.25, 1e-11], [0.25, 1.25, 1e-11]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        self.assertEqual(intersection_pairs(almost_parallel, faces, np.ones(2, dtype=bool), 1e-12, 10), {(0, 1)})
        shared_point = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, .2, 4e-9], [-1, .8, 4e-9]])
        adjacent_faces = np.array([[0, 1, 2], [0, 3, 4]])
        self.assertEqual(intersection_pairs(shared_point, adjacent_faces, np.ones(2, dtype=bool), 1e-8, 10), set())

    def test_uv_stretch_uses_log_singular_values_and_all_layers(self) -> None:
        base = fixture()
        profile = QAProfile(frozenset(base.vid.tolist()), 10)
        distorted = replace(base, V=base.V * 1.1)
        result = evaluate(base, CandidatePatch.between(base, distorted), profile)
        row = result["checks"]["uv_stretch"]["layers"]["UV"]
        self.assertAlmostEqual(row["p95_log"], np.log(1.1))
        self.assertAlmostEqual(row["max_log"], np.log(1.1))
        self.assertEqual(result["status"], "needs_review")
        second = {**base.uv_layers, "mirrored": base.uv_layers["UV"] * [-1, 1], "collapsed": np.zeros((6, 2))}
        base = replace(base, uv_layers=second)
        rows = evaluate(base, CandidatePatch.between(base, base), profile)["checks"]["uv_stretch"]["layers"]
        self.assertEqual(rows["mirrored"]["status"], "pass")
        self.assertEqual(rows["collapsed"]["reason"], "invalid_uv")
        self.assertEqual(rows["collapsed"]["invalid_uv"], 2)
        self.assertEqual(weighted_p95(np.array([1, 100]), np.array([100, 1])), 1)

    def test_volume_requires_closed_oriented_component(self) -> None:
        base = fixture([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], uv=False)
        profile = QAProfile(frozenset(base.vid.tolist()), 10)
        candidate = replace(base, V=base.V * 1.01)
        result = evaluate(base, CandidatePatch.between(base, candidate), profile)
        row = result["checks"]["area_volume"]["components"][0]
        self.assertAlmostEqual(row["volume"]["before_signed_m3"], 1 / 6)
        self.assertAlmostEqual(row["volume"]["relative_change"], 1.01**3 - 1)
        self.assertEqual(result["status"], "needs_review")
        open_base = fixture()
        opened = evaluate(open_base, CandidatePatch.between(open_base, open_base), profile)
        self.assertEqual(opened["checks"]["area_volume"]["components"][0]["volume"]["status"], "n/a")

    def test_approved_target_and_clean_are_separate(self) -> None:
        base = fixture()
        profile = QAProfile(frozenset(base.vid.tolist()), 10, target_vids=(1, 2, 3),
                            target_plane=(np.array([0.0, 0.0, 1.0]), -0.5), clean_positions=base.V)
        row = evaluate(base, CandidatePatch.between(base, base), profile)["checks"]["target_error"]
        self.assertEqual(row["approved_plane_m"]["after"]["rms"], 0.5)
        self.assertEqual(row["clean_reference_m"]["after"]["rms"], 0)

    def test_changed_protected_attributes_and_duplicate_vid_are_rejected(self) -> None:
        base = fixture()
        profile = QAProfile(frozenset(base.vid.tolist()), 10)
        changed = replace(base, hashes={**base.hashes, "protected_attribute_hash": "changed"})
        self.assertEqual(evaluate(base, CandidatePatch.between(base, changed), profile)["status"], "rejected")
        duplicate = CandidatePatch(np.array([1, 1]), base.V[:2], base.hashes)
        self.assertEqual(evaluate(base, duplicate, profile)["status"], "rejected")


class SnapshotTest(unittest.TestCase):
    def test_transaction_rejects_stale_version_and_restores_bytes(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, staged = root / "source.blend", root / "staged.blend"
            source.write_bytes(b"baseline bytes")
            staged.write_bytes(b"candidate bytes")
            sha = file_hash(source)
            meta = save_snapshot(root, "asset", fixture(), source, sha)
            restored, reread = load_snapshot(root, "asset", sha)
            self.assertEqual(meta, reread)
            np.testing.assert_array_equal(restored.V, fixture().V)
            with self.assertRaises(ValueError):
                commit(root, "asset", sha, staged, "trial.blend", "needs_review")
            receipt = commit(root, "asset", sha, staged, "trial.blend", "accepted")
            working = root / "working" / "trial.blend"
            with self.assertRaises(ValueError):
                rollback(root, receipt["audit_id"], sha)
            expected = receipt["committed_file_sha256"]
            working.write_bytes(b"intervening edit")
            with self.assertRaises(ValueError):
                rollback(root, receipt["audit_id"], expected)
            self.assertEqual(working.read_bytes(), b"intervening edit")
            working.write_bytes(staged.read_bytes())
            rollback(root, receipt["audit_id"], expected)
            self.assertEqual(working.read_bytes(), source.read_bytes())
            with self.assertRaises(ValueError):
                rollback(root, receipt["audit_id"], expected)

    def test_snapshot_hash_tampering_is_rejected(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.blend"
            source.write_bytes(b"baseline bytes")
            sha = file_hash(source)
            save_snapshot(root, "asset", fixture(), source, sha)
            metadata = root / "snapshots" / "asset" / f"{sha}.meta.json"
            data = json.loads(metadata.read_text(encoding="utf-8"))
            data["snapshot_npz_sha256"] = "wrong"
            metadata.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_snapshot(root, "asset", sha)


if __name__ == "__main__":
    unittest.main()
