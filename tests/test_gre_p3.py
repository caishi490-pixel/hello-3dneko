# SPDX-License-Identifier: Apache-2.0
"""Pure NumPy tests for P3 detection, relation fitting and shared features."""

import math
import unittest

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Run P3 unit tests with a Python interpreter that has NumPy") from error

from gre.ransac import detect_plane, face_neighbors, iteration_count
from gre.relations import Infeasible, Observation, Plane, Relation, project_junctions, relation_error, solve_planes


class DetectionTest(unittest.TestCase):
    def test_adaptive_iteration_count_and_ceiling(self) -> None:
        self.assertEqual(iteration_count(.999, 1, 3, 256), 1)
        self.assertEqual(iteration_count(.999, .5, 3, 256), math.ceil(math.log(.001)/math.log(.875)))
        self.assertEqual(iteration_count(.999, .1, 3, 18), 18)
        self.assertEqual(iteration_count(.999, 0, 3, 18), 18)

    def test_neighbors_use_shared_edges(self) -> None:
        self.assertEqual(face_neighbors(((0, 1, 2), (2, 1, 3), (0, 4, 5))), ((1,), (0,), ()))

    def test_area_normal_consistency_and_connected_support(self) -> None:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0],
                           [5, 0, 0], [5, 1, 0], [6, 0, 0]], dtype=float)
        normals = np.tile([0., 0., 1.], (len(points), 1))
        normals[4] = [1, 0, 0]
        graph = ((1, 2), (0, 3), (0, 3), (1, 2, 4), (3,), (6, 7), (5, 7), (5, 6))
        areas = np.array([2., 2., 2., 2., 1., 1., 1., 1.])
        for seed in (0, 1, 7, 42, 100):
            result = detect_plane(points, normals, areas, graph, seed=seed, distance_m=1e-6)
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result.inliers, [0, 1, 2, 3])
            self.assertAlmostEqual(result.coverage, 8/12)
            self.assertGreater(result.minimum_normal_cosine, .99)
        self.assertIsNone(detect_plane(points, normals, areas, graph, seed=1, distance_m=1e-6, minimum_coverage=.8))


class RelationsTest(unittest.TestCase):
    @staticmethod
    def observations() -> dict[str, Observation]:
        z0 = np.array([[-1, -1, 0], [-1, 1, 0.01], [1, -1, 0], [1, 1, 0.01]], dtype=float)
        z1 = z0 + [0, 0, 0.31]
        x = z0[:, [2, 1, 0]]
        return {"bottom": Observation(Plane(np.array([0., 0., 1.]), 0.), z0, np.ones(4)),
                "top": Observation(Plane(np.array([0., 0., 1.]), -0.31), z1, np.ones(4)),
                "side": Observation(Plane(np.array([1., 0., 0.]), 0.), x, np.ones(4))}

    def test_joint_unit_normals_parallel_perpendicular_and_offset(self) -> None:
        links = (Relation("parallel", "bottom", "top"), Relation("perpendicular", "bottom", "side"),
                 Relation("offset", "bottom", "top", .3))
        planes, _ = solve_planes(self.observations(), links)
        for plane in planes.values():
            self.assertAlmostEqual(np.linalg.norm(plane.normal), 1)
        for link in links:
            result = relation_error(planes, link)
            self.assertLess(result["deviation_deg"], .01)
            self.assertLess(result.get("offset_error_m", 0.), 1e-5)

    def test_conflicting_relationships_are_infeasible(self) -> None:
        links = (Relation("parallel", "bottom", "top"), Relation("perpendicular", "bottom", "top"))
        with self.assertRaises(Infeasible):
            solve_planes(self.observations(), links)

    def test_shared_line_and_corner_use_all_planes_and_keep_locks(self) -> None:
        planes = {axis: Plane(row, -float(amount)) for axis, row, amount in zip("xyz", np.eye(3), (1, 2, 3))}
        V = np.array([[10, 20, 30], [9, 8, 7], [4, 5, 6], [-0.0, 2, 4]], dtype=float)
        memberships = {0: set("xy"), 1: set("xyz"), 2: set("z")}
        groups = {frozenset("xy"), frozenset("xyz")}
        moved, summary = project_junctions(V, np.arange(3), np.array([1., 1., .5]), memberships, planes, groups)
        np.testing.assert_allclose(moved[:3], [[1, 2, 30], [1, 2, 3], [4, 5, 4.5]])
        self.assertEqual(moved[3].tobytes(), V[3].tobytes())
        self.assertEqual(summary["line_core_vertices"], 1)
        self.assertEqual(summary["corner_core_vertices"], 1)

    def test_shared_planes_must_be_approved_and_feasible(self) -> None:
        V = np.array([[0., 0., 0.]])
        planes = {"a": Plane(np.array([1., 0., 0.]), 0), "b": Plane(np.array([1., 0., 0.]), -1)}
        for approved in (set(), {frozenset(("a", "b"))}):
            with self.assertRaises(Infeasible):
                project_junctions(V, np.array([0]), np.array([1.]), {0: {"a", "b"}}, planes, approved)


if __name__ == "__main__":
    unittest.main()
