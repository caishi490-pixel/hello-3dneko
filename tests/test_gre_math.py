# SPDX-License-Identifier: Apache-2.0
"""Pure NumPy tests. Run with unittest or pytest, without Blender's bpy module."""

import unittest

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Run the GRE suite with a Python runtime that has NumPy") from error

from gre.geometry import (adjacency, bbox_diagonal, closed_locked_rim, distance_statistics,
                          face_edges, face_geometry, geodesic_distances, partition_region)
from gre.plane import DegeneratePlane, fit_locked_rim, fit_plane, project_region


class GeometryTest(unittest.TestCase):
    def test_analysis_mapping_and_areas(self) -> None:
        V = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float)
        n, area, vertex = face_geometry(V, np.array([[0, 1, 2], [0, 2, 3]]), np.array([1, 1]), 2)
        np.testing.assert_allclose(n, [[0, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(area, [0, 2])
        np.testing.assert_allclose(vertex, [2 / 3, 1 / 3, 2 / 3, 1 / 3])
        self.assertAlmostEqual(bbox_diagonal(V), np.sqrt(5))

    def test_unique_edges_and_median_metric(self) -> None:
        V = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float)
        edges = face_edges(((0, 1, 2, 3),))
        self.assertEqual(edges, {(0, 1), (1, 2), (2, 3), (0, 3)})
        graph, h = adjacency(V, edges)
        self.assertEqual(h, 1.5)
        self.assertEqual(graph[0], [(1, 2.0), (3, 1.0)])
        with self.assertRaises(ValueError):
            adjacency(V, {(0, 0)})

    def test_rim_must_be_single_closed_loop_with_all_vertices_locked(self) -> None:
        faces = ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
        locked = {0, 1, 2, 3}
        np.testing.assert_array_equal(closed_locked_rim(faces, locked), [0, 1, 2, 3])
        self.assertIsNone(closed_locked_rim(faces, {0, 1, 2}))
        self.assertIsNone(closed_locked_rim(faces, locked | {4}))
        self.assertIsNone(closed_locked_rim(faces + ((5, 6, 7),), locked | {5, 6, 7}))
        self.assertIsNone(closed_locked_rim(faces + (faces[0],), locked))

    def test_geodesic_and_rings_with_disconnected_components(self) -> None:
        graph = {0: [(1, 1)], 1: [(0, 1), (2, 2)], 2: [(1, 2), (3, 3)], 3: [(2, 3)],
                 4: [(5, 2)], 5: [(4, 2)], 6: [(7, 2)], 7: [(6, 2)]}
        self.assertEqual(geodesic_distances(graph, {0}), {0: 0, 1: 1, 2: 3, 3: 6})
        core, ring, weights = partition_region(graph, {0, 6}, 2)
        np.testing.assert_array_equal(core, [3, 4, 5])
        np.testing.assert_array_equal(ring, [1, 2])
        np.testing.assert_allclose(weights, [2 / 27, 0.5])
        core, ring, _ = partition_region(graph, {0, 6}, 0)
        np.testing.assert_array_equal(core, [1, 2, 3, 4, 5, 7])
        self.assertEqual(ring.size, 0)

    def test_clean_reference_error_statistics(self) -> None:
        stats = distance_statistics(np.array([[3, 4, 0], [0, 0, 0]], dtype=float))
        self.assertAlmostEqual(stats["rms"], np.sqrt(12.5))
        self.assertAlmostEqual(stats["p95"], 4.75)
        self.assertEqual(stats["max"], 5.0)


class PlaneTest(unittest.TestCase):
    def test_rotated_plane_and_zero_weight_outlier(self) -> None:
        points = np.array([[0, 0, 1], [2, 0, 3], [0, 2, 1], [2, 2, 3], [3, 3, -4]], dtype=float)
        normal, d, rms, ratio = fit_plane(points, np.array([1, 2, 3, 4, 0]))
        np.testing.assert_allclose(points[:4] @ normal + d, 0, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1)
        self.assertAlmostEqual(rms, 0)
        self.assertAlmostEqual(ratio, 0)

    def test_nonplanarity_and_degenerate_inputs(self) -> None:
        isotropic = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        _, _, rms, ratio = fit_plane(isotropic, np.ones(8))
        self.assertAlmostEqual(rms, 1)
        self.assertAlmostEqual(ratio, 1)
        with self.assertRaises(DegeneratePlane):
            fit_plane(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float), np.ones(3))
        with self.assertRaises(DegeneratePlane):
            fit_plane(isotropic, -np.ones(8))

    def test_rim_anchor_checks_flatness_and_agreement(self) -> None:
        rim = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
        points = np.vstack((rim, [[-0.5, -0.5, 0.06], [0.5, -0.5, -0.06],
                                  [0.5, 0.5, 0.06], [-0.5, 0.5, -0.06]]))
        weights = np.ones(len(points))
        area_fit = fit_plane(points, weights)
        anchor = fit_locked_rim(points, weights, rim, weights[:4], area_fit, 0.1)
        self.assertIsNotNone(anchor)
        np.testing.assert_allclose(anchor[0], [0, 0, 1])
        self.assertAlmostEqual(anchor[1], 0)

        warped = rim.copy()
        warped[0, 2] = 0.02
        self.assertIsNone(fit_locked_rim(points, weights, warped, weights[:4], area_fit, 0.1))
        tilted = rim.copy()
        tilted[:, 2] = 0.1 * rim[:, 0]
        self.assertIsNone(fit_locked_rim(points, weights, tilted, weights[:4], area_fit, 0.1))
        self.assertIsNone(fit_locked_rim(points, weights, rim + [0, 0, 0.05], weights[:4], area_fit, 0.1))
        self.assertIsNone(fit_locked_rim(points, weights, rim[:2], weights[:2], area_fit, 0.1))

    def test_project_only_selected_core_and_weighted_ring(self) -> None:
        V = np.array([[0, 0, 2], [0, 1, 2], [0, 1, 2], [-0.0, 0, 2]], dtype=float)
        moved = project_region(V, np.array([0]), np.array([1]), np.array([0.0, 0.0, 1.0]), -1, np.array([0.25]))
        np.testing.assert_allclose(moved[:, 2], [1, 1.75, 2, 2])
        self.assertEqual(moved[2:].tobytes(), V[2:].tobytes())
        np.testing.assert_array_equal(V[:, 2], [2, 2, 2, 2])

    def test_combined_projection_matches_repeated_passes(self) -> None:
        V = np.array([[2, 0, 2], [0, 1, 2], [0, 1, 3], [1, 3, 1], [-0.0, 0, 2]], dtype=float)
        indices = (np.array([0]), np.array([1, 2, 3]))
        n = np.array([1, 0, 1]) / np.sqrt(2)
        falloff = np.array([0.0, 0.25, 1.0])
        sequential = V
        for _ in range(16):
            sequential = project_region(sequential, *indices, n, -0.5, falloff)
        combined = project_region(V, *indices, n, -0.5, falloff, passes=16)
        np.testing.assert_allclose(combined, sequential, atol=1e-14, rtol=0)
        self.assertEqual(combined[-1].tobytes(), V[-1].tobytes())


if __name__ == "__main__":
    unittest.main()
