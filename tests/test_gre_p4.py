# SPDX-License-Identifier: Apache-2.0
"""Pure NumPy tests for local filtering, masking and the cumulative budget."""

import unittest

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise unittest.SkipTest('Run P4 unit tests with a Python interpreter that has NumPy') from error

from gre.denoise import bilateral_vertex, laplacian, pack_neighbors, robust_normals, sharp_vertices, taubin


class DenoiseTest(unittest.TestCase):
    def setUp(self):
        self.V = np.array([[0.,0.,1.],[-1.,0.,-0.],[1.,0.,0.]])
        self.adj = pack_neighbors([[1,2],[0],[0]])
        self.mask = np.array([1.,0.,0.])

    def test_laplacian_stops_at_last_accepted_iteration(self):
        fit = laplacian(self.V,self.adj,self.mask,.5,3,budget=.6)
        self.assertTrue(fit.stopped_by_budget)
        self.assertEqual(fit.iterations,1)
        self.assertAlmostEqual(fit.proposed_peak_m,.75)
        self.assertAlmostEqual(fit.V[0,2],.5)
        self.assertEqual(fit.V[1:].tobytes(),self.V[1:].tobytes())

    def test_taubin_two_steps_preserve_frozen_words(self):
        fit = taubin(self.V,self.adj,self.mask,.5,-.4,1)
        self.assertAlmostEqual(fit.V[0,2],.7)
        self.assertEqual(fit.V[1:].tobytes(),self.V[1:].tobytes())

    def test_bilateral_matches_normal_height_formula_and_budget(self):
        points = np.array([[0.,0.,0.],[-1.,0.,.1],[1.,0.,.3]])
        normals = np.broadcast_to([0,0,1.],points.shape)
        fit = bilateral_vertex(points,normals,self.adj,1.,.2,1,self.mask)
        w = np.exp(-.5*(np.array([1.01,1.09])+(np.array([.1,.3])/.2)**2))
        self.assertAlmostEqual(fit.V[0,2],float(w@np.array([.1,.3])/w.sum()))
        self.assertEqual(fit.V[0,:2].tolist(),[0,0])
        rejected = bilateral_vertex(points,normals,self.adj,1.,.2,2,self.mask,budget=.05)
        self.assertTrue(rejected.stopped_by_budget)
        self.assertEqual(rejected.iterations,0)
        self.assertEqual(rejected.V.tobytes(),points.tobytes())

    def test_normal_smoothing_is_area_weighted_and_finite(self):
        V = np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,2],[0.,0,0]])
        normals = robust_normals(V,np.array([[0,1,2],[0,3,1]]),pack_neighbors([[1,2,3],[0,2,3],[0,1],[0,1],[]]))
        base = np.array([0,2,1])/np.sqrt(5)
        y_weight = 2 * np.exp(-.5*((1-base[1])/.12)**2)
        expected = np.array([0,2,1]) + 3*base + y_weight*np.array([0,1,0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(normals[0],expected)
        np.testing.assert_array_equal(normals[-1],np.zeros(3))

    def test_sharp_closed_edges_and_open_edges(self):
        faces = ((0,2,1),(0,1,3),(1,2,3),(2,0,3))
        normals = np.tile(np.array([0.,0,1.]),(4,1))
        self.assertEqual(sharp_vertices(faces,normals,45),set())
        normals[0] = [1,0,0]
        self.assertEqual(sharp_vertices(faces,normals,45),{0,1,2})
        self.assertEqual(sharp_vertices(((0,1,2),),normals[:1],45),{0,1,2})


if __name__ == '__main__':
    unittest.main()
