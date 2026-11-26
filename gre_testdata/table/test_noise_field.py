# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests runnable from pytest or Blender's bundled Python."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_field import distances, normal_offsets, transition_weights


def test_geodesic_rings() -> None:
    graph = {0: [(1, 1)], 1: [(0, 1), (2, 2)], 2: [(1, 2), (3, 3)], 3: [(2, 3)]}
    assert distances(graph, {0}) == {0: 0, 1: 1, 2: 3, 3: 6}
    assert transition_weights(graph, {0}, 0) == {0: 0, 1: 1, 2: 1, 3: 1}
    weights = transition_weights(graph, {0}, 2)
    np.testing.assert_allclose([weights[index] for index in range(4)], [0, 2 / 27, 0.5, 1])
    assert transition_weights(graph, {0}, 3) == dict.fromkeys(graph, 0)


def test_seed_peak_and_support() -> None:
    points = np.column_stack((np.linspace(0, 1, 30), np.zeros((30, 2))))
    weights = np.linspace(0, 1, 30)
    wave = normal_offsets(points, weights, 0.02, 0.32, 42)
    assert wave[0] == 0
    assert np.isclose(np.max(np.abs(wave)), 0.02)
    np.testing.assert_array_equal(wave, normal_offsets(points, weights, 0.02, 0.32, 42))
    assert not np.array_equal(wave, normal_offsets(points, weights, 0.02, 0.32, 43))
    assert not np.array_equal(wave, normal_offsets(points, weights, 0.02, 0.42, 42))


if __name__ == "__main__":
    test_geodesic_rings()
    test_seed_peak_and_support()
    print("Noise-field unit tests: 2 passed")
