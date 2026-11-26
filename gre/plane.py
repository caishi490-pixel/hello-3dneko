# SPDX-License-Identifier: Apache-2.0
"""Area-weighted orthogonal plane fitting and local projection."""

import numpy as np


PlaneFit = tuple[np.ndarray, float, float, float]


class DegeneratePlane(ValueError):
    """The selected points do not span a plane."""


def fit_plane(points: np.ndarray, weights: np.ndarray) -> PlaneFit:
    """Return unit normal, offset, fit RMS and lambda_min / lambda_middle."""
    if (points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(weights).all() or np.any(weights < 0)
            or np.count_nonzero(weights) < 3):
        raise DegeneratePlane("Expected at least three finite, positively weighted points")
    weights = weights / np.sum(weights)
    center = weights @ points
    _, singular, axes = np.linalg.svd((points - center) * np.sqrt(weights[:, None]), full_matrices=False)
    if singular[0] == 0 or singular[1] <= singular[0] * 1e-12:
        raise DegeneratePlane("Selected points are collinear or coincident")
    normal = axes[2].copy()
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    return normal, float(-(normal @ center)), float(singular[2]), float((singular[2] / singular[1])**2)


def fit_locked_rim(points: np.ndarray, weights: np.ndarray, rim: np.ndarray, rim_weights: np.ndarray,
                   area_fit: PlaneFit, h: float) -> PlaneFit | None:
    """Use the rim only when it provides a flat target consistent with the whole region."""
    try:
        fit = fit_plane(rim, rim_weights)
    except DegeneratePlane:
        return None
    normal, d, _, _ = fit
    area_normal, _, area_rms, _ = area_fit
    origin_distance = float(rim[0] @ normal + d)
    # Require the complete perimeter to support the plane, including sparsely sampled parts.
    rim_error = np.max(np.abs((rim - rim[0]) @ normal + origin_distance))
    if rim_error > 0.02 * h or abs(normal @ area_normal) < np.cos(np.deg2rad(3)):
        return None
    residual = (points - rim[0]) @ normal + origin_distance
    rim_region_rms = float(np.sqrt(np.average(residual**2, weights=weights)))
    if rim_region_rms > 1.05 * area_rms + 1e-12 * h:
        return None
    return fit


def project_region(V: np.ndarray, region: np.ndarray, ring: np.ndarray, n: np.ndarray,
                   d: float, falloff: np.ndarray, *, passes: int = 1) -> np.ndarray:
    """Apply repeated projection onto one fixed plane without copying the mesh per pass."""
    projected = V.copy()
    if passes > 1:
        # Each virtual pass retains (1 − falloff) of the signed ring residual.
        falloff = 1 - (1 - falloff)**passes
    for indices, weights in ((region, 1.0), (ring, falloff[:, None])):
        signed = V[indices] @ n + d
        projected[indices] -= signed[:, None] * n * weights
    return projected
