"""Deterministic, conserved polygon budgets for independent mesh parts."""

import math


class BudgetInfeasible(ValueError):
    code = "budget_infeasible"

    def __init__(self, required, available):
        self.required = required
        self.available = available
        super().__init__(f"{self.code}: required={required}, available={available}")


def allocate_object_budgets(
    parts_metrics, total_target_faces, *, min_per_mesh=300, strict_min_floor=False,
    area_exponent=0.65,
) -> dict[str, int]:
    """Mix normalized area-power and original face-count shares, rounding by remainder."""
    if total_target_faces < 0 or min_per_mesh < 0 or not math.isfinite(area_exponent) or area_exponent <= 0:
        raise ValueError("Budgets must be nonnegative and area_exponent must be positive and finite")
    if not parts_metrics:
        return {}
    parts = sorted(parts_metrics, key=lambda item: item["part_id"])
    ids = [item["part_id"] for item in parts]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate part_id in budget allocation")
    count = len(parts)
    if count * min_per_mesh > total_target_faces:
        if strict_min_floor:
            raise BudgetInfeasible(count * min_per_mesh, total_target_faces)
        min_per_mesh = min(min_per_mesh, total_target_faces // count, max(16, total_target_faces // (2 * count)))
    remainder = total_target_faces - count * min_per_mesh
    areas = [max(0.0, float(p.get("surface_area", 0))) ** area_exponent for p in parts]
    faces = [max(0, int(p.get("faces_raw", 0))) for p in parts]
    area_total, face_total = sum(areas), sum(faces)
    exact = [
        remainder * ((a / area_total if area_total else 1 / count)
                     + (f / face_total if face_total else 1 / count)) / 2
        for a, f in zip(areas, faces)
    ]
    extras = [int(n) for n in exact]
    for index in sorted(range(count), key=lambda i: (extras[i] - exact[i], ids[i]))[:remainder - sum(extras)]:
        extras[index] += 1
    return {part_id: min_per_mesh + extra for part_id, extra in zip(ids, extras)}
