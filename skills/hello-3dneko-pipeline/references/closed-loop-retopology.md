# Closed-Loop Retopology, 6-Pillar Validation & Outcome Evolution

Empirical playbook distilled from benchmarking closed-loop autonomous retopology across stylized rigid props (`corset_belt_raw.glb`, `500,000` raw tris -> `17,392` quads) and thin multi-layer pleated garments (`pleated_dress_raw.glb`, `500,000` raw tris -> `15,589` quads) via `run_pipeline.py retopo` and `run_pipeline.py experiments`.

## 1. Outcome Routing Rule (Mandatory)

Every closed-loop run branches deterministically based on independent 6-pillar validation:
- **SUCCESS (`score == 100.0`)**: Record verified parameters, pre/post steps, and 6-pillar metrics into local knowledge ledgers (`knowledge/learned_recipes.jsonl`).
- **FAILED (Attempt Rejected)**: Roll back to the pre-attempt `.blend`/`.glb` snapshot and record `failure_classes` and banned parameter signatures.
- **PARTIAL_SUCCESS (`score >= 50.0` at budget limit)**: Preserve passing capabilities cleanly apart from unresolved blockers.
- **TOTAL_FAILURE (`score < 50.0` or runtime crash)**: Ban the failing `route_family` and pivot immediately to an alternative algorithm family (`quadriflow_route` -> `voxel_route` -> `decimate_route` -> `multi_part_shell_split`).

---

## 2. Success Recipes Crystallized

### Recipe A: Solid / Sculpted Props (`quadriflow_smooth_post_clean_bake`)
Benchmark result: `500,000` tris -> `17,392` quads (`34,780` GLB tris, `99.98%` `.blend` quads, `82.65%` `.glb`-recoverable quads, `83.65%` valence-4, `0` non-manifold/degenerate faces, `0.084%` mean surface drift, `1.0` bake coverage, metallic ORM preserved):
1. **Pre-sanitize**: `remove_doubles(1e-4)`, remove loose verts/edges, prune tiny debris shells, recalc normals.
2. **Quadriflow Remesh (`use_preserve_sharp=False, smooth_normals=True, use_mesh_symmetry=False`)**: Produces clean edge loops on asymmetric props without symmetry distortion.
3. **Post-Retopo Sanitize & Manifold Heal**: `remove_doubles(2e-4)`, collapse edges `< 1e-4`, delete faces `< 1e-9`, purge pinch verts / shared-3-vert GLB diagonal conflicts, recalc normals.

### Recipe B: Thin Hollow Multi-Layer Garments (`quadriflow_healed_sharp_shrink_bake`)
Benchmark result: `500,000` tris -> `15,589` faces (`31,135` GLB tris, `99.72%` `.blend` quads, `69.19%` `.glb`-recoverable quads, `72.49%` valence-4, `0` welded non-manifold edges, `0` degenerate faces, `0` debris, `0.1303%` mean surface drift, `1.5173%` bbox drift, `1.0` bake coverage):
1. **Pre-Heal Pinch Vertices & Unsafe-Fill Guard**: Purge faces incident to multi-face edges (`>2`), bow-tie edges, and pinch vertices (`not v.is_manifold`) BEFORE `Quadriflow`. Test each boundary hole fill inside `holes_fill` and revert any cap face introducing new non-manifold topology.
2. **Sharp High-Budget Quadriflow + Reference Shrinkwrap**: `use_preserve_sharp=True, smooth_normals=False` at high target budget limits vertical shrinkage on thin hems (`3.30% -> 1.52%` bbox drift) while preserving `valence4_ratio >= 0.72`.
3. **GLTF-Triangulated Collision Heal**: Purge folded 2-edge/2-face pockets, face pairs sharing `>= 3` vertices, and `KDTree`-matched `FIXED`/`SHORT_EDGE` + `remove_doubles(1.15e-4)` sliver conflicts; prune isolated micro-shells (`<= 48` verts).
4. **Full ORM Metallic + Backface-Safe Cycles Bake**:
   - Smart UV unwrap (`angle_limit_deg=86.0, island_margin=0.004`).
   - Emit-bake `Metallic` BEFORE zeroing `Metallic=0.0` on reference for albedo `DIFFUSE`, then pack `ORM` (`G=Roughness, B=Metallic`).
   - Tight thin-ruffle bake cage (`cage_extrusion=0.003, max_ray_distance=0.018`), clamp backface normals (`B < 0.68 -> (0.5, 0.5, 1.0)`), and dilate black ray-miss RGB texels cleanly without faking `Alpha`.
5. **Independent 6-Pillar Validation**: Re-open exported `.glb` in a fresh Blender session; check geometric fidelity (`BVHTree`), quad & valence-4 flow, zero non-manifold/degen/debris, poly budget, UV/material coverage, and inspect rendered PNGs (`QCWireMod.use_even_offset = False`).

---

## 3. Recorded Failure Causes (`Do Not Repeat`)

- **Hidden GLB-Only Non-Manifold Edges (`shared >= 3 verts` Quad Diagonals)**: Always purge `>= 3`-shared-vertex neighbors and run `KDTree` `FIXED`/`SHORT_EDGE` + `1.15e-4` weld checks in `.blend` before export.
- **Unconditional `holes_fill` Breaking Quadriflow**: Always purge non-manifold pinch vertices first and roll back unsafe cap faces.
- **Post-Voxel Edge Collapse Punching Holes**: Never call edge-collapsing `sanitize_mesh` right after `Voxel Remesh`.
- **Tangent-Normal Inversion & Cycles Metallic Albedo Blackout**: Use tight cages, clamp `B < 0.68`, and emit-bake `Metallic` before albedo bake.
- **glTF Import Name Collision**: Track objects via `obj.as_pointer()`, not `.name` strings.

---

> **Open-Source Boundary Note**: This public reference stops at closed-loop Agent Retopology (`retopo`). Downstream closed-loop modules (`uv-bake`, `cloth-rig`, and multi-pose automatic weighting) remain unverified/withheld as proprietary IP of the project author.
