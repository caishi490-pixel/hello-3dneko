---
name: hello-3dneko-pipeline
description: >-
  End-to-end workflow for building, cleaning, rigging, AI-polishing, and
  UV/texture-optimizing animatable anime and stylized 3D characters using
  A-pose turnarounds, modular breakdown sheets, Hyper3D Rodin,
  AI image editing, and headless Blender subprocess execution.
---

# Hello 3DNeko Character & Garment Pipeline

Standard workflow for turning 2D character designs into clean, animatable, high-clarity 3D stylized assets while avoiding common multi-view reconstruction defects, GLB rigging traps, and low-resolution UV blur.

Remember the core quality bottleneck once silhouettes work: **"Three parts modeling, seven parts textures/materials/shading"** — macro silhouette alone still looks plastic and blurry until eyes/decals get dedicated UV resolution and glossy PBR glare is replaced by Toon/Cel shading.

## Stage 1: 2D Turnaround, Basemesh Proportions & Modular Breakdown Sheets

Before generating or sculpting a body mesh, anchor macro silhouette and polygon budget against the proportional guidelines in [basemesh-proportions-blueprint.md](./references/basemesh-proportions-blueprint.md) (`6.90` heads tall, `Shoulder 1.04 : Waist 0.53 : Hip 1.00`, `12–14` quad limb loops, `Face + Hands = 55.6%` of basemesh poly budget, `8` separate face material layers, zero-poly UV-painted tight stockings, `<= 4` weights/vertex).

1. **Enforce a `40°` Open A-Pose** (or extract from `90°` T-Pose masters): Keep arms abducted roughly `40°` away from the coat/skirt so automatic skinning weights do not stick sleeves onto skirts.
2. **Audit Left/Right Consistency Before 3D**: Check asymmetric elements across Front, Side, and Back views before invoking 3D generation.
3. **Generate Unoccluded Breakdown Sheets**: Produce standalone front/back sheets of layered garments and isolated props/hair pieces on solid `#FFFFFF`.
4. **Verify**: Visually inspect all turnaround angles for L/R consistency before spending paid 3D generation credits.

## Stage 2: Route Selection, Geometry Cleanup & Weight Repair

- **Mechanical & ROM Grayblock (`Blender` procedural / `Quad` basemesh)**: Best when verifying squat/walk/jump clearance. To convert a quad AI mesh into a single watertight basemesh without losing UVs, run `tris_convert_to_quads` followed by `remove_doubles(1e-4)` (UVs reside on face-corner loops).
- **Stylized Volume & Modular Props (`Rodin Gen-2.5-High`)**: Use `Raw` for sculpted silhouettes/props, `BANG` for part separation, and `Rigging` for baseline weights/clips.

When cleaning generated `.glb` assets in Blender:
1. **Smooth Without Cracking UV Seams**: Use a spatial-welded vertex graph (`round(wp, 4)`) + loop UV remap instead of naive smoothing on un-welded `.glb` meshes.
2. **Surgical Geometry Removal (`visual-cull`)**: Never hardcode 3D XYZ boxes or approve cuts from bright color renders alone. Follow [visual-surgical-cull-sop.md](./references/visual-surgical-cull-sop.md):
   - Run `python run_pipeline.py visual-cull --mode capture` to get 10×10 grid renders + exact raycast Face-ID buffers.
   - **Prefer `--mode bisect`** (`view:x0,y0->x1,y1:drop=left|right|above|below`) to slice straight macro-silhouette channels (e.g., vertical leg corridors) with automatic UV & weight interpolation, or use `--mode preview --xray-garment` + `--mode apply` for irregular cavities.
   - **Mandatory Hard FAIL Gate**: Always inspect the dark-matte body + `0.85px` orange wireframe audit (`04_after_bisect_wireframe.png` / `04_after_wireframe_xray.png`) and iterate until zero wrapped wires or floating rings remain.
3. **Strip A-Pose Shell Weight Bleed**: Remove stray `ForeArm`/`Hand`/`Finger` weights on torso/skirts and enable `use_deform_preserve_volume = True` when matching DQS setups (prefer LBS matching glTF targets).
4. **Verify**: Check manifold integrity and render headless multi-view rest + extreme ROM poses.

When teaching or checking an agent's binding decisions, use [the rigging practice lab](./references/rig-agent-learning-lab.md). Diagnose the part and joint from measured motion, change only the affected weights, and use an independently re-imported GLB to check attachment, garment coverage and package completeness.

## Stage 3: AI Aesthetic Polish, Image Generation & UV Texture Optimization

Follow [ai-polish-and-uv-optimization.md](./references/ai-polish-and-uv-optimization.md):
1. **Audit UV Density First** using Python/Pillow (`audit_uv_atlas`).
2. **Use Paired `3D Render + 2D Concept`** for coordinate-locked `images/edits`.
3. **Split Material Slots & Reallocate UVs** (`1024x1024` dedicated eye maps or head-rigged iris geometry).
4. **Apply Toon/Cel Shading & Bake Before `.glb` Export**.

## Stage 4: Operational & Environment Guardrails

See [blender-rodin-pitfalls.md](./references/blender-rodin-pitfalls.md):
- Match Principled shaders via `node.type == 'BSDF_PRINCIPLED'`.
- Never reset scenes inside a live Blender MCP session; run batch jobs via headless CLI subprocesses (`blender -b --python ...`).
- Persist `generation_id` before polling, wrap `@hyper3d/cli` cleanly, enforce zero auto-retry on paid calls, and never write keys or signed URLs to logs.

## Runnable Orchestrator & Closed-Loop Engines

Use `run_pipeline.py` in the Hello 3DNeko workspace:
- Full-chain CLI runner: `python run_pipeline.py init ...` -> `python run_pipeline.py run --project <dir> [--dry-run]`
- Closed-loop 6-pillar retopology & bake: `python run_pipeline.py retopo --input <raw.glb> --out <dir>` — see [closed-loop-retopology.md](./references/closed-loop-retopology.md).
- Autonomous UV & Baking: `python run_pipeline.py uv-bake --high <high.glb> --low <low.blend> --out <dir>` and `python run_pipeline.py experiments`.
- Visual Surgical Bisect & Face-ID Cull: `python run_pipeline.py visual-cull --input <scene.blend> --out <dir> --mode capture|bisect|preview|apply [...]` — see [visual-surgical-cull-sop.md](./references/visual-surgical-cull-sop.md).
- Stateful Agent Mesh Surgeon (Interactive Edit Mode): `python run_pipeline.py surgeon --session <dir> --action open|look|select-box|select-linked|select-backface|grow|shrink|delete|bisect|clean|undo|export [...]` — renders a 3-panel triptych (`Grid+Selection | Clay MatCap with Backface Warning | Dark-Matte Body + 0.85px Wireframe`) on every step with automatic snapshot `undo`. Always run mesh surgery **before** quad retopology whenever possible so the final topology stays clean quads.

## Outcome Evolution Rule

Route every closed-loop run into local knowledge logs (`knowledge/`):
- **Success**: Save verified parameters and recipes.
- **Failure**: Roll back cleanly and record `failure_classes` and banned signatures.
- **Partial Success**: Separate passing pillars and reusable operators from remaining blockers.
- **Total Failure**: Ban the failing `route_family` and pivot immediately to an alternative route.
