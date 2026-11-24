# Visual Surgical Bisect & Face-ID Cull SOP (For Multimodal Agents)

Use `python run_pipeline.py visual-cull` whenever an AI-generated or retopologized garment contains unwanted internal webbing, fused leg pouches, armpit bridges, or stray interior walls that would clip during rigging/animation.

**Never hardcode 3D XYZ bounding boxes.** Always drive surgery from rendered 10×10 grid images and verify with the mandatory **dark-matte body + thin orange wireframe audit** (`04_after_bisect_wireframe.png` / `04_after_wireframe_xray.png`).

---

## 1. Choose Your Surgical Mode

| Situation | Recommended Mode | Why |
| --- | --- | --- |
| Clean vertical leg channels, hem trims, flat separation along macro silhouette lines (**Preferred**) | **`--mode bisect`** | Uses `bmesh.ops.bisect_plane` along screen-projected macro lines. Produces **ruler-straight cut edges**, preserves **interpolated UVs & vertex weights**, and avoids jagged triangle edges or stray micro-rings. |
| Irregular organic blobs or non-planar cavities where straight planes cannot isolate the defect | **`--mode preview --xray-garment`** → **`--mode apply`** | Combines multi-layer X-Ray raycasting + sub-pixel face-centroid sweep + 3D-welded cut-adjacent orphan island cleanup. |

---

## 2. Standard Closed-Loop Workflow (3 Steps)

### Step 1 — Capture 10×10 Grid Views & Exact Face-ID Buffers
Always start with `capture` so the workspace has `capture_meta.json`, `01_grid_capture_sheet.png`, `grid_<view>.png`, and occlusion-aware `face_id_<view>.npy`:

```bash
python run_pipeline.py visual-cull \
  --input path/to/fitted_garment.blend \
  --out path/to/work_dir \
  --mode capture \
  --views front,front_skirt,bottom_up,back
```

Inspect `grid_front_skirt.png` and `grid_bottom_up.png`:
- Origin `(0.0, 0.0)` is **top-left**, `(1.0, 1.0)` is **bottom-right**.
- Columns `A..J` map to `x = 0.0..1.0` (`0.1` each); rows `0..9` map to `y = 0.0..1.0` (`0.1` each).

---

### Step 2A — Preferred Route: Surgical Plane Bisect (`--mode bisect`)
Define 2D knife lines directly from the grid image using:
```text
view:x0,y0->x1,y1:drop=left|right|above|below
```
- **Multiple `--bisect` planes form a half-space intersection**: each plane first slices a straight cut seam (without deleting outside the intersection), then only faces inside the `drop` side of **all** planes (and within `--bisect-depth-m` from the front surface) are dropped.
- **Always add `--exclude "back:0.0,0.0,1.0,1.0"`** when cutting front/inner leg pouches so outer back-skirt layers stay 100% untouched.

Example (slicing a clean vertical leg channel while preserving front apron, side ruffles, and back skirt):
```bash
python run_pipeline.py visual-cull \
  --input path/to/fitted_garment.blend \
  --out path/to/work_dir \
  --mode bisect \
  --bisect "front_skirt:0.39,0.38->0.39,0.65:drop=right" \
  --bisect "front_skirt:0.61,0.38->0.61,0.65:drop=left" \
  --bisect "front_skirt:0.35,0.385->0.65,0.385:drop=below" \
  --exclude "back:0.0,0.0,1.0,1.0" \
  --bisect-depth-m 0.22 \
  --orphan-max-pct 3.0
```

Depth guideline (`--bisect-depth-m`):
- `0.12–0.14` m covers front wall only (may leave back wall of a thigh tube behind the leg).
- `0.20–0.24` m cleanly spans front apron clearance + full thigh diameter while stopping before outer back skirt layers.

---

### Step 2B — Alternative Route: X-Ray Face-ID Pick (`preview` → `apply`)
When cutting irregular regions:
1. Run `preview --xray-garment` with `--pick` (`view:x0,y0,x1,y1` or `view:E4:F5`) and `--exclude` protection boxes:
```bash
python run_pipeline.py visual-cull \
  --input path/to/fitted_garment.blend \
  --out path/to/work_dir \
  --mode preview \
  --xray-garment \
  --xray-depth-m 0.12 \
  --pick "front_skirt:0.39,0.385,0.61,0.61" \
  --pick "bottom_up:0.39,0.40,0.61,0.60" \
  --exclude "back:0.0,0.0,1.0,1.0" \
  --exclude "front_skirt:0.0,0.0,1.0,0.38" \
  --grow-hops 1
```
2. Visually inspect `02_preview_red_mask.png` to confirm bright red covers only the target defect and never spills onto outer ruffles or back layers.
3. Execute UV-seam-safe deletion + 3D-welded cut-adjacent orphan cleanup:
```bash
python run_pipeline.py visual-cull \
  --input path/to/fitted_garment.blend \
  --out path/to/work_dir \
  --mode apply \
  --mask-json path/to/work_dir/selected_faces.json \
  --orphan-max-pct 3.0
```

---

## 3. Mandatory Verification Gate (Hard FAIL Law)

**Never sign off using bright color renders (`03_after_cull.png` / `03_after_bisect.png`) alone.** Thin 1–2 mm residual rings or back-side webbing easily blend into bright textures or stockings and only explode once bones move.

Before marking a cut as `PASS`, the Agent **MUST** inspect:
1. **`04_after_bisect_wireframe.png`** (or `04_after_wireframe_xray.png`):
   - Body/occluders are rendered as **dark matte charcoal**; garment is rendered as **0.85px glowing orange wireframe**.
   - **Hard FAIL**: Any orange wire ring wrapping across a limb, floating fragment in the cut channel, or unintended hole in the back skirt / outer ruffles.
   - If FAIL, adjust `--bisect` line / `--bisect-depth-m` (or `--pick` / `--xray-depth-m`) and re-run (up to 3 iterations) until the dark-matte limb silhouettes are **100% wire-free**.
2. **`surgical_bisect_report.json`** (or `visual_cull_report.json`):
   - Confirm `post_faces < pre_faces` and check `pruned_orphan_islands` & `sliced_edges`.
3. **UV-Seam Safety**:
   - The script never welds the exported mesh and only prunes 3D-welded orphan islands that directly touch the cut boundary (`<= orphan_max_pct`), guaranteeing untouched UV islands, sleeves, and accessories remain intact.

---

## 4. Stateful Edit-Mode Workbench (`run_pipeline.py surgeon`)

When a mesh has **double-walled inner shells, flipped black normals, or complex localized defects** that need iterative inspection (like a human working in Blender Edit Mode), use the stateful **`surgeon`** workbench instead of one-shot cuts.

**Pipeline Order Rule**: Whenever cleaning a raw AI garment, run `surgeon` **BEFORE** quad retopology (`run_pipeline.py retopo`) so the final retopologized mesh has clean, unbroken quad loops instead of shredded cut triangles.

Every `surgeon` action automatically updates `<session>/renders/inspect_triptych.png`:
- **Left Panel**: 10×10 `A0..J9` Grid + Bright Red current selection
- **Middle Panel**: Clay MatCap (body hidden; backfacing inner walls tinted dark crimson so flipped normals & hidden double-walls pop out immediately)
- **Right Panel**: Dark-Matte Body + `0.85px` Orange Garment Wireframe

### Interactive Command Sequence Example
```bash
# 1. Open session (auto-saves step_000 snapshot + triptych)
python run_pipeline.py surgeon --session work/sg1 --action open --input garment.blend --view front_skirt --zoom 1.4

# 2. Zoom in or switch view to inspect inner lining closely
python run_pipeline.py surgeon --session work/sg1 --action look --view bottom_up --zoom 1.8

# 3. Select inner pouch / linked shell / backfacing inner wall (supports --op set|add|sub)
python run_pipeline.py surgeon --session work/sg1 --action select-box --box "0.38,0.40,0.62,0.60" --xray --depth-m 0.14
python run_pipeline.py surgeon --session work/sg1 --action select-linked --point "E5" --angle-deg 38 --op add
python run_pipeline.py surgeon --session work/sg1 --action select-backface --box "0.35,0.35,0.65,0.65" --op add
python run_pipeline.py surgeon --session work/sg1 --action grow --steps 1

# 4. Delete selected faces (auto-prunes cut-adjacent floating rings & saves snapshot)
python run_pipeline.py surgeon --session work/sg1 --action delete --orphan-max-pct 3.0

# 5. Or slice razor-straight planes in current view
python run_pipeline.py surgeon --session work/sg1 --action bisect \
  --line "0.39,0.38->0.39,0.65:drop=right" \
  --line "0.61,0.38->0.61,0.65:drop=left" \
  --line "0.35,0.385->0.65,0.385:drop=below" \
  --depth-m 0.22

# 6. Mistake? Instant 1-second rollback to previous snapshot
python run_pipeline.py surgeon --session work/sg1 --action undo

# 7. Clean degenerate zero-area slivers + recalculate outward normals (+ optional --quadify)
python run_pipeline.py surgeon --session work/sg1 --action clean --quadify

# 8. Export final clean .blend & .glb
python run_pipeline.py surgeon --session work/sg1 --action export --out-blend out.blend --out-glb out.glb
```

---

## 5. Self-Contained Agent Practice Dojo (`run_pipeline.py dojo`)

Before operating on production assets (or when testing a new Agent setup with zero external private models), run the built-in **Surgeon Dojo** practice course and achieve `status: PASS` (`>= 90 / 100`):

```bash
# 1. Generate procedural practice dress (outer ruffled skirt + fused thigh pouches + sneaky knee rings + webbing)
python run_pipeline.py dojo --dojo-dir projects/dojo_run --action setup

# 2. Open the practice dress in `surgeon` and inspect `renders/inspect_triptych.png`
python run_pipeline.py surgeon --session projects/dojo_run/sess --action open \
  --input projects/dojo_run/dojo_practice_dress.blend --view front_skirt --zoom 1.2

# 3. Inspect grid + clay + wireframe triptych, remove all inner defects without harming the outer/back skirt, clean & export
#    (e.g. select-box --xray + delete, or bisect + clean + export)

# 4. Run automated grader (100-point rubric: defect removal >=98%, outer skirt >=95%, 0 floating islands, 0 degenerate faces)
python run_pipeline.py dojo --dojo-dir projects/dojo_run --action grade \
  --candidate projects/dojo_run/sess/dojo_solved.blend \
  --session projects/dojo_run/sess
```

### Exam 2 — 500k-Tri High-Poly Quad Retopology (`--exam retopo_500k`)
Tests reducing a messy ~500,000-triangle AI high-poly garment (sculpted ruffled corset + asymmetric slender side ribbon + junk object) down to a clean `2.5k–22k` quad mesh without truncating the asymmetric ribbon:

```bash
# 1. Generate ~500k-tri practice high-poly asset
python run_pipeline.py dojo --dojo-dir projects/dojo_500k --exam retopo_500k --action setup

# 2. Run autonomous closed-loop 6-pillar quad retopology
python run_pipeline.py retopo \
  --input projects/dojo_500k/dojo_500k_highpoly.glb \
  --out projects/dojo_500k/retopo_out

# 3. Auto-grade against 6-pillar rubric (poly budget, >=90% quads, >=72% val4, <=5% bidirectional silhouette/ribbon dev, 0 non-manifold)
python run_pipeline.py dojo --dojo-dir projects/dojo_500k --exam retopo_500k --action grade \
  --candidate projects/dojo_500k/retopo_out/deliverables/dojo_500k_highpoly_retopo.blend
```



