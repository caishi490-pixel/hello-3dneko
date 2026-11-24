# AI Aesthetic Polish, Image Generation & UV Texture Optimization

Use this workflow when a 3D character's proportions, rig, and silhouette already work, but renders look blurry around the eyes/face, show residual baking/sculpt stains, or suffer from glossy plastic PBR highlights.

## 1. Visual & UV Atlas Audit (Before Paying for Edits)

Do not guess why a model looks rough — measure the UV space and inspect closeups first:
1. **Locate UV Pixel Starvation**: Use Python/Pillow to inspect exported diffuse/roughness atlases, locate key facial islands (eyes, brows, mouth), compute their actual pixel bounding boxes, and save zoomed crops alongside the full atlas.
   - Example benchmark: In an auto-packed `2048x2048` atlas, an entire eyeball iris can shrink to ~`20x20` pixels (`~10px` radius), making macro closeups inevitably blurry regardless of mesh polygon count.
2. **Render Clean Reference Views**: Render orthographic or matched perspective front/side/closeup views over a clean white/neutral backdrop via headless Blender (`blender -b`). Note any post-sculpt color stains left on the texture and any angle-dependent specular washout on smooth roughness maps.

## 2. AI Visual Polish via Controlled `images/edits`

Drive image-editing APIs with **both** the messy 3D render and the authoritative 2D character sheet:
1. **Input Pairing**: Pass `[current_3D_render, official_2D_turnaround]` into `images/edits` (or equivalent multi-image edit endpoint).
2. **Coordinate-Locking Prompt Rules**:
   - Explicitly forbid changing silhouette outlines, body/limb ratios, camera framing, or facial feature center coordinates.
   - Restrict edits strictly to: sharpening iris gradients/highlights, removing diffuse muddiness/stains, unifying palette harmony, and translating realism glare into crisp anime cel lighting.
3. **Choose Scope Intentionally**:
   - Use **whole-front polish** as a camera-projection target or Toon shader visual reference (`Target Look`).
   - Use **standalone high-res local generation** (front-facing iris pair, embroidery motifs, clean fur/fabric swatches) when an entire atlas cannot spare pixels for tiny focal details.

## 3. UV Optimization & Reprojection Writeback (4-Step Loop)

### Step A: Isolate Focal Regions into Dedicated Material Slots
- In Blender Edit Mode, select eye/iris/sclera polygons (or key coat motifs) and assign dedicated material slots instead of sharing a crowded full-body UV sheet.
- Re-unwrap focal faces so each eye/motif occupies a dedicated high-resolution map (e.g., `1024x1024`), OR hide baked flat eyes and parent thin curved eyeball/iris meshes directly to the `Head` bone.
- Ensure seam vertices follow clean welded topology first (see [blender-rodin-pitfalls.md](./blender-rodin-pitfalls.md)) so edits do not open UV seam cracks.

### Step B: Write Back Polished Imagery via the Right Path
- **Path 1 — Camera / Stencil Reprojection**: Project coordinate-locked AI-polished renders back from their exact camera angles onto expanded UV islands.
- **Path 2 — Targeted UV Inpainting + Stain Cleanup**: Extract the neighborhood around dirty pixels, clean-sample surrounding skin/cloth tones, and enforce generous **Margin / Bleed** padding outside UV shells to prevent mipmap seam lines.
- **Path 3 — Roughness / Metallic Recalibration**: Raise diffuse skin, hair, and wool/cloth roughness values while keeping metallic/specular accents strictly on jewelry or metal trim.

### Step C: Switch to Anime Toon / Cel Shading
- Replace glossy realistic PBR skin/fabric setups with controlled anime Toon/Cel shading.

### Step D: Bake & Verify Before Export
- Bake finalized multi-material/procedural results into clean image maps before exporting `.glb`, and re-render headless matched-camera closeups (`Before` vs `After`).

## 4. Agent Automation Pattern & Budget Safety

1. **Audit Script**: Run Python/Pillow + headless Blender to dump UV density stats and zoomed defect crops.
2. **Single Guarded AI Edit Call**: Submit paired renders + 2D reference with strict coordinate lock; inspect alignment locally first.
3. **Headless Blender Writeback**: Split slots, remap UVs, adjust roughness/Toon nodes, bake maps, and export `.glb`.
4. **Budget Rule**: Pre-check crops locally before paid calls, persist outputs immediately, and enforce zero automatic retry loops on paid APIs.
