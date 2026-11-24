# Work Order: Stage 4 — AI Aesthetic Polish & UV/Material Optimization (`生图 + 优化`)

You are the **Image Polishing & UV/Texture Optimization Engineer** in `Hello 3DNeko`.

## Context & Objective
Stage 3 cleaned the 3D mesh/rig in Blender and exported baseline white-bg renders plus the extracted UV texture atlas and pixel-density audit (`uv_audit.json`). Your job is to turn the rough 3D textures into crisp anime quality without breaking 3D silhouette alignment.

- **Character**: `{{CHARACTER_NAME}}` (`{{CHARACTER_ID}}`)
- **Output Directory**: `{{OUT_DIR}}`
- **Inputs Provided (inspect with `view_image` and Python/Pillow)**:
  - UV Audit JSON: `{{UV_AUDIT_JSON}}`
  - Extracted Diffuse Atlas: `{{DIFFUSE_IN}}`
  - Extracted Roughness/Metallic Atlas: `{{ROUGHNESS_IN}}`
  - Baseline 3D Renders (`front`, `face_closeup`, etc.): attached via `-i` and listed in `{{RENDERS_DIR}}`
  - Authoritative 2D Turnaround Reference: `{{TURNAROUND_REF}}`

## Required Deliverables (Save Directly Under `{{OUT_DIR}}`)
1. **High-Res Dedicated Eye/Iris Texture(s)** via built-in `image_gen`:
   - `{{EYE_LEFT_OUT}}` (square `1024x1024`, clean front-facing anime iris + sclera/highlight art matching the character's Left Eye color/starlight design)
   - `{{EYE_RIGHT_OUT}}` (square `1024x1024`, clean front-facing anime iris + sclera/highlight art matching the character's Right Eye color/design)
2. **Coordinate-Locked AI Polished Face/Front Target** via built-in `image_gen`:
   - `{{POLISHED_FACE_OUT}}`
   - **CRITICAL COORDINATE RULE**: Edit from the attached baseline 3D face closeup render + 2D turnaround reference. Strictly preserve exact silhouette, camera perspective, and facial anchor coordinates — only sharpen iris clarity/gradient, clean muddy diffuse patches, and unify anime cel lighting.
3. **Optimized UV Atlas Maps** via Python/Pillow scripts (using `{{UV_AUDIT_JSON}}` + generated assets):
   - `{{DIFFUSE_OPT_OUT}}`: Cleaned diffuse atlas with residual post-sculpt stains/rings painted out cleanly using surrounding clean tones and generous edge margin/bleed (zero dark seam borders).
   - `{{ROUGHNESS_OPT_OUT}}`: Recalibrated metallic-roughness map raising skin/hair/cloth roughness (e.g. matte/soft satin green channel `>= 205/255`, zero metallic bleed on skin) to eliminate specular washout while keeping jewelry trim crisp.
4. **Stage 5 Blender Apply Spec (`{{APPLY_SPEC_OUT}}`)**:
   Write a valid JSON file following this structure:
   ```json
   {
     "diffuse_override": "{{DIFFUSE_OPT_OUT}}",
     "roughness_override": "{{ROUGHNESS_OPT_OUT}}",
     "eye_left_texture": "{{EYE_LEFT_OUT}}",
     "eye_right_texture": "{{EYE_RIGHT_OUT}}",
     "split_eye_materials": true,
     "toon_shader_blend": true,
     "skin_roughness_floor": 0.82,
     "normal_strength": 0.35,
     "bake_before_glb": true,
     "bake_resolution": 2048
   }
   ```

## Hard Constraints
- Copy every generated asset out of `$CODEX_HOME/generated_images/` directly into the exact target paths above.
- Inspect generated images and zoomed diffs with `view_image` before finishing.
- Do not modify upstream `03_rig_clean` `.blend`/`.glb` files directly; Stage 5 applies your assets in headless Blender cleanly.
- Return JSON conforming strictly to `codex_output_schema.json`.
