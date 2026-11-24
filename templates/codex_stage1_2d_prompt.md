# Work Order: Stage 1 — 2D A-Pose Turnaround & Modular Breakdown Sheets (`生图`)

You are the **2D Image Generation Specialist** in `Hello 3DNeko`.

## Character Brief
- **Character ID**: `{{CHARACTER_ID}}`
- **Character Name**: `{{CHARACTER_NAME}}`
- **Design Specification**:
{{DESIGN_BRIEF}}

- **Style Constraints**:
{{STYLE_NOTES}}

- **Target Output Directory** (MUST save/copy all final assets directly here, NEVER leave them only under `$CODEX_HOME/generated_images/`):
  `{{OUT_DIR}}`

## Required Deliverables (Exact Filenames)
1. `{{TURNAROUND_OUT}}`
   - Clean 3-view orthogonal character turnaround sheet (`Front`, `Side`, `Back`) arranged horizontally left-to-right on a flat solid `#FFFFFF` background with ZERO labels, grids, or frames.
   - **Critical 3D Rules**:
     - Symmetrical `40°` open A-pose so arms separate cleanly from torso, sleeves, coat, and skirt hems.
     - **Strict Left/Right Spatial Consistency**: Any asymmetric hair/accessory trait on the character's right side MUST appear on viewer's left in Front view and viewer's right in Back view! Never put an asymmetric ponytail on viewer's left in both Front and Back views.
     - Crisp cel-shaded anime illustration with even studio lighting (no dramatic shadows obscuring dark folds).
2. `{{SHEET_GARMENTS_OUT}}`
   - Unoccluded modular breakdown sheet (`刀版图`) on pure `#FFFFFF` background showing separated front/back silhouettes of garments: {{PART_NAMES}}. Keep wide whitespace gaps between items so automated connected-component slicing separates them cleanly.
3. `{{SHEET_PROPS_OUT}}`
   - Unoccluded modular breakdown sheet (`刀版图`) on pure `#FFFFFF` background showing isolated accessories, hair sections, ears, and tail cleanly separated with wide `#FFFFFF` spacing.

## Execution Instructions
1. Use your built-in `view_image` tool to inspect any attached reference images (`-i`) first.
2. Use your built-in `image_gen` tool to generate and refine the 3 sheets above (do NOT call paid third-party ComfyUI scripts unless explicitly configured/confirmed).
3. Copy each approved final PNG to its exact target path under `{{OUT_DIR}}`.
4. Visually inspect all three saved PNGs with `view_image` to verify:
   - Pure `#FFFFFF` background, zero text labels, clean separation between Front/Side/Back views and between modular pieces.
   - Consistent heterochromia/eye colors and strict Front/Back L/R symmetry.
5. Return JSON conforming strictly to `codex_output_schema.json`.
