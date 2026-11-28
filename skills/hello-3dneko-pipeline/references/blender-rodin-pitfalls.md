# Blender & AI 3D CLI Technical Pitfalls Checklist

Use headless Blender with Cycles device detection in the order OptiX, CUDA, HIP, oneAPI, CPU. Set `HELLO_3DNEKO_BLENDER_EXE` to the executable path when auto-discovery fails. Stage bake and export I/O on short ASCII paths. Remove isolated bone shapes before export and check `junk_objects == []` after reload. Run `pack-glb` to repair zero tangents in an existing GLB.

## 1. Blender Mesh Cleanup & `.glb` UV Seam Trap

- **Symptom**: Running raw `bmesh.ops.smooth_vert` on `.glb` meshes splits open gaps across surfaces.
- **Root Cause**: glTF exporters split coincident spatial vertices along UV seams so each face corner can carry distinct UV coordinates; naive vertex smoothing shrinks both sides inward independently.
- **Fixes**:
  - **Preserve existing UV atlas while smoothing**: Build a spatial-welded vertex topology graph keyed by quantized world coordinates (`round(wp, 4)`), compute smoothed offsets on the welded graph, propagate deltas back to split coincident vertices, and apply non-destructive loop UV reprojection.
  - **Convert `Quad` AI mesh to clean editable basemesh**: Run `tris_convert_to_quads` followed by `remove_doubles(1e-4)` — Blender stores UVs on face-corner loops so welding seam vertices yields a single manifold without losing UVs.
  - **Texture pigment stays after geometry removal**: Flattening 3D sculpt blobs does not erase baked diffuse stains underneath; schedule a UV/texture cleanup pass after sculpt repair.

## 2. A-Pose Automatic Skinning "Shell Weight Bleed"

- **Symptom**: Raising arms pulls outer coat hems or lower skirt folds sideways with forearms/hands.
- **Fix**: Scan skirt/coat bounds near resting A-pose arms for stray `ForeArm`, `Hand`, and `Finger` vertex weights, purge/re-normalize onto `Hips`/`Spine`/`Leg`, and keep `<= 4` normalized influences per vertex.

## 3. Blender 4.x/5.x Scripting & Live MCP Rules

- **Localized UI Node Lookup**: Non-English UIs rename default nodes; always match shader nodes by `node.type == "BSDF_PRINCIPLED"`, never hardcoded localized names.
- **Do Not Reset Factory Settings in Live Blender MCP**: Never call `bpy.ops.wm.read_factory_settings(use_empty=True)` inside a connected Blender MCP session — batch automated builds and renders via headless CLI subprocesses (`blender -b --python ...`).
- **Animation & Modifier Ordering**: Call `scene.frame_set(frame)` before `pose.bones` transforms; apply `DataTransfer` before `Armature`; select explicitly before `convert(target='MESH')`; call `view_layer.update()` after mutating object transforms.
- **glTF Export & Procedural Materials**: Bake procedural shader graphs (`Wave`/`Noise`/ramps) into image maps prior to exporting `.glb`.

## 4. Paid AI CLI & BYOK Discipline

- **Windows Wrapper Invocation**: Wrap `.cmd` CLIs cleanly in `cmd /c "<tool>.cmd ..."` on Windows.
- **Long-Running Jobs (`BANG`)**: Pass `poll --timeout 600` for split tasks.
- **Persist Before Polling & Zero Auto-Retry**: Save `generation_id` into `manifest.json` immediately before polling, never wrap paid endpoints in automatic retry loops, trip `CircuitBreaker` on consecutive errors, and run logs through `redact_secrets()`.
