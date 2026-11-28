---
name: hello-3dneko-pipeline
description: Triage, repair, retopologize, bake, rig, and verify stylized 3D assets using available local Blender backends and independent quality gates.
---

# Hello 3DNeko Adaptive 3D Asset Pipeline v2.1

## Start with a per-part profile

```powershell
python run_pipeline.py triage --input "<asset.glb>" --out "<work>"
```

Read `asset_profile.json` before choosing a route. Check `available_backends`, `scene_summary`, and every `parts[]` record. Use the recommended route and respect `forbidden_routes`. Refer to [adaptive triage and routing](./references/adaptive-triage-and-routing.md) for the runbook.

## Choose a route

| Asset signal | Route |
|---|---|
| Clean hard surface | Use `retopo` with `autoremesher_aniso_route` when recommended and `autoremesher_cli` is true. Sharp QuadriFlow is the available Blender fallback. Use GRE when existing topology must stay fixed. [Hard surface and GRE](./references/hard-surface-and-gre-sop.md) |
| Thin sheet or slender ribbon | Smooth QuadriFlow with normal-aware reference projection and per-shell coverage. [Closed-loop retopology](./references/closed-loop-retopology.md) |
| Separated or non-manifold strands | Isolate affected shells, declare separation groups and gap rays, and repair them individually. Use `hair-rebuild` with strand control points when reconstruction is appropriate. [Hair and cloth rigging](./references/hair-and-cloth-rig-sop.md) |
| Occluded pouches or mixed geometry | Use `visual-cull` or `surgeon` for targeted removal. Verify the dark-matte wireframe and cut boundaries before retopology. [Visual surgical culling](./references/visual-surgical-cull-sop.md) |
| Attached asset parts | Allocate a single face budget across parts and use one shared atlas. [Closed-loop retopology](./references/closed-loop-retopology.md) |
| Existing UVs and a face reduction request | Use `retopo --pymeshlab-operation qecd-with-texture` for a static mesh when `pymeshlab_worker` is available. Preserve chart boundaries and material slots. [UV baking and HD retexture](./references/uv-bake-and-hd-retex-sop.md) |
| Weak UV coverage or blurry textures | Use `uv-bake` for PBR/AO or `uv-hd-retex prepare --region-profile <json>` for fixed-camera reprojection. [UV baking and HD retexture](./references/uv-bake-and-hd-retex-sop.md) |
| Stylized garment attachment | Use `cloth-rig` to fit to the body, transfer weights, and validate four poses on the re-imported export. [Hair and cloth rigging](./references/hair-and-cloth-rig-sop.md) |
| New character basemesh | Check the 40° open A-pose and proportions before generating or rigging. [Basemesh proportions](./references/basemesh-proportions-blueprint.md) |

Set `HELLO_3DNEKO_AUTOREMESHER_EXE` to a local AutoRemesher 1.2.0 executable or make it available on `PATH`. Set `HELLO_3DNEKO_PYMESHLAB_PYTHON` to the separate PyMeshLab Python environment. Triage probes both backends. [PyMeshLab setup and operations](../../gpl_workers/README.md)

## Verify before delivery

- Keep the Apache-2.0 control plane and GRE math core free of Blender and GPL-worker imports. Send Blender work through the headless GPL worker boundary.
- Use the welded UV-free OBJ exchange for geometry processing. Use the UV-preserving Blender exchange for full face-corner UVs, seams, and material slots. Run retopology and PBR baking workers on short paths.
- Check declared gap rays and distinct shell groups after preprocessing and on final output. Require local sample coverage for each input shell. Label an explicit single-shell envelope as an envelope.
- Bake the shared atlas once per channel. Inspect valid ray hits, cross-layer hits, slope decode error when calibrated, and confirmed tangent sign corrections separately.
- Re-open the GLB, check `junk_objects == []`, and repair zero tangents with `pack-glb`. Cycles probes OptiX, CUDA, HIP, oneAPI, and CPU in order. [Blender and Rodin pitfalls](./references/blender-rodin-pitfalls.md)
- Record verified recipes and failing routes in local knowledge with `task_type` and `asset_tags`. Roll back on gate failure and consider compatible routes. Keep local files, model names, credentials, and signed URLs out of public artifacts.
