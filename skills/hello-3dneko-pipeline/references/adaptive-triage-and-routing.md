# Adaptive triage and routing

Run `python run_pipeline.py triage --input "<asset>" --out "<work>"` for each new asset. Read the schema 2.1 `asset_profile.json`. Use `part_id` to refer to a part and `object_name` when operating on its Blender object. Use `scene_summary.welded_non_manifold_edges` for aggregate reporting and `parts[]` for routing decisions.

- A clean, non-thin armor plate can use `autoremesher_aniso_route` when detected, with sharp QuadriFlow as fallback. A thin, slender ribbon can use `quadriflow_smooth_flow_refine_bake` with tightly checked projection. A dirty multi-shell tassel needs isolated repair or controlled reconstruction. Follow its `forbidden_routes`.
- `blender`, `blender_mesh_tools`, `blender_quadriflow`, and `cycles_devices_detected` report the worker's available capabilities. Choose an implemented route and review the actual report after running it.
- `autoremesher_cli` reports whether the local executable supports the required flags. Set `HELLO_3DNEKO_AUTOREMESHER_EXE` or use `PATH`. `pymeshlab_worker` reports whether the isolated Python runtime supports QECD, hole closing and Alpha Wrap. [PyMeshLab setup and operations](../../../gpl_workers/README.md)
- Confirm art intent before destructive edits. For a mixed assembly, isolate only the required parts and record each part's budget, tags, and gate results. Keep knowledge for `single_prop` independent of `multi_part`, and keep outer envelope recipes independent of gap-preserving assets.
