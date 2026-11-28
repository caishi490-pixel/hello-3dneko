# Hard surface retopology and GRE

For a clean, non-thin part with high planar dominance, follow `triage`. Run `retopo` to dispatch `autoremesher_aniso_route` when `available_backends.autoremesher_cli` is true. Each eligible part receives its own quota and uses a short-path welded OBJ. Check planar profiles, sharp edges, local coverage for every shell, and the total face budget. Sharp QuadriFlow is the Blender fallback. The route uses Blender for the shared UV atlas and bake.

For topology-locked repair, run GRE using the source mesh, matching immutable reference snapshot, and original region and relation annotations. Follow the repository's [GRE runbook](../../../gre/README.md) for Blender worker invocation and independent QA.

To apply GRE after a retopology, first create a reference snapshot for the **new** mesh. Assign stable `gre_vid` and `gre_fid` identifiers and build fresh region and relation annotations. Then inject the test perturbation, run the GRE worker, and run independent QA. Report the lock count measured against that snapshot.
