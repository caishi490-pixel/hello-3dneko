# Hair and cloth rigging

Profile geometry first. Use `hair-craft --mode audit` for hair hygiene and `hair-rebuild --json "<strand control points>" --out "<work>"` for intentional strand reconstruction. For separated strands, map required groups to source shells and supply clear world-space gap segments. Check both preprocessing and the final result with the [shell contract](./closed-loop-retopology.md).

Run `python run_pipeline.py cloth-rig --body "<rigged body>" --garment "<garment>" --out "<work>"`. Check rest fit and weight normalization, then measure clip, penetration, stretch, and collapse in the extreme poses. Diagnose the affected part and joint from their motion. Use an independently re-imported GLB to check attachment, garment coverage, and package contents.

Use the existing `dojo --exam surgery` activity to practice inspection of occluded geometry before binding. Grade the actual export and archive the scorecard with the relevant QC views.
