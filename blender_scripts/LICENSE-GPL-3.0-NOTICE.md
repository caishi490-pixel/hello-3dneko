# Blender Subprocess Runner Scripts — License & Process Isolation Notice

All Python scripts inside `blender_scripts/` import Blender's `bpy`, `bmesh`, or `mathutils` APIs and are executed **inside headless Blender subprocesses** (`blender -b --python ...`).

- Therefore, files inside `blender_scripts/` are licensed under **GNU General Public License v3.0 or later (`GPL-3.0-or-later`)**, in strict compatibility with the Blender Foundation's GNU GPL terms.
- The external orchestrator (`run_pipeline.py`, `pipeline/`, `workflows/`, `templates/`, `tests/`) communicates strictly across OS process boundaries using standard JSON specifications and 3D asset files (`.glb` / `.json`) without importing `bpy`, and remains licensed under **Apache License 2.0** (see root `LICENSE`).
