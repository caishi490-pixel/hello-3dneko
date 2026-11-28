# Blender Subprocess Runner Scripts — License & Process Isolation Notice

All Python scripts inside `blender_scripts/` import Blender's `bpy`, `bmesh`, or `mathutils` APIs and are executed **inside headless Blender subprocesses** (`blender -b --python ...`).

- Therefore, files inside `blender_scripts/` are licensed under **GNU General Public License v3.0 or later (`GPL-3.0-or-later`)**, in strict compatibility with the Blender Foundation's GNU GPL terms.
- The external orchestrator (`run_pipeline.py`, `pipeline/`, `workflows/`), the templates and the standalone test driver remain under **Apache License 2.0**. Blender fixture scripts execute in the Blender worker process.
- The Apache-2.0 GRE core includes `ransac.py`, `relations.py`, `denoise.py`, `geometry.py`, `qa.py`, `snapshot.py`, `plan.py`, `plane.py`, `collision.py`, `report.py`, `pipeline.py`, and `__init__.py`.
- GRE worker entry points and worker I/O are `blender_io.py`, `pipeline_worker.py`, `cli.py`, `p3_cli.py`, `p4_cli.py`, and `qa_cli.py` under `gre/`. They operate on the Blender side of the process boundary.
