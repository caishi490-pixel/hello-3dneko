# Isolated PyMeshLab worker

This worker is licensed under GPL-3.0-or-later. The Apache-2.0 controller starts it in a separate Python process with a short-path JSON job and NumPy exchange files. The controller and Blender do not import PyMeshLab.

Use a standalone Python environment containing PyMeshLab 2025.7.post1. Set `HELLO_3DNEKO_PYMESHLAB_PYTHON` to that environment's executable, or run the CLI with the environment's Python. Confirm activation with `triage` and `available_backends.pymeshlab_worker`.

| Retopo operation | Output |
|---|---|
| `--pymeshlab-operation qecd-with-texture` | Triangle reduction of a static UV-mapped mesh. Set `--max-faces` for the scene budget. The worker preserves chart boundary vertices, face-corner UV layers, material assignments and seam edges. Skin weights and shape keys are outside this route. |
| `--pymeshlab-operation close-holes` | Manifold boundary repair with self-intersection checks, up to three passes and 30 edges per supported hole. Retopo bakes the repaired model from the original. |
| `--pymeshlab-operation alpha-wrap` | A watertight exterior proxy. Pass `--shell-contract` with `{"single_shell_envelope": true}`. The same contract selects this route automatically when the worker is available. |

Alpha Wrap rejects a missing envelope declaration or any declared separation requirement before the worker starts. QECD and hole repair detect non-manifold input before filtering. The worker exports only triangles, avoiding the native polygonal face reader.
