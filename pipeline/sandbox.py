"""Short, isolated Blender I/O paths and atomic file publication."""

from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import tempfile
from uuid import uuid4


class ShortPathWorkspace:
    def __init__(self, destination, *, temp_root=None):
        self.destination = Path(destination).resolve()
        self._root = Path(temp_root or tempfile.gettempdir()) / "nk_s"
        self._temporary = None

    def __enter__(self):
        if not str(self._root).isascii() or len(str(self._root)) > 78:
            raise ValueError("Set TEMP to a short ASCII directory for Blender I/O")
        self._root.mkdir(parents=True, exist_ok=True)
        prefix = sha256(os.fsencode(self.destination)).hexdigest()[:8] + "-"
        self._temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=self._root)
        self.root = Path(self._temporary.name)
        return self

    def __exit__(self, *exc):
        self._temporary.cleanup()

    def path(self, relative):
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root.resolve()):
            raise ValueError("Path escapes short workspace")
        if len(str(result)) >= 120 or not str(result).isascii():
            raise ValueError("Blender workspace path must be ASCII and under 120 characters")
        return result

    @staticmethod
    def publish_file(source, destination):
        """Copy on the destination volume before the atomic replacement."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.parent / f".{uuid4().hex}.tmp"
        try:
            shutil.copy2(source, pending)
            pending.replace(destination)
        finally:
            pending.unlink(missing_ok=True)

    def publish(self, relative, destination=None):
        source = self.path(relative)
        destination = Path(destination) if destination is not None else self.destination / relative
        if source.is_dir():
            for item in sorted(source.rglob("*")):
                if item.is_file():
                    self.publish_file(item, destination / item.relative_to(source))
        else:
            self.publish_file(source, destination)


_PLAN_INPUTS = {"input_path", "reference_glb", "low_poly_input", "high_poly_reference"}
_PLAN_OUTPUTS = {
    "textures_dir": "tex", "uv_layout_dir": "uv", "output_blend": "c.blend",
    "output_glb": "c.glb", "execution_log": "exec.json",
}
_VALIDATOR_INPUTS = {
    "--candidate-glb", "--candidate-blend", "--reference-glb", "--high-reference",
    "--criteria-json", "--textures-dir",
}
_VALIDATOR_OUTPUTS = {
    "--renders-dir": "qc", "--out-json": "verdict.json", "--uv-report-json": "uv.json",
    "--bake-report-json": "bake.json",
}


def run_staged_blender_job(args, workspace, invoke):
    """Stage worker inputs and atomically publish outputs from a successful subprocess."""
    input_paths, output_paths = {}, {}

    def stage_input(path):
        if not path:
            return path
        source = Path(path)
        if not source.exists():
            return path
        if path not in input_paths:
            alias = workspace.path(f"in{len(input_paths)}{source.suffix if source.is_file() else ''}")
            if source.is_dir():
                shutil.copytree(source, alias)
            else:
                shutil.copy2(source, alias)
            input_paths[path] = str(alias)
        return input_paths[path]

    def stage_output(path, alias):
        target = str(workspace.path(alias))
        output_paths[target] = path
        return target

    if args[0] == "--plan":
        plan_source = Path(args[1])
        payload = json.loads(plan_source.read_text(encoding="utf-8-sig"))
        for key in sorted(_PLAN_INPUTS & payload.keys()):
            payload[key] = stage_input(payload[key])
        for key, alias in _PLAN_OUTPUTS.items():
            if key in payload:
                payload[key] = stage_output(payload[key], alias)
        staged = workspace.path("plan.json")
        staged.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        worker_args = ["--plan", str(staged)]
    else:
        worker_args = []
        for option, value in zip(args[::2], args[1::2]):
            if option in _VALIDATOR_INPUTS:
                value = stage_input(value)
            if option in _VALIDATOR_OUTPUTS:
                value = stage_output(value, _VALIDATOR_OUTPUTS[option])
            worker_args.extend((option, value))

    ok, log = invoke(worker_args)
    remapping = {**{alias: original for original, alias in input_paths.items()}, **output_paths}

    def restore_paths(value):
        if isinstance(value, dict):
            return {k: restore_paths(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore_paths(v) for v in value]
        if isinstance(value, str):
            for alias, original in remapping.items():
                if value == alias:
                    return original
                if value.startswith((alias + "/", alias + os.sep)):
                    return str(Path(original) / value[len(alias) + 1:])
        return value

    if ok:
        if (glb := workspace.path("c.glb")).is_file() and str(glb) in output_paths:
            from .glb_postprocess import repair_zero_tangents
            repair_zero_tangents(glb)
        for alias, original in output_paths.items():
            source = Path(alias)
            if not source.exists():
                continue
            if source.suffix == ".json":
                payload = restore_paths(json.loads(source.read_text(encoding="utf-8")))
                source.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            workspace.publish(source.name, original)
    return ok, log


def _export_exchange(mode, source, destination, blender_exe, *parameters):
    from .config import DEFAULT_BLENDER_EXE, REPO_ROOT
    from .topology_agent_loop import _run_blender

    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    with ShortPathWorkspace(destination.parent) as workspace:
        name = "swap" + destination.suffix.lower()
        ok, log = _run_blender(
            blender_exe or DEFAULT_BLENDER_EXE, REPO_ROOT / "blender_scripts" / "mesh_exchange.py",
            [mode, str(source), str(workspace.path(name)), *parameters], workspace.path("worker.log"),
        )
        if not ok:
            raise RuntimeError(f"Mesh exchange failed:\n{log[-1000:]}")
        workspace.publish(name, destination)
        if mode == "welded":
            workspace.publish("weld_report.json")
    return destination


def export_welded_geometry_obj(src, out_obj, weld_dist=1e-4, *, blender_exe=None):
    return _export_exchange("welded", src, out_obj, blender_exe, "--weld-dist", str(weld_dist))


def export_uv_preserving_mesh(src, out_mesh, *, blender_exe=None):
    return _export_exchange("uv", src, out_mesh, blender_exe)
