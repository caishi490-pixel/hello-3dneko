"""Process boundary for the optional AutoRemesher geometry CLI."""

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


AUTOREMESHER_ROUTE = "autoremesher_aniso_route"
AUTOREMESHER_TIMEOUT = 150
PYMESHLAB_ROUTES = {
    "qecd-with-texture": "pymeshlab_qecd_with_texture",
    "close-holes": "pymeshlab_close_holes",
    "alpha-wrap": "single_shell_alpha_wrap",
}
_PYMESHLAB_SCRIPT = Path(__file__).resolve().parents[1] / "gpl_workers" / "pymeshlab_worker.py"
_CLI_OPTIONS = ("--input", "--output", "--target-quads", "--sharp-edge", "--adaptivity", "--anisotropy")


def permits_alpha_wrap(contract):
    contract = contract or {}
    separated = (contract.get("separation_groups") or contract.get("gap_rays")
                 or contract.get("require_gap_preservation")
                 or "require_gap_preservation" in contract.get("asset_tags", []))
    return contract.get("single_shell_envelope") is True and not separated


class AutoRemesherError(RuntimeError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


class AutoRemesherAdapter:
    def __init__(self, executable):
        self.executable = executable

    @classmethod
    def discover(cls):
        configured = os.environ.get("HELLO_3DNEKO_AUTOREMESHER_EXE", "").strip()
        executable = shutil.which(configured or "autoremesher")
        if not executable:
            return None
        executable = str(Path(executable).resolve())
        try:
            probe = subprocess.run([executable, "--help"], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        help_text = probe.stdout + probe.stderr
        if probe.returncode == 0 and all(option in help_text for option in _CLI_OPTIONS):
            return cls(executable)
        return None

    def remesh(self, source, output, *, target_quads, timeout=AUTOREMESHER_TIMEOUT):
        source, output = Path(source).resolve(), Path(output).resolve()
        if source == output or target_quads < 1:
            raise ValueError("AutoRemesher requires distinct paths and a positive quad target")
        output.unlink(missing_ok=True)
        command = [
            self.executable, "--input", str(source), "--output", str(output),
            "--target-quads", str(target_quads), "--sharp-edge", "60",
            "--adaptivity", "1.0", "--anisotropy", "1.0",
        ]
        started = time.monotonic()
        try:
            result = subprocess.run(command, cwd=source.parent, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", check=False,
                                    timeout=min(timeout, AUTOREMESHER_TIMEOUT))
        except subprocess.TimeoutExpired as exc:
            output.unlink(missing_ok=True)
            raise AutoRemesherError("autoremesher_timeout", f"exceeded {exc.timeout:.2f}s") from exc
        except OSError as exc:
            raise AutoRemesherError("autoremesher_execution_failed", str(exc)) from exc
        if result.returncode or not output.is_file() or not output.stat().st_size:
            output.unlink(missing_ok=True)
            detail = (result.stderr.strip() or result.stdout.strip())[-400:]
            raise AutoRemesherError("autoremesher_execution_failed", f"exit={result.returncode}, output={detail}")
        return round(time.monotonic() - started, 3)


class PyMeshLabAdapter:
    def __init__(self, python):
        self.python = python

    def _run(self, arguments, timeout):
        return subprocess.run([self.python, "-I", str(_PYMESHLAB_SCRIPT), *arguments],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, check=False)

    @classmethod
    def discover(cls):
        configured = os.environ.get("HELLO_3DNEKO_PYMESHLAB_PYTHON", "").strip()
        python = shutil.which(configured or sys.executable)
        if not python:
            return None
        # Keep a virtualenv's launcher path so it retains its package search path.
        adapter = cls(os.path.abspath(python))
        try:
            probe = adapter._run(["--probe"], timeout=8)
            info = json.loads(probe.stdout.splitlines()[-1]) if not probe.returncode else {}
        except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
            return None
        return adapter if info.get("backend") == "pymeshlab" else None

    def process(self, workspace, operation, *, target_faces=None, single_shell_envelope=False):
        job_path = workspace.path("job.json")
        output = workspace.path("out.npz")
        output.unlink(missing_ok=True)
        job_path.write_text(json.dumps({
            "operation": operation, "input": str(workspace.path("in.npz")), "output": str(output),
            "target_faces": target_faces, "single_shell_envelope": single_shell_envelope,
        }), encoding="utf-8")
        try:
            result = self._run(["--job", str(job_path)], timeout=180)
        except subprocess.TimeoutExpired as exc:
            output.unlink(missing_ok=True)
            raise RuntimeError("pymeshlab_timeout: worker exceeded 180s") from exc
        if result.returncode or not output.is_file():
            output.unlink(missing_ok=True)
            raise RuntimeError((result.stderr.strip() or result.stdout.strip())[-700:]
                               or f"pymeshlab_execution_failed: exit={result.returncode}")
        return json.loads(result.stdout.splitlines()[-1])
