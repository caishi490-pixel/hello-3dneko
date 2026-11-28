"""Headless Blender runner encapsulating CLI execution and dry-run previews."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .config import require_blender


def run_blender_script(
    blender_exe: str,
    script_path: Path,
    job_config: dict[str, Any],
    job_config_path: Path,
    log_path: Path,
    timeout: int = 900,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write a JSON job spec and invoke `blender.exe -b --python <script> -- --config <json>`."""
    job_config_path = Path(job_config_path)
    log_path = Path(log_path)
    job_config_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job_config_path.write_text(json.dumps(job_config, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        blender_exe,
        "-b",
        "--python-exit-code",
        "1",
        "--python",
        str(script_path),
        "--",
        "--config",
        str(job_config_path),
    ]
    if dry_run:
        return {"status": "dry_run", "command": cmd, "job_config_path": str(job_config_path)}

    cmd[0] = require_blender(blender_exe)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    combined = proc.stdout + ("\n---STDERR---\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender failed (exit {proc.returncode}) running {script_path.name}. "
            f"See log: {log_path}\nTail:\n{combined[-1000:]}"
        )
    return {
        "status": "completed",
        "command": cmd,
        "log_path": str(log_path),
        "job_config_path": str(job_config_path),
    }
