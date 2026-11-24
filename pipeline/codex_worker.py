"""Codex worker adapter for 2D generation and UV/texture optimization (Hello 3DNeko)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .guards import CircuitBreaker, PaidCallGuardError, redact_secrets


DEFAULT_CODEX_BRIDGE_SCRIPT = os.environ.get(
    "HELLO_3DNEKO_CODEX_BRIDGE_SCRIPT",
    os.environ.get("CODEX_BRIDGE_SCRIPT", ""),
).strip()
DEFAULT_COLLAB_PYTHON = os.environ.get(
    "HELLO_3DNEKO_CODEX_BRIDGE_PYTHON",
    sys.executable,
).strip()
DEFAULT_SCHEMA_PATH = REPO_ROOT / "templates" / "codex_output_schema.json"


@dataclass
class CodexWorkOrder:
    task_id: str
    stage: str
    prompt_text: str
    work_dir: Path
    input_images: list[Path] = field(default_factory=list)
    expected_outputs: list[Path] = field(default_factory=list)
    mode: str = "cli"  # "cli" or "desktop"
    sandbox: str = "workspace-write"
    model: str = ""
    codex_cmd: str = "codex.cmd"
    timeout_seconds: int = 900
    schema_path: Path = DEFAULT_SCHEMA_PATH


def render_template(template_path: Path, replacements: dict[str, str]) -> str:
    """Substitute {{KEY}} placeholders cleanly without touching literal braces."""
    text = Path(template_path).read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def build_codex_exec_command(order: CodexWorkOrder, last_msg_path: Path) -> list[str]:
    """Build deterministic headless `codex exec` CLI arguments for Windows."""
    cmd = [
        "cmd",
        "/c",
        order.codex_cmd,
        "exec",
        "--skip-git-repo-check",
        "-C",
        str(order.work_dir),
        "-s",
        order.sandbox,
        "--output-schema",
        str(order.schema_path),
        "-o",
        str(last_msg_path),
        "--json",
    ]
    if order.model:
        cmd.extend(["-m", order.model])
    for img in order.input_images:
        cmd.extend(["-i", str(img)])
    cmd.append("-")
    return cmd


def _check_desktop_cdp_alive(port: int = 9222) -> None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as resp:
            pages = json.load(resp)
            if not pages:
                raise RuntimeError("Codex Desktop CDP returned empty page list.")
    except Exception as exc:
        raise PaidCallGuardError(
            f"Codex desktop mode unavailable on 127.0.0.1:{port} ({exc}). "
            "Launch Codex Desktop with --remote-debugging-port=9222 or use --codex-mode cli."
        ) from exc


def run_codex_work_order(
    order: CodexWorkOrder,
    order_dir: Path,
    breaker: CircuitBreaker,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Dispatch a single structured work order to Codex without auto-retry."""
    order_dir = Path(order_dir)
    order_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = order_dir / f"{order.task_id}_prompt.md"
    prompt_file.write_text(order.prompt_text, encoding="utf-8")

    missing_inputs = [str(p) for p in order.input_images if not Path(p).is_file()]
    last_msg_path = order_dir / f"{order.task_id}_response.json"
    events_path = order_dir / f"{order.task_id}_events.jsonl"
    cmd = build_codex_exec_command(order, last_msg_path)

    preview = {
        "task_id": order.task_id,
        "stage": order.stage,
        "mode": order.mode,
        "work_dir": str(order.work_dir),
        "prompt_file": str(prompt_file),
        "input_images": [str(p) for p in order.input_images],
        "missing_inputs": missing_inputs,
        "expected_outputs": [str(p) for p in order.expected_outputs],
        "command": cmd if order.mode == "cli" else [str(DEFAULT_COLLAB_PYTHON), str(DEFAULT_CODEX_BRIDGE_SCRIPT or "<HELLO_3DNEKO_CODEX_BRIDGE_SCRIPT>"), "--send", "<prompt_file>"],
    }

    if dry_run:
        plan_path = order_dir / f"{order.task_id}_dry_run.json"
        plan_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "dry_run", **preview, "plan_path": str(plan_path)}

    if missing_inputs:
        raise FileNotFoundError(f"Missing Codex input images: {missing_inputs}")

    breaker.check_before_call(f"codex:{order.task_id}")
    if order.mode == "desktop":
        if not DEFAULT_CODEX_BRIDGE_SCRIPT or not Path(DEFAULT_CODEX_BRIDGE_SCRIPT).is_file():
            raise PaidCallGuardError(
                "Desktop mode requires HELLO_3DNEKO_CODEX_BRIDGE_SCRIPT pointing to an existing bridge script, "
                "or use --codex-mode cli."
            )
        _check_desktop_cdp_alive()
        bridge_cmd = [
            str(DEFAULT_COLLAB_PYTHON),
            str(DEFAULT_CODEX_BRIDGE_SCRIPT),
            "--send",
            order.prompt_text,
        ]
        try:
            proc = subprocess.run(
                bridge_cmd,
                capture_output=True,
                text=True,
                timeout=order.timeout_seconds,
                check=False,
            )
            ok = proc.returncode == 0
            breaker.record_attempt(ok)
            stdout_clean = redact_secrets(proc.stdout)
            events_path.write_text(stdout_clean, encoding="utf-8")
            if not ok:
                raise PaidCallGuardError(f"Desktop bridge failed ({proc.returncode}): {redact_secrets(proc.stderr)}")
            return {"status": "completed", "transport": "desktop", "raw_output": stdout_clean, "events_path": str(events_path)}
        except PaidCallGuardError:
            raise
        except Exception as exc:
            breaker.record_attempt(False)
            raise PaidCallGuardError(f"Desktop bridge execution error (stopped without retry): {exc}") from exc

    if order.mode != "cli":
        raise ValueError(f"Unsupported codex mode: {order.mode}")

    try:
        proc = subprocess.run(
            cmd,
            input=order.prompt_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=order.timeout_seconds,
            check=False,
        )
        events_path.write_text(redact_secrets(proc.stdout), encoding="utf-8")
        if proc.returncode != 0:
            breaker.record_attempt(False)
            raise PaidCallGuardError(
                f"codex exec failed ({proc.returncode}): {redact_secrets(proc.stderr[:600])}"
            )
        breaker.record_attempt(True)
    except PaidCallGuardError:
        raise
    except Exception as exc:
        breaker.record_attempt(False)
        raise PaidCallGuardError(f"codex exec error (stopped without retry): {exc}") from exc

    if not last_msg_path.is_file():
        raise PaidCallGuardError(f"codex exec finished without writing {last_msg_path}")

    parsed = json.loads(last_msg_path.read_text(encoding="utf-8"))
    missing_outputs = [str(p) for p in order.expected_outputs if not Path(p).is_file()]
    return {
        "status": parsed.get("status", "completed"),
        "transport": "cli",
        "response": parsed,
        "response_path": str(last_msg_path),
        "events_path": str(events_path),
        "missing_expected_outputs": missing_outputs,
    }
