"""Idempotent Hyper3D Rodin CLI client (`generation_id` persisted before poll, zero auto-retry)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .config import ManifestStore
from .guards import CircuitBreaker, PaidCallGuardError, redact_secrets


@dataclass
class RodinJobSpec:
    name: str
    action: str  # "generate", "bang", or "rigging"
    images: list[Path] = field(default_factory=list)
    tier: str = "Gen-2.5-High"
    mesh_mode: str = "Raw"
    prompt: str = ""
    target_job_name: str = ""
    bang_strength: int = 12
    bang_instruction: str = ""
    animation_presets: list[str] = field(
        default_factory=lambda: ["human_000025", "human_000000", "human_000030", "human_000088"]
    )
    poll_timeout: int = 600


def _run_hyper3d(cmd_name: str, args: list[str]) -> dict[str, Any]:
    cmd = ["cmd", "/c", cmd_name, "--output", "json"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise PaidCallGuardError(
            f"hyper3d failed ({proc.returncode}): "
            f"{redact_secrets(proc.stdout[:500])} {redact_secrets(proc.stderr[:500])}".strip()
        )
    return json.loads(proc.stdout)


def _build_submit_args(spec: RodinJobSpec, rodin_jobs: dict[str, Any]) -> list[str]:
    if spec.action == "generate":
        args = [
            "generate",
            "--tier",
            spec.tier,
            "--mesh-mode",
            spec.mesh_mode,
            "--texture-delight",
            "--format",
            "glb",
        ]
        if spec.prompt:
            args.extend(["--prompt", spec.prompt])
        for img in spec.images:
            if not Path(img).is_file():
                raise FileNotFoundError(f"Missing Rodin input image for '{spec.name}': {img}")
            args.extend(["--image", str(img)])
        return args

    source_gid = rodin_jobs.get(spec.target_job_name, {}).get("generation_id")
    if not source_gid:
        raise RuntimeError(f"Rodin job '{spec.name}' requires completed generation_id from '{spec.target_job_name}'.")

    if spec.action == "bang":
        args = ["bang", source_gid, "--strength", str(spec.bang_strength), "--format", "glb"]
        if spec.bang_instruction:
            args.extend(["--instruction", spec.bang_instruction])
        return args

    if spec.action == "rigging":
        args = [
            "rigging",
            source_gid,
            "--bone-template",
            "human",
            "--skinning-method",
            "shell",
            "--format",
            "glb",
        ]
        for preset in spec.animation_presets:
            args.extend(["--animation-preset", preset])
        return args

    raise ValueError(f"Unsupported Rodin action: {spec.action}")


def run_rodin_batch(
    specs: list[RodinJobSpec],
    out_root: Path,
    store: ManifestStore,
    breaker: CircuitBreaker,
    hyper3d_cmd: str = "hyper3d.cmd",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute or dry-run Rodin 3D jobs with strict idempotency and zero auto-retry."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = store.load()
    jobs_state: dict[str, Any] = manifest.setdefault("rodin_jobs", {})

    if dry_run:
        return {
            "status": "dry_run",
            "jobs": [
                {
                    "name": s.name,
                    "action": s.action,
                    "tier": s.tier,
                    "mesh_mode": s.mesh_mode,
                    "images": [str(p) for p in s.images],
                    "existing_generation_id": jobs_state.get(s.name, {}).get("generation_id"),
                }
                for s in specs
            ],
        }

    # Phase 1: Submit missing jobs and persist generation_id BEFORE polling.
    for spec in specs:
        entry = jobs_state.get(spec.name, {})
        if entry.get("generation_id"):
            continue
        args = _build_submit_args(spec, jobs_state)
        breaker.check_before_call(f"rodin_submit:{spec.name}")
        try:
            resp = _run_hyper3d(hyper3d_cmd, args)
            breaker.record_attempt(True)
        except Exception:
            breaker.record_attempt(False)
            raise
        jobs_state[spec.name] = {
            "action": spec.action,
            "generation_id": resp["generation_id"],
            "status": resp.get("status", "queued"),
        }
        store.save(manifest)

    # Phase 2: Poll & download completed assets cleanly without storing signed URLs.
    for spec in specs:
        entry = jobs_state[spec.name]
        if entry.get("downloaded"):
            continue
        gid = entry["generation_id"]
        poll_resp = _run_hyper3d(hyper3d_cmd, ["poll", gid, "--timeout", str(spec.poll_timeout)])
        entry["status"] = poll_resp.get("status")
        store.save(manifest)
        if entry["status"] != "completed":
            raise PaidCallGuardError(f"Rodin job '{spec.name}' ({gid}) ended with status: {entry['status']}")

        res = _run_hyper3d(hyper3d_cmd, ["result", gid])
        job_dir = out_root / spec.name
        job_dir.mkdir(parents=True, exist_ok=True)
        downloaded: dict[str, dict[str, Any]] = {}
        for item in res.get("files", []):
            fn = item["name"]
            dst = job_dir / fn
            r = requests.get(item["url"], timeout=120)
            r.raise_for_status()
            dst.write_bytes(r.content)
            downloaded[fn] = {"path": str(dst), "bytes": dst.stat().st_size}
        entry["downloaded"] = downloaded
        store.save(manifest)

    return {"status": "completed", "jobs": jobs_state}
