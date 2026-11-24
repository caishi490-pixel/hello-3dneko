"""Configuration schemas, project scaffolding, atomic manifest store, and ledger logging."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return default


def _discover_blender_exe() -> str:
    """Resolve Blender executable via env vars, PATH, or standard OS install folders."""
    explicit = _env_first("HELLO_3DNEKO_BLENDER_EXE", "ANIME3D_BLENDER_EXE", "BLENDER_EXE")
    if explicit:
        return explicit

    found = shutil.which("blender")
    if found:
        return found

    if os.name == "nt":
        for prog_env in ("ProgramFiles", "ProgramW6432"):
            base = os.environ.get(prog_env, "").strip()
            if not base:
                continue
            bf_dir = Path(base) / "Blender Foundation"
            if bf_dir.is_dir():
                for candidate in sorted(bf_dir.glob("Blender */blender.exe"), reverse=True):
                    return str(candidate)
        return "blender.exe"

    mac_app = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac_app.is_file():
        return str(mac_app)
    return "blender"


DEFAULT_BLENDER_EXE = _discover_blender_exe()
DEFAULT_HYPER3D_CMD = _env_first(
    "HELLO_3DNEKO_HYPER3D_CMD",
    "ANIME3D_HYPER3D_CMD",
    default="hyper3d.cmd" if os.name == "nt" else "hyper3d",
)
DEFAULT_CODEX_CMD = _env_first(
    "HELLO_3DNEKO_CODEX_CMD",
    "ANIME3D_CODEX_CMD",
    default="codex.cmd" if os.name == "nt" else "codex",
)
DEFAULT_COMFY_KEY_FILE = _env_first(
    "HELLO_3DNEKO_COMFY_KEY_FILE",
    "ANIME3D_COMFY_KEY_FILE",
    "COMFY_API_KEY_FILE",
    default="",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class ToolPaths:
    blender_exe: str = DEFAULT_BLENDER_EXE
    hyper3d_cmd: str = DEFAULT_HYPER3D_CMD
    codex_cmd: str = DEFAULT_CODEX_CMD
    comfy_key_file: str = DEFAULT_COMFY_KEY_FILE


@dataclass
class CodexSettings:
    mode: str = "cli"  # "cli" (codex exec) or "desktop" (CDP 9222 bridge)
    sandbox: str = "workspace-write"
    model: str = ""
    timeout_seconds: int = 900
    image_backend: str = "codex_builtin"  # "codex_builtin" or "comfy"


@dataclass
class BlenderSettings:
    target_height: float = 1.56
    render_samples: int = 48
    render_resolution: int = 896
    cycles_device_type: str = "OPTIX"
    bake_resolution: int = 2048
    eye_map_resolution: int = 1024


@dataclass
class ProjectConfig:
    character_id: str
    character_name: str
    design_brief: str
    style_notes: str = "Soft cel-shaded anime illustration with 40-degree open A-pose and #FFFFFF background."
    reference_images: list[str] = field(default_factory=list)
    part_names: list[str] = field(
        default_factory=lambda: ["outer_coat", "inner_dress", "waist_corset", "tail_or_prop"]
    )
    tools: ToolPaths = field(default_factory=ToolPaths)
    codex: CodexSettings = field(default_factory=CodexSettings)
    blender: BlenderSettings = field(default_factory=BlenderSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        tools_raw = data.get("tools") or {}
        codex_raw = data.get("codex") or {}
        blender_raw = data.get("blender") or {}
        return cls(
            character_id=data["character_id"],
            character_name=data["character_name"],
            design_brief=data.get("design_brief", ""),
            style_notes=data.get("style_notes", cls.style_notes),
            reference_images=list(data.get("reference_images") or []),
            part_names=list(data.get("part_names") or ["outer_coat", "inner_dress", "waist_corset"]),
            tools=ToolPaths(**tools_raw),
            codex=CodexSettings(**codex_raw),
            blender=BlenderSettings(**blender_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectLayout:
    root: Path
    config_path: Path
    manifest_path: Path
    ledger_path: Path
    breaker_path: Path
    dir_2d: Path
    dir_2d_slices: Path
    dir_3d_rodin: Path
    dir_rig_clean: Path
    dir_rig_renders: Path
    dir_codex_polish: Path
    dir_codex_orders: Path
    dir_final_bake: Path
    dir_final_renders: Path

    @classmethod
    def for_root(cls, project_root: Path) -> "ProjectLayout":
        root = Path(project_root).resolve()
        return cls(
            root=root,
            config_path=root / "project.json",
            manifest_path=root / "manifest.json",
            ledger_path=root / "credits_ledger.jsonl",
            breaker_path=root / ".circuit_breaker.json",
            dir_2d=root / "01_2d_design",
            dir_2d_slices=root / "01_2d_design" / "slices",
            dir_3d_rodin=root / "02_rodin_3d",
            dir_rig_clean=root / "03_rig_clean",
            dir_rig_renders=root / "03_rig_clean" / "renders",
            dir_codex_polish=root / "04_codex_polish",
            dir_codex_orders=root / "04_codex_polish" / "work_orders",
            dir_final_bake=root / "05_final_bake",
            dir_final_renders=root / "05_final_bake" / "renders",
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.root,
            self.dir_2d,
            self.dir_2d_slices,
            self.dir_3d_rodin,
            self.dir_rig_clean,
            self.dir_rig_renders,
            self.dir_codex_polish,
            self.dir_codex_orders,
            self.dir_final_bake,
            self.dir_final_renders,
        ):
            path.mkdir(parents=True, exist_ok=True)


class ManifestStore:
    """Atomic JSON manifest manager preserving stage & paid-job state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"stages": {}, "rodin_jobs": {}, "codex_jobs": {}, "updated_at": now_iso()}
        return json.loads(self.path.read_text(encoding="utf-8-sig"))

    def save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def update_stage(self, stage_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = self.load()
        data.setdefault("stages", {})[stage_name] = {**payload, "updated_at": now_iso()}
        self.save(data)
        return data


def append_ledger(ledger_path: Path, event: dict[str, Any]) -> None:
    """Append a structured stage/credit event without secrets."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": now_iso(), **event}
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def init_project(
    projects_dir: Path,
    config: ProjectConfig,
    overwrite: bool = False,
) -> ProjectLayout:
    """Initialize a structured character workspace and manifest."""
    layout = ProjectLayout.for_root(Path(projects_dir) / config.character_id)
    if layout.config_path.exists() and not overwrite:
        raise FileExistsError(f"Project already exists: {layout.root}")
    layout.ensure_dirs()
    layout.config_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store = ManifestStore(layout.manifest_path)
    if not layout.manifest_path.exists() or overwrite:
        store.save(
            {
                "character_id": config.character_id,
                "character_name": config.character_name,
                "stages": {
                    "init": {
                        "status": "completed",
                        "artifacts": {"project_json": str(layout.config_path)},
                    }
                },
                "rodin_jobs": {},
                "codex_jobs": {},
            }
        )
    append_ledger(layout.ledger_path, {"event": "init", "character_id": config.character_id})
    return layout


def load_project(project_dir: Path) -> tuple[ProjectLayout, ProjectConfig, ManifestStore]:
    """Load config, layout, and manifest from an existing project folder."""
    layout = ProjectLayout.for_root(project_dir)
    if not layout.config_path.is_file():
        raise FileNotFoundError(f"Missing project.json in {layout.root}")
    cfg = ProjectConfig.from_dict(json.loads(layout.config_path.read_text(encoding="utf-8-sig")))
    return layout, cfg, ManifestStore(layout.manifest_path)
