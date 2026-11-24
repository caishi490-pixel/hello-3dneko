"""Unified 7-Metric Experiment Ledger for Closed-Loop Agent Retopology Benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class ExperimentRecord:
    time: str
    task_type: str  # e.g. "retopo"
    asset: str
    work_dir: str
    autonomous_completed: bool
    human_interventions: int
    auto_rework_count: int
    iterations_used: int
    elapsed_seconds: float
    final_score: float
    passed: bool
    winning_strategy_id: str
    failure_reasons: list[str]
    seed: int
    config_hash: str
    reproducible_command: str


def compute_config_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def load_experiments(knowledge_dir: Path) -> list[dict[str, Any]]:
    ledger = knowledge_dir / "experiment_ledger.jsonl"
    if not ledger.is_file():
        return []
    rows = []
    for line in ledger.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return rows


def compute_ledger_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total_runs": 0,
            "autonomous_completion_rate_pct": 0.0,
            "total_human_interventions": 0,
            "avg_auto_rework_count": 0.0,
            "avg_elapsed_seconds": 0.0,
            "avg_final_score": 0.0,
            "reproducible_seed_coverage_pct": 0.0,
        }
    completed = sum(1 for r in rows if r.get("autonomous_completed"))
    interventions = sum(int(r.get("human_interventions", 0)) for r in rows)
    avg_rework = round(sum(int(r.get("auto_rework_count", 0)) for r in rows) / total, 2)
    avg_time = round(sum(float(r.get("elapsed_seconds", 0.0)) for r in rows) / total, 2)
    avg_score = round(sum(float(r.get("final_score", 0.0)) for r in rows) / total, 2)
    repro = sum(1 for r in rows if r.get("config_hash") and r.get("seed") is not None)
    return {
        "total_runs": total,
        "autonomous_completion_rate_pct": round(100.0 * completed / total, 2),
        "total_human_interventions": interventions,
        "avg_auto_rework_count": avg_rework,
        "avg_elapsed_seconds": avg_time,
        "avg_final_score": avg_score,
        "reproducible_seed_coverage_pct": round(100.0 * repro / total, 2),
    }


def rebuild_scorecard_markdown(knowledge_dir: Path) -> Path:
    rows = load_experiments(knowledge_dir)
    summary = compute_ledger_summary(rows)
    md_path = knowledge_dir / "EXPERIMENT_SCORECARD.md"
    lines = [
        "# 统一实验记录与自主完成度总览 (Unified Experiment Scorecard)",
        "",
        f"- 更新时间: `{_now_iso()}`",
        f"- 累计实验次数: `{summary['total_runs']}`",
        f"- Agent 自主完成率: `{summary['autonomous_completion_rate_pct']}%`",
        f"- 累计人工干预次数: `{summary['total_human_interventions']}`",
        f"- 平均自动返工次数: `{summary['avg_auto_rework_count']}`",
        f"- 单件平均耗时: `{summary['avg_elapsed_seconds']}s`",
        f"- 平均质量评分: `{summary['avg_final_score']}`",
        f"- 参数/种子可复现覆盖率: `{summary['reproducible_seed_coverage_pct']}%`",
        "",
        "| 时间 | 类型 | 资产 | 自主完成 | 人工干预 | 返工次数 | 耗时(s) | 评分 | 获胜策略 | 失败原因 | 配置哈希 / Seed |",
        "| --- | --- | --- | :---: | :---: | :---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in rows:
        reasons = "; ".join(r.get("failure_reasons") or []) or "-"
        if len(reasons) > 90:
            reasons = reasons[:87] + "..."
        lines.append(
            f"| {r.get('time', '')} | `{r.get('task_type', '')}` | `{r.get('asset', '')}` "
            f"| {'YES' if r.get('autonomous_completed') else 'NO'} "
            f"| {r.get('human_interventions', 0)} | {r.get('auto_rework_count', 0)} "
            f"| {r.get('elapsed_seconds', 0.0)} | {r.get('final_score', 0.0)} "
            f"| `{r.get('winning_strategy_id') or '-'}` | {reasons} "
            f"| `{r.get('config_hash', '')}` (seed={r.get('seed', 0)}) |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _sanitize_ledger_path(path: Path | str) -> str:
    """Prefer workspace-relative POSIX paths in logs to avoid leaking host usernames."""
    repo_root = Path(__file__).resolve().parent.parent
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return p.name or str(p)


def record_experiment(
    knowledge_dir: Path,
    *,
    task_type: str,
    asset: str,
    work_dir: Path,
    autonomous_completed: bool,
    human_interventions: int = 0,
    iterations_used: int = 1,
    elapsed_seconds: float = 0.0,
    final_score: float = 0.0,
    passed: bool = False,
    winning_strategy_id: str = "",
    failure_reasons: list[str] | None = None,
    seed: int = 0,
    config_payload: dict[str, Any] | None = None,
    reproducible_command: str = "",
) -> dict[str, Any]:
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    rec = ExperimentRecord(
        time=_now_iso(),
        task_type=task_type,
        asset=Path(asset).name,
        work_dir=_sanitize_ledger_path(work_dir),
        autonomous_completed=bool(autonomous_completed),
        human_interventions=int(human_interventions),
        auto_rework_count=max(0, int(iterations_used) - 1),
        iterations_used=int(iterations_used),
        elapsed_seconds=round(float(elapsed_seconds), 2),
        final_score=round(float(final_score), 2),
        passed=bool(passed),
        winning_strategy_id=winning_strategy_id,
        failure_reasons=list(failure_reasons or []),
        seed=int(seed),
        config_hash=compute_config_hash(config_payload or {"task_type": task_type, "asset": asset, "seed": seed}),
        reproducible_command=reproducible_command,
    )
    ledger = knowledge_dir / "experiment_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    scorecard_md = rebuild_scorecard_markdown(knowledge_dir)
    rows = load_experiments(knowledge_dir)
    return {
        "record": asdict(rec),
        "summary": compute_ledger_summary(rows),
        "ledger_jsonl": str(ledger),
        "scorecard_md": str(scorecard_md),
    }
