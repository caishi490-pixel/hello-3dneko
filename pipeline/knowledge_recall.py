"""Cross-run knowledge recall engine: reads learned_recipes.jsonl & failure_lessons.jsonl to guide planner priors."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def _row_matches_task(row: dict[str, Any], task_type: str) -> bool:
    row_type = str(row.get("task_type") or "").strip().lower()
    if row_type:
        return row_type == task_type
    # Legacy retopo entries omitted task_type before uv_bake/cloth_rig were added
    return task_type == "retopo"


def load_strategy_priors(
    knowledge_dir: Path | None,
    task_type: str,
    asset_tags: set[str] | None = None,
) -> dict[str, Any]:
    """Aggregate historical wins & failures into deterministic strategy priority scores.

    Returns:
        {
            "preferred_order": [strategy_id, ...],
            "deprioritized": [strategy_id, ...],
            "scores": {strategy_id: {"wins": int, "fails": int, "avg_score": float, "priority": float}},
            "recipes_read": int,
            "failures_read": int,
        }
    """
    if knowledge_dir is None:
        return {
            "preferred_order": [],
            "deprioritized": [],
            "scores": {},
            "recipes_read": 0,
            "failures_read": 0,
        }

    kdir = Path(knowledge_dir)
    recipes = [r for r in _iter_jsonl(kdir / "learned_recipes.jsonl") if _row_matches_task(r, task_type)]
    failures = [r for r in _iter_jsonl(kdir / "failure_lessons.jsonl") if _row_matches_task(r, task_type)]

    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"wins": 0.0, "fails": 0.0, "score_sum": 0.0, "tag_bonus": 0.0}
    )
    tags = asset_tags or set()

    for r in recipes:
        sid = str(r.get("strategy_id") or "").strip()
        if not sid:
            continue
        sc = float(r.get("score") or 90.0)
        st = stats[sid]
        st["wins"] += 1.0
        st["score_sum"] += sc
        if tags:
            row_blob = json.dumps(r, ensure_ascii=False).lower()
            if any(t.lower() in row_blob for t in tags):
                st["tag_bonus"] += 8.0

    for f in failures:
        sid = str(f.get("banned_strategy") or f.get("strategy_id") or "").strip()
        if not sid:
            continue
        st = stats[sid]
        st["fails"] += 1.0

    scored: dict[str, dict[str, Any]] = {}
    for sid, st in stats.items():
        wins = int(st["wins"])
        fails = int(st["fails"])
        avg_score = round(st["score_sum"] / wins, 2) if wins > 0 else 0.0
        # Reward proven wins & high quality scores; softly penalize repeated failures without permanent lockout
        win_rate = wins / max(1, wins + fails)
        priority = round((avg_score * 0.6 + win_rate * 40.0 + min(24.0, st["tag_bonus"])) - min(30.0, fails * 4.0), 2)
        scored[sid] = {
            "wins": wins,
            "fails": fails,
            "avg_score": avg_score,
            "win_rate": round(win_rate, 3),
            "priority": priority,
        }

    preferred = [
        sid
        for sid, meta in sorted(scored.items(), key=lambda kv: (-kv[1]["priority"], -kv[1]["wins"], kv[0]))
        if meta["wins"] > 0 and meta["priority"] > 20.0
    ]
    deprioritized = [
        sid
        for sid, meta in sorted(scored.items(), key=lambda kv: (kv[1]["priority"], kv[0]))
        if meta["wins"] == 0 and meta["fails"] >= 2
    ]

    return {
        "preferred_order": preferred,
        "deprioritized": deprioritized,
        "scores": scored,
        "recipes_read": len(recipes),
        "failures_read": len(failures),
    }


def apply_priors_to_candidates(
    candidates: list[tuple[str, Any, str]],
    priors: dict[str, Any] | None,
) -> list[tuple[str, Any, str]]:
    """Reorder planner candidates using recalled priors while preserving deterministic fallback order."""
    if not priors:
        return candidates
    pref_rank = {sid: idx for idx, sid in enumerate(priors.get("preferred_order", []))}
    deprio = set(priors.get("deprioritized", []))
    scores = priors.get("scores", {})

    indexed = list(enumerate(candidates))

    def _sort_key(item: tuple[int, tuple[str, Any, str]]):
        orig_idx, (sid, _, _) = item
        if sid in pref_rank:
            return (0, pref_rank[sid], orig_idx)
        if sid in deprio:
            return (2, orig_idx, 0)
        return (1, orig_idx, 0)

    reordered: list[tuple[str, Any, str]] = []
    for _, (sid, payload, rationale) in sorted(indexed, key=_sort_key):
        if sid in pref_rank:
            meta = scores.get(sid, {})
            rationale = (
                f"[Knowledge Recall: {meta.get('wins', 0)} prior wins, avg={meta.get('avg_score', 0)}] "
                + rationale
            )
        reordered.append((sid, payload, rationale))
    return reordered
