"""Safety guards: paid API zero-auto-retry idempotency, circuit breaker, secret redaction."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


_SECRET_PATTERNS = [
    re.compile(r"comfyui-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Mask potential API tokens or signed URL query strings before saving logs."""
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[REDACTED_TOKEN]", cleaned)
    cleaned = re.sub(
        r"(https?://[^\s\"']+\?)[^\s\"']+",
        r"\1[REDACTED_QUERY]",
        cleaned,
    )
    return cleaned


class PaidCallGuardError(RuntimeError):
    """Raised when a paid call fails or violates idempotency/circuit-breaker rules."""


class CircuitBreaker:
    """Sliding-window call rate & consecutive failure fuse."""

    def __init__(
        self,
        state_path: Path,
        max_consecutive_failures: int = 3,
        max_calls_in_window: int = 10,
        window_seconds: int = 180,
    ) -> None:
        self.state_path = Path(state_path)
        self.max_consecutive_failures = max_consecutive_failures
        self.max_calls_in_window = max_calls_in_window
        self.window_seconds = window_seconds

    def _load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"consecutive_failures": 0, "call_timestamps": []}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"consecutive_failures": 0, "call_timestamps": []}

    def _save(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def check_before_call(self, label: str) -> None:
        """Raise PaidCallGuardError if failure or burst thresholds are tripped."""
        state = self._load()
        failures = int(state.get("consecutive_failures", 0))
        if failures >= self.max_consecutive_failures:
            raise PaidCallGuardError(
                f"Circuit breaker tripped before '{label}': "
                f"{failures} consecutive failures recorded."
            )
        now = time.time()
        cutoff = now - self.window_seconds
        recent = [float(t) for t in state.get("call_timestamps", []) if float(t) >= cutoff]
        if len(recent) >= self.max_calls_in_window:
            raise PaidCallGuardError(
                f"Circuit breaker tripped before '{label}': "
                f"{len(recent)} calls within {self.window_seconds}s window."
            )

    def record_attempt(self, succeeded: bool) -> None:
        """Record call outcome atomically without auto-retrying."""
        now = time.time()
        state = self._load()
        cutoff = now - self.window_seconds
        recent = [float(t) for t in state.get("call_timestamps", []) if float(t) >= cutoff]
        recent.append(now)
        state["call_timestamps"] = recent
        state["consecutive_failures"] = 0 if succeeded else int(state.get("consecutive_failures", 0)) + 1
        self._save(state)
