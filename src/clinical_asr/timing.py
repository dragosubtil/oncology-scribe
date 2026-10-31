"""Task clocks and conservative, machine-local estimates from successful runs."""

from __future__ import annotations

import hashlib
import html
import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path


def _positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


class TimingHistory:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.machine = hashlib.sha256(
            (platform.node() + platform.processor() + platform.platform()).encode()
        ).hexdigest()
        self.records = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("machine") == self.machine and isinstance(data.get("records"), dict):
                for key, rows in data["records"].items():
                    if isinstance(rows, list):
                        valid = [row for row in rows if isinstance(row, list) and len(row) == 2
                                 and all(_positive(value) for value in row)]
                        self.records[key] = valid[-12:]
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    def record(self, key: str, units: float, seconds: float):
        if not _positive(units) or not _positive(seconds):
            raise ValueError("Timing measurements must be positive and finite")
        with self._lock:
            self.records[key] = (self.records.get(key, []) + [[units, seconds]])[-12:]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"machine": self.machine, "records": self.records},
                                                allow_nan=False), encoding="utf-8")
                temporary.replace(self.path)
            except OSError:
                pass

    def estimate(self, key: str, units: float) -> float | None:
        if not _positive(units):
            return None
        with self._lock:
            rows = [row for row in self.records.get(key, []) if 0.5 <= units / row[0] <= 2]
        if len(rows) < 3:
            return None
        return statistics.median(seconds / size for size, seconds in rows) * units


def begin(label: str, expected_seconds: float | None = None, now: float | None = None) -> dict:
    return {"label": label, "status": "queued", "queued_at": time.monotonic() if now is None else now,
            "expected_seconds": expected_seconds}


def processing(state: dict, now: float | None = None) -> dict:
    return state | {"status": "processing", "started_at": time.monotonic() if now is None else now}


def finish(state: dict, succeeded: bool, now: float | None = None) -> dict:
    ended = time.monotonic() if now is None else now
    return state | {"status": "completed" if succeeded else "failed", "ended_at": ended,
                    "processing_seconds": max(0, ended - state.get("started_at", ended))}


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def render(state: dict | None, now: float | None = None, generation: int = 0) -> str:
    if not state:
        return (f'<div class="task-status task-idle" role="status" data-task-clock '
                f'data-generation="{generation}" data-status="idle">Ready</div>')
    current = time.monotonic() if now is None else now
    ended = state.get("ended_at", current)
    elapsed = max(0, ended - state["queued_at"])
    label = html.escape(state["label"])
    status = state["status"]
    terminal = status in {"completed", "failed"}
    parts = [f"<strong>{label}: {status.title()}</strong>",
             f'<span data-clock-elapsed>{"Total" if terminal else "Elapsed"} {_duration(elapsed)}</span>']
    if status in {"completed", "failed"}:
        parts.append(f"Processing {_duration(state['processing_seconds'])}")
    elif state.get("expected_seconds") is None:
        parts.append('<span data-clock-eta>ETA: Not enough timing data</span>')
    elif status == "queued":
        parts.append(f"Estimated processing {_duration(state['expected_seconds'])}; queue time unknown")
    else:
        remaining = state["expected_seconds"] - max(0, current - state["started_at"])
        parts.append('<span data-clock-eta>' + (f"ETA ~{_duration(remaining)}" if remaining > 0
                                               else "Taking longer than estimated") + '</span>')
    expected = state.get("expected_seconds")
    processed = max(0, current - state.get("started_at", current))
    return (f'<div class="task-status task-{status}" role="status" data-task-clock '
            f'data-generation="{state.get("generation", generation)}" data-status="{status}" '
            f'data-task-kind="{html.escape(str(state.get("kind") or ""))}" '
            f'data-event-order="{state.get("event_order") or 0}" '
            f'data-elapsed="{elapsed}" data-processed="{processed}" '
            f'data-expected="{expected if expected is not None else ""}">'
            + "<span>" + "</span><span>".join(parts) + "</span></div>")
