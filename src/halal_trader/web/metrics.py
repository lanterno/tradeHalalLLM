"""Read-only metrics computed from the JSON log file.

The bot already writes structured records via `core/observability.py`
and the event constants in `core/events.py`. The dashboard's
``/api/metrics/*`` endpoints lazily tail the log file and aggregate
what's there — no Prometheus, no extra infrastructure.

Bounded by ``max_lines`` so a multi-GB log doesn't read into memory.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from halal_trader.core import events

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINES = 50_000


@dataclass(frozen=True)
class CycleMetrics:
    window_seconds: int
    count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    failed: int
    halted: int


@dataclass(frozen=True)
class LlmMetrics:
    window_seconds: int
    calls: int
    total_tokens: int
    p50_ms: float | None
    p95_ms: float | None
    by_provider: dict[str, dict[str, float | int]]


def _tail(path: Path, max_lines: int) -> Iterator[str]:
    """Yield up to ``max_lines`` lines from the END of ``path`` newest-first.

    For a typical 10MB rotating log the file is small enough to read in
    one pass; we don't bother seeking. If the file grows past the
    rotation cap something else is wrong.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        # `deque(..., maxlen=N)` keeps the last N lines in O(1) memory.
        for line in deque(f, maxlen=max_lines):
            yield line.rstrip("\n")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _within(record: dict, since: datetime) -> bool:
    ts = record.get("timestamp")
    if not isinstance(ts, str):
        return False
    try:
        # `python-json-logger` emits asctime; tolerate both ISO and asctime forms.
        if "T" in ts:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except ValueError:
        return False
    return dt >= since


def _iter_records(path: Path, max_lines: int) -> Iterator[dict]:
    for line in _tail(path, max_lines):
        try:
            yield json.loads(line)
        except Exception:
            continue


def cycle_metrics(
    log_path: Path,
    *,
    window_seconds: int = 3600,
    max_lines: int = DEFAULT_MAX_LINES,
    now: datetime | None = None,
) -> CycleMetrics:
    """Compute cycle latency percentiles + failure / halt counts in the window."""
    now = now or datetime.now(UTC)
    since = now - timedelta(seconds=window_seconds)

    elapsed: list[float] = []
    failed = 0
    halted = 0
    for record in _iter_records(log_path, max_lines):
        if not _within(record, since):
            continue
        event = record.get("event")
        if event == events.CYCLE_COMPLETE:
            ms = record.get("elapsed_ms")
            if isinstance(ms, (int, float)):
                elapsed.append(float(ms))
        elif event == events.CYCLE_FAILED:
            failed += 1
        elif event == events.CYCLE_HALTED:
            halted += 1

    return CycleMetrics(
        window_seconds=window_seconds,
        count=len(elapsed),
        p50_ms=_percentile(elapsed, 0.50),
        p95_ms=_percentile(elapsed, 0.95),
        p99_ms=_percentile(elapsed, 0.99),
        failed=failed,
        halted=halted,
    )


def llm_metrics(
    log_path: Path,
    *,
    window_seconds: int = 86400,
    max_lines: int = DEFAULT_MAX_LINES,
    now: datetime | None = None,
) -> LlmMetrics:
    """Aggregate LLM call counts + tokens + latency by provider in the window."""
    now = now or datetime.now(UTC)
    since = now - timedelta(seconds=window_seconds)

    elapsed_all: list[float] = []
    by_provider: dict[str, dict[str, float | int]] = {}
    total_tokens = 0
    calls = 0

    for record in _iter_records(log_path, max_lines):
        if not _within(record, since):
            continue
        if record.get("event") != events.LLM_CALL_COMPLETE:
            continue
        provider = str(record.get("provider") or "unknown")
        tokens = record.get("tokens")
        ms = record.get("elapsed_ms")

        bucket = by_provider.setdefault(
            provider,
            {"calls": 0, "tokens": 0, "elapsed_ms_list": []},
        )
        bucket["calls"] = int(bucket["calls"]) + 1
        if isinstance(tokens, (int, float)):
            bucket["tokens"] = int(bucket["tokens"]) + int(tokens)
            total_tokens += int(tokens)
        if isinstance(ms, (int, float)):
            bucket["elapsed_ms_list"].append(float(ms))
            elapsed_all.append(float(ms))
        calls += 1

    # Collapse the per-provider list into p50.
    for bucket in by_provider.values():
        ms_list = bucket.pop("elapsed_ms_list", [])
        bucket["p50_ms"] = _percentile(ms_list, 0.50) or 0.0

    return LlmMetrics(
        window_seconds=window_seconds,
        calls=calls,
        total_tokens=total_tokens,
        p50_ms=_percentile(elapsed_all, 0.50),
        p95_ms=_percentile(elapsed_all, 0.95),
        by_provider=by_provider,
    )
