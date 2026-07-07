"""Tests for web/metrics.py — JSON-tail derived cycle + LLM stats."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from halal_trader.core import events
from halal_trader.web import metrics


def _write_log(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "halal_trader.log"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _record(event: str, *, ts: datetime, **extra) -> dict:
    return {
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "level": "INFO",
        "name": "halal_trader.test",
        "message": event,
        "event": event,
        **extra,
    }


def test_cycle_metrics_empty_log(tmp_path):
    log = _write_log(tmp_path, [])
    m = metrics.cycle_metrics(log, window_seconds=3600)
    assert m.count == 0 and m.p50_ms is None
    assert m.failed == 0 and m.halted == 0


def test_cycle_metrics_basic_percentiles(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(events.CYCLE_COMPLETE, ts=now - timedelta(minutes=i), elapsed_ms=ms)
        for i, ms in enumerate([100, 200, 300, 400, 500])
    ]
    log = _write_log(tmp_path, records)

    m = metrics.cycle_metrics(log, window_seconds=3600, now=now)
    assert m.count == 5
    assert m.p50_ms == 300
    assert m.p95_ms == 500
    assert m.p99_ms == 500


def test_cycle_metrics_window_boundary_excludes_old(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(events.CYCLE_COMPLETE, ts=now - timedelta(minutes=10), elapsed_ms=100),
        _record(events.CYCLE_COMPLETE, ts=now - timedelta(hours=2), elapsed_ms=999),
    ]
    log = _write_log(tmp_path, records)

    m = metrics.cycle_metrics(log, window_seconds=3600, now=now)
    assert m.count == 1
    assert m.p50_ms == 100


def test_cycle_metrics_counts_failed_and_halted(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(events.CYCLE_COMPLETE, ts=now, elapsed_ms=100),
        _record(events.CYCLE_FAILED, ts=now),
        _record(events.CYCLE_FAILED, ts=now),
        _record(events.CYCLE_HALTED, ts=now),
    ]
    log = _write_log(tmp_path, records)
    m = metrics.cycle_metrics(log, window_seconds=3600, now=now)
    assert m.count == 1
    assert m.failed == 2
    assert m.halted == 1


def test_llm_metrics_aggregates_by_provider(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(
            events.LLM_CALL_COMPLETE,
            ts=now - timedelta(minutes=10),
            provider="openai",
            elapsed_ms=400,
            tokens=500,
        ),
        _record(
            events.LLM_CALL_COMPLETE,
            ts=now - timedelta(minutes=5),
            provider="openai",
            elapsed_ms=600,
            tokens=700,
        ),
        _record(
            events.LLM_CALL_COMPLETE,
            ts=now,
            provider="ollama",
            elapsed_ms=800,
            tokens=None,
        ),
    ]
    log = _write_log(tmp_path, records)
    m = metrics.llm_metrics(log, window_seconds=3600, now=now)
    assert m.calls == 3
    assert m.total_tokens == 1200
    assert m.by_provider["openai"]["calls"] == 2
    assert m.by_provider["openai"]["tokens"] == 1200
    # Index-based p50 over [400, 600] picks the lower of the pair.
    assert m.by_provider["openai"]["p50_ms"] == 400
    assert m.by_provider["ollama"]["calls"] == 1
    assert m.by_provider["ollama"]["tokens"] == 0  # None tokens summed as 0


def test_llm_metrics_sums_cost_usd(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(
            events.LLM_CALL_COMPLETE,
            ts=now - timedelta(minutes=5),
            provider="z-ai/glm-5.2",
            elapsed_ms=400,
            tokens=500,
            cost_usd=0.0003,
        ),
        _record(
            events.LLM_CALL_COMPLETE,
            ts=now,
            provider="z-ai/glm-5.2",
            elapsed_ms=600,
            tokens=700,
            cost_usd=0.0005,
        ),
        # A record with no cost_usd must not break the sum.
        _record(
            events.LLM_CALL_COMPLETE,
            ts=now,
            provider="z-ai/glm-5.2",
            elapsed_ms=500,
            tokens=100,
        ),
    ]
    log = _write_log(tmp_path, records)
    m = metrics.llm_metrics(log, window_seconds=3600, now=now)
    assert m.calls == 3
    assert m.total_cost_usd == 0.0008


def test_recent_rejections_parses_and_categorizes(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(
            events.CYCLE_NO_ACTION,
            ts=now - timedelta(minutes=5),
            cycle_id="cyc-1",
            reasons=[
                "AAPL:Position size for AAPL exceeds 20% limit (held $10k)",
                "ADBE:recent-close cooldown: ADBE sold 15 min ago",
            ],
        ),
        _record(
            events.CYCLE_NO_ACTION,
            ts=now,
            cycle_id="cyc-2",
            reasons=["INTU:stop-loss re-entry gate: INTU was stopped out 30m ago"],
        ),
        # Outside the window — must be excluded.
        _record(
            events.CYCLE_NO_ACTION,
            ts=now - timedelta(hours=48),
            cycle_id="old",
            reasons=["MSFT:stale"],
        ),
    ]
    log = _write_log(tmp_path, records)
    rows = metrics.recent_rejections(log, window_seconds=3600, now=now)
    assert len(rows) == 3  # 2 from cyc-1 + 1 from cyc-2, old one excluded
    # Newest-first: cyc-2's row leads.
    assert rows[0]["symbol"] == "INTU"
    assert rows[0]["category"] == "stop_loss_reentry"
    cats = {r["symbol"]: r["category"] for r in rows}
    assert cats["AAPL"] == "concentration_cap"
    assert cats["ADBE"] == "recent_close_cooldown"
    # The "SYMBOL:" prefix is stripped from the surfaced reason.
    assert rows[0]["reason"].startswith("stop-loss re-entry gate")


def test_recent_rejections_ignores_non_ticker_prefix(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(
            events.CYCLE_NO_ACTION,
            ts=now,
            cycle_id="c",
            reasons=["some free-form reason: with a colon but no ticker"],
        ),
    ]
    log = _write_log(tmp_path, records)
    rows = metrics.recent_rejections(log, window_seconds=3600, now=now)
    assert len(rows) == 1
    assert rows[0]["symbol"] is None
    assert rows[0]["reason"] == "some free-form reason: with a colon but no ticker"


def test_llm_metrics_skips_unknown_events(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    records = [
        _record(events.CYCLE_COMPLETE, ts=now, elapsed_ms=100),
        _record(
            events.LLM_CALL_COMPLETE,
            ts=now,
            provider="openai",
            elapsed_ms=400,
            tokens=500,
        ),
    ]
    log = _write_log(tmp_path, records)
    m = metrics.llm_metrics(log, window_seconds=3600, now=now)
    assert m.calls == 1


def test_metrics_tolerate_malformed_lines(tmp_path):
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    p = tmp_path / "halal_trader.log"
    with p.open("w", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write(json.dumps(_record(events.CYCLE_COMPLETE, ts=now, elapsed_ms=42)) + "\n")
        f.write("{broken json\n")
    m = metrics.cycle_metrics(p, window_seconds=3600, now=now)
    assert m.count == 1


def test_cycle_metrics_handles_missing_file(tmp_path):
    m = metrics.cycle_metrics(tmp_path / "does-not-exist.log")
    assert m.count == 0
