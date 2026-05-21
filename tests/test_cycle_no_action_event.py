"""Pin the cycle.no_action event for fully-rejected plans.

On 2026-05-21 12:00 ET the bot ran cycle-08b16f0b: LLM proposed 2
buys, both rejected by the position cap, 0 trades executed but the
LLM call cost ~4s anyway. Without a structured event for this case,
wasted cycles are invisible in the JSON log.
"""

from __future__ import annotations

import logging

import pytest

from halal_trader.core import events
from halal_trader.trading.cycle import TradingCycleService


def _cycle(monkeypatch) -> TradingCycleService:
    """Build a minimally-wired cycle service. _handle_execution_results
    only touches self._self_review + self._notifier, both of which we
    set to None to exercise the no-op + event-emission path."""
    svc = TradingCycleService.__new__(TradingCycleService)
    svc._self_review = None  # type: ignore[attr-defined]
    svc._notifier = None  # type: ignore[attr-defined]
    return svc


@pytest.mark.asyncio
async def test_all_rejected_emits_no_action(monkeypatch, caplog):
    svc = _cycle(monkeypatch)
    results = [
        {"symbol": "QCOM", "action": "buy", "status": "rejected", "reason": "Max positions"},
        {"symbol": "TXN", "action": "buy", "status": "rejected", "reason": "Max positions"},
    ]
    with caplog.at_level(logging.WARNING, logger="halal_trader.trading.cycle"):
        await svc._handle_execution_results(results)

    no_action_records = [
        r for r in caplog.records if getattr(r, "event", None) == events.CYCLE_NO_ACTION
    ]
    assert len(no_action_records) == 1
    rec = no_action_records[0]
    assert rec.proposed == 2  # type: ignore[attr-defined]
    assert rec.rejected == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mixed_results_no_event(monkeypatch, caplog):
    svc = _cycle(monkeypatch)
    results = [
        {"symbol": "QCOM", "action": "buy", "status": "rejected", "reason": "Max positions"},
        {"symbol": "NVDA", "action": "buy", "status": "filled", "filled_quantity": 10},
    ]
    with caplog.at_level(logging.WARNING, logger="halal_trader.trading.cycle"):
        await svc._handle_execution_results(results)

    no_action_records = [
        r for r in caplog.records if getattr(r, "event", None) == events.CYCLE_NO_ACTION
    ]
    assert no_action_records == []


@pytest.mark.asyncio
async def test_empty_plan_no_event(monkeypatch, caplog):
    """An empty plan (LLM had nothing to do) is silent — only fully-
    rejected plans surface the wasted-cycle event."""
    svc = _cycle(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="halal_trader.trading.cycle"):
        await svc._handle_execution_results([])

    no_action_records = [
        r for r in caplog.records if getattr(r, "event", None) == events.CYCLE_NO_ACTION
    ]
    assert no_action_records == []


@pytest.mark.asyncio
async def test_skipped_counts_as_no_action(monkeypatch, caplog):
    """A `skipped` result (e.g. close_position with no open position) counts
    toward no-action since no trade happened."""
    svc = _cycle(monkeypatch)
    results = [
        {"symbol": "AAPL", "action": "sell", "status": "skipped", "reason": "no open position"},
    ]
    with caplog.at_level(logging.WARNING, logger="halal_trader.trading.cycle"):
        await svc._handle_execution_results(results)

    no_action_records = [
        r for r in caplog.records if getattr(r, "event", None) == events.CYCLE_NO_ACTION
    ]
    assert len(no_action_records) == 1
