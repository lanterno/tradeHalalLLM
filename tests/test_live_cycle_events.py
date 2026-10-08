"""Wave I wiring tests — live cycle events flow through the EventBus.

The ``/ws/cycle`` route + ``EventBus`` itself ship in round-4/5; the
per-stage ``cycle.stage.*`` events publish via the cycle pipeline.
This commit added the missing event sources — executor fills, monitor
exits, LLM call completions — so the dashboard's live stream actually
shows what the bot is doing right now.

The dashboard frontend (a "Live" page rendering the stream as a
collapsible tree) is deferred — the backend half (the data) is what
these tests pin.
"""

from __future__ import annotations

import asyncio

import pytest

from halal_trader.core.event_bus import EventBus

# ── EventBus subscribe helper ───────────────────────────────────


async def _capture(
    bus: EventBus,
    pattern: str,
    *,
    timeout: float = 1.0,
    n: int = 1,
) -> list:
    """Drain ``n`` events from the bus matching ``pattern``, with timeout."""
    captured: list = []

    async def _drain() -> None:
        async for event in bus.subscribe(pattern):
            captured.append(event)
            if len(captured) >= n:
                return

    try:
        await asyncio.wait_for(_drain(), timeout=timeout)
    except TimeoutError:
        pass
    return captured


# ── BaseLLM._record_usage publishes llm.call.complete ───────────


# ── Topic glob filtering ────────────────────────────────────────


@pytest.mark.asyncio
async def test_bus_subscriber_can_filter_by_trade_glob() -> None:
    """The /ws/cycle handler accepts a ``topic`` query param. Confirm
    glob filtering works end-to-end on the bus side (the WS handler is
    a thin pass-through to ``bus.subscribe``)."""
    bus = EventBus()

    captured: list = []

    async def _trade_only() -> None:
        async for event in bus.subscribe("trade.*"):
            captured.append(event)
            if len(captured) >= 2:
                return

    sub_task = asyncio.create_task(_trade_only())
    await asyncio.sleep(0)
    # Mix matching and non-matching events.
    await bus.publish("cycle.stage.start", {"name": "x"})
    await bus.publish("trade.buy.placed", {"symbol": "AAPL"})
    await bus.publish("cycle.stage.end", {"name": "x"})
    await bus.publish("trade.exit.stop_loss", {"trade_id": 1})
    try:
        await asyncio.wait_for(sub_task, timeout=0.5)
    except TimeoutError:
        pass
    topics = [e.topic for e in captured]
    assert "trade.buy.placed" in topics
    assert "trade.exit.stop_loss" in topics
    assert "cycle.stage.start" not in topics
    assert "cycle.stage.end" not in topics


# ── Event constants are correct ─────────────────────────────────


def test_event_constants_referenced_by_wave_i_match() -> None:
    """The publish topics this wave introduced match the canonical
    constants in ``core/events.py``. Drift on these would break the
    dashboard's filter strings."""
    from halal_trader.core import events

    assert events.TRADE_BUY_PLACED == "trade.buy.placed"
    assert events.TRADE_SELL_PLACED == "trade.sell.placed"
    assert events.TRADE_EXIT_SL == "trade.exit.stop_loss"
    assert events.TRADE_EXIT_TP == "trade.exit.take_profit"
    assert events.LLM_CALL_COMPLETE == "llm.call.complete"
