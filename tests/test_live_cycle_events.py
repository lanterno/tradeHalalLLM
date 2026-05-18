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
from unittest.mock import AsyncMock, MagicMock

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
    except asyncio.TimeoutError:
        pass
    return captured


# ── BaseLLM._record_usage publishes llm.call.complete ───────────


def test_base_llm_attach_bus_stores_ref() -> None:
    """attach_bus is the only public seam for wiring the bus into a
    provider after construction. Pin the ref-stash behaviour."""
    from halal_trader.core.llm.base import BaseLLM

    class _Dummy(BaseLLM):
        async def generate(self, prompt, system=None):  # type: ignore[override]
            return ""

    llm = _Dummy(model="x")
    assert llm._bus is None
    bus = EventBus()
    llm.attach_bus(bus)
    assert llm._bus is bus


@pytest.mark.asyncio
async def test_record_usage_publishes_llm_call_complete() -> None:
    """A non-zero-elapsed usage event with provider+model triggers a
    bus publish of ``llm.call.complete``."""
    from halal_trader.core.llm.base import BaseLLM, CallUsage

    class _Dummy(BaseLLM):
        async def generate(self, prompt, system=None):  # type: ignore[override]
            return ""

    bus = EventBus()
    llm = _Dummy(model="claude-x")
    llm.attach_bus(bus)

    async def _trigger() -> None:
        # Subscribe first, then fire the synchronous _record_usage.
        # _record_usage schedules a task on the running loop.
        usage = CallUsage(
            provider="anthropic",
            model="claude-x",
            input_tokens=100,
            output_tokens=20,
            elapsed_ms=1234,
        )
        llm._record_usage(usage)
        # Yield once so the scheduled publish runs.
        await asyncio.sleep(0.01)

    sub_task = asyncio.create_task(_capture(bus, "llm.call.complete", n=1))
    await asyncio.sleep(0)  # let subscribe register
    await _trigger()
    events = await sub_task
    assert events, "no llm.call.complete event published"
    assert events[0].payload["provider"] == "anthropic"
    assert events[0].payload["model"] == "claude-x"
    assert events[0].payload["elapsed_ms"] == 1234


@pytest.mark.asyncio
async def test_record_usage_swallows_publish_without_loop() -> None:
    """No running loop → silent skip (some test contexts call _record_usage
    synchronously from outside an event loop)."""
    from halal_trader.core.llm.base import BaseLLM, CallUsage

    class _Dummy(BaseLLM):
        async def generate(self, prompt, system=None):  # type: ignore[override]
            return ""

    bus = MagicMock()
    bus.publish = AsyncMock()
    llm = _Dummy(model="x")
    llm.attach_bus(bus)
    # Simulate "no running loop" by patching get_running_loop to raise.
    import asyncio as _aio

    orig = _aio.get_running_loop
    _aio.get_running_loop = MagicMock(side_effect=RuntimeError("no loop"))
    try:
        llm._record_usage(
            CallUsage(provider="x", model="m", elapsed_ms=10),
        )
    finally:
        _aio.get_running_loop = orig
    bus.publish.assert_not_called()  # silent skip


# ── CryptoExecutor publishes trade.* events ─────────────────────


@pytest.mark.asyncio
async def test_executor_publish_event_is_a_noop_without_bus() -> None:
    """Default-constructed executor (no bus) → _publish_event is a no-op."""
    from halal_trader.crypto.executor import CryptoExecutor

    ex = CryptoExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.25,
        max_simultaneous_positions=5,
    )
    # Must not raise even though no bus is wired.
    await ex._publish_event("trade.buy.placed", {"pair": "BTCUSDT"})


@pytest.mark.asyncio
async def test_executor_publish_event_routes_to_bus() -> None:
    """With a bus wired, _publish_event forwards to bus.publish."""
    from halal_trader.crypto.executor import CryptoExecutor

    bus = EventBus()
    ex = CryptoExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.25,
        max_simultaneous_positions=5,
        bus=bus,
    )
    sub_task = asyncio.create_task(_capture(bus, "trade.buy.placed", n=1))
    await asyncio.sleep(0)
    await ex._publish_event(
        "trade.buy.placed",
        {"pair": "BTCUSDT", "order_id": "x", "status": "filled"},
    )
    events = await sub_task
    assert events
    assert events[0].payload["pair"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_executor_publish_swallows_bus_failure() -> None:
    """A failing bus.publish must not propagate — the executor's fill
    path can't be blocked by a stuck /ws/cycle subscriber."""
    from halal_trader.crypto.executor import CryptoExecutor

    bus = MagicMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    ex = CryptoExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.25,
        max_simultaneous_positions=5,
        bus=bus,
    )
    # Must not raise.
    await ex._publish_event("trade.buy.placed", {"pair": "BTCUSDT"})


# ── PositionMonitor publishes trade.exit.* ─────────────────────


@pytest.mark.asyncio
async def test_monitor_publish_event_routes_to_bus() -> None:
    from halal_trader.crypto.monitor import PositionMonitor

    bus = EventBus()
    mon = PositionMonitor(
        broker=MagicMock(),
        repo=MagicMock(),
        ws_manager=MagicMock(),
        bus=bus,
    )
    sub_task = asyncio.create_task(_capture(bus, "trade.exit.stop_loss", n=1))
    await asyncio.sleep(0)
    await mon._publish_event(
        "trade.exit.stop_loss",
        {"trade_id": 42, "pair": "BTCUSDT", "price": 50_000.0},
    )
    events = await sub_task
    assert events
    assert events[0].payload["trade_id"] == 42


@pytest.mark.asyncio
async def test_monitor_publish_event_noop_without_bus() -> None:
    from halal_trader.crypto.monitor import PositionMonitor

    mon = PositionMonitor(
        broker=MagicMock(),
        repo=MagicMock(),
        ws_manager=MagicMock(),
    )
    await mon._publish_event("trade.exit.take_profit", {"trade_id": 1})  # no raise


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
    await bus.publish("trade.buy.placed", {"pair": "BTCUSDT"})
    await bus.publish("cycle.stage.end", {"name": "x"})
    await bus.publish("trade.exit.stop_loss", {"trade_id": 1})
    try:
        await asyncio.wait_for(sub_task, timeout=0.5)
    except asyncio.TimeoutError:
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
