"""Executor — sells-first, halal defense-in-depth, breaker skip, no fabricated fills."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.execution.breaker import PerAssetBreaker
from halabot.execution.feasibility import FeasibilityConfig
from halabot.execution.orders import ExecutionContext, Executor
from halabot.execution.venue import FakeVenue
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType
from halabot.policy.policy import TradeProposal

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _prop(asset, side, delta, *, version=1) -> TradeProposal:
    return TradeProposal(
        asset=asset, side=side, target_weight=max(0.0, delta), current_weight=0.0,
        weight_delta=delta, reason="test", belief_version=version,
    )


def _build(*, prices=None, fail=None, breaker=None):
    venue = FakeVenue(clock_ts=T0, prices=prices or {}, fail_assets=fail or set())
    bus = InProcessEventBus(InMemoryEventLog())
    clock = FakeClock(T0)
    events: list[Event] = []
    bus.subscribe(set(EventType), lambda e: _cap(events, e))
    ex = Executor(
        venue=venue, bus=bus, clock=clock,
        breaker=breaker or PerAssetBreaker(),
        feasibility=FeasibilityConfig(min_notional_usd=50.0, lot_step=1.0),
    )
    return ex, venue, events


async def _cap(sink, e):
    sink.append(e)


def _ctx(equity=100_000.0, bp=100_000.0, held=None, halal_ok=None):
    held = held or {}
    return ExecutionContext(
        equity=equity, buying_power=bp,
        position_qty=lambda a: held.get(a, 0.0), halal_ok=halal_ok,
    )


@pytest.mark.asyncio
async def test_buy_places_order_and_emits_fill():
    ex, venue, events = _build(prices={"NVDA": 100.0})
    await ex.execute([_prop("NVDA", "buy", 0.1)], _ctx())  # 0.1 × 100k = $10k → 100 sh
    fills = [e for e in events if e.type == EventType.ORDER_FILLED]
    assert len(fills) == 1
    assert fills[0].payload["filled_quantity"] == 100.0
    assert fills[0].payload["belief_version"] == 1
    assert fills[0].payload["engine_owner"] == "belief"


@pytest.mark.asyncio
async def test_sells_execute_before_buys():
    ex, venue, events = _build(prices={"NVDA": 100.0, "AAPL": 50.0})
    props = [_prop("NVDA", "buy", 0.1), _prop("AAPL", "sell", -0.1)]
    await ex.execute(props, _ctx(held={"AAPL": 10.0}))
    # FakeVenue records order placement order; the sell must come first.
    assert [o.side for o in venue.placed][0] == "sell"


@pytest.mark.asyncio
async def test_halal_gate_blocks_buy_in_order_path():
    ex, _, events = _build(prices={"HOOD": 100.0})
    ctx = _ctx(halal_ok=lambda a: a != "HOOD")  # HOOD not tradeable
    await ex.execute([_prop("HOOD", "buy", 0.1)], ctx)
    assert not any(e.type == EventType.ORDER_FILLED for e in events)  # INV-7 defense


@pytest.mark.asyncio
async def test_venue_error_rejects_without_fabricating_fill():
    ex, _, events = _build(prices={"NVDA": 100.0}, fail={"NVDA"})
    await ex.execute([_prop("NVDA", "buy", 0.1)], _ctx())
    assert any(e.type == EventType.ORDER_REJECTED for e in events)
    assert not any(e.type == EventType.ORDER_FILLED for e in events)  # no $0 invent


@pytest.mark.asyncio
async def test_open_breaker_skips_asset():
    breaker = PerAssetBreaker(threshold=1)
    breaker.record_error("NVDA", T0)  # opens
    ex, venue, events = _build(prices={"NVDA": 100.0}, breaker=breaker)
    await ex.execute([_prop("NVDA", "buy", 0.1)], _ctx())
    assert venue.placed == []  # quarantined — never placed


@pytest.mark.asyncio
async def test_repeated_venue_errors_open_breaker():
    ex, _, _ = _build(prices={"NVDA": 100.0}, fail={"NVDA"}, breaker=PerAssetBreaker(threshold=2))
    await ex.execute([_prop("NVDA", "buy", 0.1)], _ctx())  # error 1
    await ex.execute([_prop("NVDA", "buy", 0.1)], _ctx())  # error 2 → opens
    assert ex._breaker.is_open("NVDA", T0)
