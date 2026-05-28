"""ShadowOutcomeTracker — hypothetical fills → outcomes with realized return (PG)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from halabot.belief.schema import BeliefState, Regime
from halabot.belief.store import PgBeliefStore
from halabot.learning.shadow_outcomes import ShadowOutcomeTracker
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.db import outcome as outcome_table
from halabot.platform.event_log import PgEventLog
from halabot.platform.events import EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


async def _propose(bus, clock, asset, side, weight_delta, price, *, belief_version=1):
    await bus.publish(
        new_event(
            clock, EventType.POLICY_TRADE_PROPOSED, source="policy.shadow", asset=asset,
            payload={"side": side, "weight_delta": weight_delta, "price": price,
                     "belief_version": belief_version, "reason": "test"},
        )
    )


async def _outcomes(engine):
    async with engine.connect() as conn:
        rows = (await conn.execute(sa.select(outcome_table).order_by(outcome_table.c.id))).all()
    return [dict(r._mapping) for r in rows]


def _tracker(halabot_engine):
    bus = InProcessEventBus(PgEventLog(halabot_engine))
    store = PgBeliefStore(halabot_engine)
    tracker = ShadowOutcomeTracker(bus=bus, engine=halabot_engine, store=store)
    tracker.start()
    return bus, store, tracker


@pytest.mark.asyncio
async def test_buy_then_sell_records_winning_outcome(halabot_engine):
    bus, _, tracker = _tracker(halabot_engine)
    clock = FakeClock(T0)
    await _propose(bus, clock, "NVDA", "buy", 0.10, 100.0)
    clock.set(T0 + timedelta(hours=3))
    await _propose(bus, clock, "NVDA", "sell", -0.10, 110.0)  # +10%
    rows = await _outcomes(halabot_engine)
    assert len(rows) == 1
    assert rows[0]["return_pct"] == pytest.approx(0.10)
    assert rows[0]["label"] == 1  # win
    assert rows[0]["hold_seconds"] == 3 * 3600
    assert tracker.closed_count == 1


@pytest.mark.asyncio
async def test_loss_is_labeled_zero(halabot_engine):
    bus, _, _ = _tracker(halabot_engine)
    clock = FakeClock(T0)
    await _propose(bus, clock, "NVDA", "buy", 0.10, 100.0)
    await _propose(bus, clock, "NVDA", "sell", -0.10, 95.0)  # -5%
    rows = await _outcomes(halabot_engine)
    assert rows[0]["return_pct"] == pytest.approx(-0.05)
    assert rows[0]["label"] == 0


@pytest.mark.asyncio
async def test_vwap_blends_on_adds(halabot_engine):
    bus, _, _ = _tracker(halabot_engine)
    clock = FakeClock(T0)
    await _propose(bus, clock, "NVDA", "buy", 0.10, 100.0)
    await _propose(bus, clock, "NVDA", "buy", 0.10, 120.0)  # VWAP → 110
    await _propose(bus, clock, "NVDA", "sell", -0.20, 110.0)  # exit at VWAP → 0%
    rows = await _outcomes(halabot_engine)
    assert len(rows) == 1
    assert rows[0]["entry_price"] == pytest.approx(110.0)
    assert rows[0]["return_pct"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_partial_reduce_then_full_close(halabot_engine):
    bus, _, _ = _tracker(halabot_engine)
    clock = FakeClock(T0)
    await _propose(bus, clock, "NVDA", "buy", 0.20, 100.0)
    await _propose(bus, clock, "NVDA", "sell", -0.10, 110.0)  # close half
    await _propose(bus, clock, "NVDA", "sell", -0.10, 120.0)  # close rest
    rows = await _outcomes(halabot_engine)
    assert len(rows) == 2
    assert [r["closed_weight"] for r in rows] == [pytest.approx(0.10), pytest.approx(0.10)]
    assert rows[1]["return_pct"] == pytest.approx(0.20)  # vs same 100 entry


@pytest.mark.asyncio
async def test_sell_without_position_is_noop(halabot_engine):
    bus, _, _ = _tracker(halabot_engine)
    clock = FakeClock(T0)
    await _propose(bus, clock, "NVDA", "sell", -0.10, 100.0)  # never opened
    assert await _outcomes(halabot_engine) == []


@pytest.mark.asyncio
async def test_missing_price_is_skipped(halabot_engine):
    bus, _, _ = _tracker(halabot_engine)
    clock = FakeClock(T0)
    await bus.publish(
        new_event(
            clock, EventType.POLICY_TRADE_PROPOSED, source="policy.shadow", asset="NVDA",
            payload={"side": "buy", "weight_delta": 0.1, "price": None, "belief_version": 1},
        )
    )
    await _propose(bus, clock, "NVDA", "sell", -0.1, 110.0)
    assert await _outcomes(halabot_engine) == []  # no entry was recorded


@pytest.mark.asyncio
async def test_entry_belief_snapshot_captured(halabot_engine):
    bus, store, _ = _tracker(halabot_engine)
    b = BeliefState.neutral("NVDA")
    b.regime = Regime.TRENDING_UP
    b.conviction = 0.7
    v = await store.put(b)  # version 1
    clock = FakeClock(T0)
    await _propose(bus, clock, "NVDA", "buy", 0.10, 100.0, belief_version=v)
    await _propose(bus, clock, "NVDA", "sell", -0.10, 110.0, belief_version=v)
    rows = await _outcomes(halabot_engine)
    assert rows[0]["entry_belief"]["regime"] == "trending_up"
    assert rows[0]["entry_belief"]["conviction"] == 0.7
