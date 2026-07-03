"""End-to-end: bar/news observations → CognitionRouter → continuous beliefs.

Proves the Phase-2 "always-on understanding" loop with the real stack (in-memory
store/bus/updater + real regime classifier, level engine, indicator + news
interpreters). Read-only: only belief.* events are emitted, never orders.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.belief.evidence import ContinuousCalendar
from halabot.belief.schema import Direction, Regime
from halabot.belief.store import InMemoryBeliefStore
from halabot.belief.updater import BeliefUpdater, UpdaterConfig
from halabot.cognition.bars import BarBuffer
from halabot.cognition.interpreters import IndicatorInterpreter, NewsLexiconInterpreter
from halabot.cognition.level_engine import BarLevelEngine
from halabot.cognition.regime import EvidenceRegimeClassifier
from halabot.cognition.router import CognitionRouter
from halabot.conviction.raw import IdentityCalibrator
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


class _BufferPrices:
    """Last close from the buffer — a realistic price source for the monitor."""

    def __init__(self, buffer: BarBuffer):
        self._buffer = buffer

    def last_price(self, asset: str) -> float | None:
        closes = self._buffer.closes(asset)
        return closes[-1] if closes else None


class _NoPositions:
    def has_position(self, asset: str) -> bool:
        return False


class _Thesis:
    def __init__(self):
        self.calls = 0

    async def write(self, belief) -> str:
        self.calls += 1
        return "thesis"


class _LLM:
    def available(self) -> bool:
        return True

    def breaker_open(self) -> bool:
        return False


def _build():
    store = InMemoryBeliefStore()
    bus = InProcessEventBus(InMemoryEventLog())
    clock = FakeClock(T0)
    buffer = BarBuffer()
    updater = BeliefUpdater(
        store=store,
        bus=bus,
        clock=clock,
        calendar=ContinuousCalendar(),
        regime=EvidenceRegimeClassifier(),
        levels=BarLevelEngine(buffer),
        calibrator=IdentityCalibrator(),
        thesis_writer=_Thesis(),
        prices=_BufferPrices(buffer),
        positions=_NoPositions(),
        llm=_LLM(),
        config=UpdaterConfig(),
    )
    router = CognitionRouter(bus=bus, updater=updater, buffer=buffer,
                             interpreters=[IndicatorInterpreter(buffer), NewsLexiconInterpreter()])
    router.start()
    captured: list[Event] = []
    bus.subscribe(
        {EventType.BELIEF_UPDATED, EventType.BELIEF_INVALIDATED}, lambda e: _cap(captured, e)
    )
    return bus, clock, store, captured


async def _cap(sink, e):
    sink.append(e)


async def _feed_uptrend(bus, clock, *, asset="NVDA", n=30, start=100.0):
    for i in range(n):
        clock.advance(timedelta(minutes=1))
        c = start + i
        await bus.publish(
            new_event(
                clock, EventType.OBSERVATION_BAR, source="alpaca", asset=asset,
                payload={"o": c, "h": c + 1, "low": c - 1, "c": c, "v": 1000.0},
            )
        )


@pytest.mark.asyncio
async def test_uptrend_bars_form_a_bullish_belief():
    bus, clock, store, events = _build()
    await _feed_uptrend(bus, clock)

    b = await store.get("NVDA")
    assert b is not None
    assert b.direction == Direction.LONG_BIAS
    assert b.regime == Regime.TRENDING_UP
    assert b.conviction > 0.0
    assert b.levels.invalidation is not None        # ATR-based stop established
    assert b.version >= 1
    # only belief.* emitted — the loop never produced an order event (read-only)
    assert events and all(
        e.type in {EventType.BELIEF_UPDATED, EventType.BELIEF_INVALIDATED} for e in events
    )


@pytest.mark.asyncio
async def test_bullish_news_raises_conviction_and_adds_evidence():
    bus, clock, store, _ = _build()
    await _feed_uptrend(bus, clock)
    before = (await store.get("NVDA")).conviction

    clock.advance(timedelta(minutes=1))
    await bus.publish(
        new_event(
            clock, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
            payload={"lexicon_polarity": 0.9, "headline": "surprise beat", "url": "http://n"},
        )
    )
    after = await store.get("NVDA")
    assert after.conviction >= before
    assert any(e.source == "news" for e in after.evidence)


@pytest.mark.asyncio
async def test_heartbeat_decays_conviction_without_new_data():
    bus, clock, store, _ = _build()
    await _feed_uptrend(bus, clock)
    before = (await store.get("NVDA")).conviction

    # Long quiet gap, then a heartbeat → decay-only update fades conviction (R-08).
    clock.advance(timedelta(hours=12))
    await bus.publish(new_event(clock, EventType.SYSTEM_HEARTBEAT, source="heartbeat"))
    after = (await store.get("NVDA")).conviction
    assert after < before


@pytest.mark.asyncio
async def test_router_ignores_assetless_and_unknown_safely():
    bus, clock, store, _ = _build()
    # A heartbeat with no known assets yet must not raise.
    await bus.publish(new_event(clock, EventType.SYSTEM_HEARTBEAT, source="heartbeat"))
    assert await store.all_active() == []


@pytest.mark.asyncio
async def test_macro_observation_lands_in_catalysts_pending():
    """observation.macro → router side-channel → BeliefState.catalysts_pending
    (Task B slice 1 — the formerly dormant seam, end to end through the bus)."""
    bus, clock, store, captured = _build()
    scheduled = (T0 + timedelta(days=2)).isoformat()
    await bus.publish(
        new_event(
            clock, EventType.OBSERVATION_MACRO, source="macro-catalysts", asset="NVDA",
            payload={"kind": "CPI", "asset": "NVDA", "scheduled_for": scheduled,
                     "expected_impact": 0.9, "actual": None, "consensus": None,
                     "detail": "CPI release"},
        )
    )
    b = await store.get("NVDA")
    assert b is not None
    assert [c.kind for c in b.catalysts_pending] == ["CPI"]
    assert b.catalysts_pending[0].expected_impact == pytest.approx(0.9)
    assert any(
        e.type == EventType.BELIEF_UPDATED and e.source == "belief.catalyst" for e in captured
    )


@pytest.mark.asyncio
async def test_malformed_macro_observation_dropped_without_belief():
    bus, clock, store, captured = _build()
    await bus.publish(
        new_event(
            clock, EventType.OBSERVATION_MACRO, source="macro-catalysts", asset="NVDA",
            payload={"kind": "CPI", "asset": "NVDA", "scheduled_for": "not-a-date",
                     "expected_impact": 0.9, "actual": None, "consensus": None},
        )
    )
    assert await store.get("NVDA") is None
    assert captured == []
