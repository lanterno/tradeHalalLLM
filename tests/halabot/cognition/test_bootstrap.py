"""CognitionRouter.bootstrap — warm beliefs from the event log (Appendix F).

Replay is event-time + is_replay (no invalidation/orders), then decays each
warmed belief to `now`. Proves a restart can rebuild beliefs from history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.belief.evidence import ContinuousCalendar
from halabot.belief.schema import Direction, Levels, Regime
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
    def __init__(self, buffer: BarBuffer):
        self._buffer = buffer

    def last_price(self, asset: str) -> float | None:
        closes = self._buffer.closes(asset)
        return closes[-1] if closes else None


class _NoPositions:
    def has_position(self, asset: str) -> bool:
        return False


class _NoThesis:
    async def write(self, belief) -> str:
        return ""


class _OffLLM:
    def available(self) -> bool:
        return False

    def breaker_open(self) -> bool:
        return True


def _build(*, levels=None):
    """Return a NON-started router + its bus/store/captured-events sink.

    The router is not subscribed, so seeding bars onto the bus only appends them
    to the log; bootstrap then replays them from the log."""
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
        levels=levels or BarLevelEngine(buffer),
        calibrator=IdentityCalibrator(),
        thesis_writer=_NoThesis(),
        prices=_BufferPrices(buffer),
        positions=_NoPositions(),
        llm=_OffLLM(),
        config=UpdaterConfig(),
    )
    router = CognitionRouter(
        bus=bus,
        updater=updater,
        buffer=buffer,
        interpreters=[IndicatorInterpreter(buffer), NewsLexiconInterpreter()],
    )
    captured: list[Event] = []
    bus.subscribe(
        {EventType.BELIEF_UPDATED, EventType.BELIEF_INVALIDATED}, lambda e: _cap(captured, e)
    )
    return router, bus, store, captured


async def _cap(sink, e):
    sink.append(e)


async def _seed_uptrend(bus, *, asset="NVDA", n=30, start=100.0):
    """Append historical bars to the log (publish; no router subscribed yet)."""
    clk = FakeClock(T0)
    for _ in range(n):
        clk.advance(timedelta(minutes=1))
        c = start
        start += 1
        await bus.publish(
            new_event(
                clk, EventType.OBSERVATION_BAR, source="alpaca", asset=asset,
                payload={"o": c, "h": c + 1, "low": c - 1, "c": c, "v": 1000.0},
            )
        )


class _FixedLevels:
    """Level engine that always returns a fixed invalidation below price."""

    async def levels_for(self, asset, prev):
        return Levels(invalidation=10_000.0, stop=10_000.0)  # absurdly high → price < it


@pytest.mark.asyncio
async def test_bootstrap_warms_belief_from_history():
    router, bus, store, _ = _build()
    await _seed_uptrend(bus)
    now = T0 + timedelta(hours=2)
    warmed = await router.bootstrap(since=T0, until=now, now=now)
    assert "NVDA" in warmed
    b = await store.get("NVDA")
    assert b is not None
    assert b.direction == Direction.LONG_BIAS
    assert b.regime == Regime.TRENDING_UP
    assert b.conviction > 0.0
    assert b.last_updated == now  # decayed forward to the present


@pytest.mark.asyncio
async def test_bootstrap_suppresses_invalidation_against_history():
    """is_replay must suppress belief.invalidated — replay never trades against
    historical prices even when a level is breached."""
    router, bus, store, captured = _build(levels=_FixedLevels())  # invalidation way above price
    await _seed_uptrend(bus)
    now = T0 + timedelta(hours=2)
    await router.bootstrap(since=T0, until=now, now=now)
    assert any(e.type == EventType.BELIEF_UPDATED for e in captured)  # beliefs warmed
    assert not any(e.type == EventType.BELIEF_INVALIDATED for e in captured)  # no exit fired


@pytest.mark.asyncio
async def test_bootstrap_empty_history_is_noop():
    router, bus, store, _ = _build()
    now = T0 + timedelta(hours=2)
    warmed = await router.bootstrap(since=T0, until=now, now=now)
    assert warmed == frozenset()
    assert await store.all_active() == []
