"""PgEventLog — Postgres durability + ordered replay (requires PG on :5433)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import PgEventLog
from halabot.platform.events import Event, EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _at(minute: int) -> FakeClock:
    return FakeClock(datetime(2026, 5, 28, 12, minute, tzinfo=UTC))


async def _collect(log: PgEventLog, **kw) -> list[Event]:
    return [e async for e in log.replay(**kw)]


@pytest.mark.asyncio
async def test_append_and_replay_roundtrip(halabot_engine):
    log = PgEventLog(halabot_engine)
    e = new_event(_at(0), EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
                  payload={"headline": "beat", "score": 0.9})
    await log.append(e)
    out = await _collect(log)
    assert len(out) == 1
    got = out[0]
    assert got.id == e.id
    assert got.type == EventType.OBSERVATION_NEWS
    assert got.asset == "NVDA"
    assert got.payload == {"headline": "beat", "score": 0.9}
    assert got.correlation_id == e.correlation_id


@pytest.mark.asyncio
async def test_replay_is_ts_ordered(halabot_engine):
    log = PgEventLog(halabot_engine)
    await log.append(new_event(_at(30), EventType.OBSERVATION_BAR, source="a", asset="NVDA"))
    await log.append(new_event(_at(5), EventType.OBSERVATION_BAR, source="a", asset="NVDA"))
    out = await _collect(log)
    assert [e.ts.minute for e in out] == [5, 30]


@pytest.mark.asyncio
async def test_replay_filters(halabot_engine):
    log = PgEventLog(halabot_engine)
    await log.append(new_event(_at(0), EventType.OBSERVATION_NEWS, source="a", asset="NVDA"))
    await log.append(new_event(_at(1), EventType.OBSERVATION_BAR, source="a", asset="NVDA"))
    await log.append(new_event(_at(2), EventType.OBSERVATION_BAR, source="a", asset="MSFT"))
    by_type = await _collect(log, types={EventType.OBSERVATION_NEWS})
    assert [e.type for e in by_type] == [EventType.OBSERVATION_NEWS]
    by_asset = await _collect(log, asset="MSFT")
    assert [e.asset for e in by_asset] == ["MSFT"]
    since = await _collect(log, since=T0 + timedelta(minutes=1))
    assert {e.ts.minute for e in since} == {1, 2}


@pytest.mark.asyncio
async def test_causation_chain_persists(halabot_engine):
    log = PgEventLog(halabot_engine)
    root = new_event(_at(0), EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA")
    child = new_event(_at(1), EventType.BELIEF_UPDATED, source="belief", asset="NVDA",
                      causation=root)
    await log.append(root)
    await log.append(child)
    out = await _collect(log, asset="NVDA")
    child_row = next(e for e in out if e.type == EventType.BELIEF_UPDATED)
    assert child_row.causation_id == root.id
    assert child_row.correlation_id == root.correlation_id


@pytest.mark.asyncio
async def test_bus_durably_persists_then_replays(halabot_engine):
    """End-to-end: publish through the bus → row in Postgres → replayable."""
    bus = InProcessEventBus(PgEventLog(halabot_engine))
    seen: list[Event] = []
    bus.subscribe({EventType.OBSERVATION_BAR}, lambda e: _record(seen, e))
    e = new_event(_at(0), EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    await bus.publish(e)
    assert len(seen) == 1                       # dispatched
    replayed = [x async for x in bus.replay(types={EventType.OBSERVATION_BAR})]
    assert [x.id for x in replayed] == [e.id]   # durably persisted


async def _record(sink: list, e: Event) -> None:
    sink.append(e)
