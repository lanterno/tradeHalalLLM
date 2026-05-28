"""InMemoryEventLog — append + ordered, filtered replay (INV-5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import EventType, new_event


def _at(minute: int):
    return FakeClock(datetime(2026, 5, 28, 12, minute, tzinfo=UTC))


async def _collect(log, **kw):
    return [e async for e in log.replay(**kw)]


@pytest.mark.asyncio
async def test_append_and_replay_roundtrip():
    log = InMemoryEventLog()
    e = new_event(_at(0), EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    await log.append(e)
    assert len(log) == 1
    out = await _collect(log)
    assert [x.id for x in out] == [e.id]


@pytest.mark.asyncio
async def test_replay_is_event_time_ordered_regardless_of_append_order():
    log = InMemoryEventLog()
    late = new_event(_at(30), EventType.OBSERVATION_BAR, source="a", asset="NVDA")
    early = new_event(_at(5), EventType.OBSERVATION_BAR, source="a", asset="NVDA")
    await log.append(late)   # appended out of ts order
    await log.append(early)
    out = await _collect(log)
    assert [x.ts.minute for x in out] == [5, 30]  # replays in ts order


@pytest.mark.asyncio
async def test_replay_filters_since_and_until():
    log = InMemoryEventLog()
    for m in (0, 10, 20, 30):
        await log.append(new_event(_at(m), EventType.OBSERVATION_PRICE, source="a", asset="NVDA"))
    base = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
    out = await _collect(
        log, since=base + timedelta(minutes=10), until=base + timedelta(minutes=20)
    )
    assert [x.ts.minute for x in out] == [10, 20]


@pytest.mark.asyncio
async def test_replay_filters_by_type():
    log = InMemoryEventLog()
    await log.append(new_event(_at(0), EventType.OBSERVATION_NEWS, source="a", asset="NVDA"))
    await log.append(new_event(_at(1), EventType.OBSERVATION_BAR, source="a", asset="NVDA"))
    out = await _collect(log, types={EventType.OBSERVATION_NEWS})
    assert [x.type for x in out] == [EventType.OBSERVATION_NEWS]


@pytest.mark.asyncio
async def test_replay_filters_by_asset():
    log = InMemoryEventLog()
    await log.append(new_event(_at(0), EventType.OBSERVATION_BAR, source="a", asset="NVDA"))
    await log.append(new_event(_at(1), EventType.OBSERVATION_BAR, source="a", asset="MSFT"))
    out = await _collect(log, asset="MSFT")
    assert [x.asset for x in out] == ["MSFT"]
