"""PollingSource — fetch → map → emit, dedup, error tolerance."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from halabot.perception.poll import PollingSource
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType, new_event

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


class _FakeNews(PollingSource):
    """Maps {'url','asset'} dicts to news observations, deduped by url."""

    def __init__(self, batches: list[list[dict]], **kw):
        super().__init__("fake-news", interval_s=0, **kw)
        self._batches = batches
        self._i = 0
        self.fetch_error = False

    async def fetch(self) -> list[Any]:
        if self.fetch_error:
            raise RuntimeError("feed down")
        batch = self._batches[self._i] if self._i < len(self._batches) else []
        self._i += 1
        return batch

    def to_event(self, raw: Any) -> Event | None:
        if raw.get("skip"):
            return None
        return new_event(
            CLOCK, EventType.OBSERVATION_NEWS, source="fake-news",
            asset=raw["asset"], payload={"url": raw["url"]},
        )

    def dedup_key(self, raw: Any) -> str | None:
        return raw.get("url")


async def _emit_to(sink: list[Event]):
    async def emit(e: Event) -> None:
        sink.append(e)
    return emit


@pytest.mark.asyncio
async def test_poll_once_emits_mapped_events():
    src = _FakeNews([[{"url": "a", "asset": "NVDA"}, {"url": "b", "asset": "MSFT"}]])
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 2
    assert {e.asset for e in sink} == {"NVDA", "MSFT"}


@pytest.mark.asyncio
async def test_poll_dedups_repeated_keys_across_ticks():
    src = _FakeNews([
        [{"url": "a", "asset": "NVDA"}],
        [{"url": "a", "asset": "NVDA"}, {"url": "b", "asset": "NVDA"}],  # 'a' repeats
    ])
    sink: list[Event] = []
    emit = await _emit_to(sink)
    await src.poll_once(emit)
    await src.poll_once(emit)
    assert [e.payload["url"] for e in sink] == ["a", "b"]  # 'a' not re-emitted


@pytest.mark.asyncio
async def test_poll_once_swallows_fetch_error():
    src = _FakeNews([[{"url": "a", "asset": "NVDA"}]])
    src.fetch_error = True
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 0 and sink == []  # transient feed failure → tick skipped, no crash


@pytest.mark.asyncio
async def test_poll_drops_items_mapped_to_none():
    src = _FakeNews([[{"url": "a", "asset": "NVDA", "skip": True}, {"url": "b", "asset": "NVDA"}]])
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 1
    assert sink[0].payload["url"] == "b"
