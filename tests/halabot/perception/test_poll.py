"""PollingSource — fetch → map → emit, dedup, error tolerance."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from halabot.perception.dedup import InMemoryDedupStore
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
            CLOCK,
            EventType.OBSERVATION_NEWS,
            source="fake-news",
            asset=raw["asset"],
            payload={"url": raw["url"]},
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
    src = _FakeNews(
        [
            [{"url": "a", "asset": "NVDA"}],
            [{"url": "a", "asset": "NVDA"}, {"url": "b", "asset": "NVDA"}],  # 'a' repeats
        ]
    )
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
async def test_a_large_feed_window_is_not_re_emitted():
    # The seen set used to drop an arbitrary half of itself at 2,000 keys while
    # the feed still returned those items, re-publishing ~1,000 every poll.
    window = [{"url": f"u{i}", "asset": "NVDA"} for i in range(3000)]
    src = _FakeNews([window, window, window])
    sink: list[Event] = []
    emit = await _emit_to(sink)
    assert await src.poll_once(emit) == 3000
    assert await src.poll_once(emit) == 0
    assert await src.poll_once(emit) == 0


@pytest.mark.asyncio
async def test_a_key_is_forgotten_only_after_the_feed_drops_it_for_the_ttl():
    now = [0.0]
    src = _FakeNews(
        [[{"url": "a", "asset": "NVDA"}]] * 3 + [[], [{"url": "a", "asset": "NVDA"}]],
        seen_ttl_s=100.0,
        monotonic=lambda: now[0],
    )
    sink: list[Event] = []
    emit = await _emit_to(sink)
    for _ in range(3):  # still in the feed well past one TTL: remembered throughout
        await src.poll_once(emit)
        now[0] += 80.0
    assert len(sink) == 1
    now[0] += 200.0
    await src.poll_once(emit)  # gone from the feed for longer than the TTL
    assert "a" not in src._seen
    await src.poll_once(emit)  # (with no persisted store, it would be new again)
    assert len(sink) == 2


@pytest.mark.asyncio
async def test_a_memory_miss_asks_the_persisted_store_first():
    store = InMemoryDedupStore()
    src = _FakeNews([[{"url": "b", "asset": "NVDA"}], [{"url": "a", "asset": "NVDA"}]])
    src._dedup = store
    sink: list[Event] = []
    emit = await _emit_to(sink)
    await src.poll_once(emit)  # primes from the (empty) store
    await store.add("fake-news", ["a"])  # seen since, e.g. before memory was pruned
    await src.poll_once(emit)
    assert [e.payload["url"] for e in sink] == ["b"]


@pytest.mark.asyncio
async def test_poll_drops_items_mapped_to_none():
    src = _FakeNews([[{"url": "a", "asset": "NVDA", "skip": True}, {"url": "b", "asset": "NVDA"}]])
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 1
    assert sink[0].payload["url"] == "b"
