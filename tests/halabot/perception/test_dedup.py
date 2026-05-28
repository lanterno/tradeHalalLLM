"""Persisted perception dedup — store roundtrip + restart survival (INV-2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.perception.dedup import InMemoryDedupStore, PgDedupStore
from halabot.perception.poll import PollingSource
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType, new_event

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


class _ListSource(PollingSource):
    """Emits one observation.news per item; deduped by the item's key."""

    def __init__(self, items, *, dedup_store=None):
        super().__init__("test-source", interval_s=1.0, dedup_store=dedup_store)
        self._items = items

    async def fetch(self):
        return list(self._items)

    def to_event(self, raw) -> Event:
        return new_event(
            CLOCK, EventType.OBSERVATION_NEWS, source="test", asset=raw["asset"],
            payload={"headline": raw["key"], "url": raw["key"]},
        )

    def dedup_key(self, raw) -> str:
        return raw["key"]


async def _collect(source) -> list[Event]:
    out: list[Event] = []

    async def emit(e: Event) -> None:
        out.append(e)

    await source.poll_once(emit)
    return out


# ── InMemoryDedupStore ──
@pytest.mark.asyncio
async def test_inmemory_store_roundtrip():
    store = InMemoryDedupStore()
    assert await store.load("ns") == set()
    await store.add("ns", ["a", "b"])
    assert await store.load("ns") == {"a", "b"}
    await store.add("other", ["c"])
    assert await store.load("ns") == {"a", "b"}  # namespaced


# ── PollingSource dedup behavior (in-memory) ──
@pytest.mark.asyncio
async def test_source_dedups_within_session():
    store = InMemoryDedupStore()
    src = _ListSource([{"asset": "NVDA", "key": "u1"}, {"asset": "NVDA", "key": "u2"}],
                      dedup_store=store)
    first = await _collect(src)
    second = await _collect(src)  # same items again
    assert len(first) == 2
    assert second == []  # all already seen


@pytest.mark.asyncio
async def test_dedup_survives_restart_via_store():
    """A NEW source instance sharing the store does not re-emit seen items."""
    store = InMemoryDedupStore()
    items = [{"asset": "NVDA", "key": "u1"}, {"asset": "AAPL", "key": "u2"}]
    first = await _collect(_ListSource(items, dedup_store=store))
    assert len(first) == 2
    # "restart": fresh source, same persisted store → nothing re-emitted.
    restarted = await _collect(_ListSource(items, dedup_store=store))
    assert restarted == []


@pytest.mark.asyncio
async def test_no_store_means_in_memory_only():
    src = _ListSource([{"asset": "NVDA", "key": "u1"}])
    assert len(await _collect(src)) == 1
    assert await _collect(src) == []  # still dedups in-process
    # a fresh instance (no shared store) WOULD re-emit — that's the gap the store closes
    assert len(await _collect(_ListSource([{"asset": "NVDA", "key": "u1"}]))) == 1


# ── PgDedupStore (real Postgres) ──
@pytest.mark.asyncio
async def test_pg_store_roundtrip_and_restart(halabot_engine):
    store = PgDedupStore(halabot_engine)
    # The _ListSource's namespace is its name, "test-source".
    ns = "test-source"
    await store.add(ns, ["NVDA:http://x", "AAPL:http://y"])
    loaded = await store.load(ns)
    assert {"NVDA:http://x", "AAPL:http://y"} <= loaded

    # A source primed from the store skips those keys (restart survival).
    src = _ListSource(
        [{"asset": "NVDA", "key": "NVDA:http://x"}, {"asset": "T", "key": "T:http://z"}],
        dedup_store=store,
    )
    emitted = await _collect(src)
    keys = {e.payload["url"] for e in emitted}
    assert keys == {"T:http://z"}  # the persisted NVDA key was skipped


@pytest.mark.asyncio
async def test_pg_store_upsert_is_idempotent(halabot_engine):
    store = PgDedupStore(halabot_engine)
    ns = "dup-test"
    await store.add(ns, ["k1"])
    await store.add(ns, ["k1"])  # ON CONFLICT → no error, refreshes seen_at
    assert "k1" in await store.load(ns)
