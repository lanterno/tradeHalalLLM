"""FinnhubNewsSource — maps company-news to observation.news, deduped."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from halabot.perception.sources.finnhub_news import FinnhubNewsSource, _lexicon_polarity
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, by_symbol):
        self._by = by_symbol
        self.fail: set[str] = set()

    async def get(self, url, params):
        sym = params["symbol"]
        if sym in self.fail:
            raise httpx.ConnectError("down")
        return _Resp(self._by.get(sym, []))

    async def aclose(self):
        pass


async def _universe(symbols):
    async def u() -> list[str]:
        return symbols
    return u


async def _emit_to(sink: list[Event]):
    async def emit(e: Event) -> None:
        sink.append(e)
    return emit


def _item(headline, url, **kw):
    return {"headline": headline, "url": url, "summary": kw.get("summary", ""),
            "datetime": kw.get("datetime", 1), "source": kw.get("source", "Reuters")}


async def _src(by_symbol, symbols):
    client = _FakeClient(by_symbol)
    src = FinnhubNewsSource(
        "key", await _universe(symbols), CLOCK, per_symbol_spacing_s=0, client=client
    )
    return src, client


@pytest.mark.asyncio
async def test_disabled_without_key_emits_nothing():
    src = FinnhubNewsSource("", await _universe(["NVDA"]), CLOCK, client=_FakeClient({}))
    assert not src.enabled
    assert await src.poll_once(await _emit_to([])) == 0


@pytest.mark.asyncio
async def test_emits_news_observations_with_polarity():
    src, _ = await _src({"NVDA": [_item("NVDA beats earnings", "u1")]}, ["NVDA"])
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 1
    e = sink[0]
    assert e.type == EventType.OBSERVATION_NEWS
    assert e.asset == "NVDA"
    assert e.payload["url"] == "u1"
    # polarity is the deterministic lexicon mapping of the headline
    assert e.payload["lexicon_polarity"] == _lexicon_polarity("NVDA beats earnings")


@pytest.mark.asyncio
async def test_dedups_by_url_across_polls():
    src, client = await _src({"NVDA": [_item("h1", "u1")]}, ["NVDA"])
    sink: list[Event] = []
    emit = await _emit_to(sink)
    await src.poll_once(emit)
    client._by["NVDA"] = [_item("h1", "u1"), _item("h2", "u2")]  # u1 repeats
    await src.poll_once(emit)
    assert [e.payload["url"] for e in sink] == ["u1", "u2"]


@pytest.mark.asyncio
async def test_drops_items_without_headline_or_url():
    src, _ = await _src({"NVDA": [_item("", "u1"), _item("h2", ""), _item("h3", "u3")]}, ["NVDA"])
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 1 and sink[0].payload["url"] == "u3"


@pytest.mark.asyncio
async def test_one_symbol_failure_isolated():
    src, client = await _src(
        {"NVDA": [_item("h", "u1")], "MSFT": [_item("m", "u2")]}, ["NVDA", "MSFT"]
    )
    client.fail = {"NVDA"}
    sink: list[Event] = []
    n = await src.poll_once(await _emit_to(sink))
    assert n == 1 and sink[0].asset == "MSFT"


def test_polarity_mapping_values():
    # neutral abstains (None); directional tags map to ±0.5
    from halabot.perception.sources.finnhub_news import _POLARITY

    assert _POLARITY["neutral"] is None
    assert _POLARITY["positive"] == 0.5
    assert _POLARITY["negative"] == -0.5
