"""Tests for the Yahoo Finance equities news collector + cycle stage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle_pipeline import CycleState
from halal_trader.core.cycle_stages import FetchStockNewsStage
from halal_trader.sentiment.events import NewsEvent
from halal_trader.sentiment.stocks_news import (
    StockNewsCollector,
    _epoch_to_iso,
    _parse_news_payload,
)

# ── Payload parsing ────────────────────────────────────────────


def test_parse_news_payload_maps_yahoo_shape_to_news_event():
    """Yahoo's ``news[]`` items become :class:`NewsEvent` with the
    expected field mapping."""
    payload = {
        "news": [
            {
                "title": "AAPL beats Q4 expectations",
                "publisher": "Reuters",
                "link": "https://news.example/aapl-q4",
                "providerPublishTime": 1715000000,
            }
        ]
    }
    events = _parse_news_payload("AAPL", payload)
    assert len(events) == 1
    ev = events[0]
    assert ev.title == "AAPL beats Q4 expectations"
    assert ev.source == "Reuters"
    assert ev.url == "https://news.example/aapl-q4"
    assert ev.affected_pairs == ["AAPL"]
    # "beats" is in the positive lexicon — the per-headline classifier
    # picks it up and overrides the legacy hardcoded "neutral".
    assert ev.sentiment == "positive"
    assert ev.importance == "normal"
    # ISO 8601 UTC.
    assert ev.published_at.startswith("2024-05-06T")
    assert ev.published_at.endswith("+00:00")


def test_parse_news_payload_drops_items_missing_required_fields():
    """Sponsored entries / malformed items without title or link are
    silently dropped — the cycle must not fail because of a bad
    response."""
    payload = {
        "news": [
            {"publisher": "Sponsor", "link": "https://ad.example/x"},  # no title
            {"title": "AAPL up", "publisher": "Reuters"},  # no link
            {
                "title": "Valid",
                "link": "https://news.example/v",
                "publisher": "AP",
                "providerPublishTime": 1715000000,
            },
        ]
    }
    events = _parse_news_payload("AAPL", payload)
    assert [e.title for e in events] == ["Valid"]


def test_parse_news_payload_handles_missing_publisher_with_fallback():
    """A news item without ``publisher`` falls back to a generic
    ``Yahoo Finance`` label instead of crashing the parse."""
    payload = {
        "news": [
            {
                "title": "TSLA recall",
                "link": "https://news.example/tsla",
                "providerPublishTime": 1715000000,
            }
        ]
    }
    events = _parse_news_payload("TSLA", payload)
    assert events[0].source == "Yahoo Finance"


def test_parse_news_payload_handles_empty_response():
    assert _parse_news_payload("AAPL", {}) == []
    assert _parse_news_payload("AAPL", {"news": []}) == []


def test_parse_news_payload_handles_missing_timestamp():
    """``providerPublishTime`` can be missing — leave ``published_at``
    empty so downstream lexical sort de-prioritises it."""
    payload = {
        "news": [
            {
                "title": "MSFT investor day",
                "publisher": "Reuters",
                "link": "https://news.example/msft",
            }
        ]
    }
    events = _parse_news_payload("MSFT", payload)
    assert events[0].published_at == ""


def test_epoch_to_iso_is_utc():
    assert _epoch_to_iso(1715000000).endswith("+00:00")


# ── Collector cache + fetch_for_symbols ──────────────────────────


@pytest.mark.asyncio
async def test_fetch_for_symbols_sorts_newest_first(monkeypatch):
    """Aggregated results across symbols sort newest first so the
    LLM prompt's ``limit=6`` keeps the most relevant headlines."""
    collector = StockNewsCollector()

    async def fake_fetch(self, symbol):  # noqa: ARG001
        return {
            "AAPL": [
                NewsEvent(
                    title="old AAPL",
                    source="A",
                    url="u",
                    published_at="2024-01-01T00:00:00+00:00",
                    sentiment="neutral",
                    affected_pairs=["AAPL"],
                )
            ],
            "MSFT": [
                NewsEvent(
                    title="new MSFT",
                    source="B",
                    url="u",
                    published_at="2024-06-01T00:00:00+00:00",
                    sentiment="neutral",
                    affected_pairs=["MSFT"],
                )
            ],
        }.get(symbol, [])

    monkeypatch.setattr(StockNewsCollector, "_fetch_one", fake_fetch)
    events = await collector.fetch_for_symbols(["AAPL", "MSFT"])
    assert [e.title for e in events] == ["new MSFT", "old AAPL"]


@pytest.mark.asyncio
async def test_fetch_for_symbols_swallows_per_symbol_failure(monkeypatch):
    """A failing fetch for one symbol must not abort the whole pass —
    other symbols' results still flow through."""
    collector = StockNewsCollector()

    async def fake_fetch(self, symbol):  # noqa: ARG001
        if symbol == "AAPL":
            raise RuntimeError("Yahoo 503")
        return [
            NewsEvent(
                title=f"{symbol} news",
                source="A",
                url="u",
                published_at="2024-06-01T00:00:00+00:00",
                sentiment="neutral",
                affected_pairs=[symbol],
            )
        ]

    monkeypatch.setattr(StockNewsCollector, "_fetch_one", fake_fetch)
    events = await collector.fetch_for_symbols(["AAPL", "MSFT"])
    assert [e.title for e in events] == ["MSFT news"]  # AAPL silently dropped


# ── FetchStockNewsStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_stock_news_stage_no_collector_is_no_op():
    """``news_collector=None`` is the dev / no-network path — the stage
    must yield an empty news_text without raising."""
    stage = FetchStockNewsStage(news_collector=None)
    state = CycleState(halal_pairs=["AAPL", "MSFT"])
    out = await stage.run(state)
    assert out.news_text == ""


@pytest.mark.asyncio
async def test_fetch_stock_news_stage_empty_universe_is_no_op():
    """Empty ``halal_pairs`` — skip the fetch (no point querying for
    no symbols)."""
    collector = MagicMock()
    collector.fetch_for_symbols = AsyncMock()
    stage = FetchStockNewsStage(news_collector=collector)
    state = CycleState(halal_pairs=[])
    out = await stage.run(state)
    assert out.news_text == ""
    collector.fetch_for_symbols.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_stock_news_stage_renders_events():
    """Happy path: collector returns events, the stage runs them
    through :func:`format_news_for_prompt`."""
    collector = MagicMock()
    collector.fetch_for_symbols = AsyncMock(
        return_value=[
            NewsEvent(
                title="AAPL Q4 beat",
                source="Reuters",
                url="u",
                published_at="2024-06-01T00:00:00+00:00",
                sentiment="positive",
                affected_pairs=["AAPL"],
                importance="hot",
            )
        ]
    )
    stage = FetchStockNewsStage(news_collector=collector)
    state = CycleState(halal_pairs=["AAPL"])
    out = await stage.run(state)
    assert "AAPL Q4 beat" in out.news_text
    assert "Reuters" in out.news_text


@pytest.mark.asyncio
async def test_fetch_stock_news_stage_swallows_collector_failure():
    """Network blip in the collector must not abort the cycle — the
    stage logs and emits an empty block."""
    collector = MagicMock()
    collector.fetch_for_symbols = AsyncMock(side_effect=RuntimeError("Yahoo 502"))
    stage = FetchStockNewsStage(news_collector=collector)
    state = CycleState(halal_pairs=["AAPL"])
    out = await stage.run(state)
    assert out.news_text == ""
