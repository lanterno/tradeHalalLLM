"""Stock catalyst feed tests — multi-source aggregation + prompt formatting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from halal_trader.trading.catalysts import (
    Catalyst,
    StockCatalystFeed,
    format_catalysts_for_prompt,
)


def _cat(symbol="AAPL", kind="news", **kw):
    base = dict(
        symbol=symbol,
        kind=kind,
        title="x",
        timestamp=datetime.now(timezone.utc),
        sentiment="neutral",
        source="src",
    )
    base.update(kw)
    return Catalyst(**base)


# ── format_catalysts_for_prompt ────────────────────────────────


def test_format_empty_returns_empty_string():
    assert format_catalysts_for_prompt([]) == ""


def test_format_renders_glyphs_and_meta():
    cats = [
        _cat(title="ETF approved", kind="news", sentiment="positive", source="Bloomberg"),
        _cat(title="Q1 earnings beat", kind="earnings", sentiment="positive"),
        _cat(title="CFO sells $5M", kind="insider_sell", sentiment="negative"),
    ]
    text = format_catalysts_for_prompt(cats)
    assert "📰" in text
    assert "📊" in text
    assert "▼" in text
    assert "ETF approved" in text


def test_format_drops_stale_entries():
    fresh = _cat(title="fresh news", timestamp=datetime.now(timezone.utc))
    stale = _cat(title="stale news", timestamp=datetime.now(timezone.utc) - timedelta(days=2))
    text = format_catalysts_for_prompt([fresh, stale], max_age_hours=24)
    assert "fresh news" in text
    assert "stale news" not in text


def test_format_filters_by_symbol_set():
    aapl = _cat(symbol="AAPL", title="aapl thing")
    msft = _cat(symbol="MSFT", title="msft thing")
    text = format_catalysts_for_prompt([aapl, msft], symbols=["AAPL"])
    assert "aapl thing" in text
    assert "msft thing" not in text


def test_format_respects_limit():
    cats = [_cat(title=f"item-{i}") for i in range(20)]
    text = format_catalysts_for_prompt(cats, limit=3)
    assert text.count("\n") == 2  # 3 lines


def test_naive_timestamps_assumed_utc():
    """A tz-naive datetime mustn't be silently dropped from the freshness window."""
    ts = datetime.now()  # tz-naive
    cat = _cat(title="naive ts", timestamp=ts)
    text = format_catalysts_for_prompt([cat])
    assert "naive ts" in text


# ── StockCatalystFeed aggregation ─────────────────────────────


async def test_feed_no_sources_returns_empty():
    feed = StockCatalystFeed()
    assert await feed.fetch_all(["AAPL"]) == []


async def test_feed_no_symbols_returns_empty():
    src = MagicMock()
    src.fetch = AsyncMock(return_value=[_cat()])
    feed = StockCatalystFeed([src])
    assert await feed.fetch_all([]) == []
    src.fetch.assert_not_called()


async def test_feed_combines_multiple_sources_sorted_newest_first():
    older = _cat(title="older", timestamp=datetime.now(timezone.utc) - timedelta(hours=2))
    newer = _cat(title="newer", timestamp=datetime.now(timezone.utc))

    s1 = MagicMock()
    s1.fetch = AsyncMock(return_value=[older])
    s2 = MagicMock()
    s2.fetch = AsyncMock(return_value=[newer])

    feed = StockCatalystFeed([s1, s2])
    result = await feed.fetch_all(["AAPL"])
    assert [c.title for c in result] == ["newer", "older"]


async def test_feed_swallows_per_source_exceptions():
    bad = MagicMock()
    bad.fetch = AsyncMock(side_effect=RuntimeError("api down"))
    good = MagicMock()
    good.fetch = AsyncMock(return_value=[_cat(title="ok")])

    feed = StockCatalystFeed([bad, good])
    result = await feed.fetch_all(["AAPL"])
    assert [c.title for c in result] == ["ok"]
