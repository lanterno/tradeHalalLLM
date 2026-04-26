"""Tests for the new cycle helpers wiring microstructure + news into the prompt."""

from __future__ import annotations

from unittest.mock import MagicMock

from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.sentiment.events import NewsEvent
from halal_trader.sentiment.feed import RecentNewsFeed


def _bare_cycle(*, news_feed=None) -> CryptoCycleService:
    """Build a cycle with only the helpers' dependencies — no live broker."""
    # All deps are MagicMock — the helpers under test don't touch them.
    return CryptoCycleService(
        broker=MagicMock(),
        screener=MagicMock(),
        strategy=MagicMock(),
        executor=MagicMock(),
        portfolio=MagicMock(),
        news_feed=news_feed,
    )


def test_microstructure_text_renders_per_pair_summary():
    cycle = _bare_cycle()
    orderbooks = {
        "BTCUSDT": {"bids": [[100.0, 10.0]], "asks": [[101.0, 1.0]]},
        "ETHUSDT": {"bids": [[200.0, 1.0]], "asks": [[201.0, 1.0]]},
    }
    text = cycle._build_microstructure_text(orderbooks)
    # Each pair gets one line; BTC has lopsided book → bid-heavy label.
    assert "BTCUSDT" in text
    assert "ETHUSDT" in text
    assert "bid-heavy" in text


def test_microstructure_text_skips_unusable_books():
    cycle = _bare_cycle()
    orderbooks = {
        "X": {"bids": [], "asks": []},  # empty → orderbook_features returns None
    }
    assert cycle._build_microstructure_text(orderbooks) == ""


def test_news_text_empty_when_no_feed_wired():
    cycle = _bare_cycle(news_feed=None)
    assert cycle._build_news_text(["BTCUSDT"]) == ""


def test_news_text_returns_formatted_snapshot_from_feed():
    feed = RecentNewsFeed(capacity=5, max_age_seconds=3600)
    feed.push(
        NewsEvent(
            title="ETF approved",
            source="Bloomberg",
            url="https://x",
            published_at="2026-04-26T12:00:00",
            sentiment="positive",
            affected_pairs=["BTCUSDT"],
            importance="hot",
        )
    )
    cycle = _bare_cycle(news_feed=feed)
    text = cycle._build_news_text(["BTCUSDT"])
    assert "ETF approved" in text
    assert "Bloomberg" in text


def test_news_text_pair_filter_keeps_unscoped_market_news():
    feed = RecentNewsFeed()
    feed.push(
        NewsEvent(
            title="Macro: Fed pause",
            source="Reuters",
            url="https://x",
            published_at="2026-04-26T13:00:00",
            sentiment="positive",
            affected_pairs=[],  # general market
        )
    )
    cycle = _bare_cycle(news_feed=feed)
    text = cycle._build_news_text(["BTCUSDT"])
    assert "Macro" in text


def test_news_text_handles_feed_exception_gracefully():
    """A blowing-up feed must not crash the cycle — return empty string."""
    feed = MagicMock()
    feed.snapshot.side_effect = RuntimeError("boom")
    cycle = _bare_cycle(news_feed=feed)
    assert cycle._build_news_text(["BTCUSDT"]) == ""


def test_last_indicators_cache_initially_none():
    cycle = _bare_cycle()
    assert cycle.last_indicators_cache is None
