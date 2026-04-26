"""RecentNewsFeed + prompt-formatter tests."""

from __future__ import annotations

import time

from halal_trader.sentiment.events import NewsEvent
from halal_trader.sentiment.feed import RecentNewsFeed, format_news_for_prompt


def _ev(title="x", sentiment="neutral", pairs=None, importance="normal", source="cp"):
    return NewsEvent(
        title=title,
        source=source,
        url=f"http://example.com/{title}",
        published_at="2026-04-26T12:00:00",
        sentiment=sentiment,
        affected_pairs=pairs or [],
        importance=importance,
    )


def test_feed_caps_capacity():
    feed = RecentNewsFeed(capacity=3, max_age_seconds=3600)
    for i in range(10):
        feed.push(_ev(title=f"e{i}"))
    snap = feed.snapshot()
    assert len(snap) == 3
    assert snap[-1].title == "e9"


def test_feed_stale_entries_pruned_on_read(monkeypatch):
    feed = RecentNewsFeed(capacity=10, max_age_seconds=60)

    t0 = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t0[0])
    feed.push(_ev(title="old"))
    t0[0] = 1200.0  # 200s later — beyond 60s window
    feed.push(_ev(title="fresh"))

    snap = feed.snapshot()
    assert [e.title for e in snap] == ["fresh"]


def test_feed_clear():
    feed = RecentNewsFeed()
    feed.push(_ev(title="x"))
    feed.clear()
    assert feed.snapshot() == []


def test_format_empty_returns_empty_string():
    assert format_news_for_prompt([]) == ""


def test_format_basic_bullets_with_glyphs():
    events = [
        _ev(title="ETF approved", sentiment="positive", source="Bloomberg"),
        _ev(title="Exchange hacked", sentiment="negative", source="Reuters"),
    ]
    text = format_news_for_prompt(events)
    assert "▲" in text  # positive glyph
    assert "▼" in text  # negative glyph
    assert "ETF approved" in text
    assert "Bloomberg" in text


def test_format_emits_importance_and_pairs():
    ev = _ev(
        title="SEC enforcement action",
        sentiment="negative",
        importance="breaking",
        pairs=["BTCUSDT", "ETHUSDT"],
    )
    text = format_news_for_prompt([ev])
    assert "BREAKING" in text
    assert "BTCUSDT" in text
    assert "ETHUSDT" in text


def test_format_filter_by_pair():
    events = [
        _ev(title="BTC news", pairs=["BTCUSDT"]),
        _ev(title="DOGE meme rally", pairs=["DOGEUSDT"]),
    ]
    text = format_news_for_prompt(events, pair_filter=["BTCUSDT"])
    assert "BTC news" in text
    assert "DOGE" not in text


def test_format_filter_keeps_unscoped_events():
    """Events without affected_pairs are general-market — keep them."""
    events = [
        _ev(title="Macro: Fed cuts rates", pairs=[]),
        _ev(title="DOGE rally", pairs=["DOGEUSDT"]),
    ]
    text = format_news_for_prompt(events, pair_filter=["BTCUSDT"])
    assert "Macro" in text
    assert "DOGE" not in text


def test_format_respects_limit():
    events = [_ev(title=f"e{i}") for i in range(20)]
    text = format_news_for_prompt(events, limit=3)
    # Only the most recent 3 appear.
    assert text.count("\n") == 2  # 3 lines = 2 newlines
    assert "e19" in text
    assert "e0" not in text


def test_pair_filter_with_no_matches_returns_empty():
    events = [_ev(title="DOGE", pairs=["DOGEUSDT"])]
    assert format_news_for_prompt(events, pair_filter=["BTCUSDT"]) == ""
