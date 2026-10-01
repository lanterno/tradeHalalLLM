"""News prompt-formatter tests."""

from __future__ import annotations

from halal_trader.sentiment.events import NewsEvent
from halal_trader.sentiment.feed import format_news_for_prompt


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
