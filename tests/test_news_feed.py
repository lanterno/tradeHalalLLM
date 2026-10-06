"""News prompt-formatter tests."""

from __future__ import annotations

from halal_trader.sentiment.events import NewsEvent
from halal_trader.sentiment.feed import format_news_for_prompt


def _ev(title="x", sentiment="neutral", symbols=None, importance="normal", source="cp"):
    return NewsEvent(
        title=title,
        source=source,
        url=f"http://example.com/{title}",
        published_at="2026-04-26T12:00:00",
        sentiment=sentiment,
        symbols=symbols or [],
        importance=importance,
    )


def test_format_empty_returns_empty_string():
    assert format_news_for_prompt([]) == ""


def test_format_basic_bullets_with_glyphs():
    events = [
        _ev(title="Earnings beat", sentiment="positive", source="Bloomberg"),
        _ev(title="Guidance cut", sentiment="negative", source="Reuters"),
    ]
    text = format_news_for_prompt(events)
    assert "▲" in text  # positive glyph
    assert "▼" in text  # negative glyph
    assert "Earnings beat" in text
    assert "Bloomberg" in text


def test_format_emits_importance_and_symbols():
    ev = _ev(
        title="SEC enforcement action",
        sentiment="negative",
        importance="breaking",
        symbols=["AAPL", "MSFT"],
    )
    text = format_news_for_prompt([ev])
    assert "BREAKING" in text
    assert "AAPL" in text
    assert "MSFT" in text


def test_format_filter_by_symbol():
    events = [
        _ev(title="AAPL news", symbols=["AAPL"]),
        _ev(title="TSLA rally", symbols=["TSLA"]),
    ]
    text = format_news_for_prompt(events, symbol_filter=["AAPL"])
    assert "AAPL news" in text
    assert "TSLA" not in text


def test_format_filter_keeps_unscoped_events():
    """Events without symbols are general-market — keep them."""
    events = [
        _ev(title="Macro: Fed cuts rates", symbols=[]),
        _ev(title="TSLA rally", symbols=["TSLA"]),
    ]
    text = format_news_for_prompt(events, symbol_filter=["AAPL"])
    assert "Macro" in text
    assert "TSLA" not in text


def test_format_respects_limit():
    events = [_ev(title=f"e{i}") for i in range(20)]
    text = format_news_for_prompt(events, limit=3)
    # Only the most recent 3 appear.
    assert text.count("\n") == 2  # 3 lines = 2 newlines
    assert "e19" in text
    assert "e0" not in text


def test_symbol_filter_with_no_matches_returns_empty():
    events = [_ev(title="TSLA", symbols=["TSLA"])]
    assert format_news_for_prompt(events, symbol_filter=["AAPL"]) == ""
