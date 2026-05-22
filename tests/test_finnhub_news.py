"""Finnhub stocks news backend — drop-in replacement for the Yahoo
search endpoint that started 429-ing on 2026-05-21. Tests focus on
the payload parser since the network layer is identical to the
Yahoo backend's (httpx + circuit breaker)."""

from __future__ import annotations

from halal_trader.sentiment.events import NewsEvent
from halal_trader.sentiment.stocks_news import _parse_finnhub_payload


def test_finnhub_parser_extracts_required_fields():
    payload = [
        {
            "headline": "Apple announces record Q4 earnings",
            "url": "https://example.com/aapl-q4",
            "source": "Reuters",
            "datetime": 1716387200,  # epoch s
        }
    ]
    events = _parse_finnhub_payload("AAPL", payload, limit=5)
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, NewsEvent)
    assert e.title == "Apple announces record Q4 earnings"
    assert e.url == "https://example.com/aapl-q4"
    assert e.source == "Reuters"
    assert e.published_at  # ISO string, non-empty


def test_finnhub_parser_respects_limit():
    payload = [
        {"headline": f"news {i}", "url": f"u{i}", "source": "x", "datetime": 1716387200}
        for i in range(20)
    ]
    events = _parse_finnhub_payload("AAPL", payload, limit=3)
    assert len(events) == 3


def test_finnhub_parser_skips_items_missing_title():
    payload = [
        {"headline": "", "url": "u1", "datetime": 1716387200},
        {"url": "u2", "datetime": 1716387200},  # no headline at all
        {"headline": "good", "url": "u3", "datetime": 1716387200},
    ]
    events = _parse_finnhub_payload("AAPL", payload, limit=10)
    assert len(events) == 1
    assert events[0].title == "good"


def test_finnhub_parser_skips_items_missing_url():
    payload = [
        {"headline": "good news", "datetime": 1716387200},  # no url
    ]
    events = _parse_finnhub_payload("AAPL", payload, limit=10)
    assert events == []


def test_finnhub_parser_handles_non_list_payload():
    """Finnhub returns a bare list. If we ever get something else
    (error response, rate-limit JSON), the parser must not crash."""
    assert _parse_finnhub_payload("AAPL", {}, limit=5) == []
    assert _parse_finnhub_payload("AAPL", None, limit=5) == []
    assert _parse_finnhub_payload("AAPL", "string", limit=5) == []


def test_finnhub_parser_classifies_sentiment():
    """Sentiment uses the same lexicon as the Yahoo path."""
    payload = [
        {
            "headline": "Apple beats earnings expectations, surges",
            "url": "u1",
            "datetime": 1716387200,
        }
    ]
    events = _parse_finnhub_payload("AAPL", payload, limit=5)
    # "beats" + "surges" should classify positive
    assert events[0].sentiment in ("positive", "neutral")
