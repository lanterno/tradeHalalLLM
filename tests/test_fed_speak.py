"""Tests for the Fed-speak sentiment scorer + RSS fetcher."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from halal_trader.trading.fed_speak import (
    FedSpeakFetcher,
    FedSpeakSignal,
    FedSpeech,
    aggregate_signal,
    fed_speak_to_catalysts,
    format_fed_speak_for_prompt,
    parse_rss,
    score_text,
)

# ── Lexicon scorer ───────────────────────────────────────────────


def test_score_text_empty() -> None:
    h, d = score_text("")
    assert h == 0.0 and d == 0.0


def test_score_text_hawkish_keywords() -> None:
    h, d = score_text("The committee will tighten policy further as inflation remains elevated")
    assert h > 0
    assert h > d


def test_score_text_dovish_keywords() -> None:
    h, d = score_text(
        "We see scope to ease policy as growth is softening and inflation is moderating"
    )
    assert d > 0
    assert d > h


def test_score_text_mixed_balances() -> None:
    text = "Inflation remains elevated but growth is also softening — patience is warranted"
    h, d = score_text(text)
    assert h > 0 and d > 0


# ── Aggregate ────────────────────────────────────────────────────


def _speech(title: str, summary: str, hours_ago: int = 1) -> FedSpeech:
    return FedSpeech(
        title=title,
        timestamp=datetime.now(UTC) - timedelta(hours=hours_ago),
        speaker="Test",
        url="https://example.com",
        summary=summary,
    )


def test_aggregate_empty() -> None:
    sig = aggregate_signal([])
    assert sig.n_speeches == 0
    assert sig.label == "no_data"
    assert sig.net_drift == 0.0


def test_aggregate_hawkish_drift() -> None:
    speeches = [
        _speech("hawk1", "We must tighten policy further given persistent inflation"),
        _speech("hawk2", "Restrictive stance is warranted to prevent overheating"),
    ]
    sig = aggregate_signal(speeches)
    assert sig.label in ("hawkish_drift", "mildly_hawkish")
    assert sig.net_drift > 0
    assert "hawk" in sig.most_hawkish_quote.lower()


def test_aggregate_dovish_drift() -> None:
    speeches = [
        _speech("dove1", "Growth is softening and we see scope to ease policy"),
        _speech("dove2", "Inflation is moderating; cuts may be appropriate"),
    ]
    sig = aggregate_signal(speeches)
    assert sig.label in ("dovish_drift", "mildly_dovish")
    assert sig.net_drift < 0
    assert "dove" in sig.most_dovish_quote.lower()


def test_aggregate_balanced() -> None:
    speeches = [
        _speech("mixed", "Patience is warranted; we will be data dependent"),
    ]
    sig = aggregate_signal(speeches)
    assert sig.label == "balanced"


# ── RSS parser ───────────────────────────────────────────────────


def _build_rss(items: list[dict]) -> str:
    parts = ['<?xml version="1.0"?><rss><channel>']
    for it in items:
        parts.append(
            f"<item>"
            f"<title>{it['title']}</title>"
            f"<link>{it.get('link', 'https://x.com')}</link>"
            f"<pubDate>{it['pub']}</pubDate>"
            f"<description>{it.get('desc', '')}</description>"
            f"</item>"
        )
    parts.append("</channel></rss>")
    return "".join(parts)


def _rfc822(when: datetime) -> str:
    return when.strftime("%a, %d %b %Y %H:%M:%S +0000")


def test_parse_rss_extracts_items_inside_window() -> None:
    now = datetime.now(UTC)
    items = [
        {
            "title": "Powell — outlook",
            "pub": _rfc822(now - timedelta(hours=2)),
            "desc": "Inflation is elevated",
        },
        {"title": "Old speech", "pub": _rfc822(now - timedelta(hours=200)), "desc": "stale"},
    ]
    speeches = parse_rss(_build_rss(items), window_hours=24)
    assert len(speeches) == 1
    assert "Powell" in speeches[0].title


def test_parse_rss_handles_unparseable_date() -> None:
    items = [
        {"title": "x", "pub": "not-a-date", "desc": "y"},
    ]
    speeches = parse_rss(_build_rss(items))
    assert speeches == []


def test_parse_rss_strips_html() -> None:
    items = [
        {
            "title": "x",
            "pub": _rfc822(datetime.now(UTC) - timedelta(hours=1)),
            "desc": "<p>Inflation is elevated</p>",
        }
    ]
    speeches = parse_rss(_build_rss(items))
    assert "<p>" not in speeches[0].summary


def test_parse_rss_empty_input() -> None:
    assert parse_rss("") == []


# ── Fetcher ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetcher_returns_signal_from_mock_rss() -> None:
    items = [
        {
            "title": "Powell - outlook",
            "pub": _rfc822(datetime.now(UTC) - timedelta(hours=1)),
            "desc": "We must tighten policy further as inflation is elevated",
        }
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_build_rss(items))

    f = FedSpeakFetcher()
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sig = await f.fetch()
    assert sig.n_speeches == 1
    assert sig.label in ("hawkish_drift", "mildly_hawkish", "balanced")
    await f.aclose()


@pytest.mark.asyncio
async def test_fetcher_caches_within_ttl() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=_build_rss([]))

    f = FedSpeakFetcher()
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await f.fetch()
    await f.fetch()
    await f.fetch()
    assert calls == 1
    await f.aclose()


@pytest.mark.asyncio
async def test_fetcher_http_error_returns_no_data_signal() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    f = FedSpeakFetcher()
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sig = await f.fetch()
    assert sig.label == "no_data"
    await f.aclose()


# ── Prompt formatting ────────────────────────────────────────────


def test_format_empty_returns_empty() -> None:
    sig = FedSpeakSignal(
        n_speeches=0, hawkish_score=0, dovish_score=0, net_drift=0, label="no_data"
    )
    assert format_fed_speak_for_prompt(sig) == ""


def test_format_renders_quote_lines() -> None:
    sig = FedSpeakSignal(
        n_speeches=2,
        hawkish_score=4.0,
        dovish_score=1.0,
        net_drift=3.0,
        label="hawkish_drift",
        most_hawkish_quote="Powell — restrictive stance",
        most_dovish_quote="Williams — gradual approach",
    )
    text = format_fed_speak_for_prompt(sig)
    assert "hawkish_drift" in text
    assert "Powell" in text
    assert "Williams" in text


# ── Catalyst integration ─────────────────────────────────────────


def test_fed_speak_to_catalysts_emits_per_symbol() -> None:
    sig = FedSpeakSignal(
        n_speeches=3,
        hawkish_score=5.0,
        dovish_score=1.0,
        net_drift=4.0,
        label="hawkish_drift",
    )
    cats = fed_speak_to_catalysts(sig, ["AAPL", "MSFT"])
    assert len(cats) == 2
    assert {c.symbol for c in cats} == {"AAPL", "MSFT"}
    assert all(c.kind == "fed_speak" for c in cats)


def test_fed_speak_to_catalysts_no_data_returns_empty() -> None:
    sig = FedSpeakSignal(
        n_speeches=0, hawkish_score=0, dovish_score=0, net_drift=0, label="no_data"
    )
    assert fed_speak_to_catalysts(sig, ["AAPL"]) == []
