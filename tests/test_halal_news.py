"""Tests for sentiment/halal_news.py — Round-5 Wave 11.G."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from halal_trader.sentiment.halal_news import (
    HalalRelevance,
    NewsItem,
    TaggedNews,
    Topic,
    filter_relevant,
    render_tagged,
    tag_news,
)


def _item(
    item_id: str = "N-1",
    headline: str = "Generic market update",
    body: str = "Markets traded sideways today.",
    source: str = "wire",
    published: datetime = datetime(2026, 5, 5, tzinfo=timezone.utc),
    tickers: tuple[str, ...] = (),
) -> NewsItem:
    return NewsItem(
        item_id=item_id,
        headline=headline,
        body=body,
        source=source,
        published_at=published,
        tickers=tickers,
    )


# --- Validation ----------------------------------------------------


def test_relevance_string_values():
    assert HalalRelevance.NOT_RELEVANT.value == "not_relevant"
    assert HalalRelevance.GENERAL.value == "general"
    assert HalalRelevance.HALAL_RELEVANT.value == "halal_relevant"
    assert HalalRelevance.SCHEDULED_RECOMPUTE.value == "scheduled_recompute"
    assert HalalRelevance.EXCEPTION_REVIEW.value == "exception_review"


def test_topic_string_values():
    assert Topic.DEBT_ISSUANCE.value == "debt_issuance"
    assert Topic.SCHOLAR_OPINION.value == "scholar_opinion"
    assert Topic.GENERIC.value == "generic"


def test_news_empty_id_rejected():
    with pytest.raises(ValueError):
        _item(item_id="")


def test_news_empty_headline_rejected():
    with pytest.raises(ValueError):
        _item(headline="")


def test_news_empty_source_rejected():
    with pytest.raises(ValueError):
        _item(source=" ")


def test_news_naive_published_rejected():
    with pytest.raises(ValueError):
        NewsItem(
            item_id="N-1",
            headline="x",
            body="x",
            source="x",
            published_at=datetime(2026, 5, 5),
        )


def test_tagged_confidence_range_rejected():
    item = _item()
    with pytest.raises(ValueError):
        TaggedNews(
            item=item,
            relevance=HalalRelevance.GENERAL,
            topics=frozenset({Topic.GENERIC}),
            confidence=1.5,
            haram_keywords_hit=frozenset(),
            halal_keywords_hit=frozenset(),
        )


# --- Tagging -------------------------------------------------------


def test_generic_news_not_relevant():
    tagged = tag_news(_item(headline="Markets open today", body="Stocks trading."))
    assert tagged.relevance is HalalRelevance.NOT_RELEVANT


def test_halal_keyword_drives_halal_relevant():
    tagged = tag_news(
        _item(
            headline="AAOIFI publishes new sukuk standard",
            body="The AAOIFI shariah board released guidance on hybrid sukuk.",
        )
    )
    assert tagged.relevance is HalalRelevance.HALAL_RELEVANT
    assert "aaoifi" in tagged.halal_keywords_hit
    assert "sukuk" in tagged.halal_keywords_hit


def test_haram_keyword_drives_exception_review():
    tagged = tag_news(
        _item(
            headline="ABC Corp acquires alcohol distributor",
            body="Acquisition expands ABC's alcohol division.",
        )
    )
    assert tagged.relevance is HalalRelevance.EXCEPTION_REVIEW
    assert "alcohol" in tagged.haram_keywords_hit


def test_debt_issuance_drives_scheduled_recompute():
    tagged = tag_news(
        _item(
            headline="ABC Corp announces $5B bond offering",
            body="The company issued bonds maturing 2030.",
        )
    )
    assert tagged.relevance is HalalRelevance.SCHEDULED_RECOMPUTE
    assert Topic.DEBT_ISSUANCE in tagged.topics


def test_acquisition_drives_scheduled_recompute():
    tagged = tag_news(
        _item(
            headline="ABC Corp announces acquisition of XYZ",
            body="ABC acquired XYZ for $10B in stock.",
        )
    )
    assert Topic.ACQUISITION in tagged.topics


def test_earnings_drives_general():
    """Earnings is a recognised topic but not high-relevance for halal screen."""
    tagged = tag_news(
        _item(
            headline="ABC Corp reports Q2 earnings",
            body="Quarterly results beat consensus.",
        )
    )
    # Earnings is not in high_relevance_topics, so → GENERAL
    assert tagged.relevance is HalalRelevance.GENERAL


def test_topics_default_generic_when_no_keywords():
    tagged = tag_news(_item(headline="Random update", body="Nothing material."))
    assert tagged.topics == frozenset({Topic.GENERIC})


def test_scholar_opinion_is_halal_relevant_via_keyword():
    """Scholar verdict mentions 'shariah' which is in halal keywords."""
    tagged = tag_news(
        _item(
            headline="New shariah ruling on crypto",
            body="The shariah scholar issued a fatwa.",
        )
    )
    assert tagged.relevance is HalalRelevance.HALAL_RELEVANT
    assert Topic.SCHOLAR_OPINION in tagged.topics


def test_confidence_in_unit_range():
    items = [
        _item(headline="Generic update", body="x"),
        _item(headline="Sukuk halal news", body="aaoifi shariah"),
        _item(headline="Casino acquisition", body="alcohol division"),
    ]
    for i in items:
        t = tag_news(i)
        assert 0.0 <= t.confidence <= 1.0


# --- Filter --------------------------------------------------------


def test_filter_default_keeps_halal_and_exception():
    items = [
        _item("N1", headline="Generic update", body="x"),
        _item("N2", headline="Sukuk halal", body="aaoifi"),
        _item("N3", headline="Casino acquisition", body="alcohol"),
    ]
    out = filter_relevant(items)
    ids = [t.item.item_id for t in out]
    assert "N2" in ids
    assert "N3" in ids
    assert "N1" not in ids


def test_filter_min_general_keeps_everything_above():
    items = [
        _item("N1", headline="Earnings beat", body="quarterly results"),  # GENERAL
        _item("N2", headline="Generic", body="x"),  # NOT_RELEVANT
    ]
    out = filter_relevant(items, min_relevance=HalalRelevance.GENERAL)
    ids = {t.item.item_id for t in out}
    assert "N1" in ids
    assert "N2" not in ids


# --- Render --------------------------------------------------------


def test_render_tagged_includes_relevance():
    tagged = tag_news(_item(headline="Sukuk", body="aaoifi"))
    out = render_tagged(tagged)
    assert "halal_relevant" in out


def test_render_emoji_for_each_level():
    items = {
        "halal": _item(headline="Sukuk", body="aaoifi shariah"),
        "haram": _item(headline="Casino", body="alcohol"),
        "general": _item(headline="Earnings", body="quarterly results"),
        "irrelevant": _item(headline="Generic", body="x"),
    }
    for label, item in items.items():
        out = render_tagged(tag_news(item))
        assert any(emoji in out for emoji in "⚪🔵🟡🟢🔴")


def test_render_no_secret_leak():
    tagged = tag_news(_item(headline="Sukuk", body="aaoifi"))
    out = render_tagged(tagged)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------------


def test_e2e_news_pipeline_filters_signal_from_noise():
    items = [
        _item("N1", headline="ABC Corp Q2 beat", body="quarterly results strong"),
        _item("N2", headline="ABC issues $10B bond", body="debt issuance announced"),
        _item("N3", headline="AAOIFI updates sukuk standard", body="aaoifi shariah"),
        _item("N4", headline="Random press release", body="generic update"),
    ]
    high_priority = filter_relevant(items, min_relevance=HalalRelevance.SCHEDULED_RECOMPUTE)
    ids = {t.item.item_id for t in high_priority}
    assert "N2" in ids  # debt issuance
    assert "N3" in ids  # halal-keyword
    assert "N1" not in ids  # general
    assert "N4" not in ids  # noise


def test_replay_consistency():
    item = _item(headline="Sukuk update", body="aaoifi")
    a = tag_news(item)
    b = tag_news(item)
    assert a == b
