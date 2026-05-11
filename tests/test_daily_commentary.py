"""Tests for community/daily_commentary.py — Round-5 Wave 17.D."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from halal_trader.community.daily_commentary import (
    CommentaryFeed,
    CommentaryPost,
    ModerationOutcome,
    ModerationResult,
    Severity,
    Source,
    Topic,
    append_post,
    filter_feed,
    moderate_commentary,
    render_feed,
    render_post,
    supersede,
)


def _post(
    post_id: str = "P1",
    feed_date: date = date(2026, 5, 10),
    topic: Topic = Topic.MOVERS,
    source: Source = Source.LLM,
    author: str = "alice",
    headline: str = "AAPL up 3% on earnings beat",
    body: str = "AAPL reported strong fundamentals and beat estimates.",
    severity: Severity = Severity.INFO,
    tickers: tuple[str, ...] = ("AAPL",),
    posted_at: datetime | None = None,
    supersedes_id: str = "",
) -> CommentaryPost:
    return CommentaryPost(
        post_id=post_id,
        feed_date=feed_date,
        topic=topic,
        source=source,
        author_id=author,
        headline=headline,
        body=body,
        severity=severity,
        tickers=tickers,
        posted_at=posted_at,
        supersedes_id=supersedes_id,
    )


# --- CommentaryPost validation -----------------------------------------


def test_post_valid():
    p = _post()
    assert p.topic is Topic.MOVERS


def test_post_default_posted_at_is_feed_date_midnight():
    p = _post()
    assert p.posted_at == datetime(2026, 5, 10, 0, 0)


def test_post_explicit_posted_at_preserved():
    p = _post(posted_at=datetime(2026, 5, 10, 14, 30))
    assert p.posted_at == datetime(2026, 5, 10, 14, 30)


def test_post_empty_id_rejected():
    with pytest.raises(ValueError):
        _post(post_id="")


def test_post_empty_author_rejected():
    with pytest.raises(ValueError):
        _post(author="")


def test_post_empty_headline_rejected():
    with pytest.raises(ValueError):
        _post(headline=" ")


def test_post_long_headline_rejected():
    with pytest.raises(ValueError):
        _post(headline="x" * 250)


def test_post_empty_body_rejected():
    with pytest.raises(ValueError):
        _post(body=" ")


def test_post_long_body_rejected():
    with pytest.raises(ValueError):
        _post(body="x" * 6000)


def test_post_empty_ticker_rejected():
    with pytest.raises(ValueError):
        _post(tickers=("AAPL", " "))


def test_post_long_ticker_rejected():
    with pytest.raises(ValueError):
        _post(tickers=("X" * 20,))


def test_post_immutable():
    p = _post()
    with pytest.raises(AttributeError):
        p.headline = "x"  # type: ignore[misc]


# --- moderate_commentary -----------------------------------------------


def test_moderate_clean_passes():
    p = _post()
    res = moderate_commentary(p)
    assert res.outcome is ModerationOutcome.PASS
    assert not res.reasons


def test_moderate_blocks_gambling_lexicon():
    p = _post(body="Buy AAPL, this is a casino trade")
    res = moderate_commentary(p)
    assert res.outcome is ModerationOutcome.BLOCK
    assert any("gambling-style" in r for r in res.reasons)


def test_moderate_warns_leveraged_margin():
    p = _post(body="Up 10% via leveraged margin position")
    res = moderate_commentary(p)
    assert res.outcome is ModerationOutcome.WARN


def test_moderate_block_dominates_warn():
    """Pin: BLOCK + WARN matches → BLOCK wins."""
    p = _post(body="Casino trade with leveraged margin")
    res = moderate_commentary(p)
    assert res.outcome is ModerationOutcome.BLOCK


def test_moderate_case_insensitive():
    p = _post(body="LOTTERY pick of the day")
    res = moderate_commentary(p)
    assert res.outcome is ModerationOutcome.BLOCK


def test_moderate_lexicon_override():
    custom: dict[str, tuple[ModerationOutcome, str]] = {
        "yikes": (ModerationOutcome.WARN, "test phrase")
    }
    p = _post(body="yikes that move was sharp")
    res = moderate_commentary(p, lexicon=custom)
    assert res.outcome is ModerationOutcome.WARN


def test_moderate_block_in_headline():
    p = _post(headline="Casino vibes today", body="benign")
    res = moderate_commentary(p)
    assert res.outcome is ModerationOutcome.BLOCK


# --- CommentaryFeed validation -----------------------------------------


def test_feed_empty_valid():
    f = CommentaryFeed()
    assert f.posts == ()


def test_feed_duplicate_id_rejected():
    p1 = _post(post_id="P1")
    p2 = _post(post_id="P1", headline="Other")
    with pytest.raises(ValueError):
        CommentaryFeed(posts=(p1, p2))


def test_feed_out_of_order_rejected():
    p1 = _post(post_id="P1", posted_at=datetime(2026, 5, 10, 9, 0))
    p2 = _post(post_id="P2", posted_at=datetime(2026, 5, 10, 8, 0))
    with pytest.raises(ValueError):
        CommentaryFeed(posts=(p1, p2))


# --- append_post -------------------------------------------------------


def test_append_basic():
    feed = CommentaryFeed()
    feed2, result = append_post(feed, _post())
    assert len(feed2.posts) == 1
    assert result.outcome is ModerationOutcome.PASS


def test_append_blocked_post_raises():
    feed = CommentaryFeed()
    p = _post(body="Casino-style yolo trade")
    with pytest.raises(ValueError):
        append_post(feed, p)


def test_append_warning_post_still_added():
    feed = CommentaryFeed()
    p = _post(body="A leveraged margin position is risky")
    feed2, result = append_post(feed, p)
    assert len(feed2.posts) == 1
    assert result.outcome is ModerationOutcome.WARN


def test_append_duplicate_id_rejected():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1"))
    with pytest.raises(ValueError):
        append_post(feed, _post(post_id="P1", headline="Other"))


def test_append_supersedes_unknown_rejected():
    feed = CommentaryFeed()
    bad = _post(post_id="P2", supersedes_id="P-missing")
    with pytest.raises(ValueError):
        append_post(feed, bad)


def test_append_double_supersede_rejected():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1"))
    feed, _ = append_post(
        feed,
        _post(
            post_id="P2",
            supersedes_id="P1",
            posted_at=datetime(2026, 5, 10, 12, 0),
        ),
    )
    with pytest.raises(ValueError):
        append_post(
            feed,
            _post(
                post_id="P3",
                supersedes_id="P1",
                posted_at=datetime(2026, 5, 10, 13, 0),
            ),
        )


def test_append_custom_moderator():
    feed = CommentaryFeed()
    p = _post()

    def always_block(_: CommentaryPost) -> ModerationResult:
        return ModerationResult(outcome=ModerationOutcome.BLOCK, reasons=("plug-in",))

    with pytest.raises(ValueError):
        append_post(feed, p, moderator=always_block)


# --- supersede helper --------------------------------------------------


def test_supersede_requires_supersedes_id():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1"))
    with pytest.raises(ValueError):
        supersede(feed, new_post=_post(post_id="P2"))


def test_supersede_appends_with_link():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1"))
    feed, _ = supersede(
        feed,
        new_post=_post(
            post_id="P2",
            supersedes_id="P1",
            posted_at=datetime(2026, 5, 10, 12, 0),
        ),
    )
    assert len(feed.posts) == 2


# --- filter_feed -------------------------------------------------------


def test_filter_by_topic():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1", topic=Topic.MOVERS))
    feed, _ = append_post(
        feed,
        _post(
            post_id="P2",
            topic=Topic.SUKUK,
            posted_at=datetime(2026, 5, 10, 12, 0),
        ),
    )
    out = filter_feed(feed, topic=Topic.SUKUK)
    assert len(out) == 1
    assert out[0].post_id == "P2"


def test_filter_by_source():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1", source=Source.OPERATOR))
    feed, _ = append_post(
        feed,
        _post(
            post_id="P2",
            source=Source.COMMUNITY,
            posted_at=datetime(2026, 5, 10, 12, 0),
        ),
    )
    out = filter_feed(feed, source=Source.OPERATOR)
    assert len(out) == 1


def test_filter_by_severity():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1", severity=Severity.URGENT))
    feed, _ = append_post(
        feed,
        _post(
            post_id="P2",
            severity=Severity.INFO,
            posted_at=datetime(2026, 5, 10, 12, 0),
        ),
    )
    out = filter_feed(feed, severity=Severity.URGENT)
    assert len(out) == 1
    assert out[0].post_id == "P1"


def test_filter_by_ticker():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1", tickers=("AAPL", "MSFT")))
    feed, _ = append_post(
        feed,
        _post(
            post_id="P2",
            tickers=("GOOG",),
            posted_at=datetime(2026, 5, 10, 12, 0),
        ),
    )
    out = filter_feed(feed, ticker="AAPL")
    assert len(out) == 1


def test_filter_by_date_range():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1", feed_date=date(2026, 5, 10)))
    feed, _ = append_post(
        feed,
        _post(
            post_id="P2",
            feed_date=date(2026, 6, 10),
            posted_at=datetime(2026, 6, 10, 12, 0),
        ),
    )
    out = filter_feed(feed, date_from=date(2026, 6, 1))
    assert len(out) == 1


# --- Render --------------------------------------------------------------


def test_render_post_no_secret_leak():
    p = _post(author="alice@example.com")
    out = render_post(p)
    assert "alice@example.com" not in out


def test_render_post_severity_emoji():
    assert "📝" in render_post(_post(severity=Severity.INFO))
    assert "📌" in render_post(_post(severity=Severity.NOTABLE))
    assert "🚨" in render_post(_post(severity=Severity.URGENT))


def test_render_post_truncates_body():
    p = _post(body="x" * 500)
    out = render_post(p, body_chars=50)
    assert "…" in out


def test_render_post_includes_tickers():
    p = _post(tickers=("AAPL", "MSFT"))
    out = render_post(p)
    assert "AAPL" in out
    assert "MSFT" in out


def test_render_post_supersede_marker():
    feed = CommentaryFeed()
    feed, _ = append_post(feed, _post(post_id="P1"))
    feed, _ = supersede(
        feed,
        new_post=_post(
            post_id="P2",
            supersedes_id="P1",
            posted_at=datetime(2026, 5, 10, 12, 0),
        ),
    )
    out = render_post(feed.posts[1])
    assert "supersedes=P1" in out


def test_render_feed_empty():
    assert "empty" in render_feed(CommentaryFeed())


def test_render_feed_top_n_caps():
    feed = CommentaryFeed()
    for i in range(5):
        feed, _ = append_post(
            feed,
            _post(
                post_id=f"P{i}",
                posted_at=datetime(2026, 5, 10, 9 + i, 0),
            ),
        )
    out = render_feed(feed, top_n=2)
    assert "P3" in out
    assert "P4" in out
    assert "P0" not in out
