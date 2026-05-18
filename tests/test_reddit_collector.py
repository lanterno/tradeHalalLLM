"""Tests for the cache + match-post logic in :class:`RedditCollector`.

The actual PRAW fetch is integration; this file covers the bits that
don't need network: cache-TTL hit, the lazy-praw-import disable, the
keyword-match → mention extraction, and aggregate score math.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from halal_trader.sentiment.reddit import (
    RedditCollector,
    RedditMention,
    RedditSentimentData,
)


def _make_post(
    *,
    title: str = "post",
    selftext: str = "",
    score: int = 10,
    subreddit_name: str = "CryptoCurrency",
    created_utc: float = 0.0,
    permalink: str = "/r/x/comments/1/abc/",
) -> SimpleNamespace:
    """Build a PRAW-shape stub the matcher can walk."""
    return SimpleNamespace(
        title=title,
        selftext=selftext,
        score=score,
        subreddit=SimpleNamespace(display_name=subreddit_name),
        created_utc=created_utc,
        permalink=permalink,
        removed_by_category=None,
    )


def _collector() -> RedditCollector:
    return RedditCollector(
        client_id="x",
        client_secret="y",
        trading_pairs=["BTCUSDT", "ETHUSDT"],
        cache_ttl_seconds=300,
    )


# ── cache + disable paths ─────────────────────────────────────


@pytest.mark.asyncio
async def test_collect_returns_empty_when_praw_unavailable():
    """If PRAW can't be imported, collect() returns empty without
    crashing — sentiment is best-effort."""
    c = _collector()
    # `_ensure_reddit` will see _reddit is None and try import. If praw
    # isn't installed in the test env it'll log a warning and stay None.
    # Force the disabled state regardless of import success:
    c._reddit = None

    def _no_init(self):  # noqa: ANN001
        return  # leave _reddit as None

    c._ensure_reddit = _no_init.__get__(c)  # type: ignore[method-assign]
    out = await c.collect()
    assert out == {}


@pytest.mark.asyncio
async def test_collect_returns_cached_within_ttl():
    c = _collector()
    cached = {"BTCUSDT": RedditSentimentData(pair="BTCUSDT")}
    c._cache = cached
    c._cache_time = time.monotonic()  # fresh
    out = await c.collect()
    assert out is cached  # exact same object


# ── _match_post keyword detection ─────────────────────────────


def test_match_post_attaches_btc_post_to_btcusdt_pair():
    c = _collector()
    pair_data = {p: RedditSentimentData(pair=p) for p in c._trading_pairs}
    post = _make_post(title="Bitcoin breaks $50K", score=100)
    c._match_post(post, pair_data)
    assert len(pair_data["BTCUSDT"].mentions) == 1
    assert len(pair_data["ETHUSDT"].mentions) == 0
    m = pair_data["BTCUSDT"].mentions[0]
    assert m.score == 100
    assert m.url.endswith("/r/x/comments/1/abc/")


def test_match_post_searches_selftext_too():
    """Body text counts — not just the title."""
    c = _collector()
    pair_data = {p: RedditSentimentData(pair=p) for p in c._trading_pairs}
    post = _make_post(title="Question", selftext="Anyone bullish on ETH?")
    c._match_post(post, pair_data)
    assert len(pair_data["ETHUSDT"].mentions) == 1
    assert len(pair_data["BTCUSDT"].mentions) == 0


def test_match_post_attaches_to_multiple_pairs_when_both_mentioned():
    c = _collector()
    pair_data = {p: RedditSentimentData(pair=p) for p in c._trading_pairs}
    post = _make_post(title="BTC vs ETH narrative")
    c._match_post(post, pair_data)
    assert len(pair_data["BTCUSDT"].mentions) == 1
    assert len(pair_data["ETHUSDT"].mentions) == 1


def test_match_post_skips_pair_when_keyword_absent():
    c = _collector()
    pair_data = {p: RedditSentimentData(pair=p) for p in c._trading_pairs}
    post = _make_post(title="Random news about gold")
    c._match_post(post, pair_data)
    assert pair_data["BTCUSDT"].mentions == []
    assert pair_data["ETHUSDT"].mentions == []


def test_match_post_truncates_body_at_500_chars():
    c = _collector()
    pair_data = {p: RedditSentimentData(pair=p) for p in c._trading_pairs}
    post = _make_post(title="Bitcoin", selftext="x" * 1_000)
    c._match_post(post, pair_data)
    assert len(pair_data["BTCUSDT"].mentions[0].body) == 500


# ── RedditSentimentData aggregate fields ──────────────────────


def test_aggregate_fields_default_empty():
    """The aggregate fields default to a zero-state — cycle code can
    safely read them even before any mention is recorded."""
    d = RedditSentimentData(pair="BTCUSDT")
    assert d.mention_count == 0
    assert d.avg_score == 0.0
    assert d.top_posts == []


def test_reddit_mention_dataclass_round_trip():
    m = RedditMention(
        title="t",
        body="b",
        score=42,
        subreddit="CryptoCurrency",
        created_utc=1_700_000_000.0,
        url="https://reddit.com/r/x",
    )
    assert m.score == 42
    assert m.url.startswith("https://")
