"""Tests for :mod:`sentiment.scoring` — composite signal builder + formatter.

`SentimentScorer.compute_composite` weights Reddit + CryptoPanic into
a single signal that the cycle prompt consumes; `format_sentiment_for_prompt`
renders the prompt block. Both are pure — no DB, no network.
"""

from __future__ import annotations

from halal_trader.sentiment.scoring import (
    SentimentScorer,
    SentimentSignal,
    format_sentiment_for_prompt,
)

# ── compute_composite — empty / single-source paths ──────────


def test_compute_composite_no_sources_returns_empty_signal():
    """Zero mentions and zero news → empty signal (no `data_sources`)."""
    sig = SentimentScorer().compute_composite(pair="BTCUSDT")
    assert sig.pair == "BTCUSDT"
    assert sig.data_sources == []
    assert sig.score == 0.0


def test_compute_composite_reddit_only():
    sig = SentimentScorer().compute_composite(
        pair="BTCUSDT",
        reddit_mentions=10,
        reddit_avg_score=60.0,
    )
    assert "reddit" in sig.data_sources
    assert "cryptopanic" not in sig.data_sources
    # Reddit score = (60 - 10) / 50 = 1.0 (clamped at 1.0)
    assert sig.score == 1.0


def test_compute_composite_cryptopanic_only():
    sig = SentimentScorer().compute_composite(
        pair="BTCUSDT",
        news_sentiment=0.6,
        news_count=5,
    )
    assert "cryptopanic" in sig.data_sources
    assert "reddit" not in sig.data_sources
    # Single source weighted at 1.2 → effectively 0.6.
    assert abs(sig.score - 0.6) < 1e-9


def test_compute_composite_both_sources_blended():
    """Reddit (weight 1.0 baseline) + News (weight 1.2). Reddit score
    floor at -1, news at +0.5 → blended toward news."""
    sig = SentimentScorer().compute_composite(
        pair="BTCUSDT",
        reddit_mentions=2,  # buzz < 1.5 → weight 1.0
        reddit_avg_score=10.0,  # 10 → score 0
        news_sentiment=0.5,
        news_count=10,
    )
    # (0*1.0 + 0.5*1.2) / 2.2 ≈ 0.273
    assert 0.2 < sig.score < 0.3


# ── confidence ─────────────────────────────────────────────────


def test_confidence_caps_at_1():
    """30 mentions + 5 news = 35 → confidence min(1.0, 35/20) = 1.0."""
    sig = SentimentScorer().compute_composite(
        pair="BTCUSDT",
        reddit_mentions=30,
        reddit_avg_score=10.0,
        news_sentiment=0.0,
        news_count=5,
    )
    assert sig.confidence == 1.0


def test_confidence_proportional_for_low_volume():
    """3 mentions + 1 news = 4 → confidence 4/20 = 0.2."""
    sig = SentimentScorer().compute_composite(
        pair="BTCUSDT",
        reddit_mentions=3,
        reddit_avg_score=10.0,
        news_sentiment=0.0,
        news_count=1,
    )
    assert sig.confidence == 0.2


# ── buzz heuristic ─────────────────────────────────────────────


def test_buzz_first_call_returns_one():
    """Without history, buzz defaults to 1.0 (no spike claim)."""
    s = SentimentScorer()
    sig = s.compute_composite(pair="BTCUSDT", reddit_mentions=10, reddit_avg_score=10.0)
    assert sig.buzz == 1.0


def test_buzz_spikes_when_mentions_exceed_history_avg():
    """Build history of low mentions, then a big mention spike → buzz > 1."""
    s = SentimentScorer()
    for _ in range(5):
        s.compute_composite(pair="BTCUSDT", reddit_mentions=2, reddit_avg_score=10.0)
    sig = s.compute_composite(pair="BTCUSDT", reddit_mentions=20, reddit_avg_score=10.0)
    assert sig.buzz > 5.0  # 20 / 2 = 10


def test_buzz_history_capped_at_168_entries():
    """Rolling 7-day-of-hourly window = 168 entries max."""
    s = SentimentScorer()
    for i in range(200):
        s.compute_composite(pair="BTCUSDT", reddit_mentions=i, reddit_avg_score=10.0)
    assert len(s._buzz_history["BTCUSDT"]) == 168


# ── narratives + headlines ─────────────────────────────────


def test_top_narratives_capped_at_three():
    sig = SentimentScorer().compute_composite(
        pair="BTCUSDT",
        reddit_mentions=10,
        reddit_avg_score=10.0,
        reddit_top_posts=["a", "b", "c", "d", "e"],
    )
    assert sig.top_narratives == ["a", "b", "c"]


def test_news_headlines_capped_at_three():
    sig = SentimentScorer().compute_composite(
        pair="BTCUSDT",
        news_sentiment=0.5,
        news_count=10,
        news_headlines=["h1", "h2", "h3", "h4", "h5"],
    )
    assert sig.news_headlines == ["h1", "h2", "h3"]


# ── format_sentiment_for_prompt ────────────────────────────


def test_format_empty_returns_sentinel():
    assert format_sentiment_for_prompt({}) == "No sentiment data available."


def test_format_skips_signals_with_no_data_sources():
    """An empty signal (no sources) is not rendered — the cycle would
    otherwise show noise rows for pairs nobody mentioned."""
    sig = SentimentSignal(pair="BTCUSDT")  # no data_sources
    out = format_sentiment_for_prompt({"BTCUSDT": sig})
    assert out == "No sentiment data available."


def test_format_renders_bullish_label_above_threshold():
    sig = SentimentSignal(pair="BTCUSDT", score=0.5, data_sources=["reddit"])
    out = format_sentiment_for_prompt({"BTCUSDT": sig})
    assert "BULLISH" in out


def test_format_renders_bearish_label_below_threshold():
    sig = SentimentSignal(pair="BTCUSDT", score=-0.5, data_sources=["reddit"])
    out = format_sentiment_for_prompt({"BTCUSDT": sig})
    assert "BEARISH" in out


def test_format_renders_neutral_within_threshold():
    """|score| ≤ 0.1 → NEUTRAL (avoid over-reading low-confidence noise)."""
    sig = SentimentSignal(pair="BTCUSDT", score=0.05, data_sources=["reddit"])
    out = format_sentiment_for_prompt({"BTCUSDT": sig})
    assert "NEUTRAL" in out


def test_format_appends_high_buzz_label_at_3x():
    sig = SentimentSignal(pair="BTCUSDT", score=0.0, buzz=3.5, data_sources=["reddit"])
    out = format_sentiment_for_prompt({"BTCUSDT": sig})
    assert "HIGH BUZZ" in out


def test_format_appends_elevated_buzz_label_at_2x():
    sig = SentimentSignal(pair="BTCUSDT", score=0.0, buzz=2.5, data_sources=["reddit"])
    out = format_sentiment_for_prompt({"BTCUSDT": sig})
    assert "ELEVATED BUZZ" in out
