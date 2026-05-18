"""Tests for sentiment/social_nlp.py — Round-5 Wave 11.A."""

from __future__ import annotations

import pytest

from halal_trader.sentiment.social_nlp import (
    Sentiment,
    SentimentPolicy,
    SentimentScore,
    aggregate_scores,
    extract_tickers,
    render_score,
    score_message,
)

# --- Validation -------------------------------------------------


def test_sentiment_string_values():
    assert Sentiment.BULLISH.value == "bullish"
    assert Sentiment.NEUTRAL.value == "neutral"
    assert Sentiment.BEARISH.value == "bearish"


def test_default_policy():
    p = SentimentPolicy()
    assert p.bullish_threshold == 0.20
    assert p.bearish_threshold == -0.20


def test_policy_zero_bullish_rejected():
    with pytest.raises(ValueError):
        SentimentPolicy(bullish_threshold=0.0)


def test_policy_positive_bearish_rejected():
    with pytest.raises(ValueError):
        SentimentPolicy(bearish_threshold=0.10)


def test_policy_low_caps_rejected():
    with pytest.raises(ValueError):
        SentimentPolicy(caps_amplification=0.5)


def test_policy_zero_negation_window_rejected():
    with pytest.raises(ValueError):
        SentimentPolicy(negation_window=0)


def test_score_negative_rejected():
    with pytest.raises(ValueError):
        SentimentScore(score=-1.5, sentiment=Sentiment.BEARISH, tickers=(), tokens_evaluated=1)


def test_score_negative_tokens_rejected():
    with pytest.raises(ValueError):
        SentimentScore(score=0.0, sentiment=Sentiment.NEUTRAL, tickers=(), tokens_evaluated=-1)


# --- Ticker extraction ------------------------------------------


def test_extract_tickers_basic():
    assert extract_tickers("Buying $AAPL and $MSFT") == ("AAPL", "MSFT")


def test_extract_tickers_lowercase_skipped():
    assert extract_tickers("$aapl") == ()


def test_extract_tickers_long_skipped():
    """Tickers > 6 chars not captured."""
    assert extract_tickers("$ABCDEFG") == ()


def test_extract_tickers_empty():
    assert extract_tickers("no tickers here") == ()


# --- Empty / edge cases ----------------------------------------


def test_empty_message_neutral():
    s = score_message("")
    assert s.sentiment is Sentiment.NEUTRAL
    assert s.score == 0.0


def test_whitespace_neutral():
    s = score_message("   \n  ")
    assert s.sentiment is Sentiment.NEUTRAL


def test_no_signal_words_neutral():
    s = score_message("market is open today")
    assert s.sentiment is Sentiment.NEUTRAL


# --- Bullish detection ----------------------------------------


def test_bullish_word_detected():
    s = score_message("$AAPL is bullish today")
    assert s.sentiment is Sentiment.BULLISH


def test_rocket_emoji_bullish():
    s = score_message("$BTC 🚀")
    assert s.sentiment is Sentiment.BULLISH


def test_diamond_emoji_bullish():
    s = score_message("$GME 💎")
    assert s.sentiment is Sentiment.BULLISH


def test_multiple_bull_words_amplifies():
    s = score_message("rally moon breakout")
    assert s.sentiment is Sentiment.BULLISH
    assert s.score > 0.5


# --- Bearish detection ---------------------------------------


def test_bearish_word_detected():
    s = score_message("$AAPL is bearish")
    assert s.sentiment is Sentiment.BEARISH


def test_crash_word_bearish():
    s = score_message("market is going to crash")
    assert s.sentiment is Sentiment.BEARISH


def test_red_emoji_bearish():
    s = score_message("$BTC 📉 🐻")
    assert s.sentiment is Sentiment.BEARISH


def test_rugpull_strongly_bearish():
    s = score_message("rugpull confirmed")
    assert s.sentiment is Sentiment.BEARISH
    assert s.score < -0.5


# --- Negation handling ---------------------------------------


def test_negation_flips_sentiment():
    """'not bullish' should not be bullish."""
    s = score_message("not bullish at all")
    # 'bullish' weight 1.0 → flipped to -1.0 → sentiment BEARISH
    assert s.sentiment is Sentiment.BEARISH


def test_negation_window_ends():
    """Negation only affects next 3 tokens."""
    s = score_message("not. one two three bullish")
    # 'bullish' is token 5 → outside window 1-3 → still bullish
    assert s.sentiment is Sentiment.BULLISH


# --- Intensifiers --------------------------------------------


def test_intensifier_increases_score():
    weak = score_message("buy")
    strong = score_message("very buy")
    assert abs(strong.score) >= abs(weak.score)


def test_huge_intensifier():
    s = score_message("huge rally")
    assert s.score > 0.5


# --- Caps amplification --------------------------------------


def test_caps_amplifies_score():
    """ALL CAPS bullish word scores at least as high; clipping at 1.0."""
    lower = score_message("buy")  # weight 0.6 — leaves room for caps amplification
    upper = score_message("BUY")
    assert upper.score > lower.score


# --- Tokens evaluated ----------------------------------------


def test_tokens_evaluated_counts_signal_only():
    s = score_message("$AAPL is bullish today")
    # Only 'bullish' is a signal token
    assert s.tokens_evaluated == 1


def test_tokens_evaluated_zero_for_no_signal():
    s = score_message("the market is open")
    assert s.tokens_evaluated == 0


# --- Score range -----------------------------------------------


def test_score_in_unit_interval():
    s = score_message("moon rocket diamond hands hodl")
    assert -1.0 <= s.score <= 1.0


def test_score_extreme_negative():
    s = score_message("CRASH RUGPULL BLOOD")
    assert s.score < 0
    assert -1.0 <= s.score


# --- Aggregate ------------------------------------------------


def test_aggregate_empty_neutral():
    a = aggregate_scores([])
    assert a.sentiment is Sentiment.NEUTRAL
    assert a.score == 0


def test_aggregate_averages():
    bullish = score_message("rally")
    bearish = score_message("crash")
    a = aggregate_scores([bullish, bearish])
    assert -0.3 < a.score < 0.3


def test_aggregate_unions_tickers():
    a = score_message("$AAPL bullish")
    b = score_message("$MSFT moon")
    agg = aggregate_scores([a, b])
    assert "AAPL" in agg.tickers
    assert "MSFT" in agg.tickers


def test_aggregate_dedupes_tickers():
    a = score_message("$AAPL bullish")
    b = score_message("$AAPL moon")
    agg = aggregate_scores([a, b])
    assert agg.tickers == ("AAPL",)


# --- Render --------------------------------------------------


def test_render_bullish_emoji():
    s = score_message("rally moon")
    assert "🟢" in render_score(s)


def test_render_bearish_emoji():
    s = score_message("crash dump")
    assert "🔴" in render_score(s)


def test_render_includes_tickers():
    s = score_message("$AAPL bullish")
    out = render_score(s)
    assert "$AAPL" in out


def test_render_no_secret_leak():
    s = score_message("rally")
    out = render_score(s)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------


def test_e2e_typical_reddit_thread_aggregated():
    messages = [
        "$AAPL to the moon 🚀",
        "$AAPL bullish breakout",
        "$AAPL dump incoming",
        "$AAPL rally rally rally",
    ]
    scores = [score_message(m) for m in messages]
    agg = aggregate_scores(scores)
    assert agg.sentiment is Sentiment.BULLISH
    assert "AAPL" in agg.tickers


def test_replay_consistency():
    a = score_message("$AAPL bullish")
    b = score_message("$AAPL bullish")
    assert a == b
