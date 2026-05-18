"""Tests for the financial-headline lexicon classifier."""

from __future__ import annotations

import pytest

from halal_trader.sentiment.headline_polarity import (
    NEGATIVE_TOKENS,
    POSITIVE_TOKENS,
    classify_headline,
    score_headline,
)

# ── score_headline (raw bucket sums) ─────────────────────────


def test_score_empty_string_returns_zero():
    assert score_headline("") == (0.0, 0.0)


def test_score_whitespace_only_returns_zero():
    assert score_headline("   \n\t") == (0.0, 0.0)


def test_score_no_lexicon_hits_returns_zero():
    """A plain neutral headline (no lexicon tokens) scores 0, 0."""
    assert score_headline("Company files quarterly report") == (0.0, 0.0)


def test_score_case_insensitive():
    """``BEATS`` and ``Beats`` and ``beats`` all score identically —
    the lexicon match is case-insensitive."""
    a = score_headline("AAPL BEATS expectations")
    b = score_headline("AAPL Beats expectations")
    c = score_headline("AAPL beats expectations")
    assert a == b == c
    assert a[0] > 0  # positive score
    assert a[1] == 0


def test_score_counts_every_occurrence():
    """Repeating a token adds its weight each time — a headline that
    says "beats" twice scores 2× the per-token weight."""
    one = score_headline("beats")
    two = score_headline("beats and beats again")
    assert two[0] == pytest.approx(2 * one[0])


def test_score_separate_buckets_for_positive_and_negative():
    """Mixed headline: positive and negative weights accumulate
    independently so the classifier can see the magnitude on each
    side, not just the net."""
    pos, neg = score_headline("Earnings beat but stock plunges on guidance miss")
    assert pos > 0
    assert neg > 0


# ── classify_headline (3-way label) ──────────────────────────


def test_classify_empty_is_neutral():
    assert classify_headline("") == "neutral"


def test_classify_no_lexicon_hits_is_neutral():
    assert classify_headline("Board meeting scheduled for next month") == "neutral"


def test_classify_strong_positive_headline():
    """Multiple strong positive tokens — clearly bullish."""
    assert classify_headline("AAPL beats earnings, raises guidance, upgraded") == "positive"


def test_classify_strong_negative_headline():
    """Multiple strong negative tokens — clearly bearish."""
    assert (
        classify_headline("TSLA plunges as Q4 misses, guidance slashed, fraud probe opens")
        == "negative"
    )


def test_classify_single_strong_token_flips_polarity():
    """One ``beats`` (weight 1.5) clears the +0.6 threshold — no
    second confirmer required for strong-weight tokens."""
    assert classify_headline("AAPL beats") == "positive"


def test_classify_single_strong_negative_flips_polarity():
    """One ``plunges`` (weight 1.5) flips negative without a second
    confirmer."""
    assert classify_headline("TSLA plunges") == "negative"


def test_classify_single_weak_token_stays_neutral():
    """A weak token (e.g. ``"rises"`` at +0.5) is BELOW the +0.6
    threshold — single weak hit doesn't flip the label. Forces the
    classifier to wait for confirming context.
    """
    assert classify_headline("Stock rises in early trading") == "neutral"


def test_classify_two_weak_positive_tokens_flip():
    """Two weak tokens sum past the threshold — two ``"rises"``
    accumulates to 1.0, which clears 0.6."""
    assert classify_headline("Stock rises, gains broaden across sector") == "positive"


def test_classify_balanced_positive_and_negative_stays_neutral():
    """``"beats"`` (+1.5) + ``"misses"`` (+1.5 to negative bucket) = net 0.
    A genuinely mixed headline stays neutral rather than picking a
    coin-flip side."""
    assert classify_headline("AAPL beats Q4 but misses on guidance") == "neutral"


def test_classify_analyst_rating_buy_is_positive():
    """``"buy"`` rating (+0.6) just clears the threshold — analyst
    upgrades to a buy rating should read as positive."""
    assert classify_headline("Analyst raises target with Buy rating") == "positive"


def test_classify_analyst_rating_sell_is_negative():
    """Symmetric: ``"sell"`` rating (+0.6 to negative bucket) flips
    negative."""
    assert classify_headline("Firm downgrades stock to Sell on weakness") == "negative"


# ── Lexicon hygiene ──────────────────────────────────────────


def test_no_overlap_between_positive_and_negative_lexicons():
    """A token in both lexicons would double-count and produce
    nonsense scores. Pin so an additive edit doesn't accidentally
    add ``"strong"`` to both buckets."""
    overlap = set(POSITIVE_TOKENS) & set(NEGATIVE_TOKENS)
    assert overlap == set()


def test_all_positive_weights_are_positive_and_negative_weights_positive():
    """Convention: weights are stored as positive floats in both
    lexicons; the scorer adds them to the respective bucket. A
    negative weight in either lexicon would be a bug (the buckets
    are already polarity-separated)."""
    for token, weight in POSITIVE_TOKENS.items():
        assert weight > 0, f"POSITIVE_TOKENS[{token!r}] should be > 0, got {weight}"
    for token, weight in NEGATIVE_TOKENS.items():
        assert weight > 0, f"NEGATIVE_TOKENS[{token!r}] should be > 0, got {weight}"


# ── Robustness ───────────────────────────────────────────────


def test_classify_handles_punctuation_and_extra_whitespace():
    """Yahoo headlines often have stray punctuation. The tokeniser
    splits on word boundaries so ``"beats,"`` matches ``"beats"``."""
    assert classify_headline("AAPL beats, beats, beats") == "positive"


def test_classify_does_not_match_substring_inside_unrelated_word():
    """``"beats"`` should match the verb but NOT match inside e.g.
    ``"beatscape"`` — the regex requires whole-word boundaries.
    Pin so a future regex tweak doesn't introduce false positives."""
    # "heartbeat" contains "beat" — must NOT match "beat" (whole-word)
    # because the regex captures contiguous letter runs, not arbitrary
    # substrings.
    out = classify_headline("Company gauges heartbeat of consumer demand")
    assert out == "neutral"
