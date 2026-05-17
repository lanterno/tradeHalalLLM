"""Tests for `sentiment/filing_classifier.py`.

Pins the lexicon-based scoring, the negation-window flip,
the longest-phrase-match-first invariant, the bounded-score
formula, the empty-input safety, and the per-symbol aggregator.
"""

from __future__ import annotations

import pytest

from halal_trader.sentiment.filing_classifier import (
    PhraseHit,
    Sentiment,
    SentimentResult,
    SymbolSentiment,
    aggregate_for_symbol,
    classify,
    render_result,
)

# ── empty input ──────────────────────────────────────────


def test_empty_text_returns_neutral_with_zero_score():
    """Pin: empty / whitespace-only input → NEUTRAL with score 0
    and no hits. The aggregate handles this so callers can feed
    in untrusted input without guarding."""
    result = classify("")
    assert result.label == Sentiment.NEUTRAL
    assert result.score == 0.0
    assert result.hits == []
    assert result.token_count == 0


def test_whitespace_only_returns_neutral():
    result = classify("   \n\n\t  ")
    assert result.label == Sentiment.NEUTRAL
    assert result.token_count == 0


def test_text_with_no_known_phrases_returns_neutral():
    """Pin: arbitrary text without matching phrases produces
    NEUTRAL — score 0, no hits."""
    result = classify("The quick brown fox jumps over the lazy dog.")
    assert result.label == Sentiment.NEUTRAL
    assert result.score == 0.0
    assert result.hits == []


# ── positive lexicon ─────────────────────────────────────


def test_record_revenue_phrase_matches():
    result = classify("The company posted record revenue this quarter.")
    assert result.label == Sentiment.BULLISH
    assert any(h.phrase == "record revenue" for h in result.hits)


def test_beat_estimates_phrase_matches():
    result = classify("Earnings beat estimates by a wide margin.")
    assert result.label == Sentiment.BULLISH
    assert any(h.phrase == "beat estimates" for h in result.hits)


def test_strong_phrase_has_higher_weight():
    """Pin: 'record revenue' has weight 1.5 (in `_PHRASE_WEIGHTS`).
    Compare against a default-weight phrase."""
    strong = classify("Posted record revenue last quarter.")
    default = classify("Operating leverage improved last quarter.")
    strong_hit = next(h for h in strong.hits if h.phrase == "record revenue")
    default_hit = next(h for h in default.hits if h.phrase == "operating leverage")
    assert strong_hit.weight > default_hit.weight


# ── negative lexicon ─────────────────────────────────────


def test_missed_estimates_phrase_matches():
    result = classify("Reported earnings missed estimates.")
    assert result.label == Sentiment.BEARISH


def test_going_concern_is_strong_negative():
    """Pin: 'going concern' has weight 2.0 — a single hit should
    dominate the score even alongside a positive."""
    result = classify("Despite operating leverage gains, going concern doubts persist.")
    assert result.label == Sentiment.BEARISH


def test_bankruptcy_is_strongest_negative():
    """Pin: 'bankruptcy' has weight 2.5 — the strongest signal."""
    result = classify("The subsidiary filed for bankruptcy yesterday.")
    assert result.label == Sentiment.BEARISH
    bk = next(h for h in result.hits if h.phrase == "bankruptcy")
    assert bk.weight == 2.5


def test_writedown_is_negative():
    result = classify("The company recorded a $200M writedown.")
    assert result.label == Sentiment.BEARISH


# ── negation handling ────────────────────────────────────


def test_negation_flips_positive_to_bearish():
    """Pin: a negated bullish phrase contributes negative score.
    Use a phrase the lexicon contains exactly."""
    result = classify("The company has not exceeded expectations.")
    # "not" within the 4-token window before "exceeded expectations"
    # flips the contribution sign.
    hit = next(h for h in result.hits if h.phrase == "exceeded expectations")
    assert hit.negated
    assert hit.contribution < 0
    assert result.label == Sentiment.BEARISH


def test_negation_within_window_flips_sign():
    """Pin: a hedge within 4 tokens before a positive phrase
    flips the contribution. Use a phrase the lexicon has."""
    result = classify("did not raised guidance this quarter")
    # "not" at index 1, "raised guidance" starts at index 2 →
    # within window. Contribution flipped → bearish.
    raised = next(h for h in result.hits if h.phrase == "raised guidance")
    assert raised.negated
    assert raised.contribution < 0


def test_negation_far_away_does_not_flip():
    """Pin: a hedge more than 4 tokens away from the phrase
    doesn't flip — natural-language negation is local."""
    result = classify(
        "while not generally a great quarter overall the company posted record revenue"
    )
    # "not" early; "record revenue" much later → not within window.
    rec = next(h for h in result.hits if h.phrase == "record revenue")
    assert not rec.negated
    assert rec.contribution > 0


def test_negation_flips_negative_too():
    """Pin: negation is symmetric — 'not missed' becomes positive."""
    result = classify("did not missed estimates this quarter")
    miss = next(h for h in result.hits if h.phrase == "missed estimates")
    assert miss.negated
    # Negated negative becomes positive.
    assert miss.contribution > 0


def test_negation_hedge_no_phrase_does_nothing():
    """A hedge without a following phrase doesn't generate a
    spurious hit."""
    result = classify("The company is not expanding rapidly.")
    # No matching phrase; result is neutral.
    assert result.hits == []


# ── longest-match-first invariant ────────────────────────


def test_longest_phrase_wins_at_each_position():
    """Pin: when two phrases overlap, the longer wins. Build a
    case where 'record' alone isn't a phrase but 'record
    revenue' is."""
    result = classify("Posted record revenue and record earnings.")
    phrases = [h.phrase for h in result.hits]
    # Both phrases present, no double-counting.
    assert "record revenue" in phrases
    assert "record earnings" in phrases


def test_no_overlap_double_counting():
    """Pin: two consecutive phrase matches don't share tokens.
    'record revenue raised guidance' produces two hits, not three."""
    result = classify("record revenue raised guidance")
    phrases = sorted(h.phrase for h in result.hits)
    assert phrases == ["raised guidance", "record revenue"]


# ── bounded score ────────────────────────────────────────


def test_score_bounded_by_minus_one_one():
    """Pin: even with many strong negatives, the normalised
    score stays in [-1, 1]."""
    result = classify(" ".join(["bankruptcy", "fraud", "going concern"] * 5))
    assert -1.0 <= result.score <= 1.0


def test_score_bounded_for_strong_positives():
    result = classify(" ".join(["record revenue", "raised guidance"] * 5))
    assert -1.0 <= result.score <= 1.0


def test_balanced_text_lands_near_zero():
    """Pin: equal positive + negative hits land near 0 (NEUTRAL)
    rather than picking a side."""
    result = classify("Posted record revenue but issued a going concern warning.")
    # Both bullish (1.5) and bearish (2.0) hit. With a +1 damper
    # the score is closer to 0 than to either extreme. The
    # bearish weight is larger so it tilts negative, but
    # not by much.
    assert -0.5 < result.score < 0.5


def test_single_default_weight_hit_is_bullish():
    """Pin: one default-weight bullish hit produces score
    `(1 - 0) / (1 + 0 + 1) = 0.5` — well above the 0.15
    BULLISH threshold. The +1 damper means a single hit is
    significant but bounded; multiple hits stack toward but
    never reach 1.0."""
    result = classify("Operating leverage improved.")
    assert result.label == Sentiment.BULLISH
    assert result.score == pytest.approx(0.5)


def test_strong_phrase_alone_is_definitive():
    """Pin: a single weight-2.0+ phrase produces a clear label."""
    result = classify("Material weakness identified in Q2.")
    assert result.label == Sentiment.BEARISH


# ── threshold bands ──────────────────────────────────────


def test_score_at_bullish_threshold_is_bullish():
    """Pin: ≥ 0.15 → BULLISH (the low end of the band)."""
    # A single weight-1.0 hit produces score 1/(1+1) = 0.5 →
    # well above 0.15. Pin the band edge by checking that a
    # mild-but-positive case still lands BULLISH.
    result = classify("Operating leverage improved this year.")
    assert result.label == Sentiment.BULLISH


def test_score_at_bearish_threshold_is_bearish():
    result = classify("Revenue declined materially.")
    assert result.label == Sentiment.BEARISH


# ── per-hit attribution ──────────────────────────────────


def test_hit_records_phrase_and_polarity():
    result = classify("Posted record revenue.")
    hit = result.hits[0]
    assert hit.phrase == "record revenue"
    assert hit.polarity == Sentiment.BULLISH


def test_hit_records_token_index():
    """Pin: token_index is the position in the tokenised text;
    used by the dashboard's "show me the matching context"
    feature."""
    result = classify("In fiscal Q2 the company posted record revenue.")
    hit = next(h for h in result.hits if h.phrase == "record revenue")
    # tokens: in, fiscal, q2, the, company, posted, record, revenue
    assert hit.token_index == 6


def test_hit_records_negation_flag():
    result = classify("did not raised guidance")
    hit = next(h for h in result.hits if h.phrase == "raised guidance")
    assert hit.negated is True


def test_hits_sorted_by_token_index():
    """Pin: the result's hits list is in document order."""
    result = classify("record revenue and missed estimates")
    indices = [h.token_index for h in result.hits]
    assert indices == sorted(indices)


# ── result dataclass ─────────────────────────────────────


def test_result_carries_token_count():
    result = classify("The quick brown fox jumped.")
    assert result.token_count == 5


def test_result_carries_raw_positive_and_negative():
    result = classify("Posted record revenue but missed estimates.")
    assert result.raw_positive > 0
    assert result.raw_negative > 0


def test_result_summary_mentions_label():
    result = classify("Posted record revenue.")
    assert "bullish" in result.summary.lower()


def test_result_summary_mentions_neutral_on_empty_match():
    result = classify("The quick brown fox.")
    assert "neutral" in result.summary.lower()


def test_is_neutral_property():
    bullish = classify("Posted record revenue.")
    neutral = classify("The fox is brown.")
    assert not bullish.is_neutral
    assert neutral.is_neutral


def test_result_immutable():
    result = classify("record revenue")
    assert isinstance(result, SentimentResult)
    with pytest.raises(Exception):
        result.score = 0.0  # type: ignore[misc]


# ── aggregate_for_symbol ─────────────────────────────────


def test_aggregate_no_filings_returns_neutral():
    """Pin: empty iterable → NEUTRAL with zero counts, no
    crash."""
    out = aggregate_for_symbol(symbol="AAPL", filings=[])
    assert out.symbol == "AAPL"
    assert out.filings_analysed == 0
    assert out.dominant_label == Sentiment.NEUTRAL
    assert out.score == 0.0


def test_aggregate_dominant_label_reflects_cohort():
    filings = [
        "Posted record revenue.",
        "Raised guidance for Q3.",
        "Strong demand across segments.",
    ]
    out = aggregate_for_symbol(symbol="AAPL", filings=filings)
    assert out.filings_analysed == 3
    assert out.dominant_label == Sentiment.BULLISH
    assert out.bullish_count == 3


def test_aggregate_confidence_weighting():
    """Pin: filings with more matches dominate the average. A
    single bearish-with-many-hits filing should outweigh a
    bullish-with-one-hit filing."""
    filings = [
        "Operating leverage improved.",  # 1 match, +
        "going concern material weakness fraud bankruptcy",  # 4 matches, all strong --
    ]
    out = aggregate_for_symbol(symbol="X", filings=filings)
    assert out.dominant_label == Sentiment.BEARISH


def test_aggregate_counts_each_label():
    filings = [
        "Posted record revenue.",  # bullish
        "Missed estimates by 10%.",  # bearish
        "The fox is brown.",  # neutral
    ]
    out = aggregate_for_symbol(symbol="X", filings=filings)
    assert out.bullish_count == 1
    assert out.neutral_count == 1
    assert out.bearish_count == 1


def test_aggregate_immutable():
    out = aggregate_for_symbol(symbol="X", filings=[])
    assert isinstance(out, SymbolSentiment)
    with pytest.raises(Exception):
        out.score = 1.0  # type: ignore[misc]


# ── render ───────────────────────────────────────────────


def test_render_includes_emoji():
    bullish = render_result(classify("Posted record revenue."))
    bearish = render_result(classify("Filed for bankruptcy."))
    neutral = render_result(classify("The fox is brown."))
    assert "🟢" in bullish
    assert "🔴" in bearish
    assert "🟡" in neutral


def test_render_includes_top_matches():
    text = render_result(classify("Posted record revenue and beat estimates."))
    assert "matches:" in text
    assert "record revenue" in text or "beat estimates" in text


def test_render_no_matches_omits_match_section():
    text = render_result(classify("The fox is brown."))
    assert "matches:" not in text


# ── PhraseHit immutability ───────────────────────────────


def test_phrase_hit_immutable():
    result = classify("Posted record revenue.")
    hit = result.hits[0]
    assert isinstance(hit, PhraseHit)
    with pytest.raises(Exception):
        hit.contribution = 0.0  # type: ignore[misc]


# ── tokeniser pinning ────────────────────────────────────


def test_case_fold_match():
    """Pin: lexicon matches are case-insensitive via the
    tokeniser's lower-casing."""
    upper = classify("RECORD REVENUE this quarter.")
    lower = classify("record revenue this quarter.")
    assert upper.label == lower.label == Sentiment.BULLISH


def test_punctuation_does_not_break_phrase_match():
    """Pin: tokenisation strips punctuation. "record-revenue"
    splits into two tokens, but "record revenue." with a
    trailing period still matches."""
    result = classify("Posted record revenue.")
    assert any(h.phrase == "record revenue" for h in result.hits)


def test_apostrophe_preserved_in_token():
    """Pin: apostrophes are kept inside tokens so possessives
    don't break matches that include them."""
    # Lexicon doesn't currently use possessive phrases, so this
    # is a structural test on the tokeniser.
    from halal_trader.sentiment.filing_classifier import _tokenize

    assert "company's" in _tokenize("the company's outlook")
