"""Tests for sentiment/earnings_nlp.py — Round-5 Wave 11.D."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.sentiment.earnings_nlp import (
    DisclosureChange,
    DisclosureChangeType,
    SegmentSnapshot,
    SpeakerRole,
    SpeakerTurn,
    ToneClass,
    default_bearish_lexicon,
    default_bullish_lexicon,
    default_uncertainty_lexicon,
    diff_segments,
    render_call,
    render_change,
    score_call,
    score_turn,
)


def _turn(
    turn_id: int = 0,
    speaker_name: str = "alice",
    role: SpeakerRole = SpeakerRole.CEO,
    text: str = "We had a normal quarter.",
) -> SpeakerTurn:
    return SpeakerTurn(
        turn_id=turn_id,
        speaker_name=speaker_name,
        role=role,
        text=text,
    )


# --- Lexicons ----------------------------------------------------------


def test_lexicons_non_empty():
    assert default_bullish_lexicon()
    assert default_bearish_lexicon()
    assert default_uncertainty_lexicon()


# --- SpeakerTurn validation -------------------------------------------


def test_turn_valid():
    t = _turn()
    assert t.role is SpeakerRole.CEO


def test_turn_negative_id_rejected():
    with pytest.raises(ValueError):
        _turn(turn_id=-1)


def test_turn_empty_speaker_rejected():
    with pytest.raises(ValueError):
        _turn(speaker_name="")


def test_turn_empty_text_rejected():
    with pytest.raises(ValueError):
        _turn(text=" ")


def test_turn_immutable():
    t = _turn()
    with pytest.raises(AttributeError):
        t.text = "x"  # type: ignore[misc]


# --- score_turn — clean turn ------------------------------------------


def test_score_neutral_turn():
    t = _turn(text="The weather was nice today.")
    s = score_turn(t)
    assert s.tone_class is ToneClass.NEUTRAL
    assert s.bullish_hits == 0
    assert s.bearish_hits == 0


def test_score_bullish_turn():
    t = _turn(text="We saw record revenue and strong demand. Outperform our guidance.")
    s = score_turn(t)
    assert s.tone_score > 0
    assert s.tone_class in (ToneClass.BULLISH, ToneClass.VERY_BULLISH)


def test_score_bearish_turn():
    t = _turn(text="We faced headwinds and had to cut guidance amid weakness.")
    s = score_turn(t)
    assert s.tone_score < 0
    assert s.tone_class in (ToneClass.BEARISH, ToneClass.VERY_BEARISH)


def test_score_mixed_turn():
    t = _turn(text="Strong demand offset by some headwinds; raised guidance overall.")
    s = score_turn(t)
    # 2 bullish ('strong demand', 'raised guidance')
    # + 1 bearish ('headwinds') → +1/3 ≈ 0.33 → BULLISH.
    assert s.bullish_hits == 2
    assert s.bearish_hits == 1
    assert s.tone_class is ToneClass.BULLISH


def test_score_uncertainty_pin():
    """Pin: uncertainty_score = uncertainty_hits / total_words."""
    t = _turn(
        text="We expect that we may see uncertain conditions ahead.",
    )
    s = score_turn(t)
    assert s.uncertainty_hits >= 1
    assert s.uncertainty_score > 0


def test_score_word_count_pinned():
    t = _turn(text="One two three four five.")
    s = score_turn(t)
    assert s.word_count == 5


def test_score_thresholds_very_bearish():
    """tone ≤ -0.6 → VERY_BEARISH."""
    t = _turn(
        text=(
            "We faced headwinds, weakness, decline, supply constraints, "
            "guidance cut, margin compression, writedown, impairment, "
            "macro uncertainty."
        )
    )
    s = score_turn(t)
    assert s.tone_class is ToneClass.VERY_BEARISH


def test_score_thresholds_very_bullish():
    """tone ≥ 0.6 → VERY_BULLISH."""
    t = _turn(
        text=(
            "Strong demand, record revenue, raised guidance, operating "
            "leverage, outperform, beat estimates, accelerating growth, "
            "robust pipeline, improving margins, exceeded expectations, "
            "tailwinds, expanding share gains."
        )
    )
    s = score_turn(t)
    assert s.tone_class is ToneClass.VERY_BULLISH


def test_score_custom_lexicon():
    t = _turn(text="The frobnicator was high quality.")
    s = score_turn(
        t,
        bullish_lexicon=("frobnicator",),
        bearish_lexicon=(),
        uncertainty_lexicon=(),
    )
    assert s.tone_class is ToneClass.VERY_BULLISH


# --- score_call -------------------------------------------------------


def test_score_call_aggregate():
    turns = [
        _turn(turn_id=0, role=SpeakerRole.CEO, text="Strong demand and tailwinds."),
        _turn(turn_id=1, role=SpeakerRole.CFO, text="We expect stable margins."),
    ]
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=turns)
    assert score.n_turns == 2
    assert score.aggregate_tone > 0


def test_score_call_empty_rejected():
    with pytest.raises(ValueError):
        score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=[])


def test_score_call_empty_ticker_rejected():
    with pytest.raises(ValueError):
        score_call(ticker="", call_date=date(2026, 5, 10), turns=[_turn()])


def test_score_call_cfo_high_uncertainty_flagged():
    """Pin: CFO uncertainty > 4% triggers high flag."""
    cfo_turn = _turn(
        turn_id=1,
        role=SpeakerRole.CFO,
        # Stuff lots of uncertainty markers into a short text → high score.
        text=(
            "We may could should may could maybe potentially. We expect we "
            "believe should could may. Subject to uncertain conditions."
        ),
    )
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=[cfo_turn])
    assert score.cfo_uncertainty_high


def test_score_call_no_cfo_zero_uncertainty():
    turns = [
        _turn(turn_id=0, role=SpeakerRole.CEO, text="We delivered."),
    ]
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=turns)
    assert score.cfo_uncertainty_score == 0.0
    assert not score.cfo_uncertainty_high


def test_tone_trajectory_lengths_match_turns():
    turns = [_turn(turn_id=i, text="Strong demand.") for i in range(3)]
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=turns)
    assert len(score.tone_trajectory) == 3


def test_tone_trajectory_monotone_when_consistent():
    """Pin: all-bullish turns → trajectory stays at +1.0."""
    turns = [_turn(turn_id=i, text="Strong demand and beat estimates.") for i in range(3)]
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=turns)
    for v in score.tone_trajectory:
        assert v == pytest.approx(1.0)


def test_tone_trajectory_shifts_with_negative():
    turns = [
        _turn(turn_id=0, text="Strong demand and beat estimates."),
        _turn(turn_id=1, text="But headwinds and weakness ahead."),
    ]
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=turns)
    # First → 1.0; second → mix.
    assert score.tone_trajectory[0] == pytest.approx(1.0)
    assert score.tone_trajectory[-1] < 1.0


# --- SegmentSnapshot validation --------------------------------------


def test_segment_valid():
    s = SegmentSnapshot(segment_name="Cloud", revenue_usd=1_000_000.0, tone=ToneClass.BULLISH)
    assert s.tone is ToneClass.BULLISH


def test_segment_empty_name_rejected():
    with pytest.raises(ValueError):
        SegmentSnapshot(segment_name=" ", revenue_usd=0, tone=ToneClass.NEUTRAL)


def test_segment_negative_revenue_rejected():
    with pytest.raises(ValueError):
        SegmentSnapshot(segment_name="Cloud", revenue_usd=-1.0, tone=ToneClass.NEUTRAL)


# --- diff_segments ----------------------------------------------------


def test_diff_new_segment():
    prior: list[SegmentSnapshot] = []
    current = [SegmentSnapshot(segment_name="AI", revenue_usd=1.0, tone=ToneClass.BULLISH)]
    changes = diff_segments(prior, current)
    assert any(c.type is DisclosureChangeType.NEW_SEGMENT for c in changes)


def test_diff_retired_segment():
    prior = [SegmentSnapshot(segment_name="Wearables", revenue_usd=1.0, tone=ToneClass.NEUTRAL)]
    current: list[SegmentSnapshot] = []
    changes = diff_segments(prior, current)
    assert any(c.type is DisclosureChangeType.RETIRED_SEGMENT for c in changes)


def test_diff_tone_degraded():
    prior = [SegmentSnapshot(segment_name="Cloud", revenue_usd=1.0, tone=ToneClass.BULLISH)]
    current = [SegmentSnapshot(segment_name="Cloud", revenue_usd=1.0, tone=ToneClass.BEARISH)]
    changes = diff_segments(prior, current)
    assert any(c.type is DisclosureChangeType.TONE_DEGRADED for c in changes)


def test_diff_tone_improved():
    prior = [SegmentSnapshot(segment_name="Cloud", revenue_usd=1.0, tone=ToneClass.BEARISH)]
    current = [SegmentSnapshot(segment_name="Cloud", revenue_usd=1.0, tone=ToneClass.BULLISH)]
    changes = diff_segments(prior, current)
    assert any(c.type is DisclosureChangeType.TONE_IMPROVED for c in changes)


def test_diff_no_changes_emits_nothing():
    s = SegmentSnapshot(segment_name="Cloud", revenue_usd=1.0, tone=ToneClass.BULLISH)
    changes = diff_segments([s], [s])
    assert changes == ()


def test_diff_sorted_deterministic():
    prior = [
        SegmentSnapshot(segment_name="A", revenue_usd=1.0, tone=ToneClass.BULLISH),
        SegmentSnapshot(segment_name="B", revenue_usd=1.0, tone=ToneClass.BULLISH),
    ]
    current = [
        SegmentSnapshot(segment_name="B", revenue_usd=1.0, tone=ToneClass.BEARISH),
        SegmentSnapshot(segment_name="A", revenue_usd=1.0, tone=ToneClass.NEUTRAL),
    ]
    changes = diff_segments(prior, current)
    # Sorted by (type, name); both are TONE_DEGRADED here.
    assert [c.segment_name for c in changes] == ["A", "B"]


# --- Render -----------------------------------------------------------


def test_render_call_includes_emoji():
    turns = [_turn(text="Strong demand and record revenue.")]
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=turns)
    out = render_call(score)
    assert "📞" in out
    assert "AAPL" in out


def test_render_call_uncertainty_flag():
    cfo_turn = _turn(
        role=SpeakerRole.CFO,
        text="We may could should may could maybe. We expect we believe.",
    )
    score = score_call(ticker="AAPL", call_date=date(2026, 5, 10), turns=[cfo_turn])
    out = render_call(score)
    if score.cfo_uncertainty_high:
        assert "HIGH" in out


def test_render_change():
    change = DisclosureChange(
        type=DisclosureChangeType.TONE_DEGRADED,
        segment_name="Cloud",
        detail="tone bullish → bearish",
    )
    out = render_change(change)
    assert "Cloud" in out
    assert "tone_degraded" in out
