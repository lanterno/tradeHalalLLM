"""Tests for `core/lessons_card.py` (lessons-learned card renderer).

Pins the verdict classifier (six buckets), the lesson-generation
heuristics, the markdown rendering format, and the partial-input
graceful-degradation contract (legacy trades without exit
indicators must still produce a useful card).
"""

from __future__ import annotations

from halal_trader.core.lessons_card import (
    IndicatorVector,
    LessonCard,
    LessonCardInput,
    render,
)


def _input(**overrides):
    """Build a `LessonCardInput` with sensible defaults; tests only
    specify what they need to differ."""
    base = {
        "pair": "BTCUSDT",
        "side": "buy",
        "quantity": 0.01,
        "entry_price": 60_000.0,
        "exit_price": 60_600.0,
        "return_pct": 0.01,
        "exit_reason": "take_profit",
        "llm_reasoning": "Bullish thesis based on RSI bounce.",
        "confidence": 0.7,
        "entry_indicators": IndicatorVector(
            rsi_14=45,
            macd_histogram=0.001,
            volume_ratio=1.0,
            atr_14=300,
            bb_position=0.5,
        ),
        "exit_indicators": IndicatorVector(
            rsi_14=55,
            macd_histogram=0.002,
            volume_ratio=1.0,
            atr_14=300,
            bb_position=0.6,
        ),
    }
    base.update(overrides)
    return LessonCardInput(**base)


# ── classifier ────────────────────────────────────────────


def test_winner_with_calm_indicators_is_thesis_intact():
    """Default scenario: positive return + calm indicator state →
    'winner_thesis_intact'."""
    card = render(_input())
    assert card.verdict == "winner_thesis_intact"


def test_winner_at_overbought_rsi_is_lucky():
    """If we won despite RSI ≥ 70 at entry, the card calls it a
    'lucky' winner — the bot bought near a top and got bailed out."""
    inp = _input(
        return_pct=0.02,
        entry_indicators=IndicatorVector(rsi_14=72, macd_histogram=0.001, volume_ratio=1.0),
    )
    card = render(inp)
    assert card.verdict == "winner_lucky"


def test_winner_at_upper_bb_is_lucky():
    """BB near upper band is the alternate over-extension signal."""
    inp = _input(
        return_pct=0.02,
        entry_indicators=IndicatorVector(rsi_14=55, bb_position=0.9, volume_ratio=1.0),
    )
    card = render(inp)
    assert card.verdict == "winner_lucky"


def test_loser_with_big_indicator_delta_is_thesis_invalidated():
    """A losing trade where the indicators moved a lot between
    entry and exit gets 'thesis invalidated' — regime changed."""
    inp = _input(
        return_pct=-0.02,
        exit_reason="stop_loss",
        entry_indicators=IndicatorVector(rsi_14=45, macd_histogram=0.002, volume_ratio=1.0),
        exit_indicators=IndicatorVector(rsi_14=25, macd_histogram=-0.001, volume_ratio=1.0),
    )
    card = render(inp)
    assert card.verdict == "loser_thesis_invalidated"


def test_loser_with_small_indicator_delta_is_noise():
    """When the indicator vector barely moved but we still lost,
    the trade is 'noise' — random walk killed it."""
    inp = _input(
        return_pct=-0.005,
        exit_reason="stop_loss",
        entry_indicators=IndicatorVector(rsi_14=45, macd_histogram=0.001, volume_ratio=1.0),
        exit_indicators=IndicatorVector(rsi_14=46, macd_histogram=0.001, volume_ratio=1.0),
    )
    card = render(inp)
    assert card.verdict == "loser_noise"


def test_winner_without_exit_snapshot_falls_back_to_winner():
    """Legacy trades pre-snapshot must still classify; pin the
    fallback so a partial-data row produces a useful card."""
    card = render(_input(exit_indicators=None))
    assert card.verdict == "winner"


def test_loser_without_exit_snapshot_falls_back_to_loser():
    inp = _input(return_pct=-0.01, exit_indicators=None)
    card = render(inp)
    assert card.verdict == "loser"


def test_unknown_verdict_when_return_pct_missing():
    """A trade row with no return_pct (filled but not yet closed)
    must not blow up; pin the 'unknown' bucket."""
    card = render(_input(return_pct=None))
    assert card.verdict == "unknown"


# ── lessons heuristics ────────────────────────────────────


def test_high_confidence_loser_noise_emits_conviction_lesson():
    inp = _input(
        return_pct=-0.005,
        exit_reason="stop_loss",
        confidence=0.85,
        entry_indicators=IndicatorVector(rsi_14=45, macd_histogram=0.001, volume_ratio=1.0),
        exit_indicators=IndicatorVector(rsi_14=46, macd_histogram=0.001, volume_ratio=1.0),
    )
    card = render(inp)
    assert any("conviction" in lesson.lower() for lesson in card.lessons)


def test_winner_lucky_emits_extended_entry_lesson():
    inp = _input(
        return_pct=0.02,
        entry_indicators=IndicatorVector(rsi_14=75, volume_ratio=1.0),
    )
    card = render(inp)
    assert any("extended" in lesson.lower() or "luck" in lesson.lower() for lesson in card.lessons)


def test_stop_loss_at_overbought_entry_emits_filter_lesson():
    """Stop-loss exit on RSI ≥ 70 entry → suggest filtering future
    buys above RSI 70."""
    inp = _input(
        return_pct=-0.02,
        exit_reason="stop_loss",
        entry_indicators=IndicatorVector(rsi_14=72, macd_histogram=0.001, volume_ratio=1.0),
        # exit indicators close enough to entry to bucket as 'noise'
        exit_indicators=IndicatorVector(rsi_14=70, macd_histogram=0.001, volume_ratio=1.0),
    )
    card = render(inp)
    assert any(
        "rsi 70" in lesson.lower() or "overbought" in lesson.lower() for lesson in card.lessons
    )


def test_low_volume_loser_emits_volume_filter_lesson():
    inp = _input(
        return_pct=-0.01,
        exit_reason="stop_loss",
        entry_indicators=IndicatorVector(rsi_14=50, macd_histogram=0.001, volume_ratio=0.3),
        exit_indicators=IndicatorVector(rsi_14=51, macd_histogram=0.001, volume_ratio=0.3),
    )
    card = render(inp)
    assert any("volume" in lesson.lower() for lesson in card.lessons)


def test_high_volume_winner_emits_confirmation_lesson():
    inp = _input(
        return_pct=0.02,
        entry_indicators=IndicatorVector(rsi_14=55, volume_ratio=2.5, macd_histogram=0.001),
        exit_indicators=IndicatorVector(rsi_14=58, volume_ratio=2.5, macd_histogram=0.001),
    )
    card = render(inp)
    assert any("volume" in lesson.lower() for lesson in card.lessons)


def test_trailing_stop_winner_emits_dont_loosen_lesson():
    inp = _input(
        return_pct=0.05,
        exit_reason="trailing_stop",
        entry_indicators=IndicatorVector(rsi_14=50, volume_ratio=1.5, macd_histogram=0.001),
        exit_indicators=IndicatorVector(rsi_14=58, volume_ratio=1.5, macd_histogram=0.001),
    )
    card = render(inp)
    assert any("trailing" in lesson.lower() for lesson in card.lessons)


def test_lessons_capped_at_three():
    """The card must never overwhelm an operator — pin the cap so
    a future heuristic addition doesn't produce a wall of text."""
    # Construct a case that would match many heuristics if uncapped.
    inp = _input(
        return_pct=-0.02,
        exit_reason="stop_loss",
        confidence=0.85,
        entry_indicators=IndicatorVector(rsi_14=72, volume_ratio=0.3, macd_histogram=0.001),
        exit_indicators=IndicatorVector(rsi_14=71, volume_ratio=0.3, macd_histogram=0.001),
    )
    card = render(inp)
    assert len(card.lessons) <= 3


def test_winner_thesis_intact_with_calm_data_can_have_no_lessons():
    """A textbook winner with no actionable signal need not generate
    any lesson bullet — silence is acceptable, not a bug."""
    inp = _input(
        return_pct=0.005,
        entry_indicators=IndicatorVector(rsi_14=50, volume_ratio=1.0, macd_histogram=0.001),
        exit_indicators=IndicatorVector(rsi_14=52, volume_ratio=1.0, macd_histogram=0.001),
        confidence=0.5,
    )
    card = render(inp)
    # Must not crash; lesson list may be empty.
    assert isinstance(card.lessons, list)


# ── output structure ──────────────────────────────────────


def test_card_includes_indicator_deltas_when_both_snapshots_present():
    card = render(_input())
    assert card.indicator_deltas is not None
    # Default fixture: entry RSI 45, exit RSI 55 → delta +10.
    assert card.indicator_deltas["rsi_14"] == 10


def test_card_indicator_deltas_none_when_exit_missing():
    card = render(_input(exit_indicators=None))
    assert card.indicator_deltas is None


def test_card_entry_indicators_dict_includes_all_keys():
    """The dashboard renders a fixed table of indicator keys —
    the dict must always carry the same keys, with None for missing
    fields. Pin so a refactor doesn't accidentally drop one."""
    card = render(_input(entry_indicators=IndicatorVector(rsi_14=50)))
    assert set(card.entry_indicators.keys()) == {
        "rsi_14",
        "macd_histogram",
        "volume_ratio",
        "atr_14",
        "bb_position",
    }


def test_card_handles_no_entry_indicators_at_all():
    """A trade with neither snapshot must still produce a card, just
    without the indicator sections."""
    inp = _input(entry_indicators=None, exit_indicators=None)
    card = render(inp)
    assert card.entry_indicators == {}
    assert card.indicator_deltas is None


# ── markdown output ───────────────────────────────────────


def test_markdown_includes_pair_and_verdict():
    card = render(_input())
    assert "BTCUSDT" in card.markdown
    assert card.verdict in card.markdown


def test_markdown_uses_green_emoji_for_winner():
    card = render(_input(return_pct=0.02))
    assert "🟢" in card.markdown


def test_markdown_uses_red_emoji_for_loser():
    inp = _input(return_pct=-0.02, exit_reason="stop_loss")
    card = render(inp)
    assert "🔴" in card.markdown


def test_markdown_truncates_long_rationale():
    """Long LLM rationales must not blow past notifier limits."""
    long = "Long rationale. " * 100
    card = render(_input(llm_reasoning=long))
    # Find the rationale line — must be under 250 chars total.
    rationale_lines = [line for line in card.markdown.split("\n") if line.startswith("> ")]
    assert rationale_lines, "rationale line missing"
    assert len(rationale_lines[0]) < 250


def test_markdown_includes_lesson_bullets_when_present():
    inp = _input(
        return_pct=-0.02,
        exit_reason="stop_loss",
        confidence=0.85,
        entry_indicators=IndicatorVector(rsi_14=45, macd_histogram=0.001, volume_ratio=1.0),
        exit_indicators=IndicatorVector(rsi_14=46, macd_histogram=0.001, volume_ratio=1.0),
    )
    card = render(inp)
    assert "•" in card.markdown


def test_card_is_a_lessons_card_instance():
    card = render(_input())
    assert isinstance(card, LessonCard)
