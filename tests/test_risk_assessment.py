"""Tests for education/risk_assessment.py — Round-5 Wave 20.G."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.education.risk_assessment import (
    AllocationBand,
    Option,
    Question,
    QuestionAxis,
    Response,
    RiskProfile,
    assess,
    default_quiz,
    render_result,
)


def _opt(key: str = "a", label: str = "ans", score: int = 0) -> Option:
    return Option(key=key, label=label, score=score)


def _q(
    question_id: str = "Q1",
    axis: QuestionAxis = QuestionAxis.DRAWDOWN_TOLERANCE,
    prompt: str = "How would you react?",
    options: tuple[Option, ...] | None = None,
    weight: float = 1.0,
) -> Question:
    if options is None:
        options = (
            _opt(key="a", label="low", score=0),
            _opt(key="b", label="mid", score=2),
            _opt(key="c", label="high", score=4),
        )
    return Question(
        question_id=question_id,
        axis=axis,
        prompt=prompt,
        options=options,
        weight=weight,
    )


# --- Option validation -----------------------------------------------


def test_option_valid():
    o = _opt(score=3)
    assert o.score == 3


def test_option_empty_key_rejected():
    with pytest.raises(ValueError):
        _opt(key="")


def test_option_empty_label_rejected():
    with pytest.raises(ValueError):
        _opt(label="")


def test_option_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        _opt(score=5)
    with pytest.raises(ValueError):
        _opt(score=-1)


# --- Question validation ---------------------------------------------


def test_question_valid():
    q = _q()
    assert q.weight == 1.0


def test_question_empty_id_rejected():
    with pytest.raises(ValueError):
        _q(question_id="")


def test_question_empty_prompt_rejected():
    with pytest.raises(ValueError):
        _q(prompt="")


def test_question_no_options_rejected():
    with pytest.raises(ValueError):
        _q(options=())


def test_question_duplicate_keys_rejected():
    bad = (
        _opt(key="a", label="x", score=0),
        _opt(key="a", label="y", score=2),
    )
    with pytest.raises(ValueError):
        _q(options=bad)


def test_question_zero_weight_rejected():
    with pytest.raises(ValueError):
        _q(weight=0)


def test_question_single_score_rejected():
    """Pin: a question must span ≥ 2 distinct scores."""
    bad = (
        _opt(key="a", label="x", score=2),
        _opt(key="b", label="y", score=2),
    )
    with pytest.raises(ValueError):
        _q(options=bad)


# --- AllocationBand validation ----------------------------------------


def test_allocation_band_valid():
    b = AllocationBand(equity_min=0.3, equity_max=0.5, sukuk_min=0.5, sukuk_max=0.7)
    assert b.equity_max == 0.5


def test_allocation_band_inverted_rejected():
    with pytest.raises(ValueError):
        AllocationBand(equity_min=0.5, equity_max=0.3, sukuk_min=0.5, sukuk_max=0.7)


def test_allocation_band_infeasible_rejected():
    """Bands that never sum to 1 are rejected."""
    with pytest.raises(ValueError):
        AllocationBand(equity_min=0.2, equity_max=0.3, sukuk_min=0.2, sukuk_max=0.3)


# --- assess basic ----------------------------------------------------


def test_assess_all_low_scores_yields_conservative():
    questions = (
        _q(question_id="Q1"),
        _q(question_id="Q2"),
    )
    responses = (
        Response(question_id="Q1", option_key="a"),
        Response(question_id="Q2", option_key="a"),
    )
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.profile is RiskProfile.CONSERVATIVE


def test_assess_mid_yields_balanced():
    questions = (_q(question_id="Q1"), _q(question_id="Q2"))
    responses = (
        Response(question_id="Q1", option_key="b"),
        Response(question_id="Q2", option_key="b"),
    )
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.profile is RiskProfile.BALANCED


def test_assess_all_high_scores_yields_aggressive():
    questions = (_q(question_id="Q1"), _q(question_id="Q2"))
    responses = (
        Response(question_id="Q1", option_key="c"),
        Response(question_id="Q2", option_key="c"),
    )
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.profile is RiskProfile.AGGRESSIVE


def test_assess_raw_score_matches_weights():
    """Pin: raw_score = Σ (weight × option_score)."""
    questions = (
        _q(question_id="Q1", weight=1.0),
        _q(question_id="Q2", weight=2.0),
    )
    responses = (
        Response(question_id="Q1", option_key="c"),  # 4 × 1 = 4
        Response(question_id="Q2", option_key="b"),  # 2 × 2 = 4
    )
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.raw_score == pytest.approx(8.0)


def test_assess_max_score_uses_max_option_per_question():
    questions = (
        _q(question_id="Q1", weight=1.0),
        _q(question_id="Q2", weight=2.0),
    )
    responses = (
        Response(question_id="Q1", option_key="a"),
        Response(question_id="Q2", option_key="a"),
    )
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    # max_score = 4 × 1 + 4 × 2 = 12.
    assert result.max_score == pytest.approx(12.0)


def test_assess_normalised_in_unit_interval():
    questions = (_q(),)
    responses = (Response(question_id="Q1", option_key="b"),)
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert 0.0 <= result.normalised <= 1.0


def test_assess_empty_candidate_rejected():
    questions = (_q(),)
    responses = (Response(question_id="Q1", option_key="a"),)
    with pytest.raises(ValueError):
        assess(
            questions,
            responses,
            candidate_id="",
            assessment_date=date(2026, 5, 11),
        )


def test_assess_empty_questions_rejected():
    with pytest.raises(ValueError):
        assess([], [], candidate_id="alice", assessment_date=date(2026, 5, 11))


def test_assess_missing_response_rejected():
    questions = (_q(question_id="Q1"), _q(question_id="Q2"))
    responses = (Response(question_id="Q1", option_key="a"),)
    with pytest.raises(ValueError):
        assess(
            questions,
            responses,
            candidate_id="alice",
            assessment_date=date(2026, 5, 11),
        )


def test_assess_unknown_option_key_rejected():
    questions = (_q(question_id="Q1"),)
    responses = (Response(question_id="Q1", option_key="z"),)
    with pytest.raises(ValueError):
        assess(
            questions,
            responses,
            candidate_id="alice",
            assessment_date=date(2026, 5, 11),
        )


def test_assess_unknown_question_id_rejected():
    questions = (_q(question_id="Q1"),)
    responses = (
        Response(question_id="Q1", option_key="a"),
        Response(question_id="NOPE", option_key="a"),
    )
    with pytest.raises(ValueError):
        assess(
            questions,
            responses,
            candidate_id="alice",
            assessment_date=date(2026, 5, 11),
        )


def test_assess_duplicate_response_rejected():
    questions = (_q(question_id="Q1"),)
    responses = (
        Response(question_id="Q1", option_key="a"),
        Response(question_id="Q1", option_key="b"),
    )
    with pytest.raises(ValueError):
        assess(
            questions,
            responses,
            candidate_id="alice",
            assessment_date=date(2026, 5, 11),
        )


# --- Allocation pins -------------------------------------------------


def test_conservative_default_allocation():
    questions = (_q(),)
    responses = (Response(question_id="Q1", option_key="a"),)
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.allocation.equity_max == 0.40
    assert result.allocation.sukuk_min == 0.60


def test_aggressive_default_allocation():
    questions = (_q(),)
    responses = (Response(question_id="Q1", option_key="c"),)
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.allocation.equity_min == 0.70
    assert result.allocation.sukuk_max == 0.30


def test_allocation_override():
    custom = {
        RiskProfile.CONSERVATIVE: AllocationBand(
            equity_min=0.0, equity_max=0.20, sukuk_min=0.80, sukuk_max=1.0
        ),
        RiskProfile.BALANCED: AllocationBand(
            equity_min=0.30, equity_max=0.60, sukuk_min=0.40, sukuk_max=0.70
        ),
        RiskProfile.AGGRESSIVE: AllocationBand(
            equity_min=0.80, equity_max=1.00, sukuk_min=0.00, sukuk_max=0.20
        ),
    }
    questions = (_q(),)
    responses = (Response(question_id="Q1", option_key="c"),)
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
        allocation_overrides=custom,
    )
    assert result.allocation.equity_max == 1.00


# --- axis_scores ----------------------------------------------------


def test_axis_scores_per_axis_pinned():
    questions = (
        _q(question_id="Q-DD", axis=QuestionAxis.DRAWDOWN_TOLERANCE),
        _q(question_id="Q-TH", axis=QuestionAxis.TIME_HORIZON),
    )
    responses = (
        Response(question_id="Q-DD", option_key="a"),  # 0
        Response(question_id="Q-TH", option_key="c"),  # 4
    )
    result = assess(
        questions,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.axis_scores[QuestionAxis.DRAWDOWN_TOLERANCE] == 0.0
    assert result.axis_scores[QuestionAxis.TIME_HORIZON] == 1.0


# --- default_quiz ---------------------------------------------------


def test_default_quiz_has_five_questions():
    quiz = default_quiz()
    assert len(quiz) == 5
    axes = {q.axis for q in quiz}
    assert axes == set(QuestionAxis)


def test_default_quiz_all_low_aggregates_to_conservative():
    quiz = default_quiz()
    responses = tuple(
        Response(question_id=q.question_id, option_key=q.options[0].key) for q in quiz
    )
    result = assess(
        quiz,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.profile is RiskProfile.CONSERVATIVE


def test_default_quiz_all_high_aggregates_to_aggressive():
    quiz = default_quiz()
    responses = tuple(
        Response(question_id=q.question_id, option_key=q.options[-1].key) for q in quiz
    )
    result = assess(
        quiz,
        responses,
        candidate_id="alice",
        assessment_date=date(2026, 5, 11),
    )
    assert result.profile is RiskProfile.AGGRESSIVE


# --- Render --------------------------------------------------------


def test_render_no_secret_leak():
    quiz = default_quiz()
    responses = tuple(
        Response(question_id=q.question_id, option_key=q.options[0].key) for q in quiz
    )
    result = assess(
        quiz,
        responses,
        candidate_id="alice@example.com",
        assessment_date=date(2026, 5, 11),
    )
    out = render_result(result)
    assert "alice@example.com" not in out


def test_render_profile_emoji():
    quiz = default_quiz()
    responses = tuple(
        Response(question_id=q.question_id, option_key=q.options[0].key) for q in quiz
    )
    result = assess(quiz, responses, candidate_id="alice", assessment_date=date(2026, 5, 11))
    out = render_result(result)
    assert "🛡️" in out
