"""Tests for `ml/active_learning.py`.

Pins each component scorer, the weighted total, the stable
sort behaviour, the cold-start zero-component handling, the
input validation, and the render output.
"""

from __future__ import annotations

import pytest

from halal_trader.ml.active_learning import (
    Priority,
    ScorerWeights,
    TradeCase,
    render_queue,
    score_case,
    select_top_n,
)


def _case(
    *,
    trade_id: str = "t-001",
    pair: str = "BTCUSDT",
    predicted_return: float = 0.02,
    actual_return: float = 0.01,
    confidence: float = 0.7,
    age_seconds: float = 3600.0,
    indicator_outlier_score: float | None = None,
    rationale: str = "",
) -> TradeCase:
    return TradeCase(
        trade_id=trade_id,
        pair=pair,
        predicted_return=predicted_return,
        actual_return=actual_return,
        confidence=confidence,
        age_seconds=age_seconds,
        indicator_outlier_score=indicator_outlier_score,
        rationale=rationale,
    )


# ── TradeCase validation ─────────────────────────────────


def test_case_rejects_confidence_outside_zero_one():
    with pytest.raises(ValueError, match="confidence"):
        _case(confidence=-0.1)
    with pytest.raises(ValueError, match="confidence"):
        _case(confidence=1.5)


def test_case_rejects_negative_age():
    with pytest.raises(ValueError, match="age_seconds"):
        _case(age_seconds=-1.0)


def test_case_accepts_none_outlier_score():
    """Pin: None is the explicit "not measured" sentinel — must
    not crash."""
    c = _case(indicator_outlier_score=None)
    p = score_case(c)
    # outlier component should be 0 when not measured
    assert p.components["outlier"] == 0.0


# ── ScorerWeights validation ─────────────────────────────


def test_weights_reject_negative_components():
    with pytest.raises(ValueError, match="weight"):
        ScorerWeights(confidence_error=-0.1)
    with pytest.raises(ValueError, match="weight"):
        ScorerWeights(sign_disagreement=-1.0)


def test_weights_reject_non_positive_half_life():
    with pytest.raises(ValueError, match="half_life"):
        ScorerWeights(recency_half_life_seconds=0.0)


# ── confidence_error scorer ──────────────────────────────


def test_confidence_error_scales_with_confidence_and_magnitude():
    """Pin: a high-confidence (0.9) prediction that missed by 5%
    scores higher than a low-confidence (0.3) one with the same
    miss."""
    high = _case(predicted_return=0.05, actual_return=0.0, confidence=0.9)
    low = _case(predicted_return=0.05, actual_return=0.0, confidence=0.3)
    assert (
        score_case(high).components["confidence_error"]
        > score_case(low).components["confidence_error"]
    )


def test_confidence_error_is_zero_when_prediction_is_perfect():
    """Predicted matched actual exactly; the confidence-error
    contribution should be zero regardless of confidence."""
    c = _case(predicted_return=0.02, actual_return=0.02, confidence=1.0)
    p = score_case(c)
    assert p.components["confidence_error"] == 0.0


def test_confidence_error_capped_by_internal_limit():
    """A 5σ-event miss must not dominate the score. Pin the cap."""
    c = _case(predicted_return=0.05, actual_return=-1.0, confidence=1.0)
    p = score_case(c)
    # Internal cap is 2.0 before weight; weight is 0.4 by default.
    assert p.components["confidence_error"] <= 2.0


# ── sign_disagreement scorer ─────────────────────────────


def test_sign_disagreement_zero_when_signs_match():
    c = _case(predicted_return=0.05, actual_return=0.02)
    p = score_case(c)
    assert p.components["sign_disagreement"] == 0.0


def test_sign_disagreement_zero_when_either_is_zero():
    """Pin: a "no opinion" zero-prediction can't be 'wrong-
    direction' — the contribution stays zero."""
    c = _case(predicted_return=0.0, actual_return=0.05)
    assert score_case(c).components["sign_disagreement"] == 0.0
    c2 = _case(predicted_return=0.05, actual_return=0.0)
    assert score_case(c2).components["sign_disagreement"] == 0.0


def test_sign_disagreement_fires_when_predicted_buy_but_lost():
    """Pin: predicted +5%, actual -3% is the canonical 'embarrassing
    confident wrong-direction' trade. Must score above 0."""
    c = _case(predicted_return=0.05, actual_return=-0.03)
    p = score_case(c)
    assert p.components["sign_disagreement"] > 0


def test_sign_disagreement_scales_with_magnitude():
    small = _case(predicted_return=0.01, actual_return=-0.01)
    big = _case(predicted_return=0.10, actual_return=-0.10)
    assert (
        score_case(big).components["sign_disagreement"]
        > score_case(small).components["sign_disagreement"]
    )


# ── outlier scorer ───────────────────────────────────────


def test_outlier_passes_through_when_supplied():
    c = _case(indicator_outlier_score=0.7)
    p = score_case(c)
    expected = 0.15 * 0.7  # default weight × score
    assert p.components["outlier"] == pytest.approx(expected)


def test_outlier_clamps_above_one():
    """Pin: operator may pass an un-normalised z-score; clamp."""
    c = _case(indicator_outlier_score=5.0)
    p = score_case(c)
    # After clamp = 1.0, weighted by 0.15 default
    assert p.components["outlier"] == pytest.approx(0.15)


def test_outlier_clamps_below_zero():
    c = _case(indicator_outlier_score=-2.0)
    assert score_case(c).components["outlier"] == 0.0


# ── recency scorer ───────────────────────────────────────


def test_recency_decays_with_age():
    fresh = _case(age_seconds=0.0)
    old = _case(age_seconds=14 * 24 * 3600.0)  # 2 weeks; default 7d HL
    assert score_case(fresh).components["recency"] > score_case(old).components["recency"]


def test_recency_half_life_pins_decay():
    """Pin: at t = half-life, the recency score is exactly 0.5×
    the t=0 score. Numerical sanity check."""
    weights = ScorerWeights(
        confidence_error=0.0,
        sign_disagreement=0.0,
        outlier=0.0,
        recency=1.0,
        recency_half_life_seconds=100.0,
    )
    fresh = _case(age_seconds=0.0)
    half = _case(age_seconds=100.0)
    fresh_recency = score_case(fresh, weights=weights).components["recency"]
    half_recency = score_case(half, weights=weights).components["recency"]
    assert half_recency == pytest.approx(fresh_recency * 0.5, rel=1e-6)


def test_custom_half_life_speeds_up_decay():
    """Aggressive half-life (1h) reduces a 1d-old case's score
    to near-zero."""
    weights = ScorerWeights(
        confidence_error=0.0,
        sign_disagreement=0.0,
        outlier=0.0,
        recency=1.0,
        recency_half_life_seconds=3600.0,
    )
    day_old = _case(age_seconds=86400.0)
    p = score_case(day_old, weights=weights)
    assert p.components["recency"] < 0.01


# ── total scoring + weights ──────────────────────────────


def test_score_total_is_sum_of_weighted_components():
    p = score_case(_case())
    expected = sum(p.components.values())
    assert p.score == pytest.approx(expected)


def test_zero_weights_zero_out_components():
    """Pin: a weight of 0 zeros that component completely. Lets
    operators selectively disable a contribution."""
    weights = ScorerWeights(
        confidence_error=0.0,
        sign_disagreement=0.0,
        outlier=0.0,
        recency=1.0,
    )
    p = score_case(_case(), weights=weights)
    assert p.components["confidence_error"] == 0.0
    assert p.components["sign_disagreement"] == 0.0
    assert p.components["outlier"] == 0.0


def test_explanation_picks_dominant_component():
    """Pin: the reason field surfaces the largest contribution.
    Build a case where sign-disagreement dominates."""
    c = _case(
        predicted_return=0.10,  # confident BUY
        actual_return=-0.10,  # big loser
        confidence=0.9,
        age_seconds=30 * 24 * 3600.0,  # very old → low recency
    )
    p = score_case(c)
    # Either sign_disagreement or confidence_error dominates; both
    # are valid explanations of "the model was confidently wrong".
    assert p.reason in (
        "predicted direction was wrong",
        "high-confidence prediction missed by a wide margin",
    )


def test_explanation_handles_no_signal():
    """A case where every component is zero should report
    'no notable signal' rather than crashing or picking arbitrarily."""
    weights = ScorerWeights(
        confidence_error=0.0,
        sign_disagreement=0.0,
        outlier=0.0,
        recency=0.0,
    )
    p = score_case(_case(), weights=weights)
    assert p.reason == "no notable signal"


# ── select_top_n ─────────────────────────────────────────


def test_select_top_n_returns_n_priorities_ranked_desc():
    cases = [
        _case(trade_id="boring", predicted_return=0.01, actual_return=0.01),
        _case(
            trade_id="awful",
            predicted_return=0.10,
            actual_return=-0.10,
            confidence=0.9,
        ),
        _case(
            trade_id="middling",
            predicted_return=0.05,
            actual_return=0.0,
            confidence=0.7,
        ),
    ]
    top = select_top_n(cases, n=2)
    assert len(top) == 2
    assert top[0].score >= top[1].score
    # The "awful" confident wrong-direction case should be #1.
    assert top[0].case.trade_id == "awful"


def test_select_top_n_with_n_larger_than_input():
    """Pin: requesting more than available returns all available
    in ranked order rather than padding or raising."""
    cases = [_case(trade_id="only")]
    top = select_top_n(cases, n=10)
    assert len(top) == 1


def test_select_top_n_rejects_zero_or_negative_n():
    with pytest.raises(ValueError, match="n must be positive"):
        select_top_n([_case()], n=0)
    with pytest.raises(ValueError, match="n must be positive"):
        select_top_n([_case()], n=-1)


def test_select_top_n_stable_on_ties():
    """Pin: two cases with identical scores keep their input order
    so older trades get reviewed first when scores tie."""
    cases = [
        _case(
            trade_id="first",
            predicted_return=0.0,
            actual_return=0.0,
            confidence=0.0,
            age_seconds=10000.0,
        ),
        _case(
            trade_id="second",
            predicted_return=0.0,
            actual_return=0.0,
            confidence=0.0,
            age_seconds=10000.0,
        ),
    ]
    top = select_top_n(cases, n=2)
    assert top[0].case.trade_id == "first"
    assert top[1].case.trade_id == "second"


def test_select_top_n_handles_empty_input():
    assert select_top_n([], n=5) == []


# ── output structure ─────────────────────────────────────


def test_priority_carries_components_dict():
    p = score_case(_case())
    assert set(p.components.keys()) == {
        "confidence_error",
        "sign_disagreement",
        "outlier",
        "recency",
    }


def test_priority_is_immutable():
    p = score_case(_case())
    assert isinstance(p, Priority)
    with pytest.raises(Exception):
        p.score = 999.0  # type: ignore[misc]


def test_trade_case_is_immutable():
    c = _case()
    with pytest.raises(Exception):
        c.confidence = 0.99  # type: ignore[misc]


# ── render_queue ─────────────────────────────────────────


def test_render_includes_each_case():
    cases = [
        _case(trade_id="a", pair="BTCUSDT"),
        _case(trade_id="b", pair="ETHUSDT"),
    ]
    text = render_queue(select_top_n(cases, n=2))
    assert "a" in text
    assert "b" in text
    assert "BTCUSDT" in text
    assert "ETHUSDT" in text


def test_render_includes_score():
    p = score_case(_case())
    text = render_queue([p])
    # Score formatted to 3 decimals
    assert f"{p.score:.3f}" in text


def test_render_includes_explanation_arrow():
    p = score_case(_case())
    text = render_queue([p])
    assert "→" in text


def test_render_handles_empty_queue():
    text = render_queue([])
    assert "(empty)" in text
