"""Tests for `core/committee.py` (model-voting committee).

Pins each voting policy (majority / weighted / unanimous), the
conservative tiebreak rule, the BUY-confidence-is-min vs
close-confidence-is-mean asymmetry, the median-quantity rule,
the input validation, and the render output.
"""

from __future__ import annotations

import pytest

from halal_trader.core.committee import (
    Action,
    CommitteeVerdict,
    VariantAttribution,
    VariantDecision,
    VotingPolicy,
    render_verdict,
    vote,
)


def _decision(
    *,
    variant: str = "gpt-4o",
    action: Action | str = Action.BUY,
    confidence: float = 0.7,
    quantity: float = 0.01,
) -> VariantDecision:
    return VariantDecision(
        variant=variant,
        action=action,
        confidence=confidence,
        quantity=quantity,
    )


# ── VariantDecision validation ───────────────────────────


def test_decision_rejects_confidence_outside_zero_one():
    with pytest.raises(ValueError, match="confidence"):
        _decision(confidence=-0.1)
    with pytest.raises(ValueError, match="confidence"):
        _decision(confidence=1.5)


def test_decision_rejects_negative_quantity():
    with pytest.raises(ValueError, match="quantity"):
        _decision(quantity=-1.0)


def test_decision_accepts_action_as_string():
    """Pin: a JSON-loaded decision (lossy on the enum) must feed
    in directly."""
    d = _decision(action="buy")
    v = vote([d])
    assert v.action == Action.BUY


def test_unknown_action_string_rejected():
    """Pin: typo'd action surfaces immediately rather than
    silently degrading."""
    with pytest.raises(ValueError, match="unknown action"):
        vote([_decision(action="bbuy")])


# ── empty input ──────────────────────────────────────────


def test_empty_decisions_returns_hold():
    """Pin: no variants → HOLD with zero confidence. A no-variants
    cycle (every model failed) must not accidentally promote a
    blank vote into BUY/SELL."""
    v = vote([])
    assert v.action == Action.HOLD
    assert v.confidence == 0.0
    assert v.quantity == 0.0
    assert "no variants" in v.reason.lower()


# ── majority voting ──────────────────────────────────────


def test_majority_picks_most_common_action():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.SELL),
    ]
    v = vote(decisions, policy=VotingPolicy.MAJORITY)
    assert v.action == Action.BUY


def test_majority_tiebreak_picks_more_conservative():
    """Pin: HOLD > SELL > BUY. A 1-1-1 split → HOLD; a 2-2 BUY
    vs SELL → SELL (more conservative than BUY)."""
    three_way = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.SELL),
        _decision(variant="c", action=Action.HOLD),
    ]
    assert vote(three_way).action == Action.HOLD

    two_two = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.SELL),
        _decision(variant="d", action=Action.SELL),
    ]
    assert vote(two_two).action == Action.SELL


def test_majority_unanimous_buy_returns_buy():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.BUY),
    ]
    v = vote(decisions)
    assert v.action == Action.BUY


# ── weighted voting ──────────────────────────────────────


def test_weighted_uses_weights_not_counts():
    """Pin: a high-weight BUY can outvote two lower-weight
    SELLs — the WEIGHTED policy uses sums of weights, not counts."""
    decisions = [
        _decision(variant="strong", action=Action.BUY),
        _decision(variant="weak1", action=Action.SELL),
        _decision(variant="weak2", action=Action.SELL),
    ]
    v = vote(
        decisions,
        policy=VotingPolicy.WEIGHTED,
        weights={"strong": 5.0, "weak1": 1.0, "weak2": 1.0},
    )
    assert v.action == Action.BUY


def test_weighted_default_weight_is_one():
    """Pin: variants without an explicit weight default to 1.0."""
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.SELL),
    ]
    # No weights dict → all 1.0 → BUY 2-1 SELL → BUY wins.
    v = vote(decisions, policy=VotingPolicy.WEIGHTED)
    assert v.action == Action.BUY


def test_weighted_clamps_negative_weights_to_zero():
    """A buggy config with a negative weight must not silently
    invert the contribution."""
    decisions = [
        _decision(variant="bug", action=Action.BUY),
        _decision(variant="ok", action=Action.SELL),
    ]
    # Negative weight on the BUY → clamps to 0; SELL (weight 1) wins.
    v = vote(
        decisions,
        policy=VotingPolicy.WEIGHTED,
        weights={"bug": -10.0, "ok": 1.0},
    )
    assert v.action == Action.SELL


def test_weighted_zero_total_defaults_to_hold():
    """If every weight is zero (every variant abstains), the
    committee can't tell — must default to HOLD."""
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.SELL),
    ]
    v = vote(
        decisions,
        policy=VotingPolicy.WEIGHTED,
        weights={"a": 0.0, "b": 0.0},
    )
    assert v.action == Action.HOLD


def test_weighted_tiebreak_picks_more_conservative():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.SELL),
    ]
    # Equal weights → tie → conservative → SELL.
    v = vote(
        decisions,
        policy=VotingPolicy.WEIGHTED,
        weights={"a": 1.0, "b": 1.0},
    )
    assert v.action == Action.SELL


# ── unanimous voting ─────────────────────────────────────


def test_unanimous_returns_buy_when_all_agree():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.BUY),
    ]
    v = vote(decisions, policy=VotingPolicy.UNANIMOUS)
    assert v.action == Action.BUY


def test_unanimous_holds_on_any_dissent():
    """Pin: every variant must agree on the same non-HOLD action.
    A 3-of-4 BUY with one SELL → HOLD."""
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.BUY),
        _decision(variant="d", action=Action.SELL),
    ]
    v = vote(decisions, policy=VotingPolicy.UNANIMOUS)
    assert v.action == Action.HOLD
    assert "split" in v.reason.lower()


def test_unanimous_returns_hold_when_all_say_hold():
    decisions = [
        _decision(variant="a", action=Action.HOLD),
        _decision(variant="b", action=Action.HOLD),
    ]
    v = vote(decisions, policy=VotingPolicy.UNANIMOUS)
    assert v.action == Action.HOLD
    assert "every variant said HOLD" in v.reason


# ── BUY confidence aggregation ──────────────────────────


def test_buy_confidence_is_minimum_across_agreeing_variants():
    """Pin: a 3-of-5 BUY where the lowest agreeing variant says
    0.4 should size off 0.4, not 0.9 — opening risk needs the
    strictest test."""
    decisions = [
        _decision(variant="a", action=Action.BUY, confidence=0.9),
        _decision(variant="b", action=Action.BUY, confidence=0.4),
        _decision(variant="c", action=Action.BUY, confidence=0.7),
        _decision(variant="d", action=Action.SELL, confidence=0.95),
        _decision(variant="e", action=Action.SELL, confidence=0.5),
    ]
    v = vote(decisions)
    assert v.action == Action.BUY
    assert v.confidence == 0.4  # min of agreeing variants


def test_buy_quantity_is_median_across_agreeing_variants():
    """Pin: median is more robust to outsized-bet outliers than
    mean."""
    decisions = [
        _decision(variant="a", action=Action.BUY, quantity=0.01),
        _decision(variant="b", action=Action.BUY, quantity=0.02),
        _decision(variant="c", action=Action.BUY, quantity=10.0),  # outlier
    ]
    v = vote(decisions)
    assert v.quantity == 0.02  # median, not (0.01+0.02+10)/3


def test_sell_confidence_is_mean_across_agreeing_variants():
    """Pin: SELL confidence uses the mean — closing risk averages
    confidence; the asymmetry with BUY is intentional."""
    decisions = [
        _decision(variant="a", action=Action.SELL, confidence=0.6),
        _decision(variant="b", action=Action.SELL, confidence=0.8),
    ]
    v = vote(decisions)
    assert v.action == Action.SELL
    assert v.confidence == pytest.approx(0.7)


def test_sell_quantity_is_zero():
    """Pin: SELL quantity is set by the executor (which knows the
    operator's open position), not by the committee."""
    decisions = [
        _decision(variant="a", action=Action.SELL, quantity=0.5),
        _decision(variant="b", action=Action.SELL, quantity=0.5),
    ]
    v = vote(decisions)
    assert v.quantity == 0.0


def test_hold_confidence_is_mean():
    decisions = [
        _decision(variant="a", action=Action.HOLD, confidence=0.5),
        _decision(variant="b", action=Action.HOLD, confidence=0.7),
    ]
    v = vote(decisions)
    assert v.confidence == pytest.approx(0.6)


# ── attribution ──────────────────────────────────────────


def test_attribution_carries_one_entry_per_variant():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.SELL),
        _decision(variant="c", action=Action.HOLD),
    ]
    v = vote(decisions)
    assert len(v.attributions) == 3
    assert {a.variant for a in v.attributions} == {"a", "b", "c"}


def test_attribution_marks_agreement_correctly():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.SELL),
    ]
    v = vote(decisions)
    assert v.action == Action.BUY
    by_variant = {a.variant: a for a in v.attributions}
    assert by_variant["a"].agreed_with_committee
    assert by_variant["b"].agreed_with_committee
    assert not by_variant["c"].agreed_with_committee


def test_attribution_carries_assigned_weight():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
    ]
    v = vote(
        decisions,
        policy=VotingPolicy.WEIGHTED,
        weights={"a": 2.0, "b": 0.5},
    )
    by_variant = {a.variant: a for a in v.attributions}
    assert by_variant["a"].weight == 2.0
    assert by_variant["b"].weight == 0.5


def test_agreement_count_matches_attributions():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.SELL),
    ]
    v = vote(decisions)
    assert v.agreement_count == 2
    assert v.total_variants == 3


# ── verdict structure ────────────────────────────────────


def test_verdict_is_immutable():
    v = vote([_decision()])
    assert isinstance(v, CommitteeVerdict)
    with pytest.raises(Exception):
        v.action = Action.SELL  # type: ignore[misc]


def test_attribution_is_immutable():
    v = vote([_decision()])
    a = v.attributions[0]
    assert isinstance(a, VariantAttribution)
    with pytest.raises(Exception):
        a.confidence = 0.0  # type: ignore[misc]


# ── render_verdict ───────────────────────────────────────


def test_render_includes_action_and_emoji():
    v = vote([_decision(action=Action.BUY)])
    text = render_verdict(v)
    assert "BUY" in text
    assert "🟢" in text


def test_render_uses_red_for_sell():
    v = vote([_decision(action=Action.SELL)])
    text = render_verdict(v)
    assert "🔴" in text


def test_render_uses_yellow_for_hold():
    v = vote([])
    text = render_verdict(v)
    assert "🟡" in text


def test_render_includes_variant_attribution():
    decisions = [
        _decision(variant="gpt-4o", action=Action.BUY),
        _decision(variant="claude", action=Action.SELL),
    ]
    text = render_verdict(vote(decisions))
    assert "gpt-4o" in text
    assert "claude" in text


def test_render_marks_agreeing_variants_distinctly():
    decisions = [
        _decision(variant="a", action=Action.BUY),
        _decision(variant="b", action=Action.BUY),
        _decision(variant="c", action=Action.SELL),
    ]
    text = render_verdict(vote(decisions))
    # ✓ for agreeing, · for dissenting
    assert "✓" in text
    assert "·" in text


def test_render_includes_buy_confidence_and_qty():
    v = vote(
        [
            _decision(variant="a", action=Action.BUY, confidence=0.7, quantity=0.5),
            _decision(variant="b", action=Action.BUY, confidence=0.6, quantity=0.7),
        ]
    )
    text = render_verdict(v)
    # Confidence: 60% (min); Qty: 0.6 (median)
    assert "60%" in text
    assert "0.6" in text
