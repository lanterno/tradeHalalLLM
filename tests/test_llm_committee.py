"""Tests for core/llm_committee.py — Round-5 Wave 8.A."""

from __future__ import annotations

import pytest

from halal_trader.core.llm_committee import (
    AgentRole,
    AgentVote,
    CommitteePolicy,
    CommitteeVerdict,
    Stance,
    aggregate,
    render_verdict,
)


def _vote(role: AgentRole, stance: Stance, confidence: float = 0.7) -> AgentVote:
    return AgentVote(role=role, stance=stance, confidence=confidence)


# --- Enum string-value pins -------------------------------------------------


def test_role_string_values():
    assert AgentRole.BULL.value == "bull"
    assert AgentRole.BEAR.value == "bear"
    assert AgentRole.QUANT.value == "quant"
    assert AgentRole.HALAL_JUDGE.value == "halal_judge"
    assert AgentRole.MACRO.value == "macro"
    assert AgentRole.OPERATOR_OVERRIDE.value == "operator_override"


def test_stance_string_values():
    assert Stance.BUY.value == "buy"
    assert Stance.HOLD.value == "hold"
    assert Stance.SELL.value == "sell"
    assert Stance.SKIP.value == "skip"


# --- Validation -------------------------------------------------------------


def test_vote_negative_confidence_rejected():
    with pytest.raises(ValueError):
        AgentVote(role=AgentRole.BULL, stance=Stance.BUY, confidence=-0.1)


def test_vote_above_one_confidence_rejected():
    with pytest.raises(ValueError):
        AgentVote(role=AgentRole.BULL, stance=Stance.BUY, confidence=1.5)


def test_default_policy_loads():
    p = CommitteePolicy()
    assert p.halal_judge_veto_on_skip is True
    assert p.require_quorum == 3


def test_policy_negative_weight_rejected():
    with pytest.raises(ValueError):
        CommitteePolicy(role_weights={AgentRole.BULL: -1.0})


def test_policy_zero_quorum_rejected():
    with pytest.raises(ValueError):
        CommitteePolicy(require_quorum=0)


def test_policy_default_halal_weight_is_highest_among_role_specialists():
    """HALAL_JUDGE weighted at least as high as quant."""
    p = CommitteePolicy()
    assert p.role_weights[AgentRole.HALAL_JUDGE] >= p.role_weights[AgentRole.QUANT]


# --- Quorum -----------------------------------------------------------------


def test_below_quorum_returns_skip():
    votes = [_vote(AgentRole.BULL, Stance.BUY)]
    v = aggregate(votes)
    assert v.stance is Stance.SKIP
    assert v.confidence == 0.0


def test_at_quorum_aggregates():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY),
        _vote(AgentRole.BEAR, Stance.HOLD),
        _vote(AgentRole.QUANT, Stance.BUY),
    ]
    v = aggregate(votes)
    # Quorum met → not SKIP unless veto
    assert v.stance is not Stance.SKIP


# --- Halal-judge veto -------------------------------------------------------


def test_halal_judge_skip_vetoes():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 0.95),
        _vote(AgentRole.QUANT, Stance.BUY, 0.95),
        _vote(AgentRole.MACRO, Stance.BUY, 0.95),
        _vote(AgentRole.HALAL_JUDGE, Stance.SKIP, 0.85),
    ]
    v = aggregate(votes)
    assert v.stance is Stance.SKIP
    assert v.veto_invoked is True


def test_halal_judge_buy_does_not_veto():
    votes = [
        _vote(AgentRole.BULL, Stance.SELL, 0.9),
        _vote(AgentRole.BEAR, Stance.SELL, 0.9),
        _vote(AgentRole.QUANT, Stance.SELL, 0.9),
        _vote(AgentRole.HALAL_JUDGE, Stance.BUY, 0.7),
    ]
    v = aggregate(votes)
    # Halal-judge BUY doesn't veto SELL — but with HALAL_JUDGE weight 2.0 + others
    # the scores depend on weights. Just verify no veto invoked.
    assert v.veto_invoked is False


def test_veto_disabled_in_policy():
    """When veto_on_skip is False, halal SKIP vote is just one vote."""
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 0.95),
        _vote(AgentRole.QUANT, Stance.BUY, 0.95),
        _vote(AgentRole.MACRO, Stance.BUY, 0.95),
        _vote(AgentRole.HALAL_JUDGE, Stance.SKIP, 0.5),
    ]
    v = aggregate(votes, policy=CommitteePolicy(halal_judge_veto_on_skip=False))
    assert v.veto_invoked is False
    # Three BUY votes should outweigh the SKIP
    assert v.stance is Stance.BUY


# --- Weighted aggregation ---------------------------------------------------


def test_unanimous_buy_votes_buy():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 1.0),
        _vote(AgentRole.QUANT, Stance.BUY, 1.0),
        _vote(AgentRole.MACRO, Stance.BUY, 1.0),
        _vote(AgentRole.HALAL_JUDGE, Stance.BUY, 1.0),
    ]
    v = aggregate(votes)
    assert v.stance is Stance.BUY


def test_split_with_quant_and_halal_outweighs_bull():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 0.9),
        _vote(AgentRole.BEAR, Stance.SELL, 0.6),
        _vote(AgentRole.QUANT, Stance.SELL, 0.9),
        _vote(AgentRole.HALAL_JUDGE, Stance.SELL, 0.9),
    ]
    v = aggregate(votes)
    # QUANT (1.5) + HALAL (2.0) at 0.9 each = 3.15 SELL
    # BULL (1.0) at 0.9 BUY = 0.9
    assert v.stance is Stance.SELL


def test_operator_override_dominates():
    """OPERATOR_OVERRIDE has weight 5.0, dominates everything."""
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 1.0),
        _vote(AgentRole.QUANT, Stance.BUY, 1.0),
        _vote(AgentRole.HALAL_JUDGE, Stance.BUY, 1.0),  # not SKIP, no veto
        _vote(AgentRole.OPERATOR_OVERRIDE, Stance.HOLD, 1.0),
    ]
    v = aggregate(votes)
    assert v.stance is Stance.HOLD


def test_tied_resolves_to_hold():
    """When two stances tie, the safe choice is HOLD."""
    p = CommitteePolicy(
        role_weights={
            AgentRole.BULL: 1.0,
            AgentRole.BEAR: 1.0,
            AgentRole.QUANT: 1.0,
            AgentRole.HALAL_JUDGE: 1.0,
        },
        require_quorum=3,
        halal_judge_veto_on_skip=False,
    )
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 1.0),
        _vote(AgentRole.BEAR, Stance.SELL, 1.0),
        _vote(AgentRole.QUANT, Stance.HOLD, 0.5),
    ]
    v = aggregate(votes, policy=p)
    # BUY=1.0, SELL=1.0 tied → HOLD
    assert v.stance is Stance.HOLD


# --- Confidence -------------------------------------------------------------


def test_confidence_in_unit_interval():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 0.5),
        _vote(AgentRole.QUANT, Stance.BUY, 0.5),
        _vote(AgentRole.MACRO, Stance.BUY, 0.5),
    ]
    v = aggregate(votes)
    assert 0.0 <= v.confidence <= 1.0


def test_high_unanimous_confidence_high():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 1.0),
        _vote(AgentRole.QUANT, Stance.BUY, 1.0),
        _vote(AgentRole.MACRO, Stance.BUY, 1.0),
        _vote(AgentRole.HALAL_JUDGE, Stance.BUY, 1.0),
    ]
    v = aggregate(votes)
    assert v.confidence > 0.5


# --- Verdict invariants ----------------------------------------------------


def test_verdict_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        CommitteeVerdict(
            stance=Stance.BUY,
            confidence=1.5,
            votes=(),
            veto_invoked=False,
            weighted_scores={s: 0.0 for s in Stance},
        )


# --- Render -----------------------------------------------------------------


def test_render_verdict_includes_summary():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 0.8),
        _vote(AgentRole.QUANT, Stance.BUY, 0.7),
        _vote(AgentRole.MACRO, Stance.HOLD, 0.5),
    ]
    v = aggregate(votes)
    out = render_verdict(v)
    assert "Committee verdict" in out
    assert "bull" in out
    assert "quant" in out
    assert "macro" in out


def test_render_veto_marker():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 1.0),
        _vote(AgentRole.QUANT, Stance.BUY, 1.0),
        _vote(AgentRole.MACRO, Stance.BUY, 1.0),
        _vote(AgentRole.HALAL_JUDGE, Stance.SKIP, 1.0),
    ]
    v = aggregate(votes)
    out = render_verdict(v)
    assert "HALAL VETO" in out


def test_render_no_secret_leak():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY),
        _vote(AgentRole.QUANT, Stance.BUY),
        _vote(AgentRole.MACRO, Stance.BUY),
    ]
    v = aggregate(votes)
    out = render_verdict(v)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_typical_committee_decision_buy():
    votes = [
        AgentVote(role=AgentRole.BULL, stance=Stance.BUY, confidence=0.8, rationale="momentum"),
        AgentVote(role=AgentRole.BEAR, stance=Stance.HOLD, confidence=0.6, rationale="overbought"),
        AgentVote(role=AgentRole.QUANT, stance=Stance.BUY, confidence=0.85, rationale="signal +"),
        AgentVote(role=AgentRole.MACRO, stance=Stance.BUY, confidence=0.7, rationale="risk-on"),
        AgentVote(
            role=AgentRole.HALAL_JUDGE,
            stance=Stance.BUY,
            confidence=0.9,
            rationale="all clauses pass",
        ),
    ]
    v = aggregate(votes)
    assert v.stance is Stance.BUY


def test_replay_consistency():
    votes = [
        _vote(AgentRole.BULL, Stance.BUY, 0.7),
        _vote(AgentRole.QUANT, Stance.BUY, 0.7),
        _vote(AgentRole.MACRO, Stance.BUY, 0.7),
    ]
    a = aggregate(votes)
    b = aggregate(votes)
    assert a == b
