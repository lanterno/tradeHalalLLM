"""Tests for core/red_team.py — Round-5 Wave 8.B."""

from __future__ import annotations

import pytest

from halal_trader.core.red_team import (
    Concern,
    RedTeamArgument,
    RedTeamPolicy,
    RedTeamStance,
    RedTeamVerdict,
    aggregate_redteam,
    render_verdict,
)

# --- Validation -----------------------------


def test_concern_string_values():
    assert Concern.LOGICAL_FALLACY.value == "logical_fallacy"
    assert Concern.OVERCONFIDENCE.value == "overconfidence"
    assert Concern.HISTORICAL_ANALOGUE.value == "historical_analogue"
    assert Concern.TAIL_RISK.value == "tail_risk"
    assert Concern.DATA_GAP.value == "data_gap"


def test_stance_string_values():
    assert RedTeamStance.PROCEED.value == "proceed"
    assert RedTeamStance.CAUTION.value == "caution"
    assert RedTeamStance.VETO.value == "veto"


def test_argument_severity_outside_unit_rejected():
    with pytest.raises(ValueError):
        RedTeamArgument(concern=Concern.TAIL_RISK, severity=1.5, summary="x")


def test_argument_empty_summary_rejected():
    with pytest.raises(ValueError):
        RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.5, summary="")


def test_default_policy():
    p = RedTeamPolicy()
    assert p.veto_severity_threshold == 0.7
    assert p.caution_severity_threshold == 0.4


def test_policy_unsorted_thresholds_rejected():
    with pytest.raises(ValueError):
        RedTeamPolicy(caution_severity_threshold=0.8, veto_severity_threshold=0.5)


def test_policy_zero_caution_rejected():
    with pytest.raises(ValueError):
        RedTeamPolicy(caution_severity_threshold=0.0)


def test_verdict_invalid_severity_rejected():
    with pytest.raises(ValueError):
        RedTeamVerdict(
            stance=RedTeamStance.PROCEED,
            arguments=(),
            max_severity=1.5,
            committee_confidence=0.5,
        )


# --- Aggregation -----------------------


def test_no_arguments_proceed():
    v = aggregate_redteam([], committee_confidence=0.8)
    assert v.stance is RedTeamStance.PROCEED
    assert v.max_severity == 0


def test_low_severity_proceed():
    args = [RedTeamArgument(concern=Concern.OVERCONFIDENCE, severity=0.2, summary="minor")]
    v = aggregate_redteam(args, committee_confidence=0.8)
    assert v.stance is RedTeamStance.PROCEED


def test_moderate_severity_caution():
    args = [RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.5, summary="risk")]
    v = aggregate_redteam(args, committee_confidence=0.8)
    assert v.stance is RedTeamStance.CAUTION


def test_high_severity_low_committee_confidence_veto():
    args = [RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.9, summary="severe")]
    v = aggregate_redteam(args, committee_confidence=0.3)
    assert v.stance is RedTeamStance.VETO


def test_high_severity_high_confidence_only_caution():
    """Even severe red-team arguments don't VETO if committee is highly confident."""
    args = [RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.9, summary="severe")]
    v = aggregate_redteam(args, committee_confidence=0.85)
    assert v.stance is RedTeamStance.CAUTION


def test_max_severity_recorded():
    args = [
        RedTeamArgument(concern=Concern.OVERCONFIDENCE, severity=0.3, summary="x"),
        RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.8, summary="y"),
    ]
    v = aggregate_redteam(args, committee_confidence=0.5)
    assert v.max_severity == 0.8


def test_committee_confidence_outside_unit_rejected():
    with pytest.raises(ValueError):
        aggregate_redteam([], committee_confidence=1.5)


# --- Render --------------------------


def test_render_proceed_check_emoji():
    v = aggregate_redteam([], committee_confidence=0.9)
    assert "✅" in render_verdict(v)


def test_render_caution_warning_emoji():
    args = [RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.5, summary="x")]
    v = aggregate_redteam(args, committee_confidence=0.8)
    assert "⚠️" in render_verdict(v)


def test_render_veto_stop_emoji():
    args = [RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.9, summary="x")]
    v = aggregate_redteam(args, committee_confidence=0.3)
    assert "⛔" in render_verdict(v)


def test_render_arguments_sorted_by_severity():
    args = [
        RedTeamArgument(concern=Concern.OVERCONFIDENCE, severity=0.3, summary="A"),
        RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.8, summary="B"),
    ]
    v = aggregate_redteam(args, committee_confidence=0.3)
    out = render_verdict(v)
    # Highest severity (B) appears first
    assert out.index("B") < out.index("A")


def test_render_no_secret_leak():
    v = aggregate_redteam([], committee_confidence=0.5)
    out = render_verdict(v)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E --------------------------


def test_e2e_overconfident_committee_redteam_caution():
    """Red team flags overconfidence in a marginally bullish committee."""
    args = [
        RedTeamArgument(
            concern=Concern.OVERCONFIDENCE,
            severity=0.55,
            summary="committee weighted growth too heavily; ignored vol regime",
        ),
        RedTeamArgument(
            concern=Concern.HISTORICAL_ANALOGUE,
            severity=0.50,
            summary="similar setup in 2018 saw a 20% drawdown",
        ),
    ]
    v = aggregate_redteam(args, committee_confidence=0.65)
    assert v.stance is RedTeamStance.CAUTION


def test_e2e_committee_uncertain_with_strong_redteam_vetoes():
    """Low-confidence committee + strong red-team severity → VETO."""
    args = [
        RedTeamArgument(
            concern=Concern.TAIL_RISK,
            severity=0.85,
            summary="fed-pivot scenario assigns 30% probability of -15%",
        )
    ]
    v = aggregate_redteam(args, committee_confidence=0.40)
    assert v.stance is RedTeamStance.VETO


def test_replay_consistency():
    args = [RedTeamArgument(concern=Concern.TAIL_RISK, severity=0.5, summary="x")]
    a = aggregate_redteam(args, committee_confidence=0.6)
    b = aggregate_redteam(args, committee_confidence=0.6)
    assert a == b
