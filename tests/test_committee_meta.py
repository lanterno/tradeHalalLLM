"""Tests for core/committee_meta.py — Round-5 Wave 8.G."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.core.committee_memory import OutcomeLabel, RegimeTag
from halal_trader.core.committee_meta import (
    ResolvedDecision,
    RoleReliability,
    best_role_for_setup,
    reliability_table,
    render_table,
)
from halal_trader.core.llm_committee import AgentRole, AgentVote, Stance


def _vote(
    role: AgentRole = AgentRole.BULL,
    stance: Stance = Stance.BUY,
    confidence: float = 0.7,
) -> AgentVote:
    return AgentVote(role=role, stance=stance, confidence=confidence)


def _decision(
    ticker: str = "AAPL",
    decision_date: date = date(2026, 5, 1),
    setup_class: object = RegimeTag.BULL_TREND,
    final_stance: Stance = Stance.BUY,
    votes: tuple[AgentVote, ...] | None = None,
    outcome: OutcomeLabel = OutcomeLabel.WIN,
    return_pct: float = 0.05,
) -> ResolvedDecision:
    if votes is None:
        votes = (_vote(),)
    return ResolvedDecision(
        ticker=ticker,
        decision_date=decision_date,
        setup_class=setup_class,
        final_stance=final_stance,
        votes=votes,
        outcome=outcome,
        return_pct=return_pct,
    )


# --- ResolvedDecision validation ----------------------------------------


def test_decision_valid():
    d = _decision()
    assert d.ticker == "AAPL"


def test_decision_empty_ticker_rejected():
    with pytest.raises(ValueError):
        _decision(ticker="")


def test_decision_no_votes_rejected():
    with pytest.raises(ValueError):
        _decision(votes=tuple())


def test_decision_open_outcome_rejected():
    """Pin: ResolvedDecision must have a closed outcome."""
    with pytest.raises(ValueError):
        _decision(outcome=OutcomeLabel.OPEN, return_pct=0.0)


def test_decision_win_negative_return_rejected():
    with pytest.raises(ValueError):
        _decision(outcome=OutcomeLabel.WIN, return_pct=-0.05)


def test_decision_loss_positive_return_rejected():
    with pytest.raises(ValueError):
        _decision(outcome=OutcomeLabel.LOSS, return_pct=0.05)


def test_decision_immutable():
    d = _decision()
    with pytest.raises(AttributeError):
        d.ticker = "X"  # type: ignore[misc]


# --- reliability_table — empty ------------------------------------------


def test_reliability_empty():
    table = reliability_table([], today=date(2026, 5, 1))
    assert table == ()


def test_reliability_invalid_half_life():
    with pytest.raises(ValueError):
        reliability_table([], today=date(2026, 5, 1), half_life_days=0)


# --- Stance-correctness pin ---------------------------------------------


def test_buy_stance_correct_when_outcome_win():
    decisions = [_decision(votes=(_vote(stance=Stance.BUY, confidence=0.9),))]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    bull_row = next(r for r in table if r.role is AgentRole.BULL)
    assert bull_row.accuracy == 1.0


def test_buy_stance_wrong_when_outcome_loss():
    decisions = [
        _decision(
            votes=(_vote(stance=Stance.BUY, confidence=0.9),),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        )
    ]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    bull_row = next(r for r in table if r.role is AgentRole.BULL)
    assert bull_row.accuracy == 0.0


def test_sell_stance_correct_when_outcome_loss():
    decisions = [
        _decision(
            votes=(_vote(role=AgentRole.BEAR, stance=Stance.SELL, confidence=0.8),),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        )
    ]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    bear_row = next(r for r in table if r.role is AgentRole.BEAR)
    assert bear_row.accuracy == 1.0


def test_hold_correct_when_outcome_flat():
    decisions = [
        _decision(
            votes=(_vote(stance=Stance.HOLD, confidence=0.5),),
            outcome=OutcomeLabel.FLAT,
            return_pct=0.0,
        )
    ]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    bull_row = next(r for r in table if r.role is AgentRole.BULL)
    assert bull_row.accuracy == 1.0


def test_skip_correct_when_outcome_loss():
    """Pin: SKIP is right if the trade would have been a loss."""
    decisions = [
        _decision(
            votes=(_vote(stance=Stance.SKIP, confidence=0.7),),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        )
    ]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    bull_row = next(r for r in table if r.role is AgentRole.BULL)
    assert bull_row.accuracy == 1.0


# --- Per-(role, setup) stratification -----------------------------------


def test_stratification_per_setup():
    """Pin: separate (role, setup) bins."""
    bull_bull = _decision(
        setup_class=RegimeTag.BULL_TREND,
        votes=(_vote(role=AgentRole.BULL, stance=Stance.BUY),),
        outcome=OutcomeLabel.WIN,
        return_pct=0.05,
    )
    bull_bear = _decision(
        setup_class=RegimeTag.BEAR_TREND,
        votes=(_vote(role=AgentRole.BULL, stance=Stance.BUY),),
        outcome=OutcomeLabel.LOSS,
        return_pct=-0.05,
    )
    table = reliability_table([bull_bull, bull_bear], today=date(2026, 5, 1))
    by_setup = {r.setup_class: r for r in table if r.role is AgentRole.BULL}
    assert by_setup[RegimeTag.BULL_TREND].accuracy == 1.0
    assert by_setup[RegimeTag.BEAR_TREND].accuracy == 0.0


def test_stratification_per_role_in_same_setup():
    """Same setup, multiple roles vote; each gets its own row."""
    decisions = [
        _decision(
            votes=(
                _vote(role=AgentRole.BULL, stance=Stance.BUY, confidence=0.9),
                _vote(role=AgentRole.BEAR, stance=Stance.SELL, confidence=0.4),
                _vote(role=AgentRole.QUANT, stance=Stance.BUY, confidence=0.6),
            ),
            outcome=OutcomeLabel.WIN,
            return_pct=0.05,
        )
    ]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    by_role = {r.role: r for r in table}
    assert by_role[AgentRole.BULL].accuracy == 1.0
    assert by_role[AgentRole.BEAR].accuracy == 0.0
    assert by_role[AgentRole.QUANT].accuracy == 1.0


# --- Confidence calibration metrics -------------------------------------


def test_overconfidence_gap_zero_when_calibrated():
    """Same confidence on right + wrong → gap = 0."""
    decisions = [
        _decision(
            votes=(_vote(stance=Stance.BUY, confidence=0.7),),
            outcome=OutcomeLabel.WIN,
            return_pct=0.05,
        ),
        _decision(
            decision_date=date(2026, 5, 2),
            votes=(_vote(stance=Stance.BUY, confidence=0.7),),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        ),
    ]
    table = reliability_table(decisions, today=date(2026, 5, 5))
    row = next(r for r in table if r.role is AgentRole.BULL)
    assert abs(row.overconfidence_gap) < 0.05


def test_overconfidence_gap_positive_when_overconfident():
    """High confidence on losses, low confidence on wins → positive gap."""
    decisions = [
        _decision(
            decision_date=date(2026, 5, 1),
            votes=(_vote(stance=Stance.BUY, confidence=0.4),),
            outcome=OutcomeLabel.WIN,
            return_pct=0.05,
        ),
        _decision(
            decision_date=date(2026, 5, 2),
            votes=(_vote(stance=Stance.BUY, confidence=0.9),),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        ),
    ]
    table = reliability_table(decisions, today=date(2026, 5, 5))
    row = next(r for r in table if r.role is AgentRole.BULL)
    assert row.overconfidence_gap > 0.4


def test_avg_correct_confidence_zero_when_no_correct():
    decisions = [
        _decision(
            votes=(_vote(stance=Stance.BUY, confidence=0.9),),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        )
    ]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    row = next(r for r in table if r.role is AgentRole.BULL)
    assert row.avg_correct_confidence == 0.0


def test_avg_wrong_confidence_zero_when_no_wrong():
    decisions = [
        _decision(
            votes=(_vote(stance=Stance.BUY, confidence=0.9),),
            outcome=OutcomeLabel.WIN,
            return_pct=0.05,
        )
    ]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    row = next(r for r in table if r.role is AgentRole.BULL)
    assert row.avg_wrong_confidence == 0.0


# --- Recency decay -------------------------------------------------------


def test_recency_decay_old_dominated_by_recent():
    """Lots of old wins + 1 recent loss → accuracy < 50% under HL=60."""
    decisions = []
    for _ in range(10):
        decisions.append(
            _decision(
                decision_date=date(2025, 1, 1),
                votes=(_vote(stance=Stance.BUY, confidence=0.7),),
                outcome=OutcomeLabel.WIN,
                return_pct=0.05,
            )
        )
    decisions.append(
        _decision(
            decision_date=date(2026, 4, 30),
            votes=(_vote(stance=Stance.BUY, confidence=0.7),),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        )
    )
    table = reliability_table(decisions, today=date(2026, 5, 1), half_life_days=60)
    row = next(r for r in table if r.role is AgentRole.BULL)
    # 10 wins ~485d ago: each w ≈ 0.5^8.08 ≈ 0.00367; total ≈ 0.0367.
    # 1 loss 1 day ago: w ≈ 0.989. Total ≈ 1.026; accuracy ≈ 0.0367/1.026 ≈ 3.6%.
    assert row.accuracy < 0.20


# --- Significance threshold ---------------------------------------------


def test_is_significant_threshold():
    r = RoleReliability(
        role=AgentRole.BULL,
        setup_class=RegimeTag.BULL_TREND,
        n_samples=5,
        n_effective=2.5,
        accuracy=0.5,
        avg_correct_confidence=0.5,
        avg_wrong_confidence=0.5,
        overconfidence_gap=0.0,
    )
    assert not r.is_significant(min_n_effective=3.0)
    assert r.is_significant(min_n_effective=2.0)


# --- best_role_for_setup --------------------------------------------------


def test_best_role_for_setup_returns_top():
    """Pin: most-accurate role for a setup wins."""
    decisions = []
    for _ in range(5):
        decisions.append(
            _decision(
                votes=(
                    _vote(role=AgentRole.BULL, stance=Stance.BUY),
                    _vote(role=AgentRole.QUANT, stance=Stance.BUY),
                ),
                outcome=OutcomeLabel.WIN,
                return_pct=0.05,
            )
        )
    # Add a few BULL wrongs.
    for d in range(2):
        decisions.append(
            _decision(
                decision_date=date(2026, 5, 1) + timedelta(days=d + 1),
                votes=(
                    _vote(role=AgentRole.BULL, stance=Stance.BUY),
                    _vote(role=AgentRole.QUANT, stance=Stance.SELL),
                ),
                outcome=OutcomeLabel.LOSS,
                return_pct=-0.05,
            )
        )
    table = reliability_table(decisions, today=date(2026, 5, 10))
    best = best_role_for_setup(table, RegimeTag.BULL_TREND)
    assert best is not None
    # QUANT was right on the winners (BUY) AND right on the losers (SELL),
    # so it should beat BULL.
    assert best.role is AgentRole.QUANT


def test_best_role_for_setup_returns_none_when_insignificant():
    decisions = [_decision()]  # n=1, below threshold
    table = reliability_table(decisions, today=date(2026, 5, 1))
    assert best_role_for_setup(table, RegimeTag.BULL_TREND, min_n_effective=5) is None


def test_best_role_for_setup_unknown_setup_returns_none():
    decisions = [_decision()]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    assert best_role_for_setup(table, RegimeTag.RANGE) is None


# --- Render --------------------------------------------------------------


def test_render_empty_table():
    out = render_table(())
    assert "no resolved" in out


def test_render_contains_summary():
    decisions = [_decision() for _ in range(3)]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    out = render_table(table)
    assert "🧪" in out
    assert "acc=" in out
    assert "overconf-gap" in out


def test_render_marks_significant():
    decisions = [_decision() for _ in range(5)]
    table = reliability_table(decisions, today=date(2026, 5, 1))
    out = render_table(table)
    assert "✓" in out
