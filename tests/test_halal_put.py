"""Tests for halal/halal_put.py — Round-5 Wave 4.F."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.halal.halal_put import (
    ConditionType,
    ExerciseCondition,
    HalalPutTerms,
    HedgeProposal,
    MarketObservation,
    can_exercise,
    evaluate_condition,
    exercise,
    propose_hedge,
    render_proposal,
    render_terms,
)


def _terms(
    contract_id: str = "C1",
    hedger_id: str = "alice",
    promisor_id: str = "bob",
    underlying: str = "AAPL",
    quantity: float = 100.0,
    spot: float = 200.0,
    strike: float = 180.0,
    expiry: date = date(2026, 12, 1),
    issue_date: date = date(2026, 6, 1),
    arboun: float = 200.0,
    conditions: tuple[ExerciseCondition, ...] | None = None,
    require_all: bool = True,
) -> HalalPutTerms:
    if conditions is None:
        conditions = (ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180.0),)
    return HalalPutTerms(
        contract_id=contract_id,
        hedger_id=hedger_id,
        promisor_id=promisor_id,
        underlying=underlying,
        quantity=quantity,
        spot_at_issue=spot,
        strike=strike,
        expiry=expiry,
        issue_date=issue_date,
        arboun_paid=arboun,
        conditions=conditions,
        require_all_conditions=require_all,
    )


# --- ExerciseCondition validation ----------------------------------------


def test_condition_price_below_valid():
    c = ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=100.0)
    assert c.threshold == 100.0


def test_condition_price_below_negative_threshold_rejected():
    with pytest.raises(ValueError):
        ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=-10.0)


def test_condition_drawdown_invalid_threshold():
    with pytest.raises(ValueError):
        ExerciseCondition(condition_type=ConditionType.DRAWDOWN_OVER, threshold=1.5)
    with pytest.raises(ValueError):
        ExerciseCondition(condition_type=ConditionType.DRAWDOWN_OVER, threshold=0.0)


def test_condition_vol_invalid_threshold():
    with pytest.raises(ValueError):
        ExerciseCondition(condition_type=ConditionType.VOL_ABOVE, threshold=10.0)


def test_condition_time_elapsed_invalid_threshold():
    with pytest.raises(ValueError):
        ExerciseCondition(condition_type=ConditionType.TIME_ELAPSED, threshold=-1.0)


def test_condition_negative_window_rejected():
    with pytest.raises(ValueError):
        ExerciseCondition(
            condition_type=ConditionType.PRICE_BELOW,
            threshold=100.0,
            window_days=-1,
        )


# --- HalalPutTerms validation --------------------------------------------


def test_terms_valid():
    t = _terms()
    assert t.protection_cap() == 100.0 * 180.0


def test_terms_strike_above_spot_rejected():
    """Pin: out-of-the-money put only."""
    with pytest.raises(ValueError):
        _terms(spot=100.0, strike=120.0)


def test_terms_strike_equals_spot_allowed():
    t = _terms(spot=100.0, strike=100.0)
    assert t.strike == 100.0


def test_terms_self_dealing_rejected():
    with pytest.raises(ValueError):
        _terms(hedger_id="x", promisor_id="x")


def test_terms_no_conditions_rejected():
    """Pin: at least one exercise condition required."""
    with pytest.raises(ValueError):
        _terms(conditions=())


def test_terms_duplicate_condition_type_under_all_rejected():
    cs = (
        ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180),
        ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=170),
    )
    with pytest.raises(ValueError):
        _terms(conditions=cs, require_all=True)


def test_terms_duplicate_condition_type_under_any_allowed():
    cs = (
        ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180),
        ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=170),
    )
    t = _terms(conditions=cs, require_all=False)
    assert len(t.conditions) == 2


def test_terms_expiry_before_issue_rejected():
    with pytest.raises(ValueError):
        _terms(expiry=date(2026, 5, 1), issue_date=date(2026, 6, 1))


def test_terms_negative_arboun_rejected():
    with pytest.raises(ValueError):
        _terms(arboun=-1.0)


def test_terms_immutable():
    t = _terms()
    with pytest.raises(AttributeError):
        t.strike = 0.0  # type: ignore[misc]


# --- MarketObservation validation ----------------------------------------


def test_observation_valid():
    o = MarketObservation(
        spot=180.0,
        drawdown_from_peak=0.10,
        realised_volatility=0.30,
        days_since_issue=5,
    )
    assert o.spot == 180.0


def test_observation_drawdown_out_of_range():
    with pytest.raises(ValueError):
        MarketObservation(spot=100.0, drawdown_from_peak=1.5)


def test_observation_negative_vol_rejected():
    with pytest.raises(ValueError):
        MarketObservation(spot=100.0, realised_volatility=-0.1)


def test_observation_negative_days_rejected():
    with pytest.raises(ValueError):
        MarketObservation(spot=100.0, days_since_issue=-1)


# --- evaluate_condition ---------------------------------------------------


def test_evaluate_price_below_true():
    cond = ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180.0)
    obs = MarketObservation(spot=170.0)
    assert evaluate_condition(cond, obs) is True


def test_evaluate_price_below_false():
    cond = ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180.0)
    obs = MarketObservation(spot=190.0)
    assert evaluate_condition(cond, obs) is False


def test_evaluate_drawdown_true():
    cond = ExerciseCondition(condition_type=ConditionType.DRAWDOWN_OVER, threshold=0.10)
    obs = MarketObservation(spot=100.0, drawdown_from_peak=0.15)
    assert evaluate_condition(cond, obs) is True


def test_evaluate_vol_above_true():
    cond = ExerciseCondition(condition_type=ConditionType.VOL_ABOVE, threshold=0.30)
    obs = MarketObservation(spot=100.0, realised_volatility=0.45)
    assert evaluate_condition(cond, obs) is True


def test_evaluate_time_elapsed_true():
    cond = ExerciseCondition(condition_type=ConditionType.TIME_ELAPSED, threshold=10.0)
    obs = MarketObservation(spot=100.0, days_since_issue=15)
    assert evaluate_condition(cond, obs) is True


def test_evaluate_time_elapsed_boundary():
    cond = ExerciseCondition(condition_type=ConditionType.TIME_ELAPSED, threshold=10.0)
    obs = MarketObservation(spot=100.0, days_since_issue=10)
    assert evaluate_condition(cond, obs) is True


# --- can_exercise + exercise ----------------------------------------------


def test_can_exercise_all_conditions_true():
    t = _terms(
        require_all=True,
        conditions=(
            ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180.0),
            ExerciseCondition(condition_type=ConditionType.DRAWDOWN_OVER, threshold=0.10),
        ),
    )
    obs = MarketObservation(spot=170.0, drawdown_from_peak=0.15)
    assert can_exercise(t, obs)


def test_can_exercise_one_missing_under_all_fails():
    t = _terms(
        require_all=True,
        conditions=(
            ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180.0),
            ExerciseCondition(condition_type=ConditionType.DRAWDOWN_OVER, threshold=0.20),
        ),
    )
    obs = MarketObservation(spot=170.0, drawdown_from_peak=0.10)
    assert not can_exercise(t, obs)


def test_can_exercise_any_one_under_any_passes():
    t = _terms(
        require_all=False,
        conditions=(
            ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180.0),
            ExerciseCondition(condition_type=ConditionType.VOL_ABOVE, threshold=0.30),
        ),
    )
    obs = MarketObservation(spot=190.0, realised_volatility=0.40)
    assert can_exercise(t, obs)


def test_exercise_payout_arithmetic():
    t = _terms(strike=180.0, quantity=100.0)
    obs = MarketObservation(spot=170.0)
    res = exercise(t, obs)
    # Payout = (180 - 170) × 100 = 1000.
    assert res.payout == pytest.approx(1000.0)
    assert res.is_in_the_money


def test_exercise_payout_capped_at_protection_cap():
    """Pin: payout cannot exceed quantity × strike (the cap)."""
    t = _terms(strike=180.0, quantity=100.0)
    obs = MarketObservation(spot=0.01)  # Catastrophic drop.
    res = exercise(t, obs)
    assert res.payout <= t.protection_cap() + 1e-6


def test_exercise_returns_arboun_by_default():
    t = _terms(arboun=200.0)
    obs = MarketObservation(spot=170.0)
    res = exercise(t, obs)
    assert res.arboun_returned == 200.0


def test_exercise_can_forfeit_arboun_via_flag():
    t = _terms(arboun=200.0)
    obs = MarketObservation(spot=170.0)
    res = exercise(t, obs, return_arboun_on_exercise=False)
    assert res.arboun_returned == 0.0


def test_exercise_raises_when_conditions_unmet():
    t = _terms(
        conditions=(ExerciseCondition(condition_type=ConditionType.PRICE_BELOW, threshold=180.0),),
    )
    obs = MarketObservation(spot=200.0)
    with pytest.raises(ValueError):
        exercise(t, obs)


# --- propose_hedge --------------------------------------------------------


def test_propose_basic():
    p = propose_hedge(
        contract_id="C1",
        hedger_id="alice",
        promisor_id="bob",
        underlying="AAPL",
        quantity=100.0,
        spot=200.0,
        issue_date=date(2026, 6, 1),
        horizon_days=180,
    )
    assert isinstance(p, HedgeProposal)
    assert p.terms.strike == pytest.approx(180.0)  # 90% of 200
    assert p.terms.expiry == date(2026, 6, 1) + timedelta(days=180)
    assert not p.terms.require_all_conditions
    # Two default conditions: drawdown + vol.
    assert len(p.terms.conditions) == 2


def test_propose_arboun_arithmetic():
    p = propose_hedge(
        contract_id="C1",
        hedger_id="alice",
        promisor_id="bob",
        underlying="AAPL",
        quantity=100.0,
        spot=200.0,
        issue_date=date(2026, 6, 1),
        horizon_days=180,
        arboun_pct=0.02,
    )
    # Arboun = 100 × 200 × 0.02 = 400.
    assert p.terms.arboun_paid == pytest.approx(400.0)


def test_propose_expected_tail_payout_at_default_drawdown():
    p = propose_hedge(
        contract_id="C1",
        hedger_id="alice",
        promisor_id="bob",
        underlying="AAPL",
        quantity=100.0,
        spot=200.0,
        issue_date=date(2026, 6, 1),
        horizon_days=180,
    )
    # spot_at_tail = 200 × 0.80 = 160; intrinsic = 180 - 160 = 20; payout = 100×20 = 2000.
    assert p.expected_payout_at_drawdown == pytest.approx(2000.0)


def test_propose_invalid_strike_pct():
    with pytest.raises(ValueError):
        propose_hedge(
            contract_id="C1",
            hedger_id="alice",
            promisor_id="bob",
            underlying="AAPL",
            quantity=100.0,
            spot=200.0,
            issue_date=date(2026, 6, 1),
            horizon_days=180,
            strike_pct=1.5,
        )


def test_propose_invalid_arboun_pct():
    with pytest.raises(ValueError):
        propose_hedge(
            contract_id="C1",
            hedger_id="alice",
            promisor_id="bob",
            underlying="AAPL",
            quantity=100.0,
            spot=200.0,
            issue_date=date(2026, 6, 1),
            horizon_days=180,
            arboun_pct=0.20,
        )


def test_propose_invalid_horizon():
    with pytest.raises(ValueError):
        propose_hedge(
            contract_id="C1",
            hedger_id="alice",
            promisor_id="bob",
            underlying="AAPL",
            quantity=100.0,
            spot=200.0,
            issue_date=date(2026, 6, 1),
            horizon_days=0,
        )


def test_propose_includes_notes():
    p = propose_hedge(
        contract_id="C1",
        hedger_id="alice",
        promisor_id="bob",
        underlying="AAPL",
        quantity=100.0,
        spot=200.0,
        issue_date=date(2026, 6, 1),
        horizon_days=180,
    )
    assert len(p.notes) >= 2
    assert any("Arboun" in n for n in p.notes)


# --- Render ---------------------------------------------------------------


def test_render_terms_no_secret_leak():
    """Pin: render output masks party_id."""
    t = _terms(
        hedger_id="alice@example.com",
        promisor_id="bob@example.com",
    )
    out = render_terms(t)
    assert "alice@example.com" not in out
    assert "bob@example.com" not in out
    assert "@example" not in out


def test_render_terms_contains_strike_and_quantity():
    t = _terms(quantity=100.0, strike=180.0)
    out = render_terms(t)
    assert "100.00" in out
    assert "180.00" in out


def test_render_proposal_includes_tail_payout_and_notes():
    p = propose_hedge(
        contract_id="C1",
        hedger_id="alice",
        promisor_id="bob",
        underlying="AAPL",
        quantity=100.0,
        spot=200.0,
        issue_date=date(2026, 6, 1),
        horizon_days=180,
    )
    out = render_proposal(p)
    assert "drawdown" in out.lower()
    assert "Notes" in out
