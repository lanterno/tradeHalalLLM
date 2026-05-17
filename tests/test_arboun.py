"""Tests for halal/arboun.py — Round-5 Wave 4.B."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.arboun import (
    ArbounInputs,
    ArbounIssue,
    ArbounPolicy,
    ExerciseDecision,
    StructuringResult,
    decide_exercise,
    render_exercise,
    render_structure,
    structure_arboun,
)


def _inputs(**overrides) -> ArbounInputs:
    base = {
        "arboun_id": "ARB-001",
        "buyer": "BotOperator",
        "seller": "BinanceSpot",
        "underlying": "BTC",
        "underlying_is_halal": True,
        "quantity": 1.0,
        "purchase_price_per_unit": 60000.0,
        "down_payment_amount": 6000.0,  # 10% of 60k
        "promise_date": date(2026, 5, 1),
        "exercise_date": date(2026, 8, 1),
        "down_payment_held_in_interest_account": False,
    }
    base.update(overrides)
    return ArbounInputs(**base)


# --- Validation -------------------------------------------------------------


def test_issue_string_values():
    assert ArbounIssue.DOWN_PAYMENT_TOO_SMALL.value == "down_payment_too_small"
    assert ArbounIssue.DOWN_PAYMENT_TOO_LARGE.value == "down_payment_too_large"
    assert ArbounIssue.EXERCISE_DATE_NOT_FUTURE.value == "exercise_date_not_future"
    assert ArbounIssue.EXERCISE_DATE_TOO_FAR.value == "exercise_date_too_far"
    assert ArbounIssue.NEGATIVE_QUANTITY.value == "negative_quantity"
    assert ArbounIssue.NEGATIVE_PRICE.value == "negative_price"
    assert ArbounIssue.UNDERLYING_NOT_HALAL.value == "underlying_not_halal"
    assert ArbounIssue.DOWN_PAYMENT_EARNS_INTEREST.value == "down_payment_earns_interest"


def test_default_policy():
    p = ArbounPolicy()
    assert p.min_down_payment_pct == 0.03
    assert p.max_down_payment_pct == 0.25
    assert p.max_term_days == 180


def test_policy_min_geq_max_rejected():
    with pytest.raises(ValueError):
        ArbounPolicy(min_down_payment_pct=0.30, max_down_payment_pct=0.25)


def test_policy_zero_min_rejected():
    with pytest.raises(ValueError):
        ArbounPolicy(min_down_payment_pct=0.0)


def test_policy_zero_term_rejected():
    with pytest.raises(ValueError):
        ArbounPolicy(max_term_days=0)


def test_inputs_empty_id_rejected():
    with pytest.raises(ValueError):
        _inputs(arboun_id="")


def test_inputs_empty_underlying_rejected():
    with pytest.raises(ValueError):
        _inputs(underlying="")


# --- Structure ------------------------------------------------------------


def test_clean_arboun_passes():
    r = structure_arboun(_inputs())
    assert r.is_valid


def test_arboun_haram_underlying_blocked():
    r = structure_arboun(_inputs(underlying_is_halal=False))
    assert ArbounIssue.UNDERLYING_NOT_HALAL in r.issues


def test_arboun_too_small_down_payment_blocked():
    r = structure_arboun(_inputs(down_payment_amount=600.0))  # 1% of 60k
    assert ArbounIssue.DOWN_PAYMENT_TOO_SMALL in r.issues


def test_arboun_too_large_down_payment_blocked():
    r = structure_arboun(_inputs(down_payment_amount=30000.0))  # 50% of 60k
    assert ArbounIssue.DOWN_PAYMENT_TOO_LARGE in r.issues


def test_arboun_zero_quantity_blocked():
    r = structure_arboun(_inputs(quantity=0.0))
    assert ArbounIssue.NEGATIVE_QUANTITY in r.issues


def test_arboun_negative_price_blocked():
    r = structure_arboun(_inputs(purchase_price_per_unit=-1.0))
    assert ArbounIssue.NEGATIVE_PRICE in r.issues


def test_arboun_negative_down_payment_blocked():
    r = structure_arboun(_inputs(down_payment_amount=-100.0))
    assert ArbounIssue.NEGATIVE_PRICE in r.issues


def test_arboun_interest_account_blocked():
    r = structure_arboun(_inputs(down_payment_held_in_interest_account=True))
    assert ArbounIssue.DOWN_PAYMENT_EARNS_INTEREST in r.issues


def test_arboun_exercise_in_past_blocked():
    r = structure_arboun(
        _inputs(promise_date=date(2026, 5, 1), exercise_date=date(2026, 4, 1))
    )
    assert ArbounIssue.EXERCISE_DATE_NOT_FUTURE in r.issues


def test_arboun_exercise_too_far_blocked():
    r = structure_arboun(
        _inputs(promise_date=date(2026, 1, 1), exercise_date=date(2027, 1, 1))
    )
    assert ArbounIssue.EXERCISE_DATE_TOO_FAR in r.issues


def test_arboun_at_max_term_passes():
    """Exactly 180 days should pass."""
    r = structure_arboun(
        _inputs(promise_date=date(2026, 1, 1), exercise_date=date(2026, 6, 30))
    )
    assert ArbounIssue.EXERCISE_DATE_TOO_FAR not in r.issues


def test_arboun_at_min_pct_passes():
    r = structure_arboun(_inputs(down_payment_amount=1800.0))  # 3% of 60k
    assert ArbounIssue.DOWN_PAYMENT_TOO_SMALL not in r.issues


def test_arboun_at_max_pct_passes():
    r = structure_arboun(_inputs(down_payment_amount=15000.0))  # 25% of 60k
    assert ArbounIssue.DOWN_PAYMENT_TOO_LARGE not in r.issues


def test_arboun_records_down_payment_pct():
    r = structure_arboun(_inputs(down_payment_amount=6000.0, purchase_price_per_unit=60000, quantity=1))
    assert r.down_payment_pct == pytest.approx(0.10)


def test_result_invariant_invalid_with_no_issues_rejected():
    with pytest.raises(ValueError):
        StructuringResult(
            arboun_id="x", issues=frozenset(), is_valid=False, down_payment_pct=0.10
        )


def test_result_invariant_valid_with_issues_rejected():
    with pytest.raises(ValueError):
        StructuringResult(
            arboun_id="x",
            issues=frozenset({ArbounIssue.NEGATIVE_PRICE}),
            is_valid=True,
            down_payment_pct=0.10,
        )


# --- Exercise decision ----------------------------------------------------


def test_exercise_when_in_the_money():
    inp = _inputs(quantity=2.0, purchase_price_per_unit=60000, down_payment_amount=6000)
    decision = decide_exercise(inp, settlement_price_per_unit=70000)
    assert decision.exercised is True
    assert decision.payoff == 20000.0  # (70-60)*1000 * 2


def test_forfeit_when_out_of_the_money():
    inp = _inputs(quantity=1.0, purchase_price_per_unit=60000, down_payment_amount=6000)
    decision = decide_exercise(inp, settlement_price_per_unit=50000)
    assert decision.exercised is False
    assert decision.payoff == -6000.0  # forfeit down-payment


def test_at_money_forfeits_when_loss_smaller_than_dp():
    """At-money: exercise payoff is 0, forfeit is -down_payment. Forfeit (-6000) < 0 → exercise."""
    inp = _inputs(purchase_price_per_unit=60000, down_payment_amount=6000)
    decision = decide_exercise(inp, settlement_price_per_unit=60000)
    # exercise_payoff (0) > forfeit_payoff (-6000) → exercise
    assert decision.exercised is True
    assert decision.payoff == 0.0


def test_below_money_but_above_breakeven_exercises():
    """If settlement < purchase but loss < down-payment, exercise still rational."""
    inp = _inputs(quantity=1.0, purchase_price_per_unit=60000, down_payment_amount=6000)
    # At settlement=57k, exercise loses (57k-60k) = -$3k vs forfeit -$6k → exercise
    decision = decide_exercise(inp, settlement_price_per_unit=57000)
    assert decision.exercised is True


def test_decide_negative_settlement_rejected():
    inp = _inputs()
    with pytest.raises(ValueError):
        decide_exercise(inp, settlement_price_per_unit=-1.0)


def test_exercise_decision_negative_settlement_rejected():
    with pytest.raises(ValueError):
        ExerciseDecision(arboun_id="x", settlement_price_per_unit=-1.0, exercised=False, payoff=0.0)


# --- Render --------------------------------------------------------------


def test_render_structure_clean():
    inp = _inputs()
    r = structure_arboun(inp)
    out = render_structure(inp, r)
    assert "✅" in out
    assert "BTC" in out
    assert "10.0%" in out


def test_render_structure_invalid_lists_issues():
    inp = _inputs(underlying_is_halal=False)
    r = structure_arboun(inp)
    out = render_structure(inp, r)
    assert "❌" in out
    assert "underlying_not_halal" in out


def test_render_exercise_in_money():
    inp = _inputs()
    decision = decide_exercise(inp, settlement_price_per_unit=70000)
    out = render_exercise(decision)
    assert "exercised" in out


def test_render_exercise_out_of_money():
    inp = _inputs()
    decision = decide_exercise(inp, settlement_price_per_unit=40000)
    out = render_exercise(decision)
    assert "forfeited" in out


def test_render_no_secret_leak():
    inp = _inputs()
    r = structure_arboun(inp)
    out = render_structure(inp, r)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ----------------------------------------------------------------


def test_e2e_btc_arboun_call_exercise():
    """Bullish 3-month BTC view via Arboun, in-the-money at expiry."""
    inp = _inputs(
        quantity=1.0, purchase_price_per_unit=60000.0, down_payment_amount=6000.0
    )
    r = structure_arboun(inp)
    assert r.is_valid
    decision = decide_exercise(inp, settlement_price_per_unit=72000.0)
    assert decision.exercised
    assert decision.payoff == 12000.0


def test_replay_consistency():
    inp = _inputs()
    a = structure_arboun(inp)
    b = structure_arboun(inp)
    assert a == b
