"""Tests for halal/waad.py — Round-5 Wave 4.A."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.waad import (
    PayoffAtExpiry,
    StructuringPolicy,
    WaadDirection,
    WaadInputs,
    WaadIssue,
    detect_bilateral_pair,
    render_waad,
    structure_pair,
    structure_waad,
    synthetic_call_payoff,
    synthetic_put_payoff,
)


def _call_inputs(**overrides) -> WaadInputs:
    base = {
        "waad_id": "WAAD-001",
        "direction": WaadDirection.PROMISE_TO_BUY,
        "promisor": "BotOperator",
        "promisee": "BinanceSpot",
        "underlying": "BTC",
        "quantity": 1.0,
        "strike_price": 65000.0,
        "market_price": 65000.0,
        "promise_date": date(2026, 5, 1),
        "exercise_date": date(2026, 8, 1),
        "premium_paid": 0.0,
    }
    base.update(overrides)
    return WaadInputs(**base)


def _put_inputs(**overrides) -> WaadInputs:
    base = {
        "waad_id": "WAAD-002",
        "direction": WaadDirection.PROMISE_TO_SELL,
        "promisor": "BotOperator",
        "promisee": "BinanceSpot",
        "underlying": "BTC",
        "quantity": 1.0,
        "strike_price": 60000.0,
        "market_price": 65000.0,
        "promise_date": date(2026, 5, 1),
        "exercise_date": date(2026, 8, 1),
        "premium_paid": 0.0,
    }
    base.update(overrides)
    return WaadInputs(**base)


# --- Enum + validation -----------------------------------------------------


def test_direction_string_values():
    assert WaadDirection.PROMISE_TO_BUY.value == "promise_to_buy"
    assert WaadDirection.PROMISE_TO_SELL.value == "promise_to_sell"


def test_issue_string_values():
    assert WaadIssue.BILATERAL_WAAD_BAN.value == "bilateral_waad_ban"
    assert WaadIssue.PREMIUM_CHARGED.value == "premium_charged"
    assert WaadIssue.STRIKE_OFF_MARKET.value == "strike_off_market"
    assert WaadIssue.EXERCISE_DATE_NOT_FUTURE.value == "exercise_date_not_future"
    assert WaadIssue.EXERCISE_DATE_TOO_FAR.value == "exercise_date_too_far"
    assert WaadIssue.QUANTITY_NON_POSITIVE.value == "quantity_non_positive"
    assert WaadIssue.STRIKE_NON_POSITIVE.value == "strike_non_positive"
    assert WaadIssue.EMPTY_PROMISOR.value == "empty_promisor"
    assert WaadIssue.EMPTY_PROMISEE.value == "empty_promisee"


def test_default_policy_max_term_one_year():
    p = StructuringPolicy()
    assert p.max_term_days == 365


def test_policy_zero_term_rejected():
    with pytest.raises(ValueError):
        StructuringPolicy(max_term_days=0)


def test_policy_zero_off_market_rejected():
    with pytest.raises(ValueError):
        StructuringPolicy(strike_off_market_pct=0.0)


def test_policy_above_one_off_market_rejected():
    with pytest.raises(ValueError):
        StructuringPolicy(strike_off_market_pct=1.1)


def test_inputs_empty_id_rejected():
    with pytest.raises(ValueError):
        _call_inputs(waad_id="")


def test_inputs_empty_underlying_rejected():
    with pytest.raises(ValueError):
        _call_inputs(underlying=" ")


def test_inputs_zero_market_rejected():
    with pytest.raises(ValueError):
        _call_inputs(market_price=0.0)


# --- Single Wa'd -----------------------------------------------------------


def test_clean_call_passes():
    r = structure_waad(_call_inputs())
    assert r.is_valid


def test_clean_put_passes():
    r = structure_waad(_put_inputs())
    assert r.is_valid


def test_premium_charged_blocks():
    r = structure_waad(_call_inputs(premium_paid=100.0))
    assert WaadIssue.PREMIUM_CHARGED in r.issues
    assert not r.is_valid


def test_negative_quantity_blocked():
    r = structure_waad(_call_inputs(quantity=-1.0))
    assert WaadIssue.QUANTITY_NON_POSITIVE in r.issues


def test_zero_strike_blocked():
    r = structure_waad(_call_inputs(strike_price=0.0))
    assert WaadIssue.STRIKE_NON_POSITIVE in r.issues


def test_empty_promisor_blocked():
    r = structure_waad(_call_inputs(promisor=" "))
    assert WaadIssue.EMPTY_PROMISOR in r.issues


def test_empty_promisee_blocked():
    r = structure_waad(_call_inputs(promisee=""))
    assert WaadIssue.EMPTY_PROMISEE in r.issues


def test_exercise_in_past_blocked():
    r = structure_waad(_call_inputs(exercise_date=date(2026, 4, 30), promise_date=date(2026, 5, 1)))
    assert WaadIssue.EXERCISE_DATE_NOT_FUTURE in r.issues


def test_exercise_too_far_blocked():
    r = structure_waad(_call_inputs(promise_date=date(2026, 5, 1), exercise_date=date(2027, 6, 1)))
    assert WaadIssue.EXERCISE_DATE_TOO_FAR in r.issues


def test_at_max_term_passes():
    r = structure_waad(_call_inputs(promise_date=date(2026, 5, 1), exercise_date=date(2027, 5, 1)))
    assert WaadIssue.EXERCISE_DATE_TOO_FAR not in r.issues


def test_strike_off_market_blocked():
    """Strike at 200% of market triggers off-market flag."""
    r = structure_waad(_call_inputs(strike_price=130000.0, market_price=65000.0))
    assert WaadIssue.STRIKE_OFF_MARKET in r.issues


def test_strike_at_market_passes():
    r = structure_waad(_call_inputs(strike_price=65000.0, market_price=65000.0))
    assert WaadIssue.STRIKE_OFF_MARKET not in r.issues


def test_strike_within_50pct_passes():
    r = structure_waad(_call_inputs(strike_price=70000.0, market_price=65000.0))
    assert WaadIssue.STRIKE_OFF_MARKET not in r.issues


# --- Bilateral Wa'd detection ----------------------------------------------


def test_bilateral_pair_same_parties_opposite_directions_detected():
    a = _call_inputs(
        waad_id="A", promisor="X", promisee="Y", direction=WaadDirection.PROMISE_TO_BUY
    )
    b = _call_inputs(
        waad_id="B", promisor="Y", promisee="X", direction=WaadDirection.PROMISE_TO_SELL
    )
    assert detect_bilateral_pair(a, b)


def test_bilateral_pair_different_parties_not_bilateral():
    a = _call_inputs(
        waad_id="A", promisor="X", promisee="Y", direction=WaadDirection.PROMISE_TO_BUY
    )
    b = _call_inputs(
        waad_id="B", promisor="Z", promisee="Y", direction=WaadDirection.PROMISE_TO_SELL
    )
    assert not detect_bilateral_pair(a, b)


def test_bilateral_pair_same_direction_not_bilateral():
    a = _call_inputs(
        waad_id="A", promisor="X", promisee="Y", direction=WaadDirection.PROMISE_TO_BUY
    )
    b = _call_inputs(
        waad_id="B", promisor="X", promisee="Y", direction=WaadDirection.PROMISE_TO_BUY
    )
    assert not detect_bilateral_pair(a, b)


def test_bilateral_pair_different_underlying_not_bilateral():
    a = _call_inputs(
        waad_id="A",
        promisor="X",
        promisee="Y",
        direction=WaadDirection.PROMISE_TO_BUY,
        underlying="BTC",
    )
    b = _call_inputs(
        waad_id="B",
        promisor="Y",
        promisee="X",
        direction=WaadDirection.PROMISE_TO_SELL,
        underlying="ETH",
    )
    assert not detect_bilateral_pair(a, b)


def test_structure_pair_flags_bilateral():
    a = _call_inputs(
        waad_id="A", promisor="X", promisee="Y", direction=WaadDirection.PROMISE_TO_BUY
    )
    b = _call_inputs(
        waad_id="B", promisor="Y", promisee="X", direction=WaadDirection.PROMISE_TO_SELL
    )
    ra, rb = structure_pair(a, b)
    assert WaadIssue.BILATERAL_WAAD_BAN in ra.issues
    assert WaadIssue.BILATERAL_WAAD_BAN in rb.issues
    assert not ra.is_valid
    assert not rb.is_valid


def test_structure_pair_distinct_counterparties_passes():
    """Two Wa'ds with different counterparties — bot can promise A→buy, B→sell separately."""
    a = _call_inputs(
        waad_id="A", promisor="Bot", promisee="X", direction=WaadDirection.PROMISE_TO_BUY
    )
    b = _call_inputs(
        waad_id="B", promisor="Bot", promisee="Y", direction=WaadDirection.PROMISE_TO_SELL
    )
    ra, rb = structure_pair(a, b)
    assert ra.is_valid
    assert rb.is_valid


# --- Synthetic payoffs -----------------------------------------------------


def test_call_in_the_money():
    waad = _call_inputs(strike_price=60000.0, quantity=2.0)
    payoff = synthetic_call_payoff(waad, settlement_price=65000.0)
    assert payoff.payoff == 10000.0  # (65k - 60k) * 2


def test_call_out_of_money_zero_payoff():
    waad = _call_inputs(strike_price=70000.0)
    payoff = synthetic_call_payoff(waad, settlement_price=65000.0)
    assert payoff.payoff == 0.0


def test_call_at_money_zero_payoff():
    waad = _call_inputs(strike_price=65000.0)
    payoff = synthetic_call_payoff(waad, settlement_price=65000.0)
    assert payoff.payoff == 0.0


def test_call_requires_promise_to_buy_direction():
    waad = _put_inputs()  # promise_to_sell
    with pytest.raises(ValueError):
        synthetic_call_payoff(waad, settlement_price=65000.0)


def test_call_negative_settlement_rejected():
    waad = _call_inputs()
    with pytest.raises(ValueError):
        synthetic_call_payoff(waad, settlement_price=-1.0)


def test_put_in_the_money():
    waad = _put_inputs(strike_price=70000.0, quantity=2.0)
    payoff = synthetic_put_payoff(waad, settlement_price=60000.0)
    assert payoff.payoff == 20000.0  # (70k - 60k) * 2


def test_put_out_of_money_zero():
    waad = _put_inputs(strike_price=60000.0)
    payoff = synthetic_put_payoff(waad, settlement_price=65000.0)
    assert payoff.payoff == 0.0


def test_put_requires_promise_to_sell():
    waad = _call_inputs()
    with pytest.raises(ValueError):
        synthetic_put_payoff(waad, settlement_price=60000.0)


def test_payoff_negative_settlement_in_dataclass_rejected():
    with pytest.raises(ValueError):
        PayoffAtExpiry(waad_id="x", settlement_price=-1.0, payoff=0.0)


# --- Render ----------------------------------------------------------------


def test_render_clean_call():
    inp = _call_inputs()
    r = structure_waad(inp)
    out = render_waad(inp, r)
    assert "✅" in out
    assert "promise_to_buy" in out
    assert "BTC" in out


def test_render_invalid_lists_issues():
    inp = _call_inputs(premium_paid=10.0)
    r = structure_waad(inp)
    out = render_waad(inp, r)
    assert "❌" in out
    assert "premium_charged" in out


def test_render_no_secret_leak():
    inp = _call_inputs()
    r = structure_waad(inp)
    out = render_waad(inp, r)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_synthetic_btc_call_via_waad():
    """Bullish 3-month BTC view via PROMISE_TO_BUY at $65k."""
    inp = _call_inputs(strike_price=65000.0, quantity=2.0)
    r = structure_waad(inp)
    assert r.is_valid
    payoff = synthetic_call_payoff(inp, settlement_price=72000.0)
    assert payoff.payoff == 14000.0


def test_e2e_synthetic_btc_put_via_waad():
    """Bearish 3-month BTC view via PROMISE_TO_SELL at $65k."""
    inp = _put_inputs(strike_price=65000.0, quantity=1.0)
    r = structure_waad(inp)
    assert r.is_valid
    payoff = synthetic_put_payoff(inp, settlement_price=58000.0)
    assert payoff.payoff == 7000.0


def test_replay_consistency():
    inp = _call_inputs()
    a = structure_waad(inp)
    b = structure_waad(inp)
    assert a == b
