"""Tests for the idle-cash treasury policy."""

from __future__ import annotations

import pytest

from halal_trader.core.treasury import (
    DEFAULT_HALAL_INSTRUMENTS,
    TreasuryPolicy,
    estimate_annual_yield_usd,
    plan_idle_cash,
)


def _policy(**overrides) -> TreasuryPolicy:
    base = dict(min_idle_pct=0.10, deploy_threshold_usd=100.0, redeem_threshold_usd=50.0)
    base.update(overrides)
    return TreasuryPolicy(**base)


# ── Policy validation ────────────────────────────────────────────


def test_policy_default_is_halal() -> None:
    pol = TreasuryPolicy()
    assert pol.target_instrument in DEFAULT_HALAL_INSTRUMENTS


def test_policy_rejects_non_halal_target() -> None:
    with pytest.raises(ValueError):
        TreasuryPolicy(target_instrument="SPY")  # not in halal allow-list


def test_policy_rejects_invalid_min_idle_pct() -> None:
    with pytest.raises(ValueError):
        TreasuryPolicy(min_idle_pct=1.5)
    with pytest.raises(ValueError):
        TreasuryPolicy(min_idle_pct=-0.1)


def test_custom_halal_instruments() -> None:
    pol = TreasuryPolicy(
        target_instrument="MYSUK",
        halal_instruments=("MYSUK", "SPSK"),
    )
    assert pol.is_halal("MYSUK")
    assert pol.is_halal("spsk")  # case-insensitive


# ── Plan ─────────────────────────────────────────────────────────


def test_plan_deploys_when_cash_far_above_floor() -> None:
    plan = plan_idle_cash(
        cash_balance=1000.0,
        positions_value=0.0,
        current_treasury_value=0.0,
        policy=_policy(),
    )
    assert plan.action == "deploy"
    # equity 1000, floor 100 -> deploy 900
    assert plan.amount_usd == pytest.approx(900.0)
    assert plan.cash_target == pytest.approx(100.0)
    assert plan.treasury_target == pytest.approx(900.0)


def test_plan_redeems_when_cash_below_floor() -> None:
    plan = plan_idle_cash(
        cash_balance=50.0,
        positions_value=900.0,
        current_treasury_value=200.0,
        policy=_policy(),  # floor = 0.1 * 1150 = 115
    )
    assert plan.action == "redeem"
    assert plan.amount_usd == pytest.approx(65.0, abs=0.01)


def test_plan_holds_when_within_thresholds() -> None:
    plan = plan_idle_cash(
        cash_balance=200.0,
        positions_value=800.0,
        current_treasury_value=0.0,
        policy=_policy(min_idle_pct=0.15, deploy_threshold_usd=100.0),
        # floor = 0.15 * 1000 = 150; excess = 50 < threshold -> hold
    )
    assert plan.action == "hold"
    assert plan.is_noop


def test_plan_holds_when_redeem_below_threshold() -> None:
    plan = plan_idle_cash(
        cash_balance=80.0,
        positions_value=20.0,
        current_treasury_value=20.0,
        policy=_policy(min_idle_pct=0.10, redeem_threshold_usd=50.0),
        # equity 120, floor 12. cash 80 > floor → excess 68 < deploy 100 → hold
    )
    assert plan.action == "hold"


def test_plan_holds_when_no_treasury_to_redeem() -> None:
    plan = plan_idle_cash(
        cash_balance=0.0,
        positions_value=900.0,
        current_treasury_value=0.0,
        policy=_policy(min_idle_pct=0.10, redeem_threshold_usd=10.0),
    )
    assert plan.action == "hold"
    assert "below redeem threshold" in plan.reason or "under redeem" in plan.reason


def test_plan_zero_equity_holds() -> None:
    plan = plan_idle_cash(
        cash_balance=0.0,
        positions_value=0.0,
        current_treasury_value=0.0,
        policy=_policy(),
    )
    assert plan.action == "hold"
    assert plan.amount_usd == 0


def test_plan_caps_redeem_at_treasury_value() -> None:
    plan = plan_idle_cash(
        cash_balance=10.0,
        positions_value=900.0,
        current_treasury_value=30.0,
        policy=_policy(min_idle_pct=0.20, redeem_threshold_usd=10.0),
        # floor = 0.20 * 940 = 188; shortfall 178; treasury only 30
    )
    assert plan.action == "redeem"
    assert plan.amount_usd == pytest.approx(30.0)
    assert plan.treasury_target == 0


def test_plan_writes_pre_post_snapshot() -> None:
    plan = plan_idle_cash(
        cash_balance=1000.0,
        positions_value=0.0,
        current_treasury_value=0.0,
        policy=_policy(),
    )
    assert plan.cash_before == 1000.0
    assert plan.treasury_before == 0.0
    assert plan.cash_target + plan.treasury_target == pytest.approx(1000.0)


# ── Yield estimator ──────────────────────────────────────────────


def test_estimate_yield_basic() -> None:
    assert estimate_annual_yield_usd(1000.0, apy=0.04) == pytest.approx(40.0)


def test_estimate_yield_partial_year() -> None:
    assert estimate_annual_yield_usd(1000.0, apy=0.04, days=180) == pytest.approx(
        1000.0 * 0.04 * 180 / 365, abs=0.01
    )


def test_estimate_yield_zero_when_no_treasury() -> None:
    assert estimate_annual_yield_usd(0.0) == 0.0


def test_estimate_yield_zero_apy() -> None:
    assert estimate_annual_yield_usd(1000.0, apy=0.0) == 0.0
