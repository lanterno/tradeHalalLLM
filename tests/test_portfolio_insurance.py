"""Tests for halal/portfolio_insurance.py — Round-5 Wave 13.B."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.portfolio_insurance import (
    HedgeMode,
    HedgePolicy,
    PortfolioPosition,
    compose_hedge,
    render_plan,
)


def _pos(
    symbol: str = "AAPL",
    qty: float = 100.0,
    price: float = 200.0,
) -> PortfolioPosition:
    return PortfolioPosition(symbol=symbol, quantity=qty, market_price=price)


# --- Validation ----------------------------------------------------------


def test_hedge_mode_string_values():
    assert HedgeMode.FULL_FLOOR.value == "full_floor"
    assert HedgeMode.PARTIAL_FLOOR.value == "partial_floor"
    assert HedgeMode.TAIL_ONLY.value == "tail_only"


def test_default_policy():
    p = HedgePolicy()
    assert p.mode is HedgeMode.PARTIAL_FLOOR
    assert p.floor_pct == 0.90
    assert p.tail_threshold_pct == 0.80
    assert p.coverage_ratio == 1.0


def test_policy_floor_zero_rejected():
    with pytest.raises(ValueError):
        HedgePolicy(floor_pct=0.0)


def test_policy_floor_one_rejected():
    with pytest.raises(ValueError):
        HedgePolicy(floor_pct=1.0)


def test_policy_tail_geq_floor_rejected():
    with pytest.raises(ValueError):
        HedgePolicy(floor_pct=0.80, tail_threshold_pct=0.80)


def test_policy_zero_coverage_rejected():
    with pytest.raises(ValueError):
        HedgePolicy(coverage_ratio=0.0)


def test_policy_above_one_coverage_rejected():
    with pytest.raises(ValueError):
        HedgePolicy(coverage_ratio=1.5)


def test_policy_zero_term_rejected():
    with pytest.raises(ValueError):
        HedgePolicy(hedge_term_days=0)


def test_position_empty_symbol_rejected():
    with pytest.raises(ValueError):
        PortfolioPosition(symbol="", quantity=10, market_price=100)


def test_position_zero_qty_rejected():
    with pytest.raises(ValueError):
        PortfolioPosition(symbol="A", quantity=0, market_price=100)


def test_position_zero_price_rejected():
    with pytest.raises(ValueError):
        PortfolioPosition(symbol="A", quantity=10, market_price=0)


def test_position_market_value():
    p = _pos(qty=100, price=200)
    assert p.market_value == 20000


# --- Compose hedge -------------------------------------------------------


def test_empty_portfolio_returns_empty_plan():
    plan = compose_hedge(
        [], promisor="Bot", counterparty="Counterparty", today=date(2026, 5, 1)
    )
    assert plan.waads == ()
    assert plan.portfolio_value == 0


def test_compose_partial_floor_one_position():
    positions = [_pos(qty=100, price=200)]  # value $20k, floor $18k
    plan = compose_hedge(
        positions, promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    assert len(plan.waads) == 1
    assert plan.portfolio_value == 20000
    assert plan.floor_value == 18000
    # Strike at 90% of $200 = $180
    assert plan.waads[0].strike_price == 180.0


def test_compose_full_floor_uses_floor_strike():
    plan = compose_hedge(
        [_pos(price=100)],
        promisor="Bot",
        counterparty="C",
        today=date(2026, 5, 1),
        policy=HedgePolicy(mode=HedgeMode.FULL_FLOOR, floor_pct=0.95),
    )
    assert plan.waads[0].strike_price == 95.0


def test_compose_tail_only_uses_tail_strike():
    plan = compose_hedge(
        [_pos(price=100)],
        promisor="Bot",
        counterparty="C",
        today=date(2026, 5, 1),
        policy=HedgePolicy(
            mode=HedgeMode.TAIL_ONLY, floor_pct=0.90, tail_threshold_pct=0.75
        ),
    )
    assert plan.waads[0].strike_price == 75.0


def test_compose_coverage_ratio_scales_qty():
    plan = compose_hedge(
        [_pos(qty=100, price=200)],
        promisor="Bot",
        counterparty="C",
        today=date(2026, 5, 1),
        policy=HedgePolicy(coverage_ratio=0.5),
    )
    assert plan.waads[0].quantity == 50.0


def test_compose_multiple_positions():
    positions = [_pos(symbol="A", qty=100, price=200), _pos(symbol="B", qty=50, price=300)]
    plan = compose_hedge(
        positions, promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    assert len(plan.waads) == 2
    assert plan.portfolio_value == 100 * 200 + 50 * 300


def test_compose_all_waads_valid_when_inputs_clean():
    plan = compose_hedge(
        [_pos()], promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    assert plan.all_valid()


def test_compose_expected_payoff_at_floor_positive():
    """Partial-floor plan should yield positive payoff at the tail threshold."""
    plan = compose_hedge(
        [_pos(qty=100, price=200)],
        promisor="Bot",
        counterparty="C",
        today=date(2026, 5, 1),
    )
    assert plan.expected_payoff_at_floor >= 0


# --- Render -------------------------------------------------------------


def test_render_plan_includes_summary():
    plan = compose_hedge(
        [_pos()], promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    out = render_plan(plan)
    assert "Hedge plan" in out
    assert "AAPL" in out


def test_render_plan_uses_shield_emoji_when_valid():
    plan = compose_hedge(
        [_pos()], promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    out = render_plan(plan)
    assert "🛡️" in out


def test_render_no_secret_leak():
    plan = compose_hedge(
        [_pos()], promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    out = render_plan(plan)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------------------


def test_e2e_diversified_portfolio_partial_floor_hedge():
    """Hedge a 3-position portfolio at 90% floor."""
    positions = [
        _pos(symbol="AAPL", qty=100, price=200),
        _pos(symbol="MSFT", qty=50, price=400),
        _pos(symbol="GOOGL", qty=30, price=150),
    ]
    plan = compose_hedge(
        positions,
        promisor="Bot",
        counterparty="DerivCounterparty",
        today=date(2026, 5, 1),
    )
    assert plan.all_valid()
    assert len(plan.waads) == 3
    assert plan.portfolio_value == 100 * 200 + 50 * 400 + 30 * 150


def test_replay_consistency():
    a = compose_hedge(
        [_pos()], promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    b = compose_hedge(
        [_pos()], promisor="Bot", counterparty="C", today=date(2026, 5, 1)
    )
    assert a == b
