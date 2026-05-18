"""Tests for ml/liquidity_risk.py — Round-5 Wave 13.F."""

from __future__ import annotations

import pytest

from halal_trader.ml.liquidity_risk import (
    LiquidityAssessment,
    LiquidityInputs,
    LiquidityPolicy,
    LiquidityTier,
    assess_liquidity,
    liquidity_adjusted_var,
    render_assessment,
)


def _inputs(**overrides) -> LiquidityInputs:
    base = {
        "symbol": "AAPL",
        "bid_ask_spread_bps": 2.0,
        "average_daily_volume": 100_000_000.0,
        "market_depth_at_top": 50_000.0,
        "position_size": 100_000.0,  # 0.1% of ADV
    }
    base.update(overrides)
    return LiquidityInputs(**base)


# --- Validation ---------------------------------------------


def test_tier_string_values():
    assert LiquidityTier.DEEP.value == "deep"
    assert LiquidityTier.NORMAL.value == "normal"
    assert LiquidityTier.THIN.value == "thin"
    assert LiquidityTier.ILLIQUID.value == "illiquid"


def test_inputs_empty_symbol_rejected():
    with pytest.raises(ValueError):
        _inputs(symbol="")


def test_inputs_negative_spread_rejected():
    with pytest.raises(ValueError):
        _inputs(bid_ask_spread_bps=-1)


def test_inputs_negative_position_rejected():
    with pytest.raises(ValueError):
        _inputs(position_size=-1)


def test_default_policy():
    p = LiquidityPolicy()
    assert p.deep_spread_max_bps == 5.0


def test_policy_unsorted_spread_rejected():
    with pytest.raises(ValueError):
        LiquidityPolicy(deep_spread_max_bps=50.0, normal_spread_max_bps=10.0)


def test_policy_unsorted_pct_rejected():
    with pytest.raises(ValueError):
        LiquidityPolicy(deep_pos_pct_adv=0.10, normal_pos_pct_adv=0.05)


def test_assessment_score_outside_unit_rejected():
    with pytest.raises(ValueError):
        LiquidityAssessment(
            symbol="A",
            score=1.5,
            tier=LiquidityTier.DEEP,
            position_pct_of_adv=0.001,
            estimated_liquidation_cost_pct=0.001,
        )


# --- Tier laddering -----------------------------------------


def test_deep_for_tight_spread_small_pos():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=1.0, position_size=10000))
    assert a.tier is LiquidityTier.DEEP


def test_normal_tier():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=15.0, position_size=2_000_000))
    assert a.tier is LiquidityTier.NORMAL


def test_thin_tier():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=50.0, position_size=10_000_000))
    assert a.tier is LiquidityTier.THIN


def test_illiquid_for_huge_position():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=200.0, position_size=30_000_000))
    assert a.tier is LiquidityTier.ILLIQUID


def test_zero_adv_treated_as_illiquid():
    a = assess_liquidity(_inputs(average_daily_volume=0.0, position_size=1000))
    assert a.tier is LiquidityTier.ILLIQUID
    assert a.position_pct_of_adv > 1e6  # treated as ~infinite


# --- Score --------------------------------------------------


def test_deep_high_score():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=1.0, position_size=10000))
    assert a.score > 0.95


def test_illiquid_low_score():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=200.0, position_size=30_000_000))
    assert a.score < 0.10


# --- Liquidation cost --------------------------------------


def test_liq_cost_includes_half_spread():
    """At 10 bps spread, half-spread = 5 bps = 0.0005."""
    a = assess_liquidity(_inputs(bid_ask_spread_bps=10.0, position_size=10000))
    assert a.estimated_liquidation_cost_pct >= 0.0005


def test_liq_cost_grows_with_position_size():
    small = assess_liquidity(_inputs(position_size=10000))
    large = assess_liquidity(_inputs(position_size=10_000_000))
    assert large.estimated_liquidation_cost_pct > small.estimated_liquidation_cost_pct


def test_liq_cost_capped_for_illiquid():
    a = assess_liquidity(_inputs(average_daily_volume=0))
    assert a.estimated_liquidation_cost_pct <= 0.30


# --- Liquidity-adjusted VaR -------------------------------


def test_liquidity_adjusted_var_inflates_base():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=200.0, position_size=20_000_000))
    base = 10000.0
    adjusted = liquidity_adjusted_var(base, a)
    assert adjusted > base


def test_liquidity_adjusted_var_deep_minimal_inflation():
    a = assess_liquidity(_inputs(bid_ask_spread_bps=1.0, position_size=10000))
    base = 10000.0
    adjusted = liquidity_adjusted_var(base, a)
    # Tight spread + small position → minimal liquidation cost → ~base
    assert adjusted < base * 1.01


def test_liquidity_adjusted_var_negative_base_rejected():
    a = assess_liquidity(_inputs())
    with pytest.raises(ValueError):
        liquidity_adjusted_var(-1.0, a)


# --- Render -----------------------------------------------


def test_render_includes_summary():
    a = assess_liquidity(_inputs())
    out = render_assessment(a)
    assert "AAPL" in out
    assert "liquidity" in out


def test_render_no_secret_leak():
    a = assess_liquidity(_inputs())
    out = render_assessment(a)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ------------------------------------------------


def test_e2e_thin_name_inflates_var():
    """A 1% of ADV position in a 50 bps spread name has noticeable liq cost."""
    a = assess_liquidity(_inputs(bid_ask_spread_bps=50.0, position_size=1_000_000))
    adjusted = liquidity_adjusted_var(10000.0, a)
    assert adjusted > 10000


def test_replay_consistency():
    a = assess_liquidity(_inputs())
    b = assess_liquidity(_inputs())
    assert a == b
