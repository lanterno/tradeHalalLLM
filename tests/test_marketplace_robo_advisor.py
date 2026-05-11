"""Tests for marketplace/robo_advisor.py — Round-5 Wave 21.E.

Distinct from `tests/test_robo_advisor.py` which exercises the
web-side robo-advisor surface.
"""

from __future__ import annotations

import pytest

from halal_trader.education.risk_assessment import RiskProfile
from halal_trader.marketplace.robo_advisor import (
    AllocationBand,
    FeeBracket,
    ModelPortfolio,
    TimeHorizon,
    annual_fee_for,
    band_for,
    build_plan,
    map_profile_to_model,
    needs_rebalance,
    render_plan,
)

# --- AllocationBand validation -----------------------------


def test_band_valid():
    b = AllocationBand(
        equity_min=0.4,
        equity_max=0.6,
        sukuk_min=0.3,
        sukuk_max=0.5,
        cash_min=0.05,
        cash_max=0.15,
    )
    assert b.equity_min == 0.4


def test_band_inverted_rejected():
    with pytest.raises(ValueError):
        AllocationBand(
            equity_min=0.6,
            equity_max=0.4,
            sukuk_min=0.3,
            sukuk_max=0.5,
            cash_min=0.0,
            cash_max=0.1,
        )


def test_band_infeasible_rejected():
    with pytest.raises(ValueError):
        AllocationBand(
            equity_min=0.5,
            equity_max=0.6,
            sukuk_min=0.5,
            sukuk_max=0.6,
            cash_min=0.1,
            cash_max=0.2,
        )


# --- band_for ---------------------------------------------


def test_band_for_each_model():
    for m in ModelPortfolio:
        b = band_for(m)
        assert b.equity_max >= b.equity_min


def test_defensive_low_equity():
    b = band_for(ModelPortfolio.DEFENSIVE)
    assert b.equity_max <= 0.30


def test_aggressive_high_equity():
    b = band_for(ModelPortfolio.AGGRESSIVE_GROWTH)
    assert b.equity_min >= 0.85


# --- map_profile_to_model --------------------------------


def test_map_conservative_short_defensive():
    assert (
        map_profile_to_model(RiskProfile.CONSERVATIVE, TimeHorizon.SHORT)
        is ModelPortfolio.DEFENSIVE
    )


def test_map_aggressive_long_growth_aggressive():
    assert (
        map_profile_to_model(RiskProfile.AGGRESSIVE, TimeHorizon.LONG)
        is ModelPortfolio.AGGRESSIVE_GROWTH
    )


def test_map_balanced_medium_balanced():
    assert map_profile_to_model(RiskProfile.BALANCED, TimeHorizon.MEDIUM) is ModelPortfolio.BALANCED


def test_map_conservative_long_balanced():
    """Pin: even a CONSERVATIVE investor goes BALANCED on LONG horizon."""
    assert (
        map_profile_to_model(RiskProfile.CONSERVATIVE, TimeHorizon.LONG) is ModelPortfolio.BALANCED
    )


def test_map_override():
    overrides = {
        (RiskProfile.CONSERVATIVE, TimeHorizon.SHORT): ModelPortfolio.BALANCED,
    }
    assert (
        map_profile_to_model(RiskProfile.CONSERVATIVE, TimeHorizon.SHORT, overrides=overrides)
        is ModelPortfolio.BALANCED
    )


def test_map_completeness_all_combinations():
    for p in RiskProfile:
        for h in TimeHorizon:
            map_profile_to_model(p, h)


# --- FeeBracket validation -------------------------------


def test_fee_bracket_valid():
    b = FeeBracket(aum_min_usd=0, aum_max_usd=50_000.0, annual_fee_pct=0.005)
    assert b.annual_fee_pct == 0.005


def test_fee_bracket_negative_min_rejected():
    with pytest.raises(ValueError):
        FeeBracket(aum_min_usd=-1, aum_max_usd=1000, annual_fee_pct=0.005)


def test_fee_bracket_inverted_rejected():
    with pytest.raises(ValueError):
        FeeBracket(aum_min_usd=1000, aum_max_usd=500, annual_fee_pct=0.005)


def test_fee_bracket_excessive_pct_rejected():
    """Pin: fees ≥ 5%/yr are not Wakalah-style."""
    with pytest.raises(ValueError):
        FeeBracket(aum_min_usd=0, aum_max_usd=1000, annual_fee_pct=0.10)


# --- annual_fee_for ------------------------------------


def test_annual_fee_zero_aum():
    assert annual_fee_for(0) == 0.0


def test_annual_fee_single_bracket():
    fee = annual_fee_for(25_000.0)
    assert fee == pytest.approx(125.0)


def test_annual_fee_crosses_brackets():
    fee = annual_fee_for(100_000.0)
    assert fee == pytest.approx(425.0)


def test_annual_fee_top_bracket():
    fee = annual_fee_for(1_000_000.0)
    assert fee == pytest.approx(2825.0)


def test_annual_fee_negative_aum_rejected():
    with pytest.raises(ValueError):
        annual_fee_for(-1.0)


def test_annual_fee_empty_schedule_rejected():
    with pytest.raises(ValueError):
        annual_fee_for(100.0, schedule=())


def test_annual_fee_custom_schedule():
    schedule = (FeeBracket(aum_min_usd=0, aum_max_usd=None, annual_fee_pct=0.01),)
    fee = annual_fee_for(10_000.0, schedule=schedule)
    assert fee == pytest.approx(100.0)


# --- build_plan ---------------------------------------


def test_build_plan_basic():
    plan = build_plan(
        plan_id="P1",
        user_id="alice",
        profile=RiskProfile.BALANCED,
        horizon=TimeHorizon.MEDIUM,
        aum_usd=100_000.0,
    )
    assert plan.model is ModelPortfolio.BALANCED
    assert plan.annual_fee_usd == pytest.approx(425.0)


def test_build_plan_empty_id_rejected():
    with pytest.raises(ValueError):
        build_plan(
            plan_id="",
            user_id="alice",
            profile=RiskProfile.BALANCED,
            horizon=TimeHorizon.MEDIUM,
            aum_usd=10_000.0,
        )


def test_build_plan_negative_aum_rejected():
    with pytest.raises(ValueError):
        build_plan(
            plan_id="P1",
            user_id="alice",
            profile=RiskProfile.BALANCED,
            horizon=TimeHorizon.MEDIUM,
            aum_usd=-1.0,
        )


def test_build_plan_invalid_drift_rejected():
    with pytest.raises(ValueError):
        build_plan(
            plan_id="P1",
            user_id="alice",
            profile=RiskProfile.BALANCED,
            horizon=TimeHorizon.MEDIUM,
            aum_usd=10_000.0,
            drift_threshold_pct=0,
        )


# --- needs_rebalance ---------------------------------


def test_no_rebalance_when_within_bands():
    plan = build_plan(
        plan_id="P1",
        user_id="alice",
        profile=RiskProfile.BALANCED,
        horizon=TimeHorizon.MEDIUM,
        aum_usd=100_000.0,
    )
    assert not needs_rebalance(
        plan,
        actual_equity_pct=0.50,
        actual_sukuk_pct=0.40,
        actual_cash_pct=0.10,
    )


def test_rebalance_when_equity_above_band():
    plan = build_plan(
        plan_id="P1",
        user_id="alice",
        profile=RiskProfile.BALANCED,
        horizon=TimeHorizon.MEDIUM,
        aum_usd=100_000.0,
        drift_threshold_pct=0.05,
    )
    assert needs_rebalance(
        plan,
        actual_equity_pct=0.70,
        actual_sukuk_pct=0.25,
        actual_cash_pct=0.05,
    )


def test_rebalance_within_threshold_buffer_does_not_fire():
    plan = build_plan(
        plan_id="P1",
        user_id="alice",
        profile=RiskProfile.BALANCED,
        horizon=TimeHorizon.MEDIUM,
        aum_usd=100_000.0,
        drift_threshold_pct=0.05,
    )
    assert not needs_rebalance(
        plan,
        actual_equity_pct=0.64,
        actual_sukuk_pct=0.30,
        actual_cash_pct=0.06,
    )


def test_rebalance_invalid_percentages_rejected():
    plan = build_plan(
        plan_id="P1",
        user_id="alice",
        profile=RiskProfile.BALANCED,
        horizon=TimeHorizon.MEDIUM,
        aum_usd=100_000.0,
    )
    with pytest.raises(ValueError):
        needs_rebalance(
            plan,
            actual_equity_pct=1.5,
            actual_sukuk_pct=0.3,
            actual_cash_pct=0.1,
        )


# --- Render -----------------------------------------


def test_render_plan_no_secret_leak():
    plan = build_plan(
        plan_id="P1",
        user_id="alice@example.com",
        profile=RiskProfile.BALANCED,
        horizon=TimeHorizon.MEDIUM,
        aum_usd=100_000.0,
    )
    out = render_plan(plan)
    assert "alice@example.com" not in out


def test_render_plan_model_emoji():
    plan = build_plan(
        plan_id="P1",
        user_id="alice",
        profile=RiskProfile.AGGRESSIVE,
        horizon=TimeHorizon.LONG,
        aum_usd=100_000.0,
    )
    out = render_plan(plan)
    assert "🚀" in out


def test_render_plan_fee_format():
    plan = build_plan(
        plan_id="P1",
        user_id="alice",
        profile=RiskProfile.BALANCED,
        horizon=TimeHorizon.MEDIUM,
        aum_usd=100_000.0,
    )
    out = render_plan(plan)
    assert "bps" in out
