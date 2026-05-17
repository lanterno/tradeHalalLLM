"""Tests for the halal robo-advisor engine."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.web.robo_advisor import (
    CurrentAllocation,
    HalalAssetClass,
    Holding,
    RebalanceTrade,
    RiskProfile,
    TargetAllocation,
    compute_rebalance,
    compute_target_allocation,
    render_rebalance_plan,
    render_target_allocation,
)


def _holdings(weights: dict[HalalAssetClass, float]) -> CurrentAllocation:
    return CurrentAllocation(holdings=tuple(Holding(asset=k, weight=v) for k, v in weights.items()))


# ---------------------------------------------------------------------------
# TargetAllocation validation
# ---------------------------------------------------------------------------


def _full_weights(scale: float = 1.0) -> dict[HalalAssetClass, float]:
    """A complete, sum-to-1.0 weight dict."""

    base = {
        HalalAssetClass.HALAL_EQUITY: 0.60,
        HalalAssetClass.SUKUK: 0.25,
        HalalAssetClass.HALAL_COMMODITIES: 0.07,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.03,
    }
    return {k: v * scale for k, v in base.items()}


def test_target_allocation_accepts_valid_weights() -> None:
    a = TargetAllocation(weights=_full_weights())
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == 0.60


def test_target_allocation_rejects_missing_class() -> None:
    weights = _full_weights()
    del weights[HalalAssetClass.CASH]
    with pytest.raises(ValueError, match="missing asset class"):
        TargetAllocation(weights=weights)


def test_target_allocation_rejects_negative_weight() -> None:
    weights = _full_weights()
    weights[HalalAssetClass.CASH] = -0.01
    with pytest.raises(ValueError, match="must be in"):
        TargetAllocation(weights=weights)


def test_target_allocation_rejects_weight_above_1() -> None:
    weights = _full_weights()
    weights[HalalAssetClass.HALAL_EQUITY] = 1.01
    with pytest.raises(ValueError, match="must be in"):
        TargetAllocation(weights=weights)


def test_target_allocation_rejects_sum_below_tolerance() -> None:
    weights = _full_weights(scale=0.99)
    with pytest.raises(ValueError, match="sum to 1.0"):
        TargetAllocation(weights=weights)


def test_target_allocation_rejects_sum_above_tolerance() -> None:
    weights = _full_weights(scale=1.01)
    with pytest.raises(ValueError, match="sum to 1.0"):
        TargetAllocation(weights=weights)


def test_target_allocation_accepts_tiny_float_drift() -> None:
    """0.50 + 0.30 + 0.10 + 0.05 + 0.05 = 1.0000…1 should not reject."""

    weights = {
        HalalAssetClass.HALAL_EQUITY: 0.50,
        HalalAssetClass.SUKUK: 0.30,
        HalalAssetClass.HALAL_COMMODITIES: 0.10,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.05,
    }
    a = TargetAllocation(weights=weights)
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == 0.50


# ---------------------------------------------------------------------------
# Holding + CurrentAllocation validation
# ---------------------------------------------------------------------------


def test_holding_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="must be in"):
        Holding(asset=HalalAssetClass.CASH, weight=-0.01)


def test_holding_rejects_weight_above_1() -> None:
    with pytest.raises(ValueError, match="must be in"):
        Holding(asset=HalalAssetClass.CASH, weight=1.01)


def test_current_allocation_rejects_duplicate_asset() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CurrentAllocation(
            holdings=(
                Holding(asset=HalalAssetClass.CASH, weight=0.5),
                Holding(asset=HalalAssetClass.CASH, weight=0.5),
                # other classes missing too — but duplicate fires first
            )
        )


def test_current_allocation_rejects_missing_class() -> None:
    holdings = tuple(
        Holding(asset=k, weight=v)
        for k, v in _full_weights().items()
        if k is not HalalAssetClass.CASH
    )
    with pytest.raises(ValueError, match="missing asset class"):
        CurrentAllocation(holdings=holdings)


def test_current_allocation_rejects_sum_below_tolerance() -> None:
    weights = _full_weights(scale=0.95)
    with pytest.raises(ValueError, match="sum to 1.0"):
        _holdings(weights)


def test_current_allocation_weight_for_returns_field() -> None:
    c = _holdings(_full_weights())
    assert c.weight_for(HalalAssetClass.HALAL_EQUITY) == 0.60


# ---------------------------------------------------------------------------
# compute_target_allocation — glide path semantics
# ---------------------------------------------------------------------------


def test_far_horizon_uses_far_anchor_for_aggressive() -> None:
    """≥ 30 years → far-horizon anchor."""

    a = compute_target_allocation(profile=RiskProfile.AGGRESSIVE, years_to_target=40)
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == pytest.approx(0.80)


def test_near_horizon_uses_near_anchor_for_aggressive() -> None:
    """≤ 1 year → near-horizon anchor."""

    a = compute_target_allocation(profile=RiskProfile.AGGRESSIVE, years_to_target=0.5)
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == pytest.approx(0.30)


def test_zero_years_clamps_to_near_horizon() -> None:
    a = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=0)
    # MODERATE near-horizon equity is 0.20
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == pytest.approx(0.20)


def test_far_horizon_at_exactly_30_uses_far_anchor() -> None:
    a = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=30)
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == pytest.approx(0.60)


def test_near_horizon_at_exactly_1_uses_near_anchor() -> None:
    a = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=1)
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == pytest.approx(0.20)


def test_glide_path_interpolates_linearly() -> None:
    """At 15.5 years (midpoint), MODERATE equity = midpoint of 60% and 20%."""

    a = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=15.5)
    expected_eq = (0.60 + 0.20) / 2  # exact midpoint
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == pytest.approx(expected_eq, abs=0.01)


def test_glide_path_monotonic_equity_decreases() -> None:
    """Pin: longer horizon → more equity. Strict inequality."""

    short = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=2)
    medium = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=15)
    long = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=30)
    assert (
        short.weight_for(HalalAssetClass.HALAL_EQUITY)
        < medium.weight_for(HalalAssetClass.HALAL_EQUITY)
        < long.weight_for(HalalAssetClass.HALAL_EQUITY)
    )


def test_glide_path_monotonic_cash_increases() -> None:
    """Pin: longer horizon → less cash. As target approaches, cash share grows."""

    short = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=2)
    long = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=30)
    assert short.weight_for(HalalAssetClass.CASH) > long.weight_for(HalalAssetClass.CASH)


def test_aggressive_has_more_equity_than_conservative() -> None:
    """Pin: across all horizons, AGGRESSIVE > MODERATE > CONSERVATIVE for equity."""

    for years in (5, 15, 30):
        cons = compute_target_allocation(profile=RiskProfile.CONSERVATIVE, years_to_target=years)
        mod = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=years)
        agg = compute_target_allocation(profile=RiskProfile.AGGRESSIVE, years_to_target=years)
        cons_eq = cons.weight_for(HalalAssetClass.HALAL_EQUITY)
        mod_eq = mod.weight_for(HalalAssetClass.HALAL_EQUITY)
        agg_eq = agg.weight_for(HalalAssetClass.HALAL_EQUITY)
        assert cons_eq < mod_eq < agg_eq


def test_compute_target_rejects_negative_years() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=-1)


def test_glide_path_weights_always_sum_to_1() -> None:
    """Pin: at every horizon, weights sum to 1.0 within tolerance."""

    for profile in RiskProfile:
        for years in (0, 0.5, 1, 5, 10, 15, 20, 25, 30, 50):
            a = compute_target_allocation(profile=profile, years_to_target=years)
            total = sum(a.weights.values())
            assert 0.999 <= total <= 1.001, f"{profile} @ {years}y: {total}"


# ---------------------------------------------------------------------------
# Rebalance — drift threshold pin
# ---------------------------------------------------------------------------


def test_no_drift_returns_noop() -> None:
    """Pin: when current matches target exactly, no-op plan."""

    target = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=20)
    current = _holdings(target.weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    assert plan.is_noop is True
    assert plan.trades == ()


def test_below_threshold_drift_returns_noop() -> None:
    """Pin: 4.99% drift below 5% threshold → no-op."""

    target_weights = {
        HalalAssetClass.HALAL_EQUITY: 0.60,
        HalalAssetClass.SUKUK: 0.25,
        HalalAssetClass.HALAL_COMMODITIES: 0.07,
        HalalAssetClass.HALAL_REIT: 0.05,
        HalalAssetClass.CASH: 0.03,
    }
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    # shift 4% from equity → cash; max drift is 4%
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.04
    current_weights[HalalAssetClass.CASH] += 0.04
    current = _holdings(current_weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    assert plan.is_noop is True
    assert plan.max_drift_pct == pytest.approx(4.0)


def test_at_threshold_drift_triggers_rebalance() -> None:
    """Pin: exactly 5% drift triggers (boundary inclusive)."""

    target_weights = _full_weights()
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.05
    current_weights[HalalAssetClass.CASH] += 0.05
    current = _holdings(current_weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    assert plan.is_noop is False


def test_above_threshold_drift_triggers_rebalance() -> None:
    target_weights = _full_weights()
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.10
    current_weights[HalalAssetClass.CASH] += 0.10
    current = _holdings(current_weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    assert plan.is_noop is False
    assert plan.max_drift_pct == pytest.approx(10.0)


def test_rebalance_trades_have_correct_deltas() -> None:
    target_weights = _full_weights()
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.10
    current_weights[HalalAssetClass.CASH] += 0.10
    current = _holdings(current_weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    by_asset = {t.asset: t for t in plan.trades}
    eq = by_asset[HalalAssetClass.HALAL_EQUITY]
    assert eq.is_buy is True
    assert eq.delta == pytest.approx(0.10)
    cash = by_asset[HalalAssetClass.CASH]
    assert cash.is_sell is True
    assert cash.delta == pytest.approx(-0.10)


def test_custom_threshold_flows_through() -> None:
    """A 3% threshold catches a 4% drift that the default 5% wouldn't."""

    target_weights = _full_weights()
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.04
    current_weights[HalalAssetClass.CASH] += 0.04
    current = _holdings(current_weights)
    default_plan = compute_rebalance(user_id="user-1", current=current, target=target)
    assert default_plan.is_noop is True
    strict_plan = compute_rebalance(
        user_id="user-1", current=current, target=target, threshold_pct=3.0
    )
    assert strict_plan.is_noop is False


def test_rebalance_rejects_zero_threshold() -> None:
    target = TargetAllocation(weights=_full_weights())
    current = _holdings(_full_weights())
    with pytest.raises(ValueError, match="threshold_pct"):
        compute_rebalance(user_id="user-1", current=current, target=target, threshold_pct=0)


def test_rebalance_rejects_negative_threshold() -> None:
    target = TargetAllocation(weights=_full_weights())
    current = _holdings(_full_weights())
    with pytest.raises(ValueError, match="threshold_pct"):
        compute_rebalance(user_id="user-1", current=current, target=target, threshold_pct=-1.0)


def test_rebalance_rejects_empty_user_id() -> None:
    target = TargetAllocation(weights=_full_weights())
    current = _holdings(_full_weights())
    with pytest.raises(ValueError, match="user_id"):
        compute_rebalance(user_id="", current=current, target=target)


def test_rebalance_trade_buy_sell_neutral_classification() -> None:
    t_buy = RebalanceTrade(
        asset=HalalAssetClass.HALAL_EQUITY,
        current_weight=0.5,
        target_weight=0.6,
        delta=0.1,
    )
    t_sell = RebalanceTrade(
        asset=HalalAssetClass.HALAL_EQUITY,
        current_weight=0.6,
        target_weight=0.5,
        delta=-0.1,
    )
    t_neutral = RebalanceTrade(
        asset=HalalAssetClass.HALAL_EQUITY,
        current_weight=0.5,
        target_weight=0.5,
        delta=0.0,
    )
    assert t_buy.is_buy is True and t_buy.is_sell is False
    assert t_sell.is_sell is True and t_sell.is_buy is False
    assert t_neutral.is_buy is False and t_neutral.is_sell is False


def test_rebalance_max_drift_correctly_computed() -> None:
    """Pin: max_drift_pct equals max |delta| × 100 across asset classes."""

    target_weights = _full_weights()
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    # 8% drift in equity, 3% drift in sukuk, etc.
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.08
    current_weights[HalalAssetClass.SUKUK] -= 0.03
    current_weights[HalalAssetClass.CASH] += 0.11
    current = _holdings(current_weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    # max drift is in cash (+11%)
    assert plan.max_drift_pct == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# HalalAssetClass — closed-set guarantee
# ---------------------------------------------------------------------------


def test_halal_asset_class_set_is_exactly_five() -> None:
    """Pin: the closed set of halal asset classes — adding a new one is
    a code-review change (the structural friction that prevents
    accidentally allocating to non-halal categories at runtime)."""

    classes = list(HalalAssetClass)
    assert len(classes) == 5
    assert HalalAssetClass.HALAL_EQUITY in classes
    assert HalalAssetClass.SUKUK in classes
    assert HalalAssetClass.HALAL_COMMODITIES in classes
    assert HalalAssetClass.HALAL_REIT in classes
    assert HalalAssetClass.CASH in classes


def test_no_conventional_bond_class_in_enum() -> None:
    """Pin: the enum has no conventional-bond class. A future
    contributor cannot accidentally allocate to bonds — the type
    system rejects it."""

    values = {c.value for c in HalalAssetClass}
    assert "bonds" not in values
    assert "conventional_bonds" not in values
    assert "fixed_income" not in values


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_target_allocation_is_frozen() -> None:
    a = TargetAllocation(weights=_full_weights())
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.weights = {}  # type: ignore[misc]


def test_holding_is_frozen() -> None:
    h = Holding(asset=HalalAssetClass.CASH, weight=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.weight = 0.5  # type: ignore[misc]


def test_current_allocation_is_frozen() -> None:
    c = _holdings(_full_weights())
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.holdings = ()  # type: ignore[misc]


def test_rebalance_plan_is_frozen() -> None:
    target = TargetAllocation(weights=_full_weights())
    current = _holdings(_full_weights())
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.is_noop = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB serialisation
# ---------------------------------------------------------------------------


def test_risk_profile_string_values() -> None:
    assert RiskProfile.CONSERVATIVE.value == "conservative"
    assert RiskProfile.MODERATE.value == "moderate"
    assert RiskProfile.AGGRESSIVE.value == "aggressive"


def test_halal_asset_class_string_values() -> None:
    assert HalalAssetClass.HALAL_EQUITY.value == "halal_equity"
    assert HalalAssetClass.SUKUK.value == "sukuk"
    assert HalalAssetClass.HALAL_COMMODITIES.value == "halal_commodities"
    assert HalalAssetClass.HALAL_REIT.value == "halal_reit"
    assert HalalAssetClass.CASH.value == "cash"


# ---------------------------------------------------------------------------
# Render output — pinned no-USD contract
# ---------------------------------------------------------------------------


def test_render_target_allocation_includes_profile_and_horizon() -> None:
    a = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=20)
    text = render_target_allocation(profile=RiskProfile.MODERATE, years_to_target=20, allocation=a)
    assert "moderate" in text
    assert "⚖️" in text
    assert "20.0y horizon" in text
    assert "halal_equity" in text


def test_render_rebalance_plan_noop() -> None:
    target = TargetAllocation(weights=_full_weights())
    current = _holdings(_full_weights())
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    text = render_rebalance_plan(plan)
    assert "♻️" in text
    assert "no rebalance needed" in text
    assert "user-1" in text


def test_render_rebalance_plan_with_trades() -> None:
    target_weights = _full_weights()
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.10
    current_weights[HalalAssetClass.CASH] += 0.10
    current = _holdings(current_weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    text = render_rebalance_plan(plan)
    assert "rebalance required" in text
    assert "↑" in text
    assert "↓" in text
    assert "halal_equity" in text


def test_render_no_usd_in_output() -> None:
    """Pin no-USD contract: render shows weight % only, never $ amounts."""

    a = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=20)
    text_a = render_target_allocation(
        profile=RiskProfile.MODERATE, years_to_target=20, allocation=a
    )
    assert "$" not in text_a

    target_weights = _full_weights()
    target = TargetAllocation(weights=target_weights)
    current_weights = dict(target_weights)
    current_weights[HalalAssetClass.HALAL_EQUITY] -= 0.10
    current_weights[HalalAssetClass.CASH] += 0.10
    current = _holdings(current_weights)
    plan = compute_rebalance(user_id="user-1", current=current, target=target)
    text_p = render_rebalance_plan(plan)
    assert "$" not in text_p
    assert "USD" not in text_p


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_typical_user_30y_to_retirement_aggressive() -> None:
    """A 30-year-to-retirement aggressive user should get the
    far-horizon high-equity allocation."""

    a = compute_target_allocation(profile=RiskProfile.AGGRESSIVE, years_to_target=30)
    assert a.weight_for(HalalAssetClass.HALAL_EQUITY) == pytest.approx(0.80)
    assert a.weight_for(HalalAssetClass.SUKUK) == pytest.approx(0.10)


def test_typical_user_5y_to_retirement_conservative() -> None:
    """A 5-year-to-retirement conservative user should be majority sukuk + cash."""

    a = compute_target_allocation(profile=RiskProfile.CONSERVATIVE, years_to_target=5)
    sukuk_plus_cash = a.weight_for(HalalAssetClass.SUKUK) + a.weight_for(HalalAssetClass.CASH)
    equity = a.weight_for(HalalAssetClass.HALAL_EQUITY)
    assert sukuk_plus_cash > equity


def test_full_rebalance_lifecycle() -> None:
    """A user opens with 100% cash; over time rebalances toward
    target as funds get deployed."""

    target = compute_target_allocation(profile=RiskProfile.MODERATE, years_to_target=20)
    initial_holdings = _holdings(
        {
            HalalAssetClass.HALAL_EQUITY: 0.0,
            HalalAssetClass.SUKUK: 0.0,
            HalalAssetClass.HALAL_COMMODITIES: 0.0,
            HalalAssetClass.HALAL_REIT: 0.0,
            HalalAssetClass.CASH: 1.0,
        }
    )
    plan = compute_rebalance(user_id="user-1", current=initial_holdings, target=target)
    assert plan.is_noop is False
    # the plan should buy every non-cash asset and sell cash
    by_asset = {t.asset: t for t in plan.trades}
    assert by_asset[HalalAssetClass.CASH].is_sell is True
    assert by_asset[HalalAssetClass.HALAL_EQUITY].is_buy is True
    assert by_asset[HalalAssetClass.SUKUK].is_buy is True
