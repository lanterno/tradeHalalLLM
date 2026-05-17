"""Tests for halal/hedge_basket.py — Round-5 Wave 13.C."""

from __future__ import annotations

import pytest

from halal_trader.halal.hedge_basket import (
    BasketAllocation,
    BasketPolicy,
    BasketWeighting,
    HedgeAsset,
    compose,
    render_basket,
)

# --- Validation -------------------------------------------------


def test_hedge_asset_string_values():
    assert HedgeAsset.GOLD.value == "gold"
    assert HedgeAsset.SILVER.value == "silver"
    assert HedgeAsset.SUKUK.value == "sukuk"
    assert HedgeAsset.GOLD_BACKED_STABLECOIN.value == "gold_backed_stablecoin"
    assert HedgeAsset.HALAL_CASH.value == "halal_cash"


def test_weighting_string_values():
    assert BasketWeighting.EQUAL.value == "equal"
    assert BasketWeighting.RISK_PARITY.value == "risk_parity"
    assert BasketWeighting.CUSTOM.value == "custom"


def test_default_policy():
    p = BasketPolicy()
    assert p.weighting is BasketWeighting.RISK_PARITY
    assert p.hedge_ratio == 0.30


def test_policy_zero_ratio_rejected():
    with pytest.raises(ValueError):
        BasketPolicy(hedge_ratio=0.0)


def test_policy_above_one_ratio_rejected():
    with pytest.raises(ValueError):
        BasketPolicy(hedge_ratio=1.5)


def test_policy_custom_without_weights_rejected():
    with pytest.raises(ValueError):
        BasketPolicy(weighting=BasketWeighting.CUSTOM)


def test_policy_custom_unnormalised_rejected():
    with pytest.raises(ValueError):
        BasketPolicy(
            weighting=BasketWeighting.CUSTOM,
            custom_weights={HedgeAsset.GOLD: 0.5, HedgeAsset.SILVER: 0.3},
        )


def test_policy_custom_negative_rejected():
    with pytest.raises(ValueError):
        BasketPolicy(
            weighting=BasketWeighting.CUSTOM,
            custom_weights={HedgeAsset.GOLD: -0.5, HedgeAsset.SILVER: 1.5},
        )


def test_allocation_weight_outside_unit_rejected():
    with pytest.raises(ValueError):
        BasketAllocation(asset=HedgeAsset.GOLD, weight=1.5, notional=100)


def test_allocation_negative_notional_rejected():
    with pytest.raises(ValueError):
        BasketAllocation(asset=HedgeAsset.GOLD, weight=0.5, notional=-1)


# --- Compose ---------------------------------------------------


def test_compose_empty_assets():
    basket = compose(100000.0, assets=[])
    assert basket.allocations == ()
    assert basket.hedge_notional == 0


def test_compose_negative_portfolio_rejected():
    with pytest.raises(ValueError):
        compose(-1.0, assets=[HedgeAsset.GOLD])


def test_compose_equal_weighting():
    basket = compose(
        100000.0,
        assets=[HedgeAsset.GOLD, HedgeAsset.SUKUK],
        policy=BasketPolicy(weighting=BasketWeighting.EQUAL, hedge_ratio=0.30),
    )
    assert all(a.weight == pytest.approx(0.5) for a in basket.allocations)
    assert basket.hedge_notional == 30000.0


def test_compose_risk_parity_lower_vol_higher_weight():
    """SUKUK has lower vol than GOLD → SUKUK gets larger weight under risk parity."""
    basket = compose(
        100000.0,
        assets=[HedgeAsset.GOLD, HedgeAsset.SUKUK],
        policy=BasketPolicy(weighting=BasketWeighting.RISK_PARITY),
    )
    by_asset = {a.asset: a for a in basket.allocations}
    assert by_asset[HedgeAsset.SUKUK].weight > by_asset[HedgeAsset.GOLD].weight


def test_compose_custom_weighting():
    basket = compose(
        100000.0,
        assets=[HedgeAsset.GOLD, HedgeAsset.SUKUK],
        policy=BasketPolicy(
            weighting=BasketWeighting.CUSTOM,
            hedge_ratio=0.40,
            custom_weights={HedgeAsset.GOLD: 0.7, HedgeAsset.SUKUK: 0.3},
        ),
    )
    by_asset = {a.asset: a for a in basket.allocations}
    assert by_asset[HedgeAsset.GOLD].weight == pytest.approx(0.7)
    assert by_asset[HedgeAsset.SUKUK].weight == pytest.approx(0.3)
    assert basket.hedge_notional == 40000.0


def test_compose_zero_portfolio_zero_basket():
    basket = compose(0.0, assets=[HedgeAsset.GOLD])
    assert basket.hedge_notional == 0


def test_compose_total_weight_sums_to_one():
    basket = compose(
        100000.0,
        assets=[HedgeAsset.GOLD, HedgeAsset.SILVER, HedgeAsset.SUKUK, HedgeAsset.HALAL_CASH],
    )
    total = sum(a.weight for a in basket.allocations)
    assert total == pytest.approx(1.0)


def test_compose_allocation_notional_proportional():
    basket = compose(
        100000.0,
        assets=[HedgeAsset.GOLD, HedgeAsset.SUKUK],
        policy=BasketPolicy(weighting=BasketWeighting.EQUAL, hedge_ratio=0.20),
    )
    total_notional = sum(a.notional for a in basket.allocations)
    assert total_notional == pytest.approx(20000.0)


# --- Render ---------------------------------------------------


def test_render_includes_summary():
    basket = compose(100000.0, assets=[HedgeAsset.GOLD, HedgeAsset.SUKUK])
    out = render_basket(basket)
    assert "Hedge basket" in out
    assert "$100000" in out


def test_render_lists_assets():
    basket = compose(100000.0, assets=[HedgeAsset.GOLD, HedgeAsset.SUKUK])
    out = render_basket(basket)
    assert "gold" in out
    assert "sukuk" in out


def test_render_no_secret_leak():
    basket = compose(100000.0, assets=[HedgeAsset.GOLD])
    out = render_basket(basket)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ----------------------------------------------------


def test_e2e_balanced_defensive_basket():
    """Build a 5-asset defensive basket targeting 30% of $1M portfolio."""
    basket = compose(
        1000000.0,
        assets=[
            HedgeAsset.GOLD,
            HedgeAsset.SILVER,
            HedgeAsset.SUKUK,
            HedgeAsset.GOLD_BACKED_STABLECOIN,
            HedgeAsset.HALAL_CASH,
        ],
    )
    assert basket.hedge_notional == 300000.0
    assert len(basket.allocations) == 5


def test_replay_consistency():
    a = compose(100000.0, assets=[HedgeAsset.GOLD])
    b = compose(100000.0, assets=[HedgeAsset.GOLD])
    assert a == b
