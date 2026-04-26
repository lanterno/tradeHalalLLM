"""Vol-aware slippage + confidence-weighted quantity tests."""

import pytest

from halal_trader.crypto.slippage import (
    SlippageInputs,
    confidence_weighted_quantity,
    estimate_fill,
)


def _inputs(**overrides):
    base = dict(
        side="buy",
        notional_usd=1000.0,
        atr_pct=0.02,
        atr_baseline=0.02,
        recent_quote_volume_usd=1_000_000.0,
        baseline_slippage_pct=0.0005,
    )
    base.update(overrides)
    return SlippageInputs(**base)


def test_baseline_buy_fill_above_price():
    r = estimate_fill(price=100.0, inputs=_inputs())
    assert r.fill_price > 100
    assert r.slippage_pct >= 0.0005


def test_baseline_sell_fill_below_price():
    r = estimate_fill(price=100.0, inputs=_inputs(side="sell"))
    assert r.fill_price < 100


def test_high_vol_multiplies_baseline_slippage():
    """ATR 4% vs baseline 2% → vol_multiplier 2× → baseline slippage doubles."""
    r = estimate_fill(price=100.0, inputs=_inputs(atr_pct=0.04, recent_quote_volume_usd=0))
    assert r.components["vol_multiplier"] == pytest.approx(2.0)
    assert r.slippage_pct == pytest.approx(0.001, rel=1e-6)


def test_low_vol_clamped_at_floor():
    r = estimate_fill(price=100.0, inputs=_inputs(atr_pct=0.001, recent_quote_volume_usd=0))
    assert r.components["vol_multiplier"] == pytest.approx(0.5)


def test_high_vol_clamped_at_ceiling():
    r = estimate_fill(price=100.0, inputs=_inputs(atr_pct=1.0, recent_quote_volume_usd=0))
    assert r.components["vol_multiplier"] == pytest.approx(4.0)


def test_size_impact_scales_with_volume_share():
    """Trade is 10% of recent vol → impact ≈ 10bp."""
    r = estimate_fill(
        price=100.0,
        inputs=_inputs(notional_usd=100_000, recent_quote_volume_usd=1_000_000),
    )
    assert r.components["size_impact"] == pytest.approx(0.0001)


def test_size_impact_capped_at_ceiling():
    """Impossibly-large order shouldn't produce a negative or zero fill."""
    r = estimate_fill(
        price=100.0,
        inputs=_inputs(notional_usd=10_000_000, recent_quote_volume_usd=1_000),
    )
    assert r.components["size_impact"] <= 0.01
    assert r.fill_price > 0


def test_zero_volume_disables_size_impact():
    """No recent volume signal → size_impact = 0 (don't fabricate cost)."""
    r = estimate_fill(price=100.0, inputs=_inputs(recent_quote_volume_usd=0))
    assert r.components["size_impact"] == 0.0


def test_invalid_side_raises():
    with pytest.raises(ValueError, match="side"):
        estimate_fill(price=100.0, inputs=_inputs(side="hold"))


def test_invalid_price_raises():
    with pytest.raises(ValueError, match="price"):
        estimate_fill(price=0.0, inputs=_inputs())


# ── Confidence weighting ───────────────────────────────────────


def test_confidence_midpoint_leaves_quantity_unchanged():
    assert confidence_weighted_quantity(1.0, confidence=0.5) == 1.0


def test_confidence_max_lifts_to_ceiling():
    assert confidence_weighted_quantity(1.0, confidence=1.0) == 1.5


def test_confidence_min_floors_quantity():
    assert confidence_weighted_quantity(1.0, confidence=0.0) == 0.5


def test_confidence_above_one_clamped():
    assert confidence_weighted_quantity(1.0, confidence=2.0) == 1.5


def test_confidence_below_zero_clamped():
    assert confidence_weighted_quantity(1.0, confidence=-1.0) == 0.5


def test_zero_base_quantity_stays_zero():
    """Modulator only — never invents a trade."""
    assert confidence_weighted_quantity(0.0, confidence=1.0) == 0.0


def test_custom_floor_ceiling_respected():
    assert confidence_weighted_quantity(1.0, confidence=1.0, floor=0.8, ceiling=1.2) == 1.2
    assert confidence_weighted_quantity(1.0, confidence=0.0, floor=0.8, ceiling=1.2) == 0.8
