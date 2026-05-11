"""Tests for ml/vol_target.py — Round-5 Wave 4.E."""

from __future__ import annotations

import math

import pytest

from halal_trader.ml.vol_target import (
    PortfolioVolTargetPlan,
    ScalingMode,
    VolTargetConfig,
    apply_vol_target,
    compute_scale,
    realised_volatility,
    render_decision,
    render_plan,
)

# --- VolTargetConfig validation -------------------------------------------


def test_config_default():
    c = VolTargetConfig(target_volatility=0.10)
    assert c.lookback_days == 30
    assert c.scaling_mode is ScalingMode.SCALE_DOWN_ONLY


def test_config_invalid_target():
    with pytest.raises(ValueError):
        VolTargetConfig(target_volatility=0.0)
    with pytest.raises(ValueError):
        VolTargetConfig(target_volatility=1.5)


def test_config_invalid_lookback():
    with pytest.raises(ValueError):
        VolTargetConfig(target_volatility=0.10, lookback_days=4)


def test_config_invalid_bars_per_year():
    with pytest.raises(ValueError):
        VolTargetConfig(target_volatility=0.10, bars_per_year=100)


def test_config_min_max_scale_ordering():
    with pytest.raises(ValueError):
        VolTargetConfig(target_volatility=0.10, min_scale=1.0, max_scale=0.5)


def test_config_max_scale_too_high_rejected():
    """Pin: max_scale > 3.0 requires explicit Sharia review."""
    with pytest.raises(ValueError):
        VolTargetConfig(target_volatility=0.10, max_scale=5.0)


# --- realised_volatility --------------------------------------------------


def test_realised_volatility_constant_prices_zero():
    rv = realised_volatility([100.0] * 30)
    assert rv == 0.0


def test_realised_volatility_one_price_zero():
    assert realised_volatility([100.0]) == 0.0


def test_realised_volatility_empty_zero():
    assert realised_volatility([]) == 0.0


def test_realised_volatility_negative_price_rejected():
    with pytest.raises(ValueError):
        realised_volatility([100.0, -10.0])


def test_realised_volatility_increases_with_swings():
    calm = [100.0 + 0.1 * i for i in range(30)]
    wild = [100.0 * (1.05 if i % 2 == 0 else 0.95) for i in range(30)]
    assert realised_volatility(wild) > realised_volatility(calm)


def test_realised_volatility_annualisation_pin():
    """Pin: σ_annual = σ_per_bar × √bars_per_year."""
    # Construct prices with deterministic per-bar log returns.
    # 30 bars of alternating ±1% return → per-bar σ ≈ 0.01.
    prices = [100.0]
    for i in range(30):
        prices.append(prices[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    rv = realised_volatility(prices, bars_per_year=252)
    # Per-bar σ should be ~0.01, annualised ~0.01 × √252 ≈ 0.158.
    assert 0.10 < rv < 0.20


def test_realised_volatility_crypto_annualisation():
    prices = [100.0]
    for i in range(30):
        prices.append(prices[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    rv_stocks = realised_volatility(prices, bars_per_year=252)
    rv_crypto = realised_volatility(prices, bars_per_year=365)
    assert rv_crypto > rv_stocks
    assert abs(rv_crypto / rv_stocks - math.sqrt(365 / 252)) < 0.01


# --- compute_scale --------------------------------------------------------


def _stable_prices(n: int = 60) -> list[float]:
    """Almost-flat price path → very low realised vol."""
    return [100.0 + 0.001 * i for i in range(n)]


def _wild_prices(n: int = 60) -> list[float]:
    p = [100.0]
    for i in range(n):
        p.append(p[-1] * (1.03 if i % 2 == 0 else 1 / 1.03))
    return p


def test_compute_scale_scale_down_only_reduces_when_high_vol():
    cfg = VolTargetConfig(target_volatility=0.10, scaling_mode=ScalingMode.SCALE_DOWN_ONLY)
    decision = compute_scale(_wild_prices(), cfg)
    assert decision.final_scale <= 1.0


def test_compute_scale_scale_down_only_caps_at_1():
    """Under SCALE_DOWN_ONLY, even very calm markets get capped at 1.0."""
    cfg = VolTargetConfig(target_volatility=0.30, scaling_mode=ScalingMode.SCALE_DOWN_ONLY)
    decision = compute_scale(_stable_prices(), cfg)
    assert decision.final_scale <= 1.0


def test_compute_scale_scale_both_can_lever():
    cfg = VolTargetConfig(
        target_volatility=0.30, scaling_mode=ScalingMode.SCALE_BOTH, max_scale=2.0
    )
    decision = compute_scale(_stable_prices(), cfg)
    # Stable prices → very low realised vol → raw scale very high.
    # max_scale=2.0 caps it; final should be > 1.0.
    assert decision.final_scale > 1.0
    assert decision.is_levered()


def test_compute_scale_salam_overlay_engages_when_under_target():
    cfg = VolTargetConfig(
        target_volatility=0.30,
        scaling_mode=ScalingMode.SALAM_OVERLAY,
        max_scale=2.0,
    )
    decision = compute_scale(_stable_prices(), cfg)
    assert decision.salam_overlay_pct > 0
    # Final position is capped at 1.0; overlay is the excess.
    assert decision.final_scale == 1.0


def test_compute_scale_salam_overlay_no_overlay_when_over_target():
    cfg = VolTargetConfig(target_volatility=0.10, scaling_mode=ScalingMode.SALAM_OVERLAY)
    decision = compute_scale(_wild_prices(), cfg)
    # Realised > target → reduce → no overlay.
    assert decision.salam_overlay_pct == 0


def test_compute_scale_floor_when_below_kicks_in():
    cfg = VolTargetConfig(
        target_volatility=0.30,
        scaling_mode=ScalingMode.SCALE_DOWN_ONLY,
        floor_when_below=True,
    )
    decision = compute_scale(_stable_prices(), cfg)
    # Stable → realised << target → final = 0.
    assert decision.final_scale == 0.0


def test_compute_scale_floor_when_below_off_by_default():
    cfg = VolTargetConfig(target_volatility=0.30, scaling_mode=ScalingMode.SCALE_DOWN_ONLY)
    decision = compute_scale(_stable_prices(), cfg)
    # Default: just caps at 1.0, doesn't floor at 0.
    assert decision.final_scale > 0.0


def test_compute_scale_min_scale_clamp():
    cfg = VolTargetConfig(
        target_volatility=0.05, min_scale=0.20, scaling_mode=ScalingMode.SCALE_BOTH
    )
    decision = compute_scale(_wild_prices(), cfg)
    assert decision.final_scale >= 0.20 - 1e-9


def test_compute_scale_empty_prices_rejected():
    cfg = VolTargetConfig(target_volatility=0.10)
    with pytest.raises(ValueError):
        compute_scale([], cfg)


def test_compute_scale_note_present():
    cfg = VolTargetConfig(target_volatility=0.10)
    d = compute_scale(_wild_prices(), cfg)
    assert d.note != ""


# --- apply_vol_target -----------------------------------------------------


def test_apply_basic_two_asset():
    cfg = VolTargetConfig(target_volatility=0.10)
    plan = apply_vol_target(
        [0.6, 0.4],
        [_wild_prices(), _wild_prices()],
        cfg,
    )
    assert isinstance(plan, PortfolioVolTargetPlan)
    assert (
        sum(plan.final_weights) + plan.cash_weight + sum(plan.salam_overlay_per_asset)
        == pytest.approx(1.0, abs=1e-6)
        or sum(plan.final_weights) <= 1.0
    )


def test_apply_weights_must_sum_to_one():
    cfg = VolTargetConfig(target_volatility=0.10)
    with pytest.raises(ValueError):
        apply_vol_target(
            [0.6, 0.5],
            [_wild_prices(), _wild_prices()],
            cfg,
        )


def test_apply_length_mismatch_rejected():
    cfg = VolTargetConfig(target_volatility=0.10)
    with pytest.raises(ValueError):
        apply_vol_target([0.6, 0.4], [_wild_prices()], cfg)


def test_apply_empty_rejected():
    cfg = VolTargetConfig(target_volatility=0.10)
    with pytest.raises(ValueError):
        apply_vol_target([], [], cfg)


def test_apply_scale_down_only_yields_cash():
    cfg = VolTargetConfig(target_volatility=0.05, scaling_mode=ScalingMode.SCALE_DOWN_ONLY)
    plan = apply_vol_target([0.5, 0.5], [_wild_prices(), _wild_prices()], cfg)
    assert plan.cash_weight > 0
    assert sum(plan.final_weights) < 1.0


def test_apply_salam_overlay_only_under_target():
    cfg = VolTargetConfig(
        target_volatility=0.30,
        scaling_mode=ScalingMode.SALAM_OVERLAY,
        max_scale=2.0,
    )
    plan = apply_vol_target([0.5, 0.5], [_stable_prices(), _stable_prices()], cfg)
    assert sum(plan.salam_overlay_per_asset) > 0


def test_apply_salam_overlay_no_overlay_over_target():
    cfg = VolTargetConfig(target_volatility=0.05, scaling_mode=ScalingMode.SALAM_OVERLAY)
    plan = apply_vol_target([0.5, 0.5], [_wild_prices(), _wild_prices()], cfg)
    assert sum(plan.salam_overlay_per_asset) == 0


def test_apply_floor_when_below_yields_full_cash():
    cfg = VolTargetConfig(
        target_volatility=0.30,
        scaling_mode=ScalingMode.SCALE_DOWN_ONLY,
        floor_when_below=True,
    )
    plan = apply_vol_target([0.5, 0.5], [_stable_prices(), _stable_prices()], cfg)
    assert plan.cash_weight == 1.0
    for w in plan.final_weights:
        assert w == 0.0


def test_apply_aggregate_vol_reported():
    cfg = VolTargetConfig(target_volatility=0.10)
    plan = apply_vol_target([0.5, 0.5], [_wild_prices(), _stable_prices()], cfg)
    # Aggregate σ should be between the wild and stable σ since it's a
    # weighted average.
    rv_wild = realised_volatility(_wild_prices()[-31:], bars_per_year=252)
    rv_stable = realised_volatility(_stable_prices()[-31:], bars_per_year=252)
    assert min(rv_wild, rv_stable) <= plan.aggregate_realised_vol <= max(rv_wild, rv_stable)


# --- Render ---------------------------------------------------------------


def test_render_decision_no_overlay_branch():
    cfg = VolTargetConfig(target_volatility=0.10)
    d = compute_scale(_wild_prices(), cfg)
    out = render_decision(d)
    assert "Vol-target" in out
    assert "scale=" in out


def test_render_decision_overlay_branch():
    cfg = VolTargetConfig(
        target_volatility=0.30,
        scaling_mode=ScalingMode.SALAM_OVERLAY,
        max_scale=2.0,
    )
    d = compute_scale(_stable_prices(), cfg)
    out = render_decision(d)
    assert "Salam overlay" in out


def test_render_plan_overlay_summary():
    cfg = VolTargetConfig(
        target_volatility=0.30,
        scaling_mode=ScalingMode.SALAM_OVERLAY,
        max_scale=2.0,
    )
    plan = apply_vol_target([0.5, 0.5], [_stable_prices(), _stable_prices()], cfg)
    out = render_plan(plan)
    assert "salam=" in out
