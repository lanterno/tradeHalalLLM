"""Tests for trading/exec_ab.py — Round-5 Wave 12.I."""

from __future__ import annotations

import pytest

from halal_trader.trading.exec_ab import (
    Fill,
    Side,
    compare_cohorts,
    power_estimate,
    render_compare,
    render_power,
    slippage_bps,
    summarise_cohort,
)


def _fill(
    fill_id: str = "F1",
    algo: str = "twap",
    side: Side = Side.BUY,
    arrival: float = 100.0,
    fill_price: float = 100.0,
    quantity: float = 100.0,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        algo_label=algo,
        side=side,
        arrival_mid=arrival,
        fill_price=fill_price,
        quantity=quantity,
    )


# --- Fill validation ---------------------------------------------------


def test_fill_valid():
    f = _fill()
    assert f.algo_label == "twap"


def test_fill_empty_id_rejected():
    with pytest.raises(ValueError):
        _fill(fill_id="")


def test_fill_empty_algo_rejected():
    with pytest.raises(ValueError):
        _fill(algo=" ")


def test_fill_zero_arrival_rejected():
    with pytest.raises(ValueError):
        _fill(arrival=0)


def test_fill_immutable():
    f = _fill()
    with pytest.raises(AttributeError):
        f.fill_price = 0  # type: ignore[misc]


# --- slippage_bps — signed-positive-bad pin ----------------------------


def test_slippage_buy_above_arrival_positive():
    """BUY paid up → positive slippage (bad)."""
    f = _fill(side=Side.BUY, arrival=100.0, fill_price=101.0)
    # (101 - 100) / 100 × 1e4 = 100bps.
    assert slippage_bps(f) == pytest.approx(100.0)


def test_slippage_buy_below_arrival_negative():
    """BUY filled below arrival → negative slippage (good)."""
    f = _fill(side=Side.BUY, arrival=100.0, fill_price=99.0)
    assert slippage_bps(f) == pytest.approx(-100.0)


def test_slippage_sell_below_arrival_positive():
    """SELL filled below arrival → positive slippage (bad)."""
    f = _fill(side=Side.SELL, arrival=100.0, fill_price=99.0)
    # (100 - 99) / 100 × 1e4 = 100bps.
    assert slippage_bps(f) == pytest.approx(100.0)


def test_slippage_sell_above_arrival_negative():
    """SELL filled above arrival → negative slippage (good)."""
    f = _fill(side=Side.SELL, arrival=100.0, fill_price=101.0)
    assert slippage_bps(f) == pytest.approx(-100.0)


def test_slippage_zero_at_arrival():
    f = _fill(arrival=100.0, fill_price=100.0)
    assert slippage_bps(f) == 0.0


# --- summarise_cohort --------------------------------------------------


def test_summarise_cohort_basic():
    fills = [
        _fill(fill_id="F1", algo="twap", arrival=100, fill_price=101),
        _fill(fill_id="F2", algo="twap", arrival=100, fill_price=102),
        _fill(fill_id="F3", algo="twap", arrival=100, fill_price=99),
    ]
    stats = summarise_cohort("twap", fills)
    assert stats.n_fills == 3
    # mean of [100, 200, -100] = 200/3 ≈ 66.67.
    assert stats.mean_slippage_bps == pytest.approx(200 / 3)
    assert stats.median_slippage_bps == pytest.approx(100.0)


def test_summarise_cohort_empty_rejected():
    with pytest.raises(ValueError):
        summarise_cohort("twap", [])


def test_summarise_cohort_label_mismatch_rejected():
    fills = [_fill(algo="vwap")]
    with pytest.raises(ValueError):
        summarise_cohort("twap", fills)


def test_summarise_total_quantity():
    fills = [
        _fill(fill_id="F1", quantity=100),
        _fill(fill_id="F2", quantity=200),
    ]
    stats = summarise_cohort("twap", fills)
    assert stats.total_quantity == 300.0


def test_summarise_even_n_median_is_average():
    fills = [
        _fill(fill_id="F1", arrival=100, fill_price=101),
        _fill(fill_id="F2", arrival=100, fill_price=103),
    ]
    stats = summarise_cohort("twap", fills)
    # Slippages: 100, 300 → median = 200.
    assert stats.median_slippage_bps == pytest.approx(200.0)


# --- compare_cohorts ---------------------------------------------------


def _cohort(
    label: str,
    arrivals_fills: list[tuple[float, float]],
) -> list[Fill]:
    return [
        _fill(fill_id=f"{label}-{i}", algo=label, arrival=a, fill_price=p)
        for i, (a, p) in enumerate(arrivals_fills)
    ]


def test_compare_no_difference_includes_zero():
    """Equal cohorts → CI brackets zero → not significant."""
    fills_a = _cohort("twap", [(100, 100.5)] * 30)
    fills_b = _cohort("vwap", [(100, 100.5)] * 30)
    result = compare_cohorts(fills_a, fills_b, n_bootstrap=200, seed=42)
    assert result.delta_a_minus_b_bps == pytest.approx(0.0)
    assert not result.is_significant


def test_compare_clear_difference_significant():
    """Cohort A consistently worse → positive delta, CI excludes zero."""
    fills_a = _cohort("twap", [(100, 102)] * 30)  # 200bps each
    fills_b = _cohort("vwap", [(100, 100.5)] * 30)  # 50bps each
    result = compare_cohorts(fills_a, fills_b, n_bootstrap=500, seed=42)
    assert result.delta_a_minus_b_bps > 0
    assert result.is_significant
    # Bracket should be entirely above zero.
    assert result.ci_low_bps > 0


def test_compare_seed_deterministic():
    fills_a = _cohort("twap", [(100, 101 + i * 0.1) for i in range(20)])
    fills_b = _cohort("vwap", [(100, 100.5)] * 20)
    r1 = compare_cohorts(fills_a, fills_b, seed=42, n_bootstrap=200)
    r2 = compare_cohorts(fills_a, fills_b, seed=42, n_bootstrap=200)
    assert r1.ci_low_bps == r2.ci_low_bps
    assert r1.ci_high_bps == r2.ci_high_bps


def test_compare_different_seeds_differ():
    fills_a = _cohort("twap", [(100, 101 + i * 0.1) for i in range(20)])
    fills_b = _cohort("vwap", [(100, 100.5)] * 20)
    r1 = compare_cohorts(fills_a, fills_b, seed=42, n_bootstrap=200)
    r2 = compare_cohorts(fills_a, fills_b, seed=999, n_bootstrap=200)
    # Means are equal (deterministic input), but CI bounds will vary.
    assert (r1.ci_low_bps != r2.ci_low_bps) or (r1.ci_high_bps != r2.ci_high_bps)


def test_compare_empty_cohort_rejected():
    with pytest.raises(ValueError):
        compare_cohorts([], _cohort("vwap", [(100, 100.5)]))


def test_compare_invalid_confidence_rejected():
    fills_a = _cohort("twap", [(100, 101)] * 20)
    fills_b = _cohort("vwap", [(100, 101)] * 20)
    with pytest.raises(ValueError):
        compare_cohorts(fills_a, fills_b, confidence=0.0)
    with pytest.raises(ValueError):
        compare_cohorts(fills_a, fills_b, confidence=1.0)


def test_compare_too_few_bootstraps_rejected():
    fills_a = _cohort("twap", [(100, 101)] * 20)
    fills_b = _cohort("vwap", [(100, 101)] * 20)
    with pytest.raises(ValueError):
        compare_cohorts(fills_a, fills_b, n_bootstrap=10)


def test_compare_label_inferred_from_first_fill():
    fills_a = _cohort("twap", [(100, 101)] * 20)
    fills_b = _cohort("vwap", [(100, 101)] * 20)
    result = compare_cohorts(fills_a, fills_b, n_bootstrap=200, seed=42)
    assert result.label_a == "twap"
    assert result.label_b == "vwap"


def test_compare_explicit_labels_used():
    fills_a = _cohort("twap", [(100, 101)] * 20)
    fills_b = _cohort("vwap", [(100, 101)] * 20)
    result = compare_cohorts(fills_a, fills_b, label_a="A", label_b="B", n_bootstrap=200, seed=42)
    assert result.label_a == "A"
    assert result.label_b == "B"


def test_compare_negative_delta_for_better_a():
    """If A is BETTER (lower slippage), delta < 0."""
    fills_a = _cohort("twap", [(100, 100.1)] * 30)  # 10bps
    fills_b = _cohort("vwap", [(100, 102)] * 30)  # 200bps
    result = compare_cohorts(fills_a, fills_b, n_bootstrap=500, seed=42)
    assert result.delta_a_minus_b_bps < 0
    assert result.is_significant


# --- power_estimate ----------------------------------------------------


def test_power_well_powered():
    """Large sample sizes + meaningful target → z >> 1.96 → powered."""
    p = power_estimate(
        cohort_a_std_bps=50.0,
        cohort_b_std_bps=50.0,
        n_a=1000,
        n_b=1000,
        target_delta_bps=10.0,
    )
    assert p.is_well_powered


def test_power_underpowered_small_sample():
    p = power_estimate(
        cohort_a_std_bps=100.0,
        cohort_b_std_bps=100.0,
        n_a=5,
        n_b=5,
        target_delta_bps=10.0,
    )
    assert not p.is_well_powered


def test_power_invalid_n_rejected():
    with pytest.raises(ValueError):
        power_estimate(50.0, 50.0, n_a=0, n_b=10, target_delta_bps=10.0)


def test_power_negative_std_rejected():
    with pytest.raises(ValueError):
        power_estimate(-1.0, 50.0, n_a=10, n_b=10, target_delta_bps=10.0)


def test_power_zero_target_rejected():
    with pytest.raises(ValueError):
        power_estimate(50.0, 50.0, n_a=10, n_b=10, target_delta_bps=0)


def test_power_target_negative_uses_abs():
    """Pin: power should treat -10bps target same as +10bps."""
    p_pos = power_estimate(50.0, 50.0, n_a=100, n_b=100, target_delta_bps=10.0)
    p_neg = power_estimate(50.0, 50.0, n_a=100, n_b=100, target_delta_bps=-10.0)
    assert p_pos.estimated_z == pytest.approx(p_neg.estimated_z)


# --- Render -----------------------------------------------------------


def test_render_compare_significant():
    fills_a = _cohort("twap", [(100, 102)] * 30)
    fills_b = _cohort("vwap", [(100, 100.5)] * 30)
    result = compare_cohorts(fills_a, fills_b, n_bootstrap=500, seed=42)
    out = render_compare(result)
    assert "🆚" in out
    assert "significant" in out


def test_render_compare_inconclusive():
    fills_a = _cohort("twap", [(100, 100.5)] * 30)
    fills_b = _cohort("vwap", [(100, 100.5)] * 30)
    result = compare_cohorts(fills_a, fills_b, n_bootstrap=500, seed=42)
    out = render_compare(result)
    assert "inconclusive" in out


def test_render_power_format():
    p = power_estimate(50.0, 50.0, n_a=1000, n_b=1000, target_delta_bps=10.0)
    out = render_power(p)
    assert "⚡" in out
    assert "powered" in out
