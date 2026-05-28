"""Feasibility — lot rounding, min-notional floor, buying-power, sells."""

from __future__ import annotations

from halabot.execution.feasibility import FeasibilityConfig, feasible_buy, feasible_sell

CFG = FeasibilityConfig(min_notional_usd=50.0, lot_step=1.0)


def test_buy_rounds_down_to_whole_shares():
    f = feasible_buy(250.0, 100.0, buying_power=10_000.0, cfg=CFG)
    assert f.ok and f.quantity == 2.0  # floor(2.5) = 2 shares
    assert f.notional_usd == 200.0


def test_buy_below_min_notional_rejected():
    f = feasible_buy(40.0, 100.0, buying_power=10_000.0, cfg=CFG)
    assert not f.ok and "min notional" in f.reason


def test_buy_capped_by_buying_power():
    f = feasible_buy(10_000.0, 100.0, buying_power=150.0, cfg=CFG)
    assert f.ok and f.quantity == 1.0  # only $150 BP → 1 share ($100)


def test_buy_that_rounds_to_zero_rejected():
    f = feasible_buy(60.0, 100.0, buying_power=10_000.0, cfg=CFG)
    # $60 / $100 = 0.6 → floor to 0 whole shares → reject
    assert not f.ok


def test_fractional_lot_step_for_crypto():
    cfg = FeasibilityConfig(min_notional_usd=50.0, lot_step=0.001)
    f = feasible_buy(100.0, 30_000.0, buying_power=10_000.0, cfg=cfg)
    assert f.ok and f.quantity > 0 and f.notional_usd >= 50.0


def test_non_positive_price_rejected():
    assert not feasible_buy(100.0, 0.0, buying_power=1000.0, cfg=CFG).ok


def test_sell_always_feasible_for_held_qty():
    f = feasible_sell(5.0, cfg=CFG)
    assert f.ok and f.quantity == 5.0


def test_sell_nothing_rejected():
    assert not feasible_sell(0.0, cfg=CFG).ok
