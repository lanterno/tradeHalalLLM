"""Tests for the cross-sectional factor core."""

from __future__ import annotations

from halal_trader.core.factors import FactorScore, rank_factors


def _m(chg20d, atr, price, adx):
    return {"chg20d": chg20d, "atr": atr, "price": price, "adx": adx}


def test_clear_winner_ranks_first():
    # WIN: strongest momentum, lowest vol, strongest trend.
    metrics = {
        "WIN": _m(chg20d=12.0, atr=1.0, price=100.0, adx=40.0),
        "MID": _m(chg20d=3.0, atr=3.0, price=100.0, adx=20.0),
        "LOSE": _m(chg20d=-5.0, atr=6.0, price=100.0, adx=10.0),
    }
    ranked = rank_factors(metrics)
    assert [f.symbol for f in ranked] == ["WIN", "MID", "LOSE"]
    assert ranked[0].composite > ranked[-1].composite


def test_scores_are_cross_sectional_zscores():
    metrics = {
        "A": _m(10.0, 2.0, 100.0, 30.0),
        "B": _m(0.0, 2.0, 100.0, 30.0),
        "C": _m(-10.0, 2.0, 100.0, 30.0),
    }
    ranked = rank_factors(metrics)
    # momentum z-scores should be symmetric around 0 (mean-centred).
    by_sym = {f.symbol: f for f in ranked}
    assert by_sym["B"].momentum == 0.0  # the mean
    assert by_sym["A"].momentum > 0 > by_sym["C"].momentum
    # vol + trend identical across the universe → zero contribution.
    assert by_sym["A"].low_vol == 0.0
    assert by_sym["A"].trend_quality == 0.0


def test_low_vol_factor_prefers_lower_atr():
    metrics = {
        "CALM": _m(5.0, 1.0, 100.0, 25.0),  # 1% ATR
        "WILD": _m(5.0, 8.0, 100.0, 25.0),  # 8% ATR
    }
    ranked = rank_factors(metrics)
    by_sym = {f.symbol: f for f in ranked}
    assert by_sym["CALM"].low_vol > by_sym["WILD"].low_vol
    assert ranked[0].symbol == "CALM"


def test_empty_universe():
    assert rank_factors({}) == []


def test_missing_fields_degrade_to_zero():
    metrics = {"A": {"chg20d": 5.0}, "B": {}}  # B has nothing
    ranked = rank_factors(metrics)
    assert len(ranked) == 2
    assert all(isinstance(f, FactorScore) for f in ranked)
    assert ranked[0].symbol == "A"  # the only one with momentum


def test_weights_shift_ranking():
    metrics = {
        "MOM": _m(20.0, 5.0, 100.0, 10.0),  # great momentum, bad vol/trend
        "TREND": _m(0.0, 5.0, 100.0, 40.0),  # flat momentum, great trend
    }
    mom_heavy = rank_factors(metrics, weights=(1.0, 0.0, 0.0))
    trend_heavy = rank_factors(metrics, weights=(0.0, 0.0, 1.0))
    assert mom_heavy[0].symbol == "MOM"
    assert trend_heavy[0].symbol == "TREND"
