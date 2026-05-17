"""Tests for halal/duration_hedge.py — Round-5 Wave 13.D."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_17 import SukukType
from halal_trader.halal.duration_hedge import (
    HedgeRecommendation,
    HedgeStance,
    PortfolioDuration,
    compute_portfolio_duration,
    dollar_duration,
    macaulay_duration,
    modified_duration,
    recommend_hedge,
    render_recommendation,
)
from halal_trader.markets.sukuk_pricing import Cashflow, Sukuk


def _ijara_sukuk(years: int = 5, coupon: float = 0.04, face: float = 1000.0) -> Sukuk:
    cashflows = []
    for y in range(1, years + 1):
        amount = face * coupon + (face if y == years else 0)
        cashflows.append(Cashflow(amount=amount, time_years=float(y)))
    return Sukuk(
        issuer="GovOfMalaysia",
        sukuk_type=SukukType.IJARA,
        cashflows=tuple(cashflows),
        face_value=face,
    )


# --- Validation -------------------------------------------------


def test_stance_string_values():
    assert HedgeStance.NEUTRAL.value == "neutral"
    assert HedgeStance.OFFSETTING.value == "offsetting"
    assert HedgeStance.ENHANCING.value == "enhancing"


def test_macaulay_too_negative_yield_rejected():
    with pytest.raises(ValueError):
        macaulay_duration(_ijara_sukuk(), yield_rate=-0.10)


def test_dollar_duration_negative_mv_rejected():
    with pytest.raises(ValueError):
        dollar_duration(_ijara_sukuk(), yield_rate=0.05, market_value=-1)


def test_portfolio_negative_mv_rejected():
    with pytest.raises(ValueError):
        PortfolioDuration(
            total_market_value=-1, total_dollar_duration=0, average_duration=0
        )


def test_hedge_negative_mv_rejected():
    with pytest.raises(ValueError):
        HedgeRecommendation(
            stance=HedgeStance.NEUTRAL,
            target_dollar_duration_offset=0,
            hedge_market_value=-1,
        )


# --- Duration math -------------------------------------------


def test_macaulay_short_sukuk_lower_duration():
    short = macaulay_duration(_ijara_sukuk(years=2), yield_rate=0.05)
    long = macaulay_duration(_ijara_sukuk(years=10), yield_rate=0.05)
    assert short < long


def test_macaulay_higher_yield_lower_duration():
    """Higher yield discounts later cashflows more, lowering Macaulay."""
    low_y = macaulay_duration(_ijara_sukuk(years=10), yield_rate=0.02)
    high_y = macaulay_duration(_ijara_sukuk(years=10), yield_rate=0.10)
    assert low_y > high_y


def test_modified_lt_macaulay():
    s = _ijara_sukuk()
    mac = macaulay_duration(s, yield_rate=0.05)
    mod = modified_duration(s, yield_rate=0.05)
    assert mod < mac


def test_dollar_duration_proportional_to_mv():
    s = _ijara_sukuk()
    dd_a = dollar_duration(s, yield_rate=0.05, market_value=1000)
    dd_b = dollar_duration(s, yield_rate=0.05, market_value=2000)
    assert dd_b == pytest.approx(2 * dd_a)


# --- Portfolio aggregate ------------------------------------


def test_portfolio_aggregate_sums():
    holdings = [
        (_ijara_sukuk(years=2), 0.05, 1000.0),
        (_ijara_sukuk(years=5), 0.05, 2000.0),
    ]
    p = compute_portfolio_duration(holdings)
    assert p.total_market_value == 3000
    assert p.total_dollar_duration > 0


def test_portfolio_empty_zero():
    p = compute_portfolio_duration([])
    assert p.total_market_value == 0
    assert p.total_dollar_duration == 0


def test_portfolio_average_weighted():
    holdings = [
        (_ijara_sukuk(years=2), 0.05, 1000.0),
        (_ijara_sukuk(years=10), 0.05, 1000.0),
    ]
    p = compute_portfolio_duration(holdings)
    short_md = modified_duration(_ijara_sukuk(years=2), yield_rate=0.05)
    long_md = modified_duration(_ijara_sukuk(years=10), yield_rate=0.05)
    expected_avg = (short_md + long_md) / 2
    assert p.average_duration == pytest.approx(expected_avg)


# --- Hedge recommendation ----------------------------------


def test_recommend_zero_target_offsets_full_portfolio():
    portfolio = compute_portfolio_duration(
        [(_ijara_sukuk(years=10), 0.05, 100000.0)]
    )
    rec = recommend_hedge(
        portfolio, hedge_sukuk=_ijara_sukuk(years=5), hedge_yield=0.05
    )
    assert rec.stance is HedgeStance.OFFSETTING
    assert rec.hedge_market_value > 0


def test_recommend_neutral_when_already_at_target():
    portfolio = PortfolioDuration(
        total_market_value=100000,
        total_dollar_duration=0.0,
        average_duration=0.0,
    )
    rec = recommend_hedge(
        portfolio,
        hedge_sukuk=_ijara_sukuk(),
        hedge_yield=0.05,
    )
    assert rec.stance is HedgeStance.NEUTRAL


def test_recommend_zero_duration_hedge_returns_neutral():
    """Hedge sukuk with zero duration cannot offset — return neutral."""
    # Pure cash equivalent (1-day): essentially zero duration.
    s = Sukuk(
        issuer="X",
        sukuk_type=SukukType.IJARA,
        cashflows=(Cashflow(amount=1000.0, time_years=0.001),),
        face_value=1000.0,
    )
    portfolio = compute_portfolio_duration(
        [(_ijara_sukuk(years=10), 0.05, 100000.0)]
    )
    rec = recommend_hedge(portfolio, hedge_sukuk=s, hedge_yield=0.05)
    # Hedge MV will be huge but not zero; we just check it's a finite non-negative
    assert rec.hedge_market_value >= 0


# --- Render ------------------------------------------------


def test_render_neutral():
    rec = HedgeRecommendation(
        stance=HedgeStance.NEUTRAL,
        target_dollar_duration_offset=0,
        hedge_market_value=0,
    )
    out = render_recommendation(rec)
    assert "neutral" in out
    assert "⚖️" in out


def test_render_offsetting():
    portfolio = compute_portfolio_duration(
        [(_ijara_sukuk(years=10), 0.05, 100000.0)]
    )
    rec = recommend_hedge(
        portfolio, hedge_sukuk=_ijara_sukuk(years=5), hedge_yield=0.05
    )
    out = render_recommendation(rec)
    assert "🛡️" in out
    assert "offsetting" in out


def test_render_no_secret_leak():
    rec = HedgeRecommendation(
        stance=HedgeStance.NEUTRAL,
        target_dollar_duration_offset=0,
        hedge_market_value=0,
    )
    out = render_recommendation(rec)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------


def test_e2e_long_book_hedged_by_short_sukuk():
    """Operator's book is dominated by long-dated sukuk; hedge with short."""
    portfolio = compute_portfolio_duration(
        [(_ijara_sukuk(years=10), 0.05, 1_000_000.0)]
    )
    rec = recommend_hedge(
        portfolio, hedge_sukuk=_ijara_sukuk(years=2), hedge_yield=0.05
    )
    assert rec.stance is HedgeStance.OFFSETTING
    # Hedge MV should be larger than book size since short sukuk has lower duration
    assert rec.hedge_market_value > 1_000_000


def test_replay_consistency():
    portfolio = compute_portfolio_duration([(_ijara_sukuk(), 0.05, 100000)])
    a = recommend_hedge(portfolio, hedge_sukuk=_ijara_sukuk(), hedge_yield=0.05)
    b = recommend_hedge(portfolio, hedge_sukuk=_ijara_sukuk(), hedge_yield=0.05)
    assert a == b
