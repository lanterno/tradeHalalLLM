"""Tests for halal/fx_hedge.py — Round-5 Wave 13.E."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.halal.fx_hedge import (
    CurrencyExposure,
    FXHedgeLeg,
    FXHedgePolicy,
    FXSpotRate,
    HedgeHorizon,
    net_exposure_by_currency,
    plan_fx_hedge,
    render_plan,
)


def _rate(base: str, quote: str, r: float) -> FXSpotRate:
    return FXSpotRate(base=base, quote=quote, rate=r)


# --- CurrencyExposure validation ----------------------------------------


def test_exposure_valid():
    e = CurrencyExposure(currency="USD", amount=1000.0)
    assert e.currency == "USD"


def test_exposure_invalid_currency_length():
    with pytest.raises(ValueError):
        CurrencyExposure(currency="US", amount=1000.0)
    with pytest.raises(ValueError):
        CurrencyExposure(currency="USDX", amount=1000.0)


def test_exposure_lowercase_rejected():
    with pytest.raises(ValueError):
        CurrencyExposure(currency="usd", amount=1000.0)


def test_exposure_zero_amount_rejected():
    with pytest.raises(ValueError):
        CurrencyExposure(currency="USD", amount=0.0)


def test_exposure_negative_allowed():
    e = CurrencyExposure(currency="USD", amount=-500.0)
    assert e.amount == -500.0


# --- FXSpotRate validation -----------------------------------------------


def test_rate_valid():
    r = _rate("USD", "SAR", 3.75)
    assert r.rate == 3.75


def test_rate_same_base_quote_rejected():
    with pytest.raises(ValueError):
        _rate("USD", "USD", 1.0)


def test_rate_negative_rejected():
    with pytest.raises(ValueError):
        _rate("USD", "SAR", -1.0)


# --- FXHedgePolicy validation --------------------------------------------


def test_policy_valid():
    p = FXHedgePolicy(base_currency="USD")
    assert p.horizon is HedgeHorizon.MEDIUM


def test_policy_invalid_base_currency():
    with pytest.raises(ValueError):
        FXHedgePolicy(base_currency="X")


def test_policy_t_plus_2_rejected():
    """Pin: AAOIFI Standard 1 — bay' al-sarf requires hand-to-hand;
    T+2 is rejected."""
    with pytest.raises(ValueError):
        FXHedgePolicy(base_currency="USD", spot_settlement_days=2)


def test_policy_unreasonable_fee_rejected():
    with pytest.raises(ValueError):
        FXHedgePolicy(base_currency="USD", wakalah_fee_bps=200.0)


def test_policy_negative_min_notional_rejected():
    with pytest.raises(ValueError):
        FXHedgePolicy(base_currency="USD", min_notional=-1.0)


# --- net_exposure_by_currency --------------------------------------------


def test_net_exposure_combines_long_short():
    exposures = [
        CurrencyExposure(currency="USD", amount=1000.0),
        CurrencyExposure(currency="USD", amount=-300.0),
        CurrencyExposure(currency="SAR", amount=500.0),
    ]
    out = net_exposure_by_currency(exposures)
    assert len(out) == 2
    by_cur = {e.currency: e.amount for e in out}
    assert by_cur["USD"] == 700.0
    assert by_cur["SAR"] == 500.0


def test_net_exposure_drops_fully_cancelled():
    exposures = [
        CurrencyExposure(currency="USD", amount=1000.0),
        CurrencyExposure(currency="USD", amount=-1000.0),
        CurrencyExposure(currency="SAR", amount=500.0),
    ]
    out = net_exposure_by_currency(exposures)
    currencies = {e.currency for e in out}
    assert "USD" not in currencies
    assert "SAR" in currencies


def test_net_exposure_sorted():
    exposures = [
        CurrencyExposure(currency="MYR", amount=100.0),
        CurrencyExposure(currency="AED", amount=200.0),
        CurrencyExposure(currency="USD", amount=300.0),
    ]
    out = net_exposure_by_currency(exposures)
    assert [e.currency for e in out] == ["AED", "MYR", "USD"]


# --- plan_fx_hedge -------------------------------------------------------


def test_plan_basic_two_legs():
    exposures = [
        CurrencyExposure(currency="SAR", amount=37500.0),
        CurrencyExposure(currency="AED", amount=36730.0),
    ]
    rates = [
        _rate("USD", "SAR", 3.75),
        _rate("USD", "AED", 3.673),
    ]
    pol = FXHedgePolicy(base_currency="USD")
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    assert plan.hedge_count() == 2
    # SAR notional in USD = 37500 / 3.75 = 10000.
    sar_leg = next(leg for leg in plan.legs if leg.currency == "SAR")
    assert sar_leg.notional_in_base == pytest.approx(10000.0)


def test_plan_skips_base_currency_exposure():
    exposures = [
        CurrencyExposure(currency="USD", amount=1000.0),
        CurrencyExposure(currency="SAR", amount=37500.0),
    ]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="USD")
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    assert plan.hedge_count() == 1
    assert plan.legs[0].currency == "SAR"


def test_plan_skips_below_min_notional():
    exposures = [
        CurrencyExposure(currency="SAR", amount=100.0),  # ~$26 — below min
        CurrencyExposure(currency="AED", amount=37000.0),  # ~$10070
    ]
    rates = [
        _rate("USD", "SAR", 3.75),
        _rate("USD", "AED", 3.673),
    ]
    pol = FXHedgePolicy(base_currency="USD", min_notional=1000.0)
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    assert plan.hedge_count() == 1
    assert "SAR" in plan.skipped_currencies


def test_plan_dates_pin_horizon_max_days():
    exposures = [CurrencyExposure(currency="SAR", amount=37500.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(
        base_currency="USD",
        horizon=HedgeHorizon.SHORT,
        spot_settlement_days=1,
    )
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    leg = plan.legs[0]
    # spot = plan + 1 = 2026-06-02
    assert leg.spot_settlement_date == date(2026, 6, 2)
    # roll = spot + 30 = 2026-07-02
    assert leg.waad_roll_date == leg.spot_settlement_date + timedelta(days=30)


def test_plan_t_plus_zero():
    exposures = [CurrencyExposure(currency="SAR", amount=37500.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="USD", spot_settlement_days=0)
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    leg = plan.legs[0]
    assert leg.spot_settlement_date == date(2026, 6, 1)


def test_plan_inverse_rate_lookup():
    """Pin: SAR/USD inverse should be discovered from a USD/SAR quote."""
    exposures = [CurrencyExposure(currency="USD", amount=10000.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="SAR")
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    # USD notional in SAR = 10000 × 3.75 = 37500.
    assert plan.legs[0].notional_in_base == pytest.approx(37500.0)


def test_plan_missing_rate_raises():
    exposures = [CurrencyExposure(currency="EUR", amount=1000.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="USD")
    with pytest.raises(ValueError):
        plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))


def test_plan_wakalah_fee_arithmetic():
    """Pin: fee = notional × bps / 1e4."""
    exposures = [CurrencyExposure(currency="SAR", amount=37500.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="USD", wakalah_fee_bps=10.0)
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    # Notional = 10000 USD; 10 bps = 0.1% → fee = 10.
    assert plan.legs[0].wakalah_fee == pytest.approx(10.0)
    assert plan.total_wakalah_fee == pytest.approx(10.0)


def test_plan_short_negative_exposure_treated_same():
    """Hedging long $1000 and short $1000 produces same notional."""
    exposures_long = [CurrencyExposure(currency="SAR", amount=37500.0)]
    exposures_short = [CurrencyExposure(currency="SAR", amount=-37500.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="USD")
    plan_long = plan_fx_hedge(exposures_long, rates, pol, plan_date=date(2026, 6, 1))
    plan_short = plan_fx_hedge(exposures_short, rates, pol, plan_date=date(2026, 6, 1))
    assert plan_long.legs[0].notional_in_base == plan_short.legs[0].notional_in_base


def test_plan_horizon_long_180_days():
    exposures = [CurrencyExposure(currency="SAR", amount=37500.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(
        base_currency="USD",
        horizon=HedgeHorizon.LONG,
        spot_settlement_days=0,
    )
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    leg = plan.legs[0]
    assert (leg.waad_roll_date - leg.spot_settlement_date).days == 180


# --- FXHedgeLeg validation -----------------------------------------------


def test_leg_roll_before_spot_rejected():
    with pytest.raises(ValueError):
        FXHedgeLeg(
            currency="SAR",
            notional_in_base=10000.0,
            spot_rate_to_base=3.75,
            spot_settlement_date=date(2026, 6, 5),
            waad_roll_date=date(2026, 6, 1),
            wakalah_fee=10.0,
        )


def test_leg_negative_notional_rejected():
    with pytest.raises(ValueError):
        FXHedgeLeg(
            currency="SAR",
            notional_in_base=-1.0,
            spot_rate_to_base=3.75,
            spot_settlement_date=date(2026, 6, 1),
            waad_roll_date=date(2026, 7, 1),
            wakalah_fee=10.0,
        )


# --- Render --------------------------------------------------------------


def test_render_plan_contains_summary():
    exposures = [CurrencyExposure(currency="SAR", amount=37500.0)]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="USD")
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    out = render_plan(plan)
    assert "💱" in out
    assert "USD" in out
    assert "SAR" in out


def test_render_plan_skipped_currencies_listed():
    exposures = [
        CurrencyExposure(currency="SAR", amount=10.0),  # below min
    ]
    rates = [_rate("USD", "SAR", 3.75)]
    pol = FXHedgePolicy(base_currency="USD", min_notional=1000.0)
    plan = plan_fx_hedge(exposures, rates, pol, plan_date=date(2026, 6, 1))
    out = render_plan(plan)
    assert "Skipped" in out
    assert "SAR" in out
