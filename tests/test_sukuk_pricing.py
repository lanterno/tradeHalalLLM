"""Tests for markets/sukuk_pricing.py — Round-5 Wave 3.B."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_17 import SukukType
from halal_trader.markets.sukuk_pricing import (
    Cashflow,
    CurvePoint,
    PricingResult,
    ProfitRateCurve,
    Sukuk,
    price_sukuk,
    render_pricing,
    yield_to_maturity,
)


def _flat_curve(rate: float = 0.05) -> ProfitRateCurve:
    return ProfitRateCurve(
        points=(
            CurvePoint(tenor_years=0.25, rate=rate),
            CurvePoint(tenor_years=1.0, rate=rate),
            CurvePoint(tenor_years=5.0, rate=rate),
            CurvePoint(tenor_years=10.0, rate=rate),
        )
    )


def _ijara_sukuk(face: float = 1000.0, profit_rate: float = 0.05, years: int = 5) -> Sukuk:
    """5y annual-pay Ijara sukuk."""
    cashflows: list[Cashflow] = []
    annual_profit = face * profit_rate
    for y in range(1, years + 1):
        amount = annual_profit + (face if y == years else 0)
        cashflows.append(Cashflow(amount=amount, time_years=float(y)))
    return Sukuk(
        issuer="GovOfMalaysia",
        sukuk_type=SukukType.IJARA,
        cashflows=tuple(cashflows),
        face_value=face,
    )


# --- Curve validation -------------------------------------------------------


def test_curve_point_validation_zero_tenor_rejected():
    with pytest.raises(ValueError):
        CurvePoint(tenor_years=0.0, rate=0.05)


def test_curve_point_validation_negative_tenor_rejected():
    with pytest.raises(ValueError):
        CurvePoint(tenor_years=-1.0, rate=0.05)


def test_curve_point_unreasonable_rate_rejected():
    with pytest.raises(ValueError):
        CurvePoint(tenor_years=1.0, rate=2.0)


def test_curve_empty_rejected():
    with pytest.raises(ValueError):
        ProfitRateCurve(points=())


def test_curve_unsorted_rejected():
    with pytest.raises(ValueError):
        ProfitRateCurve(
            points=(
                CurvePoint(tenor_years=2.0, rate=0.04),
                CurvePoint(tenor_years=1.0, rate=0.03),
            )
        )


def test_curve_duplicate_tenor_rejected():
    with pytest.raises(ValueError):
        ProfitRateCurve(
            points=(
                CurvePoint(tenor_years=1.0, rate=0.04),
                CurvePoint(tenor_years=1.0, rate=0.05),
            )
        )


def test_curve_empty_currency_rejected():
    with pytest.raises(ValueError):
        ProfitRateCurve(points=(CurvePoint(tenor_years=1.0, rate=0.05),), base_currency="")


def test_curve_immutable():
    c = _flat_curve()
    with pytest.raises(AttributeError):
        c.base_currency = "EUR"  # type: ignore[misc]


# --- Curve interpolation ----------------------------------------------------


def test_curve_interpolate_at_anchor():
    c = _flat_curve(0.05)
    assert c.interpolate(1.0) == pytest.approx(0.05)


def test_curve_interpolate_between_points():
    c = ProfitRateCurve(
        points=(
            CurvePoint(tenor_years=1.0, rate=0.04),
            CurvePoint(tenor_years=2.0, rate=0.06),
        )
    )
    # Linear between (1,0.04) and (2,0.06) → 1.5y → 0.05
    assert c.interpolate(1.5) == pytest.approx(0.05)


def test_curve_interpolate_below_lowest_flat():
    c = ProfitRateCurve(
        points=(
            CurvePoint(tenor_years=1.0, rate=0.04),
            CurvePoint(tenor_years=2.0, rate=0.06),
        )
    )
    assert c.interpolate(0.5) == pytest.approx(0.04)


def test_curve_interpolate_above_highest_flat():
    c = ProfitRateCurve(
        points=(
            CurvePoint(tenor_years=1.0, rate=0.04),
            CurvePoint(tenor_years=2.0, rate=0.06),
        )
    )
    assert c.interpolate(10.0) == pytest.approx(0.06)


def test_curve_interpolate_negative_tenor_rejected():
    c = _flat_curve()
    with pytest.raises(ValueError):
        c.interpolate(-1.0)


# --- Cashflow + Sukuk validation -------------------------------------------


def test_cashflow_zero_time_rejected():
    with pytest.raises(ValueError):
        Cashflow(amount=10.0, time_years=0.0)


def test_sukuk_empty_issuer_rejected():
    with pytest.raises(ValueError):
        Sukuk(
            issuer="",
            sukuk_type=SukukType.IJARA,
            cashflows=(Cashflow(amount=10.0, time_years=1.0),),
            face_value=1000.0,
        )


def test_sukuk_negative_face_rejected():
    with pytest.raises(ValueError):
        Sukuk(
            issuer="X",
            sukuk_type=SukukType.IJARA,
            cashflows=(Cashflow(amount=10.0, time_years=1.0),),
            face_value=-1.0,
        )


def test_sukuk_no_cashflows_rejected():
    with pytest.raises(ValueError):
        Sukuk(
            issuer="X",
            sukuk_type=SukukType.IJARA,
            cashflows=(),
            face_value=1000.0,
        )


def test_sukuk_unsorted_cashflows_rejected():
    with pytest.raises(ValueError):
        Sukuk(
            issuer="X",
            sukuk_type=SukukType.IJARA,
            cashflows=(
                Cashflow(amount=10.0, time_years=2.0),
                Cashflow(amount=10.0, time_years=1.0),
            ),
            face_value=1000.0,
        )


# --- Pricing math -----------------------------------------------------------


def test_pricing_at_par_when_curve_matches_coupon():
    """5y 5% Ijara on a flat 5% curve should price at par (within DCF discounting)."""
    sukuk = _ijara_sukuk(face=1000.0, profit_rate=0.05, years=5)
    curve = _flat_curve(0.05)
    result = price_sukuk(sukuk, curve)
    # Continuous compounding ≠ exact par; should be close to face value.
    assert result.present_value == pytest.approx(1000.0, rel=0.05)


def test_pricing_higher_curve_discounts_below_par():
    sukuk = _ijara_sukuk(face=1000.0, profit_rate=0.05, years=5)
    curve = _flat_curve(0.10)  # higher discount → lower PV
    result = price_sukuk(sukuk, curve)
    assert result.present_value < 1000.0


def test_pricing_lower_curve_pushes_above_par():
    sukuk = _ijara_sukuk(face=1000.0, profit_rate=0.05, years=5)
    curve = _flat_curve(0.01)
    result = price_sukuk(sukuk, curve)
    assert result.present_value > 1000.0


def test_pricing_murabaha_returns_face_value():
    sukuk = Sukuk(
        issuer="X",
        sukuk_type=SukukType.MURABAHA,
        cashflows=(Cashflow(amount=1100.0, time_years=1.0),),
        face_value=1000.0,
    )
    result = price_sukuk(sukuk, _flat_curve())
    assert result.present_value == 1000.0
    assert result.secondary_tradable is False
    assert result.used_curve_rates == ()


def test_pricing_salam_returns_face_value():
    sukuk = Sukuk(
        issuer="X",
        sukuk_type=SukukType.SALAM,
        cashflows=(Cashflow(amount=1100.0, time_years=1.0),),
        face_value=1000.0,
    )
    result = price_sukuk(sukuk, _flat_curve())
    assert result.present_value == 1000.0
    assert result.secondary_tradable is False


def test_pricing_returns_curve_rates_used():
    sukuk = _ijara_sukuk(years=3)
    result = price_sukuk(sukuk, _flat_curve(0.05))
    assert len(result.used_curve_rates) == 3
    assert all(r == pytest.approx(0.05) for r in result.used_curve_rates)


def test_pricing_negative_pv_rejected():
    with pytest.raises(ValueError):
        PricingResult(
            issuer="x",
            present_value=-1.0,
            secondary_tradable=True,
            used_curve_rates=(0.05,),
            accrued_profit=0.0,
        )


# --- YTM bisection ----------------------------------------------------------


def test_ytm_recovers_curve_yield_at_par():
    sukuk = _ijara_sukuk(face=1000.0, profit_rate=0.05, years=5)
    curve = _flat_curve(0.05)
    pv = price_sukuk(sukuk, curve).present_value
    ytm = yield_to_maturity(sukuk, market_price=pv)
    assert ytm == pytest.approx(0.05, abs=1e-4)


def test_ytm_above_pv_implies_lower_yield():
    sukuk = _ijara_sukuk(face=1000.0, profit_rate=0.05, years=5)
    curve = _flat_curve(0.05)
    pv = price_sukuk(sukuk, curve).present_value
    ytm_high = yield_to_maturity(sukuk, market_price=pv * 1.05)
    assert ytm_high < 0.05


def test_ytm_below_pv_implies_higher_yield():
    sukuk = _ijara_sukuk(face=1000.0, profit_rate=0.05, years=5)
    curve = _flat_curve(0.05)
    pv = price_sukuk(sukuk, curve).present_value
    ytm_low = yield_to_maturity(sukuk, market_price=pv * 0.95)
    assert ytm_low > 0.05


def test_ytm_zero_market_price_rejected():
    sukuk = _ijara_sukuk()
    with pytest.raises(ValueError):
        yield_to_maturity(sukuk, market_price=0.0)


def test_ytm_murabaha_rejected():
    sukuk = Sukuk(
        issuer="X",
        sukuk_type=SukukType.MURABAHA,
        cashflows=(Cashflow(amount=1100.0, time_years=1.0),),
        face_value=1000.0,
    )
    with pytest.raises(ValueError):
        yield_to_maturity(sukuk, market_price=1000.0)


# --- Render -----------------------------------------------------------------


def test_render_tradable():
    sukuk = _ijara_sukuk()
    result = price_sukuk(sukuk, _flat_curve())
    out = render_pricing(result)
    assert "GovOfMalaysia" in out
    assert "💰" in out


def test_render_non_tradable():
    sukuk = Sukuk(
        issuer="MurabahaCo",
        sukuk_type=SukukType.MURABAHA,
        cashflows=(Cashflow(amount=1100.0, time_years=1.0),),
        face_value=1000.0,
    )
    result = price_sukuk(sukuk, _flat_curve())
    out = render_pricing(result)
    assert "non-tradable" in out
    assert "⏸" in out


# --- E2E --------------------------------------------------------------------


def test_e2e_ijara_pricing_and_ytm_roundtrip():
    sukuk = _ijara_sukuk(face=1000.0, profit_rate=0.04, years=10)
    curve = _flat_curve(0.06)  # higher discount → discount price
    result = price_sukuk(sukuk, curve)
    assert result.present_value < 1000.0
    ytm = yield_to_maturity(sukuk, market_price=result.present_value)
    # YTM should be near curve's flat 6%
    assert ytm == pytest.approx(0.06, abs=1e-3)


def test_replay_consistency():
    sukuk = _ijara_sukuk()
    curve = _flat_curve()
    a = price_sukuk(sukuk, curve)
    b = price_sukuk(sukuk, curve)
    assert a == b
