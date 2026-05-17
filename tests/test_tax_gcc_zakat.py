"""Tests for core/tax_gcc_zakat.py — Round-5 Wave 18.D."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.tax_gcc_zakat import (
    GccCountry,
    GccZakatReport,
    build_report,
    default_currency_for,
    default_policy_for,
    default_zakat_rate_for,
    render_report,
)
from halal_trader.halal.zakat import NisabBasis, ZakatCalculation


def _calc(net: float = 100000.0, owed: float = 2500.0) -> ZakatCalculation:
    return ZakatCalculation(
        net_assets=net,
        nisab_value=5000.0,
        meets_nisab=True,
        zakat_owed=owed,
        basis_used=NisabBasis.SILVER,
        reporting_currency="SAR",
        hawl_due_date=date(2027, 1, 1),
    )


# --- Validation -------------------------------------------


def test_country_string_values():
    assert GccCountry.SAUDI.value == "saudi"
    assert GccCountry.UAE.value == "uae"
    assert GccCountry.BAHRAIN.value == "bahrain"
    assert GccCountry.KUWAIT.value == "kuwait"
    assert GccCountry.QATAR.value == "qatar"
    assert GccCountry.OMAN.value == "oman"


def test_default_currency_saudi_sar():
    assert default_currency_for(GccCountry.SAUDI) == "SAR"


def test_default_currency_uae_aed():
    assert default_currency_for(GccCountry.UAE) == "AED"


def test_default_zakat_rate_2_5():
    for c in GccCountry:
        assert default_zakat_rate_for(c) == 0.025


def test_default_policy_uses_country_rate():
    p = default_policy_for(GccCountry.SAUDI)
    assert p.zakat_rate == 0.025


def test_report_email_handle_rejected():
    with pytest.raises(ValueError):
        GccZakatReport(
            country=GccCountry.SAUDI,
            operator_handle="ops@example.com",
            reporting_currency="SAR",
            reporting_period_start=date(2026, 1, 1),
            reporting_period_end=date(2026, 12, 31),
            zakat_calculation=_calc(),
            additional_charity_paid=0,
        )


def test_report_period_inversion_rejected():
    with pytest.raises(ValueError):
        GccZakatReport(
            country=GccCountry.SAUDI,
            operator_handle="op-1",
            reporting_currency="SAR",
            reporting_period_start=date(2026, 12, 31),
            reporting_period_end=date(2026, 1, 1),
            zakat_calculation=_calc(),
            additional_charity_paid=0,
        )


def test_report_negative_charity_rejected():
    with pytest.raises(ValueError):
        GccZakatReport(
            country=GccCountry.SAUDI,
            operator_handle="op-1",
            reporting_currency="SAR",
            reporting_period_start=date(2026, 1, 1),
            reporting_period_end=date(2026, 12, 31),
            zakat_calculation=_calc(),
            additional_charity_paid=-1,
        )


def test_report_empty_currency_rejected():
    with pytest.raises(ValueError):
        GccZakatReport(
            country=GccCountry.SAUDI,
            operator_handle="op-1",
            reporting_currency="",
            reporting_period_start=date(2026, 1, 1),
            reporting_period_end=date(2026, 12, 31),
            zakat_calculation=_calc(),
            additional_charity_paid=0,
        )


# --- Build -----------------------------------------------


def test_build_uses_default_currency():
    report = build_report(
        country=GccCountry.UAE,
        operator_handle="op-1",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(),
    )
    assert report.reporting_currency == "AED"


def test_build_overrides_currency():
    report = build_report(
        country=GccCountry.SAUDI,
        operator_handle="op-1",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(),
        reporting_currency="USD",
    )
    assert report.reporting_currency == "USD"


def test_build_records_charity_paid():
    report = build_report(
        country=GccCountry.SAUDI,
        operator_handle="op-1",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(),
        additional_charity_paid=500.0,
    )
    assert report.additional_charity_paid == 500.0


# --- Render -----------------------------------------------


def test_render_includes_country_and_amounts():
    report = build_report(
        country=GccCountry.SAUDI,
        operator_handle="op-1",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(net=100000, owed=2500),
    )
    out = render_report(report)
    assert "SAUDI" in out
    assert "100000" in out
    assert "2500" in out
    assert "SAR" in out


def test_render_no_secret_leak():
    report = build_report(
        country=GccCountry.SAUDI,
        operator_handle="op-1",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(),
    )
    out = render_report(report)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization", "IBAN", "Bank-"):
        assert token not in out


# --- E2E ----------------------------------------------


def test_e2e_uae_zakat_only_report():
    report = build_report(
        country=GccCountry.UAE,
        operator_handle="emirates-trader",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(),
        additional_charity_paid=1000.0,
    )
    out = render_report(report)
    assert "UAE" in out
    assert "AED" in out
    assert "additional_charity_paid: 1000" in out


def test_replay_consistency():
    a = build_report(
        country=GccCountry.SAUDI,
        operator_handle="op-1",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(),
    )
    b = build_report(
        country=GccCountry.SAUDI,
        operator_handle="op-1",
        reporting_period_start=date(2026, 1, 1),
        reporting_period_end=date(2026, 12, 31),
        zakat_calculation=_calc(),
    )
    assert a == b
