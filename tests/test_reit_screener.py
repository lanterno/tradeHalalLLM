"""Tests for the REIT-specific Shariah screener."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.halal.reit_screener import (
    DEFAULT_THRESHOLDS,
    REITFinancials,
    REITPropertyType,
    REITScreenStatus,
    REITThresholds,
    TenantCategory,
    TenantContribution,
    render_screen_result,
    screen_reit,
)


def _financials(
    *,
    symbol: str = "TEST",
    name: str = "Test REIT",
    property_type: REITPropertyType = REITPropertyType.RESIDENTIAL,
    market_cap_usd: float = 1_000_000_000.0,
    interest_bearing_debt_usd: float = 100_000_000.0,
    liquid_assets_usd: float = 50_000_000.0,
    rental_income_total_usd: float = 100_000_000.0,
    tenants: tuple[TenantContribution, ...] = (),
) -> REITFinancials:
    return REITFinancials(
        symbol=symbol,
        name=name,
        property_type=property_type,
        market_cap_usd=market_cap_usd,
        interest_bearing_debt_usd=interest_bearing_debt_usd,
        liquid_assets_usd=liquid_assets_usd,
        rental_income_total_usd=rental_income_total_usd,
        tenants=tenants,
    )


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------


def test_default_thresholds_match_aaoifi() -> None:
    assert DEFAULT_THRESHOLDS.debt_to_marketcap_pct == 33.0
    assert DEFAULT_THRESHOLDS.npi_to_total_income_pct == 5.0


def test_thresholds_reject_zero_debt() -> None:
    with pytest.raises(ValueError, match="debt_to_marketcap_pct"):
        REITThresholds(debt_to_marketcap_pct=0.0)


def test_thresholds_reject_negative_npi() -> None:
    with pytest.raises(ValueError, match="npi_to_total_income_pct"):
        REITThresholds(npi_to_total_income_pct=-1.0)


def test_thresholds_reject_above_100() -> None:
    with pytest.raises(ValueError):
        REITThresholds(debt_to_marketcap_pct=101.0)


# ---------------------------------------------------------------------------
# TenantContribution validation
# ---------------------------------------------------------------------------


def test_tenant_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TenantContribution(name="", category=TenantCategory.HALAL, rental_income_pct=10.0)


def test_tenant_rejects_whitespace_only_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TenantContribution(name="   ", category=TenantCategory.HALAL, rental_income_pct=10.0)


def test_tenant_rejects_negative_pct() -> None:
    with pytest.raises(ValueError, match="rental_income_pct"):
        TenantContribution(
            name="Bank A", category=TenantCategory.CONVENTIONAL_BANK, rental_income_pct=-1.0
        )


def test_tenant_rejects_pct_above_100() -> None:
    with pytest.raises(ValueError, match="rental_income_pct"):
        TenantContribution(
            name="Bank A", category=TenantCategory.CONVENTIONAL_BANK, rental_income_pct=101.0
        )


# ---------------------------------------------------------------------------
# REITFinancials validation
# ---------------------------------------------------------------------------


def test_financials_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        _financials(symbol="")


def test_financials_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _financials(name="")


def test_financials_rejects_negative_market_cap() -> None:
    with pytest.raises(ValueError, match="market_cap_usd"):
        _financials(market_cap_usd=-1.0)


def test_financials_rejects_negative_debt() -> None:
    with pytest.raises(ValueError, match="interest_bearing_debt_usd"):
        _financials(interest_bearing_debt_usd=-1.0)


def test_financials_rejects_negative_liquid() -> None:
    with pytest.raises(ValueError, match="liquid_assets_usd"):
        _financials(liquid_assets_usd=-1.0)


def test_financials_rejects_negative_rental_income() -> None:
    with pytest.raises(ValueError, match="rental_income_total_usd"):
        _financials(rental_income_total_usd=-1.0)


def test_financials_rejects_tenant_pct_sum_above_100() -> None:
    with pytest.raises(ValueError, match="sum to"):
        _financials(
            tenants=(
                TenantContribution(name="A", category=TenantCategory.HALAL, rental_income_pct=60.0),
                TenantContribution(name="B", category=TenantCategory.HALAL, rental_income_pct=50.0),
            )
        )


def test_financials_accepts_tenant_pct_sum_at_100() -> None:
    # exactly 100% is valid
    f = _financials(
        tenants=(
            TenantContribution(name="A", category=TenantCategory.HALAL, rental_income_pct=60.0),
            TenantContribution(name="B", category=TenantCategory.HALAL, rental_income_pct=40.0),
        )
    )
    assert sum(t.rental_income_pct for t in f.tenants) == pytest.approx(100.0)


def test_financials_accepts_tenant_pct_sum_below_100() -> None:
    # under-100% is the documented "halal other tenants implicit" case
    f = _financials(
        tenants=(
            TenantContribution(
                name="Suspicious Bank",
                category=TenantCategory.CONVENTIONAL_BANK,
                rental_income_pct=3.0,
            ),
        )
    )
    assert sum(t.rental_income_pct for t in f.tenants) == 3.0


# ---------------------------------------------------------------------------
# Screen status outcomes
# ---------------------------------------------------------------------------


def test_residential_with_low_debt_is_halal() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.RESIDENTIAL))
    assert result.status is REITScreenStatus.HALAL
    assert result.debt_to_marketcap_pct == pytest.approx(10.0)
    assert result.npi_pct == 0.0
    assert result.failures == ()


def test_office_with_low_debt_is_halal() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.OFFICE))
    assert result.status is REITScreenStatus.HALAL


def test_industrial_passes_without_tenants() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.INDUSTRIAL))
    assert result.status is REITScreenStatus.HALAL


def test_data_center_passes_without_tenants() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.DATA_CENTER))
    assert result.status is REITScreenStatus.HALAL


def test_self_storage_passes_without_tenants() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.SELF_STORAGE))
    assert result.status is REITScreenStatus.HALAL


def test_healthcare_passes_without_tenants() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.HEALTHCARE))
    assert result.status is REITScreenStatus.HALAL


def test_debt_above_threshold_is_not_halal() -> None:
    result = screen_reit(
        _financials(
            interest_bearing_debt_usd=400_000_000.0,  # 40% of market cap
        )
    )
    assert result.status is REITScreenStatus.NOT_HALAL
    assert any("debt" in f for f in result.failures)


def test_debt_at_threshold_inclusive_is_halal() -> None:
    # 33% debt — boundary is INCLUSIVE per the pinned semantics
    result = screen_reit(_financials(interest_bearing_debt_usd=330_000_000.0))
    assert result.status is REITScreenStatus.HALAL


def test_debt_just_above_threshold_is_not_halal() -> None:
    # the inclusivity pin's matching reverse: anything strictly above fails
    result = screen_reit(_financials(interest_bearing_debt_usd=330_000_001.0))
    assert result.status is REITScreenStatus.NOT_HALAL


def test_zero_market_cap_is_insufficient_data() -> None:
    result = screen_reit(_financials(market_cap_usd=0.0))
    assert result.status is REITScreenStatus.INSUFFICIENT_DATA
    assert result.debt_to_marketcap_pct is None
    assert result.npi_pct is None


def test_liquid_assets_above_70_pct_fails() -> None:
    # mis-classified financial-shell as REIT
    result = screen_reit(
        _financials(
            liquid_assets_usd=800_000_000.0,  # 80% of market cap
        )
    )
    assert result.status is REITScreenStatus.NOT_HALAL
    assert any("liquid assets" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Tenant-mix outcomes
# ---------------------------------------------------------------------------


def test_retail_mall_without_tenants_is_doubtful() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.RETAIL_MALL))
    assert result.status is REITScreenStatus.DOUBTFUL
    assert any("tenant breakdown missing" in w for w in result.warnings)


def test_diversified_without_tenants_is_doubtful() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.DIVERSIFIED))
    assert result.status is REITScreenStatus.DOUBTFUL


def test_retail_mall_with_clean_tenants_is_halal() -> None:
    tenants = (
        TenantContribution(
            name="Halal Goods Co.", category=TenantCategory.HALAL, rental_income_pct=80.0
        ),
        TenantContribution(name="Bookstore", category=TenantCategory.HALAL, rental_income_pct=15.0),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.RETAIL_MALL, tenants=tenants))
    assert result.status is REITScreenStatus.HALAL
    assert result.npi_pct == 0.0
    assert result.purification_pct == 0.0


def test_retail_mall_with_marginal_npi_is_halal_with_purification() -> None:
    # 3% from a conventional bank tenant → under the 5% cap, passes,
    # but the operator must purify 3% of dividends
    tenants = (
        TenantContribution(
            name="Bank Branch",
            category=TenantCategory.CONVENTIONAL_BANK,
            rental_income_pct=3.0,
        ),
        TenantContribution(
            name="Halal Anchor", category=TenantCategory.HALAL, rental_income_pct=80.0
        ),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.RETAIL_MALL, tenants=tenants))
    assert result.status is REITScreenStatus.HALAL
    assert result.npi_pct == pytest.approx(3.0)
    assert result.purification_pct == pytest.approx(3.0)


def test_retail_mall_with_npi_above_threshold_is_not_halal() -> None:
    tenants = (
        TenantContribution(
            name="Casino Wing",
            category=TenantCategory.ALCOHOL_GAMBLING,
            rental_income_pct=8.0,
        ),
        TenantContribution(
            name="Halal Anchor", category=TenantCategory.HALAL, rental_income_pct=80.0
        ),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.RETAIL_MALL, tenants=tenants))
    assert result.status is REITScreenStatus.NOT_HALAL
    assert any("npi" in f for f in result.failures)


def test_retail_mall_npi_at_threshold_inclusive_is_halal() -> None:
    # exactly 5% npi — boundary inclusive
    tenants = (
        TenantContribution(
            name="Bank",
            category=TenantCategory.CONVENTIONAL_BANK,
            rental_income_pct=5.0,
        ),
        TenantContribution(name="Halal", category=TenantCategory.HALAL, rental_income_pct=80.0),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.RETAIL_MALL, tenants=tenants))
    assert result.status is REITScreenStatus.HALAL
    assert result.purification_pct == pytest.approx(5.0)


def test_inherently_halal_property_with_marginal_npi_purifies() -> None:
    # an office tower with a conventional-insurance tenant
    tenants = (
        TenantContribution(
            name="Insurer Inc.",
            category=TenantCategory.INSURANCE_CONVENTIONAL,
            rental_income_pct=2.5,
        ),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.OFFICE, tenants=tenants))
    assert result.status is REITScreenStatus.HALAL
    assert result.npi_pct == pytest.approx(2.5)
    assert result.purification_pct == pytest.approx(2.5)


def test_inherently_halal_property_with_excess_npi_is_not_halal() -> None:
    tenants = (
        TenantContribution(
            name="Tobacco Distributor",
            category=TenantCategory.TOBACCO,
            rental_income_pct=10.0,
        ),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.OFFICE, tenants=tenants))
    assert result.status is REITScreenStatus.NOT_HALAL


# ---------------------------------------------------------------------------
# Inherently-doubtful property types
# ---------------------------------------------------------------------------


def test_hotel_with_no_tenants_is_doubtful() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.HOTEL))
    assert result.status is REITScreenStatus.DOUBTFUL
    assert any("hotel" in w.lower() for w in result.warnings)


def test_hotel_with_clean_tenants_still_doubtful() -> None:
    # even with a halal tenant list, the underlying hospitality business
    # raises shariah questions the screener can't resolve
    tenants = (
        TenantContribution(
            name="Halal Cafe", category=TenantCategory.HALAL, rental_income_pct=100.0
        ),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.HOTEL, tenants=tenants))
    assert result.status is REITScreenStatus.DOUBTFUL


def test_hotel_with_excess_npi_is_not_halal() -> None:
    tenants = (
        TenantContribution(
            name="Bar",
            category=TenantCategory.ALCOHOL_GAMBLING,
            rental_income_pct=15.0,
        ),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.HOTEL, tenants=tenants))
    assert result.status is REITScreenStatus.NOT_HALAL


def test_specialty_property_is_doubtful() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.SPECIALTY))
    assert result.status is REITScreenStatus.DOUBTFUL


# ---------------------------------------------------------------------------
# Multiple-failure aggregation
# ---------------------------------------------------------------------------


def test_multiple_failures_all_listed() -> None:
    tenants = (
        TenantContribution(
            name="Casino",
            category=TenantCategory.ALCOHOL_GAMBLING,
            rental_income_pct=20.0,
        ),
    )
    result = screen_reit(
        _financials(
            property_type=REITPropertyType.RETAIL_MALL,
            interest_bearing_debt_usd=500_000_000.0,  # 50%
            liquid_assets_usd=800_000_000.0,  # 80%
            tenants=tenants,
        )
    )
    assert result.status is REITScreenStatus.NOT_HALAL
    # at least three distinct rule failures captured
    assert len(result.failures) >= 3


# ---------------------------------------------------------------------------
# Threshold customisation
# ---------------------------------------------------------------------------


def test_stricter_debt_threshold_flips_verdict() -> None:
    f = _financials(interest_bearing_debt_usd=320_000_000.0)  # 32%
    assert screen_reit(f).status is REITScreenStatus.HALAL
    strict = REITThresholds(debt_to_marketcap_pct=30.0)
    assert screen_reit(f, thresholds=strict).status is REITScreenStatus.NOT_HALAL


def test_stricter_npi_threshold_flips_verdict() -> None:
    tenants = (
        TenantContribution(
            name="Bank",
            category=TenantCategory.CONVENTIONAL_BANK,
            rental_income_pct=4.0,
        ),
    )
    f = _financials(property_type=REITPropertyType.RETAIL_MALL, tenants=tenants)
    assert screen_reit(f).status is REITScreenStatus.HALAL
    strict = REITThresholds(npi_to_total_income_pct=3.0)
    assert screen_reit(f, thresholds=strict).status is REITScreenStatus.NOT_HALAL


# ---------------------------------------------------------------------------
# Frozen-dataclass invariants
# ---------------------------------------------------------------------------


def test_screen_result_is_frozen() -> None:
    result = screen_reit(_financials())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = REITScreenStatus.NOT_HALAL  # type: ignore[misc]


def test_financials_is_frozen() -> None:
    f = _financials()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.market_cap_usd = 1.0  # type: ignore[misc]


def test_thresholds_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THRESHOLDS.debt_to_marketcap_pct = 50.0  # type: ignore[misc]


def test_tenant_is_frozen() -> None:
    t = TenantContribution(name="A", category=TenantCategory.HALAL, rental_income_pct=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.rental_income_pct = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Status enum string values pinned for JSON serialisation
# ---------------------------------------------------------------------------


def test_status_string_values_pinned() -> None:
    assert REITScreenStatus.HALAL.value == "halal"
    assert REITScreenStatus.NOT_HALAL.value == "not_halal"
    assert REITScreenStatus.DOUBTFUL.value == "doubtful"
    assert REITScreenStatus.INSUFFICIENT_DATA.value == "insufficient_data"


def test_property_type_string_values_pinned() -> None:
    assert REITPropertyType.RESIDENTIAL.value == "residential"
    assert REITPropertyType.RETAIL_MALL.value == "retail_mall"
    assert REITPropertyType.HOTEL.value == "hotel"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_halal_result() -> None:
    result = screen_reit(_financials())
    text = render_screen_result(result)
    assert "✅" in text
    assert "TEST" in text
    assert "HALAL" in text
    assert "debt/marketcap" in text


def test_render_not_halal_result() -> None:
    result = screen_reit(_financials(interest_bearing_debt_usd=500_000_000.0))
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text
    assert "failures:" in text


def test_render_doubtful_result() -> None:
    result = screen_reit(_financials(property_type=REITPropertyType.HOTEL))
    text = render_screen_result(result)
    assert "⚠️" in text
    assert "DOUBTFUL" in text
    assert "warnings:" in text


def test_render_insufficient_data_result() -> None:
    result = screen_reit(_financials(market_cap_usd=0.0))
    text = render_screen_result(result)
    assert "❓" in text
    assert "INSUFFICIENT_DATA" in text


def test_render_includes_purification_when_nonzero() -> None:
    tenants = (
        TenantContribution(
            name="Bank",
            category=TenantCategory.CONVENTIONAL_BANK,
            rental_income_pct=2.0,
        ),
    )
    result = screen_reit(_financials(property_type=REITPropertyType.OFFICE, tenants=tenants))
    text = render_screen_result(result)
    assert "purification" in text


def test_render_omits_purification_when_zero() -> None:
    result = screen_reit(_financials())  # clean residential, no NPI
    text = render_screen_result(result)
    assert "purification" not in text


# ---------------------------------------------------------------------------
# End-to-end realistic case
# ---------------------------------------------------------------------------


def test_real_world_simon_property_like_case() -> None:
    """A retail mall with mixed tenants: anchor halal, sub-tenant
    bank, sub-tenant cinema → multiple non-permissible exposure
    summing to 8% > the 5% cap, NOT_HALAL.
    """

    tenants = (
        TenantContribution(
            name="Halal Anchor Department Store",
            category=TenantCategory.HALAL,
            rental_income_pct=60.0,
        ),
        TenantContribution(
            name="Conventional Bank Branch",
            category=TenantCategory.CONVENTIONAL_BANK,
            rental_income_pct=4.0,
        ),
        TenantContribution(
            name="Cinema Multiplex",
            category=TenantCategory.CINEMA,
            rental_income_pct=4.0,
        ),
        TenantContribution(
            name="Halal Restaurants Cluster",
            category=TenantCategory.HALAL,
            rental_income_pct=20.0,
        ),
    )
    result = screen_reit(
        _financials(
            symbol="SPGX",
            name="Simon-Like Property Trust",
            property_type=REITPropertyType.RETAIL_MALL,
            interest_bearing_debt_usd=200_000_000.0,  # 20% — passes
            tenants=tenants,
        )
    )
    assert result.status is REITScreenStatus.NOT_HALAL
    assert result.npi_pct == pytest.approx(8.0)
    assert any("npi" in f for f in result.failures)
