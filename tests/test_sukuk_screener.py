"""Tests for the sukuk Shariah screener."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.halal.sukuk_screener import (
    DEFAULT_THRESHOLDS,
    AssetClass,
    CouponType,
    IssuerType,
    SukukIssue,
    SukukScreenResult,
    SukukStructure,
    SukukThresholds,
    SukukVerdict,
    render_screen_result,
    screen_sukuk,
)


def _issue(
    *,
    isin: str = "XS1234567890",
    issuer_name: str = "Saudi Sovereign",
    issuer_type: IssuerType = IssuerType.SOVEREIGN,
    structure: SukukStructure = SukukStructure.IJARAH,
    asset_class: AssetClass = AssetClass.REAL_ESTATE,
    coupon_type: CouponType = CouponType.RENTAL_INCOME,
    is_aaoifi_certified: bool = True,
    is_asset_backed: bool = True,
    maturity_days: int = 1825,  # 5y
    expected_yield_pct: float = 4.5,
) -> SukukIssue:
    return SukukIssue(
        isin=isin,
        issuer_name=issuer_name,
        issuer_type=issuer_type,
        structure=structure,
        asset_class=asset_class,
        coupon_type=coupon_type,
        is_aaoifi_certified=is_aaoifi_certified,
        is_asset_backed=is_asset_backed,
        maturity_days=maturity_days,
        expected_yield_pct=expected_yield_pct,
    )


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------


def test_default_thresholds() -> None:
    assert DEFAULT_THRESHOLDS.require_aaoifi_certified is True
    assert DEFAULT_THRESHOLDS.accept_murabaha is True
    assert DEFAULT_THRESHOLDS.accept_asset_based is True
    assert DEFAULT_THRESHOLDS.min_maturity_days == 1


def test_thresholds_reject_negative_min_maturity() -> None:
    with pytest.raises(ValueError, match="min_maturity_days"):
        SukukThresholds(min_maturity_days=-1)


def test_thresholds_accept_zero_min_maturity() -> None:
    """Pin: zero is valid (operator wants to include any tenor)."""

    t = SukukThresholds(min_maturity_days=0)
    assert t.min_maturity_days == 0


# ---------------------------------------------------------------------------
# SukukIssue validation
# ---------------------------------------------------------------------------


def test_issue_rejects_empty_isin() -> None:
    with pytest.raises(ValueError, match="isin"):
        _issue(isin="")


def test_issue_rejects_empty_issuer_name() -> None:
    with pytest.raises(ValueError, match="issuer_name"):
        _issue(issuer_name="")


def test_issue_rejects_negative_maturity() -> None:
    with pytest.raises(ValueError, match="maturity_days"):
        _issue(maturity_days=-1)


def test_issue_rejects_negative_yield() -> None:
    with pytest.raises(ValueError, match="expected_yield_pct"):
        _issue(expected_yield_pct=-1.0)


# ---------------------------------------------------------------------------
# Hard rejections — unconditional NOT_HALAL gates
# ---------------------------------------------------------------------------


def test_interest_coupon_is_not_halal() -> None:
    """The riba red line — INTEREST coupon is unconditionally NOT_HALAL."""

    result = screen_sukuk(_issue(coupon_type=CouponType.INTEREST))
    assert result.verdict is SukukVerdict.NOT_HALAL
    assert any("interest" in f or "riba" in f for f in result.failures)


def test_prohibited_operations_underlying_is_not_halal() -> None:
    result = screen_sukuk(_issue(asset_class=AssetClass.PROHIBITED_OPERATIONS))
    assert result.verdict is SukukVerdict.NOT_HALAL
    assert any("prohibited" in f for f in result.failures)


def test_financial_receivables_underlying_is_not_halal() -> None:
    """Pin: financial-receivables = conventional debt dressed as sukuk."""

    result = screen_sukuk(_issue(asset_class=AssetClass.FINANCIAL_RECEIVABLES))
    assert result.verdict is SukukVerdict.NOT_HALAL
    assert any("conventional debt" in f for f in result.failures)


def test_interest_coupon_overrides_other_clean_flags() -> None:
    """Pin: even with everything else clean, INTEREST coupon → NOT_HALAL."""

    result = screen_sukuk(
        _issue(
            issuer_type=IssuerType.SOVEREIGN,
            structure=SukukStructure.IJARAH,
            asset_class=AssetClass.REAL_ESTATE,
            coupon_type=CouponType.INTEREST,
            is_aaoifi_certified=True,
            is_asset_backed=True,
        )
    )
    assert result.verdict is SukukVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# INSUFFICIENT_DATA — UNKNOWN asset class
# ---------------------------------------------------------------------------


def test_unknown_asset_class_returns_insufficient_data() -> None:
    result = screen_sukuk(_issue(asset_class=AssetClass.UNKNOWN))
    assert result.verdict is SukukVerdict.INSUFFICIENT_DATA
    assert any("UNKNOWN" in w for w in result.warnings)


def test_unknown_asset_class_returns_insufficient_data_even_with_clean_flags() -> None:
    result = screen_sukuk(
        _issue(
            asset_class=AssetClass.UNKNOWN,
            structure=SukukStructure.IJARAH,
            coupon_type=CouponType.RENTAL_INCOME,
            is_aaoifi_certified=True,
        )
    )
    assert result.verdict is SukukVerdict.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# HALAL — every check passes
# ---------------------------------------------------------------------------


def test_clean_ijarah_sovereign_real_estate_is_halal() -> None:
    """Default ijarah-real-estate-sovereign-rental-aaoifi-asset-backed → HALAL."""

    result = screen_sukuk(_issue())
    assert result.verdict is SukukVerdict.HALAL
    assert result.failures == ()
    assert result.warnings == ()


def test_clean_musharaka_supranational_infrastructure_is_halal() -> None:
    result = screen_sukuk(
        _issue(
            issuer_name="Islamic Development Bank",
            issuer_type=IssuerType.SUPRANATIONAL,
            structure=SukukStructure.MUSHARAKA,
            asset_class=AssetClass.INFRASTRUCTURE,
            coupon_type=CouponType.PROFIT_SHARE,
        )
    )
    assert result.verdict is SukukVerdict.HALAL


def test_clean_wakala_corporate_halal_is_halal() -> None:
    result = screen_sukuk(
        _issue(
            issuer_name="Halal Real Estate Co.",
            issuer_type=IssuerType.CORPORATE_HALAL,
            structure=SukukStructure.WAKALA,
            asset_class=AssetClass.REAL_ESTATE,
            coupon_type=CouponType.PROFIT_SHARE,
        )
    )
    assert result.verdict is SukukVerdict.HALAL


def test_fixed_profit_rate_coupon_is_halal_when_aaoifi_compliant() -> None:
    """Pin: FIXED_PROFIT_RATE is HALAL when contractually structured —
    AAOIFI permits a benchmark-pegged rental even though the rate
    references LIBOR or similar. The pin guards against a sloppy
    implementation that would conflate "uses LIBOR" with "is interest".
    """

    result = screen_sukuk(_issue(coupon_type=CouponType.FIXED_PROFIT_RATE))
    assert result.verdict is SukukVerdict.HALAL


def test_clean_istisna_is_halal() -> None:
    result = screen_sukuk(
        _issue(
            structure=SukukStructure.ISTISNA,
            asset_class=AssetClass.AIRCRAFT,
            coupon_type=CouponType.PROFIT_SHARE,
        )
    )
    assert result.verdict is SukukVerdict.HALAL


def test_clean_salam_is_halal() -> None:
    result = screen_sukuk(
        _issue(
            structure=SukukStructure.SALAM,
            asset_class=AssetClass.UTILITIES,
            coupon_type=CouponType.PROFIT_SHARE,
        )
    )
    assert result.verdict is SukukVerdict.HALAL


def test_clean_hybrid_is_halal() -> None:
    result = screen_sukuk(_issue(structure=SukukStructure.HYBRID))
    assert result.verdict is SukukVerdict.HALAL


def test_every_halal_asset_class_passes_with_clean_flags() -> None:
    halal_classes = [
        AssetClass.REAL_ESTATE,
        AssetClass.INFRASTRUCTURE,
        AssetClass.EQUIPMENT,
        AssetClass.AIRCRAFT,
        AssetClass.VEHICLES,
        AssetClass.POWER_PLANTS,
        AssetClass.UTILITIES,
    ]
    for c in halal_classes:
        assert screen_sukuk(_issue(asset_class=c)).verdict is SukukVerdict.HALAL, c


# ---------------------------------------------------------------------------
# DOUBTFUL — soft warnings drive doubtful
# ---------------------------------------------------------------------------


def test_not_aaoifi_certified_is_doubtful() -> None:
    result = screen_sukuk(_issue(is_aaoifi_certified=False))
    assert result.verdict is SukukVerdict.DOUBTFUL
    assert any("AAOIFI-certified" in w for w in result.warnings)


def test_conventional_bank_issuer_is_doubtful() -> None:
    result = screen_sukuk(_issue(issuer_type=IssuerType.CONVENTIONAL_BANK))
    assert result.verdict is SukukVerdict.DOUBTFUL
    assert any("conventional bank" in w for w in result.warnings)


def test_corporate_mixed_issuer_is_doubtful() -> None:
    result = screen_sukuk(_issue(issuer_type=IssuerType.CORPORATE_MIXED))
    assert result.verdict is SukukVerdict.DOUBTFUL
    assert any("mixed business" in w for w in result.warnings)


def test_short_maturity_is_doubtful() -> None:
    """Pin: very short tenor flagged as money-market-shaped."""

    result = screen_sukuk(_issue(maturity_days=0))
    # default min_maturity_days=1 means 0d trips the warning
    assert result.verdict is SukukVerdict.DOUBTFUL
    assert any("money-market" in w for w in result.warnings)


def test_murabaha_under_strict_mode_is_doubtful() -> None:
    """Strict-Hanafi operators reject MURABAHA → DOUBTFUL."""

    strict = SukukThresholds(accept_murabaha=False)
    result = screen_sukuk(_issue(structure=SukukStructure.MURABAHA), thresholds=strict)
    assert result.verdict is SukukVerdict.DOUBTFUL
    assert any("murabaha" in w.lower() for w in result.warnings)


def test_murabaha_default_mode_is_halal() -> None:
    """Pin: under default thresholds, MURABAHA passes (AAOIFI-permitted)."""

    result = screen_sukuk(_issue(structure=SukukStructure.MURABAHA))
    assert result.verdict is SukukVerdict.HALAL


def test_asset_based_under_strict_mode_is_doubtful() -> None:
    strict = SukukThresholds(accept_asset_based=False)
    result = screen_sukuk(_issue(is_asset_backed=False), thresholds=strict)
    assert result.verdict is SukukVerdict.DOUBTFUL
    assert any("asset-based" in w for w in result.warnings)


def test_asset_based_default_mode_is_halal() -> None:
    """Pin: under default thresholds, asset-based sukuk passes."""

    result = screen_sukuk(_issue(is_asset_backed=False))
    assert result.verdict is SukukVerdict.HALAL


def test_aaoifi_uncertified_with_override_is_halal() -> None:
    """Operator-override: relax the AAOIFI requirement for sovereign issues."""

    relaxed = SukukThresholds(require_aaoifi_certified=False)
    result = screen_sukuk(_issue(is_aaoifi_certified=False), thresholds=relaxed)
    assert result.verdict is SukukVerdict.HALAL


# ---------------------------------------------------------------------------
# Multiple-warning aggregation
# ---------------------------------------------------------------------------


def test_multiple_warnings_all_listed() -> None:
    result = screen_sukuk(
        _issue(
            issuer_type=IssuerType.CONVENTIONAL_BANK,
            is_aaoifi_certified=False,
            is_asset_backed=False,
            maturity_days=0,
        ),
        thresholds=SukukThresholds(accept_asset_based=False),
    )
    assert result.verdict is SukukVerdict.DOUBTFUL
    assert len(result.warnings) >= 4


# ---------------------------------------------------------------------------
# Multi-failure aggregation on hard rejections
# ---------------------------------------------------------------------------


def test_multiple_failures_listed_when_hard_rejected() -> None:
    result = screen_sukuk(
        _issue(
            asset_class=AssetClass.PROHIBITED_OPERATIONS,
            coupon_type=CouponType.INTEREST,
        )
    )
    assert result.verdict is SukukVerdict.NOT_HALAL
    assert len(result.failures) >= 2


# ---------------------------------------------------------------------------
# Threshold customisation flow
# ---------------------------------------------------------------------------


def test_strict_thresholds_flip_marginal_verdicts() -> None:
    issue = _issue(
        structure=SukukStructure.MURABAHA,
        is_asset_backed=False,
    )
    assert screen_sukuk(issue).verdict is SukukVerdict.HALAL
    strict = SukukThresholds(accept_murabaha=False, accept_asset_based=False)
    assert screen_sukuk(issue, thresholds=strict).verdict is SukukVerdict.DOUBTFUL


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_screen_result_is_frozen() -> None:
    result = screen_sukuk(_issue())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = SukukVerdict.NOT_HALAL  # type: ignore[misc]


def test_issue_is_frozen() -> None:
    i = _issue()
    with pytest.raises(dataclasses.FrozenInstanceError):
        i.maturity_days = 0  # type: ignore[misc]


def test_thresholds_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THRESHOLDS.require_aaoifi_certified = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB serialisation
# ---------------------------------------------------------------------------


def test_structure_string_values() -> None:
    assert SukukStructure.IJARAH.value == "ijarah"
    assert SukukStructure.MUSHARAKA.value == "musharaka"
    assert SukukStructure.MUDARABA.value == "mudaraba"
    assert SukukStructure.WAKALA.value == "wakala"
    assert SukukStructure.MURABAHA.value == "murabaha"
    assert SukukStructure.SALAM.value == "salam"
    assert SukukStructure.ISTISNA.value == "istisna"
    assert SukukStructure.HYBRID.value == "hybrid"


def test_asset_class_string_values() -> None:
    assert AssetClass.REAL_ESTATE.value == "real_estate"
    assert AssetClass.INFRASTRUCTURE.value == "infrastructure"
    assert AssetClass.FINANCIAL_RECEIVABLES.value == "financial_receivables"
    assert AssetClass.PROHIBITED_OPERATIONS.value == "prohibited_operations"
    assert AssetClass.UNKNOWN.value == "unknown"


def test_issuer_type_string_values() -> None:
    assert IssuerType.SOVEREIGN.value == "sovereign"
    assert IssuerType.SUPRANATIONAL.value == "supranational"
    assert IssuerType.CORPORATE_HALAL.value == "corporate_halal"
    assert IssuerType.CORPORATE_MIXED.value == "corporate_mixed"
    assert IssuerType.CONVENTIONAL_BANK.value == "conventional_bank"


def test_coupon_type_string_values() -> None:
    assert CouponType.PROFIT_SHARE.value == "profit_share"
    assert CouponType.RENTAL_INCOME.value == "rental_income"
    assert CouponType.FIXED_PROFIT_RATE.value == "fixed_profit_rate"
    assert CouponType.INTEREST.value == "interest"


def test_verdict_string_values() -> None:
    assert SukukVerdict.HALAL.value == "halal"
    assert SukukVerdict.NOT_HALAL.value == "not_halal"
    assert SukukVerdict.DOUBTFUL.value == "doubtful"
    assert SukukVerdict.INSUFFICIENT_DATA.value == "insufficient_data"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_halal_result() -> None:
    result = screen_sukuk(_issue())
    text = render_screen_result(result)
    assert "✅" in text
    assert "XS1234567890" in text
    assert "HALAL" in text
    assert "ijarah" in text
    assert "real_estate" in text
    assert "rental_income" in text


def test_render_not_halal_result() -> None:
    result = screen_sukuk(_issue(coupon_type=CouponType.INTEREST))
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text
    assert "failures:" in text


def test_render_doubtful_result() -> None:
    result = screen_sukuk(_issue(is_aaoifi_certified=False))
    text = render_screen_result(result)
    assert "⚠️" in text
    assert "DOUBTFUL" in text
    assert "warnings:" in text


def test_render_insufficient_data_result() -> None:
    result = screen_sukuk(_issue(asset_class=AssetClass.UNKNOWN))
    text = render_screen_result(result)
    assert "❓" in text
    assert "INSUFFICIENT_DATA" in text


def test_render_includes_maturity_days() -> None:
    result = screen_sukuk(_issue(maturity_days=1825))
    text = render_screen_result(result)
    assert "1825d" in text


# ---------------------------------------------------------------------------
# End-to-end: real-world cases
# ---------------------------------------------------------------------------


def test_real_world_saudi_sovereign_5y_ijarah_is_halal() -> None:
    """A typical Saudi sovereign 5-year ijarah sukuk."""

    result = screen_sukuk(
        SukukIssue(
            isin="XS2389654321",
            issuer_name="Kingdom of Saudi Arabia",
            issuer_type=IssuerType.SOVEREIGN,
            structure=SukukStructure.IJARAH,
            asset_class=AssetClass.REAL_ESTATE,
            coupon_type=CouponType.RENTAL_INCOME,
            is_aaoifi_certified=True,
            is_asset_backed=True,
            maturity_days=1825,
            expected_yield_pct=4.75,
        )
    )
    assert result.verdict is SukukVerdict.HALAL


def test_real_world_dubai_islamic_bank_sukuk_is_doubtful() -> None:
    """A Dubai Islamic Bank-issued corporate sukuk — DIB is shariah-
    compliant but registered as a conventional banking entity for
    operational reasons. Operators must verify per their scholar profile."""

    result = screen_sukuk(
        SukukIssue(
            isin="XS9876543210",
            issuer_name="Dubai Islamic Bank",
            issuer_type=IssuerType.CONVENTIONAL_BANK,
            structure=SukukStructure.WAKALA,
            asset_class=AssetClass.INFRASTRUCTURE,
            coupon_type=CouponType.FIXED_PROFIT_RATE,
            is_aaoifi_certified=True,
            is_asset_backed=False,
            maturity_days=1095,
            expected_yield_pct=5.0,
        )
    )
    # CONVENTIONAL_BANK issuer warning fires; not promoted to NOT_HALAL
    assert result.verdict is SukukVerdict.DOUBTFUL


def test_real_world_synthetic_credit_linked_sukuk_is_not_halal() -> None:
    """A 'sukuk' that's actually a credit-linked note over a pool
    of conventional loans — conventional debt dressed as sukuk."""

    result = screen_sukuk(
        SukukIssue(
            isin="FAKE0000001",
            issuer_name="Synthetic Issuer SPV",
            issuer_type=IssuerType.CORPORATE_MIXED,
            structure=SukukStructure.MURABAHA,
            asset_class=AssetClass.FINANCIAL_RECEIVABLES,
            coupon_type=CouponType.FIXED_PROFIT_RATE,
            is_aaoifi_certified=False,
            is_asset_backed=False,
            maturity_days=730,
            expected_yield_pct=8.0,
        )
    )
    assert result.verdict is SukukVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# Result shape sanity
# ---------------------------------------------------------------------------


def test_result_carries_all_fields() -> None:
    result = screen_sukuk(_issue())
    assert isinstance(result, SukukScreenResult)
    assert result.isin == "XS1234567890"
    assert result.issuer_name == "Saudi Sovereign"
    assert result.structure is SukukStructure.IJARAH
    assert result.asset_class is AssetClass.REAL_ESTATE
    assert result.coupon_type is CouponType.RENTAL_INCOME
    assert result.is_aaoifi_certified is True
    assert result.is_asset_backed is True
    assert result.maturity_days == 1825
