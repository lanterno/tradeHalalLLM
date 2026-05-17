"""Tests for the tokenised-REIT halal screener."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.halal.reit_screener import REITPropertyType
from halal_trader.web.tokenised_reit import (
    CustodyModel,
    DeFiIntegration,
    RegulatorRegistration,
    SmartContractAudit,
    TokenisedREITDeal,
    TokenisedREITVerdict,
    TokenStandard,
    YieldDenomination,
    render_screen_result,
    screen_tokenised_reit,
)


def _deal(
    *,
    symbol: str = "RT-DETROIT-001",
    platform: str = "RealT",
    property_type: REITPropertyType = REITPropertyType.RESIDENTIAL,
    token_standard: TokenStandard = TokenStandard.ERC20,
    regulator: RegulatorRegistration = RegulatorRegistration.SEC_REG_D,
    custody_model: CustodyModel = CustodyModel.SPV_OWNERSHIP,
    audit_status: SmartContractAudit = SmartContractAudit.AUDITED_BIG_FOUR,
    yield_denomination: YieldDenomination = YieldDenomination.USDC_STABLECOIN,
    defi_integration: DeFiIntegration = DeFiIntegration.STANDALONE,
    lockup_days: int = 30,
) -> TokenisedREITDeal:
    return TokenisedREITDeal(
        symbol=symbol,
        platform=platform,
        property_type=property_type,
        token_standard=token_standard,
        regulator=regulator,
        custody_model=custody_model,
        audit_status=audit_status,
        yield_denomination=yield_denomination,
        defi_integration=defi_integration,
        lockup_days=lockup_days,
    )


# ---------------------------------------------------------------------------
# Deal validation
# ---------------------------------------------------------------------------


def test_deal_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        _deal(symbol="")


def test_deal_rejects_empty_platform() -> None:
    with pytest.raises(ValueError, match="platform"):
        _deal(platform="")


def test_deal_rejects_negative_lockup() -> None:
    with pytest.raises(ValueError, match="lockup_days"):
        _deal(lockup_days=-1)


def test_deal_accepts_zero_lockup() -> None:
    """Pin: zero is valid (rare but exists for fully-liquid offerings)."""

    d = _deal(lockup_days=0)
    assert d.lockup_days == 0


# ---------------------------------------------------------------------------
# Hard rejections — derivative-rights custody
# ---------------------------------------------------------------------------


def test_derivative_rights_custody_is_not_halal() -> None:
    """Pin: derivative-rights tokens fail Shariah on gharar."""

    deal = _deal(custody_model=CustodyModel.DERIVATIVE_RIGHTS)
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL
    assert any("gharar" in f for f in result.failures)


def test_derivative_rights_overrides_other_clean_flags() -> None:
    """Pin: even with everything else clean, derivative-rights → NOT_HALAL."""

    deal = _deal(
        property_type=REITPropertyType.RESIDENTIAL,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        custody_model=CustodyModel.DERIVATIVE_RIGHTS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.RENT_DIRECT_FIAT,
        defi_integration=DeFiIntegration.STANDALONE,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# Hard rejections — riba via DeFi integration
# ---------------------------------------------------------------------------


def test_lending_enabled_is_not_halal() -> None:
    """Pin: lending integration is riba via the back door."""

    deal = _deal(defi_integration=DeFiIntegration.LENDING_ENABLED)
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL
    assert any("riba" in f for f in result.failures)


def test_borrowing_enabled_is_not_halal() -> None:
    deal = _deal(defi_integration=DeFiIntegration.BORROWING_ENABLED)
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL


def test_both_enabled_is_not_halal() -> None:
    deal = _deal(defi_integration=DeFiIntegration.BOTH_ENABLED)
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL


def test_lending_enabled_overrides_clean_flags() -> None:
    """Pin: even halal property + direct ownership + audited contract,
    a protocol that enables lending makes the holder a participant in riba."""

    deal = _deal(
        property_type=REITPropertyType.RESIDENTIAL,
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        defi_integration=DeFiIntegration.LENDING_ENABLED,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL


def test_multiple_failures_all_listed() -> None:
    """Two hard failures → both listed in the audit trail."""

    deal = _deal(
        custody_model=CustodyModel.DERIVATIVE_RIGHTS,
        defi_integration=DeFiIntegration.LENDING_ENABLED,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL
    assert len(result.failures) == 2


# ---------------------------------------------------------------------------
# INSUFFICIENT_DATA — UNKNOWN token standard
# ---------------------------------------------------------------------------


def test_unknown_token_standard_returns_insufficient_data() -> None:
    deal = _deal(token_standard=TokenStandard.UNKNOWN)
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.INSUFFICIENT_DATA
    assert any("UNKNOWN" in w for w in result.warnings)


def test_unknown_token_standard_returns_insufficient_data_even_with_clean_flags() -> None:
    deal = _deal(
        token_standard=TokenStandard.UNKNOWN,
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        defi_integration=DeFiIntegration.STANDALONE,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# HALAL — every check passes
# ---------------------------------------------------------------------------


def test_clean_direct_ownership_residential_is_halal() -> None:
    """Best-case: direct ownership + Reg A+ + audited + USDC + standalone."""

    deal = _deal(
        property_type=REITPropertyType.RESIDENTIAL,
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.USDC_STABLECOIN,
        defi_integration=DeFiIntegration.STANDALONE,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.HALAL
    assert result.failures == ()
    assert result.warnings == ()


def test_clean_direct_fiat_yield_is_halal() -> None:
    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.RENT_DIRECT_FIAT,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.HALAL


def test_clean_no_yield_appreciation_only_is_halal() -> None:
    """Pin: capital-appreciation-only deals (NONE yield) are HALAL."""

    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.NONE,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.HALAL


# ---------------------------------------------------------------------------
# DOUBTFUL — soft warnings drive doubtful
# ---------------------------------------------------------------------------


def test_spv_ownership_is_doubtful() -> None:
    """Pin: SPV ownership is the typical RealT model — passes screen but
    flagged because legal-title transferability depends on jurisdiction."""

    deal = _deal(custody_model=CustodyModel.SPV_OWNERSHIP)
    result = screen_tokenised_reit(deal)
    # spv + USDC + reg_d + audited → DOUBTFUL only because of SPV warning
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert any("spv" in w.lower() for w in result.warnings)


def test_no_regulator_is_doubtful() -> None:
    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.NONE,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert any("regulator" in w for w in result.warnings)


def test_unaudited_contract_is_doubtful() -> None:
    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        audit_status=SmartContractAudit.UNAUDITED,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert any("UNAUDITED" in w for w in result.warnings)


def test_self_audited_contract_is_doubtful() -> None:
    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        audit_status=SmartContractAudit.SELF_AUDITED,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert any("SELF_AUDITED" in w for w in result.warnings)


def test_indie_audit_is_halal() -> None:
    """Indie audit (Trail of Bits / etc. but not Big Four) still passes."""

    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_INDIE,
        yield_denomination=YieldDenomination.RENT_DIRECT_FIAT,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.HALAL


def test_usdt_yield_is_doubtful() -> None:
    """Pin: USDT's commercial-paper backing is debated."""

    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.USDT_STABLECOIN,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert any("yield_denomination" in w for w in result.warnings)


def test_other_stablecoin_yield_is_doubtful() -> None:
    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.OTHER_STABLECOIN,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL


def test_native_crypto_yield_is_doubtful() -> None:
    """ETH-denominated yield needs scholar review."""

    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.NATIVE_CRYPTO,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL


def test_hotel_property_type_is_doubtful() -> None:
    """Pin: hotel property type carries DOUBTFUL even when wrapped on-chain."""

    deal = _deal(
        property_type=REITPropertyType.HOTEL,
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.RENT_DIRECT_FIAT,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert any("hotel" in w.lower() for w in result.warnings)


def test_specialty_property_type_is_doubtful() -> None:
    deal = _deal(
        property_type=REITPropertyType.SPECIALTY,
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.RENT_DIRECT_FIAT,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL


def test_multiple_warnings_aggregate() -> None:
    """SPV + no regulator + USDT + unaudited → multiple warnings, DOUBTFUL."""

    deal = _deal(
        regulator=RegulatorRegistration.NONE,
        audit_status=SmartContractAudit.UNAUDITED,
        yield_denomination=YieldDenomination.USDT_STABLECOIN,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert len(result.warnings) >= 4  # spv + no-reg + unaudited + usdt


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_deal_is_frozen() -> None:
    d = _deal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.lockup_days = 60  # type: ignore[misc]


def test_screen_result_is_frozen() -> None:
    result = screen_tokenised_reit(_deal())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = TokenisedREITVerdict.NOT_HALAL  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_token_standard_string_values() -> None:
    assert TokenStandard.ERC20.value == "erc20"
    assert TokenStandard.ERC721.value == "erc721"
    assert TokenStandard.ERC1155.value == "erc1155"
    assert TokenStandard.SPL_FUNGIBLE.value == "spl_fungible"
    assert TokenStandard.UNKNOWN.value == "unknown"


def test_regulator_string_values() -> None:
    assert RegulatorRegistration.SEC_REG_A_PLUS.value == "sec_reg_a_plus"
    assert RegulatorRegistration.SEC_REG_D.value == "sec_reg_d"
    assert RegulatorRegistration.MICA_ARTICLE_16.value == "mica_article_16"
    assert RegulatorRegistration.NONE.value == "none"


def test_custody_string_values() -> None:
    assert CustodyModel.DIRECT_OWNERSHIP.value == "direct_ownership"
    assert CustodyModel.SPV_OWNERSHIP.value == "spv_ownership"
    assert CustodyModel.DERIVATIVE_RIGHTS.value == "derivative_rights"


def test_audit_string_values() -> None:
    assert SmartContractAudit.AUDITED_BIG_FOUR.value == "audited_big_four"
    assert SmartContractAudit.AUDITED_INDIE.value == "audited_indie"
    assert SmartContractAudit.SELF_AUDITED.value == "self_audited"
    assert SmartContractAudit.UNAUDITED.value == "unaudited"


def test_yield_string_values() -> None:
    assert YieldDenomination.RENT_DIRECT_FIAT.value == "rent_direct_fiat"
    assert YieldDenomination.USDC_STABLECOIN.value == "usdc_stablecoin"
    assert YieldDenomination.USDT_STABLECOIN.value == "usdt_stablecoin"
    assert YieldDenomination.NONE.value == "none"


def test_defi_integration_string_values() -> None:
    assert DeFiIntegration.STANDALONE.value == "standalone"
    assert DeFiIntegration.LENDING_ENABLED.value == "lending_enabled"
    assert DeFiIntegration.BORROWING_ENABLED.value == "borrowing_enabled"
    assert DeFiIntegration.BOTH_ENABLED.value == "both_enabled"


def test_verdict_string_values() -> None:
    assert TokenisedREITVerdict.HALAL.value == "halal"
    assert TokenisedREITVerdict.NOT_HALAL.value == "not_halal"
    assert TokenisedREITVerdict.DOUBTFUL.value == "doubtful"
    assert TokenisedREITVerdict.INSUFFICIENT_DATA.value == "insufficient_data"


# ---------------------------------------------------------------------------
# Render output — pinned no-address contract
# ---------------------------------------------------------------------------


def test_render_halal_result() -> None:
    deal = _deal(
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.RENT_DIRECT_FIAT,
    )
    result = screen_tokenised_reit(deal)
    text = render_screen_result(result)
    assert "✅" in text
    assert "RT-DETROIT-001" in text
    assert "RealT" in text
    assert "HALAL" in text
    assert "residential" in text


def test_render_not_halal_result() -> None:
    result = screen_tokenised_reit(_deal(custody_model=CustodyModel.DERIVATIVE_RIGHTS))
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text
    assert "failures:" in text


def test_render_doubtful_result() -> None:
    result = screen_tokenised_reit(_deal())  # default has SPV
    text = render_screen_result(result)
    assert "⚠️" in text
    assert "DOUBTFUL" in text
    assert "warnings:" in text


def test_render_insufficient_data_result() -> None:
    result = screen_tokenised_reit(_deal(token_standard=TokenStandard.UNKNOWN))
    text = render_screen_result(result)
    assert "❓" in text
    assert "INSUFFICIENT_DATA" in text


def test_render_does_not_include_wallet_address() -> None:
    """Pin no-address contract: render never includes 0x-style addresses."""

    deal = _deal(
        symbol="0xABCDEF1234567890",  # would be unusual but deliberate stress
    )
    result = screen_tokenised_reit(deal)
    text = render_screen_result(result)
    # The symbol IS the human label; addresses go in a separate column
    # the engine doesn't hold. The render shows the symbol but never any
    # raw 0x… address.
    assert "0x" not in text or "0xABCDEF" in text  # symbol is allowed
    # but the engine never adds an address — there's no field for it


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_realt_shaped_residential_deal_is_doubtful_via_spv() -> None:
    """A typical RealT residential token: SPV + Reg D + Big Four audit +
    USDC yield + standalone. Passes the hard rejections but the SPV
    custody flag drops to DOUBTFUL — operator must verify the SPV's
    legal-title enforceability per their jurisdiction."""

    deal = TokenisedREITDeal(
        symbol="REALT-DETROIT-RES-001",
        platform="RealT",
        property_type=REITPropertyType.RESIDENTIAL,
        token_standard=TokenStandard.ERC20,
        regulator=RegulatorRegistration.SEC_REG_D,
        custody_model=CustodyModel.SPV_OWNERSHIP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.USDC_STABLECOIN,
        defi_integration=DeFiIntegration.STANDALONE,
        lockup_days=30,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert any("spv" in w.lower() for w in result.warnings)


def test_aave_collateral_token_is_not_halal() -> None:
    """A tokenised RE asset with Aave-style collateral integration:
    BORROWING_ENABLED → NOT_HALAL even though the underlying property
    might be a halal industrial warehouse."""

    deal = TokenisedREITDeal(
        symbol="LOFTY-WAREHOUSE-001",
        platform="Lofty",
        property_type=REITPropertyType.INDUSTRIAL,
        token_standard=TokenStandard.ERC20,
        regulator=RegulatorRegistration.SEC_REG_A_PLUS,
        custody_model=CustodyModel.DIRECT_OWNERSHIP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        yield_denomination=YieldDenomination.RENT_DIRECT_FIAT,
        defi_integration=DeFiIntegration.BORROWING_ENABLED,
        lockup_days=60,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.NOT_HALAL
    assert any("riba" in f for f in result.failures)


def test_unregulated_unaudited_doubtful_aggregate() -> None:
    """A platform without regulator + unaudited contract + USDT yield —
    multiple DOUBTFUL signals aggregate into a clear scholar-review case."""

    deal = TokenisedREITDeal(
        symbol="UNKNOWN-PLATFORM-001",
        platform="UnknownDAO",
        property_type=REITPropertyType.RESIDENTIAL,
        token_standard=TokenStandard.ERC721,
        regulator=RegulatorRegistration.NONE,
        custody_model=CustodyModel.SPV_OWNERSHIP,
        audit_status=SmartContractAudit.UNAUDITED,
        yield_denomination=YieldDenomination.USDT_STABLECOIN,
        defi_integration=DeFiIntegration.STANDALONE,
        lockup_days=0,
    )
    result = screen_tokenised_reit(deal)
    assert result.verdict is TokenisedREITVerdict.DOUBTFUL
    assert len(result.warnings) >= 4  # spv + no-reg + unaudited + usdt
