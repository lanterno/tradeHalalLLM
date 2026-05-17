"""Tests for the on-chain settlement-route screener."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.web.settlement_route import (
    DEFAULT_POLICY,
    DEXProtocol,
    IntermediateTokenStatus,
    LiquidityModel,
    MEVProtection,
    SettlementChain,
    SettlementRoute,
    SettlementRoutePolicy,
    SettlementScreenResult,
    SettlementVerdict,
    render_screen_result,
    screen_settlement_route,
)
from halal_trader.web.tokenised_reit import SmartContractAudit


def _route(
    *,
    chain: SettlementChain = SettlementChain.ETHEREUM_MAINNET,
    protocol: DEXProtocol = DEXProtocol.UNISWAP_V3,
    liquidity_model: LiquidityModel = LiquidityModel.CONCENTRATED_LP,
    audit_status: SmartContractAudit = SmartContractAudit.AUDITED_BIG_FOUR,
    mev_protection: MEVProtection = MEVProtection.FLASHBOTS,
    slippage_tolerance_pct: float = 0.5,
    intermediate_token: IntermediateTokenStatus = IntermediateTokenStatus.NONE,
    crosses_bridge: bool = False,
    expected_route_hops: int = 1,
) -> SettlementRoute:
    return SettlementRoute(
        chain=chain,
        protocol=protocol,
        liquidity_model=liquidity_model,
        audit_status=audit_status,
        mev_protection=mev_protection,
        slippage_tolerance_pct=slippage_tolerance_pct,
        intermediate_token=intermediate_token,
        crosses_bridge=crosses_bridge,
        expected_route_hops=expected_route_hops,
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_values() -> None:
    p = DEFAULT_POLICY
    assert p.max_acceptable_slippage_pct == 5.0
    assert p.soft_slippage_warning_pct == 2.0
    assert p.flag_off_chain_orderbook is True


def test_policy_rejects_zero_max_slippage() -> None:
    with pytest.raises(ValueError, match="max_acceptable_slippage_pct"):
        SettlementRoutePolicy(max_acceptable_slippage_pct=0)


def test_policy_rejects_negative_max_slippage() -> None:
    with pytest.raises(ValueError, match="max_acceptable_slippage_pct"):
        SettlementRoutePolicy(max_acceptable_slippage_pct=-1)


def test_policy_rejects_above_10_max_slippage() -> None:
    """Pin: > 10% slippage exceeds gharar threshold."""

    with pytest.raises(ValueError, match="gharar"):
        SettlementRoutePolicy(max_acceptable_slippage_pct=11.0)


def test_policy_accepts_at_10_max_slippage() -> None:
    """Pin: exactly 10% boundary inclusive."""

    p = SettlementRoutePolicy(max_acceptable_slippage_pct=10.0)
    assert p.max_acceptable_slippage_pct == 10.0


def test_policy_rejects_zero_soft_warning() -> None:
    with pytest.raises(ValueError, match="soft_slippage_warning_pct"):
        SettlementRoutePolicy(soft_slippage_warning_pct=0)


def test_policy_rejects_soft_above_max() -> None:
    """Pin: soft warning cannot exceed max acceptable."""

    with pytest.raises(ValueError, match="cannot exceed"):
        SettlementRoutePolicy(max_acceptable_slippage_pct=2.0, soft_slippage_warning_pct=3.0)


# ---------------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------------


def test_route_rejects_negative_slippage() -> None:
    with pytest.raises(ValueError, match="slippage_tolerance_pct"):
        _route(slippage_tolerance_pct=-1)


def test_route_rejects_zero_hops() -> None:
    with pytest.raises(ValueError, match="expected_route_hops"):
        _route(expected_route_hops=0)


def test_route_accepts_zero_slippage() -> None:
    """Pin: zero slippage is valid (rare but exists for stable-stable swaps)."""

    r = _route(slippage_tolerance_pct=0.0)
    assert r.slippage_tolerance_pct == 0.0


# ---------------------------------------------------------------------------
# Hard rejections — slippage above max
# ---------------------------------------------------------------------------


def test_excessive_slippage_is_not_halal() -> None:
    """Pin: 6% slippage > 5% default max → NOT_HALAL."""

    route = _route(slippage_tolerance_pct=6.0)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.NOT_HALAL
    assert any("slippage_tolerance" in f for f in result.failures)


def test_slippage_at_max_threshold_is_halal() -> None:
    """Pin: exactly at threshold (5.0%) is allowed."""

    route = _route(slippage_tolerance_pct=5.0)
    result = screen_settlement_route(route)
    # Will be DOUBTFUL because 5% > 2% soft warning, but not NOT_HALAL
    assert result.verdict is SettlementVerdict.DOUBTFUL


def test_strict_slippage_policy_flips_verdict() -> None:
    strict = SettlementRoutePolicy(max_acceptable_slippage_pct=1.0, soft_slippage_warning_pct=0.5)
    route = _route(slippage_tolerance_pct=2.0)
    assert screen_settlement_route(route, policy=strict).verdict is SettlementVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# Hard rejections — forbidden intermediate token
# ---------------------------------------------------------------------------


def test_forbidden_intermediate_token_is_not_halal() -> None:
    """Pin: routing through interest-bearing wrapper / synthetic → NOT_HALAL."""

    route = _route(intermediate_token=IntermediateTokenStatus.FORBIDDEN)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.NOT_HALAL
    assert any("forbidden" in f for f in result.failures)


def test_forbidden_intermediate_overrides_clean_flags() -> None:
    """Pin: even with all-other-clean, FORBIDDEN intermediate → NOT_HALAL."""

    route = _route(
        liquidity_model=LiquidityModel.CONCENTRATED_LP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        mev_protection=MEVProtection.FLASHBOTS,
        slippage_tolerance_pct=0.5,
        intermediate_token=IntermediateTokenStatus.FORBIDDEN,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# INSUFFICIENT_DATA — UNKNOWN liquidity model
# ---------------------------------------------------------------------------


def test_unknown_liquidity_model_is_insufficient_data() -> None:
    route = _route(liquidity_model=LiquidityModel.UNKNOWN)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.INSUFFICIENT_DATA
    assert any("UNKNOWN" in w for w in result.warnings)


def test_unknown_liquidity_model_overrides_other_clean_flags() -> None:
    route = _route(
        liquidity_model=LiquidityModel.UNKNOWN,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        mev_protection=MEVProtection.FLASHBOTS,
        slippage_tolerance_pct=0.5,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# HALAL — every check passes
# ---------------------------------------------------------------------------


def test_clean_uniswap_v3_swap_is_halal() -> None:
    """Best-case: Uniswap V3 + audited + MEV protection + low slippage + direct."""

    result = screen_settlement_route(_route())
    assert result.verdict is SettlementVerdict.HALAL
    assert result.failures == ()
    assert result.warnings == ()


def test_clean_curve_stablecoin_swap_is_halal() -> None:
    route = _route(
        protocol=DEXProtocol.CURVE,
        liquidity_model=LiquidityModel.STABLE_CURVE,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_clean_jupiter_solana_swap_is_halal() -> None:
    route = _route(
        chain=SettlementChain.SOLANA_MAINNET,
        protocol=DEXProtocol.JUPITER,
        liquidity_model=LiquidityModel.CONCENTRATED_LP,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_clean_dydx_v4_orderbook_on_chain_is_halal() -> None:
    """Pin: ORDER_BOOK_ON_CHAIN passes (dYdX v4 / Hyperliquid model)."""

    route = _route(
        chain=SettlementChain.ETHEREUM_MAINNET,
        protocol=DEXProtocol.OTHER,
        liquidity_model=LiquidityModel.ORDER_BOOK_ON_CHAIN,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_clean_with_halal_intermediate_is_halal() -> None:
    route = _route(
        intermediate_token=IntermediateTokenStatus.HALAL_STABLECOIN,
        expected_route_hops=2,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


# ---------------------------------------------------------------------------
# DOUBTFUL — soft warnings drive doubtful
# ---------------------------------------------------------------------------


def test_off_chain_orderbook_is_doubtful() -> None:
    """Pin: ORDER_BOOK_OFF_CHAIN raises gharar concerns."""

    route = _route(liquidity_model=LiquidityModel.ORDER_BOOK_OFF_CHAIN)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("off_chain" in w for w in result.warnings)


def test_off_chain_orderbook_can_be_disabled_via_policy() -> None:
    """Operator override: don't flag off-chain matched orders."""

    relaxed = SettlementRoutePolicy(flag_off_chain_orderbook=False)
    route = _route(liquidity_model=LiquidityModel.ORDER_BOOK_OFF_CHAIN)
    result = screen_settlement_route(route, policy=relaxed)
    assert result.verdict is SettlementVerdict.HALAL


def test_unaudited_contract_is_doubtful() -> None:
    route = _route(audit_status=SmartContractAudit.UNAUDITED)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("UNAUDITED" in w for w in result.warnings)


def test_self_audited_contract_is_doubtful() -> None:
    route = _route(audit_status=SmartContractAudit.SELF_AUDITED)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("SELF_AUDITED" in w for w in result.warnings)


def test_indie_audit_passes() -> None:
    """Pin: indie audit (Trail of Bits etc.) passes as HALAL."""

    route = _route(audit_status=SmartContractAudit.AUDITED_INDIE)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_no_mev_protection_is_doubtful() -> None:
    route = _route(mev_protection=MEVProtection.NONE)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("mev_protection=none" in w for w in result.warnings)


def test_native_mev_protection_passes() -> None:
    """CowSwap-style native protection passes."""

    route = _route(mev_protection=MEVProtection.NATIVE_PROTECTION)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_mev_blocker_passes() -> None:
    route = _route(mev_protection=MEVProtection.MEV_BLOCKER)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_debated_stablecoin_intermediate_is_doubtful() -> None:
    """Pin: USDT-shaped commercial-paper-backed stablecoin → DOUBTFUL."""

    route = _route(intermediate_token=IntermediateTokenStatus.DEBATED_STABLECOIN)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("debated stablecoin" in w for w in result.warnings)


def test_bridge_use_is_doubtful() -> None:
    """Pin: cross-chain bridge use adds gharar warning."""

    route = _route(crosses_bridge=True)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("bridge" in w for w in result.warnings)


def test_high_hops_count_is_doubtful() -> None:
    """Pin: > 3 hops adds compounding gharar warning."""

    route = _route(expected_route_hops=4)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("hops" in w for w in result.warnings)


def test_three_hops_passes() -> None:
    """Pin: exactly 3 hops is allowed (boundary)."""

    route = _route(expected_route_hops=3)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_slippage_above_soft_threshold_is_doubtful() -> None:
    """Pin: 3% slippage > 2% soft default → DOUBTFUL."""

    route = _route(slippage_tolerance_pct=3.0)
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("soft threshold" in w for w in result.warnings)


def test_slippage_at_soft_threshold_is_halal() -> None:
    """Pin: exactly at soft threshold (2%) does not trigger warning."""

    route = _route(slippage_tolerance_pct=2.0)
    result = screen_settlement_route(route)
    # No warning for slippage; verdict should be HALAL
    assert result.verdict is SettlementVerdict.HALAL


def test_multiple_warnings_aggregate() -> None:
    """A swap with multiple soft concerns aggregates warnings."""

    route = _route(
        audit_status=SmartContractAudit.UNAUDITED,
        mev_protection=MEVProtection.NONE,
        slippage_tolerance_pct=4.0,
        intermediate_token=IntermediateTokenStatus.DEBATED_STABLECOIN,
        crosses_bridge=True,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert len(result.warnings) >= 5  # unaudited + no_mev + slippage + debated + bridge


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_route_is_frozen() -> None:
    r = _route()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.slippage_tolerance_pct = 99.0  # type: ignore[misc]


def test_screen_result_is_frozen() -> None:
    result = screen_settlement_route(_route())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = SettlementVerdict.NOT_HALAL  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.max_acceptable_slippage_pct = 50.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_chain_string_values() -> None:
    assert SettlementChain.ETHEREUM_MAINNET.value == "ethereum_mainnet"
    assert SettlementChain.ETHEREUM_OPTIMISM.value == "ethereum_optimism"
    assert SettlementChain.SOLANA_MAINNET.value == "solana_mainnet"
    assert SettlementChain.POLYGON.value == "polygon"


def test_protocol_string_values() -> None:
    assert DEXProtocol.UNISWAP_V3.value == "uniswap_v3"
    assert DEXProtocol.UNISWAP_V4.value == "uniswap_v4"
    assert DEXProtocol.CURVE.value == "curve"
    assert DEXProtocol.JUPITER.value == "jupiter"
    assert DEXProtocol.AGGREGATOR_1INCH.value == "aggregator_1inch"


def test_liquidity_model_string_values() -> None:
    assert LiquidityModel.CONSTANT_PRODUCT.value == "constant_product"
    assert LiquidityModel.CONCENTRATED_LP.value == "concentrated_lp"
    assert LiquidityModel.STABLE_CURVE.value == "stable_curve"
    assert LiquidityModel.ORDER_BOOK_ON_CHAIN.value == "order_book_on_chain"
    assert LiquidityModel.ORDER_BOOK_OFF_CHAIN.value == "order_book_off_chain"


def test_mev_protection_string_values() -> None:
    assert MEVProtection.NATIVE_PROTECTION.value == "native_protection"
    assert MEVProtection.FLASHBOTS.value == "flashbots"
    assert MEVProtection.MEV_BLOCKER.value == "mev_blocker"
    assert MEVProtection.NONE.value == "none"


def test_intermediate_string_values() -> None:
    assert IntermediateTokenStatus.NONE.value == "none"
    assert IntermediateTokenStatus.HALAL_STABLECOIN.value == "halal_stablecoin"
    assert IntermediateTokenStatus.DEBATED_STABLECOIN.value == "debated_stablecoin"
    assert IntermediateTokenStatus.FORBIDDEN.value == "forbidden"


def test_verdict_string_values() -> None:
    assert SettlementVerdict.HALAL.value == "halal"
    assert SettlementVerdict.NOT_HALAL.value == "not_halal"
    assert SettlementVerdict.DOUBTFUL.value == "doubtful"
    assert SettlementVerdict.INSUFFICIENT_DATA.value == "insufficient_data"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_halal() -> None:
    result = screen_settlement_route(_route())
    text = render_screen_result(result)
    assert "✅" in text
    assert "uniswap_v3" in text
    assert "ethereum_mainnet" in text
    assert "HALAL" in text


def test_render_not_halal() -> None:
    result = screen_settlement_route(_route(slippage_tolerance_pct=8.0))
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text
    assert "failures:" in text


def test_render_doubtful() -> None:
    result = screen_settlement_route(_route(mev_protection=MEVProtection.NONE))
    text = render_screen_result(result)
    assert "⚠️" in text
    assert "DOUBTFUL" in text
    assert "warnings:" in text


def test_render_insufficient_data() -> None:
    result = screen_settlement_route(_route(liquidity_model=LiquidityModel.UNKNOWN))
    text = render_screen_result(result)
    assert "❓" in text
    assert "INSUFFICIENT_DATA" in text


def test_render_includes_slippage() -> None:
    result = screen_settlement_route(_route(slippage_tolerance_pct=0.5))
    text = render_screen_result(result)
    assert "0.50%" in text


def test_render_no_address_or_tx_hash() -> None:
    """Pin: render never includes 0x... addresses or tx hashes."""

    result = screen_settlement_route(_route())
    text = render_screen_result(result)
    assert "0x" not in text  # no address-shaped strings
    assert "tx" not in text.lower() or "tx hash" not in text.lower()


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_typical_uniswap_v4_swap_with_flashbots_is_halal() -> None:
    """Best-case: routing a halal-screened token swap through Uniswap V4
    on Optimism with Flashbots MEV protection, 0.5% slippage, no bridge."""

    route = SettlementRoute(
        chain=SettlementChain.ETHEREUM_OPTIMISM,
        protocol=DEXProtocol.UNISWAP_V4,
        liquidity_model=LiquidityModel.CONCENTRATED_LP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        mev_protection=MEVProtection.FLASHBOTS,
        slippage_tolerance_pct=0.5,
        intermediate_token=IntermediateTokenStatus.NONE,
        crosses_bridge=False,
        expected_route_hops=1,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_jupiter_solana_with_usdc_intermediate_is_halal() -> None:
    """Realistic: Jupiter aggregator on Solana routing through USDC."""

    route = SettlementRoute(
        chain=SettlementChain.SOLANA_MAINNET,
        protocol=DEXProtocol.JUPITER,
        liquidity_model=LiquidityModel.CONCENTRATED_LP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        mev_protection=MEVProtection.NATIVE_PROTECTION,
        slippage_tolerance_pct=1.0,
        intermediate_token=IntermediateTokenStatus.HALAL_STABLECOIN,
        crosses_bridge=False,
        expected_route_hops=2,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.HALAL


def test_cross_chain_bridge_swap_is_doubtful() -> None:
    """ETH → Polygon via bridge for stablecoin liquidity → DOUBTFUL."""

    route = SettlementRoute(
        chain=SettlementChain.POLYGON,
        protocol=DEXProtocol.UNISWAP_V3,
        liquidity_model=LiquidityModel.CONCENTRATED_LP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        mev_protection=MEVProtection.FLASHBOTS,
        slippage_tolerance_pct=0.5,
        intermediate_token=IntermediateTokenStatus.HALAL_STABLECOIN,
        crosses_bridge=True,
        expected_route_hops=2,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert any("bridge" in w for w in result.warnings)


def test_unaudited_high_slippage_no_mev_protection_is_doubtful_aggregate() -> None:
    """Worst-case-doubtful: every DOUBTFUL signal fires."""

    route = SettlementRoute(
        chain=SettlementChain.ETHEREUM_MAINNET,
        protocol=DEXProtocol.OTHER,
        liquidity_model=LiquidityModel.CONSTANT_PRODUCT,
        audit_status=SmartContractAudit.UNAUDITED,
        mev_protection=MEVProtection.NONE,
        slippage_tolerance_pct=4.5,
        intermediate_token=IntermediateTokenStatus.DEBATED_STABLECOIN,
        crosses_bridge=True,
        expected_route_hops=4,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.DOUBTFUL
    assert len(result.warnings) >= 6


def test_excessive_slippage_blocks_settlement() -> None:
    """A 7% slippage swap → NOT_HALAL even with all-other-clean."""

    route = SettlementRoute(
        chain=SettlementChain.ETHEREUM_MAINNET,
        protocol=DEXProtocol.UNISWAP_V3,
        liquidity_model=LiquidityModel.CONCENTRATED_LP,
        audit_status=SmartContractAudit.AUDITED_BIG_FOUR,
        mev_protection=MEVProtection.FLASHBOTS,
        slippage_tolerance_pct=7.0,
        intermediate_token=IntermediateTokenStatus.NONE,
        crosses_bridge=False,
        expected_route_hops=1,
    )
    result = screen_settlement_route(route)
    assert result.verdict is SettlementVerdict.NOT_HALAL


def test_screen_result_carries_chain_protocol() -> None:
    """Result preserves the route metadata for the audit trail."""

    result = screen_settlement_route(
        _route(
            chain=SettlementChain.ETHEREUM_BASE,
            protocol=DEXProtocol.UNISWAP_V4,
        )
    )
    assert isinstance(result, SettlementScreenResult)
    assert result.chain is SettlementChain.ETHEREUM_BASE
    assert result.protocol is DEXProtocol.UNISWAP_V4
