"""Decentralised on-chain settlement route screener.

The roadmap pins on-chain settlement as the long-term escape from
centralised-exchange counterparty risk: instead of trusting Binance
to honour fills, the bot routes a halal-token swap through a DEX
on Ethereum / Solana / a Layer 2. The full integration is deferred
to a follow-up — a live router needs both the swap-construction
side AND the per-cycle gas/MEV strategy — but the **route-screen
logic** is operator-supplied / CSV-fed pure-Python, exactly the
isolated-module pattern of every Round-4 wave.

DEX settlement raises concerns the centralised-exchange path
doesn't:

- **Liquidity model**. Constant-product AMM + concentrated-LP
  are HALAL (the swap is a price-discovered exchange of two
  permissible assets); off-chain order books raise gharar
  questions because the matched price isn't fixed at submission
  time, and the relayer is a counterparty of unclear shariah
  status.
- **MEV exposure**. A swap submitted to the public mempool
  without MEV protection can be sandwich-attacked: the
  searcher front-runs the swap, the price moves, the user
  pays the worse price. Operationally inferior even if not
  haram per se; the screener flags as DOUBTFUL when no
  protection is configured.
- **Slippage tolerance**. A 5% slippage tolerance is widely
  accepted; > 10% raises gharar concerns because the user
  consenting to a 10% price-move-on-fill is consenting to a
  highly uncertain outcome.
- **Smart contract audit**. Same as Wave 12.E tokenised REIT —
  the contract is the custody mechanism for the swap; an
  unaudited router is a gharar concern.
- **Cross-chain bridge use**. Bridges add a layer of custody
  + a layer of liveness assumption (the bridge validators must
  remain honest); pinned as DOUBTFUL warning so operators
  understand the additional risk.
- **Routing token**. A swap that routes through a non-halal
  intermediate token (a synthetic asset, an interest-bearing
  token wrapper, a non-allocated stablecoin) inherits the
  concerns of that token. Pinned as DOUBTFUL warning.

Pinned semantics:
- **HALAL requires every check pass.** Direct DEX + audited
  contract + halal liquidity model + reasonable slippage + MEV
  protection + no bridge + no riba intermediate routing token.
- **NOT_HALAL is unconditional** for negative slippage tolerance
  (impossible — must be positive), missing required fields, or
  routing through an explicitly-marked riba intermediate.
- **DOUBTFUL** for unaudited contracts, no MEV protection,
  slippage > soft threshold, ORDER_BOOK with off-chain matching,
  cross-chain bridge use, routing through debated stablecoins.
- **INSUFFICIENT_DATA** when liquidity model UNKNOWN.
- **Render output never includes wallet addresses, contract
  addresses, or transaction hashes**. Wallet addresses are PII
  (Wave 11.D); contract addresses are operationally sensitive
  (phishing-link risk per the Wave 12.E precedent); transaction
  hashes are user-correlatable. The render shows protocol +
  chain + verdict + per-flag warnings — the operator audits the
  on-chain side via blockchain explorer separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Re-use the contract-audit ladder from Wave 12.E — a router contract
# audited by Trail of Bits + Quantstamp passes the same standard as
# a tokenised-RE custody contract.
from halal_trader.web.tokenised_reit import SmartContractAudit


class SettlementChain(str, Enum):
    """Layer-1 / Layer-2 the swap settles on.

    Pinned BCP-style string values for JSON / DB stability. The
    set is closed at the type level; operators add new chains via
    code review (a typo can't silently route to an unsupported
    chain).
    """

    ETHEREUM_MAINNET = "ethereum_mainnet"
    ETHEREUM_OPTIMISM = "ethereum_optimism"
    ETHEREUM_ARBITRUM = "ethereum_arbitrum"
    ETHEREUM_BASE = "ethereum_base"
    POLYGON = "polygon"
    SOLANA_MAINNET = "solana_mainnet"


class DEXProtocol(str, Enum):
    """The DEX router / aggregator the swap uses.

    `AGGREGATOR_1INCH` and `AGGREGATOR_JUPITER` route across
    multiple liquidity sources — operator's intent is "best
    price"; the screener's concern is the intermediate hops
    each protocol takes.
    """

    UNISWAP_V3 = "uniswap_v3"
    UNISWAP_V4 = "uniswap_v4"
    SUSHISWAP = "sushiswap"
    CURVE = "curve"
    BALANCER = "balancer"
    AGGREGATOR_1INCH = "aggregator_1inch"
    JUPITER = "jupiter"
    RAYDIUM = "raydium"
    OTHER = "other"


class LiquidityModel(str, Enum):
    """How the protocol prices the swap.

    Pinned: CONSTANT_PRODUCT + CONCENTRATED_LP + STABLE_CURVE are
    HALAL because the swap is a price-discovered exchange of two
    permissible tokens at a deterministic price. ORDER_BOOK_OFF_
    CHAIN raises gharar questions because the matched price isn't
    fixed at submission time, and the relayer's shariah status is
    unclear.
    """

    CONSTANT_PRODUCT = "constant_product"  # Uniswap V2 style
    CONCENTRATED_LP = "concentrated_lp"  # Uniswap V3 / V4
    STABLE_CURVE = "stable_curve"  # Curve / Balancer stable pools
    ORDER_BOOK_ON_CHAIN = "order_book_on_chain"  # dYdX V4 / Hyperliquid
    ORDER_BOOK_OFF_CHAIN = "order_book_off_chain"  # 0x / matchers
    UNKNOWN = "unknown"


class MEVProtection(str, Enum):
    """How the swap is protected from sandwich / front-running.

    Pinned: NONE drives DOUBTFUL warning. The screener doesn't
    treat MEV exposure as haram per se (the protocol works
    correctly), but operationally inferior swap execution is
    flagged so the operator understands the trade-off.
    """

    NATIVE_PROTECTION = "native_protection"  # CowSwap / built-in
    FLASHBOTS = "flashbots"  # private mempool
    MEV_BLOCKER = "mev_blocker"  # third-party private RPC
    NONE = "none"  # public mempool, exposed


class IntermediateTokenStatus(str, Enum):
    """Halal status of the routing token (e.g., USDC vs USDT).

    Pinned: NONE means a direct A→B swap, no intermediate token.
    HALAL_STABLECOIN, HALAL_NATIVE pass; DEBATED_STABLECOIN
    (USDT) drives DOUBTFUL; FORBIDDEN drives NOT_HALAL.
    """

    NONE = "none"
    HALAL_STABLECOIN = "halal_stablecoin"  # USDC / RLUSD / cash-backed
    HALAL_NATIVE = "halal_native"  # ETH / SOL / etc.
    DEBATED_STABLECOIN = "debated_stablecoin"  # USDT / commercial paper
    FORBIDDEN = "forbidden"  # interest-bearing wrapper / synthetic


_FORBIDDEN_INTERMEDIATES: frozenset[IntermediateTokenStatus] = frozenset(
    {IntermediateTokenStatus.FORBIDDEN}
)


class SettlementVerdict(str, Enum):
    """Screen verdict.

    Pinned string values for JSON / DB stability — dashboard +
    exception-queue UI key on these literals.
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    DOUBTFUL = "doubtful"
    INSUFFICIENT_DATA = "insufficient_data"


# Liquidity models that pass the structural halal test.
_HALAL_LIQUIDITY_MODELS: frozenset[LiquidityModel] = frozenset(
    {
        LiquidityModel.CONSTANT_PRODUCT,
        LiquidityModel.CONCENTRATED_LP,
        LiquidityModel.STABLE_CURVE,
        LiquidityModel.ORDER_BOOK_ON_CHAIN,
    }
)


@dataclass(frozen=True)
class SettlementRoutePolicy:
    """Operator-tunable policy.

    `max_acceptable_slippage_pct` defaults to 5% — operators on
    thin-liquidity exotic pairs may bump to 8% with explicit
    scholar review. > 10% is rejected at construction (gharar
    threshold).
    `flag_off_chain_orderbook` defaults to True — the engine
    flags off-chain matched orders as DOUBTFUL.
    """

    max_acceptable_slippage_pct: float = 5.0
    soft_slippage_warning_pct: float = 2.0
    flag_off_chain_orderbook: bool = True
    require_mev_protection: bool = False  # DOUBTFUL warning, not blocking

    def __post_init__(self) -> None:
        if self.max_acceptable_slippage_pct <= 0:
            raise ValueError("max_acceptable_slippage_pct must be positive")
        if self.max_acceptable_slippage_pct > 10.0:
            raise ValueError(
                f"max_acceptable_slippage_pct {self.max_acceptable_slippage_pct} > 10 "
                "exceeds gharar threshold"
            )
        if self.soft_slippage_warning_pct <= 0:
            raise ValueError("soft_slippage_warning_pct must be positive")
        if self.soft_slippage_warning_pct > self.max_acceptable_slippage_pct:
            raise ValueError("soft_slippage_warning_pct cannot exceed max_acceptable_slippage_pct")


DEFAULT_POLICY = SettlementRoutePolicy()


@dataclass(frozen=True)
class SettlementRoute:
    """A proposed on-chain swap route."""

    chain: SettlementChain
    protocol: DEXProtocol
    liquidity_model: LiquidityModel
    audit_status: SmartContractAudit
    mev_protection: MEVProtection
    slippage_tolerance_pct: float
    intermediate_token: IntermediateTokenStatus
    crosses_bridge: bool
    expected_route_hops: int

    def __post_init__(self) -> None:
        if self.slippage_tolerance_pct < 0:
            raise ValueError("slippage_tolerance_pct must be non-negative")
        if self.expected_route_hops < 1:
            raise ValueError("expected_route_hops must be at least 1")


@dataclass(frozen=True)
class SettlementScreenResult:
    """Screen verdict + supporting flags + audit notes."""

    chain: SettlementChain
    protocol: DEXProtocol
    verdict: SettlementVerdict
    slippage_tolerance_pct: float
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def screen_settlement_route(
    route: SettlementRoute,
    *,
    policy: SettlementRoutePolicy = DEFAULT_POLICY,
) -> SettlementScreenResult:
    """Apply the on-chain settlement-route halal screen.

    Returns a `SettlementScreenResult` with verdict + per-rule
    failure / warning lists for the audit trail.
    """

    failures: list[str] = []
    warnings: list[str] = []

    # Hard rejections.
    if route.intermediate_token in _FORBIDDEN_INTERMEDIATES:
        failures.append(
            f"intermediate_token={route.intermediate_token.value}: "
            "routing through forbidden token (interest-bearing wrapper / synthetic)"
        )

    if route.slippage_tolerance_pct > policy.max_acceptable_slippage_pct:
        failures.append(
            f"slippage_tolerance {route.slippage_tolerance_pct:.2f}% "
            f"exceeds {policy.max_acceptable_slippage_pct:.2f}% gharar threshold"
        )

    # INSUFFICIENT_DATA gate.
    if route.liquidity_model is LiquidityModel.UNKNOWN:
        return SettlementScreenResult(
            chain=route.chain,
            protocol=route.protocol,
            verdict=SettlementVerdict.INSUFFICIENT_DATA,
            slippage_tolerance_pct=route.slippage_tolerance_pct,
            failures=tuple(failures),
            warnings=("liquidity_model is UNKNOWN — operator must verify before routing",),
        )

    if failures:
        return SettlementScreenResult(
            chain=route.chain,
            protocol=route.protocol,
            verdict=SettlementVerdict.NOT_HALAL,
            slippage_tolerance_pct=route.slippage_tolerance_pct,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Soft warnings — drive DOUBTFUL.
    if route.liquidity_model not in _HALAL_LIQUIDITY_MODELS:
        if (
            route.liquidity_model is LiquidityModel.ORDER_BOOK_OFF_CHAIN
            and policy.flag_off_chain_orderbook
        ):
            warnings.append(
                "liquidity_model=order_book_off_chain: matched price not fixed "
                "at submission (gharar concern); off-chain relayer's shariah "
                "status unclear"
            )

    if route.audit_status is SmartContractAudit.UNAUDITED:
        warnings.append("smart contract is UNAUDITED: custody mechanism gharar concern")
    elif route.audit_status is SmartContractAudit.SELF_AUDITED:
        warnings.append("smart contract is SELF_AUDITED: third-party audit recommended")

    if route.mev_protection is MEVProtection.NONE:
        warnings.append(
            "mev_protection=none: swap exposed to sandwich / front-running; "
            "operationally inferior execution"
        )

    if route.intermediate_token is IntermediateTokenStatus.DEBATED_STABLECOIN:
        warnings.append(
            "routing through debated stablecoin (commercial-paper-backed): "
            "needs scholar review for halal status"
        )

    if route.crosses_bridge:
        warnings.append(
            "route crosses cross-chain bridge: additional gharar from bridge "
            "custody + validator-honesty assumption"
        )

    if (
        route.slippage_tolerance_pct > policy.soft_slippage_warning_pct
        and route.slippage_tolerance_pct <= policy.max_acceptable_slippage_pct
    ):
        warnings.append(
            f"slippage_tolerance {route.slippage_tolerance_pct:.2f}% above "
            f"{policy.soft_slippage_warning_pct:.2f}% soft threshold; "
            "operator should verify expected fill matches intent"
        )

    if route.expected_route_hops > 3:
        warnings.append(
            f"route uses {route.expected_route_hops} hops: each hop adds "
            "gharar + slippage compounding; consider direct route"
        )

    if warnings:
        verdict = SettlementVerdict.DOUBTFUL
    else:
        verdict = SettlementVerdict.HALAL

    return SettlementScreenResult(
        chain=route.chain,
        protocol=route.protocol,
        verdict=verdict,
        slippage_tolerance_pct=route.slippage_tolerance_pct,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


_VERDICT_EMOJI: dict[SettlementVerdict, str] = {
    SettlementVerdict.HALAL: "✅",
    SettlementVerdict.NOT_HALAL: "❌",
    SettlementVerdict.DOUBTFUL: "⚠️",
    SettlementVerdict.INSUFFICIENT_DATA: "❓",
}


def render_screen_result(result: SettlementScreenResult) -> str:
    """Format the screen result for ops display.

    Pinned no-address contract: never includes wallet addresses,
    contract addresses, or transaction hashes. Operators audit
    the on-chain side via blockchain explorer separately.
    """

    emoji = _VERDICT_EMOJI[result.verdict]
    lines = [
        f"{emoji} {result.protocol.value} on {result.chain.value} — {result.verdict.value.upper()}",
        f"  slippage tolerance: {result.slippage_tolerance_pct:.2f}%",
    ]
    if result.failures:
        lines.append("  failures:")
        for f in result.failures:
            lines.append(f"    · {f}")
    if result.warnings:
        lines.append("  warnings:")
        for w in result.warnings:
            lines.append(f"    · {w}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "DEXProtocol",
    "IntermediateTokenStatus",
    "LiquidityModel",
    "MEVProtection",
    "SettlementChain",
    "SettlementRoute",
    "SettlementRoutePolicy",
    "SettlementScreenResult",
    "SettlementVerdict",
    "render_screen_result",
    "screen_settlement_route",
]
