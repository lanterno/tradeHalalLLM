"""Tokenised real-estate REIT screener.

Tokenised real-estate platforms (RealT, Lofty, Propy fractional)
sell on-chain tokens that represent fractional ownership of
specific physical properties. The roadmap defers full integration
because none of these platforms hold AAOIFI certification yet,
but the **screening logic** is operator-supplied / CSV-fed pure-
Python — exactly the isolated-module pattern of every other
Round-4 wave (1.G commodities, 1.H sukuk, 1.I REIT, 12.A robo,
12.D halal-VC).

Tokenised REITs raise concerns the traditional REIT screen
(Wave 1.I) doesn't catch:

- **Custody model**. The token might represent direct fractional
  ownership of the deed (RealT-style SPV per property), or a
  derivative right that tracks the property's price without
  legal ownership. Derivative-rights tokens fail Shariah on
  gharar — the holder doesn't actually own anything physical.
- **Settlement certainty**. Most US states still don't recognise
  NFT-based property records as legal title; the on-chain
  transfer is only as binding as the off-chain SPV resolution.
- **Regulator backing**. Unregistered tokenised offerings
  (SEC Reg D / Reg A+ / MiCA Article 16) carry both legal risk
  AND a stronger gharar argument because the off-chain
  enforceability is unclear.
- **Smart contract audit**. An unaudited contract is a malware
  vector — even if the underlying property is halal, the
  custody mechanism failing is a form of gharar.
- **Yield denomination**. Many tokenised RE platforms pay
  rental yield in stablecoins; the stablecoin's own halal
  status matters (USDT's reserves are partially in commercial
  paper; USDC is fully cash-backed; PYUSD / RLUSD vary).
- **DeFi integration**. The classic riba-via-back-door pattern
  for tokenised RE: the platform allows holders to lend out
  their tokens as collateral for interest-bearing loans, or
  borrow against them. Both make the holder a participant in
  riba even if the underlying property is halal.

Pinned semantics:
- **HALAL requires every check pass**. Direct ownership +
  registered regulator + audited contract + halal yield
  denomination + standalone (no lending/borrowing). Any
  failure flips to NOT_HALAL or DOUBTFUL.
- **NOT_HALAL is unconditional** for DERIVATIVE_RIGHTS custody
  (no actual ownership of physical asset → gharar), and for
  LENDING_ENABLED OR BORROWING_ENABLED DeFi integration (riba
  via back door even if the underlying is halal). The
  asymmetric treatment matters: a halal warehouse REIT that
  the operator can't borrow against is HALAL; the same REIT
  with on-chain borrowing enabled is NOT_HALAL because the
  *protocol* enables riba even if the operator doesn't
  personally borrow.
- **DOUBTFUL** for unregistered offerings, unaudited contracts,
  non-halal stablecoin yields. Operators can opt-in via Wave
  2.F scholar review queue.
- **INSUFFICIENT_DATA** when token_standard is UNKNOWN or any
  required field is undisclosed.
- **Render output never includes wallet addresses or token
  contract IDs**. Wallet addresses are PII (Wave 11.D); token
  contract addresses are also operationally sensitive (a
  malicious renderer could inject phishing links). Mirrors the
  no-PII pattern of Wave 11.D + 11.C + 3.B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Re-use the property-type taxonomy from Wave 1.I — the underlying
# physical asset's halal classification doesn't change just because
# it's tokenised.
from halal_trader.halal.reit_screener import REITPropertyType


class TokenStandard(str, Enum):
    """The on-chain token standard.

    Pinned string values for JSON / DB stability. UNKNOWN is the
    explicit "unrecognised standard" sentinel for INSUFFICIENT_DATA.
    """

    ERC20 = "erc20"  # fungible Ethereum-style
    ERC721 = "erc721"  # NFT, single-property
    ERC1155 = "erc1155"  # multi-token, batched-property
    SPL_FUNGIBLE = "spl_fungible"  # Solana fungible
    NATIVE_OTHER = "native_other"  # Cosmos / Polkadot / etc.
    UNKNOWN = "unknown"


class RegulatorRegistration(str, Enum):
    """How the offering is regulated.

    Pinned: an unregistered offering (NONE) lands DOUBTFUL — operator
    must opt-in via Wave 2.F scholar review.
    """

    SEC_REG_A_PLUS = "sec_reg_a_plus"
    SEC_REG_D = "sec_reg_d"
    MICA_ARTICLE_16 = "mica_article_16"
    OTHER = "other"
    NONE = "none"


class CustodyModel(str, Enum):
    """How the token relates to the physical property.

    `DIRECT_OWNERSHIP` — token = legal fractional deed (rare; only
    a few US states recognise this).
    `SPV_OWNERSHIP` — token = unit in an SPV (LLC) that owns the
    property; transfer of token transfers SPV unit. RealT model.
    `DERIVATIVE_RIGHTS` — token = price tracker without legal
    ownership. NOT_HALAL by construction.
    """

    DIRECT_OWNERSHIP = "direct_ownership"
    SPV_OWNERSHIP = "spv_ownership"
    DERIVATIVE_RIGHTS = "derivative_rights"


class SmartContractAudit(str, Enum):
    """Audit status of the token contract."""

    AUDITED_BIG_FOUR = "audited_big_four"  # Trail of Bits / Quantstamp / etc.
    AUDITED_INDIE = "audited_indie"
    SELF_AUDITED = "self_audited"
    UNAUDITED = "unaudited"


class YieldDenomination(str, Enum):
    """How rental yield is paid.

    Stablecoin payments need scholar review for the stablecoin's
    own halal status; some are cash-backed (acceptable), some are
    backed by interest-bearing instruments (not).
    """

    RENT_DIRECT_FIAT = "rent_direct_fiat"
    USDC_STABLECOIN = "usdc_stablecoin"  # cash-backed; generally OK
    USDT_STABLECOIN = "usdt_stablecoin"  # commercial paper; debated
    OTHER_STABLECOIN = "other_stablecoin"
    NATIVE_CRYPTO = "native_crypto"  # ETH / SOL / etc.
    NONE = "none"  # no yield (capital-appreciation only)


# Stablecoins generally accepted by Islamic-finance scholars for
# rental-yield payment. Operators extend via code review (the set
# is closed at the type level so a runtime config drift can't
# silently approve a non-halal stablecoin).
_HALAL_YIELD_DENOMINATIONS: frozenset[YieldDenomination] = frozenset(
    {
        YieldDenomination.RENT_DIRECT_FIAT,
        YieldDenomination.USDC_STABLECOIN,
        YieldDenomination.NONE,
    }
)


class DeFiIntegration(str, Enum):
    """Whether the protocol enables on-chain lending / borrowing.

    Pinned: LENDING_ENABLED or BORROWING_ENABLED is NOT_HALAL by
    construction — the protocol enables riba even if the operator
    doesn't personally lend / borrow.
    """

    STANDALONE = "standalone"  # token transfer only
    LENDING_ENABLED = "lending_enabled"  # holders can lend out
    BORROWING_ENABLED = "borrowing_enabled"  # use as collateral
    BOTH_ENABLED = "both_enabled"


_RIBA_DEFI_INTEGRATIONS: frozenset[DeFiIntegration] = frozenset(
    {
        DeFiIntegration.LENDING_ENABLED,
        DeFiIntegration.BORROWING_ENABLED,
        DeFiIntegration.BOTH_ENABLED,
    }
)


class TokenisedREITVerdict(str, Enum):
    """Screen verdict.

    Pinned string values for JSON / DB stability — dashboard +
    exception-queue UI key on these literals.
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    DOUBTFUL = "doubtful"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class TokenisedREITDeal:
    """The minimum data the screener needs.

    `lockup_days` matters because tokenised RE platforms typically
    impose 30-90 day holding periods to discourage short-term
    speculation; operator-side disclosure is required for the
    user-consent flow.
    """

    symbol: str
    platform: str  # RealT / Lofty / Propy / etc.
    property_type: REITPropertyType
    token_standard: TokenStandard
    regulator: RegulatorRegistration
    custody_model: CustodyModel
    audit_status: SmartContractAudit
    yield_denomination: YieldDenomination
    defi_integration: DeFiIntegration
    lockup_days: int

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.platform or not self.platform.strip():
            raise ValueError("platform must be non-empty")
        if self.lockup_days < 0:
            raise ValueError("lockup_days must be non-negative")


@dataclass(frozen=True)
class TokenisedREITScreenResult:
    """Screen verdict + supporting flags + audit notes."""

    symbol: str
    platform: str
    property_type: REITPropertyType
    verdict: TokenisedREITVerdict
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def screen_tokenised_reit(deal: TokenisedREITDeal) -> TokenisedREITScreenResult:
    """Apply the tokenised-REIT halal screen.

    Returns a `TokenisedREITScreenResult` with verdict + per-rule
    failure / warning lists for the audit trail.
    """

    failures: list[str] = []
    warnings: list[str] = []

    # Hard rejections — unconditional NOT_HALAL gates.
    if deal.custody_model is CustodyModel.DERIVATIVE_RIGHTS:
        failures.append("custody=derivative_rights: holder owns no physical asset (gharar)")

    if deal.defi_integration in _RIBA_DEFI_INTEGRATIONS:
        failures.append(
            f"defi_integration={deal.defi_integration.value}: "
            "protocol enables riba via lending / borrowing"
        )

    # Property-level hard rejection: hotel and specialty under the
    # Wave 1.I REIT framework are DOUBTFUL not NOT_HALAL — but the
    # tokenised wrapper doesn't add anything that flips the property-
    # level verdict.

    # INSUFFICIENT_DATA gate.
    if deal.token_standard is TokenStandard.UNKNOWN:
        return TokenisedREITScreenResult(
            symbol=deal.symbol,
            platform=deal.platform,
            property_type=deal.property_type,
            verdict=TokenisedREITVerdict.INSUFFICIENT_DATA,
            failures=tuple(failures),
            warnings=("token_standard is UNKNOWN — operator must verify before allocating",),
        )

    if failures:
        return TokenisedREITScreenResult(
            symbol=deal.symbol,
            platform=deal.platform,
            property_type=deal.property_type,
            verdict=TokenisedREITVerdict.NOT_HALAL,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Soft warnings — drive DOUBTFUL.
    if deal.regulator is RegulatorRegistration.NONE:
        warnings.append(
            "no regulator registration: unregistered offerings carry legal + gharar concerns"
        )

    if deal.audit_status is SmartContractAudit.UNAUDITED:
        warnings.append("smart contract is UNAUDITED: malware risk + custody gharar")
    elif deal.audit_status is SmartContractAudit.SELF_AUDITED:
        warnings.append("smart contract is SELF_AUDITED: third-party audit recommended")

    if deal.yield_denomination not in _HALAL_YIELD_DENOMINATIONS:
        warnings.append(
            f"yield_denomination={deal.yield_denomination.value}: "
            "stablecoin / native-crypto yield needs scholar review for halal status"
        )

    if deal.custody_model is CustodyModel.SPV_OWNERSHIP:
        warnings.append(
            "custody=spv_ownership: legal title held by SPV; "
            "transfer enforceability depends on jurisdiction"
        )

    # Hotel / specialty property types carry their own DOUBTFUL flag
    # from the Wave 1.I framework — surface the warning here.
    if deal.property_type in {
        REITPropertyType.HOTEL,
        REITPropertyType.SPECIALTY,
    }:
        warnings.append(
            f"{deal.property_type.value} property type requires scholar review "
            "regardless of tokenisation wrapper"
        )

    if warnings:
        verdict = TokenisedREITVerdict.DOUBTFUL
    else:
        verdict = TokenisedREITVerdict.HALAL

    return TokenisedREITScreenResult(
        symbol=deal.symbol,
        platform=deal.platform,
        property_type=deal.property_type,
        verdict=verdict,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


_VERDICT_EMOJI: dict[TokenisedREITVerdict, str] = {
    TokenisedREITVerdict.HALAL: "✅",
    TokenisedREITVerdict.NOT_HALAL: "❌",
    TokenisedREITVerdict.DOUBTFUL: "⚠️",
    TokenisedREITVerdict.INSUFFICIENT_DATA: "❓",
}


def render_screen_result(result: TokenisedREITScreenResult) -> str:
    """Format the screen result for ops display.

    Pinned no-address contract: never includes wallet addresses
    or token contract addresses. Operators audit the on-chain
    side via blockchain explorer separately.
    """

    emoji = _VERDICT_EMOJI[result.verdict]
    lines = [
        f"{emoji} {result.symbol} ({result.platform}) — {result.verdict.value.upper()}",
        f"  property: {result.property_type.value}",
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
    "CustodyModel",
    "DeFiIntegration",
    "RegulatorRegistration",
    "SmartContractAudit",
    "TokenStandard",
    "TokenisedREITDeal",
    "TokenisedREITScreenResult",
    "TokenisedREITVerdict",
    "YieldDenomination",
    "render_screen_result",
    "screen_tokenised_reit",
]
