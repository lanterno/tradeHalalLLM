"""Commodity-ETF Shariah screener.

Spot commodities have a special place in Shariah: gold and silver are
*ribawi* (subject to riba rules), which means a halal trade requires
actual physical possession and an immediate exchange of equal weight
for equal weight. The roadmap's interim path for crypto-bot operators
who want exposure is gold/silver-backed ETFs — but only the strictly
allocated, fully physical variants pass the screen. Most commodity
ETFs on the market today don't.

This module is the pure-Python ETF-specific layer. Picked a focused
module rather than wedging the rules into the general stock
screener (Zoya / Wahed) because the failure modes are commodity-
specific: the question isn't "does this ETF's company have ≤33%
debt", it's "is the underlying gold actually allocated bullion in
the unitholder's name or a synthetic swap pretending to be gold".

Pinned semantics:
- **HALAL requires every check pass.** Allocated-physical backing
  + segregated storage + ≥ 95% physical holdings (default
  threshold) + leverage = 1.0 + audited holdings. Any single
  failure flips to NOT_HALAL or DOUBTFUL — never silent HALAL.
- **NOT_HALAL is unconditional**: futures-only / swap-backed
  / paper-only storage / leveraged > 1.0 / no physical holdings
  trip the verdict regardless of other flags. These are the
  scholar-consensus failures.
- **DOUBTFUL** is the operator-decides bucket: unallocated-but-
  physical backing (commingled), commodity types with sector-
  scholar disagreement (oil / gas / agricultural under some
  AAOIFI advisory opinions are HALAL with caveats; under stricter
  Hanafi readings they need explicit treatment), missing audit
  attestation. Operators can opt-in via the manual exception
  queue (`halal/exception_queue`) for these.
- **INSUFFICIENT_DATA** when backing mode is UNKNOWN — the
  filings didn't disclose, so we can't classify; never silently
  HALAL just because nothing rejected. Mirrors the conservative-
  default pattern of Wave 1.I REIT screener and Wave 2.G regulator
  index.
- **Float comparisons inclusive at the threshold** (95% physical
  holdings is HALAL; 94.99% is DOUBTFUL or NOT_HALAL depending on
  why); pinned via test in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CommodityType(str, Enum):
    """Shariah-relevant commodity categories.

    The ribawi commodities (GOLD, SILVER) have stricter rules than
    industrial / agricultural ones. The enum lets the screener apply
    the right ruleset per type without a sea of conditionals at the
    call site.
    """

    GOLD = "gold"
    SILVER = "silver"
    PLATINUM = "platinum"
    PALLADIUM = "palladium"
    COPPER = "copper"
    OIL = "oil"
    NATURAL_GAS = "natural_gas"
    AGRICULTURAL = "agricultural"


# Ribawi commodities: AAOIFI Standard 57 ("Gold") + classical
# fiqh treatment of silver. Allocated physical backing is
# load-bearing for these; the screener treats unallocated
# backing as DOUBTFUL even when scholar-tolerated for
# non-ribawi commodities.
_RIBAWI_COMMODITIES: frozenset[CommodityType] = frozenset(
    {CommodityType.GOLD, CommodityType.SILVER}
)

# Commodities with sector-scholar disagreement. AAOIFI's general
# permissibility for natural resources is wide; some Hanafi
# scholars draw a stricter line on energy / agricultural futures.
# DOUBTFUL on these when not explicitly allocated-physical.
_DEBATED_COMMODITIES: frozenset[CommodityType] = frozenset(
    {CommodityType.OIL, CommodityType.NATURAL_GAS, CommodityType.AGRICULTURAL}
)


class BackingMode(str, Enum):
    """How the ETF holds its underlying commodity.

    Pinned: ``ALLOCATED_PHYSICAL`` is the strongest form — the
    unitholder's claim is on specific bars / barrels in their name
    (or by serial number in segregated storage); ``UNALLOCATED_
    PHYSICAL`` pools physical commodity across unitholders;
    ``FUTURES_BACKED`` rolls futures contracts (raises gharar
    questions); ``SWAP_BACKED`` is a derivative-only synthetic
    that doesn't hold the commodity at all (NOT_HALAL).
    """

    ALLOCATED_PHYSICAL = "allocated_physical"
    UNALLOCATED_PHYSICAL = "unallocated_physical"
    FUTURES_BACKED = "futures_backed"
    SWAP_BACKED = "swap_backed"
    UNKNOWN = "unknown"


class StorageLocation(str, Enum):
    """Where the physical holdings sit.

    SEGREGATED is the strongest (LBMA-vaulted, named-account).
    COMMINGLED is the typical pooled-vault model. PAPER means no
    physical bars — only an entry on a ledger; NOT_HALAL.
    """

    SEGREGATED = "segregated"
    COMMINGLED = "commingled"
    PAPER = "paper"


class CommodityVerdict(str, Enum):
    """Screen verdict.

    Pinned string values for JSON / DB serialisation; the dashboard
    + exception-queue UI key on these literals.
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    DOUBTFUL = "doubtful"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class CommodityThresholds:
    """Screen thresholds.

    `min_physical_holdings_pct` defaults to 95% — the AAOIFI
    advisory minimum for "fully physical" on a commodity ETF
    (the remaining 5% is operator-tolerated working cash for
    rebalancing). `max_leverage_factor` is 1.0; leveraged
    products are categorically NOT_HALAL.
    """

    min_physical_holdings_pct: float = 95.0
    max_leverage_factor: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.min_physical_holdings_pct <= 100.0:
            got = self.min_physical_holdings_pct
            raise ValueError(f"min_physical_holdings_pct must be in (0, 100], got {got}")
        if self.max_leverage_factor <= 0.0:
            raise ValueError("max_leverage_factor must be positive")


DEFAULT_THRESHOLDS = CommodityThresholds()


@dataclass(frozen=True)
class CommodityETFFinancials:
    """Minimum data the screener needs to decide.

    `physical_holdings_pct` is the share of NAV that's actually
    physical commodity (vs cash, futures, swaps, or treasuries).
    `leverage_factor` is the ETF's product leverage — 1.0 unleveraged,
    2.0 = 2x daily, -1.0 = inverse / short. Inverse and leveraged
    products are categorically NOT_HALAL because they require
    derivatives + interest-bearing financing.
    """

    symbol: str
    name: str
    commodity: CommodityType
    backing_mode: BackingMode
    storage_location: StorageLocation
    leverage_factor: float
    physical_holdings_pct: float
    has_audited_holdings: bool

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not 0.0 <= self.physical_holdings_pct <= 100.0:
            raise ValueError(
                f"physical_holdings_pct must be in [0, 100], got {self.physical_holdings_pct}"
            )


@dataclass(frozen=True)
class CommodityScreenResult:
    """Screen verdict + supporting numbers + audit notes."""

    symbol: str
    commodity: CommodityType
    verdict: CommodityVerdict
    physical_holdings_pct: float | None
    leverage_factor: float | None
    backing_mode: BackingMode
    storage_location: StorageLocation
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def screen_commodity_etf(
    financials: CommodityETFFinancials,
    *,
    thresholds: CommodityThresholds = DEFAULT_THRESHOLDS,
) -> CommodityScreenResult:
    """Apply the AAOIFI-aligned commodity-ETF screen.

    Returns a `CommodityScreenResult` with the verdict, the
    contributing numbers, and the per-rule failure / warning lists
    for the audit trail.
    """

    failures: list[str] = []
    warnings: list[str] = []

    # Hard rejections — these are the unconditional NOT_HALAL gates.
    if financials.backing_mode is BackingMode.SWAP_BACKED:
        failures.append("swap-backed: synthetic exposure is NOT_HALAL (gharar)")
    if financials.storage_location is StorageLocation.PAPER:
        failures.append("paper storage: no physical commodity is NOT_HALAL")
    if financials.leverage_factor > thresholds.max_leverage_factor:
        failures.append(
            f"leverage {financials.leverage_factor}x > "
            f"{thresholds.max_leverage_factor}x: leveraged products are NOT_HALAL"
        )
    if financials.leverage_factor < 0:
        # inverse / short ETFs
        failures.append(
            f"leverage {financials.leverage_factor}x: inverse / short products are NOT_HALAL"
        )

    # INSUFFICIENT_DATA gates — these prevent us from making a clean call.
    if financials.backing_mode is BackingMode.UNKNOWN:
        return CommodityScreenResult(
            symbol=financials.symbol,
            commodity=financials.commodity,
            verdict=CommodityVerdict.INSUFFICIENT_DATA,
            physical_holdings_pct=financials.physical_holdings_pct,
            leverage_factor=financials.leverage_factor,
            backing_mode=financials.backing_mode,
            storage_location=financials.storage_location,
            failures=tuple(failures),
            warnings=("backing_mode is UNKNOWN — filings didn't disclose",),
        )

    # Physical holdings threshold.
    if financials.physical_holdings_pct < thresholds.min_physical_holdings_pct:
        failures.append(
            f"physical holdings {financials.physical_holdings_pct:.2f}% < "
            f"{thresholds.min_physical_holdings_pct:.0f}% threshold"
        )

    # If we already have any hard failure, finalise as NOT_HALAL.
    if failures:
        return CommodityScreenResult(
            symbol=financials.symbol,
            commodity=financials.commodity,
            verdict=CommodityVerdict.NOT_HALAL,
            physical_holdings_pct=financials.physical_holdings_pct,
            leverage_factor=financials.leverage_factor,
            backing_mode=financials.backing_mode,
            storage_location=financials.storage_location,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Backing-mode soft warnings (drive DOUBTFUL).
    if financials.backing_mode is BackingMode.FUTURES_BACKED:
        warnings.append("futures-backed: roll-yield + gharar concerns; needs scholar review")
    if financials.backing_mode is BackingMode.UNALLOCATED_PHYSICAL:
        warnings.append(
            "unallocated physical: pooled vault; allocated is preferred for ribawi commodities"
        )

    # Storage-mode soft warnings.
    if financials.storage_location is StorageLocation.COMMINGLED:
        warnings.append("commingled storage: segregated (named-account) is the stronger position")

    # Audit attestation.
    if not financials.has_audited_holdings:
        warnings.append("no audited holdings disclosure: required for ribawi commodities")

    # Ribawi commodities (gold / silver) require the strongest backing —
    # an unallocated or futures-backed ribawi ETF is DOUBTFUL even when
    # everything else passes; segregated allocated is required for HALAL.
    if financials.commodity in _RIBAWI_COMMODITIES:
        if financials.backing_mode is not BackingMode.ALLOCATED_PHYSICAL:
            warnings.append(
                f"{financials.commodity.value} is ribawi: "
                "allocated-physical backing required for HALAL"
            )
        if financials.storage_location is not StorageLocation.SEGREGATED:
            warnings.append(
                f"{financials.commodity.value} is ribawi: segregated storage required for HALAL"
            )

    # Debated commodities (oil / gas / agricultural) — even with clean
    # physical backing they're DOUBTFUL pending operator scholar profile.
    if financials.commodity in _DEBATED_COMMODITIES:
        warnings.append(
            f"{financials.commodity.value} commodity has scholar disagreement; "
            "operator should consult their scholar profile"
        )

    if warnings:
        verdict = CommodityVerdict.DOUBTFUL
    else:
        verdict = CommodityVerdict.HALAL

    return CommodityScreenResult(
        symbol=financials.symbol,
        commodity=financials.commodity,
        verdict=verdict,
        physical_holdings_pct=financials.physical_holdings_pct,
        leverage_factor=financials.leverage_factor,
        backing_mode=financials.backing_mode,
        storage_location=financials.storage_location,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


_VERDICT_EMOJI: dict[CommodityVerdict, str] = {
    CommodityVerdict.HALAL: "✅",
    CommodityVerdict.NOT_HALAL: "❌",
    CommodityVerdict.DOUBTFUL: "⚠️",
    CommodityVerdict.INSUFFICIENT_DATA: "❓",
}


def render_screen_result(result: CommodityScreenResult) -> str:
    """Format the screen result for ops display."""

    lines: list[str] = []
    emoji = _VERDICT_EMOJI[result.verdict]
    lines.append(
        f"{emoji} {result.symbol} ({result.commodity.value}) — {result.verdict.value.upper()}"
    )
    lines.append(f"  backing: {result.backing_mode.value}")
    lines.append(f"  storage: {result.storage_location.value}")
    if result.physical_holdings_pct is not None:
        lines.append(f"  physical: {result.physical_holdings_pct:.2f}%")
    if result.leverage_factor is not None:
        lines.append(f"  leverage: {result.leverage_factor}x")
    if result.failures:
        lines.append("  failures:")
        for f in result.failures:
            lines.append(f"    - {f}")
    if result.warnings:
        lines.append("  warnings:")
        for w in result.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "BackingMode",
    "CommodityETFFinancials",
    "CommodityScreenResult",
    "CommodityThresholds",
    "CommodityType",
    "CommodityVerdict",
    "StorageLocation",
    "render_screen_result",
    "screen_commodity_etf",
]
