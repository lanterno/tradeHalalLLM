"""Sukuk (Islamic-bond) Shariah screener.

Sukuk are the halal alternative to conventional bonds. Instead of
lending money in exchange for fixed interest, a sukuk represents
proportional ownership in a tangible asset, lease, or partnership;
the holder earns returns from rental income, profit share, or
trading margin rather than riba. The roadmap defers the live
ingest path (Bloomberg / Refinitiv vendor cost) but the screening
*logic* is operator-supplied or CSV-fed pure-Python — exactly the
isolated-module pattern of Wave 1.G commodities, 1.I REIT, and
2.G regulator-index.

Picked a focused module rather than the general stock screener
because the failure modes are sukuk-specific: the question isn't
"does the issuer have ≤33% conventional debt", it's "is this
specific issue structured under one of the AAOIFI-approved
contracts, backed by halal underlying assets, with a profit-share
or rental coupon (not interest), and certified by a recognised
shariah board".

Pinned semantics:
- **HALAL requires every check pass.** AAOIFI-approved structure
  + AAOIFI-certified issuance + halal-asset-class underlying +
  profit-share / rental coupon (not interest) + permissible
  issuer + asset-backed (true sale) preferred. Any single failure
  flips to NOT_HALAL or DOUBTFUL — never silent HALAL.
- **NOT_HALAL is unconditional**: INTEREST coupon (the riba red
  line), PROHIBITED_OPERATIONS asset class (alcohol / gambling /
  conventional banking operations), FINANCIAL_RECEIVABLES asset
  class (conventional debt portfolio dressed up as sukuk).
- **DOUBTFUL** is the operator-decides bucket: not AAOIFI-
  certified (the issuance may be technically structured right
  but lacks the third-party scholar attestation), MURABAHA
  structure under strict mode (some scholars dispute its
  permissibility for tradeable sukuk because the underlying
  cost-plus contract is debt-like once issued), CONVENTIONAL_BANK
  issuer (the sukuk itself can be structured halal but the
  issuer's other ops disqualify it for stricter operators),
  asset-based (vs asset-backed) under strict mode, very short
  tenor (treat as money-market instrument; needs different
  treatment).
- **INSUFFICIENT_DATA** when asset_class is UNKNOWN — the
  prospectus didn't disclose the underlying assets clearly, so
  we can't classify; never silently HALAL just because nothing
  rejected. Mirrors the conservative-default pattern of Wave 1.G
  commodities, 1.I REIT, and 2.G regulator index.
- **FIXED_PROFIT_RATE coupon is HALAL when contractually structured**
  even when the rate is benchmark-pegged (LIBOR-pegged) — AAOIFI
  permits a contractually-fixed ijarah rental that *uses* a
  benchmark for transparency without making the rental itself an
  interest payment. The pin matters because a sloppy implementation
  would conflate "uses LIBOR" with "is interest" and reject every
  modern sukuk; AAOIFI is explicit on the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SukukStructure(str, Enum):
    """AAOIFI-approved sukuk structures.

    `IJARAH` (lease-based) and `MUSHARAKA` (partnership) are the
    most universally accepted. `MURABAHA` (cost-plus) is contested
    for tradeable sukuk because once issued it's debt-like.
    `HYBRID` covers Wakala-Murabaha and similar mixed structures.
    """

    IJARAH = "ijarah"
    MUSHARAKA = "musharaka"
    MUDARABA = "mudaraba"
    WAKALA = "wakala"
    MURABAHA = "murabaha"
    SALAM = "salam"
    ISTISNA = "istisna"
    HYBRID = "hybrid"


# Structures with broad scholar consensus.
_BROADLY_ACCEPTED_STRUCTURES: frozenset[SukukStructure] = frozenset(
    {
        SukukStructure.IJARAH,
        SukukStructure.MUSHARAKA,
        SukukStructure.MUDARABA,
        SukukStructure.WAKALA,
        SukukStructure.SALAM,
        SukukStructure.ISTISNA,
    }
)

# Structures with sector-scholar disagreement on tradability.
_CONTESTED_STRUCTURES: frozenset[SukukStructure] = frozenset({SukukStructure.MURABAHA})


class AssetClass(str, Enum):
    """The underlying tangible asset backing the sukuk.

    Halal asset classes (REAL_ESTATE / INFRASTRUCTURE / EQUIPMENT /
    AIRCRAFT / VEHICLES / POWER_PLANTS / UTILITIES) all pass.
    `FINANCIAL_RECEIVABLES` is the canonical NOT_HALAL case — a
    pool of conventional loans dressed up as sukuk; the underlying
    is debt and the income is interest, regardless of structure.
    `PROHIBITED_OPERATIONS` covers alcohol / gambling / conventional
    banking operations / pork / arms — categorically NOT_HALAL.
    """

    REAL_ESTATE = "real_estate"
    INFRASTRUCTURE = "infrastructure"
    EQUIPMENT = "equipment"
    AIRCRAFT = "aircraft"
    VEHICLES = "vehicles"
    POWER_PLANTS = "power_plants"
    UTILITIES = "utilities"
    FINANCIAL_RECEIVABLES = "financial_receivables"
    PROHIBITED_OPERATIONS = "prohibited_operations"
    UNKNOWN = "unknown"


# Halal asset classes — the underlying backing is tangible and
# permissible.
_HALAL_ASSET_CLASSES: frozenset[AssetClass] = frozenset(
    {
        AssetClass.REAL_ESTATE,
        AssetClass.INFRASTRUCTURE,
        AssetClass.EQUIPMENT,
        AssetClass.AIRCRAFT,
        AssetClass.VEHICLES,
        AssetClass.POWER_PLANTS,
        AssetClass.UTILITIES,
    }
)

# Asset classes that unconditionally reject the sukuk regardless
# of structure / coupon / certification.
_FORBIDDEN_ASSET_CLASSES: frozenset[AssetClass] = frozenset(
    {AssetClass.FINANCIAL_RECEIVABLES, AssetClass.PROHIBITED_OPERATIONS}
)


class IssuerType(str, Enum):
    """Issuer category — affects DOUBTFUL bucketing.

    `SOVEREIGN` (GCC, Indonesia, Malaysia, etc.) and `SUPRANATIONAL`
    (Islamic Development Bank, IFC) carry the strongest issuer
    permissibility. `CORPORATE_HALAL` is a corporate with
    documented halal-only operations. `CORPORATE_MIXED` triggers
    a DOUBTFUL warning under strict mode. `CONVENTIONAL_BANK`
    issuing sukuk is the most contested case — the sukuk can be
    structured halal but the issuer's other operations disqualify
    it for stricter operators.
    """

    SOVEREIGN = "sovereign"
    SUPRANATIONAL = "supranational"
    CORPORATE_HALAL = "corporate_halal"
    CORPORATE_MIXED = "corporate_mixed"
    CONVENTIONAL_BANK = "conventional_bank"


class CouponType(str, Enum):
    """How the sukuk's periodic returns are structured.

    `PROFIT_SHARE` (Mudaraba / Musharaka) and `RENTAL_INCOME`
    (Ijarah) are the cleanest. `FIXED_PROFIT_RATE` is contractual
    and AAOIFI-permitted even when benchmark-pegged for
    transparency. `INTEREST` is the riba red line — categorically
    NOT_HALAL.
    """

    PROFIT_SHARE = "profit_share"
    RENTAL_INCOME = "rental_income"
    FIXED_PROFIT_RATE = "fixed_profit_rate"
    INTEREST = "interest"


class SukukVerdict(str, Enum):
    """Screen verdict.

    Pinned string values for JSON / DB serialisation; the dashboard
    + exception-queue UI key on these literals.
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    DOUBTFUL = "doubtful"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class SukukThresholds:
    """Operator-tunable screen thresholds.

    `require_aaoifi_certified=True` is the default — operators
    overriding to False accept structurally-correct issues without
    third-party attestation (e.g., a sovereign issue from a country
    that uses its own AAOIFI-equivalent body). `accept_murabaha`
    defaults to True (AAOIFI-permitted); strict-Hanafi operators
    set False. `accept_asset_based` defaults to True (the bulk of
    GCC sukuk are asset-based not asset-backed); strict operators
    set False to require true-sale asset-backed structures.
    `min_maturity_days` filters out money-market-instrument-shaped
    issues that need different treatment.
    """

    require_aaoifi_certified: bool = True
    accept_murabaha: bool = True
    accept_asset_based: bool = True
    min_maturity_days: int = 1

    def __post_init__(self) -> None:
        if self.min_maturity_days < 0:
            raise ValueError("min_maturity_days must be non-negative")


DEFAULT_THRESHOLDS = SukukThresholds()


@dataclass(frozen=True)
class SukukIssue:
    """Minimum data the screener needs to decide.

    `is_asset_backed` distinguishes true-sale sukuk (holders
    actually own the asset; default-recourse is the asset)
    from asset-based sukuk (issuer retains beneficial ownership;
    holders only have a contractual claim on the income stream).
    Most modern GCC sukuk are asset-based; AAOIFI permits but
    flags asset-backed as the stronger structure.
    """

    isin: str
    issuer_name: str
    issuer_type: IssuerType
    structure: SukukStructure
    asset_class: AssetClass
    coupon_type: CouponType
    is_aaoifi_certified: bool
    is_asset_backed: bool
    maturity_days: int
    expected_yield_pct: float

    def __post_init__(self) -> None:
        if not self.isin or not self.isin.strip():
            raise ValueError("isin must be non-empty")
        if not self.issuer_name or not self.issuer_name.strip():
            raise ValueError("issuer_name must be non-empty")
        if self.maturity_days < 0:
            raise ValueError("maturity_days must be non-negative")
        if self.expected_yield_pct < 0:
            raise ValueError("expected_yield_pct must be non-negative")


@dataclass(frozen=True)
class SukukScreenResult:
    """Screen verdict + supporting numbers + audit notes."""

    isin: str
    issuer_name: str
    structure: SukukStructure
    asset_class: AssetClass
    coupon_type: CouponType
    verdict: SukukVerdict
    is_aaoifi_certified: bool
    is_asset_backed: bool
    maturity_days: int
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def screen_sukuk(
    issue: SukukIssue,
    *,
    thresholds: SukukThresholds = DEFAULT_THRESHOLDS,
) -> SukukScreenResult:
    """Apply the AAOIFI-aligned sukuk screen.

    Returns a `SukukScreenResult` with the verdict, the contributing
    flags, and per-rule failure / warning lists for the audit trail.
    """

    failures: list[str] = []
    warnings: list[str] = []

    # Hard rejections — unconditional NOT_HALAL gates.
    if issue.coupon_type is CouponType.INTEREST:
        failures.append("interest coupon: riba is NOT_HALAL unconditionally")

    if issue.asset_class is AssetClass.PROHIBITED_OPERATIONS:
        failures.append(
            "prohibited operations underlying: NOT_HALAL "
            "(alcohol / gambling / conventional banking ops / pork / arms)"
        )
    if issue.asset_class is AssetClass.FINANCIAL_RECEIVABLES:
        failures.append(
            "financial-receivables underlying: conventional debt portfolio dressed as sukuk"
        )

    # INSUFFICIENT_DATA gate.
    if issue.asset_class is AssetClass.UNKNOWN:
        return SukukScreenResult(
            isin=issue.isin,
            issuer_name=issue.issuer_name,
            structure=issue.structure,
            asset_class=issue.asset_class,
            coupon_type=issue.coupon_type,
            verdict=SukukVerdict.INSUFFICIENT_DATA,
            is_aaoifi_certified=issue.is_aaoifi_certified,
            is_asset_backed=issue.is_asset_backed,
            maturity_days=issue.maturity_days,
            failures=tuple(failures),
            warnings=("asset_class is UNKNOWN — prospectus didn't disclose",),
        )

    # If we already have any hard failure, finalise as NOT_HALAL.
    if failures:
        return SukukScreenResult(
            isin=issue.isin,
            issuer_name=issue.issuer_name,
            structure=issue.structure,
            asset_class=issue.asset_class,
            coupon_type=issue.coupon_type,
            verdict=SukukVerdict.NOT_HALAL,
            is_aaoifi_certified=issue.is_aaoifi_certified,
            is_asset_backed=issue.is_asset_backed,
            maturity_days=issue.maturity_days,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Soft warnings — drive DOUBTFUL.
    if thresholds.require_aaoifi_certified and not issue.is_aaoifi_certified:
        warnings.append("not AAOIFI-certified: lacks third-party shariah-board attestation")

    if issue.structure in _CONTESTED_STRUCTURES and not thresholds.accept_murabaha:
        warnings.append(
            f"{issue.structure.value} structure under strict mode: "
            "some scholars dispute tradability"
        )

    if issue.issuer_type is IssuerType.CONVENTIONAL_BANK:
        warnings.append(
            "issuer is a conventional bank: sukuk structurally permissible "
            "but issuer's non-sukuk operations disqualify under strict screens"
        )
    elif issue.issuer_type is IssuerType.CORPORATE_MIXED:
        warnings.append(
            "issuer has mixed business operations: "
            "operator should verify the sukuk's specific use of proceeds"
        )

    if not issue.is_asset_backed and not thresholds.accept_asset_based:
        warnings.append(
            "asset-based (not asset-backed) under strict mode: "
            "true-sale asset-backed is the stronger position"
        )

    if issue.maturity_days < thresholds.min_maturity_days:
        warnings.append(
            f"maturity {issue.maturity_days}d below "
            f"{thresholds.min_maturity_days}d threshold: "
            "treat as money-market instrument"
        )

    if warnings:
        verdict = SukukVerdict.DOUBTFUL
    else:
        verdict = SukukVerdict.HALAL

    return SukukScreenResult(
        isin=issue.isin,
        issuer_name=issue.issuer_name,
        structure=issue.structure,
        asset_class=issue.asset_class,
        coupon_type=issue.coupon_type,
        verdict=verdict,
        is_aaoifi_certified=issue.is_aaoifi_certified,
        is_asset_backed=issue.is_asset_backed,
        maturity_days=issue.maturity_days,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


_VERDICT_EMOJI: dict[SukukVerdict, str] = {
    SukukVerdict.HALAL: "✅",
    SukukVerdict.NOT_HALAL: "❌",
    SukukVerdict.DOUBTFUL: "⚠️",
    SukukVerdict.INSUFFICIENT_DATA: "❓",
}


def render_screen_result(result: SukukScreenResult) -> str:
    """Format the screen result for ops display."""

    lines: list[str] = []
    emoji = _VERDICT_EMOJI[result.verdict]
    lines.append(f"{emoji} {result.isin} ({result.issuer_name}) — {result.verdict.value.upper()}")
    lines.append(f"  structure: {result.structure.value}")
    lines.append(f"  asset_class: {result.asset_class.value}")
    lines.append(f"  coupon: {result.coupon_type.value}")
    lines.append(f"  aaoifi_certified: {result.is_aaoifi_certified}")
    lines.append(f"  asset_backed: {result.is_asset_backed}")
    lines.append(f"  maturity: {result.maturity_days}d")
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
    "AssetClass",
    "CouponType",
    "IssuerType",
    "SukukIssue",
    "SukukScreenResult",
    "SukukStructure",
    "SukukThresholds",
    "SukukVerdict",
    "render_screen_result",
    "screen_sukuk",
]
