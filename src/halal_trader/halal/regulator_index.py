"""Regional regulator halal-index ingest.

Local-market regulators publish authoritative halal-compliance lists
for their listed equities:

- **Saudi Arabia** — Tadawul Halal Index (`TADAWUL`); Capital Market
  Authority (`CMA_HALAL`) broader screening.
- **Pakistan** — KSE-Meezan Index 30 (`KMI30`); Securities & Exchange
  Commission of Pakistan (`SECP_HALAL`) broader screening.

These indices outrank a third-party screener (Zoya / Wahed) for
listings on their home markets — the regulator has access to the
issuer's full filings and its verdict is the legally-recognised
shariah classification in-country. Operators trading Saudi or
Pakistani equities should fold the regional verdict into the
consensus screener as a high-priority source.

This module is the pure-Python ingestion + matching layer. It
defines the data shapes (a `RegulatorIndex` is a `RegulatorSource`,
a fetch timestamp, and a tuple of per-symbol `IndexListing`s),
provides a per-symbol screener (`screen_with_regulator`) that
respects market authority + staleness, and ships rendering helpers
for ops display. No HTTP / DB / async — the *fetcher* (live API
client / cron / CSV import) is a follow-up; this module exercises
the matching logic in isolation so the fetcher sees a stable
contract.

Pinned semantics:
- **Authority** is per-market: TADAWUL / CMA_HALAL only carry
  weight for `Market.SAUDI`; KMI30 / SECP_HALAL only for
  `Market.PAKISTAN`. Querying TADAWUL with `market=Market.OTHER`
  returns `UNKNOWN` with no source, *not* a silent NOT_HALAL —
  cross-market authority is a category error, not a failed screen.
- **Absence ≠ NOT_HALAL.** A symbol not present in the regulator's
  index is `UNKNOWN`, not NOT_HALAL — the index is a positive list
  (these are halal), not a negative list (these are forbidden).
  Mistaking absence for forbidden would silently disqualify every
  newly-listed Saudi equity until the next quarterly index update.
- **Staleness ladder**: listings within `stale_days` (default 90)
  are fresh; between `stale_days` and `expired_days` (default 365)
  carry a stale warning but the verdict still applies; older than
  `expired_days` are demoted to `UNKNOWN` with the staleness in
  warnings — past a year the underlying corporate facts may have
  shifted enough that the index hasn't re-confirmed.
- **Symbol normalisation** at lookup is case-insensitive +
  whitespace-stripped; Saudi tickers (4-digit numeric like `1010`
  Riyad Bank) and Pakistani tickers (alphanumeric like `HBL` Habib
  Bank) coexist because matching is on the literal string after
  normalisation, not market-specific parsing.
- **Multiple covering indices** combine via "any HALAL → HALAL"
  (regulator coverage is additive: if either Tadawul or CMA flags
  the symbol HALAL, that's enough). Conflicts (one HALAL, one
  NOT_HALAL) resolve conservatively to NOT_HALAL with both sources
  recorded — the same conservative-tiebreak philosophy as Wave 2.B
  consensus and Wave 4.J committee voting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RegulatorSource(str, Enum):
    """Regulator-published halal indices the bot can ingest."""

    TADAWUL = "tadawul"
    CMA_HALAL = "cma_halal"
    KMI30 = "kmi_30"
    SECP_HALAL = "secp_halal"


class Market(str, Enum):
    """Markets the bot can screen against regional regulators.

    `OTHER` covers any market without a regional authority in this
    registry (US, UK, EU, etc.). Regional regulator screening on
    those markets returns UNKNOWN by design — operator falls back to
    the global screener (Zoya / Wahed).
    """

    SAUDI = "saudi"
    PAKISTAN = "pakistan"
    OTHER = "other"


# Which market each regulator has authority over. The screener uses
# this to decide whether a regulator's index applies to a candidate
# symbol — a TADAWUL row is irrelevant to a Pakistani stock and
# vice-versa.
_AUTHORITY: dict[RegulatorSource, Market] = {
    RegulatorSource.TADAWUL: Market.SAUDI,
    RegulatorSource.CMA_HALAL: Market.SAUDI,
    RegulatorSource.KMI30: Market.PAKISTAN,
    RegulatorSource.SECP_HALAL: Market.PAKISTAN,
}


def regulator_market(source: RegulatorSource) -> Market:
    """Return the market a regulator has authority over."""

    return _AUTHORITY[source]


class RegulatorVerdict(str, Enum):
    """Per-symbol verdict from a regulator's index.

    `UNKNOWN` is the explicit "not present in this index" verdict —
    distinct from `NOT_HALAL` so consensus callers can route correctly
    (UNKNOWN → fall back to other sources; NOT_HALAL → reject).
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IndexListing:
    """One row from a regulator's published halal index."""

    symbol: str
    verdict: RegulatorVerdict
    listed_at: datetime
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.listed_at.tzinfo is None:
            raise ValueError("listed_at must be timezone-aware")


@dataclass(frozen=True)
class RegulatorIndex:
    """A snapshot of one regulator's halal index.

    The bot ingests one index per regulator per fetch cycle (Tadawul
    publishes quarterly; KMI-30 reviews semi-annually). Multiple
    indices in `screen_with_regulator(...)` cover both providers for
    a single market — the screener combines them per the conservative-
    tiebreak rule documented in the module docstring.
    """

    source: RegulatorSource
    fetched_at: datetime
    listings: tuple[IndexListing, ...]

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware")
        seen: set[str] = set()
        for listing in self.listings:
            key = listing.symbol.strip().upper()
            if key in seen:
                raise ValueError(f"duplicate symbol {key!r} in {self.source.value} index")
            seen.add(key)

    def lookup(self, symbol: str) -> IndexListing | None:
        """Case-insensitive + whitespace-stripped symbol match."""

        target = symbol.strip().upper()
        if not target:
            return None
        for listing in self.listings:
            if listing.symbol.strip().upper() == target:
                return listing
        return None

    @property
    def market(self) -> Market:
        return _AUTHORITY[self.source]


@dataclass(frozen=True)
class RegulatorThresholds:
    """Staleness ladder.

    `stale_days` is the warning threshold; `expired_days` demotes the
    verdict to UNKNOWN. Operators tracking a fast-moving market
    (Pakistan revises its KMI-30 every 6 months) typically set
    `expired_days=180` rather than the conservative-default 365.
    """

    stale_days: int = 90
    expired_days: int = 365

    def __post_init__(self) -> None:
        if self.stale_days <= 0:
            raise ValueError("stale_days must be positive")
        if self.expired_days <= 0:
            raise ValueError("expired_days must be positive")
        if self.expired_days < self.stale_days:
            raise ValueError(
                f"expired_days ({self.expired_days}) must be >= stale_days ({self.stale_days})"
            )


DEFAULT_THRESHOLDS = RegulatorThresholds()


@dataclass(frozen=True)
class RegulatorScreenResult:
    """The screen verdict + supporting numbers + audit notes."""

    symbol: str
    market: Market
    sources: tuple[RegulatorSource, ...]
    verdict: RegulatorVerdict
    oldest_listing_age_days: int | None
    is_stale: bool
    is_expired: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)
    matched_listings: tuple[IndexListing, ...] = field(default_factory=tuple)


def _combine_verdicts(verdicts: list[RegulatorVerdict]) -> RegulatorVerdict:
    """Conservative-tiebreak combine.

    Pinned: NOT_HALAL > UNKNOWN > HALAL is the override order when
    multiple sources disagree. The semantic: "any single regulator
    saying NOT_HALAL is enough to disqualify" mirrors Wave 2.B
    consensus's strict mode and ensures the bot never trades a
    symbol any covering regulator has rejected.
    """

    if not verdicts:
        return RegulatorVerdict.UNKNOWN
    if RegulatorVerdict.NOT_HALAL in verdicts:
        return RegulatorVerdict.NOT_HALAL
    if RegulatorVerdict.HALAL in verdicts:
        return RegulatorVerdict.HALAL
    return RegulatorVerdict.UNKNOWN


def screen_with_regulator(
    *,
    symbol: str,
    market: Market,
    indices: tuple[RegulatorIndex, ...],
    now: datetime,
    thresholds: RegulatorThresholds = DEFAULT_THRESHOLDS,
) -> RegulatorScreenResult:
    """Match a symbol against the regional regulators that cover its market.

    Returns a `RegulatorScreenResult` with the combined verdict, the
    list of regulators that contributed, the oldest listing age (so
    the operator can see how recent the regulator coverage is), and
    staleness flags.

    The function is pure: it does no I/O. Callers fetch indices and
    pass them in; the staleness check uses the provided `now` for
    determinism.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be non-empty")

    warnings: list[str] = []
    matched: list[IndexListing] = []
    contributing_sources: list[RegulatorSource] = []
    contributing_verdicts: list[RegulatorVerdict] = []
    listing_ages_days: list[int] = []

    if market is Market.OTHER:
        return RegulatorScreenResult(
            symbol=symbol,
            market=market,
            sources=(),
            verdict=RegulatorVerdict.UNKNOWN,
            oldest_listing_age_days=None,
            is_stale=False,
            is_expired=False,
            warnings=(
                "no regional regulator covers Market.OTHER; fall back to the global screener",
            ),
        )

    for index in indices:
        if index.market is not market:
            # Cross-market authority guard: a TADAWUL row does not
            # govern a Pakistani stock. Silently skip.
            continue
        listing = index.lookup(symbol)
        if listing is None:
            continue
        age = (now - listing.listed_at).days
        listing_ages_days.append(age)
        matched.append(listing)
        contributing_sources.append(index.source)

        if age >= thresholds.expired_days:
            warnings.append(
                f"{index.source.value} listing for {symbol!r} is {age}d old "
                f"(>= {thresholds.expired_days}d expiry); demoted to UNKNOWN"
            )
            contributing_verdicts.append(RegulatorVerdict.UNKNOWN)
        elif age >= thresholds.stale_days:
            warnings.append(
                f"{index.source.value} listing for {symbol!r} is {age}d old "
                f"(>= {thresholds.stale_days}d stale)"
            )
            contributing_verdicts.append(listing.verdict)
        else:
            contributing_verdicts.append(listing.verdict)

    verdict = _combine_verdicts(contributing_verdicts)

    if not contributing_sources and any(idx.market is market for idx in indices):
        # We had a covering index but the symbol wasn't in any of
        # them — the index is a positive list, so absence is UNKNOWN
        # (not NOT_HALAL). Surface that as an explicit warning.
        warnings.append(
            f"{symbol!r} not present in any covering regulator index for "
            f"{market.value}; absence is UNKNOWN, not NOT_HALAL"
        )

    if not contributing_sources and not any(idx.market is market for idx in indices):
        warnings.append(
            f"no regulator index covering {market.value} was provided; "
            "fall back to the global screener"
        )

    oldest_age = max(listing_ages_days) if listing_ages_days else None
    is_stale = oldest_age is not None and oldest_age >= thresholds.stale_days
    is_expired = oldest_age is not None and oldest_age >= thresholds.expired_days

    return RegulatorScreenResult(
        symbol=symbol,
        market=market,
        sources=tuple(contributing_sources),
        verdict=verdict,
        oldest_listing_age_days=oldest_age,
        is_stale=is_stale,
        is_expired=is_expired,
        warnings=tuple(warnings),
        matched_listings=tuple(matched),
    )


_VERDICT_EMOJI: dict[RegulatorVerdict, str] = {
    RegulatorVerdict.HALAL: "✅",
    RegulatorVerdict.NOT_HALAL: "❌",
    RegulatorVerdict.UNKNOWN: "❓",
}


def render_screen_result(result: RegulatorScreenResult) -> str:
    """Format the result for ops display."""

    lines: list[str] = []
    emoji = _VERDICT_EMOJI[result.verdict]
    lines.append(
        f"{emoji} {result.symbol} ({result.market.value}) — {result.verdict.value.upper()}"
    )
    if result.sources:
        sources_list = ", ".join(s.value for s in result.sources)
        lines.append(f"  sources: {sources_list}")
    if result.oldest_listing_age_days is not None:
        suffix = " (expired)" if result.is_expired else (" (stale)" if result.is_stale else "")
        lines.append(f"  oldest listing: {result.oldest_listing_age_days}d ago{suffix}")
    if result.warnings:
        lines.append("  warnings:")
        for w in result.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)


def listing_age_days(listing: IndexListing, now: datetime) -> int:
    """Convenience: age of one listing in days."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return (now - listing.listed_at).days


def newest_index(indices: tuple[RegulatorIndex, ...]) -> RegulatorIndex | None:
    """Convenience: most-recently-fetched index, or None if empty."""

    if not indices:
        return None
    return max(indices, key=lambda idx: idx.fetched_at)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "IndexListing",
    "Market",
    "RegulatorIndex",
    "RegulatorScreenResult",
    "RegulatorSource",
    "RegulatorThresholds",
    "RegulatorVerdict",
    "listing_age_days",
    "newest_index",
    "regulator_market",
    "render_screen_result",
    "screen_with_regulator",
]
