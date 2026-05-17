"""Per-jurisdiction halal-screen routing engine — Round-5 Wave 2.G.

Round-5 Wave 2 builds out per-jurisdiction halal screening (CMA, SCA,
SC Malaysia, ISSI, etc.). Each registry produces an authoritative
verdict for issues listed on its home market — but the operator
trading from a *different* jurisdiction may need to combine multiple
verdicts conjunctively (a US operator trading Saudi names should
satisfy both AAOIFI/global screen *and* CMA approval).

This module ships the **routing engine**: given the operator's
jurisdiction + the candidate symbol's listing market, it selects
which sources apply + how to combine them.

Pinned semantics:

- **Closed-set Jurisdiction ladder.** Adding a jurisdiction is a code
  review change.
- **Closed-set RoutingMode ladder.** STRICTEST (every applicable
  source must pass) / ANY (any single applicable source passes) /
  HOME_ONLY (only the listing-market source applies).
- **Default mode is STRICTEST** — pinned in tests. Most operators
  want the conjunctive interpretation; ANY only fits a permissive
  retail operator who explicitly opts in.
- **Mismatch rules.** A symbol whose listing market has no source in
  the registry returns `INSUFFICIENT_DATA`, never `IMPERMISSIBLE`.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Jurisdiction(str, Enum):
    """Closed-set operator + listing-market jurisdictions."""

    SAUDI_ARABIA = "saudi_arabia"
    UAE = "uae"
    MALAYSIA = "malaysia"
    INDONESIA = "indonesia"
    PAKISTAN = "pakistan"
    BAHRAIN = "bahrain"
    UK = "uk"
    USA = "usa"
    EU = "eu"
    GLOBAL = "global"  # AAOIFI / Zoya / IdealRatings — applies anywhere


class RoutingMode(str, Enum):
    """Closed-set composition modes for combining verdicts."""

    STRICTEST = "strictest"
    ANY = "any"
    HOME_ONLY = "home_only"


class JurisdictionVerdict(str, Enum):
    """Per-source verdict ladder."""

    PERMISSIBLE = "permissible"
    IMPERMISSIBLE = "impermissible"
    UNKNOWN = "unknown"


class CompositeOutcome(str, Enum):
    """Final routed outcome."""

    APPROVED = "approved"
    BLOCKED = "blocked"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class SourceVerdict:
    """A single source's verdict for a candidate symbol."""

    source_name: str
    jurisdiction: Jurisdiction
    verdict: JurisdictionVerdict
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source_name or not self.source_name.strip():
            raise ValueError("source_name must be non-empty")


@dataclass(frozen=True)
class RoutingPolicy:
    """Operator-tunable routing policy."""

    operator_jurisdiction: Jurisdiction
    mode: RoutingMode = RoutingMode.STRICTEST
    require_global_consensus: bool = True

    def __post_init__(self) -> None:
        # Operators with mode=ANY who also require global is contradictory; allow
        # but document. No validation needed beyond enum ranges.
        pass


@dataclass(frozen=True)
class RoutingResult:
    """Result of routing a single symbol through the engine."""

    symbol: str
    listing_market: Jurisdiction
    applicable_sources: tuple[SourceVerdict, ...]
    outcome: CompositeOutcome
    blocking_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")


# Map of (operator, listing_market) → which jurisdictions' sources apply.
# A frozen module-level mapping rather than a function so it's auditable.
_APPLICABILITY: dict[tuple[Jurisdiction, Jurisdiction], frozenset[Jurisdiction]] = {
    # Local listings — listing-market source + global only
    (Jurisdiction.SAUDI_ARABIA, Jurisdiction.SAUDI_ARABIA): frozenset(
        {Jurisdiction.SAUDI_ARABIA, Jurisdiction.GLOBAL}
    ),
    (Jurisdiction.UAE, Jurisdiction.UAE): frozenset({Jurisdiction.UAE, Jurisdiction.GLOBAL}),
    (Jurisdiction.MALAYSIA, Jurisdiction.MALAYSIA): frozenset(
        {Jurisdiction.MALAYSIA, Jurisdiction.GLOBAL}
    ),
    (Jurisdiction.INDONESIA, Jurisdiction.INDONESIA): frozenset(
        {Jurisdiction.INDONESIA, Jurisdiction.GLOBAL}
    ),
    (Jurisdiction.PAKISTAN, Jurisdiction.PAKISTAN): frozenset(
        {Jurisdiction.PAKISTAN, Jurisdiction.GLOBAL}
    ),
}


def applicable_sources(
    operator: Jurisdiction, listing_market: Jurisdiction
) -> frozenset[Jurisdiction]:
    """Return the set of source-jurisdictions whose verdicts apply.

    Default rule: listing-market + global. Operator-jurisdiction adds a
    source only when explicitly listed in `_APPLICABILITY`.
    """
    explicit = _APPLICABILITY.get((operator, listing_market))
    if explicit is not None:
        return explicit
    # Default cross-border: home of the issue + global. Operator's own
    # jurisdiction does NOT add a source (a US operator trading a Saudi
    # name doesn't need US Shariah advisory — there isn't one).
    return frozenset({listing_market, Jurisdiction.GLOBAL})


def route_symbol(
    symbol: str,
    listing_market: Jurisdiction,
    verdicts: Iterable[SourceVerdict],
    *,
    policy: RoutingPolicy,
) -> RoutingResult:
    """Run a single symbol through the routing engine."""
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be non-empty")

    if policy.mode is RoutingMode.HOME_ONLY:
        applicable = frozenset({listing_market})
    else:
        applicable = applicable_sources(policy.operator_jurisdiction, listing_market)

    relevant = tuple(v for v in verdicts if v.jurisdiction in applicable)
    relevant_known = tuple(v for v in relevant if v.verdict is not JurisdictionVerdict.UNKNOWN)

    if not relevant_known:
        return RoutingResult(
            symbol=symbol,
            listing_market=listing_market,
            applicable_sources=relevant,
            outcome=CompositeOutcome.INSUFFICIENT_DATA,
            blocking_sources=(),
        )

    blockers = tuple(
        v.source_name for v in relevant_known if v.verdict is JurisdictionVerdict.IMPERMISSIBLE
    )

    if policy.mode is RoutingMode.STRICTEST or policy.mode is RoutingMode.HOME_ONLY:
        if blockers:
            outcome = CompositeOutcome.BLOCKED
        elif (
            policy.mode is RoutingMode.STRICTEST
            and policy.require_global_consensus
            and not any(v.jurisdiction is Jurisdiction.GLOBAL for v in relevant_known)
        ):
            # HOME_ONLY by definition skips global; only enforce consensus in STRICTEST.
            outcome = CompositeOutcome.INSUFFICIENT_DATA
        else:
            outcome = CompositeOutcome.APPROVED
    else:  # ANY
        any_pass = any(v.verdict is JurisdictionVerdict.PERMISSIBLE for v in relevant_known)
        outcome = CompositeOutcome.APPROVED if any_pass else CompositeOutcome.BLOCKED

    return RoutingResult(
        symbol=symbol,
        listing_market=listing_market,
        applicable_sources=relevant,
        outcome=outcome,
        blocking_sources=blockers,
    )


def route_batch(
    candidates: Iterable[tuple[str, Jurisdiction, tuple[SourceVerdict, ...]]],
    *,
    policy: RoutingPolicy,
) -> tuple[RoutingResult, ...]:
    return tuple(
        route_symbol(sym, market, verdicts, policy=policy) for sym, market, verdicts in candidates
    )


def filter_approved(results: Iterable[RoutingResult]) -> tuple[RoutingResult, ...]:
    return tuple(r for r in results if r.outcome is CompositeOutcome.APPROVED)


def filter_blocked(results: Iterable[RoutingResult]) -> tuple[RoutingResult, ...]:
    return tuple(r for r in results if r.outcome is CompositeOutcome.BLOCKED)


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_result(result: RoutingResult) -> str:
    emoji = {
        CompositeOutcome.APPROVED: "✅",
        CompositeOutcome.BLOCKED: "❌",
        CompositeOutcome.INSUFFICIENT_DATA: "❔",
    }[result.outcome]
    head = f"{emoji} {result.symbol} [{result.listing_market.value}] → {result.outcome.value}"
    lines = [head]
    for v in result.applicable_sources:
        verdict_emoji = {
            JurisdictionVerdict.PERMISSIBLE: "✓",
            JurisdictionVerdict.IMPERMISSIBLE: "✗",
            JurisdictionVerdict.UNKNOWN: "?",
        }[v.verdict]
        suffix = f" ({v.note})" if v.note else ""
        lines.append(f"  {verdict_emoji} {v.source_name} [{v.jurisdiction.value}]{suffix}")
    return _scrub("\n".join(lines))
