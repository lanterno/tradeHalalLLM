"""Dark-pool detection + halal-aware avoidance — Round-5 Wave 12.F.

Dark pools (off-exchange ATSes / SDPs) execute large blocks without
pre-trade transparency. They're often desirable for size-sensitive
trades, but for a halal-sensitive portfolio they raise two concerns:

1. **Counterparty visibility** is poor — you may unknowingly cross
   with conventional-leverage counterparties (interest-bearing margin)
   even when your own side is halal.
2. **Some pools host non-halal flow disproportionately** (alcohol /
   tobacco / interest-bearing fund flow).

This module is the **detection layer + opt-out gate**. It does not
itself execute orders; it classifies prints + returns a routing
verdict the caller respects.

Pinned semantics:

- **Closed-set PrintType ladder** — LIT / DARK_TRF / DARK_INTERNAL /
  AUCTION. TRF prints are post-trade-reported off-exchange; INTERNAL
  prints come from a broker's own retail-internalisation engine.
- **Closed-set RouteVerdict ladder** — ALLOW / WARN / OPT_OUT.
- **Detection from print metadata** — venue code + condition flags.
  The platform's broker adapter must populate `is_dark_indicator`,
  `is_internal_indicator`, `is_auction_indicator`. Without flags,
  the classifier defaults to LIT.
- **Aggregate dark-fraction** is the share of volume that printed
  dark over a window. Pinned: > 30% triggers WARN, > 50% triggers
  OPT_OUT.
- **Halal-sensitive flag** is operator-set per ticker. If true, the
  threshold drops (10% / 25%) — the operator wants stricter avoidance.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class PrintType(str, Enum):
    """Closed-set print type ladder."""

    LIT = "lit"
    DARK_TRF = "dark_trf"
    DARK_INTERNAL = "dark_internal"
    AUCTION = "auction"


_DARK_TYPES: frozenset[PrintType] = frozenset({PrintType.DARK_TRF, PrintType.DARK_INTERNAL})


def is_dark(print_type: PrintType) -> bool:
    return print_type in _DARK_TYPES


class RouteVerdict(str, Enum):
    """Closed-set routing verdict ladder."""

    ALLOW = "allow"
    WARN = "warn"
    OPT_OUT = "opt_out"


@dataclass(frozen=True)
class TradePrint:
    """One trade print observed on the tape."""

    print_id: str
    ticker: str
    timestamp: datetime
    price: float
    quantity: float
    venue_code: str
    is_dark_indicator: bool = False
    """True if the venue/condition flag indicates an off-exchange print."""
    is_internal_indicator: bool = False
    """True if the print is broker-internalised retail flow."""
    is_auction_indicator: bool = False

    def __post_init__(self) -> None:
        if not self.print_id or not self.print_id.strip():
            raise ValueError("print_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if not self.venue_code or not self.venue_code.strip():
            raise ValueError("venue_code must be non-empty")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


def classify_print(p: TradePrint) -> PrintType:
    """Classify a print by indicator priority.

    Pinned priority (highest specificity wins):
    1. is_internal_indicator → DARK_INTERNAL
    2. is_auction_indicator → AUCTION
    3. is_dark_indicator → DARK_TRF
    4. else → LIT
    """
    if p.is_internal_indicator:
        return PrintType.DARK_INTERNAL
    if p.is_auction_indicator:
        return PrintType.AUCTION
    if p.is_dark_indicator:
        return PrintType.DARK_TRF
    return PrintType.LIT


@dataclass(frozen=True)
class TickerDarkProfile:
    """Output of `summarise_ticker`."""

    ticker: str
    window_seconds: int
    n_prints: int
    total_volume: float
    dark_volume: float
    dark_fraction: float
    """In [0, 1]; share of `total_volume` that printed dark."""
    internal_fraction: float
    """In [0, 1]; share of `total_volume` from DARK_INTERNAL specifically."""


def summarise_ticker(
    prints: Iterable[TradePrint],
    *,
    ticker: str,
    window_seconds: int,
    as_of: datetime,
) -> TickerDarkProfile:
    """Compute dark-fraction over a `window_seconds` lookback ending at `as_of`."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    cutoff = as_of - timedelta(seconds=window_seconds)
    relevant = [p for p in prints if p.ticker == ticker and cutoff <= p.timestamp <= as_of]
    n = len(relevant)
    total = sum(p.quantity for p in relevant)
    dark = sum(p.quantity for p in relevant if is_dark(classify_print(p)))
    internal = sum(p.quantity for p in relevant if classify_print(p) is PrintType.DARK_INTERNAL)
    if total < 1e-12:
        dark_frac = 0.0
        internal_frac = 0.0
    else:
        dark_frac = dark / total
        internal_frac = internal / total
    return TickerDarkProfile(
        ticker=ticker,
        window_seconds=window_seconds,
        n_prints=n,
        total_volume=total,
        dark_volume=dark,
        dark_fraction=dark_frac,
        internal_fraction=internal_frac,
    )


@dataclass(frozen=True)
class RouteDecision:
    """Output of `decide_route`."""

    ticker: str
    verdict: RouteVerdict
    dark_fraction: float
    halal_sensitive: bool
    reason: str


# Default thresholds by halal_sensitive flag.
_THRESHOLDS_NORMAL = (0.30, 0.50)  # WARN above 30%, OPT_OUT above 50%
_THRESHOLDS_SENSITIVE = (0.10, 0.25)  # tighter for halal-sensitive trades


def decide_route(
    profile: TickerDarkProfile,
    *,
    halal_sensitive: bool = False,
    overrides: tuple[float, float] | None = None,
) -> RouteDecision:
    """Decide ALLOW / WARN / OPT_OUT for the next order on this ticker.

    Pinned: halal-sensitive trades use tighter (10% / 25%) thresholds;
    standard thresholds are 30% / 50%. Operators can override the
    pair via `overrides=(warn, opt_out)`.
    """
    if overrides is not None:
        warn_th, opt_th = overrides
    elif halal_sensitive:
        warn_th, opt_th = _THRESHOLDS_SENSITIVE
    else:
        warn_th, opt_th = _THRESHOLDS_NORMAL
    if not 0.0 <= warn_th <= opt_th <= 1.0:
        raise ValueError("thresholds must satisfy 0 ≤ warn ≤ opt_out ≤ 1")
    df = profile.dark_fraction
    if df > opt_th:
        verdict = RouteVerdict.OPT_OUT
        reason = f"dark_fraction {df * 100:.2f}% > opt_out threshold {opt_th * 100:.0f}%"
    elif df > warn_th:
        verdict = RouteVerdict.WARN
        reason = (
            f"dark_fraction {df * 100:.2f}% > warn threshold "
            f"{warn_th * 100:.0f}% (allowed; flag operator)"
        )
    else:
        verdict = RouteVerdict.ALLOW
        reason = (
            f"dark_fraction {df * 100:.2f}% within tolerance "
            f"({warn_th * 100:.0f}%/{opt_th * 100:.0f}%)"
        )
    return RouteDecision(
        ticker=profile.ticker,
        verdict=verdict,
        dark_fraction=df,
        halal_sensitive=halal_sensitive,
        reason=reason,
    )


def filter_lit_only(
    prints: Iterable[TradePrint],
) -> tuple[TradePrint, ...]:
    """Convenience: drop dark prints, keep LIT + AUCTION."""
    return tuple(p for p in prints if not is_dark(classify_print(p)))


_VERDICT_EMOJI: dict[RouteVerdict, str] = {
    RouteVerdict.ALLOW: "✅",
    RouteVerdict.WARN: "⚠️",
    RouteVerdict.OPT_OUT: "🛑",
}


def render_decision(decision: RouteDecision) -> str:
    sensitive = " [halal-sensitive]" if decision.halal_sensitive else ""
    return (
        f"{_VERDICT_EMOJI[decision.verdict]} {decision.ticker}: "
        f"{decision.verdict.value}{sensitive}\n  • {decision.reason}"
    )


def render_profile(profile: TickerDarkProfile) -> str:
    return (
        f"📊 {profile.ticker}: {profile.n_prints} prints / "
        f"{profile.total_volume:,.0f} qty in {profile.window_seconds}s; "
        f"dark={profile.dark_fraction * 100:.2f}% "
        f"(internal={profile.internal_fraction * 100:.2f}%)"
    )
