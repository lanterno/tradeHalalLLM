"""Halal options-strategy screener.

The roadmap pins options strategies as scholar-debated: classical
fiqh treats most options trading as gharar (excessive uncertainty)
because the option contract is paying for a *right*, not an
asset, and the seller is committing to deliver something they
may not own. Modern AAOIFI guidance and a growing minority of
scholars carve out a narrow permissible set:

- **Covered calls** (writing call against owned shares) — the
  writer DOES own the underlying, so selling the right to
  deliver it isn't gharar; some scholars permit, citing the
  classical sale of `urbun` (earnest money). The premium is
  treated as a fee for waiting, not interest.
- **Cash-secured puts** (writing put backed by cash equal to
  strike × 100) — the writer can deliver the cash to buy the
  shares; treated as a contractual undertaking to buy, similar
  to murabaha pre-commitment. More-permissive operators allow.
- **Protective puts** (buying put as insurance against owned
  shares) — buying insurance is not gharar (the buyer pays a
  defined premium for a defined coverage); most scholars
  permit.
- **Naked calls / puts** — selling the right to deliver
  something you don't own / committing to buy without the cash
  → unconditional gharar; classical and modern consensus
  reject.
- **Spreads / iron condors / butterflies** — multiple-leg
  strategies with at least one short naked leg → reject by
  the same logic.

The full integration (live options chain ingest, strategy
construction, exit management) is deferred — the bot doesn't
trade options today and won't until the SSB issues a clear
ruling. The screener ships **the rule** so when SSB approves a
specific strategy, the dashboard can route requests through
this gate first.

Pinned semantics:
- **Closed strategy enum.** `OptionStrategy` lists every
  strategy the screener recognises; non-listed combinations
  fall through to OTHER and route to UNKNOWN. The closed-set
  pin matters because options have many exotic combinations
  (calendar spreads, ratio spreads, jade lizards) and the
  default for any unfamiliar combination must be safe-block.
- **NAKED_CALL / NAKED_PUT unconditionally NOT_HALAL.** No
  scholar ruling reverses these; classical + modern consensus.
- **Multi-leg spreads with any short leg unconditionally
  NOT_HALAL.** A bull-call spread (buy lower-strike call +
  sell higher-strike call) embeds a naked-call short leg.
  Iron condor / butterfly likewise.
- **COVERED_CALL / CASH_SECURED_PUT DOUBTFUL pending SSB
  ruling.** Engine returns DOUBTFUL with a "scholar profile
  must approve" warning. Operators with an SSB ruling cite
  the ruling_id in the audit metadata when escalating to HALAL
  via the Wave 11.B governance engine.
- **PROTECTIVE_PUT HALAL.** Buying insurance is the cleanest
  case; the buyer pays a defined premium for a defined
  coverage, no gharar.
- **Render output never includes contract symbols, strike
  prices, or expiration dates.** Mirrors no-PII patterns of
  Wave 11.D + 11.C + 3.B + 12.E + 12.G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OptionStrategy(str, Enum):
    """Recognised option-strategy categories.

    Pinned string values for JSON / DB stability. The OTHER
    catchall surfaces unknown combinations as UNKNOWN verdict.
    """

    # Single-leg
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    NAKED_CALL = "naked_call"
    NAKED_PUT = "naked_put"

    # Covered (against underlying)
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    PROTECTIVE_PUT = "protective_put"

    # Multi-leg
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

    OTHER = "other"


# Strategies that include any short leg without underlying / cash
# backing — gharar via classical and modern consensus.
_GHARAR_NAKED_STRATEGIES: frozenset[OptionStrategy] = frozenset(
    {
        OptionStrategy.NAKED_CALL,
        OptionStrategy.NAKED_PUT,
    }
)

# Multi-leg spreads with embedded short legs — fail for the same
# reason as the standalone naked legs (the spread's long leg
# doesn't redeem the short leg).
_GHARAR_SPREAD_STRATEGIES: frozenset[OptionStrategy] = frozenset(
    {
        OptionStrategy.BULL_CALL_SPREAD,
        OptionStrategy.BEAR_PUT_SPREAD,
        OptionStrategy.IRON_CONDOR,
        OptionStrategy.BUTTERFLY,
        OptionStrategy.STRADDLE,
        OptionStrategy.STRANGLE,
    }
)

# Strategies under scholar disagreement — DOUBTFUL pending the
# operator's SSB ruling (Wave 11.B governance engine).
_DEBATED_STRATEGIES: frozenset[OptionStrategy] = frozenset(
    {
        OptionStrategy.COVERED_CALL,
        OptionStrategy.CASH_SECURED_PUT,
        OptionStrategy.LONG_CALL,
        OptionStrategy.LONG_PUT,
    }
)

# Strategies most scholars permit.
_PERMITTED_STRATEGIES: frozenset[OptionStrategy] = frozenset(
    {
        OptionStrategy.PROTECTIVE_PUT,
    }
)


class OptionsScreenVerdict(str, Enum):
    """Screen verdict.

    Pinned string values for JSON / DB stability.
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    DOUBTFUL = "doubtful"
    HALAL_WITH_CONDITIONS = "halal_with_conditions"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OptionsScreenPolicy:
    """Operator-tunable policy.

    `ssb_ruling_id` lets operators cite their Wave 11.B Shariah
    Supervisory Board's ruling that approves COVERED_CALL +
    CASH_SECURED_PUT under specific conditions. When set, the
    screener returns HALAL_WITH_CONDITIONS for those strategies
    instead of DOUBTFUL — the cited ruling becomes the audit-trail
    justification.

    `permitted_with_ssb_ruling` is the explicit allow-list of
    debated strategies the operator's SSB has approved (subset of
    `_DEBATED_STRATEGIES`). Empty by default — operator must
    actively opt-in.
    """

    ssb_ruling_id: str = ""
    permitted_with_ssb_ruling: frozenset[OptionStrategy] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # If the operator names specific strategies, they must have
        # cited a ruling_id; otherwise the conditional approval is
        # not auditable.
        if self.permitted_with_ssb_ruling and not self.ssb_ruling_id:
            raise ValueError(
                "permitted_with_ssb_ruling requires ssb_ruling_id "
                "(strategies need a citable ruling for audit)"
            )
        # Operator can only opt-in to debated strategies; can't
        # opt-in to forbidden naked / spread strategies.
        forbidden_attempts = self.permitted_with_ssb_ruling & (
            _GHARAR_NAKED_STRATEGIES | _GHARAR_SPREAD_STRATEGIES
        )
        if forbidden_attempts:
            attempted = sorted(s.value for s in forbidden_attempts)
            raise ValueError(f"cannot SSB-approve gharar strategies: {attempted}")


DEFAULT_POLICY = OptionsScreenPolicy()


@dataclass(frozen=True)
class OptionsScreenRequest:
    """A proposed option strategy to screen."""

    strategy: OptionStrategy
    underlying_symbol: str
    underlying_is_halal_screened: bool

    def __post_init__(self) -> None:
        if not self.underlying_symbol or not self.underlying_symbol.strip():
            raise ValueError("underlying_symbol must be non-empty")


@dataclass(frozen=True)
class OptionsScreenResult:
    """Screen verdict + supporting flags + audit notes."""

    strategy: OptionStrategy
    underlying_symbol: str
    verdict: OptionsScreenVerdict
    ssb_ruling_cited: str = ""
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def screen_options_strategy(
    request: OptionsScreenRequest,
    *,
    policy: OptionsScreenPolicy = DEFAULT_POLICY,
) -> OptionsScreenResult:
    """Apply the halal options-strategy screen.

    Returns an `OptionsScreenResult` with verdict + per-rule
    failure / warning lists for the audit trail.
    """

    failures: list[str] = []
    warnings: list[str] = []

    # First gate: the underlying must be halal-screened. Even a
    # PROTECTIVE_PUT on a non-halal-screened underlying is suspect
    # because the operator is still holding a non-halal position.
    if not request.underlying_is_halal_screened:
        failures.append(
            f"underlying {request.underlying_symbol} is not halal-screened: "
            "the option strategy inherits the underlying's haram status"
        )
        return OptionsScreenResult(
            strategy=request.strategy,
            underlying_symbol=request.underlying_symbol,
            verdict=OptionsScreenVerdict.NOT_HALAL,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Naked / spread strategies — categorical NOT_HALAL.
    if request.strategy in _GHARAR_NAKED_STRATEGIES:
        failures.append(
            f"{request.strategy.value} involves selling a right without "
            "underlying or cash backing (gharar)"
        )
        return OptionsScreenResult(
            strategy=request.strategy,
            underlying_symbol=request.underlying_symbol,
            verdict=OptionsScreenVerdict.NOT_HALAL,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    if request.strategy in _GHARAR_SPREAD_STRATEGIES:
        failures.append(
            f"{request.strategy.value} embeds a short leg without underlying / "
            "cash backing; the long leg does not redeem the gharar of the short"
        )
        return OptionsScreenResult(
            strategy=request.strategy,
            underlying_symbol=request.underlying_symbol,
            verdict=OptionsScreenVerdict.NOT_HALAL,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Permitted strategies — HALAL.
    if request.strategy in _PERMITTED_STRATEGIES:
        return OptionsScreenResult(
            strategy=request.strategy,
            underlying_symbol=request.underlying_symbol,
            verdict=OptionsScreenVerdict.HALAL,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Debated strategies — check operator's SSB approval.
    if request.strategy in _DEBATED_STRATEGIES:
        if request.strategy in policy.permitted_with_ssb_ruling:
            return OptionsScreenResult(
                strategy=request.strategy,
                underlying_symbol=request.underlying_symbol,
                verdict=OptionsScreenVerdict.HALAL_WITH_CONDITIONS,
                ssb_ruling_cited=policy.ssb_ruling_id,
                failures=tuple(failures),
                warnings=(
                    f"{request.strategy.value} approved under SSB ruling "
                    f"{policy.ssb_ruling_id!r}; verify the ruling's specific "
                    "conditions are honoured (premium-only, no rolling-over, etc.)",
                ),
            )
        warnings.append(
            f"{request.strategy.value} is scholar-debated; default verdict "
            "is DOUBTFUL until operator's SSB issues a permitting ruling"
        )
        return OptionsScreenResult(
            strategy=request.strategy,
            underlying_symbol=request.underlying_symbol,
            verdict=OptionsScreenVerdict.DOUBTFUL,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # OTHER catchall.
    return OptionsScreenResult(
        strategy=request.strategy,
        underlying_symbol=request.underlying_symbol,
        verdict=OptionsScreenVerdict.UNKNOWN,
        warnings=(
            f"strategy {request.strategy.value} not in recognised categories; "
            "operator must classify before allocating",
        ),
    )


_VERDICT_EMOJI: dict[OptionsScreenVerdict, str] = {
    OptionsScreenVerdict.HALAL: "✅",
    OptionsScreenVerdict.NOT_HALAL: "❌",
    OptionsScreenVerdict.DOUBTFUL: "⚠️",
    OptionsScreenVerdict.HALAL_WITH_CONDITIONS: "📋",
    OptionsScreenVerdict.UNKNOWN: "❓",
}


def render_screen_result(result: OptionsScreenResult) -> str:
    """Format the screen result for ops display.

    Pinned no-strike / no-expiry contract: the result never
    includes contract symbols (full OPRA-style), strike prices,
    or expiration dates. Operators audit those via their broker
    side; the engine surfaces only the strategy classification.
    """

    emoji = _VERDICT_EMOJI[result.verdict]
    lines = [
        f"{emoji} {result.underlying_symbol} {result.strategy.value} "
        f"— {result.verdict.value.upper()}",
    ]
    if result.ssb_ruling_cited:
        lines.append(f"  ssb_ruling: {result.ssb_ruling_cited}")
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
    "OptionStrategy",
    "OptionsScreenPolicy",
    "OptionsScreenRequest",
    "OptionsScreenResult",
    "OptionsScreenVerdict",
    "render_screen_result",
    "screen_options_strategy",
]
