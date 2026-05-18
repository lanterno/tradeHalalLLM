"""Portfolio risk aggregator + pre-trade risk gate.

Auxiliary primitive complementing Wave 4.G Bayesian portfolio
optimisation + the crypto monitor's per-position SL/TP enforcement.
Wave 4.G ships HRP weights; the monitor enforces SL/TP per position;
this module is the **pure-Python aggregator** that answers a
different question: "given our open positions + their stop-losses,
what's our total at-risk capital, and can we afford to add this new
buy without breaching account-level risk caps?"

Picked a focused aggregator over scattering risk math across the
cycle / executor / monitor because (a) the pre-trade gate is the
load-bearing safety check before placing a buy — a single function
`assert_within_risk_budget(positions, *, new_position, account_value,
policy)` that the executor calls before submission means a single
inspectable check rather than scattered conditionals; (b) the
account-level risk caps (max total at-risk %, max single-position
%, max position count) are the guards against concentration
failure modes — encoding them as policy means a misconfigured
"let me put 50% in one symbol" gets caught at the gate; (c) the
aggregation is pure: deterministic for given (positions,
account_value), so the dashboard's "account risk" tile is
inspectable + replay-able for the audit trail.

Pinned semantics:
- **Per-position risk = notional × distance-to-stop-loss.** The
  amount of capital lost if SL hits. Pure function; pinned via
  test against known-result calculation.
- **Account-level caps:**
  - max_total_at_risk_pct (default 6%): total at-risk USD / account
    value <= 6% (the conventional "no more than 6% of capital at
    risk across all positions" rule).
  - max_single_position_risk_pct (default 2%): no single position
    risks more than 2% of account.
  - max_position_count (default 20): no more than 20 simultaneous
    positions (concentration vs spread tradeoff).
- **Pre-trade gate is forward-looking.** Simulates "after this
  new buy, are we still under all 3 caps?" rather than checking
  current state.
- **Render output never includes broker account numbers, position
  IDs, or operator-side fields.** Mirrors no-secret patterns.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class RiskGateOutcome(str, Enum):
    """Outcome of pre-trade risk gate.

    Pinned string values for JSON / DB stability.
    """

    APPROVED = "approved"
    REJECTED_TOTAL_RISK = "rejected_total_risk"
    REJECTED_SINGLE_POSITION = "rejected_single_position"
    REJECTED_POSITION_COUNT = "rejected_position_count"


_DEFAULT_MAX_TOTAL_AT_RISK_PCT = 0.06  # 6%
_DEFAULT_MAX_SINGLE_POSITION_RISK_PCT = 0.02  # 2%
_DEFAULT_MAX_POSITION_COUNT = 20


@dataclass(frozen=True)
class RiskPolicy:
    """Operator-tunable risk caps."""

    max_total_at_risk_pct: float = _DEFAULT_MAX_TOTAL_AT_RISK_PCT
    max_single_position_risk_pct: float = _DEFAULT_MAX_SINGLE_POSITION_RISK_PCT
    max_position_count: int = _DEFAULT_MAX_POSITION_COUNT

    def __post_init__(self) -> None:
        if not 0.0 < self.max_total_at_risk_pct <= 1.0:
            raise ValueError(
                f"max_total_at_risk_pct {self.max_total_at_risk_pct} must be in (0, 1]"
            )
        if not 0.0 < self.max_single_position_risk_pct <= 1.0:
            raise ValueError("max_single_position_risk_pct must be in (0, 1]")
        if self.max_single_position_risk_pct > self.max_total_at_risk_pct:
            raise ValueError(
                "max_single_position_risk_pct cannot exceed "
                "max_total_at_risk_pct (single position can't risk "
                "more than the total cap)"
            )
        if self.max_position_count <= 0:
            raise ValueError("max_position_count must be positive")


DEFAULT_POLICY = RiskPolicy()


@dataclass(frozen=True)
class Position:
    """One open position.

    `notional_usd` is the position's USD size (qty × entry price).
    `entry_price` and `stop_loss_price` are in symbol-quote units
    (USDT for crypto, USD for stocks). The at-risk USD is computed
    as `notional_usd × |entry - SL| / entry`.
    """

    symbol: str
    notional_usd: float
    entry_price: float
    stop_loss_price: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.notional_usd <= 0:
            raise ValueError("notional_usd must be positive")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.stop_loss_price <= 0:
            raise ValueError("stop_loss_price must be positive")
        if self.stop_loss_price >= self.entry_price:
            # We model long positions only at this layer — SL must
            # be below entry. Short positions would invert the
            # relationship; they're a deferred follow-up.
            raise ValueError(
                f"stop_loss_price {self.stop_loss_price} must be < "
                f"entry_price {self.entry_price} (long position)"
            )


def at_risk_usd(position: Position) -> float:
    """Compute the USD at risk if SL hits.

    For a long position at entry $100, SL $95, $1000 notional:
    at_risk = $1000 × (100 - 95) / 100 = $50.
    """

    return (
        position.notional_usd
        * (position.entry_price - position.stop_loss_price)
        / position.entry_price
    )


@dataclass(frozen=True)
class RiskSnapshot:
    """Aggregated account-level risk view at a point in time."""

    position_count: int
    total_notional_usd: float
    total_at_risk_usd: float
    account_value_usd: float
    total_at_risk_pct: float
    max_single_position_risk_pct: float
    largest_position_symbol: str  # "" if no positions
    largest_position_risk_usd: float

    def __post_init__(self) -> None:
        if self.position_count < 0:
            raise ValueError("position_count must be non-negative")
        if self.total_notional_usd < 0:
            raise ValueError("total_notional_usd must be non-negative")
        if self.total_at_risk_usd < 0:
            raise ValueError("total_at_risk_usd must be non-negative")
        if self.account_value_usd <= 0:
            raise ValueError("account_value_usd must be positive")


def aggregate_risk(
    positions: Iterable[Position],
    *,
    account_value_usd: float,
) -> RiskSnapshot:
    """Compute the account-level risk snapshot.

    Pure: deterministic for given (positions, account_value).
    """

    if account_value_usd <= 0:
        raise ValueError("account_value_usd must be positive")

    pos_list = list(positions)
    total_notional = sum(p.notional_usd for p in pos_list)
    total_at_risk = sum(at_risk_usd(p) for p in pos_list)
    total_at_risk_pct = total_at_risk / account_value_usd

    largest_symbol = ""
    largest_risk_usd = 0.0
    max_single_pct = 0.0
    for pos in pos_list:
        risk = at_risk_usd(pos)
        if risk > largest_risk_usd:
            largest_risk_usd = risk
            largest_symbol = pos.symbol
        position_pct = risk / account_value_usd
        if position_pct > max_single_pct:
            max_single_pct = position_pct

    return RiskSnapshot(
        position_count=len(pos_list),
        total_notional_usd=total_notional,
        total_at_risk_usd=total_at_risk,
        account_value_usd=account_value_usd,
        total_at_risk_pct=total_at_risk_pct,
        max_single_position_risk_pct=max_single_pct,
        largest_position_symbol=largest_symbol,
        largest_position_risk_usd=largest_risk_usd,
    )


@dataclass(frozen=True)
class RiskGateDecision:
    """Outcome of pre-trade risk gate."""

    outcome: RiskGateOutcome
    message: str
    projected_total_at_risk_pct: float

    def __post_init__(self) -> None:
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")


def evaluate_pre_trade_gate(
    *,
    open_positions: Iterable[Position],
    new_position: Position,
    account_value_usd: float,
    policy: RiskPolicy = DEFAULT_POLICY,
) -> RiskGateDecision:
    """Decide whether to approve a new position given account-level caps.

    Simulates the post-trade state: total positions = current + 1,
    total at-risk = current + new_position's risk. Approves only if
    all 3 caps hold post-trade.
    """

    if account_value_usd <= 0:
        raise ValueError("account_value_usd must be positive")

    open_list = list(open_positions)
    current_position_count = len(open_list)

    # Simulate post-trade state
    projected_count = current_position_count + 1
    if projected_count > policy.max_position_count:
        return RiskGateDecision(
            outcome=RiskGateOutcome.REJECTED_POSITION_COUNT,
            message=(
                f"would breach max_position_count: {projected_count} > {policy.max_position_count}"
            ),
            projected_total_at_risk_pct=0.0,
        )

    new_risk = at_risk_usd(new_position)
    new_position_pct = new_risk / account_value_usd
    if new_position_pct > policy.max_single_position_risk_pct:
        return RiskGateDecision(
            outcome=RiskGateOutcome.REJECTED_SINGLE_POSITION,
            message=(
                f"new position risks {new_position_pct:.2%} of account, "
                f"above cap {policy.max_single_position_risk_pct:.2%}"
            ),
            projected_total_at_risk_pct=0.0,
        )

    current_total_risk = sum(at_risk_usd(p) for p in open_list)
    projected_total_risk = current_total_risk + new_risk
    projected_total_pct = projected_total_risk / account_value_usd
    if projected_total_pct > policy.max_total_at_risk_pct:
        return RiskGateDecision(
            outcome=RiskGateOutcome.REJECTED_TOTAL_RISK,
            message=(
                f"projected total at-risk {projected_total_pct:.2%} of "
                f"account, above cap {policy.max_total_at_risk_pct:.2%}"
            ),
            projected_total_at_risk_pct=projected_total_pct,
        )

    return RiskGateDecision(
        outcome=RiskGateOutcome.APPROVED,
        message=(
            f"approved: projected total at-risk "
            f"{projected_total_pct:.2%} (cap "
            f"{policy.max_total_at_risk_pct:.2%})"
        ),
        projected_total_at_risk_pct=projected_total_pct,
    )


_OUTCOME_EMOJI: dict[RiskGateOutcome, str] = {
    RiskGateOutcome.APPROVED: "✅",
    RiskGateOutcome.REJECTED_TOTAL_RISK: "🚫",
    RiskGateOutcome.REJECTED_SINGLE_POSITION: "⚠️",
    RiskGateOutcome.REJECTED_POSITION_COUNT: "📊",
}


def render_snapshot(snapshot: RiskSnapshot) -> str:
    """Format the account-level risk snapshot for ops display.

    No-secret-leak: never includes broker account numbers / position
    IDs / operator-side fields. Shows symbols + USD amounts +
    percentages.
    """

    largest_str = (
        f"  largest: {snapshot.largest_position_symbol} "
        f"(${snapshot.largest_position_risk_usd:.2f}, "
        f"{snapshot.max_single_position_risk_pct:.2%})"
        if snapshot.largest_position_symbol
        else "  largest: —"
    )
    return (
        f"📊 Account risk snapshot\n"
        f"  positions: {snapshot.position_count}\n"
        f"  total notional: ${snapshot.total_notional_usd:.2f}\n"
        f"  total at-risk: ${snapshot.total_at_risk_usd:.2f} "
        f"({snapshot.total_at_risk_pct:.2%} of account)\n"
        f"{largest_str}"
    )


def render_decision(decision: RiskGateDecision) -> str:
    """Format a pre-trade gate decision for ops display."""

    emoji = _OUTCOME_EMOJI[decision.outcome]
    return f"{emoji} {decision.outcome.value}: {decision.message}"


__all__ = [
    "DEFAULT_POLICY",
    "Position",
    "RiskGateDecision",
    "RiskGateOutcome",
    "RiskPolicy",
    "RiskSnapshot",
    "aggregate_risk",
    "at_risk_usd",
    "evaluate_pre_trade_gate",
    "render_decision",
    "render_snapshot",
]
