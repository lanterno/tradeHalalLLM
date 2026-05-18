"""Wa'd (unilateral promise) structuring engine — Round-5 Wave 4.A.

A Wa'd is a unilateral promise to enter into a future transaction at a
specified price + time. AAOIFI Standard 49 + 53 explicitly allow a
single Wa'd as the building block for halal call / put-equivalent
payoffs — the structure is permissible because (a) it is a *promise*,
not a contract, and (b) the corresponding obligation is one-sided.

What's *not* permissible:

- **Bilateral Wa'd** — both parties promise to each other; this
  collapses into a forward contract. Standard 49 cl. 7 explicitly
  bans it.
- **Wa'd with attached premium** — paying for the promise itself
  collapses it into an option contract (ribawi).
- **Salam-overlay Wa'd** — using a Wa'd to disguise a Salam contract
  with payment delayed.

This module ships the **structuring engine** that validates a
proposed Wa'd against the rules and returns the issues. The engine
also constructs the synthetic-call / synthetic-put payoffs operators
use as halal alternatives to conventional options.

Pinned semantics:

- **Closed-set Direction ladder** (PROMISE_TO_BUY / PROMISE_TO_SELL).
- **Closed-set WaadIssue ladder.**
- **Single-direction only.** The engine refuses to compose two Wa'ds
  *between the same two parties* — that's a bilateral Wa'd. Two Wa'ds
  with *different* counterparties are permitted (the bot can
  separately promise to buy from A and promise to sell to B).
- **No premium charged for the promise itself.**
- **Fair-value strike at promise-time.** A grossly off-market strike
  evidences underlying speculation rather than risk-management
  intent (Standard 49 cl. 11) — the engine flags it.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class WaadDirection(str, Enum):
    PROMISE_TO_BUY = "promise_to_buy"
    PROMISE_TO_SELL = "promise_to_sell"


class WaadIssue(str, Enum):
    """Closed-set issues a proposed Wa'd structure can carry."""

    BILATERAL_WAAD_BAN = "bilateral_waad_ban"
    PREMIUM_CHARGED = "premium_charged"
    STRIKE_OFF_MARKET = "strike_off_market"
    EXERCISE_DATE_NOT_FUTURE = "exercise_date_not_future"
    EXERCISE_DATE_TOO_FAR = "exercise_date_too_far"
    QUANTITY_NON_POSITIVE = "quantity_non_positive"
    STRIKE_NON_POSITIVE = "strike_non_positive"
    EMPTY_PROMISOR = "empty_promisor"
    EMPTY_PROMISEE = "empty_promisee"


@dataclass(frozen=True)
class StructuringPolicy:
    """Operator-tunable thresholds."""

    max_term_days: int = 365
    strike_off_market_pct: float = 0.50  # >50% off market = flagged

    def __post_init__(self) -> None:
        if self.max_term_days <= 0:
            raise ValueError("max_term_days must be positive")
        if not 0.0 < self.strike_off_market_pct <= 1.0:
            raise ValueError("strike_off_market_pct must be in (0, 1]")


@dataclass(frozen=True)
class WaadInputs:
    """Inputs for a proposed Wa'd."""

    waad_id: str
    direction: WaadDirection
    promisor: str
    promisee: str
    underlying: str
    quantity: float
    strike_price: float
    market_price: float
    promise_date: date
    exercise_date: date
    premium_paid: float = 0.0

    def __post_init__(self) -> None:
        if not self.waad_id or not self.waad_id.strip():
            raise ValueError("waad_id must be non-empty")
        if not self.underlying or not self.underlying.strip():
            raise ValueError("underlying must be non-empty")
        if self.market_price <= 0:
            raise ValueError("market_price must be positive")


@dataclass(frozen=True)
class StructuringResult:
    """Result of running a Wa'd through the structurer."""

    waad_id: str
    issues: frozenset[WaadIssue]
    is_valid: bool

    def __post_init__(self) -> None:
        if self.is_valid and self.issues:
            raise ValueError("is_valid=True but issues non-empty")
        if (not self.is_valid) and not self.issues:
            raise ValueError("is_valid=False but issues empty")


def structure_waad(
    inputs: WaadInputs, *, policy: StructuringPolicy | None = None
) -> StructuringResult:
    """Validate a single Wa'd against AAOIFI Standard 49 / 53."""
    pol = policy if policy is not None else StructuringPolicy()
    issues: set[WaadIssue] = set()

    if not inputs.promisor.strip():
        issues.add(WaadIssue.EMPTY_PROMISOR)
    if not inputs.promisee.strip():
        issues.add(WaadIssue.EMPTY_PROMISEE)
    if inputs.quantity <= 0:
        issues.add(WaadIssue.QUANTITY_NON_POSITIVE)
    if inputs.strike_price <= 0:
        issues.add(WaadIssue.STRIKE_NON_POSITIVE)
    if inputs.premium_paid > 0:
        issues.add(WaadIssue.PREMIUM_CHARGED)

    delta = inputs.exercise_date - inputs.promise_date
    if delta <= timedelta(0):
        issues.add(WaadIssue.EXERCISE_DATE_NOT_FUTURE)
    if delta > timedelta(days=pol.max_term_days):
        issues.add(WaadIssue.EXERCISE_DATE_TOO_FAR)

    if inputs.strike_price > 0 and inputs.market_price > 0:
        deviation = abs(inputs.strike_price - inputs.market_price) / inputs.market_price
        if deviation > pol.strike_off_market_pct:
            issues.add(WaadIssue.STRIKE_OFF_MARKET)

    return StructuringResult(
        waad_id=inputs.waad_id,
        issues=frozenset(issues),
        is_valid=len(issues) == 0,
    )


def detect_bilateral_pair(a: WaadInputs, b: WaadInputs) -> bool:
    """Detect a bilateral-Wa'd pair — same two parties, opposing directions, same underlying."""
    if a.underlying != b.underlying:
        return False
    if a.direction is b.direction:
        return False
    parties_a = {a.promisor, a.promisee}
    parties_b = {b.promisor, b.promisee}
    return parties_a == parties_b


def structure_pair(
    a: WaadInputs,
    b: WaadInputs,
    *,
    policy: StructuringPolicy | None = None,
) -> tuple[StructuringResult, StructuringResult]:
    """Validate two Wa'ds together; flags BILATERAL_WAAD_BAN when applicable."""
    ra = structure_waad(a, policy=policy)
    rb = structure_waad(b, policy=policy)
    if detect_bilateral_pair(a, b):
        ra = StructuringResult(
            waad_id=a.waad_id,
            issues=ra.issues | {WaadIssue.BILATERAL_WAAD_BAN},
            is_valid=False,
        )
        rb = StructuringResult(
            waad_id=b.waad_id,
            issues=rb.issues | {WaadIssue.BILATERAL_WAAD_BAN},
            is_valid=False,
        )
    return ra, rb


# --- Synthetic payoffs ------------------------------------------------------


@dataclass(frozen=True)
class PayoffAtExpiry:
    """Settlement-date payoff for a Wa'd-based synthetic position."""

    waad_id: str
    settlement_price: float
    payoff: float

    def __post_init__(self) -> None:
        if self.settlement_price < 0:
            raise ValueError("settlement_price cannot be negative")


def synthetic_call_payoff(waad: WaadInputs, settlement_price: float) -> PayoffAtExpiry:
    """A long-call-equivalent: PROMISE_TO_BUY at strike → exercise if S > K."""
    if waad.direction is not WaadDirection.PROMISE_TO_BUY:
        raise ValueError("synthetic_call_payoff requires PROMISE_TO_BUY direction")
    if settlement_price < 0:
        raise ValueError("settlement_price cannot be negative")
    payoff = max(settlement_price - waad.strike_price, 0.0) * waad.quantity
    return PayoffAtExpiry(
        waad_id=waad.waad_id,
        settlement_price=settlement_price,
        payoff=payoff,
    )


def synthetic_put_payoff(waad: WaadInputs, settlement_price: float) -> PayoffAtExpiry:
    """A long-put-equivalent: PROMISE_TO_SELL at strike → exercise if S < K."""
    if waad.direction is not WaadDirection.PROMISE_TO_SELL:
        raise ValueError("synthetic_put_payoff requires PROMISE_TO_SELL direction")
    if settlement_price < 0:
        raise ValueError("settlement_price cannot be negative")
    payoff = max(waad.strike_price - settlement_price, 0.0) * waad.quantity
    return PayoffAtExpiry(
        waad_id=waad.waad_id,
        settlement_price=settlement_price,
        payoff=payoff,
    )


# --- Render -----------------------------------------------------------------


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


def render_waad(inputs: WaadInputs, result: StructuringResult) -> str:
    emoji = "✅" if result.is_valid else "❌"
    head = (
        f"{emoji} {inputs.waad_id} {inputs.direction.value} "
        f"{inputs.quantity:.2f} {inputs.underlying} @ {inputs.strike_price:.2f} "
        f"(market: {inputs.market_price:.2f})"
    )
    lines = [
        head,
        f"  promisor: {inputs.promisor} → promisee: {inputs.promisee}",
        f"  exercise: {inputs.exercise_date.isoformat()}",
    ]
    for issue in sorted(result.issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))
