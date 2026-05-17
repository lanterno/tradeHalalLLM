"""Arboun (down-payment with forfeit option) — Round-5 Wave 4.B.

Arboun is the classical fiqh construct closest to a conventional call
option: the buyer pays a non-refundable down-payment ("arboun") for
the right (not obligation) to complete a purchase at a fixed price
within a window. If the buyer walks, the down-payment is forfeit; if
they exercise, the down-payment counts toward the purchase price.

AAOIFI Standard 20 + Standard 53 explicitly admit Arboun, with
guardrails:

- The down-payment must be reasonable (not so small as to be a fee
  for the option, not so large as to make non-exercise punitive).
- The exercise window must be specified.
- The underlying asset must itself be halal.
- The down-payment cannot earn interest while held.

This module ships the **structuring engine + payoff helpers**.
Persistence + broker dispatch live above.

Pinned semantics:

- **Closed-set ArbounIssue ladder** — 8 enumerated issues.
- **Down-payment ratio band** — operator-tunable; defaults pin
  reasonable AAOIFI ranges (3% min, 25% max of purchase price).
- **`exercise_payoff`** = (settlement_price - purchase_price) × qty
  if exercised, else - down_payment.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class ArbounIssue(str, Enum):
    """Closed-set issues an Arboun structure can carry."""

    DOWN_PAYMENT_TOO_SMALL = "down_payment_too_small"
    DOWN_PAYMENT_TOO_LARGE = "down_payment_too_large"
    EXERCISE_DATE_NOT_FUTURE = "exercise_date_not_future"
    EXERCISE_DATE_TOO_FAR = "exercise_date_too_far"
    NEGATIVE_QUANTITY = "negative_quantity"
    NEGATIVE_PRICE = "negative_price"
    UNDERLYING_NOT_HALAL = "underlying_not_halal"
    DOWN_PAYMENT_EARNS_INTEREST = "down_payment_earns_interest"


@dataclass(frozen=True)
class ArbounPolicy:
    """Operator-tunable Arboun policy.

    Defaults pin the AAOIFI Standard 20 reasonable-ratio guidance
    (3% to 25% of purchase price); operators with stricter scholar
    guidance can narrow.
    """

    min_down_payment_pct: float = 0.03
    max_down_payment_pct: float = 0.25
    max_term_days: int = 180  # 6 months — classical-fiqh ceiling

    def __post_init__(self) -> None:
        if not 0.0 < self.min_down_payment_pct < self.max_down_payment_pct <= 1.0:
            raise ValueError("0 < min_down_payment_pct < max_down_payment_pct <= 1.0")
        if self.max_term_days <= 0:
            raise ValueError("max_term_days must be positive")


@dataclass(frozen=True)
class ArbounInputs:
    """Inputs for a proposed Arboun."""

    arboun_id: str
    buyer: str
    seller: str
    underlying: str
    underlying_is_halal: bool
    quantity: float
    purchase_price_per_unit: float
    down_payment_amount: float
    promise_date: date
    exercise_date: date
    down_payment_held_in_interest_account: bool = False

    def __post_init__(self) -> None:
        if not self.arboun_id or not self.arboun_id.strip():
            raise ValueError("arboun_id must be non-empty")
        if not self.underlying or not self.underlying.strip():
            raise ValueError("underlying must be non-empty")


@dataclass(frozen=True)
class StructuringResult:
    """Result of running an Arboun through the structurer."""

    arboun_id: str
    issues: frozenset[ArbounIssue]
    is_valid: bool
    down_payment_pct: float

    def __post_init__(self) -> None:
        if self.is_valid and self.issues:
            raise ValueError("is_valid=True but issues non-empty")
        if (not self.is_valid) and not self.issues:
            raise ValueError("is_valid=False but issues empty")


def structure_arboun(
    inputs: ArbounInputs, *, policy: ArbounPolicy | None = None
) -> StructuringResult:
    """Validate an Arboun against AAOIFI Standard 20 + 53."""
    pol = policy if policy is not None else ArbounPolicy()
    issues: set[ArbounIssue] = set()

    if inputs.quantity <= 0:
        issues.add(ArbounIssue.NEGATIVE_QUANTITY)
    if inputs.purchase_price_per_unit <= 0:
        issues.add(ArbounIssue.NEGATIVE_PRICE)
    if inputs.down_payment_amount < 0:
        issues.add(ArbounIssue.NEGATIVE_PRICE)
    if not inputs.underlying_is_halal:
        issues.add(ArbounIssue.UNDERLYING_NOT_HALAL)
    if inputs.down_payment_held_in_interest_account:
        issues.add(ArbounIssue.DOWN_PAYMENT_EARNS_INTEREST)

    delta = inputs.exercise_date - inputs.promise_date
    if delta <= timedelta(0):
        issues.add(ArbounIssue.EXERCISE_DATE_NOT_FUTURE)
    if delta > timedelta(days=pol.max_term_days):
        issues.add(ArbounIssue.EXERCISE_DATE_TOO_FAR)

    purchase_total = inputs.purchase_price_per_unit * inputs.quantity
    down_pct = inputs.down_payment_amount / purchase_total if purchase_total > 0 else 0.0
    if purchase_total > 0:
        if down_pct < pol.min_down_payment_pct:
            issues.add(ArbounIssue.DOWN_PAYMENT_TOO_SMALL)
        if down_pct > pol.max_down_payment_pct:
            issues.add(ArbounIssue.DOWN_PAYMENT_TOO_LARGE)

    return StructuringResult(
        arboun_id=inputs.arboun_id,
        issues=frozenset(issues),
        is_valid=len(issues) == 0,
        down_payment_pct=down_pct,
    )


@dataclass(frozen=True)
class ExerciseDecision:
    """The exercise / forfeit decision at expiry."""

    arboun_id: str
    settlement_price_per_unit: float
    exercised: bool
    payoff: float

    def __post_init__(self) -> None:
        if self.settlement_price_per_unit < 0:
            raise ValueError("settlement_price_per_unit must be non-negative")


def decide_exercise(inputs: ArbounInputs, settlement_price_per_unit: float) -> ExerciseDecision:
    """Decide whether to exercise the Arboun at expiry.

    Rational buyer exercises if settlement_price > purchase_price (the
    asset is worth more than the locked-in price, so completing the
    purchase realises a gain larger than the down-payment forfeit).
    """
    if settlement_price_per_unit < 0:
        raise ValueError("settlement_price_per_unit must be non-negative")

    # Net payoff if exercised: (S - K) × qty — down-payment counts toward purchase.
    # Net payoff if forfeited: -down_payment (sunk cost).
    exercise_payoff = (settlement_price_per_unit - inputs.purchase_price_per_unit) * inputs.quantity
    forfeit_payoff = -inputs.down_payment_amount

    if exercise_payoff > forfeit_payoff:
        return ExerciseDecision(
            arboun_id=inputs.arboun_id,
            settlement_price_per_unit=settlement_price_per_unit,
            exercised=True,
            payoff=exercise_payoff,
        )
    return ExerciseDecision(
        arboun_id=inputs.arboun_id,
        settlement_price_per_unit=settlement_price_per_unit,
        exercised=False,
        payoff=forfeit_payoff,
    )


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


def render_structure(inputs: ArbounInputs, result: StructuringResult) -> str:
    emoji = "✅" if result.is_valid else "❌"
    head = (
        f"{emoji} {inputs.arboun_id} arboun: buy {inputs.quantity:.2f} "
        f"{inputs.underlying} @ {inputs.purchase_price_per_unit:.2f} "
        f"(down-payment {result.down_payment_pct * 100:.1f}%)"
    )
    lines = [head]
    for issue in sorted(result.issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))


def render_exercise(decision: ExerciseDecision) -> str:
    state = "exercised" if decision.exercised else "forfeited"
    return _scrub(
        f"⚖ {decision.arboun_id} {state} @ "
        f"{decision.settlement_price_per_unit:.2f} → ${decision.payoff:+.2f}"
    )
