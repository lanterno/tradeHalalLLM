"""Tail-risk halal put — Round-5 Wave 4.F.

Conventional puts grant the holder a one-sided right to sell at a strike
— this is permissible iff structured as a Wa'd (unilateral promise) by
the writer rather than a sold contract. The Wa'd-based halal-put has
distinct terms:

- A counterparty (the *promisor*) issues a binding promise to buy the
  hedger's shares at a strike if specific conditions hold.
- The hedger pays no upfront premium (or, more often, an Arboun
  down-payment that doubles as good-faith collateral and is
  forfeited if the hedger doesn't exercise — see `halal/arboun.py`).
- Conditions are explicit and verifiable (e.g. "spot below strike for 5
  consecutive days" or "VIX > 40 on exercise date") — the operator
  must observe the condition objectively before invoking the promise.

This module is the **hedge planner**. It composes with `halal/waad.py`
for the underlying contract validation; this layer translates an
operator's tail-risk view into a concrete Wa'd specification.

Pinned semantics:

- **Closed-set ConditionType**: PRICE_BELOW / DRAWDOWN_OVER /
  VOL_ABOVE / TIME_ELAPSED. Open conditions are forbidden — every
  trigger must be checkable from observable market data.
- **Equity hedges only.** Salam covers fungible commodities (Wave
  4.C); halal-puts cover non-fungible equities + ETFs.
- **Strike must be ≤ spot at issuance.** Out-of-the-money puts only
  — protects against a hedger writing themselves a free option.
- **Cumulative protection cap** = `quantity × strike` per contract.
  The hedger cannot extract more than the floor.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — counterparty IDs masked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class ConditionType(str, Enum):
    """Closed-set trigger conditions for the Wa'd-based put."""

    PRICE_BELOW = "price_below"
    DRAWDOWN_OVER = "drawdown_over"
    VOL_ABOVE = "vol_above"
    TIME_ELAPSED = "time_elapsed"


@dataclass(frozen=True)
class ExerciseCondition:
    """A single observable condition for exercising the Wa'd-put."""

    condition_type: ConditionType
    threshold: float
    window_days: int = 0
    """For PRICE_BELOW / DRAWDOWN_OVER / VOL_ABOVE this is the number of
    consecutive bars the threshold must hold before the condition is
    considered met. For TIME_ELAPSED it is unused."""

    def __post_init__(self) -> None:
        if self.condition_type is ConditionType.PRICE_BELOW:
            if self.threshold <= 0:
                raise ValueError("PRICE_BELOW threshold must be positive")
        elif self.condition_type is ConditionType.DRAWDOWN_OVER:
            if not 0.0 < self.threshold <= 1.0:
                raise ValueError("DRAWDOWN_OVER threshold must be in (0, 1]")
        elif self.condition_type is ConditionType.VOL_ABOVE:
            if not 0.0 < self.threshold < 5.0:
                raise ValueError("VOL_ABOVE threshold must be in (0, 5)")
        elif self.condition_type is ConditionType.TIME_ELAPSED:
            if self.threshold <= 0:
                raise ValueError("TIME_ELAPSED threshold must be positive")
        if self.window_days < 0:
            raise ValueError("window_days must be ≥ 0")


@dataclass(frozen=True)
class HalalPutTerms:
    """The negotiated halal-put hedge terms."""

    contract_id: str
    hedger_id: str
    promisor_id: str
    underlying: str
    quantity: float
    spot_at_issue: float
    strike: float
    expiry: date
    issue_date: date
    arboun_paid: float = 0.0
    conditions: tuple[ExerciseCondition, ...] = field(default_factory=tuple)
    require_all_conditions: bool = True
    """If True (the default), every condition must hold to exercise; if
    False, any one condition is enough (OR semantics)."""

    def __post_init__(self) -> None:
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if not self.hedger_id or not self.hedger_id.strip():
            raise ValueError("hedger_id must be non-empty")
        if not self.promisor_id or not self.promisor_id.strip():
            raise ValueError("promisor_id must be non-empty")
        if self.hedger_id == self.promisor_id:
            raise ValueError("hedger and promisor must be distinct parties")
        if not self.underlying or not self.underlying.strip():
            raise ValueError("underlying must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.spot_at_issue <= 0:
            raise ValueError("spot_at_issue must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.strike > self.spot_at_issue + 1e-9:
            raise ValueError("strike must be ≤ spot_at_issue (out-of-the-money put only)")
        if self.expiry <= self.issue_date:
            raise ValueError("expiry must be after issue_date")
        if self.arboun_paid < 0:
            raise ValueError("arboun_paid must be non-negative")
        if not self.conditions:
            raise ValueError("at least one exercise condition is required")
        # Defensive: forbid duplicate condition types when require_all_conditions
        # is True — operators that want stacked conditions of the same type
        # should pass them as one OR-set instead.
        if self.require_all_conditions:
            seen: set[ConditionType] = set()
            for c in self.conditions:
                if c.condition_type in seen:
                    raise ValueError("duplicate condition_type with require_all_conditions=True")
                seen.add(c.condition_type)

    def protection_cap(self) -> float:
        """The maximum exercise notional = quantity × strike."""
        return self.quantity * self.strike


@dataclass(frozen=True)
class MarketObservation:
    """Snapshot of observed market state used to test conditions."""

    spot: float
    drawdown_from_peak: float = 0.0
    realised_volatility: float = 0.0
    days_since_issue: int = 0

    def __post_init__(self) -> None:
        if self.spot < 0:
            raise ValueError("spot cannot be negative")
        if not 0.0 <= self.drawdown_from_peak <= 1.0:
            raise ValueError("drawdown_from_peak must be in [0, 1]")
        if self.realised_volatility < 0:
            raise ValueError("realised_volatility must be non-negative")
        if self.days_since_issue < 0:
            raise ValueError("days_since_issue must be ≥ 0")


def evaluate_condition(
    condition: ExerciseCondition,
    observation: MarketObservation,
) -> bool:
    """True iff the observation satisfies the condition.

    `window_days` is used by callers that pre-compute "consecutive
    days the threshold held" — the operator passes the rolling
    metric in `observation` and `evaluate_condition` returns the
    instantaneous check.
    """
    if condition.condition_type is ConditionType.PRICE_BELOW:
        return observation.spot < condition.threshold
    if condition.condition_type is ConditionType.DRAWDOWN_OVER:
        return observation.drawdown_from_peak > condition.threshold
    if condition.condition_type is ConditionType.VOL_ABOVE:
        return observation.realised_volatility > condition.threshold
    if condition.condition_type is ConditionType.TIME_ELAPSED:
        return observation.days_since_issue >= int(condition.threshold)
    raise ValueError(f"unknown condition_type {condition.condition_type}")


def can_exercise(terms: HalalPutTerms, observation: MarketObservation) -> bool:
    """Return True iff the hedger may invoke the Wa'd promise given the
    current market observation.

    AND vs OR semantics is governed by `terms.require_all_conditions`.
    """
    results = [evaluate_condition(c, observation) for c in terms.conditions]
    if terms.require_all_conditions:
        return all(results)
    return any(results)


@dataclass(frozen=True)
class ExerciseResult:
    """Output of `exercise`."""

    payout: float
    arboun_returned: float
    """Arboun is forfeited iff the hedger declines to exercise. If the
    hedger exercises, the Arboun is credited back (operator-tunable;
    default contract terms return Arboun on exercise to avoid double
    payment)."""
    is_in_the_money: bool


def exercise(
    terms: HalalPutTerms,
    observation: MarketObservation,
    *,
    return_arboun_on_exercise: bool = True,
) -> ExerciseResult:
    """Compute the payout if the hedger exercises now.

    Payout = max(strike - spot, 0) × quantity, capped by protection_cap.
    If conditions are not satisfied, raises — the caller must check
    `can_exercise` first.
    """
    if not can_exercise(terms, observation):
        raise ValueError("conditions not satisfied; cannot exercise")
    intrinsic = max(0.0, terms.strike - observation.spot)
    payout = min(terms.protection_cap(), intrinsic * terms.quantity)
    arboun_returned = terms.arboun_paid if return_arboun_on_exercise else 0.0
    return ExerciseResult(
        payout=payout,
        arboun_returned=arboun_returned,
        is_in_the_money=intrinsic > 0,
    )


@dataclass(frozen=True)
class HedgeProposal:
    """Output of `propose_hedge` — a recommended HalalPutTerms set."""

    terms: HalalPutTerms
    expected_payout_at_drawdown: float
    """Expected payout if the underlying hits a typical tail scenario
    (default: -20%)."""
    notes: tuple[str, ...]


def propose_hedge(
    *,
    contract_id: str,
    hedger_id: str,
    promisor_id: str,
    underlying: str,
    quantity: float,
    spot: float,
    issue_date: date,
    horizon_days: int,
    strike_pct: float = 0.90,
    arboun_pct: float = 0.01,
    tail_drawdown: float = 0.20,
    drawdown_trigger: float = 0.10,
    vol_trigger: float = 0.40,
) -> HedgeProposal:
    """Build a default HalalPutTerms set with reasonable conditions.

    Defaults:
    - Strike at 90% of spot (10% out-of-the-money)
    - Arboun = 1% of notional (acts as good-faith collateral)
    - Conditions: drawdown > 10% OR realised vol > 40% — operator can
      tighten or loosen via the kwargs.
    - Expiry = issue_date + horizon_days
    - require_all_conditions=False (the OR semantics matches a real
      tail-risk hedge — either signal alone justifies exercise).

    Notes are generated to remind the operator what to verify before
    signing.
    """
    if not 0.5 <= strike_pct <= 1.0:
        raise ValueError("strike_pct must be in [0.5, 1.0]")
    if not 0.0 <= arboun_pct < 0.10:
        raise ValueError("arboun_pct must be in [0, 0.10)")
    if not 0.0 < drawdown_trigger < 1.0:
        raise ValueError("drawdown_trigger must be in (0, 1)")
    if not 0.0 < vol_trigger < 5.0:
        raise ValueError("vol_trigger must be in (0, 5)")
    if not 0.0 < tail_drawdown < 1.0:
        raise ValueError("tail_drawdown must be in (0, 1)")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    strike = spot * strike_pct
    notional = quantity * spot
    arboun = notional * arboun_pct
    expiry = issue_date + timedelta(days=horizon_days)
    conditions = (
        ExerciseCondition(
            condition_type=ConditionType.DRAWDOWN_OVER,
            threshold=drawdown_trigger,
            window_days=3,
        ),
        ExerciseCondition(
            condition_type=ConditionType.VOL_ABOVE,
            threshold=vol_trigger,
            window_days=5,
        ),
    )
    terms = HalalPutTerms(
        contract_id=contract_id,
        hedger_id=hedger_id,
        promisor_id=promisor_id,
        underlying=underlying,
        quantity=quantity,
        spot_at_issue=spot,
        strike=strike,
        expiry=expiry,
        issue_date=issue_date,
        arboun_paid=arboun,
        conditions=conditions,
        require_all_conditions=False,
    )
    # Expected payout at tail_drawdown — model spot at -tail% of issue.
    spot_at_tail = spot * (1 - tail_drawdown)
    intrinsic_tail = max(0.0, strike - spot_at_tail)
    expected_tail_payout = min(terms.protection_cap(), intrinsic_tail * quantity)
    notes: tuple[str, ...] = (
        "OR conditions: any one trigger is enough — confirm operator intent before signing.",
        "Arboun is forfeited if the hedger does not exercise — verify "
        "operator's commitment to cover the worst-case forfeit.",
        "Strike is OUT-of-the-money: protection only kicks in below "
        f"{strike:.2f} (vs spot {spot:.2f}).",
    )
    return HedgeProposal(
        terms=terms,
        expected_payout_at_drawdown=expected_tail_payout,
        notes=notes,
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_terms(terms: HalalPutTerms) -> str:
    """Operator-readable summary; counterparty IDs masked."""
    cond_lines: list[str] = []
    for c in terms.conditions:
        cond_lines.append(f"    - {c.condition_type.value} {c.threshold} (window {c.window_days}d)")
    semantics = "ALL" if terms.require_all_conditions else "ANY"
    head = (
        f"🛡️ Halal-put: {terms.quantity:.2f} {terms.underlying} "
        f"@ strike {terms.strike:.2f} (spot {terms.spot_at_issue:.2f}), "
        f"expiry {terms.expiry.isoformat()}"
    )
    body = [
        head,
        f"  • Hedger: {_mask(terms.hedger_id)} ↔ Promisor: {_mask(terms.promisor_id)}",
        f"  • Arboun: {terms.arboun_paid:.2f}",
        f"  • Conditions ({semantics}):",
        *cond_lines,
        f"  • Protection cap: {terms.protection_cap():.2f}",
    ]
    return "\n".join(body)


def render_proposal(proposal: HedgeProposal) -> str:
    """Render a proposal — terms + expected tail payout + notes."""
    lines = [render_terms(proposal.terms)]
    lines.append(
        f"  • Expected payout at -20% drawdown: {proposal.expected_payout_at_drawdown:.2f}"
    )
    lines.append("  • Notes:")
    for note in proposal.notes:
        lines.append(f"    - {note}")
    return "\n".join(lines)
