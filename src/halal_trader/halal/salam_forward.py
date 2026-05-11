"""Salam forwards for hedging — Round-5 Wave 4.C.

Salam (pre-paid forward) is the AAOIFI-recognised mechanism for
hedging future declining-price exposure on a *fungible* asset. The
seller (the hedger) pre-receives cash for a fungible quantity; the
buyer (the counterparty) pre-pays and takes delivery later.

In a halal trading platform this means:
- A long-position holder who wants downside protection enters a Salam
  contract as the *seller*. They receive cash now, owe a fungible
  delivery later. If the price falls, the cash they received exceeds
  the cost of fulfilling at maturity → hedged.
- A counterparty (often another platform user, or the platform's
  treasury sleeve) pays cash now and receives delivery later. They
  are taking the long-forward leg.

This module is the **hedge planner** + **counterparty matching engine**.
It composes with `halal/salam_istisna.py` for the actual contract
structuring (full prepayment + fungibility validation).

Pinned semantics:

- **Salam fungibility check is hard.** Equities are *not* fungible
  under classical fiqh — only commodities, currencies, and
  AAOIFI-classified fungible-equivalents qualify. Equity hedges
  must use the Wa'd-put construct (Wave 4.F) instead.
- **Full prepayment required at contract initiation.** Pinned per
  AAOIFI Standard 10 cl. 3.1.
- **Delivery date in the future, ≤ 12 months ahead** — Standard 10
  cl. 4.4 limits Salam tenors to typical commercial windows.
- **Counterparty matching is FIFO + risk-tolerance aware.** Two
  parties match if their delivery date / quantity / asset overlap;
  oldest open request wins on tie.
- **Plan output is deterministic** — pure-Python; no clock /
  network side-effects.
- **No-secret-leak pin** on render — counterparty IDs are masked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum


# AAOIFI-classified fungible asset classes. Equities + NFTs are
# excluded — fungible by classical fiqh requires standardised,
# interchangeable units (grain, metal weight, currency unit).
class FungibleClass(str, Enum):
    """Closed-set asset class permissible as Salam underlying."""

    GRAIN = "grain"  # wheat, rice, barley
    SOFT_COMMODITY = "soft_commodity"  # sugar, coffee, cocoa
    BASE_METAL = "base_metal"  # copper, aluminium, palladium
    PRECIOUS_METAL = "precious_metal"  # gold, silver, platinum
    CURRENCY = "currency"  # USD, EUR (subject to spot-immediate rules)
    ENERGY = "energy"  # crude oil, natural gas (where halal)


class HedgeIntent(str, Enum):
    """Why the operator is entering the Salam."""

    DOWNSIDE_PROTECTION = "downside_protection"
    INCOME = "income"  # selling forward at a premium
    INVENTORY_FUND = "inventory_fund"  # raise cash now against future inventory


@dataclass(frozen=True)
class HedgeRequest:
    """A request to enter a Salam forward as the seller (hedger)."""

    request_id: str
    party_id: str
    asset_class: FungibleClass
    asset_symbol: str
    quantity: float
    spot_price: float
    delivery_date: date
    request_date: date
    intent: HedgeIntent = HedgeIntent.DOWNSIDE_PROTECTION
    risk_tolerance: float = 0.5  # 0.0 conservative, 1.0 aggressive

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.party_id or not self.party_id.strip():
            raise ValueError("party_id must be non-empty")
        if not self.asset_symbol or not self.asset_symbol.strip():
            raise ValueError("asset_symbol must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.spot_price <= 0:
            raise ValueError("spot_price must be positive")
        if self.delivery_date <= self.request_date:
            raise ValueError("delivery_date must be after request_date")
        # AAOIFI Standard 10 cl. 4.4: tenors ≤ 12 months are standard.
        if (self.delivery_date - self.request_date).days > 365:
            raise ValueError("delivery_date > 365 days ahead exceeds standard Salam tenor")
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError("risk_tolerance must be in [0, 1]")


@dataclass(frozen=True)
class CounterpartyOffer:
    """A counterparty's offer to pay cash now for future delivery (the
    buy-leg of the Salam)."""

    offer_id: str
    party_id: str
    asset_class: FungibleClass
    asset_symbol: str
    max_quantity: float
    earliest_delivery: date
    latest_delivery: date
    offer_date: date
    discount_rate: float = 0.05  # discount applied to spot for prepayment

    def __post_init__(self) -> None:
        if not self.offer_id or not self.offer_id.strip():
            raise ValueError("offer_id must be non-empty")
        if not self.party_id or not self.party_id.strip():
            raise ValueError("party_id must be non-empty")
        if not self.asset_symbol or not self.asset_symbol.strip():
            raise ValueError("asset_symbol must be non-empty")
        if self.max_quantity <= 0:
            raise ValueError("max_quantity must be positive")
        if self.earliest_delivery > self.latest_delivery:
            raise ValueError("earliest_delivery must be ≤ latest_delivery")
        if self.earliest_delivery <= self.offer_date:
            raise ValueError("earliest_delivery must be after offer_date")
        if not 0.0 <= self.discount_rate < 0.50:
            raise ValueError("discount_rate must be in [0, 0.50)")


@dataclass(frozen=True)
class SalamPlan:
    """Output of `plan_salam` — the executed contract terms."""

    request_id: str
    offer_id: str
    asset_symbol: str
    asset_class: FungibleClass
    quantity: float
    delivery_date: date
    prepayment_amount: float
    spot_price: float
    discount_applied: float
    expected_pnl_at_delivery: float

    def is_full_prepayment(self) -> bool:
        """AAOIFI Standard 10 cl. 3.1 pin — Salam is fully prepaid."""
        return self.prepayment_amount > 0


@dataclass(frozen=True)
class MatchResult:
    """Output of `match_counterparties`."""

    plans: tuple[SalamPlan, ...]
    unmatched_requests: tuple[HedgeRequest, ...]
    unmatched_offers: tuple[CounterpartyOffer, ...]

    def fully_matched(self) -> bool:
        return not self.unmatched_requests and not self.unmatched_offers


def plan_salam(
    request: HedgeRequest,
    offer: CounterpartyOffer,
    *,
    expected_price_at_delivery: float | None = None,
) -> SalamPlan:
    """Build the contract terms for a single matched (request, offer) pair.

    Validates that the pair is structurally compatible:
    - Same asset_class + asset_symbol
    - Quantity must be ≤ offer.max_quantity
    - Delivery date must lie within offer's [earliest, latest] window
    - Asset must be fungible (FungibleClass enforces this; equities
      are excluded by type system).

    Computes the prepayment amount as `quantity × (spot - discount)`.
    Estimates P&L at delivery if `expected_price_at_delivery` provided.
    """
    if request.asset_class is not offer.asset_class:
        raise ValueError("asset_class mismatch")
    if request.asset_symbol != offer.asset_symbol:
        raise ValueError("asset_symbol mismatch")
    if request.quantity > offer.max_quantity:
        raise ValueError("request quantity exceeds offer max_quantity")
    if not (offer.earliest_delivery <= request.delivery_date <= offer.latest_delivery):
        raise ValueError("delivery_date outside offer window")
    discount = request.spot_price * offer.discount_rate
    forward_unit_price = request.spot_price - discount
    prepayment = request.quantity * forward_unit_price
    if expected_price_at_delivery is not None:
        # P&L for the seller (hedger): they received `prepayment`,
        # owe `quantity × expected_price` worth of asset at delivery.
        # Net P&L = prepayment - quantity × expected_price (in asset $).
        # Positive P&L → hedge protected against price drop.
        pnl = prepayment - request.quantity * expected_price_at_delivery
    else:
        pnl = 0.0
    return SalamPlan(
        request_id=request.request_id,
        offer_id=offer.offer_id,
        asset_symbol=request.asset_symbol,
        asset_class=request.asset_class,
        quantity=request.quantity,
        delivery_date=request.delivery_date,
        prepayment_amount=prepayment,
        spot_price=request.spot_price,
        discount_applied=discount,
        expected_pnl_at_delivery=pnl,
    )


def match_counterparties(
    requests: Iterable[HedgeRequest],
    offers: Iterable[CounterpartyOffer],
) -> MatchResult:
    """FIFO match requests to offers, oldest first on tie.

    A request matches an offer iff:
    - Same asset_class + asset_symbol
    - Request.delivery_date ∈ [offer.earliest_delivery, offer.latest_delivery]
    - Request.quantity ≤ offer.max_quantity (offer is consumed if equal,
      reduced if less — handled by tracking remaining).

    Each request consumes at most one offer. We don't split a request
    across multiple offers in this primitive — operators see the
    unmatched residual and re-submit if they want partial fill semantics.
    """
    req_list = sorted(requests, key=lambda r: r.request_date)
    off_list = sorted(offers, key=lambda o: o.offer_date)
    plans: list[SalamPlan] = []
    used_offer_ids: set[str] = set()
    unmatched_reqs: list[HedgeRequest] = []
    for req in req_list:
        chosen: CounterpartyOffer | None = None
        for off in off_list:
            if off.offer_id in used_offer_ids:
                continue
            if off.asset_class is not req.asset_class:
                continue
            if off.asset_symbol != req.asset_symbol:
                continue
            if not (off.earliest_delivery <= req.delivery_date <= off.latest_delivery):
                continue
            if req.quantity > off.max_quantity:
                continue
            chosen = off
            break
        if chosen is None:
            unmatched_reqs.append(req)
            continue
        plans.append(plan_salam(req, chosen))
        used_offer_ids.add(chosen.offer_id)
    unmatched_offers = tuple(o for o in off_list if o.offer_id not in used_offer_ids)
    return MatchResult(
        plans=tuple(plans),
        unmatched_requests=tuple(unmatched_reqs),
        unmatched_offers=unmatched_offers,
    )


def _mask(party_id: str) -> str:
    """Mask the operator-/counterparty-ID for the no-secret-leak render."""
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_plan(plan: SalamPlan) -> str:
    """Operator-readable summary of one Salam plan."""
    return (
        f"📑 Salam: {plan.quantity:.2f} {plan.asset_symbol} "
        f"({plan.asset_class.value}) → delivery {plan.delivery_date.isoformat()}\n"
        f"  • Prepayment: {plan.prepayment_amount:.2f} (discount {plan.discount_applied:.2f})\n"
        f"  • Spot ref: {plan.spot_price:.2f}\n"
        f"  • Expected P&L at delivery: {plan.expected_pnl_at_delivery:.2f}"
    )


def render_match(match: MatchResult) -> str:
    """Operator-readable summary of a match round."""
    head = (
        f"🤝 Match round: {len(match.plans)} matched, "
        f"{len(match.unmatched_requests)} unmatched req, "
        f"{len(match.unmatched_offers)} unmatched offers"
    )
    lines = [head]
    for p in match.plans:
        lines.append(
            f"  • [{p.request_id}↔{p.offer_id}] "
            f"{p.quantity:.2f} {p.asset_symbol} → "
            f"{p.delivery_date.isoformat()} "
            f"prepay={p.prepayment_amount:.2f}"
        )
    if match.unmatched_requests:
        lines.append("  Unmatched requests:")
        for r in match.unmatched_requests:
            lines.append(
                f"  • [{r.request_id}] party={_mask(r.party_id)} "
                f"{r.quantity:.2f} {r.asset_symbol} "
                f"by {r.delivery_date.isoformat()}"
            )
    return "\n".join(lines)
