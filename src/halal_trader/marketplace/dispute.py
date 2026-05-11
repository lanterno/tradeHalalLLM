"""Refund + dispute resolution for paid signals/strategies — Round-5 Wave 21.H.

When a subscriber claims a signal materially misrepresents itself,
they file a dispute. This module is the **FSM + refund computer**:

1. Subscriber files a dispute citing a reason.
2. Platform/operator reviews; decides REFUND / PARTIAL_REFUND /
   REJECT / WARN_AUTHOR.
3. The amount-owed computation respects the original Wakalah fee
   structure (flat, not performance-based).

Pinned semantics:

- **Closed-set DisputeReason** — MISREPRESENTATION / NON_DELIVERY /
  HALAL_VIOLATION / PERFORMANCE_CLAIM_FALSE / OTHER.
- **Closed-set DisputeStatus FSM** — FILED → UNDER_REVIEW → RESOLVED
  (terminal) with WITHDRAWN as an alternate terminal state filed by
  subscriber.
- **Closed-set ResolutionOutcome** — FULL_REFUND / PARTIAL_REFUND /
  REJECT / WARN_AUTHOR.
- **Refund amount** is bounded by the original `fee_paid` and the
  optional `service_days_provided` ratio for PARTIAL_REFUND.
- **Author cannot file a dispute against their own listing.**
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


class DisputeReason(str, Enum):
    """Closed-set reason ladder."""

    MISREPRESENTATION = "misrepresentation"
    NON_DELIVERY = "non_delivery"
    HALAL_VIOLATION = "halal_violation"
    PERFORMANCE_CLAIM_FALSE = "performance_claim_false"
    OTHER = "other"


class DisputeStatus(str, Enum):
    """Closed-set status FSM."""

    FILED = "filed"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    WITHDRAWN = "withdrawn"


class ResolutionOutcome(str, Enum):
    """Closed-set outcome ladder."""

    FULL_REFUND = "full_refund"
    PARTIAL_REFUND = "partial_refund"
    REJECT = "reject"
    WARN_AUTHOR = "warn_author"


_LEGAL_TRANSITIONS: dict[DisputeStatus, set[DisputeStatus]] = {
    DisputeStatus.FILED: {DisputeStatus.UNDER_REVIEW, DisputeStatus.WITHDRAWN},
    DisputeStatus.UNDER_REVIEW: {DisputeStatus.RESOLVED, DisputeStatus.WITHDRAWN},
    DisputeStatus.RESOLVED: set(),
    DisputeStatus.WITHDRAWN: set(),
}


@dataclass(frozen=True)
class Dispute:
    """A subscriber's dispute filing."""

    dispute_id: str
    subscription_id: str
    subscriber_id: str
    author_id: str
    fee_paid_usd: float
    service_days_provided: int
    """How many days of service were delivered before the dispute. Used
    for PARTIAL_REFUND."""
    total_service_days: int
    """Total subscription period in days."""
    reason: DisputeReason
    rationale: str
    filed_at: datetime
    status: DisputeStatus = DisputeStatus.FILED
    resolution: ResolutionOutcome | None = None
    refund_amount_usd: float = 0.0
    resolved_at: datetime | None = None
    operator_notes: str = ""

    def __post_init__(self) -> None:
        if not self.dispute_id or not self.dispute_id.strip():
            raise ValueError("dispute_id must be non-empty")
        if not self.subscription_id or not self.subscription_id.strip():
            raise ValueError("subscription_id must be non-empty")
        if not self.subscriber_id or not self.subscriber_id.strip():
            raise ValueError("subscriber_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if self.subscriber_id == self.author_id:
            raise ValueError("subscriber and author must be distinct parties")
        if self.fee_paid_usd < 0:
            raise ValueError("fee_paid_usd must be non-negative")
        if self.service_days_provided < 0:
            raise ValueError("service_days_provided must be ≥ 0")
        if self.total_service_days <= 0:
            raise ValueError("total_service_days must be positive")
        if self.service_days_provided > self.total_service_days:
            raise ValueError("service_days_provided cannot exceed total_service_days")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty")
        if len(self.rationale) > 2000:
            raise ValueError("rationale must be ≤ 2000 chars")
        if self.refund_amount_usd < 0:
            raise ValueError("refund_amount_usd must be non-negative")
        if self.refund_amount_usd > self.fee_paid_usd + 1e-9:
            raise ValueError("refund cannot exceed fee_paid")
        # Resolution + status consistency.
        if self.status is DisputeStatus.RESOLVED and self.resolution is None:
            raise ValueError("RESOLVED dispute must have a resolution")
        if self.status is not DisputeStatus.RESOLVED and self.resolution is not None:
            raise ValueError("resolution can only be set on RESOLVED disputes")
        if self.status is DisputeStatus.RESOLVED and self.resolved_at is None:
            raise ValueError("RESOLVED dispute must have resolved_at")
        if self.resolved_at is not None and self.resolved_at < self.filed_at:
            raise ValueError("resolved_at cannot precede filed_at")


def transition(
    dispute: Dispute,
    *,
    new_status: DisputeStatus,
    at: datetime,
    resolution: ResolutionOutcome | None = None,
    refund_amount_usd: float | None = None,
    operator_notes: str = "",
) -> Dispute:
    """Transition the dispute through the FSM.

    Pinned legal moves:
        FILED → UNDER_REVIEW
        FILED → WITHDRAWN (subscriber-initiated)
        UNDER_REVIEW → RESOLVED (requires resolution + refund)
        UNDER_REVIEW → WITHDRAWN
    """
    if new_status not in _LEGAL_TRANSITIONS[dispute.status]:
        raise ValueError(f"illegal transition {dispute.status.value} → {new_status.value}")
    new_resolution = dispute.resolution
    new_refund = dispute.refund_amount_usd
    new_resolved_at = dispute.resolved_at
    notes = operator_notes or dispute.operator_notes
    if new_status is DisputeStatus.RESOLVED:
        if resolution is None:
            raise ValueError("RESOLVED transition requires a resolution")
        new_resolution = resolution
        new_resolved_at = at
        new_refund = compute_refund(
            dispute,
            outcome=resolution,
            override_refund_usd=refund_amount_usd,
        )
    return replace(
        dispute,
        status=new_status,
        resolution=new_resolution,
        refund_amount_usd=new_refund,
        resolved_at=new_resolved_at,
        operator_notes=notes,
    )


def compute_refund(
    dispute: Dispute,
    *,
    outcome: ResolutionOutcome,
    override_refund_usd: float | None = None,
) -> float:
    """Compute the refund per outcome.

    Pinned:
    - FULL_REFUND → fee_paid.
    - PARTIAL_REFUND → fee_paid × (1 - service_days_provided/total_service_days).
      Operators can override with `override_refund_usd` (within bounds).
    - REJECT → 0.
    - WARN_AUTHOR → 0 (author is notified but no money moves).
    """
    if override_refund_usd is not None:
        if override_refund_usd < 0:
            raise ValueError("override_refund_usd must be non-negative")
        if override_refund_usd > dispute.fee_paid_usd + 1e-9:
            raise ValueError("override_refund_usd cannot exceed fee_paid")
        return override_refund_usd
    if outcome is ResolutionOutcome.FULL_REFUND:
        return dispute.fee_paid_usd
    if outcome is ResolutionOutcome.PARTIAL_REFUND:
        used_frac = dispute.service_days_provided / dispute.total_service_days
        return dispute.fee_paid_usd * (1.0 - used_frac)
    return 0.0


def is_terminal(status: DisputeStatus) -> bool:
    return status in (DisputeStatus.RESOLVED, DisputeStatus.WITHDRAWN)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_OUTCOME_EMOJI: dict[ResolutionOutcome, str] = {
    ResolutionOutcome.FULL_REFUND: "💸",
    ResolutionOutcome.PARTIAL_REFUND: "💵",
    ResolutionOutcome.REJECT: "🚫",
    ResolutionOutcome.WARN_AUTHOR: "⚠️",
}


_STATUS_EMOJI: dict[DisputeStatus, str] = {
    DisputeStatus.FILED: "📨",
    DisputeStatus.UNDER_REVIEW: "🔎",
    DisputeStatus.RESOLVED: "✅",
    DisputeStatus.WITHDRAWN: "↩️",
}


def render_dispute(dispute: Dispute) -> str:
    head = (
        f"{_STATUS_EMOJI[dispute.status]} {dispute.dispute_id} "
        f"[{dispute.status.value}] reason={dispute.reason.value}\n"
        f"  Subscriber {_mask(dispute.subscriber_id)} → "
        f"Author {_mask(dispute.author_id)} "
        f"(${dispute.fee_paid_usd:.2f}, "
        f"{dispute.service_days_provided}/{dispute.total_service_days}d)"
    )
    if dispute.resolution is not None:
        head += (
            f"\n  Resolution: {_OUTCOME_EMOJI[dispute.resolution]} "
            f"{dispute.resolution.value} "
            f"refund=${dispute.refund_amount_usd:.2f}"
        )
    return head
