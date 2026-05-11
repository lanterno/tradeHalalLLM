"""GDPR / CCPA / India DPDP data-subject request workflow — Round-5 Wave 19.G.

Data-subject rights vary by jurisdiction:
- **GDPR (EU)**: access, rectification, erasure, portability, objection.
- **CCPA (California)**: access, deletion, opt-out.
- **DPDP (India)**: access, correction, erasure, grievance.

This module is the **request FSM + scope-of-data computer + retention-
exception checker**. The flow:

1. Subject files a request (closed-set Jurisdiction × RequestKind).
2. Platform marks the request VERIFYING (identity check) → IN_PROGRESS
   → COMPLETED, or REJECTED at any earlier point.
3. For ERASURE requests, the scope is computed: which user-data
   categories are deletable, which must be retained (audit / legal
   hold / regulatory retention) per `RetentionPolicy`.

Pinned semantics:

- **Closed-set Jurisdiction** — GDPR_EU / CCPA_CA / DPDP_IN.
- **Closed-set RequestKind** — ACCESS / RECTIFICATION / ERASURE /
  PORTABILITY / OBJECTION / OPT_OUT / GRIEVANCE.
- **Closed-set RequestStatus FSM** — FILED → VERIFYING → IN_PROGRESS →
  COMPLETED, with REJECTED as alternate terminal state.
- **Each (jurisdiction, kind) pair has an explicit SLA in days**
  per the regulator's published timelines.
- **DataCategory ladder** is closed; deletability is per-(category,
  jurisdiction) tunable; the audit-trail category is non-deletable
  by default (legal hold).
- **Pure-Python deterministic.**
- **No-secret-leak pin** — subject IDs masked in render.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from enum import Enum


class Jurisdiction(str, Enum):
    """Closed-set jurisdiction ladder."""

    GDPR_EU = "gdpr_eu"
    CCPA_CA = "ccpa_ca"
    DPDP_IN = "dpdp_in"


class RequestKind(str, Enum):
    """Closed-set data-subject request kind."""

    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    OBJECTION = "objection"
    OPT_OUT = "opt_out"
    GRIEVANCE = "grievance"


class RequestStatus(str, Enum):
    """Closed-set request FSM ladder."""

    FILED = "filed"
    VERIFYING = "verifying"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"


class DataCategory(str, Enum):
    """Closed-set categories of user data the platform holds."""

    ACCOUNT_PROFILE = "account_profile"
    TRADE_HISTORY = "trade_history"
    KYC_DOCUMENTS = "kyc_documents"
    AUDIT_TRAIL = "audit_trail"
    PAYMENT_RECORDS = "payment_records"
    COMMUNICATIONS = "communications"
    LEARNER_PROGRESS = "learner_progress"
    PORTFOLIO_SNAPSHOTS = "portfolio_snapshots"


# SLA per (jurisdiction, kind) in days. Sourced from each regulator's
# published timelines. Operator-tunable if regs change.
_SLA_DAYS: dict[tuple[Jurisdiction, RequestKind], int] = {
    # GDPR: 30 days for most (extendable to 90 with cause).
    (Jurisdiction.GDPR_EU, RequestKind.ACCESS): 30,
    (Jurisdiction.GDPR_EU, RequestKind.RECTIFICATION): 30,
    (Jurisdiction.GDPR_EU, RequestKind.ERASURE): 30,
    (Jurisdiction.GDPR_EU, RequestKind.PORTABILITY): 30,
    (Jurisdiction.GDPR_EU, RequestKind.OBJECTION): 30,
    # CCPA: 45 days for access + deletion.
    (Jurisdiction.CCPA_CA, RequestKind.ACCESS): 45,
    (Jurisdiction.CCPA_CA, RequestKind.ERASURE): 45,
    (Jurisdiction.CCPA_CA, RequestKind.OPT_OUT): 15,
    # DPDP: 30 days for most, with grievance redressal in 7 days.
    (Jurisdiction.DPDP_IN, RequestKind.ACCESS): 30,
    (Jurisdiction.DPDP_IN, RequestKind.RECTIFICATION): 30,
    (Jurisdiction.DPDP_IN, RequestKind.ERASURE): 30,
    (Jurisdiction.DPDP_IN, RequestKind.GRIEVANCE): 7,
}


def sla_days(jurisdiction: Jurisdiction, kind: RequestKind) -> int | None:
    """Return the SLA in days for this (jurisdiction, kind), or None
    if the regulator does not recognise the kind."""
    return _SLA_DAYS.get((jurisdiction, kind))


# Deletability per (category, jurisdiction). True = deletable on
# ERASURE; False = retention-required.
def _default_deletable(category: DataCategory, jurisdiction: Jurisdiction) -> bool:
    if category is DataCategory.AUDIT_TRAIL:
        return False  # Legal hold / regulator retention.
    if category is DataCategory.PAYMENT_RECORDS:
        # Most jurisdictions require ≥ 5-7 years of payment records.
        return False
    if category is DataCategory.KYC_DOCUMENTS:
        # KYC is held for AML compliance (≥ 5 years post-relationship-end).
        return False
    # Everything else is deletable on ERASURE.
    return True


@dataclass(frozen=True)
class RetentionPolicy:
    """Operator-tunable retention policy override.

    Maps (category, jurisdiction) → deletable bool. Falls back to
    `_default_deletable` if not present.
    """

    overrides: dict[tuple[DataCategory, Jurisdiction], bool] = field(default_factory=dict)

    def is_deletable(self, category: DataCategory, jurisdiction: Jurisdiction) -> bool:
        if (category, jurisdiction) in self.overrides:
            return self.overrides[(category, jurisdiction)]
        return _default_deletable(category, jurisdiction)


@dataclass(frozen=True)
class DataSubjectRequest:
    """A data-subject request in flight."""

    request_id: str
    subject_id: str
    jurisdiction: Jurisdiction
    kind: RequestKind
    filed_at: date
    status: RequestStatus = RequestStatus.FILED
    verified_at: date | None = None
    completed_at: date | None = None
    rejection_reason: str = ""
    operator_notes: str = ""

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.subject_id or not self.subject_id.strip():
            raise ValueError("subject_id must be non-empty")
        # SLA must be defined for this (jurisdiction, kind).
        if sla_days(self.jurisdiction, self.kind) is None:
            raise ValueError(
                f"{self.kind.value} is not a recognised request under {self.jurisdiction.value}"
            )
        if self.verified_at is not None and self.verified_at < self.filed_at:
            raise ValueError("verified_at must be ≥ filed_at")
        if self.completed_at is not None and self.completed_at < self.filed_at:
            raise ValueError("completed_at must be ≥ filed_at")
        if self.status is RequestStatus.REJECTED and not self.rejection_reason.strip():
            raise ValueError("REJECTED requires non-empty rejection_reason")
        if self.status is RequestStatus.REJECTED and self.completed_at is None:
            raise ValueError("REJECTED requires completed_at")
        if self.status is RequestStatus.COMPLETED and self.completed_at is None:
            raise ValueError("COMPLETED requires completed_at")

    def sla_days(self) -> int:
        return sla_days(self.jurisdiction, self.kind)  # type: ignore[return-value]

    def deadline(self) -> date:
        return self.filed_at + timedelta(days=self.sla_days())

    def is_overdue(self, as_of: date) -> bool:
        if self.status in (RequestStatus.COMPLETED, RequestStatus.REJECTED):
            return False
        return as_of > self.deadline()


_LEGAL_TRANSITIONS: dict[RequestStatus, set[RequestStatus]] = {
    RequestStatus.FILED: {RequestStatus.VERIFYING, RequestStatus.REJECTED},
    RequestStatus.VERIFYING: {
        RequestStatus.IN_PROGRESS,
        RequestStatus.REJECTED,
    },
    RequestStatus.IN_PROGRESS: {
        RequestStatus.COMPLETED,
        RequestStatus.REJECTED,
    },
    RequestStatus.COMPLETED: set(),
    RequestStatus.REJECTED: set(),
}


def transition(
    request: DataSubjectRequest,
    *,
    new_status: RequestStatus,
    at: date,
    rejection_reason: str = "",
    operator_notes: str = "",
) -> DataSubjectRequest:
    """Move the request through the FSM."""
    if new_status not in _LEGAL_TRANSITIONS[request.status]:
        raise ValueError(f"illegal transition {request.status.value} → {new_status.value}")
    verified_at = request.verified_at
    completed_at = request.completed_at
    notes = operator_notes or request.operator_notes
    reason = rejection_reason or request.rejection_reason
    if new_status is RequestStatus.VERIFYING:
        verified_at = at
    if new_status is RequestStatus.COMPLETED:
        completed_at = at
    if new_status is RequestStatus.REJECTED:
        completed_at = at
        if not reason.strip():
            raise ValueError("REJECTED transition requires rejection_reason")
    return replace(
        request,
        status=new_status,
        verified_at=verified_at,
        completed_at=completed_at,
        rejection_reason=reason,
        operator_notes=notes,
    )


@dataclass(frozen=True)
class ErasureScope:
    """Output of `erasure_scope`."""

    deletable_categories: tuple[DataCategory, ...]
    retained_categories: tuple[DataCategory, ...]


def erasure_scope(
    request: DataSubjectRequest,
    *,
    held_categories: Sequence[DataCategory],
    policy: RetentionPolicy | None = None,
) -> ErasureScope:
    """Compute which categories will be deleted vs retained.

    Raises if the request kind is not ERASURE.
    """
    if request.kind is not RequestKind.ERASURE:
        raise ValueError("erasure_scope only valid for ERASURE requests")
    pol = policy if policy is not None else RetentionPolicy()
    deletable: list[DataCategory] = []
    retained: list[DataCategory] = []
    for cat in held_categories:
        if pol.is_deletable(cat, request.jurisdiction):
            deletable.append(cat)
        else:
            retained.append(cat)
    return ErasureScope(
        deletable_categories=tuple(deletable),
        retained_categories=tuple(retained),
    )


@dataclass(frozen=True)
class PortabilityExport:
    """Output of `portability_export` — declarative; the deployment
    layer materialises the actual bytes."""

    request_id: str
    subject_id: str
    categories: tuple[DataCategory, ...]
    format: str = "json"

    def __post_init__(self) -> None:
        if self.format not in ("json", "csv"):
            raise ValueError("format must be 'json' or 'csv'")


def portability_export(
    request: DataSubjectRequest,
    *,
    held_categories: Sequence[DataCategory],
    format: str = "json",
) -> PortabilityExport:
    """Build the declarative export spec."""
    if request.kind not in (RequestKind.PORTABILITY, RequestKind.ACCESS):
        raise ValueError("portability_export only valid for PORTABILITY or ACCESS")
    # Cannot include AUDIT_TRAIL in a portability export — keep operator
    # data privileged.
    cats = tuple(c for c in held_categories if c is not DataCategory.AUDIT_TRAIL)
    return PortabilityExport(
        request_id=request.request_id,
        subject_id=request.subject_id,
        categories=cats,
        format=format,
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[RequestStatus, str] = {
    RequestStatus.FILED: "📨",
    RequestStatus.VERIFYING: "🔎",
    RequestStatus.IN_PROGRESS: "⚙️",
    RequestStatus.COMPLETED: "✅",
    RequestStatus.REJECTED: "❌",
}


def render_request(request: DataSubjectRequest, *, as_of: date | None = None) -> str:
    head = (
        f"{_STATUS_EMOJI[request.status]} {request.request_id} "
        f"[{request.jurisdiction.value}/{request.kind.value}] "
        f"subject {_mask(request.subject_id)} filed {request.filed_at.isoformat()}, "
        f"due {request.deadline().isoformat()}"
    )
    if as_of is not None and request.is_overdue(as_of):
        head += " ⚠️ OVERDUE"
    if request.status is RequestStatus.REJECTED:
        head += f"\n  Rejected: {request.rejection_reason}"
    return head


def render_erasure_scope(scope: ErasureScope) -> str:
    return (
        f"🗑️ Erasure scope: "
        f"delete={[c.value for c in scope.deletable_categories]}, "
        f"retain={[c.value for c in scope.retained_categories]}"
    )
