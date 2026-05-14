"""GDPR / CCPA privacy engine.

Once the bot grows past single-operator-laptop mode (Wave 3.A user
accounts; Wave 3.B per-user vault) and acquires EU or California
users, the deployment is on the hook for data-subject rights:

- **GDPR Article 15** — Right of Access: a user can request a copy
  of every piece of personal data the controller holds about them.
- **GDPR Article 16** — Right of Rectification: the user can correct
  inaccurate data.
- **GDPR Article 17** — Right to Erasure ("right to be forgotten"):
  the user can request deletion, subject to controller's legal
  obligations.
- **CCPA §1798.105** — California's near-equivalent right of
  deletion with similar carve-outs.

This module is the pure-Python data-subject-rights engine that
classifies every kind of data the bot holds, attaches a retention
policy + legal basis, and produces deterministic export / deletion
plans. The actual SQL the cycle / dashboard / vault uses to persist
those categories is a follow-up; the engine ships the contract
first so the persistence layer has a stable target.

Pinned semantics:
- **Legal-obligation categories cannot be hard-deleted.** Trade
  audit rows + halal-screening receipts + purification ledger
  entries carry a legal-basis of `LEGAL_OBLIGATION` (tax / AML /
  shariah-audit retention requirements); a deletion request for
  these returns `DENIED` with the legal basis explained, and the
  engine offers `ANONYMISE` as the alternative — the bot scrubs
  PII fields and keeps the financial / shariah audit trail
  intact. Pinned via test.
- **Consent-basis data MUST be deletable on request.** A user
  who withdraws consent for marketing analytics or feature-
  optimisation telemetry triggers a hard-delete of those rows;
  the engine never silently retains under "implied legitimate
  interest" because the user explicitly revoked consent.
- **Export must enumerate every category the user has data in.**
  A partial export is a regulatory failure mode — pinned via
  test that the export plan covers every category the user has
  ever populated, not just the recent ones.
- **Default retention is the shortest justifiable.** Operators
  override per category for their jurisdiction's specific
  retention requirements (e.g., FINRA 6-year trade retention in
  the US) — but the defaults are the legal minimum, not the
  comfortable-for-the-business maximum.
- **Render output never contains raw personal data** for user-
  facing receipts; the receipt summarises action + category +
  count, not the field values themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class DataCategory(str, Enum):
    """Categories of personal data the bot holds.

    Pinned string values for DB / JSON serialisation stability —
    a future schema migration that renames `pii` to `account_pii`
    would orphan the retention-policy and consent rows.
    """

    PII = "pii"  # name / email / phone / address
    AUTH_CREDENTIALS = "auth_credentials"  # password hashes / OAuth tokens
    BROKER_KEYS = "broker_keys"  # the encrypted secrets vault rows
    TRADING_HISTORY = "trading_history"  # order / fill / P&L rows
    HALAL_AUDIT = "halal_audit"  # screener decisions + receipts
    PURIFICATION_LEDGER = "purification_ledger"  # required by shariah audit
    LLM_PROMPTS = "llm_prompts"  # what the user asked the LLM
    LLM_RESPONSES = "llm_responses"  # what the LLM said back
    USAGE_TELEMETRY = "usage_telemetry"  # quota counters / latency rows
    MARKETING_ANALYTICS = "marketing_analytics"  # how user finds the bot
    DEVICE_FINGERPRINT = "device_fingerprint"  # IP + user-agent for fraud
    SUPPORT_TICKETS = "support_tickets"  # customer support history


class LegalBasis(str, Enum):
    """GDPR Article 6 lawful-basis values.

    Pinned string values for audit-log stability. The
    `LEGAL_OBLIGATION` basis is the load-bearing one for the
    "cannot be deleted" rule — most jurisdictions require
    financial-trade records to be retained for 5-7 years for tax
    / AML / regulatory audit, regardless of user request.
    """

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGITIMATE_INTEREST = "legitimate_interest"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTEREST = "vital_interest"
    PUBLIC_TASK = "public_task"


class DeletionAction(str, Enum):
    """Outcome of a deletion request for one category.

    `HARD_DELETE` removes the rows; `ANONYMISE` redacts PII fields
    while preserving the financial / shariah audit trail; `DENIED`
    means the request can't be fulfilled (legal obligation +
    operator-policy that doesn't permit anonymisation as a fallback).
    """

    HARD_DELETE = "hard_delete"
    ANONYMISE = "anonymise"
    DENIED = "denied"


# Default mapping of category → legal basis. Operators override for
# their specific jurisdiction. The defaults pin the conservative
# read of GDPR + CCPA: anything required for a contractual or legal
# obligation is locked; everything else falls under consent.
_DEFAULT_LEGAL_BASIS: dict[DataCategory, LegalBasis] = {
    DataCategory.PII: LegalBasis.CONTRACT,
    DataCategory.AUTH_CREDENTIALS: LegalBasis.CONTRACT,
    DataCategory.BROKER_KEYS: LegalBasis.CONTRACT,
    DataCategory.TRADING_HISTORY: LegalBasis.LEGAL_OBLIGATION,
    DataCategory.HALAL_AUDIT: LegalBasis.LEGAL_OBLIGATION,
    DataCategory.PURIFICATION_LEDGER: LegalBasis.LEGAL_OBLIGATION,
    DataCategory.LLM_PROMPTS: LegalBasis.CONSENT,
    DataCategory.LLM_RESPONSES: LegalBasis.CONSENT,
    DataCategory.USAGE_TELEMETRY: LegalBasis.LEGITIMATE_INTEREST,
    DataCategory.MARKETING_ANALYTICS: LegalBasis.CONSENT,
    DataCategory.DEVICE_FINGERPRINT: LegalBasis.LEGITIMATE_INTEREST,
    DataCategory.SUPPORT_TICKETS: LegalBasis.CONTRACT,
}

# Default retention. Note the asymmetry: TRADING_HISTORY is held
# for 7 years (FINRA / SEC), HALAL_AUDIT 7 years (shariah audit
# cycles), MARKETING_ANALYTICS 90 days (the legal minimum for
# functional adversion), USAGE_TELEMETRY 30 days (long enough for
# quota debugging, short enough to avoid building a profile).
_DEFAULT_RETENTION_DAYS: dict[DataCategory, int] = {
    DataCategory.PII: 365 * 7,
    DataCategory.AUTH_CREDENTIALS: 365 * 7,
    DataCategory.BROKER_KEYS: 365 * 7,
    DataCategory.TRADING_HISTORY: 365 * 7,
    DataCategory.HALAL_AUDIT: 365 * 7,
    DataCategory.PURIFICATION_LEDGER: 365 * 7,
    DataCategory.LLM_PROMPTS: 365,
    DataCategory.LLM_RESPONSES: 365,
    DataCategory.USAGE_TELEMETRY: 30,
    DataCategory.MARKETING_ANALYTICS: 90,
    DataCategory.DEVICE_FINGERPRINT: 90,
    DataCategory.SUPPORT_TICKETS: 365 * 3,
}


@dataclass(frozen=True)
class RetentionPolicy:
    """Per-category retention + legal-basis registry.

    The operator's deployment configures one `RetentionPolicy` at
    startup; the engine reads it for every export / deletion
    request. Default factories produce the conservative defaults
    above; operators override per-category for their jurisdiction.
    """

    retention_days: dict[DataCategory, int] = field(
        default_factory=lambda: dict(_DEFAULT_RETENTION_DAYS)
    )
    legal_basis: dict[DataCategory, LegalBasis] = field(
        default_factory=lambda: dict(_DEFAULT_LEGAL_BASIS)
    )

    def __post_init__(self) -> None:
        for cat in DataCategory:
            if cat not in self.retention_days:
                raise ValueError(f"retention_days missing category {cat.value!r}")
            if cat not in self.legal_basis:
                raise ValueError(f"legal_basis missing category {cat.value!r}")
            if self.retention_days[cat] <= 0:
                raise ValueError(f"retention_days for {cat.value!r} must be positive")

    def is_overdue(self, category: DataCategory, *, recorded_at: datetime, now: datetime) -> bool:
        """True if the row is past its retention horizon."""

        if recorded_at.tzinfo is None or now.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")
        return now - recorded_at >= timedelta(days=self.retention_days[category])


def default_retention_policy() -> RetentionPolicy:
    """Construct a fresh policy with conservative defaults.

    Operators should call this once at startup and persist the
    result; bumping a retention later is fine but lowering it
    requires re-running the deletion sweep against the new
    threshold.
    """

    return RetentionPolicy()


@dataclass(frozen=True)
class CategoryHolding:
    """How much data the user has in one category.

    Returned by the persistence layer; the engine reads `count` to
    decide whether to include the category in an export plan.
    `oldest_recorded_at` lets the engine surface "your oldest
    {category} row dates from {date}" in the user-facing receipt.
    """

    category: DataCategory
    count: int
    oldest_recorded_at: datetime | None = None
    newest_recorded_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")


@dataclass(frozen=True)
class ConsentRecord:
    """One user's consent state for one category.

    Pinned: `revoked_at` distinguishes "user never gave consent"
    (no record) from "user gave consent then withdrew" (record
    present, granted_at + revoked_at both set). The engine treats
    a revoked record as "no consent" for legal-basis purposes —
    consent is revocable.
    """

    user_id: str
    category: DataCategory
    granted_at: datetime
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.granted_at.tzinfo is None:
            raise ValueError("granted_at must be timezone-aware")
        if self.revoked_at is not None:
            if self.revoked_at.tzinfo is None:
                raise ValueError("revoked_at must be timezone-aware when set")
            if self.revoked_at < self.granted_at:
                raise ValueError("revoked_at cannot be before granted_at")

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


@dataclass(frozen=True)
class CategoryExportEntry:
    """One row in an export plan."""

    category: DataCategory
    legal_basis: LegalBasis
    count: int
    retention_days: int
    oldest_recorded_at: datetime | None
    newest_recorded_at: datetime | None


@dataclass(frozen=True)
class ExportPlan:
    """The result of an Article 15 (Right of Access) request.

    The engine produces the plan; the persistence layer iterates
    through `entries` and ships the actual rows. Pin: the plan
    contains *every* category the user has data in, never a
    partial subset.
    """

    user_id: str
    requested_at: datetime
    entries: tuple[CategoryExportEntry, ...]

    @property
    def total_count(self) -> int:
        return sum(e.count for e in self.entries)


@dataclass(frozen=True)
class CategoryDeletionEntry:
    """One row in a deletion plan."""

    category: DataCategory
    legal_basis: LegalBasis
    action: DeletionAction
    count: int
    reason: str = ""


@dataclass(frozen=True)
class DeletionPlan:
    """The result of an Article 17 (Right to Erasure) request.

    Each category gets its own action: HARD_DELETE for consent-
    or contract-basis with the user actively requesting deletion;
    ANONYMISE for legal-obligation rows where the financial /
    shariah audit trail must persist but PII fields can be
    redacted; DENIED only for legal-obligation when the
    operator's policy doesn't permit anonymisation.
    """

    user_id: str
    requested_at: datetime
    entries: tuple[CategoryDeletionEntry, ...]

    @property
    def deleted_count(self) -> int:
        return sum(e.count for e in self.entries if e.action is DeletionAction.HARD_DELETE)

    @property
    def anonymised_count(self) -> int:
        return sum(e.count for e in self.entries if e.action is DeletionAction.ANONYMISE)

    @property
    def denied_count(self) -> int:
        return sum(e.count for e in self.entries if e.action is DeletionAction.DENIED)


def build_export_plan(
    *,
    user_id: str,
    holdings: tuple[CategoryHolding, ...],
    policy: RetentionPolicy,
    requested_at: datetime,
) -> ExportPlan:
    """Build an Article 15 export plan covering every populated category.

    Pin: every category with `count > 0` appears in the plan. A
    category with `count == 0` is excluded — the user has no data
    there to export. The plan is sorted by category enum order for
    deterministic output.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if requested_at.tzinfo is None:
        raise ValueError("requested_at must be timezone-aware")

    entries: list[CategoryExportEntry] = []
    by_cat = {h.category: h for h in holdings}
    for cat in DataCategory:
        h = by_cat.get(cat)
        if h is None or h.count == 0:
            continue
        entries.append(
            CategoryExportEntry(
                category=cat,
                legal_basis=policy.legal_basis[cat],
                count=h.count,
                retention_days=policy.retention_days[cat],
                oldest_recorded_at=h.oldest_recorded_at,
                newest_recorded_at=h.newest_recorded_at,
            )
        )
    return ExportPlan(user_id=user_id, requested_at=requested_at, entries=tuple(entries))


def _decide_deletion(
    *,
    category: DataCategory,
    policy: RetentionPolicy,
    holding: CategoryHolding,
    consent_active: bool,
    operator_allows_anonymise: bool,
) -> CategoryDeletionEntry:
    """Decide the action for one category given the legal basis."""

    basis = policy.legal_basis[category]
    if basis is LegalBasis.LEGAL_OBLIGATION:
        if operator_allows_anonymise:
            return CategoryDeletionEntry(
                category=category,
                legal_basis=basis,
                action=DeletionAction.ANONYMISE,
                count=holding.count,
                reason=(
                    f"{category.value} held under legal obligation "
                    "(audit / regulatory retention); PII redacted, "
                    "audit trail preserved"
                ),
            )
        return CategoryDeletionEntry(
            category=category,
            legal_basis=basis,
            action=DeletionAction.DENIED,
            count=holding.count,
            reason=(
                f"{category.value} held under legal obligation; "
                "operator policy does not permit anonymisation as fallback"
            ),
        )
    if basis is LegalBasis.CONTRACT:
        # Contract-basis data (e.g., broker keys) is held while the
        # contract is active; the user closing their account is the
        # contract terminating, so contract-basis rows hard-delete.
        return CategoryDeletionEntry(
            category=category,
            legal_basis=basis,
            action=DeletionAction.HARD_DELETE,
            count=holding.count,
            reason=(
                f"{category.value} held under contract; contract terminating with deletion request"
            ),
        )
    if basis is LegalBasis.CONSENT:
        # Consent-basis: deletable on request, full stop.
        return CategoryDeletionEntry(
            category=category,
            legal_basis=basis,
            action=DeletionAction.HARD_DELETE,
            count=holding.count,
            reason=(
                f"{category.value} held under consent; "
                + ("user revoking consent" if consent_active else "no active consent")
            ),
        )
    if basis is LegalBasis.LEGITIMATE_INTEREST:
        # Legitimate interest is overridden by an explicit Article 17
        # request — the user's right to erasure outweighs the
        # legitimate-interest balancing test (per ICO guidance).
        return CategoryDeletionEntry(
            category=category,
            legal_basis=basis,
            action=DeletionAction.HARD_DELETE,
            count=holding.count,
            reason=(
                f"{category.value} held under legitimate interest; "
                "user's Article 17 right overrides"
            ),
        )
    # VITAL_INTEREST and PUBLIC_TASK shouldn't apply to a trading
    # bot; if they ever do, deny by default (operator must
    # explicitly opt-in).
    return CategoryDeletionEntry(
        category=category,
        legal_basis=basis,
        action=DeletionAction.DENIED,
        count=holding.count,
        reason=(
            f"{category.value} held under {basis.value}; operator must override to permit deletion"
        ),
    )


def build_deletion_plan(
    *,
    user_id: str,
    holdings: tuple[CategoryHolding, ...],
    policy: RetentionPolicy,
    requested_at: datetime,
    consents: tuple[ConsentRecord, ...] = (),
    operator_allows_anonymise: bool = True,
) -> DeletionPlan:
    """Build an Article 17 deletion plan for a user's data.

    The engine evaluates each category's legal basis and emits a
    `CategoryDeletionEntry` with the resulting action. Categories
    with `count == 0` are omitted (nothing to delete).
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if requested_at.tzinfo is None:
        raise ValueError("requested_at must be timezone-aware")

    consent_by_cat: dict[DataCategory, ConsentRecord] = {}
    for c in consents:
        if c.user_id != user_id:
            continue
        # The most recent consent record per category wins.
        prior = consent_by_cat.get(c.category)
        if prior is None or c.granted_at >= prior.granted_at:
            consent_by_cat[c.category] = c

    entries: list[CategoryDeletionEntry] = []
    by_cat = {h.category: h for h in holdings}
    for cat in DataCategory:
        h = by_cat.get(cat)
        if h is None or h.count == 0:
            continue
        consent = consent_by_cat.get(cat)
        consent_active = consent.is_active if consent is not None else False
        entries.append(
            _decide_deletion(
                category=cat,
                policy=policy,
                holding=h,
                consent_active=consent_active,
                operator_allows_anonymise=operator_allows_anonymise,
            )
        )
    return DeletionPlan(user_id=user_id, requested_at=requested_at, entries=tuple(entries))


def revoke_consent(consent: ConsentRecord, *, now: datetime) -> ConsentRecord:
    """Return a new ConsentRecord with `revoked_at = now`.

    Pinned: re-revoking an already-revoked consent is a no-op
    (returns the input unchanged) — operators retrying the
    revoke endpoint shouldn't keep advancing the revoked-at
    timestamp.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if consent.revoked_at is not None:
        return consent
    return ConsentRecord(
        user_id=consent.user_id,
        category=consent.category,
        granted_at=consent.granted_at,
        revoked_at=now,
    )


_ACTION_EMOJI: dict[DeletionAction, str] = {
    DeletionAction.HARD_DELETE: "🗑️",
    DeletionAction.ANONYMISE: "🫥",
    DeletionAction.DENIED: "🔒",
}


def render_export_plan(plan: ExportPlan) -> str:
    """Format an export plan for the user-facing receipt.

    Pinned no-PII contract: the receipt summarises action +
    category + count, never the field values themselves.
    """

    lines: list[str] = [
        f"📦 Export plan for {plan.user_id}",
        f"  requested: {plan.requested_at.isoformat()}",
        f"  total rows: {plan.total_count}",
    ]
    if not plan.entries:
        lines.append("  (no data on file)")
        return "\n".join(lines)
    for e in plan.entries:
        lines.append(
            f"  · {e.category.value} ({e.legal_basis.value}): "
            f"{e.count} rows, retained {e.retention_days}d"
        )
    return "\n".join(lines)


def render_deletion_plan(plan: DeletionPlan) -> str:
    """Format a deletion plan for the user-facing receipt."""

    lines: list[str] = [
        f"🗑 Deletion plan for {plan.user_id}",
        f"  requested: {plan.requested_at.isoformat()}",
        f"  hard-deleted: {plan.deleted_count}",
        f"  anonymised: {plan.anonymised_count}",
        f"  denied: {plan.denied_count}",
    ]
    if not plan.entries:
        lines.append("  (no data on file)")
        return "\n".join(lines)
    for e in plan.entries:
        emoji = _ACTION_EMOJI[e.action]
        lines.append(
            f"  {emoji} {e.category.value} ({e.legal_basis.value}): "
            f"{e.count} rows — {e.action.value.upper()}"
        )
        if e.reason:
            lines.append(f"      {e.reason}")
    return "\n".join(lines)


__all__ = [
    "CategoryDeletionEntry",
    "CategoryExportEntry",
    "CategoryHolding",
    "ConsentRecord",
    "DataCategory",
    "DeletionAction",
    "DeletionPlan",
    "ExportPlan",
    "LegalBasis",
    "RetentionPolicy",
    "build_deletion_plan",
    "build_export_plan",
    "default_retention_policy",
    "render_deletion_plan",
    "render_export_plan",
    "revoke_consent",
]


# Re-export for convenience: operators commonly want to compute
# "is this row past its retention horizon?" without instantiating a
# full RetentionPolicy.
def is_overdue_default(category: DataCategory, *, recorded_at: datetime, now: datetime) -> bool:
    """Convenience wrapper that uses the default retention policy."""

    return default_retention_policy().is_overdue(category, recorded_at=recorded_at, now=now)


__all__.append("is_overdue_default")
