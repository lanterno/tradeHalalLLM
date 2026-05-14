"""KYC / AML state engine.

Most jurisdictions require financial platforms to verify user
identity (Know Your Customer) and screen against sanctions /
politically-exposed-persons lists (Anti-Money-Laundering) before
permitting real-money activity. The roadmap defers the live
vendor integration (Persona / Sumsub / Onfido / Trulioo) because
each charges per-verification, but the state machine + activity-
gating logic is operator-supplied pure-Python — exactly the
isolated-module pattern of Wave 1.G commodities, 1.I REIT, 2.G
regulator-index, 1.H sukuk, and 11.D privacy.

Picked a focused state engine over wedging the rules into the
authn layer because the same user can hold different KYC tiers
for different purposes (an email-verified user can paper-trade;
identity-verified can deposit real money; enhanced-due-diligence
is required for high-risk profiles), and the gate decisions need
to be deterministic + jurisdiction-aware + auditable.

Pinned semantics:
- **Default = paper trading only.** A `UserKYCState` with no
  verification permits SIGNUP + DEMO + PAPER_TRADING but rejects
  every real-money activity. The conservative default protects
  the bot from "I forgot to wire up the gate" failures: if the
  call site forgets to check `permits(...)`, the only operations
  available are the safe ones.
- **Real-money trading requires IDENTITY_VERIFIED minimum.**
  REAL_MONEY_DEPOSIT / TRADING / WITHDRAW all require at least
  identity verification. Higher-risk jurisdictions (EU, UAE)
  layer ADDRESS_VERIFIED on top via the `JurisdictionRequirement`
  override.
- **Sanctions MATCH blocks everything.** A confirmed sanctions
  hit blocks every activity except SIGNUP (so the user can be
  notified of the rejection); the operator's compliance team
  reviews and either confirms (permanent block) or marks
  FALSE_POSITIVE (gate re-opens). Pinned via test that no
  override path lets a MATCH user trade.
- **Expired KYC (> 1 year by default) blocks new real-money
  activity** but still permits the user to withdraw their
  existing balance — pinned because trapping a user's funds
  during their KYC re-verification is operationally awful and
  legally questionable. WITHDRAW is allowed under expired KYC,
  but DEPOSIT and new trades are not.
- **HIGH risk score requires ENHANCED_DUE_DILIGENCE.** The risk
  score is operator-supplied (could be vendor-calculated, could
  be model-calculated, could be hand-set by compliance ops);
  HIGH overrides the default tier ladder and requires the
  enhanced check before any real-money activity.
- **Render output never includes ID document data.** The user-
  facing receipt summarises level + status + last-verified date,
  never the document number, photo, or address fields. Mirrors
  the no-PII pattern of Wave 11.D privacy + Wave 8.D OTLP
  translator + Wave 3.B vault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class KYCLevel(str, Enum):
    """Verification tier the user has completed.

    Each level is a strict superset of the previous: IDENTITY_
    VERIFIED implies EMAIL_VERIFIED; ADDRESS_VERIFIED implies
    IDENTITY; ENHANCED implies ADDRESS. The numeric ordering on
    the `int_value` property is what gates check against
    `JurisdictionRequirement.minimum_level`.
    """

    NONE = "none"
    EMAIL_VERIFIED = "email_verified"
    IDENTITY_VERIFIED = "identity_verified"
    ADDRESS_VERIFIED = "address_verified"
    ENHANCED_DUE_DILIGENCE = "enhanced_due_diligence"

    @property
    def int_value(self) -> int:
        return _LEVEL_ORDER[self]


_LEVEL_ORDER: dict[KYCLevel, int] = {
    KYCLevel.NONE: 0,
    KYCLevel.EMAIL_VERIFIED: 1,
    KYCLevel.IDENTITY_VERIFIED: 2,
    KYCLevel.ADDRESS_VERIFIED: 3,
    KYCLevel.ENHANCED_DUE_DILIGENCE: 4,
}


class KYCStatus(str, Enum):
    """Verification workflow status.

    `VERIFIED` is the only status that grants access; all others
    block. `EXPIRED` is the rolling-renewal status — the user was
    verified once but the verification has aged past the renewal
    horizon. `REJECTED` is a hard fail (e.g., document didn't
    match). `UNDER_REVIEW` is the operator-compliance-side hold.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    EXPIRED = "expired"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class RiskLevel(str, Enum):
    """AML risk score tier.

    `HIGH` requires ENHANCED_DUE_DILIGENCE before any real-money
    activity per FATF guidance. `LOW` and `MEDIUM` follow the
    standard tier ladder.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SanctionsOutcome(str, Enum):
    """Result of the sanctions / PEP screening.

    `MATCH` blocks all activity except SIGNUP pending operator
    review. `FALSE_POSITIVE` is set by compliance ops after
    investigation — the engine treats it as CLEAR for gating.
    """

    CLEAR = "clear"
    MATCH = "match"
    FALSE_POSITIVE = "false_positive"


class Activity(str, Enum):
    """User-initiated actions the engine gates."""

    SIGNUP = "signup"
    DEMO_TRADING = "demo_trading"
    PAPER_TRADING = "paper_trading"
    REAL_MONEY_DEPOSIT = "real_money_deposit"
    REAL_MONEY_TRADING = "real_money_trading"
    WITHDRAW = "withdraw"


# Activities that don't require any KYC.
_KYC_FREE_ACTIVITIES: frozenset[Activity] = frozenset(
    {Activity.SIGNUP, Activity.DEMO_TRADING, Activity.PAPER_TRADING}
)

# Activities that require fresh (non-expired) KYC.
_REAL_MONEY_INFLOW_ACTIVITIES: frozenset[Activity] = frozenset(
    {Activity.REAL_MONEY_DEPOSIT, Activity.REAL_MONEY_TRADING}
)


@dataclass(frozen=True)
class JurisdictionRequirement:
    """Per-jurisdiction KYC level requirement.

    Operators register one entry per jurisdiction they support.
    The default `minimum_level` is IDENTITY_VERIFIED for real-
    money activity; stricter jurisdictions (EU under MiFID II,
    UAE under VARA) bump to ADDRESS_VERIFIED.
    """

    jurisdiction: str
    minimum_level_for_real_money: KYCLevel = KYCLevel.IDENTITY_VERIFIED

    def __post_init__(self) -> None:
        if not self.jurisdiction or not self.jurisdiction.strip():
            raise ValueError("jurisdiction must be non-empty")


# Default jurisdictions the bot supports; operators extend.
_DEFAULT_JURISDICTIONS: dict[str, JurisdictionRequirement] = {
    "US": JurisdictionRequirement(
        jurisdiction="US", minimum_level_for_real_money=KYCLevel.IDENTITY_VERIFIED
    ),
    "GB": JurisdictionRequirement(
        jurisdiction="GB", minimum_level_for_real_money=KYCLevel.ADDRESS_VERIFIED
    ),
    "EU": JurisdictionRequirement(
        jurisdiction="EU", minimum_level_for_real_money=KYCLevel.ADDRESS_VERIFIED
    ),
    "AE": JurisdictionRequirement(
        jurisdiction="AE", minimum_level_for_real_money=KYCLevel.ADDRESS_VERIFIED
    ),
    "SA": JurisdictionRequirement(
        jurisdiction="SA", minimum_level_for_real_money=KYCLevel.ADDRESS_VERIFIED
    ),
    "PK": JurisdictionRequirement(
        jurisdiction="PK", minimum_level_for_real_money=KYCLevel.IDENTITY_VERIFIED
    ),
    "MY": JurisdictionRequirement(
        jurisdiction="MY", minimum_level_for_real_money=KYCLevel.IDENTITY_VERIFIED
    ),
}


@dataclass(frozen=True)
class KYCPolicy:
    """Operator-tunable policy.

    `expiry_days` is the rolling renewal horizon for verified
    KYC; defaults to 365d (the FATF guidance cap). Operators in
    stricter jurisdictions (UAE) drop to 180d.
    """

    expiry_days: int = 365
    jurisdictions: dict[str, JurisdictionRequirement] = field(
        default_factory=lambda: dict(_DEFAULT_JURISDICTIONS)
    )

    def __post_init__(self) -> None:
        if self.expiry_days <= 0:
            raise ValueError("expiry_days must be positive")


def default_policy() -> KYCPolicy:
    return KYCPolicy()


@dataclass(frozen=True)
class UserKYCState:
    """One user's current KYC + AML state.

    `verified_at` is None until first verification completes; on
    re-verification the operator persists the new timestamp. The
    state is otherwise deterministic from the inputs — the engine
    is stateless, the caller persists the state row.
    """

    user_id: str
    jurisdiction: str
    level: KYCLevel
    status: KYCStatus
    risk_level: RiskLevel
    sanctions_outcome: SanctionsOutcome
    verified_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.jurisdiction or not self.jurisdiction.strip():
            raise ValueError("jurisdiction must be non-empty")
        if self.verified_at is not None and self.verified_at.tzinfo is None:
            raise ValueError("verified_at must be timezone-aware when set")


@dataclass(frozen=True)
class GateDecision:
    """Outcome of a `permits()` check.

    `allowed=True` means the activity proceeds. `reason` carries
    the audit-log explanation either way; an allowed decision's
    reason might be "user has identity_verified, jurisdiction US
    requires identity_verified, sanctions clear" — useful for
    after-the-fact compliance review.
    """

    user_id: str
    activity: Activity
    allowed: bool
    reason: str
    required_level: KYCLevel | None = None
    actual_level: KYCLevel | None = None


def is_expired(state: UserKYCState, *, now: datetime, policy: KYCPolicy) -> bool:
    """True if the verification is past the rolling renewal horizon."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if state.verified_at is None:
        return False  # never verified → not "expired", just not started
    return now - state.verified_at >= timedelta(days=policy.expiry_days)


def permits(
    state: UserKYCState,
    *,
    activity: Activity,
    now: datetime,
    policy: KYCPolicy,
) -> GateDecision:
    """Decide whether the user can perform the activity.

    The function is pure: it does no I/O. Callers persist the
    `UserKYCState` and pass it in; the engine returns a deterministic
    GateDecision the call site uses to allow / reject the action.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    # Sanctions MATCH is the most-restrictive gate — block everything
    # except SIGNUP. SIGNUP must remain available so the operator can
    # show the user the rejection reason; every other activity is
    # blocked pending operator review.
    if state.sanctions_outcome is SanctionsOutcome.MATCH:
        if activity is Activity.SIGNUP:
            return GateDecision(
                user_id=state.user_id,
                activity=activity,
                allowed=True,
                reason="signup permitted to surface sanctions-review notification",
            )
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason=(
                "sanctions screening MATCH: all activities blocked "
                "pending operator compliance review"
            ),
        )

    # KYC-free activities (signup / demo / paper) are permitted
    # regardless of KYC level — the bot wants users to be able to
    # explore before going through verification.
    if activity in _KYC_FREE_ACTIVITIES:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=True,
            reason=f"{activity.value} does not require KYC",
        )

    # Real-money activities — check status, expiry, level,
    # jurisdiction, risk.
    if state.status is KYCStatus.REJECTED:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason="KYC verification was REJECTED",
        )
    if state.status is KYCStatus.UNDER_REVIEW:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason="KYC verification UNDER_REVIEW pending operator decision",
        )
    if state.status is KYCStatus.NOT_STARTED:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason="KYC NOT_STARTED — user must complete verification first",
        )
    if state.status is KYCStatus.IN_PROGRESS:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason="KYC IN_PROGRESS — verification not yet complete",
        )

    # Status is VERIFIED or EXPIRED below this line.
    expired = is_expired(state, now=now, policy=policy) or state.status is KYCStatus.EXPIRED

    # WITHDRAW under expired KYC is permitted — the engine
    # explicitly allows the user to retrieve their existing balance
    # rather than trapping their funds during re-verification.
    if expired and activity is Activity.WITHDRAW:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=True,
            reason=(
                "KYC expired but WITHDRAW permitted to avoid trapping user funds; "
                "user must re-verify before further deposits or trades"
            ),
        )

    if expired:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason=(
                f"KYC expired (>{policy.expiry_days}d since verified_at); "
                "user must re-verify before {activity}"
            ).format(activity=activity.value),
        )

    # HIGH risk requires ENHANCED_DUE_DILIGENCE for real-money inflow.
    if (
        state.risk_level is RiskLevel.HIGH
        and activity in _REAL_MONEY_INFLOW_ACTIVITIES
        and state.level is not KYCLevel.ENHANCED_DUE_DILIGENCE
    ):
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason=("HIGH risk score requires ENHANCED_DUE_DILIGENCE before real-money inflow"),
            required_level=KYCLevel.ENHANCED_DUE_DILIGENCE,
            actual_level=state.level,
        )

    # Jurisdiction-specific minimum level for real-money inflow.
    requirement = policy.jurisdictions.get(state.jurisdiction)
    if requirement is None:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason=(
                f"jurisdiction {state.jurisdiction!r} not registered in policy; "
                "operator must add a JurisdictionRequirement to permit real-money activity"
            ),
        )

    minimum = requirement.minimum_level_for_real_money
    # Withdrawal under verified+non-expired KYC always permitted
    # (operator-friendly: don't trap funds; the deposit gate already
    # restricted who could put money in).
    if activity is Activity.WITHDRAW:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=True,
            reason="WITHDRAW permitted under verified KYC",
            required_level=minimum,
            actual_level=state.level,
        )

    if state.level.int_value < minimum.int_value:
        return GateDecision(
            user_id=state.user_id,
            activity=activity,
            allowed=False,
            reason=(
                f"level {state.level.value} below jurisdiction "
                f"{state.jurisdiction} minimum {minimum.value} for {activity.value}"
            ),
            required_level=minimum,
            actual_level=state.level,
        )

    return GateDecision(
        user_id=state.user_id,
        activity=activity,
        allowed=True,
        reason=(
            f"level {state.level.value} ≥ jurisdiction "
            f"{state.jurisdiction} minimum {minimum.value}; sanctions clear"
        ),
        required_level=minimum,
        actual_level=state.level,
    )


_STATUS_EMOJI: dict[KYCStatus, str] = {
    KYCStatus.NOT_STARTED: "⚪",
    KYCStatus.IN_PROGRESS: "🟡",
    KYCStatus.VERIFIED: "✅",
    KYCStatus.EXPIRED: "⏰",
    KYCStatus.REJECTED: "❌",
    KYCStatus.UNDER_REVIEW: "🔍",
}


def render_user_state(state: UserKYCState) -> str:
    """Render-safe user-facing summary.

    Pinned no-PII contract: never includes the underlying ID
    document number / photo / address fields. Operators audit
    via the underlying database directly when full data is
    needed; the engine's render is for the user's own dashboard
    and Slack / Telegram audit channels.
    """

    emoji = _STATUS_EMOJI[state.status]
    lines = [
        f"{emoji} {state.user_id} — {state.level.value}",
        f"  status: {state.status.value}",
        f"  jurisdiction: {state.jurisdiction}",
        f"  risk_level: {state.risk_level.value}",
        f"  sanctions: {state.sanctions_outcome.value}",
    ]
    if state.verified_at is not None:
        lines.append(f"  verified_at: {state.verified_at.isoformat()}")
    else:
        lines.append("  verified_at: never")
    return "\n".join(lines)


def render_decision(decision: GateDecision) -> str:
    """Render the gate decision for audit logs."""

    emoji = "✅" if decision.allowed else "🚫"
    line = (
        f"{emoji} {decision.user_id} {decision.activity.value} "
        f"— {'ALLOWED' if decision.allowed else 'BLOCKED'}"
    )
    if decision.actual_level is not None and decision.required_level is not None:
        line += (
            f" (level {decision.actual_level.value} vs required {decision.required_level.value})"
        )
    return f"{line}\n  reason: {decision.reason}"


__all__ = [
    "Activity",
    "GateDecision",
    "JurisdictionRequirement",
    "KYCLevel",
    "KYCPolicy",
    "KYCStatus",
    "RiskLevel",
    "SanctionsOutcome",
    "UserKYCState",
    "default_policy",
    "is_expired",
    "permits",
    "render_decision",
    "render_user_state",
]
