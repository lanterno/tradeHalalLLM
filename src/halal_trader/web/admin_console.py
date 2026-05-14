"""Admin console view-model.

The roadmap pins Wave 3.G: "Hosted-tenant admin: see all users,
their LLM spend, their portfolio P&L, halt-switch any user's
bot, billing override. Restricted to configured admin emails."
This module is the **pure-Python aggregator + admin-action audit
layer** that composes the multi-tenant primitives landed in earlier
waves (3.A auth, 3.C quotas, 3.F billing) into a single view-model
the admin route consumes.

Picked a focused view-model over a "hand-roll admin route logic"
approach because (a) the admin-email gate is a single load-bearing
authorization check that should not be re-derived at every route
(if the admin list logic drifts between routes, an admin de-listed
in one place but still trusted in another is an audit-failure
class), (b) every admin action against a tenant's bot must be
attributed + reasoned + timestamped — the audit row needs to
survive a future "why did the bot halt at 14:22?" question against
a stable schema, (c) the aggregation totals (active users, total
LLM spend, halt count) are read-only summaries that the admin
console renders the same way the operator email summary renders
its end-of-day report — pure functions of the input snapshot.

Pinned semantics:
- **Admin email match is case-insensitive after whitespace strip.**
  Operators add admin emails through env config; mixed-case + extra
  spaces are normal data-entry slop, and rejecting `Admin@Foo.com`
  while accepting `admin@foo.com` would be an opaque misconfiguration.
- **Non-admin email → AdminAuthorizationError.** Every admin action
  passes through `audit_admin_action` which checks the admin list
  before producing the audit row; a non-admin caller never gets to
  the action's side-effect logic.
- **Reason required for destructive actions.** HALT_USER,
  OVERRIDE_TIER, SUSPEND_USER, REVOKE_SESSION require non-empty
  rationale; RESUME_USER + a future read-only inspection action
  don't. Pinned to keep the audit trail meaningful — "halted" with
  no reason is a mystery to the next on-call.
- **Render output never includes session tokens, Stripe customer IDs,
  API keys, password hashes, or scrypt salts.** Mirrors the
  no-secret patterns of Wave 3.B vault + Wave 8.D OTLP + Wave 12.G
  co-pilot + Wave 3.F billing.
- **Aggregation is deterministic.** Same input list + same `now` →
  same `AdminView`. The admin route can render the same view twice
  and not get inconsistent counts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from halal_trader.web.billing_state import SubscriptionStatus
from halal_trader.web.quotas import Tier


class AdminAuthorizationError(Exception):
    """Raised when a non-admin email attempts an admin action."""

    def __init__(self, email: str) -> None:
        super().__init__(f"email {email!r} is not on the admin list")
        self.email = email


class AdminAction(str, Enum):
    """Admin actions the console can authorise.

    Pinned string values for JSON / DB stability. The audit row
    keys on these literals; the admin route maps HTTP verbs to
    these.
    """

    HALT_USER = "halt_user"
    RESUME_USER = "resume_user"
    OVERRIDE_TIER = "override_tier"
    REVOKE_SESSION = "revoke_session"
    SUSPEND_USER = "suspend_user"
    INSPECT_USER = "inspect_user"


_DESTRUCTIVE_ACTIONS: frozenset[AdminAction] = frozenset(
    {
        AdminAction.HALT_USER,
        AdminAction.OVERRIDE_TIER,
        AdminAction.REVOKE_SESSION,
        AdminAction.SUSPEND_USER,
    }
)


def _normalise_email(email: str) -> str:
    """Lowercase + whitespace-strip an email for comparison."""

    return email.strip().lower()


@dataclass(frozen=True)
class AdminEmailList:
    """The configured set of admin emails.

    Operators populate this from env / config at boot; the list is
    immutable after construction so a runtime mutation can't silently
    elevate a user. Email match is case-insensitive after whitespace
    strip.
    """

    emails: frozenset[str]

    def __post_init__(self) -> None:
        if not self.emails:
            raise ValueError("admin email list must be non-empty")
        for email in self.emails:
            if not email or not email.strip():
                raise ValueError("admin email must be non-empty")
            if "@" not in email:
                raise ValueError(f"admin email {email!r} missing @")
        # Re-freeze to the normalised form so contains-checks below
        # don't re-normalise on every call.
        normalised = frozenset(_normalise_email(e) for e in self.emails)
        object.__setattr__(self, "emails", normalised)

    def is_admin(self, email: str) -> bool:
        """Case-insensitive admin-list membership check."""

        return _normalise_email(email) in self.emails


@dataclass(frozen=True)
class UserSummary:
    """One user's row in the admin view.

    Carries the cross-cutting facts the admin needs to triage:
    auth/billing state, today's LLM spend, halt status, last
    activity. Specifically does NOT carry: password hash, scrypt
    salt, session tokens, broker API keys, Stripe customer ID,
    invoice amounts. The no-secret-leak contract is pinned via
    test against the rendered output.
    """

    user_id: str
    email: str
    effective_tier: Tier
    subscription_status: SubscriptionStatus
    llm_usd_used_today: float
    llm_usd_limit_today: float
    halt_active: bool
    last_active_at: datetime | None
    joined_at: datetime

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.email or not self.email.strip():
            raise ValueError("email must be non-empty")
        if "@" not in self.email:
            raise ValueError(f"email {self.email!r} missing @")
        if self.llm_usd_used_today < 0:
            raise ValueError("llm_usd_used_today must be non-negative")
        if self.llm_usd_limit_today < 0:
            raise ValueError("llm_usd_limit_today must be non-negative")
        if self.joined_at.tzinfo is None:
            raise ValueError("joined_at must be timezone-aware")
        if self.last_active_at is not None and self.last_active_at.tzinfo is None:
            raise ValueError("last_active_at must be timezone-aware when set")

    @property
    def llm_usage_fraction(self) -> float:
        """LLM USD usage as a fraction of today's limit; 0.0 if no limit."""

        if self.llm_usd_limit_today <= 0:
            return 0.0
        return self.llm_usd_used_today / self.llm_usd_limit_today


@dataclass(frozen=True)
class AdminActionRequest:
    """An admin's request to act on a target user.

    `performed_by` is the admin's email (validated against the
    AdminEmailList at audit time); `reason` is required for
    destructive actions (HALT/OVERRIDE/SUSPEND/REVOKE) — pinned
    so a future "why was user X halted" question has an answer.
    `target_tier` is required for OVERRIDE_TIER.
    """

    action_id: str
    action: AdminAction
    target_user_id: str
    performed_by: str
    reason: str
    performed_at: datetime
    target_tier: Tier | None = None

    def __post_init__(self) -> None:
        if not self.action_id or not self.action_id.strip():
            raise ValueError("action_id must be non-empty")
        if not self.target_user_id or not self.target_user_id.strip():
            raise ValueError("target_user_id must be non-empty")
        if not self.performed_by or not self.performed_by.strip():
            raise ValueError("performed_by must be non-empty")
        if "@" not in self.performed_by:
            raise ValueError(f"performed_by {self.performed_by!r} missing @")
        if self.performed_at.tzinfo is None:
            raise ValueError("performed_at must be timezone-aware")
        if self.action in _DESTRUCTIVE_ACTIONS:
            if not self.reason or not self.reason.strip():
                raise ValueError(f"reason required for destructive action {self.action.value}")
        if self.action is AdminAction.OVERRIDE_TIER and self.target_tier is None:
            raise ValueError("OVERRIDE_TIER requires target_tier")


@dataclass(frozen=True)
class AdminActionRecord:
    """Immutable audit row produced by `audit_admin_action`."""

    action_id: str
    action: AdminAction
    target_user_id: str
    performed_by: str
    reason: str
    performed_at: datetime
    target_tier: Tier | None = None


def audit_admin_action(
    request: AdminActionRequest,
    *,
    admin_emails: AdminEmailList,
) -> AdminActionRecord:
    """Validate the admin and produce an audit-trail record.

    Raises `AdminAuthorizationError` if `request.performed_by` is
    not on the admin list. The caller's side-effect path (actually
    halting the user, overriding the tier, etc.) is downstream of
    this — if `audit_admin_action` raises, the side effect never
    happens.
    """

    if not admin_emails.is_admin(request.performed_by):
        raise AdminAuthorizationError(request.performed_by)

    return AdminActionRecord(
        action_id=request.action_id,
        action=request.action,
        target_user_id=request.target_user_id,
        performed_by=_normalise_email(request.performed_by),
        reason=request.reason,
        performed_at=request.performed_at,
        target_tier=request.target_tier,
    )


_ACTIVE_WINDOW_DEFAULT = timedelta(hours=24)


@dataclass(frozen=True)
class AdminView:
    """Aggregated admin console view-model.

    `users` is sorted by joined_at ascending (operators expect
    consistent ordering across renders); aggregate counters
    derived deterministically from the same input list.
    """

    generated_at: datetime
    users: tuple[UserSummary, ...]
    total_users: int
    active_users_24h: int
    halt_active_count: int
    total_llm_usd_today: float
    tier_breakdown: tuple[tuple[Tier, int], ...]

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")


def build_admin_view(
    users: Iterable[UserSummary],
    *,
    now: datetime,
    active_window: timedelta = _ACTIVE_WINDOW_DEFAULT,
) -> AdminView:
    """Aggregate a user iterable into an AdminView.

    Pure: deterministic for a given input + `now` + `active_window`.
    The `active_window` defaults to the same rolling 24-hour window
    Wave 3.C uses for quota accounting; pass a different window
    (e.g. `timedelta(days=7)`) for a weekly-active aggregate.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if active_window <= timedelta(0):
        raise ValueError("active_window must be positive")

    user_list = sorted(users, key=lambda u: u.joined_at)
    user_tuple = tuple(user_list)

    active_threshold = now - active_window
    active = sum(
        1
        for u in user_tuple
        if u.last_active_at is not None and u.last_active_at >= active_threshold
    )
    halted = sum(1 for u in user_tuple if u.halt_active)
    total_spend = sum(u.llm_usd_used_today for u in user_tuple)

    counts: dict[Tier, int] = {tier: 0 for tier in Tier}
    for user in user_tuple:
        counts[user.effective_tier] += 1
    breakdown = tuple((tier, counts[tier]) for tier in Tier)

    return AdminView(
        generated_at=now,
        users=user_tuple,
        total_users=len(user_tuple),
        active_users_24h=active,
        halt_active_count=halted,
        total_llm_usd_today=total_spend,
        tier_breakdown=breakdown,
    )


_TIER_EMOJI: dict[Tier, str] = {
    Tier.FREE: "🆓",
    Tier.PRO: "💼",
    Tier.ENTERPRISE: "🏢",
}


_STATUS_EMOJI: dict[SubscriptionStatus, str] = {
    SubscriptionStatus.TRIALING: "🆓",
    SubscriptionStatus.ACTIVE: "✅",
    SubscriptionStatus.PAST_DUE: "⏰",
    SubscriptionStatus.GRACE_PERIOD: "⚠️",
    SubscriptionStatus.CANCELLED: "🛑",
    SubscriptionStatus.EXPIRED: "❌",
}


def render_user_summary(user: UserSummary) -> str:
    """Format one user row for ops display.

    Pinned no-secret-leak: never includes session tokens / API
    keys / password hashes / Stripe customer IDs / invoice
    amounts. Shows user_id + email + tier emoji + status + LLM
    spend fraction + halt flag + last active. The render contract
    is regression-pinned.
    """

    tier_emoji = _TIER_EMOJI[user.effective_tier]
    status_emoji = _STATUS_EMOJI[user.subscription_status]
    halt_marker = " 🛑HALTED" if user.halt_active else ""
    pct = user.llm_usage_fraction * 100.0
    last_active = (
        user.last_active_at.date().isoformat() if user.last_active_at is not None else "never"
    )
    return (
        f"{tier_emoji}{status_emoji} {user.user_id} ({user.email}) — "
        f"{user.effective_tier.value} / {user.subscription_status.value} | "
        f"LLM ${user.llm_usd_used_today:.2f}/${user.llm_usd_limit_today:.2f} "
        f"({pct:.0f}%) | active: {last_active}{halt_marker}"
    )


def render_admin_view(view: AdminView) -> str:
    """Format the full admin view-model for ops display.

    Header (totals + tier breakdown) + per-user rows. No-secret-leak
    pinned across the full render.
    """

    lines = [
        f"Admin view @ {view.generated_at.isoformat()}",
        f"  total: {view.total_users}, "
        f"active 24h: {view.active_users_24h}, "
        f"halted: {view.halt_active_count}",
        f"  LLM today: ${view.total_llm_usd_today:.2f}",
    ]
    if view.tier_breakdown:
        breakdown_str = ", ".join(
            f"{_TIER_EMOJI[tier]}{tier.value}={count}" for tier, count in view.tier_breakdown
        )
        lines.append(f"  tiers: {breakdown_str}")
    lines.append("")
    for user in view.users:
        lines.append(render_user_summary(user))
    return "\n".join(lines)


def render_action_record(record: AdminActionRecord) -> str:
    """Format an admin action record for ops display.

    Pinned: shows action / target / performed_by / reason / time.
    Never includes secrets even when the action is REVOKE_SESSION
    (the session token is not on the record).
    """

    lines = [
        f"⚙️ {record.action.value} → {record.target_user_id}",
        f"  by: {record.performed_by} @ {record.performed_at.isoformat()}",
    ]
    if record.target_tier is not None:
        lines.append(f"  target tier: {record.target_tier.value}")
    if record.reason:
        lines.append(f"  reason: {record.reason}")
    return "\n".join(lines)


__all__ = [
    "AdminAction",
    "AdminActionRecord",
    "AdminActionRequest",
    "AdminAuthorizationError",
    "AdminEmailList",
    "AdminView",
    "UserSummary",
    "audit_admin_action",
    "build_admin_view",
    "render_action_record",
    "render_admin_view",
    "render_user_summary",
]
