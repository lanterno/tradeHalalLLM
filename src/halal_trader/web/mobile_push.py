"""Mobile push notification policy + delivery state machine.

The roadmap pins Wave 3.J: "Native iOS/Android app for monitoring +
halt-switch + push notifications. The existing `/api/mobile/summary`
is the contract. Wire the operator's phone to receive trade fills +
risk halts + daily summary as native push." This module is the
**pure-Python policy + delivery-state engine** that the FastAPI
push-dispatcher route consumes; the actual APNs / FCM SDK calls
happen operator-side once the SDK integration lands.

Picked a focused policy + state engine over wiring directly into
APNs/FCM because (a) the notification kinds (trade_fill /
risk_halt / daily_summary) have very different urgency profiles —
risk_halt is critical and bypasses quiet hours, daily_summary
respects quiet hours absolutely; encoding the priority + quiet-hour
rules once means the dispatcher can't accidentally send a daily
summary at 3am, (b) device registration has a platform-specific
token format (APNs hex, FCM string) that the API contract enforces
at registration time so the SDK call doesn't fail mid-dispatch,
(c) delivery state (PENDING → SENT → DELIVERED → FAILED) tracked
deterministically gives operators an audit trail for "why didn't
my phone buzz at 14:22?" without correlating across APNs + FCM
provider logs, (d) per-user rate limiting prevents accidental DoS
from a noisy strategy ("100 trade-fill notifications in 60 seconds"
floods the operator's phone and is a usability failure).

Pinned semantics:
- **Three notification kinds with priority ladder.** RISK_HALT
  is critical (bypasses quiet hours + rate limit); TRADE_FILL is
  normal (respects quiet hours, rate-limited); DAILY_SUMMARY is
  low (respects quiet hours absolutely).
- **Quiet hours default 22:00-07:00 in the user's timezone.**
  Operator-tunable; CRITICAL bypasses; pinned via test that
  TRADE_FILL at 23:00 quiet-rejects but RISK_HALT at 23:00 sends.
- **Rate limit: 30 non-critical notifications per hour per user.**
  Operator-tunable; CRITICAL bypasses; rolling 60-minute window.
- **Delivery state machine: PENDING → SENT → DELIVERED → FAILED.**
  Status transitions one-way; pinned to keep the audit trail honest.
- **Render output never includes device tokens / APNs / FCM keys.**
  Mirrors no-secret patterns of upstream waves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum


class NotificationKind(str, Enum):
    """Pinned notification kinds with priority ordering.

    Pinned string values for JSON / DB stability.
    """

    TRADE_FILL = "trade_fill"
    RISK_HALT = "risk_halt"
    DAILY_SUMMARY = "daily_summary"


class Priority(str, Enum):
    """Priority tiers. CRITICAL bypasses quiet hours + rate limits."""

    LOW = "low"
    NORMAL = "normal"
    CRITICAL = "critical"


_KIND_PRIORITY: dict[NotificationKind, Priority] = {
    NotificationKind.RISK_HALT: Priority.CRITICAL,
    NotificationKind.TRADE_FILL: Priority.NORMAL,
    NotificationKind.DAILY_SUMMARY: Priority.LOW,
}


class Platform(str, Enum):
    """Mobile platforms. Pinned values."""

    IOS = "ios"
    ANDROID = "android"


class DeliveryStatus(str, Enum):
    """Delivery state machine."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"


_DELIVERY_ORDER: tuple[DeliveryStatus, ...] = (
    DeliveryStatus.PENDING,
    DeliveryStatus.SENT,
    DeliveryStatus.DELIVERED,
)


_DEFAULT_QUIET_START = time(22, 0)
_DEFAULT_QUIET_END = time(7, 0)
_DEFAULT_RATE_LIMIT_PER_HOUR = 30


@dataclass(frozen=True)
class NotificationPolicy:
    """Operator-tunable notification policy."""

    quiet_hours_start: time = _DEFAULT_QUIET_START
    quiet_hours_end: time = _DEFAULT_QUIET_END
    rate_limit_per_hour: int = _DEFAULT_RATE_LIMIT_PER_HOUR

    def __post_init__(self) -> None:
        if self.rate_limit_per_hour <= 0:
            raise ValueError("rate_limit_per_hour must be positive")
        if self.quiet_hours_start == self.quiet_hours_end:
            raise ValueError(
                "quiet_hours_start and quiet_hours_end must differ "
                "(use 23:59 for 'no quiet hours' approximation)"
            )


DEFAULT_POLICY = NotificationPolicy()


def _is_in_quiet_hours(
    *,
    now: datetime,
    policy: NotificationPolicy,
) -> bool:
    """Return True if `now` falls within the policy's quiet hours.

    Handles wraparound (22:00-07:00 spans midnight).
    """

    current = now.time()
    start = policy.quiet_hours_start
    end = policy.quiet_hours_end
    if start < end:
        # Same-day window (e.g. 12:00-15:00)
        return start <= current < end
    # Wraparound window (e.g. 22:00-07:00)
    return current >= start or current < end


@dataclass(frozen=True)
class DeviceRegistration:
    """Mobile device registration row.

    `device_token` is the platform-specific push token (APNs hex string
    on iOS, FCM token on Android). The dataclass deliberately doesn't
    log it — the no-secret render pin asserts the token never appears
    in render output.
    """

    user_id: str
    device_id: str
    platform: Platform
    device_token: str
    timezone_offset_minutes: int  # user's local UTC offset
    registered_at: datetime
    revoked: bool = False

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.device_id or not self.device_id.strip():
            raise ValueError("device_id must be non-empty")
        if not self.device_token or not self.device_token.strip():
            raise ValueError("device_token must be non-empty")
        if not -720 <= self.timezone_offset_minutes <= 840:
            raise ValueError(
                f"timezone_offset_minutes {self.timezone_offset_minutes} out of [-720, 840]"
            )
        if self.registered_at.tzinfo is None:
            raise ValueError("registered_at must be timezone-aware")
        # Per-platform token shape sanity (loose check)
        if self.platform is Platform.IOS:
            # APNs tokens are typically 64-character hex strings
            stripped = self.device_token.strip()
            if not all(c in "0123456789abcdefABCDEF" for c in stripped):
                raise ValueError(f"iOS device_token must be hex; got {stripped[:8]!r}...")


@dataclass(frozen=True)
class NotificationRequest:
    """Operator-or-cycle-side request to send a notification."""

    notification_id: str
    user_id: str
    kind: NotificationKind
    title: str
    body: str
    requested_at: datetime

    def __post_init__(self) -> None:
        if not self.notification_id or not self.notification_id.strip():
            raise ValueError("notification_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.body or not self.body.strip():
            raise ValueError("body must be non-empty")
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        if len(self.title) > 100:
            raise ValueError(f"title too long ({len(self.title)} > 100)")
        if len(self.body) > 1000:
            raise ValueError(f"body too long ({len(self.body)} > 1000)")


class GateOutcome(str, Enum):
    """Why a notification was sent or held."""

    SEND = "send"
    HOLD_QUIET_HOURS = "hold_quiet_hours"
    HOLD_RATE_LIMIT = "hold_rate_limit"
    HOLD_NO_DEVICE = "hold_no_device"


@dataclass(frozen=True)
class GateDecision:
    """Outcome of gate checks for one notification."""

    outcome: GateOutcome
    message: str

    def __post_init__(self) -> None:
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")


def priority_for(kind: NotificationKind) -> Priority:
    return _KIND_PRIORITY[kind]


def _user_local_now(request: NotificationRequest, *, registration: DeviceRegistration) -> datetime:
    """Convert the request's requested_at to the user's local time."""

    local_tz = timezone(timedelta(minutes=registration.timezone_offset_minutes))
    return request.requested_at.astimezone(local_tz)


def evaluate_gate(
    request: NotificationRequest,
    *,
    registration: DeviceRegistration | None,
    recent_send_count: int,
    policy: NotificationPolicy = DEFAULT_POLICY,
) -> GateDecision:
    """Decide whether to dispatch a notification given the gates.

    `recent_send_count` is the count of non-critical notifications the
    user has received in the last hour (the dispatcher tracks this);
    used for rate-limit gate.

    `registration` is the user's active mobile device (None if user
    has no registered device — HOLD_NO_DEVICE).

    Pinned: CRITICAL bypasses quiet hours AND rate limits. NORMAL +
    LOW respect both.
    """

    if registration is None or registration.revoked:
        return GateDecision(
            outcome=GateOutcome.HOLD_NO_DEVICE,
            message=f"user {request.user_id!r} has no active mobile device",
        )

    priority = priority_for(request.kind)
    if priority is Priority.CRITICAL:
        return GateDecision(
            outcome=GateOutcome.SEND,
            message=f"critical {request.kind.value} bypasses quiet hours + rate limit",
        )

    # Quiet hours check
    user_local = _user_local_now(request, registration=registration)
    if _is_in_quiet_hours(now=user_local, policy=policy):
        return GateDecision(
            outcome=GateOutcome.HOLD_QUIET_HOURS,
            message=(
                f"{request.kind.value} held: user in quiet hours "
                f"({policy.quiet_hours_start}-{policy.quiet_hours_end})"
            ),
        )

    # Rate-limit check
    if recent_send_count >= policy.rate_limit_per_hour:
        return GateDecision(
            outcome=GateOutcome.HOLD_RATE_LIMIT,
            message=(
                f"rate limit hit: {recent_send_count} sends in last hour "
                f"(cap {policy.rate_limit_per_hour})"
            ),
        )

    return GateDecision(
        outcome=GateOutcome.SEND,
        message=f"{request.kind.value} clear to send",
    )


@dataclass(frozen=True)
class DeliveryRecord:
    """Per-notification audit row.

    Operations (`mark_sent`, `mark_delivered`, `mark_failed`) return
    new state; the audit trail is the immutable history of status
    transitions.
    """

    notification_id: str
    status: DeliveryStatus
    last_updated_at: datetime
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not self.notification_id or not self.notification_id.strip():
            raise ValueError("notification_id must be non-empty")
        if self.last_updated_at.tzinfo is None:
            raise ValueError("last_updated_at must be timezone-aware")
        if self.status is DeliveryStatus.FAILED and not self.failure_reason.strip():
            raise ValueError("FAILED status requires failure_reason")
        if self.status is not DeliveryStatus.FAILED and self.failure_reason:
            raise ValueError("non-FAILED status must have empty failure_reason")


class DeliveryOrderError(Exception):
    """Raised when a status transition violates the state machine order."""

    def __init__(self, current: DeliveryStatus, attempted: DeliveryStatus) -> None:
        super().__init__(f"cannot transition from {current.value} to {attempted.value}")
        self.current = current
        self.attempted = attempted


def start_delivery(*, notification_id: str, now: datetime) -> DeliveryRecord:
    """Create the initial PENDING delivery record."""

    if not notification_id or not notification_id.strip():
        raise ValueError("notification_id must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return DeliveryRecord(
        notification_id=notification_id,
        status=DeliveryStatus.PENDING,
        last_updated_at=now,
    )


def _check_forward(current: DeliveryStatus, target: DeliveryStatus) -> None:
    """Ensure `target` is one step forward from `current` in canonical order."""

    if current is DeliveryStatus.FAILED:
        raise DeliveryOrderError(current, target)
    cur_idx = _DELIVERY_ORDER.index(current)
    target_idx = _DELIVERY_ORDER.index(target)
    if target_idx != cur_idx + 1:
        raise DeliveryOrderError(current, target)


def mark_sent(record: DeliveryRecord, *, now: datetime) -> DeliveryRecord:
    """Move PENDING → SENT (after APNs/FCM accepts the dispatch)."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    _check_forward(record.status, DeliveryStatus.SENT)
    return DeliveryRecord(
        notification_id=record.notification_id,
        status=DeliveryStatus.SENT,
        last_updated_at=now,
    )


def mark_delivered(record: DeliveryRecord, *, now: datetime) -> DeliveryRecord:
    """Move SENT → DELIVERED (after APNs/FCM confirms delivery)."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    _check_forward(record.status, DeliveryStatus.DELIVERED)
    return DeliveryRecord(
        notification_id=record.notification_id,
        status=DeliveryStatus.DELIVERED,
        last_updated_at=now,
    )


def mark_failed(
    record: DeliveryRecord,
    *,
    now: datetime,
    failure_reason: str,
) -> DeliveryRecord:
    """Move any pre-FAILED status → FAILED. FAILED is terminal."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not failure_reason or not failure_reason.strip():
        raise ValueError("failure_reason must be non-empty")
    if record.status is DeliveryStatus.FAILED:
        raise DeliveryOrderError(record.status, DeliveryStatus.FAILED)
    return DeliveryRecord(
        notification_id=record.notification_id,
        status=DeliveryStatus.FAILED,
        last_updated_at=now,
        failure_reason=failure_reason,
    )


def count_recent_sends(
    records: Iterable[DeliveryRecord],
    *,
    user_notification_ids: Iterable[str],
    now: datetime,
    window: timedelta = timedelta(hours=1),
) -> int:
    """Count notifications sent for the user in the last `window`.

    Used by the gate to compute the rate-limit input. The dispatcher
    builds the full record set + the user's notification id list and
    passes both — keeps the gate logic pure.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cutoff = now - window
    user_ids = set(user_notification_ids)
    return sum(
        1
        for r in records
        if r.notification_id in user_ids
        and r.status in (DeliveryStatus.SENT, DeliveryStatus.DELIVERED)
        and r.last_updated_at >= cutoff
    )


_KIND_EMOJI: dict[NotificationKind, str] = {
    NotificationKind.TRADE_FILL: "💸",
    NotificationKind.RISK_HALT: "🚨",
    NotificationKind.DAILY_SUMMARY: "📊",
}


_PRIORITY_EMOJI: dict[Priority, str] = {
    Priority.LOW: "🔵",
    Priority.NORMAL: "🟡",
    Priority.CRITICAL: "🔴",
}


_STATUS_EMOJI: dict[DeliveryStatus, str] = {
    DeliveryStatus.PENDING: "⏳",
    DeliveryStatus.SENT: "📤",
    DeliveryStatus.DELIVERED: "✅",
    DeliveryStatus.FAILED: "❌",
}


def render_request(request: NotificationRequest) -> str:
    """Format a notification request for ops display.

    No-secret-leak: never includes device_token / APNs key / FCM key.
    """

    kind_emoji = _KIND_EMOJI[request.kind]
    pri_emoji = _PRIORITY_EMOJI[priority_for(request.kind)]
    return (
        f"{kind_emoji}{pri_emoji} {request.title}\n"
        f"  user: {request.user_id}\n"
        f"  body: {request.body}\n"
        f"  kind: {request.kind.value}\n"
        f"  requested: {request.requested_at.isoformat()}"
    )


def render_delivery(record: DeliveryRecord) -> str:
    """Format a delivery record for ops display."""

    emoji = _STATUS_EMOJI[record.status]
    line = f"{emoji} {record.notification_id} — {record.status.value}"
    if record.failure_reason:
        line += f" — {record.failure_reason}"
    return line


__all__ = [
    "DEFAULT_POLICY",
    "DeliveryOrderError",
    "DeliveryRecord",
    "DeliveryStatus",
    "DeviceRegistration",
    "GateDecision",
    "GateOutcome",
    "NotificationKind",
    "NotificationPolicy",
    "NotificationRequest",
    "Platform",
    "Priority",
    "count_recent_sends",
    "evaluate_gate",
    "mark_delivered",
    "mark_failed",
    "mark_sent",
    "priority_for",
    "render_delivery",
    "render_request",
    "start_delivery",
]
