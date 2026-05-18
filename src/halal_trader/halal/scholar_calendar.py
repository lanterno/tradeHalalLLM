"""Scholar consultation calendar.

Auxiliary primitive complementing Wave 2.F scholar review workflow
+ Wave 11.B SSB governance. Wave 2.F handles ad-hoc verdicts on the
exception queue; Wave 11.B tracks the SSB board's structured
governance meetings; this module is the **pure-Python scheduler**
for the regular cadence of scholar consultations: annual halal
compliance audit, quarterly portfolio review, plus ad-hoc
consultations triggered by exception-queue overflow.

Picked a focused calendar over a generic ical / Google Calendar
integration because (a) the consultation kinds (ANNUAL_AUDIT /
QUARTERLY_REVIEW / AD_HOC) have very different cadence + lead-time
requirements: an annual audit needs months of advance scheduling
across multiple stakeholder calendars; a quarterly review needs ~2
weeks; an ad-hoc consultation may need same-week turnaround;
encoding the cadence as policy means the dashboard surfaces "your
next quarterly review is overdue" without operator interpretation;
(b) the reminder ladder (30d / 7d / 1d before) is the load-bearing
"will the scholar actually show up?" attribute — encoding it
explicitly means a missed reminder gets caught at the audit-trail
layer rather than after the consultation date passes; (c)
deterministic state transitions (SCHEDULED → CONFIRMED → COMPLETED)
mirror the lifecycle pattern used in 8.E incident-response and 6.I
distillation deployment, giving operators one mental model.

Pinned semantics:
- **Closed-set ConsultationKind enum.** ANNUAL_AUDIT / QUARTERLY_REVIEW
  / AD_HOC. Adding a kind is a code review change.
- **State machine: SCHEDULED → CONFIRMED → COMPLETED (or → CANCELLED).**
  Forward-only on the happy path; CANCELLED is terminal from any
  pre-COMPLETED state.
- **Reminder ladder by lead-time: 30d, 7d, 1d before scheduled date.**
  Operator-tunable but defaults match standard cadence-meeting
  best-practice. The is-due-for-reminder check is pure: deterministic
  for given (consultation, now).
- **Annual audit must be CONFIRMED at least 60 days before the
  scheduled date.** A non-confirmed annual audit at 59 days out
  flags overdue (the scholar's calendar fills up).
- **Render output never includes scholar contact emails or
  meeting URLs.** Only the public consultation summary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ConsultationKind(str, Enum):
    """Kind of scholar consultation.

    Pinned string values for JSON / DB stability. Adding a kind is
    a code review change.
    """

    ANNUAL_AUDIT = "annual_audit"
    QUARTERLY_REVIEW = "quarterly_review"
    AD_HOC = "ad_hoc"


class ConsultationStatus(str, Enum):
    """Lifecycle status."""

    SCHEDULED = "scheduled"  # Booked but scholar hasn't confirmed
    CONFIRMED = "confirmed"  # Scholar confirmed attendance
    COMPLETED = "completed"  # Consultation happened; minutes recorded
    CANCELLED = "cancelled"  # Cancelled before completion (terminal)


_HAPPY_PATH_ORDER: tuple[ConsultationStatus, ...] = (
    ConsultationStatus.SCHEDULED,
    ConsultationStatus.CONFIRMED,
    ConsultationStatus.COMPLETED,
)


_DEFAULT_REMINDER_LEAD_TIMES: tuple[timedelta, ...] = (
    timedelta(days=30),
    timedelta(days=7),
    timedelta(days=1),
)


_DEFAULT_ANNUAL_CONFIRM_LEAD_TIME = timedelta(days=60)


@dataclass(frozen=True)
class CalendarPolicy:
    """Operator-tunable calendar policy."""

    reminder_lead_times: tuple[timedelta, ...] = _DEFAULT_REMINDER_LEAD_TIMES
    annual_confirm_lead_time: timedelta = _DEFAULT_ANNUAL_CONFIRM_LEAD_TIME

    def __post_init__(self) -> None:
        if not self.reminder_lead_times:
            raise ValueError("reminder_lead_times must be non-empty")
        for lead in self.reminder_lead_times:
            if lead <= timedelta(0):
                raise ValueError(f"reminder lead time {lead} must be positive")
        # Lead times should be in descending order (30d → 7d → 1d)
        sorted_leads = sorted(self.reminder_lead_times, key=lambda t: -t.total_seconds())
        if list(self.reminder_lead_times) != sorted_leads:
            raise ValueError("reminder_lead_times must be in descending order (longest lead first)")
        if self.annual_confirm_lead_time <= timedelta(0):
            raise ValueError("annual_confirm_lead_time must be positive")


DEFAULT_POLICY = CalendarPolicy()


class StatusTransitionError(Exception):
    """Raised on invalid status transition."""

    def __init__(
        self,
        consultation_id: str,
        current: ConsultationStatus,
        attempted: ConsultationStatus,
    ) -> None:
        super().__init__(
            f"consultation {consultation_id!r}: cannot transition from "
            f"{current.value} to {attempted.value}"
        )
        self.consultation_id = consultation_id
        self.current = current
        self.attempted = attempted


@dataclass(frozen=True)
class Consultation:
    """One scheduled scholar consultation."""

    consultation_id: str
    kind: ConsultationKind
    scholar_handle: str  # public scholar identifier (not contact email)
    scheduled_at: datetime
    status: ConsultationStatus
    last_status_at: datetime
    topic: str
    minutes_url: str = ""  # set when COMPLETED

    def __post_init__(self) -> None:
        if not self.consultation_id or not self.consultation_id.strip():
            raise ValueError("consultation_id must be non-empty")
        if not self.scholar_handle or not self.scholar_handle.strip():
            raise ValueError("scholar_handle must be non-empty")
        if not self.topic or not self.topic.strip():
            raise ValueError("topic must be non-empty")
        if self.scheduled_at.tzinfo is None:
            raise ValueError("scheduled_at must be timezone-aware")
        if self.last_status_at.tzinfo is None:
            raise ValueError("last_status_at must be timezone-aware")
        # COMPLETED requires a minutes URL (the audit trail of what
        # was discussed); other statuses must NOT have minutes_url
        if self.status is ConsultationStatus.COMPLETED:
            if not self.minutes_url.strip():
                raise ValueError("COMPLETED status requires non-empty minutes_url")
        else:
            if self.minutes_url:
                raise ValueError(f"{self.status.value} status must not have minutes_url")


def schedule_consultation(
    *,
    consultation_id: str,
    kind: ConsultationKind,
    scholar_handle: str,
    scheduled_at: datetime,
    topic: str,
    now: datetime,
) -> Consultation:
    """Create a fresh SCHEDULED consultation."""

    if not consultation_id or not consultation_id.strip():
        raise ValueError("consultation_id must be non-empty")
    if not scholar_handle or not scholar_handle.strip():
        raise ValueError("scholar_handle must be non-empty")
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if scheduled_at.tzinfo is None:
        raise ValueError("scheduled_at must be timezone-aware")
    if scheduled_at <= now:
        raise ValueError(f"scheduled_at {scheduled_at} must be in the future (now={now})")
    return Consultation(
        consultation_id=consultation_id,
        kind=kind,
        scholar_handle=scholar_handle,
        scheduled_at=scheduled_at,
        status=ConsultationStatus.SCHEDULED,
        last_status_at=now,
        topic=topic,
    )


def confirm_consultation(consultation: Consultation, *, now: datetime) -> Consultation:
    """Move SCHEDULED → CONFIRMED."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if consultation.status is not ConsultationStatus.SCHEDULED:
        raise StatusTransitionError(
            consultation.consultation_id,
            consultation.status,
            ConsultationStatus.CONFIRMED,
        )
    return Consultation(
        consultation_id=consultation.consultation_id,
        kind=consultation.kind,
        scholar_handle=consultation.scholar_handle,
        scheduled_at=consultation.scheduled_at,
        status=ConsultationStatus.CONFIRMED,
        last_status_at=now,
        topic=consultation.topic,
    )


def complete_consultation(
    consultation: Consultation, *, now: datetime, minutes_url: str
) -> Consultation:
    """Move CONFIRMED → COMPLETED. Requires minutes_url."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not minutes_url or not minutes_url.strip():
        raise ValueError("minutes_url must be non-empty for COMPLETED")
    if consultation.status is not ConsultationStatus.CONFIRMED:
        raise StatusTransitionError(
            consultation.consultation_id,
            consultation.status,
            ConsultationStatus.COMPLETED,
        )
    return Consultation(
        consultation_id=consultation.consultation_id,
        kind=consultation.kind,
        scholar_handle=consultation.scholar_handle,
        scheduled_at=consultation.scheduled_at,
        status=ConsultationStatus.COMPLETED,
        last_status_at=now,
        topic=consultation.topic,
        minutes_url=minutes_url,
    )


def cancel_consultation(consultation: Consultation, *, now: datetime) -> Consultation:
    """Move any pre-COMPLETED status → CANCELLED. Terminal."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if consultation.status in (
        ConsultationStatus.COMPLETED,
        ConsultationStatus.CANCELLED,
    ):
        raise StatusTransitionError(
            consultation.consultation_id,
            consultation.status,
            ConsultationStatus.CANCELLED,
        )
    return Consultation(
        consultation_id=consultation.consultation_id,
        kind=consultation.kind,
        scholar_handle=consultation.scholar_handle,
        scheduled_at=consultation.scheduled_at,
        status=ConsultationStatus.CANCELLED,
        last_status_at=now,
        topic=consultation.topic,
    )


def is_due_for_reminder(
    consultation: Consultation,
    *,
    now: datetime,
    last_reminder_at: datetime | None = None,
    policy: CalendarPolicy = DEFAULT_POLICY,
) -> bool:
    """True if a reminder should be fired now.

    Returns True if `now` is at or past one of the lead-time
    thresholds AND `last_reminder_at` predates that threshold (so
    we don't re-fire the same reminder).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if last_reminder_at is not None and last_reminder_at.tzinfo is None:
        raise ValueError("last_reminder_at must be timezone-aware when set")
    # Only SCHEDULED + CONFIRMED consultations need reminders
    if consultation.status not in (
        ConsultationStatus.SCHEDULED,
        ConsultationStatus.CONFIRMED,
    ):
        return False

    for lead in policy.reminder_lead_times:
        threshold = consultation.scheduled_at - lead
        if now >= threshold:
            # We've passed this threshold; was a reminder already sent?
            if last_reminder_at is None or last_reminder_at < threshold:
                return True
    return False


def annual_audit_overdue_for_confirmation(
    consultation: Consultation,
    *,
    now: datetime,
    policy: CalendarPolicy = DEFAULT_POLICY,
) -> bool:
    """True if an ANNUAL_AUDIT is still SCHEDULED within
    `annual_confirm_lead_time` of its scheduled date.

    Pin: annual audits need scholars to confirm at least 60 days
    out (their calendars fill up). A non-confirmed annual audit
    at 59 days out is operationally risky.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if consultation.kind is not ConsultationKind.ANNUAL_AUDIT:
        return False
    if consultation.status is not ConsultationStatus.SCHEDULED:
        return False
    deadline = consultation.scheduled_at - policy.annual_confirm_lead_time
    return now >= deadline


def filter_due_for_reminder(
    consultations: Iterable[Consultation],
    *,
    now: datetime,
    last_reminders: dict[str, datetime] | None = None,
    policy: CalendarPolicy = DEFAULT_POLICY,
) -> tuple[Consultation, ...]:
    """Return consultations due for a reminder, sorted by scheduled_at."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    last_reminders_map = last_reminders or {}
    due = [
        c
        for c in consultations
        if is_due_for_reminder(
            c,
            now=now,
            last_reminder_at=last_reminders_map.get(c.consultation_id),
            policy=policy,
        )
    ]
    return tuple(sorted(due, key=lambda c: c.scheduled_at))


def upcoming(
    consultations: Iterable[Consultation],
    *,
    now: datetime,
    horizon: timedelta = timedelta(days=90),
) -> tuple[Consultation, ...]:
    """Return consultations scheduled within the horizon, sorted ascending."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if horizon <= timedelta(0):
        raise ValueError("horizon must be positive")
    cutoff = now + horizon
    upcoming_list = [
        c
        for c in consultations
        if c.status in (ConsultationStatus.SCHEDULED, ConsultationStatus.CONFIRMED)
        and now <= c.scheduled_at <= cutoff
    ]
    return tuple(sorted(upcoming_list, key=lambda c: c.scheduled_at))


_KIND_EMOJI: dict[ConsultationKind, str] = {
    ConsultationKind.ANNUAL_AUDIT: "📋",
    ConsultationKind.QUARTERLY_REVIEW: "📅",
    ConsultationKind.AD_HOC: "💬",
}


_STATUS_EMOJI: dict[ConsultationStatus, str] = {
    ConsultationStatus.SCHEDULED: "🗓️",
    ConsultationStatus.CONFIRMED: "✅",
    ConsultationStatus.COMPLETED: "📝",
    ConsultationStatus.CANCELLED: "❌",
}


def render_consultation(consultation: Consultation) -> str:
    """Format a consultation for ops display.

    No-secret-leak: the dataclass deliberately doesn't carry scholar
    contact emails or meeting URLs (operator-side state); render
    surfaces only public fields. The minutes_url is intentionally
    rendered when present because it's the public audit trail.
    """

    kind_emoji = _KIND_EMOJI[consultation.kind]
    status_emoji = _STATUS_EMOJI[consultation.status]
    lines = [
        f"{kind_emoji}{status_emoji} {consultation.consultation_id} "
        f"({consultation.kind.value}) — {consultation.status.value}",
        f"  scholar: {consultation.scholar_handle}",
        f"  topic: {consultation.topic}",
        f"  scheduled: {consultation.scheduled_at.isoformat()}",
    ]
    if consultation.minutes_url:
        lines.append(f"  minutes: {consultation.minutes_url}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "CalendarPolicy",
    "Consultation",
    "ConsultationKind",
    "ConsultationStatus",
    "StatusTransitionError",
    "annual_audit_overdue_for_confirmation",
    "cancel_consultation",
    "complete_consultation",
    "confirm_consultation",
    "filter_due_for_reminder",
    "is_due_for_reminder",
    "render_consultation",
    "schedule_consultation",
    "upcoming",
]
