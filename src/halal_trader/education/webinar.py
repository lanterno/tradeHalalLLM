"""Scholar webinar coordinator — Round-5 Wave 20.C.

Monthly live Q&A with the platform's scholars. This module is the
**scheduling + question intake + RSVP lifecycle**:

- A `Webinar` represents one scheduled session (scholar + topic +
  time + capacity).
- `QuestionSubmission` records flow from the audience pre-event;
  scholar can mark each as ANSWERED / DEFERRED / DECLINED.
- `Attendance` records RSVPs; capacity enforced.

Pinned semantics:

- **Closed-set WebinarStatus FSM** — SCHEDULED → REGISTRATION_OPEN →
  LIVE → CONCLUDED, with CANCELLED as alternate terminal from
  SCHEDULED, REGISTRATION_OPEN, or LIVE.
- **Closed-set QuestionStatus FSM** — SUBMITTED → ANSWERED / DEFERRED
  / DECLINED.
- **Question moderation**: questions reference the platform's
  chat-moderation lexicon (caller-supplied predicate); BLOCKED
  questions never enter the queue.
- **Pure-Python deterministic.**
- **No-secret-leak pin** — IDs masked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


class WebinarStatus(str, Enum):
    """Closed-set webinar FSM ladder."""

    SCHEDULED = "scheduled"
    REGISTRATION_OPEN = "registration_open"
    LIVE = "live"
    CONCLUDED = "concluded"
    CANCELLED = "cancelled"


class QuestionStatus(str, Enum):
    """Closed-set question lifecycle ladder."""

    SUBMITTED = "submitted"
    ANSWERED = "answered"
    DEFERRED = "deferred"
    DECLINED = "declined"


@dataclass(frozen=True)
class Webinar:
    """One scheduled scholar webinar."""

    webinar_id: str
    scholar_id: str
    title: str
    topic: str
    scheduled_at: datetime
    duration_minutes: int
    capacity: int
    registration_opens_at: datetime
    status: WebinarStatus = WebinarStatus.SCHEDULED
    cancelled_reason: str = ""

    def __post_init__(self) -> None:
        if not self.webinar_id or not self.webinar_id.strip():
            raise ValueError("webinar_id must be non-empty")
        if not self.scholar_id or not self.scholar_id.strip():
            raise ValueError("scholar_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if len(self.title) > 200:
            raise ValueError("title must be ≤ 200 chars")
        if not self.topic.strip():
            raise ValueError("topic must be non-empty")
        if len(self.topic) > 100:
            raise ValueError("topic must be ≤ 100 chars")
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")
        if self.duration_minutes > 240:
            raise ValueError("duration > 240 min suspicious")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.capacity > 10_000:
            raise ValueError("capacity > 10_000 suspicious")
        if self.registration_opens_at > self.scheduled_at:
            raise ValueError("registration_opens_at must be ≤ scheduled_at")
        if self.status is WebinarStatus.CANCELLED and not self.cancelled_reason.strip():
            raise ValueError("CANCELLED requires cancelled_reason")
        if self.status is not WebinarStatus.CANCELLED and self.cancelled_reason.strip():
            raise ValueError("cancelled_reason only set when CANCELLED")


@dataclass(frozen=True)
class Attendance:
    """One user's registration."""

    attendance_id: str
    webinar_id: str
    user_id: str
    registered_at: datetime
    attended: bool = False

    def __post_init__(self) -> None:
        if not self.attendance_id or not self.attendance_id.strip():
            raise ValueError("attendance_id must be non-empty")
        if not self.webinar_id or not self.webinar_id.strip():
            raise ValueError("webinar_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")


@dataclass(frozen=True)
class QuestionSubmission:
    """One pre-event question submitted by an attendee."""

    question_id: str
    webinar_id: str
    asker_id: str
    text: str
    submitted_at: datetime
    status: QuestionStatus = QuestionStatus.SUBMITTED
    scholar_response: str = ""

    def __post_init__(self) -> None:
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        if not self.webinar_id or not self.webinar_id.strip():
            raise ValueError("webinar_id must be non-empty")
        if not self.asker_id or not self.asker_id.strip():
            raise ValueError("asker_id must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if len(self.text) > 1000:
            raise ValueError("text must be ≤ 1000 chars")
        if self.status is QuestionStatus.ANSWERED and not self.scholar_response.strip():
            raise ValueError("ANSWERED requires non-empty scholar_response")


# --- Webinar FSM ------------------------------------


_LEGAL_WEBINAR_TRANSITIONS: dict[WebinarStatus, set[WebinarStatus]] = {
    WebinarStatus.SCHEDULED: {
        WebinarStatus.REGISTRATION_OPEN,
        WebinarStatus.CANCELLED,
    },
    WebinarStatus.REGISTRATION_OPEN: {
        WebinarStatus.LIVE,
        WebinarStatus.CANCELLED,
    },
    WebinarStatus.LIVE: {
        WebinarStatus.CONCLUDED,
        WebinarStatus.CANCELLED,
    },
    WebinarStatus.CONCLUDED: set(),
    WebinarStatus.CANCELLED: set(),
}


def open_registration(webinar: Webinar) -> Webinar:
    if webinar.status is not WebinarStatus.SCHEDULED:
        raise ValueError(f"open_registration illegal from {webinar.status.value}")
    return replace(webinar, status=WebinarStatus.REGISTRATION_OPEN)


def go_live(webinar: Webinar) -> Webinar:
    if webinar.status is not WebinarStatus.REGISTRATION_OPEN:
        raise ValueError(f"go_live illegal from {webinar.status.value}")
    return replace(webinar, status=WebinarStatus.LIVE)


def conclude(webinar: Webinar) -> Webinar:
    if webinar.status is not WebinarStatus.LIVE:
        raise ValueError(f"conclude illegal from {webinar.status.value}")
    return replace(webinar, status=WebinarStatus.CONCLUDED)


def cancel(webinar: Webinar, *, reason: str) -> Webinar:
    if WebinarStatus.CANCELLED not in _LEGAL_WEBINAR_TRANSITIONS[webinar.status]:
        raise ValueError(f"cancel illegal from {webinar.status.value}")
    if not reason.strip():
        raise ValueError("reason must be non-empty")
    if len(reason) > 500:
        raise ValueError("reason too long")
    return replace(webinar, status=WebinarStatus.CANCELLED, cancelled_reason=reason)


# --- Attendance + capacity ---------------------------


def register_attendee(
    webinar: Webinar,
    attendances: Iterable[Attendance],
    *,
    attendance_id: str,
    user_id: str,
    registered_at: datetime,
) -> Attendance:
    """Register a new attendee. Capacity + status checks enforced."""
    if webinar.status not in (
        WebinarStatus.REGISTRATION_OPEN,
        WebinarStatus.LIVE,
    ):
        raise ValueError(f"cannot register to a {webinar.status.value} webinar")
    atts = tuple(attendances)
    existing = [a for a in atts if a.webinar_id == webinar.webinar_id]
    if any(a.user_id == user_id for a in existing):
        raise ValueError(f"{user_id} already registered")
    if len(existing) + 1 > webinar.capacity:
        raise ValueError(f"capacity {webinar.capacity} exceeded")
    return Attendance(
        attendance_id=attendance_id,
        webinar_id=webinar.webinar_id,
        user_id=user_id,
        registered_at=registered_at,
    )


def mark_attended(attendance: Attendance) -> Attendance:
    if attendance.attended:
        return attendance
    return replace(attendance, attended=True)


def attendance_count(webinar_id: str, attendances: Iterable[Attendance]) -> int:
    return sum(1 for a in attendances if a.webinar_id == webinar_id)


# --- Question submission + moderation -----------------


def submit_question(
    webinar: Webinar,
    *,
    question_id: str,
    asker_id: str,
    text: str,
    submitted_at: datetime,
    is_text_acceptable: Callable[[str], bool] | None = None,
) -> QuestionSubmission:
    """Submit a question; caller provides a moderation predicate.

    Pinned: questions can only be submitted while SCHEDULED,
    REGISTRATION_OPEN, or LIVE.
    """
    if webinar.status not in (
        WebinarStatus.SCHEDULED,
        WebinarStatus.REGISTRATION_OPEN,
        WebinarStatus.LIVE,
    ):
        raise ValueError(f"cannot submit a question to a {webinar.status.value} webinar")
    if is_text_acceptable is not None and not is_text_acceptable(text):
        raise ValueError("question text failed moderation")
    return QuestionSubmission(
        question_id=question_id,
        webinar_id=webinar.webinar_id,
        asker_id=asker_id,
        text=text,
        submitted_at=submitted_at,
    )


def answer_question(question: QuestionSubmission, *, response: str) -> QuestionSubmission:
    if question.status is not QuestionStatus.SUBMITTED:
        raise ValueError(f"answer illegal from {question.status.value}")
    if not response.strip():
        raise ValueError("response must be non-empty")
    if len(response) > 5000:
        raise ValueError("response too long")
    return replace(
        question,
        status=QuestionStatus.ANSWERED,
        scholar_response=response,
    )


def defer_question(question: QuestionSubmission) -> QuestionSubmission:
    if question.status is not QuestionStatus.SUBMITTED:
        raise ValueError(f"defer illegal from {question.status.value}")
    return replace(question, status=QuestionStatus.DEFERRED)


def decline_question(question: QuestionSubmission) -> QuestionSubmission:
    if question.status is not QuestionStatus.SUBMITTED:
        raise ValueError(f"decline illegal from {question.status.value}")
    return replace(question, status=QuestionStatus.DECLINED)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[WebinarStatus, str] = {
    WebinarStatus.SCHEDULED: "📅",
    WebinarStatus.REGISTRATION_OPEN: "📝",
    WebinarStatus.LIVE: "🔴",
    WebinarStatus.CONCLUDED: "✅",
    WebinarStatus.CANCELLED: "🚫",
}


def render_webinar(webinar: Webinar, *, n_attendees: int | None = None) -> str:
    head = (
        f"{_STATUS_EMOJI[webinar.status]} {webinar.webinar_id} "
        f"[{webinar.status.value}] {webinar.topic}: {webinar.title}\n"
        f"  Scholar: {_mask(webinar.scholar_id)} | "
        f"{webinar.scheduled_at.isoformat()} ({webinar.duration_minutes} min) | "
        f"capacity {webinar.capacity}"
    )
    if n_attendees is not None:
        head += f" | registered {n_attendees}"
    if webinar.status is WebinarStatus.CANCELLED:
        head += f"\n  Cancelled: {webinar.cancelled_reason}"
    return head
