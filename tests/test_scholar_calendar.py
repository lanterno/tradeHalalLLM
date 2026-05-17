"""Tests for `halal_trader.halal.scholar_calendar`.

Auxiliary primitive complementing Wave 2.F + Wave 11.B. Covers:
consultation kinds + lifecycle, reminder ladder, annual-audit
confirmation deadline, no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.halal.scholar_calendar import (
    DEFAULT_POLICY,
    CalendarPolicy,
    Consultation,
    ConsultationKind,
    ConsultationStatus,
    StatusTransitionError,
    annual_audit_overdue_for_confirmation,
    cancel_consultation,
    complete_consultation,
    confirm_consultation,
    filter_due_for_reminder,
    is_due_for_reminder,
    render_consultation,
    schedule_consultation,
    upcoming,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_consultation_kind_string_values_pinned() -> None:
    assert ConsultationKind.ANNUAL_AUDIT.value == "annual_audit"
    assert ConsultationKind.QUARTERLY_REVIEW.value == "quarterly_review"
    assert ConsultationKind.AD_HOC.value == "ad_hoc"


def test_consultation_status_string_values_pinned() -> None:
    assert ConsultationStatus.SCHEDULED.value == "scheduled"
    assert ConsultationStatus.CONFIRMED.value == "confirmed"
    assert ConsultationStatus.COMPLETED.value == "completed"
    assert ConsultationStatus.CANCELLED.value == "cancelled"


# --------------------------- CalendarPolicy ----------------------------------


def test_default_policy_pins() -> None:
    """Pin: 30d/7d/1d reminder ladder + 60d annual confirm window."""

    assert DEFAULT_POLICY.reminder_lead_times == (
        timedelta(days=30),
        timedelta(days=7),
        timedelta(days=1),
    )
    assert DEFAULT_POLICY.annual_confirm_lead_time == timedelta(days=60)


def test_policy_rejects_empty_lead_times() -> None:
    with pytest.raises(ValueError, match="reminder_lead_times"):
        CalendarPolicy(reminder_lead_times=())


def test_policy_rejects_zero_lead() -> None:
    with pytest.raises(ValueError, match="lead time"):
        CalendarPolicy(reminder_lead_times=(timedelta(0),))


def test_policy_rejects_non_descending_leads() -> None:
    """Pin: lead times must be in descending order (longest first)."""

    with pytest.raises(ValueError, match="descending"):
        CalendarPolicy(
            reminder_lead_times=(timedelta(days=7), timedelta(days=30)),
        )


def test_policy_rejects_zero_annual_confirm() -> None:
    with pytest.raises(ValueError, match="annual_confirm_lead_time"):
        CalendarPolicy(annual_confirm_lead_time=timedelta(0))


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.annual_confirm_lead_time = timedelta(days=1)  # type: ignore[misc]


# --------------------------- Consultation validation -------------------------


def _consultation(**overrides: object) -> Consultation:
    base: dict[str, object] = {
        "consultation_id": "c1",
        "kind": ConsultationKind.QUARTERLY_REVIEW,
        "scholar_handle": "mufti_x",
        "scheduled_at": T0 + timedelta(days=14),
        "status": ConsultationStatus.SCHEDULED,
        "last_status_at": T0,
        "topic": "Q2 portfolio review",
    }
    base.update(overrides)
    return Consultation(**base)  # type: ignore[arg-type]


def test_consultation_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="consultation_id"):
        _consultation(consultation_id="")


def test_consultation_rejects_empty_scholar_handle() -> None:
    with pytest.raises(ValueError, match="scholar_handle"):
        _consultation(scholar_handle="")


def test_consultation_rejects_empty_topic() -> None:
    with pytest.raises(ValueError, match="topic"):
        _consultation(topic="")


def test_consultation_rejects_naive_scheduled_at() -> None:
    with pytest.raises(ValueError, match="scheduled_at"):
        _consultation(scheduled_at=datetime(2026, 5, 1))


def test_consultation_completed_requires_minutes_url() -> None:
    """Pin: COMPLETED status requires minutes_url for audit trail."""

    with pytest.raises(ValueError, match="minutes_url"):
        _consultation(
            status=ConsultationStatus.COMPLETED,
            minutes_url="",
        )


def test_consultation_non_completed_must_not_have_minutes_url() -> None:
    """Pin: minutes_url is COMPLETED-only field."""

    with pytest.raises(ValueError, match="minutes_url"):
        _consultation(
            status=ConsultationStatus.SCHEDULED,
            minutes_url="https://minutes.example",
        )


def test_consultation_is_frozen() -> None:
    c = _consultation()
    with pytest.raises(FrozenInstanceError):
        c.topic = "other"  # type: ignore[misc]


# --------------------------- schedule_consultation ---------------------------


def test_schedule_basic() -> None:
    c = schedule_consultation(
        consultation_id="c1",
        kind=ConsultationKind.QUARTERLY_REVIEW,
        scholar_handle="mufti_x",
        scheduled_at=T0 + timedelta(days=14),
        topic="Q2 review",
        now=T0,
    )
    assert c.status is ConsultationStatus.SCHEDULED
    assert c.minutes_url == ""


def test_schedule_rejects_past_date() -> None:
    """Pin: scheduling for a past date is operator error."""

    with pytest.raises(ValueError, match="scheduled_at"):
        schedule_consultation(
            consultation_id="c1",
            kind=ConsultationKind.QUARTERLY_REVIEW,
            scholar_handle="x",
            scheduled_at=T0 - timedelta(days=1),
            topic="x",
            now=T0,
        )


def test_schedule_rejects_now_as_scheduled_at() -> None:
    """Pin: scheduling at exactly `now` is also rejected."""

    with pytest.raises(ValueError, match="scheduled_at"):
        schedule_consultation(
            consultation_id="c1",
            kind=ConsultationKind.QUARTERLY_REVIEW,
            scholar_handle="x",
            scheduled_at=T0,
            topic="x",
            now=T0,
        )


def test_schedule_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        schedule_consultation(
            consultation_id="c1",
            kind=ConsultationKind.AD_HOC,
            scholar_handle="x",
            scheduled_at=T0 + timedelta(days=1),
            topic="x",
            now=datetime(2026, 5, 1),
        )


def test_schedule_rejects_naive_scheduled_at() -> None:
    with pytest.raises(ValueError, match="scheduled_at"):
        schedule_consultation(
            consultation_id="c1",
            kind=ConsultationKind.AD_HOC,
            scholar_handle="x",
            scheduled_at=datetime(2026, 6, 1),  # naive
            topic="x",
            now=T0,
        )


# --------------------------- confirm_consultation ----------------------------


def test_confirm_from_scheduled() -> None:
    c = _consultation(status=ConsultationStatus.SCHEDULED)
    c = confirm_consultation(c, now=T0 + timedelta(days=1))
    assert c.status is ConsultationStatus.CONFIRMED


def test_confirm_already_confirmed_rejected() -> None:
    c = _consultation(status=ConsultationStatus.CONFIRMED)
    with pytest.raises(StatusTransitionError):
        confirm_consultation(c, now=T0)


def test_confirm_completed_rejected() -> None:
    """Pin: cannot revert from COMPLETED."""

    c = _consultation(
        status=ConsultationStatus.COMPLETED,
        minutes_url="https://m.example",
    )
    with pytest.raises(StatusTransitionError):
        confirm_consultation(c, now=T0)


# --------------------------- complete_consultation ---------------------------


def test_complete_from_confirmed() -> None:
    c = _consultation(status=ConsultationStatus.CONFIRMED)
    c = complete_consultation(c, now=T0 + timedelta(days=15), minutes_url="https://minutes.example")
    assert c.status is ConsultationStatus.COMPLETED
    assert c.minutes_url == "https://minutes.example"


def test_complete_requires_minutes_url() -> None:
    c = _consultation(status=ConsultationStatus.CONFIRMED)
    with pytest.raises(ValueError, match="minutes_url"):
        complete_consultation(c, now=T0, minutes_url="")


def test_complete_skip_from_scheduled_rejected() -> None:
    """Pin: cannot skip CONFIRMED → COMPLETED."""

    c = _consultation(status=ConsultationStatus.SCHEDULED)
    with pytest.raises(StatusTransitionError):
        complete_consultation(c, now=T0, minutes_url="https://m.example")


# --------------------------- cancel_consultation -----------------------------


def test_cancel_from_scheduled() -> None:
    c = _consultation(status=ConsultationStatus.SCHEDULED)
    c = cancel_consultation(c, now=T0 + timedelta(days=1))
    assert c.status is ConsultationStatus.CANCELLED


def test_cancel_from_confirmed() -> None:
    c = _consultation(status=ConsultationStatus.CONFIRMED)
    c = cancel_consultation(c, now=T0 + timedelta(days=1))
    assert c.status is ConsultationStatus.CANCELLED


def test_cancel_completed_rejected() -> None:
    """Pin: cannot cancel a completed consultation."""

    c = _consultation(
        status=ConsultationStatus.COMPLETED,
        minutes_url="https://m.example",
    )
    with pytest.raises(StatusTransitionError):
        cancel_consultation(c, now=T0)


def test_cancel_already_cancelled_rejected() -> None:
    c = _consultation(status=ConsultationStatus.CANCELLED)
    with pytest.raises(StatusTransitionError):
        cancel_consultation(c, now=T0)


# --------------------------- is_due_for_reminder ----------------------------


def test_due_30_days_before() -> None:
    """Pin: at 30d threshold, reminder due if not yet sent."""

    c = _consultation(
        scheduled_at=T0 + timedelta(days=30),  # exactly 30 days out
        status=ConsultationStatus.SCHEDULED,
    )
    assert is_due_for_reminder(c, now=T0) is True


def test_not_due_31_days_before() -> None:
    """Pin: 31d out is before any reminder threshold."""

    c = _consultation(
        scheduled_at=T0 + timedelta(days=31),
        status=ConsultationStatus.SCHEDULED,
    )
    assert is_due_for_reminder(c, now=T0) is False


def test_due_7_days_before_if_30d_already_sent() -> None:
    """Pin: 30d reminder already sent; 7d threshold triggers next."""

    c = _consultation(
        scheduled_at=T0 + timedelta(days=7),  # exactly 7 days out
        status=ConsultationStatus.SCHEDULED,
    )
    last_reminder = T0 - timedelta(days=10)  # before 7d threshold
    assert is_due_for_reminder(c, now=T0, last_reminder_at=last_reminder) is True


def test_not_due_if_reminder_at_or_past_active_threshold() -> None:
    """Pin: reminder already sent AT or past the active threshold → not due.

    Scheduled at T0+7d → 7d threshold at T0. A reminder sent at exactly
    T0 (or later) means we already fired for the 7d threshold; don't
    fire again.
    """

    c = _consultation(
        scheduled_at=T0 + timedelta(days=7),
        status=ConsultationStatus.SCHEDULED,
    )
    # Reminder sent at the 7d threshold itself — already fired
    last_reminder = T0
    assert is_due_for_reminder(c, now=T0, last_reminder_at=last_reminder) is False


def test_due_1_day_before() -> None:
    c = _consultation(
        scheduled_at=T0 + timedelta(days=1),
        status=ConsultationStatus.SCHEDULED,
    )
    assert is_due_for_reminder(c, now=T0) is True


def test_completed_not_due() -> None:
    """Pin: COMPLETED consultations don't fire reminders."""

    c = _consultation(
        scheduled_at=T0 + timedelta(days=1),
        status=ConsultationStatus.COMPLETED,
        minutes_url="https://m.example",
    )
    assert is_due_for_reminder(c, now=T0) is False


def test_cancelled_not_due() -> None:
    """Pin: CANCELLED consultations don't fire reminders."""

    c = _consultation(
        scheduled_at=T0 + timedelta(days=1),
        status=ConsultationStatus.CANCELLED,
    )
    assert is_due_for_reminder(c, now=T0) is False


def test_due_for_reminder_naive_now_rejected() -> None:
    c = _consultation()
    with pytest.raises(ValueError, match="now"):
        is_due_for_reminder(c, now=datetime(2026, 5, 1))


# --------------------------- annual_audit_overdue_for_confirmation -----------


def test_annual_audit_overdue_at_60_day_boundary() -> None:
    """Pin: at exactly 60 days out, an unconfirmed annual audit is overdue."""

    c = _consultation(
        kind=ConsultationKind.ANNUAL_AUDIT,
        status=ConsultationStatus.SCHEDULED,
        scheduled_at=T0 + timedelta(days=60),
    )
    assert annual_audit_overdue_for_confirmation(c, now=T0) is True


def test_annual_audit_not_overdue_at_61_days() -> None:
    """Pin: 61 days out is just before the deadline."""

    c = _consultation(
        kind=ConsultationKind.ANNUAL_AUDIT,
        status=ConsultationStatus.SCHEDULED,
        scheduled_at=T0 + timedelta(days=61),
    )
    assert annual_audit_overdue_for_confirmation(c, now=T0) is False


def test_quarterly_review_never_overdue_for_annual_check() -> None:
    """Pin: only ANNUAL_AUDIT triggers this check."""

    c = _consultation(
        kind=ConsultationKind.QUARTERLY_REVIEW,
        status=ConsultationStatus.SCHEDULED,
        scheduled_at=T0 + timedelta(days=10),  # very close
    )
    assert annual_audit_overdue_for_confirmation(c, now=T0) is False


def test_confirmed_annual_audit_not_overdue() -> None:
    """Pin: only SCHEDULED status flags overdue (confirmed is fine)."""

    c = _consultation(
        kind=ConsultationKind.ANNUAL_AUDIT,
        status=ConsultationStatus.CONFIRMED,
        scheduled_at=T0 + timedelta(days=30),
    )
    assert annual_audit_overdue_for_confirmation(c, now=T0) is False


# --------------------------- filter_due_for_reminder -------------------------


def test_filter_due_returns_sorted_by_scheduled() -> None:
    c1 = _consultation(
        consultation_id="c1",
        scheduled_at=T0 + timedelta(days=7),
        status=ConsultationStatus.SCHEDULED,
    )
    c2 = _consultation(
        consultation_id="c2",
        scheduled_at=T0 + timedelta(days=1),
        status=ConsultationStatus.SCHEDULED,
    )
    due = filter_due_for_reminder([c1, c2], now=T0)
    ids = [c.consultation_id for c in due]
    # c2 is sooner, so it sorts first
    assert ids == ["c2", "c1"]


def test_filter_due_with_last_reminders() -> None:
    """Pin: last_reminders dict suppresses already-fired reminders.

    c1 has no prior reminder → due (30d threshold of T0+7d is at T0-23d,
    crossed). c2 has a reminder at the active 1d threshold (T0 for
    scheduled T0+1d) → already fired, not due.
    """

    c1 = _consultation(
        consultation_id="c1",
        scheduled_at=T0 + timedelta(days=7),
        status=ConsultationStatus.SCHEDULED,
    )
    c2 = _consultation(
        consultation_id="c2",
        scheduled_at=T0 + timedelta(days=1),
        status=ConsultationStatus.SCHEDULED,
    )
    last = {
        "c2": T0,  # reminder fired at the 1d threshold itself
    }
    due = filter_due_for_reminder([c1, c2], now=T0, last_reminders=last)
    ids = {c.consultation_id for c in due}
    # c1 still due (no prior reminder); c2 NOT due (last_reminder at threshold)
    assert ids == {"c1"}


def test_filter_due_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="now"):
        filter_due_for_reminder([], now=datetime(2026, 5, 1))


# --------------------------- upcoming ----------------------------------------


def test_upcoming_within_90_day_horizon() -> None:
    soon = _consultation(
        consultation_id="soon",
        scheduled_at=T0 + timedelta(days=14),
        status=ConsultationStatus.SCHEDULED,
    )
    far = _consultation(
        consultation_id="far",
        scheduled_at=T0 + timedelta(days=120),
        status=ConsultationStatus.SCHEDULED,
    )
    result = upcoming([soon, far], now=T0)
    ids = {c.consultation_id for c in result}
    assert ids == {"soon"}


def test_upcoming_excludes_completed() -> None:
    """Pin: upcoming shows only SCHEDULED + CONFIRMED."""

    completed = _consultation(
        consultation_id="completed",
        scheduled_at=T0 + timedelta(days=14),
        status=ConsultationStatus.COMPLETED,
        minutes_url="https://m.example",
    )
    cancelled = _consultation(
        consultation_id="cancelled",
        scheduled_at=T0 + timedelta(days=14),
        status=ConsultationStatus.CANCELLED,
    )
    confirmed = _consultation(
        consultation_id="confirmed",
        scheduled_at=T0 + timedelta(days=14),
        status=ConsultationStatus.CONFIRMED,
    )
    result = upcoming([completed, cancelled, confirmed], now=T0)
    ids = {c.consultation_id for c in result}
    assert ids == {"confirmed"}


def test_upcoming_sorted_ascending() -> None:
    later = _consultation(
        consultation_id="later",
        scheduled_at=T0 + timedelta(days=30),
    )
    sooner = _consultation(
        consultation_id="sooner",
        scheduled_at=T0 + timedelta(days=7),
    )
    result = upcoming([later, sooner], now=T0)
    ids = [c.consultation_id for c in result]
    assert ids == ["sooner", "later"]


def test_upcoming_custom_horizon() -> None:
    """Custom 14-day horizon excludes 30-days-out."""

    far = _consultation(
        scheduled_at=T0 + timedelta(days=30),
    )
    result = upcoming([far], now=T0, horizon=timedelta(days=14))
    assert result == ()


def test_upcoming_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="now"):
        upcoming([], now=datetime(2026, 5, 1))


def test_upcoming_zero_horizon_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        upcoming([], now=T0, horizon=timedelta(0))


# --------------------------- render ------------------------------------------


def test_render_includes_kind_and_status_emoji() -> None:
    c = _consultation(kind=ConsultationKind.ANNUAL_AUDIT)
    out = render_consultation(c)
    assert "📋" in out  # annual audit
    assert "🗓️" in out  # scheduled


def test_render_includes_topic_and_scholar() -> None:
    c = _consultation(scholar_handle="mufti_faraz", topic="REIT screening")
    out = render_consultation(c)
    assert "mufti_faraz" in out
    assert "REIT screening" in out


def test_render_includes_minutes_url_when_completed() -> None:
    c = _consultation(
        status=ConsultationStatus.COMPLETED,
        minutes_url="https://minutes.example/q1",
    )
    out = render_consultation(c)
    assert "https://minutes.example/q1" in out


def test_render_omits_minutes_when_not_completed() -> None:
    c = _consultation(status=ConsultationStatus.SCHEDULED)
    out = render_consultation(c)
    assert "minutes:" not in out


def test_render_no_secret_leak() -> None:
    """Pin: the dataclass deliberately doesn't carry scholar contact
    emails or meeting URLs (operator-side state)."""

    c = _consultation()
    out = render_consultation(c)
    assert "@" not in out
    assert "zoom.us" not in out.lower()
    assert "meet.google" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_annual_audit_full_lifecycle() -> None:
    """Real-world: annual audit scheduled 90d out, confirmed 60d out,
    completed on the day, minutes recorded."""

    audit = schedule_consultation(
        consultation_id="audit_2026",
        kind=ConsultationKind.ANNUAL_AUDIT,
        scholar_handle="mufti_faraz",
        scheduled_at=T0 + timedelta(days=90),
        topic="Annual halal compliance audit 2026",
        now=T0,
    )
    # 30 days later (60 days out): scholar confirms
    confirmed = confirm_consultation(audit, now=T0 + timedelta(days=30))
    assert confirmed.status is ConsultationStatus.CONFIRMED

    # Day of consultation: complete with minutes
    completed = complete_consultation(
        confirmed,
        now=T0 + timedelta(days=90),
        minutes_url="https://docs.halal-trader.dev/audit/2026",
    )
    assert completed.status is ConsultationStatus.COMPLETED


def test_e2e_overdue_annual_audit_caught() -> None:
    """Pin: an annual audit at 50 days out without confirmation flags overdue."""

    audit = schedule_consultation(
        consultation_id="audit",
        kind=ConsultationKind.ANNUAL_AUDIT,
        scholar_handle="mufti_x",
        scheduled_at=T0 + timedelta(days=50),
        topic="Annual audit",
        now=T0,
    )
    assert annual_audit_overdue_for_confirmation(audit, now=T0) is True


def test_e2e_reminder_ladder_fires() -> None:
    """Real-world: 30d → reminder; advance 23 days; 7d → reminder."""

    consultation = schedule_consultation(
        consultation_id="q2_review",
        kind=ConsultationKind.QUARTERLY_REVIEW,
        scholar_handle="mufti_x",
        scheduled_at=T0 + timedelta(days=30),
        topic="Q2 review",
        now=T0,
    )
    # First reminder fires at 30d threshold
    assert is_due_for_reminder(consultation, now=T0) is True
    last_reminder = T0  # operator records it

    # 23 days later: at 7d threshold, second reminder
    next_check = T0 + timedelta(days=23)
    assert is_due_for_reminder(consultation, now=next_check, last_reminder_at=last_reminder) is True


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal consultation states."""

    def build() -> Consultation:
        c = schedule_consultation(
            consultation_id="c1",
            kind=ConsultationKind.QUARTERLY_REVIEW,
            scholar_handle="x",
            scheduled_at=T0 + timedelta(days=14),
            topic="x",
            now=T0,
        )
        return confirm_consultation(c, now=T0 + timedelta(days=1))

    a = build()
    b = build()
    assert a == b
