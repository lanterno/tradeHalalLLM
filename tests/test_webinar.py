"""Tests for education/webinar.py — Round-5 Wave 20.C."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.education.webinar import (
    Attendance,
    QuestionStatus,
    Webinar,
    WebinarStatus,
    answer_question,
    attendance_count,
    cancel,
    conclude,
    decline_question,
    defer_question,
    go_live,
    mark_attended,
    open_registration,
    register_attendee,
    render_webinar,
    submit_question,
)


def _webinar(
    webinar_id: str = "W1",
    scholar_id: str = "scholar-ali",
    title: str = "Monthly Halal Q&A",
    topic: str = "Sukuk Structures",
    scheduled_at: datetime = datetime(2026, 6, 15, 18, 0),
    duration_minutes: int = 60,
    capacity: int = 100,
    registration_opens_at: datetime = datetime(2026, 6, 1, 0, 0),
    status: WebinarStatus = WebinarStatus.SCHEDULED,
    cancelled_reason: str = "",
) -> Webinar:
    return Webinar(
        webinar_id=webinar_id,
        scholar_id=scholar_id,
        title=title,
        topic=topic,
        scheduled_at=scheduled_at,
        duration_minutes=duration_minutes,
        capacity=capacity,
        registration_opens_at=registration_opens_at,
        status=status,
        cancelled_reason=cancelled_reason,
    )


# --- Webinar validation -----------------------


def test_webinar_valid():
    w = _webinar()
    assert w.status is WebinarStatus.SCHEDULED


def test_webinar_empty_id_rejected():
    with pytest.raises(ValueError):
        _webinar(webinar_id="")


def test_webinar_long_title_rejected():
    with pytest.raises(ValueError):
        _webinar(title="x" * 300)


def test_webinar_empty_topic_rejected():
    with pytest.raises(ValueError):
        _webinar(topic=" ")


def test_webinar_long_topic_rejected():
    with pytest.raises(ValueError):
        _webinar(topic="x" * 200)


def test_webinar_zero_duration_rejected():
    with pytest.raises(ValueError):
        _webinar(duration_minutes=0)


def test_webinar_long_duration_rejected():
    with pytest.raises(ValueError):
        _webinar(duration_minutes=500)


def test_webinar_zero_capacity_rejected():
    with pytest.raises(ValueError):
        _webinar(capacity=0)


def test_webinar_registration_after_scheduled_rejected():
    with pytest.raises(ValueError):
        _webinar(
            scheduled_at=datetime(2026, 6, 1, 18, 0),
            registration_opens_at=datetime(2026, 7, 1, 0, 0),
        )


def test_webinar_cancelled_without_reason_rejected():
    with pytest.raises(ValueError):
        _webinar(status=WebinarStatus.CANCELLED, cancelled_reason="")


def test_webinar_immutable():
    w = _webinar()
    with pytest.raises(AttributeError):
        w.capacity = 200  # type: ignore[misc]


# --- FSM transitions --------------------------


def test_open_registration_from_scheduled():
    w = _webinar()
    w2 = open_registration(w)
    assert w2.status is WebinarStatus.REGISTRATION_OPEN


def test_open_registration_from_other_rejected():
    w = open_registration(_webinar())
    with pytest.raises(ValueError):
        open_registration(w)


def test_go_live_from_registration_open():
    w = open_registration(_webinar())
    w2 = go_live(w)
    assert w2.status is WebinarStatus.LIVE


def test_go_live_from_scheduled_rejected():
    w = _webinar()
    with pytest.raises(ValueError):
        go_live(w)


def test_conclude_from_live():
    w = go_live(open_registration(_webinar()))
    w2 = conclude(w)
    assert w2.status is WebinarStatus.CONCLUDED


def test_concluded_terminal():
    w = conclude(go_live(open_registration(_webinar())))
    with pytest.raises(ValueError):
        go_live(w)


def test_cancel_from_scheduled():
    w = _webinar()
    c = cancel(w, reason="scholar unavailable")
    assert c.status is WebinarStatus.CANCELLED


def test_cancel_from_registration_open():
    w = open_registration(_webinar())
    c = cancel(w, reason="emergency")
    assert c.status is WebinarStatus.CANCELLED


def test_cancel_from_live():
    w = go_live(open_registration(_webinar()))
    c = cancel(w, reason="technical")
    assert c.status is WebinarStatus.CANCELLED


def test_cancel_from_concluded_rejected():
    w = conclude(go_live(open_registration(_webinar())))
    with pytest.raises(ValueError):
        cancel(w, reason="too late")


def test_cancel_empty_reason_rejected():
    w = _webinar()
    with pytest.raises(ValueError):
        cancel(w, reason=" ")


# --- Attendance ------------------------------


def test_register_basic():
    w = open_registration(_webinar())
    a = register_attendee(
        w,
        [],
        attendance_id="A1",
        user_id="bob",
        registered_at=datetime(2026, 6, 2, 10, 0),
    )
    assert a.user_id == "bob"


def test_register_to_scheduled_rejected():
    w = _webinar()  # SCHEDULED
    with pytest.raises(ValueError):
        register_attendee(
            w,
            [],
            attendance_id="A1",
            user_id="bob",
            registered_at=datetime(2026, 6, 2),
        )


def test_register_to_cancelled_rejected():
    w = cancel(_webinar(), reason="out")
    with pytest.raises(ValueError):
        register_attendee(
            w,
            [],
            attendance_id="A1",
            user_id="bob",
            registered_at=datetime(2026, 6, 2),
        )


def test_register_capacity_enforced():
    w = open_registration(_webinar(capacity=2))
    a1 = register_attendee(
        w,
        [],
        attendance_id="A1",
        user_id="bob",
        registered_at=datetime(2026, 6, 2),
    )
    a2 = register_attendee(
        w,
        [a1],
        attendance_id="A2",
        user_id="charlie",
        registered_at=datetime(2026, 6, 3),
    )
    with pytest.raises(ValueError):
        register_attendee(
            w,
            [a1, a2],
            attendance_id="A3",
            user_id="dave",
            registered_at=datetime(2026, 6, 4),
        )


def test_register_duplicate_user_rejected():
    w = open_registration(_webinar())
    a = register_attendee(
        w,
        [],
        attendance_id="A1",
        user_id="bob",
        registered_at=datetime(2026, 6, 2),
    )
    with pytest.raises(ValueError):
        register_attendee(
            w,
            [a],
            attendance_id="A2",
            user_id="bob",
            registered_at=datetime(2026, 6, 3),
        )


def test_mark_attended():
    a = Attendance(
        attendance_id="A1",
        webinar_id="W1",
        user_id="bob",
        registered_at=datetime(2026, 6, 2),
    )
    a2 = mark_attended(a)
    assert a2.attended is True


def test_mark_attended_idempotent():
    a = Attendance(
        attendance_id="A1",
        webinar_id="W1",
        user_id="bob",
        registered_at=datetime(2026, 6, 2),
        attended=True,
    )
    a2 = mark_attended(a)
    assert a2.attended is True


def test_attendance_count():
    records = [
        Attendance(
            attendance_id="A1",
            webinar_id="W1",
            user_id="bob",
            registered_at=datetime(2026, 6, 2),
        ),
        Attendance(
            attendance_id="A2",
            webinar_id="W2",
            user_id="bob",
            registered_at=datetime(2026, 6, 3),
        ),
    ]
    assert attendance_count("W1", records) == 1


# --- Question submission --------------------


def test_submit_question_basic():
    w = open_registration(_webinar())
    q = submit_question(
        w,
        question_id="Q1",
        asker_id="bob",
        text="Is Sukuk-Murabaha tradable secondary?",
        submitted_at=datetime(2026, 6, 5),
    )
    assert q.status is QuestionStatus.SUBMITTED


def test_submit_question_long_text_rejected():
    w = open_registration(_webinar())
    with pytest.raises(ValueError):
        submit_question(
            w,
            question_id="Q1",
            asker_id="bob",
            text="x" * 1500,
            submitted_at=datetime(2026, 6, 5),
        )


def test_submit_question_to_concluded_rejected():
    w = conclude(go_live(open_registration(_webinar())))
    with pytest.raises(ValueError):
        submit_question(
            w,
            question_id="Q1",
            asker_id="bob",
            text="Q?",
            submitted_at=datetime(2026, 7, 1),
        )


def test_submit_question_moderation_blocks():
    w = open_registration(_webinar())
    with pytest.raises(ValueError):
        submit_question(
            w,
            question_id="Q1",
            asker_id="bob",
            text="Inappropriate text",
            submitted_at=datetime(2026, 6, 5),
            is_text_acceptable=lambda t: "Inappropriate" not in t,
        )


def test_submit_question_to_live_allowed():
    w = go_live(open_registration(_webinar()))
    q = submit_question(
        w,
        question_id="Q1",
        asker_id="bob",
        text="Live Q?",
        submitted_at=datetime(2026, 6, 15, 18, 5),
    )
    assert q.status is QuestionStatus.SUBMITTED


# --- Question lifecycle ---------------------


def test_answer_question_basic():
    w = open_registration(_webinar())
    q = submit_question(
        w,
        question_id="Q1",
        asker_id="bob",
        text="Q?",
        submitted_at=datetime(2026, 6, 5),
    )
    q2 = answer_question(q, response="Here is the answer...")
    assert q2.status is QuestionStatus.ANSWERED


def test_answer_empty_response_rejected():
    w = open_registration(_webinar())
    q = submit_question(
        w,
        question_id="Q1",
        asker_id="bob",
        text="Q?",
        submitted_at=datetime(2026, 6, 5),
    )
    with pytest.raises(ValueError):
        answer_question(q, response=" ")


def test_answer_already_answered_rejected():
    w = open_registration(_webinar())
    q = answer_question(
        submit_question(
            w,
            question_id="Q1",
            asker_id="bob",
            text="Q?",
            submitted_at=datetime(2026, 6, 5),
        ),
        response="ans",
    )
    with pytest.raises(ValueError):
        answer_question(q, response="another")


def test_defer_question_basic():
    w = open_registration(_webinar())
    q = submit_question(
        w,
        question_id="Q1",
        asker_id="bob",
        text="Q?",
        submitted_at=datetime(2026, 6, 5),
    )
    q2 = defer_question(q)
    assert q2.status is QuestionStatus.DEFERRED


def test_defer_already_answered_rejected():
    w = open_registration(_webinar())
    q = answer_question(
        submit_question(
            w,
            question_id="Q1",
            asker_id="bob",
            text="Q?",
            submitted_at=datetime(2026, 6, 5),
        ),
        response="ans",
    )
    with pytest.raises(ValueError):
        defer_question(q)


def test_decline_question():
    w = open_registration(_webinar())
    q = submit_question(
        w,
        question_id="Q1",
        asker_id="bob",
        text="Q?",
        submitted_at=datetime(2026, 6, 5),
    )
    q2 = decline_question(q)
    assert q2.status is QuestionStatus.DECLINED


# --- Render -------------------------------


def test_render_webinar_status_emoji():
    w = _webinar()
    out = render_webinar(w)
    assert "📅" in out


def test_render_webinar_no_secret_leak():
    w = _webinar(scholar_id="scholar-ali@example.com")
    out = render_webinar(w)
    assert "scholar-ali@example.com" not in out


def test_render_webinar_with_attendee_count():
    w = _webinar()
    out = render_webinar(w, n_attendees=42)
    assert "registered 42" in out


def test_render_cancelled_shows_reason():
    w = cancel(_webinar(), reason="scholar travelling")
    out = render_webinar(w)
    assert "Cancelled" in out
    assert "travelling" in out
