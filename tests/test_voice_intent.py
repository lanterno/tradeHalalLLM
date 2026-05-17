"""Tests for `halal_trader.web.voice_intent` (Wave 5.E).

Covers: voice intent classification, synonym matching, destructive
intent confirmation gate, expiry, no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.voice_intent import (
    ConfirmationExpiredError,
    IntentRecognition,
    NotDestructiveError,
    VoiceIntent,
    classify_intent,
    confirm_recognition,
    is_destructive,
    is_executable,
    recognize,
    render_recognition,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_voice_intent_string_values_pinned() -> None:
    assert VoiceIntent.HALT.value == "halt"
    assert VoiceIntent.RESUME.value == "resume"
    assert VoiceIntent.STATUS.value == "status"
    assert VoiceIntent.DRAWDOWN_QUERY.value == "drawdown_query"
    assert VoiceIntent.HALT_AND_CLOSE_ALL.value == "halt_and_close_all"
    assert VoiceIntent.UNKNOWN.value == "unknown"


# --------------------------- classify_intent: HALT ---------------------------


def test_classify_halt_basic() -> None:
    assert classify_intent("halt the bot") is VoiceIntent.HALT


def test_classify_stop_bot() -> None:
    assert classify_intent("stop the bot") is VoiceIntent.HALT


def test_classify_kill_bot() -> None:
    assert classify_intent("kill the bot") is VoiceIntent.HALT


def test_classify_pause_bot() -> None:
    assert classify_intent("pause the bot") is VoiceIntent.HALT


def test_classify_halt_trading() -> None:
    assert classify_intent("halt trading") is VoiceIntent.HALT


def test_classify_stop_trading() -> None:
    assert classify_intent("stop trading now") is VoiceIntent.HALT


def test_classify_halt_with_punctuation() -> None:
    """Pin: "halt!" and "halt." both match."""

    assert classify_intent("halt the bot!") is VoiceIntent.HALT
    assert classify_intent("Stop the bot.") is VoiceIntent.HALT


# --------------------------- classify_intent: HALT_AND_CLOSE_ALL -------------


def test_classify_halt_and_close_all() -> None:
    assert classify_intent("halt and close all positions") is VoiceIntent.HALT_AND_CLOSE_ALL


def test_classify_stop_and_close_all() -> None:
    assert classify_intent("stop the bot and close all positions") is VoiceIntent.HALT_AND_CLOSE_ALL


def test_classify_emergency_exit() -> None:
    assert classify_intent("emergency exit") is VoiceIntent.HALT_AND_CLOSE_ALL


def test_classify_liquidate_everything() -> None:
    assert classify_intent("liquidate everything now") is VoiceIntent.HALT_AND_CLOSE_ALL


def test_classify_close_all_takes_priority_over_halt() -> None:
    """Pin: HALT_AND_CLOSE_ALL is matched before HALT.

    "halt and close all" must NOT classify as HALT, because the
    operator's intent is to also close positions.
    """

    result = classify_intent("halt and close all positions")
    assert result is VoiceIntent.HALT_AND_CLOSE_ALL
    assert result is not VoiceIntent.HALT


# --------------------------- classify_intent: RESUME -------------------------


def test_classify_resume_bot() -> None:
    assert classify_intent("resume the bot") is VoiceIntent.RESUME


def test_classify_resume_trading() -> None:
    assert classify_intent("resume trading") is VoiceIntent.RESUME


def test_classify_start_bot() -> None:
    assert classify_intent("start the bot") is VoiceIntent.RESUME


def test_classify_unhalt() -> None:
    assert classify_intent("unhalt") is VoiceIntent.RESUME


def test_classify_continue_trading() -> None:
    assert classify_intent("continue trading") is VoiceIntent.RESUME


# --------------------------- classify_intent: STATUS -------------------------


def test_classify_status() -> None:
    assert classify_intent("status") is VoiceIntent.STATUS


def test_classify_how_doing() -> None:
    assert classify_intent("how is the bot doing") is VoiceIntent.STATUS


def test_classify_what_running() -> None:
    assert classify_intent("what is running right now") is VoiceIntent.STATUS


def test_classify_current_state() -> None:
    assert classify_intent("show current state") is VoiceIntent.STATUS


# --------------------------- classify_intent: DRAWDOWN_QUERY -----------------


def test_classify_drawdown() -> None:
    assert classify_intent("drawdown") is VoiceIntent.DRAWDOWN_QUERY


def test_classify_how_down_today() -> None:
    assert classify_intent("how down are we today") is VoiceIntent.DRAWDOWN_QUERY


def test_classify_how_much_lost() -> None:
    assert classify_intent("how much have we lost") is VoiceIntent.DRAWDOWN_QUERY


def test_classify_down_today() -> None:
    assert classify_intent("are we down today") is VoiceIntent.DRAWDOWN_QUERY


def test_classify_performance_today() -> None:
    assert classify_intent("performance today") is VoiceIntent.DRAWDOWN_QUERY


# --------------------------- classify_intent: UNKNOWN ------------------------


def test_classify_empty_string() -> None:
    assert classify_intent("") is VoiceIntent.UNKNOWN


def test_classify_whitespace() -> None:
    assert classify_intent("   ") is VoiceIntent.UNKNOWN


def test_classify_random_text() -> None:
    """Pin: text that doesn't match any pattern returns UNKNOWN."""

    assert classify_intent("hello world") is VoiceIntent.UNKNOWN


def test_classify_partial_match_doesnt_match() -> None:
    """Pin: "halt" alone (without bot/trading) doesn't match HALT.

    We require both "halt" AND a target word to avoid accidental
    triggering on a fragment like "halt? what does that mean".
    """

    # "halt" without "bot" or "trading" doesn't match HALT
    assert classify_intent("halt") is VoiceIntent.UNKNOWN


def test_classify_case_insensitive() -> None:
    """Pin: matching is case-insensitive."""

    assert classify_intent("HALT THE BOT") is VoiceIntent.HALT
    assert classify_intent("Stop The Bot") is VoiceIntent.HALT


# --------------------------- is_destructive ----------------------------------


def test_halt_is_destructive() -> None:
    assert is_destructive(VoiceIntent.HALT) is True


def test_halt_and_close_all_is_destructive() -> None:
    assert is_destructive(VoiceIntent.HALT_AND_CLOSE_ALL) is True


def test_resume_is_not_destructive() -> None:
    """Pin: RESUME is recovery, not destructive — no confirmation needed."""

    assert is_destructive(VoiceIntent.RESUME) is False


def test_status_is_not_destructive() -> None:
    assert is_destructive(VoiceIntent.STATUS) is False


def test_drawdown_query_is_not_destructive() -> None:
    assert is_destructive(VoiceIntent.DRAWDOWN_QUERY) is False


def test_unknown_is_not_destructive() -> None:
    assert is_destructive(VoiceIntent.UNKNOWN) is False


# --------------------------- IntentRecognition validation --------------------


def test_recognition_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="recognition_id"):
        IntentRecognition(
            recognition_id="",
            intent=VoiceIntent.HALT,
            raw_text="halt the bot",
            recognized_at=T0,
            expires_at=T0 + timedelta(seconds=10),
        )


def test_recognition_rejects_empty_raw_text() -> None:
    with pytest.raises(ValueError, match="raw_text"):
        IntentRecognition(
            recognition_id="r1",
            intent=VoiceIntent.HALT,
            raw_text="",
            recognized_at=T0,
            expires_at=T0 + timedelta(seconds=10),
        )


def test_recognition_rejects_naive_recognized_at() -> None:
    with pytest.raises(ValueError, match="recognized_at"):
        IntentRecognition(
            recognition_id="r1",
            intent=VoiceIntent.HALT,
            raw_text="halt",
            recognized_at=datetime(2026, 5, 1),
            expires_at=T0 + timedelta(seconds=10),
        )


def test_recognition_rejects_naive_expires_at() -> None:
    with pytest.raises(ValueError, match="expires_at"):
        IntentRecognition(
            recognition_id="r1",
            intent=VoiceIntent.HALT,
            raw_text="halt",
            recognized_at=T0,
            expires_at=datetime(2026, 5, 1, 12, 0, 10),
        )


def test_recognition_rejects_expires_before_recognized() -> None:
    """Pin: expires_at must be in the future."""

    with pytest.raises(ValueError, match="expires_at"):
        IntentRecognition(
            recognition_id="r1",
            intent=VoiceIntent.HALT,
            raw_text="halt",
            recognized_at=T0,
            expires_at=T0 - timedelta(seconds=1),
        )


def test_recognition_is_frozen() -> None:
    r = IntentRecognition(
        recognition_id="r1",
        intent=VoiceIntent.HALT,
        raw_text="halt the bot",
        recognized_at=T0,
        expires_at=T0 + timedelta(seconds=10),
    )
    with pytest.raises(FrozenInstanceError):
        r.confirmed_at = T0  # type: ignore[misc]


# --------------------------- recognize ---------------------------------------


def test_recognize_classifies_halt() -> None:
    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    assert r.intent is VoiceIntent.HALT
    assert r.confirmed_at is None
    assert r.expires_at == T0 + timedelta(seconds=10)


def test_recognize_classifies_unknown() -> None:
    r = recognize(recognition_id="r1", raw_text="hello world", now=T0)
    assert r.intent is VoiceIntent.UNKNOWN


def test_recognize_custom_window() -> None:
    r = recognize(
        recognition_id="r1",
        raw_text="halt the bot",
        now=T0,
        confirmation_window=timedelta(seconds=30),
    )
    assert r.expires_at == T0 + timedelta(seconds=30)


def test_recognize_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="recognition_id"):
        recognize(recognition_id="", raw_text="halt", now=T0)


def test_recognize_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="raw_text"):
        recognize(recognition_id="r1", raw_text="", now=T0)


def test_recognize_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        recognize(
            recognition_id="r1",
            raw_text="halt",
            now=datetime(2026, 5, 1),
        )


def test_recognize_rejects_zero_window() -> None:
    with pytest.raises(ValueError, match="confirmation_window"):
        recognize(
            recognition_id="r1",
            raw_text="halt",
            now=T0,
            confirmation_window=timedelta(0),
        )


# --------------------------- confirm_recognition -----------------------------


def test_confirm_destructive_within_window() -> None:
    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    confirmed = confirm_recognition(r, now=T0 + timedelta(seconds=5))
    assert confirmed.confirmed_at == T0 + timedelta(seconds=5)


def test_confirm_at_boundary_succeeds() -> None:
    """Pin: confirmation at exactly expires_at is allowed (not strictly after)."""

    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    confirmed = confirm_recognition(r, now=r.expires_at)
    assert confirmed.confirmed_at == r.expires_at


def test_confirm_after_window_rejected() -> None:
    """Pin: confirmation past expires_at raises ConfirmationExpiredError."""

    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    with pytest.raises(ConfirmationExpiredError) as exc_info:
        confirm_recognition(r, now=T0 + timedelta(seconds=15))
    assert exc_info.value.recognition_id == "r1"


def test_confirm_non_destructive_rejected() -> None:
    """Pin: non-destructive intents don't need confirmation."""

    r = recognize(recognition_id="r1", raw_text="status", now=T0)
    with pytest.raises(NotDestructiveError) as exc_info:
        confirm_recognition(r, now=T0)
    assert exc_info.value.intent is VoiceIntent.STATUS


def test_confirm_already_confirmed_rejected() -> None:
    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    r = confirm_recognition(r, now=T0 + timedelta(seconds=2))
    with pytest.raises(ValueError, match="already confirmed"):
        confirm_recognition(r, now=T0 + timedelta(seconds=3))


def test_confirm_rejects_naive_now() -> None:
    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    with pytest.raises(ValueError, match="now"):
        confirm_recognition(r, now=datetime(2026, 5, 1))


# --------------------------- is_executable ----------------------------------


def test_executable_unknown_is_false() -> None:
    r = recognize(recognition_id="r1", raw_text="hello world", now=T0)
    assert is_executable(r, now=T0) is False


def test_executable_non_destructive_is_true() -> None:
    """Pin: STATUS / RESUME / DRAWDOWN_QUERY are immediately executable."""

    r = recognize(recognition_id="r1", raw_text="status", now=T0)
    assert is_executable(r, now=T0) is True


def test_executable_destructive_unconfirmed_is_false() -> None:
    """Pin: HALT without confirmation is NOT executable."""

    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    assert is_executable(r, now=T0) is False


def test_executable_destructive_confirmed_is_true() -> None:
    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    r = confirm_recognition(r, now=T0 + timedelta(seconds=2))
    assert is_executable(r, now=T0 + timedelta(seconds=3)) is True


def test_executable_destructive_after_expiry_is_false() -> None:
    """Pin: even confirmed, executing after the window closes is disallowed."""

    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    r = confirm_recognition(r, now=T0 + timedelta(seconds=2))
    # Way past expiry
    assert is_executable(r, now=T0 + timedelta(seconds=60)) is False


# --------------------------- render_recognition ------------------------------


def test_render_includes_intent_and_text() -> None:
    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    out = render_recognition(r)
    assert "halt" in out
    assert "halt the bot" in out


def test_render_includes_intent_emoji() -> None:
    """Pin: HALT shows 🛑; HALT_AND_CLOSE_ALL shows 🚨."""

    r_halt = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    r_close = recognize(recognition_id="r2", raw_text="halt and close all positions", now=T0)
    assert "🛑" in render_recognition(r_halt)
    assert "🚨" in render_recognition(r_close)


def test_render_destructive_marker() -> None:
    """Pin: unconfirmed destructive shows 'needs confirm' marker."""

    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    out = render_recognition(r)
    assert "DESTRUCTIVE" in out
    assert "confirm" in out.lower()


def test_render_confirmed_marker() -> None:
    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    r = confirm_recognition(r, now=T0 + timedelta(seconds=2))
    out = render_recognition(r)
    assert "confirmed" in out.lower()


def test_render_no_destructive_marker_for_safe_intent() -> None:
    r = recognize(recognition_id="r1", raw_text="status", now=T0)
    out = render_recognition(r)
    assert "DESTRUCTIVE" not in out


def test_render_no_secret_leak() -> None:
    """Pin: render never includes audio file paths / device IDs."""

    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    out = render_recognition(r)
    assert ".wav" not in out
    assert ".mp3" not in out
    assert "device_id" not in out.lower()
    assert "audio_path" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_status_query_runs_immediately() -> None:
    """Real-world: operator says "status" — bot returns dashboard
    snapshot immediately, no confirmation needed."""

    r = recognize(recognition_id="r1", raw_text="status", now=T0)
    assert r.intent is VoiceIntent.STATUS
    assert is_executable(r, now=T0) is True


def test_e2e_halt_requires_confirmation() -> None:
    """Real-world: operator says "halt the bot" — bot waits for
    "yes confirm" before acting."""

    r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
    assert r.intent is VoiceIntent.HALT
    # Cannot execute yet
    assert is_executable(r, now=T0) is False
    # Operator confirms 3s later
    r = confirm_recognition(r, now=T0 + timedelta(seconds=3))
    # Now executable
    assert is_executable(r, now=T0 + timedelta(seconds=4)) is True


def test_e2e_emergency_exit_classified_correctly() -> None:
    """Real-world: operator says "emergency exit" during incident —
    classified as HALT_AND_CLOSE_ALL, confirmation required."""

    r = recognize(recognition_id="r1", raw_text="emergency exit", now=T0)
    assert r.intent is VoiceIntent.HALT_AND_CLOSE_ALL
    assert is_destructive(r.intent) is True
    assert is_executable(r, now=T0) is False


def test_e2e_misheard_command_returns_unknown() -> None:
    """Real-world: Whisper transcribes garbage — UNKNOWN, no action."""

    r = recognize(recognition_id="r1", raw_text="what time is it", now=T0)
    assert r.intent is VoiceIntent.UNKNOWN
    assert is_executable(r, now=T0) is False


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal recognitions."""

    def build() -> IntentRecognition:
        r = recognize(recognition_id="r1", raw_text="halt the bot", now=T0)
        return confirm_recognition(r, now=T0 + timedelta(seconds=2))

    a = build()
    b = build()
    assert a == b
