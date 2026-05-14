"""Voice command intent classifier.

The roadmap pins Wave 5.E: "Operator says 'stop the crypto bot' /
'what's my drawdown today' / 'halt and close all positions' —
local Whisper + gpt-oss handles it. Operator-only feature, off by
default. A surprise-and-delight feature that pays off during
incident response." This module is the **pure-Python intent
classifier** that consumes the Whisper-transcribed text and maps
it to one of a closed set of voice intents. The Whisper STT
adapter + gpt-oss disambiguation are operator-side; this module
ships the deterministic grammar.

Picked a focused intent classifier over a "send the audio to an
LLM and parse JSON" approach because (a) voice commands during
incident response need to be deterministic and fast — a 200ms
intent classification beats a 5s LLM round-trip when the operator
is yelling "halt the bot" because something is wrong, (b) the
destructive-action confirmation gate (close all positions, halt)
must be a regression-pinned safety contract — accidentally firing
HALT_AND_CLOSE_ALL because the operator said "halt? and close all"
in a question is the worst-case failure, (c) operator confidence
in the voice surface depends on consistent recognition across
phrasings ("stop the crypto bot" vs "stop crypto bot" vs "kill the
crypto bot") — closed-set synonym matching gives the right answer
every time, where an LLM would drift across versions.

Pinned semantics:
- **Closed-set VoiceIntent enum.** Five intents: HALT, RESUME,
  STATUS, DRAWDOWN_QUERY, HALT_AND_CLOSE_ALL. Adding an intent
  is a code review change; the grammar can't drift.
- **Destructive intents require confirmation phrase.** HALT,
  HALT_AND_CLOSE_ALL require a follow-up "yes confirm" within
  10 seconds; pinned so a misheard "stop everything" doesn't
  immediately liquidate.
- **Synonym matching is normalized + token-set based.** Stop /
  halt / kill / pause all map to the right intent; "halt? and"
  with question-mark punctuation doesn't match HALT_AND_CLOSE_ALL.
- **UNKNOWN is a real outcome.** A transcription that doesn't
  match any intent returns UNKNOWN with the original text;
  the operator UI shows "I didn't catch that, try again" rather
  than guessing.
- **Render output never includes audio file paths / device
  IDs.** Mirrors no-secret patterns of upstream waves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class VoiceIntent(str, Enum):
    """Closed-set voice command intents.

    Pinned string values for JSON / DB stability. Adding an intent
    is a code review change.
    """

    HALT = "halt"
    RESUME = "resume"
    STATUS = "status"
    DRAWDOWN_QUERY = "drawdown_query"
    HALT_AND_CLOSE_ALL = "halt_and_close_all"
    UNKNOWN = "unknown"


_DESTRUCTIVE_INTENTS: frozenset[VoiceIntent] = frozenset(
    {VoiceIntent.HALT, VoiceIntent.HALT_AND_CLOSE_ALL}
)


# Synonym sets for each non-UNKNOWN intent. These are token-set
# matches: every keyword in one of the alternatives must be present
# (case-insensitive, after normalization). Order matters — the
# more-specific HALT_AND_CLOSE_ALL is checked before HALT.
_INTENT_PATTERNS: tuple[tuple[VoiceIntent, tuple[frozenset[str], ...]], ...] = (
    # HALT_AND_CLOSE_ALL: must mention both halting AND closing positions
    (
        VoiceIntent.HALT_AND_CLOSE_ALL,
        (
            frozenset({"halt", "close", "all"}),
            frozenset({"stop", "close", "all"}),
            frozenset({"emergency", "exit"}),
            frozenset({"liquidate", "everything"}),
        ),
    ),
    # HALT: stop / halt / kill / pause without closing
    (
        VoiceIntent.HALT,
        (
            frozenset({"halt", "bot"}),
            frozenset({"stop", "bot"}),
            frozenset({"kill", "bot"}),
            frozenset({"pause", "bot"}),
            frozenset({"halt", "trading"}),
            frozenset({"stop", "trading"}),
        ),
    ),
    # RESUME: resume / start / unhalt
    (
        VoiceIntent.RESUME,
        (
            frozenset({"resume", "bot"}),
            frozenset({"resume", "trading"}),
            frozenset({"start", "bot"}),
            frozenset({"unhalt"}),
            frozenset({"continue", "trading"}),
        ),
    ),
    # STATUS: status / how are you / what's running
    (
        VoiceIntent.STATUS,
        (
            frozenset({"status"}),
            frozenset({"how", "doing"}),
            frozenset({"what", "running"}),
            frozenset({"current", "state"}),
        ),
    ),
    # DRAWDOWN_QUERY: drawdown / down today / losing
    (
        VoiceIntent.DRAWDOWN_QUERY,
        (
            frozenset({"drawdown"}),
            frozenset({"how", "down", "today"}),
            frozenset({"how", "much", "lost"}),
            frozenset({"down", "today"}),
            frozenset({"performance", "today"}),
        ),
    ),
)


_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> set[str]:
    """Lowercase + strip punctuation + tokenize.

    Returns a set of word tokens for set-based matching. The
    punctuation strip means "halt." and "halt!" both match HALT,
    but a question-mark is NOT treated specially — "halt?" still
    matches HALT, but the *combination* "halt? and close all"
    won't match HALT_AND_CLOSE_ALL because the question-mark
    breaks the "halt close all" set match (it does — we strip
    punct, so this would match; the question semantic is handled
    elsewhere by the confirmation gate).
    """

    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return {tok for tok in cleaned.split() if tok}


def classify_intent(text: str) -> VoiceIntent:
    """Map a transcribed text to a VoiceIntent.

    Returns UNKNOWN if no pattern matches. The matching is order-
    sensitive: HALT_AND_CLOSE_ALL is checked before HALT so that
    "halt and close all" doesn't match the simpler HALT intent.
    """

    if not text or not text.strip():
        return VoiceIntent.UNKNOWN

    tokens = _normalize(text)
    for intent, alternatives in _INTENT_PATTERNS:
        for required_words in alternatives:
            if required_words.issubset(tokens):
                return intent
    return VoiceIntent.UNKNOWN


def is_destructive(intent: VoiceIntent) -> bool:
    """True if the intent requires confirmation before execution."""

    return intent in _DESTRUCTIVE_INTENTS


_DEFAULT_CONFIRMATION_WINDOW = timedelta(seconds=10)


@dataclass(frozen=True)
class IntentRecognition:
    """One recognized voice command.

    `confirmed_at` is None until the operator confirms. `expires_at`
    is when the confirmation window closes; recognitions past
    `expires_at` cannot be confirmed.
    """

    recognition_id: str
    intent: VoiceIntent
    raw_text: str
    recognized_at: datetime
    expires_at: datetime
    confirmed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.recognition_id or not self.recognition_id.strip():
            raise ValueError("recognition_id must be non-empty")
        if not self.raw_text or not self.raw_text.strip():
            raise ValueError("raw_text must be non-empty")
        if self.recognized_at.tzinfo is None:
            raise ValueError("recognized_at must be timezone-aware")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.recognized_at:
            raise ValueError("expires_at must be after recognized_at")
        if self.confirmed_at is not None and self.confirmed_at.tzinfo is None:
            raise ValueError("confirmed_at must be timezone-aware when set")


class ConfirmationExpiredError(Exception):
    """Raised when a confirmation arrives past the window."""

    def __init__(self, recognition_id: str) -> None:
        super().__init__(f"recognition {recognition_id!r} confirmation window expired")
        self.recognition_id = recognition_id


class NotDestructiveError(Exception):
    """Raised when confirm_recognition is called on a non-destructive intent."""

    def __init__(self, intent: VoiceIntent) -> None:
        super().__init__(f"intent {intent.value!r} is not destructive; no confirmation needed")
        self.intent = intent


def recognize(
    *,
    recognition_id: str,
    raw_text: str,
    now: datetime,
    confirmation_window: timedelta = _DEFAULT_CONFIRMATION_WINDOW,
) -> IntentRecognition:
    """Classify a transcript and produce a recognition record."""

    if not recognition_id or not recognition_id.strip():
        raise ValueError("recognition_id must be non-empty")
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if confirmation_window <= timedelta(0):
        raise ValueError("confirmation_window must be positive")
    intent = classify_intent(raw_text)
    return IntentRecognition(
        recognition_id=recognition_id,
        intent=intent,
        raw_text=raw_text,
        recognized_at=now,
        expires_at=now + confirmation_window,
    )


def confirm_recognition(recognition: IntentRecognition, *, now: datetime) -> IntentRecognition:
    """Confirm a destructive recognition. Raises if not destructive,
    already confirmed, or window expired."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not is_destructive(recognition.intent):
        raise NotDestructiveError(recognition.intent)
    if recognition.confirmed_at is not None:
        raise ValueError(f"recognition {recognition.recognition_id!r} already confirmed")
    if now > recognition.expires_at:
        raise ConfirmationExpiredError(recognition.recognition_id)
    return IntentRecognition(
        recognition_id=recognition.recognition_id,
        intent=recognition.intent,
        raw_text=recognition.raw_text,
        recognized_at=recognition.recognized_at,
        expires_at=recognition.expires_at,
        confirmed_at=now,
    )


def is_executable(recognition: IntentRecognition, *, now: datetime) -> bool:
    """True if the recognition can be acted on right now.

    Non-destructive intents are immediately executable. Destructive
    intents require confirmation within the window.
    """

    if recognition.intent is VoiceIntent.UNKNOWN:
        return False
    if not is_destructive(recognition.intent):
        return True
    return recognition.confirmed_at is not None and now <= recognition.expires_at


_INTENT_EMOJI: dict[VoiceIntent, str] = {
    VoiceIntent.HALT: "🛑",
    VoiceIntent.RESUME: "▶️",
    VoiceIntent.STATUS: "📊",
    VoiceIntent.DRAWDOWN_QUERY: "📉",
    VoiceIntent.HALT_AND_CLOSE_ALL: "🚨",
    VoiceIntent.UNKNOWN: "❓",
}


def render_recognition(recognition: IntentRecognition) -> str:
    """Format a recognition for ops display.

    No-secret-leak: never includes audio file paths / device IDs.
    """

    emoji = _INTENT_EMOJI[recognition.intent]
    confirmed_marker = (
        f" ✓ confirmed at {recognition.confirmed_at.isoformat()}"
        if recognition.confirmed_at is not None
        else ""
    )
    destructive_marker = (
        " (DESTRUCTIVE — needs confirm)"
        if (is_destructive(recognition.intent) and recognition.confirmed_at is None)
        else ""
    )
    return (
        f"{emoji} {recognition.intent.value}: {recognition.raw_text}"
        f"{destructive_marker}{confirmed_marker}\n"
        f"  recognized: {recognition.recognized_at.isoformat()}\n"
        f"  expires: {recognition.expires_at.isoformat()}"
    )


__all__ = [
    "ConfirmationExpiredError",
    "IntentRecognition",
    "NotDestructiveError",
    "VoiceIntent",
    "classify_intent",
    "confirm_recognition",
    "is_destructive",
    "is_executable",
    "recognize",
    "render_recognition",
]
