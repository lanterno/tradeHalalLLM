"""Halal-friendly chat moderation — Round-5 Wave 17.F.

Live ticker-rooms need a permissive-but-firm filter that keeps gambling-
style talk and haram references out without becoming a censorship
nightmare. The classifier ladders messages PASS / WARN / BLOCK based
on:

1. **Lexicon hits** — closed-set categories of haram-coded language.
2. **Hype intensity** — ALL-CAPS density, exclamation runs, rocket-
   emoji clusters. Hype alone never BLOCKs but it can compound into a
   WARN / BLOCK alongside lexicon hits.
3. **Repetition** — same user spamming the same message body within
   `repeat_window_seconds` triggers a WARN.
4. **Self-promotion** — non-author URLs / cashtag promo without a
   posted-from-author flag.

This module is **policy + classifier**; downstream the room hands off
the result for display + escalation. The lexicon is operator-tunable
and starts deliberately small so 17.D can defer to this module once
both are wired.

Pinned semantics:

- **Closed-set ModerationOutcome ladder** — PASS / WARN / BLOCK.
- **BLOCK > WARN > PASS** (strictly monotone severity).
- **Pure-functional** — no state. The repetition-detector accepts a
  history window from the caller, never holds it.
- **Operator-tunable lexicon** — `default_lexicon()` returns a
  customisable copy.
- **No-secret-leak pin** — render output redacts user identifiers
  beyond first-2 / last-2 mask.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ModerationOutcome(str, Enum):
    """Closed-set outcome ladder."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


def _max_outcome(a: ModerationOutcome, b: ModerationOutcome) -> ModerationOutcome:
    order = {
        ModerationOutcome.PASS: 0,
        ModerationOutcome.WARN: 1,
        ModerationOutcome.BLOCK: 2,
    }
    return a if order[a] >= order[b] else b


class HaramCategory(str, Enum):
    """Closed-set haram-language categories."""

    GAMBLING = "gambling"
    RIBA = "riba"
    HARAM_SECTOR = "haram_sector"
    SPECULATION_HYPE = "speculation_hype"


def default_lexicon() -> dict[str, tuple[HaramCategory, ModerationOutcome]]:
    """Return the platform's default lexicon. Operators can copy +
    extend in their config."""
    return {
        # Gambling
        "casino": (HaramCategory.GAMBLING, ModerationOutcome.BLOCK),
        "lottery": (HaramCategory.GAMBLING, ModerationOutcome.BLOCK),
        "yolo bet": (HaramCategory.GAMBLING, ModerationOutcome.BLOCK),
        "roulette": (HaramCategory.GAMBLING, ModerationOutcome.BLOCK),
        "all-in yolo": (HaramCategory.GAMBLING, ModerationOutcome.BLOCK),
        # Riba
        "guaranteed return": (HaramCategory.RIBA, ModerationOutcome.BLOCK),
        "fixed coupon yield": (HaramCategory.RIBA, ModerationOutcome.WARN),
        "leveraged margin": (HaramCategory.RIBA, ModerationOutcome.WARN),
        "interest income": (HaramCategory.RIBA, ModerationOutcome.WARN),
        "shorting borrow": (HaramCategory.RIBA, ModerationOutcome.WARN),
        # Haram sector
        "alcohol stock": (HaramCategory.HARAM_SECTOR, ModerationOutcome.WARN),
        "tobacco play": (HaramCategory.HARAM_SECTOR, ModerationOutcome.WARN),
        "pork producer": (HaramCategory.HARAM_SECTOR, ModerationOutcome.WARN),
        # Speculation hype
        "moonshot": (HaramCategory.SPECULATION_HYPE, ModerationOutcome.WARN),
        "10-bagger": (HaramCategory.SPECULATION_HYPE, ModerationOutcome.WARN),
        "to the moon": (HaramCategory.SPECULATION_HYPE, ModerationOutcome.WARN),
    }


@dataclass(frozen=True)
class ChatMessage:
    """A single user chat message."""

    message_id: str
    user_id: str
    ticker_room: str
    body: str
    posted_at: datetime

    def __post_init__(self) -> None:
        if not self.message_id or not self.message_id.strip():
            raise ValueError("message_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.ticker_room or not self.ticker_room.strip():
            raise ValueError("ticker_room must be non-empty")
        if not self.body.strip():
            raise ValueError("body must be non-empty")
        if len(self.body) > 1000:
            raise ValueError("body must be ≤ 1000 chars")


@dataclass(frozen=True)
class ModerationResult:
    """Output of `classify`."""

    message_id: str
    outcome: ModerationOutcome
    matched_phrases: tuple[str, ...]
    matched_categories: tuple[HaramCategory, ...]
    reasons: tuple[str, ...]
    hype_score: float
    """0–1; > 0.5 contributes to WARN; > 0.8 + lexicon hit BLOCKs."""


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_ROCKET = "🚀"


def _hype_score(body: str) -> float:
    """Heuristic hype score in [0, 1].

    Combines:
    - ALL-CAPS density (caps letters / total letters)
    - exclamation runs (!! and !!!)
    - rocket-emoji density
    """
    letters = [c for c in body if c.isalpha()]
    if not letters:
        return 0.0
    caps_density = sum(1 for c in letters if c.isupper()) / len(letters)
    excl_runs = body.count("!!") + 2 * body.count("!!!")
    rockets = body.count(_ROCKET)
    excl_norm = min(1.0, excl_runs / 3.0)
    rocket_norm = min(1.0, rockets / 3.0)
    score = 0.5 * caps_density + 0.25 * excl_norm + 0.25 * rocket_norm
    return min(1.0, score)


def _lexicon_hits(
    body: str,
    lexicon: dict[str, tuple[HaramCategory, ModerationOutcome]],
) -> tuple[
    list[str],
    list[HaramCategory],
    ModerationOutcome,
    list[str],
]:
    """Return (matched_phrases, matched_categories, worst_outcome, reasons)."""
    text = body.lower()
    phrases: list[str] = []
    cats: list[HaramCategory] = []
    reasons: list[str] = []
    worst = ModerationOutcome.PASS
    for phrase, (cat, outcome) in lexicon.items():
        if phrase in text:
            phrases.append(phrase)
            cats.append(cat)
            reasons.append(f"matched '{phrase}' ({cat.value})")
            worst = _max_outcome(worst, outcome)
    return phrases, cats, worst, reasons


def classify(
    message: ChatMessage,
    *,
    lexicon: dict[str, tuple[HaramCategory, ModerationOutcome]] | None = None,
    history: Sequence[ChatMessage] = (),
    repeat_window_seconds: int = 60,
) -> ModerationResult:
    """Classify a single message.

    `history` is the *recent* same-room messages (caller-supplied), used
    for repetition detection. Pure-functional: classify has no state.
    """
    table = lexicon if lexicon is not None else default_lexicon()
    phrases, cats, worst, reasons = _lexicon_hits(message.body, table)
    hype = _hype_score(message.body)
    if hype > 0.50:
        reasons.append(f"hype_score={hype:.2f}")
        if worst is ModerationOutcome.PASS:
            worst = ModerationOutcome.WARN
        elif hype > 0.80 and worst is ModerationOutcome.WARN:
            worst = ModerationOutcome.BLOCK
    # Repetition: same user, same body in window.
    if history:
        cutoff = message.posted_at - timedelta(seconds=repeat_window_seconds)
        recent_user_msgs = [
            h
            for h in history
            if h.user_id == message.user_id
            and h.posted_at >= cutoff
            and h.message_id != message.message_id
        ]
        if any(h.body.strip().lower() == message.body.strip().lower() for h in recent_user_msgs):
            reasons.append("repeated body within window")
            worst = _max_outcome(worst, ModerationOutcome.WARN)
    # Self-promotion: bare URL without ticker context.
    if _URL_RE.search(message.body) and message.ticker_room.upper() not in message.body.upper():
        reasons.append("URL posted without ticker context")
        worst = _max_outcome(worst, ModerationOutcome.WARN)
    return ModerationResult(
        message_id=message.message_id,
        outcome=worst,
        matched_phrases=tuple(phrases),
        matched_categories=tuple(cats),
        reasons=tuple(reasons),
        hype_score=hype,
    )


def classify_batch(
    messages: Iterable[ChatMessage],
    *,
    lexicon: dict[str, tuple[HaramCategory, ModerationOutcome]] | None = None,
    repeat_window_seconds: int = 60,
) -> tuple[ModerationResult, ...]:
    """Classify each message; the running history is built on the fly
    so repetition is detected across the batch."""
    history: list[ChatMessage] = []
    out: list[ModerationResult] = []
    for m in messages:
        out.append(
            classify(
                m,
                lexicon=lexicon,
                history=tuple(history),
                repeat_window_seconds=repeat_window_seconds,
            )
        )
        history.append(m)
    return tuple(out)


def filter_passing(
    messages: Iterable[ChatMessage],
    results: Iterable[ModerationResult],
) -> tuple[ChatMessage, ...]:
    """Return only messages whose result is PASS."""
    by_id = {r.message_id: r for r in results}
    return tuple(
        m
        for m in messages
        if by_id.get(
            m.message_id,
            ModerationResult(
                message_id="",
                outcome=ModerationOutcome.BLOCK,
                matched_phrases=(),
                matched_categories=(),
                reasons=("no result",),
                hype_score=0.0,
            ),
        ).outcome
        is ModerationOutcome.PASS
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_OUTCOME_EMOJI: dict[ModerationOutcome, str] = {
    ModerationOutcome.PASS: "✅",
    ModerationOutcome.WARN: "⚠️",
    ModerationOutcome.BLOCK: "🛑",
}


def render_result(result: ModerationResult) -> str:
    """Operator-readable summary."""
    head = (
        f"{_OUTCOME_EMOJI[result.outcome]} [{result.message_id}] "
        f"{result.outcome.value} "
        f"(hype={result.hype_score:.2f})"
    )
    if not result.reasons:
        return head
    lines = [head]
    for r in result.reasons:
        lines.append(f"  • {r}")
    return "\n".join(lines)
