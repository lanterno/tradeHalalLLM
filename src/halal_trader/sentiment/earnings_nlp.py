"""Earnings call NLP — Round-5 Wave 11.D.

Custom lexicon-based scorer for earnings-call transcripts. Surfaces:

1. **CFO uncertainty markers** — hedging language ("may", "we expect",
   "headwinds", "challenges") that historically correlates with
   downside surprises.
2. **Confidence markers** — bullish language ("strong demand",
   "raised guidance", "operating leverage").
3. **Segment-level disclosure changes** — quarter-over-quarter diff
   counting which segments newly appear / disappear / change tone.
4. **Tone time-series** — cumulative sentiment per speaker turn so
   operators can see whether tone shifted mid-call (a CFO that
   started bullish and ended hedging is a different signal than one
   who was hedging the whole way).

This module is a **pure-Python lexicon scorer**. No external NLP
model. The lexicon is operator-tunable; defaults are calibrated
against the standard earnings-call literature (Tetlock, Loughran-
McDonald financial sentiment word lists, plus halal-specific
language tilts).

Pinned semantics:

- **Closed-set ToneClass ladder** — VERY_BEARISH / BEARISH / NEUTRAL /
  BULLISH / VERY_BULLISH. Used both per-turn and call-aggregate.
- **Closed-set DisclosureChange** — NEW_SEGMENT / RETIRED_SEGMENT /
  TONE_DEGRADED / TONE_IMPROVED.
- **Tone score = (bullish_hits − bearish_hits) / max(1, total_hits)**.
  Pinned in [-1, 1]; emits ToneClass via fixed thresholds.
- **Uncertainty score = uncertainty_hits / total_words**. Pinned in
  [0, 1]; the threshold for "high uncertainty" is 0.04 (4% of words
  are hedging).
- **Pure-Python deterministic.**
- **No-secret-leak pin** — speaker names masked in render.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum


class ToneClass(str, Enum):
    """Closed-set tone ladder."""

    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class SpeakerRole(str, Enum):
    """Closed-set speaker-role ladder."""

    CEO = "ceo"
    CFO = "cfo"
    OPERATING_OFFICER = "operating_officer"
    ANALYST = "analyst"
    OTHER = "other"


class DisclosureChangeType(str, Enum):
    """Closed-set segment-disclosure change ladder."""

    NEW_SEGMENT = "new_segment"
    RETIRED_SEGMENT = "retired_segment"
    TONE_DEGRADED = "tone_degraded"
    TONE_IMPROVED = "tone_improved"


# --- Lexicons -----------------------------------------------------------


def default_bullish_lexicon() -> tuple[str, ...]:
    """Bullish words/phrases. Lowercase; substring-matched."""
    return (
        "strong demand",
        "record revenue",
        "raised guidance",
        "operating leverage",
        "outperform",
        "beat estimates",
        "accelerating growth",
        "robust pipeline",
        "improving margins",
        "exceeded expectations",
        "tailwinds",
        "expanding",
        "share gains",
    )


def default_bearish_lexicon() -> tuple[str, ...]:
    """Bearish words/phrases."""
    return (
        "headwinds",
        "challenges",
        "weakness",
        "decline",
        "missed estimates",
        "guidance cut",
        "lowered guidance",
        "softness",
        "macro uncertainty",
        "supply constraints",
        "margin compression",
        "underperform",
        "writedown",
        "impairment",
    )


def default_uncertainty_lexicon() -> tuple[str, ...]:
    """CFO uncertainty / hedging markers."""
    return (
        "we expect",
        "may",
        "could",
        "uncertain",
        "we believe",
        "should",
        "potentially",
        "approximately",
        "if conditions",
        "subject to",
        "depending on",
        "not entirely clear",
        "too early",
        "too soon",
    )


@dataclass(frozen=True)
class SpeakerTurn:
    """One contiguous speaker turn in the transcript."""

    turn_id: int
    speaker_name: str
    role: SpeakerRole
    text: str

    def __post_init__(self) -> None:
        if self.turn_id < 0:
            raise ValueError("turn_id must be ≥ 0")
        if not self.speaker_name or not self.speaker_name.strip():
            raise ValueError("speaker_name must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")


@dataclass(frozen=True)
class TurnScore:
    """Output of `score_turn`."""

    turn_id: int
    role: SpeakerRole
    bullish_hits: int
    bearish_hits: int
    uncertainty_hits: int
    word_count: int
    tone_score: float  # [-1, 1]
    tone_class: ToneClass
    uncertainty_score: float  # [0, 1]


def _count_phrase_hits(text: str, phrases: Sequence[str]) -> int:
    """Substring-count phrase hits in lowercase text."""
    n = 0
    text_l = text.lower()
    for p in phrases:
        if p in text_l:
            n += text_l.count(p)
    return n


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _tone_class(score: float) -> ToneClass:
    if score <= -0.6:
        return ToneClass.VERY_BEARISH
    if score <= -0.2:
        return ToneClass.BEARISH
    if score < 0.2:
        return ToneClass.NEUTRAL
    if score < 0.6:
        return ToneClass.BULLISH
    return ToneClass.VERY_BULLISH


def score_turn(
    turn: SpeakerTurn,
    *,
    bullish_lexicon: Sequence[str] | None = None,
    bearish_lexicon: Sequence[str] | None = None,
    uncertainty_lexicon: Sequence[str] | None = None,
) -> TurnScore:
    """Score a single speaker turn."""
    bull_lex = bullish_lexicon if bullish_lexicon is not None else default_bullish_lexicon()
    bear_lex = bearish_lexicon if bearish_lexicon is not None else default_bearish_lexicon()
    unc_lex = (
        uncertainty_lexicon if uncertainty_lexicon is not None else default_uncertainty_lexicon()
    )
    bull_hits = _count_phrase_hits(turn.text, bull_lex)
    bear_hits = _count_phrase_hits(turn.text, bear_lex)
    unc_hits = _count_phrase_hits(turn.text, unc_lex)
    words = max(1, _word_count(turn.text))
    total_tone = bull_hits + bear_hits
    if total_tone == 0:
        tone = 0.0
    else:
        tone = (bull_hits - bear_hits) / total_tone
    unc_score = unc_hits / words
    return TurnScore(
        turn_id=turn.turn_id,
        role=turn.role,
        bullish_hits=bull_hits,
        bearish_hits=bear_hits,
        uncertainty_hits=unc_hits,
        word_count=words,
        tone_score=tone,
        tone_class=_tone_class(tone),
        uncertainty_score=unc_score,
    )


@dataclass(frozen=True)
class CallScore:
    """Aggregate score for an entire earnings call."""

    ticker: str
    call_date: date
    n_turns: int
    aggregate_tone: float
    aggregate_tone_class: ToneClass
    cfo_uncertainty_score: float
    """High-water mark across CFO turns (worst-case)."""
    cfo_uncertainty_high: bool
    """True iff cfo_uncertainty_score > 0.04 (4% threshold)."""
    tone_trajectory: tuple[float, ...]
    """Per-turn cumulative tone — operator can see if mood shifted
    mid-call."""


def _aggregate_tone(turns: Sequence[TurnScore]) -> float:
    total_bull = sum(t.bullish_hits for t in turns)
    total_bear = sum(t.bearish_hits for t in turns)
    total = total_bull + total_bear
    if total == 0:
        return 0.0
    return (total_bull - total_bear) / total


def _cumulative_trajectory(turns: Sequence[TurnScore]) -> tuple[float, ...]:
    out: list[float] = []
    running_bull = 0
    running_bear = 0
    for t in turns:
        running_bull += t.bullish_hits
        running_bear += t.bearish_hits
        total = running_bull + running_bear
        if total == 0:
            out.append(0.0)
        else:
            out.append((running_bull - running_bear) / total)
    return tuple(out)


def score_call(
    *,
    ticker: str,
    call_date: date,
    turns: Sequence[SpeakerTurn],
    bullish_lexicon: Sequence[str] | None = None,
    bearish_lexicon: Sequence[str] | None = None,
    uncertainty_lexicon: Sequence[str] | None = None,
) -> CallScore:
    """Score an entire call from a turn sequence."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be non-empty")
    if not turns:
        raise ValueError("turns must be non-empty")
    scored = tuple(
        score_turn(
            t,
            bullish_lexicon=bullish_lexicon,
            bearish_lexicon=bearish_lexicon,
            uncertainty_lexicon=uncertainty_lexicon,
        )
        for t in turns
    )
    agg_tone = _aggregate_tone(scored)
    cfo_turns = [t for t in scored if t.role is SpeakerRole.CFO]
    cfo_unc = max((t.uncertainty_score for t in cfo_turns), default=0.0)
    return CallScore(
        ticker=ticker,
        call_date=call_date,
        n_turns=len(scored),
        aggregate_tone=agg_tone,
        aggregate_tone_class=_tone_class(agg_tone),
        cfo_uncertainty_score=cfo_unc,
        cfo_uncertainty_high=cfo_unc > 0.04,
        tone_trajectory=_cumulative_trajectory(scored),
    )


@dataclass(frozen=True)
class SegmentSnapshot:
    """One quarter's segment disclosure."""

    segment_name: str
    revenue_usd: float
    tone: ToneClass

    def __post_init__(self) -> None:
        if not self.segment_name or not self.segment_name.strip():
            raise ValueError("segment_name must be non-empty")
        if self.revenue_usd < 0:
            raise ValueError("revenue_usd must be non-negative")


@dataclass(frozen=True)
class DisclosureChange:
    """Output of `diff_segments` — one detected change."""

    type: DisclosureChangeType
    segment_name: str
    detail: str


_TONE_ORDER: dict[ToneClass, int] = {
    ToneClass.VERY_BEARISH: 0,
    ToneClass.BEARISH: 1,
    ToneClass.NEUTRAL: 2,
    ToneClass.BULLISH: 3,
    ToneClass.VERY_BULLISH: 4,
}


def diff_segments(
    prior: Sequence[SegmentSnapshot],
    current: Sequence[SegmentSnapshot],
) -> tuple[DisclosureChange, ...]:
    """Return changes between two quarters' segment disclosures.

    NEW_SEGMENT: appears in `current` but not in `prior`.
    RETIRED_SEGMENT: appears in `prior` but not in `current`.
    TONE_DEGRADED: same name, current tone < prior tone (in ladder).
    TONE_IMPROVED: same name, current tone > prior tone.
    Same name with same tone → no entry emitted.
    """
    prior_by_name = {s.segment_name: s for s in prior}
    cur_by_name = {s.segment_name: s for s in current}
    out: list[DisclosureChange] = []
    # NEW_SEGMENT
    for name, c in cur_by_name.items():
        if name not in prior_by_name:
            out.append(
                DisclosureChange(
                    type=DisclosureChangeType.NEW_SEGMENT,
                    segment_name=name,
                    detail=f"new segment with tone={c.tone.value}",
                )
            )
    # RETIRED_SEGMENT
    for name, p in prior_by_name.items():
        if name not in cur_by_name:
            out.append(
                DisclosureChange(
                    type=DisclosureChangeType.RETIRED_SEGMENT,
                    segment_name=name,
                    detail=f"segment retired (was {p.tone.value})",
                )
            )
    # Tone changes
    for name in cur_by_name.keys() & prior_by_name.keys():
        c = cur_by_name[name]
        p = prior_by_name[name]
        if _TONE_ORDER[c.tone] < _TONE_ORDER[p.tone]:
            out.append(
                DisclosureChange(
                    type=DisclosureChangeType.TONE_DEGRADED,
                    segment_name=name,
                    detail=f"tone {p.tone.value} → {c.tone.value}",
                )
            )
        elif _TONE_ORDER[c.tone] > _TONE_ORDER[p.tone]:
            out.append(
                DisclosureChange(
                    type=DisclosureChangeType.TONE_IMPROVED,
                    segment_name=name,
                    detail=f"tone {p.tone.value} → {c.tone.value}",
                )
            )
    out.sort(key=lambda c: (c.type.value, c.segment_name))
    return tuple(out)


def _mask(name: str) -> str:
    if len(name) <= 4:
        return "***"
    return name[:2] + "…" + name[-2:]


_TONE_EMOJI: dict[ToneClass, str] = {
    ToneClass.VERY_BEARISH: "🔴🔴",
    ToneClass.BEARISH: "🔴",
    ToneClass.NEUTRAL: "⚪",
    ToneClass.BULLISH: "🟢",
    ToneClass.VERY_BULLISH: "🟢🟢",
}


def render_call(score: CallScore) -> str:
    cfo_flag = " ⚠️ HIGH" if score.cfo_uncertainty_high else ""
    return (
        f"📞 {score.ticker} {score.call_date.isoformat()}: "
        f"{_TONE_EMOJI[score.aggregate_tone_class]} "
        f"{score.aggregate_tone_class.value} "
        f"(tone={score.aggregate_tone:+.2f}, n={score.n_turns}), "
        f"CFO uncertainty={score.cfo_uncertainty_score:.3f}{cfo_flag}"
    )


def render_change(change: DisclosureChange) -> str:
    return f"📋 {change.type.value}: {change.segment_name} — {change.detail}"
