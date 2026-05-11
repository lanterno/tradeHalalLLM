"""Halal trading academy course tracker — Round-5 Wave 20.A.

Curriculum is organised as **Tracks** (e.g. "Halal Foundations") which
contain **Lessons** (sequenced units). A `LearnerProgress` record per
user tracks lesson completions; helpers compute tier (BEGINNER →
INTERMEDIATE → ADVANCED) based on completion counts.

This module is the **catalogue + progress engine**. No persistence;
the deployment layer owns DB writes.

Pinned semantics:

- **Closed-set Tier ladder**: BEGINNER / INTERMEDIATE / ADVANCED /
  EXPERT.
- **Closed-set LessonKind**: VIDEO / TEXT / INTERACTIVE / QUIZ.
- **Tier promotion is monotone non-decreasing** — once a learner
  qualifies for INTERMEDIATE, no amount of skipped completions can
  send them back to BEGINNER.
- **Lessons within a track are strictly ordered** — completing
  lesson N requires lesson N-1 done first (the `enforce_order` flag
  on `Track` toggles this; default True).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — learner IDs masked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class Tier(str, Enum):
    """Closed-set tier ladder."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


_TIER_ORDER: dict[Tier, int] = {
    Tier.BEGINNER: 0,
    Tier.INTERMEDIATE: 1,
    Tier.ADVANCED: 2,
    Tier.EXPERT: 3,
}


class LessonKind(str, Enum):
    """Closed-set lesson kind."""

    VIDEO = "video"
    TEXT = "text"
    INTERACTIVE = "interactive"
    QUIZ = "quiz"


@dataclass(frozen=True)
class Lesson:
    """A single lesson within a track."""

    lesson_id: str
    title: str
    kind: LessonKind
    minutes_estimated: int
    order_index: int
    """Position within the track; strictly ascending."""

    def __post_init__(self) -> None:
        if not self.lesson_id or not self.lesson_id.strip():
            raise ValueError("lesson_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if len(self.title) > 120:
            raise ValueError("title must be ≤ 120 chars")
        if self.minutes_estimated <= 0:
            raise ValueError("minutes_estimated must be positive")
        if self.order_index < 0:
            raise ValueError("order_index must be ≥ 0")


@dataclass(frozen=True)
class Track:
    """A sequenced set of lessons targeting a tier."""

    track_id: str
    title: str
    tier: Tier
    lessons: tuple[Lesson, ...]
    enforce_order: bool = True

    def __post_init__(self) -> None:
        if not self.track_id or not self.track_id.strip():
            raise ValueError("track_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.lessons:
            raise ValueError("track must have at least one lesson")
        # Lesson IDs unique within a track.
        ids = [L.lesson_id for L in self.lessons]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate lesson_id within track")
        # order_index strictly ascending.
        orders = [L.order_index for L in self.lessons]
        if orders != sorted(orders):
            raise ValueError("lessons must be ordered by order_index")
        if len(set(orders)) != len(orders):
            raise ValueError("order_index must be unique within track")

    def total_minutes(self) -> int:
        return sum(L.minutes_estimated for L in self.lessons)


@dataclass(frozen=True)
class Catalog:
    """A frozen catalogue of tracks."""

    tracks: tuple[Track, ...]

    def __post_init__(self) -> None:
        ids = [t.track_id for t in self.tracks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate track_id in catalog")

    def by_id(self, track_id: str) -> Track | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def by_tier(self, tier: Tier) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if t.tier is tier)


@dataclass(frozen=True)
class CompletionRecord:
    """One lesson completion event."""

    learner_id: str
    track_id: str
    lesson_id: str
    completed_on: date

    def __post_init__(self) -> None:
        if not self.learner_id or not self.learner_id.strip():
            raise ValueError("learner_id must be non-empty")
        if not self.track_id or not self.track_id.strip():
            raise ValueError("track_id must be non-empty")
        if not self.lesson_id or not self.lesson_id.strip():
            raise ValueError("lesson_id must be non-empty")


@dataclass(frozen=True)
class LearnerProgress:
    """A learner's progress across all tracks."""

    learner_id: str
    completions: tuple[CompletionRecord, ...] = ()
    qualified_tier: Tier = Tier.BEGINNER
    """Operator-pinned current tier — monotone non-decreasing."""

    def __post_init__(self) -> None:
        if not self.learner_id or not self.learner_id.strip():
            raise ValueError("learner_id must be non-empty")
        # Every completion record's learner_id must match.
        for c in self.completions:
            if c.learner_id != self.learner_id:
                raise ValueError("completion.learner_id must match progress.learner_id")

    def completed_lessons_in(self, track_id: str) -> tuple[str, ...]:
        return tuple(c.lesson_id for c in self.completions if c.track_id == track_id)

    def is_lesson_completed(self, track_id: str, lesson_id: str) -> bool:
        return any(c.track_id == track_id and c.lesson_id == lesson_id for c in self.completions)


# Tier-promotion thresholds: completing the listed number of tracks at
# the previous tier qualifies the learner for the next tier. Track
# "completion" means *all* lessons in the track done.
_PROMOTION_THRESHOLDS: dict[Tier, int] = {
    Tier.BEGINNER: 0,  # default starting tier
    Tier.INTERMEDIATE: 2,  # complete ≥ 2 beginner tracks
    Tier.ADVANCED: 2,  # complete ≥ 2 intermediate tracks
    Tier.EXPERT: 2,  # complete ≥ 2 advanced tracks
}


def is_track_complete(progress: LearnerProgress, track: Track) -> bool:
    """True iff every lesson in `track` is in the progress."""
    done = set(progress.completed_lessons_in(track.track_id))
    return all(L.lesson_id in done for L in track.lessons)


def n_tracks_completed_at_tier(progress: LearnerProgress, catalog: Catalog, tier: Tier) -> int:
    return sum(1 for t in catalog.by_tier(tier) if is_track_complete(progress, t))


def compute_qualified_tier(progress: LearnerProgress, catalog: Catalog) -> Tier:
    """Walk the tier ladder; promote as long as the learner has
    completed enough tracks at the prior tier.

    Pinned monotone: never returns a tier lower than `progress.qualified_tier`.
    """
    new_tier = Tier.BEGINNER
    if (
        n_tracks_completed_at_tier(progress, catalog, Tier.BEGINNER)
        >= _PROMOTION_THRESHOLDS[Tier.INTERMEDIATE]
    ):
        new_tier = Tier.INTERMEDIATE
    if (
        new_tier is Tier.INTERMEDIATE
        and n_tracks_completed_at_tier(progress, catalog, Tier.INTERMEDIATE)
        >= _PROMOTION_THRESHOLDS[Tier.ADVANCED]
    ):
        new_tier = Tier.ADVANCED
    if (
        new_tier is Tier.ADVANCED
        and n_tracks_completed_at_tier(progress, catalog, Tier.ADVANCED)
        >= _PROMOTION_THRESHOLDS[Tier.EXPERT]
    ):
        new_tier = Tier.EXPERT
    if _TIER_ORDER[new_tier] < _TIER_ORDER[progress.qualified_tier]:
        return progress.qualified_tier
    return new_tier


def complete_lesson(
    progress: LearnerProgress,
    catalog: Catalog,
    *,
    track_id: str,
    lesson_id: str,
    completed_on: date,
) -> LearnerProgress:
    """Mark a lesson complete; returns a new frozen LearnerProgress.

    Pinned:
    - Track must exist in catalog.
    - Lesson must exist in the track.
    - If track.enforce_order: every prior-indexed lesson must already
      be complete.
    - Duplicate completion is idempotent (no-op).
    - Tier is recomputed and monotone-non-decreasing.
    """
    track = catalog.by_id(track_id)
    if track is None:
        raise ValueError(f"unknown track_id {track_id}")
    lesson = next((L for L in track.lessons if L.lesson_id == lesson_id), None)
    if lesson is None:
        raise ValueError(f"unknown lesson_id {lesson_id} in track {track_id}")
    if progress.is_lesson_completed(track_id, lesson_id):
        return progress
    if track.enforce_order:
        done = set(progress.completed_lessons_in(track_id))
        for prior in track.lessons:
            if prior.order_index >= lesson.order_index:
                break
            if prior.lesson_id not in done:
                raise ValueError(
                    f"lesson {lesson_id} requires prior lesson "
                    f"{prior.lesson_id} (order {prior.order_index})"
                )
    new_record = CompletionRecord(
        learner_id=progress.learner_id,
        track_id=track_id,
        lesson_id=lesson_id,
        completed_on=completed_on,
    )
    new_completions = (*progress.completions, new_record)
    new_progress = replace(progress, completions=new_completions)
    new_tier = compute_qualified_tier(new_progress, catalog)
    return replace(new_progress, qualified_tier=new_tier)


@dataclass(frozen=True)
class TrackProgressView:
    """Operator-readable per-track snapshot."""

    track_id: str
    track_title: str
    tier: Tier
    n_lessons: int
    n_completed: int
    percent_complete: float
    is_complete: bool


def per_track_progress(
    progress: LearnerProgress, catalog: Catalog
) -> tuple[TrackProgressView, ...]:
    """Return one view per track in the catalogue."""
    out: list[TrackProgressView] = []
    for t in catalog.tracks:
        done = set(progress.completed_lessons_in(t.track_id))
        n_done = sum(1 for L in t.lessons if L.lesson_id in done)
        pct = n_done / len(t.lessons) if t.lessons else 0.0
        out.append(
            TrackProgressView(
                track_id=t.track_id,
                track_title=t.title,
                tier=t.tier,
                n_lessons=len(t.lessons),
                n_completed=n_done,
                percent_complete=pct,
                is_complete=is_track_complete(progress, t),
            )
        )
    return tuple(out)


def _mask(learner_id: str) -> str:
    if len(learner_id) <= 4:
        return "***"
    return learner_id[:2] + "…" + learner_id[-2:]


_TIER_EMOJI: dict[Tier, str] = {
    Tier.BEGINNER: "🌱",
    Tier.INTERMEDIATE: "🌿",
    Tier.ADVANCED: "🌳",
    Tier.EXPERT: "🏆",
}


def render_progress(
    progress: LearnerProgress,
    views: Sequence[TrackProgressView],
) -> str:
    head = (
        f"📚 Learner {_mask(progress.learner_id)} "
        f"{_TIER_EMOJI[progress.qualified_tier]} {progress.qualified_tier.value}"
    )
    lines = [head]
    for v in views:
        check = "✅" if v.is_complete else "▶️"
        lines.append(
            f"  {check} {v.track_title} ({v.tier.value}): "
            f"{v.n_completed}/{v.n_lessons} "
            f"({v.percent_complete * 100:.0f}%)"
        )
    return "\n".join(lines)
