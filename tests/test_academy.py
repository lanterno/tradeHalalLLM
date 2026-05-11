"""Tests for education/academy.py — Round-5 Wave 20.A."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.education.academy import (
    Catalog,
    CompletionRecord,
    LearnerProgress,
    Lesson,
    LessonKind,
    Tier,
    Track,
    complete_lesson,
    compute_qualified_tier,
    is_track_complete,
    n_tracks_completed_at_tier,
    per_track_progress,
    render_progress,
)


def _lesson(
    lesson_id: str = "L1",
    title: str = "Intro",
    kind: LessonKind = LessonKind.VIDEO,
    minutes: int = 10,
    order_index: int = 0,
) -> Lesson:
    return Lesson(
        lesson_id=lesson_id,
        title=title,
        kind=kind,
        minutes_estimated=minutes,
        order_index=order_index,
    )


def _track(
    track_id: str = "T1",
    title: str = "Halal Foundations",
    tier: Tier = Tier.BEGINNER,
    n_lessons: int = 3,
    enforce_order: bool = True,
) -> Track:
    lessons = tuple(
        _lesson(lesson_id=f"L{i}", title=f"Lesson {i}", order_index=i) for i in range(n_lessons)
    )
    return Track(
        track_id=track_id,
        title=title,
        tier=tier,
        lessons=lessons,
        enforce_order=enforce_order,
    )


def _catalog() -> Catalog:
    return Catalog(
        tracks=(
            _track(track_id="B1", tier=Tier.BEGINNER),
            _track(track_id="B2", tier=Tier.BEGINNER),
            _track(track_id="I1", tier=Tier.INTERMEDIATE),
            _track(track_id="I2", tier=Tier.INTERMEDIATE),
            _track(track_id="A1", tier=Tier.ADVANCED),
            _track(track_id="A2", tier=Tier.ADVANCED),
        )
    )


# --- Lesson validation -------------------------------------------------


def test_lesson_valid():
    L = _lesson()
    assert L.kind is LessonKind.VIDEO


def test_lesson_empty_id_rejected():
    with pytest.raises(ValueError):
        _lesson(lesson_id="")


def test_lesson_long_title_rejected():
    with pytest.raises(ValueError):
        _lesson(title="x" * 200)


def test_lesson_zero_minutes_rejected():
    with pytest.raises(ValueError):
        _lesson(minutes=0)


def test_lesson_negative_order_rejected():
    with pytest.raises(ValueError):
        _lesson(order_index=-1)


def test_lesson_immutable():
    L = _lesson()
    with pytest.raises(AttributeError):
        L.title = "x"  # type: ignore[misc]


# --- Track validation -------------------------------------------------


def test_track_valid():
    t = _track()
    assert t.total_minutes() == 30


def test_track_empty_lessons_rejected():
    with pytest.raises(ValueError):
        Track(
            track_id="T1",
            title="X",
            tier=Tier.BEGINNER,
            lessons=(),
        )


def test_track_duplicate_lesson_id_rejected():
    bad = (
        _lesson(lesson_id="L1", order_index=0),
        _lesson(lesson_id="L1", order_index=1),
    )
    with pytest.raises(ValueError):
        Track(track_id="T1", title="X", tier=Tier.BEGINNER, lessons=bad)


def test_track_unsorted_lessons_rejected():
    bad = (
        _lesson(lesson_id="L1", order_index=1),
        _lesson(lesson_id="L2", order_index=0),
    )
    with pytest.raises(ValueError):
        Track(track_id="T1", title="X", tier=Tier.BEGINNER, lessons=bad)


def test_track_duplicate_order_rejected():
    bad = (
        _lesson(lesson_id="L1", order_index=0),
        _lesson(lesson_id="L2", order_index=0),
    )
    with pytest.raises(ValueError):
        Track(track_id="T1", title="X", tier=Tier.BEGINNER, lessons=bad)


# --- Catalog validation -----------------------------------------------


def test_catalog_duplicate_track_id_rejected():
    bad = (_track(track_id="T1"), _track(track_id="T1"))
    with pytest.raises(ValueError):
        Catalog(tracks=bad)


def test_catalog_by_id():
    c = _catalog()
    assert c.by_id("B1") is not None
    assert c.by_id("nonexistent") is None


def test_catalog_by_tier():
    c = _catalog()
    beginner = c.by_tier(Tier.BEGINNER)
    assert len(beginner) == 2
    assert all(t.tier is Tier.BEGINNER for t in beginner)


# --- LearnerProgress validation ---------------------------------------


def test_progress_valid_empty():
    p = LearnerProgress(learner_id="alice")
    assert p.qualified_tier is Tier.BEGINNER


def test_progress_empty_learner_rejected():
    with pytest.raises(ValueError):
        LearnerProgress(learner_id=" ")


def test_progress_mismatched_completion_rejected():
    bad = CompletionRecord(
        learner_id="bob",
        track_id="B1",
        lesson_id="L0",
        completed_on=date(2026, 5, 1),
    )
    with pytest.raises(ValueError):
        LearnerProgress(learner_id="alice", completions=(bad,))


def test_progress_helpers():
    p = LearnerProgress(
        learner_id="alice",
        completions=(
            CompletionRecord(
                learner_id="alice",
                track_id="B1",
                lesson_id="L0",
                completed_on=date(2026, 5, 1),
            ),
        ),
    )
    assert p.completed_lessons_in("B1") == ("L0",)
    assert p.is_lesson_completed("B1", "L0")
    assert not p.is_lesson_completed("B1", "L1")


# --- is_track_complete ------------------------------------------------


def test_is_track_complete_true_when_all_done():
    catalog = _catalog()
    p = LearnerProgress(
        learner_id="alice",
        completions=tuple(
            CompletionRecord(
                learner_id="alice",
                track_id="B1",
                lesson_id=f"L{i}",
                completed_on=date(2026, 5, 1),
            )
            for i in range(3)
        ),
    )
    t = catalog.by_id("B1")
    assert t is not None
    assert is_track_complete(p, t)


def test_is_track_complete_false_when_partial():
    catalog = _catalog()
    p = LearnerProgress(
        learner_id="alice",
        completions=(
            CompletionRecord(
                learner_id="alice",
                track_id="B1",
                lesson_id="L0",
                completed_on=date(2026, 5, 1),
            ),
        ),
    )
    t = catalog.by_id("B1")
    assert t is not None
    assert not is_track_complete(p, t)


# --- complete_lesson — order enforcement -----------------------------


def test_complete_lesson_first_lesson():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p2 = complete_lesson(p, catalog, track_id="B1", lesson_id="L0", completed_on=date(2026, 5, 1))
    assert p2.is_lesson_completed("B1", "L0")


def test_complete_lesson_skipping_prior_rejected_when_enforced():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    with pytest.raises(ValueError):
        complete_lesson(p, catalog, track_id="B1", lesson_id="L2", completed_on=date(2026, 5, 1))


def test_complete_lesson_skipping_allowed_when_unenforced():
    track = _track(track_id="X", enforce_order=False)
    catalog = Catalog(tracks=(track,))
    p = LearnerProgress(learner_id="alice")
    p2 = complete_lesson(p, catalog, track_id="X", lesson_id="L2", completed_on=date(2026, 5, 1))
    assert p2.is_lesson_completed("X", "L2")


def test_complete_lesson_unknown_track_rejected():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    with pytest.raises(ValueError):
        complete_lesson(
            p,
            catalog,
            track_id="NOPE",
            lesson_id="L0",
            completed_on=date(2026, 5, 1),
        )


def test_complete_lesson_unknown_lesson_rejected():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    with pytest.raises(ValueError):
        complete_lesson(
            p,
            catalog,
            track_id="B1",
            lesson_id="NOPE",
            completed_on=date(2026, 5, 1),
        )


def test_complete_lesson_duplicate_is_idempotent():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = complete_lesson(p, catalog, track_id="B1", lesson_id="L0", completed_on=date(2026, 5, 1))
    p2 = complete_lesson(p, catalog, track_id="B1", lesson_id="L0", completed_on=date(2026, 5, 2))
    assert len(p2.completions) == 1


# --- Tier promotion ---------------------------------------------------


def _complete_track(p: LearnerProgress, catalog: Catalog, track_id: str) -> LearnerProgress:
    track = catalog.by_id(track_id)
    assert track is not None
    for L in track.lessons:
        p = complete_lesson(
            p,
            catalog,
            track_id=track_id,
            lesson_id=L.lesson_id,
            completed_on=date(2026, 5, 1),
        )
    return p


def test_promotion_to_intermediate_after_two_beginner_tracks():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = _complete_track(p, catalog, "B1")
    p = _complete_track(p, catalog, "B2")
    assert p.qualified_tier is Tier.INTERMEDIATE


def test_promotion_one_beginner_keeps_beginner():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = _complete_track(p, catalog, "B1")
    assert p.qualified_tier is Tier.BEGINNER


def test_promotion_to_advanced():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = _complete_track(p, catalog, "B1")
    p = _complete_track(p, catalog, "B2")
    p = _complete_track(p, catalog, "I1")
    p = _complete_track(p, catalog, "I2")
    assert p.qualified_tier is Tier.ADVANCED


def test_promotion_to_expert():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    for tid in ("B1", "B2", "I1", "I2", "A1", "A2"):
        p = _complete_track(p, catalog, tid)
    assert p.qualified_tier is Tier.EXPERT


def test_tier_monotone_non_decreasing():
    """Pin: even with manual qualified_tier override, recompute never drops."""
    catalog = _catalog()
    p = LearnerProgress(
        learner_id="alice",
        qualified_tier=Tier.ADVANCED,  # operator override
    )
    # No completions → recompute would normally return BEGINNER.
    new_tier = compute_qualified_tier(p, catalog)
    assert new_tier is Tier.ADVANCED


# --- per_track_progress ----------------------------------------------


def test_per_track_progress_basic():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = complete_lesson(p, catalog, track_id="B1", lesson_id="L0", completed_on=date(2026, 5, 1))
    views = per_track_progress(p, catalog)
    by_id = {v.track_id: v for v in views}
    assert by_id["B1"].n_completed == 1
    assert by_id["B1"].percent_complete == pytest.approx(1 / 3)
    assert not by_id["B1"].is_complete
    assert by_id["B2"].n_completed == 0


def test_per_track_progress_complete():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = _complete_track(p, catalog, "B1")
    views = per_track_progress(p, catalog)
    by_id = {v.track_id: v for v in views}
    assert by_id["B1"].is_complete
    assert by_id["B1"].percent_complete == 1.0


# --- n_tracks_completed_at_tier --------------------------------------


def test_n_tracks_completed_at_tier_counts_only_full():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = _complete_track(p, catalog, "B1")
    p = complete_lesson(p, catalog, track_id="B2", lesson_id="L0", completed_on=date(2026, 5, 1))
    assert n_tracks_completed_at_tier(p, catalog, Tier.BEGINNER) == 1


# --- Render -----------------------------------------------------------


def test_render_no_secret_leak():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice@example.com")
    views = per_track_progress(p, catalog)
    out = render_progress(p, views)
    assert "alice@example.com" not in out


def test_render_includes_tier_emoji():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    views = per_track_progress(p, catalog)
    out = render_progress(p, views)
    assert "🌱" in out  # beginner


def test_render_includes_complete_check():
    catalog = _catalog()
    p = LearnerProgress(learner_id="alice")
    p = _complete_track(p, catalog, "B1")
    views = per_track_progress(p, catalog)
    out = render_progress(p, views)
    assert "✅" in out
