"""Halal-CFA equivalent curriculum — Round-5 Wave 20.I.

A structured 3-level certification ladder modelled after the CFA but
focused on halal markets:

- **Level I (HCA-I)** — Foundations: AAOIFI standards, halal screening,
  sukuk fundamentals, riba/gharar/maysir, basic portfolio theory.
- **Level II (HCA-II)** — Application: equity analysis, sukuk pricing,
  Mudarabah/Musharakah portfolio construction, risk modelling.
- **Level III (HCA-III)** — Mastery: structured products (Wa'd, Salam,
  Arboun), institutional Islamic finance, regulatory frameworks.

This module is the **curriculum schema + enrolment FSM + eligibility
computer**:

1. A `Curriculum` holds 3 `Level`s; each level has N `Module`s; each
   module has M `Topic`s (the smallest unit of study).
2. A `LearnerEnrolment` records the user's progress through the
   curriculum; topics get marked complete; modules + levels are
   considered complete when all child elements are.
3. Exam eligibility for a level requires completion of all prior
   levels' coursework + all of that level's modules.

Pinned semantics:

- **Closed-set CertLevel ladder** — HCA_I / HCA_II / HCA_III.
- **Closed-set EnrolmentStatus FSM** — NOT_ENROLLED → ENROLLED →
  ELIGIBLE → CERTIFIED, with WITHDRAWN as alternate terminal.
- **Strict prerequisite chain**: HCA-II enrolment requires HCA-I
  CERTIFIED; HCA-III requires HCA-II CERTIFIED.
- **Topic completion is monotone** — once marked complete, can't
  be reverted within an enrolment.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class CertLevel(str, Enum):
    """Closed-set certification level ladder."""

    HCA_I = "hca_i"
    HCA_II = "hca_ii"
    HCA_III = "hca_iii"


_LEVEL_ORDER: dict[CertLevel, int] = {
    CertLevel.HCA_I: 1,
    CertLevel.HCA_II: 2,
    CertLevel.HCA_III: 3,
}


_PREREQ: dict[CertLevel, CertLevel | None] = {
    CertLevel.HCA_I: None,
    CertLevel.HCA_II: CertLevel.HCA_I,
    CertLevel.HCA_III: CertLevel.HCA_II,
}


class EnrolmentStatus(str, Enum):
    """Closed-set enrolment FSM."""

    NOT_ENROLLED = "not_enrolled"
    ENROLLED = "enrolled"
    ELIGIBLE = "eligible"
    """All coursework complete; learner can sit the exam."""
    CERTIFIED = "certified"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class Topic:
    """Smallest unit of study."""

    topic_id: str
    title: str
    estimated_hours: float

    def __post_init__(self) -> None:
        if not self.topic_id or not self.topic_id.strip():
            raise ValueError("topic_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if len(self.title) > 200:
            raise ValueError("title must be ≤ 200 chars")
        if self.estimated_hours <= 0:
            raise ValueError("estimated_hours must be positive")
        if self.estimated_hours > 100:
            raise ValueError("estimated_hours > 100 suspicious")


@dataclass(frozen=True)
class Module:
    """A group of topics."""

    module_id: str
    title: str
    topics: tuple[Topic, ...]

    def __post_init__(self) -> None:
        if not self.module_id or not self.module_id.strip():
            raise ValueError("module_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.topics:
            raise ValueError("module must have at least one topic")
        ids = [t.topic_id for t in self.topics]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate topic_id within module")

    def total_hours(self) -> float:
        return sum(t.estimated_hours for t in self.topics)


@dataclass(frozen=True)
class Level:
    """One certification level (HCA-I / II / III)."""

    level: CertLevel
    modules: tuple[Module, ...]
    pass_threshold_pct: float = 0.70
    """Operator-tunable; default 70%."""

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("level must have at least one module")
        ids = [m.module_id for m in self.modules]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate module_id within level")
        if not 0.0 < self.pass_threshold_pct < 1.0:
            raise ValueError("pass_threshold_pct must be in (0, 1)")

    def total_hours(self) -> float:
        return sum(m.total_hours() for m in self.modules)

    def all_topic_ids(self) -> tuple[str, ...]:
        return tuple(t.topic_id for m in self.modules for t in m.topics)


@dataclass(frozen=True)
class Curriculum:
    """The full 3-level curriculum."""

    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        if len(self.levels) != 3:
            raise ValueError("Curriculum must have exactly 3 levels")
        seen: set[CertLevel] = set()
        for L in self.levels:
            if L.level in seen:
                raise ValueError(f"duplicate level {L.level.value}")
            seen.add(L.level)
        # Topic IDs unique globally — prevents cross-level conflicts.
        all_ids: list[str] = []
        for L in self.levels:
            all_ids.extend(L.all_topic_ids())
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("duplicate topic_id across curriculum")

    def by_level(self, level: CertLevel) -> Level:
        for L in self.levels:
            if L.level is level:
                return L
        raise ValueError(f"level {level.value} not in curriculum")

    def topic_to_level(self, topic_id: str) -> CertLevel | None:
        for L in self.levels:
            if topic_id in L.all_topic_ids():
                return L.level
        return None


@dataclass(frozen=True)
class LearnerEnrolment:
    """One learner's enrolment in one level."""

    enrolment_id: str
    learner_id: str
    level: CertLevel
    enrolled_on: date
    completed_topic_ids: tuple[str, ...] = ()
    status: EnrolmentStatus = EnrolmentStatus.ENROLLED
    certified_on: date | None = None
    withdrawn_on: date | None = None

    def __post_init__(self) -> None:
        if not self.enrolment_id or not self.enrolment_id.strip():
            raise ValueError("enrolment_id must be non-empty")
        if not self.learner_id or not self.learner_id.strip():
            raise ValueError("learner_id must be non-empty")
        if self.status is EnrolmentStatus.NOT_ENROLLED:
            raise ValueError("LearnerEnrolment must not be NOT_ENROLLED")
        if len(set(self.completed_topic_ids)) != len(self.completed_topic_ids):
            raise ValueError("duplicate completed topic_id")
        if self.status is EnrolmentStatus.CERTIFIED and self.certified_on is None:
            raise ValueError("CERTIFIED requires certified_on")
        if self.status is EnrolmentStatus.WITHDRAWN and self.withdrawn_on is None:
            raise ValueError("WITHDRAWN requires withdrawn_on")
        if self.certified_on is not None and self.certified_on < self.enrolled_on:
            raise ValueError("certified_on must be ≥ enrolled_on")
        if self.withdrawn_on is not None and self.withdrawn_on < self.enrolled_on:
            raise ValueError("withdrawn_on must be ≥ enrolled_on")


# --- Enrolment + completion -------------------------


def enrol(
    curriculum: Curriculum,
    *,
    enrolment_id: str,
    learner_id: str,
    level: CertLevel,
    enrolled_on: date,
    prior_enrolments: Iterable[LearnerEnrolment] = (),
) -> LearnerEnrolment:
    """Create a new ENROLLED record. Enforces the prerequisite chain.

    Pinned: HCA-II requires HCA-I CERTIFIED; HCA-III requires HCA-II
    CERTIFIED. HCA-I has no prerequisite.
    """
    prereq = _PREREQ[level]
    prior_t = tuple(e for e in prior_enrolments if e.learner_id == learner_id)
    # Already enrolled or certified at this level?
    for e in prior_t:
        if e.level is level and e.status in (
            EnrolmentStatus.ENROLLED,
            EnrolmentStatus.ELIGIBLE,
            EnrolmentStatus.CERTIFIED,
        ):
            raise ValueError(f"already enrolled or certified at {level.value}")
    if prereq is not None:
        if not any(e.level is prereq and e.status is EnrolmentStatus.CERTIFIED for e in prior_t):
            raise ValueError(f"prerequisite {prereq.value} not certified")
    return LearnerEnrolment(
        enrolment_id=enrolment_id,
        learner_id=learner_id,
        level=level,
        enrolled_on=enrolled_on,
        status=EnrolmentStatus.ENROLLED,
    )


def complete_topic(
    enrolment: LearnerEnrolment,
    curriculum: Curriculum,
    *,
    topic_id: str,
) -> LearnerEnrolment:
    """Mark a topic complete. Idempotent if already complete.

    Pinned: topic must belong to this enrolment's level.
    """
    if enrolment.status not in (
        EnrolmentStatus.ENROLLED,
        EnrolmentStatus.ELIGIBLE,
    ):
        raise ValueError(f"cannot complete topic in {enrolment.status.value} state")
    level_obj = curriculum.by_level(enrolment.level)
    if topic_id not in level_obj.all_topic_ids():
        raise ValueError(f"topic {topic_id} not in level {enrolment.level.value}")
    if topic_id in enrolment.completed_topic_ids:
        return enrolment
    new_completed = (*enrolment.completed_topic_ids, topic_id)
    # Auto-promote to ELIGIBLE if all topics done.
    all_topics = set(level_obj.all_topic_ids())
    if all_topics.issubset(set(new_completed)):
        new_status = EnrolmentStatus.ELIGIBLE
    else:
        new_status = enrolment.status
    return replace(
        enrolment,
        completed_topic_ids=new_completed,
        status=new_status,
    )


def percent_complete(enrolment: LearnerEnrolment, curriculum: Curriculum) -> float:
    """Return [0, 1] — fraction of the level's topics completed."""
    level_obj = curriculum.by_level(enrolment.level)
    total = len(level_obj.all_topic_ids())
    if total == 0:
        return 0.0
    return len(enrolment.completed_topic_ids) / total


def certify(
    enrolment: LearnerEnrolment,
    *,
    on: date,
    exam_score_pct: float,
    curriculum: Curriculum,
) -> LearnerEnrolment:
    """ELIGIBLE → CERTIFIED if exam_score ≥ level's pass_threshold."""
    if enrolment.status is not EnrolmentStatus.ELIGIBLE:
        raise ValueError(f"certify illegal from {enrolment.status.value}")
    if not 0.0 <= exam_score_pct <= 1.0:
        raise ValueError("exam_score_pct must be in [0, 1]")
    level_obj = curriculum.by_level(enrolment.level)
    if exam_score_pct < level_obj.pass_threshold_pct - 1e-9:
        raise ValueError(
            f"exam score {exam_score_pct * 100:.2f}% below threshold "
            f"{level_obj.pass_threshold_pct * 100:.2f}%"
        )
    if on < enrolment.enrolled_on:
        raise ValueError("certify on must be ≥ enrolled_on")
    return replace(enrolment, status=EnrolmentStatus.CERTIFIED, certified_on=on)


def withdraw(enrolment: LearnerEnrolment, *, on: date) -> LearnerEnrolment:
    if enrolment.status in (
        EnrolmentStatus.CERTIFIED,
        EnrolmentStatus.WITHDRAWN,
    ):
        raise ValueError(f"withdraw illegal from {enrolment.status.value}")
    if on < enrolment.enrolled_on:
        raise ValueError("withdrawn_on must be ≥ enrolled_on")
    return replace(enrolment, status=EnrolmentStatus.WITHDRAWN, withdrawn_on=on)


def is_exam_eligible(enrolment: LearnerEnrolment, curriculum: Curriculum) -> bool:
    """True iff status is ELIGIBLE (all topics done)."""
    return enrolment.status is EnrolmentStatus.ELIGIBLE


def highest_certified_level(
    enrolments: Iterable[LearnerEnrolment],
) -> CertLevel | None:
    best: CertLevel | None = None
    for e in enrolments:
        if e.status is EnrolmentStatus.CERTIFIED:
            if best is None or _LEVEL_ORDER[e.level] > _LEVEL_ORDER[best]:
                best = e.level
    return best


# --- Render ----------------------------------------


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[EnrolmentStatus, str] = {
    EnrolmentStatus.NOT_ENROLLED: "⬜",
    EnrolmentStatus.ENROLLED: "📖",
    EnrolmentStatus.ELIGIBLE: "🟡",
    EnrolmentStatus.CERTIFIED: "🎓",
    EnrolmentStatus.WITHDRAWN: "↩️",
}


_LEVEL_EMOJI: dict[CertLevel, str] = {
    CertLevel.HCA_I: "1️⃣",
    CertLevel.HCA_II: "2️⃣",
    CertLevel.HCA_III: "3️⃣",
}


def render_enrolment(enrolment: LearnerEnrolment, curriculum: Curriculum) -> str:
    pct = percent_complete(enrolment, curriculum)
    return (
        f"{_STATUS_EMOJI[enrolment.status]} "
        f"{_LEVEL_EMOJI[enrolment.level]} "
        f"{_mask(enrolment.learner_id)} "
        f"[{enrolment.status.value}]: "
        f"{len(enrolment.completed_topic_ids)} topics "
        f"({pct * 100:.0f}%)"
    )
