"""Architecture deep-dive video catalogue + production state machine.

The roadmap pins Wave 9.E: "Record 5-10 minute videos walking
through key parts of the architecture: the cycle pipeline, the
halal screener, the LLM ensemble, the replay store. Posted
publicly." This module is the **pure-Python topic registry +
production state machine** the operator consults to track which
architecture explainers exist, which are recorded vs published,
and which prerequisites each viewer needs.

Picked a focused topic catalogue + state machine over an
ad-hoc spreadsheet because (a) the prerequisites graph (the LLM
ensemble video assumes the viewer watched the cycle pipeline
video first) needs deterministic ordering — a regression-pinned
DAG that operators can render as a "watch path" guide; (b) the
production lifecycle (drafted → recorded → edited → published)
mirrors the wave 6.I distillation deployment + 8.C DR drill state
machines so operators have one mental model for "promoting a
video to public-release"; (c) duration estimates per topic are
factual catalogue data — encoding them once means the CDN /
hosting cost projection consults one source.

Pinned semantics:
- **Closed-set TopicArea enum.** Adding an area is a code review
  change so the viewer's mental map of "what's covered?" stays
  stable as new videos land.
- **Production lifecycle is forward-only.** DRAFTED → RECORDED →
  EDITED → PUBLISHED; once PUBLISHED, content can be revised
  via a NEW topic record (not by editing the published one).
- **Prerequisites are a strict DAG.** A topic can't be its own
  prereq (cycle); validation rejects cycles at construction.
- **Duration in the 5-10 minute band per roadmap.** Topics
  shorter than 5min or longer than 10min raise at construction —
  the roadmap pins the format intentionally.
- **Render output never includes operator credentials, hosting
  service tokens, or raw transcripts.** Mirrors no-secret
  patterns of upstream waves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class TopicArea(str, Enum):
    """Closed-set architecture topic areas.

    Pinned string values for JSON / DB stability. The four roadmap-
    pinned areas (cycle pipeline, halal screener, LLM ensemble,
    replay store) plus a few foundational + advanced ones that
    naturally expand the catalogue.
    """

    CYCLE_PIPELINE = "cycle_pipeline"
    HALAL_SCREENER = "halal_screener"
    LLM_ENSEMBLE = "llm_ensemble"
    REPLAY_STORE = "replay_store"
    PURIFICATION_LEDGER = "purification_ledger"
    KILL_SWITCH = "kill_switch"
    BROKER_PLUGIN = "broker_plugin"
    OBSERVABILITY = "observability"


class ProductionStatus(str, Enum):
    """Video production lifecycle.

    Pinned values; forward-only. PUBLISHED is terminal — content
    revisions require a new topic record so the audit trail of
    "what was published when" stays clean.
    """

    DRAFTED = "drafted"
    RECORDED = "recorded"
    EDITED = "edited"
    PUBLISHED = "published"


_PRODUCTION_ORDER: tuple[ProductionStatus, ...] = (
    ProductionStatus.DRAFTED,
    ProductionStatus.RECORDED,
    ProductionStatus.EDITED,
    ProductionStatus.PUBLISHED,
)


_MIN_DURATION = timedelta(minutes=5)
_MAX_DURATION = timedelta(minutes=10)


class ProductionTransitionError(Exception):
    """Raised when a status transition violates forward-only order."""

    def __init__(self, current: ProductionStatus, attempted: ProductionStatus) -> None:
        super().__init__(f"cannot transition from {current.value} to {attempted.value}")
        self.current = current
        self.attempted = attempted


class PrerequisiteCycleError(Exception):
    """Raised when a topic's prerequisites would form a cycle."""

    def __init__(self, topic_id: str, cycle_path: tuple[str, ...]) -> None:
        super().__init__(f"topic {topic_id!r} prerequisites form cycle: {' → '.join(cycle_path)}")
        self.topic_id = topic_id
        self.cycle_path = cycle_path


@dataclass(frozen=True)
class VideoTopic:
    """One architecture deep-dive video topic.

    `prerequisites` is a frozenset of topic IDs the viewer should
    have watched first. The prerequisite check is a separate
    function (cycle detection requires graph context).
    """

    topic_id: str
    title: str
    area: TopicArea
    estimated_duration: timedelta
    prerequisites: frozenset[str]
    status: ProductionStatus
    drafted_at: datetime
    last_status_at: datetime

    def __post_init__(self) -> None:
        if not self.topic_id or not self.topic_id.strip():
            raise ValueError("topic_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not _MIN_DURATION <= self.estimated_duration <= _MAX_DURATION:
            raise ValueError(
                f"estimated_duration must be in [5min, 10min], got {self.estimated_duration}"
            )
        if self.drafted_at.tzinfo is None:
            raise ValueError("drafted_at must be timezone-aware")
        if self.last_status_at.tzinfo is None:
            raise ValueError("last_status_at must be timezone-aware")
        if self.last_status_at < self.drafted_at:
            raise ValueError("last_status_at must be >= drafted_at")
        # Self-prereq check (the trivial cycle case)
        if self.topic_id in self.prerequisites:
            raise PrerequisiteCycleError(self.topic_id, (self.topic_id, self.topic_id))


def draft_topic(
    *,
    topic_id: str,
    title: str,
    area: TopicArea,
    estimated_duration: timedelta,
    prerequisites: Iterable[str] = (),
    now: datetime,
) -> VideoTopic:
    """Build a fresh DRAFTED topic record."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return VideoTopic(
        topic_id=topic_id,
        title=title,
        area=area,
        estimated_duration=estimated_duration,
        prerequisites=frozenset(prerequisites),
        status=ProductionStatus.DRAFTED,
        drafted_at=now,
        last_status_at=now,
    )


def _check_forward(current: ProductionStatus, target: ProductionStatus) -> None:
    """Ensure target is exactly one step forward in canonical order."""

    cur_idx = _PRODUCTION_ORDER.index(current)
    try:
        target_idx = _PRODUCTION_ORDER.index(target)
    except ValueError as exc:
        raise ProductionTransitionError(current, target) from exc
    if target_idx != cur_idx + 1:
        raise ProductionTransitionError(current, target)


def advance_topic(topic: VideoTopic, to_status: ProductionStatus, *, now: datetime) -> VideoTopic:
    """Move a topic forward one production step.

    DRAFTED → RECORDED → EDITED → PUBLISHED. Skipping or
    backtracking raises ProductionTransitionError. PUBLISHED is
    terminal.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    _check_forward(topic.status, to_status)
    return VideoTopic(
        topic_id=topic.topic_id,
        title=topic.title,
        area=topic.area,
        estimated_duration=topic.estimated_duration,
        prerequisites=topic.prerequisites,
        status=to_status,
        drafted_at=topic.drafted_at,
        last_status_at=now,
    )


def is_publishable(topic: VideoTopic) -> bool:
    """True if topic.status is PUBLISHED."""

    return topic.status is ProductionStatus.PUBLISHED


def assert_no_prereq_cycle(topics: Iterable[VideoTopic]) -> None:
    """Validate the full catalogue's prerequisite graph for cycles.

    Operators run this before publishing the catalogue so a
    misconfigured prereq edge fails CI rather than ships.
    """

    by_id = {t.topic_id: t for t in topics}

    def visit(topic_id: str, in_progress: tuple[str, ...]) -> None:
        if topic_id in in_progress:
            cycle_start_idx = in_progress.index(topic_id)
            cycle_path = in_progress[cycle_start_idx:] + (topic_id,)
            raise PrerequisiteCycleError(topic_id, cycle_path)
        topic = by_id.get(topic_id)
        if topic is None:
            return  # External prereq (not in this batch); fine
        for prereq in topic.prerequisites:
            visit(prereq, in_progress + (topic_id,))

    for topic in topics:
        visit(topic.topic_id, ())


def watch_path(target_topic_id: str, *, topics: Iterable[VideoTopic]) -> tuple[str, ...]:
    """Return a topological-order watch path that ends at target.

    The returned tuple is the set of topics the viewer should watch
    in order before (and including) the target. Useful for the
    "before you watch X, watch Y first" UI hint.
    """

    by_id = {t.topic_id: t for t in topics}
    if target_topic_id not in by_id:
        raise KeyError(f"topic {target_topic_id!r} not in catalogue")

    visited: set[str] = set()
    order: list[str] = []

    def visit(topic_id: str, stack: tuple[str, ...]) -> None:
        if topic_id in visited:
            return
        if topic_id in stack:
            cycle_start = stack.index(topic_id)
            cycle_path = stack[cycle_start:] + (topic_id,)
            raise PrerequisiteCycleError(topic_id, cycle_path)
        topic = by_id.get(topic_id)
        if topic is None:
            return
        for prereq in sorted(topic.prerequisites):
            visit(prereq, stack + (topic_id,))
        visited.add(topic_id)
        order.append(topic_id)

    visit(target_topic_id, ())
    return tuple(order)


def total_runtime(topics: Iterable[VideoTopic]) -> timedelta:
    """Sum the estimated durations across a topic set.

    Operators consult this for "if a viewer watches every video,
    what's the total runtime?" — useful for landing-page copy.
    """

    return sum((t.estimated_duration for t in topics), start=timedelta())


def published_topics(
    topics: Iterable[VideoTopic],
) -> tuple[VideoTopic, ...]:
    """Return only PUBLISHED topics (sorted by topic_id for determinism)."""

    return tuple(
        sorted(
            (t for t in topics if t.status is ProductionStatus.PUBLISHED),
            key=lambda t: t.topic_id,
        )
    )


_AREA_EMOJI: dict[TopicArea, str] = {
    TopicArea.CYCLE_PIPELINE: "🔄",
    TopicArea.HALAL_SCREENER: "✅",
    TopicArea.LLM_ENSEMBLE: "🧠",
    TopicArea.REPLAY_STORE: "💾",
    TopicArea.PURIFICATION_LEDGER: "🤲",
    TopicArea.KILL_SWITCH: "🛑",
    TopicArea.BROKER_PLUGIN: "🔌",
    TopicArea.OBSERVABILITY: "📊",
}


_STATUS_EMOJI: dict[ProductionStatus, str] = {
    ProductionStatus.DRAFTED: "📝",
    ProductionStatus.RECORDED: "🎥",
    ProductionStatus.EDITED: "✂️",
    ProductionStatus.PUBLISHED: "🚀",
}


def render_topic(topic: VideoTopic) -> str:
    """Format a topic for ops display.

    No-secret-leak: never includes operator credentials, hosting
    service tokens, or raw transcripts. Shows topic metadata only.
    """

    area_emoji = _AREA_EMOJI[topic.area]
    status_emoji = _STATUS_EMOJI[topic.status]
    duration_min = topic.estimated_duration.total_seconds() / 60.0
    prereqs_str = ", ".join(sorted(topic.prerequisites)) if topic.prerequisites else "—"
    return (
        f"{status_emoji}{area_emoji} {topic.topic_id}: {topic.title}\n"
        f"  area: {topic.area.value}\n"
        f"  duration: {duration_min:.1f} min\n"
        f"  prerequisites: {prereqs_str}\n"
        f"  status: {topic.status.value}"
    )


__all__ = [
    "PrerequisiteCycleError",
    "ProductionStatus",
    "ProductionTransitionError",
    "TopicArea",
    "VideoTopic",
    "advance_topic",
    "assert_no_prereq_cycle",
    "draft_topic",
    "is_publishable",
    "published_topics",
    "render_topic",
    "total_runtime",
    "watch_path",
]
