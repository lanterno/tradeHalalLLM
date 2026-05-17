"""Tests for `halal_trader.web.video_catalogue` (Wave 9.E).

Covers: topic registry, production lifecycle, prerequisite DAG +
cycle detection, watch path computation, no-secret render.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.video_catalogue import (
    PrerequisiteCycleError,
    ProductionStatus,
    ProductionTransitionError,
    TopicArea,
    VideoTopic,
    advance_topic,
    assert_no_prereq_cycle,
    draft_topic,
    is_publishable,
    published_topics,
    render_topic,
    total_runtime,
    watch_path,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
_DEFAULT_DURATION = timedelta(minutes=7)


# --------------------------- Enum string pins --------------------------------


def test_topic_area_string_values_pinned() -> None:
    assert TopicArea.CYCLE_PIPELINE.value == "cycle_pipeline"
    assert TopicArea.HALAL_SCREENER.value == "halal_screener"
    assert TopicArea.LLM_ENSEMBLE.value == "llm_ensemble"
    assert TopicArea.REPLAY_STORE.value == "replay_store"
    assert TopicArea.PURIFICATION_LEDGER.value == "purification_ledger"
    assert TopicArea.KILL_SWITCH.value == "kill_switch"
    assert TopicArea.BROKER_PLUGIN.value == "broker_plugin"
    assert TopicArea.OBSERVABILITY.value == "observability"


def test_production_status_string_values_pinned() -> None:
    assert ProductionStatus.DRAFTED.value == "drafted"
    assert ProductionStatus.RECORDED.value == "recorded"
    assert ProductionStatus.EDITED.value == "edited"
    assert ProductionStatus.PUBLISHED.value == "published"


# --------------------------- VideoTopic validation ---------------------------


def _topic(**overrides: object) -> VideoTopic:
    base: dict[str, object] = {
        "topic_id": "t1",
        "title": "The cycle pipeline explained",
        "area": TopicArea.CYCLE_PIPELINE,
        "estimated_duration": _DEFAULT_DURATION,
        "prerequisites": frozenset(),
        "status": ProductionStatus.DRAFTED,
        "drafted_at": T0,
        "last_status_at": T0,
    }
    base.update(overrides)
    return VideoTopic(**base)  # type: ignore[arg-type]


def test_topic_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="topic_id"):
        _topic(topic_id="")


def test_topic_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title"):
        _topic(title="")


def test_topic_accepts_5min_duration_lower_boundary() -> None:
    """Pin: roadmap pins 5-10 min band; 5min inclusive."""

    t = _topic(estimated_duration=timedelta(minutes=5))
    assert t.estimated_duration == timedelta(minutes=5)


def test_topic_accepts_10min_duration_upper_boundary() -> None:
    """Pin: 10min inclusive."""

    t = _topic(estimated_duration=timedelta(minutes=10))
    assert t.estimated_duration == timedelta(minutes=10)


def test_topic_rejects_below_5min() -> None:
    with pytest.raises(ValueError, match="estimated_duration"):
        _topic(estimated_duration=timedelta(minutes=4, seconds=59))


def test_topic_rejects_above_10min() -> None:
    """Pin: roadmap intentionally caps at 10 min."""

    with pytest.raises(ValueError, match="estimated_duration"):
        _topic(estimated_duration=timedelta(minutes=11))


def test_topic_rejects_naive_drafted_at() -> None:
    with pytest.raises(ValueError, match="drafted_at"):
        _topic(drafted_at=datetime(2026, 5, 1))


def test_topic_rejects_last_status_before_drafted() -> None:
    with pytest.raises(ValueError, match="last_status_at"):
        _topic(last_status_at=T0 - timedelta(seconds=1))


def test_topic_rejects_self_prereq() -> None:
    """Pin: topic can't be its own prerequisite (trivial cycle)."""

    with pytest.raises(PrerequisiteCycleError):
        _topic(topic_id="t1", prerequisites=frozenset({"t1"}))


def test_topic_is_frozen() -> None:
    t = _topic()
    with pytest.raises(FrozenInstanceError):
        t.title = "other"  # type: ignore[misc]


# --------------------------- draft_topic -------------------------------------


def test_draft_topic_basic() -> None:
    t = draft_topic(
        topic_id="cycle",
        title="The cycle pipeline",
        area=TopicArea.CYCLE_PIPELINE,
        estimated_duration=_DEFAULT_DURATION,
        now=T0,
    )
    assert t.status is ProductionStatus.DRAFTED


def test_draft_topic_with_prerequisites() -> None:
    t = draft_topic(
        topic_id="llm",
        title="The LLM ensemble",
        area=TopicArea.LLM_ENSEMBLE,
        estimated_duration=_DEFAULT_DURATION,
        prerequisites=["cycle"],
        now=T0,
    )
    assert "cycle" in t.prerequisites


def test_draft_topic_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        draft_topic(
            topic_id="t1",
            title="x",
            area=TopicArea.CYCLE_PIPELINE,
            estimated_duration=_DEFAULT_DURATION,
            now=datetime(2026, 5, 1),
        )


# --------------------------- advance_topic -----------------------------------


def test_advance_drafted_to_recorded() -> None:
    t = _topic()
    t = advance_topic(t, ProductionStatus.RECORDED, now=T0 + timedelta(days=1))
    assert t.status is ProductionStatus.RECORDED


def test_advance_full_lifecycle_to_published() -> None:
    t = _topic()
    t = advance_topic(t, ProductionStatus.RECORDED, now=T0)
    t = advance_topic(t, ProductionStatus.EDITED, now=T0)
    t = advance_topic(t, ProductionStatus.PUBLISHED, now=T0)
    assert t.status is ProductionStatus.PUBLISHED


def test_advance_skip_rejected() -> None:
    """Pin: cannot skip DRAFTED → EDITED."""

    t = _topic()
    with pytest.raises(ProductionTransitionError):
        advance_topic(t, ProductionStatus.EDITED, now=T0)


def test_advance_skip_to_published_rejected() -> None:
    """Pin: PUBLISHED requires three explicit forward steps."""

    t = _topic()
    with pytest.raises(ProductionTransitionError):
        advance_topic(t, ProductionStatus.PUBLISHED, now=T0)


def test_advance_backwards_rejected() -> None:
    """Pin: forward-only — can't go back to DRAFTED from RECORDED."""

    t = _topic()
    t = advance_topic(t, ProductionStatus.RECORDED, now=T0)
    with pytest.raises(ProductionTransitionError):
        advance_topic(t, ProductionStatus.DRAFTED, now=T0)


def test_advance_published_terminal() -> None:
    """Pin: PUBLISHED is terminal — no more transitions allowed.

    Content revisions require a NEW topic record so the audit trail
    of "what was published when" stays clean.
    """

    t = _topic()
    t = advance_topic(t, ProductionStatus.RECORDED, now=T0)
    t = advance_topic(t, ProductionStatus.EDITED, now=T0)
    t = advance_topic(t, ProductionStatus.PUBLISHED, now=T0)
    # Nothing valid to advance to
    with pytest.raises(ProductionTransitionError):
        advance_topic(t, ProductionStatus.PUBLISHED, now=T0)


def test_advance_naive_now_rejected() -> None:
    t = _topic()
    with pytest.raises(ValueError, match="now"):
        advance_topic(t, ProductionStatus.RECORDED, now=datetime(2026, 5, 1))


def test_advance_returns_new_state() -> None:
    """Pin: state operations return new state (immutable)."""

    original = _topic()
    new_state = advance_topic(original, ProductionStatus.RECORDED, now=T0 + timedelta(days=2))
    assert original.status is ProductionStatus.DRAFTED
    assert new_state.status is ProductionStatus.RECORDED
    assert new_state.last_status_at == T0 + timedelta(days=2)


# --------------------------- is_publishable ----------------------------------


def test_publishable_only_published() -> None:
    """Pin: only PUBLISHED is publishable."""

    drafted = _topic(status=ProductionStatus.DRAFTED)
    recorded = _topic(status=ProductionStatus.RECORDED, last_status_at=T0)
    edited = _topic(status=ProductionStatus.EDITED, last_status_at=T0)
    published = _topic(status=ProductionStatus.PUBLISHED, last_status_at=T0)
    assert is_publishable(drafted) is False
    assert is_publishable(recorded) is False
    assert is_publishable(edited) is False
    assert is_publishable(published) is True


# --------------------------- assert_no_prereq_cycle --------------------------


def test_no_cycle_with_linear_chain() -> None:
    """A → B → C (no cycle)."""

    a = _topic(topic_id="a", prerequisites=frozenset())
    b = _topic(topic_id="b", prerequisites=frozenset({"a"}))
    c = _topic(topic_id="c", prerequisites=frozenset({"b"}))
    assert_no_prereq_cycle([a, b, c])


def test_no_cycle_with_diamond() -> None:
    """A → B, A → C, B + C → D (DAG, no cycle)."""

    a = _topic(topic_id="a", prerequisites=frozenset())
    b = _topic(topic_id="b", prerequisites=frozenset({"a"}))
    c = _topic(topic_id="c", prerequisites=frozenset({"a"}))
    d = _topic(topic_id="d", prerequisites=frozenset({"b", "c"}))
    assert_no_prereq_cycle([a, b, c, d])


def test_cycle_caught_two_topic() -> None:
    """A → B → A (cycle of length 2)."""

    a = _topic(topic_id="a", prerequisites=frozenset({"b"}))
    b = _topic(topic_id="b", prerequisites=frozenset({"a"}))
    with pytest.raises(PrerequisiteCycleError):
        assert_no_prereq_cycle([a, b])


def test_cycle_caught_three_topic() -> None:
    """A → B → C → A (cycle of length 3)."""

    a = _topic(topic_id="a", prerequisites=frozenset({"c"}))
    b = _topic(topic_id="b", prerequisites=frozenset({"a"}))
    c = _topic(topic_id="c", prerequisites=frozenset({"b"}))
    with pytest.raises(PrerequisiteCycleError) as exc_info:
        assert_no_prereq_cycle([a, b, c])
    assert len(exc_info.value.cycle_path) >= 3


def test_external_prereq_doesnt_raise() -> None:
    """Pin: a prereq that's not in the batch (external reference) is
    accepted — operators iteratively add topics."""

    a = _topic(topic_id="a", prerequisites=frozenset({"external"}))
    assert_no_prereq_cycle([a])  # no raise


# --------------------------- watch_path --------------------------------------


def test_watch_path_no_prereqs() -> None:
    """A topic with no prereqs has a single-element watch path."""

    a = _topic(topic_id="a", prerequisites=frozenset())
    path = watch_path("a", topics=[a])
    assert path == ("a",)


def test_watch_path_linear_chain() -> None:
    """A → B → C: watching C requires A then B then C."""

    a = _topic(topic_id="a", prerequisites=frozenset())
    b = _topic(topic_id="b", prerequisites=frozenset({"a"}))
    c = _topic(topic_id="c", prerequisites=frozenset({"b"}))
    path = watch_path("c", topics=[a, b, c])
    assert path == ("a", "b", "c")


def test_watch_path_diamond() -> None:
    """A → B, A → C, B + C → D: watching D includes A once."""

    a = _topic(topic_id="a", prerequisites=frozenset())
    b = _topic(topic_id="b", prerequisites=frozenset({"a"}))
    c = _topic(topic_id="c", prerequisites=frozenset({"a"}))
    d = _topic(topic_id="d", prerequisites=frozenset({"b", "c"}))
    path = watch_path("d", topics=[a, b, c, d])
    # A appears exactly once; before B, C, D
    assert path[-1] == "d"
    assert path.count("a") == 1
    a_idx = path.index("a")
    b_idx = path.index("b")
    c_idx = path.index("c")
    d_idx = path.index("d")
    assert a_idx < b_idx
    assert a_idx < c_idx
    assert b_idx < d_idx
    assert c_idx < d_idx


def test_watch_path_unknown_topic_raises() -> None:
    a = _topic(topic_id="a", prerequisites=frozenset())
    with pytest.raises(KeyError):
        watch_path("nonexistent", topics=[a])


def test_watch_path_external_prereq_skipped() -> None:
    """Pin: external prereq is skipped, not raised — the catalogue
    builder sees only what's in the batch."""

    a = _topic(topic_id="a", prerequisites=frozenset({"external"}))
    path = watch_path("a", topics=[a])
    assert "external" not in path
    assert "a" in path


def test_watch_path_cycle_raises() -> None:
    """Pin: watch_path also detects cycles."""

    a = _topic(topic_id="a", prerequisites=frozenset({"b"}))
    b = _topic(topic_id="b", prerequisites=frozenset({"a"}))
    with pytest.raises(PrerequisiteCycleError):
        watch_path("a", topics=[a, b])


# --------------------------- total_runtime -----------------------------------


def test_total_runtime_basic() -> None:
    a = _topic(estimated_duration=timedelta(minutes=7))
    b = _topic(
        topic_id="t2",
        estimated_duration=timedelta(minutes=8),
    )
    assert total_runtime([a, b]) == timedelta(minutes=15)


def test_total_runtime_empty() -> None:
    assert total_runtime([]) == timedelta()


# --------------------------- published_topics --------------------------------


def test_published_topics_filters() -> None:
    drafted = _topic(topic_id="t_d", status=ProductionStatus.DRAFTED)
    published = _topic(
        topic_id="t_p",
        status=ProductionStatus.PUBLISHED,
        last_status_at=T0,
    )
    result = published_topics([drafted, published])
    ids = [t.topic_id for t in result]
    assert ids == ["t_p"]


def test_published_topics_sorted() -> None:
    """Pin: returned in topic_id-sorted order."""

    p1 = _topic(
        topic_id="zebra",
        status=ProductionStatus.PUBLISHED,
        last_status_at=T0,
    )
    p2 = _topic(
        topic_id="alpha",
        status=ProductionStatus.PUBLISHED,
        last_status_at=T0,
    )
    result = published_topics([p1, p2])
    assert [t.topic_id for t in result] == ["alpha", "zebra"]


# --------------------------- render_topic ------------------------------------


def test_render_topic_includes_id_and_title() -> None:
    t = _topic(topic_id="cycle_pipeline", title="The cycle pipeline")
    out = render_topic(t)
    assert "cycle_pipeline" in out
    assert "The cycle pipeline" in out


def test_render_topic_area_emoji() -> None:
    cycle = _topic(area=TopicArea.CYCLE_PIPELINE)
    halal = _topic(area=TopicArea.HALAL_SCREENER)
    llm = _topic(area=TopicArea.LLM_ENSEMBLE)
    halt = _topic(area=TopicArea.KILL_SWITCH)
    assert "🔄" in render_topic(cycle)
    assert "✅" in render_topic(halal)
    assert "🧠" in render_topic(llm)
    assert "🛑" in render_topic(halt)


def test_render_topic_status_emoji() -> None:
    drafted = _topic(status=ProductionStatus.DRAFTED)
    published = _topic(status=ProductionStatus.PUBLISHED, last_status_at=T0)
    assert "📝" in render_topic(drafted)
    assert "🚀" in render_topic(published)


def test_render_topic_includes_duration() -> None:
    t = _topic(estimated_duration=timedelta(minutes=7, seconds=30))
    out = render_topic(t)
    assert "7.5 min" in out


def test_render_topic_includes_prerequisites() -> None:
    t = _topic(prerequisites=frozenset({"prereq1", "prereq2"}))
    out = render_topic(t)
    assert "prereq1" in out
    assert "prereq2" in out


def test_render_topic_no_prereqs_dash() -> None:
    """Pin: empty prereqs render as `—` not blank."""

    t = _topic(prerequisites=frozenset())
    out = render_topic(t)
    assert "—" in out


def test_render_topic_no_secret_leak() -> None:
    """Pin: render never includes operator credentials, hosting
    service tokens, or raw transcripts."""

    t = _topic()
    out = render_topic(t)
    assert "api_key" not in out.lower()
    assert "youtube" not in out.lower()
    assert "vimeo" not in out.lower()
    assert "transcript" not in out.lower()
    assert "bearer" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_topic_lifecycle() -> None:
    """Real-world: cycle pipeline video drafted → recorded → edited → published."""

    t = draft_topic(
        topic_id="cycle_pipeline_v1",
        title="The cycle pipeline (deep-dive)",
        area=TopicArea.CYCLE_PIPELINE,
        estimated_duration=timedelta(minutes=8),
        now=T0,
    )
    assert is_publishable(t) is False
    t = advance_topic(t, ProductionStatus.RECORDED, now=T0 + timedelta(days=7))
    t = advance_topic(t, ProductionStatus.EDITED, now=T0 + timedelta(days=14))
    t = advance_topic(t, ProductionStatus.PUBLISHED, now=T0 + timedelta(days=21))
    assert is_publishable(t) is True


def test_e2e_realistic_catalogue_dag() -> None:
    """Real-world: build a 4-topic catalogue with the roadmap-named
    topics, validate no cycles, compute watch paths."""

    cycle = draft_topic(
        topic_id="cycle",
        title="The cycle pipeline",
        area=TopicArea.CYCLE_PIPELINE,
        estimated_duration=timedelta(minutes=7),
        now=T0,
    )
    halal = draft_topic(
        topic_id="halal",
        title="The halal screener",
        area=TopicArea.HALAL_SCREENER,
        estimated_duration=timedelta(minutes=8),
        prerequisites=["cycle"],
        now=T0,
    )
    llm = draft_topic(
        topic_id="llm",
        title="The LLM ensemble",
        area=TopicArea.LLM_ENSEMBLE,
        estimated_duration=timedelta(minutes=9),
        prerequisites=["cycle"],
        now=T0,
    )
    replay = draft_topic(
        topic_id="replay",
        title="The replay store",
        area=TopicArea.REPLAY_STORE,
        estimated_duration=timedelta(minutes=6),
        prerequisites=["cycle"],
        now=T0,
    )

    catalogue = [cycle, halal, llm, replay]
    assert_no_prereq_cycle(catalogue)

    # Watch path for halal includes cycle first
    halal_path = watch_path("halal", topics=catalogue)
    assert halal_path == ("cycle", "halal")

    # Total runtime is 30 min
    assert total_runtime(catalogue) == timedelta(minutes=30)


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal topic states."""

    def build() -> VideoTopic:
        t = draft_topic(
            topic_id="cycle",
            title="The cycle pipeline",
            area=TopicArea.CYCLE_PIPELINE,
            estimated_duration=timedelta(minutes=7),
            now=T0,
        )
        return advance_topic(t, ProductionStatus.RECORDED, now=T0)

    a = build()
    b = build()
    assert a == b
