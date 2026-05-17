"""Tests for `core/cycle_timeline.py` (per-cycle stage timeline
aggregator).

Pins the start/end pairing semantic, the open-stage and orphan-end
recovery contracts, the duration computation (prefer elapsed_ms;
fall back to wall-clock), the bottleneck ranking, the
`from_bus_payload` adapter, and the markdown render shape.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.core.cycle_timeline import (
    CycleTimeline,
    StageBottleneck,
    StageEvent,
    StageEventType,
    StageRun,
    build_timeline,
)


def _start(name: str, at: datetime, **attrs: str) -> StageEvent:
    return StageEvent(at=at, event_type=StageEventType.START, name=name, attrs=attrs)


def _end(
    name: str,
    at: datetime,
    *,
    elapsed_ms: float | None = None,
    error: str | None = None,
) -> StageEvent:
    return StageEvent(
        at=at,
        event_type=StageEventType.END,
        name=name,
        elapsed_ms=elapsed_ms,
        error=error,
    )


# ── empty input ──────────────────────────────────────────


def test_empty_events_returns_empty_timeline():
    """Pin: no events → no stages → no error. The dashboard
    'no data yet' state renders off `stage_count == 0`."""
    t = build_timeline([])
    assert isinstance(t, CycleTimeline)
    assert t.stage_count == 0
    assert t.stages == []
    assert t.bottlenecks == []
    assert t.total_duration_ms == 0.0


# ── basic pairing ───────────────────────────────────────


def test_single_stage_pairs_start_and_end():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t1 = t0 + timedelta(milliseconds=240)
    t = build_timeline(
        [
            _start("broker.fetch", t0),
            _end("broker.fetch", t1, elapsed_ms=240.0),
        ]
    )
    assert t.stage_count == 1
    s = t.stages[0]
    assert s.name == "broker.fetch"
    assert s.start_at == t0
    assert s.end_at == t1
    assert s.duration_ms == 240.0
    assert s.status == "ok"


def test_multiple_stages_in_order():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("a", t0),
            _end("a", t0 + timedelta(milliseconds=10), elapsed_ms=10.0),
            _start("b", t0 + timedelta(milliseconds=10)),
            _end("b", t0 + timedelta(milliseconds=20), elapsed_ms=10.0),
        ]
    )
    assert [s.name for s in t.stages] == ["a", "b"]


def test_total_duration_spans_first_start_to_last_end():
    """Total wall-clock from earliest start to latest end."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("a", t0),
            _end("a", t0 + timedelta(seconds=1), elapsed_ms=1000.0),
            _start("b", t0 + timedelta(seconds=1)),
            _end("b", t0 + timedelta(seconds=3), elapsed_ms=2000.0),
        ]
    )
    # 3000ms wall-clock; 3000ms summed across stages — they happen
    # to match here because no overlap.
    assert t.total_duration_ms == 3000.0


# ── duration source preference ───────────────────────────


def test_duration_uses_elapsed_ms_when_present():
    """Pin: precise pipeline measurement wins over wall-clock
    derivation. Wall-clock can be off by a few ms when the bus
    publish delays the END event slightly."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("x", t0),
            # END 200ms later wall-clock, but elapsed_ms says 150ms.
            _end("x", t0 + timedelta(milliseconds=200), elapsed_ms=150.0),
        ]
    )
    assert t.stages[0].duration_ms == 150.0


def test_duration_falls_back_to_wall_clock_without_elapsed_ms():
    """Legacy / partial replay rows have no elapsed_ms — must
    still produce a usable duration rather than None."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("x", t0),
            _end("x", t0 + timedelta(milliseconds=200)),
        ]
    )
    assert t.stages[0].duration_ms == 200.0


# ── error / open stage handling ──────────────────────────


def test_end_with_error_marks_status_error():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("flaky", t0),
            _end(
                "flaky",
                t0 + timedelta(milliseconds=50),
                elapsed_ms=50.0,
                error="ConnectionError",
            ),
        ]
    )
    s = t.stages[0]
    assert s.status == "error"
    assert s.error == "ConnectionError"
    assert t.error_count == 1


def test_start_without_end_marks_status_open():
    """Pin: a cycle killed mid-stage X must show stage X in the
    timeline as `open`, not be silently dropped."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("a", t0),
            _end("a", t0 + timedelta(milliseconds=10), elapsed_ms=10.0),
            _start("killed_in_progress", t0 + timedelta(milliseconds=10)),
            # no matching end
        ]
    )
    open_stage = next(s for s in t.stages if s.name == "killed_in_progress")
    assert open_stage.status == "open"
    assert open_stage.duration_ms is None
    assert open_stage.end_at is None
    assert t.open_count == 1


def test_orphan_end_synthesises_zero_width_run():
    """Live bus occasionally drops a START under backpressure;
    the END always carries elapsed_ms. Pin the recovery: the
    stage is rendered as a zero-wall-clock run with the
    elapsed_ms duration."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _end("orphan", t0, elapsed_ms=42.0),
        ]
    )
    assert t.stage_count == 1
    s = t.stages[0]
    assert s.start_at == s.end_at == t0
    assert s.duration_ms == 42.0
    assert s.status == "ok"


def test_two_starts_same_name_first_one_flushed_as_open():
    """Defensive: if a refactor accidentally publishes two
    consecutive starts for the same name without an end in
    between, the first is flushed as `open` so it's not silently
    overwritten."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("dup", t0),
            _start("dup", t0 + timedelta(milliseconds=10)),
            _end("dup", t0 + timedelta(milliseconds=20), elapsed_ms=10.0),
        ]
    )
    # First start → open; second start + end → ok.
    statuses = [s.status for s in t.stages]
    assert "open" in statuses
    assert "ok" in statuses


# ── bottleneck ranking ───────────────────────────────────


def test_bottleneck_sorts_by_descending_duration():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    events = []
    for name, ms in [("fast", 5), ("slow", 500), ("medium", 50)]:
        events.append(_start(name, t0))
        events.append(_end(name, t0 + timedelta(milliseconds=ms), elapsed_ms=ms))
    t = build_timeline(events)
    names = [b.name for b in t.bottlenecks]
    assert names[0] == "slow"
    assert names[-1] == "fast"


def test_bottleneck_capped_at_top_5_by_default():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    events = []
    for i in range(10):
        events.append(_start(f"s{i}", t0))
        events.append(_end(f"s{i}", t0 + timedelta(milliseconds=i + 1), elapsed_ms=i + 1))
    t = build_timeline(events)
    assert len(t.bottlenecks) == 5


def test_bottleneck_excludes_open_stages():
    """Open stages have unknown duration — they shouldn't fake
    a 0ms entry in the ranking. Pin the exclusion."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("done", t0),
            _end("done", t0 + timedelta(milliseconds=10), elapsed_ms=10.0),
            _start("open_stage", t0 + timedelta(milliseconds=10)),
        ]
    )
    names = {b.name for b in t.bottlenecks}
    assert "open_stage" not in names


def test_bottleneck_pct_of_total_sums_below_one():
    """Sanity: the percentages can't sum above 100% (they may
    sum below 100% if stages overlapped or the total includes
    inter-stage gaps)."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("a", t0),
            _end("a", t0 + timedelta(milliseconds=100), elapsed_ms=100.0),
            _start("b", t0 + timedelta(milliseconds=100)),
            _end("b", t0 + timedelta(milliseconds=200), elapsed_ms=100.0),
        ]
    )
    total_pct = sum(b.pct_of_total for b in t.bottlenecks)
    assert total_pct <= 1.0 + 1e-9


# ── from_bus_payload adapter ─────────────────────────────


def test_from_bus_payload_parses_start_topic():
    at = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    ev = StageEvent.from_bus_payload(
        "cycle.stage.start",
        {"name": "test_stage", "extra": "value"},
        at=at,
    )
    assert ev.event_type == StageEventType.START
    assert ev.name == "test_stage"
    assert ev.attrs == {"extra": "value"}


def test_from_bus_payload_parses_end_topic_with_elapsed_ms():
    at = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    ev = StageEvent.from_bus_payload(
        "cycle.stage.end",
        {"name": "x", "elapsed_ms": 123.5, "error": "boom"},
        at=at,
    )
    assert ev.event_type == StageEventType.END
    assert ev.elapsed_ms == 123.5
    assert ev.error == "boom"
    # canonical fields stripped from attrs
    assert "elapsed_ms" not in ev.attrs
    assert "error" not in ev.attrs


def test_from_bus_payload_rejects_unknown_topic():
    """Pin: a wiring bug that publishes the wrong topic must
    surface immediately rather than silently producing an empty
    timeline."""
    with pytest.raises(ValueError, match="unrecognised stage topic"):
        StageEvent.from_bus_payload("cycle.cycle.complete", {}, at=datetime.now(UTC))


# ── markdown render ──────────────────────────────────────


def test_markdown_includes_total_duration():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("a", t0),
            _end("a", t0 + timedelta(milliseconds=500), elapsed_ms=500.0),
        ]
    )
    assert "500ms" in t.markdown or "0.50s" in t.markdown
    assert "Total" in t.markdown


def test_markdown_renders_seconds_for_long_durations():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("a", t0),
            _end("a", t0 + timedelta(seconds=2, milliseconds=500), elapsed_ms=2500.0),
        ]
    )
    assert "2.50s" in t.markdown


def test_markdown_includes_bottleneck_table():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    events = []
    for name, ms in [("a", 100), ("b", 200)]:
        events.append(_start(name, t0))
        events.append(_end(name, t0 + timedelta(milliseconds=ms), elapsed_ms=ms))
    t = build_timeline(events)
    assert "## Top stages by duration" in t.markdown


def test_markdown_marks_errored_stages():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("flaky", t0),
            _end(
                "flaky",
                t0 + timedelta(milliseconds=50),
                elapsed_ms=50.0,
                error="boom",
            ),
        ]
    )
    assert "✘" in t.markdown
    assert "boom" in t.markdown


def test_markdown_marks_open_stages():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline([_start("never_finished", t0)])
    assert "no end event recorded" in t.markdown


def test_markdown_includes_health_line_when_errors_or_opens():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline([_start("never_finished", t0)])
    assert "Health:" in t.markdown
    assert "unfinished" in t.markdown


def test_markdown_omits_health_line_when_clean():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("a", t0),
            _end("a", t0 + timedelta(milliseconds=10), elapsed_ms=10.0),
        ]
    )
    assert "Health:" not in t.markdown


# ── output structure ─────────────────────────────────────


def test_timeline_is_immutable():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [_start("a", t0), _end("a", t0 + timedelta(milliseconds=10), elapsed_ms=10.0)]
    )
    assert isinstance(t, CycleTimeline)
    with pytest.raises(Exception):
        t.total_duration_ms = 0.0  # type: ignore[misc]


def test_stage_run_is_immutable():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [_start("a", t0), _end("a", t0 + timedelta(milliseconds=10), elapsed_ms=10.0)]
    )
    s = t.stages[0]
    assert isinstance(s, StageRun)
    with pytest.raises(Exception):
        s.name = "x"  # type: ignore[misc]


def test_bottleneck_is_a_dataclass():
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [_start("a", t0), _end("a", t0 + timedelta(milliseconds=10), elapsed_ms=10.0)]
    )
    assert isinstance(t.bottlenecks[0], StageBottleneck)


def test_attrs_pass_through_to_stage_run():
    """Pin so a future stage type that carries extra context
    (e.g. `pair=BTCUSDT` on a per-pair stage) flows through to
    the timeline render."""
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t = build_timeline(
        [
            _start("compute", t0, pair="BTCUSDT"),
            _end("compute", t0 + timedelta(milliseconds=10), elapsed_ms=10.0),
        ]
    )
    assert t.stages[0].attrs == {"pair": "BTCUSDT"}
