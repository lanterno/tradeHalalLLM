"""Per-cycle stage timeline aggregator.

Round-4 wave 5.A: the existing `core/cycle_pipeline.py` already
publishes ``cycle.stage.start`` / ``cycle.stage.end`` events on the
bus and the ``/ws/cycle`` WebSocket streams them live. This module
is the **off-line replay layer** — given a flat list of stage
events from one cycle (live, replay store, or a JSON dump), build
a structured `CycleTimeline` with per-stage durations, bottleneck
identification, and a renderer.

The aggregator is the data side of the dashboard's "click any
historical cycle, see a Gantt-shaped timeline" feature. Building
it as a pure function on `StageEvent` records means the same logic
serves the live stream (drop-in over the WebSocket payload) and
the replay store (one-shot per cycle on demand) — no SQL, no
async, no WebSocket plumbing in this file.

Why the explicit StageEvent dataclass rather than parsing the bus
payload directly: the bus delivers loose dicts; the aggregator
wants tight typing so a missing ``elapsed_ms`` field surfaces as
"not measured" rather than KeyError-during-render. The
`from_bus_payload` classmethod handles the conversion at the
boundary.

Halal alignment: timing data is observability only — never feeds
back into a sizing or entry decision. The aggregator is read-only.

Pure-Python; no NumPy, no DB, no async. Frozen dataclasses safe
to cache for the dashboard's "this cycle's timeline" tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable, Mapping, Sequence

# ── Inputs ────────────────────────────────────────────────


class StageEventType(str, Enum):
    """The two event types the cycle pipeline emits per stage.

    Matches the `cycle.stage.start` / `cycle.stage.end` topic
    suffixes published by `core/cycle_pipeline.stage(...)`.
    """

    START = "start"
    END = "end"


@dataclass(frozen=True)
class StageEvent:
    """One stage start / end observation.

    ``elapsed_ms`` is set on the END event (and may be None on
    legacy / partial replay store rows). ``error`` is set on END
    when the stage body raised; the existing pipeline also
    swallows the exception on stages flagged `swallow=True`, so
    the error string is informational, not a kill signal.
    """

    at: datetime
    event_type: StageEventType
    name: str
    elapsed_ms: float | None = None
    error: str | None = None
    attrs: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_bus_payload(
        cls,
        topic: str,
        payload: Mapping[str, object],
        *,
        at: datetime,
    ) -> "StageEvent":
        """Adapter from the live bus payload shape.

        Topics are ``cycle.stage.start`` / ``cycle.stage.end``;
        anything else raises so a wiring bug surfaces immediately
        rather than silently producing an empty timeline.
        """
        if topic.endswith(".start"):
            event_type = StageEventType.START
        elif topic.endswith(".end"):
            event_type = StageEventType.END
        else:
            raise ValueError(f"unrecognised stage topic {topic!r}")
        name = str(payload.get("name", "<unnamed>"))
        elapsed_raw = payload.get("elapsed_ms")
        elapsed: float | None
        if isinstance(elapsed_raw, (int, float)):
            elapsed = float(elapsed_raw)
        else:
            elapsed = None
        error = payload.get("error")
        # Strip the canonical fields from the attrs slice so
        # downstream attrs don't carry redundant copies.
        skip = {"name", "elapsed_ms", "error"}
        attrs = {k: str(v) for k, v in payload.items() if k not in skip}
        return cls(
            at=at,
            event_type=event_type,
            name=name,
            elapsed_ms=elapsed,
            error=str(error) if error is not None else None,
            attrs=attrs,
        )


# ── Outputs ───────────────────────────────────────────────


@dataclass(frozen=True)
class StageRun:
    """One stage's full execution as observed in the timeline.

    ``duration_ms`` is the canonical duration: prefers the END
    event's `elapsed_ms` (the pipeline measures it precisely with
    `time.monotonic()`); falls back to wall-clock `(end_at -
    start_at)` if `elapsed_ms` is missing — pin so legacy events
    still produce a usable number rather than None.

    ``status`` is `"ok"` / `"error"` / `"open"` (start observed
    but no matching end — usually means the cycle was killed
    mid-flight).
    """

    name: str
    start_at: datetime
    end_at: datetime | None
    duration_ms: float | None
    status: str
    error: str | None
    attrs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StageBottleneck:
    """One entry in the "where did the time go" ranking."""

    name: str
    duration_ms: float
    pct_of_total: float


@dataclass(frozen=True)
class CycleTimeline:
    """The composed view, ready for rendering / JSON-serialising.

    ``total_duration_ms`` is wall-clock from the first stage's
    start to the last stage's end (or to the open stage's start
    if the cycle didn't complete). ``stages`` is in order-of-
    first-observation — the dashboard renders a Gantt-shaped
    chart from this list directly.
    """

    cycle_started_at: datetime
    cycle_ended_at: datetime | None
    total_duration_ms: float
    stage_count: int
    error_count: int
    open_count: int
    stages: list[StageRun]
    bottlenecks: list[StageBottleneck]
    markdown: str = ""


# ── Aggregation ──────────────────────────────────────────


def _pair_events(events: Sequence[StageEvent]) -> list[StageRun]:
    """Pair start/end events into `StageRun` records.

    Pin: a START without a matching END is reported as
    ``status="open"`` rather than dropped — the operator wants
    to see *that* the cycle was killed mid-stage X, not a
    timeline that silently omits the killer. An END without a
    matching START is treated as a synthetic start at the END's
    own timestamp (with `duration_ms` from `elapsed_ms`), since
    the live bus does occasionally drop a START event under
    backpressure but the END always carries a precise duration.
    """
    pending_starts: dict[str, StageEvent] = {}
    runs: list[StageRun] = []
    for ev in events:
        if ev.event_type == StageEventType.START:
            # If there's already an unclosed start with the same
            # name, flush it as "open" so it's not lost.
            existing = pending_starts.pop(ev.name, None)
            if existing is not None:
                runs.append(
                    StageRun(
                        name=existing.name,
                        start_at=existing.at,
                        end_at=None,
                        duration_ms=None,
                        status="open",
                        error=None,
                        attrs=existing.attrs,
                    )
                )
            pending_starts[ev.name] = ev
            continue
        # END
        start = pending_starts.pop(ev.name, None)
        if start is not None:
            duration_ms: float | None
            if ev.elapsed_ms is not None:
                duration_ms = ev.elapsed_ms
            else:
                duration_ms = (ev.at - start.at).total_seconds() * 1000.0
            runs.append(
                StageRun(
                    name=ev.name,
                    start_at=start.at,
                    end_at=ev.at,
                    duration_ms=duration_ms,
                    status="error" if ev.error else "ok",
                    error=ev.error,
                    attrs=start.attrs,
                )
            )
        else:
            # Orphan END (live-bus drop). Synthesise a zero-width
            # start at the END's instant; duration comes from
            # `elapsed_ms` if available.
            runs.append(
                StageRun(
                    name=ev.name,
                    start_at=ev.at,
                    end_at=ev.at,
                    duration_ms=ev.elapsed_ms,
                    status="error" if ev.error else "ok",
                    error=ev.error,
                    attrs=ev.attrs,
                )
            )

    # Drain remaining unclosed starts as `open`.
    for start in pending_starts.values():
        runs.append(
            StageRun(
                name=start.name,
                start_at=start.at,
                end_at=None,
                duration_ms=None,
                status="open",
                error=None,
                attrs=start.attrs,
            )
        )
    return runs


def _bottlenecks(
    stages: Sequence[StageRun], total_ms: float, *, top_n: int = 5
) -> list[StageBottleneck]:
    """Top-N stages by duration. Pin: open stages count as 0
    duration here — not knowing what they took is itself a
    diagnosis the operator should see in `open_count`, not
    a wild guess we extrapolate."""
    timed = [s for s in stages if s.duration_ms is not None]
    timed.sort(key=lambda s: s.duration_ms or 0.0, reverse=True)
    out: list[StageBottleneck] = []
    for s in timed[:top_n]:
        pct = (s.duration_ms or 0.0) / total_ms if total_ms > 0 else 0.0
        out.append(
            StageBottleneck(
                name=s.name,
                duration_ms=s.duration_ms or 0.0,
                pct_of_total=pct,
            )
        )
    return out


def _format_ms(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    if ms >= 1000.0:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


def _render_markdown(timeline: CycleTimeline) -> str:
    """Build a paste-ready markdown summary."""
    lines = [
        f"# Cycle timeline — {timeline.cycle_started_at:%Y-%m-%d %H:%M:%S UTC}",
        "",
        f"**Total:** {_format_ms(timeline.total_duration_ms)} across {timeline.stage_count} stages",
    ]
    flag_parts = []
    if timeline.error_count:
        flag_parts.append(f"{timeline.error_count} errored")
    if timeline.open_count:
        flag_parts.append(f"{timeline.open_count} unfinished")
    if flag_parts:
        lines.append(f"**Health:** {' · '.join(flag_parts)}")

    if timeline.bottlenecks:
        lines.append("")
        lines.append("## Top stages by duration")
        lines.append("")
        lines.append("| Stage | Duration | % of cycle |")
        lines.append("| --- | --- | --- |")
        for b in timeline.bottlenecks:
            lines.append(f"| {b.name} | {_format_ms(b.duration_ms)} | {b.pct_of_total:.1%} |")

    if timeline.stages:
        lines.append("")
        lines.append("## Stage-by-stage")
        lines.append("")
        for s in timeline.stages:
            marker = {"ok": "✔", "error": "✘", "open": "·"}.get(s.status, "?")
            duration = _format_ms(s.duration_ms)
            line = f"- {marker} `{s.name}` — {duration}"
            if s.status == "error" and s.error:
                line += f" (error: `{s.error}`)"
            elif s.status == "open":
                line += " (no end event recorded)"
            lines.append(line)
    return "\n".join(lines)


def build_timeline(events: Iterable[StageEvent]) -> CycleTimeline:
    """Compose a structured timeline from a flat event stream.

    ``events`` must be in the order they were emitted. Empty
    input returns an empty timeline rooted at the unix epoch —
    the dashboard's "no data" state renders cleanly off the
    `stage_count == 0` check.
    """
    events_list = list(events)
    if not events_list:
        epoch = datetime.fromtimestamp(0, tz=None)
        return CycleTimeline(
            cycle_started_at=epoch,
            cycle_ended_at=None,
            total_duration_ms=0.0,
            stage_count=0,
            error_count=0,
            open_count=0,
            stages=[],
            bottlenecks=[],
            markdown="",
        )

    stages = _pair_events(events_list)
    started_at = events_list[0].at
    completed_ends = [s.end_at for s in stages if s.end_at is not None]
    if completed_ends:
        ended_at = max(completed_ends)
        total_ms = (ended_at - started_at).total_seconds() * 1000.0
    else:
        ended_at = None
        # Fallback: if every stage is open, the cycle's wall-clock
        # length is not yet defined — report 0.0 rather than
        # negative or NaN.
        total_ms = 0.0

    error_count = sum(1 for s in stages if s.status == "error")
    open_count = sum(1 for s in stages if s.status == "open")
    bottlenecks = _bottlenecks(stages, total_ms)
    timeline = CycleTimeline(
        cycle_started_at=started_at,
        cycle_ended_at=ended_at,
        total_duration_ms=total_ms,
        stage_count=len(stages),
        error_count=error_count,
        open_count=open_count,
        stages=stages,
        bottlenecks=bottlenecks,
    )
    md = _render_markdown(timeline)
    return CycleTimeline(
        cycle_started_at=timeline.cycle_started_at,
        cycle_ended_at=timeline.cycle_ended_at,
        total_duration_ms=timeline.total_duration_ms,
        stage_count=timeline.stage_count,
        error_count=timeline.error_count,
        open_count=timeline.open_count,
        stages=timeline.stages,
        bottlenecks=timeline.bottlenecks,
        markdown=md,
    )


__all__ = [
    "CycleTimeline",
    "StageBottleneck",
    "StageEvent",
    "StageEventType",
    "StageRun",
    "build_timeline",
]
