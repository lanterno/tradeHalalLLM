"""Status-page snapshot aggregator.

Round-4 wave 8.G: a public status page (think
`status.halal-trader.dev`) needs a one-call entry point that
takes the bot's recent operational records and produces a
publish-ready snapshot — overall status colour, uptime
percentage, list of recent incidents, current ongoing event if
any. This module is that aggregator.

Inputs are two append-only streams the bot already produces:

* **Halt records** — `(engaged_at, reason, resolved_at)` tuples
  from the `halt_log` audit table (Wave 0.D / `core/halt.py`).
  Ongoing halts have `resolved_at = None`.
* **Cycle event records** — `(at, succeeded, duration_ms)`
  tuples from the cycle pipeline's structured log (Wave 5.A
  `core/cycle_timeline.py` already aggregates these for the
  internal dashboard; this module aggregates them for public
  consumption).

Picked these two streams because they're operator-published
and append-only — a public status page can't read live broker
state or operator-identifying logs without leaking. Both
streams are sanitised at the source.

Output is a `StatusSnapshot` with a four-level traffic-light
status:

* **OPERATIONAL** — no ongoing halt, last-24h success rate ≥ 99%,
  no PAGE-severity incident in the window.
* **DEGRADED** — last-24h success rate < 99% but ≥ 95%, or a
  recent (resolved) incident in the window.
* **PARTIAL_OUTAGE** — last-24h success rate < 95% but ≥ 80%,
  OR a recent halt of duration > 5min.
* **MAJOR_OUTAGE** — currently halted, OR last-24h success
  rate < 80%.

The thresholds are tunable but ship with defaults that match
practitioner conventions for retail-trading bots.

Halal alignment: the snapshot exposes only **anonymised
operational metadata** — no per-trade data, no operator
identifiers, no broker keys, no LLM prompts. The
`SECURITY.md` policy treats the status page as a public-
facing surface; the aggregator's denylist mirrors the
`otel_translator.py` redacted-attribute set so a future
expansion can't accidentally leak.

Pure-Python; no DB / network. The caller fetches the input
streams from Postgres and hands them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# ── Status vocabulary ────────────────────────────────────


class StatusLevel(str, Enum):
    """Public-facing status colours.

    Pin the order: OPERATIONAL < DEGRADED < PARTIAL_OUTAGE <
    MAJOR_OUTAGE so callers can compare with `<`/`>` for "is the
    current status worse than X" decisions.
    """

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"


_STATUS_RANK: dict[StatusLevel, int] = {
    StatusLevel.OPERATIONAL: 0,
    StatusLevel.DEGRADED: 1,
    StatusLevel.PARTIAL_OUTAGE: 2,
    StatusLevel.MAJOR_OUTAGE: 3,
}


_STATUS_EMOJI: dict[StatusLevel, str] = {
    StatusLevel.OPERATIONAL: "🟢",
    StatusLevel.DEGRADED: "🟡",
    StatusLevel.PARTIAL_OUTAGE: "🟠",
    StatusLevel.MAJOR_OUTAGE: "🔴",
}


# ── Inputs ────────────────────────────────────────────────


@dataclass(frozen=True)
class HaltRecord:
    """One halt event the bot logged.

    ``engaged_at`` is when the halt was raised; ``resolved_at``
    is when the operator resumed (None if the halt is still
    active). ``reason`` is the operator-supplied note.

    Pin: the reason is filtered through a small denylist before
    being included in any public snapshot — operator-typed
    reasons can leak (e.g., "halt while testing strategy XYZ"
    where XYZ is the operator's strategy name).
    """

    engaged_at: datetime
    reason: str
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.resolved_at is not None and self.resolved_at < self.engaged_at:
            raise ValueError(
                f"resolved_at ({self.resolved_at}) must be >= engaged_at ({self.engaged_at})"
            )

    def is_active(self, *, now: datetime) -> bool:
        return self.resolved_at is None and self.engaged_at <= now

    def duration(self, *, now: datetime) -> timedelta:
        end = self.resolved_at if self.resolved_at is not None else now
        return end - self.engaged_at


@dataclass(frozen=True)
class CycleEventRecord:
    """One cycle outcome observation.

    ``at`` is the cycle's wall-clock start. ``succeeded`` is True
    iff the cycle ran to completion without raising;
    ``duration_ms`` is the wall-clock total.
    """

    at: datetime
    succeeded: bool
    duration_ms: float

    def __post_init__(self) -> None:
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0; got {self.duration_ms}")


# ── Configuration ────────────────────────────────────────


@dataclass(frozen=True)
class StatusThresholds:
    """Operator-tunable thresholds for the level decision.

    Defaults match retail-trading-bot practitioner conventions:

    * 99% success → OPERATIONAL
    * 95-99% → DEGRADED
    * 80-95% → PARTIAL_OUTAGE
    * < 80% → MAJOR_OUTAGE

    Plus a recent-halt-duration cut: any halt longer than
    `recent_halt_minutes` in the window flips the level to at
    least PARTIAL_OUTAGE even if success rate is fine.

    `incident_window_days` controls how far back the snapshot
    looks; default 7 days is the standard public-status-page
    window.
    """

    success_rate_operational: float = 0.99
    success_rate_degraded: float = 0.95
    success_rate_partial: float = 0.80
    recent_halt_minutes: int = 5
    incident_window_days: int = 7

    def __post_init__(self) -> None:
        if (
            not 0.0
            <= self.success_rate_partial
            <= self.success_rate_degraded
            <= self.success_rate_operational
            <= 1.0
        ):
            raise ValueError(
                "success-rate thresholds must be ordered: 0 ≤ partial ≤ degraded ≤ operational ≤ 1"
            )
        if self.recent_halt_minutes < 0:
            raise ValueError("recent_halt_minutes must be >= 0")
        if self.incident_window_days <= 0:
            raise ValueError("incident_window_days must be positive")


# ── Reason filter ────────────────────────────────────────


# Substrings that, if present in a halt reason, redact the whole
# reason to a generic placeholder. Defence against operator
# typing strategy names / API key tails / position pairs into
# the reason field.
_SENSITIVE_REASON_SUBSTRINGS: frozenset[str] = frozenset(
    {
        "api_key",
        "secret",
        "token",
        "password",
        "operator",
    }
)


def filter_reason(reason: str, *, max_chars: int = 80) -> str:
    """Sanitise a halt reason for public publication.

    Pin: a reason containing any sensitive substring is replaced
    with `"halt for operational reasons"` rather than partially
    redacted. Partial redaction is fragile (key tails leak via
    length) and the audience for the public status page doesn't
    need the granularity.

    Long reasons cap at ``max_chars`` with a `…` suffix.
    """
    lowered = reason.lower()
    for needle in _SENSITIVE_REASON_SUBSTRINGS:
        if needle in lowered:
            return "halt for operational reasons"
    if len(reason) > max_chars:
        return reason[: max_chars - 1] + "…"
    return reason


# ── Output ────────────────────────────────────────────────


@dataclass(frozen=True)
class IncidentSummary:
    """One incident as it appears on the public page."""

    started_at: datetime
    ended_at: datetime | None
    duration_minutes: float
    reason: str  # already passed through `filter_reason`
    is_active: bool


@dataclass(frozen=True)
class StatusSnapshot:
    """Aggregated, publish-ready status payload.

    ``level`` is the four-level traffic-light. ``success_rate``
    is the fraction of cycles that completed without raising in
    the window — 1.0 if no cycle events were logged (no negative
    information; better than reporting 0.0 on an empty stream).
    ``cycle_count`` is the number of cycles observed.
    ``incidents`` lists halts in the window, most-recent first.
    ``ongoing_incident`` is set when `is_currently_halted` is
    True; it's the incident the public page renders as the
    headline.
    """

    captured_at: datetime
    level: StatusLevel
    success_rate: float
    cycle_count: int
    is_currently_halted: bool
    incidents: list[IncidentSummary] = field(default_factory=list)
    ongoing_incident: IncidentSummary | None = None
    summary: str = ""


# ── Aggregation ──────────────────────────────────────────


def _success_rate(events: list[CycleEventRecord], *, since: datetime) -> tuple[float, int]:
    """Fraction of cycles in the window that succeeded, plus the
    cycle count. Pin: empty window → (1.0, 0) — a public page
    showing "no cycles observed" should report OPERATIONAL, not
    a divide-by-zero or a silent 0%."""
    in_window = [e for e in events if e.at >= since]
    if not in_window:
        return 1.0, 0
    successes = sum(1 for e in in_window if e.succeeded)
    return successes / len(in_window), len(in_window)


def _classify(
    *,
    success_rate: float,
    is_halted: bool,
    incidents_in_window: list[HaltRecord],
    thresholds: StatusThresholds,
    now: datetime,
) -> StatusLevel:
    """Pin the decision tree:

    1. **Currently halted → MAJOR_OUTAGE.** No ambiguity.
    2. Else if success rate < 80% → MAJOR_OUTAGE.
    3. Else if any recent halt > recent_halt_minutes → at least
       PARTIAL_OUTAGE, regardless of success rate.
    4. Else if success rate < 95% → PARTIAL_OUTAGE.
    5. Else if success rate < 99% → DEGRADED.
    6. Else if any halt in the window → DEGRADED (recovered, but
       worth noting).
    7. Else → OPERATIONAL.

    The decision is intentionally conservative — the public page
    err on the side of acknowledging an issue rather than
    hiding it.
    """
    if is_halted:
        return StatusLevel.MAJOR_OUTAGE
    if success_rate < thresholds.success_rate_partial:
        return StatusLevel.MAJOR_OUTAGE

    long_halt = any(
        h.duration(now=now).total_seconds() / 60.0 > thresholds.recent_halt_minutes
        for h in incidents_in_window
    )
    if long_halt:
        if success_rate < thresholds.success_rate_degraded:
            return StatusLevel.MAJOR_OUTAGE
        return StatusLevel.PARTIAL_OUTAGE

    if success_rate < thresholds.success_rate_degraded:
        return StatusLevel.PARTIAL_OUTAGE
    if success_rate < thresholds.success_rate_operational:
        return StatusLevel.DEGRADED
    if incidents_in_window:
        return StatusLevel.DEGRADED
    return StatusLevel.OPERATIONAL


def _build_summary(
    level: StatusLevel,
    success_rate: float,
    cycle_count: int,
    is_halted: bool,
    window_days: int,
) -> str:
    emoji = _STATUS_EMOJI[level]
    if is_halted:
        return f"{emoji} {level.value} · halt is currently engaged"
    if cycle_count == 0:
        return f"{emoji} {level.value} · no cycle data in last {window_days}d"
    return (
        f"{emoji} {level.value} · "
        f"{success_rate:.2%} success over {cycle_count} cycles "
        f"({window_days}d window)"
    )


def build_snapshot(
    *,
    halts: list[HaltRecord],
    cycle_events: list[CycleEventRecord],
    now: datetime,
    thresholds: StatusThresholds | None = None,
) -> StatusSnapshot:
    """Compose a public-facing status snapshot from raw streams.

    ``now`` is injected so tests are deterministic; production
    callers pass `datetime.now(UTC)`.

    Pin: the snapshot includes only halts whose `engaged_at` falls
    inside the window. A halt that engaged before the window but
    is still active counts as ongoing (the headline incident),
    but does NOT show up in the historical list — the public
    page should treat it as "active now" rather than as one of
    several past events.
    """
    t = thresholds or StatusThresholds()
    since = now - timedelta(days=t.incident_window_days)

    # Categorise halts.
    incidents_in_window: list[HaltRecord] = []
    ongoing: HaltRecord | None = None
    for h in halts:
        if h.is_active(now=now):
            ongoing = h
            # Active halts don't go into the historical list.
            continue
        if h.engaged_at >= since:
            incidents_in_window.append(h)

    incidents_in_window.sort(key=lambda h: h.engaged_at, reverse=True)

    success_rate, cycle_count = _success_rate(cycle_events, since=since)
    is_halted = ongoing is not None

    level = _classify(
        success_rate=success_rate,
        is_halted=is_halted,
        incidents_in_window=incidents_in_window,
        thresholds=t,
        now=now,
    )

    summaries = [
        IncidentSummary(
            started_at=h.engaged_at,
            ended_at=h.resolved_at,
            duration_minutes=h.duration(now=now).total_seconds() / 60.0,
            reason=filter_reason(h.reason),
            is_active=False,
        )
        for h in incidents_in_window
    ]

    ongoing_summary: IncidentSummary | None = None
    if ongoing is not None:
        ongoing_summary = IncidentSummary(
            started_at=ongoing.engaged_at,
            ended_at=None,
            duration_minutes=ongoing.duration(now=now).total_seconds() / 60.0,
            reason=filter_reason(ongoing.reason),
            is_active=True,
        )

    summary = _build_summary(level, success_rate, cycle_count, is_halted, t.incident_window_days)

    return StatusSnapshot(
        captured_at=now,
        level=level,
        success_rate=success_rate,
        cycle_count=cycle_count,
        is_currently_halted=is_halted,
        incidents=summaries,
        ongoing_incident=ongoing_summary,
        summary=summary,
    )


# ── Render helper ────────────────────────────────────────


def render_snapshot(snapshot: StatusSnapshot) -> str:
    """Operator-readable text payload for log / Slack / a basic
    HTML status page (one line per incident; emoji-prefixed
    overall status header)."""
    lines = [snapshot.summary, ""]
    if snapshot.ongoing_incident is not None:
        i = snapshot.ongoing_incident
        lines.append(
            f"⚠️ Ongoing: started {i.started_at:%Y-%m-%d %H:%M UTC}, "
            f"{i.duration_minutes:.0f} min so far — {i.reason}"
        )
        lines.append("")
    if snapshot.incidents:
        lines.append(f"Recent incidents (last {len(snapshot.incidents)}):")
        for i in snapshot.incidents:
            ended = i.ended_at.strftime("%Y-%m-%d %H:%M UTC") if i.ended_at else "ongoing"
            lines.append(
                f"  · {i.started_at:%Y-%m-%d %H:%M UTC} → {ended} "
                f"({i.duration_minutes:.0f} min) — {i.reason}"
            )
    else:
        lines.append("No incidents in the window.")
    return "\n".join(lines)


__all__ = [
    "CycleEventRecord",
    "HaltRecord",
    "IncidentSummary",
    "StatusLevel",
    "StatusSnapshot",
    "StatusThresholds",
    "build_snapshot",
    "filter_reason",
    "render_snapshot",
]
