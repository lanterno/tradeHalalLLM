"""Incident response state machine.

Auxiliary primitive for Wave 8.E ("On-call runbook + alerting").
Wave 8.E ships the alert router + seed runbooks; this module is
the **pure-Python incident lifecycle engine** that tracks an
alert from open through post-mortem, enforcing severity-driven
ack timelines and post-mortem requirements.

Picked a focused state engine over a generic ticketing tool
because (a) severity-driven ack SLAs (sev1 needs ack within 5
minutes; sev4 within 24 hours) need deterministic enforcement —
the dashboard tile "are any sev1 unacked past SLA?" should
answer in O(N) of incidents, not require a ticket-system query;
(b) the lifecycle has a strict order (OPEN → ACK → MITIGATED
→ RESOLVED → POSTMORTEM_PUBLISHED) and skipping is the most
common operator error during an outage; (c) post-mortem is
required for sev1/sev2 incidents — a closed sev1 without a
post-mortem published within 7 days is a process failure that
must be visible on the dashboard.

Pinned semantics:
- **Severity ladder: SEV1 (production down) → SEV4 (informational).**
  SEV1 ack within 5min; SEV2 within 30min; SEV3 within 4h;
  SEV4 within 24h. Operator-tunable via policy.
- **Lifecycle is forward-only.** OPEN → ACK → MITIGATED →
  RESOLVED → POSTMORTEM_PUBLISHED. Skipping ahead raises;
  reverting raises.
- **Post-mortem required for sev1 / sev2.** Within 7 days
  of RESOLVED. SEV3 / SEV4 don't require post-mortems.
- **Acker / resolver / author identity required.** Every state
  transition records who decided it; an ack without a name is
  rejected so the audit trail can't be anonymous.
- **Render output never includes raw error messages from the
  alert source — only the incident summary + status emoji.**
  Mirrors no-secret patterns of upstream waves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class Severity(str, Enum):
    """Severity ladder (highest priority first by ack SLA).

    Pinned string values for JSON / DB stability. Adding a severity
    is a code review change.
    """

    SEV1 = "sev1"  # Production down or data loss
    SEV2 = "sev2"  # Major degradation
    SEV3 = "sev3"  # Minor degradation
    SEV4 = "sev4"  # Informational / cleanup


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.SEV4: 0,
    Severity.SEV3: 1,
    Severity.SEV2: 2,
    Severity.SEV1: 3,
}


_DEFAULT_ACK_SLAS: dict[Severity, timedelta] = {
    Severity.SEV1: timedelta(minutes=5),
    Severity.SEV2: timedelta(minutes=30),
    Severity.SEV3: timedelta(hours=4),
    Severity.SEV4: timedelta(hours=24),
}


_POSTMORTEM_REQUIRED: frozenset[Severity] = frozenset({Severity.SEV1, Severity.SEV2})


_DEFAULT_POSTMORTEM_DEADLINE = timedelta(days=7)


class IncidentStatus(str, Enum):
    """Forward-only lifecycle.

    Pinned values. POSTMORTEM_PUBLISHED is terminal.
    """

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    POSTMORTEM_PUBLISHED = "postmortem_published"


_STATUS_ORDER: tuple[IncidentStatus, ...] = (
    IncidentStatus.OPEN,
    IncidentStatus.ACKNOWLEDGED,
    IncidentStatus.MITIGATED,
    IncidentStatus.RESOLVED,
    IncidentStatus.POSTMORTEM_PUBLISHED,
)


@dataclass(frozen=True)
class IncidentPolicy:
    """Operator-tunable policy."""

    ack_slas: dict[Severity, timedelta] = None  # type: ignore[assignment]
    postmortem_deadline: timedelta = _DEFAULT_POSTMORTEM_DEADLINE

    def __post_init__(self) -> None:
        # Allow None default → use _DEFAULT_ACK_SLAS frozen copy
        if self.ack_slas is None:
            object.__setattr__(self, "ack_slas", dict(_DEFAULT_ACK_SLAS))
        for sev, sla in self.ack_slas.items():  # type: ignore[union-attr]
            if not isinstance(sev, Severity):
                raise ValueError(f"ack_slas key {sev!r} must be Severity")
            if sla <= timedelta(0):
                raise ValueError(f"ack SLA for {sev.value} must be positive")
        # All severities covered
        for sev in Severity:
            if sev not in self.ack_slas:  # type: ignore[union-attr]
                raise ValueError(f"ack_slas missing entry for {sev.value}")
        if self.postmortem_deadline <= timedelta(0):
            raise ValueError("postmortem_deadline must be positive")


DEFAULT_POLICY = IncidentPolicy()


class StatusTransitionError(Exception):
    """Raised when a status transition violates lifecycle ordering."""

    def __init__(self, current: IncidentStatus, attempted: IncidentStatus) -> None:
        super().__init__(f"cannot transition from {current.value} to {attempted.value}")
        self.current = current
        self.attempted = attempted


class PostmortemNotRequiredError(Exception):
    """Raised when publish_postmortem is called on sev3/sev4 incident."""

    def __init__(self, severity: Severity) -> None:
        super().__init__(f"severity {severity.value!r} doesn't require a post-mortem")
        self.severity = severity


@dataclass(frozen=True)
class Incident:
    """One incident record.

    Operations (`acknowledge`, `mitigate`, `resolve`, `publish_postmortem`)
    return a new state. The dataclass is frozen; the audit trail
    is the immutable history of transitions.
    """

    incident_id: str
    severity: Severity
    summary: str
    opened_at: datetime
    status: IncidentStatus
    last_status_at: datetime
    acker: str = ""
    resolver: str = ""
    postmortem_author: str = ""

    def __post_init__(self) -> None:
        if not self.incident_id or not self.incident_id.strip():
            raise ValueError("incident_id must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.opened_at.tzinfo is None:
            raise ValueError("opened_at must be timezone-aware")
        if self.last_status_at.tzinfo is None:
            raise ValueError("last_status_at must be timezone-aware")
        if self.last_status_at < self.opened_at:
            raise ValueError("last_status_at must be >= opened_at")
        # Per-status attribution requirements
        if self.status is IncidentStatus.OPEN:
            if self.acker or self.resolver or self.postmortem_author:
                raise ValueError("OPEN status must not have acker / resolver / author")
        elif self.status is IncidentStatus.ACKNOWLEDGED:
            if not self.acker.strip():
                raise ValueError("ACKNOWLEDGED requires non-empty acker")
            if self.resolver or self.postmortem_author:
                raise ValueError("ACKNOWLEDGED must not have resolver / postmortem_author")
        elif self.status is IncidentStatus.MITIGATED:
            if not self.acker.strip():
                raise ValueError("MITIGATED requires acker (carried forward)")
        elif self.status is IncidentStatus.RESOLVED:
            if not self.acker.strip() or not self.resolver.strip():
                raise ValueError("RESOLVED requires acker + non-empty resolver")
        elif self.status is IncidentStatus.POSTMORTEM_PUBLISHED:
            if (
                not self.acker.strip()
                or not self.resolver.strip()
                or not self.postmortem_author.strip()
            ):
                raise ValueError("POSTMORTEM_PUBLISHED requires acker + resolver + author")


def open_incident(
    *,
    incident_id: str,
    severity: Severity,
    summary: str,
    now: datetime,
) -> Incident:
    """Open a new incident in OPEN status."""

    if not incident_id or not incident_id.strip():
        raise ValueError("incident_id must be non-empty")
    if not summary or not summary.strip():
        raise ValueError("summary must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return Incident(
        incident_id=incident_id,
        severity=severity,
        summary=summary,
        opened_at=now,
        status=IncidentStatus.OPEN,
        last_status_at=now,
    )


def _check_forward(current: IncidentStatus, target: IncidentStatus) -> None:
    cur_idx = _STATUS_ORDER.index(current)
    target_idx = _STATUS_ORDER.index(target)
    if target_idx != cur_idx + 1:
        raise StatusTransitionError(current, target)


def acknowledge(incident: Incident, *, acker: str, now: datetime) -> Incident:
    """Move OPEN → ACKNOWLEDGED. Records the acker name."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not acker or not acker.strip():
        raise ValueError("acker must be non-empty")
    _check_forward(incident.status, IncidentStatus.ACKNOWLEDGED)
    return Incident(
        incident_id=incident.incident_id,
        severity=incident.severity,
        summary=incident.summary,
        opened_at=incident.opened_at,
        status=IncidentStatus.ACKNOWLEDGED,
        last_status_at=now,
        acker=acker,
    )


def mitigate(incident: Incident, *, now: datetime) -> Incident:
    """Move ACKNOWLEDGED → MITIGATED."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    _check_forward(incident.status, IncidentStatus.MITIGATED)
    return Incident(
        incident_id=incident.incident_id,
        severity=incident.severity,
        summary=incident.summary,
        opened_at=incident.opened_at,
        status=IncidentStatus.MITIGATED,
        last_status_at=now,
        acker=incident.acker,
    )


def resolve(incident: Incident, *, resolver: str, now: datetime) -> Incident:
    """Move MITIGATED → RESOLVED. Records the resolver name."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not resolver or not resolver.strip():
        raise ValueError("resolver must be non-empty")
    _check_forward(incident.status, IncidentStatus.RESOLVED)
    return Incident(
        incident_id=incident.incident_id,
        severity=incident.severity,
        summary=incident.summary,
        opened_at=incident.opened_at,
        status=IncidentStatus.RESOLVED,
        last_status_at=now,
        acker=incident.acker,
        resolver=resolver,
    )


def publish_postmortem(incident: Incident, *, author: str, now: datetime) -> Incident:
    """Move RESOLVED → POSTMORTEM_PUBLISHED.

    Pinned: only sev1 / sev2 require post-mortems; calling this on
    sev3 / sev4 raises PostmortemNotRequiredError (operators can
    skip the postmortem step for low-severity incidents).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not author or not author.strip():
        raise ValueError("author must be non-empty")
    if incident.severity not in _POSTMORTEM_REQUIRED:
        raise PostmortemNotRequiredError(incident.severity)
    _check_forward(incident.status, IncidentStatus.POSTMORTEM_PUBLISHED)
    return Incident(
        incident_id=incident.incident_id,
        severity=incident.severity,
        summary=incident.summary,
        opened_at=incident.opened_at,
        status=IncidentStatus.POSTMORTEM_PUBLISHED,
        last_status_at=now,
        acker=incident.acker,
        resolver=incident.resolver,
        postmortem_author=author,
    )


def is_ack_overdue(
    incident: Incident,
    *,
    now: datetime,
    policy: IncidentPolicy = DEFAULT_POLICY,
) -> bool:
    """True if incident is OPEN past its severity-specific ack SLA.

    Pinned: only OPEN status can be overdue; ACKNOWLEDGED+ have
    been acked already.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if incident.status is not IncidentStatus.OPEN:
        return False
    elapsed = now - incident.opened_at
    sla = policy.ack_slas[incident.severity]  # type: ignore[index]
    return elapsed > sla


def is_postmortem_overdue(
    incident: Incident,
    *,
    now: datetime,
    policy: IncidentPolicy = DEFAULT_POLICY,
) -> bool:
    """True if a sev1/sev2 incident is RESOLVED past the postmortem deadline.

    Sev3 / sev4 don't require post-mortems, so they're never overdue.
    Already-published incidents are never overdue.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if incident.severity not in _POSTMORTEM_REQUIRED:
        return False
    if incident.status is not IncidentStatus.RESOLVED:
        return False
    elapsed = now - incident.last_status_at
    return elapsed > policy.postmortem_deadline


def severity_outranks(a: Severity, b: Severity) -> bool:
    """True if `a` is strictly more severe than `b` (SEV1 > SEV4)."""

    return _SEVERITY_ORDER[a] > _SEVERITY_ORDER[b]


def filter_overdue(
    incidents: Iterable[Incident],
    *,
    now: datetime,
    policy: IncidentPolicy = DEFAULT_POLICY,
) -> tuple[Incident, ...]:
    """Return overdue incidents sorted by severity (highest first).

    Includes both ack-overdue (OPEN past SLA) and postmortem-overdue
    (RESOLVED sev1/sev2 past deadline) incidents.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    overdue = [
        inc
        for inc in incidents
        if is_ack_overdue(inc, now=now, policy=policy)
        or is_postmortem_overdue(inc, now=now, policy=policy)
    ]
    return tuple(sorted(overdue, key=lambda i: -_SEVERITY_ORDER[i.severity]))


_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.SEV1: "🔴",
    Severity.SEV2: "🟠",
    Severity.SEV3: "🟡",
    Severity.SEV4: "🔵",
}


_STATUS_EMOJI: dict[IncidentStatus, str] = {
    IncidentStatus.OPEN: "⚠️",
    IncidentStatus.ACKNOWLEDGED: "👀",
    IncidentStatus.MITIGATED: "🛡️",
    IncidentStatus.RESOLVED: "✅",
    IncidentStatus.POSTMORTEM_PUBLISHED: "📋",
}


def render_incident(incident: Incident) -> str:
    """Format an incident for ops display.

    No-secret-leak: render shows summary + status + severity emoji
    + attribution. Never includes raw alert payload / stack traces /
    log lines (those go in the operator's runbook drill, not the
    incident card).
    """

    sev_emoji = _SEVERITY_EMOJI[incident.severity]
    status_emoji = _STATUS_EMOJI[incident.status]
    lines = [
        f"{sev_emoji}{status_emoji} {incident.incident_id} "
        f"({incident.severity.value}) — {incident.status.value}",
        f"  summary: {incident.summary}",
        f"  opened: {incident.opened_at.isoformat()}",
    ]
    if incident.acker:
        lines.append(f"  acker: {incident.acker}")
    if incident.resolver:
        lines.append(f"  resolver: {incident.resolver}")
    if incident.postmortem_author:
        lines.append(f"  postmortem author: {incident.postmortem_author}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "Incident",
    "IncidentPolicy",
    "IncidentStatus",
    "PostmortemNotRequiredError",
    "Severity",
    "StatusTransitionError",
    "acknowledge",
    "filter_overdue",
    "is_ack_overdue",
    "is_postmortem_overdue",
    "mitigate",
    "open_incident",
    "publish_postmortem",
    "render_incident",
    "resolve",
    "severity_outranks",
]
