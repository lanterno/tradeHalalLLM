"""Tests for `halal_trader.ops.incident_response` (Wave 8.E aux).

Covers: severity ladder, lifecycle state machine, ack SLA enforcement,
postmortem-required-for-sev1/sev2 pin, no-secret render.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.ops.incident_response import (
    DEFAULT_POLICY,
    Incident,
    IncidentPolicy,
    IncidentStatus,
    PostmortemNotRequiredError,
    Severity,
    StatusTransitionError,
    acknowledge,
    filter_overdue,
    is_ack_overdue,
    is_postmortem_overdue,
    mitigate,
    open_incident,
    publish_postmortem,
    render_incident,
    resolve,
    severity_outranks,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_severity_string_values_pinned() -> None:
    assert Severity.SEV1.value == "sev1"
    assert Severity.SEV2.value == "sev2"
    assert Severity.SEV3.value == "sev3"
    assert Severity.SEV4.value == "sev4"


def test_incident_status_string_values_pinned() -> None:
    assert IncidentStatus.OPEN.value == "open"
    assert IncidentStatus.ACKNOWLEDGED.value == "acknowledged"
    assert IncidentStatus.MITIGATED.value == "mitigated"
    assert IncidentStatus.RESOLVED.value == "resolved"
    assert IncidentStatus.POSTMORTEM_PUBLISHED.value == "postmortem_published"


# --------------------------- IncidentPolicy ----------------------------------


def test_default_policy_ack_slas() -> None:
    """Pin: roadmap-pinned ack SLAs."""

    assert DEFAULT_POLICY.ack_slas[Severity.SEV1] == timedelta(minutes=5)
    assert DEFAULT_POLICY.ack_slas[Severity.SEV2] == timedelta(minutes=30)
    assert DEFAULT_POLICY.ack_slas[Severity.SEV3] == timedelta(hours=4)
    assert DEFAULT_POLICY.ack_slas[Severity.SEV4] == timedelta(hours=24)


def test_default_policy_postmortem_deadline() -> None:
    """Pin: 7-day postmortem deadline."""

    assert DEFAULT_POLICY.postmortem_deadline == timedelta(days=7)


def test_policy_rejects_zero_ack_sla() -> None:
    custom = {
        Severity.SEV1: timedelta(0),
        Severity.SEV2: timedelta(minutes=30),
        Severity.SEV3: timedelta(hours=4),
        Severity.SEV4: timedelta(hours=24),
    }
    with pytest.raises(ValueError, match="ack SLA"):
        IncidentPolicy(ack_slas=custom)


def test_policy_rejects_missing_severity() -> None:
    """Pin: every severity must have an ack SLA."""

    custom = {
        Severity.SEV1: timedelta(minutes=5),
        # missing SEV2/SEV3/SEV4
    }
    with pytest.raises(ValueError, match="missing"):
        IncidentPolicy(ack_slas=custom)


def test_policy_rejects_zero_postmortem_deadline() -> None:
    with pytest.raises(ValueError, match="postmortem_deadline"):
        IncidentPolicy(postmortem_deadline=timedelta(0))


# --------------------------- severity_outranks -------------------------------


def test_sev1_outranks_sev4() -> None:
    """Pin: SEV1 is strictly more severe than SEV4."""

    assert severity_outranks(Severity.SEV1, Severity.SEV4) is True


def test_sev1_outranks_sev2() -> None:
    assert severity_outranks(Severity.SEV1, Severity.SEV2) is True


def test_sev3_does_not_outrank_sev2() -> None:
    assert severity_outranks(Severity.SEV3, Severity.SEV2) is False


def test_sev1_does_not_outrank_self() -> None:
    """Pin: severity_outranks is strict (>, not >=)."""

    assert severity_outranks(Severity.SEV1, Severity.SEV1) is False


# --------------------------- Incident validation -----------------------------


def test_incident_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="incident_id"):
        Incident(
            incident_id="",
            severity=Severity.SEV1,
            summary="prod down",
            opened_at=T0,
            status=IncidentStatus.OPEN,
            last_status_at=T0,
        )


def test_incident_rejects_empty_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        Incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="",
            opened_at=T0,
            status=IncidentStatus.OPEN,
            last_status_at=T0,
        )


def test_incident_rejects_naive_opened_at() -> None:
    with pytest.raises(ValueError, match="opened_at"):
        Incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            opened_at=datetime(2026, 5, 1),
            status=IncidentStatus.OPEN,
            last_status_at=T0,
        )


def test_incident_rejects_last_status_before_opened() -> None:
    with pytest.raises(ValueError, match="last_status_at"):
        Incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            opened_at=T0,
            status=IncidentStatus.OPEN,
            last_status_at=T0 - timedelta(seconds=1),
        )


def test_open_status_rejects_acker() -> None:
    """Pin: OPEN status must not have acker / resolver / author set."""

    with pytest.raises(ValueError, match="OPEN"):
        Incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            opened_at=T0,
            status=IncidentStatus.OPEN,
            last_status_at=T0,
            acker="alice",
        )


def test_acknowledged_requires_acker() -> None:
    with pytest.raises(ValueError, match="ACKNOWLEDGED"):
        Incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            opened_at=T0,
            status=IncidentStatus.ACKNOWLEDGED,
            last_status_at=T0,
        )


def test_resolved_requires_acker_and_resolver() -> None:
    with pytest.raises(ValueError, match="RESOLVED"):
        Incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            opened_at=T0,
            status=IncidentStatus.RESOLVED,
            last_status_at=T0,
            acker="alice",  # resolver missing
        )


def test_postmortem_status_requires_all_three() -> None:
    """Pin: POSTMORTEM_PUBLISHED requires acker + resolver + author."""

    with pytest.raises(ValueError, match="POSTMORTEM_PUBLISHED"):
        Incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            opened_at=T0,
            status=IncidentStatus.POSTMORTEM_PUBLISHED,
            last_status_at=T0,
            acker="alice",
            resolver="bob",
            # postmortem_author missing
        )


def test_incident_is_frozen() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="prod down",
        now=T0,
    )
    with pytest.raises(FrozenInstanceError):
        inc.summary = "other"  # type: ignore[misc]


# --------------------------- open_incident -----------------------------------


def test_open_incident_basic() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="prod down",
        now=T0,
    )
    assert inc.status is IncidentStatus.OPEN
    assert inc.acker == ""


def test_open_incident_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="incident_id"):
        open_incident(
            incident_id="",
            severity=Severity.SEV1,
            summary="x",
            now=T0,
        )


def test_open_incident_rejects_empty_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        open_incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="",
            now=T0,
        )


def test_open_incident_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        open_incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            now=datetime(2026, 5, 1),
        )


# --------------------------- acknowledge -------------------------------------


def test_acknowledge_open_to_acked() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="prod down",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0 + timedelta(minutes=2))
    assert inc.status is IncidentStatus.ACKNOWLEDGED
    assert inc.acker == "alice"


def test_acknowledge_requires_acker() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    with pytest.raises(ValueError, match="acker"):
        acknowledge(inc, acker="", now=T0)


def test_acknowledge_skip_rejected() -> None:
    """Pin: cannot ack an already-acked incident."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    with pytest.raises(StatusTransitionError):
        acknowledge(inc, acker="bob", now=T0)


# --------------------------- mitigate / resolve ------------------------------


def test_mitigate_from_acked() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0 + timedelta(minutes=15))
    assert inc.status is IncidentStatus.MITIGATED
    assert inc.acker == "alice"  # carried forward


def test_mitigate_skip_from_open_rejected() -> None:
    """Pin: cannot mitigate without acking first."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    with pytest.raises(StatusTransitionError):
        mitigate(inc, now=T0)


def test_resolve_from_mitigated() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0)
    inc = resolve(inc, resolver="bob", now=T0 + timedelta(hours=1))
    assert inc.status is IncidentStatus.RESOLVED
    assert inc.resolver == "bob"


def test_resolve_requires_resolver() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0)
    with pytest.raises(ValueError, match="resolver"):
        resolve(inc, resolver="", now=T0)


def test_resolve_skip_from_acked_rejected() -> None:
    """Pin: cannot skip MITIGATED → RESOLVED."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    with pytest.raises(StatusTransitionError):
        resolve(inc, resolver="bob", now=T0)


# --------------------------- publish_postmortem ------------------------------


def test_publish_postmortem_sev1() -> None:
    """Pin: SEV1 supports postmortem publication."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0)
    inc = resolve(inc, resolver="bob", now=T0)
    inc = publish_postmortem(inc, author="charlie", now=T0 + timedelta(days=3))
    assert inc.status is IncidentStatus.POSTMORTEM_PUBLISHED
    assert inc.postmortem_author == "charlie"


def test_publish_postmortem_sev2_supported() -> None:
    """Pin: SEV2 also requires postmortem."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV2,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0)
    inc = resolve(inc, resolver="bob", now=T0)
    inc = publish_postmortem(inc, author="charlie", now=T0)
    assert inc.status is IncidentStatus.POSTMORTEM_PUBLISHED


def test_publish_postmortem_sev3_rejected() -> None:
    """Pin: SEV3 doesn't require/support postmortem.

    Calling publish_postmortem on a sev3 raises so operators can't
    accidentally bloat the postmortem queue with low-severity items.
    """

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV3,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0)
    inc = resolve(inc, resolver="bob", now=T0)
    with pytest.raises(PostmortemNotRequiredError) as exc_info:
        publish_postmortem(inc, author="charlie", now=T0)
    assert exc_info.value.severity is Severity.SEV3


def test_publish_postmortem_sev4_rejected() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV4,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0)
    inc = resolve(inc, resolver="bob", now=T0)
    with pytest.raises(PostmortemNotRequiredError):
        publish_postmortem(inc, author="charlie", now=T0)


def test_publish_postmortem_requires_author() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    inc = mitigate(inc, now=T0)
    inc = resolve(inc, resolver="bob", now=T0)
    with pytest.raises(ValueError, match="author"):
        publish_postmortem(inc, author="", now=T0)


def test_publish_postmortem_skip_from_open_rejected() -> None:
    """Pin: postmortem requires RESOLVED state."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    with pytest.raises(StatusTransitionError):
        publish_postmortem(inc, author="charlie", now=T0)


# --------------------------- is_ack_overdue ----------------------------------


def test_ack_overdue_sev1_at_5min_boundary_not_overdue() -> None:
    """Pin: 5min exactly is NOT overdue (>, not >=)."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    assert is_ack_overdue(inc, now=T0 + timedelta(minutes=5)) is False


def test_ack_overdue_sev1_past_5min() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    assert is_ack_overdue(inc, now=T0 + timedelta(minutes=6)) is True


def test_ack_overdue_sev2_at_30min_pin() -> None:
    """Pin: SEV2 SLA is 30min."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV2,
        summary="x",
        now=T0,
    )
    assert is_ack_overdue(inc, now=T0 + timedelta(minutes=29)) is False
    assert is_ack_overdue(inc, now=T0 + timedelta(minutes=31)) is True


def test_ack_overdue_sev3_at_4hr_pin() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV3,
        summary="x",
        now=T0,
    )
    assert is_ack_overdue(inc, now=T0 + timedelta(hours=3, minutes=59)) is False
    assert is_ack_overdue(inc, now=T0 + timedelta(hours=5)) is True


def test_ack_overdue_sev4_at_24hr_pin() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV4,
        summary="x",
        now=T0,
    )
    assert is_ack_overdue(inc, now=T0 + timedelta(hours=23)) is False
    assert is_ack_overdue(inc, now=T0 + timedelta(hours=25)) is True


def test_ack_overdue_acked_never_overdue() -> None:
    """Pin: only OPEN status can be ack-overdue."""

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    # Even way past the SLA, an acked incident is not "ack-overdue"
    assert is_ack_overdue(inc, now=T0 + timedelta(hours=10)) is False


def test_ack_overdue_rejects_naive_now() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    with pytest.raises(ValueError, match="now"):
        is_ack_overdue(inc, now=datetime(2026, 5, 1))


# --------------------------- is_postmortem_overdue ---------------------------


def _resolved_sev(severity: Severity, *, resolved_at: datetime) -> Incident:
    inc = open_incident(
        incident_id="i1",
        severity=severity,
        summary="x",
        now=resolved_at - timedelta(hours=1),
    )
    inc = acknowledge(inc, acker="alice", now=resolved_at - timedelta(hours=1))
    inc = mitigate(inc, now=resolved_at - timedelta(minutes=30))
    inc = resolve(inc, resolver="bob", now=resolved_at)
    return inc


def test_postmortem_overdue_sev1_past_7days() -> None:
    inc = _resolved_sev(Severity.SEV1, resolved_at=T0)
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=8)) is True


def test_postmortem_overdue_sev1_at_7days_boundary_not_overdue() -> None:
    """Pin: 7 days exactly is NOT overdue."""

    inc = _resolved_sev(Severity.SEV1, resolved_at=T0)
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=7)) is False


def test_postmortem_overdue_sev2_past_7days() -> None:
    inc = _resolved_sev(Severity.SEV2, resolved_at=T0)
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=10)) is True


def test_postmortem_overdue_sev3_never() -> None:
    """Pin: SEV3 doesn't require postmortem; never overdue."""

    inc = _resolved_sev(Severity.SEV3, resolved_at=T0)
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=30)) is False


def test_postmortem_overdue_sev4_never() -> None:
    inc = _resolved_sev(Severity.SEV4, resolved_at=T0)
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=30)) is False


def test_postmortem_overdue_published_never() -> None:
    """Pin: already-published postmortems are never overdue."""

    inc = _resolved_sev(Severity.SEV1, resolved_at=T0)
    inc = publish_postmortem(inc, author="charlie", now=T0 + timedelta(days=2))
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=30)) is False


def test_postmortem_overdue_open_never() -> None:
    """Pin: only RESOLVED sev1/sev2 can be postmortem-overdue.

    OPEN / ACKNOWLEDGED / MITIGATED states aren't ready for postmortem
    yet, so they're never "postmortem-overdue".
    """

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=30)) is False


# --------------------------- filter_overdue ----------------------------------


def test_filter_overdue_combines_ack_and_postmortem() -> None:
    """Both ack-overdue and postmortem-overdue surface."""

    sev1_overdue = open_incident(
        incident_id="i_ack",
        severity=Severity.SEV1,
        summary="prod down",
        now=T0 - timedelta(minutes=10),
    )
    sev2_pm_overdue = _resolved_sev(Severity.SEV2, resolved_at=T0 - timedelta(days=10))
    fresh = open_incident(
        incident_id="i_fresh",
        severity=Severity.SEV3,
        summary="x",
        now=T0,
    )

    overdue = filter_overdue([sev1_overdue, sev2_pm_overdue, fresh], now=T0)
    overdue_ids = {i.incident_id for i in overdue}
    assert "i_ack" in overdue_ids
    # sev2 ID was the default i1 from the helper
    assert any(i.incident_id == "i1" for i in overdue)
    assert "i_fresh" not in overdue_ids


def test_filter_overdue_sorts_by_severity_descending() -> None:
    """Pin: SEV1 first, then SEV2, etc."""

    sev2 = open_incident(
        incident_id="i_sev2",
        severity=Severity.SEV2,
        summary="x",
        now=T0 - timedelta(hours=1),
    )
    sev1 = open_incident(
        incident_id="i_sev1",
        severity=Severity.SEV1,
        summary="x",
        now=T0 - timedelta(minutes=10),
    )
    overdue = filter_overdue([sev2, sev1], now=T0)
    assert overdue[0].severity is Severity.SEV1
    assert overdue[1].severity is Severity.SEV2


def test_filter_overdue_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="now"):
        filter_overdue([], now=datetime(2026, 5, 1))


# --------------------------- render ------------------------------------------


def test_render_severity_emoji() -> None:
    """Pin: SEV1=🔴, SEV2=🟠, SEV3=🟡, SEV4=🔵."""

    sev1 = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    sev4 = open_incident(
        incident_id="i4",
        severity=Severity.SEV4,
        summary="x",
        now=T0,
    )
    assert "🔴" in render_incident(sev1)
    assert "🔵" in render_incident(sev4)


def test_render_status_emoji() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    assert "⚠️" in render_incident(inc)
    inc = acknowledge(inc, acker="alice", now=T0)
    assert "👀" in render_incident(inc)


def test_render_includes_attribution_when_set() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    inc = acknowledge(inc, acker="alice", now=T0)
    out = render_incident(inc)
    assert "alice" in out


def test_render_no_attribution_when_open() -> None:
    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="x",
        now=T0,
    )
    out = render_incident(inc)
    assert "acker:" not in out
    assert "resolver:" not in out


def test_render_no_secret_leak() -> None:
    """Pin: render shows summary + status + severity emoji + attribution.

    Never includes raw alert payload / stack traces / log lines /
    API keys.
    """

    inc = open_incident(
        incident_id="i1",
        severity=Severity.SEV1,
        summary="prod down",
        now=T0,
    )
    out = render_incident(inc)
    assert "stack" not in out.lower()
    assert "traceback" not in out.lower()
    assert "api_key" not in out.lower()
    assert "bearer" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_sev1_lifecycle() -> None:
    """Real-world: SEV1 alert → ack 2min → mitigate 30min → resolve 1hr →
    postmortem published 3 days later."""

    inc = open_incident(
        incident_id="i_sev1",
        severity=Severity.SEV1,
        summary="cycle service stopped responding",
        now=T0,
    )
    # Ack within SLA (5 min)
    inc = acknowledge(inc, acker="alice", now=T0 + timedelta(minutes=2))
    assert is_ack_overdue(inc, now=T0 + timedelta(minutes=10)) is False
    # Mitigate within 30min
    inc = mitigate(inc, now=T0 + timedelta(minutes=30))
    # Resolve within 1hr
    inc = resolve(inc, resolver="alice", now=T0 + timedelta(hours=1))
    # Postmortem within 7 days
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=3)) is False
    inc = publish_postmortem(inc, author="bob", now=T0 + timedelta(days=3))
    assert inc.status is IncidentStatus.POSTMORTEM_PUBLISHED


def test_e2e_sev1_postmortem_overdue() -> None:
    """Real-world: sev1 resolved but no postmortem published 14 days
    later — overdue."""

    inc = _resolved_sev(Severity.SEV1, resolved_at=T0)
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=14)) is True


def test_e2e_sev1_unacked_pages_oncall() -> None:
    """Pin: sev1 unacked at 6 min triggers ack-overdue (pages on-call)."""

    inc = open_incident(
        incident_id="i_sev1",
        severity=Severity.SEV1,
        summary="prod down",
        now=T0,
    )
    assert is_ack_overdue(inc, now=T0 + timedelta(minutes=6)) is True


def test_e2e_sev3_no_postmortem_required() -> None:
    """Pin: sev3 incident lifecycle ends at RESOLVED; no postmortem
    needed; not flagged overdue regardless of time elapsed."""

    inc = _resolved_sev(Severity.SEV3, resolved_at=T0)
    assert is_postmortem_overdue(inc, now=T0 + timedelta(days=30)) is False
    with pytest.raises(PostmortemNotRequiredError):
        publish_postmortem(inc, author="bob", now=T0 + timedelta(days=2))


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal incident states."""

    def build() -> Incident:
        inc = open_incident(
            incident_id="i1",
            severity=Severity.SEV1,
            summary="x",
            now=T0,
        )
        inc = acknowledge(inc, acker="alice", now=T0)
        return mitigate(inc, now=T0)

    a = build()
    b = build()
    assert a == b
