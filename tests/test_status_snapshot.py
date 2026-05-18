"""Tests for `core/status_snapshot.py`.

Pins the four-level classification decision tree, the
ongoing-vs-historical halt split, the empty-stream → OPERATIONAL
contract, the reason-filter denylist, and the threshold-
validation rules.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.core.status_snapshot import (
    CycleEventRecord,
    HaltRecord,
    IncidentSummary,
    StatusLevel,
    StatusSnapshot,
    StatusThresholds,
    build_snapshot,
    filter_reason,
    render_snapshot,
)

_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _ago(**kwargs) -> datetime:
    """Build a datetime relative to _NOW."""
    return _NOW - timedelta(**kwargs)


def _halt(
    *,
    engaged_minutes_ago: float,
    duration_minutes: float | None = None,
    reason: str = "operator halt",
) -> HaltRecord:
    engaged = _NOW - timedelta(minutes=engaged_minutes_ago)
    if duration_minutes is None:
        resolved: datetime | None = None
    else:
        resolved = engaged + timedelta(minutes=duration_minutes)
    return HaltRecord(engaged_at=engaged, reason=reason, resolved_at=resolved)


def _cycle(
    *,
    minutes_ago: float,
    succeeded: bool = True,
    duration_ms: float = 100.0,
) -> CycleEventRecord:
    return CycleEventRecord(
        at=_NOW - timedelta(minutes=minutes_ago),
        succeeded=succeeded,
        duration_ms=duration_ms,
    )


# ── threshold validation ─────────────────────────────────


def test_thresholds_reject_misordered_success_rates():
    """Pin: thresholds must form a non-decreasing ladder. A
    config where degraded < partial would silently classify
    every cycle as PARTIAL_OUTAGE."""
    with pytest.raises(ValueError, match="ordered"):
        StatusThresholds(
            success_rate_partial=0.99,
            success_rate_degraded=0.95,
            success_rate_operational=0.99,
        )


def test_thresholds_reject_negative_recent_halt_minutes():
    with pytest.raises(ValueError, match="recent_halt_minutes"):
        StatusThresholds(recent_halt_minutes=-1)


def test_thresholds_reject_zero_window_days():
    with pytest.raises(ValueError, match="incident_window_days"):
        StatusThresholds(incident_window_days=0)


# ── HaltRecord validation ────────────────────────────────


def test_halt_record_rejects_resolved_before_engaged():
    """Pin: data corruption check — a halt resolved before it
    engaged is impossible; surface immediately rather than
    producing a negative duration."""
    with pytest.raises(ValueError, match="resolved_at"):
        HaltRecord(
            engaged_at=_NOW,
            reason="x",
            resolved_at=_NOW - timedelta(minutes=5),
        )


def test_halt_active_when_unresolved():
    h = HaltRecord(engaged_at=_ago(minutes=10), reason="x")
    assert h.is_active(now=_NOW)


def test_halt_not_active_when_resolved():
    h = HaltRecord(
        engaged_at=_ago(minutes=10),
        reason="x",
        resolved_at=_ago(minutes=5),
    )
    assert not h.is_active(now=_NOW)


def test_halt_duration_uses_now_when_unresolved():
    h = HaltRecord(engaged_at=_ago(minutes=10), reason="x")
    duration = h.duration(now=_NOW)
    assert duration == timedelta(minutes=10)


def test_halt_duration_uses_resolved_when_set():
    h = HaltRecord(
        engaged_at=_ago(minutes=10),
        reason="x",
        resolved_at=_ago(minutes=7),
    )
    assert h.duration(now=_NOW) == timedelta(minutes=3)


# ── CycleEventRecord validation ──────────────────────────


def test_cycle_event_rejects_negative_duration():
    with pytest.raises(ValueError, match="duration_ms"):
        CycleEventRecord(at=_NOW, succeeded=True, duration_ms=-1.0)


# ── filter_reason ────────────────────────────────────────


def test_filter_reason_redacts_api_key_substring():
    """Pin: any reason containing a sensitive substring is
    fully replaced rather than partially redacted. Partial
    redaction leaks key length / structure."""
    assert filter_reason("debug api_key rotation") == "halt for operational reasons"


def test_filter_reason_redacts_secret_substring():
    assert filter_reason("rotating SECRET token") == "halt for operational reasons"


def test_filter_reason_redacts_password_substring():
    assert filter_reason("changed db password") == "halt for operational reasons"


def test_filter_reason_passes_through_clean_text():
    assert filter_reason("scheduled maintenance") == "scheduled maintenance"


def test_filter_reason_caps_long_text():
    long_reason = "x" * 200
    out = filter_reason(long_reason, max_chars=80)
    assert len(out) == 80
    assert out.endswith("…")


def test_filter_reason_handles_empty_string():
    """Pin: empty input passes through (an unset reason should
    still publish as a blank summary, not crash)."""
    assert filter_reason("") == ""


def test_filter_reason_case_insensitive_match():
    """Pin: substring match is lower-cased so 'API_KEY' / 'api_key'
    / 'Api_Key' all redact."""
    assert filter_reason("rotating API_KEY") == "halt for operational reasons"


# ── empty streams → OPERATIONAL ──────────────────────────


def test_empty_streams_yield_operational():
    """Pin: no-data is OPERATIONAL (success_rate defaults to
    1.0 on empty cycle stream). A public page showing 'no cycles
    observed' shouldn't alarm."""
    snap = build_snapshot(halts=[], cycle_events=[], now=_NOW)
    assert snap.level == StatusLevel.OPERATIONAL
    assert snap.success_rate == 1.0
    assert snap.cycle_count == 0
    assert not snap.is_currently_halted
    assert snap.incidents == []
    assert snap.ongoing_incident is None


# ── classification: ongoing halt ─────────────────────────


def test_ongoing_halt_yields_major_outage():
    """Pin: a currently-active halt is MAJOR_OUTAGE regardless
    of success rate."""
    halts = [_halt(engaged_minutes_ago=30)]
    snap = build_snapshot(
        halts=halts,
        cycle_events=[_cycle(minutes_ago=i) for i in range(60)],
        now=_NOW,
    )
    assert snap.level == StatusLevel.MAJOR_OUTAGE
    assert snap.is_currently_halted
    assert snap.ongoing_incident is not None


def test_ongoing_halt_ignored_in_historical_list():
    """Pin: an active halt is the *headline*, not a historical
    incident. The historical list should not contain it."""
    halts = [_halt(engaged_minutes_ago=30)]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    assert snap.ongoing_incident is not None
    assert snap.incidents == []


# ── classification: success-rate ladder ──────────────────


def test_99_success_rate_no_halts_is_operational():
    cycles = [_cycle(minutes_ago=i) for i in range(100)]
    snap = build_snapshot(halts=[], cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.OPERATIONAL


def test_97_success_rate_yields_degraded():
    cycles = [_cycle(minutes_ago=i, succeeded=(i % 33 != 0)) for i in range(100)]
    # ~3 failures out of 100 → 97% success
    snap = build_snapshot(halts=[], cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.DEGRADED


def test_90_success_rate_yields_partial_outage():
    cycles = [_cycle(minutes_ago=i, succeeded=(i % 10 != 0)) for i in range(100)]
    # ~10% failures → 90% success
    snap = build_snapshot(halts=[], cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.PARTIAL_OUTAGE


def test_70_success_rate_yields_major_outage():
    cycles = [_cycle(minutes_ago=i, succeeded=(i % 10 >= 3)) for i in range(100)]
    # 30% failures → 70% success
    snap = build_snapshot(halts=[], cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.MAJOR_OUTAGE


# ── classification: long-halt rule ───────────────────────


def test_long_recovered_halt_flips_to_partial_outage():
    """Pin: a halt longer than `recent_halt_minutes` (default 5
    min) flips the level to at least PARTIAL_OUTAGE even when
    success rate is fine."""
    halts = [
        _halt(engaged_minutes_ago=120, duration_minutes=10, reason="bad day"),
    ]
    cycles = [_cycle(minutes_ago=i) for i in range(100)]  # 100% success
    snap = build_snapshot(halts=halts, cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.PARTIAL_OUTAGE


def test_short_recovered_halt_only_yields_degraded():
    """Pin: a < 5 min halt counts as a "blip" — DEGRADED, not
    PARTIAL_OUTAGE. The level acknowledges the incident without
    over-reporting."""
    halts = [
        _halt(engaged_minutes_ago=120, duration_minutes=2, reason="quick fix"),
    ]
    cycles = [_cycle(minutes_ago=i) for i in range(100)]
    snap = build_snapshot(halts=halts, cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.DEGRADED


def test_long_halt_with_low_success_rate_yields_major_outage():
    """Long halt + low success rate stacks → MAJOR_OUTAGE."""
    halts = [
        _halt(engaged_minutes_ago=120, duration_minutes=10, reason="x"),
    ]
    cycles = [_cycle(minutes_ago=i, succeeded=(i % 10 != 0)) for i in range(100)]
    # 90% success (degraded) + long halt (partial) → degraded by
    # success ladder, partial via halt ladder, but the long-halt
    # check overrides up to PARTIAL_OUTAGE; success-rate-degraded
    # check elevates to MAJOR_OUTAGE per the decision tree.
    snap = build_snapshot(halts=halts, cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.MAJOR_OUTAGE


# ── window inclusion ─────────────────────────────────────


def test_halts_outside_window_excluded():
    """Pin: incident_window_days=7 by default. A halt 30d ago
    must not appear in the snapshot."""
    halts = [
        _halt(
            engaged_minutes_ago=30 * 24 * 60,  # 30 days ago
            duration_minutes=10,
        ),
    ]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    assert snap.incidents == []
    assert snap.ongoing_incident is None
    assert snap.level == StatusLevel.OPERATIONAL


def test_cycle_events_outside_window_excluded():
    """Old failures should not affect the snapshot's success
    rate."""
    cycles = [
        # 30 days ago, all failures
        _cycle(minutes_ago=30 * 24 * 60 + i, succeeded=False)
        for i in range(50)
    ]
    snap = build_snapshot(halts=[], cycle_events=cycles, now=_NOW)
    assert snap.level == StatusLevel.OPERATIONAL
    assert snap.cycle_count == 0  # all out of window


def test_custom_window_days_changes_inclusion():
    halts = [_halt(engaged_minutes_ago=10 * 24 * 60, duration_minutes=10)]
    # Default 7d window → halt excluded
    snap_default = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    assert snap_default.incidents == []
    # 14d window → halt included
    snap_extended = build_snapshot(
        halts=halts,
        cycle_events=[],
        now=_NOW,
        thresholds=StatusThresholds(incident_window_days=14),
    )
    assert len(snap_extended.incidents) == 1


# ── ordering ─────────────────────────────────────────────


def test_incidents_sorted_most_recent_first():
    halts = [
        _halt(engaged_minutes_ago=120, duration_minutes=2, reason="oldest"),
        _halt(engaged_minutes_ago=30, duration_minutes=2, reason="newest"),
        _halt(engaged_minutes_ago=60, duration_minutes=2, reason="middle"),
    ]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    assert [i.reason for i in snap.incidents] == [
        "newest",
        "middle",
        "oldest",
    ]


# ── reason filtering in incidents ────────────────────────


def test_incident_reasons_pass_through_filter():
    halts = [
        _halt(
            engaged_minutes_ago=30,
            duration_minutes=2,
            reason="rotated api_key",
        )
    ]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    assert snap.incidents[0].reason == "halt for operational reasons"


# ── snapshot output ──────────────────────────────────────


def test_snapshot_records_captured_at():
    snap = build_snapshot(halts=[], cycle_events=[], now=_NOW)
    assert snap.captured_at == _NOW


def test_snapshot_summary_contains_emoji():
    snap = build_snapshot(halts=[], cycle_events=[], now=_NOW)
    assert "🟢" in snap.summary


def test_snapshot_summary_mentions_halt_when_active():
    halts = [_halt(engaged_minutes_ago=30)]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    assert "halt" in snap.summary.lower()


def test_snapshot_summary_mentions_cycle_count():
    cycles = [_cycle(minutes_ago=i) for i in range(50)]
    snap = build_snapshot(halts=[], cycle_events=cycles, now=_NOW)
    assert "50 cycles" in snap.summary


def test_snapshot_summary_handles_no_cycle_data():
    snap = build_snapshot(halts=[], cycle_events=[], now=_NOW)
    assert "no cycle data" in snap.summary.lower()


# ── status level ordering ────────────────────────────────


def test_status_level_str_values_documented():
    """Pin the string values so a future enum addition can't
    silently break the public-API stability."""
    assert StatusLevel.OPERATIONAL.value == "operational"
    assert StatusLevel.DEGRADED.value == "degraded"
    assert StatusLevel.PARTIAL_OUTAGE.value == "partial_outage"
    assert StatusLevel.MAJOR_OUTAGE.value == "major_outage"


# ── output structure ─────────────────────────────────────


def test_snapshot_immutable():
    snap = build_snapshot(halts=[], cycle_events=[], now=_NOW)
    assert isinstance(snap, StatusSnapshot)
    with pytest.raises(Exception):
        snap.level = StatusLevel.MAJOR_OUTAGE  # type: ignore[misc]


def test_incident_summary_immutable():
    halts = [_halt(engaged_minutes_ago=30, duration_minutes=2)]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    inc = snap.incidents[0]
    assert isinstance(inc, IncidentSummary)
    with pytest.raises(Exception):
        inc.reason = "tampered"  # type: ignore[misc]


# ── render ───────────────────────────────────────────────


def test_render_includes_summary_line():
    snap = build_snapshot(halts=[], cycle_events=[], now=_NOW)
    text = render_snapshot(snap)
    assert "🟢" in text
    assert "operational" in text


def test_render_handles_ongoing_incident():
    halts = [_halt(engaged_minutes_ago=10)]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    text = render_snapshot(snap)
    assert "⚠️" in text
    assert "Ongoing" in text


def test_render_lists_recent_incidents():
    halts = [
        _halt(engaged_minutes_ago=30, duration_minutes=2, reason="x"),
        _halt(engaged_minutes_ago=60, duration_minutes=3, reason="y"),
    ]
    snap = build_snapshot(halts=halts, cycle_events=[], now=_NOW)
    text = render_snapshot(snap)
    assert "Recent incidents" in text
    assert "x" in text
    assert "y" in text


def test_render_says_no_incidents_on_clean_window():
    snap = build_snapshot(halts=[], cycle_events=[], now=_NOW)
    text = render_snapshot(snap)
    assert "No incidents" in text
