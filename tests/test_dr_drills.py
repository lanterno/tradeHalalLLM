"""Tests for `halal_trader.ops.dr_drills` (Wave 8.C).

Covers: drill kinds + canonical step sequences, step ordering,
drill failure short-circuit, cadence overdue logic, no-secret
render contract, immutability + replay-ability.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.ops.dr_drills import (
    DEFAULT_POLICY,
    DrillAlreadyFailedError,
    DrillKind,
    DrillPolicy,
    DrillRun,
    DrillStatus,
    StepOutOfOrderError,
    StepRecord,
    StepStatus,
    UnknownStepError,
    days_overdue,
    is_overdue,
    record_step,
    render_drill,
    start_drill,
    steps_for,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_drill_kind_string_values_pinned() -> None:
    assert DrillKind.BACKUP_RESTORE.value == "backup_restore"
    assert DrillKind.CYCLE_REPLAY.value == "cycle_replay"
    assert DrillKind.BROKER_RECONCILE.value == "broker_reconcile"


def test_step_status_string_values_pinned() -> None:
    assert StepStatus.PENDING.value == "pending"
    assert StepStatus.PASSED.value == "passed"
    assert StepStatus.FAILED.value == "failed"


def test_drill_status_string_values_pinned() -> None:
    assert DrillStatus.IN_PROGRESS.value == "in_progress"
    assert DrillStatus.PASSED.value == "passed"
    assert DrillStatus.FAILED.value == "failed"


# --------------------------- steps_for ---------------------------------------


def test_backup_restore_steps_pinned() -> None:
    assert steps_for(DrillKind.BACKUP_RESTORE) == (
        "verify_backup_artifact_exists",
        "spin_up_isolated_postgres",
        "restore_dump_into_isolated_db",
        "verify_row_counts_match_expected",
        "run_smoke_query_against_restored_db",
        "tear_down_isolated_postgres",
    )


def test_cycle_replay_steps_pinned() -> None:
    assert steps_for(DrillKind.CYCLE_REPLAY) == (
        "select_recent_cycle_id",
        "fetch_replay_inputs_from_db",
        "execute_replay_via_engine",
        "diff_replay_decision_against_recorded",
        "verify_diff_is_empty",
    )


def test_broker_reconcile_steps_pinned() -> None:
    assert steps_for(DrillKind.BROKER_RECONCILE) == (
        "snapshot_local_position_state",
        "fetch_broker_position_state",
        "compute_position_delta",
        "verify_delta_within_tolerance",
    )


# --------------------------- DrillPolicy -------------------------------------


def test_default_cadence_is_30_days() -> None:
    assert DEFAULT_POLICY.cadence == timedelta(days=30)


def test_policy_rejects_zero_cadence() -> None:
    with pytest.raises(ValueError, match="cadence"):
        DrillPolicy(cadence=timedelta(0))


def test_policy_rejects_negative_cadence() -> None:
    with pytest.raises(ValueError, match="cadence"):
        DrillPolicy(cadence=timedelta(seconds=-1))


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.cadence = timedelta(days=7)  # type: ignore[misc]


# --------------------------- StepRecord --------------------------------------


def test_step_record_rejects_empty_step() -> None:
    with pytest.raises(ValueError, match="step"):
        StepRecord(step="", status=StepStatus.PASSED, decided_at=T0)


def test_step_record_rejects_naive_decided_at() -> None:
    with pytest.raises(ValueError, match="decided_at"):
        StepRecord(
            step="foo",
            status=StepStatus.PASSED,
            decided_at=datetime(2026, 5, 1),
        )


def test_step_record_rejects_pending_status() -> None:
    with pytest.raises(ValueError, match="non-pending"):
        StepRecord(step="foo", status=StepStatus.PENDING, decided_at=T0)


def test_step_record_is_frozen() -> None:
    record = StepRecord(step="foo", status=StepStatus.PASSED, decided_at=T0)
    with pytest.raises(FrozenInstanceError):
        record.status = StepStatus.FAILED  # type: ignore[misc]


def test_step_record_default_notes_empty() -> None:
    record = StepRecord(step="foo", status=StepStatus.PASSED, decided_at=T0)
    assert record.notes == ""


# --------------------------- DrillRun ----------------------------------------


def test_drill_rejects_empty_drill_id() -> None:
    with pytest.raises(ValueError, match="drill_id"):
        DrillRun(
            drill_id="",
            kind=DrillKind.BACKUP_RESTORE,
            started_at=T0,
            operator="ops",
            records=(),
        )


def test_drill_rejects_empty_operator() -> None:
    with pytest.raises(ValueError, match="operator"):
        DrillRun(
            drill_id="d1",
            kind=DrillKind.BACKUP_RESTORE,
            started_at=T0,
            operator="",
            records=(),
        )


def test_drill_rejects_naive_started_at() -> None:
    with pytest.raises(ValueError, match="started_at"):
        DrillRun(
            drill_id="d1",
            kind=DrillKind.BACKUP_RESTORE,
            started_at=datetime(2026, 5, 1),
            operator="ops",
            records=(),
        )


def test_drill_is_frozen() -> None:
    drill = start_drill(
        drill_id="d1",
        kind=DrillKind.BACKUP_RESTORE,
        operator="ops",
        now=T0,
    )
    with pytest.raises(FrozenInstanceError):
        drill.drill_id = "other"  # type: ignore[misc]


# --------------------------- start_drill -------------------------------------


def test_start_drill_basic() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BACKUP_RESTORE, operator="ops", now=T0)
    assert drill.records == ()
    assert drill.aggregate_status is DrillStatus.IN_PROGRESS
    assert drill.next_step() == "verify_backup_artifact_exists"


def test_start_drill_rejects_empty_drill_id() -> None:
    with pytest.raises(ValueError, match="drill_id"):
        start_drill(
            drill_id="",
            kind=DrillKind.BACKUP_RESTORE,
            operator="ops",
            now=T0,
        )


def test_start_drill_rejects_empty_operator() -> None:
    with pytest.raises(ValueError, match="operator"):
        start_drill(
            drill_id="d1",
            kind=DrillKind.BACKUP_RESTORE,
            operator="",
            now=T0,
        )


def test_start_drill_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        start_drill(
            drill_id="d1",
            kind=DrillKind.BACKUP_RESTORE,
            operator="ops",
            now=datetime(2026, 5, 1),
        )


# --------------------------- record_step -------------------------------------


def _walk_passes(drill: DrillRun, steps: tuple[str, ...], *, base: datetime = T0) -> DrillRun:
    """Helper: pass every step in order."""

    for i, step in enumerate(steps):
        drill = record_step(drill, step, StepStatus.PASSED, now=base + timedelta(seconds=i))
    return drill


def test_record_step_passes_first_step() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(
        drill,
        "snapshot_local_position_state",
        StepStatus.PASSED,
        now=T0,
    )
    assert drill.status_of("snapshot_local_position_state") is StepStatus.PASSED


def test_record_step_full_pass_path() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = _walk_passes(drill, steps_for(DrillKind.BROKER_RECONCILE))
    assert drill.aggregate_status is DrillStatus.PASSED


def test_record_step_out_of_order_rejected() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    with pytest.raises(StepOutOfOrderError) as exc_info:
        record_step(
            drill,
            "compute_position_delta",
            StepStatus.PASSED,
            now=T0,
        )
    assert exc_info.value.step == "compute_position_delta"
    assert exc_info.value.missing == "snapshot_local_position_state"


def test_record_step_unknown_step_rejected() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    with pytest.raises(UnknownStepError) as exc_info:
        record_step(drill, "totally_not_a_step", StepStatus.PASSED, now=T0)
    assert exc_info.value.step == "totally_not_a_step"
    assert exc_info.value.kind is DrillKind.BROKER_RECONCILE


def test_record_step_already_decided_rejected() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)
    with pytest.raises(ValueError, match="already decided"):
        record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)


def test_record_step_pending_status_rejected() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    with pytest.raises(ValueError, match="PENDING"):
        record_step(
            drill,
            "snapshot_local_position_state",
            StepStatus.PENDING,
            now=T0,
        )


def test_record_step_naive_now_rejected() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    with pytest.raises(ValueError, match="now"):
        record_step(
            drill,
            "snapshot_local_position_state",
            StepStatus.PASSED,
            now=datetime(2026, 5, 1),
        )


def test_record_step_returns_new_state_not_mutates() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    new_drill = record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)
    assert drill.records == ()
    assert len(new_drill.records) == 1


def test_record_step_records_notes() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(
        drill,
        "snapshot_local_position_state",
        StepStatus.PASSED,
        now=T0,
        notes="snapshot OK, 3 positions",
    )
    record = drill.records[0]
    assert record.notes == "snapshot OK, 3 positions"


# --------------------------- failure short-circuit ---------------------------


def test_drill_marked_failed_on_step_failure() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)
    drill = record_step(
        drill,
        "fetch_broker_position_state",
        StepStatus.FAILED,
        now=T0,
        notes="broker returned 502",
    )
    assert drill.aggregate_status is DrillStatus.FAILED


def test_failed_drill_blocks_subsequent_steps() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)
    drill = record_step(drill, "fetch_broker_position_state", StepStatus.FAILED, now=T0)
    with pytest.raises(DrillAlreadyFailedError) as exc_info:
        record_step(drill, "compute_position_delta", StepStatus.PASSED, now=T0)
    assert exc_info.value.drill_id == "d1"


def test_failed_drill_next_step_returns_none() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.FAILED, now=T0)
    assert drill.next_step() is None


# --------------------------- aggregate_status --------------------------------


def test_aggregate_status_in_progress_at_start() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    assert drill.aggregate_status is DrillStatus.IN_PROGRESS


def test_aggregate_status_in_progress_partial() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)
    assert drill.aggregate_status is DrillStatus.IN_PROGRESS


def test_aggregate_status_passed_when_all_passed() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.CYCLE_REPLAY, operator="ops", now=T0)
    drill = _walk_passes(drill, steps_for(DrillKind.CYCLE_REPLAY))
    assert drill.aggregate_status is DrillStatus.PASSED


# --------------------------- completed_at ------------------------------------


def test_completed_at_none_in_progress() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    assert drill.completed_at is None


def test_completed_at_returns_max_decided_at() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    last = T0 + timedelta(minutes=5)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)
    drill = record_step(drill, "fetch_broker_position_state", StepStatus.PASSED, now=last)
    drill = record_step(
        drill,
        "compute_position_delta",
        StepStatus.PASSED,
        now=T0 + timedelta(minutes=2),
    )
    drill = record_step(
        drill,
        "verify_delta_within_tolerance",
        StepStatus.PASSED,
        now=T0 + timedelta(minutes=4),
    )
    assert drill.completed_at == last


# --------------------------- is_overdue --------------------------------------


def test_is_overdue_true_when_never_run() -> None:
    assert is_overdue(None, now=T0) is True


def test_is_overdue_false_when_recent() -> None:
    last = T0 - timedelta(days=10)
    assert is_overdue(last, now=T0) is False


def test_is_overdue_false_at_exactly_30_days_boundary() -> None:
    """Pin: 30d exactly is NOT overdue (>, not >=)."""

    last = T0 - timedelta(days=30)
    assert is_overdue(last, now=T0) is False


def test_is_overdue_true_past_30_days() -> None:
    last = T0 - timedelta(days=31)
    assert is_overdue(last, now=T0) is True


def test_is_overdue_uses_custom_cadence() -> None:
    last = T0 - timedelta(days=10)
    strict = DrillPolicy(cadence=timedelta(days=7))
    assert is_overdue(last, now=T0, policy=strict) is True


def test_is_overdue_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        is_overdue(None, now=datetime(2026, 5, 1))


def test_is_overdue_rejects_naive_last_passed_at() -> None:
    with pytest.raises(ValueError, match="last_passed_at"):
        is_overdue(datetime(2026, 4, 1), now=T0)


# --------------------------- days_overdue ------------------------------------


def test_days_overdue_zero_when_not_overdue() -> None:
    last = T0 - timedelta(days=10)
    assert days_overdue(last, now=T0) == 0


def test_days_overdue_returns_excess_days() -> None:
    last = T0 - timedelta(days=35)
    assert days_overdue(last, now=T0) == 5


def test_days_overdue_returns_neg_one_for_never_run() -> None:
    """Pin: never-run is signalled by sentinel -1."""

    assert days_overdue(None, now=T0) == -1


# --------------------------- render_drill ------------------------------------


def test_render_includes_drill_id_and_kind() -> None:
    drill = start_drill(
        drill_id="d_2026_05",
        kind=DrillKind.BACKUP_RESTORE,
        operator="alice",
        now=T0,
    )
    out = render_drill(drill)
    assert "d_2026_05" in out
    assert "backup_restore" in out
    assert "alice" in out


def test_render_shows_step_emoji_per_status() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.PASSED, now=T0)
    out = render_drill(drill)
    assert "✅" in out
    assert "⬜" in out  # for pending


def test_render_shows_failure_emoji() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(drill, "snapshot_local_position_state", StepStatus.FAILED, now=T0)
    out = render_drill(drill)
    assert "❌" in out


def test_render_shows_notes_when_present() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(
        drill,
        "snapshot_local_position_state",
        StepStatus.PASSED,
        now=T0,
        notes="3 positions snapshotted",
    )
    out = render_drill(drill)
    assert "3 positions snapshotted" in out


def test_render_shows_next_step_when_in_progress() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    out = render_drill(drill)
    assert "next:" in out
    assert "snapshot_local_position_state" in out


def test_render_shows_completed_when_done() -> None:
    drill = start_drill(drill_id="d1", kind=DrillKind.CYCLE_REPLAY, operator="ops", now=T0)
    drill = _walk_passes(drill, steps_for(DrillKind.CYCLE_REPLAY))
    out = render_drill(drill)
    assert "completed:" in out
    assert "passed" in out


def test_render_no_secret_leak() -> None:
    """Pin: render never includes API responses / Stripe IDs / dollar amounts."""

    drill = start_drill(drill_id="d1", kind=DrillKind.BROKER_RECONCILE, operator="ops", now=T0)
    drill = record_step(
        drill,
        "snapshot_local_position_state",
        StepStatus.PASSED,
        now=T0,
        notes="snapshot OK",
    )
    out = render_drill(drill)
    assert "$" not in out
    assert "USD" not in out
    assert "cus_" not in out.lower()
    assert "sub_" not in out.lower()
    assert "api_key" not in out.lower()
    assert "bearer" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_monthly_drill_passes() -> None:
    drill = start_drill(
        drill_id="dr_2026_05",
        kind=DrillKind.BACKUP_RESTORE,
        operator="ops_oncall",
        now=T0,
    )
    drill = _walk_passes(drill, steps_for(DrillKind.BACKUP_RESTORE), base=T0)
    assert drill.aggregate_status is DrillStatus.PASSED
    assert drill.completed_at is not None
    # The next overdue check uses this completion time
    next_ok = drill.completed_at + timedelta(days=10)
    assert is_overdue(drill.completed_at, now=next_ok) is False
    next_late = drill.completed_at + timedelta(days=45)
    assert is_overdue(drill.completed_at, now=next_late) is True


def test_e2e_drill_failure_blocks_progress() -> None:
    drill = start_drill(
        drill_id="dr_2026_05",
        kind=DrillKind.BACKUP_RESTORE,
        operator="ops_oncall",
        now=T0,
    )
    # First two steps pass, then we hit a failure on step 3
    drill = record_step(drill, "verify_backup_artifact_exists", StepStatus.PASSED, now=T0)
    drill = record_step(
        drill,
        "spin_up_isolated_postgres",
        StepStatus.PASSED,
        now=T0 + timedelta(seconds=30),
    )
    drill = record_step(
        drill,
        "restore_dump_into_isolated_db",
        StepStatus.FAILED,
        now=T0 + timedelta(seconds=60),
        notes="pg_restore reported corrupt header",
    )
    assert drill.aggregate_status is DrillStatus.FAILED
    # Subsequent steps blocked
    with pytest.raises(DrillAlreadyFailedError):
        record_step(
            drill,
            "verify_row_counts_match_expected",
            StepStatus.PASSED,
            now=T0,
        )


def test_e2e_replay_consistency() -> None:
    """Pin: same operations produce equal drill states."""

    def build() -> DrillRun:
        d = start_drill(
            drill_id="d1",
            kind=DrillKind.CYCLE_REPLAY,
            operator="ops",
            now=T0,
        )
        d = record_step(d, "select_recent_cycle_id", StepStatus.PASSED, now=T0)
        d = record_step(d, "fetch_replay_inputs_from_db", StepStatus.PASSED, now=T0)
        return d

    a = build()
    b = build()
    assert a == b
