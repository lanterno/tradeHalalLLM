"""Disaster recovery drill engine.

The roadmap pins Wave 8.C: "Restore from backup, replay the last
24 hours of cycles, reconcile broker state. Documented runbook +
monthly automated drill." This module is the **pure-Python drill
state machine + scoring layer** the operator runs (manually or
via the monthly cron) to verify the bot's recovery primitives
still work after a failure scenario.

Picked a focused state machine over a "hand-roll a runbook PDF
checklist" approach because (a) DR drills must be replay-able
and outcome-pinnable — a "we ran the drill last month and it
passed" claim is worthless without the timestamped per-step
audit trail this engine produces; (b) the three drill kinds
(backup-restore / cycle-replay / broker-reconcile) share an
abstract step-state-machine but each has its own canonical step
sequence — pinning the steps in code lets the engine route the
right runbook for the right drill kind without the operator
remembering; (c) cadence enforcement (monthly drills are
worthless if the operator forgets to schedule them) is a pure
function of the last-passed timestamp + cadence policy, so the
dashboard can surface "drill is overdue" without re-implementing
the math; (d) the drill scoring (pass / fail / partial) feeds the
status page (Wave 8.G) and operator email summaries.

Pinned semantics:
- **Three canonical drill kinds.** BACKUP_RESTORE (verify Wave
  8.F backup runbook actually restores cleanly), CYCLE_REPLAY
  (verify Wave 6.E deterministic replay engine works on real
  cycle data), BROKER_RECONCILE (verify `core/reconcile.py`
  catches divergence between local and broker-side state).
  Each kind has its own canonical step sequence pinned in code.
- **Steps must complete in order.** A user can't mark "verify
  reconciliation report" as PASSED before "fetch broker state"
  is PASSED. Pinned via `record_step` raising on out-of-order.
- **Failed step blocks subsequent steps.** Once any step is
  FAILED, the drill is marked FAILED and remaining steps are
  not executable. Operator must restart the drill (new drill_id)
  after fixing the root cause.
- **Monthly cadence by default; operator-tunable.** A drill is
  overdue if (now - last_passed_at) > cadence; default 30 days.
- **Render output never includes broker API responses, account
  balances, or restored database contents.** Mirrors the
  no-secret patterns of Wave 3.B vault + 3.G admin console.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class DrillKind(str, Enum):
    """Canonical disaster recovery drill kinds.

    Pinned string values for JSON / DB stability. Adding a kind
    is a code review change.
    """

    BACKUP_RESTORE = "backup_restore"
    CYCLE_REPLAY = "cycle_replay"
    BROKER_RECONCILE = "broker_reconcile"


class StepStatus(str, Enum):
    """Per-step status. Pinned string values for JSON / DB stability."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class DrillStatus(str, Enum):
    """Aggregate drill status."""

    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"


_BACKUP_RESTORE_STEPS: tuple[str, ...] = (
    "verify_backup_artifact_exists",
    "spin_up_isolated_postgres",
    "restore_dump_into_isolated_db",
    "verify_row_counts_match_expected",
    "run_smoke_query_against_restored_db",
    "tear_down_isolated_postgres",
)


_CYCLE_REPLAY_STEPS: tuple[str, ...] = (
    "select_recent_cycle_id",
    "fetch_replay_inputs_from_db",
    "execute_replay_via_engine",
    "diff_replay_decision_against_recorded",
    "verify_diff_is_empty",
)


_BROKER_RECONCILE_STEPS: tuple[str, ...] = (
    "snapshot_local_position_state",
    "fetch_broker_position_state",
    "compute_position_delta",
    "verify_delta_within_tolerance",
)


_STEPS_BY_KIND: dict[DrillKind, tuple[str, ...]] = {
    DrillKind.BACKUP_RESTORE: _BACKUP_RESTORE_STEPS,
    DrillKind.CYCLE_REPLAY: _CYCLE_REPLAY_STEPS,
    DrillKind.BROKER_RECONCILE: _BROKER_RECONCILE_STEPS,
}


def steps_for(kind: DrillKind) -> tuple[str, ...]:
    """Return the canonical step sequence for a drill kind."""

    return _STEPS_BY_KIND[kind]


class StepOutOfOrderError(Exception):
    """Raised when a step is recorded before its prerequisites."""

    def __init__(self, step: str, missing: str) -> None:
        super().__init__(f"cannot record {step!r} before {missing!r} is done")
        self.step = step
        self.missing = missing


class DrillAlreadyFailedError(Exception):
    """Raised when a step is recorded against a drill that already failed."""

    def __init__(self, drill_id: str) -> None:
        super().__init__(f"drill {drill_id!r} already failed; restart with new drill")
        self.drill_id = drill_id


class UnknownStepError(Exception):
    """Raised when a step name isn't in the kind's canonical sequence."""

    def __init__(self, step: str, kind: DrillKind) -> None:
        super().__init__(f"step {step!r} is not in {kind.value} canonical sequence")
        self.step = step
        self.kind = kind


_DEFAULT_CADENCE = timedelta(days=30)


@dataclass(frozen=True)
class DrillPolicy:
    """Operator-tunable drill policy."""

    cadence: timedelta = _DEFAULT_CADENCE

    def __post_init__(self) -> None:
        if self.cadence <= timedelta(0):
            raise ValueError("cadence must be positive")


DEFAULT_POLICY = DrillPolicy()


@dataclass(frozen=True)
class StepRecord:
    """Per-step audit row.

    `notes` is a free-form short description (e.g. "row count
    diff: 0", "broker reported 502 retry"); pinned no-secret-leak
    contract — operators must not paste account balances or API
    responses into notes (the render helper's regression test
    asserts dollar signs and Stripe IDs don't appear).
    """

    step: str
    status: StepStatus
    decided_at: datetime
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.step or not self.step.strip():
            raise ValueError("step must be non-empty")
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        if self.status is StepStatus.PENDING:
            raise ValueError("StepRecord must record a non-pending status")


@dataclass(frozen=True)
class DrillRun:
    """One drill's state.

    Operations (`record_step`) return new state rather than
    mutating — pinned for replay-ability.
    """

    drill_id: str
    kind: DrillKind
    started_at: datetime
    operator: str
    records: tuple[StepRecord, ...]

    def __post_init__(self) -> None:
        if not self.drill_id or not self.drill_id.strip():
            raise ValueError("drill_id must be non-empty")
        if not self.operator or not self.operator.strip():
            raise ValueError("operator must be non-empty")
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")

    def status_of(self, step: str) -> StepStatus:
        """Return the recorded status of a step, or PENDING if not yet."""

        for record in self.records:
            if record.step == step:
                return record.status
        return StepStatus.PENDING

    @property
    def aggregate_status(self) -> DrillStatus:
        """Aggregate: FAILED if any failed; PASSED if all passed; else IN_PROGRESS."""

        if any(r.status is StepStatus.FAILED for r in self.records):
            return DrillStatus.FAILED
        all_steps = steps_for(self.kind)
        if all(self.status_of(s) is StepStatus.PASSED for s in all_steps):
            return DrillStatus.PASSED
        return DrillStatus.IN_PROGRESS

    def next_step(self) -> str | None:
        """The next pending step, or None if drill is done (passed or failed)."""

        if self.aggregate_status is DrillStatus.FAILED:
            return None
        for step in steps_for(self.kind):
            if self.status_of(step) is StepStatus.PENDING:
                return step
        return None

    @property
    def completed_at(self) -> datetime | None:
        """Timestamp of the final step decision, or None if in-progress."""

        status = self.aggregate_status
        if status is DrillStatus.IN_PROGRESS:
            return None
        if not self.records:
            return None
        return max(r.decided_at for r in self.records)


def start_drill(
    *,
    drill_id: str,
    kind: DrillKind,
    operator: str,
    now: datetime,
) -> DrillRun:
    """Create a fresh drill run with no records yet."""

    if not drill_id or not drill_id.strip():
        raise ValueError("drill_id must be non-empty")
    if not operator or not operator.strip():
        raise ValueError("operator must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return DrillRun(
        drill_id=drill_id,
        kind=kind,
        started_at=now,
        operator=operator,
        records=(),
    )


def record_step(
    drill: DrillRun,
    step: str,
    status: StepStatus,
    *,
    now: datetime,
    notes: str = "",
) -> DrillRun:
    """Record a step's outcome on the drill.

    Raises:
    - `UnknownStepError` if `step` isn't in the kind's canonical sequence
    - `StepOutOfOrderError` if prerequisites aren't all PASSED
    - `DrillAlreadyFailedError` if any prior step already FAILED
    - `ValueError` for naive `now` or PENDING `status` or already-decided step
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if status is StepStatus.PENDING:
        raise ValueError("cannot record PENDING status; use PASSED or FAILED")

    canonical = steps_for(drill.kind)
    if step not in canonical:
        raise UnknownStepError(step, drill.kind)

    if drill.aggregate_status is DrillStatus.FAILED:
        raise DrillAlreadyFailedError(drill.drill_id)

    if drill.status_of(step) is not StepStatus.PENDING:
        raise ValueError(f"step {step!r} already decided")

    target_index = canonical.index(step)
    for prior in canonical[:target_index]:
        if drill.status_of(prior) is not StepStatus.PASSED:
            raise StepOutOfOrderError(step, prior)

    record = StepRecord(step=step, status=status, decided_at=now, notes=notes)
    return DrillRun(
        drill_id=drill.drill_id,
        kind=drill.kind,
        started_at=drill.started_at,
        operator=drill.operator,
        records=drill.records + (record,),
    )


def is_overdue(
    last_passed_at: datetime | None,
    *,
    now: datetime,
    policy: DrillPolicy = DEFAULT_POLICY,
) -> bool:
    """True if drill is overdue (never run or last pass beyond cadence).

    Accepts None for last_passed_at (never-run case → overdue).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if last_passed_at is None:
        return True
    if last_passed_at.tzinfo is None:
        raise ValueError("last_passed_at must be timezone-aware")
    return now - last_passed_at > policy.cadence


def days_overdue(
    last_passed_at: datetime | None,
    *,
    now: datetime,
    policy: DrillPolicy = DEFAULT_POLICY,
) -> int:
    """Days past the cadence; 0 if not overdue.

    Used by the dashboard tile to show "drill is X days overdue".
    """

    if not is_overdue(last_passed_at, now=now, policy=policy):
        return 0
    if last_passed_at is None:
        # Never run — count from epoch-ish placeholder; report 0 + just return positive
        return -1  # sentinel for "never run"
    elapsed = now - last_passed_at
    overdue_by = elapsed - policy.cadence
    return max(0, overdue_by.days)


_STATUS_EMOJI: dict[StepStatus, str] = {
    StepStatus.PENDING: "⬜",
    StepStatus.PASSED: "✅",
    StepStatus.FAILED: "❌",
}


_DRILL_STATUS_EMOJI: dict[DrillStatus, str] = {
    DrillStatus.IN_PROGRESS: "🔄",
    DrillStatus.PASSED: "✅",
    DrillStatus.FAILED: "❌",
}


def render_drill(drill: DrillRun) -> str:
    """Format the drill state for ops display.

    Pinned no-secret-leak: never includes broker API responses,
    account balances, restored database contents, Stripe IDs.
    Renders kind + operator + per-step emoji + step + notes (if
    any) + aggregate status.
    """

    emoji = _DRILL_STATUS_EMOJI[drill.aggregate_status]
    lines = [
        f"{emoji} DR drill {drill.drill_id} — {drill.kind.value}",
        f"  operator: {drill.operator}",
        f"  started: {drill.started_at.isoformat()}",
        f"  status: {drill.aggregate_status.value}",
    ]

    for step in steps_for(drill.kind):
        status = drill.status_of(step)
        step_emoji = _STATUS_EMOJI[status]
        line = f"  {step_emoji} {step}"
        for record in drill.records:
            if record.step == step and record.notes:
                line += f" — {record.notes}"
                break
        lines.append(line)

    nxt = drill.next_step()
    if nxt is not None:
        lines.append(f"  next: {nxt}")

    if drill.completed_at is not None:
        lines.append(f"  completed: {drill.completed_at.isoformat()}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "DrillAlreadyFailedError",
    "DrillKind",
    "DrillPolicy",
    "DrillRun",
    "DrillStatus",
    "StepOutOfOrderError",
    "StepRecord",
    "StepStatus",
    "UnknownStepError",
    "days_overdue",
    "is_overdue",
    "record_step",
    "render_drill",
    "start_drill",
    "steps_for",
]
