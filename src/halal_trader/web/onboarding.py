"""Self-service onboarding flow.

The roadmap pins Wave 3.E: "New user lands → Google sign-in →
'connect a broker' → optional broker keys → 'pick strategy' →
optional model choice → first cycle runs in paper mode
automatically. Should be < 5 minutes from sign-in to first trade
simulated." This module is the **pure-Python state machine** the
onboarding route consumes to track each user's progress through
the flow and render the next-step prompt.

Picked a focused state machine over a "hand-roll progress checks
per route" approach because (a) the flow has a strict ordering
(can't pick a strategy before a broker is chosen) — encoding that
once means every route consults the same source of truth and
can't drift into letting a user pick a strategy with no broker;
(b) operators auditing "why did user X get stuck at step Y?" need
to replay the completion timestamps against a stable schema; (c)
the < 5-minute SLA the roadmap pins requires `time_to_first_trade`
math — a pure function of timestamps means the dashboard surfaces
the SLA breach automatically; (d) optional steps (broker keys,
model choice) are not order-skipped — they're gated optional.

Pinned semantics:
- **Steps complete in order; can't skip.** A user can't mark
  STRATEGY_CHOSEN as complete until BROKER_CHOSEN is complete.
  Pinned via `complete_step` raising on out-of-order completion.
- **Optional steps are skippable, not pre-filled.** A user with
  no broker API keys can `skip_step(BROKER_KEYS_STORED)` to advance;
  the audit row records "skipped" not "completed", so a future
  audit knows the user never provided keys.
- **First cycle is the terminal step.** FIRST_CYCLE_RUN marks
  onboarding done; `is_complete` flips True and the SLA clock
  stops via `time_to_first_trade`.
- **Five-minute SLA is operator-tunable.** Default 5 min; the
  flag fires when `time_to_first_trade > sla_threshold`.
- **Render output never includes broker API keys / Stripe IDs /
  session tokens.** Mirrors no-secret patterns of Wave 3.B vault
  + Wave 3.F billing + Wave 3.G admin console.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class OnboardingStep(str, Enum):
    """Ordered onboarding flow steps.

    Pinned string values for JSON / DB stability. Order matters —
    `_STEP_ORDER` below pins the canonical sequence and is the
    source of truth for `complete_step`'s prerequisite check.
    """

    SIGNED_IN = "signed_in"
    BROKER_CHOSEN = "broker_chosen"
    BROKER_KEYS_STORED = "broker_keys_stored"
    STRATEGY_CHOSEN = "strategy_chosen"
    MODEL_CHOSEN = "model_chosen"
    FIRST_CYCLE_RUN = "first_cycle_run"


_STEP_ORDER: tuple[OnboardingStep, ...] = (
    OnboardingStep.SIGNED_IN,
    OnboardingStep.BROKER_CHOSEN,
    OnboardingStep.BROKER_KEYS_STORED,
    OnboardingStep.STRATEGY_CHOSEN,
    OnboardingStep.MODEL_CHOSEN,
    OnboardingStep.FIRST_CYCLE_RUN,
)


_OPTIONAL_STEPS: frozenset[OnboardingStep] = frozenset(
    {OnboardingStep.BROKER_KEYS_STORED, OnboardingStep.MODEL_CHOSEN}
)


class StepStatus(str, Enum):
    """Per-step status. Pinned string values for JSON / DB stability."""

    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class StepOutOfOrderError(Exception):
    """Raised when a step is completed before its prerequisites."""

    def __init__(self, step: OnboardingStep, missing: OnboardingStep) -> None:
        super().__init__(f"cannot complete {step.value} before {missing.value} is done")
        self.step = step
        self.missing = missing


class StepNotSkippableError(Exception):
    """Raised when skip_step is called on a required step."""

    def __init__(self, step: OnboardingStep) -> None:
        super().__init__(f"step {step.value} is required and cannot be skipped")
        self.step = step


_DEFAULT_SLA = timedelta(minutes=5)


@dataclass(frozen=True)
class OnboardingPolicy:
    """Operator-tunable onboarding policy."""

    sla_threshold: timedelta = _DEFAULT_SLA

    def __post_init__(self) -> None:
        if self.sla_threshold <= timedelta(0):
            raise ValueError("sla_threshold must be positive")


DEFAULT_POLICY = OnboardingPolicy()


@dataclass(frozen=True)
class StepRecord:
    """Per-step audit row.

    `decided_at` is the timestamp the step was completed or
    skipped; `status` distinguishes the two so an audit can
    answer "did the user actually configure broker keys, or
    skip them?".
    """

    step: OnboardingStep
    status: StepStatus
    decided_at: datetime

    def __post_init__(self) -> None:
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        if self.status is StepStatus.PENDING:
            raise ValueError("StepRecord must record a non-pending status")


@dataclass(frozen=True)
class OnboardingState:
    """One user's onboarding progress.

    `started_at` is the SIGNED_IN timestamp (the SLA clock starts
    here); `records` is the immutable history of step decisions.
    Operations (`complete_step`, `skip_step`) return a new state
    rather than mutating — pinned for replay-ability.
    """

    user_id: str
    started_at: datetime
    records: tuple[StepRecord, ...]

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")

    def status_of(self, step: OnboardingStep) -> StepStatus:
        """Return the recorded status of a step (or PENDING if not yet)."""

        for record in self.records:
            if record.step is step:
                return record.status
        return StepStatus.PENDING

    @property
    def is_complete(self) -> bool:
        """True if FIRST_CYCLE_RUN is COMPLETED."""

        return self.status_of(OnboardingStep.FIRST_CYCLE_RUN) is StepStatus.COMPLETED

    def next_step(self) -> OnboardingStep | None:
        """The next pending step in canonical order, or None if done."""

        for step in _STEP_ORDER:
            if self.status_of(step) is StepStatus.PENDING:
                return step
        return None


def start_onboarding(*, user_id: str, now: datetime) -> OnboardingState:
    """Create a fresh onboarding state with SIGNED_IN already completed."""

    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    record = StepRecord(
        step=OnboardingStep.SIGNED_IN,
        status=StepStatus.COMPLETED,
        decided_at=now,
    )
    return OnboardingState(user_id=user_id, started_at=now, records=(record,))


def _next_pending_for(state: OnboardingState, step: OnboardingStep) -> OnboardingStep | None:
    """Return the first pending step at-or-before `step` in canonical order.

    Used to detect out-of-order completion: if `step` is BROKER_KEYS_STORED
    but BROKER_CHOSEN is still PENDING, the prerequisite check fails.
    """

    target_index = _STEP_ORDER.index(step)
    for prior in _STEP_ORDER[:target_index]:
        if state.status_of(prior) is StepStatus.PENDING and prior not in _OPTIONAL_STEPS:
            return prior
    return None


def complete_step(
    state: OnboardingState,
    step: OnboardingStep,
    *,
    now: datetime,
) -> OnboardingState:
    """Mark a step COMPLETED. Raises if prerequisites unmet."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if state.status_of(step) is not StepStatus.PENDING:
        raise ValueError(f"step {step.value} already decided")

    missing = _next_pending_for(state, step)
    if missing is not None:
        raise StepOutOfOrderError(step, missing)

    record = StepRecord(step=step, status=StepStatus.COMPLETED, decided_at=now)
    return OnboardingState(
        user_id=state.user_id,
        started_at=state.started_at,
        records=state.records + (record,),
    )


def skip_step(
    state: OnboardingState,
    step: OnboardingStep,
    *,
    now: datetime,
) -> OnboardingState:
    """Mark an optional step SKIPPED. Raises if step is required."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if step not in _OPTIONAL_STEPS:
        raise StepNotSkippableError(step)
    if state.status_of(step) is not StepStatus.PENDING:
        raise ValueError(f"step {step.value} already decided")

    record = StepRecord(step=step, status=StepStatus.SKIPPED, decided_at=now)
    return OnboardingState(
        user_id=state.user_id,
        started_at=state.started_at,
        records=state.records + (record,),
    )


def time_to_first_trade(state: OnboardingState) -> timedelta | None:
    """Elapsed time from SIGNED_IN to FIRST_CYCLE_RUN, or None if incomplete.

    The roadmap-pinned SLA is 5 minutes; `flag_sla_breach` consults
    this and the policy threshold.
    """

    for record in state.records:
        if record.step is OnboardingStep.FIRST_CYCLE_RUN and record.status is StepStatus.COMPLETED:
            return record.decided_at - state.started_at
    return None


def flag_sla_breach(
    state: OnboardingState,
    *,
    policy: OnboardingPolicy = DEFAULT_POLICY,
) -> bool:
    """True if time_to_first_trade exceeds the policy SLA.

    Returns False if onboarding hasn't completed yet — the SLA flag
    is for *completed* flows that took too long. A user still in
    progress isn't breaching yet (they may complete on time).
    """

    elapsed = time_to_first_trade(state)
    if elapsed is None:
        return False
    return elapsed > policy.sla_threshold


def progress_pct(state: OnboardingState) -> float:
    """Fraction of steps that are COMPLETED or SKIPPED."""

    decided = sum(1 for step in _STEP_ORDER if state.status_of(step) is not StepStatus.PENDING)
    return decided / len(_STEP_ORDER)


_STEP_LABEL: dict[OnboardingStep, str] = {
    OnboardingStep.SIGNED_IN: "Sign in",
    OnboardingStep.BROKER_CHOSEN: "Choose a broker",
    OnboardingStep.BROKER_KEYS_STORED: "Store broker API keys (optional)",
    OnboardingStep.STRATEGY_CHOSEN: "Pick a strategy",
    OnboardingStep.MODEL_CHOSEN: "Choose an LLM model (optional)",
    OnboardingStep.FIRST_CYCLE_RUN: "Run your first paper-trading cycle",
}


_STATUS_EMOJI: dict[StepStatus, str] = {
    StepStatus.PENDING: "⬜",
    StepStatus.COMPLETED: "✅",
    StepStatus.SKIPPED: "⏭️",
}


def render_onboarding_state(state: OnboardingState) -> str:
    """Format the onboarding state for ops display.

    Pinned no-secret-leak: never includes broker API keys / Stripe
    customer IDs / session tokens. Renders user_id + per-step
    emoji + label + decision timestamp + progress percentage +
    next-step hint.
    """

    pct = progress_pct(state) * 100
    lines = [
        f"Onboarding for {state.user_id} — {pct:.0f}% complete",
        f"  started: {state.started_at.isoformat()}",
    ]
    for step in _STEP_ORDER:
        status = state.status_of(step)
        emoji = _STATUS_EMOJI[status]
        label = _STEP_LABEL[step]
        suffix = ""
        if status is not StepStatus.PENDING:
            for record in state.records:
                if record.step is step:
                    suffix = f" — {record.decided_at.isoformat()}"
                    break
        lines.append(f"  {emoji} {label}{suffix}")

    next_step = state.next_step()
    if next_step is not None:
        lines.append(f"  next: {_STEP_LABEL[next_step]}")
    else:
        elapsed = time_to_first_trade(state)
        if elapsed is not None:
            secs = elapsed.total_seconds()
            lines.append(f"  ⭐ done — first trade in {secs:.0f}s")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "OnboardingPolicy",
    "OnboardingState",
    "OnboardingStep",
    "StepNotSkippableError",
    "StepOutOfOrderError",
    "StepRecord",
    "StepStatus",
    "complete_step",
    "flag_sla_breach",
    "progress_pct",
    "render_onboarding_state",
    "skip_step",
    "start_onboarding",
    "time_to_first_trade",
]
