"""LLM ensemble distillation policy + orchestration.

The roadmap pins Wave 6.I: "Once the ensemble has emitted 10k+
decisions, train a compact local model (DistilBERT-sized) to
mimic its decisions. Trades 5% of accuracy for 100x faster
inference + zero LLM cost. Pin this as a long-term cost-reduction
lever." This module is the **pure-Python distillation policy +
deployment-gate engine** that decides when to trigger a
distillation run, validates the trained student model meets the
accuracy + latency gates, and tracks deployment readiness.

Picked a focused policy engine over a "single train script"
approach because (a) the trigger conditions (10k+ decisions,
sufficient diversity, no recent retraining) are decision rules,
not training code — pinning them lets a future cron-driven
pipeline consult the same gate rather than re-deriving the
trigger math, (b) the student-vs-teacher accuracy gate (≥95%
agreement on a holdout cohort) and latency gate (≥10x faster
than the teacher) are deployment-readiness criteria — the
operator's `halal-trader ml status` surfaces them as pass/fail
without re-running the model, (c) the deployment lifecycle
(SHADOW → CANARY → PRODUCTION) is a state machine identical in
shape to other waves' progression engines (3.E onboarding, 8.C
DR drills, 10.G partnerships) — keeping it consistent here
means the operator's mental model transfers directly. The
actual training step (HuggingFace `transformers` distillation
loop) is operator-side and consumes this module's policy
output as configuration.

Pinned semantics:
- **Distillation requires 10k+ teacher decisions.** The roadmap
  threshold; below it the student trains on too little data.
  Operator-tunable via `DistillationPolicy`; default 10000.
- **Decision cohort must be diverse.** A 10k-decision cohort
  where 9.5k are HOLD and only 500 are BUY/SELL gives the
  student no signal on the rare classes; the policy enforces a
  minimum class-balance ratio (default 5% per class).
- **Accuracy gate is ≥95% agreement with teacher.** Below 95%
  the student is rejected — the roadmap pins "5% of accuracy"
  as the acceptable trade for the latency gain. Operator-tunable.
- **Latency improvement gate is ≥10x faster.** A student barely
  faster than the teacher isn't worth the deployment complexity;
  the policy rejects students that don't deliver the latency
  win that justifies the project.
- **Deployment progresses SHADOW → CANARY → PRODUCTION.** Steps
  are one-at-a-time; can't promote to PRODUCTION without
  surviving CANARY. Pinned via state-machine ordering.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class DistillationDecision(str, Enum):
    """Whether to launch a distillation run.

    Pinned string values for JSON / DB stability.
    """

    SKIP_INSUFFICIENT_DECISIONS = "skip_insufficient_decisions"
    SKIP_INSUFFICIENT_DIVERSITY = "skip_insufficient_diversity"
    SKIP_RECENT_RETRAIN = "skip_recent_retrain"
    LAUNCH = "launch"


class DeploymentStage(str, Enum):
    """Student-model deployment stages.

    Pinned string values. SHADOW → CANARY → PRODUCTION; off-funnel
    REJECTED (failed a gate) and RETIRED (replaced by a newer
    student).
    """

    NOT_DEPLOYED = "not_deployed"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"
    REJECTED = "rejected"
    RETIRED = "retired"


_DEPLOYMENT_ORDER: tuple[DeploymentStage, ...] = (
    DeploymentStage.NOT_DEPLOYED,
    DeploymentStage.SHADOW,
    DeploymentStage.CANARY,
    DeploymentStage.PRODUCTION,
)


class GateOutcome(str, Enum):
    """Per-gate result. Pinned string values."""

    PASSED = "passed"
    FAILED = "failed"


_DEFAULT_MIN_DECISIONS = 10_000
_DEFAULT_MIN_CLASS_BALANCE = 0.05
_DEFAULT_MIN_RETRAIN_INTERVAL = timedelta(days=14)
_DEFAULT_ACCURACY_FLOOR = 0.95
_DEFAULT_LATENCY_SPEEDUP_FLOOR = 10.0


@dataclass(frozen=True)
class DistillationPolicy:
    """Operator-tunable distillation policy.

    `min_decisions` is the absolute minimum decision count to
    consider triggering distillation; `min_class_balance` is the
    minimum fraction each decision class must represent
    (BUY/SELL/HOLD); `min_retrain_interval` prevents thrashing
    (don't retrain more than every 14 days by default);
    `accuracy_floor` is the student-vs-teacher agreement gate;
    `latency_speedup_floor` is the latency improvement gate.
    """

    min_decisions: int = _DEFAULT_MIN_DECISIONS
    min_class_balance: float = _DEFAULT_MIN_CLASS_BALANCE
    min_retrain_interval: timedelta = _DEFAULT_MIN_RETRAIN_INTERVAL
    accuracy_floor: float = _DEFAULT_ACCURACY_FLOOR
    latency_speedup_floor: float = _DEFAULT_LATENCY_SPEEDUP_FLOOR

    def __post_init__(self) -> None:
        if self.min_decisions <= 0:
            raise ValueError("min_decisions must be positive")
        if not 0.0 < self.min_class_balance < 1.0:
            raise ValueError(f"min_class_balance {self.min_class_balance} must be in (0, 1)")
        if self.min_retrain_interval <= timedelta(0):
            raise ValueError("min_retrain_interval must be positive")
        if not 0.0 < self.accuracy_floor <= 1.0:
            raise ValueError(f"accuracy_floor {self.accuracy_floor} must be in (0, 1]")
        if self.latency_speedup_floor <= 1.0:
            raise ValueError(
                f"latency_speedup_floor {self.latency_speedup_floor} "
                f"must be > 1.0 (no point distilling without speedup)"
            )


DEFAULT_POLICY = DistillationPolicy()


@dataclass(frozen=True)
class DecisionCohort:
    """Summary of the teacher's decision history."""

    total_decisions: int
    buy_count: int
    sell_count: int
    hold_count: int
    earliest_decision_at: datetime
    latest_decision_at: datetime

    def __post_init__(self) -> None:
        if self.total_decisions < 0:
            raise ValueError("total_decisions must be non-negative")
        if self.buy_count < 0 or self.sell_count < 0 or self.hold_count < 0:
            raise ValueError("decision counts must be non-negative")
        observed = self.buy_count + self.sell_count + self.hold_count
        if observed != self.total_decisions:
            raise ValueError(f"buy+sell+hold ({observed}) != total ({self.total_decisions})")
        if self.earliest_decision_at.tzinfo is None:
            raise ValueError("earliest_decision_at must be timezone-aware")
        if self.latest_decision_at.tzinfo is None:
            raise ValueError("latest_decision_at must be timezone-aware")
        if self.latest_decision_at < self.earliest_decision_at:
            raise ValueError("latest_decision_at must be >= earliest_decision_at")

    def class_balance(self) -> float:
        """Return the smallest class fraction across BUY/SELL/HOLD.

        0.0 if no decisions yet. Used by the diversity gate.
        """

        if self.total_decisions == 0:
            return 0.0
        return min(
            self.buy_count / self.total_decisions,
            self.sell_count / self.total_decisions,
            self.hold_count / self.total_decisions,
        )


def trigger_decision(
    cohort: DecisionCohort,
    *,
    last_retrain_at: datetime | None,
    now: datetime,
    policy: DistillationPolicy = DEFAULT_POLICY,
) -> DistillationDecision:
    """Decide whether to launch a distillation run.

    Returns one of the DistillationDecision values; the operator's
    cron consumes this directly.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if last_retrain_at is not None and last_retrain_at.tzinfo is None:
        raise ValueError("last_retrain_at must be timezone-aware when set")

    if cohort.total_decisions < policy.min_decisions:
        return DistillationDecision.SKIP_INSUFFICIENT_DECISIONS

    if cohort.class_balance() < policy.min_class_balance:
        return DistillationDecision.SKIP_INSUFFICIENT_DIVERSITY

    if last_retrain_at is not None and (now - last_retrain_at) < policy.min_retrain_interval:
        return DistillationDecision.SKIP_RECENT_RETRAIN

    return DistillationDecision.LAUNCH


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome (accuracy or latency)."""

    name: str
    outcome: GateOutcome
    measured_value: float
    threshold: float
    message: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")


@dataclass(frozen=True)
class StudentValidation:
    """Validation report for a trained student model.

    `agreement_rate` is the fraction of holdout decisions where
    the student matched the teacher. `latency_p99_ms` is the
    student's 99th-percentile inference latency; `teacher_latency_p99_ms`
    is the teacher's. The latency speedup is the ratio.
    """

    student_id: str
    agreement_rate: float
    latency_p99_ms: float
    teacher_latency_p99_ms: float
    holdout_size: int
    validated_at: datetime

    def __post_init__(self) -> None:
        if not self.student_id or not self.student_id.strip():
            raise ValueError("student_id must be non-empty")
        if not 0.0 <= self.agreement_rate <= 1.0:
            raise ValueError(f"agreement_rate {self.agreement_rate} must be in [0, 1]")
        if self.latency_p99_ms <= 0:
            raise ValueError("latency_p99_ms must be positive")
        if self.teacher_latency_p99_ms <= 0:
            raise ValueError("teacher_latency_p99_ms must be positive")
        if self.holdout_size <= 0:
            raise ValueError("holdout_size must be positive")
        if self.validated_at.tzinfo is None:
            raise ValueError("validated_at must be timezone-aware")

    @property
    def latency_speedup(self) -> float:
        """Teacher latency / student latency. >1.0 means student is faster."""

        return self.teacher_latency_p99_ms / self.latency_p99_ms


def evaluate_gates(
    validation: StudentValidation,
    *,
    policy: DistillationPolicy = DEFAULT_POLICY,
) -> tuple[GateResult, ...]:
    """Run the accuracy + latency gates, returning per-gate results."""

    accuracy_passed = validation.agreement_rate >= policy.accuracy_floor
    accuracy = GateResult(
        name="accuracy",
        outcome=GateOutcome.PASSED if accuracy_passed else GateOutcome.FAILED,
        measured_value=validation.agreement_rate,
        threshold=policy.accuracy_floor,
        message=(
            f"agreement {validation.agreement_rate:.2%} "
            f"{'>=' if accuracy_passed else '<'} {policy.accuracy_floor:.2%}"
        ),
    )

    speedup = validation.latency_speedup
    latency_passed = speedup >= policy.latency_speedup_floor
    latency = GateResult(
        name="latency_speedup",
        outcome=GateOutcome.PASSED if latency_passed else GateOutcome.FAILED,
        measured_value=speedup,
        threshold=policy.latency_speedup_floor,
        message=(
            f"speedup {speedup:.1f}x "
            f"{'>=' if latency_passed else '<'} "
            f"{policy.latency_speedup_floor:.1f}x"
        ),
    )

    return (accuracy, latency)


def all_gates_passed(gates: Iterable[GateResult]) -> bool:
    return all(g.outcome is GateOutcome.PASSED for g in gates)


@dataclass(frozen=True)
class DeploymentTransition:
    """Audit row for a deployment-stage transition."""

    from_stage: DeploymentStage
    to_stage: DeploymentStage
    decided_at: datetime
    notes: str = ""

    def __post_init__(self) -> None:
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")


@dataclass(frozen=True)
class StudentDeployment:
    """One student model's deployment state.

    Operations (`promote`, `reject`, `retire`) return a new state.
    """

    student_id: str
    current_stage: DeploymentStage
    transitions: tuple[DeploymentTransition, ...]

    def __post_init__(self) -> None:
        if not self.student_id or not self.student_id.strip():
            raise ValueError("student_id must be non-empty")


class DeploymentOrderError(Exception):
    """Raised when promote() skips a stage."""

    def __init__(self, from_stage: DeploymentStage, to_stage: DeploymentStage) -> None:
        super().__init__(f"cannot promote from {from_stage.value} to {to_stage.value}")
        self.from_stage = from_stage
        self.to_stage = to_stage


def start_deployment(
    *,
    student_id: str,
    now: datetime,
) -> StudentDeployment:
    """Create a fresh deployment at NOT_DEPLOYED stage."""

    if not student_id or not student_id.strip():
        raise ValueError("student_id must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return StudentDeployment(
        student_id=student_id,
        current_stage=DeploymentStage.NOT_DEPLOYED,
        transitions=(),
    )


def promote(
    deployment: StudentDeployment,
    to_stage: DeploymentStage,
    *,
    now: datetime,
    notes: str = "",
) -> StudentDeployment:
    """Promote the student to the next deployment stage.

    Forward moves must be one step along NOT_DEPLOYED → SHADOW →
    CANARY → PRODUCTION. Skipping raises `DeploymentOrderError`.
    Cannot promote from REJECTED or RETIRED (terminal).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if deployment.current_stage in (DeploymentStage.REJECTED, DeploymentStage.RETIRED):
        raise ValueError(f"cannot promote a {deployment.current_stage.value} deployment")
    if to_stage in (DeploymentStage.REJECTED, DeploymentStage.RETIRED):
        raise ValueError("use reject() or retire() for terminal states")
    if to_stage is deployment.current_stage:
        raise ValueError(f"already at {to_stage.value}")

    cur_idx = _DEPLOYMENT_ORDER.index(deployment.current_stage)
    new_idx = _DEPLOYMENT_ORDER.index(to_stage)
    if new_idx != cur_idx + 1:
        raise DeploymentOrderError(deployment.current_stage, to_stage)

    transition = DeploymentTransition(
        from_stage=deployment.current_stage,
        to_stage=to_stage,
        decided_at=now,
        notes=notes,
    )
    return StudentDeployment(
        student_id=deployment.student_id,
        current_stage=to_stage,
        transitions=deployment.transitions + (transition,),
    )


def reject(
    deployment: StudentDeployment,
    *,
    now: datetime,
    notes: str = "",
) -> StudentDeployment:
    """Mark the deployment REJECTED (gate failed)."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if deployment.current_stage is DeploymentStage.REJECTED:
        raise ValueError("already rejected")
    transition = DeploymentTransition(
        from_stage=deployment.current_stage,
        to_stage=DeploymentStage.REJECTED,
        decided_at=now,
        notes=notes,
    )
    return StudentDeployment(
        student_id=deployment.student_id,
        current_stage=DeploymentStage.REJECTED,
        transitions=deployment.transitions + (transition,),
    )


def retire(
    deployment: StudentDeployment,
    *,
    now: datetime,
    notes: str = "",
) -> StudentDeployment:
    """Mark a previously-deployed student RETIRED (replaced by newer)."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if deployment.current_stage is DeploymentStage.RETIRED:
        raise ValueError("already retired")
    transition = DeploymentTransition(
        from_stage=deployment.current_stage,
        to_stage=DeploymentStage.RETIRED,
        decided_at=now,
        notes=notes,
    )
    return StudentDeployment(
        student_id=deployment.student_id,
        current_stage=DeploymentStage.RETIRED,
        transitions=deployment.transitions + (transition,),
    )


_STAGE_EMOJI: dict[DeploymentStage, str] = {
    DeploymentStage.NOT_DEPLOYED: "⬜",
    DeploymentStage.SHADOW: "🌑",
    DeploymentStage.CANARY: "🐤",
    DeploymentStage.PRODUCTION: "🚀",
    DeploymentStage.REJECTED: "❌",
    DeploymentStage.RETIRED: "🪦",
}


_GATE_EMOJI: dict[GateOutcome, str] = {
    GateOutcome.PASSED: "✅",
    GateOutcome.FAILED: "❌",
}


def render_validation(
    validation: StudentValidation,
    gates: Iterable[GateResult],
) -> str:
    """Format the validation + gates for ops display.

    No-secret-leak: never includes raw decision text, training
    data, or model weights — only the validation summary numbers.
    """

    gate_list = list(gates)
    lines = [
        f"📊 Distillation validation — student {validation.student_id}",
        f"  agreement: {validation.agreement_rate:.2%} on "
        f"{validation.holdout_size} holdout decisions",
        f"  latency p99: student {validation.latency_p99_ms:.1f}ms vs "
        f"teacher {validation.teacher_latency_p99_ms:.1f}ms "
        f"({validation.latency_speedup:.1f}x)",
        f"  validated: {validation.validated_at.isoformat()}",
    ]
    for gate in gate_list:
        emoji = _GATE_EMOJI[gate.outcome]
        lines.append(f"  {emoji} {gate.name}: {gate.message}")

    if all_gates_passed(gate_list):
        lines.append("  ✅ All gates passed — eligible for SHADOW deployment")
    else:
        lines.append("  ❌ Gate failure — student should be rejected")
    return "\n".join(lines)


def render_deployment(deployment: StudentDeployment) -> str:
    """Format a deployment state for ops display."""

    emoji = _STAGE_EMOJI[deployment.current_stage]
    lines = [
        f"{emoji} student {deployment.student_id} — {deployment.current_stage.value}",
        f"  transitions: {len(deployment.transitions)}",
    ]
    if deployment.transitions:
        last = deployment.transitions[-1]
        lines.append(f"  last: {last.from_stage.value} → {last.to_stage.value}")
        if last.notes:
            lines.append(f"  notes: {last.notes}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "DecisionCohort",
    "DeploymentOrderError",
    "DeploymentStage",
    "DeploymentTransition",
    "DistillationDecision",
    "DistillationPolicy",
    "GateOutcome",
    "GateResult",
    "StudentDeployment",
    "StudentValidation",
    "all_gates_passed",
    "evaluate_gates",
    "promote",
    "reject",
    "render_deployment",
    "render_validation",
    "retire",
    "start_deployment",
    "trigger_decision",
]
