"""Tests for `halal_trader.ml.distillation` (Wave 6.I).

Covers: trigger decision (insufficient decisions / diversity / recent
retrain / launch), accuracy + latency gates with boundary pins,
deployment state machine, immutability + replay-ability.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.ml.distillation import (
    DEFAULT_POLICY,
    DecisionCohort,
    DeploymentOrderError,
    DeploymentStage,
    DistillationDecision,
    DistillationPolicy,
    GateOutcome,
    StudentDeployment,
    StudentValidation,
    all_gates_passed,
    evaluate_gates,
    promote,
    reject,
    render_deployment,
    render_validation,
    retire,
    start_deployment,
    trigger_decision,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_distillation_decision_string_values_pinned() -> None:
    assert DistillationDecision.SKIP_INSUFFICIENT_DECISIONS.value == "skip_insufficient_decisions"
    assert DistillationDecision.SKIP_INSUFFICIENT_DIVERSITY.value == "skip_insufficient_diversity"
    assert DistillationDecision.SKIP_RECENT_RETRAIN.value == "skip_recent_retrain"
    assert DistillationDecision.LAUNCH.value == "launch"


def test_deployment_stage_string_values_pinned() -> None:
    assert DeploymentStage.NOT_DEPLOYED.value == "not_deployed"
    assert DeploymentStage.SHADOW.value == "shadow"
    assert DeploymentStage.CANARY.value == "canary"
    assert DeploymentStage.PRODUCTION.value == "production"
    assert DeploymentStage.REJECTED.value == "rejected"
    assert DeploymentStage.RETIRED.value == "retired"


def test_gate_outcome_string_values_pinned() -> None:
    assert GateOutcome.PASSED.value == "passed"
    assert GateOutcome.FAILED.value == "failed"


# --------------------------- DistillationPolicy ------------------------------


def test_default_policy_pins_roadmap_thresholds() -> None:
    """Pin: roadmap pins 10k decisions + 95% accuracy floor + 100x speedup
    target (we use 10x as the gate floor, with 100x as the aspiration)."""

    assert DEFAULT_POLICY.min_decisions == 10_000
    assert DEFAULT_POLICY.accuracy_floor == 0.95
    assert DEFAULT_POLICY.latency_speedup_floor == 10.0


def test_policy_rejects_zero_min_decisions() -> None:
    with pytest.raises(ValueError, match="min_decisions"):
        DistillationPolicy(min_decisions=0)


def test_policy_rejects_negative_min_decisions() -> None:
    with pytest.raises(ValueError, match="min_decisions"):
        DistillationPolicy(min_decisions=-1)


def test_policy_rejects_class_balance_at_zero() -> None:
    with pytest.raises(ValueError, match="min_class_balance"):
        DistillationPolicy(min_class_balance=0.0)


def test_policy_rejects_class_balance_at_one() -> None:
    with pytest.raises(ValueError, match="min_class_balance"):
        DistillationPolicy(min_class_balance=1.0)


def test_policy_rejects_zero_retrain_interval() -> None:
    with pytest.raises(ValueError, match="min_retrain_interval"):
        DistillationPolicy(min_retrain_interval=timedelta(0))


def test_policy_rejects_accuracy_floor_at_zero() -> None:
    with pytest.raises(ValueError, match="accuracy_floor"):
        DistillationPolicy(accuracy_floor=0.0)


def test_policy_accepts_accuracy_floor_at_one() -> None:
    p = DistillationPolicy(accuracy_floor=1.0)
    assert p.accuracy_floor == 1.0


def test_policy_rejects_speedup_at_one() -> None:
    """Pin: speedup must be > 1.0 — distilling without speedup is pointless."""

    with pytest.raises(ValueError, match="latency_speedup_floor"):
        DistillationPolicy(latency_speedup_floor=1.0)


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.min_decisions = 5  # type: ignore[misc]


# --------------------------- DecisionCohort ----------------------------------


def _cohort(**overrides: object) -> DecisionCohort:
    base: dict[str, object] = {
        "total_decisions": 12_000,
        "buy_count": 4_000,
        "sell_count": 4_000,
        "hold_count": 4_000,
        "earliest_decision_at": T0 - timedelta(days=90),
        "latest_decision_at": T0 - timedelta(days=1),
    }
    base.update(overrides)
    return DecisionCohort(**base)  # type: ignore[arg-type]


def test_cohort_rejects_negative_total() -> None:
    with pytest.raises(ValueError, match="total_decisions"):
        _cohort(total_decisions=-1)


def test_cohort_rejects_negative_class_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _cohort(buy_count=-1, total_decisions=8_000, sell_count=4_000, hold_count=4_000)


def test_cohort_rejects_inconsistent_sum() -> None:
    """buy + sell + hold must equal total."""

    with pytest.raises(ValueError, match="total"):
        _cohort(buy_count=1_000, sell_count=1_000, hold_count=1_000)


def test_cohort_rejects_naive_earliest() -> None:
    with pytest.raises(ValueError, match="earliest_decision_at"):
        _cohort(earliest_decision_at=datetime(2026, 5, 1))


def test_cohort_rejects_latest_before_earliest() -> None:
    with pytest.raises(ValueError, match="latest_decision_at"):
        _cohort(latest_decision_at=T0 - timedelta(days=200))


def test_cohort_class_balance_basic() -> None:
    cohort = _cohort(total_decisions=10_000, buy_count=2_000, sell_count=3_000, hold_count=5_000)
    assert cohort.class_balance() == 0.2  # smallest is BUY at 20%


def test_cohort_class_balance_zero_when_no_decisions() -> None:
    cohort = _cohort(total_decisions=0, buy_count=0, sell_count=0, hold_count=0)
    assert cohort.class_balance() == 0.0


def test_cohort_is_frozen() -> None:
    cohort = _cohort()
    with pytest.raises(FrozenInstanceError):
        cohort.total_decisions = 99  # type: ignore[misc]


# --------------------------- trigger_decision --------------------------------


def test_trigger_skip_insufficient_decisions() -> None:
    cohort = _cohort(total_decisions=5_000, buy_count=1_500, sell_count=1_500, hold_count=2_000)
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.SKIP_INSUFFICIENT_DECISIONS


def test_trigger_skip_at_boundary_below_10k() -> None:
    """Pin: 9999 decisions is insufficient."""

    cohort = _cohort(total_decisions=9_999, buy_count=3_333, sell_count=3_333, hold_count=3_333)
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.SKIP_INSUFFICIENT_DECISIONS


def test_trigger_launches_at_10k_boundary() -> None:
    """Pin: exactly 10000 decisions hits the inclusive boundary."""

    cohort = _cohort(total_decisions=10_000, buy_count=3_333, sell_count=3_333, hold_count=3_334)
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.LAUNCH


def test_trigger_skip_insufficient_diversity() -> None:
    """Pin: 9.5k HOLD + 0.25k BUY + 0.25k SELL is too imbalanced.

    BUY = 0.025 < 0.05 default class-balance floor.
    """

    cohort = _cohort(total_decisions=10_000, buy_count=250, sell_count=250, hold_count=9_500)
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.SKIP_INSUFFICIENT_DIVERSITY


def test_trigger_class_balance_at_5_pct_boundary_passes() -> None:
    """Pin: exactly 5% class fraction hits the inclusive boundary."""

    cohort = _cohort(total_decisions=10_000, buy_count=500, sell_count=500, hold_count=9_000)
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.LAUNCH


def test_trigger_skip_recent_retrain() -> None:
    cohort = _cohort()
    last_retrain = T0 - timedelta(days=7)  # within default 14-day cooldown
    decision = trigger_decision(cohort, last_retrain_at=last_retrain, now=T0)
    assert decision is DistillationDecision.SKIP_RECENT_RETRAIN


def test_trigger_at_14_day_boundary_launches() -> None:
    """Pin: exactly 14 days since last retrain is OK to retrigger."""

    cohort = _cohort()
    last_retrain = T0 - timedelta(days=14)
    decision = trigger_decision(cohort, last_retrain_at=last_retrain, now=T0)
    assert decision is DistillationDecision.LAUNCH


def test_trigger_no_prior_retrain_launches() -> None:
    cohort = _cohort()
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.LAUNCH


def test_trigger_rejects_naive_now() -> None:
    cohort = _cohort()
    with pytest.raises(ValueError, match="now"):
        trigger_decision(
            cohort,
            last_retrain_at=None,
            now=datetime(2026, 5, 1),
        )


def test_trigger_rejects_naive_last_retrain() -> None:
    cohort = _cohort()
    with pytest.raises(ValueError, match="last_retrain_at"):
        trigger_decision(
            cohort,
            last_retrain_at=datetime(2026, 4, 1),
            now=T0,
        )


def test_trigger_priority_decisions_before_diversity() -> None:
    """Pin: insufficient decisions check fires before diversity check.

    A 100-decision cohort with bad diversity should report
    SKIP_INSUFFICIENT_DECISIONS (the bigger blocker), not diversity.
    """

    cohort = _cohort(total_decisions=100, buy_count=0, sell_count=0, hold_count=100)
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.SKIP_INSUFFICIENT_DECISIONS


# --------------------------- StudentValidation -------------------------------


def _validation(**overrides: object) -> StudentValidation:
    base: dict[str, object] = {
        "student_id": "student_v1",
        "agreement_rate": 0.97,
        "latency_p99_ms": 5.0,
        "teacher_latency_p99_ms": 100.0,
        "holdout_size": 2_000,
        "validated_at": T0,
    }
    base.update(overrides)
    return StudentValidation(**base)  # type: ignore[arg-type]


def test_validation_rejects_empty_student_id() -> None:
    with pytest.raises(ValueError, match="student_id"):
        _validation(student_id="")


def test_validation_rejects_agreement_above_one() -> None:
    with pytest.raises(ValueError, match="agreement_rate"):
        _validation(agreement_rate=1.01)


def test_validation_rejects_negative_agreement() -> None:
    with pytest.raises(ValueError, match="agreement_rate"):
        _validation(agreement_rate=-0.01)


def test_validation_rejects_zero_latency() -> None:
    with pytest.raises(ValueError, match="latency_p99_ms"):
        _validation(latency_p99_ms=0)


def test_validation_rejects_zero_teacher_latency() -> None:
    with pytest.raises(ValueError, match="teacher_latency_p99_ms"):
        _validation(teacher_latency_p99_ms=0)


def test_validation_rejects_zero_holdout() -> None:
    with pytest.raises(ValueError, match="holdout_size"):
        _validation(holdout_size=0)


def test_validation_rejects_naive_validated_at() -> None:
    with pytest.raises(ValueError, match="validated_at"):
        _validation(validated_at=datetime(2026, 5, 1))


def test_validation_speedup_property() -> None:
    val = _validation(latency_p99_ms=10.0, teacher_latency_p99_ms=200.0)
    assert val.latency_speedup == 20.0


def test_validation_is_frozen() -> None:
    val = _validation()
    with pytest.raises(FrozenInstanceError):
        val.student_id = "other"  # type: ignore[misc]


# --------------------------- evaluate_gates ----------------------------------


def test_gates_pass_for_good_student() -> None:
    val = _validation(agreement_rate=0.97, latency_p99_ms=5.0)
    gates = evaluate_gates(val)
    assert all(g.outcome is GateOutcome.PASSED for g in gates)


def test_accuracy_gate_at_95_pct_boundary_passes() -> None:
    """Pin: agreement exactly 0.95 hits the inclusive boundary."""

    val = _validation(agreement_rate=0.95)
    gates = evaluate_gates(val)
    accuracy = next(g for g in gates if g.name == "accuracy")
    assert accuracy.outcome is GateOutcome.PASSED


def test_accuracy_gate_below_95_pct_fails() -> None:
    val = _validation(agreement_rate=0.949)
    gates = evaluate_gates(val)
    accuracy = next(g for g in gates if g.name == "accuracy")
    assert accuracy.outcome is GateOutcome.FAILED


def test_latency_gate_at_10x_boundary_passes() -> None:
    """Pin: 10x speedup hits the inclusive boundary."""

    val = _validation(latency_p99_ms=10.0, teacher_latency_p99_ms=100.0)
    gates = evaluate_gates(val)
    latency = next(g for g in gates if g.name == "latency_speedup")
    assert latency.outcome is GateOutcome.PASSED


def test_latency_gate_below_10x_fails() -> None:
    val = _validation(latency_p99_ms=11.0, teacher_latency_p99_ms=100.0)
    gates = evaluate_gates(val)
    latency = next(g for g in gates if g.name == "latency_speedup")
    assert latency.outcome is GateOutcome.FAILED


def test_gate_carries_measured_value_and_threshold() -> None:
    val = _validation(agreement_rate=0.94)
    gates = evaluate_gates(val)
    accuracy = next(g for g in gates if g.name == "accuracy")
    assert accuracy.measured_value == 0.94
    assert accuracy.threshold == 0.95


def test_all_gates_passed_helper_true() -> None:
    val = _validation()
    assert all_gates_passed(evaluate_gates(val)) is True


def test_all_gates_passed_helper_false_on_any_failure() -> None:
    val = _validation(agreement_rate=0.5)
    assert all_gates_passed(evaluate_gates(val)) is False


def test_custom_policy_flows_through() -> None:
    """Strict policy: 99% accuracy + 50x speedup."""

    strict = DistillationPolicy(accuracy_floor=0.99, latency_speedup_floor=50.0)
    val = _validation(agreement_rate=0.97, latency_p99_ms=5.0)
    gates = evaluate_gates(val, policy=strict)
    # 97% < 99%, so accuracy should fail
    assert all_gates_passed(gates) is False


# --------------------------- StudentDeployment -------------------------------


def test_start_deployment_basic() -> None:
    d = start_deployment(student_id="s1", now=T0)
    assert d.current_stage is DeploymentStage.NOT_DEPLOYED
    assert d.transitions == ()


def test_start_deployment_rejects_empty_student_id() -> None:
    with pytest.raises(ValueError, match="student_id"):
        start_deployment(student_id="", now=T0)


def test_start_deployment_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        start_deployment(student_id="s1", now=datetime(2026, 5, 1))


def test_deployment_is_frozen() -> None:
    d = start_deployment(student_id="s1", now=T0)
    with pytest.raises(FrozenInstanceError):
        d.student_id = "other"  # type: ignore[misc]


# --------------------------- promote -----------------------------------------


def test_promote_one_step_forward() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0)
    assert d.current_stage is DeploymentStage.SHADOW


def test_promote_full_path_to_production() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0)
    d = promote(d, DeploymentStage.CANARY, now=T0)
    d = promote(d, DeploymentStage.PRODUCTION, now=T0)
    assert d.current_stage is DeploymentStage.PRODUCTION
    assert len(d.transitions) == 3


def test_promote_skip_rejected() -> None:
    """Pin: cannot promote NOT_DEPLOYED → CANARY."""

    d = start_deployment(student_id="s1", now=T0)
    with pytest.raises(DeploymentOrderError):
        promote(d, DeploymentStage.CANARY, now=T0)


def test_promote_to_rejected_via_function_only() -> None:
    """Pin: cannot promote() to REJECTED — must use reject()."""

    d = start_deployment(student_id="s1", now=T0)
    with pytest.raises(ValueError, match="reject"):
        promote(d, DeploymentStage.REJECTED, now=T0)


def test_promote_to_retired_via_function_only() -> None:
    d = start_deployment(student_id="s1", now=T0)
    with pytest.raises(ValueError, match="retire"):
        promote(d, DeploymentStage.RETIRED, now=T0)


def test_promote_already_at_stage_rejected() -> None:
    d = start_deployment(student_id="s1", now=T0)
    with pytest.raises(ValueError, match="already at"):
        promote(d, DeploymentStage.NOT_DEPLOYED, now=T0)


def test_promote_from_rejected_blocked() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = reject(d, now=T0)
    with pytest.raises(ValueError, match="rejected"):
        promote(d, DeploymentStage.SHADOW, now=T0)


def test_promote_from_retired_blocked() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0)
    d = retire(d, now=T0)
    with pytest.raises(ValueError, match="retired"):
        promote(d, DeploymentStage.SHADOW, now=T0)


def test_promote_returns_new_state() -> None:
    """Pin: state operations are immutable."""

    original = start_deployment(student_id="s1", now=T0)
    new_state = promote(original, DeploymentStage.SHADOW, now=T0)
    assert original.current_stage is DeploymentStage.NOT_DEPLOYED
    assert new_state.current_stage is DeploymentStage.SHADOW


def test_promote_naive_now_rejected() -> None:
    d = start_deployment(student_id="s1", now=T0)
    with pytest.raises(ValueError, match="now"):
        promote(d, DeploymentStage.SHADOW, now=datetime(2026, 5, 1))


# --------------------------- reject ------------------------------------------


def test_reject_from_any_stage() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0)
    d = reject(d, now=T0, notes="failed shadow drift check")
    assert d.current_stage is DeploymentStage.REJECTED


def test_reject_already_rejected() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = reject(d, now=T0)
    with pytest.raises(ValueError, match="already rejected"):
        reject(d, now=T0)


# --------------------------- retire ------------------------------------------


def test_retire_from_production() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0)
    d = promote(d, DeploymentStage.CANARY, now=T0)
    d = promote(d, DeploymentStage.PRODUCTION, now=T0)
    d = retire(d, now=T0, notes="replaced by v2")
    assert d.current_stage is DeploymentStage.RETIRED


def test_retire_already_retired() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0)
    d = retire(d, now=T0)
    with pytest.raises(ValueError, match="already retired"):
        retire(d, now=T0)


# --------------------------- render ------------------------------------------


def test_render_validation_all_passed() -> None:
    val = _validation()
    out = render_validation(val, evaluate_gates(val))
    assert "student_v1" in out
    assert "97" in out  # 97%
    assert "20.0x" in out  # 100/5
    assert "✅" in out
    assert "All gates passed" in out


def test_render_validation_failed_gate() -> None:
    val = _validation(agreement_rate=0.5)
    out = render_validation(val, evaluate_gates(val))
    assert "❌" in out
    assert "rejected" in out


def test_render_validation_no_secret_leak() -> None:
    """Pin: render never includes raw decision text / training data."""

    val = _validation()
    out = render_validation(val, evaluate_gates(val))
    assert "BUY" not in out  # no raw decisions
    assert "weights" not in out.lower()
    assert "training_data" not in out.lower()


def test_render_deployment_basic() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0, notes="low-traffic shadow")
    out = render_deployment(d)
    assert "s1" in out
    assert "shadow" in out
    assert "low-traffic shadow" in out
    assert "🌑" in out


def test_render_deployment_emoji_per_stage() -> None:
    d = start_deployment(student_id="s1", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0)
    d = promote(d, DeploymentStage.CANARY, now=T0)
    d = promote(d, DeploymentStage.PRODUCTION, now=T0)
    out = render_deployment(d)
    assert "🚀" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_distillation_lifecycle() -> None:
    """Full happy-path: cohort triggers → student passes gates →
    promoted SHADOW → CANARY → PRODUCTION."""

    # 1. Trigger
    cohort = _cohort()
    decision = trigger_decision(cohort, last_retrain_at=None, now=T0)
    assert decision is DistillationDecision.LAUNCH

    # 2. Validate trained student
    val = _validation(
        student_id="student_2026_05",
        agreement_rate=0.97,
        latency_p99_ms=3.0,
        teacher_latency_p99_ms=120.0,
    )
    gates = evaluate_gates(val)
    assert all_gates_passed(gates)

    # 3. Deploy through shadow → canary → prod
    d = start_deployment(student_id="student_2026_05", now=T0)
    d = promote(d, DeploymentStage.SHADOW, now=T0 + timedelta(days=1))
    d = promote(d, DeploymentStage.CANARY, now=T0 + timedelta(days=8))
    d = promote(d, DeploymentStage.PRODUCTION, now=T0 + timedelta(days=15))
    assert d.current_stage is DeploymentStage.PRODUCTION


def test_e2e_failed_student_rejected() -> None:
    val = _validation(agreement_rate=0.85, latency_p99_ms=5.0)
    gates = evaluate_gates(val)
    assert not all_gates_passed(gates)
    d = start_deployment(student_id="student_v1", now=T0)
    d = reject(d, now=T0, notes="agreement 85% < 95% floor")
    assert d.current_stage is DeploymentStage.REJECTED


def test_e2e_replay_consistency() -> None:
    def build() -> StudentDeployment:
        d = start_deployment(student_id="s1", now=T0)
        d = promote(d, DeploymentStage.SHADOW, now=T0)
        d = promote(d, DeploymentStage.CANARY, now=T0)
        return d

    a = build()
    b = build()
    assert a == b
