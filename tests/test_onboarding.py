"""Tests for `halal_trader.web.onboarding` (Wave 3.E).

Covers: step ordering enforcement, optional step skip semantics,
SLA breach flag, progress percentage, render no-leak contract,
state immutability + replay-ability.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.onboarding import (
    DEFAULT_POLICY,
    OnboardingPolicy,
    OnboardingState,
    OnboardingStep,
    StepNotSkippableError,
    StepOutOfOrderError,
    StepRecord,
    StepStatus,
    complete_step,
    flag_sla_breach,
    progress_pct,
    render_onboarding_state,
    skip_step,
    start_onboarding,
    time_to_first_trade,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_onboarding_step_string_values_pinned() -> None:
    assert OnboardingStep.SIGNED_IN.value == "signed_in"
    assert OnboardingStep.BROKER_CHOSEN.value == "broker_chosen"
    assert OnboardingStep.BROKER_KEYS_STORED.value == "broker_keys_stored"
    assert OnboardingStep.STRATEGY_CHOSEN.value == "strategy_chosen"
    assert OnboardingStep.MODEL_CHOSEN.value == "model_chosen"
    assert OnboardingStep.FIRST_CYCLE_RUN.value == "first_cycle_run"


def test_step_status_string_values_pinned() -> None:
    assert StepStatus.PENDING.value == "pending"
    assert StepStatus.COMPLETED.value == "completed"
    assert StepStatus.SKIPPED.value == "skipped"


# --------------------------- OnboardingPolicy --------------------------------


def test_default_policy_is_5_minutes() -> None:
    assert DEFAULT_POLICY.sla_threshold == timedelta(minutes=5)


def test_policy_rejects_zero_sla() -> None:
    with pytest.raises(ValueError, match="sla_threshold"):
        OnboardingPolicy(sla_threshold=timedelta(0))


def test_policy_rejects_negative_sla() -> None:
    with pytest.raises(ValueError, match="sla_threshold"):
        OnboardingPolicy(sla_threshold=timedelta(seconds=-1))


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.sla_threshold = timedelta(hours=1)  # type: ignore[misc]


# --------------------------- StepRecord --------------------------------------


def test_step_record_rejects_naive_decided_at() -> None:
    with pytest.raises(ValueError, match="decided_at"):
        StepRecord(
            step=OnboardingStep.SIGNED_IN,
            status=StepStatus.COMPLETED,
            decided_at=datetime(2026, 5, 1),
        )


def test_step_record_rejects_pending_status() -> None:
    """A StepRecord is an audit row — pending isn't a decision."""

    with pytest.raises(ValueError, match="non-pending"):
        StepRecord(
            step=OnboardingStep.SIGNED_IN,
            status=StepStatus.PENDING,
            decided_at=T0,
        )


def test_step_record_is_frozen() -> None:
    record = StepRecord(
        step=OnboardingStep.SIGNED_IN,
        status=StepStatus.COMPLETED,
        decided_at=T0,
    )
    with pytest.raises(FrozenInstanceError):
        record.status = StepStatus.SKIPPED  # type: ignore[misc]


# --------------------------- OnboardingState ---------------------------------


def test_onboarding_state_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        OnboardingState(user_id="", started_at=T0, records=())


def test_onboarding_state_rejects_naive_started_at() -> None:
    with pytest.raises(ValueError, match="started_at"):
        OnboardingState(user_id="u1", started_at=datetime(2026, 5, 1), records=())


def test_onboarding_state_is_frozen() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    with pytest.raises(FrozenInstanceError):
        state.user_id = "other"  # type: ignore[misc]


def test_status_of_pending_for_unrecorded() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    assert state.status_of(OnboardingStep.BROKER_CHOSEN) is StepStatus.PENDING


def test_status_of_completed_after_complete() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    assert state.status_of(OnboardingStep.BROKER_CHOSEN) is StepStatus.COMPLETED


# --------------------------- start_onboarding --------------------------------


def test_start_onboarding_marks_signed_in() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    assert state.status_of(OnboardingStep.SIGNED_IN) is StepStatus.COMPLETED
    assert state.started_at == T0


def test_start_onboarding_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        start_onboarding(user_id="", now=T0)


def test_start_onboarding_rejects_whitespace_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        start_onboarding(user_id="   ", now=T0)


def test_start_onboarding_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        start_onboarding(user_id="u1", now=datetime(2026, 5, 1))


def test_start_onboarding_initial_progress_is_one_sixth() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    assert progress_pct(state) == pytest.approx(1 / 6)


def test_start_onboarding_next_step_is_broker() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    assert state.next_step() is OnboardingStep.BROKER_CHOSEN


# --------------------------- complete_step -----------------------------------


def test_complete_step_advances_state() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    assert state.status_of(OnboardingStep.BROKER_CHOSEN) is StepStatus.COMPLETED


def test_complete_step_in_full_order() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    for step in [
        OnboardingStep.BROKER_CHOSEN,
        OnboardingStep.BROKER_KEYS_STORED,
        OnboardingStep.STRATEGY_CHOSEN,
        OnboardingStep.MODEL_CHOSEN,
        OnboardingStep.FIRST_CYCLE_RUN,
    ]:
        state = complete_step(state, step, now=T0)
    assert state.is_complete


def test_complete_step_out_of_order_rejected() -> None:
    """Pin: cannot complete STRATEGY_CHOSEN before BROKER_CHOSEN."""

    state = start_onboarding(user_id="u1", now=T0)
    with pytest.raises(StepOutOfOrderError) as exc_info:
        complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)
    assert exc_info.value.step is OnboardingStep.STRATEGY_CHOSEN
    assert exc_info.value.missing is OnboardingStep.BROKER_CHOSEN


def test_complete_step_skipping_optional_allowed() -> None:
    """Pin: can complete STRATEGY_CHOSEN with BROKER_KEYS_STORED still pending."""

    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    # Don't complete BROKER_KEYS_STORED (optional)
    state = complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)
    assert state.status_of(OnboardingStep.STRATEGY_CHOSEN) is StepStatus.COMPLETED
    assert state.status_of(OnboardingStep.BROKER_KEYS_STORED) is StepStatus.PENDING


def test_complete_step_already_decided_rejected() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    with pytest.raises(ValueError, match="already decided"):
        complete_step(state, OnboardingStep.SIGNED_IN, now=T0)


def test_complete_step_rejects_naive_now() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    with pytest.raises(ValueError, match="now"):
        complete_step(
            state,
            OnboardingStep.BROKER_CHOSEN,
            now=datetime(2026, 5, 1),
        )


def test_complete_step_returns_new_state_not_mutates() -> None:
    """Pin: state operations are immutable — return new state."""

    original = start_onboarding(user_id="u1", now=T0)
    new_state = complete_step(original, OnboardingStep.BROKER_CHOSEN, now=T0)
    # Original unchanged
    assert original.status_of(OnboardingStep.BROKER_CHOSEN) is StepStatus.PENDING
    # New state has the change
    assert new_state.status_of(OnboardingStep.BROKER_CHOSEN) is StepStatus.COMPLETED


def test_complete_step_records_decision_timestamp() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    later = T0 + timedelta(minutes=2)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=later)
    found = next(r for r in state.records if r.step is OnboardingStep.BROKER_CHOSEN)
    assert found.decided_at == later


# --------------------------- skip_step ---------------------------------------


def test_skip_step_marks_skipped() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    assert state.status_of(OnboardingStep.BROKER_KEYS_STORED) is StepStatus.SKIPPED


def test_skip_step_rejects_required_step() -> None:
    """Pin: SIGNED_IN / BROKER_CHOSEN / STRATEGY_CHOSEN / FIRST_CYCLE_RUN cannot be skipped."""

    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)

    with pytest.raises(StepNotSkippableError):
        skip_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)


def test_skip_step_rejects_signed_in() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    with pytest.raises(StepNotSkippableError):
        skip_step(state, OnboardingStep.SIGNED_IN, now=T0)


def test_skip_step_rejects_first_cycle_run() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    with pytest.raises(StepNotSkippableError):
        skip_step(state, OnboardingStep.FIRST_CYCLE_RUN, now=T0)


def test_skip_step_model_chosen_is_skippable() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.MODEL_CHOSEN, now=T0)
    assert state.status_of(OnboardingStep.MODEL_CHOSEN) is StepStatus.SKIPPED


def test_skip_step_already_decided_rejected() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    with pytest.raises(ValueError, match="already decided"):
        skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)


def test_skip_step_rejects_naive_now() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    with pytest.raises(ValueError, match="now"):
        skip_step(
            state,
            OnboardingStep.BROKER_KEYS_STORED,
            now=datetime(2026, 5, 1),
        )


# --------------------------- next_step ---------------------------------------


def test_next_step_returns_first_pending_in_order() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    assert state.next_step() is OnboardingStep.BROKER_KEYS_STORED


def test_next_step_skips_decided_optional() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    assert state.next_step() is OnboardingStep.STRATEGY_CHOSEN


def test_next_step_returns_none_when_complete() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    for step in [
        OnboardingStep.BROKER_CHOSEN,
        OnboardingStep.BROKER_KEYS_STORED,
        OnboardingStep.STRATEGY_CHOSEN,
        OnboardingStep.MODEL_CHOSEN,
        OnboardingStep.FIRST_CYCLE_RUN,
    ]:
        state = complete_step(state, step, now=T0)
    assert state.next_step() is None


# --------------------------- is_complete -------------------------------------


def test_is_complete_false_when_first_cycle_pending() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    assert state.is_complete is False


def test_is_complete_true_when_first_cycle_completed() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    state = complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.MODEL_CHOSEN, now=T0)
    state = complete_step(state, OnboardingStep.FIRST_CYCLE_RUN, now=T0)
    assert state.is_complete is True


# --------------------------- time_to_first_trade -----------------------------


def test_time_to_first_trade_none_when_incomplete() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    assert time_to_first_trade(state) is None


def test_time_to_first_trade_returns_elapsed() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    state = complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.MODEL_CHOSEN, now=T0)
    finished = T0 + timedelta(minutes=4)
    state = complete_step(state, OnboardingStep.FIRST_CYCLE_RUN, now=finished)
    assert time_to_first_trade(state) == timedelta(minutes=4)


# --------------------------- flag_sla_breach ---------------------------------


def test_sla_breach_false_when_under_5_min() -> None:
    state = _completed_state(elapsed=timedelta(minutes=4))
    assert flag_sla_breach(state) is False


def test_sla_breach_false_at_exactly_5_min_boundary() -> None:
    """Pin: 5min exactly is NOT a breach (>, not >=)."""

    state = _completed_state(elapsed=timedelta(minutes=5))
    assert flag_sla_breach(state) is False


def test_sla_breach_true_when_over_5_min() -> None:
    state = _completed_state(elapsed=timedelta(minutes=5, seconds=1))
    assert flag_sla_breach(state) is True


def test_sla_breach_false_when_incomplete() -> None:
    """Pin: in-progress flow doesn't breach yet."""

    state = start_onboarding(user_id="u1", now=T0)
    assert flag_sla_breach(state) is False


def test_sla_breach_uses_custom_policy() -> None:
    state = _completed_state(elapsed=timedelta(minutes=2))
    strict = OnboardingPolicy(sla_threshold=timedelta(minutes=1))
    assert flag_sla_breach(state, policy=strict) is True


def _completed_state(*, elapsed: timedelta) -> OnboardingState:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    state = complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.MODEL_CHOSEN, now=T0)
    return complete_step(state, OnboardingStep.FIRST_CYCLE_RUN, now=T0 + elapsed)


# --------------------------- progress_pct ------------------------------------


def test_progress_pct_initial() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    assert progress_pct(state) == pytest.approx(1 / 6)


def test_progress_pct_after_complete() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    assert progress_pct(state) == pytest.approx(2 / 6)


def test_progress_pct_after_skip() -> None:
    """Pin: skipped counts as decided for progress."""

    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    assert progress_pct(state) == pytest.approx(3 / 6)


def test_progress_pct_full() -> None:
    state = _completed_state(elapsed=timedelta(minutes=4))
    assert progress_pct(state) == 1.0


# --------------------------- render_onboarding_state -------------------------


def test_render_includes_user_id_and_progress() -> None:
    state = start_onboarding(user_id="alice", now=T0)
    out = render_onboarding_state(state)
    assert "alice" in out
    assert "17%" in out  # 1/6 ≈ 17%


def test_render_shows_step_emoji() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
    out = render_onboarding_state(state)
    assert "✅" in out  # for completed
    assert "⏭️" in out  # for skipped
    assert "⬜" in out  # for pending


def test_render_shows_next_step_hint() -> None:
    state = start_onboarding(user_id="u1", now=T0)
    out = render_onboarding_state(state)
    assert "next:" in out
    assert "Choose a broker" in out


def test_render_shows_completion_marker_when_done() -> None:
    state = _completed_state(elapsed=timedelta(minutes=3))
    out = render_onboarding_state(state)
    assert "⭐" in out
    assert "180s" in out


def test_render_no_secret_leak() -> None:
    """Pin: render never includes broker keys / Stripe IDs / session tokens."""

    state = start_onboarding(user_id="u1", now=T0)
    out = render_onboarding_state(state)
    assert "api_key" not in out.lower()
    assert "secret" not in out.lower().replace("api_secret", "")
    assert "cus_" not in out.lower()
    assert "sub_" not in out.lower()
    assert "bearer" not in out.lower()
    assert "session_" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_happy_path_under_sla() -> None:
    state = start_onboarding(user_id="alice", now=T0)
    t = T0
    for step in [
        OnboardingStep.BROKER_CHOSEN,
        OnboardingStep.BROKER_KEYS_STORED,
        OnboardingStep.STRATEGY_CHOSEN,
        OnboardingStep.MODEL_CHOSEN,
        OnboardingStep.FIRST_CYCLE_RUN,
    ]:
        t += timedelta(seconds=30)
        state = complete_step(state, step, now=t)
    assert state.is_complete
    elapsed = time_to_first_trade(state)
    assert elapsed == timedelta(seconds=150)
    assert flag_sla_breach(state) is False


def test_e2e_skip_optionals() -> None:
    """Real-world: user with no broker keys / no LLM preference."""

    state = start_onboarding(user_id="bob", now=T0)
    t = T0 + timedelta(seconds=30)
    state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=t)
    state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=t)
    t += timedelta(seconds=30)
    state = complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=t)
    state = skip_step(state, OnboardingStep.MODEL_CHOSEN, now=t)
    t += timedelta(seconds=30)
    state = complete_step(state, OnboardingStep.FIRST_CYCLE_RUN, now=t)
    assert state.is_complete
    assert state.status_of(OnboardingStep.BROKER_KEYS_STORED) is StepStatus.SKIPPED
    assert state.status_of(OnboardingStep.MODEL_CHOSEN) is StepStatus.SKIPPED
    # Audit trail preserved: 2 skipped steps
    skipped = [r for r in state.records if r.status is StepStatus.SKIPPED]
    assert len(skipped) == 2


def test_e2e_slow_user_breaches_sla() -> None:
    state = _completed_state(elapsed=timedelta(minutes=8))
    assert flag_sla_breach(state) is True


def test_e2e_replay_consistency() -> None:
    """Pin: applying the same operations produces the same state."""

    def build() -> OnboardingState:
        state = start_onboarding(user_id="u1", now=T0)
        state = complete_step(state, OnboardingStep.BROKER_CHOSEN, now=T0)
        state = skip_step(state, OnboardingStep.BROKER_KEYS_STORED, now=T0)
        state = complete_step(state, OnboardingStep.STRATEGY_CHOSEN, now=T0)
        return state

    a = build()
    b = build()
    assert a == b
