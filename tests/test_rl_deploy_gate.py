"""Tests for core/rl_deploy_gate.py — Round-5 Wave 9.G."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.core.rl_deploy_gate import (
    GatePolicy,
    GateRecord,
    GateStatus,
    RejectionReason,
    ShadowMetrics,
    evaluate,
    mark_promoted,
    mark_rejected_late,
    render_record,
)


def _metrics(
    candidate_sharpe: float = 5.0,
    baseline_sharpe: float = 1.0,
    candidate_max_drawdown_pct: float = 0.10,
    n_trades: int = 500,
    shadow_started_on: date = date(2026, 2, 1),
    shadow_last_active_on: date = date(2026, 5, 11),
) -> ShadowMetrics:
    return ShadowMetrics(
        candidate_sharpe=candidate_sharpe,
        baseline_sharpe=baseline_sharpe,
        candidate_max_drawdown_pct=candidate_max_drawdown_pct,
        n_trades=n_trades,
        shadow_started_on=shadow_started_on,
        shadow_last_active_on=shadow_last_active_on,
    )


# --- GatePolicy validation ----------------------


def test_policy_default():
    p = GatePolicy()
    assert p.min_shadow_days == 90


def test_policy_invalid_shadow_days_rejected():
    with pytest.raises(ValueError):
        GatePolicy(min_shadow_days=0)


def test_policy_invalid_sharpe_delta_rejected():
    with pytest.raises(ValueError):
        GatePolicy(min_sharpe_delta=0.0)


def test_policy_invalid_drawdown_rejected():
    with pytest.raises(ValueError):
        GatePolicy(max_drawdown_pct=0.0)
    with pytest.raises(ValueError):
        GatePolicy(max_drawdown_pct=1.5)


def test_policy_invalid_trade_count_rejected():
    with pytest.raises(ValueError):
        GatePolicy(min_trade_count=0)


# --- ShadowMetrics validation -------------------


def test_metrics_valid():
    m = _metrics()
    assert m.shadow_days() == 99
    assert m.sharpe_delta() == 4.0


def test_metrics_unreasonable_sharpe_rejected():
    with pytest.raises(ValueError):
        _metrics(candidate_sharpe=100.0)
    with pytest.raises(ValueError):
        _metrics(baseline_sharpe=-100.0)


def test_metrics_drawdown_out_of_range_rejected():
    with pytest.raises(ValueError):
        _metrics(candidate_max_drawdown_pct=1.5)


def test_metrics_negative_trades_rejected():
    with pytest.raises(ValueError):
        _metrics(n_trades=-1)


def test_metrics_active_before_started_rejected():
    with pytest.raises(ValueError):
        _metrics(
            shadow_started_on=date(2026, 5, 1),
            shadow_last_active_on=date(2026, 4, 1),
        )


def test_metrics_immutable():
    m = _metrics()
    with pytest.raises(AttributeError):
        m.candidate_sharpe = 0.0  # type: ignore[misc]


# --- evaluate — clean path ----------------------


def test_evaluate_clean_eligible():
    m = _metrics()
    record = evaluate("agent-v2", "agent-v1", m)
    assert record.status is GateStatus.ELIGIBLE
    assert not record.rejection_reasons


# --- evaluate — rejection paths -----------------


def test_evaluate_insufficient_days_rejected():
    m = _metrics(
        shadow_started_on=date(2026, 4, 1),
        shadow_last_active_on=date(2026, 5, 11),  # 40 days
    )
    record = evaluate("agent-v2", "agent-v1", m)
    assert record.status is GateStatus.REJECTED
    assert RejectionReason.INSUFFICIENT_DAYS in record.rejection_reasons


def test_evaluate_sharpe_delta_too_low_rejected():
    m = _metrics(candidate_sharpe=1.5, baseline_sharpe=1.0)
    record = evaluate("agent-v2", "agent-v1", m)
    assert record.status is GateStatus.REJECTED
    assert RejectionReason.SHARPE_DELTA_TOO_LOW in record.rejection_reasons


def test_evaluate_drawdown_breach_rejected():
    m = _metrics(candidate_max_drawdown_pct=0.30)
    record = evaluate("agent-v2", "agent-v1", m)
    assert record.status is GateStatus.REJECTED
    assert RejectionReason.DRAWDOWN_BREACH in record.rejection_reasons


def test_evaluate_trade_count_too_low_rejected():
    m = _metrics(n_trades=50)
    record = evaluate("agent-v2", "agent-v1", m)
    assert record.status is GateStatus.REJECTED
    assert RejectionReason.TRADE_COUNT_TOO_LOW in record.rejection_reasons


def test_evaluate_negative_sharpe_rejected_under_default_policy():
    m = _metrics(candidate_sharpe=-0.5, baseline_sharpe=-5.0)  # +Δ but still negative
    record = evaluate("agent-v2", "agent-v1", m)
    assert record.status is GateStatus.REJECTED
    assert RejectionReason.STILL_NEGATIVE in record.rejection_reasons


def test_evaluate_negative_sharpe_allowed_when_disabled():
    m = _metrics(candidate_sharpe=-0.5, baseline_sharpe=-5.0)
    policy = GatePolicy(require_positive_sharpe=False)
    record = evaluate("agent-v2", "agent-v1", m, policy=policy)
    assert record.status is GateStatus.ELIGIBLE


def test_evaluate_combined_failures_all_captured():
    m = _metrics(
        candidate_sharpe=-1.0,
        baseline_sharpe=0.0,
        candidate_max_drawdown_pct=0.50,
        n_trades=10,
        shadow_started_on=date(2026, 5, 1),
        shadow_last_active_on=date(2026, 5, 11),
    )
    record = evaluate("agent-v2", "agent-v1", m)
    assert record.status is GateStatus.REJECTED
    # Should hit all reasons.
    assert len(record.rejection_reasons) >= 4


# --- GateRecord validation ----------------------


def test_record_same_agent_rejected():
    m = _metrics()
    with pytest.raises(ValueError):
        GateRecord(
            candidate_id="agent",
            baseline_id="agent",
            metrics=m,
        )


def test_record_rejected_without_reasons_rejected():
    m = _metrics()
    with pytest.raises(ValueError):
        GateRecord(
            candidate_id="agent-v2",
            baseline_id="agent-v1",
            metrics=m,
            status=GateStatus.REJECTED,
            rejection_reasons=(),
        )


def test_record_promoted_without_date_rejected():
    m = _metrics()
    with pytest.raises(ValueError):
        GateRecord(
            candidate_id="agent-v2",
            baseline_id="agent-v1",
            metrics=m,
            status=GateStatus.PROMOTED,
            promoted_on=None,
        )


def test_record_promoted_before_shadow_end_rejected():
    m = _metrics()
    with pytest.raises(ValueError):
        GateRecord(
            candidate_id="agent-v2",
            baseline_id="agent-v1",
            metrics=m,
            status=GateStatus.PROMOTED,
            promoted_on=m.shadow_last_active_on - timedelta(days=1),
        )


def test_record_immutable():
    record = evaluate("agent-v2", "agent-v1", _metrics())
    with pytest.raises(AttributeError):
        record.status = GateStatus.PROMOTED  # type: ignore[misc]


# --- mark_promoted ------------------------------


def test_promote_eligible():
    record = evaluate("agent-v2", "agent-v1", _metrics())
    promoted = mark_promoted(record, on=date(2026, 5, 20))
    assert promoted.status is GateStatus.PROMOTED
    assert promoted.promoted_on == date(2026, 5, 20)


def test_promote_rejected_disallowed():
    m = _metrics(candidate_sharpe=1.5, baseline_sharpe=1.0)
    record = evaluate("agent-v2", "agent-v1", m)
    with pytest.raises(ValueError):
        mark_promoted(record, on=date(2026, 5, 20))


def test_promote_before_shadow_end_rejected():
    record = evaluate("agent-v2", "agent-v1", _metrics())
    with pytest.raises(ValueError):
        mark_promoted(record, on=date(2026, 4, 1))


def test_promoted_is_terminal():
    record = mark_promoted(
        evaluate("agent-v2", "agent-v1", _metrics()),
        on=date(2026, 5, 20),
    )
    with pytest.raises(ValueError):
        mark_promoted(record, on=date(2026, 5, 21))


# --- mark_rejected_late -------------------------


def test_reject_late_from_eligible():
    record = evaluate("agent-v2", "agent-v1", _metrics())
    rejected = mark_rejected_late(record, reasons=(RejectionReason.DRAWDOWN_BREACH,))
    assert rejected.status is GateStatus.REJECTED


def test_reject_late_empty_reasons_rejected():
    record = evaluate("agent-v2", "agent-v1", _metrics())
    with pytest.raises(ValueError):
        mark_rejected_late(record, reasons=())


def test_reject_late_from_promoted_rejected():
    record = mark_promoted(
        evaluate("agent-v2", "agent-v1", _metrics()),
        on=date(2026, 5, 20),
    )
    with pytest.raises(ValueError):
        mark_rejected_late(record, reasons=(RejectionReason.DRAWDOWN_BREACH,))


# --- Render -------------------------------------


def test_render_status_emoji():
    record = evaluate("agent-v2", "agent-v1", _metrics())
    out = render_record(record)
    assert "🟢" in out


def test_render_no_secret_leak():
    record = evaluate("agent-v2@example.com", "agent-v1@example.com", _metrics())
    out = render_record(record)
    assert "agent-v2@example.com" not in out
    assert "agent-v1@example.com" not in out


def test_render_rejection_reasons_listed():
    m = _metrics(candidate_sharpe=0.5, baseline_sharpe=0.4)
    record = evaluate("agent-v2", "agent-v1", m)
    out = render_record(record)
    assert "Rejections" in out


def test_render_promoted_date():
    record = mark_promoted(
        evaluate("agent-v2", "agent-v1", _metrics()),
        on=date(2026, 5, 20),
    )
    out = render_record(record)
    assert "2026-05-20" in out
