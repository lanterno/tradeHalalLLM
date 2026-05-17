"""Tests for `ml/ci_pipeline.py`.

Pins each per-gate check, the "skipped → pass with note"
contract, the aggregate passed-iff-every-gate-passed rule, the
threshold validation, and the render output.
"""

from __future__ import annotations

import pytest

from halal_trader.crypto.backtest import BacktestResult
from halal_trader.crypto.walkforward import MonteCarloReport, WalkForwardReport
from halal_trader.ml.ci_pipeline import (
    CIPipelineReport,
    CIThresholds,
    GateOutcome,
    _ks_distance,
    check_distribution_drift,
    check_sharpe_regression,
    check_win_rate_regression,
    render_report,
    run_ci,
)


def _fold(*, max_drawdown_pct: float = 0.05, total_trades: int = 10) -> BacktestResult:
    return BacktestResult(
        pair="STUB",
        start_date="",
        end_date="",
        initial_balance=1000.0,
        final_balance=1000.0,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
    )


def _wf(
    *,
    avg_return_pct: float = 0.05,
    avg_sharpe: float = 1.0,
    win_rate: float = 0.55,
    fold_count: int = 6,
    trades_per_fold: int = 10,
) -> WalkForwardReport:
    folds = [_fold(total_trades=trades_per_fold) for _ in range(fold_count)]
    return WalkForwardReport(
        folds=folds,
        avg_return_pct=avg_return_pct,
        avg_sharpe=avg_sharpe,
        win_rate=win_rate,
        fold_count=fold_count,
    )


# ── threshold validation ─────────────────────────────────


def test_thresholds_reject_invalid_sharpe_ratio():
    with pytest.raises(ValueError, match="min_sharpe_ratio"):
        CIThresholds(min_sharpe_ratio=-0.1)
    with pytest.raises(ValueError, match="min_sharpe_ratio"):
        CIThresholds(min_sharpe_ratio=1.5)


def test_thresholds_reject_invalid_win_rate_ratio():
    with pytest.raises(ValueError, match="min_win_rate_ratio"):
        CIThresholds(min_win_rate_ratio=-1.0)


def test_thresholds_reject_zero_or_oversize_distance():
    """Pin: distance > 1.0 makes no sense (CDFs are bounded
    in [0, 1]); zero would always fail."""
    with pytest.raises(ValueError, match="distribution_distance"):
        CIThresholds(max_distribution_distance=0.0)
    with pytest.raises(ValueError, match="distribution_distance"):
        CIThresholds(max_distribution_distance=1.5)


def test_thresholds_reject_zero_min_sample_size():
    with pytest.raises(ValueError, match="min_sample_size"):
        CIThresholds(min_sample_size=0)


# ── KS distance helper ───────────────────────────────────


def test_ks_distance_zero_when_distributions_identical():
    """Pin: identical samples → distance 0."""
    assert _ks_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_ks_distance_one_when_disjoint():
    """Pin: completely disjoint samples (every value in A < every
    value in B) → distance 1."""
    a = [0.1, 0.2, 0.3]
    b = [10.0, 20.0, 30.0]
    assert _ks_distance(a, b) == 1.0


def test_ks_distance_handles_empty():
    assert _ks_distance([], [1.0]) is None
    assert _ks_distance([1.0], []) is None


def test_ks_distance_bounded_in_zero_one():
    """Pin: any two samples produce a distance in [0, 1]."""
    for a, b in [
        ([1, 2, 3], [2, 3, 4]),
        ([1, 1, 1, 1], [2, 2, 2, 2]),
        ([0, 1], [0.5]),
    ]:
        d = _ks_distance(a, b)
        assert d is not None
        assert 0.0 <= d <= 1.0


# ── Sharpe regression check ──────────────────────────────


def test_sharpe_regression_passes_when_candidate_at_or_above_threshold():
    out = check_sharpe_regression(
        candidate_sharpe=0.95,
        incumbent_sharpe=1.00,
        thresholds=CIThresholds(min_sharpe_ratio=0.90),
    )
    assert out.passed
    assert out.remediation == ""


def test_sharpe_regression_fails_when_candidate_below_threshold():
    out = check_sharpe_regression(
        candidate_sharpe=0.50,
        incumbent_sharpe=1.00,
        thresholds=CIThresholds(min_sharpe_ratio=0.90),
    )
    assert not out.passed
    assert "regressed" in out.remediation.lower()


def test_sharpe_regression_skips_on_no_incumbent():
    """Pin: cold-start with no incumbent passes with a 'skipped'
    note. Operators running fresh models shouldn't be gated."""
    out = check_sharpe_regression(
        candidate_sharpe=1.0,
        incumbent_sharpe=None,
        thresholds=CIThresholds(),
    )
    assert out.passed
    assert out.is_skipped
    assert out.remediation.startswith("skipped")


def test_sharpe_regression_skips_on_non_positive_incumbent():
    """Pin: a non-positive incumbent isn't a real baseline (could
    even cause a divide-by-zero); skip rather than divide."""
    out = check_sharpe_regression(
        candidate_sharpe=1.0,
        incumbent_sharpe=0.0,
        thresholds=CIThresholds(),
    )
    assert out.passed
    assert out.is_skipped


# ── Win-rate regression check ────────────────────────────


def test_win_rate_regression_passes_above_threshold():
    out = check_win_rate_regression(
        candidate_win_rate=0.50,
        incumbent_win_rate=0.50,
        thresholds=CIThresholds(),
    )
    assert out.passed


def test_win_rate_regression_fails_below_threshold():
    out = check_win_rate_regression(
        candidate_win_rate=0.30,
        incumbent_win_rate=0.50,
        thresholds=CIThresholds(min_win_rate_ratio=0.95),
    )
    assert not out.passed


def test_win_rate_regression_skips_on_no_incumbent():
    out = check_win_rate_regression(
        candidate_win_rate=0.50,
        incumbent_win_rate=None,
        thresholds=CIThresholds(),
    )
    assert out.passed
    assert out.is_skipped


# ── Distribution drift check ─────────────────────────────


def test_drift_passes_when_distributions_close():
    """Pin: identical samples → KS distance 0 → PASS. Use the same
    sample for both sides to guarantee a clean baseline (the
    heavy-tailed nature of trade returns means even similar but
    non-identical empirical distributions can show meaningful KS
    distance on small samples)."""
    sample = [0.01, 0.02, 0.01, 0.0, 0.015] * 10
    out = check_distribution_drift(
        candidate_returns=sample,
        incumbent_returns=list(sample),
        thresholds=CIThresholds(max_distribution_distance=0.30),
    )
    assert out.passed


def test_drift_fails_when_distributions_diverge():
    """Pin: clearly disjoint samples → KS distance ≈ 1 → FAIL."""
    out = check_distribution_drift(
        candidate_returns=[0.05] * 50,
        incumbent_returns=[-0.05] * 50,
        thresholds=CIThresholds(max_distribution_distance=0.30),
    )
    assert not out.passed
    assert "drifted" in out.remediation.lower()


def test_drift_skips_when_no_incumbent():
    out = check_distribution_drift(
        candidate_returns=[0.01] * 50,
        incumbent_returns=None,
        thresholds=CIThresholds(),
    )
    assert out.passed
    assert out.is_skipped


def test_drift_skips_when_sample_below_min():
    """Pin: KS on small samples is mostly noise; skip the check
    rather than alarming."""
    out = check_distribution_drift(
        candidate_returns=[0.01] * 5,
        incumbent_returns=[0.02] * 50,
        thresholds=CIThresholds(min_sample_size=20),
    )
    assert out.passed
    assert out.is_skipped


def test_drift_skips_when_either_side_below_min():
    """Symmetric: small incumbent triggers the same skip."""
    out = check_distribution_drift(
        candidate_returns=[0.01] * 50,
        incumbent_returns=[0.02] * 5,
        thresholds=CIThresholds(min_sample_size=20),
    )
    assert out.passed
    assert out.is_skipped


# ── run_ci aggregate ─────────────────────────────────────


def test_run_ci_passes_when_every_gate_passes():
    """End-to-end: a candidate that meets every threshold gets a
    PASS report."""
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=0.95,
        candidate_win_rate=0.50,
        candidate_returns=[0.01] * 50,
        incumbent_sharpe=1.00,
        incumbent_win_rate=0.50,
        incumbent_returns=[0.01] * 50,
    )
    assert rep.passed
    assert rep.failures == []


def test_run_ci_fails_when_any_gate_fails():
    """Single-gate failure flips the aggregate to FAIL."""
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=0.30,  # huge regression
        candidate_win_rate=0.50,
        candidate_returns=[0.01] * 50,
        incumbent_sharpe=1.00,
        incumbent_win_rate=0.50,
        incumbent_returns=[0.01] * 50,
    )
    assert not rep.passed
    assert any(f.name == "sharpe_regression" for f in rep.failures)


def test_run_ci_promotion_verdict_exposed_for_drilldown():
    """Pin: the underlying Wave 4.F PromotionVerdict is exposed on
    the report so the operator can drill into walk-forward
    failures without re-running."""
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=0.95,
        candidate_win_rate=0.50,
        candidate_returns=[0.01] * 50,
        incumbent_sharpe=1.00,
        incumbent_win_rate=0.50,
        incumbent_returns=[0.01] * 50,
    )
    assert rep.promotion_verdict is not None


def test_run_ci_returns_report_with_four_gates():
    """Walk-forward + Sharpe + win-rate + drift = 4 gates."""
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=1.0,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
    )
    assert len(rep.gates) == 4


def test_run_ci_skipped_gates_count_as_pass():
    """Pin: a fresh-model run (no incumbent) where 3 of 4 gates
    skip but walk-forward passes → overall PASS."""
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=1.0,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
        # No incumbent → 3 skip
    )
    assert rep.passed
    skipped_count = sum(1 for g in rep.gates if g.is_skipped)
    assert skipped_count == 3


def test_run_ci_with_monte_carlo_report_layered():
    """Pin: passing a Monte Carlo report through to the walk-
    forward gate composes with the rest of the CI checks."""
    mc = MonteCarloReport(
        runs=500,
        final_return_pct_mean=0.10,
        final_return_pct_p5=0.0,
        final_return_pct_p95=0.20,
        max_drawdown_pct_mean=0.05,
        max_drawdown_pct_p95=0.15,
    )
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=1.0,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
        candidate_monte_carlo=mc,
    )
    assert rep.passed


def test_run_ci_summary_reports_skip_count():
    """PASS summary mentions skipped gates so the operator knows
    coverage was partial."""
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=1.0,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
    )
    assert "skipped" in rep.summary.lower()


def test_run_ci_summary_reports_failure_count_on_fail():
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=0.30,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
        incumbent_sharpe=1.0,
    )
    assert "FAIL" in rep.summary
    assert "1" in rep.summary  # 1 of 4 gates failed


# ── output structure ─────────────────────────────────────


def test_gate_outcome_is_immutable():
    out = check_sharpe_regression(
        candidate_sharpe=1.0, incumbent_sharpe=None, thresholds=CIThresholds()
    )
    assert isinstance(out, GateOutcome)
    with pytest.raises(Exception):
        out.passed = False  # type: ignore[misc]


def test_pipeline_report_is_immutable():
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=1.0,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
    )
    assert isinstance(rep, CIPipelineReport)
    with pytest.raises(Exception):
        rep.passed = False  # type: ignore[misc]


def test_is_skipped_property_only_true_when_passed_and_skip_note():
    """Pin: a *failure* with a 'skipped' substring isn't a skip;
    the property only fires when the gate passed AND the
    remediation explicitly starts with 'skipped'."""
    out = GateOutcome(
        name="x",
        passed=False,
        actual=None,
        threshold=None,
        remediation="skipped: nope",
    )
    assert not out.is_skipped


# ── render_report ────────────────────────────────────────


def test_render_includes_overall_status():
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=1.0,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
    )
    text = render_report(rep)
    assert "PASS" in text
    assert "ML CI pipeline" in text


def test_render_marks_failures_with_cross():
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=0.30,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
        incumbent_sharpe=1.0,
    )
    text = render_report(rep)
    assert "FAIL" in text
    assert "✘" in text


def test_render_marks_skipped_gates_distinctly():
    """Pin: skipped gates use a — marker rather than ✔ so the
    operator can spot at-a-glance which gates ran 'for real'."""
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=1.0,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
        # No incumbent → skipped
    )
    text = render_report(rep)
    assert "—" in text


def test_render_includes_remediation_for_failures():
    rep = run_ci(
        candidate_walk_forward=_wf(),
        candidate_sharpe=0.30,
        candidate_win_rate=0.5,
        candidate_returns=[0.01] * 50,
        incumbent_sharpe=1.0,
    )
    text = render_report(rep)
    assert "→" in text
