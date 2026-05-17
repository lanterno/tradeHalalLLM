"""Tests for `core/promotion_gate.py`.

Pins each check's pass / fail / unmeasured behaviour, the
warning ladder for marginal fold counts, the additive composition
contract (more inputs → more potential failures, never silently
softer), the A/B opt-in semantics including the
"significant-but-worse" reject path, and the render helper output.
"""

from __future__ import annotations

import pytest

from halal_trader.core.ab_compare import ABComparison, CohortStats
from halal_trader.core.promotion_gate import (
    CheckResult,
    PromotionThresholds,
    PromotionVerdict,
    evaluate_promotion,
    render_verdict,
)
from halal_trader.crypto.backtest import BacktestResult
from halal_trader.crypto.walkforward import MonteCarloReport, WalkForwardReport


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
    fold_drawdowns: list[float] | None = None,
    trades_per_fold: int = 10,
) -> WalkForwardReport:
    if fold_drawdowns is None:
        fold_drawdowns = [0.05] * fold_count
    folds = [
        _fold(max_drawdown_pct=fold_drawdowns[i], total_trades=trades_per_fold)
        for i in range(fold_count)
    ]
    return WalkForwardReport(
        folds=folds,
        avg_return_pct=avg_return_pct,
        avg_sharpe=avg_sharpe,
        win_rate=win_rate,
        fold_count=fold_count,
    )


def _mc(
    *,
    runs: int = 500,
    max_drawdown_pct_p95: float = 0.10,
) -> MonteCarloReport:
    return MonteCarloReport(
        runs=runs,
        final_return_pct_mean=0.10,
        final_return_pct_p5=0.0,
        final_return_pct_p95=0.20,
        max_drawdown_pct_mean=0.05,
        max_drawdown_pct_p95=max_drawdown_pct_p95,
    )


def _ab(*, p_value: float | None = 0.01, mean_diff: float = 0.005) -> ABComparison:
    """Build a minimal ABComparison stub. CohortStats fields are
    irrelevant to the gate — only `p_value` + `mean_diff` matter."""
    a_stats = CohortStats(
        n_trades=100,
        win_rate=0.55,
        mean_return=0.01,
        median_return=0.005,
        std_return=0.02,
        sharpe=0.5,
        max_drawdown=-0.05,
        total_return=1.0,
        profit_factor=2.0,
    )
    b_stats = CohortStats(
        n_trades=100,
        win_rate=0.50,
        mean_return=0.005,
        median_return=0.003,
        std_return=0.02,
        sharpe=0.25,
        max_drawdown=-0.05,
        total_return=0.5,
        profit_factor=1.5,
    )
    return ABComparison(
        a=a_stats,
        b=b_stats,
        mean_diff=mean_diff,
        t_statistic=2.5,
        degrees_of_freedom=198.0,
        p_value=p_value,
        significant_at_05=p_value is not None and p_value < 0.05,
    )


# ── Walk-forward checks ──────────────────────────────────


def test_passes_with_default_thresholds_on_strong_strategy():
    """Default-threshold sanity: a strategy clearing every default
    bar passes."""
    verdict = evaluate_promotion(_wf())
    assert verdict.passed
    assert verdict.failures == []


def test_fails_when_avg_return_below_zero():
    verdict = evaluate_promotion(_wf(avg_return_pct=-0.01))
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "oos_avg_return_pct" in failure_names


def test_fails_when_sharpe_below_floor():
    verdict = evaluate_promotion(_wf(avg_sharpe=0.3))
    assert not verdict.passed
    sharpe_check = next(f for f in verdict.failures if f.name == "oos_avg_sharpe")
    assert sharpe_check.actual == 0.3
    assert "Sharpe" in sharpe_check.remediation


def test_fails_when_win_rate_below_floor():
    verdict = evaluate_promotion(_wf(win_rate=0.30))
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "oos_win_rate" in failure_names


def test_fails_when_fold_count_below_floor():
    verdict = evaluate_promotion(_wf(fold_count=2))
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "fold_count" in failure_names


def test_fails_when_worst_fold_drawdown_exceeds_cap():
    """Pin: the gate uses the WORST fold's drawdown, not the mean.
    A single fold with a 30% drawdown must kill the promotion even
    if the average is fine."""
    verdict = evaluate_promotion(_wf(fold_drawdowns=[0.05, 0.05, 0.05, 0.05, 0.05, 0.30]))
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "max_oos_drawdown_pct" in failure_names


def test_fails_when_total_trades_below_floor():
    """Sample size matters more than the average metrics — a
    1-trade fold with 100% win rate must not pass."""
    verdict = evaluate_promotion(_wf(trades_per_fold=2))
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "total_trades" in failure_names


# ── warnings ─────────────────────────────────────────────


def test_warns_on_fold_count_just_above_floor():
    """6 folds at default (floor=5) should warn but pass."""
    verdict = evaluate_promotion(_wf(fold_count=6))
    assert verdict.passed
    assert len(verdict.warnings) >= 1
    assert any("fold count" in w.lower() for w in verdict.warnings)


def test_no_warning_when_fold_count_well_above_floor():
    """≥ floor + 2 should be silent."""
    verdict = evaluate_promotion(_wf(fold_count=20))
    assert all("fold count" not in w.lower() for w in verdict.warnings)


# ── Monte Carlo checks ───────────────────────────────────


def test_mc_check_skipped_when_no_report_supplied():
    verdict = evaluate_promotion(_wf())
    check_names = {c.name for c in verdict.checks}
    assert "mc_p95_drawdown_pct" not in check_names
    assert "mc_resamples" not in check_names


def test_passes_when_mc_p95_drawdown_under_cap():
    verdict = evaluate_promotion(_wf(), monte_carlo=_mc(max_drawdown_pct_p95=0.15))
    assert verdict.passed


def test_fails_when_mc_p95_drawdown_over_cap():
    verdict = evaluate_promotion(_wf(), monte_carlo=_mc(max_drawdown_pct_p95=0.40))
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "mc_p95_drawdown_pct" in failure_names


def test_fails_when_mc_resamples_below_floor():
    verdict = evaluate_promotion(_wf(), monte_carlo=_mc(runs=10))
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "mc_resamples" in failure_names


def test_mc_field_names_match_walkforward_report_shape():
    """Pin the field names against `walkforward.MonteCarloReport`
    so a refactor of either side surfaces as a NameError, not a
    silent always-pass via getattr's None fallback."""
    mc = _mc()
    assert hasattr(mc, "max_drawdown_pct_p95")
    assert hasattr(mc, "runs")


# ── A/B opt-in ───────────────────────────────────────────


def test_ab_check_skipped_by_default():
    """Default thresholds set ab_significance_required=False — the
    check must not appear at all."""
    verdict = evaluate_promotion(_wf(), ab_comparison=_ab())
    check_names = {c.name for c in verdict.checks}
    assert "ab_significance" not in check_names


def test_ab_check_required_but_no_comparison_fails():
    th = PromotionThresholds(ab_significance_required=True)
    verdict = evaluate_promotion(_wf(), thresholds=th)
    assert not verdict.passed
    failure_names = {f.name for f in verdict.failures}
    assert "ab_significance" in failure_names


def test_ab_check_passes_when_significant_and_better():
    th = PromotionThresholds(ab_significance_required=True)
    verdict = evaluate_promotion(
        _wf(),
        thresholds=th,
        ab_comparison=_ab(p_value=0.01, mean_diff=0.005),
    )
    assert verdict.passed


def test_ab_check_fails_when_significant_but_worse():
    """Pin the regression-rejection: a strategy that's significantly
    *worse* than the live baseline must be rejected even though
    the difference is statistically real."""
    th = PromotionThresholds(ab_significance_required=True)
    verdict = evaluate_promotion(
        _wf(),
        thresholds=th,
        ab_comparison=_ab(p_value=0.01, mean_diff=-0.005),
    )
    assert not verdict.passed
    ab_failure = next(f for f in verdict.failures if f.name == "ab_significance")
    assert (
        "below" in ab_failure.remediation.lower() or "regression" in ab_failure.remediation.lower()
    )


def test_ab_check_fails_on_non_significant_pvalue():
    th = PromotionThresholds(ab_significance_required=True)
    verdict = evaluate_promotion(
        _wf(),
        thresholds=th,
        ab_comparison=_ab(p_value=0.20, mean_diff=0.005),
    )
    assert not verdict.passed
    ab_failure = next(f for f in verdict.failures if f.name == "ab_significance")
    assert "noise" in ab_failure.remediation.lower() or "p-value" in ab_failure.remediation.lower()


def test_ab_check_fails_on_none_pvalue_when_required():
    """Small-sample / scipy-missing cases produce p_value=None on
    the comparator. The gate must not silently pass."""
    th = PromotionThresholds(ab_significance_required=True)
    verdict = evaluate_promotion(
        _wf(),
        thresholds=th,
        ab_comparison=_ab(p_value=None, mean_diff=0.005),
    )
    assert not verdict.passed


# ── threshold customisation ──────────────────────────────


def test_strict_thresholds_can_reject_default_passing_strategy():
    """Operator can tighten any threshold; pin so a refactor
    doesn't accidentally fix the defaults in stone."""
    th = PromotionThresholds(min_oos_avg_sharpe=2.0)
    verdict = evaluate_promotion(_wf(avg_sharpe=1.0), thresholds=th)
    assert not verdict.passed


def test_loose_thresholds_can_accept_default_failing_strategy():
    """Symmetric: an operator can loosen too. Documented in the
    audit trail by the threshold record."""
    th = PromotionThresholds(min_oos_avg_sharpe=0.0)
    verdict = evaluate_promotion(_wf(avg_sharpe=0.1), thresholds=th)
    assert verdict.passed


# ── verdict structure ────────────────────────────────────


def test_verdict_failures_subset_matches_overall_passed():
    verdict = evaluate_promotion(_wf(avg_sharpe=0.1))
    assert not verdict.passed
    assert len(verdict.failures) >= 1
    for failure in verdict.failures:
        assert isinstance(failure, CheckResult)
        assert not failure.passed


def test_verdict_checks_includes_passes_too():
    """The `checks` list is the complete picture — passes + fails.
    The dashboard renders 'things that look good' from the passes."""
    verdict = evaluate_promotion(_wf())
    assert all(c.passed for c in verdict.checks)
    # The default verdict has 6 walk-forward checks.
    assert len(verdict.checks) == 6


def test_verdict_is_immutable():
    verdict = evaluate_promotion(_wf())
    assert isinstance(verdict, PromotionVerdict)
    with pytest.raises(Exception):
        verdict.passed = False  # type: ignore[misc]


def test_check_result_carries_actual_and_threshold():
    verdict = evaluate_promotion(_wf(avg_sharpe=0.3))
    sharpe = next(c for c in verdict.checks if c.name == "oos_avg_sharpe")
    assert sharpe.actual == 0.3
    assert sharpe.threshold == 0.5


# ── render_verdict ───────────────────────────────────────


def test_render_includes_overall_status():
    text = render_verdict(evaluate_promotion(_wf()))
    assert "PASS" in text
    assert "Promote-to-live verdict" in text


def test_render_marks_failures_with_x():
    text = render_verdict(evaluate_promotion(_wf(avg_sharpe=0.1)))
    assert "FAIL" in text
    assert "✘" in text


def test_render_includes_remediation_for_failures():
    text = render_verdict(evaluate_promotion(_wf(avg_sharpe=0.1)))
    assert "Sharpe" in text
    assert "→" in text


def test_render_lists_warnings():
    text = render_verdict(evaluate_promotion(_wf(fold_count=6)))
    assert "Warnings:" in text
    assert "fold count" in text.lower()


def test_render_handles_unmeasured_actual():
    """Pin: when a check's actual is None (data missing), the
    render must show 'n/a' rather than crashing."""
    th = PromotionThresholds(ab_significance_required=True)
    text = render_verdict(evaluate_promotion(_wf(), thresholds=th))
    assert "n/a" in text


# ── additive composition ─────────────────────────────────


def test_adding_mc_can_only_add_failures_never_remove():
    """Pin the additive contract: a strategy that passes
    walk-forward must not pass *less* with Monte Carlo added if
    MC results are good. (Adding MC can only add failures.)"""
    base = evaluate_promotion(_wf())
    with_mc = evaluate_promotion(_wf(), monte_carlo=_mc(max_drawdown_pct_p95=0.10))
    # Both pass. Both have the same walk-forward failures (none).
    assert base.passed and with_mc.passed
    base_wf_fails = {f.name for f in base.failures if not f.name.startswith("mc_")}
    with_mc_wf_fails = {f.name for f in with_mc.failures if not f.name.startswith("mc_")}
    assert base_wf_fails == with_mc_wf_fails


def test_adding_bad_mc_to_passing_wf_now_fails():
    """Symmetric: a passing walk-forward + a bad Monte Carlo must
    fail overall. Adding MC adds failures."""
    base = evaluate_promotion(_wf())
    with_bad_mc = evaluate_promotion(_wf(), monte_carlo=_mc(max_drawdown_pct_p95=0.50))
    assert base.passed
    assert not with_bad_mc.passed
