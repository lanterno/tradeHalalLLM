"""Tests for `halal_trader.ml.backtest_comparator`.

Auxiliary primitive for Wave 4.F walk-forward gate. Covers:
Welch's t-test approximation, Cohen's d, per-metric verdicts,
overall-verdict aggregation, drawdown-inverted comparison.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.ml.backtest_comparator import (
    DEFAULT_POLICY,
    BacktestResult,
    ComparisonPolicy,
    ComparisonVerdict,
    MetricComparison,
    StrategyComparison,
    compare_backtests,
    compare_metric,
    render_metric_comparison,
    render_strategy_comparison,
)

# --------------------------- Enum string pins --------------------------------


def test_verdict_string_values_pinned() -> None:
    assert ComparisonVerdict.A_WINS.value == "a_wins"
    assert ComparisonVerdict.B_WINS.value == "b_wins"
    assert ComparisonVerdict.TIE.value == "tie"
    assert ComparisonVerdict.INCONCLUSIVE.value == "inconclusive"


# --------------------------- ComparisonPolicy --------------------------------


def test_default_policy() -> None:
    assert DEFAULT_POLICY.alpha == 0.05
    assert DEFAULT_POLICY.min_samples_per_arm == 50
    assert DEFAULT_POLICY.min_effect_size == 0.1


def test_policy_rejects_alpha_at_zero() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ComparisonPolicy(alpha=0.0)


def test_policy_rejects_alpha_at_one() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ComparisonPolicy(alpha=1.0)


def test_policy_rejects_min_samples_below_2() -> None:
    """Pin: t-test undefined for n<2."""

    with pytest.raises(ValueError, match="min_samples"):
        ComparisonPolicy(min_samples_per_arm=1)


def test_policy_rejects_negative_effect_size() -> None:
    with pytest.raises(ValueError, match="min_effect_size"):
        ComparisonPolicy(min_effect_size=-0.1)


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.alpha = 0.10  # type: ignore[misc]


# --------------------------- BacktestResult ----------------------------------


def _result(**overrides: object) -> BacktestResult:
    base: dict[str, object] = {
        "strategy_id": "alpha",
        "sharpe": 1.5,
        "sharpe_std": 0.3,
        "win_rate": 0.55,
        "win_rate_std": 0.05,
        "total_return_pct": 12.0,
        "total_return_std": 2.5,
        "max_drawdown_pct": 8.0,
        "max_drawdown_std": 1.5,
        "trade_count": 100,
    }
    base.update(overrides)
    return BacktestResult(**base)  # type: ignore[arg-type]


def test_result_rejects_empty_strategy_id() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        _result(strategy_id="")


def test_result_rejects_negative_sharpe_std() -> None:
    with pytest.raises(ValueError, match="sharpe_std"):
        _result(sharpe_std=-0.1)


def test_result_rejects_win_rate_above_one() -> None:
    with pytest.raises(ValueError, match="win_rate"):
        _result(win_rate=1.01)


def test_result_rejects_negative_drawdown() -> None:
    """Pin: max_drawdown_pct is a positive magnitude (lower is better)."""

    with pytest.raises(ValueError, match="max_drawdown_pct"):
        _result(max_drawdown_pct=-0.5)


def test_result_rejects_zero_trade_count() -> None:
    with pytest.raises(ValueError, match="trade_count"):
        _result(trade_count=0)


def test_result_is_frozen() -> None:
    r = _result()
    with pytest.raises(FrozenInstanceError):
        r.sharpe = 99.0  # type: ignore[misc]


# --------------------------- compare_metric ---------------------------------


def test_compare_metric_inconclusive_low_samples() -> None:
    """Pin: insufficient samples → INCONCLUSIVE regardless of p-value."""

    result = compare_metric(
        metric_name="sharpe",
        a_mean=2.0,
        a_std=0.5,
        a_n=10,  # below default min_samples=50
        b_mean=1.0,
        b_std=0.5,
        b_n=10,
        policy=DEFAULT_POLICY,
    )
    assert result.verdict is ComparisonVerdict.INCONCLUSIVE


def test_compare_metric_inconclusive_low_effect_size() -> None:
    """Pin: tiny effect size → INCONCLUSIVE even with large samples."""

    # Cohen's d ≈ 0.05 (below default 0.1 threshold)
    result = compare_metric(
        metric_name="sharpe",
        a_mean=1.50,
        a_std=1.0,
        a_n=1000,
        b_mean=1.45,  # 0.05 difference; pooled std ≈ 1.0; d ≈ 0.05
        b_std=1.0,
        b_n=1000,
        policy=DEFAULT_POLICY,
    )
    assert result.verdict is ComparisonVerdict.INCONCLUSIVE


def test_compare_metric_a_wins_clearly() -> None:
    """A has higher Sharpe with low p-value + meaningful effect."""

    result = compare_metric(
        metric_name="sharpe",
        a_mean=2.0,
        a_std=0.5,
        a_n=100,
        b_mean=1.0,
        b_std=0.5,
        b_n=100,
        policy=DEFAULT_POLICY,
    )
    assert result.verdict is ComparisonVerdict.A_WINS
    assert result.cohens_d > 0  # positive when A > B


def test_compare_metric_b_wins() -> None:
    """B has higher value."""

    result = compare_metric(
        metric_name="sharpe",
        a_mean=1.0,
        a_std=0.5,
        a_n=100,
        b_mean=2.0,
        b_std=0.5,
        b_n=100,
    )
    assert result.verdict is ComparisonVerdict.B_WINS
    assert result.cohens_d < 0  # negative when A < B


def test_compare_metric_tie_when_p_above_alpha() -> None:
    """Pin: similar means → high p-value → TIE."""

    result = compare_metric(
        metric_name="sharpe",
        a_mean=1.5,
        a_std=2.0,
        a_n=100,
        b_mean=1.45,
        b_std=2.0,
        b_n=100,
    )
    # If effect size also < 0.1, it's INCONCLUSIVE; otherwise TIE
    # The test verifies it's not a clear winner for either
    assert result.verdict in (
        ComparisonVerdict.TIE,
        ComparisonVerdict.INCONCLUSIVE,
    )


def test_compare_metric_drawdown_inverted() -> None:
    """Pin: for drawdown, LOWER value WINS (drawdown is bad)."""

    # A has 5% drawdown, B has 15% — A wins because lower is better
    result = compare_metric(
        metric_name="max_drawdown_pct",
        a_mean=5.0,
        a_std=1.0,
        a_n=100,
        b_mean=15.0,
        b_std=1.0,
        b_n=100,
    )
    assert result.verdict is ComparisonVerdict.A_WINS


def test_compare_metric_drawdown_b_wins_when_lower() -> None:
    """B has lower drawdown → B wins."""

    result = compare_metric(
        metric_name="max_drawdown_pct",
        a_mean=15.0,
        a_std=1.0,
        a_n=100,
        b_mean=5.0,
        b_std=1.0,
        b_n=100,
    )
    assert result.verdict is ComparisonVerdict.B_WINS


def test_compare_metric_zero_variance_equal_means() -> None:
    """Pin: zero-variance + equal means → undefined t, p=1.0."""

    result = compare_metric(
        metric_name="sharpe",
        a_mean=1.5,
        a_std=0.0,
        a_n=100,
        b_mean=1.5,
        b_std=0.0,
        b_n=100,
    )
    assert result.t_statistic == 0.0
    assert result.p_value == 1.0
    assert result.cohens_d == 0.0
    # Verdict: zero effect size → INCONCLUSIVE
    assert result.verdict is ComparisonVerdict.INCONCLUSIVE


def test_compare_metric_custom_strict_policy() -> None:
    """Strict policy: alpha=0.01, min_samples=200."""

    strict = ComparisonPolicy(alpha=0.01, min_samples_per_arm=200, min_effect_size=0.5)
    # 100 samples < 200 → inconclusive even with clear difference
    result = compare_metric(
        metric_name="sharpe",
        a_mean=2.0,
        a_std=0.5,
        a_n=100,
        b_mean=1.0,
        b_std=0.5,
        b_n=100,
        policy=strict,
    )
    assert result.verdict is ComparisonVerdict.INCONCLUSIVE


# --------------------------- Welch t-test approximation ----------------------


def test_welch_known_values() -> None:
    """Pin: Welch's t-statistic against known result.

    Two samples with mean diff=2, std=1, n=100 each:
    var_diff = 1/100 + 1/100 = 0.02
    se_diff = sqrt(0.02) ≈ 0.1414
    t = 2 / 0.1414 ≈ 14.14
    """

    result = compare_metric(
        metric_name="sharpe",
        a_mean=2.0,
        a_std=1.0,
        a_n=100,
        b_mean=0.0,
        b_std=1.0,
        b_n=100,
    )
    assert result.t_statistic == pytest.approx(14.14, rel=0.01)


def test_welch_p_value_extreme() -> None:
    """Pin: huge t-statistic → tiny p-value."""

    result = compare_metric(
        metric_name="sharpe",
        a_mean=10.0,
        a_std=0.1,
        a_n=100,
        b_mean=0.0,
        b_std=0.1,
        b_n=100,
    )
    # |t| > 100 → p effectively 0
    assert result.p_value < 1e-10


def test_welch_p_value_no_difference() -> None:
    """Pin: identical means → p-value = 1.0."""

    result = compare_metric(
        metric_name="sharpe",
        a_mean=1.5,
        a_std=0.5,
        a_n=100,
        b_mean=1.5,
        b_std=0.5,
        b_n=100,
    )
    assert result.p_value == pytest.approx(1.0, abs=1e-6)


# --------------------------- Cohen's d ---------------------------------------


def test_cohens_d_positive_when_a_higher() -> None:
    result = compare_metric(
        metric_name="sharpe",
        a_mean=2.0,
        a_std=1.0,
        a_n=100,
        b_mean=1.0,
        b_std=1.0,
        b_n=100,
    )
    assert result.cohens_d > 0


def test_cohens_d_negative_when_a_lower() -> None:
    result = compare_metric(
        metric_name="sharpe",
        a_mean=1.0,
        a_std=1.0,
        a_n=100,
        b_mean=2.0,
        b_std=1.0,
        b_n=100,
    )
    assert result.cohens_d < 0


def test_cohens_d_known_value() -> None:
    """Pin: d = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)

    With mean diff=1.0, std=1.0 each, d = 1.0 / sqrt(1.0) = 1.0
    """

    result = compare_metric(
        metric_name="sharpe",
        a_mean=2.0,
        a_std=1.0,
        a_n=100,
        b_mean=1.0,
        b_std=1.0,
        b_n=100,
    )
    assert result.cohens_d == pytest.approx(1.0, abs=0.01)


# --------------------------- compare_backtests -------------------------------


def test_compare_backtests_a_clearly_wins() -> None:
    """A is dramatically better on all 4 metrics."""

    a = _result(
        strategy_id="alpha",
        sharpe=2.5,
        sharpe_std=0.3,
        win_rate=0.65,
        win_rate_std=0.04,
        total_return_pct=20.0,
        total_return_std=2.0,
        max_drawdown_pct=5.0,
        max_drawdown_std=1.0,
        trade_count=200,
    )
    b = _result(
        strategy_id="bravo",
        sharpe=0.8,
        sharpe_std=0.3,
        win_rate=0.45,
        win_rate_std=0.04,
        total_return_pct=5.0,
        total_return_std=2.0,
        max_drawdown_pct=15.0,
        max_drawdown_std=1.0,
        trade_count=200,
    )
    comparison = compare_backtests(a, b)
    assert comparison.overall_verdict is ComparisonVerdict.A_WINS
    assert comparison.sharpe.verdict is ComparisonVerdict.A_WINS
    assert comparison.max_drawdown.verdict is ComparisonVerdict.A_WINS


def test_compare_backtests_b_clearly_wins() -> None:
    a = _result(
        strategy_id="alpha",
        sharpe=0.8,
        sharpe_std=0.3,
        win_rate=0.45,
        win_rate_std=0.04,
        total_return_pct=5.0,
        total_return_std=2.0,
        max_drawdown_pct=15.0,
        max_drawdown_std=1.0,
        trade_count=200,
    )
    b = _result(
        strategy_id="bravo",
        sharpe=2.5,
        sharpe_std=0.3,
        win_rate=0.65,
        win_rate_std=0.04,
        total_return_pct=20.0,
        total_return_std=2.0,
        max_drawdown_pct=5.0,
        max_drawdown_std=1.0,
        trade_count=200,
    )
    comparison = compare_backtests(a, b)
    assert comparison.overall_verdict is ComparisonVerdict.B_WINS


def test_compare_backtests_inconclusive_when_low_samples() -> None:
    """Pin: any inconclusive metric → overall INCONCLUSIVE."""

    a = _result(strategy_id="alpha", trade_count=10)  # below 50
    b = _result(strategy_id="bravo", trade_count=10)
    comparison = compare_backtests(a, b)
    assert comparison.overall_verdict is ComparisonVerdict.INCONCLUSIVE


def test_compare_backtests_mixed_results_tie() -> None:
    """Pin: A wins Sharpe + total_return but B wins drawdown +
    win_rate → no clear overall winner.

    All four metrics have decisive verdicts; A and B each win two,
    so the overall verdict is TIE.
    """

    a = _result(
        strategy_id="alpha",
        sharpe=2.5,  # A wins Sharpe
        sharpe_std=0.3,
        win_rate=0.45,  # A loses win_rate
        win_rate_std=0.03,
        total_return_pct=18.0,  # A wins total_return
        total_return_std=1.5,
        max_drawdown_pct=15.0,  # A loses drawdown
        max_drawdown_std=1.0,
        trade_count=200,
    )
    b = _result(
        strategy_id="bravo",
        sharpe=0.8,
        sharpe_std=0.3,
        win_rate=0.65,
        win_rate_std=0.03,
        total_return_pct=8.0,
        total_return_std=1.5,
        max_drawdown_pct=5.0,
        max_drawdown_std=1.0,
        trade_count=200,
    )
    comparison = compare_backtests(a, b)
    # 2-2 split → overall TIE
    assert comparison.overall_verdict is ComparisonVerdict.TIE


def test_compare_backtests_rejects_same_strategy_id() -> None:
    """Pin: comparing a strategy to itself is meaningless."""

    a = _result(strategy_id="alpha")
    b = _result(strategy_id="alpha")
    with pytest.raises(ValueError, match="same strategy_id"):
        compare_backtests(a, b)


def test_compare_backtests_drawdown_inverted_in_overall() -> None:
    """Pin: drawdown verdict is inverted in the overall aggregator.

    If A has higher Sharpe + win_rate + return AND lower drawdown,
    A wins all 4 metrics → A_WINS overall.
    """

    a = _result(
        strategy_id="a",
        sharpe=2.0,
        sharpe_std=0.3,
        win_rate=0.60,
        win_rate_std=0.04,
        total_return_pct=15.0,
        total_return_std=2.0,
        max_drawdown_pct=5.0,  # LOWER → wins
        max_drawdown_std=1.0,
        trade_count=200,
    )
    b = _result(
        strategy_id="b",
        sharpe=1.0,
        sharpe_std=0.3,
        win_rate=0.50,
        win_rate_std=0.04,
        total_return_pct=8.0,
        total_return_std=2.0,
        max_drawdown_pct=15.0,  # HIGHER → loses
        max_drawdown_std=1.0,
        trade_count=200,
    )
    comparison = compare_backtests(a, b)
    assert comparison.max_drawdown.verdict is ComparisonVerdict.A_WINS
    assert comparison.overall_verdict is ComparisonVerdict.A_WINS


# --------------------------- StrategyComparison overall_verdict --------------


def _metric(
    *,
    name: str,
    a_value: float,
    b_value: float,
    verdict: ComparisonVerdict,
) -> MetricComparison:
    """Test helper for direct StrategyComparison construction."""

    return MetricComparison(
        metric_name=name,
        a_value=a_value,
        b_value=b_value,
        cohens_d=0.5,
        t_statistic=2.0,
        p_value=0.01,
        verdict=verdict,
    )


def test_overall_inconclusive_if_any_metric_inconclusive() -> None:
    """Pin: even if 3 of 4 metrics show A_WINS, INCONCLUSIVE wins."""

    sc = StrategyComparison(
        a_strategy_id="a",
        b_strategy_id="b",
        sharpe=_metric(
            name="sharpe",
            a_value=2.0,
            b_value=1.0,
            verdict=ComparisonVerdict.A_WINS,
        ),
        win_rate=_metric(
            name="win_rate",
            a_value=0.6,
            b_value=0.5,
            verdict=ComparisonVerdict.A_WINS,
        ),
        total_return=_metric(
            name="total_return_pct",
            a_value=15.0,
            b_value=8.0,
            verdict=ComparisonVerdict.A_WINS,
        ),
        max_drawdown=_metric(
            name="max_drawdown_pct",
            a_value=5.0,
            b_value=15.0,
            verdict=ComparisonVerdict.INCONCLUSIVE,
        ),
    )
    assert sc.overall_verdict is ComparisonVerdict.INCONCLUSIVE


def test_overall_a_wins_when_a_majority_no_loss() -> None:
    sc = StrategyComparison(
        a_strategy_id="a",
        b_strategy_id="b",
        sharpe=_metric(
            name="sharpe",
            a_value=2.0,
            b_value=1.0,
            verdict=ComparisonVerdict.A_WINS,
        ),
        win_rate=_metric(
            name="win_rate",
            a_value=0.6,
            b_value=0.5,
            verdict=ComparisonVerdict.A_WINS,
        ),
        total_return=_metric(
            name="total_return_pct",
            a_value=15.0,
            b_value=8.0,
            verdict=ComparisonVerdict.A_WINS,
        ),
        max_drawdown=_metric(
            name="max_drawdown_pct",
            a_value=5.0,
            b_value=10.0,
            verdict=ComparisonVerdict.TIE,
        ),
    )
    assert sc.overall_verdict is ComparisonVerdict.A_WINS


def test_overall_tie_when_split_wins() -> None:
    """A wins 2 metrics, B wins 2 → TIE."""

    sc = StrategyComparison(
        a_strategy_id="a",
        b_strategy_id="b",
        sharpe=_metric(
            name="sharpe",
            a_value=2.0,
            b_value=1.0,
            verdict=ComparisonVerdict.A_WINS,
        ),
        win_rate=_metric(
            name="win_rate",
            a_value=0.5,
            b_value=0.6,
            verdict=ComparisonVerdict.B_WINS,
        ),
        total_return=_metric(
            name="total_return_pct",
            a_value=15.0,
            b_value=8.0,
            verdict=ComparisonVerdict.A_WINS,
        ),
        max_drawdown=_metric(
            name="max_drawdown_pct",
            a_value=15.0,
            b_value=5.0,
            verdict=ComparisonVerdict.B_WINS,
        ),
    )
    assert sc.overall_verdict is ComparisonVerdict.TIE


def test_overall_tie_when_all_metrics_tie() -> None:
    sc = StrategyComparison(
        a_strategy_id="a",
        b_strategy_id="b",
        sharpe=_metric(
            name="sharpe",
            a_value=1.5,
            b_value=1.5,
            verdict=ComparisonVerdict.TIE,
        ),
        win_rate=_metric(
            name="win_rate",
            a_value=0.5,
            b_value=0.5,
            verdict=ComparisonVerdict.TIE,
        ),
        total_return=_metric(
            name="total_return_pct",
            a_value=10.0,
            b_value=10.0,
            verdict=ComparisonVerdict.TIE,
        ),
        max_drawdown=_metric(
            name="max_drawdown_pct",
            a_value=8.0,
            b_value=8.0,
            verdict=ComparisonVerdict.TIE,
        ),
    )
    assert sc.overall_verdict is ComparisonVerdict.TIE


# --------------------------- render ------------------------------------------


def test_render_metric_comparison_includes_emoji() -> None:
    comp = _metric(
        name="sharpe",
        a_value=2.0,
        b_value=1.0,
        verdict=ComparisonVerdict.A_WINS,
    )
    out = render_metric_comparison(comp)
    assert "🅰️" in out
    assert "sharpe" in out
    assert "a_wins" in out


def test_render_strategy_comparison_includes_all_metrics() -> None:
    a = _result(
        strategy_id="alpha",
        sharpe=2.0,
        sharpe_std=0.3,
        trade_count=200,
    )
    b = _result(strategy_id="bravo", trade_count=200)
    sc = compare_backtests(a, b)
    out = render_strategy_comparison(sc)
    assert "alpha" in out
    assert "bravo" in out
    assert "sharpe" in out
    assert "win_rate" in out
    assert "total_return_pct" in out
    assert "max_drawdown_pct" in out


def test_render_no_secret_leak() -> None:
    """Pin: render shows summary stats only — no raw return series,
    no operator-side fields."""

    comp = _metric(
        name="sharpe",
        a_value=2.0,
        b_value=1.0,
        verdict=ComparisonVerdict.A_WINS,
    )
    out = render_metric_comparison(comp)
    assert "raw_returns" not in out
    assert "trades_list" not in out
    assert "broker_id" not in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_strategy_a_b_test() -> None:
    """Real-world: operator A/B-tests strategy variation; comparator
    declares clear winner."""

    momentum = _result(
        strategy_id="momentum",
        sharpe=1.8,
        sharpe_std=0.2,
        win_rate=0.58,
        win_rate_std=0.03,
        total_return_pct=14.0,
        total_return_std=1.5,
        max_drawdown_pct=7.0,
        max_drawdown_std=1.0,
        trade_count=250,
    )
    momentum_v2 = _result(
        strategy_id="momentum_v2",
        sharpe=2.4,
        sharpe_std=0.2,
        win_rate=0.62,
        win_rate_std=0.03,
        total_return_pct=18.0,
        total_return_std=1.5,
        max_drawdown_pct=5.5,
        max_drawdown_std=1.0,
        trade_count=250,
    )
    comparison = compare_backtests(momentum, momentum_v2)
    assert comparison.overall_verdict is ComparisonVerdict.B_WINS  # momentum_v2 better


def test_e2e_inconclusive_small_pilot() -> None:
    """Real-world: operator runs 30-trade pilot, comparator says
    'wait for more data' even with apparent improvement."""

    pilot_a = _result(strategy_id="a", trade_count=30)  # below default 50
    pilot_b = _result(strategy_id="b", sharpe=2.5, trade_count=30)
    comparison = compare_backtests(pilot_a, pilot_b)
    assert comparison.overall_verdict is ComparisonVerdict.INCONCLUSIVE


def test_e2e_replay_consistency() -> None:
    """Same inputs produce equal comparison results."""

    a = _result(strategy_id="alpha", trade_count=200)
    b = _result(strategy_id="bravo", trade_count=200)
    comp1 = compare_backtests(a, b)
    comp2 = compare_backtests(a, b)
    assert comp1 == comp2
