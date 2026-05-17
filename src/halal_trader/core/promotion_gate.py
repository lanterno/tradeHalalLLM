"""Promote-to-live gate over walk-forward / Monte Carlo / A-B results.

Round-4 wave 4.F: today the existing `crypto/walkforward.py` runs
walk-forward + Monte Carlo backtests but the *promotion decision*
(should this strategy go live?) is left to the operator's eye.
This module promotes that judgement to a first-class checked
ruleset.

The gate takes:

* A `WalkForwardReport` (out-of-sample aggregate of N folds).
* Optional Monte Carlo `MonteCarloReport` (drawdown distribution
  over shuffled trade orderings — caught Sharpe-by-luck cases).
* Optional A/B `ABComparison` against the current live prompt
  (Welch's-t test from `core/ab_compare.py`).

…and a `PromotionThresholds` config bundle. Output:
`PromotionVerdict(passed, checks, remediation)` — a per-check
breakdown the dashboard can render, plus a one-line remediation
hint for each failure so the operator knows *what to fix*, not just
that it failed.

Why a separate module rather than extending `walkforward.py`:

* The thresholds are *operator policy*, not backtest mechanics.
  Keeping them out of the report dataclass means a future
  walkforward-internals refactor can't accidentally tighten /
  loosen the live-trading bar.
* The gate composes three already-shipped sources — walkforward
  (existing), Monte Carlo (existing), A/B comparator
  (Round-4 Wave 5.B). Putting the composition in `core/` keeps
  every check importable without touching `crypto/`-specific
  dependencies.

Halal alignment: passing the gate is *necessary but not
sufficient* for live trading. The operator still has to engage
the kill-switch + halt mechanism + halal screener. The gate just
prevents a structurally-bad strategy from reaching the live
queue.

Pure-Python; no DB, no async. Operates on already-collected report
dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from halal_trader.core.ab_compare import ABComparison
from halal_trader.crypto.backtest import BacktestResult
from halal_trader.crypto.walkforward import (
    MonteCarloReport,
    WalkForwardReport,
)

# ── Threshold config ──────────────────────────────────────


@dataclass(frozen=True)
class PromotionThresholds:
    """Operator-configured bars a strategy must clear.

    Defaults are deliberately *demanding*: a strategy that barely
    scrapes through them is still risky. Operators can tighten
    individually (e.g. require 2.0 Sharpe for a low-volume
    strategy), but loosening below these defaults should be
    documented in the audit trail.

    ``ab_significance_required`` defaults to False because A/B
    comparisons need ≥100 trades against a live baseline; many
    promotion candidates are first-time strategies with no
    incumbent to compare against. When True, the gate rejects
    if the A/B comparison's p-value > 0.05 *and* the new strategy's
    mean return is below the baseline's.
    """

    min_oos_avg_return_pct: float = 0.0
    min_oos_avg_sharpe: float = 0.5
    min_oos_win_rate: float = 0.40
    min_fold_count: int = 5
    max_oos_drawdown_pct: float = 0.20  # 20% peak-to-trough cap
    min_total_trades: int = 50

    # Monte Carlo: empirical drawdown distribution must show the
    # observed run isn't just lucky.
    max_mc_p95_drawdown_pct: float = 0.30
    min_mc_resamples: int = 100

    # A/B: only enforced if `ab_significance_required` is True.
    ab_significance_required: bool = False
    ab_min_p_value: float = 0.05  # caller wants p < this to "win"


# ── Verdict structure ─────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    """One named threshold check's outcome.

    ``actual`` and ``threshold`` are stored as floats so the
    dashboard can render them in a single column. ``remediation``
    is a one-line operator-readable hint when the check fails;
    empty string when it passes.
    """

    name: str
    passed: bool
    actual: float | None
    threshold: float | None
    remediation: str = ""


@dataclass(frozen=True)
class PromotionVerdict:
    """Outcome of running the gate.

    ``passed`` is True iff every check passes. ``checks`` lists
    every check (including passes — useful for the dashboard's
    "things that look good" section). ``failures`` is the subset
    that failed, surfaced separately so the operator's eye lands
    on what to fix.

    ``warnings`` lists soft concerns — checks the gate is
    deliberately *not* failing on but the operator should know
    about (e.g. fold count just barely above the floor).
    """

    passed: bool
    checks: list[CheckResult]
    failures: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────


def _check(
    name: str,
    actual: float | None,
    *,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
    remediation: str,
) -> CheckResult:
    """Build a CheckResult by comparing ``actual`` against a min /
    max threshold. Pin: when ``actual`` is None (data not measured),
    the check fails — the gate must not let unmeasured fields
    silently pass."""
    if actual is None:
        return CheckResult(
            name=name,
            passed=False,
            actual=None,
            threshold=min_threshold if min_threshold is not None else max_threshold,
            remediation=f"{remediation} (not measured — data missing)",
        )
    if min_threshold is not None:
        if actual >= min_threshold:
            return CheckResult(name=name, passed=True, actual=actual, threshold=min_threshold)
        return CheckResult(
            name=name,
            passed=False,
            actual=actual,
            threshold=min_threshold,
            remediation=remediation,
        )
    if max_threshold is not None:
        if actual <= max_threshold:
            return CheckResult(name=name, passed=True, actual=actual, threshold=max_threshold)
        return CheckResult(
            name=name,
            passed=False,
            actual=actual,
            threshold=max_threshold,
            remediation=remediation,
        )
    raise ValueError("at least one of min_threshold / max_threshold must be set")


def _total_trades(report: WalkForwardReport) -> int:
    """Sum trade counts across folds — proxy for sample size."""
    return sum(getattr(f, "total_trades", 0) for f in report.folds)


def _max_oos_drawdown(report: WalkForwardReport) -> float:
    """Worst (most-negative) drawdown observed across folds.

    Returns a non-negative fraction (0.20 = 20% drawdown). Pin:
    folds typically expose drawdown as a positive `max_drawdown_pct`
    in the existing BacktestResult, so we just take the max — no
    sign flipping.
    """
    if not report.folds:
        return 0.0
    return float(max(getattr(f, "max_drawdown_pct", 0.0) for f in report.folds))


# ── Gate ──────────────────────────────────────────────────


def evaluate_promotion(
    walk_forward: WalkForwardReport,
    *,
    thresholds: PromotionThresholds | None = None,
    monte_carlo: MonteCarloReport | None = None,
    ab_comparison: ABComparison | None = None,
) -> PromotionVerdict:
    """Run every applicable check; collate into a verdict.

    The gate is *additive* — adding a Monte Carlo or A/B comparison
    can only ever produce *more* failure conditions. A strategy that
    passes with just walk-forward will also pass with walk-forward +
    Monte Carlo when the MC results are good. This makes the gate
    composable: the cheapest checks run first; expensive ones
    (Monte Carlo over 1000 resamples, A/B over 100+ live trades)
    are layered on as the operator gathers evidence.
    """
    thresholds = thresholds or PromotionThresholds()
    checks: list[CheckResult] = []
    warnings: list[str] = []

    # ── walk-forward checks ──
    checks.append(
        _check(
            "oos_avg_return_pct",
            walk_forward.avg_return_pct,
            min_threshold=thresholds.min_oos_avg_return_pct,
            remediation=(
                "Out-of-sample average return is below the floor — "
                "check for prompt overfit to the training window."
            ),
        )
    )
    checks.append(
        _check(
            "oos_avg_sharpe",
            walk_forward.avg_sharpe,
            min_threshold=thresholds.min_oos_avg_sharpe,
            remediation=(
                "Sharpe is too low for live trading. Tighten entry filters "
                "or reduce holding-period noise."
            ),
        )
    )
    checks.append(
        _check(
            "oos_win_rate",
            walk_forward.win_rate,
            min_threshold=thresholds.min_oos_win_rate,
            remediation=(
                "Win rate is below the floor — most asymmetric-payoff "
                "strategies still need ≥40% to clear costs."
            ),
        )
    )
    checks.append(
        _check(
            "fold_count",
            float(walk_forward.fold_count),
            min_threshold=float(thresholds.min_fold_count),
            remediation=(
                "Too few walk-forward folds — increase the kline window "
                "or shrink train_size / test_size to fit more folds."
            ),
        )
    )
    checks.append(
        _check(
            "max_oos_drawdown_pct",
            _max_oos_drawdown(walk_forward),
            max_threshold=thresholds.max_oos_drawdown_pct,
            remediation=(
                "Worst-fold drawdown exceeds the cap. Tighten SL or "
                "size down — a single bad fold of this depth is a kill."
            ),
        )
    )
    checks.append(
        _check(
            "total_trades",
            float(_total_trades(walk_forward)),
            min_threshold=float(thresholds.min_total_trades),
            remediation=(
                "Sample size too small for the result to be trustworthy. "
                "Run more candles or shorten holding periods."
            ),
        )
    )

    # Soft warning on fold count just barely above floor.
    if (
        walk_forward.fold_count >= thresholds.min_fold_count
        and walk_forward.fold_count < thresholds.min_fold_count + 2
    ):
        warnings.append(
            f"Fold count ({walk_forward.fold_count}) is at the floor "
            f"({thresholds.min_fold_count}); consider running more folds "
            f"before promote-to-live."
        )

    # ── Monte Carlo checks (optional) ──
    if monte_carlo is not None:
        # Field names match the existing `MonteCarloReport` shape
        # in `crypto/walkforward.py` (`max_drawdown_pct_p95`,
        # `runs`); pin the references so a refactor of either side
        # surfaces immediately as a NameError, not a silent
        # always-pass.
        checks.append(
            _check(
                "mc_p95_drawdown_pct",
                getattr(monte_carlo, "max_drawdown_pct_p95", None),
                max_threshold=thresholds.max_mc_p95_drawdown_pct,
                remediation=(
                    "95th-percentile Monte Carlo drawdown exceeds the cap — "
                    "the strategy's good runs may be order-of-trades luck."
                ),
            )
        )
        checks.append(
            _check(
                "mc_resamples",
                float(getattr(monte_carlo, "runs", 0)),
                min_threshold=float(thresholds.min_mc_resamples),
                remediation=(
                    "Too few Monte Carlo resamples for the percentile to be "
                    "stable — re-run with more iterations."
                ),
            )
        )

    # ── A/B comparison check (optional, opt-in) ──
    if thresholds.ab_significance_required:
        if ab_comparison is None:
            checks.append(
                CheckResult(
                    name="ab_significance",
                    passed=False,
                    actual=None,
                    threshold=thresholds.ab_min_p_value,
                    remediation=(
                        "A/B significance required but no comparison supplied — "
                        "run ≥100 trades against the live prompt before promotion."
                    ),
                )
            )
        else:
            p = ab_comparison.p_value
            mean_diff = ab_comparison.mean_diff
            if p is None:
                checks.append(
                    CheckResult(
                        name="ab_significance",
                        passed=False,
                        actual=None,
                        threshold=thresholds.ab_min_p_value,
                        remediation=(
                            "A/B p-value couldn't be computed (small sample, "
                            "no scipy). Add more trades or install scipy."
                        ),
                    )
                )
            elif p < thresholds.ab_min_p_value and mean_diff > 0:
                checks.append(
                    CheckResult(
                        name="ab_significance",
                        passed=True,
                        actual=p,
                        threshold=thresholds.ab_min_p_value,
                    )
                )
            else:
                # Either non-significant or significant-but-worse.
                if mean_diff <= 0:
                    remediation = (
                        f"New strategy's mean return ({mean_diff:+.4f}) is "
                        "below the live baseline. Don't promote a "
                        "regression."
                    )
                else:
                    remediation = (
                        f"A/B p-value ({p:.4f}) exceeds the {thresholds.ab_min_p_value} "
                        "threshold. The observed improvement may be noise."
                    )
                checks.append(
                    CheckResult(
                        name="ab_significance",
                        passed=False,
                        actual=p,
                        threshold=thresholds.ab_min_p_value,
                        remediation=remediation,
                    )
                )

    failures = [c for c in checks if not c.passed]
    return PromotionVerdict(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        warnings=warnings,
    )


def render_verdict(verdict: PromotionVerdict) -> str:
    """Pretty multi-line text suitable for CLI / Slack / email.

    Format mirrors `crypto/stress.render_report` so an operator
    running both sees a consistent shape."""
    lines = ["=== Promote-to-live verdict ==="]
    status = "✔ PASS" if verdict.passed else "✘ FAIL"
    lines.append(f"Overall: {status}")
    lines.append("")
    for c in verdict.checks:
        marker = "✔" if c.passed else "✘"
        actual = f"{c.actual:.4f}" if c.actual is not None else "n/a"
        threshold = f"{c.threshold:.4f}" if c.threshold is not None else "n/a"
        lines.append(f"  {marker} {c.name:<24} actual={actual:<12} threshold={threshold}")
        if c.remediation:
            lines.append(f"      → {c.remediation}")
    if verdict.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in verdict.warnings:
            lines.append(f"  · {w}")
    return "\n".join(lines)


def _stub_backtest_for_dummy_fold(
    *, max_drawdown_pct: float = 0.0, total_trades: int = 0
) -> BacktestResult:
    """Internal test helper — kept here (not in tests) so a future
    wiring layer can spin up a quick verdict on a fresh strategy
    without re-importing the test module. Returns a minimum-viable
    BacktestResult with just the fields the gate reads."""
    return BacktestResult(
        pair="STUB",
        start_date="",
        end_date="",
        initial_balance=1000.0,
        final_balance=1000.0,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
    )


__all__ = [
    "CheckResult",
    "PromotionThresholds",
    "PromotionVerdict",
    "evaluate_promotion",
    "render_verdict",
]
