"""Continuous-integration pipeline for ML models.

Round-4 wave 6.D: every model commit (or candidate produced by
the Wave 4.B GA) needs a *checked* set of gates before it can
replace the production model. This module is the orchestrator:

* **Sharpe regression check** — candidate's Sharpe must not be
  meaningfully worse than the incumbent's (default: ≥ 90% of
  incumbent, i.e. no >10% regression).
* **Win-rate regression check** — symmetric guard on win rate.
* **Drift comparison** — candidate's per-trade-return distribution
  shouldn't show a meaningful population shift vs the incumbent
  (KS-test–style: a max absolute distance between empirical CDFs
  capped by `max_distribution_distance`).
* **Walk-forward gate** — composes Wave 4.F's `evaluate_promotion`
  on the candidate's walk-forward report, so the WF acceptance
  criteria from a strategy author's CI run are the same the
  promotion gate uses at promote-to-live time.

Each check returns a `GateOutcome` carrying pass/fail + a
remediation hint; the aggregate `CIPipelineReport` collates them
into the operator-facing pass/fail. A single failure blocks
promotion (additive composition — pin: more gates → more
potential failures, never less).

Why a separate orchestrator rather than extending
`promotion_gate.py`:

* `promotion_gate.py` answers "should this candidate go live?"
  (the live-trading gate). 6.D answers "does this candidate pass
  CI?" (the *commit-time* gate, run before the human even
  reviews). They share thresholds but compose differently —
  CI runs against a baseline (incumbent), not against absolute
  bars.
* CI bundles a regression test (vs incumbent) that the live gate
  doesn't run — promotion gate only checks absolute thresholds
  + an optional A/B; CI specifically guards against silent
  regressions.

Halal alignment: the CI pipeline is metadata only. It never
opens a position. A failing CI run blocks the promotion to live
trading; the existing kill-switch + halt mechanism remains the
final authority.

Pure-Python; no NumPy / scipy / DB / async. Operates on already-
collected report dataclasses + the Wave 4.F promotion gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from halal_trader.core.promotion_gate import (
    PromotionThresholds,
    PromotionVerdict,
    evaluate_promotion,
)
from halal_trader.crypto.walkforward import (
    MonteCarloReport,
    WalkForwardReport,
)

# ── Vocabulary ────────────────────────────────────────────


@dataclass(frozen=True)
class CIThresholds:
    """Operator-configured CI bars.

    Defaults capture the roadmap's "block promote-to-live if any
    metric regresses by > 10%" rule:

    * ``min_sharpe_ratio`` (default 0.90) — candidate's Sharpe must
      be at least 90% of incumbent's.
    * ``min_win_rate_ratio`` (default 0.95) — win rate must be at
      least 95% (more sensitive than Sharpe to small sample
      variability, so the floor is tighter).
    * ``max_distribution_distance`` (default 0.30) — KS-style
      max-CDF-distance; > 0.30 means the per-trade-return
      distributions are meaningfully different.
    * ``min_sample_size`` (default 20) — below this, drift
      comparison and Sharpe ratio checks return PASS with a
      "skipped" note rather than alarming on noise.
    """

    min_sharpe_ratio: float = 0.90
    min_win_rate_ratio: float = 0.95
    max_distribution_distance: float = 0.30
    min_sample_size: int = 20

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_sharpe_ratio <= 1.0:
            raise ValueError(f"min_sharpe_ratio must be in [0, 1]; got {self.min_sharpe_ratio}")
        if not 0.0 <= self.min_win_rate_ratio <= 1.0:
            raise ValueError(f"min_win_rate_ratio must be in [0, 1]; got {self.min_win_rate_ratio}")
        if not 0.0 < self.max_distribution_distance <= 1.0:
            raise ValueError(
                f"max_distribution_distance must be in (0, 1]; got {self.max_distribution_distance}"
            )
        if self.min_sample_size < 1:
            raise ValueError(f"min_sample_size must be >= 1; got {self.min_sample_size}")


@dataclass(frozen=True)
class GateOutcome:
    """One named CI check's result."""

    name: str
    passed: bool
    actual: float | None
    threshold: float | None
    remediation: str = ""

    @property
    def is_skipped(self) -> bool:
        """A check that ran but couldn't produce a verdict (e.g.
        no incumbent to compare against, sample too small) is
        recorded as passed=True with a "skipped" remediation note.
        Pin: the orchestrator never *fails* on a skip — operators
        running a fresh model with no incumbent shouldn't be
        gated."""
        return self.passed and self.remediation.startswith("skipped")


@dataclass(frozen=True)
class CIPipelineReport:
    """Aggregate verdict over every CI gate.

    ``passed`` is True iff every gate passed (skipped counts as
    pass — see `GateOutcome.is_skipped`). ``failures`` is the
    subset that hard-failed; the dashboard renders these
    prominently. ``promotion_verdict`` is the underlying Wave 4.F
    verdict for the walk-forward layer, exposed so the operator
    can drill in.
    """

    passed: bool
    gates: list[GateOutcome]
    failures: list[GateOutcome] = field(default_factory=list)
    promotion_verdict: PromotionVerdict | None = None
    summary: str = ""


# ── Helpers ───────────────────────────────────────────────


def _empirical_cdf(values: list[float]) -> list[tuple[float, float]]:
    """Build `(x, F(x))` pairs for a step-CDF.

    Pin: the CDF is right-continuous (a value `x` in the sample
    contributes 1/N to F(x) at that x). Used for the KS-style
    distance check; we don't need a parametric distribution
    because the bot's per-trade returns are heavy-tailed."""
    if not values:
        return []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return [(v, (i + 1) / n) for i, v in enumerate(sorted_vals)]


def _ks_distance(a: list[float], b: list[float]) -> float | None:
    """Maximum absolute distance between the empirical CDFs of
    two samples. Returns None when either sample is empty.

    This is the textbook two-sample KS statistic, computed
    without scipy: at each unique observation x, the absolute
    difference |F_a(x) - F_b(x)| is a candidate; the max is the
    statistic.

    Pin: the values returned by both CDFs at any point are in
    [0, 1], so the distance is bounded — `max_distribution_distance`
    threshold of 0.30 means "the two distributions disagree by
    more than 30 percentage points at some quantile"."""
    if not a or not b:
        return None
    # Combine all unique x values; at each, evaluate both CDFs.
    all_x = sorted(set(a + b))
    sorted_a = sorted(a)
    sorted_b = sorted(b)
    n_a = len(sorted_a)
    n_b = len(sorted_b)
    max_dist = 0.0
    for x in all_x:
        # Right-continuous CDF: count of values ≤ x.
        f_a = _count_le(sorted_a, x) / n_a
        f_b = _count_le(sorted_b, x) / n_b
        dist = abs(f_a - f_b)
        if dist > max_dist:
            max_dist = dist
    return max_dist


def _count_le(sorted_vals: list[float], x: float) -> int:
    """Right-continuous count: how many values in sorted_vals
    are ≤ x. Linear scan; the corpus is small enough that
    binary-search optimisation isn't worth the complexity."""
    count = 0
    for v in sorted_vals:
        if v <= x:
            count += 1
        else:
            break
    return count


# ── Per-gate checks ───────────────────────────────────────


def check_sharpe_regression(
    *,
    candidate_sharpe: float,
    incumbent_sharpe: float | None,
    thresholds: CIThresholds,
) -> GateOutcome:
    """Pin: incumbent=None means cold-start; check passes with a
    "skipped" note. The operator's first model has no baseline
    to regress against."""
    if incumbent_sharpe is None:
        return GateOutcome(
            name="sharpe_regression",
            passed=True,
            actual=candidate_sharpe,
            threshold=None,
            remediation="skipped: no incumbent Sharpe to compare against",
        )
    if incumbent_sharpe <= 0:
        # A non-positive incumbent isn't a real baseline; skip
        # rather than divide-by-zero on the ratio.
        return GateOutcome(
            name="sharpe_regression",
            passed=True,
            actual=candidate_sharpe,
            threshold=None,
            remediation="skipped: incumbent Sharpe is non-positive",
        )
    ratio = candidate_sharpe / incumbent_sharpe
    if ratio >= thresholds.min_sharpe_ratio:
        return GateOutcome(
            name="sharpe_regression",
            passed=True,
            actual=ratio,
            threshold=thresholds.min_sharpe_ratio,
        )
    return GateOutcome(
        name="sharpe_regression",
        passed=False,
        actual=ratio,
        threshold=thresholds.min_sharpe_ratio,
        remediation=(
            f"Sharpe regressed to {ratio:.2%} of incumbent "
            f"({candidate_sharpe:.4f} vs {incumbent_sharpe:.4f}); "
            f"investigate before promoting."
        ),
    )


def check_win_rate_regression(
    *,
    candidate_win_rate: float,
    incumbent_win_rate: float | None,
    thresholds: CIThresholds,
) -> GateOutcome:
    if incumbent_win_rate is None:
        return GateOutcome(
            name="win_rate_regression",
            passed=True,
            actual=candidate_win_rate,
            threshold=None,
            remediation="skipped: no incumbent win-rate to compare against",
        )
    if incumbent_win_rate <= 0:
        return GateOutcome(
            name="win_rate_regression",
            passed=True,
            actual=candidate_win_rate,
            threshold=None,
            remediation="skipped: incumbent win rate is non-positive",
        )
    ratio = candidate_win_rate / incumbent_win_rate
    if ratio >= thresholds.min_win_rate_ratio:
        return GateOutcome(
            name="win_rate_regression",
            passed=True,
            actual=ratio,
            threshold=thresholds.min_win_rate_ratio,
        )
    return GateOutcome(
        name="win_rate_regression",
        passed=False,
        actual=ratio,
        threshold=thresholds.min_win_rate_ratio,
        remediation=(
            f"Win rate regressed to {ratio:.2%} of incumbent "
            f"({candidate_win_rate:.2%} vs {incumbent_win_rate:.2%})."
        ),
    )


def check_distribution_drift(
    *,
    candidate_returns: list[float],
    incumbent_returns: list[float] | None,
    thresholds: CIThresholds,
) -> GateOutcome:
    """KS-style distribution comparison.

    Pin: when either sample is below `min_sample_size`, the check
    skips with a "small sample" note — a KS distance on 10 points
    is mostly noise."""
    if incumbent_returns is None:
        return GateOutcome(
            name="distribution_drift",
            passed=True,
            actual=None,
            threshold=thresholds.max_distribution_distance,
            remediation="skipped: no incumbent returns to compare against",
        )
    if (
        len(candidate_returns) < thresholds.min_sample_size
        or len(incumbent_returns) < thresholds.min_sample_size
    ):
        return GateOutcome(
            name="distribution_drift",
            passed=True,
            actual=None,
            threshold=thresholds.max_distribution_distance,
            remediation=(
                f"skipped: sample size below {thresholds.min_sample_size} "
                f"(candidate={len(candidate_returns)}, "
                f"incumbent={len(incumbent_returns)})"
            ),
        )
    distance = _ks_distance(candidate_returns, incumbent_returns)
    if distance is None or distance <= thresholds.max_distribution_distance:
        return GateOutcome(
            name="distribution_drift",
            passed=True,
            actual=distance,
            threshold=thresholds.max_distribution_distance,
        )
    return GateOutcome(
        name="distribution_drift",
        passed=False,
        actual=distance,
        threshold=thresholds.max_distribution_distance,
        remediation=(
            f"Per-trade-return distribution drifted "
            f"(KS distance {distance:.3f} > "
            f"{thresholds.max_distribution_distance:.3f}); the "
            f"candidate behaves materially differently."
        ),
    )


# ── Pipeline driver ──────────────────────────────────────


def run_ci(
    *,
    candidate_walk_forward: WalkForwardReport,
    candidate_sharpe: float,
    candidate_win_rate: float,
    candidate_returns: list[float],
    incumbent_sharpe: float | None = None,
    incumbent_win_rate: float | None = None,
    incumbent_returns: list[float] | None = None,
    thresholds: CIThresholds | None = None,
    promotion_thresholds: PromotionThresholds | None = None,
    candidate_monte_carlo: MonteCarloReport | None = None,
) -> CIPipelineReport:
    """Run every CI gate against the candidate.

    The walk-forward layer composes Wave 4.F's `evaluate_promotion`
    so a candidate that passes CI is also pre-cleared on the
    absolute walk-forward bars the promote-to-live gate uses.

    Pin: skipped checks (no incumbent, small sample) count as
    PASS but are flagged in the gate's `remediation` field. The
    aggregate `passed` is True iff every gate passed (skip or
    real-pass).
    """
    t = thresholds or CIThresholds()
    gates: list[GateOutcome] = []

    # 1. Walk-forward acceptance (composes Wave 4.F).
    promotion_verdict = evaluate_promotion(
        candidate_walk_forward,
        thresholds=promotion_thresholds,
        monte_carlo=candidate_monte_carlo,
    )
    gates.append(
        GateOutcome(
            name="walk_forward",
            passed=promotion_verdict.passed,
            actual=None,
            threshold=None,
            remediation=(
                ""
                if promotion_verdict.passed
                else "; ".join(f.remediation for f in promotion_verdict.failures)
            ),
        )
    )

    # 2. Sharpe regression check.
    gates.append(
        check_sharpe_regression(
            candidate_sharpe=candidate_sharpe,
            incumbent_sharpe=incumbent_sharpe,
            thresholds=t,
        )
    )

    # 3. Win-rate regression check.
    gates.append(
        check_win_rate_regression(
            candidate_win_rate=candidate_win_rate,
            incumbent_win_rate=incumbent_win_rate,
            thresholds=t,
        )
    )

    # 4. Distribution drift check.
    gates.append(
        check_distribution_drift(
            candidate_returns=candidate_returns,
            incumbent_returns=incumbent_returns,
            thresholds=t,
        )
    )

    failures = [g for g in gates if not g.passed]
    overall_passed = len(failures) == 0
    summary = _build_summary(overall_passed, gates)

    return CIPipelineReport(
        passed=overall_passed,
        gates=gates,
        failures=failures,
        promotion_verdict=promotion_verdict,
        summary=summary,
    )


def _build_summary(passed: bool, gates: list[GateOutcome]) -> str:
    skipped = sum(1 for g in gates if g.is_skipped)
    if passed:
        if skipped:
            return f"PASS — {len(gates)} gates ran, {skipped} skipped"
        return f"PASS — all {len(gates)} gates green"
    fail_count = sum(1 for g in gates if not g.passed)
    return f"FAIL — {fail_count} of {len(gates)} gates failed"


# ── Render helper ─────────────────────────────────────────


def render_report(report: CIPipelineReport) -> str:
    """CLI / Slack-ready text payload visually consistent with
    `core/promotion_gate.render_verdict` so an operator running
    both sees a familiar shape."""
    lines = ["=== ML CI pipeline ==="]
    status = "✔ PASS" if report.passed else "✘ FAIL"
    lines.append(f"Overall: {status}")
    lines.append("")
    for g in report.gates:
        marker = "✔" if g.passed else "✘"
        if g.is_skipped:
            marker = "—"
        actual = f"{g.actual:.4f}" if g.actual is not None else "n/a"
        threshold = f"{g.threshold:.4f}" if g.threshold is not None else "n/a"
        lines.append(f"  {marker} {g.name:<24} actual={actual:<12} threshold={threshold}")
        if g.remediation:
            lines.append(f"      → {g.remediation}")
    return "\n".join(lines)


__all__ = [
    "CIPipelineReport",
    "CIThresholds",
    "GateOutcome",
    "check_distribution_drift",
    "check_sharpe_regression",
    "check_win_rate_regression",
    "render_report",
    "run_ci",
]
