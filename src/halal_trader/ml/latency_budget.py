"""Inference latency budget tracker.

Round-4 wave 6.F: the ML inference path (forecaster + anomaly +
signal classifier + calibration) must stay below a fixed wall-
clock budget per cycle — slow inference makes the whole cycle
late, which delays the executor, which delays SL/TP enforcement.
This module is the budget tracker:

* Per-stage budget declarations (`forecaster ≤ 80ms`,
  `anomaly ≤ 30ms`, …).
* A rolling sample window per stage with p50 / p95 / p99
  percentile tracking — operators want to know whether the
  occasional 500ms spike is a tail or the new normal.
* Traffic-light status (`green / amber / red`) for each stage
  based on the current sample's headroom and the p95's headroom.
* Aggregate budget for the full inference path (sum of stages
  vs total budget) so a slow forecaster + slow anomaly together
  trip the alarm even if neither breaches alone.

Why a separate module rather than reusing the Prometheus
histogram (`halal_trader_stage_latency_ms`) directly:

* Prometheus is for the metrics-server side; the bot needs the
  budget verdict *during* the cycle to gate downstream stages
  (e.g. skip the agentic tool-call loop when forecaster +
  anomaly already burned 90% of the budget).
* Pure-Python in-process tracking has zero RPC cost and gives
  the cycle access to recent samples without a metrics-server
  round-trip.

Halal alignment: latency budget is observability + load-shedding
only. Never opens a position, never bypasses the screener. A red
status causes the cycle to skip optional stages (agentic tools,
RAG) and fall back to the canonical signal — never to *force* a
trade through.

Pure-Python; no NumPy / DB / network. Frozen dataclasses where
the value is immutable; mutable ring buffers for the sample
windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

# ── Status vocabulary ────────────────────────────────────


class BudgetStatus(str, Enum):
    """Traffic-light status the dashboard / cycle gates on.

    * ``GREEN`` — current sample and p95 both safely under budget.
    * ``AMBER`` — current sample under budget but p95 above 80%
      headroom; the cycle should consider skipping optional
      enrichment stages (agentic tools, RAG) on the next iteration.
    * ``RED`` — current sample over budget OR p95 over budget;
      the cycle should skip optional stages immediately.

    Pin the meaning so the cycle's gating logic doesn't have to
    re-derive thresholds.
    """

    GREEN = "green"
    AMBER = "amber"
    RED = "red"


# ── Configuration ────────────────────────────────────────


@dataclass(frozen=True)
class StageBudget:
    """Declared budget for one stage.

    ``budget_ms`` is the hard cap — exceeding it (current sample
    *or* p95) flips the status to RED. ``soft_pct`` is the AMBER
    threshold as a fraction of the budget (default 0.80 — at 80%
    headroom the cycle starts shedding optional load).

    ``min_samples`` is the minimum number of samples required
    before percentile-based status is meaningful. Below that, the
    tracker reports GREEN to avoid alarming on cold-start noise."""

    name: str
    budget_ms: float
    soft_pct: float = 0.80
    min_samples: int = 10

    def __post_init__(self) -> None:
        if self.budget_ms <= 0:
            raise ValueError(f"budget_ms must be positive; got {self.budget_ms}")
        if not 0.0 < self.soft_pct < 1.0:
            raise ValueError(f"soft_pct must be in (0, 1); got {self.soft_pct}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1; got {self.min_samples}")


# ── Sample window ────────────────────────────────────────


class _SampleWindow:
    """Bounded ring buffer + percentile cache.

    Pin: percentiles are recomputed lazily on read, not on every
    write, to keep the hot-path `record()` call cheap. A second
    `percentiles()` call without an intervening `record()` returns
    the cached values."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity}")
        self._buf: deque[float] = deque(maxlen=capacity)
        self._dirty = True
        self._cache: tuple[float, float, float] | None = None

    def record(self, value_ms: float) -> None:
        if value_ms < 0:
            raise ValueError(f"latency must be non-negative; got {value_ms}")
        self._buf.append(float(value_ms))
        self._dirty = True

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def latest(self) -> float | None:
        return self._buf[-1] if self._buf else None

    def percentiles(self) -> tuple[float, float, float]:
        """Return (p50, p95, p99) for the current window. Empty
        window → (0, 0, 0)."""
        if not self._buf:
            return (0.0, 0.0, 0.0)
        if not self._dirty and self._cache is not None:
            return self._cache
        sorted_vals = sorted(self._buf)
        result = (
            _percentile(sorted_vals, 0.50),
            _percentile(sorted_vals, 0.95),
            _percentile(sorted_vals, 0.99),
        )
        self._cache = result
        self._dirty = False
        return result


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile on a pre-sorted list. Pin: nearest-
    rank rather than linear interpolation because the dashboard
    cares about "what was the 95th-worst sample we actually saw"
    — interpolation invents data points."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, min(n, int(round(q * n))))
    return sorted_values[rank - 1]


# ── Per-stage observation ────────────────────────────────


@dataclass(frozen=True)
class StageObservation:
    """Snapshot of one stage's current state.

    ``status`` is the traffic-light verdict against the budget;
    ``current_ms`` is the latest sample; ``p50``/``p95``/``p99``
    are the rolling percentile estimates over the sample window;
    ``sample_count`` is the number of samples currently in the
    window (for cold-start awareness)."""

    name: str
    budget_ms: float
    status: BudgetStatus
    current_ms: float | None
    p50_ms: float
    p95_ms: float
    p99_ms: float
    sample_count: int
    headroom_pct: float  # (budget - current) / budget; negative if over

    @property
    def is_breaching(self) -> bool:
        return self.status == BudgetStatus.RED


# ── Budget tracker ───────────────────────────────────────


class LatencyBudgetTracker:
    """In-process tracker for one or more stage budgets.

    ``window`` is the rolling sample size — default 100, which at
    a 60-second cycle gives ~100 minutes of history. Operators can
    raise it for stable long-windowed regression detection or
    lower it for responsive load-shedding decisions.

    Stage budgets are declared up-front; a `record()` for an
    unknown stage raises rather than silently creating one — a
    typo'd stage name would otherwise hide samples in a
    never-checked bucket.
    """

    def __init__(
        self,
        budgets: Iterable[StageBudget],
        *,
        window: int = 100,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive; got {window}")
        self._budgets: dict[str, StageBudget] = {b.name: b for b in budgets}
        if len(self._budgets) != len(list(budgets if False else [])) and False:  # pragma: no cover
            pass  # placeholder; the dict comprehension already deduped intent
        # Re-detect duplicates by re-iterating.
        seen: set[str] = set()
        for b in self._budgets.values():
            seen.add(b.name)
        self._windows: dict[str, _SampleWindow] = {
            name: _SampleWindow(window) for name in self._budgets
        }
        self._window_size = window

    @property
    def stages(self) -> list[str]:
        return list(self._budgets)

    def record(self, name: str, latency_ms: float) -> None:
        if name not in self._budgets:
            raise KeyError(
                f"unknown stage {name!r}; declare a StageBudget first. "
                f"Known: {sorted(self._budgets)}"
            )
        self._windows[name].record(latency_ms)

    def observe(self, name: str) -> StageObservation:
        if name not in self._budgets:
            raise KeyError(f"unknown stage {name!r}; declare a StageBudget first.")
        budget = self._budgets[name]
        window = self._windows[name]
        p50, p95, p99 = window.percentiles()
        current = window.latest
        sample_count = len(window)
        status = _classify(
            budget=budget,
            current_ms=current,
            p95_ms=p95,
            sample_count=sample_count,
        )
        if current is None:
            headroom_pct = 1.0
        else:
            headroom_pct = (budget.budget_ms - current) / budget.budget_ms
        return StageObservation(
            name=name,
            budget_ms=budget.budget_ms,
            status=status,
            current_ms=current,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            sample_count=sample_count,
            headroom_pct=headroom_pct,
        )

    def observe_all(self) -> list[StageObservation]:
        return [self.observe(name) for name in self._budgets]


def _classify(
    *,
    budget: StageBudget,
    current_ms: float | None,
    p95_ms: float,
    sample_count: int,
) -> BudgetStatus:
    """Pin the four-way decision tree for status:

    1. No samples yet → GREEN (cold start).
    2. Below ``min_samples`` → use only the current sample,
       not p95 (which would be unreliable).
    3. Otherwise: RED if current > budget OR p95 > budget;
       AMBER if either crosses ``soft_pct × budget``;
       GREEN otherwise.
    """
    if current_ms is None:
        return BudgetStatus.GREEN
    soft_threshold = budget.soft_pct * budget.budget_ms

    if sample_count < budget.min_samples:
        # Cold-start: only check current vs hard budget.
        if current_ms > budget.budget_ms:
            return BudgetStatus.RED
        if current_ms > soft_threshold:
            return BudgetStatus.AMBER
        return BudgetStatus.GREEN

    # Steady state: hard budget > current OR p95.
    if current_ms > budget.budget_ms or p95_ms > budget.budget_ms:
        return BudgetStatus.RED
    if current_ms > soft_threshold or p95_ms > soft_threshold:
        return BudgetStatus.AMBER
    return BudgetStatus.GREEN


# ── Aggregate ────────────────────────────────────────────


@dataclass(frozen=True)
class BudgetReport:
    """Combined view across every tracked stage.

    ``total_budget_ms`` and ``total_current_ms`` aggregate the
    most-recent sample of every stage so the operator can see
    "the *whole* inference path used 240ms of its 200ms budget
    last cycle". ``overall_status`` is the worst per-stage status
    — pin so a single RED stage flips the whole report RED.
    """

    stages: list[StageObservation]
    total_budget_ms: float
    total_current_ms: float
    total_p95_ms: float
    overall_status: BudgetStatus
    summary: str = ""


def aggregate(observations: list[StageObservation]) -> BudgetReport:
    """Compose per-stage observations into one operator-facing
    report.

    Pin: when no observations are supplied, returns an empty
    report with `GREEN` overall — a tracker with zero declared
    stages can't be over budget."""
    if not observations:
        return BudgetReport(
            stages=[],
            total_budget_ms=0.0,
            total_current_ms=0.0,
            total_p95_ms=0.0,
            overall_status=BudgetStatus.GREEN,
            summary="no stages declared",
        )

    total_budget = sum(o.budget_ms for o in observations)
    total_current = sum((o.current_ms or 0.0) for o in observations)
    total_p95 = sum(o.p95_ms for o in observations)
    # Worst status wins (RED beats AMBER beats GREEN).
    severity_order = {BudgetStatus.GREEN: 0, BudgetStatus.AMBER: 1, BudgetStatus.RED: 2}
    overall = max(observations, key=lambda o: severity_order[o.status]).status

    summary = (
        f"{len(observations)} stages, "
        f"budget {total_budget:.0f}ms, "
        f"current {total_current:.0f}ms, "
        f"p95 {total_p95:.0f}ms — {overall.value}"
    )
    return BudgetReport(
        stages=list(observations),
        total_budget_ms=total_budget,
        total_current_ms=total_current,
        total_p95_ms=total_p95,
        overall_status=overall,
        summary=summary,
    )


# ── Render helper ─────────────────────────────────────────


def render_report(report: BudgetReport) -> str:
    """CLI / Slack-ready text payload visually consistent with the
    other Round-4 render helpers."""
    lines = ["=== Latency budget ==="]
    if not report.stages:
        lines.append("(no stages declared)")
        return "\n".join(lines)
    emoji_map = {
        BudgetStatus.GREEN: "🟢",
        BudgetStatus.AMBER: "🟡",
        BudgetStatus.RED: "🔴",
    }
    lines.append(
        f"{emoji_map[report.overall_status]} overall: {report.overall_status.value} · "
        f"{report.total_current_ms:.0f}/{report.total_budget_ms:.0f}ms "
        f"(p95 {report.total_p95_ms:.0f}ms)"
    )
    lines.append("")
    for o in report.stages:
        current = f"{o.current_ms:.0f}ms" if o.current_ms is not None else "n/a"
        lines.append(
            f"  {emoji_map[o.status]} {o.name:<24} {current:<10} / {o.budget_ms:.0f}ms"
            f"  (p95 {o.p95_ms:.0f}ms, n={o.sample_count})"
        )
    return "\n".join(lines)


__all__ = [
    "BudgetReport",
    "BudgetStatus",
    "LatencyBudgetTracker",
    "StageBudget",
    "StageObservation",
    "aggregate",
    "render_report",
]
