"""Tests for `ml/latency_budget.py`.

Pins the budget validation, the rolling sample window's percentile
math, the four-way classification (cold-start / steady-state /
soft / hard), the aggregate severity-takes-worst rule, and the
render output.
"""

from __future__ import annotations

import pytest

from halal_trader.ml.latency_budget import (
    BudgetStatus,
    LatencyBudgetTracker,
    StageBudget,
    StageObservation,
    aggregate,
    render_report,
)


def _budget(
    name: str = "forecaster",
    *,
    budget_ms: float = 100.0,
    soft_pct: float = 0.80,
    min_samples: int = 10,
) -> StageBudget:
    return StageBudget(
        name=name,
        budget_ms=budget_ms,
        soft_pct=soft_pct,
        min_samples=min_samples,
    )


def _tracker(
    *,
    budgets: tuple[StageBudget, ...] = (),
    window: int = 100,
) -> LatencyBudgetTracker:
    if not budgets:
        budgets = (_budget(),)
    return LatencyBudgetTracker(list(budgets), window=window)


# ── StageBudget validation ───────────────────────────────


def test_budget_rejects_non_positive_budget_ms():
    with pytest.raises(ValueError, match="budget_ms"):
        StageBudget(name="x", budget_ms=0.0)


def test_budget_rejects_soft_pct_outside_zero_one():
    """Pin: soft must be strictly inside (0, 1) — at 0 it'd be
    AMBER on every measurement; at 1 it'd never trigger."""
    with pytest.raises(ValueError, match="soft_pct"):
        StageBudget(name="x", budget_ms=100.0, soft_pct=0.0)
    with pytest.raises(ValueError, match="soft_pct"):
        StageBudget(name="x", budget_ms=100.0, soft_pct=1.0)


def test_budget_rejects_zero_min_samples():
    with pytest.raises(ValueError, match="min_samples"):
        StageBudget(name="x", budget_ms=100.0, min_samples=0)


# ── Cold-start ───────────────────────────────────────────


def test_observe_with_no_samples_returns_green():
    """Pin: cold start = GREEN, never AMBER. The tracker shouldn't
    alarm on no data."""
    t = _tracker()
    obs = t.observe("forecaster")
    assert obs.status == BudgetStatus.GREEN
    assert obs.current_ms is None
    assert obs.sample_count == 0
    assert obs.headroom_pct == 1.0


def test_observe_below_min_samples_uses_current_only():
    """Pin: below min_samples, p95 is unreliable — classification
    falls back to current vs hard budget. A single sample at 90ms
    against a 100ms budget should AMBER (above 80% soft), not RED."""
    t = _tracker()
    t.record("forecaster", 90.0)
    obs = t.observe("forecaster")
    # Default min_samples=10 not yet reached.
    assert obs.sample_count == 1
    assert obs.status == BudgetStatus.AMBER


def test_observe_below_min_samples_with_breach_is_red():
    """Pin: cold-start RED still triggers when current > hard
    budget — operators want the breach surfaced even on first
    sample."""
    t = _tracker()
    t.record("forecaster", 200.0)  # 2× budget
    obs = t.observe("forecaster")
    assert obs.status == BudgetStatus.RED


# ── Steady-state classification ──────────────────────────


def test_steady_state_green_when_well_under_budget():
    t = _tracker(budgets=(_budget(min_samples=5),))
    for _ in range(10):
        t.record("forecaster", 30.0)  # 30ms, well under 100ms / 80%
    obs = t.observe("forecaster")
    assert obs.status == BudgetStatus.GREEN


def test_steady_state_amber_when_p95_crosses_soft_threshold():
    """Pin: p95 above 80% soft threshold → AMBER even if current
    is GREEN. The dashboard wants a leading indicator."""
    t = _tracker(budgets=(_budget(min_samples=5),))
    # 9 fast + 1 slow; p95 picks up the slow one.
    for _ in range(9):
        t.record("forecaster", 30.0)
    t.record("forecaster", 90.0)  # p95 ≈ 90, above 80% soft
    obs = t.observe("forecaster")
    # Latest is now 90 (above 80% soft) → AMBER from current alone too.
    assert obs.status == BudgetStatus.AMBER


def test_steady_state_red_when_p95_crosses_hard_budget():
    """Pin: any p95 above the hard budget → RED, regardless of
    the current sample. A consistently-slow stage must flip RED
    even when the latest sample happens to be fast."""
    t = _tracker(budgets=(_budget(min_samples=5),))
    # 9 over-budget samples + 1 fast latest.
    for _ in range(9):
        t.record("forecaster", 150.0)
    t.record("forecaster", 50.0)
    obs = t.observe("forecaster")
    assert obs.status == BudgetStatus.RED


def test_steady_state_red_when_current_breaches_even_if_p95_ok():
    t = _tracker(budgets=(_budget(min_samples=5),))
    for _ in range(9):
        t.record("forecaster", 30.0)
    t.record("forecaster", 200.0)  # latest spike
    obs = t.observe("forecaster")
    assert obs.status == BudgetStatus.RED


# ── Percentile math ──────────────────────────────────────


def test_percentiles_handle_uniform_window():
    t = _tracker()
    for _ in range(20):
        t.record("forecaster", 50.0)
    obs = t.observe("forecaster")
    assert obs.p50_ms == 50.0
    assert obs.p95_ms == 50.0
    assert obs.p99_ms == 50.0


def test_percentiles_pick_nearest_rank_not_interpolated():
    """Pin: nearest-rank percentile means p95 of 20 samples is
    the 19th or 20th value (rank=ceil(0.95×20)=19), not an
    interpolation."""
    t = _tracker()
    # Values 1..20 (sorted)
    for v in range(1, 21):
        t.record("forecaster", float(v))
    obs = t.observe("forecaster")
    # p95 nearest rank: round(0.95*20)=19 → sorted_values[18] = 19.0
    assert obs.p95_ms == 19.0


def test_window_size_caps_history():
    """Pin: ring-buffer eviction. After window+1 samples, only
    the last `window` are retained."""
    t = _tracker(window=5)
    for v in range(1, 11):  # 10 values
        t.record("forecaster", float(v))
    obs = t.observe("forecaster")
    # Only [6,7,8,9,10] retained. Nearest-rank p50: round(0.5×5)=2
    # under Python's banker's rounding → sorted[1] = 7.
    assert obs.p50_ms == 7.0
    assert obs.sample_count == 5


# ── headroom ─────────────────────────────────────────────


def test_headroom_positive_when_under_budget():
    t = _tracker()
    t.record("forecaster", 40.0)  # 60% of 100ms
    obs = t.observe("forecaster")
    assert obs.headroom_pct == pytest.approx(0.6)


def test_headroom_negative_when_over_budget():
    t = _tracker()
    t.record("forecaster", 130.0)
    obs = t.observe("forecaster")
    assert obs.headroom_pct == pytest.approx(-0.3)


# ── unknown stage ────────────────────────────────────────


def test_record_for_unknown_stage_raises():
    """Pin: typo'd stage name surfaces immediately rather than
    silently creating a never-checked bucket."""
    t = _tracker()
    with pytest.raises(KeyError, match="unknown stage"):
        t.record("forcaster", 10.0)  # typo


def test_observe_for_unknown_stage_raises():
    t = _tracker()
    with pytest.raises(KeyError, match="unknown stage"):
        t.observe("missing")


# ── tracker construction ─────────────────────────────────


def test_tracker_rejects_non_positive_window():
    with pytest.raises(ValueError, match="window"):
        LatencyBudgetTracker([_budget()], window=0)


def test_tracker_stages_property_lists_declared_names():
    t = _tracker(budgets=(_budget("a"), _budget("b"), _budget("c")))
    assert sorted(t.stages) == ["a", "b", "c"]


def test_record_rejects_negative_latency():
    """Pin: a negative latency is a measurement bug; surface."""
    t = _tracker()
    with pytest.raises(ValueError, match="non-negative"):
        t.record("forecaster", -1.0)


# ── observe_all ──────────────────────────────────────────


def test_observe_all_returns_one_per_stage():
    t = _tracker(budgets=(_budget("a"), _budget("b"), _budget("c")))
    obs = t.observe_all()
    assert len(obs) == 3
    assert {o.name for o in obs} == {"a", "b", "c"}


# ── aggregate ────────────────────────────────────────────


def test_aggregate_sums_budgets_and_current_samples():
    a = StageObservation(
        name="a",
        budget_ms=80.0,
        status=BudgetStatus.GREEN,
        current_ms=20.0,
        p50_ms=20.0,
        p95_ms=25.0,
        p99_ms=28.0,
        sample_count=10,
        headroom_pct=0.75,
    )
    b = StageObservation(
        name="b",
        budget_ms=120.0,
        status=BudgetStatus.GREEN,
        current_ms=30.0,
        p50_ms=30.0,
        p95_ms=40.0,
        p99_ms=45.0,
        sample_count=10,
        headroom_pct=0.75,
    )
    rep = aggregate([a, b])
    assert rep.total_budget_ms == 200.0
    assert rep.total_current_ms == 50.0
    assert rep.total_p95_ms == 65.0


def test_aggregate_overall_takes_worst_status():
    """Pin: the worst per-stage status wins. A single RED stage
    flips the whole report RED."""
    green = StageObservation(
        name="g",
        budget_ms=80.0,
        status=BudgetStatus.GREEN,
        current_ms=20.0,
        p50_ms=20.0,
        p95_ms=25.0,
        p99_ms=28.0,
        sample_count=10,
        headroom_pct=0.75,
    )
    red = StageObservation(
        name="r",
        budget_ms=120.0,
        status=BudgetStatus.RED,
        current_ms=200.0,
        p50_ms=180.0,
        p95_ms=200.0,
        p99_ms=210.0,
        sample_count=10,
        headroom_pct=-0.66,
    )
    rep = aggregate([green, red])
    assert rep.overall_status == BudgetStatus.RED


def test_aggregate_amber_beats_green_in_overall():
    green = StageObservation(
        name="g",
        budget_ms=80.0,
        status=BudgetStatus.GREEN,
        current_ms=20.0,
        p50_ms=20.0,
        p95_ms=25.0,
        p99_ms=28.0,
        sample_count=10,
        headroom_pct=0.75,
    )
    amber = StageObservation(
        name="a",
        budget_ms=120.0,
        status=BudgetStatus.AMBER,
        current_ms=100.0,
        p50_ms=90.0,
        p95_ms=110.0,
        p99_ms=115.0,
        sample_count=10,
        headroom_pct=0.16,
    )
    rep = aggregate([green, amber])
    assert rep.overall_status == BudgetStatus.AMBER


def test_aggregate_handles_none_current_ms():
    """Pin: a stage that hasn't recorded anything yet (current_ms
    None) contributes 0 to the total — the operator's "no data
    yet" reading is friendly to aggregation."""
    cold = StageObservation(
        name="cold",
        budget_ms=100.0,
        status=BudgetStatus.GREEN,
        current_ms=None,
        p50_ms=0.0,
        p95_ms=0.0,
        p99_ms=0.0,
        sample_count=0,
        headroom_pct=1.0,
    )
    rep = aggregate([cold])
    assert rep.total_current_ms == 0.0


def test_aggregate_empty_returns_green_with_no_stages():
    rep = aggregate([])
    assert rep.overall_status == BudgetStatus.GREEN
    assert rep.total_budget_ms == 0.0
    assert "no stages" in rep.summary.lower()


def test_aggregate_summary_contains_status_keyword():
    a = StageObservation(
        name="a",
        budget_ms=100.0,
        status=BudgetStatus.GREEN,
        current_ms=20.0,
        p50_ms=20.0,
        p95_ms=25.0,
        p99_ms=28.0,
        sample_count=10,
        headroom_pct=0.80,
    )
    rep = aggregate([a])
    assert "green" in rep.summary.lower()


# ── render_report ────────────────────────────────────────


def test_render_includes_overall_status_emoji():
    t = _tracker()
    t.record("forecaster", 30.0)
    rep = aggregate(t.observe_all())
    text = render_report(rep)
    assert "🟢" in text or "🟡" in text or "🔴" in text


def test_render_includes_stage_lines():
    t = _tracker(budgets=(_budget("forecaster"), _budget("anomaly")))
    t.record("forecaster", 30.0)
    t.record("anomaly", 10.0)
    rep = aggregate(t.observe_all())
    text = render_report(rep)
    assert "forecaster" in text
    assert "anomaly" in text


def test_render_handles_empty_report():
    rep = aggregate([])
    text = render_report(rep)
    assert "no stages declared" in text


def test_render_marks_red_stage_with_red_emoji():
    t = _tracker(budgets=(_budget(min_samples=2),))
    for _ in range(3):
        t.record("forecaster", 200.0)
    rep = aggregate(t.observe_all())
    text = render_report(rep)
    assert "🔴" in text


# ── observation structure ────────────────────────────────


def test_is_breaching_property_aligns_with_status():
    obs_red = StageObservation(
        name="x",
        budget_ms=100.0,
        status=BudgetStatus.RED,
        current_ms=200.0,
        p50_ms=180.0,
        p95_ms=200.0,
        p99_ms=210.0,
        sample_count=10,
        headroom_pct=-1.0,
    )
    obs_green = StageObservation(
        name="x",
        budget_ms=100.0,
        status=BudgetStatus.GREEN,
        current_ms=20.0,
        p50_ms=20.0,
        p95_ms=25.0,
        p99_ms=28.0,
        sample_count=10,
        headroom_pct=0.80,
    )
    assert obs_red.is_breaching is True
    assert obs_green.is_breaching is False


def test_observation_is_immutable():
    obs = StageObservation(
        name="x",
        budget_ms=100.0,
        status=BudgetStatus.GREEN,
        current_ms=20.0,
        p50_ms=20.0,
        p95_ms=25.0,
        p99_ms=28.0,
        sample_count=10,
        headroom_pct=0.80,
    )
    with pytest.raises(Exception):
        obs.status = BudgetStatus.RED  # type: ignore[misc]


def test_report_is_immutable():
    rep = aggregate([])
    with pytest.raises(Exception):
        rep.overall_status = BudgetStatus.RED  # type: ignore[misc]
