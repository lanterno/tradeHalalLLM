"""Tests for the round-4 wave 7.B additions to `crypto/stress.py`.

Covers the four new generators (regime_shift / volatility_explosion /
liquidity_crunch / correlated_pair) plus their graders' integration
into `standard_scenarios()` and `grade()`.

Smoke-level: confirms the generated klines have the structural
shape each scenario advertises (a vol jump in regime_shift; near-
zero net drift in volatility_explosion; collapsed volume + wide H-L
in liquidity_crunch; observed correlation drop in correlated_pair),
and that the graders penalise / pass the right cohort of synthetic
plans.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pytest

from halal_trader.crypto.stress import (
    correlated_pair_klines,
    grade,
    liquidity_crunch_klines,
    regime_shift_klines,
    standard_scenarios,
    volatility_explosion_klines,
)

# ── Test plan shape ──────────────────────────────────────


@dataclass
class _StubDecision:
    action: str = "buy"
    confidence: float = 0.5


@dataclass
class _StubPlan:
    decisions: list = field(default_factory=list)
    market_outlook: str = ""


def _plan(action: str = "buy", confidence: float = 0.7) -> _StubPlan:
    return _StubPlan(decisions=[_StubDecision(action=action, confidence=confidence)])


def _no_op_plan() -> _StubPlan:
    return _StubPlan(decisions=[])


# ── regime_shift_klines ──────────────────────────────────


def test_regime_shift_kline_count_matches_args():
    ks = regime_shift_klines(n_calm=10, n_turbulent=15)
    assert len(ks) == 25


def test_regime_shift_actually_shifts_volatility():
    """Pin the structural promise: per-bar return std must be
    materially higher in the post-shift slice than pre-shift."""
    ks = regime_shift_klines(n_calm=40, n_turbulent=40)
    pre_returns = [(ks[i].close - ks[i - 1].close) / ks[i - 1].close for i in range(1, 40)]
    post_returns = [(ks[i].close - ks[i - 1].close) / ks[i - 1].close for i in range(40, 80)]
    pre_std = statistics.pstdev(pre_returns)
    post_std = statistics.pstdev(post_returns)
    # The default 15× bump should land at least 5× even after sampling noise.
    assert post_std > pre_std * 5


def test_regime_shift_is_seed_deterministic():
    """Stress results must be reproducible run-to-run; pin the seed
    determinism so a regression is visible as a regression."""
    a = regime_shift_klines(seed=42)
    b = regime_shift_klines(seed=42)
    assert [k.close for k in a] == [k.close for k in b]


# ── volatility_explosion_klines ─────────────────────────


def test_volatility_explosion_kline_count():
    ks = volatility_explosion_klines(n_pre=10, n_burst=20)
    assert len(ks) == 30


def test_volatility_explosion_drift_is_close_to_zero():
    """The scenario contract: high vol but ~no net drift, so the
    bot has nothing trend-y to chase. Loose tolerance — tail
    realisations can wander further than typical."""
    ks = volatility_explosion_klines(n_pre=20, n_burst=80, seed=0)
    start = ks[20].close  # right after the calm pre period
    end = ks[-1].close
    drift_pct = (end - start) / start
    # Loose bound: 50% — the scenario aims to produce *no clean
    # direction*, not zero drift. Per-bar shocks of ~2.5% can
    # accumulate; the mean-revert pull keeps the random walk from
    # exploding but doesn't force it back to start.
    assert abs(drift_pct) < 0.50


def test_volatility_explosion_bars_are_high_vol():
    """The burst section's per-bar moves should average significantly
    more than the calm pre section's."""
    ks = volatility_explosion_klines(n_pre=20, n_burst=40)
    pre_moves = [abs(ks[i].close - ks[i - 1].close) / ks[i - 1].close for i in range(1, 20)]
    burst_moves = [abs(ks[i].close - ks[i - 1].close) / ks[i - 1].close for i in range(20, 60)]
    assert statistics.mean(burst_moves) > statistics.mean(pre_moves) * 5


# ── liquidity_crunch_klines ──────────────────────────────


def test_liquidity_crunch_bars_have_collapsed_volume():
    """5% of normal — pin so a refactor doesn't accidentally
    restore full volume."""
    ks = liquidity_crunch_klines(n=20)
    avg_vol = statistics.mean(k.volume for k in ks)
    # 100 base × 0.05 = 5
    assert avg_vol == pytest.approx(5.0, abs=0.5)


def test_liquidity_crunch_bars_are_wide():
    """H-L range must be inflated relative to body size — that's
    the kline-level approximation of a wide bid-ask spread."""
    ks = liquidity_crunch_klines(n=30)
    ratios = []
    for k in ks:
        body = abs(k.close - k.open) or 1e-9
        hl_range = k.high - k.low
        ratios.append(hl_range / body)
    # Wide-bar default is body × 3 plus 0.005×close jitter; the
    # average ratio should be at least 3.
    assert statistics.mean(ratios) > 3.0


def test_liquidity_crunch_kline_count():
    ks = liquidity_crunch_klines(n=15)
    assert len(ks) == 15


# ── correlated_pair_klines ───────────────────────────────


def test_correlated_pair_returns_two_streams_of_same_length():
    a, b = correlated_pair_klines(n=50, n_breakdown_at=25)
    assert len(a) == 50
    assert len(b) == 50


def test_correlated_pair_high_correlation_pre_breakdown():
    """Pre-breakdown bars at rho=0.95 should empirically correlate
    well above 0.5 (loose bound; sampling noise on n=50)."""
    a, b = correlated_pair_klines(
        n=200, n_breakdown_at=200, pre_correlation=0.95, post_correlation=0.0
    )
    ra = [(a[i].close - a[i - 1].close) / a[i - 1].close for i in range(1, len(a))]
    rb = [(b[i].close - b[i - 1].close) / b[i - 1].close for i in range(1, len(b))]
    rho = _correlation(ra, rb)
    assert rho > 0.5


def test_correlated_pair_post_breakdown_correlation_drops():
    """Compare correlation in the pre-window (high) vs post-window
    (zero) — post should be visibly lower."""
    a, b = correlated_pair_klines(
        n=200, n_breakdown_at=100, pre_correlation=0.95, post_correlation=0.0
    )
    pre_a = [(a[i].close - a[i - 1].close) / a[i - 1].close for i in range(1, 100)]
    pre_b = [(b[i].close - b[i - 1].close) / b[i - 1].close for i in range(1, 100)]
    post_a = [(a[i].close - a[i - 1].close) / a[i - 1].close for i in range(100, 200)]
    post_b = [(b[i].close - b[i - 1].close) / b[i - 1].close for i in range(100, 200)]
    rho_pre = _correlation(pre_a, pre_b)
    rho_post = _correlation(post_a, post_b)
    assert rho_pre > rho_post + 0.3


def test_correlated_pair_rejects_invalid_correlation():
    with pytest.raises(ValueError, match="correlations"):
        correlated_pair_klines(pre_correlation=1.5)


def test_correlated_pair_rejects_invalid_breakdown_index():
    with pytest.raises(ValueError, match="n_breakdown_at"):
        correlated_pair_klines(n=50, n_breakdown_at=99)


def _correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, hand-rolled so the test is independent
    of numpy's API in case scipy goes away."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


# ── standard_scenarios extension ─────────────────────────


def test_standard_scenarios_includes_new_round4_additions():
    """Pin the names so a refactor that drops one fails loudly
    rather than silently shrinking the regression suite."""
    names = {sc.name for sc in standard_scenarios()}
    assert "regime_shift" in names
    assert "volatility_explosion" in names
    assert "liquidity_crunch" in names


def test_standard_scenarios_has_grader_for_every_entry():
    """Every scenario in the standard suite must have a grader,
    otherwise grade() falls back to severity 0 with a warning and
    the scenario silently 'passes' regardless of behaviour."""
    from halal_trader.crypto.stress import _GRADERS

    for sc in standard_scenarios():
        assert sc.name in _GRADERS, f"missing grader for {sc.name}"


# ── graders ───────────────────────────────────────────────


def test_regime_shift_grader_passes_when_no_buy():
    sc = next(s for s in standard_scenarios() if s.name == "regime_shift")
    verdict = grade(sc, _no_op_plan())
    assert verdict.severity == 0.0
    assert verdict.passed


def test_regime_shift_grader_penalises_high_confidence_buy():
    sc = next(s for s in standard_scenarios() if s.name == "regime_shift")
    verdict = grade(sc, _plan("buy", confidence=0.85))
    assert verdict.severity >= 0.5
    assert not verdict.passed


def test_regime_shift_grader_low_severity_for_low_confidence_buy():
    sc = next(s for s in standard_scenarios() if s.name == "regime_shift")
    verdict = grade(sc, _plan("buy", confidence=0.3))
    assert 0 < verdict.severity < 0.5


def test_volatility_explosion_grader_passes_on_no_op():
    sc = next(s for s in standard_scenarios() if s.name == "volatility_explosion")
    verdict = grade(sc, _no_op_plan())
    assert verdict.severity == 0.0


def test_volatility_explosion_grader_flags_confident_trade_either_direction():
    sc = next(s for s in standard_scenarios() if s.name == "volatility_explosion")
    buy_verdict = grade(sc, _plan("buy", confidence=0.85))
    sell_verdict = grade(sc, _plan("sell", confidence=0.85))
    assert buy_verdict.severity >= 0.5
    assert sell_verdict.severity >= 0.5


def test_liquidity_crunch_grader_passes_on_no_op():
    sc = next(s for s in standard_scenarios() if s.name == "liquidity_crunch")
    verdict = grade(sc, _no_op_plan())
    assert verdict.severity == 0.0


def test_liquidity_crunch_grader_penalises_high_confidence_buy():
    sc = next(s for s in standard_scenarios() if s.name == "liquidity_crunch")
    verdict = grade(sc, _plan("buy", confidence=0.85))
    assert verdict.severity >= 0.5
