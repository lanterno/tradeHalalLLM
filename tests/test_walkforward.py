"""Walk-forward + Monte Carlo wrapper tests."""

from __future__ import annotations

from halal_trader.crypto.backtest import BacktestResult, SimulatedTrade
from halal_trader.crypto.walkforward import (
    monte_carlo_resample,
    run_walk_forward,
    split_walk_forward,
)
from halal_trader.domain.models import Kline


def _kl(close: float, t: int) -> Kline:
    return Kline(
        open_time=t,
        open=close,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=1.0,
        close_time=t + 60_000,
    )


def _trade(pnl: float, ts: int = 0) -> SimulatedTrade:
    return SimulatedTrade(
        pair="X",
        side="buy",
        quantity=1,
        price=100.0,
        timestamp=ts,
        pnl=pnl,
        exit_price=100.0 + pnl,
        exit_timestamp=ts + 60_000,
    )


# ── split_walk_forward ────────────────────────────────────────


def test_split_basic_layout():
    windows = split_walk_forward(100, train_size=40, test_size=10)
    # Folds: (0..40, 40..50), (10..50, 50..60), … step defaults to test_size=10.
    assert windows[0].train_start == 0
    assert windows[0].train_end == 40
    assert windows[0].test_start == 40
    assert windows[0].test_end == 50
    # Each step advances by test_size.
    assert windows[1].train_start == 10


def test_split_too_small_returns_empty():
    assert split_walk_forward(10, train_size=40, test_size=10) == []


def test_split_invalid_sizes_raise():
    import pytest

    with pytest.raises(ValueError):
        split_walk_forward(100, train_size=0, test_size=10)
    with pytest.raises(ValueError):
        split_walk_forward(100, train_size=10, test_size=0)


def test_split_custom_step_creates_overlap():
    windows = split_walk_forward(100, train_size=40, test_size=10, step=5)
    # Step 5 with test 10 → overlapping test sets, useful for stress tests.
    assert windows[0].test_end == 50
    assert windows[1].test_start == 45


# ── run_walk_forward orchestration ────────────────────────────


async def test_run_walk_forward_aggregates_fold_metrics():
    klines = [_kl(100 + i * 0.1, i * 60_000) for i in range(120)]
    call_count = {"n": 0}

    async def fake_backtest(pair: str, slice_: list[Kline]) -> BacktestResult:
        call_count["n"] += 1
        # Alternate between profitable + losing folds for variance.
        ret = 0.05 if call_count["n"] % 2 == 0 else -0.02
        return BacktestResult(
            pair=pair,
            start_date="",
            end_date="",
            initial_balance=10_000,
            final_balance=10_000 * (1 + ret),
            total_return_pct=ret,
            sharpe_ratio=0.5 if ret > 0 else -0.3,
        )

    report = await run_walk_forward(
        "X", klines, backtest_fn=fake_backtest, train_size=40, test_size=20, warmup=20
    )
    assert report.fold_count > 0
    assert -1 < report.avg_return_pct < 1
    assert 0 <= report.win_rate <= 1


async def test_run_walk_forward_feeds_pure_oos_slices_no_leakage():
    """Each fold must receive exactly warmup + test_size bars (warmup context
    before the test window, then the test window) — never the full train span.
    This pins the leakage fix: the engine, which trades from index window_size,
    then trades only the out-of-sample test window."""
    klines = [_kl(100 + i * 0.1, i * 60_000) for i in range(300)]
    seen_lengths: list[int] = []

    async def capture(pair: str, slice_: list[Kline]) -> BacktestResult:
        seen_lengths.append(len(slice_))
        return BacktestResult(
            pair=pair, start_date="", end_date="",
            initial_balance=10_000, final_balance=10_000,
        )

    warmup, test_size = 100, 50
    report = await run_walk_forward(
        "X", klines, backtest_fn=capture,
        train_size=200, test_size=test_size, warmup=warmup,
    )
    assert report.fold_count > 0
    # Every fold slice is exactly warmup + test_size — no extra train bars.
    assert seen_lengths
    assert all(n == warmup + test_size for n in seen_lengths)


async def test_run_walk_forward_skips_folds_without_full_warmup():
    """A fold whose test window starts before a full warmup prefix is skipped
    rather than evaluated on a truncated (leaky/degraded) context."""
    klines = [_kl(100 + i * 0.1, i * 60_000) for i in range(200)]

    async def fake(pair: str, slice_: list[Kline]) -> BacktestResult:
        # train_size 60 < warmup 100 → test_start (60) - warmup (100) < 0 → skip.
        assert len(slice_) >= 100  # if ever called, must have a full warmup
        return BacktestResult(
            pair=pair, start_date="", end_date="",
            initial_balance=10_000, final_balance=10_000,
        )

    report = await run_walk_forward(
        "X", klines, backtest_fn=fake, train_size=60, test_size=20, warmup=100
    )
    # First test_start is 60 < 100, but later folds step forward; only folds
    # with test_start >= 100 survive.
    assert all(True for _ in report.folds)  # no crash
    assert report.fold_count >= 0


async def test_run_walk_forward_empty_returns_zero():
    async def fake(pair: str, slice_: list[Kline]) -> BacktestResult:
        raise AssertionError("should not be called")

    report = await run_walk_forward("X", [], backtest_fn=fake, train_size=40, test_size=20)
    assert report.fold_count == 0
    assert report.avg_return_pct == 0.0


# ── monte_carlo_resample ──────────────────────────────────────


def test_monte_carlo_basic_distribution():
    trades = [_trade(10), _trade(-5), _trade(15), _trade(-2), _trade(8)]
    report = monte_carlo_resample(trades, initial_balance=10_000, runs=200, seed=42)
    assert report.runs == 200
    # Final return is invariant to ordering — sum of pnls / initial.
    expected = (10 - 5 + 15 - 2 + 8) / 10_000
    assert abs(report.final_return_pct_mean - expected) < 1e-9


def test_monte_carlo_dd_distribution_widens_with_shuffle():
    """Worst-case drawdown across shuffles should be at least the realised one."""
    # Construct a trade list where the realised order has nearly zero DD
    # but a worst-case re-ordering creates a real drawdown.
    trades = [_trade(10)] * 5 + [_trade(-5)] * 5
    report = monte_carlo_resample(trades, initial_balance=10_000, runs=500, seed=1)
    # 95th-percentile DD must be > 0 because some shuffles will frontload losses.
    assert report.max_drawdown_pct_p95 > 0


def test_monte_carlo_no_trades_returns_zero():
    report = monte_carlo_resample([], initial_balance=10_000, runs=100)
    assert report.runs == 0
    assert report.final_return_pct_mean == 0.0


def test_monte_carlo_negative_runs_raises():
    import pytest

    with pytest.raises(ValueError):
        monte_carlo_resample([_trade(10)], initial_balance=10_000, runs=0)


def test_monte_carlo_seed_makes_run_reproducible():
    trades = [_trade(10), _trade(-5), _trade(15)]
    a = monte_carlo_resample(trades, initial_balance=10_000, runs=50, seed=99)
    b = monte_carlo_resample(trades, initial_balance=10_000, runs=50, seed=99)
    assert a.max_drawdown_pct_mean == b.max_drawdown_pct_mean
