"""The research trials ledger: every backtest counts, verdicts deflate with the count."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from halal_trader.research.ledger import PREFIX, record_backtest

N = 1500
DAYS = [date(2020, 1, 1) + timedelta(days=i) for i in range(N)]


def _series(seed: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bench = rng.normal(0.0004, 0.01, N)
    return bench + alpha + rng.normal(0.0, 0.004, N), bench


async def _record(engine, cfg: dict, seed: int, alpha: float):
    returns, bench = _series(seed, alpha)
    return await record_backtest(
        engine, strategy="t", config=cfg, days=DAYS, returns=returns, benchmark=bench
    )


@pytest.mark.asyncio
async def test_real_edge_passes_and_noise_fails(engine):
    strong = await _record(engine, {"v": 1}, seed=1, alpha=0.001)
    assert strong is not None and strong.verdict == "pass"
    noise = await _record(engine, {"v": 2}, seed=2, alpha=0.0)
    assert noise is not None and noise.verdict == "fail"
    assert noise.n_trials == 2


@pytest.mark.asyncio
async def test_rerunning_a_config_is_not_a_new_trial(engine):
    await _record(engine, {"v": 1}, seed=1, alpha=0.0002)
    again = await _record(engine, {"v": 1}, seed=3, alpha=0.0002)
    assert again is not None and again.n_trials == 1
    async with engine.connect() as conn:
        from sqlalchemy import text

        rows = (
            await conn.execute(
                text("SELECT count(*) FROM quant_trials WHERE name LIKE :p"), {"p": PREFIX + "%"}
            )
        ).scalar()
    assert rows == 2  # both runs are kept; only the count of configurations is deduplicated


@pytest.mark.asyncio
async def test_more_trials_raise_the_hurdle(engine):
    first = await _record(engine, {"v": 0}, seed=10, alpha=0.0003)
    assert first is not None
    for i in range(1, 30):
        await _record(engine, {"v": i}, seed=10 + i, alpha=0.0)
    last = await _record(engine, {"v": 0}, seed=10, alpha=0.0003)
    assert last is not None
    assert last.n_trials == 30
    assert last.hurdle_sharpe > first.hurdle_sharpe
    assert last.dsr < first.dsr


@pytest.mark.asyncio
async def test_a_run_names_the_benchmark_it_was_judged_by(engine):
    """The window and criterion say SPUS by default, and the label when one is given."""
    from sqlalchemy import text

    from halal_trader.research.ledger import BENCHMARK, CRITERION

    returns, bench = _series(5, 0.0005)
    default = await record_backtest(
        engine, strategy="t", config={"v": 1}, days=DAYS, returns=returns, benchmark=bench
    )
    labelled = await record_backtest(
        engine,
        strategy="t",
        config={"v": 2},
        days=DAYS,
        returns=returns,
        benchmark=bench,
        benchmark_label="SPY (exposure-matched)",
    )
    assert default is not None and labelled is not None
    async with engine.connect() as conn:
        rows = {
            r.id: (r.window, r.criterion)
            for r in await conn.execute(text('SELECT id, "window", criterion FROM quant_trials'))
        }
    assert rows[default.trial_id] == (f"{DAYS[0]}..{DAYS[-1]} vs {BENCHMARK}", CRITERION)
    window, criterion = rows[labelled.trial_id]
    assert window == f"{DAYS[0]}..{DAYS[-1]} vs SPY (exposure-matched)"
    assert "vs SPY (exposure-matched)" in criterion and BENCHMARK not in criterion


@pytest.mark.asyncio
async def test_degenerate_active_series_is_not_recorded(engine):
    flat = np.full(N, 0.001)
    out = await record_backtest(
        engine, strategy="t", config={}, days=DAYS, returns=flat, benchmark=flat
    )
    assert out is None
