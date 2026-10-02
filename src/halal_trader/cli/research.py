"""`halal-trader research`: backtests that decide what earns capital (research/)."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import click
import numpy as np

from halal_trader.logging import console


@click.group("research")
def research() -> None:
    """Strategy research on the stored market data (results are biased; see each command)."""


@research.command("factor-backtest")
@click.option("--top", default=30, show_default=True, help="Names held.")
@click.option("--cost-bps", default=10.0, show_default=True, help="Cost per side, basis points.")
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2017-01-01",
    show_default=True,
    help="First rebalance on or after this date (needs a year of history before it).",
)
@click.option(
    "--pit",
    is_flag=True,
    help="Point-in-time eligibility: that month's liquid universe x that quarter's screen "
    "(run `data pit-universe` and `compliance screen-history` first).",
)
@click.option("--universe", default=1000, show_default=True, help="With --pit: universe size.")
def factor_backtest_cmd(top: int, cost_bps: float, since: Any, pit: bool, universe: int) -> None:
    """S1 first look: monthly top-N halal momentum + low-vol vs SPY/SPUS/HLAL."""
    from halal_trader.research.factor_backtest import (
        BIASES,
        backtest,
        benchmark_returns,
        load_prices,
        split_reused_tickers,
        stats,
    )

    schedule: Any = None

    async def _load() -> tuple[Any, set[str]]:
        nonlocal schedule
        from sqlalchemy import text

        from halal_trader.config import get_settings
        from halal_trader.data.store import BENCHMARKS
        from halal_trader.db.models import init_db

        engine = await init_db(get_settings().database_url)
        try:
            if pit:
                from halal_trader.market_hours import today_eastern
                from halal_trader.research.pit import pit_schedule

                schedule = await pit_schedule(
                    engine, start=since.date(), end=today_eastern(), top_n=universe
                )
                if not any(schedule.eligible_from.values()):
                    raise click.ClickException(
                        "no point-in-time eligibility: run `data pit-universe` and "
                        "`compliance screen-history` first"
                    )
                prices = await load_prices(
                    engine,
                    sorted(schedule.universe | set(BENCHMARKS)),
                    since=date(since.year - 2, 1, 1),
                )
                return prices, set()
            async with engine.connect() as conn:
                halal = {
                    r.symbol
                    for r in await conn.execute(
                        text(
                            "SELECT symbol FROM halal_screen_results WHERE verdict = 'halal' "
                            "AND as_of = (SELECT max(as_of) FROM halal_screen_results)"
                        )
                    )
                }
            if not halal:
                raise click.ClickException("no halal verdicts: run `compliance screen` first")
            prices = await load_prices(
                engine, sorted(halal | set(BENCHMARKS)), since=date(since.year - 2, 1, 1)
            )
        finally:
            await engine.dispose()
        return prices, halal

    prices, halal = asyncio.run(_load())
    prices, breaks = split_reused_tickers(prices)
    pit_from = schedule.eligible_from if schedule is not None else None
    result = backtest(
        prices,
        eligible=halal,
        top_n=top,
        cost_bps=cost_bps,
        start=since.date(),
        eligible_from=pit_from,
    )
    control = backtest(
        prices,
        eligible=halal,
        top_n=1_000_000,
        cost_bps=cost_bps,
        start=since.date(),
        eligible_from=pit_from,
    )

    def line(label: str, s: Any) -> str:
        sharpe = f"{s.sharpe:5.2f}" if s.sharpe is not None else "  n/a"
        psr = f"{s.psr:.2f}" if s.psr is not None else "n/a"
        return (
            f"  {label:18} CAGR {s.cagr:+7.2%}  vol {s.volatility:6.2%}  Sharpe {sharpe}"
            f"  PSR {psr}  maxDD {s.max_drawdown:7.2%}"
        )

    eligible_label = (
        f"point-in-time top-{universe} universe x quarterly screen"
        if schedule is not None
        else f"{len(halal)} eligible"
    )
    console.print(
        f"S1 top-{top} halal momentum+low-vol, {cost_bps:g} bps/side, {eligible_label}; "
        f"{result.days[0]} -> {result.days[-1]}, avg turnover {result.avg_turnover:.0%}/rebalance"
    )
    console.print(line("strategy", result.stats))
    console.print(line("no factor (all)", control.stats))
    for bench in ("SPY", "SPUS", "HLAL"):
        b = benchmark_returns(prices, bench, result.days)
        if b is None:  # shorter history: compare on the benchmark's own window
            col = prices.symbols.index(bench) if bench in prices.symbols else None
            if col is None:
                continue
            have = [
                i
                for i, d in enumerate(result.days)
                if not np.isnan(prices.close[prices.days.index(d), col])
            ]
            if len(have) < 60:
                continue
            days = result.days[have[0] + 1 :]
            b = benchmark_returns(prices, bench, days)
            if b is None:
                continue
            console.print(line(f"strategy (from {days[0]})", stats(result.returns[have[0] + 1 :])))
        console.print(line(bench, stats(b)))
    console.print(
        "  by year: " + "  ".join(f"{y} {r:+.1%}" for y, r in sorted(result.yearly.items()))
    )
    last = max(result.holdings) if result.holdings else None
    if last:
        console.print(f"  holdings at {last}: {', '.join(result.holdings[last])}")
    if breaks:
        console.print(
            f"  reused tickers: {len(breaks)} series cut at a 10x one-day move: "
            + ", ".join(f"{s} {d}" for s, d in breaks)
        )
    if schedule is not None:
        console.print(
            "[yellow]Remaining bias:[/yellow] delisted companies cannot be mapped to SEC "
            "filings, so they are never eligible. Share of the universe unmapped, by year: "
            + "  ".join(f"{y} {s:.0%}" for y, s in sorted(schedule.unmapped.items()))
        )
        return
    console.print("[yellow]Biased upward by:[/yellow] " + "; ".join(BIASES))
