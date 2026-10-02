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
        _judge(
            prices,
            result,
            config={
                "strategy": "s1",
                "score": "z(mom 12-1) + z(-vol 63d)",
                "rebalance": "monthly",
                "weighting": "equal",
                "top": top,
                "universe": universe,
                "cost_bps": cost_bps,
                "since": str(since.date()),
            },
        )
        console.print(
            "[yellow]Remaining bias:[/yellow] delisted companies cannot be mapped to SEC "
            "filings, so they are never eligible. Share of the universe unmapped, by year: "
            + "  ".join(f"{y} {s:.0%}" for y, s in sorted(schedule.unmapped.items()))
        )
        return
    console.print("[yellow]Biased upward by:[/yellow] " + "; ".join(BIASES))


@research.command("pit-backtest")
@click.argument("strategy", type=click.Choice(["t3-cap", "t4-tilt", "t5-3factor"]))
@click.option("--cost-bps", default=10.0, show_default=True, help="Cost per side, basis points.")
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2017-01-01",
    show_default=True,
    help="First rebalance on or after this date.",
)
@click.option("--universe", default=1000, show_default=True, help="Liquidity universe size.")
def pit_backtest_cmd(strategy: str, cost_bps: float, since: Any, universe: int) -> None:
    """A pre-registered point-in-time strategy (research/strategies.py), judged vs SPUS."""
    from halal_trader.research.factor_backtest import (
        backtest_targets,
        benchmark_returns,
        split_reused_tickers,
        stats,
    )
    from halal_trader.research.strategies import STRATEGIES, TILT, TOP_N, Inputs

    async def _load() -> tuple[Any, Any, Any]:
        from halal_trader.config import get_settings
        from halal_trader.data.fundamentals import quality_by_year
        from halal_trader.data.store import BENCHMARKS
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern
        from halal_trader.research.factor_backtest import load_prices
        from halal_trader.research.pit import pit_schedule

        engine = await init_db(get_settings().database_url)
        try:
            schedule = await pit_schedule(
                engine, start=since.date(), end=today_eastern(), top_n=universe
            )
            if not any(schedule.eligible_from.values()):
                raise click.ClickException(
                    "no point-in-time eligibility: run `data pit-universe` and "
                    "`compliance screen-history` first"
                )
            quality = await quality_by_year(engine)
            if strategy != "t3-cap" and not quality:
                raise click.ClickException("no quality data: run `data fundamentals` first")
            prices = await load_prices(
                engine,
                sorted(schedule.universe | set(BENCHMARKS)),
                since=date(since.year - 2, 1, 1),
            )
            return schedule, quality, prices
        finally:
            await engine.dispose()

    schedule, quality, prices = asyncio.run(_load())
    prices, breaks = split_reused_tickers(prices)
    inputs = Inputs(prices, schedule.firms_from, quality)
    result = backtest_targets(
        prices, STRATEGIES[strategy](inputs), cost_bps=cost_bps, start=since.date()
    )
    console.print(
        f"{strategy}, {cost_bps:g} bps/side, point-in-time top-{universe} universe x quarterly "
        f"screen; {result.days[0]} -> {result.days[-1]}, "
        f"avg turnover {result.avg_turnover:.0%}/rebalance"
    )
    s = result.stats
    console.print(
        f"  strategy  CAGR {s.cagr:+7.2%}  vol {s.volatility:6.2%}  "
        f"Sharpe {s.sharpe or 0:5.2f}  maxDD {s.max_drawdown:7.2%}"
    )
    for bench in ("SPY", "SPUS", "HLAL"):
        b = benchmark_returns(prices, bench, result.days)
        if b is not None:
            bs = stats(b)
            console.print(f"  {bench:8}  CAGR {bs.cagr:+7.2%}  Sharpe {bs.sharpe or 0:5.2f}")
    console.print(
        "  by year: " + "  ".join(f"{y} {r:+.1%}" for y, r in sorted(result.yearly.items()))
    )
    last = max(result.holdings) if result.holdings else None
    if last:
        console.print(
            f"  {len(result.holdings[last])} held at {last}, largest: "
            + ", ".join(result.holdings[last][:15])
        )
    config: dict[str, Any] = {
        "strategy": strategy,
        "rebalance": "monthly",
        "universe": universe,
        "cost_bps": cost_bps,
        "since": str(since.date()),
        "quality": "gross profit / assets, calendar-year frame, used from May 1 next year",
    }
    if strategy == "t4-tilt":
        config["tilt"] = TILT
    if strategy == "t5-3factor":
        config["top"] = TOP_N
    if strategy != "t3-cap":
        config["score"] = "z(z(mom 12-1) + z(-vol 63d) + z(quality))"
    _judge(prices, result, config=config)
    if breaks:
        console.print(f"  reused tickers: {len(breaks)} series cut at a 10x one-day move")


def _judge(prices: Any, result: Any, *, config: dict[str, Any]) -> None:
    """Record a point-in-time run in the trials ledger and print its verdict."""
    from halal_trader.research.factor_backtest import benchmark_returns
    from halal_trader.research.ledger import BENCHMARK, CRITERION, record_backtest

    col = prices.symbols.index(BENCHMARK) if BENCHMARK in prices.symbols else None
    row = {d: i for i, d in enumerate(prices.days)}
    have = (
        [i for i, d in enumerate(result.days) if not np.isnan(prices.close[row[d], col])]
        if col is not None
        else []
    )
    if len(have) < 60:
        console.print(f"[yellow]not recorded: no {BENCHMARK} history to judge against[/yellow]")
        return
    days = result.days[have[0] + 1 :]
    bench = benchmark_returns(prices, BENCHMARK, days)
    if bench is None:
        console.print(f"[yellow]not recorded: gaps in {BENCHMARK} history[/yellow]")
        return

    async def _run() -> Any:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db

        engine = await init_db(get_settings().database_url)
        try:
            return await record_backtest(
                engine,
                strategy=config["strategy"],
                config=config,
                days=days,
                returns=result.returns[have[0] + 1 :],
                benchmark=bench,
            )
        finally:
            await engine.dispose()

    a = asyncio.run(_run())
    if a is None:
        console.print("[yellow]not recorded: degenerate active returns[/yellow]")
        return
    color = "green" if a.verdict == "pass" else "red"
    console.print(
        f"  from {days[0]}: strategy CAGR {a.cagr:+.2%} vs {BENCHMARK} {a.benchmark_cagr:+.2%}, "
        f"tracking error {a.tracking_error:.2%}"
    )
    console.print(
        f"  [{color}]ledger: {a.verdict.upper()}[/{color}] trial #{a.trial_id}, "
        f"{a.n_trials} distinct research trials; active Sharpe vs {BENCHMARK} "
        f"{a.active_sharpe:+.2f} against a no-skill hurdle of {a.hurdle_sharpe:.2f}, "
        f"DSR {a.dsr:.2f} ({CRITERION})"
    )


@research.command("event-backtest")
@click.argument("signal", type=click.Choice(["e1a", "e1b", "e1c", "e1-combo", "e3"]))
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2016-06-01",
    show_default=True,
    help="First entry on or after this date.",
)
def event_backtest_cmd(signal: str, since: Any) -> None:
    """A pre-registered S2 Phase C event portfolio, recorded and judged vs SPUS."""
    hold = {"e1a": 60, "e1b": 60, "e1c": 60, "e1-combo": 60, "e3": 63}[signal]
    slots = 40

    async def _run() -> tuple[Any, Any, Any, int]:
        from datetime import date as _date

        from halal_trader.config import get_settings
        from halal_trader.data.universe import universe_at
        from halal_trader.db.models import init_db
        from halal_trader.events import signals
        from halal_trader.events.portfolio import simulate
        from halal_trader.events.study import cost_bps, load_bars
        from halal_trader.market_hours import today_eastern
        from halal_trader.research.pit import pit_schedule

        engine = await init_db(get_settings().database_url)
        try:
            schedule = await pit_schedule(
                engine, start=since.date(), end=today_eastern(), top_n=1000
            )
            symbols = sorted(schedule.universe | {"SPY", "SPUS"})
            bars = await load_bars(engine, symbols)
            if signal == "e1a":
                candidates = await signals.e1a(engine)
            elif signal == "e1-combo":
                candidates = await signals.e1_combo(engine, bars)
            elif signal == "e1b":
                candidates = await signals.e1b(engine)
            elif signal == "e1c":
                candidates = await signals.e1c(engine, bars)
            else:
                candidates = await signals.e3(engine)
            ranks: dict[_date, dict[str, int]] = {}

            def cost(symbol: str, day: _date) -> float:
                return cost_bps(ranks.get(day.replace(day=1), {}).get(symbol))

            for month in sorted({d.replace(day=1) for d in bars.sessions if d >= since.date()}):
                names = await universe_at(engine, month, top_n=3000)
                ranks[month] = {s: i for i, s in enumerate(names)}
            book = simulate(
                bars,
                candidates,
                hold=hold,
                slots=slots,
                start=since.date(),
                eligible_from=schedule.eligible_from,
                cost_bps=cost,
            )
            return book, bars, candidates, len(candidates)
        finally:
            await engine.dispose()

    book, bars, candidates, n = asyncio.run(_run())
    from halal_trader.research.factor_backtest import stats

    s = stats(book.returns)
    console.print(
        f"{signal}: {n} candidates, {book.trades} trades, {book.skipped_full} turned away "
        f"(slots full), {book.skipped_ineligible} ineligible; mean exposure {book.exposure:.0%}; "
        f"{book.days[0]} -> {book.days[-1]}"
    )
    console.print(
        f"  CAGR {s.cagr:+.2%}  vol {s.volatility:.2%}  Sharpe {s.sharpe or 0:.2f}  "
        f"maxDD {s.max_drawdown:.2%}"
    )
    from halal_trader.research.ledger import BENCHMARK, record_backtest

    spus = bars.close.get(BENCHMARK, {})
    idx = [i for i, d in enumerate(book.days) if d in spus and i > 0 and book.days[i - 1] in spus]
    if len(idx) < 60:
        console.print("[yellow]not recorded: too little SPUS history[/yellow]")
        return
    days = [book.days[i] for i in idx]
    bench = np.array([spus[book.days[i]] / spus[book.days[i - 1]] - 1 for i in idx])
    rets = book.returns[idx]

    async def _record() -> Any:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db

        engine = await init_db(get_settings().database_url)
        try:
            return await record_backtest(
                engine,
                strategy=signal,
                config={
                    "strategy": signal,
                    "kind": "event portfolio",
                    "slots": slots,
                    "hold": hold,
                    "threshold": "trailing-365d 90th percentile"
                    if signal != "e3"
                    else ">=2 officers/directors, >=$100k, 30 days, no 10b5-1",
                    "costs": "7/15/30 bps one way by liquidity",
                    "since": str(since.date()),
                },
                days=days,
                returns=rets,
                benchmark=bench,
                extra={"trades": book.trades, "exposure": book.exposure},
            )
        finally:
            await engine.dispose()

    a = asyncio.run(_record())
    if a is None:
        console.print("[yellow]not recorded: degenerate active returns[/yellow]")
        return
    color = "green" if a.verdict == "pass" else "red"
    console.print(
        f"  from {days[0]}: CAGR {a.cagr:+.2%} vs {BENCHMARK} {a.benchmark_cagr:+.2%}, "
        f"tracking error {a.tracking_error:.2%}"
    )
    console.print(
        f"  [{color}]ledger: {a.verdict.upper()}[/{color}] trial #{a.trial_id}, "
        f"{a.n_trials} distinct research trials; active Sharpe {a.active_sharpe:+.2f} "
        f"vs hurdle {a.hurdle_sharpe:.2f}, DSR {a.dsr:.2f}"
    )
