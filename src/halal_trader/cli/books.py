"""`halal-trader books`: paper-forward books (research/forward_book.py)."""

from __future__ import annotations

from typing import Any

import click

from halal_trader.cli._run import run_db
from halal_trader.logging import console


@click.group("books")
def books() -> None:
    """Strategies run forward on real closing prices, with no orders."""


@books.command("create")
@click.argument("name")
@click.option("--strategy", default="s1-momentum-lowvol", show_default=True)
@click.option("--top", default=30, show_default=True, help="Names held.")
@click.option("--cost-bps", default=10.0, show_default=True, help="Per side, on turnover.")
def create_cmd(name: str, strategy: str, top: int, cost_bps: float) -> None:
    """Start a book: it is set up on the latest stored session and trades from the next."""

    async def work(engine: Any, settings: Any) -> Any:
        from halal_trader.market_hours import today_eastern
        from halal_trader.research.forward_book import advance_book, create_book

        await create_book(engine, name, strategy=strategy, top_n=top, cost_bps=cost_bps)
        return await advance_book(engine, name, through=today_eastern())

    rows = run_db(work)
    console.print(f"book {name} starts on {rows[0].day}; it rebalances at the next close")


@books.command("run")
def run_cmd() -> None:
    """The evening run by hand: top up bars, re-screen if a week old, advance every book."""

    async def work(engine: Any, settings: Any) -> Any:
        from halal_trader.market_hours import today_eastern
        from halal_trader.research.daily import run_research

        return await run_research(engine, settings, today=today_eastern())

    run = run_db(work)
    console.print(f"bars stored: {run.bars_stored}")
    console.print(f"screened: {run.screened if run.screened is not None else 'skipped (fresh)'}")
    for name, n in run.books.items():
        console.print(f"book {name}: {n} session(s) appended")
    for error in run.errors:
        console.print(f"[red]{error}[/red]")


@books.command("show")
@click.argument("name")
def show_cmd(name: str) -> None:
    """A book's NAV and statistics against SPY, SPUS and HLAL over the same sessions."""

    async def work(engine: Any, settings: Any) -> Any:
        from halal_trader.research.forward_book import report

        return await report(engine, name)

    r = run_db(work)
    console.print(f"{r.name}: started {r.started}, {r.days} session(s), NAV {r.nav:.4f}")

    def line(label: str, s: Any) -> str:
        if s is None:
            return f"  {label:6} (not enough sessions yet)"
        sharpe = f"{s.sharpe:5.2f}" if s.sharpe is not None else "  n/a"
        return (
            f"  {label:6} return {s.total_return:+7.2%}  Sharpe {sharpe}"
            f"  maxDD {s.max_drawdown:7.2%}"
        )

    console.print(line("book", r.stats))
    for symbol, s in r.benchmarks.items():
        console.print(line(symbol, s))
    if r.holdings:
        top = sorted(r.holdings.items(), key=lambda kv: -kv[1])
        console.print("  holdings: " + ", ".join(f"{s} {w:.1%}" for s, w in top))
