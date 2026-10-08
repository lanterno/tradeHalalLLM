"""`halal-trader purify`: what each dividend owes to charity (compliance/purification.py)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import click

from halal_trader.cli._run import run_db
from halal_trader.logging import console
from halal_trader.portfolio.core_account import DAY_TRADER


@click.group("purify")
def purify() -> None:
    """Purification of impermissible income in dividends."""


@purify.command("sync")
@click.option("--days", default=400, show_default=True, help="Dividend history to fetch.")
@click.option(
    "--recompute-unpaid",
    is_flag=True,
    help="Recompute every unpaid accrual under the current screen (paid ones never change).",
)
def sync_cmd(days: int, recompute_unpaid: bool) -> None:
    """Fetch dividends of everything held, then accrue the paper account and every book."""

    async def _run(engine: Any, settings: Any) -> tuple[int, dict[str, int]]:
        from halal_trader.compliance.purification import (
            accrue_book,
            accrue_paper,
            clear_unpaid,
            held_symbols,
            sync_dividends,
        )
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.market_hours import today_eastern
        from halal_trader.research.forward_book import book_names

        market = AlpacaMarketData.from_settings(settings)
        today = today_eastern()
        try:
            symbols = await held_symbols(engine, today - timedelta(days=days))
            n = await sync_dividends(
                engine, market, symbols, start=today - timedelta(days=days), end=today
            )
            if recompute_unpaid:
                await clear_unpaid(engine)
            accrued = {DAY_TRADER: len(await accrue_paper(engine, through=today))}
            for book in await book_names(engine):
                accrued[f"book:{book}"] = len(await accrue_book(engine, book, through=today))
            return n, accrued
        finally:
            await market.aclose()

    n, accrued = run_db(_run)
    console.print(
        f"{n} new dividend(s); accrued " + ", ".join(f"{k}: {v}" for k, v in accrued.items())
    )


@purify.command("report")
@click.option("--account", default=DAY_TRADER, show_default=True, help='"paper" or "book:<name>".')
@click.option("--year", type=int, default=None, help="Payment year (default: this year).")
def report_cmd(account: str, year: int | None) -> None:
    """Dividends and the amount to purify, per holding, for one payment year."""
    from halal_trader.market_hours import today_eastern

    year = year or today_eastern().year

    async def _run(engine: Any, settings: Any) -> list[Any]:
        from halal_trader.compliance.purification import report

        return await report(engine, account, year)

    from halal_trader.compliance.purification import BOOK_NOTIONAL

    lines = run_db(_run)
    unit = f" (per ${BOOK_NOTIONAL:,.0f} following the book)" if account.startswith("book:") else ""
    console.print(f"[bold]{account}[/bold], dividends payable in {year}{unit}")
    if not lines:
        console.print("  nothing accrued")
        return
    for line in lines:
        note = f"  ({line.assumed} at the 5% default)" if line.assumed else ""
        console.print(
            f"  {line.symbol:6} dividends ${line.dividends:10.2f}  purify ${line.amount:8.2f}"
            f"  over {line.payments} payment(s){note}"
        )
    console.print(
        f"  [bold]total: dividends ${sum(x.dividends for x in lines):.2f}, "
        f"purify ${sum(x.amount for x in lines):.2f}[/bold]"
    )
    console.print(
        "  The ratio counts interest income only: impermissible business revenue inside a "
        "passing company is not in SEC data, so treat this as the minimum."
    )


@purify.command("paid")
@click.option(
    "--through",
    type=click.DateTime(["%Y-%m-%d"]),
    required=True,
    help="Covers every unpaid accrual payable on or before this date.",
)
@click.option("--to", "paid_to", required=True, help="The charity, as you want it recorded.")
@click.option("--account", default=DAY_TRADER, show_default=True)
def paid_cmd(through: Any, paid_to: str, account: str) -> None:
    """Record that you have given the outstanding purification away."""

    async def _run(engine: Any, settings: Any) -> float:
        from halal_trader.compliance.purification import mark_paid

        return await mark_paid(engine, account, through=through.date(), paid_to=paid_to)

    console.print(f"marked ${run_db(_run):.2f} paid to {paid_to}")
