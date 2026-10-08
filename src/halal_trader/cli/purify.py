"""`halal-trader purify`: what each dividend owes to charity (compliance/purification.py)."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import click

from halal_trader.logging import console


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

    async def _run() -> tuple[int, dict[str, int]]:
        from halal_trader.compliance.purification import (
            accrue_book,
            accrue_paper,
            clear_unpaid,
            held_symbols,
            sync_dividends,
        )
        from halal_trader.config import get_settings
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern
        from halal_trader.research.forward_book import book_names

        settings = get_settings()
        engine = await init_db(settings.database_url)
        market = AlpacaMarketData(settings.alpaca.api_key, settings.alpaca.secret_key)
        today = today_eastern()
        try:
            symbols = await held_symbols(engine, today - timedelta(days=days))
            n = await sync_dividends(
                engine, market, symbols, start=today - timedelta(days=days), end=today
            )
            if recompute_unpaid:
                await clear_unpaid(engine)
            accrued = {"paper": len(await accrue_paper(engine, through=today))}
            for book in await book_names(engine):
                accrued[f"book:{book}"] = len(await accrue_book(engine, book, through=today))
            return n, accrued
        finally:
            await market.aclose()
            await engine.dispose()

    n, accrued = asyncio.run(_run())
    console.print(
        f"{n} new dividend(s); accrued " + ", ".join(f"{k}: {v}" for k, v in accrued.items())
    )


@purify.command("report")
@click.option("--account", default="paper", show_default=True, help='"paper" or "book:<name>".')
@click.option("--year", type=int, default=None, help="Payment year (default: this year).")
def report_cmd(account: str, year: int | None) -> None:
    """Dividends and the amount to purify, per holding, for one payment year."""
    from halal_trader.market_hours import today_eastern

    year = year or today_eastern().year

    async def _run() -> list[Any]:
        from halal_trader.compliance.purification import report
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db

        engine = await init_db(get_settings().database_url)
        try:
            return await report(engine, account, year)
        finally:
            await engine.dispose()

    from halal_trader.compliance.purification import BOOK_NOTIONAL

    lines = asyncio.run(_run())
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
@click.option("--account", default="paper", show_default=True)
def paid_cmd(through: Any, paid_to: str, account: str) -> None:
    """Record that you have given the outstanding purification away."""

    async def _run() -> float:
        from halal_trader.compliance.purification import mark_paid
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db

        engine = await init_db(get_settings().database_url)
        try:
            return await mark_paid(engine, account, through=through.date(), paid_to=paid_to)
        finally:
            await engine.dispose()

    console.print(f"marked ${asyncio.run(_run()):.2f} paid to {paid_to}")
