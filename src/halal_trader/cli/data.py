"""`halal-trader data`: the research market-data store (data/)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, TypeVar

import click

from halal_trader.logging import console

T = TypeVar("T")


def _with_store(work: Callable[[Any, Any], Awaitable[T]]) -> T:
    """Run ``work(engine, client)`` with a DB engine and a market-data client."""

    async def _run() -> T:
        from halal_trader.config import get_settings
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
        client = AlpacaMarketData(settings.alpaca.api_key, settings.alpaca.secret_key)
        try:
            return await work(engine, client)
        finally:
            await client.aclose()
            await engine.dispose()

    return asyncio.run(_run())


@click.group("data")
def data() -> None:
    """Research market data: Alpaca assets and 10 years of daily SIP bars."""


@data.command("assets")
def assets_cmd() -> None:
    """Sync Alpaca's active US-equity asset list."""
    from halal_trader.data.store import sync_assets

    n = _with_store(sync_assets)
    console.print(f"{n} active assets synced")


@data.command("backfill")
@click.option("--top", default=1500, show_default=True, help="Most liquid names to keep.")
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2016-01-01",
    show_default=True,
    help="First session to fetch.",
)
def backfill_cmd(top: int, since: Any) -> None:
    """Pick the liquid universe, then fetch raw + adjusted daily bars for it.

    Resumable: symbols already stored continue from their last session.
    """
    from halal_trader.data.store import BENCHMARKS, liquid_universe, sync_assets, update_bars
    from halal_trader.market_hours import today_eastern

    async def work(engine: Any, client: Any) -> tuple[int, dict[str, int]]:
        await sync_assets(engine, client)
        members = await liquid_universe(engine, client, top_n=top, as_of=today_eastern())
        symbols = sorted({m.symbol for m in members} | set(BENCHMARKS))
        stored = await update_bars(engine, client, symbols, since=since.date())
        return len(symbols), stored

    n, stored = _with_store(work)
    console.print(f"{n} symbols; rows stored: {stored}")


@data.command("update")
def update_cmd() -> None:
    """Top up every stored symbol to the latest closed session."""
    from sqlalchemy import text

    from halal_trader.data.store import update_bars

    async def work(engine: Any, client: Any) -> tuple[int, dict[str, int]]:
        async with engine.connect() as conn:
            symbols = [
                r.symbol for r in await conn.execute(text("SELECT DISTINCT symbol FROM daily_bars"))
            ]
        if not symbols:
            return 0, {}
        return len(symbols), await update_bars(engine, client, symbols, since=date(2016, 1, 1))

    n, stored = _with_store(work)
    console.print(
        f"{n} symbols topped up; rows stored: {stored}"
        if n
        else "No bars stored yet: run `data backfill`."
    )
