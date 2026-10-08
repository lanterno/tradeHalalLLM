"""`halal-trader data`: the research market-data store (data/)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, TypeVar

import click

from halal_trader.cli._run import run_db
from halal_trader.logging import console

T = TypeVar("T")


def _with_store(work: Callable[[Any, Any], Awaitable[T]]) -> T:
    """Run ``work(engine, client)`` with a DB engine and a market-data client."""

    async def _run(engine: Any, settings: Any) -> T:
        from halal_trader.data.alpaca_market import AlpacaMarketData

        client = AlpacaMarketData.from_settings(settings)
        try:
            return await work(engine, client)
        finally:
            await client.aclose()

    return run_db(_run)


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

    from halal_trader.data.store import update_bars

    async def work(engine: Any, client: Any) -> tuple[int, dict[str, int]]:
        from halal_trader.data.store import stored_symbols

        symbols = await stored_symbols(engine)
        if not symbols:
            return 0, {}
        return len(symbols), await update_bars(engine, client, symbols, since=date(2016, 1, 1))

    n, stored = _with_store(work)
    console.print(
        f"{n} symbols topped up; rows stored: {stored}"
        if n
        else "No bars stored yet: run `data backfill`."
    )


@data.command("pit-universe")
@click.option("--top", default=1000, show_default=True, help="Names in the universe each month.")
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2016-01-01",
    show_default=True,
    help="First month of the universe; monthly bars start a year earlier.",
)
def pit_universe_cmd(top: int, since: Any) -> None:
    """Point-in-time universe: monthly bars for every listed and delisted stock,
    the top names by trailing dollar volume each month, daily bars for all of them.

    Resumable: daily bars continue from each symbol's last stored session.
    """
    from halal_trader.data.store import BENCHMARKS, update_bars
    from halal_trader.data.universe import sync_monthly_bars, universe_history
    from halal_trader.market_hours import today_eastern

    start = since.date()

    async def work(engine: Any, client: Any) -> tuple[int, int, int, dict[str, int]]:
        symbols, rows = await sync_monthly_bars(
            engine, client, since=start.replace(year=start.year - 1)
        )
        history = await universe_history(engine, start=start, end=today_eastern(), top_n=top)
        members = sorted({s for names in history.values() for s in names} | set(BENCHMARKS))
        stored = await update_bars(engine, client, members, since=start)
        return symbols, rows, len(members), stored

    symbols, rows, members, stored = _with_store(work)
    console.print(f"monthly bars: {rows} rows for {symbols} symbols (listed and delisted)")
    console.print(f"{members} names were ever in the top {top}; daily rows stored: {stored}")


@data.command("fundamentals")
@click.option("--since", default=2014, show_default=True, help="First calendar year.")
def fundamentals_cmd(since: int) -> None:
    """Sync annual gross profit and total assets for every SEC filer (the quality input)."""

    async def _run(engine: Any, settings: Any) -> int:
        from halal_trader.compliance.sec import SecClient
        from halal_trader.data.fundamentals import sync_annual, usable_year
        from halal_trader.market_hours import today_eastern

        sec = SecClient(settings.edgar.user_agent)
        try:
            last = usable_year(today_eastern())
            return await sync_annual(sec, engine, range(since, last + 1))
        finally:
            await sec.aclose()

    console.print(f"{run_db(_run)} filer-years stored")
