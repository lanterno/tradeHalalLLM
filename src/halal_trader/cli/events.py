"""`halal-trader events`: the event store's labels and what the scores knew."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from halal_trader.logging import console


@click.group("events")
def events() -> None:
    """News and filings as dated, scored, labelled events (S2 research)."""


@events.command("report")
@click.option("--threshold", default=0.85, show_default=True, help="The reactor's trigger.")
def report_cmd(threshold: float) -> None:
    """Label matured events, then show each scorer's IC against abnormal returns."""

    async def _run() -> tuple[int, list[Any], dict[str, int]]:
        from sqlalchemy import text

        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.events.labels import label_events, report

        engine = await init_db(get_settings().database_url)
        try:
            written = await label_events(engine)
            async with engine.connect() as conn:
                counts = {
                    "events": (await conn.execute(text("SELECT count(*) FROM events"))).scalar()
                    or 0,
                    "scored": (
                        await conn.execute(text("SELECT count(*) FROM event_scores"))
                    ).scalar()
                    or 0,
                }
            return written, await report(engine, threshold=threshold), counts
        finally:
            await engine.dispose()

    written, rows, counts = asyncio.run(_run())
    console.print(f"{counts['events']} events, {counts['scored']} scores; {written} new label(s)")
    if not rows:
        console.print("[yellow]no labelled scores yet: horizons need sessions to elapse[/yellow]")
        return
    for r in rows:
        above = f"{r.above_abn:+.2%}" if r.above_abn is not None else "  n/a"
        below = f"{r.below_abn:+.2%}" if r.below_abn is not None else "  n/a"
        console.print(
            f"  {r.scorer:40} {r.horizon:>2}d  n={r.events:<5} IC {r.ic:+.3f}  "
            f">= {threshold:g}: {r.above} events, mean abn {above}; below: {below}"
        )


@events.command("backfill")
@click.argument("what", type=click.Choice(["news", "filings", "insiders", "eps", "all"]))
@click.option(
    "--rate",
    default=100,
    show_default=True,
    help="Alpaca requests per minute for news (the live bot shares the key's 200/min).",
)
def backfill_cmd(what: str, rate: int) -> None:
    """Fill the event store's history (resumable; finished units are skipped)."""

    async def _run() -> dict[str, int]:
        from halal_trader.compliance.sec import SecClient
        from halal_trader.config import get_settings
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.db.models import init_db
        from halal_trader.events import history

        settings = get_settings()
        engine = await init_db(settings.database_url)
        sec = SecClient(settings.edgar.user_agent)
        out: dict[str, int] = {}
        try:
            companies = await history.covered_companies(engine)
            if what in ("filings", "all"):
                out["filings"] = await history.backfill_filings(engine, sec, companies)
            if what in ("insiders", "all"):
                out["insiders"] = await history.backfill_insiders(engine, sec, companies)
            if what in ("eps", "all"):
                out["eps"] = await history.backfill_eps(engine, sec, companies)
            if what in ("news", "all"):
                market = AlpacaMarketData(
                    settings.alpaca.api_key,
                    settings.alpaca.secret_key,
                    min_interval_s=60.0 / max(rate, 1),
                )
                try:
                    symbols = await history.covered_symbols(engine)
                    out["news"] = await history.backfill_news(engine, market, symbols=symbols)
                finally:
                    await market.aclose()
        finally:
            await sec.aclose()
            await engine.dispose()
        return out

    for name, n in asyncio.run(_run()).items():
        console.print(f"{name}: {n} new")


@events.command("extract")
def extract_cmd() -> None:
    """Read earnings results and guidance vs consensus out of stored headlines."""

    async def _run() -> int:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.events.earnings_parse import extract_all

        engine = await init_db(get_settings().database_url)
        try:
            return await extract_all(engine)
        finally:
            await engine.dispose()

    console.print(f"{asyncio.run(_run())} earnings fact(s) extracted")
