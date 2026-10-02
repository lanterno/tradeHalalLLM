"""`halal-trader compliance`: the in-house Shariah screen (compliance/)."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import click

from halal_trader.logging import console


@click.group("compliance")
def compliance() -> None:
    """In-house AAOIFI-style screen from SEC data (research-grade until validated)."""


@compliance.command("screen")
@click.option("--symbols", default="", help="Comma-separated; default = every stock in daily_bars.")
def screen_cmd(symbols: str) -> None:
    """Screen symbols against AAOIFI business-activity and financial-ratio rules."""

    async def _run() -> list[Any]:
        from sqlalchemy import text

        from halal_trader.compliance.runner import run_screen
        from halal_trader.compliance.sec import SecClient
        from halal_trader.config import get_settings
        from halal_trader.data.store import BENCHMARKS
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern

        settings = get_settings()
        engine = await init_db(settings.database_url)
        sec = SecClient(settings.edgar.user_agent)
        try:
            chosen = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            if not chosen:
                async with engine.connect() as conn:
                    rows = await conn.execute(text("SELECT DISTINCT symbol FROM daily_bars"))
                    chosen = sorted(r.symbol for r in rows if r.symbol not in BENCHMARKS)
            if not chosen:
                raise click.ClickException("no symbols: pass --symbols or run `data backfill`")
            return await run_screen(sec, engine, chosen, today_eastern())
        finally:
            await sec.aclose()
            await engine.dispose()

    results = asyncio.run(_run())
    counts = Counter(r.verdict for r in results)
    console.print(
        f"screened {len(results)}: [green]{counts['halal']} halal[/green], "
        f"[red]{counts['not_halal']} not halal[/red], "
        f"[yellow]{counts['doubtful']} doubtful[/yellow]"
    )
    for r in results:
        if len(results) <= 40 or r.verdict != "halal":
            reason = "; ".join(r.reasons) or "passes"
            console.print(f"  {r.symbol:6} {r.verdict:9} {reason}")


@compliance.command("screen-history")
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2016-01-01",
    show_default=True,
    help="First quarter end to screen.",
)
@click.option("--top", default=1000, show_default=True, help="Universe size each quarter.")
def screen_history_cmd(since: Any, top: int) -> None:
    """Screen every past quarter end's point-in-time universe (resumable)."""

    async def _run() -> dict[Any, int]:
        from halal_trader.compliance.history import screen_history
        from halal_trader.compliance.sec import SecClient
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern

        settings = get_settings()
        engine = await init_db(settings.database_url)
        sec = SecClient(settings.edgar.user_agent)
        try:
            return await screen_history(
                sec, engine, start=since.date(), end=today_eastern(), top_n=top
            )
        finally:
            await sec.aclose()
            await engine.dispose()

    done = asyncio.run(_run())
    for as_of, halal in sorted(done.items()):
        console.print(f"  {as_of}: {halal} halal")
    console.print(f"{len(done)} quarter(s) screened")


@compliance.command("map-delisted")
def map_delisted_cmd() -> None:
    """Match tickers SEC no longer lists to their filer by name, then re-screen them."""

    async def _run() -> tuple[list[Any], dict[Any, int]]:
        from halal_trader.compliance.delisted import map_unmapped, rescreen_mapped
        from halal_trader.compliance.sec import SecClient
        from halal_trader.config import get_settings
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern

        settings = get_settings()
        engine = await init_db(settings.database_url)
        sec = SecClient(settings.edgar.user_agent)
        market = AlpacaMarketData(settings.alpaca.api_key, settings.alpaca.secret_key)
        try:
            assets = [*await market.assets(), *await market.inactive_assets()]
            matches = await map_unmapped(sec, engine, assets, today=today_eastern())
            return matches, await rescreen_mapped(sec, engine)
        finally:
            await market.aclose()
            await sec.aclose()
            await engine.dispose()

    matches, rescreened = asyncio.run(_run())
    by_status: dict[str, int] = {}
    for m in matches:
        by_status[m.status] = by_status.get(m.status, 0) + 1
    console.print(
        f"{len(matches)} unmapped ticker(s): "
        + ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
    )
    console.print(
        f"re-screened {len(rescreened)} quarter(s); newly halal per quarter: "
        + " ".join(f"{d}:{n}" for d, n in sorted(rescreened.items()))
    )


@compliance.command("rescreen-passes")
def rescreen_passes_cmd() -> None:
    """Re-screen past halal verdicts made under an older screening method."""

    async def _run() -> dict[Any, int]:
        from halal_trader.compliance.history import rescreen_passes
        from halal_trader.compliance.sec import SecClient
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
        sec = SecClient(settings.edgar.user_agent)
        try:
            return await rescreen_passes(sec, engine)
        finally:
            await sec.aclose()
            await engine.dispose()

    done = asyncio.run(_run())
    console.print(f"re-screened {sum(done.values())} verdict(s) across {len(done)} screen date(s)")


@compliance.command("validate")
def validate_cmd() -> None:
    """Compare the latest screen with SPUS and HLAL holdings (SEC N-PORT)."""

    async def _run() -> tuple[Any, list[Any]]:
        from sqlalchemy import text

        from halal_trader.compliance.etf_holdings import HALAL_ETFS, latest_holdings
        from halal_trader.compliance.sec import SecClient
        from halal_trader.compliance.validate import Verdict, compare
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
        sec = SecClient(settings.edgar.user_agent)
        try:
            holdings = [await latest_holdings(sec, etf) for etf in HALAL_ETFS]
            async with engine.connect() as conn:
                rows = await conn.execute(
                    text(
                        "SELECT symbol, verdict, reasons, metrics->>'market_cap' AS mc "
                        "FROM halal_screen_results "
                        "WHERE as_of = (SELECT max(as_of) FROM halal_screen_results)"
                    )
                )
                verdicts = {
                    r.symbol: Verdict(
                        r.symbol, r.verdict, list(r.reasons), float(r.mc) if r.mc else None
                    )
                    for r in rows
                }
        finally:
            await sec.aclose()
            await engine.dispose()
        if not verdicts:
            raise click.ClickException("no screen results yet: run `compliance screen` first")
        tickers: set[str] = set().union(*(h.tickers for h in holdings))
        return compare(verdicts, tickers), holdings

    v, holdings = asyncio.run(_run())
    for h in holdings:
        console.print(f"{h.etf}: {len(h.holdings)} equity holdings as of {h.period_end}")
    console.print(
        f"ETF names screened: {v.screened_etf_names}/{v.etf_names}; "
        f"we agree on {v.agree} ({v.agreement:.0%})"
    )
    for title, items in (
        ("ETF holds, we REJECT (possible false negatives)", v.etf_held_we_reject),
        ("ETF holds, we are DOUBTFUL (missing data)", v.etf_held_we_doubt),
        (
            "We PASS, no ETF holds, market cap >= $50B (possible false positives)",
            v.large_halal_not_in_etfs,
        ),
    ):
        console.print(f"\n[bold]{title}: {len(items)}[/bold]")
        for item in items[:40]:
            console.print(f"  {item.symbol:6} {'; '.join(item.reasons) or 'passes'}")
    if v.etf_held_not_screened:
        console.print(
            f"\nETF names outside the screened set: {len(v.etf_held_not_screened)} "
            f"(e.g. {', '.join(v.etf_held_not_screened[:10])})"
        )
