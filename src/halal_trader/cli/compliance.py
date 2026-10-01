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
