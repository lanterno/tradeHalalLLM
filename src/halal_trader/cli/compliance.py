"""`halal-trader compliance`: the in-house Shariah screen (compliance/)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import click

from halal_trader.cli._run import fail, run_db
from halal_trader.logging import console


@click.group("compliance")
def compliance() -> None:
    """In-house AAOIFI-style screen from SEC data (research-grade until validated)."""


@compliance.command("screen")
@click.option("--symbols", default="", help="Comma-separated; default = every stock in daily_bars.")
def screen_cmd(symbols: str) -> None:
    """Screen symbols against AAOIFI business-activity and financial-ratio rules."""

    async def _run(engine: Any, settings: Any) -> list[Any]:

        from halal_trader.compliance.runner import run_screen
        from halal_trader.compliance.sec import SecClient
        from halal_trader.data.store import BENCHMARKS
        from halal_trader.market_hours import today_eastern

        sec = SecClient(settings.edgar.user_agent)
        try:
            chosen = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            if not chosen:
                from halal_trader.data.store import stored_symbols

                chosen = [s for s in await stored_symbols(engine) if s not in BENCHMARKS]
            if not chosen:
                fail("no symbols: pass --symbols or run `data backfill`")
            return await run_screen(sec, engine, chosen, today_eastern())
        finally:
            await sec.aclose()

    results = run_db(_run)
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

    async def _run(engine: Any, settings: Any) -> dict[Any, int]:
        from halal_trader.compliance.history import screen_history
        from halal_trader.compliance.sec import SecClient
        from halal_trader.market_hours import today_eastern

        sec = SecClient(settings.edgar.user_agent)
        try:
            return await screen_history(
                sec, engine, start=since.date(), end=today_eastern(), top_n=top
            )
        finally:
            await sec.aclose()

    done = run_db(_run)
    for as_of, halal in sorted(done.items()):
        console.print(f"  {as_of}: {halal} halal")
    console.print(f"{len(done)} quarter(s) screened")


@compliance.command("map-delisted")
def map_delisted_cmd() -> None:
    """Match tickers SEC no longer lists to their filer by name, then re-screen them."""

    async def _run(engine: Any, settings: Any) -> tuple[list[Any], dict[Any, int]]:
        from halal_trader.compliance.delisted import map_unmapped, rescreen_mapped
        from halal_trader.compliance.sec import SecClient
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.market_hours import today_eastern

        sec = SecClient(settings.edgar.user_agent)
        market = AlpacaMarketData.from_settings(settings)
        try:
            assets = [*await market.assets(), *await market.inactive_assets()]
            matches = await map_unmapped(sec, engine, assets, today=today_eastern())
            return matches, await rescreen_mapped(sec, engine)
        finally:
            await market.aclose()
            await sec.aclose()

    matches, rescreened = run_db(_run)
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


@compliance.command("rescreen")
def rescreen_cmd() -> None:
    """Re-screen every stored verdict made under an older screening method (resumable)."""

    async def _run(engine: Any, settings: Any) -> dict[Any, int]:
        from halal_trader.compliance.history import rescreen_stale
        from halal_trader.compliance.sec import SecClient

        sec = SecClient(settings.edgar.user_agent)
        try:
            return await rescreen_stale(sec, engine)
        finally:
            await sec.aclose()

    done = run_db(_run)
    console.print(f"re-screened {sum(done.values())} verdict(s) across {len(done)} screen date(s)")


@compliance.command("rescreen-renamed")
@click.option("--dry-run", is_flag=True, help="List the affected rows; screen and store nothing.")
def rescreen_renamed_cmd(dry_run: bool) -> None:
    """Re-screen the stored rows an index's holding under a renamed company's old ticker
    changes (resumable)."""

    async def _run(engine: Any, settings: Any) -> list[Any]:
        from halal_trader.compliance.renamed import rescreen_renamed
        from halal_trader.compliance.sec import SecClient

        sec = SecClient(settings.edgar.user_agent)
        try:
            return await rescreen_renamed(sec, engine, dry_run=dry_run)
        finally:
            await sec.aclose()

    done = run_db(_run)
    for a, verdict in done:
        after = "" if verdict is None else f", re-screened {verdict}"
        console.print(
            f"  {a.as_of} {a.symbol:6} {a.stored} -> {a.replayed}{after}  [dim]{a.why}[/dim]"
        )
    dates = len({a.as_of for a, _ in done})
    did = "would be re-screened" if dry_run else "re-screened"
    console.print(f"{len(done)} row(s) across {dates} screen date(s) {did}")


@compliance.command("etf-history")
def etf_history_cmd() -> None:
    """Store every N-PORT holdings filing of SPUS and HLAL (the index veto's input)."""

    async def _run(engine: Any, settings: Any) -> int:
        from halal_trader.compliance.etf_holdings import sync_holdings
        from halal_trader.compliance.sec import SecClient

        sec = SecClient(settings.edgar.user_agent)
        try:
            return await sync_holdings(sec, engine)
        finally:
            await sec.aclose()

    console.print(f"{run_db(_run)} holdings filing(s) stored")


@compliance.command("validate")
def validate_cmd() -> None:
    """Compare the latest screen with SPUS and HLAL holdings (SEC N-PORT)."""

    async def _run(engine: Any, settings: Any) -> tuple[Any, list[Any]]:
        from sqlalchemy import text

        from halal_trader.compliance.etf_holdings import HALAL_ETFS, latest_holdings
        from halal_trader.compliance.sec import SecClient
        from halal_trader.compliance.validate import compare, verdict_of

        sec = SecClient(settings.edgar.user_agent)
        try:
            holdings = [await latest_holdings(sec, etf) for etf in HALAL_ETFS]
            async with engine.connect() as conn:
                rows = await conn.execute(
                    text(
                        "SELECT symbol, verdict, reasons, metrics->>'market_cap' AS mc, "
                        "metrics->>'implied_debt_ratio' AS implied, "
                        "metrics->>'debt_ratio' AS debt FROM halal_screen_current "
                        "WHERE as_of = (SELECT max(as_of) FROM halal_screen_results)"
                    )
                )
                verdicts = {r.symbol: verdict_of(r) for r in rows}
        finally:
            await sec.aclose()
        if not verdicts:
            fail("no screen results yet: run `compliance screen` first")
        tickers: set[str] = set().union(*(h.tickers for h in holdings))
        return compare(verdicts, tickers), holdings

    v, holdings = run_db(_run)
    for h in holdings:
        console.print(f"{h.etf}: {len(h.holdings)} equity holdings as of {h.period_end}")
    console.print(
        f"ETF names screened: {v.screened_etf_names}/{v.etf_names}; "
        f"we agree on {v.agree} ({v.agreement:.0%})"
    )
    by_kind = v.rejected_by_kind
    for title, items in (
        ("ETF holds, we REJECT on a ratio (possible false negatives)", by_kind.get("ratio", [])),
        ("ETF holds, we REJECT on activity (a judgment)", by_kind.get("activity", [])),
        ("ETF holds, the other ETF's index VETOES it", by_kind.get("veto", [])),
        ("ETF holds, we are DOUBTFUL (missing data)", by_kind.get("data", [])),
        (
            "We PASS, no ETF holds, market cap >= $50B (possible false positives)",
            v.large_halal_not_in_etfs,
        ),
        (
            "We PASS, interest expense implies debt >= 30% (possible false positives)",
            v.implied_debt_suspects,
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
