"""Insights CLI — surface the new analysis modules at the terminal.

Each subcommand is a thin Click wrapper over one analysis store
(purification ledger, catalysts, RAG,
the halal exception queue, compliance receipts).

Heavy modules (sqlmodel, ml) are imported inside command bodies
to keep ``--help`` fast — same pattern the rest of the CLI uses.
"""

from __future__ import annotations

import asyncio

import click

# ── group ────────────────────────────────────────────────────────


@click.group("insights")
def insights() -> None:
    """Run analytics over recent trades and synthetic scenarios."""


@insights.command("purification")
def purification_cmd() -> None:
    """Outstanding round-trip purification due."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.halal.round_trip_purification import (
            RoundTripLedger,
            outstanding_round_trip_due,
        )
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            ledger = RoundTripLedger(engine=engine)
            summary = await outstanding_round_trip_due(ledger)
            if summary["n_entries"] == 0:
                console.print("[yellow]No purification ledger yet — no closed wins.[/]")
                return
            console.print(f"[bold]Outstanding:[/] ${summary['total_usd']:.2f}")
            console.print(f"Disbursed total: ${summary['disbursed_total_usd']:.2f}")
            console.print(f"Total entries: {summary['n_entries']}")
            if summary["by_symbol"]:
                console.print("[bold]By symbol:[/]")
                for sym, due in sorted(summary["by_symbol"].items(), key=lambda kv: -kv[1]):
                    console.print(f"  {sym:<10} ${due:.2f}")
        finally:
            await engine.dispose()

    asyncio.run(_run())


@insights.command("catalysts")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--lookahead",
    default=24,
    show_default=True,
    help="Hours of history/forward-window for time-bound sources",
)
def catalysts_cmd(symbols: tuple[str, ...], lookahead: int) -> None:
    """Inspect what catalysts the stock cycle will surface for SYMBOLS.

    Constructs the same source bundle the live trading scheduler builds
    (FRED + EDGAR + Options-IV + Fed-speak), runs each for the given
    symbols, and prints the assembled catalyst list — exactly what the
    LLM would see in the next cycle's prompt.
    """

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.logging import console
        from halal_trader.trading.catalysts import (
            StockCatalystFeed,
            format_catalysts_for_prompt,
        )

        settings = get_settings()
        sources: list = []

        if settings.fred.api_key:
            from halal_trader.trading.fred_catalysts import (
                FREDReleaseCalendarSource,
            )

            sources.append(FREDReleaseCalendarSource(api_key=settings.fred.api_key))
            console.print("[green]✓[/] FRED enabled")
        else:
            console.print("[yellow]✗[/] FRED disabled (no FRED_API_KEY)")

        if settings.edgar.user_agent:
            from halal_trader.trading.edgar_catalysts import EDGAREightKSource

            sources.append(EDGAREightKSource(user_agent=settings.edgar.user_agent))
            console.print("[green]✓[/] EDGAR enabled")
        else:
            console.print("[yellow]✗[/] EDGAR disabled (no EDGAR_USER_AGENT)")

        from halal_trader.trading.fed_speak_adapter import FedSpeakCatalystSource

        sources.append(FedSpeakCatalystSource())
        console.print("[green]✓[/] Fed-speak enabled (always-on)")
        console.print()

        feed = StockCatalystFeed(sources=sources)
        catalysts = await feed.fetch_all([s.upper() for s in symbols])

        console.print(f"[bold]Catalysts for {', '.join(s.upper() for s in symbols)}:[/]")
        if not catalysts:
            console.print("[yellow](no catalysts in window)[/]")
            return
        text = format_catalysts_for_prompt(
            catalysts, symbols=[s.upper() for s in symbols], max_age_hours=lookahead
        )
        console.print(text or "[yellow](all catalysts older than lookahead)[/]")

        # Close any sources that hold open clients.
        for src in sources:
            if hasattr(src, "aclose"):
                try:
                    await src.aclose()
                except Exception:  # noqa: BLE001
                    pass

    asyncio.run(_run())


@insights.command("rag")
@click.option("--query", "-q", default="", help="Text to retrieve analogues for")
@click.option("--k", default=5, show_default=True)
def rag_cmd(query: str, k: int) -> None:
    """Top-K most-similar past trade rationales by cosine of hashed BoW."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core.llm.rag import format_rag_for_prompt
        from halal_trader.core.llm.rag_db import DBRationaleStore
        from halal_trader.db.models import init_db
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            store = DBRationaleStore(engine=engine)
            size = await store.size()
            if size == 0:
                console.print("[yellow]RAG store empty — close some trades first.[/]")
                return
            if not query:
                console.print(f"[bold]RAG store:[/] {size} rationale(s)")
                from sqlalchemy.ext.asyncio import async_sessionmaker
                from sqlmodel import select

                from halal_trader.db.models import RationaleRow as _Row

                sm = async_sessionmaker(engine, expire_on_commit=False)
                async with sm() as s:
                    rows = (
                        (await s.execute(select(_Row).order_by(_Row.timestamp.desc()).limit(10)))
                        .scalars()
                        .all()
                    )
                for r in rows:
                    outcome = "WIN" if r.outcome_win else "LOSS"
                    console.print(f"  {outcome} {r.outcome_pnl_pct:+.2%} {r.symbol}: {r.text[:80]}")
                return
            hits = await store.query(query, k=k, min_similarity=0.0)
            console.print(format_rag_for_prompt(hits, max_rows=k))
            agg = await store.aggregate(hits)
            console.print(
                f"\n[bold]Weighted outcome:[/] pnl={agg['weighted_pnl_pct']:+.2%} "
                f"win-rate={agg['weighted_win_rate']:.0%} (n={agg['n']})"
            )
        finally:
            await engine.dispose()

    asyncio.run(_run())


@insights.command("exceptions")
@click.option("--status", default="pending", show_default=True)
def exceptions_cmd(status: str) -> None:
    """List Sharia exception queue entries (pending by default)."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.halal.exception_queue import (
            ExceptionQueue,
            render_summary,
        )
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            q = ExceptionQueue(engine=engine)
            rows = await q.all() if status == "all" else await q.by_status(status)  # type: ignore[arg-type]
            console.print(render_summary(rows))
        finally:
            await engine.dispose()

    asyncio.run(_run())


@insights.command("explain")
@click.argument("asset_class", type=click.Choice(["stock"]))
@click.argument("trade_id", type=int)
def explain_cmd(asset_class: str, trade_id: int) -> None:
    """Render the halal-compliance explanation for one trade (Wave L)."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.halal.audit import export_receipt
        from halal_trader.halal.explainer import explain_screening
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            receipt = await export_receipt(engine, trade_id=trade_id, asset_class=asset_class)
            if receipt is None:
                console.print(f"[red]No {asset_class} trade with id {trade_id}[/]")
                return
            explanation = explain_screening(receipt.payload)
            console.print(explanation.body_md)
            if explanation.sources:
                console.print("\n[dim]Sources:[/]")
                for s in explanation.sources:
                    console.print(f"  · {s}")
        finally:
            await engine.dispose()

    asyncio.run(_run())
