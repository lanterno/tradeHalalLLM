"""``halal-trader recommend`` — the daily halal "stock of the day".

Advisory only: generates (or shows) the LLM-picked most-promising halal
stock. Never places an order. Heavy modules are imported inside the command
body so ``--help`` stays fast (matches the rest of the CLI).
"""

from __future__ import annotations

import asyncio
from typing import Any

import click

from halal_trader.logging import console


def _print_rec(rec: dict[str, Any]) -> None:
    from rich.panel import Panel

    def _lvl(v: Any) -> str:
        return f"${v:,.2f}" if isinstance(v, int | float) else "—"

    conviction = float(rec.get("conviction") or 0.0)
    body = (
        f"[bold green]{rec['symbol']}[/bold green]   "
        f"conviction [bold]{conviction:.0%}[/bold]\n"
        f"[dim]{rec.get('date', '')} · {rec.get('universe_size', 0)} candidates · "
        f"{rec.get('model') or 'llm'}[/dim]\n\n"
        f"[bold]Thesis[/bold]\n{rec.get('thesis') or '—'}\n\n"
        f"[bold]Halal note[/bold]\n{rec.get('halal_note') or '—'}\n\n"
        f"Entry {_lvl(rec.get('suggested_entry'))}   "
        f"Target [green]{_lvl(rec.get('suggested_target'))}[/green]   "
        f"Stop [red]{_lvl(rec.get('suggested_stop'))}[/red]"
    )
    if rec.get("catalysts"):
        body += f"\n\n[bold]Catalysts[/bold]\n{rec['catalysts']}"
    if rec.get("risks"):
        body += f"\n\n[bold]Risks[/bold]\n{rec['risks']}"
    console.print(
        Panel(body, title="📈 Halal Stock of the Day (advisory)", border_style="green")
    )


def _pct(v: Any) -> str:
    return f"{v:+.2f}%" if isinstance(v, int | float) else "—"


def _print_scorecard(sc: dict[str, Any], backfill: dict[str, int]) -> None:
    from rich.panel import Panel

    if not sc.get("available"):
        console.print(
            Panel(
                f"No matured picks yet ({sc.get('n_total', 0)} total, "
                f"0 scored).\nForward returns fill in over the days after each pick.",
                title="📊 Recommendation Scorecard",
                border_style="yellow",
            )
        )
        return
    hit = sc.get("hit_rate_5d")
    hit_s = f"{hit:.0%}" if isinstance(hit, int | float) else "—"
    best, worst = sc.get("best", {}), sc.get("worst", {})
    body = (
        f"[bold]{sc['n_scored']}[/bold] scored picks "
        f"(of {sc['n_total']} total)\n\n"
        f"5-day hit rate: [bold]{hit_s}[/bold]\n"
        f"Avg forward return — 1d {_pct(sc.get('avg_fwd_1d'))} · "
        f"5d {_pct(sc.get('avg_fwd_5d'))} · 20d {_pct(sc.get('avg_fwd_20d'))}\n"
        f"Avg 5d excess vs {sc.get('benchmark', 'benchmark')}: "
        f"[bold]{_pct(sc.get('avg_excess_5d'))}[/bold]\n\n"
        f"Best: [green]{best.get('symbol', '—')} {_pct(best.get('fwd_5d'))}[/green] "
        f"({best.get('date', '')})   "
        f"Worst: [red]{worst.get('symbol', '—')} {_pct(worst.get('fwd_5d'))}[/red] "
        f"({worst.get('date', '')})\n"
        f"[dim]backfill: {backfill.get('updated', 0)} updated, "
        f"{backfill.get('scored', 0)} newly scored[/dim]"
    )
    console.print(
        Panel(body, title="📊 Recommendation Scorecard (advisory)", border_style="cyan")
    )


@click.command()
@click.option(
    "--show",
    is_flag=True,
    help="Show the latest saved recommendation without regenerating (no LLM call)",
)
@click.option(
    "--scorecard",
    is_flag=True,
    help="Backfill matured picks' forward returns and print the track record (no LLM call)",
)
def recommend(show: bool, scorecard: bool) -> None:
    """Generate (or --show / --scorecard) the daily halal stock-of-the-day."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.db.repository import Repository

        settings = get_settings()
        engine = await init_db(settings.database_url)
        repo = Repository(engine)

        if show:
            rec = await repo.get_latest_recommendation()
            if rec is None:
                console.print(
                    "[yellow]No recommendation yet — run "
                    "`halal-trader recommend` to generate one.[/yellow]"
                )
                return
            _print_rec(rec)
            return

        if scorecard:
            from halal_trader.mcp.client import AlpacaMCPClient
            from halal_trader.recommendation.scorecard import (
                backfill_outcomes,
                compute_scorecard,
            )

            mcp = AlpacaMCPClient()
            await mcp.connect()
            try:
                console.print("[dim]Backfilling matured forward returns…[/dim]")
                res = await backfill_outcomes(mcp, repo)
            finally:
                await mcp.disconnect()
            sc = await compute_scorecard(repo)
            _print_scorecard(sc, res)
            return

        from halal_trader.mcp.client import AlpacaMCPClient
        from halal_trader.recommendation.engine import DailyRecommendationEngine

        mcp = AlpacaMCPClient()
        await mcp.connect()
        try:
            eng = DailyRecommendationEngine(broker=mcp, repo=repo, settings=settings)
            console.print("[dim]Analysing the halal universe…[/dim]")
            rec = await eng.generate()
        finally:
            await mcp.disconnect()
        _print_rec(rec)

    asyncio.run(_run())
