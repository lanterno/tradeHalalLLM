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
    console.print(Panel(body, title="📈 Halal Stock of the Day (advisory)", border_style="green"))


def _pct(v: Any) -> str:
    return f"{v:+.2f}%" if isinstance(v, int | float) else "—"


def _print_scorecard(sc: dict[str, Any], backfill: dict[str, int], whatif: dict[str, Any]) -> None:
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
    caveat = (
        ""
        if sc.get("sufficient")
        else f"  [yellow]⚠ thin sample — needs ≥{sc.get('min_samples', 20)} to trust[/yellow]"
    )
    body = (
        f"[bold]{sc['n_scored']}[/bold] scored picks "
        f"(of {sc['n_total']} total){caveat}\n\n"
        f"5-day hit rate: [bold]{hit_s}[/bold]\n"
        f"Avg forward return — 1d {_pct(sc.get('avg_fwd_1d'))} · "
        f"5d {_pct(sc.get('avg_fwd_5d'))} · 20d {_pct(sc.get('avg_fwd_20d'))}\n"
        f"Avg 5d excess vs {sc.get('benchmark', 'benchmark')}: "
        f"[bold]{_pct(sc.get('avg_excess_5d'))}[/bold]\n\n"
        f"Best: [green]{best.get('symbol', '—')} {_pct(best.get('fwd_5d'))}[/green] "
        f"({best.get('date', '')})   "
        f"Worst: [red]{worst.get('symbol', '—')} {_pct(worst.get('fwd_5d'))}[/red] "
        f"({worst.get('date', '')})\n"
    )
    if whatif.get("available"):
        body += (
            f"What-if (took every pick): "
            f"[bold]{_pct(whatif.get('total_return_pct'))}[/bold] vs "
            f"{sc.get('benchmark', 'bench')} {_pct(whatif.get('benchmark_return_pct'))}\n"
        )
        if whatif.get("plan_n"):
            exits = sc.get("plan_exit_counts") or {}
            ex = ", ".join(f"{k} {v}" for k, v in exits.items()) or "—"
            body += (
                f"Plan what-if (entry@open, bracket): "
                f"[bold]{_pct(whatif.get('plan_return_pct'))}[/bold] "
                f"over {whatif['plan_n']} picks · exits: {ex}\n"
            )
    if sc.get("n_with_levels"):
        first_hits = sc.get("first_hit_counts") or {}
        fh = ", ".join(f"{k} {v}" for k, v in first_hits.items()) or "—"
        levels_caveat = "" if sc.get("levels_sufficient") else " [yellow]⚠ thin[/yellow]"

        def _rate_s(v: Any) -> str:
            return f"{v:.0%}" if isinstance(v, int | float) else "—"

        body += (
            f"Plan quality ({sc['n_with_levels']} picks with levels{levels_caveat}): "
            f"target hit [green]{_rate_s(sc.get('target_hit_rate'))}[/green] · "
            f"stop hit [red]{_rate_s(sc.get('stop_hit_rate'))}[/red]\n"
            f"  first hit: {fh} · avg MFE {_pct(sc.get('avg_mfe_5d'))} · "
            f"avg MAE {_pct(sc.get('avg_mae_5d'))}\n"
        )
    if sc.get("band_n") or sc.get("candidate_band_n"):
        cov = sc.get("band_coverage_5d")
        cov_s = f"{cov:.0%}" if isinstance(cov, int | float) else "—"
        ccov = sc.get("candidate_band_coverage_5d")
        ccov_s = f"{ccov:.0%}" if isinstance(ccov, int | float) else "—"
        body += (
            f"Quant band coverage (5d path): picks [bold]{cov_s}[/bold] "
            f"(n={sc.get('band_n', 0)}) · all candidates [bold]{ccov_s}[/bold] "
            f"(n={sc.get('candidate_band_n', 0)})\n"
        )
    pick_pct = sc.get("avg_pick_percentile_5d")
    if isinstance(pick_pct, int | float):
        body += (
            f"Counterfactual: pick at the [bold]{pick_pct:.0%}[/bold] percentile "
            f"of its candidates (0.5 = random; n={sc.get('pick_percentile_n', 0)})\n"
        )
    body += (
        f"[dim]backfill: {backfill.get('updated', 0)} updated, "
        f"{backfill.get('scored', 0)} newly scored, "
        f"{backfill.get('skipped', 0)} skipped[/dim]"
    )
    console.print(Panel(body, title="📊 Recommendation Scorecard (advisory)", border_style="cyan"))


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
@click.option(
    "--audit-outcomes",
    is_flag=True,
    help=(
        "Re-label already-scored picks against a bar window that covers their "
        "rec date (repairs rows mislabeled by the old close-window anchoring)"
    ),
)
def recommend(show: bool, scorecard: bool, audit_outcomes: bool) -> None:
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

        if audit_outcomes:
            from halal_trader.mcp.client import AlpacaMCPClient
            from halal_trader.recommendation.scorecard import audit_scored_outcomes

            mcp = AlpacaMCPClient()
            await mcp.connect()
            try:
                console.print("[dim]Auditing scored picks against covering bar windows…[/dim]")
                res = await audit_scored_outcomes(mcp, repo)
            finally:
                await mcp.disconnect()
            console.print(
                f"Audit: [bold]{res['audited']}[/bold] audited, "
                f"[bold]{res['repaired']}[/bold] repaired, "
                f"{res['skipped']} marked skipped"
            )
            return

        if scorecard:
            from halal_trader.mcp.client import AlpacaMCPClient
            from halal_trader.recommendation.scorecard import (
                backfill_outcomes,
                compute_scorecard,
                whatif_equity_curve,
            )

            mcp = AlpacaMCPClient()
            await mcp.connect()
            try:
                console.print("[dim]Backfilling matured forward returns…[/dim]")
                res = await backfill_outcomes(mcp, repo)
            finally:
                await mcp.disconnect()
            sc = await compute_scorecard(repo)
            wc = await whatif_equity_curve(repo)
            _print_scorecard(sc, res, wc)
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
