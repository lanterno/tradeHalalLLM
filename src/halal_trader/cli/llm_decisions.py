"""Inspect and replay logged LLM decisions.

These commands let an operator audit the LLM trail without writing SQL:

* ``llm-decisions list`` — recent decisions with cost/token summary
* ``llm-decisions show ID`` — full row including raw response and parsed action
* ``llm-decisions cost-summary`` — daily spend roll-up by provider/model

Today we store ``prompt_summary`` and ``raw_response`` (not the full
assembled prompt), so a true "re-run against the same LLM" workflow
also needs the snapshot store at ``core/replay.py``. These commands are
the audit half of the replay story; ``halal-trader insights replay``
covers the rerun half.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import click
import sqlalchemy as sa
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import LlmDecision
from halal_trader.logging import console


def _format_cost(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.4f}"


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


@click.group("llm-decisions")
def llm_decisions() -> None:
    """Inspect logged LLM decisions (cost, tokens, prompt version, raw output)."""


@llm_decisions.command("list")
@click.option("--limit", default=20, show_default=True, help="Max rows to show.")
@click.option(
    "--prompt-version",
    default=None,
    help="Filter by prompt registry id, e.g. 'crypto.strategy.system@abc123…'.",
)
@click.option("--provider", default=None, help="Filter by provider name (anthropic, openai, …).")
def list_cmd(limit: int, prompt_version: str | None, provider: str | None) -> None:
    """List recent LLM decisions, newest first."""
    from rich.table import Table

    from halal_trader.config import get_settings
    from halal_trader.db.models import init_db

    async def _run() -> None:
        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            from sqlmodel import select

            stmt = select(LlmDecision).order_by(LlmDecision.id.desc()).limit(limit)
            if prompt_version:
                stmt = stmt.where(LlmDecision.prompt_version == prompt_version)
            if provider:
                stmt = stmt.where(LlmDecision.provider == provider)

            async with AsyncSession(engine) as session:
                rows = (await session.exec(stmt)).all()

            if not rows:
                console.print("[yellow]No matching decisions found.[/yellow]")
                return

            table = Table(title=f"Recent LLM Decisions (showing {len(rows)})")
            table.add_column("id", justify="right")
            table.add_column("timestamp")
            table.add_column("provider")
            table.add_column("model")
            table.add_column("prompt_version", style="cyan")
            table.add_column("in", justify="right")
            table.add_column("out", justify="right")
            table.add_column("cache_r", justify="right")
            table.add_column("cost", justify="right")
            table.add_column("ms", justify="right")

            for r in rows:
                table.add_row(
                    str(r.id),
                    r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    r.provider,
                    r.model,
                    r.prompt_version or "—",
                    _format_tokens(r.input_tokens),
                    _format_tokens(r.output_tokens),
                    _format_tokens(r.cache_read_tokens),
                    _format_cost(r.cost_usd),
                    str(r.execution_ms or "—"),
                )
            console.print(table)
        finally:
            await engine.dispose()

    asyncio.run(_run())


@llm_decisions.command("show")
@click.argument("decision_id", type=int)
def show_cmd(decision_id: int) -> None:
    """Print the full record for a single decision id."""
    from halal_trader.config import get_settings
    from halal_trader.db.models import init_db

    async def _run() -> None:
        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            async with AsyncSession(engine) as session:
                row = await session.get(LlmDecision, decision_id)
                if row is None:
                    console.print(f"[red]No decision found with id={decision_id}[/red]")
                    return

            console.print(f"[bold]LLM Decision #{row.id}[/bold] @ {row.timestamp}")
            console.print(f"  provider:        {row.provider}")
            console.print(f"  model:           {row.model}")
            console.print(f"  prompt_version:  {row.prompt_version or '—'}")
            console.print(
                f"  tokens:          input={_format_tokens(row.input_tokens)}, "
                f"output={_format_tokens(row.output_tokens)}, "
                f"cache_read={_format_tokens(row.cache_read_tokens)}, "
                f"cache_write={_format_tokens(row.cache_write_tokens)}"
            )
            console.print(f"  cost_usd:        {_format_cost(row.cost_usd)}")
            console.print(f"  execution_ms:    {row.execution_ms or '—'}")
            console.print(f"  prompt_summary:  {row.prompt_summary or '—'}")

            if row.symbols:
                try:
                    syms = json.loads(row.symbols)
                    console.print(f"  symbols:         {', '.join(syms) if syms else '—'}")
                except Exception:
                    console.print(f"  symbols:         {row.symbols}")

            if row.parsed_action:
                console.print("\n[bold]parsed_action:[/bold]")
                try:
                    console.print_json(row.parsed_action)
                except Exception:
                    console.print(row.parsed_action)

            if row.raw_response:
                console.print("\n[bold]raw_response:[/bold]")
                # raw_response is the LLM's serialised JSON blob — pretty-print
                # if it parses, otherwise dump the raw text so we never hide it.
                try:
                    console.print_json(row.raw_response)
                except Exception:
                    console.print(row.raw_response)

            if row.thinking:
                console.print("\n[bold]thinking:[/bold]")
                console.print(row.thinking)
        finally:
            await engine.dispose()

    asyncio.run(_run())


@llm_decisions.command("cost-summary")
@click.option("--days", default=7, show_default=True, help="Lookback window in days.")
def cost_summary_cmd(days: int) -> None:
    """Daily LLM spend totals broken down by provider + model."""
    from rich.table import Table

    from halal_trader.config import get_settings
    from halal_trader.db.models import init_db

    cutoff = datetime.now(UTC) - timedelta(days=days)

    async def _run() -> None:
        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            async with engine.connect() as conn:
                result = await conn.execute(
                    sa.text(
                        """
                        SELECT date(timestamp) AS day,
                               provider,
                               model,
                               COUNT(*) AS calls,
                               COALESCE(SUM(input_tokens), 0) AS input_tok,
                               COALESCE(SUM(output_tokens), 0) AS output_tok,
                               COALESCE(SUM(cache_read_tokens), 0) AS cache_tok,
                               COALESCE(SUM(cost_usd), 0) AS cost
                        FROM llm_decisions
                        WHERE timestamp >= :cutoff
                        GROUP BY day, provider, model
                        ORDER BY day DESC, cost DESC
                        """
                    ),
                    {"cutoff": cutoff},
                )
                rows = result.all()

            if not rows:
                console.print(f"[yellow]No decisions in the last {days} days.[/yellow]")
                return

            table = Table(title=f"LLM Cost Summary (last {days} days)")
            right_cols = {"calls", "input", "output", "cache_read", "cost"}
            cols = ("day", "provider", "model", "calls", "input", "output", "cache_read", "cost")
            for col in cols:
                table.add_column(col, justify="right" if col in right_cols else "left")

            day_totals: dict[str, float] = defaultdict(float)
            for day, provider, model, calls, in_tok, out_tok, cache_tok, cost in rows:
                table.add_row(
                    str(day),
                    provider,
                    model,
                    str(calls),
                    f"{int(in_tok):,}",
                    f"{int(out_tok):,}",
                    f"{int(cache_tok):,}",
                    f"${float(cost):.4f}",
                )
                day_totals[str(day)] += float(cost)

            console.print(table)
            console.print("\n[bold]Daily totals:[/bold]")
            for day in sorted(day_totals.keys(), reverse=True):
                console.print(f"  {day}: ${day_totals[day]:.4f}")
        finally:
            await engine.dispose()

    asyncio.run(_run())
