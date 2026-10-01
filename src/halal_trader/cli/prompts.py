"""Prompt-evolution CLI (Wave F) — operator-facing ad-hoc runs.

Mirrors the dashboard's ``/api/prompts/*`` endpoints so the operator
can list recent candidates or promote a genome from the terminal
without touching the web UI. (The ``evolve`` sweep scored genomes with
the crypto strategy's allele pool and replay fitness; it went with
crypto.)
"""

from __future__ import annotations

import asyncio

import click

from halal_trader.logging import console


@click.group("prompts")
def prompts_group() -> None:
    """Prompt-evolution genetic-algorithm operations."""


@prompts_group.command("candidates")
@click.option("--name", default=None, help="Filter by logical prompt name.")
@click.option("--limit", default=20, show_default=True)
def candidates(name: str | None, limit: int) -> None:
    """List recent prompt_genomes rows + their fitness."""
    asyncio.run(_run_candidates(name=name, limit=limit))


async def _run_candidates(*, name: str | None, limit: int) -> None:
    from halal_trader.config import get_settings
    from halal_trader.core.llm.prompt_evo_runner import list_recent_genomes
    from halal_trader.db import init_db

    settings = get_settings()
    engine = await init_db(settings.database_url)
    rows = await list_recent_genomes(engine=engine, name=name, limit=limit)
    if not rows:
        console.print("[yellow]No prompt_genomes rows yet — run `prompts evolve` first.[/yellow]")
        return
    for r in rows:
        promoted = "[green]promoted[/green]" if r.get("promoted_at") else "candidate"
        console.print(
            f"#{r['id']:>4}  {r['name']:<30}  "
            f"fitness={r.get('fitness', 0):+.4f}  "
            f"n={r.get('n_cycles', 0)}  {promoted}"
        )


@prompts_group.command("promote")
@click.argument("genome_id", type=int)
def promote(genome_id: int) -> None:
    """Mark a candidate genome as the active prompt for its slot.

    Writes ``ACTIVE_PROMPT_VERSION=<name>@genome-<id>`` to
    ``RuntimeConfig``; the next cycle picks it up.
    """
    asyncio.run(_run_promote(genome_id))


async def _run_promote(genome_id: int) -> None:
    from halal_trader.config import get_settings
    from halal_trader.core.llm.prompt_evo_runner import promote_genome
    from halal_trader.db import init_db

    settings = get_settings()
    engine = await init_db(settings.database_url)
    ok = await promote_genome(engine=engine, genome_id=genome_id)
    if ok:
        console.print(f"[green]Promoted genome #{genome_id}.[/green]")
    else:
        console.print(f"[red]Genome #{genome_id} not found.[/red]")
