"""Prompt-evolution CLI (Wave F) — operator-facing ad-hoc runs.

Mirrors the dashboard's ``/api/prompts/*`` endpoints so the operator
can run an evolution sweep, list recent candidates, or promote a
genome from the terminal without touching the web UI.
"""

from __future__ import annotations

import asyncio

import click

from halal_trader.logging import console


@click.group("prompts")
def prompts_group() -> None:
    """Prompt-evolution genetic-algorithm operations."""


@prompts_group.command("evolve")
@click.option(
    "--name",
    default="crypto.strategy.system",
    show_default=True,
    help="Logical prompt name — namespaces genome rows in `prompt_genomes`.",
)
@click.option("--generations", default=8, show_default=True, help="GA generations to run.")
@click.option("--population", default=12, show_default=True, help="Genomes per generation.")
@click.option(
    "--snapshots",
    default=200,
    show_default=True,
    help="Replay snapshot count to score each genome against.",
)
@click.option(
    "--evaluator",
    type=click.Choice(["replay_pnl", "confidence_proxy"]),
    default="replay_pnl",
    show_default=True,
    help="Which Wave F fitness function to use.",
)
def evolve(name: str, generations: int, population: int, snapshots: int, evaluator: str) -> None:
    """Run one GA sweep over recent replay snapshots and persist candidates."""
    asyncio.run(
        _run_evolve(
            name=name,
            generations=generations,
            population=population,
            snapshot_limit=snapshots,
            evaluator_name=evaluator,
        )
    )


async def _run_evolve(
    *,
    name: str,
    generations: int,
    population: int,
    snapshot_limit: int,
    evaluator_name: str,
) -> None:
    from halal_trader.config import get_settings
    from halal_trader.core.llm.prompt_evo_runner import evolve_with_replay
    from halal_trader.crypto.prompt_fitness import (
        confidence_proxy_fitness,
        replay_pnl_fitness,
    )
    from halal_trader.crypto.prompts import crypto_allele_pool
    from halal_trader.db import init_db

    settings = get_settings()
    engine = await init_db(settings.database_url)
    evaluator = (
        confidence_proxy_fitness if evaluator_name == "confidence_proxy" else replay_pnl_fitness
    )

    console.print(
        f"[yellow]Running prompt evolution: {generations}gen × "
        f"{population}pop over {snapshot_limit} snapshots, "
        f"evaluator={evaluator_name}…[/yellow]"
    )
    result = await evolve_with_replay(
        engine=engine,
        name=name,
        pool=crypto_allele_pool(),
        evaluator=evaluator,
        generations=generations,
        population_size=population,
        snapshot_limit=snapshot_limit,
    )
    console.print(
        f"[green]Best fitness: {result.best.fitness:+.4f} "
        f"(over {result.n_snapshots} snapshots)[/green]"
    )
    for slot, allele in result.best.genome.slots.items():
        truncated = allele[:80] + ("…" if len(allele) > 80 else "")
        console.print(f"  {slot}: {truncated!r}")


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
