"""Glue between the prompt-evolution GA and the replay store.

Wave F wires the existing ``prompt_evo`` GA (which is pure logic over
named slot×allele genomes) to a fitness function backed by recorded
cycle replay snapshots. The runner:

1. Pulls the most recent N replay snapshots from the DB.
2. For each candidate genome, scores it against every snapshot via
   the caller-supplied evaluator.
3. Persists the genome + its measured fitness to ``prompt_genomes``
   so the dashboard can render the candidate list.

The evaluator is intentionally a callable supplied by the caller so
this module stays pure-glue — the trading-side decides whether
"fitness" is mean confidence, win-rate-weighted Sharpe, or some
custom blend.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from halal_trader.core.llm.prompt_evo import (
    AllelePool,
    PromptGA,
    PromptGenome,
    ScoredGenome,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.core.replay import CycleSnapshot

logger = logging.getLogger(__name__)


Evaluator = Callable[[PromptGenome, "CycleSnapshot"], Awaitable[float]]


@dataclass
class FitnessRunResult:
    """Outcome of one fitness sweep across the GA population."""

    best: ScoredGenome
    n_snapshots: int


async def evolve_with_replay(
    *,
    engine: "AsyncEngine",
    name: str,
    pool: AllelePool,
    evaluator: Evaluator,
    generations: int = 8,
    population_size: int = 12,
    snapshot_limit: int = 200,
    persist: bool = True,
) -> FitnessRunResult:
    """Run the GA, scoring each genome against recent replay snapshots.

    The evaluator is called once per (genome, snapshot) pair; the
    aggregate fitness for a genome is the **mean** of its per-snapshot
    scores. Returning NaN / inf from the evaluator signals "no signal
    on this snapshot" and that pair is dropped from the average.
    """
    from halal_trader.core.replay import ReplayStore

    store = ReplayStore(engine=engine)
    cycle_ids = await store.list_cycle_ids(limit=snapshot_limit)
    snapshots = []
    for cid in cycle_ids:
        try:
            snapshots.append(await store.read(cid))
        except Exception as exc:  # noqa: BLE001
            logger.debug("skipping unreadable replay %s: %s", cid, exc)

    async def _fitness(genome: PromptGenome) -> float:
        scores: list[float] = []
        for snap in snapshots:
            try:
                v = await evaluator(genome, snap)
            except Exception as exc:  # noqa: BLE001
                logger.debug("evaluator failed on %s: %s", snap.cycle_id, exc)
                continue
            if v != v or v in (float("inf"), float("-inf")):
                continue
            scores.append(v)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    ga = PromptGA(pool=pool, fitness=_fitness, population_size=population_size)
    best = await ga.evolve(generations=generations)

    if persist and snapshots and ga.history:
        # Persist the *final* generation: those are the candidates the
        # operator promotes from. ``best`` is included since
        # ``history[-1]`` is sorted with the best at index 0.
        await _persist_generation(
            engine=engine,
            name=name,
            scored=ga.history[-1],
            n_snapshots=len(snapshots),
        )

    return FitnessRunResult(best=best, n_snapshots=len(snapshots))


async def _persist_generation(
    *,
    engine: "AsyncEngine",
    name: str,
    scored: list[ScoredGenome],
    n_snapshots: int,
) -> None:
    """Write each scored genome of the final generation to the DB."""
    from sqlmodel.ext.asyncio.session import AsyncSession

    from halal_trader.db.models import PromptGenome as PromptGenomeRow

    async with AsyncSession(engine, expire_on_commit=False) as session:
        for cand in scored:
            row = PromptGenomeRow(
                name=name,
                genome=dict(cand.genome.slots),
                fitness=cand.fitness,
                n_cycles=n_snapshots,
                parent_ids=[],
                notes=cand.notes or "",
            )
            session.add(row)
        await session.commit()


async def list_recent_genomes(
    *,
    engine: "AsyncEngine",
    name: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the most recent prompt_genomes rows."""
    from sqlmodel import col, select
    from sqlmodel.ext.asyncio.session import AsyncSession

    from halal_trader.db.models import PromptGenome as PromptGenomeRow

    async with AsyncSession(engine, expire_on_commit=False) as session:
        stmt = select(PromptGenomeRow).order_by(col(PromptGenomeRow.created_at).desc()).limit(limit)
        if name is not None:
            stmt = stmt.where(PromptGenomeRow.name == name)
        result = await session.exec(stmt)
        return [r.model_dump() for r in result.all()]


async def promote_genome(*, engine: "AsyncEngine", genome_id: int) -> bool:
    """Mark a genome as the active prompt for its slot."""
    from datetime import UTC, datetime

    from sqlmodel.ext.asyncio.session import AsyncSession

    from halal_trader.db.models import PromptGenome as PromptGenomeRow
    from halal_trader.db.repository import Repository

    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = await session.get(PromptGenomeRow, genome_id)
        if row is None:
            return False
        row.promoted_at = datetime.now(UTC)
        session.add(row)
        await session.commit()
        await session.refresh(row)

    repo = Repository(engine)
    await repo.set_runtime_config(
        "ACTIVE_PROMPT_VERSION",
        f"{row.name}@genome-{row.id}",
        set_by="prompt_evo",
    )
    return True
