"""Tests for the prompt-evolution GA runner."""

from __future__ import annotations

from halal_trader.core.llm.prompt_evo import AllelePool, PromptGenome
from halal_trader.core.llm.prompt_evo_runner import (
    evolve_with_replay,
    list_recent_genomes,
    promote_genome,
)


async def test_evolve_with_replay_persists_population(engine) -> None:
    """End-to-end: snapshot → evolve → persist → list."""
    # Seed a few replay snapshots.
    from halal_trader.core.replay import CycleSnapshot, ReplayStore

    store = ReplayStore(engine=engine)
    for i in range(3):
        snap = CycleSnapshot.from_inputs(
            cycle_id=f"cycle-{i:08x}",
            market="crypto",
            klines_by_symbol={},
            indicators_cache={},
            halal_pairs=["BTCUSDT"],
            today_pnl=0.0,
        )
        await store.write(snap)

    pool = AllelePool(
        slots={
            "tone": ["formal", "concise", "verbose"],
            "format": ["json", "yaml"],
        }
    )

    async def evaluator(genome: PromptGenome, snap) -> float:
        # Reward "concise + json" — anything else gets less.
        score = 0.0
        if genome.slots.get("tone") == "concise":
            score += 0.5
        if genome.slots.get("format") == "json":
            score += 0.5
        return score

    result = await evolve_with_replay(
        engine=engine,
        name="test_slot",
        pool=pool,
        evaluator=evaluator,
        generations=3,
        population_size=4,
    )
    assert result.n_snapshots == 3
    assert result.best.fitness > 0

    rows = await list_recent_genomes(engine=engine, name="test_slot")
    assert rows
    # The "concise + json" genome should be in there.
    best_in_db = max(rows, key=lambda r: r["fitness"])
    assert best_in_db["fitness"] >= 0.5


async def test_promote_genome_writes_runtime_config(engine) -> None:
    from halal_trader.core.replay import CycleSnapshot, ReplayStore

    store = ReplayStore(engine=engine)
    await store.write(
        CycleSnapshot.from_inputs(
            cycle_id="cycle-aaaa",
            market="crypto",
            klines_by_symbol={},
            indicators_cache={},
            halal_pairs=[],
            today_pnl=0.0,
        )
    )

    pool = AllelePool(slots={"x": ["a", "b"]})

    async def evaluator(genome: PromptGenome, snap) -> float:
        return 1.0 if genome.slots.get("x") == "b" else 0.0

    await evolve_with_replay(
        engine=engine,
        name="promote_slot",
        pool=pool,
        evaluator=evaluator,
        generations=1,
        population_size=2,
    )
    rows = await list_recent_genomes(engine=engine, name="promote_slot")
    target = rows[0]["id"]

    ok = await promote_genome(engine=engine, genome_id=target)
    assert ok is True

    from halal_trader.db.repository import Repository

    repo = Repository(engine)
    cfg = await repo.list_runtime_config()
    assert cfg.get("ACTIVE_PROMPT_VERSION", "").startswith("promote_slot@genome-")
