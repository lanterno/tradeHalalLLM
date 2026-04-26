"""Tests for prompt-evolution GA."""

from __future__ import annotations

import random

import pytest

from halal_trader.core.llm.prompt_evo import (
    AllelePool,
    PromptGA,
    PromptGenome,
    crossover,
)


def _pool() -> AllelePool:
    return AllelePool(
        slots={
            "tone": ["calm", "ruthless", "neutral"],
            "sizing": ["aggressive", "kelly", "conservative"],
            "format": ["json-only", "json+thinking"],
        }
    )


# ── Genome ────────────────────────────────────────────────────────


def test_render_uses_slots() -> None:
    g = PromptGenome(slots={"a": "1", "b": "2"})
    assert g.render("a={a}; b={b}") == "a=1; b=2"


def test_genome_eq_and_hash() -> None:
    a = PromptGenome(slots={"x": "1"})
    b = PromptGenome(slots={"x": "1"})
    c = PromptGenome(slots={"x": "2"})
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    s = {a, b, c}
    assert len(s) == 2


def test_diff_reports_mismatched_slots() -> None:
    a = PromptGenome(slots={"x": "1", "y": "2"})
    b = PromptGenome(slots={"x": "1", "y": "3"})
    d = a.diff(b)
    assert d == {"y": ("2", "3")}


# ── Pool ──────────────────────────────────────────────────────────


def test_base_genome_uses_first_allele() -> None:
    g = _pool().base_genome()
    assert g.slots == {"tone": "calm", "sizing": "aggressive", "format": "json-only"}


def test_random_genome_uses_each_slot() -> None:
    rng = random.Random(0)
    g = _pool().random_genome(rng)
    assert set(g.slots) == {"tone", "sizing", "format"}


def test_mutate_changes_one_slot() -> None:
    rng = random.Random(0)
    pool = _pool()
    base = pool.base_genome()
    mut = pool.mutate(base, rng)
    assert mut != base
    diffs = base.diff(mut)
    assert len(diffs) == 1


def test_mutate_with_no_alternatives_returns_unchanged() -> None:
    rng = random.Random(0)
    pool = AllelePool(slots={"x": ["only"]})
    g = pool.base_genome()
    assert pool.mutate(g, rng) == g


# ── Crossover ─────────────────────────────────────────────────────


def test_crossover_picks_per_slot() -> None:
    rng = random.Random(0)
    a = PromptGenome(slots={"x": "1", "y": "2"})
    b = PromptGenome(slots={"x": "3", "y": "4"})
    child = crossover(a, b, rng)
    assert child.slots["x"] in ("1", "3")
    assert child.slots["y"] in ("2", "4")


def test_crossover_handles_disjoint_slots() -> None:
    rng = random.Random(0)
    a = PromptGenome(slots={"x": "1"})
    b = PromptGenome(slots={"y": "2"})
    child = crossover(a, b, rng)
    # Both slots present (each only in one parent)
    assert "x" in child.slots
    assert "y" in child.slots


# ── GA driver ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evolve_finds_known_optimum() -> None:
    """Fitness counts how many slots match a target genome."""
    pool = _pool()
    target = PromptGenome(slots={"tone": "ruthless", "sizing": "kelly", "format": "json+thinking"})

    async def fitness(g: PromptGenome) -> float:
        return float(sum(1 for k, v in g.slots.items() if target.slots.get(k) == v))

    ga = PromptGA(
        pool=pool,
        fitness=fitness,
        population_size=8,
        elite_count=2,
        seed=42,
    )
    best = await ga.evolve(generations=12)
    assert best.fitness == 3.0
    assert best.genome == target


@pytest.mark.asyncio
async def test_evolve_caches_repeat_evaluations() -> None:
    pool = AllelePool(slots={"only": ["a", "b"]})
    calls = 0

    async def fitness(g: PromptGenome) -> float:
        nonlocal calls
        calls += 1
        return 1.0 if g.slots["only"] == "a" else 0.0

    ga = PromptGA(pool=pool, fitness=fitness, population_size=4, elite_count=1, seed=0)
    await ga.evolve(generations=6)
    # Only 2 unique genomes possible — cache must have prevented many repeats.
    assert calls <= 4
    assert ga.cache_size() == 2


@pytest.mark.asyncio
async def test_evolve_failed_fitness_scored_zero() -> None:
    pool = AllelePool(slots={"x": ["a", "b"]})

    async def fitness(_g: PromptGenome) -> float:
        raise RuntimeError("boom")

    ga = PromptGA(pool=pool, fitness=fitness, population_size=2, elite_count=1, seed=0)
    best = await ga.evolve(generations=2)
    assert best.fitness == 0.0


@pytest.mark.asyncio
async def test_evolve_seed_genomes_present_in_initial_population() -> None:
    pool = _pool()
    seed = PromptGenome(slots={"tone": "ruthless", "sizing": "conservative", "format": "json-only"})

    async def fitness(g: PromptGenome) -> float:
        return 1.0 if g == seed else 0.0

    ga = PromptGA(pool=pool, fitness=fitness, population_size=6, elite_count=1, seed=0)
    best = await ga.evolve(generations=1, seed_genomes=[seed])
    assert best.fitness == 1.0


@pytest.mark.asyncio
async def test_history_has_one_entry_per_generation() -> None:
    pool = _pool()

    async def fitness(_g: PromptGenome) -> float:
        return 0.5

    ga = PromptGA(pool=pool, fitness=fitness, population_size=4, seed=0)
    await ga.evolve(generations=3)
    # initial gen + 3 evolution gens
    assert len(ga.history) == 4
