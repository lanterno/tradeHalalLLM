"""Prompt-evolution genetic algorithm.

Treat a prompt template as a *genome*: a fixed list of named "slots"
(strategy guidelines, output format, sizing rules) each holding one
"allele" (a specific phrasing). Mutation swaps one slot's allele for
another from a curated pool; crossover splices slots from two parents.

Evaluation runs each genome through an offline fitness function — at
this scale the natural choice is a backtest or a replay of recorded
cycle snapshots (from :mod:`halal_trader.core.replay`). Fitness can be
Sharpe ratio, profit factor, or a custom blend.

What this module is:
* Pure logic for the GA — populations, mutation, crossover, selection.
* No dependency on any specific LLM, prompt, or backtest engine.

What it intentionally is *not*:
* The fitness function — the caller supplies it. That's where you wire
  to ``crypto/backtest.py`` / ``crypto/walkforward.py`` / a replay
  harness over ``ReplayStore``.

Typical usage::

    pool = AllelePool(slots=load_slot_alleles())
    ga = PromptGA(pool=pool, fitness=my_fitness, population_size=12)
    best = await ga.evolve(generations=10)
    print("best fitness:", best.fitness, "diff vs base:", best.diff(base_genome))
"""

from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Genome / pool ─────────────────────────────────────────────────


@dataclass(frozen=True)
class PromptGenome:
    """A specific prompt assembled from one allele per slot.

    ``slots`` is a stable mapping ``slot_name -> allele_text`` shared
    by every genome in the population. The actual prompt is rendered by
    the caller via :meth:`render` (or any custom template that knows
    these slot names).
    """

    slots: dict[str, str] = field(default_factory=dict)

    def render(self, template: str) -> str:
        """``template`` uses ``{slot_name}`` placeholders to interpolate."""
        return template.format(**self.slots)

    def fingerprint(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.slots.items()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PromptGenome) and self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def diff(self, other: "PromptGenome") -> dict[str, tuple[str, str]]:
        """Return ``{slot: (mine, theirs)}`` for slots that differ."""
        out: dict[str, tuple[str, str]] = {}
        for k in set(self.slots) | set(other.slots):
            mine = self.slots.get(k, "")
            theirs = other.slots.get(k, "")
            if mine != theirs:
                out[k] = (mine, theirs)
        return out


@dataclass
class AllelePool:
    """For each slot, the candidate alleles a mutation can pick from.

    The first allele in each list is the canonical default; randomising
    the *base* genome reproduces today's prompt as a population of one.
    """

    slots: dict[str, list[str]] = field(default_factory=dict)

    def base_genome(self) -> PromptGenome:
        return PromptGenome(slots={k: v[0] for k, v in self.slots.items() if v})

    def random_genome(self, rng: random.Random) -> PromptGenome:
        return PromptGenome(slots={k: rng.choice(v) for k, v in self.slots.items() if v})

    def mutate(self, genome: PromptGenome, rng: random.Random) -> PromptGenome:
        """Mutate one slot to a different allele (if possible)."""
        if not genome.slots:
            return genome
        slot = rng.choice(list(genome.slots))
        choices = [a for a in self.slots.get(slot, []) if a != genome.slots[slot]]
        if not choices:
            return genome
        new_slots = dict(genome.slots)
        new_slots[slot] = rng.choice(choices)
        return PromptGenome(slots=new_slots)


# ── Crossover ────────────────────────────────────────────────────


def crossover(a: PromptGenome, b: PromptGenome, rng: random.Random) -> PromptGenome:
    """Per-slot uniform crossover."""
    keys = sorted(set(a.slots) | set(b.slots))
    new_slots: dict[str, str] = {}
    for k in keys:
        src = a if rng.random() < 0.5 else b
        if k in src.slots:
            new_slots[k] = src.slots[k]
        elif k in a.slots:
            new_slots[k] = a.slots[k]
        elif k in b.slots:
            new_slots[k] = b.slots[k]
    return PromptGenome(slots=new_slots)


# ── Population entry ─────────────────────────────────────────────


@dataclass
class ScoredGenome:
    genome: PromptGenome
    fitness: float
    notes: str = ""

    def __lt__(self, other: "ScoredGenome") -> bool:
        return self.fitness < other.fitness


# ── GA driver ────────────────────────────────────────────────────


@dataclass
class PromptGA:
    """Tiny GA tuned for prompt-engineering at our scale.

    * Generational with elitism (top ``elite_count`` survive intact).
    * Tournament selection of size ``tournament_k``.
    * Per-genome mutation rate ``mutation_rate`` (after crossover).
    * Caches fitness scores so identical genomes don't re-run the
      (expensive) backtest.
    """

    pool: AllelePool
    fitness: Callable[[PromptGenome], Awaitable[float]]
    population_size: int = 12
    elite_count: int = 2
    tournament_k: int = 3
    mutation_rate: float = 0.4
    seed: int | None = None

    _rng: random.Random = field(init=False)
    _cache: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)
    history: list[list[ScoredGenome]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed if self.seed is not None else random.random())

    async def _score(self, genome: PromptGenome) -> float:
        fp = genome.fingerprint()
        if fp in self._cache:
            return self._cache[fp]
        try:
            score = float(await self.fitness(genome))
        except Exception as exc:  # noqa: BLE001
            logger.warning("fitness call failed for genome — scoring 0: %s", exc)
            score = 0.0
        self._cache[fp] = score
        return score

    def _initial_population(
        self, seed_genomes: Sequence[PromptGenome] | None
    ) -> list[PromptGenome]:
        seeds = list(seed_genomes or [])
        # Always include the base genome so we never *lose* the current default.
        seeds.append(self.pool.base_genome())
        # Fill with random until we hit population_size, then drop dupes.
        while len(seeds) < self.population_size:
            seeds.append(self.pool.random_genome(self._rng))
        # de-dupe while preserving order
        seen: set[tuple[tuple[str, str], ...]] = set()
        out: list[PromptGenome] = []
        for g in seeds:
            fp = g.fingerprint()
            if fp not in seen:
                seen.add(fp)
                out.append(g)
        # backfill if dedup left us short
        while len(out) < self.population_size:
            out.append(self.pool.random_genome(self._rng))
        return out[: self.population_size]

    def _tournament(self, scored: Sequence[ScoredGenome]) -> ScoredGenome:
        sample = self._rng.sample(list(scored), k=min(self.tournament_k, len(scored)))
        return max(sample, key=lambda s: s.fitness)

    async def evolve(
        self,
        generations: int = 10,
        seed_genomes: Sequence[PromptGenome] | None = None,
    ) -> ScoredGenome:
        """Run the GA and return the best-scored genome ever seen."""
        population = self._initial_population(seed_genomes)
        scored = [ScoredGenome(genome=g, fitness=await self._score(g)) for g in population]
        scored.sort(reverse=True)
        self.history.append(list(scored))
        best = scored[0]

        for gen in range(1, generations + 1):
            elite = scored[: self.elite_count]
            next_pop: list[PromptGenome] = [s.genome for s in elite]
            while len(next_pop) < self.population_size:
                p1 = self._tournament(scored).genome
                p2 = self._tournament(scored).genome
                child = crossover(p1, p2, self._rng)
                if self._rng.random() < self.mutation_rate:
                    child = self.pool.mutate(child, self._rng)
                next_pop.append(child)
            scored = [ScoredGenome(genome=g, fitness=await self._score(g)) for g in next_pop]
            scored.sort(reverse=True)
            self.history.append(list(scored))
            if scored[0].fitness > best.fitness:
                best = scored[0]
            logger.info(
                "GA gen %d/%d — best=%.4f mean=%.4f",
                gen,
                generations,
                scored[0].fitness,
                sum(s.fitness for s in scored) / len(scored),
            )
        return best

    def best(self) -> ScoredGenome | None:
        if not self.history:
            return None
        return max((s for gen in self.history for s in gen), key=lambda s: s.fitness)

    def cache_size(self) -> int:
        return len(self._cache)
