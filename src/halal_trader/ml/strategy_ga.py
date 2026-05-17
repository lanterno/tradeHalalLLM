"""Genetic algorithm over strategy genomes.

Round-4 wave 4.B: the existing `core/llm/prompt_evo*.py` evolves
*prompts*. This module extends the same idea one level up: evolve
the **strategy itself** — combinations of indicator entry filters
(RSI bands, MACD direction, BB position, volume ratio thresholds),
regime gates (only trade in a given regime), and risk caps (max
position pct, max simultaneous positions). The GA proposes new
genomes; the caller's fitness function backtests them; the GA
selects, crosses over, mutates, and iterates.

The module is the **mechanic**, not the application. It doesn't
know what a "BUY" looks like, doesn't call the broker, doesn't
read backtest results. The caller hands over a `fitness_fn` that
takes a `StrategyGenome` and returns a single number (typically
out-of-sample Sharpe or a composite score). The GA does the rest.

Why a generic GA rather than scipy / DEAP / pygmo:

* The genome is small and the population is tiny (≤50). The full
  framework's bookkeeping is more code than the algorithm itself.
* We want determinism for regression tests — every operation
  takes a `random.Random` so the GA is fully reproducible from a
  seed. Frameworks that thread a global RNG break this guarantee.
* Halal alignment: the genome bounds enforce only-long, no-leverage
  constraints (max_position_pct ≤ 1.0, no negative quantities). A
  framework's generic genome wouldn't.

Pure-Python; no NumPy / SciPy / DB / async. The fitness call is
the only blocking point — the caller is responsible for parallel
backtests if they want them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable

# ── Genome ────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyGenome:
    """One candidate strategy.

    Pure data — no behaviour. The caller's fitness function
    interprets the genome as it sees fit (a typical interpretation
    is at the top of `tests/test_strategy_ga.py`).

    All bounds are inclusive on the **bottom** end and exclusive
    on the top — so an `rsi_buy_max=30` rule fires when RSI < 30,
    matching the convention in `crypto/strategy.py`.

    ``regime_gate`` is `None` to mean "trade in any regime";
    otherwise it's the regime label the strategy demands.

    Halal invariants: ``max_position_pct ∈ (0, 1]`` (no leverage),
    ``rsi_buy_max ∈ [0, 100]``, etc. The mutation / crossover
    helpers preserve these — `_clamp` enforces them whenever a
    new genome is built.
    """

    rsi_buy_max: float = 30.0  # buy when RSI < this
    rsi_sell_min: float = 70.0  # sell when RSI > this
    macd_required_positive: bool = False  # require MACD histogram > 0
    bb_buy_below: float = 0.20  # buy only when BB position < this
    min_volume_ratio: float = 0.5  # require vol ≥ this × normal
    regime_gate: str | None = None  # only trade in this regime label
    max_position_pct: float = 0.10
    max_simultaneous_positions: int = 5
    min_confidence: float = 0.6  # discard signals below this

    def to_dict(self) -> dict[str, object]:
        return {
            "rsi_buy_max": self.rsi_buy_max,
            "rsi_sell_min": self.rsi_sell_min,
            "macd_required_positive": self.macd_required_positive,
            "bb_buy_below": self.bb_buy_below,
            "min_volume_ratio": self.min_volume_ratio,
            "regime_gate": self.regime_gate,
            "max_position_pct": self.max_position_pct,
            "max_simultaneous_positions": self.max_simultaneous_positions,
            "min_confidence": self.min_confidence,
        }


@dataclass(frozen=True)
class GenomeBounds:
    """Search-space bounds for the GA.

    Random and mutated genomes stay inside these. Operators tighten
    bounds to focus the search (e.g. fix `regime_gate` to a single
    value while only the indicator filters evolve).
    """

    rsi_buy_max_range: tuple[float, float] = (10.0, 50.0)
    rsi_sell_min_range: tuple[float, float] = (50.0, 90.0)
    bb_buy_below_range: tuple[float, float] = (0.0, 0.50)
    min_volume_ratio_range: tuple[float, float] = (0.1, 3.0)
    max_position_pct_range: tuple[float, float] = (0.01, 0.30)
    max_simultaneous_positions_range: tuple[int, int] = (1, 10)
    min_confidence_range: tuple[float, float] = (0.3, 0.9)
    available_regime_gates: tuple[str | None, ...] = (None, "uptrend", "ranging", "downtrend")


# ── Population scaffolding ────────────────────────────────


@dataclass(frozen=True)
class ScoredGenome:
    """A genome with its measured fitness."""

    genome: StrategyGenome
    fitness: float


@dataclass
class GenerationReport:
    """Summary of one generation's outcome."""

    generation: int
    best: ScoredGenome
    avg_fitness: float
    population: list[ScoredGenome] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _ensure_band(low: float, high: float) -> tuple[float, float]:
    """Pin: `rsi_buy_max < rsi_sell_min` after every mutation /
    crossover. Genomes where buy_max ≥ sell_min would never produce
    sells; swap if the GA produces them."""
    if low >= high:
        return high, low
    return low, high


# ── Genome construction ───────────────────────────────────


def random_genome(rng: random.Random, bounds: GenomeBounds | None = None) -> StrategyGenome:
    """Sample a uniform random genome within the bounds."""
    b = bounds or GenomeBounds()
    rsi_buy = rng.uniform(*b.rsi_buy_max_range)
    rsi_sell = rng.uniform(*b.rsi_sell_min_range)
    rsi_buy, rsi_sell = _ensure_band(rsi_buy, rsi_sell)
    return StrategyGenome(
        rsi_buy_max=rsi_buy,
        rsi_sell_min=rsi_sell,
        macd_required_positive=rng.random() > 0.5,
        bb_buy_below=rng.uniform(*b.bb_buy_below_range),
        min_volume_ratio=rng.uniform(*b.min_volume_ratio_range),
        regime_gate=rng.choice(list(b.available_regime_gates)),
        max_position_pct=rng.uniform(*b.max_position_pct_range),
        max_simultaneous_positions=rng.randint(*b.max_simultaneous_positions_range),
        min_confidence=rng.uniform(*b.min_confidence_range),
    )


def crossover(
    a: StrategyGenome,
    b: StrategyGenome,
    *,
    rng: random.Random,
) -> StrategyGenome:
    """Uniform crossover — each gene independently picks from
    parent A or parent B with 50/50.

    Returns a *new* frozen genome with the band invariant
    re-enforced (parents that individually satisfy buy < sell can
    cross into a genome that violates it; the GA must never let
    that escape).
    """

    def pick(field: str):
        return getattr(a if rng.random() < 0.5 else b, field)

    rsi_buy = pick("rsi_buy_max")
    rsi_sell = pick("rsi_sell_min")
    rsi_buy, rsi_sell = _ensure_band(rsi_buy, rsi_sell)
    return StrategyGenome(
        rsi_buy_max=rsi_buy,
        rsi_sell_min=rsi_sell,
        macd_required_positive=pick("macd_required_positive"),
        bb_buy_below=pick("bb_buy_below"),
        min_volume_ratio=pick("min_volume_ratio"),
        regime_gate=pick("regime_gate"),
        max_position_pct=pick("max_position_pct"),
        max_simultaneous_positions=pick("max_simultaneous_positions"),
        min_confidence=pick("min_confidence"),
    )


def mutate(
    genome: StrategyGenome,
    *,
    rng: random.Random,
    bounds: GenomeBounds | None = None,
    rate: float = 0.25,
    sigma: float = 0.15,
) -> StrategyGenome:
    """Per-gene Gaussian mutation with probability ``rate``.

    Continuous fields nudge by `gauss(0, range_width × sigma)` and
    clamp to bounds. Bool fields flip with probability ``rate``.
    Regime gate resamples from the available choices with
    probability ``rate``. Pin: every mutated genome remains within
    bounds — the GA's invariants must be transitive across all
    operations."""
    b = bounds or GenomeBounds()
    field_args = replace(genome).to_dict()

    def maybe(name: str, lo: float, hi: float) -> None:
        if rng.random() < rate:
            current = float(field_args[name])
            width = hi - lo
            field_args[name] = _clamp(current + rng.gauss(0, width * sigma), lo, hi)

    maybe("rsi_buy_max", *b.rsi_buy_max_range)
    maybe("rsi_sell_min", *b.rsi_sell_min_range)
    maybe("bb_buy_below", *b.bb_buy_below_range)
    maybe("min_volume_ratio", *b.min_volume_ratio_range)
    maybe("max_position_pct", *b.max_position_pct_range)
    maybe("min_confidence", *b.min_confidence_range)

    if rng.random() < rate:
        lo, hi = b.max_simultaneous_positions_range
        sigma_int = max(1, int((hi - lo) * sigma))
        delta = rng.choice([-sigma_int, 0, sigma_int])
        field_args["max_simultaneous_positions"] = _clamp_int(
            int(field_args["max_simultaneous_positions"]) + delta, lo, hi
        )

    if rng.random() < rate:
        field_args["macd_required_positive"] = not bool(field_args["macd_required_positive"])

    if rng.random() < rate:
        field_args["regime_gate"] = rng.choice(list(b.available_regime_gates))

    rsi_buy, rsi_sell = _ensure_band(
        float(field_args["rsi_buy_max"]), float(field_args["rsi_sell_min"])
    )
    field_args["rsi_buy_max"] = rsi_buy
    field_args["rsi_sell_min"] = rsi_sell
    return StrategyGenome(**field_args)  # type: ignore[arg-type]


# ── Selection ─────────────────────────────────────────────


def tournament_select(
    population: list[ScoredGenome],
    *,
    rng: random.Random,
    tournament_size: int = 3,
) -> ScoredGenome:
    """Tournament selection: sample `tournament_size` candidates,
    return the fittest. Pin the simplest robust choice — fitness-
    proportionate roulette can degenerate when fitness goes
    negative (a perfectly valid Sharpe outcome)."""
    if not population:
        raise ValueError("population must be non-empty")
    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive")
    sample = rng.sample(population, min(tournament_size, len(population)))
    return max(sample, key=lambda s: s.fitness)


def elitism(population: list[ScoredGenome], *, k: int = 2) -> list[ScoredGenome]:
    """Top-K survival. Keeps the best K genomes in every new
    generation so the GA never loses a found-good solution."""
    if k <= 0:
        return []
    return sorted(population, key=lambda s: s.fitness, reverse=True)[:k]


# ── GA driver ─────────────────────────────────────────────


def evolve(
    *,
    fitness_fn: Callable[[StrategyGenome], float],
    population_size: int = 30,
    generations: int = 10,
    bounds: GenomeBounds | None = None,
    elite_count: int = 2,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.25,
    tournament_size: int = 3,
    seed: int | None = None,
    seed_population: list[StrategyGenome] | None = None,
) -> list[GenerationReport]:
    """Run the GA for ``generations`` generations.

    ``fitness_fn`` is the only domain-specific input. It must be
    deterministic for the GA's seed-determinism guarantee — a
    backtest with the same data + same RNG seed must return the
    same number every call.

    ``seed_population`` lets the operator inject hand-tuned
    genomes (e.g. the live-prompt's current settings) so the GA
    can build off a known-good start rather than rolling fresh
    randoms. Anything beyond population_size is ignored; anything
    below it is padded with random genomes.

    Returns one `GenerationReport` per generation in order. Empty
    generations list (population_size=0 etc.) raises rather than
    silently producing no output.
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if generations <= 0:
        raise ValueError("generations must be positive")
    if elite_count > population_size:
        raise ValueError("elite_count must not exceed population_size")

    rng = random.Random(seed)
    bounds = bounds or GenomeBounds()

    # Build initial population.
    seeds = list(seed_population or [])
    initial = seeds[:population_size]
    while len(initial) < population_size:
        initial.append(random_genome(rng, bounds))

    population = [ScoredGenome(genome=g, fitness=float(fitness_fn(g))) for g in initial]
    reports: list[GenerationReport] = []

    for gen in range(generations):
        best = max(population, key=lambda s: s.fitness)
        avg = sum(s.fitness for s in population) / len(population)
        reports.append(
            GenerationReport(
                generation=gen,
                best=best,
                avg_fitness=avg,
                population=list(population),
            )
        )

        # Build next generation: elites + crossover/mutation children.
        next_pop: list[StrategyGenome] = [s.genome for s in elitism(population, k=elite_count)]
        while len(next_pop) < population_size:
            parent_a = tournament_select(population, rng=rng, tournament_size=tournament_size)
            parent_b = tournament_select(population, rng=rng, tournament_size=tournament_size)
            if rng.random() < crossover_rate:
                child = crossover(parent_a.genome, parent_b.genome, rng=rng)
            else:
                child = parent_a.genome
            child = mutate(child, rng=rng, bounds=bounds, rate=mutation_rate)
            next_pop.append(child)

        population = [ScoredGenome(genome=g, fitness=float(fitness_fn(g))) for g in next_pop]

    # Final report after the last generation's population is scored.
    best = max(population, key=lambda s: s.fitness)
    avg = sum(s.fitness for s in population) / len(population)
    reports.append(
        GenerationReport(
            generation=generations,
            best=best,
            avg_fitness=avg,
            population=list(population),
        )
    )
    return reports


__all__ = [
    "GenerationReport",
    "GenomeBounds",
    "ScoredGenome",
    "StrategyGenome",
    "crossover",
    "elitism",
    "evolve",
    "mutate",
    "random_genome",
    "tournament_select",
]
