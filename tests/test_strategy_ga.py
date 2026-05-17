"""Tests for `ml/strategy_ga.py` (genetic strategy generator).

Pins the bounds-respect invariant across random / mutate / crossover,
the RSI-band-must-stay-ordered safety, the seed-determinism contract,
the tournament selection logic, the elitism count, and the
end-to-end `evolve` driver against a synthetic fitness landscape
with a known optimum so the test asserts the GA *finds* it.
"""

from __future__ import annotations

import random

import pytest

from halal_trader.ml.strategy_ga import (
    GenerationReport,
    GenomeBounds,
    ScoredGenome,
    StrategyGenome,
    crossover,
    elitism,
    evolve,
    mutate,
    random_genome,
    tournament_select,
)

# ── random_genome ────────────────────────────────────────


def test_random_genome_respects_default_bounds():
    rng = random.Random(0)
    g = random_genome(rng)
    b = GenomeBounds()
    assert b.rsi_buy_max_range[0] <= g.rsi_buy_max <= b.rsi_buy_max_range[1]
    assert b.rsi_sell_min_range[0] <= g.rsi_sell_min <= b.rsi_sell_min_range[1]
    assert b.bb_buy_below_range[0] <= g.bb_buy_below <= b.bb_buy_below_range[1]
    assert b.min_volume_ratio_range[0] <= g.min_volume_ratio <= b.min_volume_ratio_range[1]
    assert b.max_position_pct_range[0] <= g.max_position_pct <= b.max_position_pct_range[1]
    assert (
        b.max_simultaneous_positions_range[0]
        <= g.max_simultaneous_positions
        <= b.max_simultaneous_positions_range[1]
    )
    assert g.regime_gate in b.available_regime_gates


def test_random_genome_enforces_rsi_band_ordering():
    """Pin: even if random sampling produces buy_max > sell_min,
    the helper swaps to maintain `buy < sell`. Every genome must
    be tradable as-is."""
    rng = random.Random(0)
    for _ in range(50):
        g = random_genome(rng)
        assert g.rsi_buy_max < g.rsi_sell_min


def test_random_genome_is_seed_deterministic():
    """Pin: same seed → same genome. Required for regression tests
    on the GA driver."""
    g1 = random_genome(random.Random(42))
    g2 = random_genome(random.Random(42))
    assert g1 == g2


def test_random_genome_respects_custom_bounds():
    """Operator can lock down a sub-search by tightening bounds."""
    rng = random.Random(0)
    bounds = GenomeBounds(
        rsi_buy_max_range=(20.0, 25.0),
        rsi_sell_min_range=(75.0, 80.0),
        available_regime_gates=("uptrend",),
    )
    for _ in range(20):
        g = random_genome(rng, bounds)
        assert 20.0 <= g.rsi_buy_max <= 25.0
        assert 75.0 <= g.rsi_sell_min <= 80.0
        assert g.regime_gate == "uptrend"


# ── crossover ────────────────────────────────────────────


def test_crossover_inherits_only_from_parents():
    """Pin: crossover must NOT introduce values not present in
    either parent. (Continuous fields: each child gene equals
    one of the two parents'.)"""
    a = StrategyGenome(rsi_buy_max=20.0, rsi_sell_min=75.0)
    b = StrategyGenome(rsi_buy_max=25.0, rsi_sell_min=80.0)
    rng = random.Random(0)
    for _ in range(30):
        child = crossover(a, b, rng=rng)
        # buy_max must equal one of the two parents'.
        assert child.rsi_buy_max in (20.0, 25.0)
        # Same for sell_min.
        assert child.rsi_sell_min in (75.0, 80.0)


def test_crossover_enforces_band_when_parents_cross_into_violation():
    """A buy_max from parent A (29) and sell_min from parent B
    (28) violates the band. Pin: crossover swaps."""
    a = StrategyGenome(rsi_buy_max=29.0, rsi_sell_min=70.0)
    b = StrategyGenome(rsi_buy_max=10.0, rsi_sell_min=28.0)
    # Try several seeds — at least one should produce the
    # cross-violation pattern, and even then the result must obey
    # the band.
    for seed in range(30):
        rng = random.Random(seed)
        child = crossover(a, b, rng=rng)
        assert child.rsi_buy_max < child.rsi_sell_min


def test_crossover_is_seed_deterministic():
    a = StrategyGenome(rsi_buy_max=20.0)
    b = StrategyGenome(rsi_buy_max=25.0)
    c1 = crossover(a, b, rng=random.Random(99))
    c2 = crossover(a, b, rng=random.Random(99))
    assert c1 == c2


# ── mutate ───────────────────────────────────────────────


def test_mutate_keeps_genome_within_bounds():
    """Pin: every mutated genome must stay inside the search bounds.
    Run many iterations to catch tail cases."""
    bounds = GenomeBounds()
    rng = random.Random(0)
    g = random_genome(rng, bounds)
    for _ in range(100):
        g = mutate(g, rng=rng, bounds=bounds, rate=1.0)  # mutate every gene
        assert bounds.rsi_buy_max_range[0] <= g.rsi_buy_max <= bounds.rsi_buy_max_range[1]
        assert bounds.rsi_sell_min_range[0] <= g.rsi_sell_min <= bounds.rsi_sell_min_range[1]
        assert g.rsi_buy_max < g.rsi_sell_min  # band invariant survives
        assert (
            bounds.max_position_pct_range[0]
            <= g.max_position_pct
            <= bounds.max_position_pct_range[1]
        )
        assert (
            bounds.max_simultaneous_positions_range[0]
            <= g.max_simultaneous_positions
            <= bounds.max_simultaneous_positions_range[1]
        )


def test_mutate_with_zero_rate_returns_unchanged_genome():
    """Pin: rate=0 must produce an identical genome."""
    rng = random.Random(0)
    g = random_genome(rng)
    mutated = mutate(g, rng=rng, rate=0.0)
    assert mutated == g


def test_mutate_is_seed_deterministic():
    rng_factory = lambda: random.Random(7)  # noqa: E731
    g = random_genome(random.Random(0))
    m1 = mutate(g, rng=rng_factory(), rate=0.5)
    m2 = mutate(g, rng=rng_factory(), rate=0.5)
    assert m1 == m2


def test_mutate_respects_custom_regime_gate_choices():
    """Pin: the regime_gate mutation only picks from the bounded
    list. A locked-down "uptrend only" search must never produce
    a different gate."""
    bounds = GenomeBounds(available_regime_gates=("uptrend",))
    rng = random.Random(0)
    g = StrategyGenome(regime_gate="uptrend")
    for _ in range(20):
        g = mutate(g, rng=rng, bounds=bounds, rate=1.0)
        assert g.regime_gate == "uptrend"


# ── tournament selection ─────────────────────────────────


def test_tournament_picks_fittest_in_sample():
    pop = [
        ScoredGenome(genome=StrategyGenome(), fitness=1.0),
        ScoredGenome(genome=StrategyGenome(), fitness=5.0),
        ScoredGenome(genome=StrategyGenome(), fitness=2.0),
    ]
    # With tournament_size = len(pop) the helper always picks the
    # max — pin the deterministic case.
    rng = random.Random(0)
    winner = tournament_select(pop, rng=rng, tournament_size=3)
    assert winner.fitness == 5.0


def test_tournament_handles_pop_smaller_than_tournament_size():
    """Pin: smaller-than-requested sample takes the whole pop
    rather than crashing."""
    pop = [ScoredGenome(genome=StrategyGenome(), fitness=1.0)]
    winner = tournament_select(pop, rng=random.Random(0), tournament_size=10)
    assert winner.fitness == 1.0


def test_tournament_rejects_empty_population():
    with pytest.raises(ValueError, match="non-empty"):
        tournament_select([], rng=random.Random(0))


def test_tournament_rejects_zero_size():
    pop = [ScoredGenome(genome=StrategyGenome(), fitness=1.0)]
    with pytest.raises(ValueError, match="positive"):
        tournament_select(pop, rng=random.Random(0), tournament_size=0)


# ── elitism ──────────────────────────────────────────────


def test_elitism_returns_top_k_descending():
    pop = [
        ScoredGenome(genome=StrategyGenome(), fitness=1.0),
        ScoredGenome(genome=StrategyGenome(), fitness=5.0),
        ScoredGenome(genome=StrategyGenome(), fitness=3.0),
        ScoredGenome(genome=StrategyGenome(), fitness=4.0),
    ]
    elites = elitism(pop, k=2)
    assert [s.fitness for s in elites] == [5.0, 4.0]


def test_elitism_with_zero_k_returns_empty():
    pop = [ScoredGenome(genome=StrategyGenome(), fitness=1.0)]
    assert elitism(pop, k=0) == []


# ── evolve driver ────────────────────────────────────────


def _peaked_fitness(target: StrategyGenome) -> callable:
    """Build a fitness function that scores higher the closer a
    genome is to ``target`` on the continuous fields. Used to
    verify the GA actually *converges* on a known optimum."""

    def fn(g: StrategyGenome) -> float:
        dist = (
            abs(g.rsi_buy_max - target.rsi_buy_max) / 50.0
            + abs(g.rsi_sell_min - target.rsi_sell_min) / 50.0
            + abs(g.max_position_pct - target.max_position_pct) / 0.30
            + abs(g.min_volume_ratio - target.min_volume_ratio) / 3.0
        )
        return -dist  # smaller distance → higher score

    return fn


def test_evolve_returns_one_report_per_generation_plus_final():
    """Pin: N generations → N + 1 reports (initial + N updates).
    Off-by-one in this loop has bitten every GA implementation."""
    fn = _peaked_fitness(StrategyGenome())
    reports = evolve(fitness_fn=fn, population_size=10, generations=3, seed=0)
    assert len(reports) == 4


def test_evolve_best_fitness_is_monotonic_non_decreasing():
    """With elitism, the best fitness in any generation must be
    ≥ the best in the previous generation."""
    target = StrategyGenome(
        rsi_buy_max=25.0,
        rsi_sell_min=75.0,
        max_position_pct=0.10,
        min_volume_ratio=1.5,
    )
    fn = _peaked_fitness(target)
    reports = evolve(fitness_fn=fn, population_size=20, generations=10, elite_count=2, seed=42)
    best_fits = [r.best.fitness for r in reports]
    for prev, curr in zip(best_fits, best_fits[1:]):
        assert curr >= prev - 1e-9


def test_evolve_converges_toward_target_in_synthetic_landscape():
    """End-to-end: feed the GA a synthetic peaked fitness landscape
    and verify the final-gen best is materially better than the
    initial-gen best. Pin so a refactor that breaks selection or
    crossover surfaces as a regression."""
    target = StrategyGenome(
        rsi_buy_max=25.0,
        rsi_sell_min=75.0,
        max_position_pct=0.10,
        min_volume_ratio=1.5,
    )
    fn = _peaked_fitness(target)
    reports = evolve(
        fitness_fn=fn,
        population_size=30,
        generations=15,
        seed=123,
    )
    initial_best = reports[0].best.fitness
    final_best = reports[-1].best.fitness
    assert final_best > initial_best
    # Should converge close to the optimum (-0.0).
    assert final_best > -0.5


def test_evolve_is_seed_deterministic():
    """Pin the regression-test guarantee: same seed + same fitness
    fn → same final genome."""
    fn = _peaked_fitness(StrategyGenome())
    a = evolve(fitness_fn=fn, population_size=10, generations=5, seed=99)
    b = evolve(fitness_fn=fn, population_size=10, generations=5, seed=99)
    assert a[-1].best.genome == b[-1].best.genome


def test_evolve_seeds_population_with_supplied_genomes():
    """Pin: operator-supplied seed genomes appear in generation 0."""
    custom = StrategyGenome(rsi_buy_max=22.0, max_position_pct=0.05)
    fn = _peaked_fitness(custom)  # custom is the optimum
    reports = evolve(
        fitness_fn=fn,
        population_size=10,
        generations=1,
        seed=0,
        seed_population=[custom],
    )
    initial_genomes = [s.genome for s in reports[0].population]
    assert custom in initial_genomes


def test_evolve_pads_short_seed_population_with_randoms():
    """Operator passes one seed; GA fills the rest with random
    genomes — the resulting population has the right size."""
    fn = _peaked_fitness(StrategyGenome())
    reports = evolve(
        fitness_fn=fn,
        population_size=10,
        generations=1,
        seed=0,
        seed_population=[StrategyGenome()],
    )
    assert len(reports[0].population) == 10


def test_evolve_rejects_invalid_settings():
    fn = _peaked_fitness(StrategyGenome())
    with pytest.raises(ValueError, match="population_size"):
        evolve(fitness_fn=fn, population_size=0, generations=1)
    with pytest.raises(ValueError, match="generations"):
        evolve(fitness_fn=fn, population_size=10, generations=0)
    with pytest.raises(ValueError, match="elite_count"):
        evolve(fitness_fn=fn, population_size=5, generations=1, elite_count=10)


# ── output structure ─────────────────────────────────────


def test_generation_report_carries_best_and_avg():
    fn = _peaked_fitness(StrategyGenome())
    reports = evolve(fitness_fn=fn, population_size=10, generations=2, seed=0)
    for r in reports:
        assert isinstance(r, GenerationReport)
        assert isinstance(r.best, ScoredGenome)
        assert r.avg_fitness <= r.best.fitness + 1e-9


def test_genome_to_dict_round_trips_every_field():
    """The dict adapter is the JSON / audit-trail shape — must
    carry every gene so a serialised genome can be fully
    reconstructed."""
    g = StrategyGenome(
        rsi_buy_max=22.0,
        rsi_sell_min=78.0,
        macd_required_positive=True,
        bb_buy_below=0.15,
        min_volume_ratio=1.2,
        regime_gate="uptrend",
        max_position_pct=0.08,
        max_simultaneous_positions=4,
        min_confidence=0.65,
    )
    d = g.to_dict()
    assert set(d.keys()) == {
        "rsi_buy_max",
        "rsi_sell_min",
        "macd_required_positive",
        "bb_buy_below",
        "min_volume_ratio",
        "regime_gate",
        "max_position_pct",
        "max_simultaneous_positions",
        "min_confidence",
    }
    assert StrategyGenome(**d) == g  # type: ignore[arg-type]


def test_genome_is_immutable():
    g = StrategyGenome()
    with pytest.raises(Exception):
        g.rsi_buy_max = 99.0  # type: ignore[misc]
