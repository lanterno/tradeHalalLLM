"""Wave F wiring tests — AllelePool, genome-aware build_prompts, fitness.

The GA primitives themselves (mutation, crossover, selection) are
covered by ``tests/test_prompt_evo*``. These tests cover the
consumer-side wiring: that the crypto AllelePool is the right shape,
that genome substitution preserves the JSON schema, that fitness
functions return sensible signals, and that the CLI imports cleanly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from halal_trader.core.llm.prompt_evo import PromptGenome
from halal_trader.core.replay import CycleSnapshot
from halal_trader.crypto.prompts import (
    SYSTEM_PROMPT,
    PromptContext,
    StrategyParams,
    _slot_alleles,
    build_prompts,
    crypto_allele_pool,
)


def _ctx() -> PromptContext:
    """Minimal PromptContext for prompt-render tests."""
    account = MagicMock()
    account.total_balance_usdt = 1000.0
    account.available_balance_usdt = 800.0
    account.in_order_usdt = 200.0
    account.usdt_free = 800.0
    return PromptContext(account=account, halal_pairs=["BTCUSDT"])


def _params() -> StrategyParams:
    return StrategyParams(
        max_position_pct=0.25,
        daily_loss_limit=0.03,
        daily_return_target=0.01,
        max_positions=5,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )


# ── AllelePool shape ─────────────────────────────────────────────


def test_pool_has_three_slots() -> None:
    """The Wave F design ships with exactly three evolvable slots."""
    pool = crypto_allele_pool()
    assert set(pool.slots.keys()) == {"role_intro", "strategy_emphasis", "decision_humility"}


def test_each_slot_has_minimum_three_alleles() -> None:
    """At <3 alleles per slot, mutation has nowhere to go."""
    pool = crypto_allele_pool()
    for slot, alleles in pool.slots.items():
        assert len(alleles) >= 3, f"slot {slot!r} has only {len(alleles)} alleles"


def test_base_genome_matches_canonical_first_allele() -> None:
    """The first allele is the canonical default — the base genome
    must reproduce today's prompt byte-for-byte."""
    pool = crypto_allele_pool()
    base = pool.base_genome()
    alleles = _slot_alleles()
    assert base.slots == {k: v[0] for k, v in alleles.items()}


def test_alleles_are_immutable_across_calls() -> None:
    """``_slot_alleles()`` returns a fresh dict so mutating it
    doesn't poison the next caller."""
    a = _slot_alleles()
    a["role_intro"].append("garbage allele")
    b = _slot_alleles()
    assert "garbage allele" not in b["role_intro"]


# ── Genome-aware build_prompts ───────────────────────────────────


def test_build_prompts_no_genome_renders_base_text() -> None:
    """Omitting the genome arg → today's prompt verbatim."""
    sys_prompt, _user = build_prompts(_ctx(), _params())
    assert "You are an expert crypto scalping AI." in sys_prompt
    assert "Prioritise liquidity" not in sys_prompt
    assert "When in doubt, hold" not in sys_prompt


def test_build_prompts_genome_substitutes_role_intro() -> None:
    """A genome with a non-canonical role_intro flows into the prompt."""
    genome = PromptGenome(
        slots={
            "role_intro": "You are a conservative crypto scalper focused on capital preservation."
        }
    )
    sys_prompt, _user = build_prompts(_ctx(), _params(), genome=genome)
    assert "conservative crypto scalper" in sys_prompt
    assert "expert crypto scalping AI" not in sys_prompt


def test_build_prompts_genome_substitutes_multiple_slots() -> None:
    """Multi-slot genome → all overrides land."""
    alleles = _slot_alleles()
    genome = PromptGenome(
        slots={
            "role_intro": alleles["role_intro"][1],
            "strategy_emphasis": alleles["strategy_emphasis"][1],
            "decision_humility": alleles["decision_humility"][1],
        }
    )
    sys_prompt, _user = build_prompts(_ctx(), _params(), genome=genome)
    assert "disciplined crypto-momentum trader" in sys_prompt
    assert "Prioritise liquidity" in sys_prompt
    assert "When in doubt, hold" in sys_prompt


def test_build_prompts_unknown_slot_in_genome_is_ignored() -> None:
    """A genome carrying a slot the prompt doesn't know about
    is silently ignored — protects against renamed slots crashing
    a live cycle that loaded an old genome from DB."""
    genome = PromptGenome(slots={"role_intro": "X.", "nonexistent_slot": "Y"})
    sys_prompt, _user = build_prompts(_ctx(), _params(), genome=genome)
    assert "X." in sys_prompt
    # No exception raised — base values used for the real slots.


def test_genome_render_still_contains_critical_json_schema() -> None:
    """Mutations must not break the JSON output contract — pin a
    sample of phrasings that the executor depends on parsing."""
    pool = crypto_allele_pool()
    for genome in [pool.base_genome()] + [
        PromptGenome(slots={k: alleles[-1] for k, alleles in _slot_alleles().items()})
    ]:
        sys_prompt, _user = build_prompts(_ctx(), _params(), genome=genome)
        assert "OUTPUT JSON SCHEMA" in sys_prompt
        assert '"action": "buy"|"sell"|"hold"' in sys_prompt
        assert "halal-compliant list" in sys_prompt


def test_system_prompt_template_uses_all_three_slots() -> None:
    """Belt-and-suspenders: every slot the AllelePool knows about
    actually appears in the rendered template."""
    for slot in _slot_alleles():
        assert "{" + slot + "}" in SYSTEM_PROMPT, f"slot {slot!r} missing from SYSTEM_PROMPT"


# ── Fitness functions ────────────────────────────────────────────


def _snap(*, today_pnl: float, equity: float, ml: str = "") -> CycleSnapshot:
    return CycleSnapshot(
        cycle_id="c1",
        cycle_started_at="2026-05-18T12:00:00+00:00",
        today_pnl=today_pnl,
        account={"total_balance_usdt": equity},
        ml_signals_text=ml,
    )


def test_replay_pnl_fitness_positive_pnl_beats_negative_pnl() -> None:
    """Sanity: a profitable cycle gets a higher fitness score."""
    from halal_trader.crypto.prompt_fitness import replay_pnl_fitness

    genome = PromptGenome(slots={"role_intro": "x"})
    win = asyncio.run(replay_pnl_fitness(genome, _snap(today_pnl=20.0, equity=1000.0)))
    lose = asyncio.run(replay_pnl_fitness(genome, _snap(today_pnl=-10.0, equity=1000.0)))
    assert win > lose


def test_replay_pnl_fitness_length_penalty_breaks_ties() -> None:
    """Genomes with same P&L → the shorter genome wins (token cost)."""
    from halal_trader.crypto.prompt_fitness import replay_pnl_fitness

    short = PromptGenome(slots={"role_intro": "x" * 20})
    long_ = PromptGenome(slots={"role_intro": "x" * 5000})
    snap = _snap(today_pnl=0.0, equity=1000.0)
    s_short = asyncio.run(replay_pnl_fitness(short, snap))
    s_long = asyncio.run(replay_pnl_fitness(long_, snap))
    assert s_short > s_long


def test_replay_pnl_fitness_zero_equity_returns_only_penalty() -> None:
    """A cold-start snapshot (equity=0) shouldn't divide by zero."""
    from halal_trader.crypto.prompt_fitness import replay_pnl_fitness

    genome = PromptGenome(slots={"role_intro": "x"})
    score = asyncio.run(replay_pnl_fitness(genome, _snap(today_pnl=100.0, equity=0.0)))
    # PnL_pct = 0 (equity guard), score = -length_penalty (small negative).
    assert -1e-3 < score <= 0.0


def test_confidence_proxy_fitness_extracts_avg_confidence() -> None:
    """Pulls the mean confidence from ml_signals_text."""
    from halal_trader.crypto.prompt_fitness import confidence_proxy_fitness

    genome = PromptGenome(slots={"role_intro": "x"})
    snap = _snap(
        today_pnl=0.0,
        equity=1000.0,
        ml="BTCUSDT: anomaly_score=0.5, confidence=0.8\nETHUSDT: confidence=0.6",
    )
    score = asyncio.run(confidence_proxy_fitness(genome, snap))
    # mean(0.8, 0.6) = 0.7; penalty is sub-microscopic.
    assert abs(score - 0.7) < 0.01


def test_confidence_proxy_fitness_empty_text_returns_zero() -> None:
    """No signals → no signal."""
    from halal_trader.crypto.prompt_fitness import confidence_proxy_fitness

    genome = PromptGenome(slots={"role_intro": "x"})
    score = asyncio.run(
        confidence_proxy_fitness(genome, _snap(today_pnl=0.0, equity=1000.0, ml=""))
    )
    assert score == 0.0


# ── CLI shape ────────────────────────────────────────────────────


def test_cli_prompts_group_registered() -> None:
    """The CLI exposes ``halal-trader prompts evolve / candidates / promote``."""
    from halal_trader.cli import cli

    # cli is a click Group — list of named subcommands.
    cmds = list(cli.commands.keys())
    assert "prompts" in cmds
    prompts_grp = cli.commands["prompts"]
    sub = list(prompts_grp.commands.keys())
    assert {"evolve", "candidates", "promote"} <= set(sub)


def test_cli_evolve_help_runs() -> None:
    """`halal-trader prompts evolve --help` parses without exception."""
    from click.testing import CliRunner

    from halal_trader.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["prompts", "evolve", "--help"])
    assert result.exit_code == 0
    assert "Run one GA sweep" in result.output


# ── Settings knobs ───────────────────────────────────────────────


def test_settings_expose_prompt_evo_knobs() -> None:
    """The nightly job reads three CryptoSettings fields — pin defaults."""
    from halal_trader.config import CryptoSettings

    s = CryptoSettings()
    assert s.prompt_evo_generations >= 1
    assert s.prompt_evo_population >= 4
    assert s.prompt_evo_snapshots >= 20


def test_settings_validate_prompt_evo_bounds() -> None:
    """Out-of-range values rejected by pydantic."""
    from halal_trader.config import CryptoSettings

    with pytest.raises(Exception):
        CryptoSettings(prompt_evo_generations=999)
    with pytest.raises(Exception):
        CryptoSettings(prompt_evo_population=2)
