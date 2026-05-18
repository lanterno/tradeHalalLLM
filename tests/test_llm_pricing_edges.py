"""Edge-case tests for :mod:`core.llm.pricing`.

`test_llm_pricing.py` covers the basic shapes (known model lookup,
unknown → DEFAULT_PRICING, ollama-style → free, compute sums input +
output, cache categories included). This file pins the remaining
contract: each supported model family's snapshot prices (so a
silent edit to the table is caught), the prefix-matching for
local models (qwen/llama/mistral get free pricing without an entry),
the once-only unknown-model warning, and the all-four-categories
rollup math.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from halal_trader.core.llm.pricing import (
    DEFAULT_PRICING,
    compute_cost_usd,
    get_pricing,
)

# ── Anthropic family snapshots ──────────────────────────────


def test_claude_opus_4_7_pricing_snapshot():
    p = get_pricing("claude-opus-4-7")
    assert p.input_per_mtok == Decimal("15.00")
    assert p.output_per_mtok == Decimal("75.00")
    assert p.cache_read_per_mtok == Decimal("1.50")
    assert p.cache_write_per_mtok == Decimal("18.75")


def test_claude_opus_4_1_matches_4_7_pricing():
    """Both opus-4 variants billed identically — pin so they don't
    diverge silently when the table is edited."""
    assert get_pricing("claude-opus-4-1") == get_pricing("claude-opus-4-7")
    assert get_pricing("claude-opus-4-20250514") == get_pricing("claude-opus-4-7")


def test_claude_sonnet_4_6_pricing_snapshot():
    p = get_pricing("claude-sonnet-4-6")
    assert p.input_per_mtok == Decimal("3.00")
    assert p.output_per_mtok == Decimal("15.00")
    assert p.cache_read_per_mtok == Decimal("0.30")
    assert p.cache_write_per_mtok == Decimal("3.75")


def test_claude_sonnet_variants_share_pricing():
    """sonnet-4-5, sonnet-4-6, and the dated sonnet-4-20250514 all
    share the table row — keeps the parity invariant explicit."""
    assert get_pricing("claude-sonnet-4-5") == get_pricing("claude-sonnet-4-6")
    assert get_pricing("claude-sonnet-4-20250514") == get_pricing("claude-sonnet-4-6")


def test_claude_haiku_4_5_pricing_snapshot():
    p = get_pricing("claude-haiku-4-5-20251001")
    assert p.input_per_mtok == Decimal("1.00")
    assert p.output_per_mtok == Decimal("5.00")
    assert p.cache_read_per_mtok == Decimal("0.10")
    assert p.cache_write_per_mtok == Decimal("1.25")


def test_anthropic_family_priced_in_descending_order():
    """Sanity check: opus > sonnet > haiku on input price — if the
    table is edited and this inverts, something is wrong."""
    opus = get_pricing("claude-opus-4-7").input_per_mtok
    sonnet = get_pricing("claude-sonnet-4-6").input_per_mtok
    haiku = get_pricing("claude-haiku-4-5-20251001").input_per_mtok
    assert opus > sonnet > haiku


# ── OpenAI family snapshots ─────────────────────────────────


def test_gpt_4o_pricing_snapshot():
    p = get_pricing("gpt-4o")
    assert p.input_per_mtok == Decimal("2.50")
    assert p.output_per_mtok == Decimal("10.00")
    assert p.cache_read_per_mtok == Decimal("1.25")


def test_gpt_4o_mini_is_cheaper_than_4o():
    """Mini variant sanity check — cheaper across all categories."""
    full = get_pricing("gpt-4o")
    mini = get_pricing("gpt-4o-mini")
    assert mini.input_per_mtok < full.input_per_mtok
    assert mini.output_per_mtok < full.output_per_mtok


def test_gpt_4_turbo_pricing_snapshot():
    p = get_pricing("gpt-4-turbo")
    assert p.input_per_mtok == Decimal("10.00")
    assert p.output_per_mtok == Decimal("30.00")


# ── Local / ollama prefix matching ──────────────────────────


def test_ollama_explicit_entry_is_free():
    p = get_pricing("ollama")
    assert p.input_per_mtok == Decimal("0")
    assert p.output_per_mtok == Decimal("0")


def test_qwen_prefix_treated_as_free_local():
    """`qwen` family runs locally on Ollama — costs $0 in electricity.
    Hits the prefix-match branch in `get_pricing`, not the explicit
    table lookup."""
    p = get_pricing("qwen2.5:32b")
    assert p.input_per_mtok == Decimal("0")


def test_llama_prefix_treated_as_free_local():
    p = get_pricing("llama3.1:70b")
    assert p.input_per_mtok == Decimal("0")


def test_mistral_prefix_treated_as_free_local():
    p = get_pricing("mistral-nemo:12b")
    assert p.input_per_mtok == Decimal("0")


def test_colon_in_model_name_triggers_local_branch():
    """Any model name with a colon (Ollama's tag separator) is treated
    as local — so a future model like `gemma3:9b` doesn't hit the
    DEFAULT (over-bill) path, even though it's not in the prefix list."""
    p = get_pricing("gemma3:9b")
    assert p.input_per_mtok == Decimal("0")


def test_ollama_prefix_treated_as_free():
    """`ollama-anything` (custom local builds tagged `ollama-foo`) hits
    the prefix branch."""
    p = get_pricing("ollama-custom-build")
    assert p.input_per_mtok == Decimal("0")


# ── Unknown-model warning + fallback ────────────────────────


def test_unknown_cloud_model_falls_back_to_default():
    p = get_pricing("frontier-model-2026")
    assert p == DEFAULT_PRICING


def test_unknown_model_warned_only_once(caplog):
    """The `_warned_unknown` set deduplicates warnings — important
    because an unknown model would log on EVERY call otherwise (spamming
    the operator's terminal). Pin the dedup so a refactor that drops
    the set is caught."""
    from halal_trader.core.llm import pricing as pricing_mod

    # Use a unique name + clear from the warned-set so the test is repeatable.
    name = "test-unique-unknown-xyz-2026"
    pricing_mod._warned_unknown.discard(name)

    with caplog.at_level(logging.WARNING):
        get_pricing(name)
        get_pricing(name)
        get_pricing(name)
    warns = [r for r in caplog.records if "Unknown LLM model" in r.message and name in r.message]
    assert len(warns) == 1


# ── compute_cost_usd math ───────────────────────────────────


def test_compute_cost_usd_zero_tokens_returns_zero():
    """All-zero call — returns Decimal('0'), not None or 0.0."""
    cost = compute_cost_usd("claude-opus-4-7")
    assert cost == Decimal("0")
    assert isinstance(cost, Decimal)


def test_compute_cost_usd_all_four_categories_summed():
    """All four token categories rolled into the total — matches
    `_PRICING['claude-opus-4-7']` × 1k tokens each.

    Per 1M:
      input:       15.00
      output:      75.00
      cache_read:   1.50
      cache_write: 18.75
    Per 1k each:
      0.015 + 0.075 + 0.0015 + 0.01875 = 0.11025
    """
    cost = compute_cost_usd(
        "claude-opus-4-7",
        input_tokens=1_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
        cache_write_tokens=1_000,
    )
    assert cost == Decimal("0.11025")


def test_compute_cost_usd_for_ollama_is_zero():
    """No matter the token volume, Ollama is free — important so a
    big local-model call doesn't blow the operator's daily budget."""
    cost = compute_cost_usd(
        "qwen2.5:32b",
        input_tokens=1_000_000_000,
        output_tokens=1_000_000_000,
    )
    assert cost == Decimal("0")


def test_compute_cost_usd_unknown_model_uses_default():
    """An unknown model is billed against DEFAULT_PRICING. Compute
    matches the table snapshot (input=$5, output=$20 per 1M)."""
    cost = compute_cost_usd("unknown-frontier", input_tokens=1_000_000)
    assert cost == DEFAULT_PRICING.input_per_mtok


def test_compute_cost_usd_decimal_precision_preserved():
    """200 input tokens × $15/1M = $0.003 — verify the Decimal math
    doesn't drift to a float-flavoured 0.0029999... result."""
    cost = compute_cost_usd("claude-opus-4-7", input_tokens=200)
    assert cost == Decimal("200") * Decimal("15.00") / Decimal(1_000_000)
    # Specifically: exactly 0.003.
    assert cost == Decimal("0.003")


def test_compute_cost_usd_only_input_tokens():
    """Pure input bill (e.g. a system-prompt cache check that returned
    no completion) — only the input row contributes."""
    cost = compute_cost_usd("gpt-4o", input_tokens=100_000)
    # 100k × 2.50/1M = 0.25
    assert cost == Decimal("0.25")


def test_compute_cost_usd_only_output_tokens():
    cost = compute_cost_usd("gpt-4o", output_tokens=100_000)
    # 100k × 10.00/1M = 1.00
    assert cost == Decimal("1.00")


def test_default_pricing_higher_than_haiku_lower_than_opus():
    """The DEFAULT is intentionally placed *above* the cheapest cloud
    model and *below* the most expensive — operator gets a noisy bill
    on an unknown model rather than a silent free pass or a panic."""
    assert DEFAULT_PRICING.input_per_mtok > get_pricing("claude-haiku-4-5-20251001").input_per_mtok
    assert DEFAULT_PRICING.input_per_mtok < get_pricing("claude-opus-4-7").input_per_mtok
