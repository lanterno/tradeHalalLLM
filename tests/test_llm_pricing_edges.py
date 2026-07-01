"""Edge-case tests for :mod:`core.llm.pricing`.

`test_llm_pricing.py` covers the basic shapes (known model lookup,
unknown → DEFAULT_PRICING, compute sums input + output, cache
categories included). This file pins the remaining contract: the two
GLM-5.2 table rows' snapshot prices (so a silent edit to the table is
caught), the removal of the old local-model free branch, the once-only
unknown-model warning, and the all-four-categories rollup math.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from halal_trader.core.llm.pricing import (
    DEFAULT_PRICING,
    compute_cost_usd,
    get_pricing,
)

# ── GLM-5.2 family snapshots ────────────────────────────────


def test_openrouter_glm_pricing_snapshot():
    """OpenRouter row (verified 2026-07-01). Cache reads are billed at
    the FULL input rate here — OpenRouter passes per-host cache
    discounts through inconsistently, so the table errs high."""
    p = get_pricing("z-ai/glm-5.2")
    assert p.input_per_mtok == Decimal("0.93")
    assert p.output_per_mtok == Decimal("3.00")
    assert p.cache_read_per_mtok == Decimal("0.93")
    assert p.cache_write_per_mtok == Decimal("0")


def test_zai_direct_glm_pricing_snapshot():
    """Z.ai direct row — published $1.40/$4.40, cached input $0.26."""
    p = get_pricing("glm-5.2")
    assert p.input_per_mtok == Decimal("1.40")
    assert p.output_per_mtok == Decimal("4.40")
    assert p.cache_read_per_mtok == Decimal("0.26")
    assert p.cache_write_per_mtok == Decimal("0")


def test_openrouter_cache_read_billed_at_input_rate():
    """Pin the errs-high policy: the OpenRouter row's cache-read rate
    equals its input rate (no discount assumed), unlike Z.ai direct
    where the published discount applies."""
    openrouter = get_pricing("z-ai/glm-5.2")
    zai = get_pricing("glm-5.2")
    assert openrouter.cache_read_per_mtok == openrouter.input_per_mtok
    assert zai.cache_read_per_mtok < zai.input_per_mtok


# ── Old local-free branch removed ───────────────────────────


def test_ollama_style_ids_now_hit_default_pricing():
    """The colon/qwen/llama/mistral free branch was removed with the
    Ollama provider — a stale local model id in an old config now gets
    the conservative DEFAULT (over-bill) instead of a silent free pass."""
    assert get_pricing("qwen2.5:32b") == DEFAULT_PRICING
    assert get_pricing("llama3.1:70b") == DEFAULT_PRICING
    assert get_pricing("mistral-nemo:12b") == DEFAULT_PRICING
    assert get_pricing("ollama-custom-build") == DEFAULT_PRICING


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
    cost = compute_cost_usd("z-ai/glm-5.2")
    assert cost == Decimal("0")
    assert isinstance(cost, Decimal)


def test_compute_cost_usd_all_four_categories_summed():
    """All four token categories rolled into the total — matches
    `_PRICING['z-ai/glm-5.2']` × 1k tokens each.

    Per 1M:
      input:       0.93
      output:      3.00
      cache_read:  0.93
      cache_write: 0
    Per 1k each:
      0.00093 + 0.003 + 0.00093 + 0 = 0.00486
    """
    cost = compute_cost_usd(
        "z-ai/glm-5.2",
        input_tokens=1_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
        cache_write_tokens=1_000,
    )
    assert cost == Decimal("0.00486")


def test_compute_cost_usd_unknown_model_uses_default():
    """An unknown model is billed against DEFAULT_PRICING. Compute
    matches the table snapshot (input=$5, output=$20 per 1M)."""
    cost = compute_cost_usd("unknown-frontier", input_tokens=1_000_000)
    assert cost == DEFAULT_PRICING.input_per_mtok


def test_compute_cost_usd_decimal_precision_preserved():
    """200 input tokens × $0.93/1M = $0.000186 — verify the Decimal math
    doesn't drift to a float-flavoured result."""
    cost = compute_cost_usd("z-ai/glm-5.2", input_tokens=200)
    assert cost == Decimal("200") * Decimal("0.93") / Decimal(1_000_000)
    # Specifically: exactly 0.000186.
    assert cost == Decimal("0.000186")


def test_compute_cost_usd_only_input_tokens():
    """Pure input bill (e.g. a prompt-cache warm-up that returned no
    completion) — only the input row contributes."""
    cost = compute_cost_usd("glm-5.2", input_tokens=100_000)
    # 100k × 1.40/1M = 0.14
    assert cost == Decimal("0.14")


def test_compute_cost_usd_only_output_tokens():
    cost = compute_cost_usd("glm-5.2", output_tokens=100_000)
    # 100k × 4.40/1M = 0.44
    assert cost == Decimal("0.44")


def test_default_pricing_sits_above_every_known_row():
    """The DEFAULT is intentionally more expensive than every known GLM
    row — an unknown model produces a noisy (over-attributed) bill that
    trips the daily cap rather than a silent under-count."""
    for model in ("z-ai/glm-5.2", "glm-5.2"):
        known = get_pricing(model)
        assert DEFAULT_PRICING.input_per_mtok > known.input_per_mtok
        assert DEFAULT_PRICING.output_per_mtok > known.output_per_mtok
