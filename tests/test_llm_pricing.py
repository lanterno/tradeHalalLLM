"""LLM pricing & CallUsage tests — cost math and unknown-model fallback."""

from decimal import Decimal

from halal_trader.core.llm.base import CallUsage
from halal_trader.core.llm.pricing import (
    DEFAULT_PRICING,
    compute_cost_usd,
    get_pricing,
)


def test_known_anthropic_model_pricing():
    p = get_pricing("claude-opus-4-7")
    assert p.input_per_mtok == Decimal("15.00")
    assert p.output_per_mtok == Decimal("75.00")


def test_unknown_model_falls_back_to_default():
    # Cloud-shaped name (no colon prefix) hits the conservative default,
    # not free-ollama-pricing — we'd rather over-bill than silently $0.
    p = get_pricing("some-unreleased-frontier-model")
    assert p == DEFAULT_PRICING


def test_ollama_style_model_is_free():
    p = get_pricing("qwen2.5:32b")
    assert p.input_per_mtok == Decimal("0")


def test_compute_cost_usd_sums_all_categories():
    # 1M input + 0.5M output on opus-4-7 = $15 + $37.50 = $52.50
    cost = compute_cost_usd(
        "claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=500_000,
    )
    assert cost == Decimal("52.50")


def test_compute_cost_usd_includes_cache_categories():
    # 1M cache reads on opus-4-7 = $1.50; 1M cache writes = $18.75.
    cost = compute_cost_usd(
        "claude-opus-4-7",
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == Decimal("20.25")


def test_compute_cost_usd_returns_decimal_not_float():
    cost = compute_cost_usd("gpt-4o", input_tokens=1000, output_tokens=500)
    assert isinstance(cost, Decimal)


def test_call_usage_total_tokens():
    u = CallUsage(input_tokens=100, output_tokens=50)
    assert u.total_tokens == 150
