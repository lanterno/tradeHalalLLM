"""LLM pricing & CallUsage tests — cost math and unknown-model fallback."""

from decimal import Decimal

from halal_trader.core.llm.base import CallUsage
from halal_trader.core.llm.pricing import (
    DEFAULT_PRICING,
    compute_cost_usd,
    get_pricing,
)


def test_known_openrouter_glm_model_pricing():
    p = get_pricing("z-ai/glm-5.2")
    assert p.input_per_mtok == Decimal("0.93")
    assert p.output_per_mtok == Decimal("3.00")


def test_unknown_model_falls_back_to_default():
    # Any unrecognised name hits the conservative default — we'd
    # rather over-bill than silently record $0.
    p = get_pricing("some-unreleased-frontier-model")
    assert p == DEFAULT_PRICING


def test_local_style_model_no_longer_free():
    """The local-model free branch (colon/qwen/llama prefixes) was
    removed with the Ollama provider — an old-config model id now gets
    the conservative default instead of a silent $0 bill."""
    p = get_pricing("qwen2.5:32b")
    assert p == DEFAULT_PRICING


def test_compute_cost_usd_sums_all_categories():
    # 1M input + 0.5M output on z-ai/glm-5.2 = $0.93 + $1.50 = $2.43
    cost = compute_cost_usd(
        "z-ai/glm-5.2",
        input_tokens=1_000_000,
        output_tokens=500_000,
    )
    assert cost == Decimal("2.43")


def test_compute_cost_usd_includes_cache_categories():
    # 1M cache reads on glm-5.2 (Z.ai) = $0.26; cache writes are $0.
    cost = compute_cost_usd(
        "glm-5.2",
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == Decimal("0.26")


def test_compute_cost_usd_returns_decimal_not_float():
    cost = compute_cost_usd("z-ai/glm-5.2", input_tokens=1000, output_tokens=500)
    assert isinstance(cost, Decimal)


def test_call_usage_total_tokens():
    u = CallUsage(input_tokens=100, output_tokens=50)
    assert u.total_tokens == 150
