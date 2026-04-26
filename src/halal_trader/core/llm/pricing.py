"""LLM token pricing — single source of truth for cost attribution.

USD per 1M tokens, current as of late 2025 / early 2026 list prices. The
table is intentionally narrow — providers change pricing several times
a year, so we want one place to bump the numbers when they shift.

Unknown models fall back to ``DEFAULT_PRICING`` (a conservative-high
estimate) and are logged once. We'd rather over-attribute cost than
silently swallow it — if a daily-cap circuit breaker trips on a slightly
inflated bill, that's a recoverable false positive; if it never trips
because we recorded $0 against an uncosted model, we lose money quietly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens for each token category."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    cache_write_per_mtok: Decimal


def _p(*nums: str) -> ModelPricing:
    return ModelPricing(*(Decimal(n) for n in nums))


# input, output, cache_read, cache_write — USD per 1M tokens.
_PRICING: dict[str, ModelPricing] = {
    # Anthropic Claude family
    "claude-opus-4-7": _p("15.00", "75.00", "1.50", "18.75"),
    "claude-opus-4-1": _p("15.00", "75.00", "1.50", "18.75"),
    "claude-opus-4-20250514": _p("15.00", "75.00", "1.50", "18.75"),
    "claude-sonnet-4-6": _p("3.00", "15.00", "0.30", "3.75"),
    "claude-sonnet-4-5": _p("3.00", "15.00", "0.30", "3.75"),
    "claude-sonnet-4-20250514": _p("3.00", "15.00", "0.30", "3.75"),
    "claude-haiku-4-5-20251001": _p("1.00", "5.00", "0.10", "1.25"),
    # OpenAI GPT-4o family
    "gpt-4o": _p("2.50", "10.00", "1.25", "2.50"),
    "gpt-4o-mini": _p("0.15", "0.60", "0.075", "0.15"),
    "gpt-4-turbo": _p("10.00", "30.00", "5.00", "10.00"),
    # Local Ollama models — cost is electricity, treat as zero.
    "ollama": _p("0", "0", "0", "0"),
}

# When we don't recognise the model, charge against this. Set high enough
# that surprise spend trips the cap, low enough that we don't false-alarm
# on small-talk completions during dev.
DEFAULT_PRICING = _p("5.00", "20.00", "0.50", "6.25")

_warned_unknown: set[str] = set()


def get_pricing(model: str) -> ModelPricing:
    """Return pricing for ``model`` or fall back to a conservative default."""
    if model in _PRICING:
        return _PRICING[model]
    # Ollama models often look like ``qwen2.5:32b`` — anything that didn't
    # match a cloud entry above is treated as local/free.
    if ":" in model or model.startswith(("ollama", "qwen", "llama", "mistral")):
        return _PRICING["ollama"]
    if model not in _warned_unknown:
        _warned_unknown.add(model)
        logger.warning("Unknown LLM model %r — using default pricing for cost tracking", model)
    return DEFAULT_PRICING


def compute_cost_usd(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal:
    """Total USD cost for a single LLM call. Returns Decimal for exact roll-up.

    Cache reads are billed at the discounted cache rate; cache writes are
    billed slightly above the input rate (Anthropic's policy). Both are
    counted ON TOP OF input_tokens — providers report them separately.
    """
    p = get_pricing(model)
    cost = (
        Decimal(input_tokens) * p.input_per_mtok
        + Decimal(output_tokens) * p.output_per_mtok
        + Decimal(cache_read_tokens) * p.cache_read_per_mtok
        + Decimal(cache_write_tokens) * p.cache_write_per_mtok
    ) / Decimal(1_000_000)
    return cost
