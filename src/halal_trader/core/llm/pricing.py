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
    # GLM-5.2 via OpenRouter (verified 2026-07-01). OpenRouter passes
    # per-host prompt-cache discounts through inconsistently, so cache
    # reads are billed at the full input rate here — errs high.
    "z-ai/glm-5.2": _p("0.93", "3.00", "0.93", "0"),
    # GLM-5.2 on Z.ai direct — published $1.40/$4.40, cached input $0.26.
    "glm-5.2": _p("1.40", "4.40", "0.26", "0"),
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
    billed slightly above the input rate (some hosts' policy). Both are
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
