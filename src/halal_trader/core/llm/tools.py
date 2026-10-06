"""Typed JSONSchema tool definitions for LLM tool-use calls.

The strategy LLM call uses native provider tool use rather than "ask for
a JSON blob and schema-repair on retry". The model emits a structured
``submit_decisions`` call with arguments validated by the schema; the SDK
returns a Python dict and we materialise the TradingPlan from it.

The tools are encoded in a provider-agnostic ``Tool`` dataclass and
projected onto the OpenAI-compatible API shape (``for_openai``), which
is what every GLM endpoint speaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tool:
    """One callable the LLM can invoke."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    def for_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolCall:
    """One tool invocation the model made."""

    name: str
    args: dict[str, Any]
    id: str | None = None  # provider-supplied call id, if any


# ── Strategy-shaped tool: legacy decisions[] schema ──────────────
# Mirrors the JSON shape today's prompt asks for, so
# ``TradingPlan.model_validate(tool_call.args)`` works without a
# translation layer.

_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["buy", "sell", "hold"],
            "description": "Trade direction.",
        },
        "symbol": {"type": "string", "description": "Stock symbol, e.g. 'AAPL'"},
        "quantity": {
            "type": "number",
            "description": "Absolute order quantity in base-asset units (e.g. BTC, not USDT).",
            "minimum": 0,
        },
        "confidence": {
            "type": "number",
            "description": "Model's confidence in the decision (0..1).",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning": {
            "type": "string",
            "description": "One-line rationale for this specific decision.",
        },
        "stop_loss": {
            "type": "number",
            "description": (
                "Optional per-decision stop-loss price (USDT, absolute). "
                "Override the default — omit to use the configured percentage."
            ),
        },
        "target_price": {
            "type": "number",
            "description": (
                "Optional per-decision take-profit price (USDT, absolute). "
                "Override the default — omit to use the configured percentage."
            ),
        },
        "thesis_tag": {
            "type": "string",
            "description": (
                "Optional setup classifier: breakout / mean_revert / momentum / "
                "trend_follow / scalp / news_catalyst."
            ),
        },
    },
    "required": ["action", "symbol", "quantity", "confidence"],
    "additionalProperties": False,
}


SUBMIT_DECISIONS_TOOL = Tool(
    name="submit_decisions",
    description=(
        "Submit the final list of trading decisions for this cycle. Call exactly "
        "once. Empty decisions list = explicit HOLD on every position."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": _DECISION_SCHEMA,
                "description": "Buy/sell/hold actions to take this cycle.",
            },
            "reasoning": {
                "type": "string",
                "description": "One concise sentence — the rationale for the plan as a whole.",
            },
            "market_outlook": {
                "type": "string",
                "description": "Brief high-level read of the current setup.",
            },
        },
        "required": ["decisions", "market_outlook"],
        "additionalProperties": False,
    },
)
