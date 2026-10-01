"""Typed JSONSchema tool definitions for LLM tool-use calls.

Wave E switches the strategy LLM call from "ask for a JSON blob and
schema-repair on retry" to native provider tool use. The model emits
a structured ``submit_decisions`` call with arguments validated by the
schema; the SDK returns a Python dict and we materialise the
TradingPlan from it.

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


# ── Agentic tools (Wave H pre-wires these) ────────────────────────


QUERY_REGIME_MEMORY_TOOL = Tool(
    name="query_regime_memory",
    description=(
        "Retrieve top-K historical days whose market regime is most similar "
        "to today's, by cosine similarity on a 10-dim regime feature vector. "
        "Use this to ask 'when has the market looked like this before, and "
        "what happened next?' before committing to a trade. Args supply the "
        "feature vector explicitly (or pass an empty dict to use today's "
        "current snapshot)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "volatility": {
                "type": "number",
                "description": "ATR / price (decimal, e.g. 0.02 for 2%).",
            },
            "trend": {
                "type": "number",
                "description": "Multi-timeframe trend alignment in [-1, +1].",
            },
            "breadth": {
                "type": "number",
                "description": "Share of universe up minus down in [-1, +1].",
            },
            "sentiment": {
                "type": "number",
                "description": "Composite news/social sentiment in [-1, +1].",
            },
            "drawdown": {
                "type": "number",
                "description": "Current drawdown from peak in [0, 1].",
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "How many analogous past regimes to return.",
            },
        },
        "additionalProperties": False,
    },
)


QUERY_RAG_TOOL = Tool(
    name="query_rag",
    description=(
        "Retrieve top-K analogous past trade rationales by semantic similarity. "
        "Returns a list of (symbol, text, outcome_pnl_pct, similarity)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-form text to match against."},
            "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


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
        "symbol": {"type": "string", "description": "Trading pair, e.g. 'BTCUSDT'"},
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
