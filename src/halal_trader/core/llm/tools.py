"""Typed JSONSchema tool definitions for LLM tool-use calls.

Wave E switches the strategy LLM call from "ask for a JSON blob and
schema-repair on retry" to native provider tool use. The model emits
a structured ``submit_plan`` call with arguments validated by the
schema; the SDK returns a Python dict and we materialise the
TradingPlan from it.

Anthropic and OpenAI both speak the same JSONSchema-flavoured tool
format; we encode the tools in a provider-agnostic ``Tool`` dataclass
and let each provider's adapter project it onto its native API shape.
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

    def for_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

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
    id: str | None = None  # provider-supplied call id (Anthropic), if any


# ── Crypto strategy tools ────────────────────────────────────────


_BUY_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "Trading pair, e.g. 'BTCUSDT'"},
        "size_pct": {
            "type": "number",
            "description": "Position size as fraction of equity (0..max_position_pct).",
            "minimum": 0,
            "maximum": 1,
        },
        "stop_loss_pct": {
            "type": "number",
            "description": "Stop-loss distance from entry as a fraction (e.g. 0.01 = 1%).",
            "minimum": 0,
            "maximum": 0.5,
        },
        "take_profit_pct": {
            "type": "number",
            "description": "Take-profit distance from entry as a fraction.",
            "minimum": 0,
            "maximum": 1.0,
        },
        "confidence": {
            "type": "number",
            "description": "Model's confidence in the trade (0..1).",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning": {"type": "string", "description": "One-paragraph rationale."},
        "thesis_tag": {
            "type": "string",
            "description": (
                "Optional setup classifier: breakout, mean_revert, momentum, "
                "trend_follow, scalp, news_catalyst."
            ),
        },
    },
    "required": ["symbol", "size_pct", "stop_loss_pct", "take_profit_pct", "reasoning"],
    "additionalProperties": False,
}


_SELL_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "Trading pair, e.g. 'BTCUSDT'"},
        "reason": {"type": "string", "description": "One-line rationale for closing."},
    },
    "required": ["symbol", "reason"],
    "additionalProperties": False,
}


SUBMIT_PLAN_TOOL = Tool(
    name="submit_plan",
    description=(
        "Submit the final trading plan for this cycle. Must call exactly once. "
        "Empty buys/sells lists are valid (HOLD)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "market_outlook": {
                "type": "string",
                "description": "One-paragraph high-level read of the current setup.",
            },
            "buys": {
                "type": "array",
                "items": _BUY_DECISION_SCHEMA,
                "description": "BUY decisions to enter this cycle.",
            },
            "sells": {
                "type": "array",
                "items": _SELL_DECISION_SCHEMA,
                "description": "SELL decisions for currently-open positions.",
            },
        },
        "required": ["market_outlook", "buys", "sells"],
        "additionalProperties": False,
    },
)


# ── Agentic tools (Wave H pre-wires these) ────────────────────────


ANALYZE_PAIR_TOOL = Tool(
    name="analyze_pair",
    description=(
        "Pull deeper data on one pair (multi-timeframe indicators, basis, "
        "whale flows). Call this if the model needs more context before a buy."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Trading pair, e.g. 'BTCUSDT'"},
        },
        "required": ["symbol"],
        "additionalProperties": False,
    },
)


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


COMPUTE_VAR_TOOL = Tool(
    name="compute_var_95",
    description=(
        "Compute 95% Value-at-Risk of a hypothetical position vector. "
        "Returns the projected single-day loss at the 5th percentile."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "symbols": {"type": "array", "items": {"type": "string"}},
            "weights": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["symbols", "weights"],
        "additionalProperties": False,
    },
)


CRYPTO_STRATEGY_TOOLS: list[Tool] = [SUBMIT_PLAN_TOOL]
CRYPTO_AGENTIC_TOOLS: list[Tool] = [
    ANALYZE_PAIR_TOOL,
    QUERY_RAG_TOOL,
    QUERY_REGIME_MEMORY_TOOL,
    COMPUTE_VAR_TOOL,
    SUBMIT_PLAN_TOOL,
]


# ── Strategy-shaped tool: legacy decisions[] schema ──────────────
# Mirrors the JSON shape today's prompt asks for, so
# ``CryptoTradingPlan.model_validate(tool_call.args)`` works without a
# translation layer. The richer :data:`SUBMIT_PLAN_TOOL` (above) is
# kept for the agentic surface in Wave H, where the model emits a
# more abstract plan and the strategy materialises it into decisions.

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
