"""Wave H agentic tool handlers for the stocks strategy.

Stocks-side counterpart to ``crypto/agent_tools.py`` — builds the
handler dict ``run_agent`` dispatches to. Only the asset-agnostic
tools (``query_rag``, ``query_regime_memory``) are wired here; the
crypto-specific ones (``analyze_pair`` reading kline-buffer data,
``compute_var_95`` building returns from minute klines) don't have
stocks-shaped equivalents and are skipped — the strategy's
``tools=[...]`` list omits them.

The shape is intentionally narrower than crypto's because stocks
threads its prompt context directly through ``analyze()`` kwargs
rather than a frozen ``PromptContext`` dataclass.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from halal_trader.core.llm.tools import ToolCall

logger = logging.getLogger(__name__)


ToolHandler = Callable[["ToolCall"], Awaitable[str]]


def build_agent_handlers(
    *,
    hub: Any | None = None,
) -> dict[str, ToolHandler]:
    """Build the handler dict the stocks agent loop dispatches to.

    Mirrors :func:`crypto.agent_tools.build_agent_handlers` but only
    covers the asset-agnostic tools. ``hub`` is the
    :class:`InsightsHub` exposing the RAG + regime memory stores.
    """
    return {
        "query_rag": _make_query_rag_handler(hub=hub),
        "query_regime_memory": _make_query_regime_memory_handler(hub=hub),
    }


def _make_query_rag_handler(*, hub: Any | None) -> ToolHandler:
    """Retrieve top-K analogous past trade rationales from the RAG store."""

    async def _handler(call: "ToolCall") -> str:
        query = str(call.args.get("query") or "").strip()
        if not query:
            return "Error: query_rag requires a non-empty 'query' string."
        k = int(call.args.get("k", 5))
        rag = getattr(hub, "rag", None) if hub is not None else None
        if rag is None:
            return "RAG store not wired this cycle — no analogous rationales available."
        try:
            from halal_trader.core.llm.rag import format_rag_for_prompt

            hits = await rag.query(query, k=k)
            if not hits:
                return f"No analogous past rationales found for query: {query!r}."
            return format_rag_for_prompt(hits, max_rows=k)
        except Exception as exc:  # noqa: BLE001
            logger.debug("query_rag failed: %s", exc)
            return f"RAG lookup failed transiently: {exc}"

    return _handler


def _make_query_regime_memory_handler(*, hub: Any | None) -> ToolHandler:
    """Top-K analogous past market regimes via pgvector cosine similarity.

    Identical semantics to the crypto-side handler — the regime memory
    store is asset-agnostic, so the same module backs both bots.
    """

    async def _handler(call: "ToolCall") -> str:
        regime = getattr(hub, "regime", None) if hub is not None else None
        if regime is None:
            return "Regime memory not wired this cycle — no historical analogues available."
        try:
            from halal_trader.ml.regime_memory import (
                RegimeFeatures,
                format_for_prompt,
            )

            args = dict(call.args or {})
            k = int(args.pop("k", 5))
            allowed = {
                "volatility",
                "trend",
                "breadth",
                "sentiment",
                "drawdown",
                "volume_ratio",
                "correlation",
                "realized_return_1d",
                "rsi",
                "spread_bps",
            }
            features_dict = {k_: float(v) for k_, v in args.items() if k_ in allowed}
            features = RegimeFeatures(**features_dict)
            hits = await regime.query(features, k=k)
            if not hits:
                return "No analogous past regimes found for the requested feature vector."
            return format_for_prompt(features, hits, max_lines=k)
        except Exception as exc:  # noqa: BLE001
            logger.debug("query_regime_memory failed: %s", exc)
            return f"Regime memory lookup failed transiently: {exc}"

    return _handler
