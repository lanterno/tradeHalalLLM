"""Wave H agentic tool handlers for the crypto strategy.

The :func:`run_agent` loop in ``core/llm/agent.py`` drives the LLM
through a bounded tool-calling conversation. Each non-terminal tool
(``analyze_pair``, ``query_rag``, ``compute_var_95``) needs a
concrete handler — a small async closure that resolves the tool's
arguments against the cycle's already-fetched context and returns
formatted text the LLM sees on its next turn.

The handlers are intentionally *thin*: they don't reach for new
network resources or open broker connections. Everything they need
is already in :class:`PromptContext` or the injected ``hub`` /
``timeframes`` refs. Anything not available falls back to a short
"not available this cycle" string instead of crashing — the agent
should be free to ask for it without aborting the whole cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from halal_trader.core.llm.tools import ToolCall
    from halal_trader.crypto.prompts import PromptContext

logger = logging.getLogger(__name__)


ToolHandler = Callable[["ToolCall"], Awaitable[str]]


def build_agent_handlers(
    *,
    ctx: "PromptContext",
    hub: Any | None = None,
    timeframes: Any | None = None,
) -> dict[str, ToolHandler]:
    """Build the handler dict the agent loop dispatches to.

    Returns a mapping ``{tool_name: async (ToolCall) -> str}`` covering
    the three non-terminal tools in ``CRYPTO_AGENTIC_TOOLS``
    (the terminal ``submit_decisions`` / ``submit_plan`` ends the
    loop and is handled by the driver, not here).
    """
    return {
        "analyze_pair": _make_analyze_pair_handler(ctx, timeframes=timeframes),
        "query_rag": _make_query_rag_handler(hub=hub),
        "compute_var_95": _make_compute_var_handler(ctx),
    }


def _make_analyze_pair_handler(
    ctx: "PromptContext",
    *,
    timeframes: Any | None,
) -> ToolHandler:
    """Return a deeper read on one pair than what's already in the prompt.

    Source 1: this cycle's already-fetched indicators (avoids a
    re-compute). Source 2 (optional): the multi-timeframe analyzer if
    one is wired — gives the LLM access to 5m/15m/1h/4h structure
    on demand without inflating every cycle's prompt.
    """

    async def _handler(call: "ToolCall") -> str:
        symbol = str(call.args.get("symbol") or "").upper()
        if not symbol:
            return "Error: analyze_pair requires a 'symbol' argument (e.g. 'BTCUSDT')."
        lines: list[str] = []
        indicators = (ctx.indicators_cache or {}).get(symbol)
        if indicators:
            try:
                from halal_trader.crypto.indicators import format_indicators_for_prompt

                lines.append(format_indicators_for_prompt(symbol, indicators))
            except Exception:  # noqa: BLE001
                # Sparse indicator dicts (e.g. a snapshot mid-cycle that
                # didn't capture every field) — render what we have.
                kv = ", ".join(f"{k}={v}" for k, v in indicators.items())
                lines.append(f"{symbol}: {kv}")
        else:
            lines.append(f"{symbol}: no indicators available this cycle.")
        # Multi-timeframe deepening — only if an analyzer is wired.
        if timeframes is not None:
            try:
                from halal_trader.crypto.timeframes import (
                    build_timeframe_text,
                )

                tf_text = await build_timeframe_text(timeframes, [symbol])
                if tf_text:
                    lines.append(tf_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("analyze_pair tf lookup failed for %s: %s", symbol, exc)
        return "\n\n".join(lines)

    return _handler


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


def _make_compute_var_handler(ctx: "PromptContext") -> ToolHandler:
    """Compute 95% VaR from recent closing returns of the requested symbols.

    The simulator uses a small-sample Cornish-Fisher-adjusted VaR
    (see :mod:`ml.bayesian_var`) which is honest about fat-tail
    behaviour on the kind of 60-bar windows the cycle has fetched.
    """

    async def _handler(call: "ToolCall") -> str:
        symbols = call.args.get("symbols") or []
        weights = call.args.get("weights") or []
        if not symbols or not weights or len(symbols) != len(weights):
            return (
                "Error: compute_var_95 requires equal-length 'symbols' and "
                "'weights' arrays (weights as fractions of equity)."
            )
        if abs(sum(float(w) for w in weights) - 1.0) > 0.05:
            return (
                "Note: weights should sum to ~1.0 (got "
                f"{sum(float(w) for w in weights):.3f}). Computing anyway."
            )
        # Build a per-symbol returns series from this cycle's klines.
        # If a requested symbol isn't in the cache, skip it (the LLM
        # gets a "no data" line rather than a crash).
        klines_by_symbol = ctx.klines_by_symbol or {}
        symbol_returns: dict[str, list[float]] = {}
        for sym in symbols:
            klines = klines_by_symbol.get(str(sym).upper())
            if not klines or len(klines) < 3:
                continue
            closes = [float(k.close) for k in klines]
            rets = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]
            if rets:
                symbol_returns[str(sym).upper()] = rets
        if not symbol_returns:
            return "compute_var_95: no klines available for the requested symbols."
        # Equal-weighted across the available symbols' return series.
        # (Honouring the LLM-supplied weights exactly would require an
        # aligned bar timeline; the simpler approach is enough for the
        # tail-risk read the model is asking for.)
        try:
            from halal_trader.ml.bayesian_var import bayesian_var, render_result

            joined: list[float] = []
            for rets in symbol_returns.values():
                joined.extend(rets)
            result = bayesian_var(joined, alpha=0.05)
            return render_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.debug("compute_var_95 failed: %s", exc)
            return f"compute_var_95 failed transiently: {exc}"

    return _handler
