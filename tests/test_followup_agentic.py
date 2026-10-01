"""Follow-up tests for Wave H deferrals — query_regime_memory tool + stocks agentic.

The base Wave H wiring is covered by ``tests/test_agentic_wiring.py``;
this file pins the two items that were explicitly listed as Wave H
deferrals in ``cleanup_roadmap.md``:

* The third tool from the original Wave H spec
  (``query_regime_memory``) is defined + handler-bound on stocks.
* The stocks-side ``TradingStrategy`` has an agentic branch with the
  asset-agnostic tools (RAG + regime memory).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.llm.tools import QUERY_REGIME_MEMORY_TOOL, ToolCall

# ── Tool definition shape ───────────────────────────────────────


def test_query_regime_memory_tool_has_stable_name() -> None:
    """The strategy's tools=[...] list references this name; pin it."""
    assert QUERY_REGIME_MEMORY_TOOL.name == "query_regime_memory"


def test_query_regime_memory_tool_schema_has_k_param() -> None:
    """The handler reads ``k`` to bound the result count; pin the
    schema so a refactor that drops it doesn't silently flood the
    LLM with results."""
    schema = QUERY_REGIME_MEMORY_TOOL.input_schema
    assert "k" in schema["properties"]
    assert schema["properties"]["k"]["maximum"] == 20


def test_query_regime_memory_openai_projection() -> None:
    """The provider-side helper (used by GLMLLM) projects the schema
    onto the OpenAI-compatible function envelope correctly."""
    openai_payload = QUERY_REGIME_MEMORY_TOOL.for_openai()
    assert openai_payload["type"] == "function"
    assert openai_payload["function"]["name"] == "query_regime_memory"


# ── Stocks-side wiring ─────────────────────────────────────────


def test_stocks_strategy_default_is_not_agentic() -> None:
    """Stocks agentic mode defaults to off."""
    from halal_trader.trading.strategy import TradingStrategy

    strat = TradingStrategy(
        llm=MagicMock(),
        repo=MagicMock(),
        llm_provider_name="x",
        max_position_pct=0.2,
        daily_loss_limit=0.02,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
    )
    assert strat._agentic_enabled is False


def test_stocks_strategy_agentic_flag_persists() -> None:
    """When enabled, the knobs are exposed for runtime inspection."""
    from halal_trader.trading.strategy import TradingStrategy

    strat = TradingStrategy(
        llm=MagicMock(),
        repo=MagicMock(),
        llm_provider_name="x",
        max_position_pct=0.2,
        daily_loss_limit=0.02,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
        agentic_enabled=True,
        agentic_max_turns=4,
        agentic_max_seconds=20.0,
    )
    assert strat._agentic_enabled is True
    assert strat._agentic_max_turns == 4
    assert strat._agentic_max_seconds == 20.0


def test_stocks_settings_expose_agentic_knobs() -> None:
    """The stocks agentic knobs exist with their documented defaults."""
    from halal_trader.config import StockSettings

    s = StockSettings()
    assert s.agentic_enabled is False
    assert s.agentic_max_turns == 5
    assert s.agentic_max_seconds == 30.0


# ── Stocks handler behaviour ───────────────────────────────────


@pytest.mark.asyncio
async def test_stocks_query_rag_handler_routes_to_store() -> None:
    """The handler routes the query to the hub's RAG store."""
    from halal_trader.trading.agent_tools import build_agent_handlers

    rag = MagicMock()
    hit = MagicMock(
        symbol="AAPL",
        text="Pre-market gap up + volume — closed +1.2%.",
        outcome_pnl_pct=0.012,
        rationale_id="r5",
        timestamp="2025-08-15T13:30:00+00:00",
    )
    rag.query = AsyncMock(return_value=[(hit, 0.74)])
    hub = MagicMock(rag=rag)

    handlers = build_agent_handlers(hub=hub)
    out = await handlers["query_rag"](
        ToolCall(name="query_rag", args={"query": "pre-market gap up large cap", "k": 3})
    )
    rag.query.assert_awaited_once()
    assert "AAPL" in out or "pre-market" in out.lower()


@pytest.mark.asyncio
async def test_stocks_query_rag_handler_no_hub() -> None:
    from halal_trader.trading.agent_tools import build_agent_handlers

    handlers = build_agent_handlers(hub=None)
    out = await handlers["query_rag"](ToolCall(name="query_rag", args={"query": "x"}))
    assert "not wired" in out.lower()


@pytest.mark.asyncio
async def test_stocks_query_rag_blank_query_rejected() -> None:
    from halal_trader.trading.agent_tools import build_agent_handlers

    handlers = build_agent_handlers(hub=MagicMock(rag=MagicMock()))
    out = await handlers["query_rag"](ToolCall(name="query_rag", args={"query": ""}))
    assert "Error" in out


@pytest.mark.asyncio
async def test_stocks_handler_set_is_rag_and_regime_memory() -> None:
    """The stocks handler dict binds exactly the two asset-agnostic tools."""
    from halal_trader.trading.agent_tools import build_agent_handlers

    handlers = build_agent_handlers(hub=None)
    assert set(handlers.keys()) == {"query_rag", "query_regime_memory"}
