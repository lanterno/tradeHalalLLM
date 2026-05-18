"""Follow-up tests for Wave H deferrals — query_regime_memory tool + stocks agentic.

The base Wave H wiring is covered by ``tests/test_agentic_wiring.py``;
this file pins the two items that were explicitly listed as Wave H
deferrals in ``cleanup_roadmap.md``:

* The third tool from the original Wave H spec
  (``query_regime_memory``) is now defined + handler-bound on both
  crypto + stocks.
* The stocks-side ``TradingStrategy`` mirrors the crypto agentic
  branch with the asset-agnostic tools (RAG + regime memory).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.llm.tools import (
    CRYPTO_AGENTIC_TOOLS,
    QUERY_REGIME_MEMORY_TOOL,
    ToolCall,
)

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


def test_query_regime_memory_in_crypto_agentic_tools() -> None:
    """The bundle constant the agentic mode picks tools from must
    include the new tool — verified by name match."""
    names = [t.name for t in CRYPTO_AGENTIC_TOOLS]
    assert "query_regime_memory" in names


def test_query_regime_memory_anthropic_projection() -> None:
    """The provider-side helpers (used by AnthropicLLM, OpenAILLM)
    project the schema correctly."""
    payload = QUERY_REGIME_MEMORY_TOOL.for_anthropic()
    assert payload["name"] == "query_regime_memory"
    assert "input_schema" in payload
    openai_payload = QUERY_REGIME_MEMORY_TOOL.for_openai()
    assert openai_payload["type"] == "function"
    assert openai_payload["function"]["name"] == "query_regime_memory"


# ── Crypto handler behaviour ────────────────────────────────────


@pytest.mark.asyncio
async def test_crypto_query_regime_memory_no_hub_returns_friendly_message() -> None:
    """Standalone / dashboard-only contexts have no hub. The handler
    must report "not wired" rather than crash the agent loop."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    handlers = build_agent_handlers(
        ctx=PromptContext(account=MagicMock(total_balance_usdt=1.0)),
        hub=None,
        timeframes=None,
    )
    out = await handlers["query_regime_memory"](ToolCall(name="query_regime_memory", args={"k": 3}))
    assert "not wired" in out.lower()


@pytest.mark.asyncio
async def test_crypto_query_regime_memory_routes_to_store_when_wired() -> None:
    """With a hub.regime store wired, the handler builds a
    RegimeFeatures from the call args + invokes ``regime.query``."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext
    from halal_trader.ml.regime_memory import RegimeSnapshot

    regime = MagicMock()
    from halal_trader.ml.regime_memory import RegimeFeatures

    snap = RegimeSnapshot(
        date="2026-03-15",
        features=RegimeFeatures(volatility=0.02, trend=0.1),
        outcome_pnl_pct=0.008,
        outcome_win_rate=0.60,
        outcome_n_trades=5,
        note="post-FOMC rally",
    )
    regime.query = AsyncMock(return_value=[(snap, 0.82)])
    hub = MagicMock(regime=regime)

    handlers = build_agent_handlers(
        ctx=PromptContext(account=MagicMock(total_balance_usdt=1.0)),
        hub=hub,
        timeframes=None,
    )
    out = await handlers["query_regime_memory"](
        ToolCall(
            name="query_regime_memory",
            args={"volatility": 0.02, "trend": 0.1, "sentiment": -0.2, "k": 3},
        )
    )
    regime.query.assert_awaited_once()
    # Check the feature dict roundtripped through RegimeFeatures.
    call_args = regime.query.await_args
    features = call_args.args[0]
    assert features.volatility == 0.02
    assert features.trend == 0.1
    assert features.sentiment == -0.2
    assert "2026-03-15" in out or "post-FOMC" in out


@pytest.mark.asyncio
async def test_crypto_query_regime_memory_drops_unknown_keys() -> None:
    """A malformed tool call (extra keys the dataclass doesn't know
    about) must not crash the loop — the handler filters by the
    allowed feature set."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    regime = MagicMock()
    regime.query = AsyncMock(return_value=[])
    hub = MagicMock(regime=regime)
    handlers = build_agent_handlers(
        ctx=PromptContext(account=MagicMock(total_balance_usdt=1.0)),
        hub=hub,
        timeframes=None,
    )
    out = await handlers["query_regime_memory"](
        ToolCall(
            name="query_regime_memory",
            args={"volatility": 0.02, "garbage_key": 999, "moon_phase": "waxing"},
        )
    )
    # Should still call query — the garbage keys are dropped, not raised.
    regime.query.assert_awaited_once()
    assert "No analogous past regimes" in out


# ── Stocks-side wiring ─────────────────────────────────────────


def test_stocks_strategy_default_is_not_agentic() -> None:
    """Stocks defaults to off, same as crypto."""
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
    """``CRYPTO_AGENTIC_*`` env vars mirror on the stocks side."""
    from halal_trader.config import StockSettings

    s = StockSettings()
    assert s.agentic_enabled is False
    assert s.agentic_max_turns == 5
    assert s.agentic_max_seconds == 30.0


# ── Stocks handler behaviour ───────────────────────────────────


@pytest.mark.asyncio
async def test_stocks_query_rag_handler_routes_to_store() -> None:
    """Mirror of the crypto handler — same RAG store, different bot."""
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
async def test_stocks_handler_set_omits_crypto_specific_tools() -> None:
    """analyze_pair and compute_var_95 don't have clean stocks
    equivalents; their absence from the stocks handler dict is
    deliberate and the strategy's tools=[...] omits them."""
    from halal_trader.trading.agent_tools import build_agent_handlers

    handlers = build_agent_handlers(hub=None)
    assert set(handlers.keys()) == {"query_rag", "query_regime_memory"}
