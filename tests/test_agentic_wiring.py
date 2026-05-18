"""Wave H wiring tests — agentic loop + tool handlers + transcript persistence.

``core/llm/agent.py:run_agent`` (the bounded multi-turn driver) is
already covered by ``tests/test_llm_agent_budget.py``. This file
covers the *consumer wiring* added in this commit:

* The new ``crypto/agent_tools.py`` handlers behave sensibly across
  the happy path + the missing-data / missing-deps degradations.
* ``BaseStrategy._run_llm_analysis(agent=...)`` runs the multi-turn
  loop instead of a single call, materialises the terminal tool's
  args into the validate pipeline, and persists the transcript on
  the ``LlmDecision`` row.
* ``agentic_enabled=False`` keeps the legacy single-call behaviour
  exactly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.llm.tools import (
    ANALYZE_PAIR_TOOL,
    QUERY_RAG_TOOL,
    SUBMIT_DECISIONS_TOOL,
    ToolCall,
)

# ── crypto/agent_tools handlers ─────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_pair_handler_reads_indicators_cache() -> None:
    """The cheapest call: the handler should never refetch what the
    cycle already pulled — it reads from PromptContext.indicators_cache."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    ctx = PromptContext(
        account=MagicMock(total_balance_usdt=10_000.0),
        indicators_cache={"BTCUSDT": {"rsi_14": 60.0, "atr_14": 100.0}},
    )
    handlers = build_agent_handlers(ctx=ctx, hub=None, timeframes=None)
    out = await handlers["analyze_pair"](ToolCall(name="analyze_pair", args={"symbol": "BTCUSDT"}))
    assert "BTCUSDT" in out
    assert "RSI" in out.upper() or "rsi" in out


@pytest.mark.asyncio
async def test_analyze_pair_handler_rejects_blank_symbol() -> None:
    """Misformed tool calls return a clear error string the LLM can read,
    not a Python exception that aborts the agent loop."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    ctx = PromptContext(account=MagicMock(total_balance_usdt=1.0))
    handlers = build_agent_handlers(ctx=ctx, hub=None, timeframes=None)
    out = await handlers["analyze_pair"](ToolCall(name="analyze_pair", args={"symbol": ""}))
    assert "Error" in out or "requires" in out


@pytest.mark.asyncio
async def test_analyze_pair_handler_no_indicators_for_symbol() -> None:
    """Unknown symbol → graceful "no data" message, not a crash."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    ctx = PromptContext(
        account=MagicMock(total_balance_usdt=1.0),
        indicators_cache={"ETHUSDT": {"rsi_14": 50.0}},
    )
    handlers = build_agent_handlers(ctx=ctx, hub=None, timeframes=None)
    out = await handlers["analyze_pair"](ToolCall(name="analyze_pair", args={"symbol": "BTCUSDT"}))
    assert "BTCUSDT" in out
    assert "no indicators" in out.lower()


@pytest.mark.asyncio
async def test_query_rag_handler_no_hub_returns_friendly_string() -> None:
    """Wire-default: the dashboard process has no hub; the handler
    must say so plainly rather than raising AttributeError."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    handlers = build_agent_handlers(
        ctx=PromptContext(account=MagicMock(total_balance_usdt=1.0)),
        hub=None,
        timeframes=None,
    )
    out = await handlers["query_rag"](ToolCall(name="query_rag", args={"query": "rsi 30 bb lower"}))
    assert "not wired" in out.lower() or "no" in out.lower()


@pytest.mark.asyncio
async def test_query_rag_handler_with_hub_routes_to_store() -> None:
    """When a hub with a RAG store is wired, the handler should run
    the query and return the formatted hits."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    rag = MagicMock()
    # One hit shaped like (RationaleRowDC, similarity) — the formatter
    # accepts dicts/dataclasses via getattr.
    hit_row = MagicMock(
        symbol="BTCUSDT",
        text="RSI 28, BB lower band — bought, closed +0.8%.",
        outcome_pnl_pct=0.008,
        rationale_id="r1",
        timestamp="2026-04-01T12:00:00+00:00",
    )
    rag.query = AsyncMock(return_value=[(hit_row, 0.78)])
    hub = MagicMock(rag=rag)

    handlers = build_agent_handlers(
        ctx=PromptContext(account=MagicMock(total_balance_usdt=1.0)),
        hub=hub,
        timeframes=None,
    )
    out = await handlers["query_rag"](
        ToolCall(name="query_rag", args={"query": "rsi 30 bb lower", "k": 3})
    )
    rag.query.assert_awaited_once()
    assert "BTCUSDT" in out or "rsi" in out.lower()


@pytest.mark.asyncio
async def test_query_rag_handler_blank_query_rejected() -> None:
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    handlers = build_agent_handlers(
        ctx=PromptContext(account=MagicMock(total_balance_usdt=1.0)),
        hub=MagicMock(rag=MagicMock()),
        timeframes=None,
    )
    out = await handlers["query_rag"](ToolCall(name="query_rag", args={"query": ""}))
    assert "Error" in out


@pytest.mark.asyncio
async def test_compute_var_handler_requires_aligned_arrays() -> None:
    """Symbols and weights must be equal-length non-empty arrays."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext

    handlers = build_agent_handlers(
        ctx=PromptContext(account=MagicMock(total_balance_usdt=1.0)),
        hub=None,
        timeframes=None,
    )
    out = await handlers["compute_var_95"](
        ToolCall(name="compute_var_95", args={"symbols": ["BTCUSDT"], "weights": []})
    )
    assert "Error" in out


@pytest.mark.asyncio
async def test_compute_var_handler_computes_from_klines() -> None:
    """With klines for the requested symbol, compute VaR end-to-end."""
    from halal_trader.crypto.agent_tools import build_agent_handlers
    from halal_trader.crypto.prompts import PromptContext
    from halal_trader.domain.models import Kline

    # Build a 30-bar series with random-walk-ish returns.
    klines = []
    price = 50_000.0
    for i in range(30):
        price *= 1.0 + (0.001 if i % 2 else -0.001)
        klines.append(
            Kline(
                open_time=i,
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1.0,
                close_time=i + 1,
            )
        )
    ctx = PromptContext(
        account=MagicMock(total_balance_usdt=10_000.0),
        klines_by_symbol={"BTCUSDT": klines},
    )
    handlers = build_agent_handlers(ctx=ctx, hub=None, timeframes=None)
    out = await handlers["compute_var_95"](
        ToolCall(
            name="compute_var_95",
            args={"symbols": ["BTCUSDT"], "weights": [1.0]},
        )
    )
    # render_result from bayesian_var includes "VaR" string
    assert "VaR" in out or "var" in out.lower() or "%" in out


# ── BaseStrategy.AgentConfig + _run_agentic ─────────────────────


@pytest.mark.asyncio
async def test_run_agentic_multi_turn_loop_records_transcript() -> None:
    """Mock 2 tool calls + 1 submit_decisions. Confirm the transcript
    has both intermediate turns and the strategy's record_decision
    receives a ``tool_transcript`` matching the loop's history."""
    from halal_trader.core.llm.base import BaseLLM
    from halal_trader.core.strategy import AgentConfig, BaseStrategy

    # Mock LLM emits: analyze_pair → query_rag → submit_decisions.
    plan_args = {
        "decisions": [
            {
                "action": "buy",
                "symbol": "BTCUSDT",
                "quantity": 0.001,
                "confidence": 0.7,
                "reasoning": "rsi oversold + RAG analogue",
            }
        ],
        "market_outlook": "constructive",
    }
    turn_outputs = [
        [ToolCall(name="analyze_pair", args={"symbol": "BTCUSDT"})],
        [ToolCall(name="query_rag", args={"query": "rsi 30 bb lower", "k": 3})],
        [ToolCall(name="submit_decisions", args=plan_args)],
    ]
    llm = MagicMock(spec=BaseLLM)
    llm.supports_tool_use = True
    llm.model = "claude-x"
    llm.last_thinking = ""
    llm.last_usage = MagicMock(cost_usd=0)
    llm.generate_tool_call = AsyncMock(side_effect=turn_outputs)

    repo = AsyncMock()
    repo.record_decision = AsyncMock(return_value=1)

    handlers = {
        "analyze_pair": AsyncMock(return_value="BTCUSDT 4h chart: bullish flag at 50k."),
        "query_rag": AsyncMock(return_value="Past analog: closed +0.8%"),
    }

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm
    strat._repo = repo
    strat._llm_provider_name = "anthropic"
    strat._llm_budget = None

    raw, transcript = await strat._run_agentic(
        user_prompt="user prompt",
        system_prompt="system prompt",
        agent=AgentConfig(
            tools=[ANALYZE_PAIR_TOOL, QUERY_RAG_TOOL, SUBMIT_DECISIONS_TOOL],
            handlers=handlers,
            terminal_tool="submit_decisions",
            max_turns=5,
            max_seconds=30.0,
        ),
    )
    assert raw == plan_args
    assert len(transcript) == 2  # two intermediate turns; terminal isn't a turn
    assert transcript[0]["tool_name"] == "analyze_pair"
    assert transcript[1]["tool_name"] == "query_rag"
    # Each turn has its handler result text recorded
    assert "bullish flag" in transcript[0]["result_text"]
    assert "+0.8%" in transcript[1]["result_text"]
    # All three LLM calls fired
    assert llm.generate_tool_call.await_count == 3


@pytest.mark.asyncio
async def test_run_llm_analysis_agent_path_persists_transcript() -> None:
    """End-to-end: the strategy's _run_llm_analysis with agent= takes
    the agentic loop, validates the terminal args through the same
    pipeline as the single-shot path, AND passes the transcript to
    ``repo.record_decision(tool_transcript=...)``."""
    from halal_trader.core.llm.base import BaseLLM
    from halal_trader.core.strategy import AgentConfig, BaseStrategy

    plan_args = {
        "decisions": [],
        "market_outlook": "flat",
        "reasoning": "no edge",
    }
    turn_outputs = [
        [ToolCall(name="analyze_pair", args={"symbol": "BTCUSDT"})],
        [ToolCall(name="submit_decisions", args=plan_args)],
    ]
    llm = MagicMock(spec=BaseLLM)
    llm.supports_tool_use = True
    llm.model = "claude-x"
    llm.last_thinking = ""
    llm.last_usage = MagicMock(cost_usd=0)
    llm.generate_tool_call = AsyncMock(side_effect=turn_outputs)

    repo = AsyncMock()
    repo.record_decision = AsyncMock(return_value=1)

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm
    strat._repo = repo
    strat._llm_provider_name = "anthropic"
    strat._llm_budget = None

    handlers = {"analyze_pair": AsyncMock(return_value="bullish 4h read")}
    plan = await strat._run_llm_analysis(
        "sys",
        "user",
        prompt_summary="x",
        validate=lambda raw: raw,
        make_empty=lambda msg: {"error": msg},
        extract_symbols=lambda p: [],
        count_actions=lambda p: {"decisions": 0},
        log_prefix="Crypto",
        agent=AgentConfig(
            tools=[ANALYZE_PAIR_TOOL, SUBMIT_DECISIONS_TOOL],
            handlers=handlers,
            terminal_tool="submit_decisions",
            max_turns=3,
            max_seconds=10.0,
        ),
    )
    assert plan == plan_args
    # The record_decision call landed with tool_transcript set
    repo.record_decision.assert_awaited()
    call_kwargs = repo.record_decision.await_args.kwargs
    assert call_kwargs.get("tool_transcript") is not None
    assert len(call_kwargs["tool_transcript"]) == 1
    assert call_kwargs["tool_transcript"][0]["tool_name"] == "analyze_pair"


@pytest.mark.asyncio
async def test_run_llm_analysis_no_agent_keeps_single_call_path() -> None:
    """Acceptance bar: ``agent=None`` reverts to the existing tool /
    JSON path with zero behavioural change (no agentic loop runs,
    transcript stays None on the recorded row)."""
    from halal_trader.core.llm.base import BaseLLM
    from halal_trader.core.strategy import BaseStrategy

    llm = MagicMock(spec=BaseLLM)
    llm.supports_tool_use = True
    llm.model = "claude-x"
    llm.last_thinking = ""
    llm.last_usage = MagicMock(cost_usd=0)
    llm.generate_tool_call = AsyncMock(
        return_value=[
            ToolCall(
                name="submit_decisions",
                args={"decisions": [], "market_outlook": "ok"},
            )
        ]
    )

    repo = AsyncMock()
    repo.record_decision = AsyncMock(return_value=1)

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm
    strat._repo = repo
    strat._llm_provider_name = "anthropic"
    strat._llm_budget = None

    await strat._run_llm_analysis(
        "sys",
        "user",
        prompt_summary="x",
        validate=lambda raw: raw,
        make_empty=lambda msg: {"error": msg},
        extract_symbols=lambda p: [],
        count_actions=lambda p: {"decisions": 0},
        tool=SUBMIT_DECISIONS_TOOL,
        agent=None,
    )
    # tool_transcript stays None
    call_kwargs = repo.record_decision.await_args.kwargs
    assert call_kwargs.get("tool_transcript") is None


# ── CryptoTradingStrategy opt-in flag ───────────────────────────


def test_strategy_default_is_not_agentic() -> None:
    """Default-constructed strategy keeps the single-call path."""
    from unittest.mock import MagicMock

    from halal_trader.crypto.strategy import CryptoTradingStrategy

    strat = CryptoTradingStrategy(
        llm=MagicMock(),
        repo=MagicMock(),
        llm_provider_name="x",
        max_position_pct=0.25,
        daily_loss_limit=0.03,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
    )
    assert strat._agentic_enabled is False


def test_strategy_agentic_flag_persists() -> None:
    """When set, the flag is exposed for runtime inspection."""
    from unittest.mock import MagicMock

    from halal_trader.crypto.strategy import CryptoTradingStrategy

    strat = CryptoTradingStrategy(
        llm=MagicMock(),
        repo=MagicMock(),
        llm_provider_name="x",
        max_position_pct=0.25,
        daily_loss_limit=0.03,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
        agentic_enabled=True,
        agentic_max_turns=7,
        agentic_max_seconds=42.0,
    )
    assert strat._agentic_enabled is True
    assert strat._agentic_max_turns == 7
    assert strat._agentic_max_seconds == 42.0


# ── LlmDecisionRepo.record_decision accepts tool_transcript ─────


@pytest.mark.asyncio
async def test_llm_decision_record_accepts_tool_transcript() -> None:
    """Smoke-check the repo signature without hitting the DB —
    confirms the kwarg threads through to the SQLModel constructor."""
    from halal_trader.db.repos.llm_decisions import LlmDecisionRepoImpl

    impl = LlmDecisionRepoImpl(engine=MagicMock())
    # Patch the AsyncSession entry point so we don't open a real DB
    # connection — the test just verifies the call signature accepts
    # the kwarg and constructs an LlmDecision row.
    import halal_trader.db.repos.llm_decisions as mod

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        def add(self, obj):
            self.last = obj

        async def commit(self):
            return None

        async def refresh(self, obj):
            obj.id = 42

    mod_orig = mod.AsyncSession
    mod.AsyncSession = lambda _engine: _FakeSession()  # type: ignore[assignment]
    try:
        out = await impl.record_decision(
            provider="anthropic",
            model="claude-x",
            tool_transcript=[
                {"turn": 1, "tool_name": "analyze_pair", "args": {"symbol": "BTCUSDT"}}
            ],
        )
        assert out == 42
    finally:
        mod.AsyncSession = mod_orig  # type: ignore[assignment]
