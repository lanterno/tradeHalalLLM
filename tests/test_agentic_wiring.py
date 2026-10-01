"""Wave H wiring tests — agentic loop + tool handlers + transcript persistence.

``core/llm/agent.py:run_agent`` (the bounded multi-turn driver) is
already covered by ``tests/test_llm_agent_budget.py``. This file
covers the *consumer wiring* added in this commit:

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
    QUERY_RAG_TOOL,
    QUERY_REGIME_MEMORY_TOOL,
    SUBMIT_DECISIONS_TOOL,
    ToolCall,
)

# ── BaseStrategy.AgentConfig + _run_agentic ─────────────────────


@pytest.mark.asyncio
async def test_run_agentic_multi_turn_loop_records_transcript() -> None:
    """Mock 2 tool calls + 1 submit_decisions. Confirm the transcript
    has both intermediate turns and the strategy's record_decision
    receives a ``tool_transcript`` matching the loop's history."""
    from halal_trader.core.llm.base import BaseLLM
    from halal_trader.core.strategy import AgentConfig, BaseStrategy

    # Mock LLM emits: query_regime_memory → query_rag → submit_decisions.
    plan_args = {
        "decisions": [
            {
                "action": "buy",
                "symbol": "AAPL",
                "quantity": 1,
                "confidence": 0.7,
                "reasoning": "rsi oversold + RAG analogue",
            }
        ],
        "market_outlook": "constructive",
    }
    turn_outputs = [
        [ToolCall(name="query_regime_memory", args={"k": 3})],
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
        "query_regime_memory": AsyncMock(
            return_value="2024-03-12 analogue: bullish flag, +1.2% next day."
        ),
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
            tools=[QUERY_REGIME_MEMORY_TOOL, QUERY_RAG_TOOL, SUBMIT_DECISIONS_TOOL],
            handlers=handlers,
            terminal_tool="submit_decisions",
            max_turns=5,
            max_seconds=30.0,
        ),
    )
    assert raw == plan_args
    assert len(transcript) == 2  # two intermediate turns; terminal isn't a turn
    assert transcript[0]["tool_name"] == "query_regime_memory"
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
        [ToolCall(name="query_regime_memory", args={"k": 3})],
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

    handlers = {"query_regime_memory": AsyncMock(return_value="bullish analogue")}
    plan = await strat._run_llm_analysis(
        "sys",
        "user",
        prompt_summary="x",
        validate=lambda raw: raw,
        make_empty=lambda msg: {"error": msg},
        extract_symbols=lambda p: [],
        count_actions=lambda p: {"decisions": 0},
        log_prefix="Stocks",
        agent=AgentConfig(
            tools=[QUERY_REGIME_MEMORY_TOOL, SUBMIT_DECISIONS_TOOL],
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
    assert call_kwargs["tool_transcript"][0]["tool_name"] == "query_regime_memory"


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
            tool_transcript=[{"turn": 1, "tool_name": "query_regime_memory", "args": {"k": 3}}],
        )
        assert out == 42
    finally:
        mod.AsyncSession = mod_orig  # type: ignore[assignment]
