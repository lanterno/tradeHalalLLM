"""Tests for the agentic multi-turn tool-calling loop."""

from __future__ import annotations

from halal_trader.core.llm.agent import AgentResult, run_agent
from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.tools import (
    CRYPTO_AGENTIC_TOOLS,
    Tool,
    ToolCall,
)


class _ScriptedLLM(BaseLLM):
    """Test double — emits a pre-baked sequence of ToolCalls."""

    def __init__(self, script: list[ToolCall]) -> None:
        super().__init__(model="scripted")
        self._script = list(script)

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return ""

    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[Tool],
        system: str | None = None,
        force_tool: str | None = None,
    ) -> list[ToolCall]:
        if force_tool:
            return [ToolCall(name=force_tool, args={})]
        if not self._script:
            return [ToolCall(name="submit_plan", args={})]
        return [self._script.pop(0)]


async def test_agent_terminates_on_submit_plan() -> None:
    plan_args = {"market_outlook": "neutral", "buys": [], "sells": []}
    llm = _ScriptedLLM([ToolCall(name="submit_plan", args=plan_args)])

    result = await run_agent(
        llm,
        system="be sharp",
        user="cycle prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={},
    )
    assert isinstance(result, AgentResult)
    assert result.final_call.name == "submit_plan"
    assert result.final_call.args == plan_args
    assert result.transcript == []
    assert result.budget_exhausted is False


async def test_agent_dispatches_intermediate_tools() -> None:
    """analyze_pair → query_rag → submit_plan."""
    plan_args = {"market_outlook": "ok", "buys": [], "sells": []}
    llm = _ScriptedLLM(
        [
            ToolCall(name="analyze_pair", args={"symbol": "BTCUSDT"}),
            ToolCall(name="query_rag", args={"query": "rsi oversold", "k": 3}),
            ToolCall(name="submit_plan", args=plan_args),
        ]
    )

    calls_seen: list[str] = []

    async def analyze(call: ToolCall) -> str:
        calls_seen.append(call.name)
        return "BTCUSDT 4h: RSI 32, MACD bullish"

    async def query_rag(call: ToolCall) -> str:
        calls_seen.append(call.name)
        return "Top hits: BTCUSDT 2026-04-15 +1.2%"

    result = await run_agent(
        llm,
        system="agent test",
        user="cycle prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": analyze, "query_rag": query_rag},
    )
    assert calls_seen == ["analyze_pair", "query_rag"]
    assert len(result.transcript) == 2
    assert result.transcript[0].tool_name == "analyze_pair"
    assert "BTCUSDT 4h" in (result.transcript[0].result_text or "")
    assert result.final_call.name == "submit_plan"
    assert result.budget_exhausted is False


async def test_agent_force_finalises_on_max_turns() -> None:
    """When the model never submits, the loop forces submit_plan."""
    # Six analyze_pair calls — exceeds max_turns=3, then we force.
    llm = _ScriptedLLM([ToolCall(name="analyze_pair", args={"symbol": "X"})] * 6)

    async def analyze(call: ToolCall) -> str:
        return "noop"

    result = await run_agent(
        llm,
        system="",
        user="cycle prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": analyze},
        max_turns=3,
    )
    assert result.budget_exhausted is True
    assert result.final_call.name == "submit_plan"
    assert len(result.transcript) == 3


async def test_agent_records_tool_handler_failure() -> None:
    plan_args = {"market_outlook": "ok", "buys": [], "sells": []}
    llm = _ScriptedLLM(
        [
            ToolCall(name="analyze_pair", args={"symbol": "X"}),
            ToolCall(name="submit_plan", args=plan_args),
        ]
    )

    async def boom(_call: ToolCall) -> str:
        raise RuntimeError("intentional")

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": boom},
    )
    assert "intentional" in result.transcript[0].error
    assert result.final_call.name == "submit_plan"
