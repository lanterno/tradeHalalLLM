"""Wave E wiring tests — tool-use across providers + strategy integration.

The provider-native ``generate_tool_call`` implementations (Anthropic,
OpenAI) are covered by existing tests in test_anthropic_tool_call.py;
this file covers the *wiring*: capability flags, FallbackLLM
delegation, the new SUBMIT_DECISIONS_TOOL schema, and the
``BaseStrategy._run_llm_analysis(tool=...)`` branch selection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.fallback import FallbackLLM
from halal_trader.core.llm.tools import (
    SUBMIT_DECISIONS_TOOL,
    SUBMIT_PLAN_TOOL,
    Tool,
    ToolCall,
)

# ── supports_tool_use capability flags ───────────────────────────


def test_anthropic_advertises_tool_use_support() -> None:
    """AnthropicLLM ships native tool-use via the Messages API."""
    from halal_trader.core.llm.anthropic import AnthropicLLM

    assert AnthropicLLM.supports_tool_use is True


def test_openai_advertises_tool_use_support() -> None:
    """OpenAILLM ships native tool-use via chat.completions."""
    from halal_trader.core.llm.openai import OpenAILLM

    assert OpenAILLM.supports_tool_use is True


def test_ollama_does_not_advertise_tool_use() -> None:
    """Ollama lacks native tool-use → falls back to generate_json."""
    from halal_trader.core.llm.ollama import OllamaLLM

    assert OllamaLLM.supports_tool_use is False


def test_base_default_is_false() -> None:
    """A new provider subclass that doesn't opt in stays safe-by-default."""
    assert BaseLLM.supports_tool_use is False


# ── SUBMIT_DECISIONS_TOOL schema shape ──────────────────────────


def test_submit_decisions_tool_required_fields() -> None:
    """The strategy assumes ``decisions`` and ``market_outlook`` are
    always present — pin so a schema refactor doesn't quietly drop them."""
    schema = SUBMIT_DECISIONS_TOOL.input_schema
    assert "decisions" in schema["required"]
    assert "market_outlook" in schema["required"]
    decision = schema["properties"]["decisions"]["items"]
    assert "action" in decision["required"]
    assert "symbol" in decision["required"]
    assert "quantity" in decision["required"]
    assert decision["properties"]["action"]["enum"] == ["buy", "sell", "hold"]


def test_submit_decisions_tool_projects_for_anthropic() -> None:
    """The provider-native helper produces the Anthropic-shaped dict."""
    payload = SUBMIT_DECISIONS_TOOL.for_anthropic()
    assert payload["name"] == "submit_decisions"
    assert "input_schema" in payload
    assert payload["input_schema"]["properties"]["decisions"]["type"] == "array"


def test_submit_decisions_tool_projects_for_openai() -> None:
    """OpenAI wraps the same schema under ``function.parameters``."""
    payload = SUBMIT_DECISIONS_TOOL.for_openai()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "submit_decisions"
    assert payload["function"]["parameters"]["properties"]["decisions"]["type"] == "array"


def test_submit_plan_tool_still_exists_for_agentic_surface() -> None:
    """The richer SUBMIT_PLAN_TOOL stays around for Wave H."""
    assert SUBMIT_PLAN_TOOL.name == "submit_plan"


# ── FallbackLLM.generate_tool_call delegation ───────────────────


@pytest.mark.asyncio
async def test_fallback_tool_call_uses_primary_when_supported() -> None:
    """Primary supports tool-use → fallback delegates directly to it."""
    primary = MagicMock(spec=BaseLLM)
    primary.supports_tool_use = True
    primary.model = "claude-x"
    primary.last_thinking = ""
    primary.last_usage = MagicMock(cost_usd=0)
    primary.generate_tool_call = AsyncMock(
        return_value=[ToolCall(name="submit_decisions", args={"decisions": []}, id="a")]
    )

    secondary = MagicMock(spec=BaseLLM)
    secondary.supports_tool_use = True
    secondary.generate_tool_call = AsyncMock(side_effect=AssertionError("should not be called"))

    fb = FallbackLLM(primary=primary, fallbacks=[secondary])
    calls = await fb.generate_tool_call(
        "prompt", tools=[SUBMIT_DECISIONS_TOOL], system="sys", force_tool="submit_decisions"
    )
    assert calls and calls[0].name == "submit_decisions"
    primary.generate_tool_call.assert_awaited_once()
    secondary.generate_tool_call.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_tool_call_falls_through_on_primary_failure() -> None:
    """Primary raises → fallback walks the chain to the next provider."""
    primary = MagicMock(spec=BaseLLM)
    primary.supports_tool_use = True
    primary.model = "primary-x"
    primary.last_thinking = ""
    primary.last_usage = MagicMock(cost_usd=0)
    primary.generate_tool_call = AsyncMock(side_effect=RuntimeError("primary 503"))

    secondary = MagicMock(spec=BaseLLM)
    secondary.supports_tool_use = True
    secondary.model = "secondary-x"
    secondary.last_thinking = ""
    secondary.last_usage = MagicMock(cost_usd=0)
    secondary.generate_tool_call = AsyncMock(
        return_value=[ToolCall(name="submit_decisions", args={"decisions": []})]
    )

    fb = FallbackLLM(primary=primary, fallbacks=[secondary])
    calls = await fb.generate_tool_call(
        "prompt", tools=[SUBMIT_DECISIONS_TOOL], force_tool="submit_decisions"
    )
    assert calls and calls[0].name == "submit_decisions"
    primary.generate_tool_call.assert_awaited_once()
    secondary.generate_tool_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_fallback_tool_call_raises_when_chain_exhausted() -> None:
    """Every provider fails → re-raise the last error so the strategy's
    empty-plan path fires (rather than silently returning [])."""
    primary = MagicMock(spec=BaseLLM)
    primary.supports_tool_use = True
    primary.model = "primary-x"
    primary.last_thinking = ""
    primary.last_usage = MagicMock(cost_usd=0)
    primary.generate_tool_call = AsyncMock(side_effect=RuntimeError("primary 503"))

    fb = FallbackLLM(primary=primary, fallbacks=[])
    with pytest.raises(RuntimeError, match="primary 503"):
        await fb.generate_tool_call("prompt", tools=[SUBMIT_DECISIONS_TOOL])


def test_fallback_supports_tool_use_property_reflects_eligible_providers() -> None:
    """The property surfaces any-of-eligible support so the strategy
    decides whether to ask for a tool call at all."""
    inner_supports = MagicMock()
    inner_supports.supports_tool_use = True
    inner_supports.model = "supports-x"
    inner_no = MagicMock()
    inner_no.supports_tool_use = False
    inner_no.model = "nosup-x"

    fb_a = FallbackLLM(primary=inner_supports, fallbacks=[])
    assert fb_a.supports_tool_use is True

    fb_b = FallbackLLM(primary=inner_no, fallbacks=[inner_supports])
    assert fb_b.supports_tool_use is True

    fb_c = FallbackLLM(primary=inner_no, fallbacks=[])
    assert fb_c.supports_tool_use is False


# ── BaseStrategy tool-use branch selection ──────────────────────


@pytest.mark.asyncio
async def test_strategy_takes_tool_path_when_llm_supports_and_tool_given() -> None:
    """LLM with ``supports_tool_use=True`` + a tool arg → tool-call path."""
    from halal_trader.core.strategy import BaseStrategy

    repo = AsyncMock()
    llm = MagicMock(spec=BaseLLM)
    llm.supports_tool_use = True
    llm.model = "claude-x"
    llm.last_thinking = ""
    llm.last_usage = MagicMock(cost_usd=0)
    llm.generate_tool_call = AsyncMock(
        return_value=[
            ToolCall(
                name="submit_decisions",
                args={"decisions": [], "market_outlook": "flat", "reasoning": "hold"},
            )
        ]
    )

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm
    strat._repo = repo
    strat._llm_provider_name = "test"
    strat._llm_budget = None

    plan = await strat._run_llm_analysis(
        "sys",
        "user",
        prompt_summary="x",
        validate=lambda raw: raw,
        make_empty=lambda msg: {"error": msg},
        extract_symbols=lambda p: [],
        count_actions=lambda p: {"decisions": 0},
        tool=SUBMIT_DECISIONS_TOOL,
    )
    llm.generate_tool_call.assert_awaited_once()
    assert getattr(llm, "generate_json", MagicMock()).call_count == 0
    assert plan["market_outlook"] == "flat"


@pytest.mark.asyncio
async def test_strategy_falls_back_to_json_when_tool_use_not_supported() -> None:
    """Ollama-shape LLM (supports_tool_use=False) → generate_json path,
    even when a tool is passed."""
    from halal_trader.core.strategy import BaseStrategy

    repo = AsyncMock()
    llm = MagicMock(spec=BaseLLM)
    llm.supports_tool_use = False
    llm.model = "llama-x"
    llm.last_thinking = ""
    llm.last_usage = MagicMock(cost_usd=0)
    llm.generate_json = AsyncMock(
        return_value={"decisions": [], "market_outlook": "flat", "reasoning": "hold"}
    )
    # Ensure the tool-call path is NOT taken.
    llm.generate_tool_call = AsyncMock(side_effect=AssertionError("should not be called"))

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm
    strat._repo = repo
    strat._llm_provider_name = "test"
    strat._llm_budget = None

    plan = await strat._run_llm_analysis(
        "sys",
        "user",
        prompt_summary="x",
        validate=lambda raw: raw,
        make_empty=lambda msg: {"error": msg},
        extract_symbols=lambda p: [],
        count_actions=lambda p: {"decisions": 0},
        tool=SUBMIT_DECISIONS_TOOL,
    )
    llm.generate_json.assert_awaited_once()
    llm.generate_tool_call.assert_not_called()
    assert plan["market_outlook"] == "flat"


@pytest.mark.asyncio
async def test_strategy_falls_back_to_json_when_no_tool_passed() -> None:
    """A caller that doesn't pass a tool kwarg keeps the legacy path."""
    from halal_trader.core.strategy import BaseStrategy

    repo = AsyncMock()
    llm = MagicMock(spec=BaseLLM)
    llm.supports_tool_use = True  # even though support is there, no tool → JSON
    llm.model = "claude-x"
    llm.last_thinking = ""
    llm.last_usage = MagicMock(cost_usd=0)
    llm.generate_json = AsyncMock(
        return_value={"decisions": [], "market_outlook": "ok", "reasoning": "h"}
    )
    llm.generate_tool_call = AsyncMock(side_effect=AssertionError("should not be called"))

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm
    strat._repo = repo
    strat._llm_provider_name = "test"
    strat._llm_budget = None

    await strat._run_llm_analysis(
        "sys",
        "user",
        prompt_summary="x",
        validate=lambda raw: raw,
        make_empty=lambda msg: {"error": msg},
        extract_symbols=lambda p: [],
        count_actions=lambda p: {"decisions": 0},
        tool=None,
    )
    llm.generate_json.assert_awaited_once()
    llm.generate_tool_call.assert_not_called()


@pytest.mark.asyncio
async def test_strategy_call_tool_picks_matching_name_when_multiple_calls() -> None:
    """If the model returns multiple tool calls, prefer the one whose
    name matches the tool we forced."""
    from halal_trader.core.strategy import BaseStrategy

    llm = MagicMock(spec=BaseLLM)
    llm.generate_tool_call = AsyncMock(
        return_value=[
            ToolCall(name="other_tool", args={"x": 1}),
            ToolCall(name="submit_decisions", args={"decisions": [], "market_outlook": "ok"}),
        ]
    )

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm

    args = await strat._call_tool(
        user_prompt="u",
        system_prompt="s",
        tool=SUBMIT_DECISIONS_TOOL,
    )
    assert args["market_outlook"] == "ok"


@pytest.mark.asyncio
async def test_strategy_call_tool_raises_when_no_calls_returned() -> None:
    """An empty tool-call list is a contract violation — propagate so
    the strategy's empty-plan path records the failure."""
    from halal_trader.core.strategy import BaseStrategy

    llm = MagicMock(spec=BaseLLM)
    llm.generate_tool_call = AsyncMock(return_value=[])

    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm

    with pytest.raises(ValueError, match="no tool calls"):
        await strat._call_tool(
            user_prompt="u",
            system_prompt="s",
            tool=Tool(name="x", description="d", input_schema={}),
        )
