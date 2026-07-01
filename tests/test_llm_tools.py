"""Tests for the typed LLM tool definitions."""

from __future__ import annotations

from halal_trader.core.llm.tools import (
    ANALYZE_PAIR_TOOL,
    CRYPTO_AGENTIC_TOOLS,
    CRYPTO_STRATEGY_TOOLS,
    QUERY_RAG_TOOL,
    SUBMIT_PLAN_TOOL,
    Tool,
    ToolCall,
)


def test_submit_plan_schema_includes_required_fields() -> None:
    schema = SUBMIT_PLAN_TOOL.input_schema
    assert "market_outlook" in schema["properties"]
    assert "buys" in schema["properties"]
    assert "sells" in schema["properties"]
    assert "market_outlook" in schema["required"]


def test_openai_projection_wraps_in_function_envelope() -> None:
    payload = SUBMIT_PLAN_TOOL.for_openai()
    assert payload["type"] == "function"
    fn = payload["function"]
    assert fn["name"] == "submit_plan"
    assert fn["parameters"]["type"] == "object"


def test_strategy_tools_collection_contains_submit_plan() -> None:
    assert SUBMIT_PLAN_TOOL in CRYPTO_STRATEGY_TOOLS


def test_agentic_tools_include_helpers_then_submit_plan() -> None:
    names = [t.name for t in CRYPTO_AGENTIC_TOOLS]
    assert names[-1] == "submit_plan"  # terminal tool
    assert ANALYZE_PAIR_TOOL.name in names
    assert QUERY_RAG_TOOL.name in names
    assert "compute_var_95" in names


def test_tool_call_dataclass_round_trips() -> None:
    call = ToolCall(name="submit_plan", args={"market_outlook": "ok", "buys": [], "sells": []})
    assert call.name == "submit_plan"
    assert call.args["market_outlook"] == "ok"
    assert call.id is None


def test_tool_can_be_serialised_for_openai() -> None:
    tool = Tool(name="t", description="d", input_schema={"type": "object", "properties": {}})
    o = tool.for_openai()
    assert o["function"]["name"] == "t"
