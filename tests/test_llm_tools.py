"""Tests for the typed LLM tool definitions."""

from __future__ import annotations

from halal_trader.core.llm.tools import (
    QUERY_RAG_TOOL,
    QUERY_REGIME_MEMORY_TOOL,
    SUBMIT_DECISIONS_TOOL,
    Tool,
    ToolCall,
)


def test_submit_decisions_schema_includes_required_fields() -> None:
    schema = SUBMIT_DECISIONS_TOOL.input_schema
    assert "decisions" in schema["properties"]
    assert schema["properties"]["decisions"]["type"] == "array"


def test_openai_projection_wraps_in_function_envelope() -> None:
    payload = SUBMIT_DECISIONS_TOOL.for_openai()
    assert payload["type"] == "function"
    fn = payload["function"]
    assert fn["name"] == "submit_decisions"
    assert fn["parameters"]["type"] == "object"


def test_agentic_helper_tools_have_stable_names() -> None:
    assert QUERY_RAG_TOOL.name == "query_rag"
    assert QUERY_REGIME_MEMORY_TOOL.name == "query_regime_memory"


def test_tool_call_dataclass_round_trips() -> None:
    call = ToolCall(name="submit_decisions", args={"decisions": [], "market_outlook": "ok"})
    assert call.name == "submit_decisions"
    assert call.args["market_outlook"] == "ok"
    assert call.id is None


def test_tool_can_be_serialised_for_openai() -> None:
    tool = Tool(name="t", description="d", input_schema={"type": "object", "properties": {}})
    o = tool.for_openai()
    assert o["function"]["name"] == "t"
