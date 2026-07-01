"""Tests for :meth:`GLMLLM.generate_tool_call` — the agentic tool-use path.

`test_glm_usage.py` covers the `generate(...)` text path and the
endpoint dialects. This file pins the parallel `generate_tool_call`
path that the Wave H agentic loop relies on: the kwargs assembly
(tools serialised via ``for_openai``, ``tool_choice`` shape when
forced), the tool_calls walker that turns SDK entries into
``ToolCall`` dataclasses (including malformed-arguments recovery),
and the usage accounting (which mirrors ``generate``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from halal_trader.core.llm.glm import GLMLLM
from halal_trader.core.llm.tools import Tool


def _tool(name: str = "submit_plan") -> Tool:
    return Tool(
        name=name,
        description="x",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _tc(name: str, arguments: str | None, *, id: str | None = "tc_001") -> SimpleNamespace:
    """One SDK tool_calls entry (function envelope + call id)."""
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=arguments))


def _stub_response(
    tool_calls: list[SimpleNamespace] | None = None,
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Build the SimpleNamespace tree the OpenAI-compat SDK emits for a
    tool-use completion."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))],
    )


def _wire(llm: GLMLLM, response: SimpleNamespace) -> AsyncMock:
    """Inject a fake AsyncOpenAI client whose chat.completions.create
    returns the response we want."""
    create_mock = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    llm._client = client
    return create_mock


def _llm() -> GLMLLM:
    return GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")


# ── kwargs assembly ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_tools_serialised_via_for_openai():
    """Each Tool is projected via ``for_openai()`` before being sent —
    every GLM endpoint speaks the OpenAI-flavour
    ``{type:'function', function:{...}}`` envelope."""
    llm = _llm()
    create_mock = _wire(llm, _stub_response([]))
    await llm.generate_tool_call("hello", tools=[_tool("submit_plan")])

    sent_tools = create_mock.await_args.kwargs["tools"]
    assert isinstance(sent_tools, list)
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["function"]["name"] == "submit_plan"
    assert sent_tools[0]["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_force_tool_sets_tool_choice():
    """``force_tool="submit_plan"`` becomes the OpenAI-shaped
    ``tool_choice={'type':'function','function':{'name':...}}`` —
    locks the wire shape in case the SDK contract shifts."""
    llm = _llm()
    create_mock = _wire(llm, _stub_response([]))
    await llm.generate_tool_call("hello", tools=[_tool("submit_plan")], force_tool="submit_plan")
    sent = create_mock.await_args.kwargs
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "submit_plan"}}


@pytest.mark.asyncio
async def test_no_force_tool_omits_tool_choice():
    """Without ``force_tool``, ``tool_choice`` must NOT be sent — let
    the model decide whether to call. An explicit None would 400 on
    strict hosts."""
    llm = _llm()
    create_mock = _wire(llm, _stub_response([]))
    await llm.generate_tool_call("hello", tools=[_tool()])
    assert "tool_choice" not in create_mock.await_args.kwargs


@pytest.mark.asyncio
async def test_extra_body_and_temperature_flow_through():
    """The endpoint dialect (extra_body) and temperature apply to the
    tool-call path exactly like ``generate()`` — a host that drops
    them can silently break forced tool_choice."""
    llm = _llm()
    create_mock = _wire(llm, _stub_response([]))
    await llm.generate_tool_call("hello", tools=[_tool()])
    kwargs = create_mock.await_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["extra_body"]["reasoning"] == {"enabled": False}
    assert kwargs["extra_body"]["provider"] == {"require_parameters": True}


@pytest.mark.asyncio
async def test_system_prompt_emits_system_role_message():
    llm = _llm()
    create_mock = _wire(llm, _stub_response([]))
    await llm.generate_tool_call("user prompt", tools=[_tool()], system="sys msg")
    messages = create_mock.await_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys msg"}
    assert messages[1] == {"role": "user", "content": "user prompt"}


@pytest.mark.asyncio
async def test_user_prompt_is_single_message_without_system():
    """The model's input is a single user-role message — multi-turn
    history is the agentic loop's job, not this method's."""
    llm = _llm()
    create_mock = _wire(llm, _stub_response([]))
    await llm.generate_tool_call("user prompt here", tools=[_tool()])
    msgs = create_mock.await_args.kwargs["messages"]
    assert msgs == [{"role": "user", "content": "user prompt here"}]


# ── tool_calls → ToolCall conversion ────────────────────────


@pytest.mark.asyncio
async def test_single_tool_call_yields_one_call():
    llm = _llm()
    _wire(
        llm,
        _stub_response(
            [
                _tc(
                    "submit_plan",
                    '{"market_outlook": "ok", "buys": [], "sells": []}',
                    id="tc_001",
                )
            ]
        ),
    )
    calls = await llm.generate_tool_call("hi", tools=[_tool()])

    assert len(calls) == 1
    call = calls[0]
    assert call.name == "submit_plan"
    assert call.args == {"market_outlook": "ok", "buys": [], "sells": []}
    assert call.id == "tc_001"


@pytest.mark.asyncio
async def test_multiple_tool_calls_all_returned():
    """The agentic loop expects all tool calls in a single response —
    e.g. analyze_pair followed by submit_plan in the same turn."""
    llm = _llm()
    _wire(
        llm,
        _stub_response(
            [
                _tc("analyze_pair", '{"symbol": "BTCUSDT"}', id="t1"),
                _tc("submit_plan", '{"buys": []}', id="t2"),
            ]
        ),
    )
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert [c.name for c in calls] == ["analyze_pair", "submit_plan"]
    assert [c.id for c in calls] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_no_tool_calls_returns_empty_list():
    """A response without tool calls (model answered in prose, or a
    host mishandled forced tool_choice) returns ``[]`` — the strategy
    layer treats that as a failed call (no-action plan), never a crash."""
    llm = _llm()
    _wire(llm, _stub_response(None))
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert calls == []


@pytest.mark.asyncio
async def test_arguments_json_string_parsed_into_args_dict():
    """OpenAI-compat endpoints deliver arguments as a JSON *string* —
    the provider parses it so downstream consumers get a dict."""
    llm = _llm()
    _wire(llm, _stub_response([_tc("submit_plan", '{"decisions": [{"symbol": "BTCUSDT"}]}')]))
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert calls[0].args == {"decisions": [{"symbol": "BTCUSDT"}]}


@pytest.mark.asyncio
async def test_malformed_arguments_json_becomes_empty_dict():
    """Defensive: a host that emits truncated/invalid JSON in
    ``function.arguments`` yields ``args={}`` rather than raising —
    the caller's schema validation then rejects the empty plan."""
    llm = _llm()
    _wire(llm, _stub_response([_tc("submit_plan", '{"buys": [unterminated')]))
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert len(calls) == 1
    assert calls[0].args == {}


@pytest.mark.asyncio
async def test_none_arguments_becomes_empty_dict():
    """``arguments=None`` (SDK quirk) also lands as ``{}``."""
    llm = _llm()
    _wire(llm, _stub_response([_tc("submit_plan", None)]))
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert calls[0].args == {}


# ── usage accounting ────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_flows_into_last_usage():
    """The tool-call path must update ``last_usage`` the same way
    ``generate()`` does — provider label, tokens, cache subtraction
    and cost."""
    llm = _llm()
    _wire(
        llm,
        _stub_response(
            [_tc("submit_plan", "{}")],
            prompt_tokens=200,
            completion_tokens=50,
            cached_tokens=120,
        ),
    )
    await llm.generate_tool_call("hi", tools=[_tool()])

    u = llm.last_usage
    assert u.provider == "glm"
    assert u.input_tokens == 80  # 200 - 120 cached
    assert u.output_tokens == 50
    assert u.cache_read_tokens == 120
    # Cost > 0 (exact numbers are pinned in the pricing tests).
    assert u.cost_usd > 0


@pytest.mark.asyncio
async def test_no_usage_field_does_not_crash():
    """Mirror the ``generate()`` defensive — some hosts / test fixtures
    return no usage block. last_usage should still be populated with
    default zeros, not raise."""
    llm = _llm()
    response = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[_tc("t", "{}")]))
        ],
    )
    _wire(llm, response)
    await llm.generate_tool_call("hi", tools=[_tool()])
    assert llm.last_usage.input_tokens == 0
    assert llm.last_usage.cost_usd == 0
