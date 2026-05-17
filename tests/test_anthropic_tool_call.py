"""Tests for :meth:`AnthropicLLM.generate_tool_call` — the agentic tool-use path.

`test_anthropic_caching.py` covers the `generate(...)` text path and
the system-prompt cache wiring. This file pins the parallel
`generate_tool_call` path that the Wave H agentic loop relies on:
the kwargs assembly (tools serialised via ``for_anthropic``,
``tool_choice`` shape when forced), the ``content`` walker that turns
``tool_use`` blocks into ``ToolCall`` dataclasses, and the usage
accounting (which mirrors ``generate``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from halal_trader.core.llm.anthropic import AnthropicLLM
from halal_trader.core.llm.tools import Tool


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Block:
    """Mirrors the SDK's content-block surface (text or tool_use)."""

    type: str
    text: str = ""
    name: str = ""
    input: dict[str, Any] | None = None
    id: str | None = None


@dataclass
class _Response:
    content: list[_Block]
    usage: _Usage | None = None


class _Messages:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _Response:
        self.last_kwargs = kwargs
        return self.response


class _StubClient:
    def __init__(self, response: _Response) -> None:
        self.messages = _Messages(response)


def _tool(name: str = "submit_plan") -> Tool:
    return Tool(
        name=name,
        description="x",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _llm(response: _Response, *, cache: bool = True) -> AnthropicLLM:
    llm = AnthropicLLM(model="claude-opus-4-7", api_key="x", enable_prompt_cache=cache)
    llm._client = _StubClient(response)
    return llm


# ── kwargs assembly ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_tools_serialised_via_for_anthropic():
    """Each Tool is projected via ``for_anthropic()`` before being sent —
    the SDK expects ``{name, description, input_schema}`` keys, not the
    OpenAI-flavour ``{type:'function', function:{...}}`` envelope."""
    response = _Response(content=[], usage=None)
    llm = _llm(response)
    await llm.generate_tool_call("hello", tools=[_tool("submit_plan")])

    sent_tools = llm._client.messages.last_kwargs["tools"]
    assert isinstance(sent_tools, list)
    assert sent_tools[0]["name"] == "submit_plan"
    assert "description" in sent_tools[0]
    assert sent_tools[0]["input_schema"]["type"] == "object"
    # Anthropic shape — no nested 'function' envelope.
    assert "function" not in sent_tools[0]


@pytest.mark.asyncio
async def test_force_tool_sets_tool_choice():
    """``force_tool="submit_plan"`` becomes ``tool_choice={'type':'tool','name':...}``
    — locks the SDK shape in case Anthropic ever changes it."""
    response = _Response(content=[], usage=None)
    llm = _llm(response)
    await llm.generate_tool_call("hello", tools=[_tool("submit_plan")], force_tool="submit_plan")
    sent = llm._client.messages.last_kwargs
    assert sent["tool_choice"] == {"type": "tool", "name": "submit_plan"}


@pytest.mark.asyncio
async def test_no_force_tool_omits_tool_choice():
    """Without ``force_tool``, ``tool_choice`` must NOT be sent — let
    the model decide whether to call. Sending an empty/None tool_choice
    would 400 with newer SDKs."""
    response = _Response(content=[], usage=None)
    llm = _llm(response)
    await llm.generate_tool_call("hello", tools=[_tool()])
    sent = llm._client.messages.last_kwargs
    assert "tool_choice" not in sent


@pytest.mark.asyncio
async def test_system_payload_uses_caching_when_enabled():
    """The same caching wiring as `generate()` must flow through to
    the tool-call path — Wave H agentic cycles re-use the static
    system prompt across many sub-calls."""
    response = _Response(content=[], usage=None)
    llm = _llm(response, cache=True)
    await llm.generate_tool_call("hello", tools=[_tool()], system="big static")
    sent = llm._client.messages.last_kwargs
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_system_payload_plain_when_caching_disabled():
    response = _Response(content=[], usage=None)
    llm = _llm(response, cache=False)
    await llm.generate_tool_call("hello", tools=[_tool()], system="big static")
    assert llm._client.messages.last_kwargs["system"] == "big static"


@pytest.mark.asyncio
async def test_user_prompt_is_single_message():
    """The model's input is a single user-role message — multi-turn
    history is the agentic loop's job, not this method's."""
    response = _Response(content=[], usage=None)
    llm = _llm(response)
    await llm.generate_tool_call("user prompt here", tools=[_tool()])
    msgs = llm._client.messages.last_kwargs["messages"]
    assert msgs == [{"role": "user", "content": "user prompt here"}]


# ── content → ToolCall conversion ───────────────────────────


@pytest.mark.asyncio
async def test_single_tool_use_block_yields_one_call():
    response = _Response(
        content=[
            _Block(
                type="tool_use",
                name="submit_plan",
                input={"market_outlook": "ok", "buys": [], "sells": []},
                id="tu_001",
            )
        ],
        usage=None,
    )
    llm = _llm(response)
    calls = await llm.generate_tool_call("hi", tools=[_tool()])

    assert len(calls) == 1
    call = calls[0]
    assert call.name == "submit_plan"
    assert call.args == {"market_outlook": "ok", "buys": [], "sells": []}
    assert call.id == "tu_001"


@pytest.mark.asyncio
async def test_multiple_tool_use_blocks_yield_multiple_calls():
    """The agentic loop expects all tool blocks in a single response —
    e.g. analyze_pair followed by submit_plan in the same turn."""
    response = _Response(
        content=[
            _Block(type="tool_use", name="analyze_pair", input={"symbol": "BTCUSDT"}, id="t1"),
            _Block(type="tool_use", name="submit_plan", input={"buys": []}, id="t2"),
        ],
    )
    llm = _llm(response)
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert [c.name for c in calls] == ["analyze_pair", "submit_plan"]
    assert [c.id for c in calls] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_text_blocks_are_ignored():
    """Anthropic interleaves text + tool_use blocks. Only tool_use
    rows become ``ToolCall``s — the prose is dropped."""
    response = _Response(
        content=[
            _Block(type="text", text="Here is my plan:"),
            _Block(type="tool_use", name="submit_plan", input={"buys": []}, id="t1"),
            _Block(type="text", text="Done."),
        ],
    )
    llm = _llm(response)
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert len(calls) == 1
    assert calls[0].name == "submit_plan"


@pytest.mark.asyncio
async def test_no_tool_use_blocks_returns_empty_list():
    """A response with only text blocks (model decided not to call
    a tool) returns ``[]`` — caller decides what to do."""
    response = _Response(content=[_Block(type="text", text="I won't call anything.")])
    llm = _llm(response)
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert calls == []


@pytest.mark.asyncio
async def test_none_input_becomes_empty_dict():
    """Defensive: an SDK quirk where ``input=None`` lands instead of
    ``{}`` shouldn't blow up downstream consumers — they get ``{}``."""
    response = _Response(
        content=[_Block(type="tool_use", name="t", input=None, id="x")],
    )
    llm = _llm(response)
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert calls[0].args == {}


@pytest.mark.asyncio
async def test_args_is_a_copy_not_aliased():
    """Mutating ``call.args`` must not mutate the SDK's response —
    upstream code may want to feed args back in next turn."""
    inner = {"buys": [{"symbol": "BTC"}]}
    response = _Response(
        content=[_Block(type="tool_use", name="t", input=inner, id="x")],
    )
    llm = _llm(response)
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    calls[0].args["buys"] = []
    assert inner["buys"] == [{"symbol": "BTC"}]  # original untouched


@pytest.mark.asyncio
async def test_missing_id_attribute_yields_none_id():
    """Older Anthropic responses or test stubs may omit the ``id``
    attribute on the block — ``getattr(..., 'id', None)`` keeps the
    contract: ToolCall.id stays None rather than raising."""

    @dataclass
    class _BlockNoId:
        type: str
        name: str = ""
        input: dict[str, Any] | None = field(default_factory=dict)

    response = _Response(content=[_BlockNoId(type="tool_use", name="t", input={})])
    llm = _llm(response)
    calls = await llm.generate_tool_call("hi", tools=[_tool()])
    assert calls[0].id is None


# ── usage accounting ────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_flows_into_last_usage():
    """The tool-call path must update ``last_usage`` the same way
    ``generate()`` does — input/output/cache tokens + cost."""
    response = _Response(
        content=[_Block(type="tool_use", name="t", input={}, id="x")],
        usage=_Usage(
            input_tokens=200,
            output_tokens=50,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=0,
        ),
    )
    llm = _llm(response)
    await llm.generate_tool_call("hi", tools=[_tool()])

    u = llm.last_usage
    assert u.input_tokens == 200
    assert u.output_tokens == 50
    assert u.cache_read_tokens == 1000
    assert u.cache_write_tokens == 0
    # Cost > 0 (don't pin the exact number — that's covered in
    # test_anthropic_caching; here we just want it computed).
    assert u.cost_usd > 0


@pytest.mark.asyncio
async def test_no_usage_field_does_not_crash():
    """Mirror the ``generate()`` defensive — older SDKs / test fixtures
    may return no usage block. last_usage should still be populated
    with default zeros, not raise."""
    response = _Response(
        content=[_Block(type="tool_use", name="t", input={}, id="x")],
        usage=None,
    )
    llm = _llm(response)
    await llm.generate_tool_call("hi", tools=[_tool()])
    assert llm.last_usage.input_tokens == 0
    assert llm.last_usage.cost_usd == 0
