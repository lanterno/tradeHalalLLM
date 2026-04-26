"""Anthropic prompt-cache wiring tests.

We don't hit the real API in tests — instead we replace the SDK client
with a stub that records the kwargs and returns a scripted response. The
contract under test is: when ``enable_prompt_cache=True`` (the default),
the system prompt is sent as a structured ``{type, text, cache_control}``
block, and ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
flow through to ``CallUsage.cache_read_tokens`` / ``cache_write_tokens``
and into the cost calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from halal_trader.core.llm.anthropic import AnthropicLLM


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Block:
    text: str


@dataclass
class _Response:
    content: list[_Block]
    usage: _Usage | None = None


class _Messages:
    """Captures the kwargs of the create() call for inspection."""

    def __init__(self, response: _Response) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _Response:
        self.last_kwargs = kwargs
        return self.response


class _StubClient:
    def __init__(self, response: _Response) -> None:
        self.messages = _Messages(response)


@pytest.fixture
def stub_response_with_cache_hit() -> _Response:
    return _Response(
        content=[_Block(text='{"action": "buy"}')],
        usage=_Usage(
            input_tokens=200,
            output_tokens=50,
            cache_read_input_tokens=1000,  # cache hit
            cache_creation_input_tokens=0,
        ),
    )


@pytest.fixture
def stub_response_with_cache_miss() -> _Response:
    return _Response(
        content=[_Block(text='{"action": "buy"}')],
        usage=_Usage(
            input_tokens=200,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=1000,  # cache write — first call
        ),
    )


async def test_caching_enabled_sends_structured_system_payload(
    stub_response_with_cache_hit,
):
    llm = AnthropicLLM(model="claude-opus-4-7", api_key="x", enable_prompt_cache=True)
    llm._client = _StubClient(stub_response_with_cache_hit)

    await llm.generate("user msg", system="static system block")

    sent = llm._client.messages.last_kwargs
    assert sent is not None
    system_payload = sent["system"]
    assert isinstance(system_payload, list)
    assert len(system_payload) == 1
    assert system_payload[0]["text"] == "static system block"
    assert system_payload[0]["cache_control"] == {"type": "ephemeral"}


async def test_caching_disabled_sends_plain_string_system(
    stub_response_with_cache_hit,
):
    llm = AnthropicLLM(model="claude-opus-4-7", api_key="x", enable_prompt_cache=False)
    llm._client = _StubClient(stub_response_with_cache_hit)

    await llm.generate("user msg", system="static system block")

    sent = llm._client.messages.last_kwargs
    assert sent["system"] == "static system block"


async def test_empty_system_never_wraps_even_with_caching(stub_response_with_cache_hit):
    """Cache-control on an empty system block is wasteful and confuses the API."""
    llm = AnthropicLLM(model="claude-opus-4-7", api_key="x", enable_prompt_cache=True)
    llm._client = _StubClient(stub_response_with_cache_hit)

    await llm.generate("user msg", system=None)
    assert llm._client.messages.last_kwargs["system"] == ""


async def test_cache_hit_flows_into_usage_and_cost(stub_response_with_cache_hit):
    llm = AnthropicLLM(model="claude-opus-4-7", api_key="x")
    llm._client = _StubClient(stub_response_with_cache_hit)

    await llm.generate("user msg", system="big static prompt")

    u = llm.last_usage
    assert u.input_tokens == 200
    assert u.output_tokens == 50
    assert u.cache_read_tokens == 1000
    assert u.cache_write_tokens == 0
    # opus-4-7: input $15, output $75, cache_read $1.50 per 1M tokens.
    # 200/1M*15 + 50/1M*75 + 1000/1M*1.5 = 0.003 + 0.00375 + 0.0015 = 0.00825
    assert u.cost_usd == Decimal("0.00825")


async def test_cache_miss_charges_at_creation_rate(stub_response_with_cache_miss):
    llm = AnthropicLLM(model="claude-opus-4-7", api_key="x")
    llm._client = _StubClient(stub_response_with_cache_miss)

    await llm.generate("user msg", system="big static prompt")

    u = llm.last_usage
    assert u.cache_read_tokens == 0
    assert u.cache_write_tokens == 1000
    # cache_write on opus-4-7 is $18.75 per 1M, so 1000 tokens = $0.01875
    # plus baseline input/output costs.
    expected = Decimal("200") * Decimal("15") / Decimal("1000000")
    expected += Decimal("50") * Decimal("75") / Decimal("1000000")
    expected += Decimal("1000") * Decimal("18.75") / Decimal("1000000")
    assert u.cost_usd == expected


async def test_missing_usage_field_is_handled(stub_response_with_cache_hit):
    """Some test fixtures or downgraded SDKs return no usage block — don't crash."""
    response = _Response(content=[_Block(text="{}")], usage=None)
    llm = AnthropicLLM(model="claude-opus-4-7", api_key="x")
    llm._client = _StubClient(response)
    await llm.generate("user msg", system="x")
    assert llm.last_usage.input_tokens == 0
    assert llm.last_usage.cost_usd == Decimal("0")
