"""Tests for :class:`OpenAILLM`'s usage + cache-credit accounting.

The Anthropic provider's caching tests live in `test_anthropic_caching.py`;
this file mirrors them for OpenAI — most importantly the cache-token
subtraction (OpenAI bundles cached tokens *inside* `prompt_tokens`,
unlike Anthropic which surfaces them separately).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from halal_trader.core.llm.openai import OpenAILLM


def _stub_response(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cached_tokens: int = 0,
    text: str = '{"ok": true}',
) -> SimpleNamespace:
    """Build the SimpleNamespace tree the real OpenAI SDK emits."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _wire(llm: OpenAILLM, response: SimpleNamespace) -> AsyncMock:
    """Inject a fake AsyncOpenAI client whose chat.completions.create
    returns the response we want."""
    create_mock = AsyncMock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm._client = client
    return create_mock


# ── usage breakdown ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_records_prompt_completion_tokens():
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=100, completion_tokens=50))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 100
    assert llm.last_usage.output_tokens == 50


@pytest.mark.asyncio
async def test_cached_tokens_subtracted_from_prompt_tokens():
    """OpenAI's cached_tokens is INSIDE prompt_tokens, not in addition.
    Subtract for accurate fresh-input billing."""
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=100, cached_tokens=30))
    await llm.generate("hello")
    # 100 - 30 = 70 fresh prompt tokens
    assert llm.last_usage.input_tokens == 70
    assert llm.last_usage.cache_read_tokens == 30


@pytest.mark.asyncio
async def test_subtraction_clamped_at_zero():
    """If cached_tokens > prompt_tokens (shouldn't happen but defensive),
    input_tokens stays at 0 rather than going negative."""
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=10, cached_tokens=50))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 0
    assert llm.last_usage.cache_read_tokens == 50


@pytest.mark.asyncio
async def test_no_cached_tokens_passthrough():
    """When cached_tokens is 0, prompt_tokens is unchanged."""
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=100, cached_tokens=0))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 100
    assert llm.last_usage.cache_read_tokens == 0


# ── cost computation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_cost_usd_populated():
    """A response with token counts produces a positive cost."""
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=1000, completion_tokens=500))
    await llm.generate("hello")
    assert llm.last_usage.cost_usd > 0


@pytest.mark.asyncio
async def test_cached_tokens_lower_cost_than_full_prompt():
    """Same total prompt_tokens, but with caching → cheaper bill."""
    full = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(full, _stub_response(prompt_tokens=1000, completion_tokens=0, cached_tokens=0))
    await full.generate("hello")
    cost_full = full.last_usage.cost_usd

    cached = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(
        cached,
        _stub_response(prompt_tokens=1000, completion_tokens=0, cached_tokens=900),
    )
    await cached.generate("hello")
    cost_cached = cached.last_usage.cost_usd

    assert cost_cached < cost_full


# ── messages shape ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_system_prompt_emits_system_role_message():
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    create_mock = _wire(llm, _stub_response())
    await llm.generate("user msg", system="sys msg")
    messages = create_mock.await_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys msg"}
    assert messages[1] == {"role": "user", "content": "user msg"}


@pytest.mark.asyncio
async def test_no_system_prompt_omits_system_role():
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    create_mock = _wire(llm, _stub_response())
    await llm.generate("user msg")
    messages = create_mock.await_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_request_uses_json_object_response_format():
    """JSON-mode is required so the strategy parser doesn't have to
    sniff prose from a free-form completion."""
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    kwargs = create_mock.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_returns_message_content_directly():
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    _wire(llm, _stub_response(text='{"plan": "buy"}'))
    out = await llm.generate("hello")
    assert out == '{"plan": "buy"}'


@pytest.mark.asyncio
async def test_returns_empty_string_when_content_none():
    """Defensive: a fluky response with `content=None` doesn't crash."""
    llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
    )
    _wire(llm, response)
    out = await llm.generate("hello")
    assert out == ""
