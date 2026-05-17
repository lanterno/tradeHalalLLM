"""Tests for :class:`OllamaLLM`'s usage + message-shape contract.

Mirrors `test_openai_usage.py` for the local provider — pinning that
the per-call message format, JSON-mode request, token-count read, and
empty-response-rejection behave correctly without spinning up Ollama.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from halal_trader.core.llm.ollama import OllamaLLM


def _stub_response(
    *,
    content: str = '{"ok": true}',
    prompt_eval_count: int | None = 100,
    eval_count: int | None = 50,
) -> dict:
    """Build the dict shape Ollama's AsyncClient.chat returns."""
    return {
        "message": {"content": content},
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def _wire(llm: OllamaLLM, response: dict) -> AsyncMock:
    """Inject a fake AsyncClient whose chat() returns the given dict."""
    chat_mock = AsyncMock(return_value=response)
    llm._client = SimpleNamespace(chat=chat_mock)
    return chat_mock


# ── usage breakdown ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_records_token_counts():
    llm = OllamaLLM(model="qwen2.5:7b")
    _wire(llm, _stub_response(prompt_eval_count=100, eval_count=50))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 100
    assert llm.last_usage.output_tokens == 50


@pytest.mark.asyncio
async def test_usage_defaults_to_zero_when_counts_missing():
    """Older Ollama builds don't surface eval counts. Defensively
    default to zero rather than crashing the cost roll-up."""
    llm = OllamaLLM(model="qwen2.5:7b")
    _wire(llm, _stub_response(prompt_eval_count=None, eval_count=None))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 0
    assert llm.last_usage.output_tokens == 0


@pytest.mark.asyncio
async def test_cost_usd_is_zero_for_local_model():
    """Ollama is free — pricing table returns $0 for local models."""
    llm = OllamaLLM(model="qwen2.5:7b")
    _wire(llm, _stub_response(prompt_eval_count=10_000, eval_count=10_000))
    await llm.generate("hello")
    assert llm.last_usage.cost_usd == 0


# ── messages shape ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_system_prompt_emits_system_role_message():
    llm = OllamaLLM(model="qwen2.5:7b")
    chat_mock = _wire(llm, _stub_response())
    await llm.generate("user msg", system="sys msg")
    messages = chat_mock.await_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys msg"}
    assert messages[1] == {"role": "user", "content": "user msg"}


@pytest.mark.asyncio
async def test_no_system_prompt_omits_system_role():
    llm = OllamaLLM(model="qwen2.5:7b")
    chat_mock = _wire(llm, _stub_response())
    await llm.generate("user msg")
    messages = chat_mock.await_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_request_uses_json_format():
    """JSON mode is required for the strategy parser to skip the
    prose-sniff step every call."""
    llm = OllamaLLM(model="qwen2.5:7b")
    chat_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    assert chat_mock.await_args.kwargs["format"] == "json"


@pytest.mark.asyncio
async def test_request_uses_low_temperature():
    """temp=0.2 is our default for stable JSON output."""
    llm = OllamaLLM(model="qwen2.5:7b")
    chat_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    options = chat_mock.await_args.kwargs["options"]
    assert options["temperature"] == 0.2


@pytest.mark.asyncio
async def test_returns_message_content_directly():
    llm = OllamaLLM(model="qwen2.5:7b")
    _wire(llm, _stub_response(content='{"plan": "buy"}'))
    out = await llm.generate("hello")
    assert out == '{"plan": "buy"}'


# ── empty response rejection ───────────────────────────────


@pytest.mark.asyncio
async def test_empty_content_raises_value_error():
    """An empty string would silently degrade to "no plan" downstream;
    raise so the FallbackLLM can switch providers."""
    llm = OllamaLLM(model="qwen2.5:7b")
    _wire(llm, _stub_response(content=""))
    with pytest.raises(ValueError, match="empty"):
        await llm.generate("hello")


@pytest.mark.asyncio
async def test_whitespace_only_content_raises_value_error():
    llm = OllamaLLM(model="qwen2.5:7b")
    _wire(llm, _stub_response(content="   \n\t  "))
    with pytest.raises(ValueError, match="empty"):
        await llm.generate("hello")
