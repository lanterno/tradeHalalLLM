"""Tests for :class:`GLMLLM`'s usage accounting + endpoint dialects.

GLM-5.2 is served through OpenAI-compatible endpoints, so the usage
tree mirrors the OpenAI shape — most importantly the cache-token
subtraction (cached tokens arrive bundled *inside* ``prompt_tokens``,
so they're subtracted to get fresh-input billing). The dialect tests
pin the per-host request differences: OpenRouter's ``reasoning`` /
``provider`` extra_body vs the Z.ai-style ``thinking`` toggle, and the
temperature-0.0 clamp that only applies off-OpenRouter.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from halal_trader.core.llm.glm import GLMLLM

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


def _stub_response(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cached_tokens: int = 0,
    text: str = '{"ok": true}',
) -> SimpleNamespace:
    """Build the SimpleNamespace tree the real OpenAI-compat SDK emits."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _wire(llm: GLMLLM, response: SimpleNamespace) -> AsyncMock:
    """Inject a fake AsyncOpenAI client whose chat.completions.create
    returns the response we want."""
    create_mock = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    llm._client = client
    return create_mock


# ── usage breakdown ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_records_prompt_completion_tokens():
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=100, completion_tokens=50))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 100
    assert llm.last_usage.output_tokens == 50


@pytest.mark.asyncio
async def test_usage_provider_label_is_glm():
    """CallUsage rows are attributed to the 'glm' provider — the
    LlmDecision table and cost roll-ups key on this label."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    _wire(llm, _stub_response())
    await llm.generate("hello")
    assert llm.last_usage.provider == "glm"
    assert llm.last_usage.model == "z-ai/glm-5.2"


@pytest.mark.asyncio
async def test_cached_tokens_subtracted_from_prompt_tokens():
    """cached_tokens is INSIDE prompt_tokens, not in addition.
    Subtract for accurate fresh-input billing."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=100, cached_tokens=30))
    await llm.generate("hello")
    # 100 - 30 = 70 fresh prompt tokens
    assert llm.last_usage.input_tokens == 70
    assert llm.last_usage.cache_read_tokens == 30


@pytest.mark.asyncio
async def test_subtraction_clamped_at_zero():
    """If cached_tokens > prompt_tokens (shouldn't happen but defensive),
    input_tokens stays at 0 rather than going negative."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=10, cached_tokens=50))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 0
    assert llm.last_usage.cache_read_tokens == 50


@pytest.mark.asyncio
async def test_no_cached_tokens_passthrough():
    """When cached_tokens is 0, prompt_tokens is unchanged."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=100, cached_tokens=0))
    await llm.generate("hello")
    assert llm.last_usage.input_tokens == 100
    assert llm.last_usage.cache_read_tokens == 0


# ── cost computation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_cost_usd_populated():
    """A response with token counts produces a positive cost."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    _wire(llm, _stub_response(prompt_tokens=1000, completion_tokens=500))
    await llm.generate("hello")
    assert llm.last_usage.cost_usd > 0


@pytest.mark.asyncio
async def test_cached_tokens_lower_cost_on_zai_pricing():
    """Same total prompt_tokens, but with caching → cheaper bill.

    Uses the Z.ai model id — its table row has a discounted cache-read
    rate ($0.26 vs $1.40 input). The OpenRouter row deliberately bills
    cache reads at the full input rate (errs high), so the discount is
    only observable on the direct-endpoint pricing.
    """
    full = GLMLLM(model="glm-5.2", api_key="sk-test", base_url=ZAI_BASE_URL)
    _wire(full, _stub_response(prompt_tokens=1000, completion_tokens=0, cached_tokens=0))
    await full.generate("hello")
    cost_full = full.last_usage.cost_usd

    cached = GLMLLM(model="glm-5.2", api_key="sk-test", base_url=ZAI_BASE_URL)
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
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    create_mock = _wire(llm, _stub_response())
    await llm.generate("user msg", system="sys msg")
    messages = create_mock.await_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys msg"}
    assert messages[1] == {"role": "user", "content": "user msg"}


@pytest.mark.asyncio
async def test_no_system_prompt_omits_system_role():
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    create_mock = _wire(llm, _stub_response())
    await llm.generate("user msg")
    messages = create_mock.await_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_request_uses_json_object_response_format():
    """JSON-mode is required so the strategy parser doesn't have to
    sniff prose from a free-form completion."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    kwargs = create_mock.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_returns_message_content_directly():
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    _wire(llm, _stub_response(text='{"plan": "buy"}'))
    out = await llm.generate("hello")
    assert out == '{"plan": "buy"}'


@pytest.mark.asyncio
async def test_returns_empty_string_when_content_none():
    """Defensive: a fluky response with `content=None` doesn't crash."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
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


# ── endpoint dialect: OpenRouter ────────────────────────────


@pytest.mark.asyncio
async def test_openrouter_sends_reasoning_and_provider_extra_body():
    """The default OpenRouter endpoint normalises the thinking toggle
    as ``reasoning.enabled`` and pins ``provider.require_parameters``
    so requests only route to hosts that honour response_format+tools."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    extra = create_mock.await_args.kwargs["extra_body"]
    assert extra["reasoning"] == {"enabled": False}
    assert extra["provider"] == {"require_parameters": True}
    # The Z.ai-style key must NOT leak onto OpenRouter requests.
    assert "thinking" not in extra


@pytest.mark.asyncio
async def test_openrouter_thinking_true_flips_reasoning_enabled():
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test", thinking=True)
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    extra = create_mock.await_args.kwargs["extra_body"]
    assert extra["reasoning"] == {"enabled": True}


@pytest.mark.asyncio
async def test_openrouter_require_parameters_false_omits_provider_key():
    """require_parameters=False must omit the provider block entirely —
    an explicit ``provider: {}`` would still constrain routing."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test", require_parameters=False)
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    extra = create_mock.await_args.kwargs["extra_body"]
    assert "provider" not in extra
    assert extra["reasoning"] == {"enabled": False}


# ── endpoint dialect: Z.ai-style (non-OpenRouter) ───────────


@pytest.mark.asyncio
async def test_zai_sends_thinking_disabled_extra_body():
    """Non-OpenRouter hosts speak the native GLM dialect —
    ``thinking: {"type": "disabled"}`` and no OpenRouter routing keys."""
    llm = GLMLLM(model="glm-5.2", api_key="sk-test", base_url=ZAI_BASE_URL)
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    extra = create_mock.await_args.kwargs["extra_body"]
    assert extra["thinking"] == {"type": "disabled"}
    assert "reasoning" not in extra
    assert "provider" not in extra


@pytest.mark.asyncio
async def test_zai_thinking_true_sends_enabled():
    llm = GLMLLM(model="glm-5.2", api_key="sk-test", base_url=ZAI_BASE_URL, thinking=True)
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    extra = create_mock.await_args.kwargs["extra_body"]
    assert extra["thinking"] == {"type": "enabled"}


# ── temperature handling ────────────────────────────────────


@pytest.mark.asyncio
async def test_temperature_zero_clamped_on_zai_endpoint():
    """Z.ai historically rejects exactly 0.0 — the classifier's greedy
    pin is clamped to 0.01 on non-OpenRouter endpoints."""
    llm = GLMLLM(model="glm-5.2", api_key="sk-test", base_url=ZAI_BASE_URL, temperature=0.0)
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    assert create_mock.await_args.kwargs["temperature"] == 0.01


@pytest.mark.asyncio
async def test_temperature_zero_not_clamped_on_openrouter():
    """OpenRouter accepts 0.0 — the clamp must NOT apply there, so the
    classifier stays truly greedy on the default endpoint."""
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test", temperature=0.0)
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    assert create_mock.await_args.kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_nonzero_temperature_passes_through_on_zai():
    """The clamp is exactly-0.0 only — the 0.2 strategy default is
    untouched on every endpoint."""
    llm = GLMLLM(model="glm-5.2", api_key="sk-test", base_url=ZAI_BASE_URL, temperature=0.2)
    create_mock = _wire(llm, _stub_response())
    await llm.generate("hello")
    assert create_mock.await_args.kwargs["temperature"] == 0.2
