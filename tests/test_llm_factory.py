"""Tests for :func:`create_llm` factory + endpoint-fallback chain assembly.

The GLMLLM provider itself is tested in `test_glm_usage.py` /
`test_glm_tool_call.py`; this file pins the wiring: the fail-loud
missing-key contract, when the FallbackLLM wrapper is added around a
second GLM *endpoint* (same model family, different host), and the
classifier stack's determinism pins (temperature 0.0, thinking off).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from halal_trader.core.llm.factory import create_classifier_llm, create_llm
from halal_trader.core.llm.fallback import FallbackLLM
from halal_trader.core.llm.glm import GLMLLM


def _settings(
    *,
    model: str = "z-ai/glm-5.2",
    api_key: str = "sk-glm",
    base_url: str = "https://openrouter.ai/api/v1",
    fallback_base_url: str = "",
    fallback_model: str = "",
    fallback_api_key: str = "",
    timeout_seconds: int = 60,
    thinking: bool = False,
    require_parameters: bool = True,
) -> SimpleNamespace:
    """Attribute-only Settings stand-in — the factory reads, never validates."""
    glm = SimpleNamespace(
        api_key=api_key,
        base_url=base_url,
        fallback_base_url=fallback_base_url,
        fallback_model=fallback_model,
        fallback_api_key=fallback_api_key,
        timeout_seconds=timeout_seconds,
        thinking=thinking,
        require_parameters=require_parameters,
    )
    return SimpleNamespace(llm=SimpleNamespace(model=model, glm=glm))


# ── primary construction ──────────────────────────────────────


def test_create_returns_glm_primary():
    out = create_llm(_settings())
    assert isinstance(out, GLMLLM)


def test_create_raises_when_api_key_missing():
    """No GLM_API_KEY fails loudly at startup — there is no other
    provider to silently degrade to anymore."""
    with pytest.raises(ValueError, match="GLM_API_KEY"):
        create_llm(_settings(api_key=""))


def test_create_passes_settings_through_to_glm():
    """Every GLM knob flows from Settings into the provider instance —
    a factory refactor that drops one would silently revert the
    operator's endpoint config to the class defaults."""
    out = create_llm(
        _settings(
            model="glm-5.2",
            api_key="sk-custom",
            base_url="https://api.z.ai/api/paas/v4",
            timeout_seconds=90,
            thinking=True,
            require_parameters=False,
        )
    )
    assert isinstance(out, GLMLLM)
    assert out.model == "glm-5.2"
    assert out.api_key == "sk-custom"
    assert out.base_url == "https://api.z.ai/api/paas/v4"
    assert out.timeout_seconds == 90
    assert out.thinking is True
    assert out.require_parameters is False


def test_strategy_llm_keeps_default_temperature():
    """Regression guard: the classifier's determinism pin must NOT
    bleed into the strategy path — create_llm stays at the 0.2 default."""
    out = create_llm(_settings())
    assert out.temperature == 0.2


# ── endpoint-fallback chain ───────────────────────────────────


def test_no_fallback_config_returns_bare_primary():
    """Neither fallback_base_url nor fallback_model set → no wrapper."""
    out = create_llm(_settings())
    assert isinstance(out, GLMLLM)
    assert not isinstance(out, FallbackLLM)


def test_distinct_fallback_base_url_wraps_in_fallback_llm():
    """A second host for the same weights (e.g. Z.ai direct behind
    OpenRouter) gives the primary a degraded path."""
    out = create_llm(
        _settings(
            fallback_base_url="https://api.z.ai/api/paas/v4",
            fallback_model="glm-5.2",
        )
    )
    assert isinstance(out, FallbackLLM)
    assert isinstance(out._primary, GLMLLM)
    assert out._primary.base_url == "https://openrouter.ai/api/v1"
    assert [f.base_url for f in out._fallbacks] == ["https://api.z.ai/api/paas/v4"]
    assert [f.model for f in out._fallbacks] == ["glm-5.2"]


def test_fallback_model_alone_is_enough_to_chain():
    """Only fallback_model set → same base_url, different model id —
    still a distinct endpoint pair, so the chain is built."""
    out = create_llm(_settings(fallback_model="z-ai/glm-5.2:nitro"))
    assert isinstance(out, FallbackLLM)
    assert out._fallbacks[0].model == "z-ai/glm-5.2:nitro"
    assert out._fallbacks[0].base_url == "https://openrouter.ai/api/v1"


def test_fallback_base_url_alone_inherits_primary_model():
    """Only fallback_base_url set → the fallback reuses the primary's
    model id (naming may still need an override per host, but the
    default is the sane one)."""
    out = create_llm(_settings(fallback_base_url="https://api.fireworks.ai/inference/v1"))
    assert isinstance(out, FallbackLLM)
    assert out._fallbacks[0].model == "z-ai/glm-5.2"


def test_identical_fallback_endpoint_is_skipped():
    """A fallback describing the same (base_url, model) pair as the
    primary is a pointless self-retry leg — warn + bare primary."""
    out = create_llm(
        _settings(
            fallback_base_url="https://openrouter.ai/api/v1",
            fallback_model="z-ai/glm-5.2",
        )
    )
    assert isinstance(out, GLMLLM)
    assert not isinstance(out, FallbackLLM)


def test_identical_fallback_ignores_trailing_slash():
    """Endpoint identity is compared with trailing slashes stripped —
    `.../v1/` vs `.../v1` is the same host, not a chain."""
    out = create_llm(
        _settings(
            fallback_base_url="https://openrouter.ai/api/v1/",
            fallback_model="z-ai/glm-5.2",
        )
    )
    assert not isinstance(out, FallbackLLM)


def test_fallback_api_key_defaults_to_primary_key():
    """Empty fallback_api_key → the fallback reuses the primary key
    (common when both endpoints are OpenRouter-routed)."""
    out = create_llm(
        _settings(api_key="sk-primary", fallback_base_url="https://api.z.ai/api/paas/v4")
    )
    assert isinstance(out, FallbackLLM)
    assert out._fallbacks[0].api_key == "sk-primary"


def test_fallback_api_key_used_when_set():
    out = create_llm(
        _settings(
            api_key="sk-primary",
            fallback_base_url="https://api.z.ai/api/paas/v4",
            fallback_api_key="sk-zai",
        )
    )
    assert isinstance(out, FallbackLLM)
    assert out._fallbacks[0].api_key == "sk-zai"


def test_fallback_inherits_shared_knobs():
    """timeout/thinking/require_parameters apply to both chain members —
    the fallback is the same workload on a different host."""
    out = create_llm(
        _settings(
            fallback_base_url="https://api.z.ai/api/paas/v4",
            timeout_seconds=45,
            thinking=True,
        )
    )
    assert isinstance(out, FallbackLLM)
    fb = out._fallbacks[0]
    assert fb.timeout_seconds == 45
    assert fb.thinking is True
    assert fb.require_parameters is True


# ── create_classifier_llm — dedicated reactor-classifier chain ──


def test_classifier_pins_temperature_to_zero():
    """The classifier stack must use greedy decoding (temperature=0.0)
    so the same headline scores reproducibly near the 0.85 entry
    threshold. The strategy LLM keeps its 0.2 default."""
    out = create_classifier_llm(_settings())
    assert isinstance(out, GLMLLM)
    assert out.temperature == 0.0


def test_classifier_pins_thinking_off_even_when_configured_on():
    """A 1-line headline needs milliseconds-cheap classification —
    thinking stays off even when the operator enables it for the
    strategy path."""
    out = create_classifier_llm(_settings(thinking=True))
    assert isinstance(out, GLMLLM)
    assert out.thinking is False


def test_classifier_chain_mirrors_endpoint_fallback():
    """The classifier gets the same endpoint chain as the strategy LLM,
    with the determinism pins applied to every chain member."""
    out = create_classifier_llm(
        _settings(fallback_base_url="https://api.z.ai/api/paas/v4", thinking=True)
    )
    assert isinstance(out, FallbackLLM)
    assert out._primary.temperature == 0.0
    assert out._primary.thinking is False
    assert all(fb.temperature == 0.0 for fb in out._fallbacks)
    assert all(fb.thinking is False for fb in out._fallbacks)


def test_classifier_raises_when_api_key_missing():
    """Same fail-loud contract as create_llm — there is no free local
    floor anymore, so a missing key must not silently no-op."""
    with pytest.raises(ValueError, match="GLM_API_KEY"):
        create_classifier_llm(_settings(api_key=""))


def test_classifier_is_separate_instance_from_strategy_llm():
    """Backoff state and usage accounting must not bleed between the
    strategy and classifier workloads — distinct instances."""
    s = _settings()
    strategy = create_llm(s)
    classifier = create_classifier_llm(s)
    assert strategy is not classifier
