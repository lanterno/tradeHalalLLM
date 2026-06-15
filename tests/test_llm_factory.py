"""Tests for :func:`create_llm` factory + fallback chain assembly.

The actual provider classes (Ollama / OpenAI / Anthropic) are tested
in their own files; this file pins the wiring: which provider gets
selected as primary, when the FallbackLLM wrapper is added, and
how unknown / unconfigured fallback providers are handled.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from halal_trader.config import LLMProvider
from halal_trader.core.llm.factory import create_llm
from halal_trader.core.llm.fallback import FallbackLLM


def _settings(
    *,
    provider: LLMProvider = LLMProvider.OLLAMA,
    fallback_providers: list[str] | None = None,
    openai_key: str = "",
    anthropic_key: str = "",
) -> MagicMock:
    s = MagicMock()
    s.llm.provider = provider
    s.llm.model = "test-model"
    s.llm.fallback_providers = fallback_providers or []
    s.llm.ollama.host = "http://localhost:11434"
    s.llm.ollama.fallback_model = ""
    s.llm.openai.api_key = openai_key
    s.llm.openai.fallback_model = ""
    s.llm.anthropic.api_key = anthropic_key
    s.llm.anthropic.fallback_model = ""
    return s


# ── primary selection ─────────────────────────────────────────


def test_create_returns_ollama_when_provider_is_ollama():
    from halal_trader.core.llm.ollama import OllamaLLM

    out = create_llm(_settings(provider=LLMProvider.OLLAMA))
    assert isinstance(out, OllamaLLM)


def test_create_returns_openai_when_configured():
    from halal_trader.core.llm.openai import OpenAILLM

    out = create_llm(_settings(provider=LLMProvider.OPENAI, openai_key="sk-test"))
    assert isinstance(out, OpenAILLM)


def test_create_returns_anthropic_when_configured():
    from halal_trader.core.llm.anthropic import AnthropicLLM

    out = create_llm(_settings(provider=LLMProvider.ANTHROPIC, anthropic_key="sk-ant"))
    assert isinstance(out, AnthropicLLM)


def test_create_raises_when_primary_unconfigured():
    """Selecting OpenAI without an API key fails loudly — better
    than silently degrading to Ollama."""
    with pytest.raises(ValueError, match="not configured"):
        create_llm(_settings(provider=LLMProvider.OPENAI))


# ── fallback chain ────────────────────────────────────────────


def test_no_fallback_returns_primary_directly():
    """Empty `fallback_providers` → no FallbackLLM wrapper."""
    out = create_llm(_settings(fallback_providers=[]))
    assert not isinstance(out, FallbackLLM)


def test_with_configured_fallback_returns_fallback_wrapper():
    out = create_llm(
        _settings(
            provider=LLMProvider.OLLAMA,
            fallback_providers=["openai"],
            openai_key="sk",
        )
    )
    assert isinstance(out, FallbackLLM)


def test_unknown_fallback_provider_is_skipped():
    """Typo in `fallback_providers` shouldn't kill the bot."""
    out = create_llm(
        _settings(
            provider=LLMProvider.OLLAMA,
            fallback_providers=["bogus-provider"],
        )
    )
    # Falls back to no-fallback path because the unknown name was skipped.
    assert not isinstance(out, FallbackLLM)


def test_fallback_same_as_primary_is_skipped():
    """Listing the same provider as primary is a no-op (avoids
    self-reference cycle)."""
    out = create_llm(
        _settings(
            provider=LLMProvider.OLLAMA,
            fallback_providers=["ollama"],
        )
    )
    assert not isinstance(out, FallbackLLM)


def test_unconfigured_fallback_provider_is_skipped():
    """Listing 'openai' as a fallback but no API key → skipped, not
    crash. Bot still runs with primary only."""
    out = create_llm(
        _settings(
            provider=LLMProvider.OLLAMA,
            fallback_providers=["openai"],
            # openai_key intentionally empty
        )
    )
    assert not isinstance(out, FallbackLLM)


def test_multiple_fallbacks_all_added_in_order():
    out = create_llm(
        _settings(
            provider=LLMProvider.OLLAMA,
            fallback_providers=["openai", "anthropic"],
            openai_key="sk-1",
            anthropic_key="sk-2",
        )
    )
    assert isinstance(out, FallbackLLM)


# ── same-provider model fallback ──────────────────────────────


def test_same_provider_model_fallback_added_when_explicitly_configured():
    """An explicit distinct OPENAI_FALLBACK_MODEL gives the OpenAI primary a
    same-key degraded path (gpt-4o -> gpt-4o-mini) even with an empty
    fallback_providers list — so a transient gpt-4o timeout doesn't yield a
    no-action cycle."""
    from halal_trader.core.llm.openai import OpenAILLM

    s = _settings(provider=LLMProvider.OPENAI, openai_key="sk-test")
    s.llm.model = "gpt-4o"
    s.llm.openai.fallback_model = "gpt-4o-mini"
    out = create_llm(s)
    assert isinstance(out, FallbackLLM)
    assert isinstance(out._primary, OpenAILLM)
    assert out._primary.model == "gpt-4o"
    assert [f.model for f in out._fallbacks] == ["gpt-4o-mini"]


def test_same_provider_model_fallback_skipped_when_unset():
    """Default (empty fallback_model) → bare primary, no wrapper (unchanged
    behavior for configs that never opted in)."""
    s = _settings(provider=LLMProvider.OPENAI, openai_key="sk-test")
    s.llm.model = "gpt-4o"
    s.llm.openai.fallback_model = ""
    out = create_llm(s)
    assert not isinstance(out, FallbackLLM)


def test_same_provider_model_fallback_skipped_when_equal_to_primary():
    """A fallback model identical to the primary is a no-op (no pointless
    same-model retry leg)."""
    s = _settings(provider=LLMProvider.OPENAI, openai_key="sk-test")
    s.llm.model = "gpt-4o-mini"
    s.llm.openai.fallback_model = "gpt-4o-mini"
    out = create_llm(s)
    assert not isinstance(out, FallbackLLM)


def test_same_provider_model_fallback_precedes_cross_provider():
    """Same-provider model fallback is tried BEFORE switching providers."""
    s = _settings(
        provider=LLMProvider.OPENAI,
        fallback_providers=["anthropic"],
        openai_key="sk-1",
        anthropic_key="sk-2",
    )
    s.llm.model = "gpt-4o"
    s.llm.openai.fallback_model = "gpt-4o-mini"
    out = create_llm(s)
    assert isinstance(out, FallbackLLM)
    # gpt-4o -> gpt-4o-mini (same key) -> Anthropic (provider switch).
    assert [f.model for f in out._fallbacks] == ["gpt-4o-mini", "claude-sonnet-4-20250514"]


# ── create_classifier_llm — dedicated reactor-classifier chain ──


def test_classifier_chain_includes_ollama_floor_always():
    """Ollama needs no API key — the chain MUST always have a floor so
    classifier resilience doesn't depend on cloud credentials. This is
    the structural fix the 2026-05-22 quota incident motivated."""
    from halal_trader.core.llm.factory import create_classifier_llm
    from halal_trader.core.llm.ollama import OllamaLLM

    # No cloud keys at all.
    out = create_classifier_llm(_settings())
    # With only Ollama eligible the factory returns a bare OllamaLLM
    # (no rotation needed when there's only one provider).
    assert isinstance(out, OllamaLLM)


def test_classifier_chain_prefers_openai_when_configured():
    """OpenAI is primary by design (cheapest at gpt-4o-mini, fastest
    classification quality at the 1-line-headline scale)."""
    from halal_trader.core.llm.factory import create_classifier_llm
    from halal_trader.core.llm.openai import OpenAILLM

    out = create_classifier_llm(_settings(openai_key="sk-test"))
    # With two providers (OpenAI + Ollama floor) we get a FallbackLLM.
    assert isinstance(out, FallbackLLM)
    assert isinstance(out._primary, OpenAILLM)
    # Ollama is the only fallback in this scenario.
    assert len(out._fallbacks) == 1


def test_classifier_chain_full_three_providers():
    """All three configured → OpenAI → Anthropic → Ollama chain."""
    from halal_trader.core.llm.anthropic import AnthropicLLM
    from halal_trader.core.llm.factory import create_classifier_llm
    from halal_trader.core.llm.ollama import OllamaLLM
    from halal_trader.core.llm.openai import OpenAILLM

    out = create_classifier_llm(
        _settings(openai_key="sk-1", anthropic_key="sk-2")
    )
    assert isinstance(out, FallbackLLM)
    assert isinstance(out._primary, OpenAILLM)
    assert len(out._fallbacks) == 2
    assert isinstance(out._fallbacks[0], AnthropicLLM)
    assert isinstance(out._fallbacks[1], OllamaLLM)


def test_classifier_chain_anthropic_primary_when_no_openai():
    """OpenAI key missing → Anthropic becomes primary, Ollama floor."""
    from halal_trader.core.llm.anthropic import AnthropicLLM
    from halal_trader.core.llm.factory import create_classifier_llm
    from halal_trader.core.llm.ollama import OllamaLLM

    out = create_classifier_llm(_settings(anthropic_key="sk-ant"))
    assert isinstance(out, FallbackLLM)
    assert isinstance(out._primary, AnthropicLLM)
    assert len(out._fallbacks) == 1
    assert isinstance(out._fallbacks[0], OllamaLLM)


def test_classifier_chain_uses_cheap_default_models():
    """The classifier-specific defaults must NOT be the strategy's
    heavyweight model — classify is a 1-line task that should burn the
    cheapest available variant per provider."""
    from halal_trader.core.llm.factory import create_classifier_llm

    out = create_classifier_llm(
        _settings(openai_key="sk-1", anthropic_key="sk-2")
    )
    assert isinstance(out, FallbackLLM)
    # OpenAI primary: gpt-4o-mini (not gpt-4o, gpt-5, etc).
    assert "mini" in out._primary.model.lower()
    # Anthropic fallback: haiku (cheap), not opus/sonnet.
    assert "haiku" in out._fallbacks[0].model.lower()


def test_classifier_chain_pins_temperature_to_zero():
    """Every provider in the classifier stack must use greedy decoding
    (temperature=0.0) so the same headline scores reproducibly. The
    strategy LLM keeps its 0.2 default — only the classifier needs to
    be deterministic near the 0.85 entry threshold."""
    from halal_trader.core.llm.factory import create_classifier_llm

    out = create_classifier_llm(
        _settings(openai_key="sk-1", anthropic_key="sk-2")
    )
    assert isinstance(out, FallbackLLM)
    assert out._primary.temperature == 0.0
    assert all(fb.temperature == 0.0 for fb in out._fallbacks)


def test_strategy_llm_keeps_default_temperature():
    """Regression guard: the determinism change must NOT bleed into the
    strategy path — create_llm providers stay at the 0.2 default."""
    out = create_llm(_settings(provider=LLMProvider.OLLAMA))
    assert out.temperature == 0.2
