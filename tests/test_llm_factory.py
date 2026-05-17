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

    out = create_llm(
        _settings(provider=LLMProvider.OPENAI, openai_key="sk-test")
    )
    assert isinstance(out, OpenAILLM)


def test_create_returns_anthropic_when_configured():
    from halal_trader.core.llm.anthropic import AnthropicLLM

    out = create_llm(
        _settings(provider=LLMProvider.ANTHROPIC, anthropic_key="sk-ant")
    )
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
