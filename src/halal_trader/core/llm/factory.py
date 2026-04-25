"""Factory + opt-in fallback chain assembly."""

from __future__ import annotations

import logging

from halal_trader.config import LLMProvider, Settings, get_settings
from halal_trader.core.llm.anthropic import AnthropicLLM
from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.fallback import FallbackLLM
from halal_trader.core.llm.ollama import OllamaLLM
from halal_trader.core.llm.openai import OpenAILLM

logger = logging.getLogger(__name__)


def _create_single_llm(provider: LLMProvider, model: str, settings: Settings) -> BaseLLM | None:
    """Create a single LLM instance for a given provider, or None if unconfigured."""
    match provider:
        case LLMProvider.OLLAMA:
            return OllamaLLM(model=model, host=settings.llm.ollama.host)
        case LLMProvider.OPENAI:
            if settings.llm.openai.api_key:
                return OpenAILLM(model=model, api_key=settings.llm.openai.api_key)
        case LLMProvider.ANTHROPIC:
            if settings.llm.anthropic.api_key:
                return AnthropicLLM(model=model, api_key=settings.llm.anthropic.api_key)
    return None


def create_llm(settings: Settings | None = None) -> BaseLLM:
    """Factory: create the appropriate LLM with opt-in fallback chain.

    Fallbacks are only created for providers explicitly listed in
    ``settings.llm.fallback_providers``.  An empty list (the default)
    means primary-only with no cloud fallback.
    """
    if settings is None:
        settings = get_settings()

    primary = _create_single_llm(settings.llm.provider, settings.llm.model, settings)
    if primary is None:
        raise ValueError(f"Primary LLM provider {settings.llm.provider.value} is not configured")

    fallback_models = {
        LLMProvider.OLLAMA: settings.llm.ollama.fallback_model or settings.llm.model,
        LLMProvider.OPENAI: settings.llm.openai.fallback_model or "gpt-4o-mini",
        LLMProvider.ANTHROPIC: (
            settings.llm.anthropic.fallback_model or "claude-sonnet-4-20250514"
        ),
    }

    fallbacks: list[BaseLLM] = []
    for name in settings.llm.fallback_providers:
        try:
            provider = LLMProvider(name.lower())
        except ValueError:
            logger.warning("Unknown fallback provider '%s' — skipping", name)
            continue
        if provider == settings.llm.provider:
            continue
        model = fallback_models.get(provider, settings.llm.model)
        fb = _create_single_llm(provider, model, settings)
        if fb is not None:
            fallbacks.append(fb)
        else:
            logger.warning(
                "Fallback provider '%s' requested but not configured (missing API key?)",
                name,
            )

    if fallbacks:
        logger.info(
            "LLM fallback chain: %s -> %s",
            type(primary).__name__,
            " -> ".join(type(f).__name__ for f in fallbacks),
        )
        return FallbackLLM(primary, fallbacks)

    logger.info("LLM provider: %s (no fallbacks)", type(primary).__name__)
    return primary
