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


# Default classifier model per provider. Chosen for cost + classification
# quality at a 1-line-headline scale (no need for strategy-tier reasoning).
# Override via env if operators want different models per provider.
_CLASSIFIER_MODEL_DEFAULTS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-haiku-4-5-20251001",
    LLMProvider.OLLAMA: "llama3.2:3b",
}

# Preferred classifier chain order: cheap cloud first, then the local
# Ollama floor so the reactor degrades to free-but-slower rather than
# failing entirely when both cloud providers are unreachable / out of
# credit. The 2026-05-22 quota incident proved single-provider classifier
# resilience isn't enough; a free floor that doesn't share the quota
# pool is the structural fix.
_CLASSIFIER_CHAIN_ORDER: list[LLMProvider] = [
    LLMProvider.OPENAI,
    LLMProvider.ANTHROPIC,
    LLMProvider.OLLAMA,
]


def create_classifier_llm(settings: Settings | None = None) -> BaseLLM:
    """Build a dedicated LLM stack for the news-headline classifier.

    Separates the classifier's failure modes from the strategy LLM so a
    quota exhaustion in one doesn't take down the other. The default
    chain prefers cheap classification-capable models, with the local
    Ollama floor always last so the reactor has *some* fallback even
    when both cloud providers fail.

    Construction rules:

    * **Cloud providers** (OpenAI, Anthropic) are only added when
      ``settings.llm.{provider}.api_key`` is truthy. Empty keys are
      skipped silently — they'd just generate auth errors at call time.
    * **Ollama** is always added — it needs no API key. If the Ollama
      host is unreachable at call time the classifier returns 0.0 like
      any other failure (handled by ``GPTHeadlineClassifier``).
    * The returned LLM is a :class:`FallbackLLM` when ≥2 providers were
      built, or the bare provider when only one. (A bare Ollama still
      works; the classifier just doesn't get rotation.)

    Raises ValueError only if not even Ollama could be constructed,
    which should never happen with the default :class:`OllamaSettings`.
    """
    if settings is None:
        settings = get_settings()

    built: list[BaseLLM] = []
    for provider in _CLASSIFIER_CHAIN_ORDER:
        # Cloud providers need an API key; Ollama doesn't.
        if provider == LLMProvider.OPENAI and not settings.llm.openai.api_key:
            continue
        if provider == LLMProvider.ANTHROPIC and not settings.llm.anthropic.api_key:
            continue
        model = _CLASSIFIER_MODEL_DEFAULTS[provider]
        instance = _create_single_llm(provider, model, settings)
        if instance is not None:
            built.append(instance)

    if not built:
        # Should be unreachable — Ollama always builds — but keep the
        # error explicit rather than handing back None.
        raise ValueError("Classifier LLM chain has no providers — this shouldn't happen")

    if len(built) == 1:
        logger.info(
            "Classifier LLM (no fallbacks): %s/%s",
            type(built[0]).__name__,
            built[0].model,
        )
        return built[0]

    primary = built[0]
    fallbacks = built[1:]
    logger.info(
        "Classifier LLM fallback chain: %s -> %s",
        type(primary).__name__,
        " -> ".join(f"{type(f).__name__}/{f.model}" for f in fallbacks),
    )
    return FallbackLLM(primary, fallbacks)
