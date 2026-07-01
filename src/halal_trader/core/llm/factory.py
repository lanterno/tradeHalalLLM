"""Factory: assemble the GLM-5.2 stack (primary + optional endpoint fallback)."""

from __future__ import annotations

import logging

from halal_trader.config import Settings, get_settings
from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.fallback import FallbackLLM
from halal_trader.core.llm.glm import GLMLLM

logger = logging.getLogger(__name__)

# Classifier scores must be reproducible: the 2026-05-22 session logged
# the same headline scoring 0.70 on one call and 0.90 on the next, which
# makes the 0.85 entry threshold a coin-flip near the boundary. Pin the
# whole classifier stack to greedy decoding so a headline lands on one
# side of the threshold deterministically. (GLMLLM clamps 0.0 → 0.01 on
# Z.ai-style endpoints, which reject exactly 0.0.)
_CLASSIFIER_TEMPERATURE = 0.0


def _build_chain(
    settings: Settings,
    *,
    temperature: float = 0.2,
    thinking: bool | None = None,
) -> BaseLLM:
    """Primary GLM endpoint, plus a FallbackLLM wrap when a second endpoint is set.

    Both chain members are GLM-5.2 — the fallback is a different *host*
    for the same model (e.g. Z.ai direct behind OpenRouter), not a
    different model family. ``thinking=None`` inherits the configured
    default; the classifier pins it off.
    """
    g = settings.llm.glm
    if not g.api_key:
        raise ValueError(
            "GLM_API_KEY is not set — the bot cannot start without it. "
            "Create a key at https://openrouter.ai/keys (default endpoint) "
            "and add GLM_API_KEY=... to .env"
        )
    think = g.thinking if thinking is None else thinking

    primary = GLMLLM(
        model=settings.llm.model,
        api_key=g.api_key,
        base_url=g.base_url,
        temperature=temperature,
        timeout_seconds=g.timeout_seconds,
        thinking=think,
        require_parameters=g.require_parameters,
    )

    if not (g.fallback_base_url or g.fallback_model):
        return primary

    fb_base = g.fallback_base_url or g.base_url
    fb_model = g.fallback_model or settings.llm.model
    if (fb_base.rstrip("/"), fb_model) == (g.base_url.rstrip("/"), settings.llm.model):
        logger.warning("GLM fallback endpoint is identical to the primary — skipping the chain")
        return primary

    fallback = GLMLLM(
        model=fb_model,
        api_key=g.fallback_api_key or g.api_key,
        base_url=fb_base,
        temperature=temperature,
        timeout_seconds=g.timeout_seconds,
        thinking=think,
        require_parameters=g.require_parameters,
    )
    logger.info(
        "GLM endpoint chain: %s (%s) -> %s (%s)",
        settings.llm.model,
        primary.base_url,
        fb_model,
        fallback.base_url,
    )
    return FallbackLLM(primary, [fallback])


def create_llm(settings: Settings | None = None) -> BaseLLM:
    """The strategy LLM: GLM-5.2 with the configured endpoint chain."""
    if settings is None:
        settings = get_settings()
    llm = _build_chain(settings)
    if isinstance(llm, GLMLLM):
        logger.info("LLM: GLM %s via %s (no endpoint fallback)", llm.model, llm.base_url)
    return llm


def create_classifier_llm(settings: Settings | None = None) -> BaseLLM:
    """A dedicated GLM stack for the news-headline classifier.

    Separate instances from the strategy LLM so backoff state and usage
    accounting don't bleed between the two workloads. Thinking is
    always off here — a 1-line headline needs milliseconds-cheap greedy
    classification, not strategy-tier reasoning.
    """
    if settings is None:
        settings = get_settings()
    return _build_chain(settings, temperature=_CLASSIFIER_TEMPERATURE, thinking=False)
