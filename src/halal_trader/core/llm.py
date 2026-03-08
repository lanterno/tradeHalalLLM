"""LLM provider abstraction supporting Ollama, OpenAI, and Anthropic."""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from halal_trader.config import LLMProvider, Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN_LOG_THRESHOLDS = (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def strip_thinking(text: str) -> tuple[str, str]:
    """Separate ``<think>`` reasoning from the final answer.

    Returns (thinking_chain, clean_body).  If no ``<think>`` tags are
    present, *thinking_chain* is empty.
    """
    parts = _THINK_RE.findall(text)
    thinking = "\n\n".join(p.strip() for p in parts if p.strip())
    body = _THINK_RE.sub("", text).strip()
    return thinking, body


def _clean_json_body(raw: str) -> str:
    """Strip markdown code fences and leading prose from a raw LLM response."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    return cleaned


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.last_thinking: str = ""
        self._daily_tokens: int = 0
        self._daily_reset_date: str = ""
        self._last_threshold_logged: int = 0

    def _track_usage(self, tokens: int) -> None:
        """Accumulate daily token usage and log at key thresholds."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_tokens = 0
            self._daily_reset_date = today
            self._last_threshold_logged = 0

        self._daily_tokens += tokens

        for threshold in _TOKEN_LOG_THRESHOLDS:
            if self._daily_tokens >= threshold > self._last_threshold_logged:
                self._last_threshold_logged = threshold
                logger.info(
                    "LLM daily token usage crossed %dk (%d total today, model: %s)",
                    threshold // 1000, self._daily_tokens, self.model,
                )
                break

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt and return the raw text response."""
        ...

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Generate a response and parse it as JSON."""
        raw = await self.generate(prompt, system)
        thinking, body = strip_thinking(raw)
        self.last_thinking = thinking
        if thinking:
            logger.debug("LLM thinking (%d chars): %.200s…", len(thinking), thinking)
        return json.loads(_clean_json_body(body))


class OllamaLLM(BaseLLM):
    """Local LLM via Ollama with thinking-mode awareness.

    Uses ``format="json"`` for fast, reliable structured output.
    Any ``<think>`` reasoning blocks the model may still emit are
    captured in ``last_thinking`` for audit and self-improvement.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        super().__init__(model)
        self.host = host
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            self._client = ollama.AsyncClient(host=self.host)
        return self._client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = await client.chat(
            model=self.model,
            messages=messages,
            format="json",
            options={"temperature": 0.2},
        )
        elapsed = time.monotonic() - t0
        logger.debug("Ollama response in %.1fs", elapsed)

        return response["message"]["content"]


class OpenAILLM(BaseLLM):
    """Cloud LLM via OpenAI API."""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        elapsed = time.monotonic() - t0
        if response.usage:
            logger.debug(
                "OpenAI response in %.1fs — %d tokens (prompt: %d, completion: %d)",
                elapsed,
                response.usage.total_tokens,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            self._track_usage(response.usage.total_tokens)
        else:
            logger.debug("OpenAI response in %.1fs", elapsed)

        return response.choices[0].message.content or ""


class AnthropicLLM(BaseLLM):
    """Cloud LLM via Anthropic API."""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()

        t0 = time.monotonic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.monotonic() - t0
        if hasattr(response, "usage") and response.usage:
            total = response.usage.input_tokens + response.usage.output_tokens
            logger.debug(
                "Anthropic response in %.1fs — input: %d, output: %d tokens",
                elapsed,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            self._track_usage(total)
        else:
            logger.debug("Anthropic response in %.1fs", elapsed)

        return response.content[0].text


class FallbackLLM(BaseLLM):
    """Wraps a primary LLM with fallback providers and exponential backoff."""

    def __init__(self, primary: BaseLLM, fallbacks: list[BaseLLM] | None = None) -> None:
        super().__init__(primary.model)
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._consecutive_failures = 0
        self._backoff_until: float = 0
        self._max_backoff_minutes = 30
        self._active_model = primary.model

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._active_model

    @model.setter
    def model(self, value: str) -> None:
        self._active_model = value

    async def generate(self, prompt: str, system: str | None = None) -> str:
        providers: list[BaseLLM] = []

        now = time.monotonic()
        if now >= self._backoff_until:
            providers.append(self._primary)
        else:
            remaining = self._backoff_until - now
            logger.debug(
                "Primary LLM in backoff for %.0fs more, trying fallbacks", remaining
            )

        providers.extend(self._fallbacks)

        last_error: Exception | None = None
        for provider in providers:
            try:
                result = await provider.generate(prompt, system)
                if provider is self._primary:
                    self._consecutive_failures = 0
                    self._backoff_until = 0
                self._active_model = provider.model
                self.last_thinking = provider.last_thinking
                return result
            except Exception as e:
                last_error = e
                provider_name = type(provider).__name__
                error_str = str(e)
                is_quota_error = "429" in error_str or "insufficient_quota" in error_str
                logger.warning("LLM provider %s failed: %s", provider_name, e)
                if provider is self._primary:
                    self._consecutive_failures += 1
                    backoff_threshold = 1 if is_quota_error else 3
                    if self._consecutive_failures >= backoff_threshold:
                        backoff_min = min(
                            2 ** (self._consecutive_failures - backoff_threshold),
                            self._max_backoff_minutes,
                        )
                        self._backoff_until = time.monotonic() + backoff_min * 60
                        logger.warning(
                            "Primary LLM failed %d times — backing off for %d minutes",
                            self._consecutive_failures, backoff_min,
                        )

        raise last_error or RuntimeError("No LLM providers available")

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Delegate to each provider's generate_json (preserving thinking-mode retry)."""
        providers: list[BaseLLM] = []

        now = time.monotonic()
        if now >= self._backoff_until:
            providers.append(self._primary)
        else:
            remaining = self._backoff_until - now
            logger.debug(
                "Primary LLM in backoff for %.0fs more, trying fallbacks", remaining
            )

        providers.extend(self._fallbacks)

        last_error: Exception | None = None
        for provider in providers:
            try:
                result = await provider.generate_json(prompt, system)
                if provider is self._primary:
                    self._consecutive_failures = 0
                    self._backoff_until = 0
                self._active_model = provider.model
                self.last_thinking = provider.last_thinking
                return result
            except Exception as e:
                last_error = e
                provider_name = type(provider).__name__
                error_str = str(e)
                is_quota_error = "429" in error_str or "insufficient_quota" in error_str
                logger.warning("LLM provider %s failed: %s", provider_name, e)
                if provider is self._primary:
                    self._consecutive_failures += 1
                    backoff_threshold = 1 if is_quota_error else 3
                    if self._consecutive_failures >= backoff_threshold:
                        backoff_min = min(
                            2 ** (self._consecutive_failures - backoff_threshold),
                            self._max_backoff_minutes,
                        )
                        self._backoff_until = time.monotonic() + backoff_min * 60
                        logger.warning(
                            "Primary LLM failed %d times — backing off for %d minutes",
                            self._consecutive_failures, backoff_min,
                        )

        raise last_error or RuntimeError("No LLM providers available")


def _create_single_llm(provider: LLMProvider, model: str, settings: Settings) -> BaseLLM | None:
    """Create a single LLM instance for a given provider, or None if unconfigured."""
    match provider:
        case LLMProvider.OLLAMA:
            return OllamaLLM(model=model, host=settings.ollama_host)
        case LLMProvider.OPENAI:
            if settings.openai_api_key:
                return OpenAILLM(model=model, api_key=settings.openai_api_key)
        case LLMProvider.ANTHROPIC:
            if settings.anthropic_api_key:
                return AnthropicLLM(model=model, api_key=settings.anthropic_api_key)
    return None


def create_llm(settings: Settings | None = None) -> BaseLLM:
    """Factory: create the appropriate LLM with automatic fallback chain."""
    if settings is None:
        settings = get_settings()

    primary = _create_single_llm(settings.llm_provider, settings.llm_model, settings)
    if primary is None:
        raise ValueError(
            f"Primary LLM provider {settings.llm_provider.value} is not configured"
        )

    fallback_models = {
        LLMProvider.OLLAMA: settings.ollama_fallback_model or settings.llm_model,
        LLMProvider.OPENAI: settings.openai_fallback_model or "gpt-4o-mini",
        LLMProvider.ANTHROPIC: settings.anthropic_fallback_model or "claude-sonnet-4-20250514",
    }

    fallbacks: list[BaseLLM] = []
    all_providers = [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.OLLAMA]
    for provider in all_providers:
        if provider == settings.llm_provider:
            continue
        model = fallback_models.get(provider, settings.llm_model)
        fb = _create_single_llm(provider, model, settings)
        if fb is not None:
            fallbacks.append(fb)

    if fallbacks:
        logger.info(
            "LLM fallback chain: %s -> %s",
            type(primary).__name__,
            " -> ".join(type(f).__name__ for f in fallbacks),
        )
        return FallbackLLM(primary, fallbacks)

    return primary
