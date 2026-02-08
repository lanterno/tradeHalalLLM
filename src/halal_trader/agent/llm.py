"""LLM provider abstraction supporting Ollama, OpenAI, and Anthropic."""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from halal_trader.config import LLMProvider, Settings, get_settings

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt and return the raw text response."""
        ...

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Generate a response and parse it as JSON."""
        raw = await self.generate(prompt, system)
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)


class OllamaLLM(BaseLLM):
    """Local LLM via Ollama."""

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
        logger.debug("Anthropic response in %.1fs", elapsed)

        return response.content[0].text


def create_llm(settings: Settings | None = None) -> BaseLLM:
    """Factory: create the appropriate LLM based on configuration."""
    if settings is None:
        settings = get_settings()

    match settings.llm_provider:
        case LLMProvider.OLLAMA:
            return OllamaLLM(model=settings.llm_model, host=settings.ollama_host)
        case LLMProvider.OPENAI:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            return OpenAILLM(model=settings.llm_model, api_key=settings.openai_api_key)
        case LLMProvider.ANTHROPIC:
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
            return AnthropicLLM(model=settings.llm_model, api_key=settings.anthropic_api_key)
        case _:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
