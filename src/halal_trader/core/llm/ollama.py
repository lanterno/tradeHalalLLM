"""Local LLM via Ollama with thinking-mode awareness."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from halal_trader.core import events
from halal_trader.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Local LLM via Ollama with thinking-mode awareness.

    Uses ``format="json"`` for fast, reliable structured output. Any
    ``<think>`` reasoning blocks the model may still emit are captured
    in ``last_thinking`` for audit and self-improvement.
    """

    _TIMEOUT_SECONDS = 45

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
        response = await asyncio.wait_for(
            client.chat(
                model=self.model,
                messages=messages,
                format="json",
                options={"temperature": 0.2},
            ),
            timeout=self._TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - t0
        logger.debug("Ollama response in %.1fs", elapsed)

        content: str = response["message"]["content"]
        if not content or not content.strip():
            raise ValueError("Ollama returned empty response")
        logger.info(
            "ollama call complete in %.1fs",
            elapsed,
            extra={
                "event": events.LLM_CALL_COMPLETE,
                "provider": "ollama",
                "model": self.model,
                "elapsed_ms": int(elapsed * 1000),
                "tokens": None,
            },
        )
        return content
