"""Cloud LLM via OpenAI API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from halal_trader.core import events
from halal_trader.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """Cloud LLM via OpenAI API."""

    _TIMEOUT_SECONDS = 30

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
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            ),
            timeout=self._TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - t0
        tokens = response.usage.total_tokens if response.usage else None
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

        logger.info(
            "openai call complete in %.1fs (tokens=%s)",
            elapsed,
            tokens,
            extra={
                "event": events.LLM_CALL_COMPLETE,
                "provider": "openai",
                "model": self.model,
                "elapsed_ms": int(elapsed * 1000),
                "tokens": tokens,
            },
        )
        return response.choices[0].message.content or ""
