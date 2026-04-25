"""Cloud LLM via Anthropic API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from halal_trader.core import events
from halal_trader.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """Cloud LLM via Anthropic API."""

    _TIMEOUT_SECONDS = 30

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
        response = await asyncio.wait_for(
            client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=self._TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - t0
        tokens: int | None = None
        if hasattr(response, "usage") and response.usage:
            tokens = response.usage.input_tokens + response.usage.output_tokens
            logger.debug(
                "Anthropic response in %.1fs — input: %d, output: %d tokens",
                elapsed,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            self._track_usage(tokens)
        else:
            logger.debug("Anthropic response in %.1fs", elapsed)

        logger.info(
            "anthropic call complete in %.1fs (tokens=%s)",
            elapsed,
            tokens,
            extra={
                "event": events.LLM_CALL_COMPLETE,
                "provider": "anthropic",
                "model": self.model,
                "elapsed_ms": int(elapsed * 1000),
                "tokens": tokens,
            },
        )
        text: str = response.content[0].text
        return text
