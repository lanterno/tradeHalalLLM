"""Cloud LLM via OpenAI API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from halal_trader.core import events
from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.core.llm.pricing import compute_cost_usd

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

        usage = CallUsage(provider="openai", model=self.model, elapsed_ms=int(elapsed * 1000))
        if response.usage:
            u = response.usage
            usage.input_tokens = getattr(u, "prompt_tokens", 0) or 0
            usage.output_tokens = getattr(u, "completion_tokens", 0) or 0
            # OpenAI surfaces cached prompt tokens via the prompt_tokens_details
            # nested object on chat completions. The category is bundled inside
            # input_tokens (not in addition to), so subtract for accurate cost.
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
                usage.cache_read_tokens = cached
                usage.input_tokens = max(0, usage.input_tokens - cached)
            usage.cost_usd = compute_cost_usd(
                self.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
            self._track_usage(usage.total_tokens)
        self.last_usage = usage

        logger.info(
            "openai call complete in %.1fs (tokens=%d, cache_read=%d, cost=$%s)",
            elapsed,
            usage.total_tokens,
            usage.cache_read_tokens,
            f"{usage.cost_usd:.4f}",
            extra={
                "event": events.LLM_CALL_COMPLETE,
                "provider": "openai",
                "model": self.model,
                "elapsed_ms": usage.elapsed_ms,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "cost_usd": float(usage.cost_usd),
            },
        )
        return response.choices[0].message.content or ""
