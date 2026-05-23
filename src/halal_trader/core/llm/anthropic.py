"""Cloud LLM via Anthropic API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from halal_trader.core import events
from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.core.llm.pricing import compute_cost_usd

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """Cloud LLM via Anthropic API."""

    # Wave E: Anthropic Messages speaks native tool use via the
    # ``tools=[...]`` + ``tool_choice`` params.
    supports_tool_use = True

    _TIMEOUT_SECONDS = 30

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        enable_prompt_cache: bool = True,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(model, temperature=temperature)
        self.api_key = api_key
        self._client: Any = None
        # Caching is on by default — there is no downside on Anthropic's
        # billing model (cache reads are cheaper than uncached input)
        # and the only requirement is that the system prompt is reused.
        # Disable in tests that need deterministic single-call behavior.
        self._enable_prompt_cache = enable_prompt_cache

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    def _system_payload(self, system: str | None) -> Any:
        """Build the ``system`` argument, opting into ephemeral cache when enabled.

        Returns a structured list when caching is on (so we can attach
        cache_control), otherwise the plain string the SDK expects.
        """
        text = system or ""
        if not text or not self._enable_prompt_cache:
            return text
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()

        t0 = time.monotonic()
        response = await asyncio.wait_for(
            client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=self._system_payload(system),
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=self._TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - t0

        usage = CallUsage(provider="anthropic", model=self.model, elapsed_ms=int(elapsed * 1000))
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            usage.input_tokens = getattr(u, "input_tokens", 0) or 0
            usage.output_tokens = getattr(u, "output_tokens", 0) or 0
            usage.cache_read_tokens = getattr(u, "cache_read_input_tokens", 0) or 0
            usage.cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0) or 0
            usage.cost_usd = compute_cost_usd(
                self.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        self._record_usage(usage)

        logger.info(
            "anthropic call complete in %.1fs (tokens=%d, cache_read=%d, cost=$%s)",
            elapsed,
            usage.total_tokens,
            usage.cache_read_tokens,
            f"{usage.cost_usd:.4f}",
            extra={
                "event": events.LLM_CALL_COMPLETE,
                "provider": "anthropic",
                "model": self.model,
                "elapsed_ms": usage.elapsed_ms,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "cost_usd": float(usage.cost_usd),
            },
        )
        text: str = response.content[0].text
        return text

    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[Any],
        system: str | None = None,
        force_tool: str | None = None,
    ) -> list[Any]:
        """Anthropic-native tool-use: returns one ToolCall per emitted tool block."""
        from halal_trader.core.llm.tools import ToolCall

        client = self._get_client()
        anthropic_tools = [t.for_anthropic() for t in tools]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "system": self._system_payload(system),
            "messages": [{"role": "user", "content": prompt}],
            "tools": anthropic_tools,
        }
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        t0 = time.monotonic()
        response = await asyncio.wait_for(
            client.messages.create(**kwargs),
            timeout=self._TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - t0

        usage = CallUsage(provider="anthropic", model=self.model, elapsed_ms=int(elapsed * 1000))
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            usage.input_tokens = getattr(u, "input_tokens", 0) or 0
            usage.output_tokens = getattr(u, "output_tokens", 0) or 0
            usage.cache_read_tokens = getattr(u, "cache_read_input_tokens", 0) or 0
            usage.cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0) or 0
            usage.cost_usd = compute_cost_usd(
                self.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        self._record_usage(usage)

        calls: list[ToolCall] = []
        for block in response.content:
            if getattr(block, "type", "") == "tool_use":
                calls.append(
                    ToolCall(
                        name=block.name,
                        args=dict(block.input or {}),
                        id=getattr(block, "id", None),
                    )
                )
        return calls
