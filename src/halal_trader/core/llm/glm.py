"""GLM-5.2 via any OpenAI-compatible endpoint (default: OpenRouter).

The bot's single LLM provider. GLM-5.2's weights are MIT-licensed and
served by many independent hosts, so the same provider class covers
OpenRouter (multi-host failover), Z.ai direct, Fireworks, Together, …
— only ``base_url`` (and the model id naming convention) changes.

Endpoint dialect differences handled here:

* **Thinking toggle** — OpenRouter normalises it as
  ``reasoning: {"enabled": bool}``; Z.ai-style endpoints use
  ``thinking: {"type": "enabled"|"disabled"}``. GLM-5.2 thinks by
  default, which blows the cycle latency budget, so the bot defaults
  it OFF.
* **OpenRouter provider routing** — ``provider.require_parameters``
  restricts routing to hosts that honour every request param
  (``response_format`` + ``tools``); without it a request can land on
  a host that silently drops JSON mode or tool calls.
* **Temperature 0.0** — Z.ai historically rejects exactly 0.0, so it
  is clamped to 0.01 on non-OpenRouter endpoints (the classifier stack
  pins temperature 0.0 for reproducible scores).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from halal_trader.core import events
from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.core.llm.pricing import compute_cost_usd
from halal_trader.core.llm.tools import ToolCall

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class GLMLLM(BaseLLM):
    """GLM-5.2 through an OpenAI-compatible chat-completions endpoint."""

    supports_tool_use = True

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.2,
        timeout_seconds: int = 60,
        thinking: bool = False,
        require_parameters: bool = True,
    ) -> None:
        super().__init__(model, temperature=temperature)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.thinking = thinking
        self.require_parameters = require_parameters
        self._client: Any = None

    # ── Endpoint dialect ───────────────────────────────────────

    def _is_openrouter(self) -> bool:
        return "openrouter" in self.base_url

    def _effective_temperature(self) -> float:
        if not self._is_openrouter() and self.temperature == 0.0:
            return 0.01
        return self.temperature

    def _extra_body(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self._is_openrouter():
            extra["reasoning"] = {"enabled": self.thinking}
            if self.require_parameters:
                extra["provider"] = {"require_parameters": True}
        else:
            extra["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        return extra

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # ── Usage accounting ───────────────────────────────────────

    def _usage_from_response(self, response: Any, elapsed: float) -> CallUsage:
        usage = CallUsage(provider="glm", model=self.model, elapsed_ms=int(elapsed * 1000))
        if response.usage:
            u = response.usage
            usage.input_tokens = getattr(u, "prompt_tokens", 0) or 0
            usage.output_tokens = getattr(u, "completion_tokens", 0) or 0
            # Cached prompt tokens arrive nested (OpenAI-compat shape) and
            # are bundled inside input_tokens, so subtract for exact cost.
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
        self._record_usage(usage)
        return usage

    # ── Generation ─────────────────────────────────────────────

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
                temperature=self._effective_temperature(),
                extra_body=self._extra_body(),
            ),
            timeout=self.timeout_seconds,
        )
        elapsed = time.monotonic() - t0
        usage = self._usage_from_response(response, elapsed)

        logger.info(
            "glm call complete in %.1fs (tokens=%d, cache_read=%d, cost=$%s)",
            elapsed,
            usage.total_tokens,
            usage.cache_read_tokens,
            f"{usage.cost_usd:.4f}",
            extra={
                "event": events.LLM_CALL_COMPLETE,
                "provider": "glm",
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

    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[Any],
        system: str | None = None,
        force_tool: str | None = None,
    ) -> list[Any]:
        """Native tool use with forced tool_choice (the strategy hot path).

        The endpoint enforces the schema and the SDK parses the
        arguments; a host that mishandles forced tool_choice yields an
        empty list here, which the strategy layer treats as a failed
        call (no-action plan), never a crash.
        """
        client = self._get_client()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [t.for_openai() for t in tools],
            "temperature": self._effective_temperature(),
            "extra_body": self._extra_body(),
        }
        if force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}

        t0 = time.monotonic()
        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=self.timeout_seconds,
        )
        elapsed = time.monotonic() - t0
        self._usage_from_response(response, elapsed)

        calls: list[ToolCall] = []
        for choice in response.choices:
            msg = choice.message
            for tc in getattr(msg, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                try:
                    args = json.loads(fn.arguments or "{}")
                except json.JSONDecodeError, TypeError:
                    args = {}
                calls.append(ToolCall(name=fn.name, args=args, id=getattr(tc, "id", None)))
        return calls
