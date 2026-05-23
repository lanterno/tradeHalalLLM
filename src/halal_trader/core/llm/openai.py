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

    # Wave E: OpenAI chat completions speak native tool use via the
    # ``tools=[...]`` + ``tool_choice`` params.
    supports_tool_use = True

    _DEFAULT_TIMEOUT_SECONDS = 30
    # Reasoning models think before responding; observed latencies on
    # gpt-5.5 are 15–28s for our prompt, regularly bumping the 30s
    # ceiling. The crypto scheduler caps each cycle at 2× the trading
    # interval (default 60s → 120s cycle budget), so we stop below that
    # so a stuck LLM call still leaves a margin for the rest of the
    # pipeline (indicators, halal check, executor).
    _REASONING_TIMEOUT_SECONDS = 90

    # Reasoning models (o1, o3, gpt-5, gpt-5.5, …) only accept the
    # default temperature; passing 0.2 returns a 400. Match by prefix
    # so future variants are picked up without code changes. Same
    # prefixes also get the extended timeout.
    _REASONING_MODEL_PREFIXES = ("o1", "o3", "gpt-5")

    def __init__(self, model: str, api_key: str, *, temperature: float = 0.2) -> None:
        super().__init__(model, temperature=temperature)
        self.api_key = api_key
        self._client: Any = None

    def _is_reasoning_model(self) -> bool:
        m = self.model.lower()
        return any(m.startswith(p) for p in self._REASONING_MODEL_PREFIXES)

    def _accepts_custom_temperature(self) -> bool:
        return not self._is_reasoning_model()

    def _timeout_seconds(self) -> int:
        return (
            self._REASONING_TIMEOUT_SECONDS
            if self._is_reasoning_model()
            else self._DEFAULT_TIMEOUT_SECONDS
        )

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

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self._accepts_custom_temperature():
            kwargs["temperature"] = self.temperature

        t0 = time.monotonic()
        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=self._timeout_seconds(),
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
        self._record_usage(usage)

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

    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[Any],
        system: str | None = None,
        force_tool: str | None = None,
    ) -> list[Any]:
        """OpenAI-native tool use: model returns one or more tool_calls.

        Avoids the round-trip of asking for JSON and schema-repairing —
        the API enforces the schema and the SDK parses the arguments
        for us. Output-token cost drops because the model doesn't emit
        JSON syntax characters as text.
        """
        import json

        from halal_trader.core.llm.tools import ToolCall

        client = self._get_client()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        openai_tools = [t.for_openai() for t in tools]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": openai_tools,
        }
        if force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        if self._accepts_custom_temperature():
            kwargs["temperature"] = self.temperature

        t0 = time.monotonic()
        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=self._timeout_seconds(),
        )
        elapsed = time.monotonic() - t0

        usage = CallUsage(provider="openai", model=self.model, elapsed_ms=int(elapsed * 1000))
        if response.usage:
            u = response.usage
            usage.input_tokens = getattr(u, "prompt_tokens", 0) or 0
            usage.output_tokens = getattr(u, "completion_tokens", 0) or 0
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
