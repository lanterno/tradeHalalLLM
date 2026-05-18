"""Base LLM contract + thinking-mode helpers shared by every provider."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_LOG_THRESHOLDS = (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class CallUsage:
    """Token + cost breakdown for a single LLM call.

    Populated by every provider on its instance after each ``generate()``
    call so callers can persist the numbers alongside the LlmDecision row.
    The fields mirror what Anthropic and OpenAI surface in their usage
    objects — providers that don't report a category (e.g. cache tokens
    on OpenAI today) leave it at zero.
    """

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    elapsed_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
        self.last_usage: CallUsage = CallUsage(model=model)
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
                    threshold // 1000,
                    self._daily_tokens,
                    self.model,
                )
                break

    def _record_usage(self, usage: CallUsage) -> None:
        """Stamp ``last_usage``, roll up daily tokens, emit Prometheus.

        Every provider's ``generate()`` should call this exactly once
        per successful call so the per-call latency histogram in
        ``core/metrics.halal_trader_llm_call_ms`` matches what the LLM
        cost-roll-up sees. Providers that report 0 tokens still get the
        latency observation (a stuck Ollama call should show up in p95
        even if it returned nothing).
        """
        from halal_trader.core.metrics import observe_llm_call

        self.last_usage = usage
        if usage.total_tokens:
            self._track_usage(usage.total_tokens)
        if usage.elapsed_ms and usage.provider and usage.model:
            observe_llm_call(
                provider=usage.provider,
                model=usage.model,
                ms=float(usage.elapsed_ms),
            )

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
        parsed: dict[str, Any] = json.loads(_clean_json_body(body))
        return parsed

    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: "list[Any]",
        system: str | None = None,
        force_tool: str | None = None,
    ) -> "list[Any]":
        """Provider-native tool-use call. Returns ToolCall instances.

        Default implementation falls back to ``generate_json`` and
        materialises a single tool call from the parsed JSON, matching
        ``force_tool`` if provided. Provider subclasses (Anthropic,
        OpenAI) override with native tool-use semantics.
        """
        from halal_trader.core.llm.tools import ToolCall

        target_tool = force_tool or (tools[0].name if tools else "submit_plan")
        # Fallback path: just ask for JSON matching the tool's schema and wrap it.
        parsed = await self.generate_json(prompt, system)
        return [ToolCall(name=target_tool, args=parsed)]
