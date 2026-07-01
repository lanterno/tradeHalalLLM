"""Provider-rotation + chain-backoff wrapper around BaseLLM."""

from __future__ import annotations

import logging
import time
from typing import Any

from halal_trader.core import events
from halal_trader.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


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
        self._chain_failures = 0
        self._chain_backoff_until: float = 0

    @property
    def model(self) -> str:
        return self._active_model

    @model.setter
    def model(self, value: str) -> None:
        self._active_model = value

    # ── Internal helpers ───────────────────────────────────────

    def _check_chain_backoff(self) -> None:
        now = time.monotonic()
        if now < self._chain_backoff_until:
            remaining = int(self._chain_backoff_until - now)
            raise RuntimeError(
                f"All LLM providers in backoff for {remaining}s more "
                f"after {self._chain_failures} consecutive full-chain failures"
            )

    def _eligible_providers(self) -> list[BaseLLM]:
        now = time.monotonic()
        providers: list[BaseLLM] = []
        if now >= self._backoff_until:
            providers.append(self._primary)
        else:
            backoff_remaining = self._backoff_until - now
            logger.debug(
                "Primary LLM in backoff for %.0fs more, trying fallbacks",
                backoff_remaining,
            )
        providers.extend(self._fallbacks)
        return providers

    def _on_provider_success(self, provider: BaseLLM) -> None:
        if provider is self._primary:
            self._consecutive_failures = 0
            self._backoff_until = 0
        self._chain_failures = 0
        self._chain_backoff_until = 0
        self._active_model = provider.model
        self.last_thinking = provider.last_thinking
        # Surface the chosen provider's usage so persistence sees the
        # real cost of the call that succeeded, not a stale primary one.
        self.last_usage = provider.last_usage

    def _on_provider_failure(self, provider: BaseLLM, error: Exception) -> None:
        provider_name = type(provider).__name__
        error_str = str(error)
        is_quota_error = "429" in error_str or "insufficient_quota" in error_str
        # ``str(error)`` is empty for several exception types (bare
        # timeouts / connection resets), which logged a useless
        # "LLM provider GLMLLM failed: " with no cause. Always surface
        # the exception type so transient blips are diagnosable.
        logger.warning("LLM provider %s failed: %s", provider_name, error_str or repr(error))
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
                    self._consecutive_failures,
                    backoff_min,
                )

    def _arm_chain_backoff(self) -> None:
        self._chain_failures += 1
        chain_backoff_sec = min(60 * 2 ** (self._chain_failures - 1), 1800)
        self._chain_backoff_until = time.monotonic() + chain_backoff_sec
        logger.error(
            "All LLM providers failed (%d consecutive) — chain backoff for %ds",
            self._chain_failures,
            chain_backoff_sec,
            extra={
                "event": events.LLM_CHAIN_BACKOFF,
                "consecutive": self._chain_failures,
                "backoff_seconds": chain_backoff_sec,
            },
        )

    # ── Public API ─────────────────────────────────────────────

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self._check_chain_backoff()
        last_error: Exception | None = None
        for provider in self._eligible_providers():
            try:
                result = await provider.generate(prompt, system)
                self._on_provider_success(provider)
                return result
            except Exception as e:
                last_error = e
                self._on_provider_failure(provider, e)
        self._arm_chain_backoff()
        raise last_error or RuntimeError("No LLM providers available")

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Delegate to each provider's generate_json (preserving thinking-mode retry)."""
        self._check_chain_backoff()
        last_error: Exception | None = None
        for provider in self._eligible_providers():
            try:
                result = await provider.generate_json(prompt, system)
                self._on_provider_success(provider)
                return result
            except Exception as e:
                last_error = e
                self._on_provider_failure(provider, e)
        self._arm_chain_backoff()
        raise last_error or RuntimeError("No LLM providers available")

    @property
    def supports_tool_use(self) -> bool:  # type: ignore[override]
        """Tool-use capability of whichever inner provider is eligible.

        Returns True if *any* eligible provider supports native tool use —
        the strategy uses this to decide whether to take the
        ``generate_tool_call`` path. Generators that don't support it
        still get tried via the JSON fallback inside
        :meth:`generate_tool_call`.
        """
        for provider in self._eligible_providers():
            if getattr(provider, "supports_tool_use", False):
                return True
        return False

    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: "list[Any]",
        system: str | None = None,
        force_tool: str | None = None,
    ) -> "list[Any]":
        """Delegate to each provider's generate_tool_call.

        Providers without native tool-use inherit the
        default ``BaseLLM.generate_tool_call`` implementation that
        materialises a single tool call from ``generate_json``. The
        Fallback layer doesn't care which path produced the call —
        downstream consumers see a uniform list of
        :class:`ToolCall` instances.
        """
        self._check_chain_backoff()
        last_error: Exception | None = None
        for provider in self._eligible_providers():
            try:
                result = await provider.generate_tool_call(
                    prompt,
                    tools=tools,
                    system=system,
                    force_tool=force_tool,
                )
                self._on_provider_success(provider)
                return result
            except Exception as e:
                last_error = e
                self._on_provider_failure(provider, e)
        self._arm_chain_backoff()
        raise last_error or RuntimeError("No LLM providers available")
