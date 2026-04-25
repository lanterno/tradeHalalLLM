"""Base strategy — shared LLM analysis orchestration for all markets."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC
from typing import Any

from halal_trader.domain.ports import LLMBackend, TradeRepository

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Extract the shared LLM → validate → record → return flow.

    Subclasses implement prompt building and plan validation;
    the orchestration (timing, audit trail, error handling) lives here.
    """

    def __init__(
        self,
        llm: LLMBackend,
        repo: TradeRepository,
        *,
        llm_provider_name: str,
        max_position_pct: float,
        daily_loss_limit: float,
        daily_return_target: float,
        max_simultaneous_positions: int,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._llm_provider_name = llm_provider_name
        self._max_position_pct = max_position_pct
        self._daily_loss_limit = daily_loss_limit
        self._daily_return_target = daily_return_target
        self._max_simultaneous_positions = max_simultaneous_positions

    async def _run_llm_analysis(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        prompt_summary: str,
        validate: Any,
        make_empty: Any,
        extract_symbols: Any,
        count_actions: Any,
        log_prefix: str = "LLM",
    ) -> Any:
        """Core analysis loop shared by all strategy subclasses.

        *validate*: ``callable(raw_dict) -> plan``
        *make_empty*: ``callable(error_msg) -> empty_plan``
        *extract_symbols*: ``callable(plan) -> list[str]``
        *count_actions*: ``callable(plan) -> dict``
        """
        t0 = time.monotonic()
        try:
            raw = await self._llm.generate_json(user_prompt, system=system_prompt)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            plan = validate(raw)

            await self._repo.record_decision(
                provider=self._llm_provider_name,
                model=self._llm.model,
                prompt_summary=prompt_summary,
                raw_response=json.dumps(raw),
                parsed_action=count_actions(plan),
                symbols=extract_symbols(plan),
                execution_ms=elapsed_ms,
                thinking=getattr(self._llm, "last_thinking", None) or None,
            )

            logger.info(
                "%s analysis complete in %dms: %s",
                log_prefix,
                elapsed_ms,
                ", ".join(f"{k} {v}" for k, v in count_actions(plan).items()),
            )
            self._on_llm_success()
            return plan

        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self._on_llm_failure(e, elapsed_ms, log_prefix)
            await self._repo.record_decision(
                provider=self._llm_provider_name,
                model=self._llm.model,
                prompt_summary=f"FAILED {log_prefix.lower()} analysis",
                raw_response=str(e),
                execution_ms=elapsed_ms,
            )
            return make_empty(str(e))

    def _on_llm_success(self) -> None:
        """Hook for subclasses to react to a successful LLM call (e.g. reset counters)."""

    def _on_llm_failure(self, error: Exception, elapsed_ms: int, prefix: str) -> None:
        """Hook for subclasses to react to a failed LLM call (e.g. circuit breaker)."""
        logger.error("%s analysis failed after %dms: %s", prefix, elapsed_ms, error)
