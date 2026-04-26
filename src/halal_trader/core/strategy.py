"""Base strategy — shared LLM analysis orchestration for all markets."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC
from typing import Any

from pydantic import ValidationError

from halal_trader.core.llm.budget import LLMBudget
from halal_trader.core.tracing import tracer
from halal_trader.db.repository import Repository
from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


# When the LLM returns JSON that fails our schema validation we get one
# free repair attempt — a follow-up call that includes the raw bad
# output plus the validation error and asks for a corrected blob.
# Empirically this rescues most "forgot a required field" failures
# without doubling the cost on the typical success path.
_REPAIR_INSTRUCTION = (
    "Your previous response could not be parsed against the required schema. "
    "Return ONLY a single JSON object that fixes the validation errors below. "
    "Do not include any prose, markdown, or explanation — just the corrected JSON. "
    "Validation errors:\n{errors}\n\n"
    "Your previous response:\n{previous}"
)


class BaseStrategy(ABC):
    """Extract the shared LLM → validate → record → return flow.

    Subclasses implement prompt building and plan validation;
    the orchestration (timing, audit trail, error handling) lives here.
    """

    def __init__(
        self,
        llm: LLMBackend,
        repo: Repository,
        *,
        llm_provider_name: str,
        max_position_pct: float,
        daily_loss_limit: float,
        daily_return_target: float,
        max_simultaneous_positions: int,
        llm_budget: LLMBudget | None = None,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._llm_provider_name = llm_provider_name
        self._max_position_pct = max_position_pct
        self._daily_loss_limit = daily_loss_limit
        self._daily_return_target = daily_return_target
        self._max_simultaneous_positions = max_simultaneous_positions
        self._llm_budget = llm_budget

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
        prompt_version: str | None = None,
    ) -> Any:
        """Core analysis loop shared by all strategy subclasses.

        *validate*: ``callable(raw_dict) -> plan``
        *make_empty*: ``callable(error_msg) -> empty_plan``
        *extract_symbols*: ``callable(plan) -> list[str]``
        *count_actions*: ``callable(plan) -> dict``
        *prompt_version*: ``"name@hash"`` short form from the prompt registry,
            persisted on the LlmDecision row so each decision is replayable
            against the exact template that produced it.
        """
        t0 = time.monotonic()
        try:
            async with tracer.aspan(
                "strategy.llm_call",
                provider=self._llm_provider_name,
                model=getattr(self._llm, "model", ""),
                prompt_version=prompt_version or "",
            ):
                initial_raw = await self._llm.generate_json(user_prompt, system=system_prompt)
            async with tracer.aspan("strategy.validate", log_prefix=log_prefix):
                plan, raw, repair_used = await self._validate_with_repair(
                    raw=initial_raw,
                    validate=validate,
                    system_prompt=system_prompt,
                    log_prefix=log_prefix,
                )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            if repair_used:
                logger.info(
                    "%s schema-repair pass succeeded — invalid JSON corrected on retry",
                    log_prefix,
                )

            usage = getattr(self._llm, "last_usage", None)
            if usage and self._llm_budget is not None and usage.cost_usd:
                await self._llm_budget.record(usage.cost_usd)
            await self._repo.record_decision(
                provider=self._llm_provider_name,
                model=self._llm.model,
                prompt_summary=prompt_summary,
                raw_response=json.dumps(raw),
                parsed_action=count_actions(plan),
                symbols=extract_symbols(plan),
                execution_ms=elapsed_ms,
                thinking=getattr(self._llm, "last_thinking", None) or None,
                prompt_version=prompt_version,
                input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                cache_read_tokens=getattr(usage, "cache_read_tokens", None) if usage else None,
                cache_write_tokens=getattr(usage, "cache_write_tokens", None) if usage else None,
                cost_usd=float(usage.cost_usd) if usage and usage.cost_usd else None,
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
                prompt_version=prompt_version,
            )
            return make_empty(str(e))

    async def _validate_with_repair(
        self,
        *,
        raw: dict[str, Any],
        validate: Any,
        system_prompt: str,
        log_prefix: str,
    ) -> tuple[Any, dict[str, Any], bool]:
        """Validate ``raw`` and, on schema failure, get one repair attempt.

        Network/transport errors propagate untouched — they're handled by
        the broad ``except`` in :meth:`_run_llm_analysis`. Only schema
        errors (``ValidationError``) and JSON-shape errors (``ValueError``,
        ``TypeError``) trigger the repair pass, because that's where a
        retry can actually help.
        """
        try:
            return validate(raw), raw, False
        except (ValidationError, ValueError, TypeError) as schema_err:
            logger.warning(
                "%s LLM output failed schema validation — attempting one repair pass: %s",
                log_prefix,
                schema_err,
            )
            repair_prompt = _REPAIR_INSTRUCTION.format(
                errors=str(schema_err),
                previous=json.dumps(raw),
            )
            try:
                repaired = await self._llm.generate_json(repair_prompt, system=system_prompt)
            except Exception as repair_err:
                # Network/parse failure on the repair call — surface the
                # *original* validation error since that's the actionable
                # signal; the repair-call failure is logged separately.
                logger.warning(
                    "%s schema-repair call failed (%s); falling back to empty plan",
                    log_prefix,
                    repair_err,
                )
                raise schema_err from repair_err
            # Re-validate the repaired blob; bubble up the new error if
            # it still fails so the caller's ``except`` records it.
            return validate(repaired), repaired, True

    def _on_llm_success(self) -> None:
        """Hook for subclasses to react to a successful LLM call (e.g. reset counters)."""

    def _on_llm_failure(self, error: Exception, elapsed_ms: int, prefix: str) -> None:
        """Hook for subclasses to react to a failed LLM call (e.g. circuit breaker)."""
        logger.error("%s analysis failed after %dms: %s", prefix, elapsed_ms, error)
