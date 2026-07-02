"""Base strategy — shared LLM analysis orchestration for all markets."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from halal_trader.core.llm.budget import LLMBudget
from halal_trader.core.llm.quota import is_quota_error
from halal_trader.core.tracing import tracer
from halal_trader.db.repos import LlmDecisionRepo
from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Wave H — knobs for the agentic tool-calling loop.

    Pass to ``_run_llm_analysis(agent=...)`` to drive the LLM through
    a bounded multi-turn conversation that ends with a terminal tool
    call. ``tools`` includes both the data-fetching tools (e.g.
    ``analyze_pair``, ``query_rag``) and the terminal one whose args
    become the raw dict the strategy's ``validate`` callback parses.
    """

    tools: list[Any] = field(default_factory=list)
    handlers: dict[str, Any] = field(default_factory=dict)
    terminal_tool: str = "submit_decisions"
    max_turns: int = 5
    max_seconds: float = 30.0


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
        repo: LlmDecisionRepo,
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
        # Optional operator alerter (AlertSink) — attached by the
        # composition root after construction (mirrors BaseLLM.attach_bus;
        # the crypto root builds its AlertSink after the strategy).
        self._alert_sink: Any | None = None

    def attach_alert_sink(self, sink: Any) -> None:
        """Wire the rate-limited operator AlertSink for LLM credit alerts.

        Without it, credit exhaustion on the strategy LLM only reaches
        the logs — the classifier's quota breaker alerts, but the
        strategy path (the one that actually trades) stayed silent
        through the 2026-06 OpenAI 429 storm.
        """
        self._alert_sink = sink

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
        tool: Any = None,
        agent: "AgentConfig | None" = None,
    ) -> Any:
        """Core analysis loop shared by all strategy subclasses.

        *validate*: ``callable(raw_dict) -> plan``
        *make_empty*: ``callable(error_msg) -> empty_plan``
        *extract_symbols*: ``callable(plan) -> list[str]``
        *count_actions*: ``callable(plan) -> dict``
        *prompt_version*: ``"name@hash"`` short form from the prompt registry,
            persisted on the LlmDecision row so each decision is replayable
            against the exact template that produced it.
        *tool* (Wave E): when provided AND ``llm.supports_tool_use`` is
            True, take the native tool-use path: the provider validates
            the schema before returning, schema-repair becomes a no-op,
            and output-token cost drops because the model doesn't emit
            JSON syntax characters. When omitted (or the backend lacks
            native tool use),
            falls back to the legacy ``generate_json`` + repair path.
        *agent* (Wave H): when provided AND ``llm.supports_tool_use``,
            drive the model through a bounded tool-calling loop
            instead of a single call. ``agent.terminal_tool``'s args
            become the raw dict ``validate`` consumes; the full
            transcript lands on the LlmDecision row for replay.
        """
        t0 = time.monotonic()
        use_agent = agent is not None and getattr(self._llm, "supports_tool_use", False)
        use_tool = (
            not use_agent and tool is not None and getattr(self._llm, "supports_tool_use", False)
        )
        transcript_dicts: list[dict[str, Any]] | None = None
        try:
            async with tracer.aspan(
                "strategy.llm_call",
                provider=self._llm_provider_name,
                model=getattr(self._llm, "model", ""),
                prompt_version=prompt_version or "",
            ):
                if use_agent:
                    assert agent is not None  # narrow for the type checker
                    initial_raw, transcript_dicts = await self._run_agentic(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        agent=agent,
                    )
                elif use_tool:
                    initial_raw = await self._call_tool(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        tool=tool,
                    )
                else:
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
                tool_transcript=transcript_dicts,
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
            if self._alert_sink is not None and is_quota_error(e):
                # Credit exhaustion is non-transient: without a top-up
                # every subsequent cycle degrades to a no-action plan.
                # AlertSink rate-limits per error_type (15-min window),
                # so 15-min/60s cycle cadences can't spam Telegram.
                try:
                    await self._alert_sink.notify(
                        "llm.quota_exhausted",
                        f"Strategy LLM out of credits "
                        f"({self._llm_provider_name}/{self._llm.model}): {e}. "
                        f"Cycles degrade to no-action plans until topped up.",
                    )
                except Exception as alert_err:  # noqa: BLE001 — alerting must never break the cycle
                    logger.warning("quota alert failed to send: %s", alert_err)
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

    async def _run_agentic(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        agent: "AgentConfig",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Drive the bounded tool-calling loop; return (terminal-args, transcript).

        Surfaces ``run_agent``'s transcript as a list of dicts ready to
        land on ``LlmDecision.tool_transcript`` (JSONB column). When
        the loop is forced to finalise on a budget exhaustion *and*
        the model refuses to submit, the returned dict is empty and
        the strategy's downstream validate path will fail-then-empty
        through ``make_empty`` — matching the single-call empty-plan
        contract.
        """
        from dataclasses import asdict

        from halal_trader.core.llm.agent import run_agent

        result = await run_agent(
            self._llm,
            system=system_prompt,
            user=user_prompt,
            tools=list(agent.tools),
            handlers=dict(agent.handlers),
            terminal_tool=agent.terminal_tool,
            max_turns=agent.max_turns,
            max_seconds=agent.max_seconds,
        )
        transcript_dicts: list[dict[str, Any]] = [asdict(t) for t in result.transcript]
        # Annotate the transcript so the dashboard can render whether
        # the loop completed normally or was force-finalised.
        if result.budget_exhausted:
            transcript_dicts.append(
                {
                    "turn": len(transcript_dicts) + 1,
                    "tool_name": "_budget_exhausted",
                    "args": {},
                    "result_text": None,
                    "elapsed_ms": result.elapsed_ms,
                    "error": "max_turns or wall-clock budget hit",
                }
            )
        return dict(result.final_call.args or {}), transcript_dicts

    async def _call_tool(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        tool: Any,
    ) -> dict[str, Any]:
        """Wave E tool-use call: invoke ``tool`` and return its args dict.

        The provider enforces the JSONSchema before returning, so the
        result is structurally valid (Pydantic validation still runs
        downstream — semantic checks like halal-pair gating happen
        post-parse). ``force_tool`` is passed so the model can't reply
        with prose; it must call the tool.
        """
        calls = await self._llm.generate_tool_call(
            user_prompt,
            tools=[tool],
            system=system_prompt,
            force_tool=tool.name,
        )
        if not calls:
            raise ValueError(f"LLM returned no tool calls (expected {tool.name!r})")
        # Prefer the matching tool call if multiple are returned; otherwise
        # take the first. The strategy schema requires exactly one
        # ``submit_*`` call per cycle so this is a defensive guard.
        match = next((c for c in calls if c.name == tool.name), calls[0])
        return dict(match.args or {})

    def _on_llm_success(self) -> None:
        """Hook for subclasses to react to a successful LLM call (e.g. reset counters)."""

    def _on_llm_failure(self, error: Exception, elapsed_ms: int, prefix: str) -> None:
        """Hook for subclasses to react to a failed LLM call (e.g. circuit breaker)."""
        logger.error("%s analysis failed after %dms: %s", prefix, elapsed_ms, error)
