"""Agentic multi-turn tool-calling loop with bounded budget.

Wave H — instead of one prompt → one decision, the LLM gets a
toolbelt and decides whether it needs more context before
committing. ``submit_plan`` is the terminal tool: emitting it ends
the loop and we materialise the plan from its arguments. The other
tools (``analyze_pair``, ``query_rag``, ``compute_var_95``) feed
extra context back into the model.

Bounds:
* ``max_turns`` — hard cap on tool calls per cycle (default 5).
* ``max_seconds`` — wall-clock cap per cycle (default 30s).
* On budget exhaustion, the loop forces ``submit_plan`` with the
  partial context the model has accumulated.

The whole transcript is persisted to ``LlmDecision.tool_transcript``
so the dashboard can render the agent's chain-of-thought per cycle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.tools import Tool, ToolCall

logger = logging.getLogger(__name__)


ToolHandler = Callable[[ToolCall], Awaitable[str]]
"""Caller-supplied function that handles one tool call and returns
a string the LLM will see on its next turn."""


@dataclass
class AgentTurn:
    """One round-trip through the loop — kept for the transcript."""

    turn: int
    tool_name: str
    args: dict[str, Any]
    result_text: str | None = None
    elapsed_ms: float = 0.0
    error: str | None = None


@dataclass
class AgentResult:
    """Outcome of a complete agentic loop."""

    final_call: ToolCall  # always submit_plan in success / forced-finalise
    transcript: list[AgentTurn] = field(default_factory=list)
    budget_exhausted: bool = False
    elapsed_ms: float = 0.0


_DEFAULT_MAX_TURNS = 5
_DEFAULT_MAX_SECONDS = 30.0


async def run_agent(
    llm: BaseLLM,
    *,
    system: str,
    user: str,
    tools: list[Tool],
    handlers: dict[str, ToolHandler],
    terminal_tool: str = "submit_plan",
    max_turns: int = _DEFAULT_MAX_TURNS,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> AgentResult:
    """Drive the LLM through a bounded tool-calling conversation.

    On each turn:
      1. Ask the model for a tool call (constrained to ``tools``).
      2. If it's the terminal tool, return its args as the result.
      3. Otherwise dispatch to the handler, append the result to the
         conversation, and loop.

    Budget exhaustion forces the terminal tool — we never return
    without a final ``submit_plan`` call. If the model refuses to
    submit even when forced, the result.final_call has empty args
    and the caller can choose how to fall back.
    """
    transcript: list[AgentTurn] = []
    history: list[dict[str, Any]] = [{"role": "user", "content": user}]
    t0 = time.monotonic()

    for turn in range(1, max_turns + 1):
        elapsed = time.monotonic() - t0
        if elapsed >= max_seconds:
            return _force_finalise(
                llm,
                system=system,
                history=history,
                tools=tools,
                terminal_tool=terminal_tool,
                transcript=transcript,
                start=t0,
                reason="wall_clock_exhausted",
            )

        try:
            calls = await asyncio.wait_for(
                llm.generate_tool_call(
                    _flatten(history),
                    tools=tools,
                    system=system,
                ),
                timeout=max_seconds - elapsed,
            )
        except asyncio.TimeoutError:
            return _force_finalise(
                llm,
                system=system,
                history=history,
                tools=tools,
                terminal_tool=terminal_tool,
                transcript=transcript,
                start=t0,
                reason="generate_timeout",
            )

        if not calls:
            return AgentResult(
                final_call=ToolCall(name=terminal_tool, args={}),
                transcript=transcript,
                budget_exhausted=False,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
            )

        call = calls[0]  # one call at a time keeps the budget interpretable
        if call.name == terminal_tool:
            return AgentResult(
                final_call=call,
                transcript=transcript,
                budget_exhausted=False,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
            )

        handler = handlers.get(call.name)
        if handler is None:
            transcript.append(
                AgentTurn(
                    turn=turn,
                    tool_name=call.name,
                    args=dict(call.args),
                    error=f"no handler for tool {call.name!r}",
                )
            )
            history.append(
                {
                    "role": "tool_result",
                    "tool": call.name,
                    "content": f"Error: tool {call.name!r} is not available.",
                }
            )
            continue

        turn_t0 = time.monotonic()
        try:
            result_text = await handler(call)
            err: str | None = None
        except Exception as exc:  # noqa: BLE001
            result_text = f"Error: {exc}"
            err = repr(exc)
        elapsed_ms = (time.monotonic() - turn_t0) * 1000.0

        transcript.append(
            AgentTurn(
                turn=turn,
                tool_name=call.name,
                args=dict(call.args),
                result_text=result_text,
                elapsed_ms=elapsed_ms,
                error=err,
            )
        )
        history.append({"role": "tool_result", "tool": call.name, "content": result_text})

    # Out of turns — force a terminal call.
    return await _force_finalise_async(
        llm,
        system=system,
        history=history,
        tools=tools,
        terminal_tool=terminal_tool,
        transcript=transcript,
        start=t0,
        reason="max_turns_exhausted",
    )


# ── Helpers ──────────────────────────────────────────────────────


def _flatten(history: list[dict[str, Any]]) -> str:
    """Render the conversation as a single user prompt for providers
    that don't yet support a multi-turn tool history natively.

    For Anthropic with tool use, the SDK ideally accepts a list of
    role-tagged messages. The MVP just flattens everything to one
    ``user`` blob — a future enhancement upgrades AnthropicLLM to
    pass the structured history directly.
    """
    parts = []
    for msg in history:
        role = msg.get("role", "user")
        if role == "tool_result":
            parts.append(f"[tool {msg.get('tool', '?')} result]\n{msg.get('content', '')}")
        else:
            parts.append(str(msg.get("content", "")))
    return "\n\n".join(parts)


def _force_finalise(
    llm: BaseLLM,
    *,
    system: str,
    history: list[dict[str, Any]],
    tools: list[Tool],
    terminal_tool: str,
    transcript: list[AgentTurn],
    start: float,
    reason: str,
) -> AgentResult:
    """Synchronous version used on hard timeouts — returns empty plan."""
    logger.warning("agent loop forcing finalise: %s", reason)
    return AgentResult(
        final_call=ToolCall(name=terminal_tool, args={}),
        transcript=transcript,
        budget_exhausted=True,
        elapsed_ms=(time.monotonic() - start) * 1000.0,
    )


async def _force_finalise_async(
    llm: BaseLLM,
    *,
    system: str,
    history: list[dict[str, Any]],
    tools: list[Tool],
    terminal_tool: str,
    transcript: list[AgentTurn],
    start: float,
    reason: str,
) -> AgentResult:
    """Issue one more call constrained to ``terminal_tool`` after budget exit.

    Falls back to an empty plan when even the forced call fails.
    """
    logger.info("agent loop forcing %s after %s", terminal_tool, reason)
    try:
        calls = await llm.generate_tool_call(
            _flatten(history) + "\n\n[Budget exhausted — submit your plan now.]",
            tools=tools,
            system=system,
            force_tool=terminal_tool,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("forced finalise failed: %s", exc)
        return _force_finalise(
            llm,
            system=system,
            history=history,
            tools=tools,
            terminal_tool=terminal_tool,
            transcript=transcript,
            start=start,
            reason=f"force_failed:{reason}",
        )
    final = calls[0] if calls else ToolCall(name=terminal_tool, args={})
    return AgentResult(
        final_call=final,
        transcript=transcript,
        budget_exhausted=True,
        elapsed_ms=(time.monotonic() - start) * 1000.0,
    )
