"""Budget + edge-case tests for the agentic loop in :mod:`core.llm.agent`.

`test_llm_agent.py` covers the happy paths (terminate-on-submit, tool
dispatch, max-turns force, handler-raises). This file pins the
remaining branches:

* Empty `generate_tool_call` response → return-without-budget-exhausted.
* Missing handler for a tool name the model called → record + continue.
* `max_seconds` wall-clock budget triggers `_force_finalise` (sync).
* Generation timeout (asyncio.TimeoutError) → force-finalise (sync).
* `_force_finalise_async` happy path on max-turns exit.
* `_force_finalise_async` swallowing a failed forced call → empty plan.
"""

from __future__ import annotations

import asyncio

from halal_trader.core.llm.agent import run_agent
from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.tools import CRYPTO_AGENTIC_TOOLS, Tool, ToolCall


class _ScriptedLLM(BaseLLM):
    """Test double — emits scripted ToolCall lists, optionally with delay."""

    def __init__(
        self,
        script: list[list[ToolCall]],
        *,
        delay_seconds: float = 0.0,
        force_response: list[ToolCall] | None = None,
        force_raises: Exception | None = None,
    ) -> None:
        super().__init__(model="scripted")
        self._script = list(script)
        self._delay = delay_seconds
        self._force_response = force_response
        self._force_raises = force_raises

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return ""

    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[Tool],
        system: str | None = None,
        force_tool: str | None = None,
    ) -> list[ToolCall]:
        if force_tool:
            if self._force_raises is not None:
                raise self._force_raises
            if self._force_response is not None:
                return self._force_response
            return [ToolCall(name=force_tool, args={"forced": True})]
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if not self._script:
            return [ToolCall(name="submit_plan", args={})]
        return self._script.pop(0)


async def test_empty_tool_calls_returns_empty_plan_without_budget_exhausted():
    """When the model returns no tool calls (e.g. text-only response with
    no `tool_use` blocks), the loop returns the terminal-tool sentinel
    with empty args and `budget_exhausted=False` — caller decides how
    to fall back. The transcript stays empty since nothing dispatched."""
    llm = _ScriptedLLM([[]])  # one turn returning no calls

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={},
    )
    assert result.final_call.name == "submit_plan"
    assert result.final_call.args == {}
    assert result.transcript == []
    assert result.budget_exhausted is False


async def test_unknown_tool_name_records_error_and_continues():
    """The model called a tool we don't have a handler for. Loop must:
    (1) record the turn with an error; (2) feed an Error message back
    in the history so the next turn sees it; (3) continue rather than
    crashing the cycle."""
    plan_args = {"market_outlook": "ok", "buys": [], "sells": []}
    llm = _ScriptedLLM(
        [
            [ToolCall(name="mystery_tool", args={"x": 1})],
            [ToolCall(name="submit_plan", args=plan_args)],
        ]
    )

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={},  # no handler for "mystery_tool"
    )
    assert len(result.transcript) == 1
    turn = result.transcript[0]
    assert turn.tool_name == "mystery_tool"
    assert turn.error is not None
    assert "no handler" in turn.error
    assert "mystery_tool" in turn.error
    assert turn.result_text is None
    # Reaches submit_plan on the next turn.
    assert result.final_call.args == plan_args
    assert result.budget_exhausted is False


async def test_wall_clock_budget_forces_finalise_with_empty_plan():
    """When `max_seconds` elapses *before* the next generate-tool-call
    even starts, the loop syncs to `_force_finalise` (NOT the async
    forced-tool retry). That path returns an empty submit_plan with
    budget_exhausted=True — there's no time to re-prompt the model."""
    # Each turn the LLM takes 0.1s; max_seconds=0.05 makes the wall-clock
    # check at the top of turn 2 trip the early-exit.
    llm = _ScriptedLLM(
        [
            [ToolCall(name="analyze_pair", args={"symbol": "X"})],
            [ToolCall(name="analyze_pair", args={"symbol": "Y"})],
        ],
        delay_seconds=0.1,
    )

    async def analyze(_call: ToolCall) -> str:
        return "ok"

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": analyze},
        max_seconds=0.05,
        max_turns=10,
    )
    assert result.budget_exhausted is True
    assert result.final_call.name == "submit_plan"
    assert result.final_call.args == {}  # empty plan from sync force


async def test_generate_timeout_force_finalises_with_empty_plan():
    """If `generate_tool_call` itself blows past the per-call timeout
    (asyncio.wait_for raises TimeoutError), the loop also goes via
    sync `_force_finalise` → empty plan + budget_exhausted=True."""

    class _SlowLLM(BaseLLM):
        async def generate(self, prompt: str, system: str | None = None) -> str:
            return ""

        async def generate_tool_call(
            self,
            prompt: str,
            *,
            tools: list[Tool],
            system: str | None = None,
            force_tool: str | None = None,
        ) -> list[ToolCall]:
            await asyncio.sleep(2.0)  # always exceeds the test's max_seconds
            return [ToolCall(name="submit_plan", args={})]

    llm = _SlowLLM(model="slow")
    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={},
        max_seconds=0.05,
    )
    assert result.budget_exhausted is True
    assert result.final_call.name == "submit_plan"
    assert result.final_call.args == {}


async def test_max_turns_exit_calls_async_force_finalise_with_real_plan():
    """When the model uses up all its turns calling intermediate tools,
    the loop's exit goes via *async* `_force_finalise_async` — which
    issues one more LLM call with `force_tool='submit_plan'` so the
    model returns a real plan, not an empty one. The forced response
    becomes the final_call; budget_exhausted is True."""
    llm = _ScriptedLLM(
        [[ToolCall(name="analyze_pair", args={"symbol": "X"})]] * 3,
        force_response=[
            ToolCall(name="submit_plan", args={"market_outlook": "forced", "buys": []})
        ],
    )

    async def analyze(_call: ToolCall) -> str:
        return "ok"

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": analyze},
        max_turns=3,
    )
    assert result.budget_exhausted is True
    assert result.final_call.name == "submit_plan"
    # The forced-call response shape (NOT an empty plan) flows through.
    assert result.final_call.args == {"market_outlook": "forced", "buys": []}
    assert len(result.transcript) == 3


async def test_async_force_finalise_swallows_exception_returns_empty_plan():
    """If even the *forced* generate_tool_call raises (network blip,
    SDK error), we don't crash the cycle — we fall back to sync
    `_force_finalise` which returns an empty plan. budget_exhausted
    stays True so the operator can spot it."""
    llm = _ScriptedLLM(
        [[ToolCall(name="analyze_pair", args={"symbol": "X"})]] * 3,
        force_raises=RuntimeError("LLM down"),
    )

    async def analyze(_call: ToolCall) -> str:
        return "ok"

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": analyze},
        max_turns=3,
    )
    assert result.budget_exhausted is True
    assert result.final_call.name == "submit_plan"
    assert result.final_call.args == {}  # exception swallowed → empty fallback


async def test_async_force_finalise_handles_empty_response():
    """If the forced call returns an empty list (model still refuses to
    submit), fall back to the empty terminal-tool sentinel. Don't index
    into an empty list."""
    llm = _ScriptedLLM(
        [[ToolCall(name="analyze_pair", args={"symbol": "X"})]] * 3,
        force_response=[],  # empty even when forced
    )

    async def analyze(_call: ToolCall) -> str:
        return "ok"

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": analyze},
        max_turns=3,
    )
    assert result.budget_exhausted is True
    assert result.final_call.name == "submit_plan"
    assert result.final_call.args == {}


async def test_args_in_transcript_are_dict_copies():
    """Mutating a transcript turn's args must not retroactively change
    the LLM's recorded ToolCall — used in audit/replay so the original
    is the source of truth."""
    original = {"symbol": "BTCUSDT", "extra": [1, 2, 3]}
    plan_args = {"market_outlook": "ok", "buys": [], "sells": []}
    llm = _ScriptedLLM(
        [
            [ToolCall(name="analyze_pair", args=original)],
            [ToolCall(name="submit_plan", args=plan_args)],
        ]
    )

    async def analyze(_call: ToolCall) -> str:
        return "ok"

    result = await run_agent(
        llm,
        system="",
        user="prompt",
        tools=CRYPTO_AGENTIC_TOOLS,
        handlers={"analyze_pair": analyze},
    )
    transcript_args = result.transcript[0].args
    assert transcript_args == original
    transcript_args["mutated"] = True
    assert "mutated" not in original  # original untouched
