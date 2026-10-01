"""INVARIANT: a lost LLM reply is recorded as a FAILED call, never a quiet HOLD.

The documented failure mode of this bot is silent no-action cycles, not
crashes. An empty or truncated tool call used to validate as an empty
TradingPlan and be persisted as a normal decision with raw_response "{}".
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.tools import SUBMIT_DECISIONS_TOOL, ToolCall
from halal_trader.core.strategy import BaseStrategy


def _strategy(tool_args: dict[str, Any]) -> tuple[BaseStrategy, AsyncMock]:
    llm = MagicMock(spec=BaseLLM)
    llm.supports_tool_use = True
    llm.model = "glm-test"
    llm.last_thinking = ""
    llm.last_usage = MagicMock(cost_usd=0)
    llm.generate_tool_call = AsyncMock(
        return_value=[ToolCall(name="submit_decisions", args=tool_args)]
    )
    repo = AsyncMock()
    strat = BaseStrategy.__new__(BaseStrategy)
    strat._llm = llm
    strat._repo = repo
    strat._llm_provider_name = "test"
    strat._alert_sink = None
    return strat, repo


async def _analyse(strat: BaseStrategy) -> dict[str, Any]:
    return await strat._run_llm_analysis(  # type: ignore[no-any-return]
        "sys",
        "user",
        prompt_summary="cycle",
        validate=lambda raw: raw,
        make_empty=lambda msg: {"error": msg},
        extract_symbols=lambda p: [],
        count_actions=lambda p: {},
        tool=SUBMIT_DECISIONS_TOOL,
    )


@pytest.mark.parametrize(
    "args",
    [
        {},  # what an unparseable reply used to collapse into
        {"decisions": []},  # truncated: market_outlook lost
    ],
)
async def test_tool_call_missing_required_keys_is_recorded_as_failed(args: dict[str, Any]) -> None:
    strat, repo = _strategy(args)

    plan = await _analyse(strat)

    assert "error" in plan  # make_empty: no decisions are acted on
    recorded = repo.record_decision.await_args.kwargs
    assert recorded["prompt_summary"].startswith("FAILED")
    assert "missing required" in recorded["raw_response"]


async def test_a_real_hold_is_still_a_success() -> None:
    strat, repo = _strategy({"decisions": [], "market_outlook": "flat"})

    plan = await _analyse(strat)

    assert plan["market_outlook"] == "flat"
    assert not repo.record_decision.await_args.kwargs["prompt_summary"].startswith("FAILED")


def _failing_strategy(error: Exception) -> tuple[BaseStrategy, AsyncMock]:
    strat, _ = _strategy({})
    strat._llm.generate_tool_call = AsyncMock(side_effect=error)  # type: ignore[attr-defined]
    sink = MagicMock()
    sink.notify = AsyncMock()
    strat._alert_sink = sink
    return strat, sink.notify


async def test_persistent_failures_alert_after_three_cycles() -> None:
    strat, notify = _failing_strategy(TimeoutError())

    await _analyse(strat)
    await _analyse(strat)
    notify.assert_not_awaited()  # two in a row can be a blip
    await _analyse(strat)

    kind, message = notify.await_args.args
    assert kind == "llm.failing"
    assert "3 consecutive" in message and "TimeoutError" in message


async def test_a_success_resets_the_failure_count() -> None:
    strat, notify = _failing_strategy(TimeoutError())
    await _analyse(strat)
    await _analyse(strat)
    strat._llm.generate_tool_call = AsyncMock(  # type: ignore[attr-defined]
        return_value=[
            ToolCall(name="submit_decisions", args={"decisions": [], "market_outlook": "x"})
        ]
    )
    await _analyse(strat)
    strat._llm.generate_tool_call = AsyncMock(side_effect=TimeoutError())  # type: ignore[attr-defined]
    await _analyse(strat)
    await _analyse(strat)

    notify.assert_not_awaited()  # never three in a row


async def test_an_exhausted_budget_does_not_double_alert() -> None:
    from halal_trader.core.llm.spend import BudgetExhausted

    strat, notify = _failing_strategy(BudgetExhausted("cap reached"))
    for _ in range(4):
        await _analyse(strat)

    notify.assert_not_awaited()  # the spend meter already alerted
