"""Tests for `BaseLLM._track_usage` daily-counter + threshold logging,
the default `generate_tool_call` fallback, and `CallUsage.total_tokens`.

`test_llm_base_helpers.py` already pins the pure `strip_thinking` /
`_clean_json_body` helpers; `test_anthropic_caching.py` and
`test_openai_usage.py` cover provider-specific usage. This file pins
the contract that lives on the abstract base — daily reset semantics,
the threshold log ladder, and the default tool-call wrapper that
non-Anthropic providers fall through to.
"""

from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from halal_trader.core.llm.base import BaseLLM, CallUsage


class _Stub(BaseLLM):
    """Minimal subclass — just satisfies the ABC and lets us script JSON."""

    def __init__(self, model: str = "stub", json_response: dict | None = None) -> None:
        import json

        super().__init__(model)
        self._json_response = json_response or {"market_outlook": "ok"}
        self._raw_text = json.dumps(self._json_response)

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return self._raw_text


# ── CallUsage.total_tokens ──────────────────────────────────


def test_call_usage_total_tokens_sums_input_and_output():
    u = CallUsage(input_tokens=200, output_tokens=50)
    assert u.total_tokens == 250


def test_call_usage_total_tokens_zero_default():
    """Both fields default to 0 — total is 0, not None."""
    u = CallUsage()
    assert u.total_tokens == 0


def test_call_usage_default_cost_is_decimal_zero():
    """The default factory must yield Decimal('0'), not 0.0 — matters
    for the LlmDecision row's NUMERIC column."""
    u = CallUsage()
    assert u.cost_usd == Decimal("0")
    assert isinstance(u.cost_usd, Decimal)


def test_call_usage_total_excludes_cache_tokens():
    """Cache tokens are billed differently and reported in their own
    fields; they don't roll into `total_tokens`."""
    u = CallUsage(input_tokens=200, output_tokens=50, cache_read_tokens=1000)
    assert u.total_tokens == 250


# ── _track_usage threshold logging ──────────────────────────


def test_track_usage_no_log_below_first_threshold(caplog):
    """Below 10k tokens the helper stays silent — nothing in the log."""
    llm = _Stub()
    with caplog.at_level(logging.INFO):
        llm._track_usage(5_000)
    assert not any("daily token usage crossed" in r.message for r in caplog.records)


def test_track_usage_logs_at_first_threshold(caplog):
    """First crossing of 10k logs once."""
    llm = _Stub()
    with caplog.at_level(logging.INFO):
        llm._track_usage(15_000)
    crossings = [r for r in caplog.records if "daily token usage crossed" in r.message]
    assert len(crossings) == 1
    assert "10k" in crossings[0].message


def test_track_usage_does_not_double_log_within_same_threshold(caplog):
    """Crossing 10k once and then adding more (still below 50k) must
    not produce another 10k log line — `_last_threshold_logged` gates it."""
    llm = _Stub()
    with caplog.at_level(logging.INFO):
        llm._track_usage(15_000)
        llm._track_usage(20_000)  # 35k cumulative, still under 50k
    crossings = [r for r in caplog.records if "daily token usage crossed" in r.message]
    assert len(crossings) == 1


def test_track_usage_logs_each_new_threshold_in_sequence(caplog):
    """Crossing 10k → 50k → 100k logs each tier exactly once."""
    llm = _Stub()
    with caplog.at_level(logging.INFO):
        llm._track_usage(15_000)  # crosses 10k
        llm._track_usage(40_000)  # 55k cumulative — crosses 50k
        llm._track_usage(50_000)  # 105k cumulative — crosses 100k
    crossings = [r.message for r in caplog.records if "daily token usage crossed" in r.message]
    assert any("10k" in m for m in crossings)
    assert any("50k" in m for m in crossings)
    assert any("100k" in m for m in crossings)


def test_track_usage_resets_counter_on_new_day(caplog, monkeypatch):
    """Crossing midnight (UTC) resets `_daily_tokens` and the
    threshold-logged tracker — tomorrow's first 10k should log again."""
    from halal_trader.core.llm import base as base_mod

    class _Dt:
        _now = "2026-04-25"

        @classmethod
        def set(cls, val: str) -> None:
            cls._now = val

        @staticmethod
        def now(tz=None):
            class _D:
                @staticmethod
                def strftime(fmt: str) -> str:
                    return _Dt._now

            return _D()

    monkeypatch.setattr(base_mod, "datetime", _Dt)

    llm = _Stub()
    with caplog.at_level(logging.INFO):
        llm._track_usage(15_000)  # day 1: crosses 10k
        # Tomorrow — fresh date triggers reset.
        _Dt.set("2026-04-26")
        llm._track_usage(15_000)  # day 2: also crosses 10k (counter reset)

    crossings = [r for r in caplog.records if "daily token usage crossed" in r.message]
    assert len(crossings) == 2  # both days logged independently
    # The internal counter on day 2 reflects the day-2 input, not cumulative.
    assert llm._daily_tokens == 15_000
    assert llm._daily_reset_date == "2026-04-26"


def test_track_usage_skips_intermediate_thresholds_on_big_call(caplog):
    """A single 600k-token call jumps past 10k/50k/100k/250k/500k all
    at once. The implementation `break`s after the first new threshold
    — pin that behaviour so a refactor that changed it (logging all
    crossed tiers at once) breaks here."""
    llm = _Stub()
    with caplog.at_level(logging.INFO):
        llm._track_usage(600_000)
    crossings = [r for r in caplog.records if "daily token usage crossed" in r.message]
    # Only one log line per `_track_usage` call (the loop breaks after match).
    assert len(crossings) == 1
    # The lowest unlogged threshold (10k) is logged first.
    assert "10k" in crossings[0].message


def test_track_usage_handles_zero(caplog):
    """A 0-token call (some providers report 0 on a no-op) shouldn't
    log anything but also shouldn't crash."""
    llm = _Stub()
    with caplog.at_level(logging.INFO):
        llm._track_usage(0)
    assert llm._daily_tokens == 0


# ── default generate_tool_call fallback ────────────────────


@pytest.mark.asyncio
async def test_default_tool_call_uses_force_tool_name():
    """Fallback path: when `force_tool` is set, the returned ToolCall
    carries that name regardless of the tools list contents."""

    class _T:
        def __init__(self, name: str) -> None:
            self.name = name

    llm = _Stub(json_response={"buys": []})
    calls = await llm.generate_tool_call(
        "hi", tools=[_T("first"), _T("second")], force_tool="submit_plan"
    )
    assert len(calls) == 1
    assert calls[0].name == "submit_plan"
    assert calls[0].args == {"buys": []}  # JSON parsed from generate()


@pytest.mark.asyncio
async def test_default_tool_call_uses_first_tool_name_when_no_force():
    """Without force, the first tool's name is used — matches the
    typical strategy-call flow where `tools=[SUBMIT_PLAN_TOOL]`."""

    class _T:
        def __init__(self, name: str) -> None:
            self.name = name

    llm = _Stub()
    calls = await llm.generate_tool_call("hi", tools=[_T("first"), _T("second")])
    assert calls[0].name == "first"


@pytest.mark.asyncio
async def test_default_tool_call_falls_back_to_submit_plan_when_no_tools():
    """Defensive: empty tools list + no force → use the conventional
    'submit_plan' name. Saves the caller from having to special-case
    the empty list."""
    llm = _Stub()
    calls = await llm.generate_tool_call("hi", tools=[])
    assert calls[0].name == "submit_plan"


@pytest.mark.asyncio
async def test_default_tool_call_args_are_parsed_json_body():
    """The ToolCall.args dict is exactly what `generate_json` returned —
    no extra wrapping, no key remapping."""

    class _T:
        name = "submit_plan"

    llm = _Stub(json_response={"market_outlook": "neutral", "buys": [], "sells": []})
    calls = await llm.generate_tool_call("hi", tools=[_T()])
    assert calls[0].args == {"market_outlook": "neutral", "buys": [], "sells": []}


# ── generate_json thinking + JSON ─────────────────────────


class _ThinkingLLM(BaseLLM):
    """Stub that returns a response with a leading <think> block."""

    def __init__(self, payload: str) -> None:
        super().__init__("stub")
        self._payload = payload

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return self._payload


@pytest.mark.asyncio
async def test_generate_json_separates_thinking_and_parses_body():
    """`generate_json` must populate `last_thinking` from the `<think>`
    block AND return a parsed dict from the body — both happen in one
    call."""
    payload = '<think>weighing the trade</think>\n{"action":"hold"}'
    llm = _ThinkingLLM(payload)
    parsed = await llm.generate_json("ignored")
    assert parsed == {"action": "hold"}
    assert llm.last_thinking == "weighing the trade"


@pytest.mark.asyncio
async def test_generate_json_clears_last_thinking_when_no_think_block():
    """Subsequent calls without a think block must reset `last_thinking`
    to "" — otherwise stale chain-of-thought from a prior call leaks
    into this call's LlmDecision row."""
    llm = _ThinkingLLM('<think>old thinking</think>\n{"a":1}')
    await llm.generate_json("first")
    assert llm.last_thinking == "old thinking"

    # Swap to a no-think payload and re-run.
    llm._payload = '{"a":2}'
    await llm.generate_json("second")
    assert llm.last_thinking == ""  # reset, not leaked


@pytest.mark.asyncio
async def test_generate_json_strips_markdown_fences() -> None:
    """`_clean_json_body` runs after `strip_thinking` — verify the
    markdown-fence strip is applied (some Ollama / OpenAI responses
    wrap JSON in ```json … ``` even when format=json is requested)."""
    payload = '```json\n{"a":1}\n```'
    llm = _ThinkingLLM(payload)
    parsed = await llm.generate_json("hi")
    assert parsed == {"a": 1}
