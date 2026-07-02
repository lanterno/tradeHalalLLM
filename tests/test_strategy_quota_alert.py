"""Strategy-LLM credit-exhaustion operator alert.

The classifier's quota breaker has alerted via AlertSink since 2026-05;
the strategy path (the one that actually trades) stayed silent through
the 2026-06 quota storm. These tests pin the new behavior: a quota-ish
LLM failure fires exactly one rate-limited ``llm.quota_exhausted``
alert, generic failures stay quiet, and a broken sink can never break
the cycle (the empty no-action plan is still returned).
"""

from __future__ import annotations

from typing import Any

import pytest

from halal_trader.core.llm.quota import QUOTA_ERROR_MARKERS, is_quota_error
from halal_trader.core.strategy import BaseStrategy


class _FailingLLM:
    def __init__(self, error: Exception) -> None:
        self.model = "z-ai/glm-5.2"
        self._error = error

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        raise self._error


class _FakeRepo:
    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []

    async def record_decision(self, **kwargs: Any) -> None:
        self.decisions.append(kwargs)


class _FakeSink:
    def __init__(self, *, explode: bool = False) -> None:
        self.notifications: list[tuple[str, str]] = []
        self._explode = explode

    async def notify(self, error_type: str, details: str) -> None:
        if self._explode:
            raise RuntimeError("telegram down")
        self.notifications.append((error_type, details))


def _strategy(llm: _FailingLLM, repo: _FakeRepo) -> BaseStrategy:
    class _Strat(BaseStrategy):
        pass

    return _Strat(
        llm,  # type: ignore[arg-type]
        repo,  # type: ignore[arg-type]
        llm_provider_name="glm",
        max_position_pct=0.2,
        daily_loss_limit=0.02,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
    )


async def _run(strategy: BaseStrategy) -> Any:
    return await strategy._run_llm_analysis(
        "system",
        "user",
        prompt_summary="test",
        validate=lambda raw: raw,
        make_empty=lambda err: {"empty": True, "error": err},
        extract_symbols=lambda plan: [],
        count_actions=lambda plan: {},
    )


# ── is_quota_error ─────────────────────────────────────────────


def test_quota_markers_matched_case_insensitively():
    assert is_quota_error(Exception("Error code: 402 - Insufficient Credits"))
    assert is_quota_error(Exception("insufficient_quota: check your plan"))
    assert is_quota_error("You exceeded your current quota.")
    assert not is_quota_error(Exception("connection reset by peer"))
    assert not is_quota_error(Exception("429 rate limit reached"))  # transient, not credits


def test_markers_cover_openai_and_openrouter_shapes():
    assert "insufficient credits" in QUOTA_ERROR_MARKERS  # OpenRouter 402
    assert "insufficient_quota" in QUOTA_ERROR_MARKERS  # OpenAI-compat


# ── alert firing ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quota_failure_fires_alert_and_returns_empty_plan():
    repo = _FakeRepo()
    strategy = _strategy(
        _FailingLLM(Exception("Error code: 402 - Insufficient credits")), repo
    )
    sink = _FakeSink()
    strategy.attach_alert_sink(sink)

    plan = await _run(strategy)

    assert plan["empty"] is True
    assert len(sink.notifications) == 1
    error_type, details = sink.notifications[0]
    assert error_type == "llm.quota_exhausted"
    assert "glm/z-ai/glm-5.2" in details
    # The failure is still recorded for the audit trail.
    assert repo.decisions and repo.decisions[0]["prompt_summary"].startswith("FAILED")


@pytest.mark.asyncio
async def test_generic_failure_does_not_alert():
    strategy = _strategy(_FailingLLM(RuntimeError("boom")), _FakeRepo())
    sink = _FakeSink()
    strategy.attach_alert_sink(sink)

    plan = await _run(strategy)

    assert plan["empty"] is True
    assert sink.notifications == []


@pytest.mark.asyncio
async def test_no_sink_attached_is_safe():
    strategy = _strategy(
        _FailingLLM(Exception("insufficient_quota")), _FakeRepo()
    )
    plan = await _run(strategy)  # must not raise
    assert plan["empty"] is True


@pytest.mark.asyncio
async def test_broken_sink_never_breaks_the_cycle():
    strategy = _strategy(
        _FailingLLM(Exception("insufficient credits")), _FakeRepo()
    )
    strategy.attach_alert_sink(_FakeSink(explode=True))

    plan = await _run(strategy)  # alert raises internally; plan still returned

    assert plan["empty"] is True
