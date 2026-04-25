"""Direct tests for crypto/backtest.py:LLMBacktestEngine.

Exercises cache hit/miss, prompt-cache-key invariance under same inputs,
trade decisions translated through the live-prompt response shape, and
graceful handling of LLM failures.
"""

from __future__ import annotations

from typing import Any

import pytest

from halal_trader.crypto.backtest import LLMBacktestEngine, _extract_decision
from halal_trader.domain.models import Kline


def _kl(close: float, open_time: int) -> Kline:
    return Kline(
        open_time=open_time,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=100.0,
        close_time=open_time + 60_000,
    )


def _series(start: float, n: int, step: float = 0.5) -> list[Kline]:
    return [_kl(start + i * step, i * 60_000) for i in range(n)]


class _StubLLM:
    """LLM that returns canned responses and records every call."""

    def __init__(self, response: dict[str, Any] | None = None):
        self._response = response or {
            "decisions": [
                {
                    "action": "buy",
                    "symbol": "BTCUSDT",
                    "quantity": 0.001,
                    "confidence": 0.9,
                    "reasoning": "stub",
                    "entry_price": 0.0,
                    "target_price": 0.0,
                    "stop_loss": 0.0,
                }
            ],
            "market_outlook": "bullish",
            "risk_notes": "",
        }
        self.calls = 0
        self.last_system: str | None = None
        self.last_prompt: str | None = None

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        self.calls += 1
        self.last_system = system
        self.last_prompt = prompt
        return self._response


@pytest.mark.asyncio
async def test_engine_runs_and_invokes_llm():
    klines = _series(100.0, 200)
    llm = _StubLLM(
        response={"decisions": [{"action": "hold", "confidence": 0.5}]}
    )
    engine = LLMBacktestEngine(llm, initial_balance=10_000.0)
    result = await engine.run("BTCUSDT", klines, window_size=100, cycle_interval=10)
    assert result is not None
    assert llm.calls > 0


@pytest.mark.asyncio
async def test_engine_uses_unified_live_prompt():
    """Prompt sent to the LLM should look like the live cycle's prompt
    (PORTFOLIO STATUS, TASK, etc.), not the old 5-line stripped form."""
    klines = _series(100.0, 200)
    llm = _StubLLM(response={"decisions": [{"action": "hold", "confidence": 0.5}]})
    engine = LLMBacktestEngine(llm, initial_balance=10_000.0)
    await engine.run("BTCUSDT", klines, window_size=100, cycle_interval=10)
    # Sanity: the prompt must include the live template's PORTFOLIO STATUS
    # section so prompt-engineering iterations transfer to live.
    assert "PORTFOLIO STATUS" in (llm.last_prompt or "")
    assert "TASK" in (llm.last_prompt or "")


@pytest.mark.asyncio
async def test_engine_caches_identical_prompts(tmp_path):
    """Two runs with the same inputs hit the cache → fewer LLM calls."""
    klines = _series(100.0, 200)
    llm = _StubLLM(response={"decisions": [{"action": "hold", "confidence": 0.5}]})
    engine = LLMBacktestEngine(
        llm, initial_balance=10_000.0, cache_dir=str(tmp_path)
    )
    await engine.run("BTCUSDT", klines, window_size=100, cycle_interval=10)
    first_calls = llm.calls

    engine2 = LLMBacktestEngine(
        llm, initial_balance=10_000.0, cache_dir=str(tmp_path)
    )
    await engine2.run("BTCUSDT", klines, window_size=100, cycle_interval=10)
    # Cache from first run survives → second run shouldn't re-hit the LLM.
    assert llm.calls == first_calls


@pytest.mark.asyncio
async def test_engine_buy_action_opens_position():
    klines = _series(100.0, 200)
    llm = _StubLLM(
        response={
            "decisions": [
                {
                    "action": "buy",
                    "symbol": "BTCUSDT",
                    "confidence": 0.9,
                    "reasoning": "test",
                }
            ]
        }
    )
    engine = LLMBacktestEngine(llm, initial_balance=10_000.0, sl_pct=0.01, tp_pct=0.02)
    result = await engine.run("BTCUSDT", klines, window_size=100, cycle_interval=20)
    # At least one trade should have been opened given a high-confidence buy.
    assert result.total_trades >= 1


@pytest.mark.asyncio
async def test_engine_handles_llm_failure_gracefully():
    """A failing LLM call shouldn't abort the whole backtest."""

    class _FailingLLM:
        calls = 0

        async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
            type(self).calls += 1
            raise RuntimeError("API down")

    klines = _series(100.0, 200)
    engine = LLMBacktestEngine(_FailingLLM(), initial_balance=10_000.0)
    result = await engine.run("BTCUSDT", klines, window_size=100, cycle_interval=10)
    assert result is not None
    assert result.total_trades == 0


# ── _extract_decision shape compatibility ─────────────────────


def test_extract_decision_live_shape():
    raw = {
        "decisions": [
            {"action": "buy", "symbol": "BTCUSDT", "confidence": 0.7, "reasoning": "r"},
            {"action": "sell", "symbol": "ETHUSDT", "confidence": 0.6},
        ]
    }
    a, c, r = _extract_decision(raw, "BTCUSDT")
    assert (a, c, r) == ("buy", 0.7, "r")


def test_extract_decision_picks_first_when_no_symbol_match():
    raw = {"decisions": [{"action": "sell", "symbol": "ETHUSDT", "confidence": 0.5}]}
    a, c, _ = _extract_decision(raw, "BTCUSDT")
    assert a == "sell"


def test_extract_decision_legacy_shape():
    """Pre-P3.1 cached responses had a flat shape; backwards compat."""
    raw = {"action": "hold", "confidence": 0.4, "reasoning": "old"}
    a, c, r = _extract_decision(raw, "BTCUSDT")
    assert (a, c, r) == ("hold", 0.4, "old")


def test_extract_decision_defaults_on_missing_keys():
    a, c, r = _extract_decision({"decisions": []}, "BTCUSDT")
    assert (a, c, r) == ("hold", 0.5, "")
