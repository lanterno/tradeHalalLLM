"""Tests for counter-factual regret."""

from __future__ import annotations

import json
from typing import Any

import pytest

from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.core.regret import (
    ClosedTradeView,
    CounterFactualVerdict,
    aggregate_regret,
    counter_factual_review,
    hindsight_optimal_size,
    hindsight_regret,
    review_closed_trades,
)

_seq = iter(range(10_000))


def _winner(size: float = 1.0, pnl: float = 0.02) -> ClosedTradeView:
    return ClosedTradeView(
        trade_id=f"win-{next(_seq):04d}",
        symbol="BTCUSDT",
        action_size_pct=size,
        pnl_pct=pnl,
    )


def _loser(size: float = 1.0, pnl: float = -0.015) -> ClosedTradeView:
    return ClosedTradeView(
        trade_id=f"lose-{next(_seq):04d}",
        symbol="ETHUSDT",
        action_size_pct=size,
        pnl_pct=pnl,
    )


# ── Hindsight regret ──────────────────────────────────────────────


def test_optimal_size_winner_full() -> None:
    assert hindsight_optimal_size(0.01) == 1.0


def test_optimal_size_loser_zero() -> None:
    assert hindsight_optimal_size(-0.01) == 0.0


def test_optimal_size_flat_zero() -> None:
    assert hindsight_optimal_size(0.0) == 0.0


def test_full_size_winner_zero_regret() -> None:
    rec = hindsight_regret(_winner(size=1.0, pnl=0.02))
    assert rec.regret == 0.0
    assert rec.note == ""


def test_zero_size_loser_zero_regret() -> None:
    rec = hindsight_regret(_loser(size=0.0, pnl=-0.015))
    assert rec.regret == 0.0


def test_small_size_winner_missed_edge_note() -> None:
    rec = hindsight_regret(_winner(size=0.2, pnl=0.02))
    assert rec.regret > 0.5
    assert "missed-edge" in rec.note


def test_full_size_loser_tail_loss_note() -> None:
    rec = hindsight_regret(_loser(size=1.0, pnl=-0.02))
    assert rec.regret == 1.0
    assert "tail-loss" in rec.note


def test_action_size_clamped() -> None:
    rec = hindsight_regret(_winner(size=1.4, pnl=0.01))
    assert rec.actual_size_pct == 1.0
    assert rec.regret == 0.0


# ── Aggregation ───────────────────────────────────────────────────


def test_aggregate_empty() -> None:
    summary = aggregate_regret([])
    assert summary.n == 0
    assert summary.mean_regret == 0.0


def test_aggregate_basic_means() -> None:
    records = [
        hindsight_regret(_winner(size=1.0, pnl=0.02)),  # 0
        hindsight_regret(_winner(size=0.0, pnl=0.02)),  # 1
        hindsight_regret(_loser(size=1.0, pnl=-0.02)),  # 1
        hindsight_regret(_loser(size=0.0, pnl=-0.02)),  # 0
    ]
    summary = aggregate_regret(records)
    assert summary.n == 4
    assert summary.mean_regret == 0.5
    assert summary.median_regret == 0.5
    assert summary.pct_high_regret == 0.5
    assert summary.missed_edge_count == 1
    assert summary.tail_loss_count == 1


def test_aggregate_groups_by_symbol() -> None:
    records = [
        hindsight_regret(_winner(size=0.0, pnl=0.02)),  # BTCUSDT: 1
        hindsight_regret(_loser(size=1.0, pnl=-0.02)),  # ETHUSDT: 1
        hindsight_regret(_winner(size=1.0, pnl=0.02)),  # BTCUSDT: 0
    ]
    summary = aggregate_regret(records)
    assert summary.by_symbol["BTCUSDT"] == 0.5
    assert summary.by_symbol["ETHUSDT"] == 1.0


def test_aggregate_groups_by_setup() -> None:
    records = [
        hindsight_regret(_winner(size=0.0, pnl=0.02)),
        hindsight_regret(_loser(size=1.0, pnl=-0.02)),
    ]
    setup_lookup = {records[0].trade_id: "breakout", records[1].trade_id: "mean_reversion"}
    summary = aggregate_regret(records, setup_lookup=setup_lookup)
    assert summary.by_setup["breakout"] == 1.0
    assert summary.by_setup["mean_reversion"] == 1.0


# ── Counter-factual LLM ───────────────────────────────────────────


class _ScriptedLLM(BaseLLM):
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        super().__init__(model="cf-stub")
        self._response = response
        self.calls = 0
        self.last_usage = CallUsage(model="cf-stub")

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return json.dumps(self._response)


@pytest.mark.asyncio
async def test_counter_factual_basic() -> None:
    llm = _ScriptedLLM({"would_repeat": False, "regret": 0.6, "alt_action": "skip"})
    v = await counter_factual_review(
        llm,
        context_excerpt="rsi=85 momentum=high",
        action="buy",
        symbol="BTCUSDT",
        size_pct=0.8,
        confidence=0.9,
    )
    assert v.would_repeat is False
    assert v.regret == 0.6
    assert v.alt_action == "skip"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_counter_factual_failure_returns_safe_default() -> None:
    llm = _ScriptedLLM(RuntimeError("api down"))
    v = await counter_factual_review(
        llm,
        context_excerpt="x",
        action="buy",
        symbol="BTCUSDT",
        size_pct=0.5,
        confidence=0.7,
    )
    assert v.would_repeat is True
    assert v.regret == 0.0
    assert v.alt_action == "unknown"


@pytest.mark.asyncio
async def test_counter_factual_clamps_regret() -> None:
    llm = _ScriptedLLM({"would_repeat": False, "regret": 5.0, "alt_action": "skip"})
    v = await counter_factual_review(
        llm, context_excerpt="x", action="buy", symbol="BTCUSDT", size_pct=1.0, confidence=0.9
    )
    assert v.regret == 1.0


@pytest.mark.asyncio
async def test_review_closed_trades_no_llm_yields_only_records() -> None:
    trades = [_winner(size=0.2, pnl=0.02), _loser(size=1.0, pnl=-0.02)]
    out = await review_closed_trades(trades)
    assert len(out) == 2
    assert all(isinstance(r, type(out[0][0])) for r, _ in out)
    assert all(cf is None for _, cf in out)


@pytest.mark.asyncio
async def test_review_closed_trades_with_llm_runs_cf() -> None:
    trades = [_winner(size=0.5, pnl=0.02)]
    llm = _ScriptedLLM({"would_repeat": True, "regret": 0.1, "alt_action": "buy"})

    async def fetch_ctx(_tid: str) -> str:
        return "ctx"

    out = await review_closed_trades(trades, fetch_context=fetch_ctx, llm=llm)
    assert len(out) == 1
    rec, cf = out[0]
    assert isinstance(cf, CounterFactualVerdict)
    assert cf.regret == 0.1
    assert rec.regret > 0.0
