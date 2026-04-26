"""Tests for ensemble LLM judge."""

from __future__ import annotations

import pytest

from halal_trader.core.llm.ensemble import (
    EnsembleVariant,
    aggregate_plans,
    run_ensemble,
)
from halal_trader.domain.models import (
    CryptoTradeDecision,
    CryptoTradingPlan,
    TradeAction,
)


def _buy(symbol: str = "BTCUSDT", qty: float = 1.0, conf: float = 0.7) -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=qty,
        confidence=conf,
        reasoning="x",
    )


def _sell(symbol: str = "ETHUSDT", qty: float = 1.0, conf: float = 0.6) -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.SELL,
        symbol=symbol,
        quantity=qty,
        confidence=conf,
        reasoning="x",
    )


def _plan(*decisions: CryptoTradeDecision, outlook: str = "") -> CryptoTradingPlan:
    return CryptoTradingPlan(decisions=list(decisions), market_outlook=outlook)


# ── Aggregation ───────────────────────────────────────────────────


def test_unanimous_keeps_decision_with_full_multiplier() -> None:
    plans = {
        "a": _plan(_buy(qty=1.0)),
        "b": _plan(_buy(qty=1.5)),
        "c": _plan(_buy(qty=0.5)),
    }
    v = aggregate_plans(plans, quorum=2)
    assert v.agreement_score == 1.0
    assert v.sizing_multiplier == 1.0
    assert len(v.consensus_plan.decisions) == 1
    # median quantity of [0.5, 1.0, 1.5] = 1.0
    assert v.consensus_plan.decisions[0].quantity == 1.0


def test_quorum_reached_partial_agreement() -> None:
    plans = {
        "a": _plan(_buy(symbol="BTCUSDT")),
        "b": _plan(_buy(symbol="BTCUSDT")),
        "c": _plan(_buy(symbol="ETHUSDT")),  # disagrees
    }
    v = aggregate_plans(plans, quorum=2)
    # 2 of 3 agreed on BTC, 1 on ETH alone -> ETH dropped
    assert len(v.consensus_plan.decisions) == 1
    assert v.consensus_plan.decisions[0].symbol == "BTCUSDT"
    assert 0.5 < v.agreement_score < 1.0
    assert 0.5 <= v.sizing_multiplier < 1.0


def test_no_quorum_drops_all() -> None:
    plans = {
        "a": _plan(_buy(symbol="BTCUSDT")),
        "b": _plan(_buy(symbol="ETHUSDT")),
        "c": _plan(_buy(symbol="SOLUSDT")),
    }
    v = aggregate_plans(plans, quorum=2)
    assert v.consensus_plan.decisions == []


def test_skip_at_threshold_zeroes_multiplier() -> None:
    plans = {
        "a": _plan(_buy(symbol="BTCUSDT")),
        "b": _plan(_buy(symbol="ETHUSDT")),  # half agreement
    }
    v = aggregate_plans(plans, quorum=1, skip_quorum_at=0.6)
    assert v.sizing_multiplier == 0.0


def test_action_disagreement_buckets_separately() -> None:
    plans = {
        "a": _plan(_buy(symbol="BTCUSDT")),
        "b": _plan(_sell(symbol="BTCUSDT")),  # same symbol, opposite action
    }
    v = aggregate_plans(plans, quorum=2)
    # both reached only 1 vote -> nothing survives
    assert v.consensus_plan.decisions == []
    assert v.counts["BTCUSDT"]["buy"] == 1
    assert v.counts["BTCUSDT"]["sell"] == 1


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_plans({}, quorum=1)


def test_consensus_uses_median_confidence() -> None:
    plans = {
        "a": _plan(_buy(qty=1.0, conf=0.5)),
        "b": _plan(_buy(qty=1.0, conf=0.7)),
        "c": _plan(_buy(qty=1.0, conf=0.9)),
    }
    v = aggregate_plans(plans, quorum=2)
    assert v.consensus_plan.decisions[0].confidence == pytest.approx(0.7)


# ── Driver ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_ensemble_concurrent_success() -> None:
    async def make_plan(symbol: str):
        return _plan(_buy(symbol=symbol))

    variants = [
        EnsembleVariant(name="hot", call=lambda: make_plan("BTCUSDT")),
        EnsembleVariant(name="cool", call=lambda: make_plan("BTCUSDT")),
        EnsembleVariant(name="contrarian", call=lambda: make_plan("ETHUSDT")),
    ]
    v = await run_ensemble(variants, quorum=2)
    assert len(v.consensus_plan.decisions) == 1
    assert v.consensus_plan.decisions[0].symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_run_ensemble_one_failure_continues() -> None:
    async def good():
        return _plan(_buy())

    async def bad():
        raise RuntimeError("variant down")

    variants = [
        EnsembleVariant(name="good", call=good),
        EnsembleVariant(name="bad", call=bad),
    ]
    v = await run_ensemble(variants, quorum=1)
    # Single survivor still produces a verdict.
    assert v.consensus_plan.decisions
    assert "good" in v.per_variant
    assert "bad" not in v.per_variant


@pytest.mark.asyncio
async def test_run_ensemble_all_failures_raises() -> None:
    async def bad():
        raise RuntimeError("nope")

    variants = [
        EnsembleVariant(name="bad1", call=bad),
        EnsembleVariant(name="bad2", call=bad),
    ]
    with pytest.raises(RuntimeError):
        await run_ensemble(variants)


@pytest.mark.asyncio
async def test_strategy_ensemble_uses_consensus_quantities() -> None:
    """End-to-end: ensemble votes are merged into the strategy's plan."""
    import json
    from unittest.mock import AsyncMock

    from halal_trader.core.llm.base import BaseLLM, CallUsage
    from halal_trader.crypto.strategy import CryptoTradingStrategy
    from halal_trader.domain.models import CryptoAccount

    class _ScriptedLLM(BaseLLM):
        def __init__(self, payload, model="stub"):
            super().__init__(model=model)
            self._payload = payload
            self.last_usage = CallUsage(model=model)

        async def generate(self, prompt, system=None):
            return json.dumps(self._payload)

    primary_payload = {
        "decisions": [
            {
                "action": "buy",
                "symbol": "BTCUSDT",
                "quantity": 1.0,
                "confidence": 0.7,
                "reasoning": "trend",
            }
        ],
        "market_outlook": "bullish",
        "risk_notes": "",
    }
    alt_payload = {
        "decisions": [
            {
                "action": "buy",
                "symbol": "BTCUSDT",
                "quantity": 0.5,
                "confidence": 0.6,
                "reasoning": "trend confirm",
            }
        ],
        "market_outlook": "bullish",
        "risk_notes": "",
    }

    repo = AsyncMock()
    repo.record_decision = AsyncMock()
    strat = CryptoTradingStrategy(
        _ScriptedLLM(primary_payload, "primary"),
        repo,
        llm_provider_name="stub",
        max_position_pct=0.25,
        daily_loss_limit=0.05,
        daily_return_target=0.01,
        max_simultaneous_positions=3,
        ensemble_llms=[_ScriptedLLM(alt_payload, "alt")],
        ensemble_quorum=2,
    )
    plan = await strat.analyze(
        account=CryptoAccount(total_balance_usdt=1000),
        positions_text="",
        halal_pairs=["BTCUSDT"],
        klines_by_symbol={},
        orderbooks={},
    )
    assert plan.decisions
    # Median of [1.0, 0.5] = 0.5 — ensemble overrides primary's 1.0.
    assert plan.decisions[0].quantity == 0.5
    assert strat.last_ensemble_verdict is not None


@pytest.mark.asyncio
async def test_run_ensemble_timeout() -> None:
    import asyncio

    async def slow():
        await asyncio.sleep(5.0)
        return _plan(_buy())

    async def fast():
        return _plan(_buy())

    variants = [
        EnsembleVariant(name="slow", call=slow),
        EnsembleVariant(name="fast", call=fast),
    ]
    v = await run_ensemble(variants, quorum=1, timeout_s=0.1)
    # Fast variant should still vote; slow timed out.
    assert "slow" not in v.per_variant
    assert "fast" in v.per_variant
