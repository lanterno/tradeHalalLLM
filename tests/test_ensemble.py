"""Tests for ensemble LLM judge."""

from __future__ import annotations

import pytest

from halal_trader.core.llm.ensemble import (
    EnsembleVariant,
    aggregate_plans,
    run_ensemble,
)
from halal_trader.domain.models import TradeAction, TradeDecision, TradingPlan


def _buy(symbol: str = "AAPL", qty: int = 10, conf: float = 0.7) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=qty,
        confidence=conf,
        reasoning="x",
    )


def _sell(symbol: str = "MSFT", qty: int = 10, conf: float = 0.6) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.SELL,
        symbol=symbol,
        quantity=qty,
        confidence=conf,
        reasoning="x",
    )


def _plan(*decisions: TradeDecision, outlook: str = "") -> TradingPlan:
    return TradingPlan(decisions=list(decisions), market_outlook=outlook)


# ── Aggregation ───────────────────────────────────────────────────


def test_unanimous_keeps_decision_with_full_multiplier() -> None:
    plans = {
        "a": _plan(_buy(qty=10)),
        "b": _plan(_buy(qty=15)),
        "c": _plan(_buy(qty=5)),
    }
    v = aggregate_plans(plans, quorum=2)
    assert v.agreement_score == 1.0
    assert v.sizing_multiplier == 1.0
    assert len(v.consensus_plan.decisions) == 1
    # median quantity of [5, 10, 15] = 10
    assert v.consensus_plan.decisions[0].quantity == 10


def test_quorum_reached_partial_agreement() -> None:
    plans = {
        "a": _plan(_buy(symbol="AAPL")),
        "b": _plan(_buy(symbol="AAPL")),
        "c": _plan(_buy(symbol="MSFT")),  # disagrees
    }
    v = aggregate_plans(plans, quorum=2)
    # 2 of 3 agreed on AAPL, 1 on MSFT alone -> MSFT dropped
    assert len(v.consensus_plan.decisions) == 1
    assert v.consensus_plan.decisions[0].symbol == "AAPL"
    assert 0.5 < v.agreement_score < 1.0
    assert 0.5 <= v.sizing_multiplier < 1.0


def test_no_quorum_drops_all() -> None:
    plans = {
        "a": _plan(_buy(symbol="AAPL")),
        "b": _plan(_buy(symbol="MSFT")),
        "c": _plan(_buy(symbol="GOOG")),
    }
    v = aggregate_plans(plans, quorum=2)
    assert v.consensus_plan.decisions == []


def test_skip_at_threshold_zeroes_multiplier() -> None:
    plans = {
        "a": _plan(_buy(symbol="AAPL")),
        "b": _plan(_buy(symbol="MSFT")),  # half agreement
    }
    v = aggregate_plans(plans, quorum=1, skip_quorum_at=0.6)
    assert v.sizing_multiplier == 0.0


def test_action_disagreement_buckets_separately() -> None:
    plans = {
        "a": _plan(_buy(symbol="AAPL")),
        "b": _plan(_sell(symbol="AAPL")),  # same symbol, opposite action
    }
    v = aggregate_plans(plans, quorum=2)
    # both reached only 1 vote -> nothing survives
    assert v.consensus_plan.decisions == []
    assert v.counts["AAPL"]["buy"] == 1
    assert v.counts["AAPL"]["sell"] == 1


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_plans({}, quorum=1)


def test_consensus_uses_median_confidence() -> None:
    plans = {
        "a": _plan(_buy(qty=10, conf=0.5)),
        "b": _plan(_buy(qty=10, conf=0.7)),
        "c": _plan(_buy(qty=10, conf=0.9)),
    }
    v = aggregate_plans(plans, quorum=2)
    assert v.consensus_plan.decisions[0].confidence == pytest.approx(0.7)


# ── Driver ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_ensemble_concurrent_success() -> None:
    async def make_plan(symbol: str):
        return _plan(_buy(symbol=symbol))

    variants = [
        EnsembleVariant(name="hot", call=lambda: make_plan("AAPL")),
        EnsembleVariant(name="cool", call=lambda: make_plan("AAPL")),
        EnsembleVariant(name="contrarian", call=lambda: make_plan("MSFT")),
    ]
    v = await run_ensemble(variants, quorum=2)
    assert len(v.consensus_plan.decisions) == 1
    assert v.consensus_plan.decisions[0].symbol == "AAPL"


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
