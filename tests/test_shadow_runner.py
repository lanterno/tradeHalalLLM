"""Tests for the shadow strategy runtime."""

from __future__ import annotations

import pytest

from halal_trader.core.shadow import ShadowLedger
from halal_trader.core.shadow_runner import (
    FrozenPromptStrategy,
    ShadowRunner,
    SimulatedShadowAccount,
)
from halal_trader.domain.models import (
    CryptoTradeDecision,
    CryptoTradingPlan,
    TradeAction,
)


# ── Simulator ────────────────────────────────────────────────────


def test_simulator_buy_adds_position() -> None:
    acct = SimulatedShadowAccount(cash=1000.0)
    d = CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol="BTCUSDT",
        quantity=2.0,
        confidence=0.6,
        reasoning="x",
    )
    acct.apply_decision(d, {"BTCUSDT": 100.0})
    assert acct.positions["BTCUSDT"] == 2.0
    assert acct.cash == 800.0


def test_simulator_buy_clamped_by_cash() -> None:
    acct = SimulatedShadowAccount(cash=100.0)
    d = CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol="BTCUSDT",
        quantity=10.0,
        confidence=0.6,
        reasoning="x",
    )
    acct.apply_decision(d, {"BTCUSDT": 100.0})
    # 10 × 100 = 1000 > 100 cash; clamp to 1.0 quantity
    assert acct.positions["BTCUSDT"] == 1.0
    assert acct.cash == 0.0


def test_simulator_sell_only_what_we_own() -> None:
    acct = SimulatedShadowAccount(cash=0.0, positions={"X": 1.0})
    d = CryptoTradeDecision(
        action=TradeAction.SELL,
        symbol="X",
        quantity=5.0,
        confidence=0.6,
        reasoning="x",
    )
    acct.apply_decision(d, {"X": 50.0})
    assert acct.positions["X"] == 0.0
    assert acct.cash == 50.0


def test_simulator_skips_unknown_price() -> None:
    acct = SimulatedShadowAccount(cash=100.0)
    d = CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol="X",
        quantity=1.0,
        confidence=0.5,
        reasoning="x",
    )
    acct.apply_decision(d, {})  # no price
    assert acct.cash == 100.0
    assert acct.positions == {}


def test_equity_with_positions() -> None:
    acct = SimulatedShadowAccount(cash=500.0, positions={"X": 2.0, "Y": 1.0})
    e = acct.equity({"X": 100.0, "Y": 50.0})
    assert e == 750.0


# ── FrozenPromptStrategy ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_frozen_strategy_tags_plan() -> None:
    class _Inner:
        async def analyze(self):
            return CryptoTradingPlan(decisions=[], market_outlook="x")

    s = FrozenPromptStrategy(inner=_Inner(), frozen_prompt_version="v0@abc")
    plan = await s.analyze()
    assert "frozen_prompt=v0@abc" in plan.risk_notes


@pytest.mark.asyncio
async def test_frozen_strategy_idempotent_tag() -> None:
    class _Inner:
        async def analyze(self):
            return CryptoTradingPlan(
                decisions=[], market_outlook="x", risk_notes="frozen_prompt=v0@abc"
            )

    s = FrozenPromptStrategy(inner=_Inner(), frozen_prompt_version="v0@abc")
    plan = await s.analyze()
    # Don't double-tag — single occurrence
    assert plan.risk_notes.count("frozen_prompt=v0@abc") == 1


# ── ShadowRunner ─────────────────────────────────────────────────


def _buy(symbol: str = "BTCUSDT", qty: float = 1.0) -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=qty,
        confidence=0.6,
        reasoning="x",
    )


@pytest.mark.asyncio
async def test_runner_writes_one_row_per_cycle() -> None:
    class _Strat:
        async def analyze(self, **kw):
            return CryptoTradingPlan(decisions=[])

    led = ShadowLedger()
    r = ShadowRunner(shadow_strategy=_Strat(), ledger=led, starting_cash=1000)
    await r.observe_cycle(
        cycle_id="c1",
        live_equity=1000,
        latest_prices={"BTCUSDT": 100},
        analyze_kwargs={},
    )
    assert led.size == 1
    assert led.entries[0].cycle_id == "c1"


@pytest.mark.asyncio
async def test_runner_simulates_buys_and_diff_emerges() -> None:
    """Live equity stays flat at $1000; shadow takes a 10% gain on the next price."""

    class _Strat:
        def __init__(self):
            self.calls = 0

        async def analyze(self, **kw):
            self.calls += 1
            if self.calls == 1:
                return CryptoTradingPlan(decisions=[_buy(qty=5)])
            return CryptoTradingPlan(decisions=[])

    led = ShadowLedger()
    r = ShadowRunner(shadow_strategy=_Strat(), ledger=led, starting_cash=1000)
    # Cycle 1: shadow buys 5 @ $100 = $500, cash=500, equity=1000
    await r.observe_cycle(
        cycle_id="c1",
        live_equity=1000,
        latest_prices={"BTCUSDT": 100},
        analyze_kwargs={},
    )
    # Cycle 2: price up 10%, shadow holds; equity = 500 + 5×110 = 1050
    await r.observe_cycle(
        cycle_id="c2",
        live_equity=1000,
        latest_prices={"BTCUSDT": 110},
        analyze_kwargs={},
    )
    assert led.size == 2
    assert led.entries[1].shadow_equity == 1050.0
    # Live diff > 0
    assert led.entries[1].diff < 0  # live trails shadow


@pytest.mark.asyncio
async def test_runner_swallows_strategy_error_records_anyway() -> None:
    class _Strat:
        async def analyze(self, **kw):
            raise RuntimeError("boom")

    led = ShadowLedger()
    r = ShadowRunner(shadow_strategy=_Strat(), ledger=led, starting_cash=1000)
    eq = await r.observe_cycle(
        cycle_id="c1",
        live_equity=1000,
        latest_prices={"BTCUSDT": 100},
        analyze_kwargs={},
    )
    assert eq == 1000.0
    assert led.size == 1
