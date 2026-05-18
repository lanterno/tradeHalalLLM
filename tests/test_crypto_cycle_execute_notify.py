"""Tests for :class:`ExecuteAndNotifyStage` branches.

Pins: empty-decisions skip (no executor call), shadow-runner observe
failure swallowed, latest-prices shape, empty-klines fallback, zero-
equity fallback, notifier failure doesn't block snapshot, snapshot
failure doesn't block notifier, snapshot skipped on rejected/sell, no
notifier wired silently skips.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle_pipeline import CycleState
from halal_trader.core.cycle_stages import ExecuteAndNotifyStage
from halal_trader.domain.models import (
    CryptoAccount,
    CryptoTradeDecision,
    CryptoTradingPlan,
    Kline,
    TradeAction,
)


def _kline(close: float = 50_000.0) -> Kline:
    return Kline(
        open_time=1,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        close_time=2,
    )


def _account(total: float = 10_000.0) -> CryptoAccount:
    return CryptoAccount(
        total_balance_usdt=total,
        available_balance_usdt=total * 0.8,
        in_order_usdt=total * 0.2,
        usdt_free=total * 0.8,
    )


def _plan_with_buy() -> CryptoTradingPlan:
    return CryptoTradingPlan(
        decisions=[
            CryptoTradeDecision(
                action=TradeAction.BUY,
                symbol="BTCUSDT",
                quantity=0.001,
                confidence=0.8,
                reasoning="test",
            )
        ],
        market_outlook="bullish",
    )


def _empty_plan() -> CryptoTradingPlan:
    return CryptoTradingPlan(decisions=[], market_outlook="hold")


def _stage(
    *,
    executor: AsyncMock | None = None,
    portfolio: AsyncMock | None = None,
    notifier=None,
    shadow_runner=None,
    shadow_kwargs: dict | None = None,
) -> ExecuteAndNotifyStage:
    return ExecuteAndNotifyStage(
        executor=executor or AsyncMock(),
        portfolio=portfolio or AsyncMock(),
        notifier=notifier,
        shadow_runner=shadow_runner,
        shadow_kwargs_builder=(lambda _s: shadow_kwargs or {}) if shadow_runner else None,
    )


def _state(
    *,
    plan: CryptoTradingPlan,
    account: CryptoAccount,
    indicators_cache: dict | None = None,
    klines_by_symbol: dict | None = None,
) -> CycleState:
    s = CycleState()
    s.plan = plan
    s.account = account
    s.indicators_cache = indicators_cache or {}
    s.klines_by_symbol = klines_by_symbol or {}
    return s


# ── Empty-decisions skip path ─────────────────────────────


@pytest.mark.asyncio
async def test_empty_decisions_skips_executor_call():
    """No decisions → ``executor.execute_plan`` must NOT be called.
    Saves a round-trip and a tracer span on hold cycles."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    stage = _stage(executor=executor)
    await stage.run(_state(plan=_empty_plan(), account=_account()))
    executor.execute_plan.assert_not_called()


# ── Shadow runner branches ────────────────────────────────


@pytest.mark.asyncio
async def test_shadow_runner_observe_failure_is_swallowed():
    """If ``shadow_runner.observe_cycle`` raises (e.g. DB hiccup), the
    cycle must continue — observation is best-effort."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock(side_effect=RuntimeError("shadow DB down"))

    stage = _stage(executor=executor, shadow_runner=shadow, shadow_kwargs={"x": 1})
    await stage.run(_state(plan=_empty_plan(), account=_account()))
    shadow.observe_cycle.assert_awaited_once()


@pytest.mark.asyncio
async def test_shadow_runner_observed_with_latest_prices_from_klines():
    """The runner gets ``latest_prices`` derived from each pair's last
    kline close — pin so the shape doesn't drift."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock()

    stage = _stage(executor=executor, shadow_runner=shadow)
    await stage.run(
        _state(
            plan=_empty_plan(),
            account=_account(total=10_000.0),
            klines_by_symbol={
                "BTCUSDT": [_kline(close=50_000.0)],
                "ETHUSDT": [_kline(close=3_000.0)],
            },
        )
    )
    kw = shadow.observe_cycle.await_args.kwargs
    assert kw["latest_prices"] == {"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0}
    assert kw["live_equity"] == 10_000.0


@pytest.mark.asyncio
async def test_shadow_runner_handles_empty_klines_per_pair():
    """A pair with no klines → 0.0 fallback in the ``latest_prices`` dict."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock()

    stage = _stage(executor=executor, shadow_runner=shadow)
    await stage.run(
        _state(plan=_empty_plan(), account=_account(), klines_by_symbol={"BTCUSDT": []})
    )
    assert shadow.observe_cycle.await_args.kwargs["latest_prices"] == {"BTCUSDT": 0.0}


@pytest.mark.asyncio
async def test_shadow_runner_zero_equity_fallback():
    """Zero/None equity (cold start) → ``live_equity`` defaults to 0.0."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock()

    stage = _stage(executor=executor, shadow_runner=shadow)
    await stage.run(
        _state(
            plan=_empty_plan(),
            account=CryptoAccount(
                total_balance_usdt=0.0,
                available_balance_usdt=0.0,
                in_order_usdt=0.0,
                usdt_free=0.0,
            ),
        )
    )
    assert shadow.observe_cycle.await_args.kwargs["live_equity"] == 0.0


# ── Notifier failure ──────────────────────────────────────


@pytest.mark.asyncio
async def test_notifier_failure_does_not_block_snapshot_recording():
    """If ``notifier.notify_trade`` raises (Telegram down), the indicator
    snapshot for ML retraining must STILL be recorded."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "action": "buy",
                "status": "filled",
                "trade_id": 42,
                "quantity": 0.001,
                "price": 50_000.0,
            }
        ]
    )

    portfolio = AsyncMock()
    portfolio.record_indicator_snapshot = AsyncMock()

    notifier = MagicMock()
    notifier.notify_trade = AsyncMock(side_effect=RuntimeError("Telegram 403"))

    stage = _stage(executor=executor, portfolio=portfolio, notifier=notifier)
    await stage.run(
        _state(
            plan=_plan_with_buy(),
            account=_account(),
            indicators_cache={"BTCUSDT": {"rsi_14": 30.0}},
            klines_by_symbol={"BTCUSDT": [_kline()]},
        )
    )
    notifier.notify_trade.assert_awaited_once()
    portfolio.record_indicator_snapshot.assert_awaited_once()


# ── Snapshot recording failure ────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_block_notifier():
    """Mirror: snapshot DB write fails → notifier still fires."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "action": "buy",
                "status": "filled",
                "trade_id": 42,
                "quantity": 0.001,
                "price": 50_000.0,
            }
        ]
    )

    portfolio = AsyncMock()
    portfolio.record_indicator_snapshot = AsyncMock(side_effect=RuntimeError("DB locked"))

    notifier = MagicMock()
    notifier.notify_trade = AsyncMock()

    stage = _stage(executor=executor, portfolio=portfolio, notifier=notifier)
    await stage.run(
        _state(
            plan=_plan_with_buy(),
            account=_account(),
            indicators_cache={"BTCUSDT": {"rsi_14": 30.0}},
        )
    )
    portfolio.record_indicator_snapshot.assert_awaited_once()
    notifier.notify_trade.assert_awaited_once()


# ── Snapshot conditions ───────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_skipped_when_status_is_rejected():
    """``status='rejected'`` → no snapshot."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "action": "buy",
                "status": "rejected",
                "trade_id": None,
                "reason": "insufficient buying power",
            }
        ]
    )
    portfolio = AsyncMock()
    portfolio.record_indicator_snapshot = AsyncMock()

    stage = _stage(executor=executor, portfolio=portfolio)
    await stage.run(
        _state(
            plan=_plan_with_buy(),
            account=_account(),
            indicators_cache={"BTCUSDT": {"rsi_14": 30.0}},
        )
    )
    portfolio.record_indicator_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_skipped_for_sell_actions():
    """``action='sell'`` → no snapshot (snapshots are entry-only)."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "action": "sell",
                "status": "filled",
                "trade_id": 99,
                "quantity": 0.001,
                "price": 50_000.0,
            }
        ]
    )
    portfolio = AsyncMock()
    portfolio.record_indicator_snapshot = AsyncMock()

    stage = _stage(executor=executor, portfolio=portfolio)
    await stage.run(
        _state(
            plan=_plan_with_buy(),
            account=_account(),
            indicators_cache={"BTCUSDT": {"rsi_14": 70.0}},
        )
    )
    portfolio.record_indicator_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_skipped_when_symbol_not_in_indicators_cache():
    """A buy fill for a pair we don't have indicators for → skip snapshot."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "action": "buy",
                "status": "filled",
                "trade_id": 42,
                "quantity": 0.001,
                "price": 50_000.0,
            }
        ]
    )
    portfolio = AsyncMock()
    portfolio.record_indicator_snapshot = AsyncMock()

    stage = _stage(executor=executor, portfolio=portfolio)
    await stage.run(_state(plan=_plan_with_buy(), account=_account()))
    portfolio.record_indicator_snapshot.assert_not_awaited()


# ── Notifier conditions ───────────────────────────────────


@pytest.mark.asyncio
async def test_notifier_fires_on_submitted_status_too():
    """Both ``submitted`` AND ``filled`` trigger the notifier."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "action": "buy",
                "status": "submitted",
                "trade_id": 42,
                "quantity": 0.001,
                "price": 50_000.0,
            }
        ]
    )

    notifier = MagicMock()
    notifier.notify_trade = AsyncMock()

    stage = _stage(executor=executor, notifier=notifier)
    await stage.run(_state(plan=_plan_with_buy(), account=_account()))
    notifier.notify_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_notifier_skipped_on_rejected_status():
    """``rejected`` → no notification."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[{"symbol": "BTCUSDT", "action": "buy", "status": "rejected"}]
    )

    notifier = MagicMock()
    notifier.notify_trade = AsyncMock()

    stage = _stage(executor=executor, notifier=notifier)
    await stage.run(_state(plan=_plan_with_buy(), account=_account()))
    notifier.notify_trade.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_notifier_wired_skips_silently():
    """``notifier=None`` → silent skip."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[
            {
                "symbol": "BTCUSDT",
                "action": "buy",
                "status": "filled",
                "trade_id": 1,
                "quantity": 1,
                "price": 1,
            }
        ]
    )

    stage = _stage(executor=executor)
    await stage.run(_state(plan=_plan_with_buy(), account=_account()))


def test_execute_and_notify_stage_has_stable_name():
    stage = _stage(executor=AsyncMock())
    assert stage.name == "execute_and_notify"
