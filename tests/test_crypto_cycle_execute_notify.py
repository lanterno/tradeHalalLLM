"""Tests for `CryptoCycleService._execute_and_notify` branches.

`test_crypto_cycle.py` has the happy-path test for "shadow_runner=None,
buy fills". This file pins the remaining branches: shadow_runner
observe-cycle raises (swallowed → cycle continues), notifier raises
(swallowed → cycle continues + snapshot still recorded), snapshot
recording raises (swallowed → notifier still called), and the
empty-decisions skip path (no executor call).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.crypto.cycle import CryptoCycleService
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


def _service(
    *,
    executor: AsyncMock | None = None,
    portfolio: AsyncMock | None = None,
    notifier=None,
    shadow_runner=None,
):
    """Build a minimal service for `_execute_and_notify` testing."""
    svc = CryptoCycleService(
        broker=MagicMock(),
        screener=AsyncMock(),
        strategy=AsyncMock(),
        executor=executor or AsyncMock(),
        portfolio=portfolio or AsyncMock(),
        ws_manager=MagicMock(),
        configured_pairs=["BTCUSDT"],
        notifier=notifier,
        shadow_runner=shadow_runner,
    )
    return svc


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


# ── Empty-decisions skip path ─────────────────────────────


@pytest.mark.asyncio
async def test_empty_decisions_skips_executor_call():
    """No decisions → executor.execute_plan must NOT be called.
    Saves a round-trip and a tracer span on hold cycles."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    svc = _service(executor=executor)
    await svc._execute_and_notify(
        _empty_plan(),
        account=_account(),
        indicators_cache={},
        klines_by_symbol={},
        shadow_kwargs={},
    )
    executor.execute_plan.assert_not_called()


# ── Shadow runner branches ────────────────────────────────


@pytest.mark.asyncio
async def test_shadow_runner_observe_failure_is_swallowed():
    """If `shadow_runner.observe_cycle` raises (e.g. DB hiccup), the
    cycle must continue — observation is best-effort. Pin so a
    refactor that re-raises the shadow error doesn't take down the
    whole cycle."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock(side_effect=RuntimeError("shadow DB down"))

    svc = _service(executor=executor, shadow_runner=shadow)
    # Must NOT raise.
    await svc._execute_and_notify(
        _empty_plan(),
        account=_account(),
        indicators_cache={},
        klines_by_symbol={},
        shadow_kwargs={"x": 1},
    )
    shadow.observe_cycle.assert_awaited_once()


@pytest.mark.asyncio
async def test_shadow_runner_observed_with_latest_prices_from_klines():
    """The runner gets `latest_prices` derived from each pair's last
    kline close — pin so the shape doesn't drift (the runner's own
    divergence math depends on it)."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock()

    svc = _service(executor=executor, shadow_runner=shadow)
    await svc._execute_and_notify(
        _empty_plan(),
        account=_account(total=10_000.0),
        indicators_cache={},
        klines_by_symbol={"BTCUSDT": [_kline(close=50_000.0)], "ETHUSDT": [_kline(close=3_000.0)]},
        shadow_kwargs={},
    )
    kw = shadow.observe_cycle.await_args.kwargs
    assert kw["latest_prices"] == {"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0}
    assert kw["live_equity"] == 10_000.0


@pytest.mark.asyncio
async def test_shadow_runner_handles_empty_klines_per_pair():
    """A pair with no klines → 0.0 fallback in the latest_prices dict
    (the runner branches on zero internally)."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock()

    svc = _service(executor=executor, shadow_runner=shadow)
    await svc._execute_and_notify(
        _empty_plan(),
        account=_account(),
        indicators_cache={},
        klines_by_symbol={"BTCUSDT": []},  # empty list
        shadow_kwargs={},
    )
    assert shadow.observe_cycle.await_args.kwargs["latest_prices"] == {"BTCUSDT": 0.0}


@pytest.mark.asyncio
async def test_shadow_runner_zero_equity_fallback():
    """If account equity is 0 (cold start), `live_equity` defaults
    to 0.0 not None — runner expects a float."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(return_value=[])

    shadow = MagicMock()
    shadow.observe_cycle = AsyncMock()

    svc = _service(executor=executor, shadow_runner=shadow)
    await svc._execute_and_notify(
        _empty_plan(),
        account=CryptoAccount(
            total_balance_usdt=0.0,
            available_balance_usdt=0.0,
            in_order_usdt=0.0,
            usdt_free=0.0,
        ),
        indicators_cache={},
        klines_by_symbol={},
        shadow_kwargs={},
    )
    assert shadow.observe_cycle.await_args.kwargs["live_equity"] == 0.0


# ── Notifier failure ──────────────────────────────────────


@pytest.mark.asyncio
async def test_notifier_failure_does_not_block_snapshot_recording():
    """If `notifier.notify_trade` raises (Telegram down), the indicator
    snapshot for ML retraining must STILL be recorded — they're
    independent best-effort steps."""
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

    svc = _service(executor=executor, portfolio=portfolio, notifier=notifier)
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={"BTCUSDT": {"rsi_14": 30.0}},
        klines_by_symbol={"BTCUSDT": [_kline()]},
        shadow_kwargs={},
    )
    # Notifier was called (and failed), but snapshot was still recorded.
    notifier.notify_trade.assert_awaited_once()
    portfolio.record_indicator_snapshot.assert_awaited_once()


# ── Snapshot recording failure ────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_block_notifier():
    """Mirror of the above: snapshot DB write fails → notifier still
    fires (operator alert is more time-sensitive than the ML label,
    which can be backfilled)."""
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

    svc = _service(executor=executor, portfolio=portfolio, notifier=notifier)
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={"BTCUSDT": {"rsi_14": 30.0}},
        klines_by_symbol={},
        shadow_kwargs={},
    )
    portfolio.record_indicator_snapshot.assert_awaited_once()  # tried + failed
    notifier.notify_trade.assert_awaited_once()  # still fired


# ── Snapshot conditions ───────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_skipped_when_status_is_rejected():
    """`status='rejected'` → no snapshot (the trade didn't happen).
    Keeps the ML retraining label set clean."""
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

    svc = _service(executor=executor, portfolio=portfolio)
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={"BTCUSDT": {"rsi_14": 30.0}},
        klines_by_symbol={},
        shadow_kwargs={},
    )
    portfolio.record_indicator_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_skipped_for_sell_actions():
    """`action='sell'` → no snapshot (snapshots are entry-only; the
    close path has its own labelling via post_close.record_close).
    Pin so a refactor that flips the action check doesn't double-snap."""
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

    svc = _service(executor=executor, portfolio=portfolio)
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={"BTCUSDT": {"rsi_14": 70.0}},
        klines_by_symbol={},
        shadow_kwargs={},
    )
    portfolio.record_indicator_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_skipped_when_symbol_not_in_indicators_cache():
    """A buy fill for a pair we don't have indicators for (e.g. cycle
    fetched klines but indicators failed) → skip snapshot rather than
    record an empty vector."""
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

    svc = _service(executor=executor, portfolio=portfolio)
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={},  # no indicators
        klines_by_symbol={},
        shadow_kwargs={},
    )
    portfolio.record_indicator_snapshot.assert_not_awaited()


# ── Notifier conditions ───────────────────────────────────


@pytest.mark.asyncio
async def test_notifier_fires_on_submitted_status_too():
    """Both `submitted` AND `filled` trigger the notifier (operator
    wants to know the moment the order is on the wire, not just
    when it fully fills)."""
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

    svc = _service(executor=executor, notifier=notifier)
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={},
        klines_by_symbol={},
        shadow_kwargs={},
    )
    notifier.notify_trade.assert_awaited_once()


@pytest.mark.asyncio
async def test_notifier_skipped_on_rejected_status():
    """`rejected` → no notification (operator already sees rejected
    trades in the log; another Telegram message would be noise)."""
    executor = AsyncMock()
    executor.execute_plan = AsyncMock(
        return_value=[{"symbol": "BTCUSDT", "action": "buy", "status": "rejected"}]
    )

    notifier = MagicMock()
    notifier.notify_trade = AsyncMock()

    svc = _service(executor=executor, notifier=notifier)
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={},
        klines_by_symbol={},
        shadow_kwargs={},
    )
    notifier.notify_trade.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_notifier_wired_skips_silently():
    """If the operator hasn't configured Telegram, `notifier=None` →
    silent skip (no AttributeError on the call site)."""
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

    svc = _service(executor=executor)  # notifier=None
    # Must not raise.
    await svc._execute_and_notify(
        _plan_with_buy(),
        account=_account(),
        indicators_cache={},
        klines_by_symbol={},
        shadow_kwargs={},
    )
