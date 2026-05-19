"""Tests for TradingCycleService — orchestration + halt threading.

Mirrors ``test_crypto_cycle.py`` for the stocks side. Focuses on the
small contracts that don't need a Postgres engine:

* ``_should_halt`` delegates to the portfolio tracker
* ``_pre_cycle_checks`` short-circuits when the market is closed
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from halal_trader.trading.cycle import TradingCycleService


def _make_service(*, should_halt: bool = False, market_open: bool = True):
    broker = AsyncMock()
    clock = MagicMock()
    clock.is_open = market_open
    clock.next_open = "2026-05-01T09:30:00-04:00"
    clock.next_close = "2026-05-01T16:00:00-04:00"
    broker.get_clock.return_value = clock

    screener = MagicMock()
    strategy = AsyncMock()
    executor = AsyncMock()
    portfolio = AsyncMock()
    portfolio.should_halt_trading.return_value = should_halt

    return TradingCycleService(
        broker=broker,
        screener=screener,
        strategy=strategy,
        executor=executor,
        portfolio=portfolio,
    )


@pytest.mark.asyncio
async def test_should_halt_returns_true_when_portfolio_halted():
    svc = _make_service(should_halt=True)
    assert await svc._should_halt() is True


@pytest.mark.asyncio
async def test_should_halt_returns_false_when_portfolio_ok():
    svc = _make_service(should_halt=False)
    assert await svc._should_halt() is False


@pytest.mark.asyncio
@patch("halal_trader.trading.cycle.is_market_open_local")
async def test_pre_cycle_checks_returns_false_when_local_closed(mock_local):
    """When ``is_market_open_local`` says closed, we skip the cycle without
    even hitting the broker — broker API calls should not run."""
    mock_local.return_value = False
    svc = _make_service()
    assert await svc._pre_cycle_checks() is False
    svc._broker.get_clock.assert_not_awaited()


@pytest.mark.asyncio
@patch("halal_trader.trading.cycle.is_market_open_local")
async def test_pre_cycle_checks_returns_false_when_broker_says_closed(mock_local):
    """Broker clock is the authoritative source — it can disagree with
    the local clock around half-days / holidays. When it says closed
    we skip even if the local check passed."""
    mock_local.return_value = True
    svc = _make_service(market_open=False)
    assert await svc._pre_cycle_checks() is False


@pytest.mark.asyncio
@patch("halal_trader.trading.cycle.is_market_open_local")
async def test_pre_cycle_checks_returns_true_when_both_open(mock_local):
    mock_local.return_value = True
    svc = _make_service(market_open=True)
    assert await svc._pre_cycle_checks() is True


@pytest.mark.asyncio
async def test_post_cycle_no_op_when_no_engine():
    """Without an engine, the post-cycle reconcile must be a clean no-op
    (the dev path runs the bot without a DB-backed reconcile)."""
    svc = _make_service()
    assert svc._engine is None  # default
    # Should not raise.
    await svc._post_cycle()


def test_constructor_threads_through_optional_kwargs():
    """The kwargs added across recent parity work all stick."""
    notifier = MagicMock()
    hub = MagicMock()
    regime = MagicMock()
    timeframes = MagicMock()
    svc = TradingCycleService(
        broker=AsyncMock(),
        screener=MagicMock(),
        strategy=AsyncMock(),
        executor=AsyncMock(),
        portfolio=AsyncMock(),
        regime_detector=regime,
        timeframe_analyzer=timeframes,
        insights_hub=hub,
        notifier=notifier,
    )
    assert svc._regime_detector is regime
    assert svc._timeframes is timeframes
    assert svc._hub is hub
    assert svc._notifier is notifier


def test_constructor_threads_analytics_and_self_review():
    """Stocks-side parity: ``analytics`` + ``self_review`` flow into the
    cycle so ``BuildPerformanceStage`` + ``BuildActiveAdjustmentsStage``
    stamp ``state.performance_text`` / ``state.active_adjustments``."""
    analytics = MagicMock()
    self_review = MagicMock()
    svc = TradingCycleService(
        broker=AsyncMock(),
        screener=MagicMock(),
        strategy=AsyncMock(),
        executor=AsyncMock(),
        portfolio=AsyncMock(),
        analytics=analytics,
        self_review=self_review,
    )
    assert svc._analytics is analytics
    assert svc._self_review is self_review


def test_constructor_analytics_and_self_review_default_to_none():
    """Both default to None — the stages handle the missing collaborator
    gracefully (emit an empty block). Pinned so a future refactor doesn't
    silently promote either to a required parameter."""
    svc = TradingCycleService(
        broker=AsyncMock(),
        screener=MagicMock(),
        strategy=AsyncMock(),
        executor=AsyncMock(),
        portfolio=AsyncMock(),
    )
    assert svc._analytics is None
    assert svc._self_review is None


# ── _handle_execution_results (Round-7 extracted helper) ─────


@pytest.mark.asyncio
async def test_handle_execution_results_records_failure_on_error():
    """``status="error"`` result bumps the self-review's per-symbol
    failure counter so the 10-failures trigger can eventually fire."""
    svc = _make_service()
    self_review = MagicMock()
    self_review.record_execution_failure = MagicMock()
    svc._self_review = self_review

    await svc._handle_execution_results(
        [
            {"symbol": "AAPL", "action": "buy", "status": "error", "reason": "RATE_LIMIT"},
        ]
    )

    self_review.record_execution_failure.assert_called_once_with("AAPL", "RATE_LIMIT")


@pytest.mark.asyncio
async def test_handle_execution_results_records_failure_on_rejected():
    """``status="rejected"`` (e.g. halal screener blocked the buy)
    also bumps the counter — repeated rejections on a symbol are
    operator-actionable signal."""
    svc = _make_service()
    self_review = MagicMock()
    self_review.record_execution_failure = MagicMock()
    svc._self_review = self_review

    await svc._handle_execution_results(
        [
            {
                "symbol": "MSFT",
                "action": "buy",
                "status": "rejected",
                "reason": "below_min_notional",
            },
        ]
    )

    self_review.record_execution_failure.assert_called_once_with("MSFT", "below_min_notional")


@pytest.mark.asyncio
async def test_handle_execution_results_skips_recording_on_filled():
    """Happy path: a filled buy must NOT bump the failure counter
    (otherwise every successful trade would trigger reviews)."""
    svc = _make_service()
    self_review = MagicMock()
    self_review.record_execution_failure = MagicMock()
    svc._self_review = self_review

    await svc._handle_execution_results(
        [
            {
                "symbol": "AAPL",
                "action": "buy",
                "status": "filled",
                "quantity": 10,
                "price": 180.0,
            },
        ]
    )

    self_review.record_execution_failure.assert_not_called()


@pytest.mark.asyncio
async def test_handle_execution_results_no_op_when_self_review_none():
    """Default wiring (``self_review=None``) — error results just log,
    no AttributeError on the missing call site."""
    svc = _make_service()
    assert svc._self_review is None  # default

    # Must not raise.
    await svc._handle_execution_results(
        [{"symbol": "AAPL", "action": "buy", "status": "error", "reason": "x"}]
    )


@pytest.mark.asyncio
async def test_handle_execution_results_swallows_recorder_exception():
    """A broken recorder must not block subsequent notifications.
    The filled result that follows the failing one must still
    trigger ``notify_trade``."""
    from unittest.mock import AsyncMock

    svc = _make_service()
    self_review = MagicMock()
    self_review.record_execution_failure = MagicMock(side_effect=RuntimeError("boom"))
    svc._self_review = self_review

    notifier = MagicMock()
    notifier.notify_trade = AsyncMock()
    svc._notifier = notifier

    await svc._handle_execution_results(
        [
            {"symbol": "AAPL", "action": "buy", "status": "error", "reason": "x"},
            {
                "symbol": "MSFT",
                "action": "buy",
                "status": "filled",
                "quantity": 5,
                "price": 400.0,
            },
        ]
    )

    # Recorder blew up but the notifier still ran on the second result.
    notifier.notify_trade.assert_awaited_once()
