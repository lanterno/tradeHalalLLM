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
