"""Reactor (news-momentum) entry execution — the "fast in" half of the
operator's fast-in/slow-out strategy (memory: strategy-fast-in-slow-out).

``TradeExecutor.execute_reactor_entry`` places a half-size paper BUY on a
high-confidence scored catalyst, but only when the stock is also up on the
session (news + price-up confluence). It routes through ``_execute_buy`` so
it inherits every risk gate and tags the row ``entry_type='reactor_momentum'``
so the slow-out lockout protects it from LLM exits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.domain.models import Account
from halal_trader.trading.executor import TradeExecutor


def _account(portfolio_value: float = 100_000.0) -> Account:
    return Account(
        equity=portfolio_value,
        buying_power=portfolio_value,
        cash=portfolio_value,
        portfolio_value=portfolio_value,
        status="ACTIVE",
    )


def _snapshot(symbol: str, latest: float, prev_close: float) -> dict:
    return {
        symbol: {
            "latest_trade": {"price": latest},
            "prev_daily_bar": {"close": prev_close},
            "daily_bar": {"open": prev_close, "close": latest},
        }
    }


def _executor(broker: MagicMock, repo: MagicMock, **kw) -> TradeExecutor:
    return TradeExecutor(
        broker,
        repo,
        max_position_pct=0.20,
        max_simultaneous_positions=10,
        max_sector_pct=0,
        reactor_entry_size_fraction=0.5,
        reactor_entry_min_intraday_change_pct=0.002,
        **kw,
    )


# ── price-confirmation + sizing gates (return before _execute_buy) ──


@pytest.mark.asyncio
async def test_reactor_entry_skips_when_no_price():
    broker = MagicMock()
    broker.get_stock_snapshot = AsyncMock(return_value={})  # no price
    executor = _executor(broker, MagicMock())

    out = await executor.execute_reactor_entry("NVDA", score=0.9, reasoning="beat")
    assert out["status"] == "skipped"
    assert "no usable price" in out["reason"]


@pytest.mark.asyncio
async def test_reactor_entry_skips_when_price_not_confirming():
    """Bullish headline but the tape is down on the session → no entry."""
    broker = MagicMock()
    broker.get_stock_snapshot = AsyncMock(
        return_value=_snapshot("NVDA", latest=195.0, prev_close=200.0)  # -2.5%
    )
    executor = _executor(broker, MagicMock())

    out = await executor.execute_reactor_entry("NVDA", score=0.95, reasoning="beat")
    assert out["status"] == "skipped"
    assert "price-confirmation failed" in out["reason"]


@pytest.mark.asyncio
async def test_reactor_entry_skips_when_below_one_share():
    """Tiny equity + high price → sized below a whole share, skip."""
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account(portfolio_value=100.0))
    broker.get_stock_snapshot = AsyncMock(
        return_value=_snapshot("NVDA", latest=900.0, prev_close=890.0)  # up, confirms
    )
    executor = _executor(broker, MagicMock())

    out = await executor.execute_reactor_entry("NVDA", score=0.9, reasoning="beat")
    assert out["status"] == "skipped"
    assert "below 1 share" in out["reason"]


# ── happy path: places a half-size, reactor-tagged BUY ──────────────


@pytest.mark.asyncio
async def test_reactor_entry_places_half_size_tagged_buy():
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account(portfolio_value=100_000.0))
    broker.get_stock_snapshot = AsyncMock(
        return_value=_snapshot("NVDA", latest=200.0, prev_close=199.0)  # +0.5%, confirms
    )
    broker.place_order = AsyncMock(return_value={"id": "ord-1"})
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "ord-1",
            "status": "filled",
            "filled_qty": "50",
            "filled_avg_price": "200.10",
        }
    )

    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=42)

    executor = _executor(broker, repo)
    # Neutralize the time-of-day + history gates so the test is deterministic.
    executor._check_market_close_lockout = lambda: None  # type: ignore[method-assign]
    executor._check_recent_close_cooldown = AsyncMock(return_value=None)  # type: ignore[method-assign]
    executor._check_sector_limit = AsyncMock(return_value=None)  # type: ignore[method-assign]

    out = await executor.execute_reactor_entry("NVDA", score=0.92, reasoning="surprise beat")

    assert out["status"] == "filled"
    # target_notional = 100_000 * 0.20 * 0.5 = 10_000 ; / $200 = 50 shares.
    assert out["quantity"] == 50
    kwargs = repo.record_trade.await_args.kwargs
    assert kwargs["entry_type"] == "reactor_momentum"
    assert kwargs["side"] == "buy"
    assert kwargs["quantity"] == 50


@pytest.mark.asyncio
async def test_reactor_entry_respects_existing_risk_gates():
    """If a gate inside _execute_buy rejects (e.g. re-entry cooldown),
    the reactor entry surfaces that rejection rather than forcing a buy."""
    broker = MagicMock()
    broker.get_account_info = AsyncMock(return_value=_account())
    broker.get_stock_snapshot = AsyncMock(
        return_value=_snapshot("NVDA", latest=200.0, prev_close=199.0)
    )
    executor = _executor(broker, MagicMock())
    executor._check_market_close_lockout = lambda: None  # type: ignore[method-assign]
    executor._check_recent_close_cooldown = AsyncMock(  # type: ignore[method-assign]
        return_value="re-entry cooldown active"
    )

    out = await executor.execute_reactor_entry("NVDA", score=0.9, reasoning="beat")
    assert out["status"] == "rejected"
    assert "cooldown" in out["reason"]


# ── intraday-change helper ──────────────────────────────────────────


def test_intraday_change_prefers_prev_close():
    executor = _executor(MagicMock(), MagicMock())
    snap = _snapshot("NVDA", latest=204.0, prev_close=200.0)
    assert executor._extract_intraday_change_pct(snap, "NVDA") == pytest.approx(0.02)


def test_intraday_change_falls_back_to_open():
    executor = _executor(MagicMock(), MagicMock())
    snap = {"NVDA": {"latest_trade": {"price": 101.0}, "daily_bar": {"open": 100.0}}}
    assert executor._extract_intraday_change_pct(snap, "NVDA") == pytest.approx(0.01)


def test_intraday_change_none_without_reference():
    executor = _executor(MagicMock(), MagicMock())
    snap = {"NVDA": {"latest_trade": {"price": 100.0}}}  # no bars
    assert executor._extract_intraday_change_pct(snap, "NVDA") is None
