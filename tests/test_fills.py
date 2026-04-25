"""Tests for core/fills.py — order fill confirmation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halal_trader.core.fills import FillResult, confirm_alpaca, confirm_binance


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


# ── Binance ────────────────────────────────────────────────────


def test_confirm_binance_fully_filled_with_fills():
    response = {
        "orderId": 12345,
        "status": "FILLED",
        "executedQty": "0.05",
        "cumulativeQuoteQty": "3415.00",
        "fills": [
            {"price": "68000.0", "qty": "0.03"},
            {"price": "68500.0", "qty": "0.02"},
        ],
    }
    result = confirm_binance(response, _ts())
    assert result.status == "filled"
    assert result.order_id == "12345"
    assert result.filled_quantity == pytest.approx(0.05)
    assert result.filled_price == pytest.approx((68000 * 0.03 + 68500 * 0.02) / 0.05)
    assert result.filled_at is not None


def test_confirm_binance_falls_back_to_cumulative():
    response = {
        "orderId": "abc",
        "status": "FILLED",
        "executedQty": "1.0",
        "cumulativeQuoteQty": "100.0",
        "fills": [],
    }
    result = confirm_binance(response, _ts())
    assert result.filled_quantity == 1.0
    assert result.filled_price == 100.0


def test_confirm_binance_partial_fill():
    response = {
        "orderId": "p1",
        "status": "PARTIALLY_FILLED",
        "executedQty": "0.5",
        "cumulativeQuoteQty": "50.0",
        "fills": [],
    }
    result = confirm_binance(response, _ts())
    assert result.status == "partially_filled"
    assert result.filled_quantity == 0.5
    assert result.filled_at is None  # only set on full fill


def test_confirm_binance_rejected():
    response = {"orderId": "r1", "status": "REJECTED"}
    result = confirm_binance(response, _ts())
    assert result.status == "rejected"
    assert result.filled_quantity == 0.0
    assert result.filled_price is None


def test_confirm_binance_pending_with_no_data():
    response = {"orderId": ""}
    result = confirm_binance(response, _ts())
    assert result.status == "pending"
    assert result.filled_quantity == 0.0


# ── Alpaca ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_confirm_alpaca_filled_immediately():
    async def poller() -> dict:
        return {"status": "filled", "filled_qty": "10", "filled_avg_price": "190.50"}

    result = await confirm_alpaca(
        poll=poller,
        order_id="alp-1",
        submitted_at=_ts(),
        timeout=5,
        interval=0.1,
    )
    assert result.status == "filled"
    assert result.order_id == "alp-1"
    assert result.filled_quantity == 10.0
    assert result.filled_price == 190.50
    assert result.filled_at is not None


@pytest.mark.asyncio
async def test_confirm_alpaca_polls_until_filled():
    states = iter(
        [
            {"status": "new", "filled_qty": "0"},
            {"status": "partially_filled", "filled_qty": "5"},
            {"status": "filled", "filled_qty": "10", "filled_avg_price": "100.0"},
        ]
    )

    async def poller() -> dict:
        return next(states)

    result = await confirm_alpaca(
        poll=poller,
        order_id="alp-2",
        submitted_at=_ts(),
        timeout=10,
        interval=0.01,
    )
    assert result.status == "filled"
    assert result.filled_quantity == 10.0


@pytest.mark.asyncio
async def test_confirm_alpaca_timeout_with_partial_fill():
    async def poller() -> dict:
        return {"status": "new", "filled_qty": "3"}

    result = await confirm_alpaca(
        poll=poller,
        order_id="alp-3",
        submitted_at=_ts(),
        timeout=0.1,
        interval=0.05,
    )
    assert result.status == "partially_filled"
    assert result.filled_quantity == 3.0


@pytest.mark.asyncio
async def test_confirm_alpaca_timeout_no_fill():
    async def poller() -> dict:
        return {"status": "new", "filled_qty": "0"}

    result = await confirm_alpaca(
        poll=poller,
        order_id="alp-4",
        submitted_at=_ts(),
        timeout=0.1,
        interval=0.05,
    )
    assert result.status == "pending"
    assert result.filled_quantity == 0.0


@pytest.mark.asyncio
async def test_confirm_alpaca_rejected_terminal():
    async def poller() -> dict:
        return {"status": "rejected"}

    result = await confirm_alpaca(
        poll=poller,
        order_id="alp-5",
        submitted_at=_ts(),
        timeout=5,
        interval=0.05,
    )
    assert result.status == "rejected"
    assert isinstance(result, FillResult)
