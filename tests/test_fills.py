"""Tests for core/fills.py — order fill confirmation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halal_trader.core.fills import FillResult, confirm_alpaca


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


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
