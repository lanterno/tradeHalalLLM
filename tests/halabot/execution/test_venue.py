"""FakeVenue — fills, positions, close; never invents a price (INV-2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.execution.venue import FakeVenue, Order, VenueError

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _venue(**kw) -> FakeVenue:
    return FakeVenue(clock_ts=T0, **kw)


@pytest.mark.asyncio
async def test_buy_fills_and_creates_position():
    v = _venue(prices={"NVDA": 100.0})
    r = await v.place(Order("NVDA", "buy", 3.0, "c1"))
    assert r.is_filled and r.filled_qty == 3.0 and r.filled_price == 100.0
    pos = {p.asset: p for p in await v.positions()}
    assert pos["NVDA"].quantity == 3.0


@pytest.mark.asyncio
async def test_close_flattens_position():
    v = _venue(prices={"NVDA": 100.0})
    await v.place(Order("NVDA", "buy", 3.0, "c1"))
    await v.close("NVDA")
    assert await v.positions() == []  # flat


@pytest.mark.asyncio
async def test_missing_quote_raises_never_invents():
    v = _venue(prices={})  # no price for NVDA
    with pytest.raises(VenueError):
        await v.snapshot("NVDA")
    with pytest.raises(VenueError):
        await v.place(Order("NVDA", "buy", 1.0, "c1"))


@pytest.mark.asyncio
async def test_fail_asset_raises():
    v = _venue(prices={"NVDA": 100.0}, fail_assets={"NVDA"})
    with pytest.raises(VenueError):
        await v.place(Order("NVDA", "buy", 1.0, "c1"))


@pytest.mark.asyncio
async def test_close_with_no_position_is_safe():
    v = _venue(prices={"NVDA": 100.0})
    r = await v.close("NVDA")
    assert r.filled_qty == 0.0  # nothing to close, no error
    assert r.status == "filled"  # well-formed (audit #1: no positional-arg corruption)
    assert r.order_id == "fake-close-NVDA"


@pytest.mark.asyncio
async def test_duplicate_client_id_does_not_double_fill():
    v = _venue(prices={"NVDA": 100.0})
    r1 = await v.place(Order("NVDA", "buy", 3.0, "dup"))
    r2 = await v.place(Order("NVDA", "buy", 3.0, "dup"))  # same client_id
    assert r1.order_id == r2.order_id  # idempotent — prior result returned
    pos = {p.asset: p for p in await v.positions()}
    assert pos["NVDA"].quantity == 3.0  # NOT 6.0 — no double-fill (audit #4)
