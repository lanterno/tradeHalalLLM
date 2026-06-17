"""Tests for `TradingCycleService._fetch_market_data`.

The stocks-side market-data sweep (snapshot + bars per symbol) has
per-symbol exception isolation and a 20-symbol cap. Untested today —
a regression here would either over-pressure Alpaca's free tier
(rate-limit storm) or silently drop symbols from the cycle's view.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.trading.cycle import TradingCycleService


def _service(broker: AsyncMock | None = None) -> TradingCycleService:
    return TradingCycleService(
        broker=broker or AsyncMock(),
        screener=MagicMock(),
        strategy=AsyncMock(),
        executor=AsyncMock(),
        portfolio=AsyncMock(),
    )


# ── Happy path ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_market_data_returns_snapshots_and_bars():
    """Both snapshots and bars are populated per symbol."""
    broker = AsyncMock()
    broker.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": 150.0}})
    broker.get_stock_bars = AsyncMock(return_value=[{"o": 150, "c": 151, "h": 152, "l": 149}])

    svc = _service(broker)
    snapshots, bars = await svc._fetch_market_data(["AAPL", "MSFT"])

    assert "AAPL" in snapshots and "MSFT" in snapshots
    assert "AAPL" in bars and "MSFT" in bars
    assert broker.get_stock_snapshot.await_count == 2
    assert broker.get_stock_bars.await_count == 2


@pytest.mark.asyncio
async def test_fetch_market_data_uses_60day_1day_bars():
    """The bar-fetch must request enough daily history to clear the 30-bar
    indicator/snapshot floor. The prior 5-day window yielded only ~3
    trading-day bars, so every ML snapshot was skipped and the risk/regime
    indicators ran near-empty. Pin 60 days (≈42 trading bars) so a refactor
    that shrinks this has to confront the floor again."""
    broker = AsyncMock()
    broker.get_stock_snapshot = AsyncMock(return_value={})
    broker.get_stock_bars = AsyncMock(return_value=[])

    svc = _service(broker)
    await svc._fetch_market_data(["AAPL"])

    broker.get_stock_bars.assert_awaited_once()
    kwargs = broker.get_stock_bars.call_args.kwargs
    assert kwargs == {"days": 60, "timeframe": "1Day"}


# ── Per-symbol exception isolation ─────────────────────────


@pytest.mark.asyncio
async def test_fetch_market_data_isolates_snapshot_failure():
    """A single snapshot failure must NOT drop the bar fetch for the
    same symbol or affect other symbols. Pin so a transient Alpaca
    blip doesn't lose the whole cycle's data."""
    broker = AsyncMock()

    async def get_snapshot(sym):
        if sym == "AAPL":
            raise RuntimeError("snapshot timeout")
        return {"latest_trade": {"price": 100.0}}

    broker.get_stock_snapshot = AsyncMock(side_effect=get_snapshot)
    broker.get_stock_bars = AsyncMock(return_value=[{"o": 100, "c": 101}])

    svc = _service(broker)
    snapshots, bars = await svc._fetch_market_data(["AAPL", "MSFT"])

    assert "AAPL" not in snapshots  # snapshot failed
    assert "AAPL" in bars  # bars still fetched
    assert "MSFT" in snapshots
    assert "MSFT" in bars


@pytest.mark.asyncio
async def test_fetch_market_data_isolates_bar_failure():
    """Mirror: a bar-fetch failure for one symbol doesn't drop the
    snapshot for the same symbol or affect others."""
    broker = AsyncMock()
    broker.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": 100.0}})

    async def get_bars(sym, **_):
        if sym == "AAPL":
            raise RuntimeError("bars timeout")
        return [{"o": 100, "c": 101}]

    broker.get_stock_bars = AsyncMock(side_effect=get_bars)

    svc = _service(broker)
    snapshots, bars = await svc._fetch_market_data(["AAPL", "MSFT"])

    assert "AAPL" in snapshots
    assert "AAPL" not in bars
    assert "MSFT" in snapshots
    assert "MSFT" in bars


@pytest.mark.asyncio
async def test_fetch_market_data_both_failed_for_one_symbol():
    """Snapshot AND bars fail for one symbol → that symbol is absent
    from both dicts; others continue normally."""
    broker = AsyncMock()

    async def fail_for_aapl(sym, **_):
        if sym == "AAPL":
            raise RuntimeError("alpaca down for AAPL")
        return {"latest_trade": {"price": 100.0}}

    broker.get_stock_snapshot = AsyncMock(side_effect=fail_for_aapl)
    broker.get_stock_bars = AsyncMock(side_effect=fail_for_aapl)

    svc = _service(broker)
    snapshots, bars = await svc._fetch_market_data(["AAPL", "MSFT"])

    assert "AAPL" not in snapshots
    assert "AAPL" not in bars
    assert "MSFT" in snapshots
    assert "MSFT" in bars


# ── 20-symbol cap ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_market_data_caps_at_20_symbols():
    """Universe sizes well above 20 are capped to 20 — pin the value
    (load-bearing for Alpaca's free-tier rate limit)."""
    broker = AsyncMock()
    broker.get_stock_snapshot = AsyncMock(return_value={})
    broker.get_stock_bars = AsyncMock(return_value=[])

    svc = _service(broker)
    universe = [f"S{i:03d}" for i in range(50)]
    snapshots, bars = await svc._fetch_market_data(universe)

    # Only the first 20 are fetched, regardless of how many are halal.
    assert broker.get_stock_snapshot.await_count == 20
    assert broker.get_stock_bars.await_count == 20
    assert len(snapshots) == 20
    assert len(bars) == 20
    # The cap is on the *first* 20 in input order — pin so a refactor
    # that randomises doesn't silently change which symbols get traded.
    assert "S000" in snapshots
    assert "S019" in snapshots
    assert "S020" not in snapshots
    assert "S049" not in snapshots


@pytest.mark.asyncio
async def test_fetch_market_data_under_cap_processes_all():
    """When the universe is smaller than the cap, all symbols are
    fetched — no off-by-one truncation."""
    broker = AsyncMock()
    broker.get_stock_snapshot = AsyncMock(return_value={})
    broker.get_stock_bars = AsyncMock(return_value=[])

    svc = _service(broker)
    await svc._fetch_market_data(["AAPL", "MSFT", "GOOG"])

    assert broker.get_stock_snapshot.await_count == 3
    assert broker.get_stock_bars.await_count == 3


# ── Edge cases ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_market_data_empty_input_returns_empty_dicts():
    broker = AsyncMock()
    broker.get_stock_snapshot = AsyncMock()
    broker.get_stock_bars = AsyncMock()

    svc = _service(broker)
    snapshots, bars = await svc._fetch_market_data([])

    assert snapshots == {}
    assert bars == {}
    broker.get_stock_snapshot.assert_not_awaited()
    broker.get_stock_bars.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_market_data_all_failures_returns_empty_dicts():
    """If Alpaca is fully down (every call fails), both dicts come
    back empty — the cycle continues with degraded data rather
    than crashing."""
    broker = AsyncMock()
    broker.get_stock_snapshot = AsyncMock(side_effect=RuntimeError("alpaca down"))
    broker.get_stock_bars = AsyncMock(side_effect=RuntimeError("alpaca down"))

    svc = _service(broker)
    snapshots, bars = await svc._fetch_market_data(["AAPL", "MSFT"])

    assert snapshots == {}
    assert bars == {}
    # Both calls were attempted for each symbol.
    assert broker.get_stock_snapshot.await_count == 2
    assert broker.get_stock_bars.await_count == 2
