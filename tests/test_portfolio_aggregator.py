"""Tests for `brokers.aggregator.PortfolioAggregator`.

The aggregator gives the dashboard / risk engine / mobile summary a
unified view across N wired brokers. Critical contract: a per-broker
failure must NOT abort the snapshot — the operator sees what they
have AND what's broken.
"""

from __future__ import annotations

import asyncio

import pytest

from halal_trader.brokers import AggregatedPortfolio, BrokerHealth, PortfolioAggregator
from halal_trader.brokers.paper import PaperCryptoBroker, PaperStockBroker

# ── Happy-path snapshot ────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_aggregator_returns_zero_equity():
    """No brokers wired → zero equity, no positions, no health rows."""
    agg = PortfolioAggregator()
    snap = await agg.snapshot()
    assert isinstance(snap, AggregatedPortfolio)
    assert snap.total_equity_usd == 0.0
    assert snap.stocks_equity_usd == 0.0
    assert snap.crypto_equity_usd == 0.0
    assert snap.stock_positions == []
    assert snap.crypto_balances_by_asset == {}
    assert snap.per_broker_equity == {}
    assert snap.per_broker_health == []
    assert snap.has_failures is False


@pytest.mark.asyncio
async def test_single_stock_broker_snapshot():
    """One stock broker → its equity is the total."""
    sb = PaperStockBroker(starting_cash=50_000.0)
    sb.set_seed_price("AAPL", 150.0)
    await sb.place_order("AAPL", "buy", 10)  # invests 1500 + slippage

    agg = PortfolioAggregator()
    agg.add_stock_broker("alpaca-paper", sb)
    snap = await agg.snapshot()

    assert snap.stocks_equity_usd > 0
    assert snap.crypto_equity_usd == 0.0
    assert snap.total_equity_usd == snap.stocks_equity_usd
    assert "alpaca-paper" in snap.per_broker_equity
    assert len(snap.stock_positions) == 1
    assert snap.stock_positions[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_single_crypto_broker_snapshot():
    cb = PaperCryptoBroker(starting_usdt=20_000.0)
    cb.set_seed_price("BTCUSDT", 50_000.0)

    agg = PortfolioAggregator()
    agg.add_crypto_broker("binance-paper", cb)
    snap = await agg.snapshot()

    assert snap.crypto_equity_usd == pytest.approx(20_000.0)
    assert snap.stocks_equity_usd == 0.0
    assert "binance-paper" in snap.per_broker_equity
    # USDT balance still in the unified balances dict.
    assert snap.crypto_balances_by_asset.get("USDT") == pytest.approx(20_000.0)


@pytest.mark.asyncio
async def test_multi_broker_unified_equity():
    """Two stock brokers + one crypto broker → equity sums correctly,
    per-broker breakdown attributed."""
    sb_a = PaperStockBroker(starting_cash=50_000.0)
    sb_b = PaperStockBroker(starting_cash=30_000.0)
    cb = PaperCryptoBroker(starting_usdt=20_000.0)

    agg = PortfolioAggregator()
    agg.add_stock_broker("alpaca", sb_a)
    agg.add_stock_broker("ibkr-paper", sb_b)
    agg.add_crypto_broker("binance", cb)

    snap = await agg.snapshot()

    # Each broker's equity attributed.
    assert snap.per_broker_equity["alpaca"] == pytest.approx(50_000.0)
    assert snap.per_broker_equity["ibkr-paper"] == pytest.approx(30_000.0)
    assert snap.per_broker_equity["binance"] == pytest.approx(20_000.0)
    # Stocks rolled together.
    assert snap.stocks_equity_usd == pytest.approx(80_000.0)
    # Crypto separate.
    assert snap.crypto_equity_usd == pytest.approx(20_000.0)
    # Total = stocks + crypto.
    assert snap.total_equity_usd == pytest.approx(100_000.0)


@pytest.mark.asyncio
async def test_aggregator_unifies_stock_positions_across_brokers():
    """Two stock brokers, each with a different position → unified
    position list. Useful when an operator wires both Alpaca paper +
    IBKR live during a staged migration."""
    sb_a = PaperStockBroker(slippage_bps=0)
    sb_b = PaperStockBroker(slippage_bps=0)
    sb_a.set_seed_price("AAPL", 150.0)
    sb_b.set_seed_price("MSFT", 410.0)
    await sb_a.place_order("AAPL", "buy", 10)
    await sb_b.place_order("MSFT", "buy", 5)

    agg = PortfolioAggregator()
    agg.add_stock_broker("a", sb_a)
    agg.add_stock_broker("b", sb_b)

    snap = await agg.snapshot()
    symbols = sorted(p.symbol for p in snap.stock_positions)
    assert symbols == ["AAPL", "MSFT"]


@pytest.mark.asyncio
async def test_aggregator_sums_balances_across_crypto_brokers():
    """Same asset on Binance + Coinbase → balances sum correctly.
    Pin the contract so a future operator with two crypto venues sees
    a unified BTC balance, not a fragmented per-broker view."""
    cb_a = PaperCryptoBroker(starting_usdt=10_000, slippage_bps=0)
    cb_b = PaperCryptoBroker(starting_usdt=10_000, slippage_bps=0)
    cb_a.set_seed_price("BTCUSDT", 50_000.0)
    cb_b.set_seed_price("BTCUSDT", 50_000.0)
    await cb_a.place_order("BTCUSDT", "buy", 0.1)  # 0.1 BTC on broker a
    await cb_b.place_order("BTCUSDT", "buy", 0.05)  # 0.05 BTC on broker b

    agg = PortfolioAggregator()
    agg.add_crypto_broker("a", cb_a)
    agg.add_crypto_broker("b", cb_b)

    snap = await agg.snapshot()
    # 0.1 + 0.05 = 0.15 BTC unified.
    assert snap.crypto_balances_by_asset["BTC"] == pytest.approx(0.15)


# ── Health + failure isolation ─────────────────────────────


@pytest.mark.asyncio
async def test_health_rows_include_every_broker():
    sb = PaperStockBroker()
    cb = PaperCryptoBroker()
    agg = PortfolioAggregator()
    agg.add_stock_broker("a", sb)
    agg.add_crypto_broker("b", cb)

    snap = await agg.snapshot()
    health_names = sorted(h.name for h in snap.per_broker_health)
    assert health_names == ["a", "b"]
    assert all(h.available for h in snap.per_broker_health)


@pytest.mark.asyncio
async def test_failed_stock_broker_isolated():
    """A stock broker raising must NOT drop the others — operator
    sees the failure flagged but the rest of the portfolio."""

    class _BrokenBroker:
        async def get_account_info(self):
            raise RuntimeError("alpaca down")

        async def get_all_positions(self):
            raise RuntimeError("alpaca down")

    healthy = PaperStockBroker(starting_cash=10_000)
    agg = PortfolioAggregator()
    agg.add_stock_broker("healthy", healthy)
    agg.add_stock_broker("broken", _BrokenBroker())

    snap = await agg.snapshot()
    # Healthy broker's equity is in the totals.
    assert snap.stocks_equity_usd == pytest.approx(10_000.0)
    assert "healthy" in snap.per_broker_equity
    assert "broken" not in snap.per_broker_equity
    # Health row for the failure is present.
    assert snap.has_failures is True
    failed = [h for h in snap.per_broker_health if not h.available]
    assert len(failed) == 1
    assert failed[0].name == "broken"
    assert "alpaca down" in failed[0].error
    # `healthy_broker_names` excludes the failure.
    assert snap.healthy_broker_names == ["healthy"]


@pytest.mark.asyncio
async def test_failed_crypto_broker_isolated():
    class _BrokenCrypto:
        async def get_account(self):
            raise RuntimeError("binance 503")

        async def get_balances(self):
            raise RuntimeError("binance 503")

    healthy = PaperCryptoBroker(starting_usdt=20_000)
    agg = PortfolioAggregator()
    agg.add_crypto_broker("healthy", healthy)
    agg.add_crypto_broker("broken", _BrokenCrypto())

    snap = await agg.snapshot()
    assert snap.crypto_equity_usd == pytest.approx(20_000.0)
    assert any(h.name == "broken" and not h.available for h in snap.per_broker_health)


@pytest.mark.asyncio
async def test_timeout_marks_broker_as_failed():
    """A slow broker (longer than `timeout_seconds`) is marked
    unavailable rather than blocking the whole snapshot."""

    class _SlowBroker:
        async def get_account_info(self):
            await asyncio.sleep(2.0)
            from halal_trader.domain.models import Account

            return Account(equity=999.0)

        async def get_all_positions(self):
            await asyncio.sleep(2.0)
            return []

    fast = PaperStockBroker(starting_cash=5_000)
    agg = PortfolioAggregator(timeout_seconds=0.05)
    agg.add_stock_broker("fast", fast)
    agg.add_stock_broker("slow", _SlowBroker())

    snap = await agg.snapshot()
    assert snap.stocks_equity_usd == pytest.approx(5_000.0)
    slow_health = next(h for h in snap.per_broker_health if h.name == "slow")
    assert slow_health.available is False


@pytest.mark.asyncio
async def test_all_brokers_failing_returns_zero_with_health_intact():
    """Apocalyptic case: every broker is down. Aggregator returns
    zero equity but the health rows tell the operator exactly what
    failed. Don't crash — the dashboard needs to render *something*."""

    class _Broken:
        async def get_account_info(self):
            raise RuntimeError("down")

        async def get_all_positions(self):
            raise RuntimeError("down")

    agg = PortfolioAggregator()
    agg.add_stock_broker("a", _Broken())
    agg.add_stock_broker("b", _Broken())

    snap = await agg.snapshot()
    assert snap.total_equity_usd == 0.0
    assert snap.has_failures
    assert len(snap.per_broker_health) == 2
    assert all(not h.available for h in snap.per_broker_health)


# ── add_*_broker contract ─────────────────────────────────


@pytest.mark.asyncio
async def test_add_stock_broker_overrides_same_name():
    """Re-adding under the same name swaps the broker — useful for
    tests + restarts that want to reset state without rebuilding the
    aggregator from scratch."""
    a = PaperStockBroker(starting_cash=1_000)
    b = PaperStockBroker(starting_cash=99_000)

    agg = PortfolioAggregator()
    agg.add_stock_broker("primary", a)
    agg.add_stock_broker("primary", b)  # override

    assert len(agg.stock_brokers) == 1
    snap = await agg.snapshot()
    assert snap.stocks_equity_usd == pytest.approx(99_000.0)


@pytest.mark.asyncio
async def test_per_broker_health_has_correct_asset_class_label():
    sb = PaperStockBroker()
    cb = PaperCryptoBroker()
    agg = PortfolioAggregator()
    agg.add_stock_broker("alpaca", sb)
    agg.add_crypto_broker("binance", cb)

    snap = await agg.snapshot()
    by_name = {h.name: h for h in snap.per_broker_health}
    assert by_name["alpaca"].asset_class == "stocks"
    assert by_name["binance"].asset_class == "crypto"


def test_broker_health_dataclass_is_frozen():
    """`BrokerHealth` is frozen — dashboard code passes it across
    threads, so accidental mutation would create flakiness."""
    h = BrokerHealth(name="x", asset_class="stocks", available=True)
    with pytest.raises(Exception):
        h.available = False  # type: ignore[misc]


def test_aggregated_portfolio_is_frozen():
    p = AggregatedPortfolio(
        total_equity_usd=100.0,
        stocks_equity_usd=80.0,
        crypto_equity_usd=20.0,
        stock_positions=[],
        crypto_balances_by_asset={},
        per_broker_equity={},
        per_broker_health=[],
    )
    with pytest.raises(Exception):
        p.total_equity_usd = 200.0  # type: ignore[misc]
