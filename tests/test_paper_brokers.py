"""Tests for the SDK-free paper brokers in `brokers/paper.py`.

`PaperStockBroker` + `PaperCryptoBroker` need to satisfy the same
Protocols (`Broker`, `CryptoBroker`) that the live adapters do —
they're a drop-in replacement for fresh-checkout / onboarding /
CI usage. Every method is covered.
"""

from __future__ import annotations

import pytest

from halal_trader.brokers import KNOWN_CRYPTO_BROKERS, KNOWN_STOCK_BROKERS
from halal_trader.brokers.paper import PaperCryptoBroker, PaperStockBroker
from halal_trader.domain.models import Account, CryptoAccount, Kline, MarketClock

# ── Registry hookup ────────────────────────────────────────


def test_paper_brokers_registered_under_default_name():
    """Both `paper` adapters must be wired into the registry —
    otherwise `BROKER=paper` / `CRYPTO_BROKER=paper` won't resolve."""
    assert "paper" in KNOWN_STOCK_BROKERS()
    assert "paper" in KNOWN_CRYPTO_BROKERS()


# ── PaperStockBroker ───────────────────────────────────────


@pytest.mark.asyncio
async def test_stock_broker_initial_account_is_starting_cash():
    b = PaperStockBroker(starting_cash=50_000.0)
    acct = await b.get_account_info()
    assert isinstance(acct, Account)
    assert acct.cash == 50_000.0
    assert acct.equity == 50_000.0
    assert acct.buying_power == 50_000.0


@pytest.mark.asyncio
async def test_stock_broker_clock_always_open():
    """Paper sessions intentionally bypass the market-hours skip —
    operators wanting closed-market behaviour use real adapters."""
    b = PaperStockBroker()
    clock = await b.get_clock()
    assert isinstance(clock, MarketClock)
    assert clock.is_open is True


@pytest.mark.asyncio
async def test_stock_broker_buy_deducts_cash_and_creates_position():
    b = PaperStockBroker(starting_cash=10_000.0, slippage_bps=0)
    b.set_seed_price("AAPL", 150.0)
    res = await b.place_order("AAPL", "buy", 10)
    assert res["status"] == "filled"
    assert res["symbol"] == "AAPL"

    acct = await b.get_account_info()
    assert acct.cash == 10_000.0 - 1500.0  # 10 * 150
    positions = await b.get_all_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10
    assert positions[0].avg_entry_price == 150.0


@pytest.mark.asyncio
async def test_stock_broker_sell_returns_cash():
    b = PaperStockBroker(starting_cash=10_000.0, slippage_bps=0)
    b.set_seed_price("AAPL", 150.0)
    await b.place_order("AAPL", "buy", 10)
    await b.place_order("AAPL", "sell", 5)

    acct = await b.get_account_info()
    # Bought 10 @150 = -1500, sold 5 @150 = +750.
    assert acct.cash == 10_000.0 - 1500.0 + 750.0
    positions = await b.get_all_positions()
    assert positions[0].qty == 5


@pytest.mark.asyncio
async def test_stock_broker_full_close_removes_position():
    """Selling the entire position drops it from the positions dict
    so `get_all_positions` doesn't return zero-qty rows."""
    b = PaperStockBroker(slippage_bps=0)
    b.set_seed_price("AAPL", 100.0)
    await b.place_order("AAPL", "buy", 5)
    await b.place_order("AAPL", "sell", 5)
    assert await b.get_all_positions() == []


@pytest.mark.asyncio
async def test_stock_broker_buy_rejected_when_insufficient_cash():
    b = PaperStockBroker(starting_cash=100.0, slippage_bps=0)
    b.set_seed_price("AAPL", 150.0)
    res = await b.place_order("AAPL", "buy", 1)
    assert res["status"] == "rejected"
    assert "insufficient cash" in res["reason"]


@pytest.mark.asyncio
async def test_stock_broker_sell_rejected_when_no_position():
    b = PaperStockBroker(slippage_bps=0)
    b.set_seed_price("AAPL", 150.0)
    res = await b.place_order("AAPL", "sell", 1)
    assert res["status"] == "rejected"
    assert "insufficient position" in res["reason"]


@pytest.mark.asyncio
async def test_stock_broker_rejected_with_no_seed_price():
    """Defensive: an order on a symbol the operator hasn't primed →
    rejected with a clear reason. Otherwise we'd compute a $0 fill."""
    b = PaperStockBroker()
    res = await b.place_order("UNKNOWN", "buy", 1)
    assert res["status"] == "rejected"
    assert "no seed price" in res["reason"]


@pytest.mark.asyncio
async def test_stock_broker_rejected_on_non_positive_qty():
    b = PaperStockBroker()
    b.set_seed_price("AAPL", 100.0)
    res = await b.place_order("AAPL", "buy", 0)
    assert res["status"] == "rejected"


@pytest.mark.asyncio
async def test_stock_broker_slippage_buys_above_seed_price():
    """`slippage_bps=10` (0.1%) → buys fill 0.1% above the seed."""
    b = PaperStockBroker(starting_cash=10_000, slippage_bps=10)
    b.set_seed_price("AAPL", 100.0)
    res = await b.place_order("AAPL", "buy", 1)
    fill_price = float(res["filled_avg_price"])
    assert fill_price == pytest.approx(100.0 * 1.001)


@pytest.mark.asyncio
async def test_stock_broker_slippage_sells_below_seed_price():
    b = PaperStockBroker(starting_cash=10_000, slippage_bps=10)
    b.set_seed_price("AAPL", 100.0)
    await b.place_order("AAPL", "buy", 1)
    res = await b.place_order("AAPL", "sell", 1)
    fill_price = float(res["filled_avg_price"])
    assert fill_price == pytest.approx(100.0 * 0.999)


@pytest.mark.asyncio
async def test_stock_broker_avg_cost_updates_on_subsequent_buys():
    """Two buys at different prices → weighted-average cost basis."""
    b = PaperStockBroker(starting_cash=100_000, slippage_bps=0)
    b.set_seed_price("AAPL", 100.0)
    await b.place_order("AAPL", "buy", 10)  # 10 @ 100 → avg 100
    b.set_seed_price("AAPL", 120.0)
    await b.place_order("AAPL", "buy", 10)  # 10 @ 120 → avg 110

    positions = await b.get_all_positions()
    assert positions[0].avg_entry_price == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_stock_broker_snapshot_returns_alpaca_shape():
    b = PaperStockBroker()
    b.set_seed_price("AAPL", 150.0)
    snap = await b.get_stock_snapshot("AAPL")
    assert "AAPL" in snap
    assert snap["AAPL"]["latest_trade"]["price"] == 150.0
    assert "latest_quote" in snap["AAPL"]
    assert "daily_bar" in snap["AAPL"]


@pytest.mark.asyncio
async def test_stock_broker_snapshot_handles_csv_symbols():
    """Alpaca-style comma-separated symbols string → multi-symbol
    snapshot."""
    b = PaperStockBroker()
    b.set_seed_price("AAPL", 150.0)
    b.set_seed_price("MSFT", 410.0)
    snap = await b.get_stock_snapshot("AAPL,MSFT")
    assert "AAPL" in snap and "MSFT" in snap


@pytest.mark.asyncio
async def test_stock_broker_bars_returns_synthetic_history():
    b = PaperStockBroker()
    b.set_seed_price("AAPL", 150.0)
    bars = await b.get_stock_bars("AAPL", days=5, timeframe="1Day")
    assert len(bars) == 5
    assert bars[0]["c"] == 150.0


@pytest.mark.asyncio
async def test_stock_broker_bars_empty_when_no_seed_price():
    b = PaperStockBroker()
    bars = await b.get_stock_bars("UNKNOWN", days=5, timeframe="1Day")
    assert bars == []


@pytest.mark.asyncio
async def test_stock_broker_close_position_sells_all():
    b = PaperStockBroker(slippage_bps=0)
    b.set_seed_price("AAPL", 100.0)
    await b.place_order("AAPL", "buy", 5)
    res = await b.close_position("AAPL")
    assert res["status"] == "filled"
    assert await b.get_all_positions() == []


@pytest.mark.asyncio
async def test_stock_broker_close_position_skips_when_empty():
    b = PaperStockBroker()
    res = await b.close_position("UNHELD")
    assert res["status"] == "skipped"


@pytest.mark.asyncio
async def test_stock_broker_close_all_handles_multiple_positions():
    b = PaperStockBroker(slippage_bps=0)
    b.set_seed_price("AAPL", 100.0)
    b.set_seed_price("MSFT", 200.0)
    await b.place_order("AAPL", "buy", 5)
    await b.place_order("MSFT", "buy", 3)
    results = await b.close_all_positions()
    assert len(results) == 2
    assert all(r["status"] == "filled" for r in results)
    assert await b.get_all_positions() == []


@pytest.mark.asyncio
async def test_stock_broker_reset_preserves_seed_prices():
    """Calling reset() should preserve the primed prices —
    onboarding flow doesn't want to re-seed every cycle."""
    b = PaperStockBroker(starting_cash=10_000, slippage_bps=0)
    b.set_seed_price("AAPL", 150.0)
    await b.place_order("AAPL", "buy", 1)

    b.reset(starting_cash=20_000)
    acct = await b.get_account_info()
    assert acct.cash == 20_000
    # Position cleared, but seed price retained:
    assert await b.get_all_positions() == []
    res = await b.place_order("AAPL", "buy", 1)
    assert res["status"] == "filled"


# ── PaperCryptoBroker ──────────────────────────────────────


@pytest.mark.asyncio
async def test_crypto_broker_initial_account():
    b = PaperCryptoBroker(starting_usdt=50_000)
    acct = await b.get_account()
    assert isinstance(acct, CryptoAccount)
    assert acct.total_balance_usdt == 50_000
    assert acct.available_balance_usdt == 50_000
    assert acct.usdt_free == 50_000


@pytest.mark.asyncio
async def test_crypto_broker_balances_include_usdt_when_no_positions():
    b = PaperCryptoBroker(starting_usdt=10_000)
    balances = await b.get_balances()
    assert len(balances) == 1
    assert balances[0].asset == "USDT"
    assert balances[0].free == 10_000


@pytest.mark.asyncio
async def test_crypto_broker_buy_deducts_usdt_and_creates_position():
    b = PaperCryptoBroker(starting_usdt=10_000, slippage_bps=0)
    b.set_seed_price("BTCUSDT", 50_000.0)
    res = await b.place_order("BTCUSDT", "buy", 0.1)
    assert res["status"] == "FILLED"
    assert res["symbol"] == "BTCUSDT"

    acct = await b.get_account()
    assert acct.available_balance_usdt == 10_000 - 5000  # 0.1 * 50000

    balances = await b.get_balances()
    btc_row = next(b for b in balances if b.asset == "BTC")
    assert btc_row.free == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_crypto_broker_busd_pair_strips_correctly():
    """BUSD-quoted pair → asset name strips BUSD suffix."""
    b = PaperCryptoBroker(starting_usdt=10_000, slippage_bps=0)
    b.set_seed_price("ETHBUSD", 3_000.0)
    await b.place_order("ETHBUSD", "buy", 1)
    balances = await b.get_balances()
    asset_names = [b.asset for b in balances]
    assert "ETH" in asset_names


@pytest.mark.asyncio
async def test_crypto_broker_klines_returns_synthetic_history():
    """`get_klines(limit=N)` returns N synthetic bars."""
    b = PaperCryptoBroker()
    b.set_seed_price("BTCUSDT", 50_000.0)
    klines = await b.get_klines("BTCUSDT", interval="1m", limit=30)
    assert len(klines) == 30
    assert all(isinstance(k, Kline) for k in klines)
    assert all(k.close == 50_000.0 for k in klines)


@pytest.mark.asyncio
async def test_crypto_broker_klines_empty_when_no_seed():
    b = PaperCryptoBroker()
    klines = await b.get_klines("UNKNOWN", limit=10)
    assert klines == []


@pytest.mark.asyncio
async def test_crypto_broker_orderbook_shape():
    b = PaperCryptoBroker()
    b.set_seed_price("BTCUSDT", 50_000.0)
    book = await b.get_order_book("BTCUSDT")
    assert "bids" in book and "asks" in book
    assert book["bids"][0][0] < book["asks"][0][0]  # bid below ask


@pytest.mark.asyncio
async def test_crypto_broker_ticker_price():
    b = PaperCryptoBroker()
    b.set_seed_price("BTCUSDT", 50_000.0)
    assert await b.get_ticker_price("BTCUSDT") == 50_000.0
    assert await b.get_ticker_price("UNKNOWN") == 0.0


@pytest.mark.asyncio
async def test_crypto_broker_buy_rejected_when_insufficient_usdt():
    b = PaperCryptoBroker(starting_usdt=100, slippage_bps=0)
    b.set_seed_price("BTCUSDT", 50_000.0)
    res = await b.place_order("BTCUSDT", "buy", 1)
    assert res["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_crypto_broker_sell_rejected_when_no_position():
    b = PaperCryptoBroker()
    b.set_seed_price("BTCUSDT", 50_000.0)
    res = await b.place_order("BTCUSDT", "sell", 0.1)
    assert res["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_crypto_broker_open_orders_empty():
    """Synchronous fills → no open orders."""
    b = PaperCryptoBroker()
    assert await b.get_open_orders() == []


@pytest.mark.asyncio
async def test_crypto_broker_cancel_order_rejected():
    """No open orders to cancel — safe noop."""
    b = PaperCryptoBroker()
    res = await b.cancel_order("BTCUSDT", "fake-order-id")
    assert res["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_crypto_broker_balances_handles_dust_position():
    """A position closed down to 0 should not appear in balances."""
    b = PaperCryptoBroker(starting_usdt=10_000, slippage_bps=0)
    b.set_seed_price("BTCUSDT", 50_000.0)
    await b.place_order("BTCUSDT", "buy", 0.1)
    await b.place_order("BTCUSDT", "sell", 0.1)
    balances = await b.get_balances()
    asset_names = [b.asset for b in balances]
    assert "BTC" not in asset_names


@pytest.mark.asyncio
async def test_crypto_broker_avg_cost_weighted_across_buys():
    b = PaperCryptoBroker(starting_usdt=100_000, slippage_bps=0)
    b.set_seed_price("BTCUSDT", 50_000.0)
    await b.place_order("BTCUSDT", "buy", 0.1)  # 50_000 → 5000 spent
    b.set_seed_price("BTCUSDT", 60_000.0)
    await b.place_order("BTCUSDT", "buy", 0.1)  # 60_000 → 6000 spent
    # Avg cost = (5000 + 6000) / 0.2 = 55_000
    assert b._state.avg_costs["BTCUSDT"] == pytest.approx(55_000.0)


# ── Cross-protocol smoke ───────────────────────────────────


@pytest.mark.asyncio
async def test_stock_paper_broker_satisfies_protocol_smoke():
    """End-to-end: fetch account, snapshot, bars, place order, get
    positions — the full bot-cycle surface works without any external
    dependency."""
    b = PaperStockBroker(starting_cash=10_000, slippage_bps=0)
    b.set_seed_price("AAPL", 150.0)

    assert (await b.get_account_info()).cash == 10_000
    snap = await b.get_stock_snapshot("AAPL")
    bars = await b.get_stock_bars("AAPL", days=3)
    res = await b.place_order("AAPL", "buy", 5)
    positions = await b.get_all_positions()

    assert snap["AAPL"]["latest_trade"]["price"] == 150.0
    assert len(bars) == 3
    assert res["status"] == "filled"
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_crypto_paper_broker_satisfies_protocol_smoke():
    b = PaperCryptoBroker(starting_usdt=100_000, slippage_bps=0)
    b.set_seed_price("BTCUSDT", 50_000.0)

    assert (await b.get_account()).total_balance_usdt == 100_000
    klines = await b.get_klines("BTCUSDT", limit=5)
    book = await b.get_order_book("BTCUSDT")
    res = await b.place_order("BTCUSDT", "buy", 0.1)
    balances = await b.get_balances()

    assert len(klines) == 5
    assert book["bids"]
    assert res["status"] == "FILLED"
    assert any(b.asset == "BTC" for b in balances)
