"""Tests for CryptoExecutor — order validation, circuit breaker, execution flow."""

from unittest.mock import AsyncMock, MagicMock

from halal_trader.crypto.exchange import SymbolFilter
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.domain.models import (
    CryptoAccount,
    CryptoBalance,
    CryptoTradeDecision,
    CryptoTradingPlan,
    TradeAction,
)


def _make_executor(
    *,
    cached_price=50000.0,
    ticker_price=50000.0,
    symbol_filter=None,
    balances=None,
    order_result=None,
    max_position_pct=0.25,
    max_simultaneous_positions=4,
    configured_pairs=None,
):
    broker = AsyncMock()
    broker.get_ticker_price.return_value = ticker_price
    broker.round_quantity = MagicMock(side_effect=lambda sym, qty: round(qty, 6))
    broker.get_cached_price = MagicMock(return_value=cached_price)
    broker.get_symbol_filter = MagicMock(return_value=symbol_filter)
    broker.get_balances.return_value = balances or [
        CryptoBalance(asset="USDT", free=5000.0),
    ]
    broker.place_order.return_value = order_result or {
        "orderId": "12345",
        "status": "FILLED",
        "fills": [{"price": str(ticker_price), "qty": "0.001"}],
    }

    repo = AsyncMock()
    repo.record_crypto_trade.return_value = 1
    repo.get_open_crypto_trades_for_pair.return_value = []

    executor = CryptoExecutor(
        broker,
        repo,
        max_position_pct=max_position_pct,
        max_simultaneous_positions=max_simultaneous_positions,
        configured_pairs=configured_pairs or ["BTCUSDT", "ETHUSDT"],
    )
    return executor, broker, repo


def _make_plan(decisions):
    return CryptoTradingPlan(
        decisions=decisions,
        market_outlook="Test",
        risk_notes="",
    )


def _make_buy(
    symbol="BTCUSDT", quantity=0.001, confidence=0.8,
    entry_price=50000, sl=49500, tp=51000,
):
    return CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=quantity,
        confidence=confidence,
        reasoning="Test buy",
        entry_price=entry_price,
        target_price=tp,
        stop_loss=sl,
    )


def _make_sell(symbol="BTCUSDT", quantity=0.001):
    return CryptoTradeDecision(
        action=TradeAction.SELL,
        symbol=symbol,
        quantity=quantity,
        confidence=0.7,
        reasoning="Test sell",
    )


def _make_account(balance=10000.0):
    return CryptoAccount(
        total_balance_usdt=balance,
        available_balance_usdt=balance * 0.8,
        in_order_usdt=balance * 0.2,
        usdt_free=balance * 0.8,
    )


class TestExecuteBuy:
    async def test_successful_buy(self):
        executor, broker, repo = _make_executor()
        plan = _make_plan([_make_buy()])
        results = await executor.execute_plan(plan, account=_make_account())

        assert len(results) == 1
        assert results[0]["action"] == "buy"
        assert results[0]["status"] == "filled"
        broker.place_order.assert_awaited_once()
        repo.record_crypto_trade.assert_awaited_once()

    async def test_buy_rejected_insufficient_usdt(self):
        executor, broker, repo = _make_executor(ticker_price=50000.0)
        account = CryptoAccount(
            total_balance_usdt=100.0,
            available_balance_usdt=1.0,
            in_order_usdt=0.0,
            usdt_free=1.0,
        )
        plan = _make_plan([_make_buy(quantity=0.01)])
        results = await executor.execute_plan(plan, account=account)

        assert results[0]["status"] == "rejected"
        assert "Insufficient" in results[0]["reason"]
        broker.place_order.assert_not_awaited()

    async def test_buy_rejected_exceeds_position_pct(self):
        executor, broker, repo = _make_executor(
            max_position_pct=0.10,
            ticker_price=50000.0,
        )
        account = _make_account(balance=10000.0)
        plan = _make_plan([_make_buy(quantity=0.05)])
        results = await executor.execute_plan(plan, account=account)

        assert results[0]["status"] == "rejected"
        assert "limit" in results[0]["reason"].lower()


class TestExecuteSell:
    async def test_successful_sell(self):
        executor, broker, repo = _make_executor(
            balances=[CryptoBalance(asset="BTC", free=0.01)]
        )
        plan = _make_plan([_make_sell()])
        results = await executor.execute_plan(plan, account=_make_account())

        assert len(results) == 1
        assert results[0]["action"] == "sell"
        broker.place_order.assert_awaited_once()

    async def test_sell_clamps_to_actual_balance(self):
        executor, broker, repo = _make_executor(
            balances=[CryptoBalance(asset="BTC", free=0.0005)]
        )
        plan = _make_plan([_make_sell(quantity=1.0)])
        await executor.execute_plan(plan, account=_make_account())

        call_args = broker.place_order.call_args
        assert call_args.kwargs["quantity"] <= 0.0005

    async def test_sell_rejected_no_balance(self):
        executor, broker, repo = _make_executor(
            balances=[CryptoBalance(asset="USDT", free=5000.0)]
        )
        plan = _make_plan([_make_sell()])
        results = await executor.execute_plan(plan, account=_make_account())

        assert results[0]["status"] == "rejected"
        assert "No" in results[0]["reason"]


class TestValidateOrder:
    def test_validates_min_notional(self):
        sf = SymbolFilter(
            min_qty=0.00001,
            max_qty=1000.0,
            step_size=0.00001,
            min_notional=10.0,
            tick_size=0.01,
            base_asset_precision=8,
            quote_asset_precision=8,
        )
        executor, broker, _ = _make_executor(symbol_filter=sf)
        err = executor._validate_order("BTCUSDT", "BUY", 0.001, 50000.0)
        assert err is None

    def test_rejects_below_min_notional(self):
        sf = SymbolFilter(
            min_qty=0.00001,
            max_qty=1000.0,
            step_size=0.00001,
            min_notional=10.0,
            tick_size=0.01,
            base_asset_precision=8,
            quote_asset_precision=8,
        )
        executor, broker, _ = _make_executor(symbol_filter=sf)
        err = executor._validate_order("BTCUSDT", "BUY", 0.00001, 100.0)
        assert err is not None
        assert "minimum notional" in err.lower()

    def test_rejects_below_min_qty(self):
        sf = SymbolFilter(
            min_qty=0.001,
            max_qty=1000.0,
            step_size=0.001,
            min_notional=1.0,
            tick_size=0.01,
            base_asset_precision=8,
            quote_asset_precision=8,
        )
        executor, broker, _ = _make_executor(symbol_filter=sf)
        err = executor._validate_order("BTCUSDT", "BUY", 0.0001, 50000.0)
        assert err is not None
        assert "minimum" in err.lower()

    def test_fallback_without_symbol_filter(self):
        executor, _, _ = _make_executor(symbol_filter=None)
        err = executor._validate_order("BTCUSDT", "BUY", 0.001, 50000.0)
        assert err is None

    def test_fallback_rejects_tiny_order(self):
        executor, _, _ = _make_executor(symbol_filter=None)
        err = executor._validate_order("BTCUSDT", "BUY", 0.00001, 10.0)
        assert err is not None


class TestCircuitBreaker:
    def test_pair_not_blocked_initially(self):
        executor, _, _ = _make_executor()
        assert executor.is_pair_blocked("BTCUSDT") is False

    def test_pair_blocked_after_threshold_errors(self):
        executor, _, _ = _make_executor()
        for _ in range(5):
            executor._record_pair_error("BTCUSDT")
        assert executor.is_pair_blocked("BTCUSDT") is True

    def test_pair_unblocked_after_cooldown(self):
        executor, _, _ = _make_executor()
        executor._cb_cooldown = 0
        for _ in range(5):
            executor._record_pair_error("BTCUSDT")
        assert executor.is_pair_blocked("BTCUSDT") is False

    async def test_blocked_pair_buy_skipped(self):
        executor, broker, _ = _make_executor()
        for _ in range(5):
            executor._record_pair_error("BTCUSDT")

        plan = _make_plan([_make_buy()])
        results = await executor.execute_plan(plan, account=_make_account())
        assert results[0]["status"] == "rejected"
        assert "circuit breaker" in results[0]["reason"]


class TestSellsFirstThenBuys:
    async def test_sells_execute_before_buys(self):
        call_order = []

        async def track_sell(*args, **kwargs):
            call_order.append("sell")
            return {
                "orderId": "s1", "status": "FILLED",
                "fills": [{"price": "50000", "qty": "0.001"}],
            }

        async def track_buy(*args, **kwargs):
            call_order.append("buy")
            return {
                "orderId": "b1", "status": "FILLED",
                "fills": [{"price": "50000", "qty": "0.001"}],
            }

        executor, broker, repo = _make_executor(
            balances=[
                CryptoBalance(asset="BTC", free=0.01),
                CryptoBalance(asset="USDT", free=5000.0),
            ]
        )

        async def dispatch_order(**kwargs):
            if kwargs.get("side") == "SELL":
                return await track_sell()
            return await track_buy()

        broker.place_order.side_effect = dispatch_order

        plan = _make_plan([_make_buy(), _make_sell()])
        await executor.execute_plan(plan, account=_make_account())

        assert call_order[0] == "sell"


class TestMaxPositions:
    async def test_rejects_buy_at_max_positions(self):
        executor, broker, repo = _make_executor(
            max_simultaneous_positions=1,
            balances=[
                CryptoBalance(asset="BTC", free=0.01),
                CryptoBalance(asset="USDT", free=5000.0),
            ],
        )
        plan = _make_plan([_make_buy(symbol="ETHUSDT")])
        results = await executor.execute_plan(plan, account=_make_account())

        assert results[0]["status"] == "rejected"
        assert "Max simultaneous" in results[0]["reason"]


class TestOrderStatusMapping:
    def test_filled(self):
        assert CryptoExecutor._map_order_status({"status": "FILLED"}) == "filled"

    def test_canceled(self):
        assert CryptoExecutor._map_order_status({"status": "CANCELED"}) == "rejected"

    def test_rejected(self):
        assert CryptoExecutor._map_order_status({"status": "REJECTED"}) == "rejected"

    def test_partial(self):
        assert CryptoExecutor._map_order_status({"status": "PARTIALLY_FILLED"}) == "partial"

    def test_other(self):
        assert CryptoExecutor._map_order_status({"status": "NEW"}) == "submitted"
