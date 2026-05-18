"""Tests for CryptoCycleService — orchestration, halt conditions, flat-market skip."""

from unittest.mock import AsyncMock, MagicMock, patch

from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.domain.models import (
    CryptoAccount,
    CryptoBalance,
    CryptoTradingPlan,
    Kline,
)


def _make_klines(close=50000.0, n=30):
    return [
        Kline(
            open_time=1700000000000 + i * 60000,
            open=close - 10 + i,
            high=close + 50,
            low=close - 50,
            close=close + i * 0.1,
            volume=1000.0 + i,
            close_time=1700000000000 + (i + 1) * 60000,
        )
        for i in range(n)
    ]


def _mock_settings(**overrides):
    """Build a MagicMock settings tree mirroring the nested config layout."""
    defaults = {
        "flat_price_threshold": 0.03,
        "flat_rsi_lower": 40.0,
        "flat_rsi_upper": 60.0,
        "flat_vol_threshold": 1.2,
        "max_consecutive_flat_skips": 5,
        "max_pairs_per_cycle": 10,
    }
    defaults.update(overrides)
    settings = MagicMock()
    settings.crypto = MagicMock()
    for k, v in defaults.items():
        setattr(settings.crypto, k, v)
    return settings


def _make_cycle_service(
    *,
    halal_pairs=None,
    klines_by_symbol=None,
    should_halt=False,
    account_balance=10000.0,
    plan=None,
    settings_overrides=None,
):
    broker = AsyncMock()
    broker.get_account.return_value = CryptoAccount(
        total_balance_usdt=account_balance,
        available_balance_usdt=account_balance * 0.8,
        in_order_usdt=account_balance * 0.2,
        usdt_free=account_balance * 0.8,
    )
    broker.get_balances.return_value = [CryptoBalance(asset="USDT", free=account_balance * 0.8)]
    broker.get_order_book.return_value = {"bids": [], "asks": []}
    broker.get_klines.return_value = _make_klines()
    broker.get_cached_price.return_value = 50000.0
    broker.format_filters_for_prompt.return_value = ""
    # Microstructure helpers — return None so the cycle skips the
    # basis/funding extension cleanly without leaving unawaited coroutines.
    broker.get_funding_signal.return_value = None

    screener = AsyncMock()
    screener.get_halal_pairs.return_value = halal_pairs or ["BTC", "ETH"]

    strategy = AsyncMock()
    strategy.analyze.return_value = plan or CryptoTradingPlan(market_outlook="Test", risk_notes="")

    executor = AsyncMock()
    executor.execute_plan.return_value = []

    portfolio = AsyncMock()
    portfolio.should_halt_trading.return_value = should_halt
    portfolio.get_open_trades.return_value = []
    portfolio.format_positions_for_prompt.return_value = "No positions"
    portfolio.get_current_pnl.return_value = 0.0

    ws = MagicMock()
    ws.get_klines.return_value = _make_klines()
    ws.get_latest_price.return_value = 50000.0

    settings = _mock_settings(**(settings_overrides or {}))

    service = CryptoCycleService(
        broker=broker,
        screener=screener,
        strategy=strategy,
        executor=executor,
        portfolio=portfolio,
        ws_manager=ws,
        configured_pairs=["BTCUSDT", "ETHUSDT"],
    )
    service._settings = settings
    return service, broker, screener, strategy, executor, portfolio


class TestRunCycle:
    @patch("halal_trader.crypto.cycle.get_settings")
    async def test_successful_cycle(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, broker, screener, strategy, executor, portfolio = _make_cycle_service()
        await svc.run_cycle()
        strategy.analyze.assert_awaited_once()

    @patch("halal_trader.crypto.cycle.get_settings")
    async def test_halts_on_loss_limit(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, _, _, strategy, _, _ = _make_cycle_service(should_halt=True)
        await svc.run_cycle()
        strategy.analyze.assert_not_awaited()

    @patch("halal_trader.crypto.cycle.get_settings")
    async def test_skips_when_no_halal_pairs(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, _, _, strategy, _, _ = _make_cycle_service(halal_pairs=[])
        svc._configured_pairs = []
        await svc.run_cycle()
        strategy.analyze.assert_not_awaited()

    @patch("halal_trader.crypto.cycle.get_settings")
    async def test_executes_plan_with_decisions(self, mock_get_settings):
        from halal_trader.domain.models import CryptoTradeDecision, TradeAction

        mock_get_settings.return_value = _mock_settings()
        plan = CryptoTradingPlan(
            decisions=[
                CryptoTradeDecision(
                    action=TradeAction.BUY,
                    symbol="BTCUSDT",
                    quantity=0.001,
                    confidence=0.8,
                    reasoning="test",
                )
            ],
            market_outlook="Bullish",
        )
        svc, _, _, strategy, executor, _ = _make_cycle_service(plan=plan)
        await svc.run_cycle()
        executor.execute_plan.assert_awaited_once()

    @patch("halal_trader.crypto.cycle.get_settings")
    async def test_records_indicator_snapshot_and_notifies_when_no_shadow_runner(
        self, mock_get_settings
    ):
        """The default config has shadow_runner=None. After a buy fills,
        the cycle must still record the indicator snapshot (ML loop) and
        call the notifier (operator alerts) — those are independent of
        the shadow ledger."""
        from halal_trader.domain.models import CryptoTradeDecision, TradeAction

        mock_get_settings.return_value = _mock_settings()
        plan = CryptoTradingPlan(
            decisions=[
                CryptoTradeDecision(
                    action=TradeAction.BUY,
                    symbol="BTCUSDT",
                    quantity=0.001,
                    confidence=0.8,
                    reasoning="test",
                )
            ],
            market_outlook="Bullish",
        )
        svc, _, _, _, executor, portfolio = _make_cycle_service(plan=plan)
        # Make the executor return a single filled-buy result so the
        # snapshot + notify code paths fire.
        executor.execute_plan.return_value = [
            {
                "status": "filled",
                "action": "buy",
                "symbol": "BTCUSDT",
                "trade_id": 42,
                "quantity": 0.001,
                "price": 50_000.0,
            }
        ]
        notifier = AsyncMock()
        svc._notifier = notifier
        svc._shadow_runner = None  # default — verifying the no-shadow path
        await svc.run_cycle()
        portfolio.record_indicator_snapshot.assert_awaited_once()
        notifier.notify_trade.assert_awaited_once_with(
            pair="BTCUSDT",
            side="buy",
            quantity=0.001,
            price=50_000.0,
            market="crypto",
            order_id="",
        )


class TestShouldSkipLlm:
    @patch("halal_trader.crypto.cycle.get_settings")
    def test_skips_when_all_flat(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, *_ = _make_cycle_service()
        indicators = {
            "BTCUSDT": {"price_change_5m": 0.01, "rsi_14": 50, "volume_ratio": 1.0},
            "ETHUSDT": {"price_change_5m": 0.005, "rsi_14": 50, "volume_ratio": 0.9},
        }
        assert svc._should_skip_llm(indicators) is True

    @patch("halal_trader.crypto.cycle.get_settings")
    def test_does_not_skip_with_rsi_signal(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, *_ = _make_cycle_service()
        indicators = {
            "BTCUSDT": {"price_change_5m": 0.01, "rsi_14": 30, "volume_ratio": 1.0},
        }
        assert svc._should_skip_llm(indicators) is False

    @patch("halal_trader.crypto.cycle.get_settings")
    def test_does_not_skip_with_price_movement(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, *_ = _make_cycle_service()
        indicators = {
            "BTCUSDT": {"price_change_5m": 0.1, "rsi_14": 50, "volume_ratio": 1.0},
        }
        assert svc._should_skip_llm(indicators) is False

    @patch("halal_trader.crypto.cycle.get_settings")
    def test_does_not_skip_with_volume_spike(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, *_ = _make_cycle_service()
        indicators = {
            "BTCUSDT": {"price_change_5m": 0.01, "rsi_14": 50, "volume_ratio": 2.0},
        }
        assert svc._should_skip_llm(indicators) is False

    @patch("halal_trader.crypto.cycle.get_settings")
    def test_skips_empty_indicators(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, *_ = _make_cycle_service()
        assert svc._should_skip_llm({}) is True

    @patch("halal_trader.crypto.cycle.get_settings")
    def test_ignores_error_indicators(self, mock_get_settings):
        mock_get_settings.return_value = _mock_settings()
        svc, *_ = _make_cycle_service()
        indicators = {
            "BTCUSDT": {"error": "No data"},
        }
        assert svc._should_skip_llm(indicators) is True


class TestGetTradeablePairs:
    """Happy-path tests for the tradeable-pairs Wave B stage. Extensive
    branch coverage lives in :mod:`tests.test_crypto_cycle_tradeable`;
    these two pin the cycle-service ↔ stage integration."""

    @patch("halal_trader.crypto.cycle.get_settings")
    async def test_returns_intersection_of_configured_and_halal(self, mock_get_settings):
        from halal_trader.core.cycle_pipeline import CycleState
        from halal_trader.core.cycle_stages import GetTradeablePairsStage

        mock_get_settings.return_value = _mock_settings()
        svc, *_ = _make_cycle_service(halal_pairs=["BTC", "SOL"])
        svc._configured_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        state = CycleState()
        await GetTradeablePairsStage(
            screener=svc._screener,
            portfolio=svc._portfolio,
            configured_pairs=svc._configured_pairs,
            max_pairs=svc._settings.crypto.max_pairs_per_cycle,
        ).run(state)
        assert "BTCUSDT" in state.halal_pairs
        assert "SOLUSDT" in state.halal_pairs
        assert "ETHUSDT" not in state.halal_pairs

    @patch("halal_trader.crypto.cycle.get_settings")
    async def test_falls_back_to_configured_when_no_halal(self, mock_get_settings):
        from halal_trader.core.cycle_pipeline import CycleState
        from halal_trader.core.cycle_stages import GetTradeablePairsStage

        mock_get_settings.return_value = _mock_settings()
        svc, _, screener, *_ = _make_cycle_service()
        screener.get_halal_pairs.return_value = []
        state = CycleState()
        await GetTradeablePairsStage(
            screener=svc._screener,
            portfolio=svc._portfolio,
            configured_pairs=svc._configured_pairs,
            max_pairs=svc._settings.crypto.max_pairs_per_cycle,
        ).run(state)
        assert state.halal_pairs == ["BTCUSDT", "ETHUSDT"]
