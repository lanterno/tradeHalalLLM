"""Tests for CryptoTradingStrategy — LLM prompt construction and circuit breaker."""

from unittest.mock import AsyncMock

from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.domain.models import CryptoAccount, CryptoTradingPlan, Kline


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


def _make_account(balance=10000.0):
    return CryptoAccount(
        total_balance_usdt=balance,
        available_balance_usdt=balance * 0.8,
        in_order_usdt=balance * 0.2,
        usdt_free=balance * 0.8,
    )


def _make_strategy(llm_response=None, llm_error=None, **kwargs):
    llm = AsyncMock()
    if llm_error:
        llm.generate_json.side_effect = llm_error
    else:
        llm.generate_json.return_value = llm_response or {
            "decisions": [],
            "market_outlook": "Test outlook",
            "risk_notes": "Test notes",
        }
    llm.model = "test-model"

    repo = AsyncMock()
    repo.record_decision.return_value = 1

    strategy = CryptoTradingStrategy(
        llm,
        repo,
        llm_provider_name="test",
        max_position_pct=kwargs.get("max_position_pct", 0.25),
        daily_loss_limit=0.03,
        daily_return_target=0.01,
        max_simultaneous_positions=kwargs.get("max_simultaneous_positions", 4),
        llm_failure_threshold=kwargs.get("llm_failure_threshold", 3),
        llm_cooldown_seconds=kwargs.get("llm_cooldown_seconds", 60),
    )
    return strategy, llm, repo


class TestAnalyze:
    async def test_returns_empty_plan_on_success(self):
        strategy, llm, repo = _make_strategy()
        plan = await strategy.analyze(
            account=_make_account(),
            positions_text="No positions",
            halal_pairs=["BTCUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines()},
            orderbooks={},
        )
        assert isinstance(plan, CryptoTradingPlan)
        assert plan.market_outlook == "Test outlook"
        llm.generate_json.assert_awaited_once()
        repo.record_decision.assert_awaited_once()

    async def test_returns_plan_with_decisions(self):
        strategy, llm, repo = _make_strategy(llm_response={
            "decisions": [
                {
                    "action": "buy",
                    "symbol": "BTCUSDT",
                    "quantity": 0.001,
                    "confidence": 0.8,
                    "reasoning": "RSI oversold",
                    "entry_price": 50000,
                    "target_price": 51000,
                    "stop_loss": 49500,
                }
            ],
            "market_outlook": "Bullish",
            "risk_notes": "Low volume",
        })
        plan = await strategy.analyze(
            account=_make_account(),
            positions_text="No positions",
            halal_pairs=["BTCUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines()},
            orderbooks={},
        )
        assert len(plan.buys) == 1
        assert plan.buys[0].symbol == "BTCUSDT"

    async def test_handles_llm_failure_gracefully(self):
        strategy, llm, repo = _make_strategy(llm_error=RuntimeError("Connection timeout"))
        plan = await strategy.analyze(
            account=_make_account(),
            positions_text="",
            halal_pairs=["BTCUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines()},
            orderbooks={},
        )
        assert isinstance(plan, CryptoTradingPlan)
        assert "failed" in plan.market_outlook.lower()
        assert len(plan.decisions) == 0

    async def test_zero_portfolio_value_fallback(self):
        strategy, _, _ = _make_strategy()
        plan = await strategy.analyze(
            account=CryptoAccount(total_balance_usdt=0),
            positions_text="",
            halal_pairs=["BTCUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines()},
            orderbooks={},
        )
        assert isinstance(plan, CryptoTradingPlan)


class TestCircuitBreaker:
    async def test_cooldown_after_threshold_failures(self):
        strategy, llm, repo = _make_strategy(
            llm_error=RuntimeError("fail"),
            llm_failure_threshold=2,
            llm_cooldown_seconds=60,
        )

        for _ in range(2):
            await strategy.analyze(
                account=_make_account(),
                positions_text="",
                halal_pairs=["BTCUSDT"],
                klines_by_symbol={"BTCUSDT": _make_klines()},
                orderbooks={},
            )

        plan = await strategy.analyze(
            account=_make_account(),
            positions_text="",
            halal_pairs=["BTCUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines()},
            orderbooks={},
        )
        assert "cooldown" in plan.market_outlook.lower()
        assert llm.generate_json.await_count == 2

    async def test_success_resets_failure_counter(self):
        strategy, llm, repo = _make_strategy(llm_failure_threshold=3)

        llm.generate_json.side_effect = RuntimeError("fail")
        await strategy.analyze(
            account=_make_account(),
            positions_text="",
            halal_pairs=["BTCUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines()},
            orderbooks={},
        )
        assert strategy._consecutive_llm_failures == 1

        llm.generate_json.side_effect = None
        llm.generate_json.return_value = {
            "decisions": [],
            "market_outlook": "ok",
            "risk_notes": "",
        }
        await strategy.analyze(
            account=_make_account(),
            positions_text="",
            halal_pairs=["BTCUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines()},
            orderbooks={},
        )
        assert strategy._consecutive_llm_failures == 0


class TestSellOnlyMode:
    async def test_sell_only_prompt_at_max_positions(self):
        strategy, llm, repo = _make_strategy(max_simultaneous_positions=2)
        await strategy.analyze(
            account=_make_account(),
            positions_text="BTC: 0.1",
            halal_pairs=["BTCUSDT", "ETHUSDT"],
            klines_by_symbol={"BTCUSDT": _make_klines(), "ETHUSDT": _make_klines()},
            orderbooks={},
            open_position_count=2,
        )
        call_args = llm.generate_json.call_args
        system_prompt = call_args.kwargs.get("system") or call_args.args[1]
        assert "SELL-ONLY MODE" in system_prompt


class TestBuildOrderbookText:
    def test_empty_orderbooks(self):
        strategy, _, _ = _make_strategy()
        text = strategy._build_orderbook_text({})
        assert "No order book" in text

    def test_formats_orderbook_with_imbalance(self):
        strategy, _, _ = _make_strategy()
        orderbooks = {
            "BTCUSDT": {
                "bids": [[50000.0, 2.0], [49990.0, 1.0]],
                "asks": [[50010.0, 0.5], [50020.0, 0.3]],
            }
        }
        text = strategy._build_orderbook_text(orderbooks)
        assert "BTCUSDT" in text
        assert "BUY pressure" in text


class TestBuildIndicatorsText:
    def test_empty_klines(self):
        strategy, _, _ = _make_strategy()
        text = strategy._build_indicators_text({}, None)
        assert "No indicator data" in text

    def test_uses_cache_when_available(self):
        strategy, _, _ = _make_strategy()
        cache = {"BTCUSDT": {"rsi_14": 45, "error": None}}
        klines = {"BTCUSDT": _make_klines()}
        text = strategy._build_indicators_text(klines, cache)
        assert "BTCUSDT" in text
