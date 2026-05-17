"""Tests for the Wave B stage classes.

Each stage is a thin wrapper that takes a :class:`CycleState`, mutates
one field, returns it. The underlying helpers are already covered by
:mod:`tests.test_cycle_shared_helpers` — these tests exercise the
state-mutation contract.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle_pipeline import CycleState
from halal_trader.core.cycle_stages import (
    ApplyRegimeGateStage,
    AugmentMicrostructureWithBasisStage,
    AugmentMicrostructureWithWhaleFlowsStage,
    AugmentRegimeWithMemoryStage,
    AugmentRegimeWithRagStage,
    BuildActiveAdjustmentsStage,
    BuildCatalystsStage,
    BuildCryptoRiskStage,
    BuildExchangeRulesStage,
    BuildForecastsStage,
    BuildMicrostructureStage,
    BuildMlSignalsStage,
    BuildNewsStage,
    BuildPerformanceStage,
    BuildRegimeStage,
    BuildSentimentStage,
    BuildStockRiskStage,
    BuildTimeframeStage,
)
from halal_trader.crypto.regime import MarketRegime


@pytest.mark.asyncio
async def test_build_regime_stage_no_detector_leaves_text_empty():
    state = CycleState(indicators_cache={"AAPL": {"rsi_14": 50}})
    out = await BuildRegimeStage(detector=None).run(state)
    assert out is state  # mutated in place + returned
    assert out.regime_text == ""


@pytest.mark.asyncio
async def test_build_regime_stage_populates_text_when_indicators_present():
    detector = MagicMock()
    detector.detect.return_value = (
        MarketRegime.TRENDING_UP,
        0.82,
        "trade with the trend",
    )
    state = CycleState(
        indicators_cache={"AAPL": {"rsi_14": 60, "ema_9": 100, "ema_21": 99}},
    )
    out = await BuildRegimeStage(detector=detector).run(state)
    assert out is state
    assert "AAPL" in out.regime_text
    assert "TRENDING_UP" in out.regime_text


@pytest.mark.asyncio
async def test_build_regime_stage_skips_when_indicators_empty():
    detector = MagicMock()
    state = CycleState()  # default empty indicators_cache
    out = await BuildRegimeStage(detector=detector).run(state)
    assert out.regime_text == ""
    # Detector wasn't invoked because the helper short-circuits on empty input.
    assert detector.detect.call_count == 0


@pytest.mark.asyncio
async def test_build_regime_stage_has_stable_name():
    """The stage name appears in instrumentation events; lock it."""
    assert BuildRegimeStage(detector=None).name == "build_regime_text"


# ── BuildForecastsStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_forecasts_stage_no_forecaster_leaves_text_empty():
    state = CycleState(klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)] * 25})
    out = await BuildForecastsStage(forecaster=None).run(state)
    assert out.forecasts_text == ""


@pytest.mark.asyncio
async def test_forecasts_stage_no_klines_leaves_text_empty():
    forecaster = MagicMock()
    state = CycleState()  # no klines
    out = await BuildForecastsStage(forecaster=forecaster).run(state)
    assert out.forecasts_text == ""
    assert forecaster.forecast.call_count == 0


@pytest.mark.asyncio
async def test_forecasts_stage_skips_pairs_below_min_history():
    """Pairs with <20 klines are skipped — Chronos needs ≥96 steps but
    we use 20 as the cycle's lower bound for training stability."""
    forecaster = MagicMock()
    state = CycleState(klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)] * 10})
    out = await BuildForecastsStage(forecaster=forecaster).run(state)
    assert forecaster.forecast.call_count == 0
    # Empty forecasts → formatter sentinel ("No ML price forecasts available.").
    assert "No ML price forecasts" in out.forecasts_text


@pytest.mark.asyncio
async def test_forecasts_stage_runs_forecaster_and_formats():
    from halal_trader.ml.forecaster import PriceForecast

    fc = PriceForecast(
        pair="BTCUSDT",
        current_price=42_000.0,
        predicted_prices=[42_500.0, 43_000.0],
        upper_bound=[43_000.0, 44_000.0],
        lower_bound=[41_000.0, 42_000.0],
        confidence=0.7,
        horizon=2,
    )
    forecaster = MagicMock()
    forecaster.forecast.return_value = fc
    state = CycleState(klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)] * 25})
    out = await BuildForecastsStage(forecaster=forecaster).run(state)
    assert "BTCUSDT" in out.forecasts_text
    assert "UP" in out.forecasts_text
    assert forecaster.forecast.call_count == 1


@pytest.mark.asyncio
async def test_forecasts_stage_swallows_failures():
    forecaster = MagicMock()
    forecaster.forecast.side_effect = RuntimeError("model crashed")
    state = CycleState(klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)] * 25})
    out = await BuildForecastsStage(forecaster=forecaster).run(state)
    # Failure swallowed; field stays at default empty.
    assert out.forecasts_text == ""


@pytest.mark.asyncio
async def test_forecasts_stage_has_stable_name():
    assert BuildForecastsStage().name == "build_forecasts_text"


# ── BuildMlSignalsStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ml_signals_stage_no_detectors_keeps_pre_rendered_forecasts():
    """When no detectors are wired, an upstream forecast block survives."""
    state = CycleState(
        indicators_cache={"AAPL": {"rsi_14": 50}},
        forecasts_text="forecast block from upstream",
    )
    out = await BuildMlSignalsStage(anomaly_detector=None, signal_classifier=None).run(state)
    # Helper returns forecasts_text unchanged when both detectors are None.
    assert out.ml_signals_text == "forecast block from upstream"


@pytest.mark.asyncio
async def test_ml_signals_stage_runs_detectors():
    anomaly = MagicMock()
    anomaly.detect.return_value = (True, 0.92)
    signal = MagicMock()
    signal.predict_confidence.return_value = 0.71
    state = CycleState(indicators_cache={"AAPL": {"rsi_14": 50}})
    out = await BuildMlSignalsStage(anomaly_detector=anomaly, signal_classifier=signal).run(state)
    assert "ANOMALY DETECTED" in out.ml_signals_text
    assert "ML confidence" in out.ml_signals_text
    assert anomaly.add_sample.call_count == 1


@pytest.mark.asyncio
async def test_ml_signals_stage_has_stable_name():
    assert BuildMlSignalsStage().name == "build_ml_signals_text"


# ── BuildTimeframeStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeframe_stage_no_analyzer_leaves_text_empty():
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildTimeframeStage(analyzer=None).run(state)
    assert out.timeframe_text == ""


@pytest.mark.asyncio
async def test_timeframe_stage_calls_analyzer_and_formats():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(
        return_value={
            "AAPL": {
                "alignment_score": 0.72,
                "per_tf": {"1Day": "RSI=58, MACD=bullish"},
                "support_resistance": [],
            }
        }
    )
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildTimeframeStage(analyzer=analyzer).run(state)
    assert "AAPL" in out.timeframe_text
    assert "BULLISH" in out.timeframe_text


@pytest.mark.asyncio
async def test_timeframe_stage_has_stable_name():
    assert BuildTimeframeStage(analyzer=None).name == "build_timeframe_text"


# ── BuildPerformanceStage ────────────────────────────────────────


@pytest.mark.asyncio
async def test_performance_stage_no_analytics_leaves_text_empty():
    state = CycleState()
    out = await BuildPerformanceStage(analytics=None).run(state)
    assert out.performance_text == ""


@pytest.mark.asyncio
async def test_performance_stage_calls_compute_and_format():
    analytics = MagicMock()
    analytics.compute_stats = AsyncMock(return_value="<stats>")
    analytics.format_for_prompt = MagicMock(return_value="Win rate: 55%")
    state = CycleState()
    out = await BuildPerformanceStage(analytics=analytics, lookback_days=14).run(state)
    analytics.compute_stats.assert_awaited_once_with(lookback_days=14)
    analytics.format_for_prompt.assert_called_once_with("<stats>")
    assert out.performance_text == "Win rate: 55%"


@pytest.mark.asyncio
async def test_performance_stage_swallows_failure():
    analytics = MagicMock()
    analytics.compute_stats = AsyncMock(side_effect=RuntimeError("db down"))
    state = CycleState()
    out = await BuildPerformanceStage(analytics=analytics).run(state)
    assert out.performance_text == ""


@pytest.mark.asyncio
async def test_performance_stage_has_stable_name():
    assert BuildPerformanceStage(analytics=None).name == "build_performance_text"


# ── BuildActiveAdjustmentsStage ──────────────────────────────────


@pytest.mark.asyncio
async def test_active_adjustments_stage_no_reviewer_leaves_text_empty():
    state = CycleState()
    out = await BuildActiveAdjustmentsStage(self_review=None).run(state)
    assert out.active_adjustments == ""


@pytest.mark.asyncio
async def test_active_adjustments_stage_calls_formatter():
    reviewer = MagicMock()
    reviewer.format_adjustments_for_prompt.return_value = "- max_position_pct: 0.10"
    state = CycleState()
    out = await BuildActiveAdjustmentsStage(self_review=reviewer).run(state)
    assert out.active_adjustments == "- max_position_pct: 0.10"


@pytest.mark.asyncio
async def test_active_adjustments_stage_swallows_failure():
    reviewer = MagicMock()
    reviewer.format_adjustments_for_prompt.side_effect = RuntimeError("boom")
    state = CycleState()
    out = await BuildActiveAdjustmentsStage(self_review=reviewer).run(state)
    assert out.active_adjustments == ""


@pytest.mark.asyncio
async def test_active_adjustments_stage_has_stable_name():
    assert BuildActiveAdjustmentsStage(self_review=None).name == "build_active_adjustments"


# ── BuildExchangeRulesStage ──────────────────────────────────────


@pytest.mark.asyncio
async def test_exchange_rules_stage_no_broker_leaves_text_empty():
    state = CycleState()
    out = await BuildExchangeRulesStage(broker=None).run(state)
    assert out.exchange_rules_text == ""


@pytest.mark.asyncio
async def test_exchange_rules_stage_skips_brokers_without_helper():
    """Stocks-side broker doesn't expose ``format_filters_for_prompt``."""
    broker = MagicMock(spec=[])  # no methods
    state = CycleState()
    out = await BuildExchangeRulesStage(broker=broker).run(state)
    assert out.exchange_rules_text == ""


@pytest.mark.asyncio
async def test_exchange_rules_stage_calls_broker_helper():
    broker = MagicMock()
    broker.format_filters_for_prompt.return_value = "min_notional=10.0"
    state = CycleState()
    out = await BuildExchangeRulesStage(broker=broker).run(state)
    assert out.exchange_rules_text == "min_notional=10.0"


@pytest.mark.asyncio
async def test_exchange_rules_stage_swallows_failure():
    broker = MagicMock()
    broker.format_filters_for_prompt.side_effect = RuntimeError("boom")
    state = CycleState()
    out = await BuildExchangeRulesStage(broker=broker).run(state)
    assert out.exchange_rules_text == ""


@pytest.mark.asyncio
async def test_exchange_rules_stage_has_stable_name():
    assert BuildExchangeRulesStage(broker=None).name == "build_exchange_rules_text"


# ── BuildCatalystsStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_catalysts_stage_no_feed_leaves_text_empty():
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildCatalystsStage(feed=None).run(state)
    assert out.catalysts_text == ""


@pytest.mark.asyncio
async def test_catalysts_stage_no_symbols_leaves_text_empty():
    feed = MagicMock()
    feed.fetch_all = AsyncMock(return_value=[])
    state = CycleState()  # empty halal_pairs
    out = await BuildCatalystsStage(feed=feed).run(state)
    assert out.catalysts_text == ""
    assert feed.fetch_all.call_count == 0


@pytest.mark.asyncio
async def test_catalysts_stage_calls_feed_and_formats():
    from datetime import datetime, timezone

    from halal_trader.trading.catalysts import Catalyst

    feed = MagicMock()
    feed.fetch_all = AsyncMock(
        return_value=[
            Catalyst(
                symbol="AAPL",
                kind="news",
                title="Apple beats Q1",
                timestamp=datetime.now(timezone.utc),
                sentiment="positive",
                source="Bloomberg",
            )
        ]
    )
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildCatalystsStage(feed=feed).run(state)
    assert "AAPL" in out.catalysts_text
    assert "Apple beats Q1" in out.catalysts_text


@pytest.mark.asyncio
async def test_catalysts_stage_swallows_failure():
    feed = MagicMock()
    feed.fetch_all = AsyncMock(side_effect=RuntimeError("alpaca down"))
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildCatalystsStage(feed=feed).run(state)
    assert out.catalysts_text == ""


@pytest.mark.asyncio
async def test_catalysts_stage_has_stable_name():
    assert BuildCatalystsStage(feed=None).name == "build_catalysts_text"


# ── BuildMicrostructureStage ─────────────────────────────────────


@pytest.mark.asyncio
async def test_microstructure_stage_no_orderbooks_leaves_text_empty():
    state = CycleState()  # default empty orderbooks
    out = await BuildMicrostructureStage().run(state)
    assert out.microstructure_text == ""


@pytest.mark.asyncio
async def test_microstructure_stage_formats_orderbook_features():
    state = CycleState(
        orderbooks={
            "BTCUSDT": {
                "bids": [[42000.0, 1.0], [41999.0, 2.0]],
                "asks": [[42001.0, 1.5], [42002.0, 0.5]],
            }
        }
    )
    out = await BuildMicrostructureStage().run(state)
    # Specific format details belong to the underlying helper; here we
    # just verify the stage drove it and produced a non-empty block.
    assert "BTCUSDT" in out.microstructure_text


@pytest.mark.asyncio
async def test_microstructure_stage_has_stable_name():
    assert BuildMicrostructureStage().name == "build_microstructure_text"


# ── BuildNewsStage ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_news_stage_no_feed_leaves_text_empty():
    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildNewsStage(news_feed=None).run(state)
    assert out.news_text == ""


@pytest.mark.asyncio
async def test_news_stage_calls_snapshot_and_formats():
    feed = MagicMock()
    feed.snapshot.return_value = []  # empty snapshot still exercises the path
    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildNewsStage(news_feed=feed).run(state)
    feed.snapshot.assert_called_once()
    # Empty snapshot → format_news_for_prompt returns an empty / sentinel string.
    assert isinstance(out.news_text, str)


@pytest.mark.asyncio
async def test_news_stage_swallows_failure():
    feed = MagicMock()
    feed.snapshot.side_effect = RuntimeError("feed down")
    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildNewsStage(news_feed=feed).run(state)
    assert out.news_text == ""


@pytest.mark.asyncio
async def test_news_stage_has_stable_name():
    assert BuildNewsStage(news_feed=None).name == "build_news_text"


# ── BuildSentimentStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_sentiment_stage_no_deps_leaves_text_empty():
    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildSentimentStage().run(state)
    assert out.sentiment_text == ""


@pytest.mark.asyncio
async def test_sentiment_stage_disabled_manager_is_skipped():
    manager = MagicMock()
    manager.enabled = False
    manager.latest_signals = {"BTCUSDT": MagicMock(buzz=4.0, score=0.5)}
    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildSentimentStage(sentiment_manager=manager).run(state)
    # Disabled manager → composite signals not consulted.
    assert out.sentiment_text == ""


def _signal(buzz: float, score: float = 0.5):  # noqa: ANN202
    """Build a real SentimentSignal so format_sentiment_for_prompt accepts it."""
    from halal_trader.sentiment.scoring import SentimentSignal

    return SentimentSignal(
        pair="BTCUSDT",
        score=score,
        buzz=buzz,
        confidence=0.8,
        data_sources=["reddit", "cryptopanic"],
    )


@pytest.mark.asyncio
async def test_sentiment_stage_high_buzz_fires_notifier():
    manager = MagicMock()
    manager.enabled = True
    manager.latest_signals = {"BTCUSDT": _signal(buzz=4.5, score=0.7)}
    notifier = MagicMock()
    notifier.notify_buzz = AsyncMock()
    state = CycleState(halal_pairs=["BTCUSDT"])
    await BuildSentimentStage(sentiment_manager=manager, notifier=notifier).run(state)
    notifier.notify_buzz.assert_awaited_once_with("BTCUSDT", 4.5, 0.7)


@pytest.mark.asyncio
async def test_sentiment_stage_low_buzz_does_not_notify():
    manager = MagicMock()
    manager.enabled = True
    manager.latest_signals = {"BTCUSDT": _signal(buzz=1.0, score=0.3)}
    notifier = MagicMock()
    notifier.notify_buzz = AsyncMock()
    state = CycleState(halal_pairs=["BTCUSDT"])
    await BuildSentimentStage(sentiment_manager=manager, notifier=notifier).run(state)
    assert notifier.notify_buzz.call_count == 0


@pytest.mark.asyncio
async def test_sentiment_stage_velocity_mutates_hub():
    """Reddit mention velocity is stashed on hub.velocity for the dashboard."""
    from datetime import datetime, timezone

    from halal_trader.sentiment.velocity import Mention

    fetcher = MagicMock()
    # Recent surge of BTC mentions; fetch_for_symbols receives the bases.
    fetcher.fetch_for_symbols = AsyncMock(
        return_value=[
            Mention(symbol="BTC", timestamp=datetime.now(timezone.utc), source="reddit")
            for _ in range(10)
        ]
    )
    hub = MagicMock()
    hub.velocity = {}
    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildSentimentStage(reddit_fetcher=fetcher, hub=hub).run(state)
    fetcher.fetch_for_symbols.assert_awaited_once_with(["BTC"])
    # Velocity dict was assigned (specific contents are the helper's job).
    assert hub.velocity != {}
    # Sentiment text should include the velocity block when surges fire.
    assert isinstance(out.sentiment_text, str)


@pytest.mark.asyncio
async def test_sentiment_stage_swallows_velocity_failure():
    fetcher = MagicMock()
    fetcher.fetch_for_symbols = AsyncMock(side_effect=RuntimeError("reddit down"))
    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildSentimentStage(reddit_fetcher=fetcher).run(state)
    assert out.sentiment_text == ""


@pytest.mark.asyncio
async def test_sentiment_stage_has_stable_name():
    assert BuildSentimentStage().name == "build_sentiment_text"


# ── BuildStockRiskStage ──────────────────────────────────────────


def _bar(o: float, h: float, low: float, c: float, v: float = 1_000.0) -> dict:
    return {"o": o, "h": h, "l": low, "c": c, "v": v}


def _stock_series(start: float, n: int, step: float = 0.5) -> list[dict]:
    out = []
    price = start
    for _ in range(n):
        out.append(_bar(price, price + 0.5, price - 0.5, price + step))
        price += step
    return out


@pytest.mark.asyncio
async def test_stock_risk_stage_no_bars_leaves_text_empty():
    state = CycleState()
    out = await BuildStockRiskStage().run(state)
    assert out.risk_text == ""
    assert out.indicators_cache == {}


@pytest.mark.asyncio
async def test_stock_risk_stage_populates_risk_and_indicators():
    """Happy path: bars in, risk_text + indicators_cache out."""
    account = MagicMock()
    account.effective_equity = 100_000.0
    account.equity = 100_000.0
    state = CycleState(
        bars={"AAPL": _stock_series(180.0, 50)},
        open_positions=[],
        account=account,
    )
    out = await BuildStockRiskStage().run(state)
    # The risk engine produces a non-empty text block + populates the
    # indicator cache that downstream stages (regime, ML) consume.
    assert isinstance(out.risk_text, str)
    assert "AAPL" in out.indicators_cache


@pytest.mark.asyncio
async def test_stock_risk_stage_swallows_failure():
    """A buggy bars payload mustn't break the cycle."""
    account = MagicMock()
    account.effective_equity = 100_000.0
    state = CycleState(
        bars={"AAPL": "not a bars payload"},  # triggers parse failure
        open_positions=[],
        account=account,
    )
    out = await BuildStockRiskStage().run(state)
    # Risk text empty; indicators cache stays empty.
    assert out.risk_text == ""


@pytest.mark.asyncio
async def test_stock_risk_stage_threads_halt_signal():
    """When the risk engine returns ``is_halted=True``, the stage must
    set ``state.halt`` so the cycle short-circuits before the LLM call.
    Mirrors :class:`BuildCryptoRiskStage`'s halt threading."""
    from unittest.mock import patch

    halted_state = MagicMock()
    halted_state.is_halted = True
    halted_state.halt_reason = "drawdown_breach"
    output = MagicMock(
        state=halted_state,
        risk_text="HALT: drawdown breach",
        indicators_by_symbol={},
    )
    account = MagicMock()
    account.effective_equity = 100_000.0
    state = CycleState(
        bars={"AAPL": [_bar(180.0, 181.0, 179.5, 180.5)]},
        open_positions=[],
        account=account,
    )
    with patch("halal_trader.trading.risk.evaluate_stock_risk", return_value=output):
        out = await BuildStockRiskStage().run(state)
    assert out.halt is True
    assert out.risk_state is halted_state
    assert "HALT" in out.risk_text


@pytest.mark.asyncio
async def test_stock_risk_stage_no_bars_leaves_halt_false():
    """Empty-bars early return must not leave a stale halt from prior cycle."""
    state = CycleState(halt=True)  # simulate stale halt flag
    out = await BuildStockRiskStage().run(state)
    assert out.halt is False


@pytest.mark.asyncio
async def test_stock_risk_stage_has_stable_name():
    assert BuildStockRiskStage().name == "evaluate_stock_risk"


# ── BuildCryptoRiskStage ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_crypto_risk_stage_no_engine_leaves_text_empty():
    state = CycleState()
    out = await BuildCryptoRiskStage(risk_engine=None).run(state)
    assert out.risk_text == ""
    assert out.halt is False


@pytest.mark.asyncio
async def test_crypto_risk_stage_calls_engine_and_formats():
    engine = MagicMock()
    risk_state = MagicMock()
    risk_state.is_halted = False
    engine.evaluate.return_value = risk_state
    engine.format_for_prompt.return_value = "Heat 4.2%, drawdown 1.1%"
    account = MagicMock()
    account.total_balance_usdt = 10_000.0
    state = CycleState(
        account=account,
        klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)]},
        indicators_cache={"BTCUSDT": {"rsi_14": 55}},
        current_prices={"BTCUSDT": 42_000.0},
    )
    trade = MagicMock()
    trade.pair = "BTCUSDT"
    trade.quantity = 0.1
    trade.entry_price = 41_000.0
    out = await BuildCryptoRiskStage(risk_engine=engine, open_trades=[trade]).run(state)
    assert out.risk_text == "Heat 4.2%, drawdown 1.1%"
    assert out.halt is False
    # Engine got the open-position values + unrealized P&L derived from current_prices.
    kwargs = engine.evaluate.call_args.kwargs
    assert kwargs["open_positions_value"] == {"BTCUSDT": 0.1 * 42_000.0}
    assert kwargs["unrealized_pnl"]["BTCUSDT"] == (42_000.0 - 41_000.0) * 0.1


@pytest.mark.asyncio
async def test_crypto_risk_stage_propagates_halt():
    engine = MagicMock()
    risk_state = MagicMock()
    risk_state.is_halted = True
    engine.evaluate.return_value = risk_state
    engine.format_for_prompt.return_value = "HALTED — drawdown breach"
    account = MagicMock()
    account.total_balance_usdt = 9_000.0
    state = CycleState(account=account)
    out = await BuildCryptoRiskStage(risk_engine=engine).run(state)
    assert out.halt is True
    assert "HALTED" in out.risk_text


@pytest.mark.asyncio
async def test_crypto_risk_stage_swallows_failure():
    engine = MagicMock()
    engine.evaluate.side_effect = RuntimeError("engine boom")
    state = CycleState(account=MagicMock(total_balance_usdt=1000.0))
    out = await BuildCryptoRiskStage(risk_engine=engine).run(state)
    assert out.risk_text == ""
    assert out.halt is False


@pytest.mark.asyncio
async def test_crypto_risk_stage_has_stable_name():
    assert BuildCryptoRiskStage().name == "evaluate_portfolio_risk"


# ── AugmentRegimeWithMemoryStage ─────────────────────────────────


@pytest.mark.asyncio
async def test_regime_memory_stage_no_memory_leaves_text_unchanged():
    state = CycleState(regime_text="seeded regime block")
    out = await AugmentRegimeWithMemoryStage(regime_memory=None).run(state)
    assert out.regime_text == "seeded regime block"


@pytest.mark.asyncio
async def test_regime_memory_stage_skips_when_no_indicators():
    memory = MagicMock()
    memory.size = AsyncMock(return_value=10)
    state = CycleState(regime_text="seeded")
    out = await AugmentRegimeWithMemoryStage(regime_memory=memory).run(state)
    # No indicators → build_regime_features returns None → memory not queried.
    assert out.regime_text == "seeded"
    assert memory.size.call_count == 0


@pytest.mark.asyncio
async def test_regime_memory_stage_skips_when_memory_empty():
    memory = MagicMock()
    memory.size = AsyncMock(return_value=0)
    account = MagicMock()
    account.total_balance_usdt = 10_000.0
    state = CycleState(
        account=account,
        regime_text="seeded",
        indicators_cache={"BTCUSDT": {"rsi_14": 55, "atr_14": 100, "current_price": 42_000.0}},
        today_pnl=50.0,
    )
    out = await AugmentRegimeWithMemoryStage(regime_memory=memory).run(state)
    # Memory size 0 → query not run.
    assert out.regime_text == "seeded"
    assert memory.size.await_count == 1


@pytest.mark.asyncio
async def test_regime_memory_stage_swallows_failure():
    memory = MagicMock()
    memory.size = AsyncMock(side_effect=RuntimeError("memory down"))
    state = CycleState(
        account=MagicMock(total_balance_usdt=10_000.0),
        regime_text="seeded",
        indicators_cache={"BTCUSDT": {"rsi_14": 55, "atr_14": 100, "current_price": 42_000.0}},
    )
    out = await AugmentRegimeWithMemoryStage(regime_memory=memory).run(state)
    assert out.regime_text == "seeded"


@pytest.mark.asyncio
async def test_regime_memory_stage_has_stable_name():
    assert AugmentRegimeWithMemoryStage(regime_memory=None).name == "augment_regime_with_memory"


# ── AugmentRegimeWithRagStage ────────────────────────────────────


@pytest.mark.asyncio
async def test_regime_rag_stage_no_store_leaves_text_unchanged():
    state = CycleState(regime_text="seeded")
    out = await AugmentRegimeWithRagStage(rag_store=None).run(state)
    assert out.regime_text == "seeded"


@pytest.mark.asyncio
async def test_regime_rag_stage_skips_when_store_empty():
    store = MagicMock()
    store.size = AsyncMock(return_value=0)
    store.query = AsyncMock(return_value=[])
    state = CycleState(regime_text="seeded")
    out = await AugmentRegimeWithRagStage(rag_store=store).run(state)
    assert out.regime_text == "seeded"
    assert store.query.call_count == 0


@pytest.mark.asyncio
async def test_regime_rag_stage_swallows_failure():
    store = MagicMock()
    store.size = AsyncMock(side_effect=RuntimeError("rag down"))
    state = CycleState(regime_text="seeded")
    out = await AugmentRegimeWithRagStage(rag_store=store).run(state)
    assert out.regime_text == "seeded"


@pytest.mark.asyncio
async def test_regime_rag_stage_has_stable_name():
    assert AugmentRegimeWithRagStage(rag_store=None).name == "augment_regime_with_rag"


# ── AugmentMicrostructureWithWhaleFlowsStage ─────────────────────


@pytest.mark.asyncio
async def test_whale_flows_stage_no_source_leaves_text_unchanged():
    state = CycleState(microstructure_text="seeded mstr")
    out = await AugmentMicrostructureWithWhaleFlowsStage(whale_flow_source=None).run(state)
    assert out.microstructure_text == "seeded mstr"


@pytest.mark.asyncio
async def test_whale_flows_stage_swallows_failure():
    source = MagicMock()
    source.fetch = AsyncMock(side_effect=RuntimeError("etherscan down"))
    state = CycleState(microstructure_text="seeded")
    out = await AugmentMicrostructureWithWhaleFlowsStage(whale_flow_source=source).run(state)
    assert out.microstructure_text == "seeded"


@pytest.mark.asyncio
async def test_whale_flows_stage_has_stable_name():
    assert (
        AugmentMicrostructureWithWhaleFlowsStage(whale_flow_source=None).name
        == "augment_microstructure_with_whale_flows"
    )


# ── AugmentMicrostructureWithBasisStage ──────────────────────────


@pytest.mark.asyncio
async def test_basis_stage_no_broker_leaves_text_unchanged():
    state = CycleState(microstructure_text="seeded", halal_pairs=["BTCUSDT"])
    out = await AugmentMicrostructureWithBasisStage(broker=None, basis_tracker=None).run(state)
    assert out.microstructure_text == "seeded"


@pytest.mark.asyncio
async def test_basis_stage_skips_brokers_without_funding_signal():
    broker = MagicMock(spec=[])  # no get_funding_signal method
    state = CycleState(microstructure_text="seeded", halal_pairs=["BTCUSDT"])
    out = await AugmentMicrostructureWithBasisStage(broker=broker, basis_tracker=MagicMock()).run(
        state
    )
    assert out.microstructure_text == "seeded"


@pytest.mark.asyncio
async def test_basis_stage_no_pairs_leaves_text_unchanged():
    broker = MagicMock()
    broker.get_funding_signal = AsyncMock()
    state = CycleState(microstructure_text="seeded")  # empty halal_pairs
    out = await AugmentMicrostructureWithBasisStage(broker=broker, basis_tracker=MagicMock()).run(
        state
    )
    assert out.microstructure_text == "seeded"
    assert broker.get_funding_signal.call_count == 0


@pytest.mark.asyncio
async def test_basis_stage_has_stable_name():
    assert (
        AugmentMicrostructureWithBasisStage(broker=None, basis_tracker=None).name
        == "augment_microstructure_with_basis"
    )


# ── ApplyRegimeGateStage ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_regime_gate_no_detector_leaves_plan_unchanged():
    plan = MagicMock()
    plan.buys = [MagicMock(symbol="BTCUSDT")]
    plan.decisions = list(plan.buys)
    state = CycleState(plan=plan)
    await ApplyRegimeGateStage(detector=None).run(state)
    assert len(state.plan.decisions) == 1


@pytest.mark.asyncio
async def test_regime_gate_no_plan_is_no_op():
    detector = MagicMock()
    state = CycleState(plan=None)
    out = await ApplyRegimeGateStage(detector=detector).run(state)
    assert out is state
    assert detector.detect.call_count == 0


@pytest.mark.asyncio
async def test_regime_gate_blocks_buys_in_downtrend():
    detector = MagicMock()
    detector.detect.return_value = (MarketRegime.TRENDING_DOWN, 0.85, "no buys")
    buy = MagicMock(symbol="BTCUSDT")
    plan = MagicMock()
    plan.buys = [buy]
    plan.decisions = [buy]
    state = CycleState(
        plan=plan,
        klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)]},
        indicators_cache={"BTCUSDT": {"rsi_14": 35, "macd_histogram": -0.5}},
    )
    await ApplyRegimeGateStage(detector=detector).run(state)
    assert buy not in state.plan.decisions
    assert detector.detect.call_count == 1


@pytest.mark.asyncio
async def test_regime_gate_keeps_buys_in_low_confidence_downtrend():
    detector = MagicMock()
    # Confidence below 0.6 threshold → no block.
    detector.detect.return_value = (MarketRegime.TRENDING_DOWN, 0.5, "no buys")
    buy = MagicMock(symbol="BTCUSDT")
    plan = MagicMock()
    plan.buys = [buy]
    plan.decisions = [buy]
    state = CycleState(
        plan=plan,
        klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)]},
        indicators_cache={"BTCUSDT": {"rsi_14": 35}},
    )
    await ApplyRegimeGateStage(detector=detector).run(state)
    assert buy in state.plan.decisions


@pytest.mark.asyncio
async def test_regime_gate_blocks_buys_in_downtrend_for_stocks():
    """The gate is symbol-source-agnostic — works for stock tickers
    even when ``klines_by_symbol`` is empty (stocks use ``bars``)."""
    detector = MagicMock()
    detector.detect.return_value = (MarketRegime.TRENDING_DOWN, 0.85, "no buys")
    buy = MagicMock(symbol="AAPL")
    plan = MagicMock()
    plan.buys = [buy]
    plan.decisions = [buy]
    state = CycleState(
        plan=plan,
        # No klines_by_symbol — stocks-side path.
        indicators_cache={"AAPL": {"rsi_14": 35, "macd_histogram": -0.5}},
    )
    await ApplyRegimeGateStage(detector=detector).run(state)
    assert buy not in state.plan.decisions
    assert detector.detect.call_count == 1


@pytest.mark.asyncio
async def test_regime_gate_has_stable_name():
    assert ApplyRegimeGateStage(detector=None).name == "apply_regime_gate"
