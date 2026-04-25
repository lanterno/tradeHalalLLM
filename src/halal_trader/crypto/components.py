"""Crypto component registry — builds the full live-trading wiring once.

The scheduler used to do this inline in a 215-line ``_create_components``
method with deeply nested conditionals for the ML / sentiment / news
branches. This module centralises the wiring; the scheduler just calls
:func:`build_components` and fans the result out into its own attributes.

Optional subsystems (sentiment, ML, news reactor) live behind small
``_build_*`` helpers; each returns ``None`` when not configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.config import Settings
from halal_trader.core.llm import create_llm
from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.safeguards import LiveModeChecker, check_live_mode_token
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.monitor import PositionMonitor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.regime import RegimeDetector
from halal_trader.crypto.risk import PortfolioRiskEngine
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.self_improve import TradeSelfReview
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.timeframes import TimeframeAnalyzer
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.db.repository import Repository
from halal_trader.ml.retrainer import RetrainingScheduler
from halal_trader.notifications.telegram import AlertSink, TelegramNotifier
from halal_trader.sentiment.events import NewsEventReactor

logger = logging.getLogger(__name__)


@dataclass
class CryptoComponents:
    """Every long-lived object the crypto cycle needs, in one bag."""

    # Live-mode safety
    live_mode_checker: LiveModeChecker

    # Brokers + market data
    binance: BinanceClient
    ws: BinanceWSManager

    # LLM
    llm: BaseLLM
    strategy: CryptoTradingStrategy

    # Order flow
    executor: CryptoExecutor
    monitor: PositionMonitor

    # Portfolio + analytics
    screener: CryptoHalalScreener
    portfolio: CryptoPortfolioTracker
    analytics: PerformanceAnalytics
    risk_engine: PortfolioRiskEngine
    self_review: TradeSelfReview
    retrainer: RetrainingScheduler

    # Notifications + ops
    notifier: TelegramNotifier
    alerts: AlertSink

    # Optional / conditionally-enabled
    sentiment_manager: Any = None
    timeframe_analyzer: Any = None
    regime_detector: RegimeDetector | None = None
    ml_forecaster: Any = None
    ml_anomaly: Any = None
    ml_signal: Any = None
    news_reactor: NewsEventReactor | None = None


# ── Optional subsystem builders ───────────────────────────────


def _build_sentiment(settings: Settings) -> Any:
    from halal_trader.sentiment.manager import SentimentManager

    return SentimentManager(
        trading_pairs=settings.crypto_pairs,
        reddit_client_id=settings.reddit_client_id,
        reddit_client_secret=settings.reddit_client_secret,
        cryptopanic_api_key=settings.cryptopanic_api_key,
        use_finbert=settings.sentiment_use_finbert,
        update_interval_seconds=settings.sentiment_update_interval_seconds,
    )


def _build_ml(settings: Settings) -> tuple[Any, Any, Any]:
    """Return (forecaster, anomaly_detector, signal_classifier) or all-None."""
    if not settings.ml_enabled:
        return None, None, None
    try:
        from halal_trader.ml.anomaly import MarketAnomalyDetector, MLSignalClassifier
        from halal_trader.ml.forecaster import PriceForecaster
        from halal_trader.ml.hub import ModelHub

        hub = ModelHub(device=settings.ml_device, models_dir=settings.ml_models_dir)
        logger.info("ML models enabled (device: %s)", settings.ml_device)
        return PriceForecaster(hub), MarketAnomalyDetector(hub), MLSignalClassifier(hub)
    except Exception as e:
        logger.warning("ML models initialization failed: %s", e)
        return None, None, None


def _build_news_reactor(settings: Settings) -> NewsEventReactor:
    return NewsEventReactor(
        api_key=settings.cryptopanic_api_key,
        trading_pairs=settings.crypto_pairs,
        poll_interval_seconds=30,
        importance_filter="hot",
    )


# ── Main builder ──────────────────────────────────────────────


async def build_components(
    *,
    settings: Settings,
    repo: Repository,
    engine: AsyncEngine | None,
    binance: BinanceClient,
    exiting_pairs: set[str],
) -> CryptoComponents:
    """Wire every long-lived crypto-bot component once.

    The scheduler still owns lifecycle (start/stop of background tasks);
    this function just constructs the objects in dependency order.
    """
    check_live_mode_token(settings, market="crypto")
    live_mode_checker = LiveModeChecker(settings=settings, market="crypto")

    await binance.connect()

    ws = BinanceWSManager(binance.client, symbols=settings.crypto_pairs)
    await ws.start()

    llm = create_llm(settings)

    screener = CryptoHalalScreener(
        repo,
        coingecko_api_key=settings.coingecko_api_key,
        min_market_cap=settings.crypto_min_market_cap,
    )

    strategy = CryptoTradingStrategy(
        llm,
        repo,
        llm_provider_name=settings.llm_provider.value,
        max_position_pct=settings.crypto_max_position_pct,
        daily_loss_limit=settings.crypto_daily_loss_limit,
        daily_return_target=settings.crypto_daily_return_target,
        max_simultaneous_positions=settings.crypto_max_simultaneous_positions,
        llm_failure_threshold=settings.crypto_llm_failure_threshold,
        llm_cooldown_seconds=settings.crypto_llm_cooldown_seconds,
    )

    executor = CryptoExecutor(
        binance,
        repo,
        max_position_pct=settings.crypto_max_position_pct,
        max_simultaneous_positions=settings.crypto_max_simultaneous_positions,
        configured_pairs=settings.crypto_pairs,
        circuit_breaker_threshold=settings.crypto_circuit_breaker_threshold,
        circuit_breaker_window=settings.crypto_circuit_breaker_window,
        circuit_breaker_cooldown=settings.crypto_circuit_breaker_cooldown,
        exiting_pairs=exiting_pairs,
    )

    portfolio = CryptoPortfolioTracker(
        binance, repo, daily_loss_limit=settings.crypto_daily_loss_limit
    )
    analytics = PerformanceAnalytics(repo)

    notifier = TelegramNotifier(
        bot_token=settings.telegram_bot_token, chat_id=settings.telegram_chat_id
    )
    alerts = AlertSink(notifier)
    if notifier.enabled:
        logger.info("Telegram notifications enabled")

    sentiment_manager = _build_sentiment(settings)
    if sentiment_manager is not None:
        await sentiment_manager.start()

    timeframe_analyzer = TimeframeAnalyzer(binance)
    regime_detector = RegimeDetector(models_dir=settings.ml_models_dir)
    ml_forecaster, ml_anomaly, ml_signal = _build_ml(settings)

    self_review = TradeSelfReview(llm, repo, strategy=strategy)
    await self_review.load_from_db()

    risk_engine = PortfolioRiskEngine(
        base_max_position_pct=settings.crypto_max_position_pct,
        max_portfolio_heat_pct=settings.crypto_max_portfolio_heat_pct,
        max_drawdown_pct=settings.crypto_max_drawdown_pct,
        high_correlation_threshold=settings.crypto_high_correlation_threshold,
        correlation_reduction_factor=settings.crypto_correlation_reduction_factor,
        atr_baseline=settings.crypto_atr_baseline,
    )

    retrainer = RetrainingScheduler(repo, models_dir=settings.ml_models_dir)

    monitor = PositionMonitor(
        broker=binance,
        repo=repo,
        ws_manager=ws,
        check_interval=settings.crypto_monitor_interval,
        trailing_stop_activation_pct=settings.crypto_trailing_stop_activation_pct,
        trailing_stop_distance_pct=settings.crypto_trailing_stop_distance_pct,
        notifier=notifier if notifier.enabled else None,
        retrainer=retrainer,
        exiting_pairs=exiting_pairs,
    )

    news_reactor = _build_news_reactor(settings)

    _ = engine  # kept in the signature so future components can read it.

    return CryptoComponents(
        live_mode_checker=live_mode_checker,
        binance=binance,
        ws=ws,
        llm=llm,
        strategy=strategy,
        executor=executor,
        monitor=monitor,
        screener=screener,
        portfolio=portfolio,
        analytics=analytics,
        risk_engine=risk_engine,
        self_review=self_review,
        retrainer=retrainer,
        notifier=notifier,
        alerts=alerts,
        sentiment_manager=sentiment_manager,
        timeframe_analyzer=timeframe_analyzer,
        regime_detector=regime_detector,
        ml_forecaster=ml_forecaster,
        ml_anomaly=ml_anomaly,
        ml_signal=ml_signal,
        news_reactor=news_reactor,
    )
