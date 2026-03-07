"""Crypto trading bot — composition root and 24/7 asyncio scheduler."""

from __future__ import annotations

import asyncio
import logging

from halal_trader.agent.llm import create_llm
from halal_trader.config import get_settings
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.monitor import PositionMonitor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository
from halal_trader.market_hours import today_eastern

logger = logging.getLogger(__name__)


class CryptoTradingBot:
    """Composition root and scheduler — wires crypto components and runs 24/7."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._binance = BinanceClient(
            api_key=self.settings.binance_api_key,
            secret_key=self.settings.binance_secret_key,
            testnet=self.settings.binance_testnet,
            configured_pairs=self.settings.crypto_pairs,
        )
        self._ws: BinanceWSManager | None = None
        self._screener: CryptoHalalScreener | None = None
        self._cycle_service: CryptoCycleService | None = None
        self._portfolio: CryptoPortfolioTracker | None = None
        self._monitor: PositionMonitor | None = None
        self._sentiment_manager = None
        self._self_review = None
        self._notifier = None
        self._running = False
        self._last_day: str | None = None

    async def initialize(self) -> None:
        """Set up all crypto trading components."""
        logger.info("Initializing crypto trading bot...")

        # Database
        engine = await init_db(str(self.settings.db_path))
        repo = Repository(engine)

        # Binance connection
        await self._binance.connect()

        # WebSocket manager for real-time klines
        self._ws = BinanceWSManager(
            self._binance.client,
            symbols=self.settings.crypto_pairs,
        )
        await self._ws.start()

        # LLM (reuse existing provider system)
        llm = create_llm(self.settings)

        # Halal screener
        self._screener = CryptoHalalScreener(
            repo,
            coingecko_api_key=self.settings.coingecko_api_key,
            min_market_cap=self.settings.crypto_min_market_cap,
        )

        # Strategy
        strategy = CryptoTradingStrategy(
            llm,
            repo,
            llm_provider_name=self.settings.llm_provider.value,
            max_position_pct=self.settings.crypto_max_position_pct,
            daily_loss_limit=self.settings.crypto_daily_loss_limit,
            daily_return_target=self.settings.crypto_daily_return_target,
            max_simultaneous_positions=self.settings.crypto_max_simultaneous_positions,
        )

        # Executor
        executor = CryptoExecutor(
            self._binance,
            repo,
            max_position_pct=self.settings.crypto_max_position_pct,
            max_simultaneous_positions=self.settings.crypto_max_simultaneous_positions,
            configured_pairs=self.settings.crypto_pairs,
        )

        # Portfolio tracker
        self._portfolio = CryptoPortfolioTracker(
            self._binance,
            repo,
            daily_loss_limit=self.settings.crypto_daily_loss_limit,
        )

        # Performance analytics
        analytics = PerformanceAnalytics(repo)

        # ── New Feature Components ─────────────────────────────

        # Telegram notifier
        from halal_trader.notifications.telegram import TelegramNotifier
        self._notifier = TelegramNotifier(
            bot_token=self.settings.telegram_bot_token,
            chat_id=self.settings.telegram_chat_id,
        )
        if self._notifier.enabled:
            logger.info("Telegram notifications enabled")

        # Sentiment manager (Reddit + CryptoPanic)
        from halal_trader.sentiment.manager import SentimentManager
        self._sentiment_manager = SentimentManager(
            trading_pairs=self.settings.crypto_pairs,
            reddit_client_id=self.settings.reddit_client_id,
            reddit_client_secret=self.settings.reddit_client_secret,
            cryptopanic_api_key=self.settings.cryptopanic_api_key,
            use_finbert=self.settings.sentiment_use_finbert,
            update_interval_seconds=self.settings.sentiment_update_interval_seconds,
        )
        await self._sentiment_manager.start()

        # Multi-timeframe analyzer
        from halal_trader.crypto.timeframes import TimeframeAnalyzer
        timeframe_analyzer = TimeframeAnalyzer(self._binance)

        # Market regime detector
        from halal_trader.crypto.regime import RegimeDetector
        regime_detector = RegimeDetector(models_dir=self.settings.ml_models_dir)

        # ML models (optional)
        ml_forecaster = None
        ml_anomaly = None
        ml_signal = None
        if self.settings.ml_enabled:
            try:
                from halal_trader.ml.anomaly import MarketAnomalyDetector, MLSignalClassifier
                from halal_trader.ml.forecaster import PriceForecaster
                from halal_trader.ml.hub import ModelHub

                hub = ModelHub(
                    device=self.settings.ml_device,
                    models_dir=self.settings.ml_models_dir,
                )
                ml_forecaster = PriceForecaster(hub)
                ml_anomaly = MarketAnomalyDetector(hub)
                ml_signal = MLSignalClassifier(hub)
                logger.info("ML models enabled (device: %s)", self.settings.ml_device)
            except Exception as e:
                logger.warning("ML models initialization failed: %s", e)

        # Self-improvement loop
        from halal_trader.crypto.self_improve import TradeSelfReview
        self._self_review = TradeSelfReview(llm, repo)

        # Cycle service (wired with all new components)
        self._cycle_service = CryptoCycleService(
            broker=self._binance,
            screener=self._screener,
            strategy=strategy,
            executor=executor,
            portfolio=self._portfolio,
            ws_manager=self._ws,
            configured_pairs=self.settings.crypto_pairs,
            analytics=analytics,
            sentiment_manager=self._sentiment_manager,
            timeframe_analyzer=timeframe_analyzer,
            regime_detector=regime_detector,
            ml_forecaster=ml_forecaster,
            ml_anomaly_detector=ml_anomaly,
            ml_signal_classifier=ml_signal,
            self_review=self._self_review,
            notifier=self._notifier if self._notifier.enabled else None,
        )

        # Position monitor (SL/TP enforcement)
        self._monitor = PositionMonitor(
            broker=self._binance,
            repo=repo,
            ws_manager=self._ws,
            check_interval=2.0,
            trailing_stop_activation_pct=0.005,
            trailing_stop_distance_pct=0.003,
        )
        await self._monitor.start()

        logger.info("Crypto trading bot initialized successfully")

    async def shutdown(self) -> None:
        """Clean up all resources."""
        logger.info("Shutting down crypto trading bot...")
        self._running = False
        if self._monitor:
            await self._monitor.stop()
        if self._sentiment_manager:
            await self._sentiment_manager.stop()
        if self._ws:
            await self._ws.stop()
        await self._binance.disconnect()
        logger.info("Crypto trading bot shut down")

    # ── Daily Routines ─────────────────────────────────────────

    async def _daily_start(self) -> None:
        """Daily start routine: refresh halal cache, record starting equity."""
        logger.info("=== CRYPTO DAILY START ===")
        try:
            await self._screener.refresh_screening()
            await self._portfolio.record_day_start()
            self._last_day = today_eastern().isoformat()
            logger.info("Crypto daily start complete")
        except Exception as e:
            logger.error("Crypto daily start failed: %s", e)

    async def _daily_end(self) -> None:
        """Daily end routine: record P&L snapshot, run self-review, send summary."""
        logger.info("=== CRYPTO DAILY END ===")
        try:
            summary = await self._portfolio.record_day_end()
            logger.info("Crypto daily summary: %s", summary)

            # Run self-review
            if self._self_review:
                try:
                    review = await self._self_review.review(lookback_days=1)
                    if review.observations:
                        logger.info(
                            "Self-review observations: %s",
                            "; ".join(review.observations[:3]),
                        )
                except Exception as e:
                    logger.debug("Self-review failed: %s", e)

            # Send daily summary via Telegram
            if self._notifier and self._notifier.enabled:
                try:
                    await self._notifier.notify_daily_summary(summary or {})
                except Exception:
                    pass
        except Exception as e:
            logger.error("Crypto daily end failed: %s", e)

    async def _check_day_rollover(self) -> None:
        """Check if we've crossed into a new Eastern day and handle rollover."""
        today = today_eastern().isoformat()
        if self._last_day is not None and today != self._last_day:
            # End the previous day and start the new one
            await self._daily_end()
            await self._daily_start()

    # ── Main Loop ──────────────────────────────────────────────

    async def run(self) -> None:
        """Start the crypto trading bot with continuous 1-minute cycles."""
        await self.initialize()
        self._running = True

        # Initial daily start
        await self._daily_start()

        interval = self.settings.crypto_trading_interval_seconds

        logger.info(
            "Crypto bot started — interval: %ds, target: %.1f%%, loss limit: %.1f%%, "
            "pairs: %s, testnet: %s",
            interval,
            self.settings.crypto_daily_return_target * 100,
            self.settings.crypto_daily_loss_limit * 100,
            ", ".join(self.settings.crypto_pairs),
            self.settings.binance_testnet,
        )

        try:
            while self._running:
                cycle_start = asyncio.get_event_loop().time()

                # Check for day rollover (UTC midnight)
                await self._check_day_rollover()

                # Check if self-review should trigger (consecutive losses)
                if self._self_review:
                    try:
                        if await self._self_review.should_trigger_review():
                            logger.info("Consecutive losses detected — triggering self-review")
                            await self._self_review.review(lookback_days=1)
                    except Exception:
                        pass

                # Run trading cycle
                await self._cycle_service.run_cycle()

                # Sleep for remaining interval time
                elapsed = asyncio.get_event_loop().time() - cycle_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    logger.debug(
                        "Cycle took %.1fs, sleeping %.1fs until next cycle",
                        elapsed,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(
                        "Cycle took %.1fs (exceeds %ds interval), running next immediately",
                        elapsed,
                        interval,
                    )

        except KeyboardInterrupt, asyncio.CancelledError:
            logger.info("Crypto bot interrupted")
        finally:
            await self._daily_end()
            await self.shutdown()

    async def run_once(self) -> None:
        """Run a single crypto trading cycle (useful for testing)."""
        await self.initialize()
        try:
            await self._daily_start()
            await self._cycle_service.run_cycle()
        finally:
            await self.shutdown()
