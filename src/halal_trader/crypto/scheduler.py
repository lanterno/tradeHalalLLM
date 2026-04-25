"""Crypto trading bot — composition root and 24/7 asyncio scheduler."""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from halal_trader.core.llm import create_llm
from halal_trader.core.scheduler import BaseTradingBot
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.monitor import PositionMonitor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.market_hours import today_eastern

logger = logging.getLogger(__name__)


class CryptoTradingBot(BaseTradingBot):
    """Composition root and scheduler — wires crypto components and runs 24/7."""

    def __init__(self) -> None:
        super().__init__()
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
        self._last_day: str | None = None
        self._exiting_pairs: set[str] = set()

    async def _create_components(self) -> None:
        """Create crypto-specific trading components."""
        logger.info("Initializing crypto trading bot...")

        repo = self._repo
        assert repo is not None

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
            llm_failure_threshold=self.settings.crypto_llm_failure_threshold,
            llm_cooldown_seconds=self.settings.crypto_llm_cooldown_seconds,
        )

        # Executor
        executor = CryptoExecutor(
            self._binance,
            repo,
            max_position_pct=self.settings.crypto_max_position_pct,
            max_simultaneous_positions=self.settings.crypto_max_simultaneous_positions,
            configured_pairs=self.settings.crypto_pairs,
            circuit_breaker_threshold=self.settings.crypto_circuit_breaker_threshold,
            circuit_breaker_window=self.settings.crypto_circuit_breaker_window,
            circuit_breaker_cooldown=self.settings.crypto_circuit_breaker_cooldown,
            exiting_pairs=self._exiting_pairs,
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

        # Telegram notifier + rate-limited error sink
        from halal_trader.notifications.telegram import AlertSink, TelegramNotifier

        self._notifier = TelegramNotifier(
            bot_token=self.settings.telegram_bot_token,
            chat_id=self.settings.telegram_chat_id,
        )
        self._alerts = AlertSink(self._notifier)
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

        # Self-improvement loop (load saved adjustments from DB)
        from halal_trader.crypto.self_improve import TradeSelfReview

        self._self_review = TradeSelfReview(llm, repo, strategy=strategy)
        await self._self_review.load_from_db()

        # Portfolio-level risk engine
        from halal_trader.crypto.risk import PortfolioRiskEngine

        risk_engine = PortfolioRiskEngine(
            base_max_position_pct=self.settings.crypto_max_position_pct,
            max_portfolio_heat_pct=self.settings.crypto_max_portfolio_heat_pct,
            max_drawdown_pct=self.settings.crypto_max_drawdown_pct,
            high_correlation_threshold=self.settings.crypto_high_correlation_threshold,
            correlation_reduction_factor=self.settings.crypto_correlation_reduction_factor,
            atr_baseline=self.settings.crypto_atr_baseline,
        )

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
            risk_engine=risk_engine,
            alerts=self._alerts,
            engine=self._engine,
        )

        # ML retrainer (labels closed trades and retrains models)
        from halal_trader.ml.retrainer import RetrainingScheduler

        self._retrainer = RetrainingScheduler(
            repo,
            models_dir=self.settings.ml_models_dir,
        )

        # Position monitor (SL/TP enforcement)
        notifier_for_monitor = self._notifier if self._notifier.enabled else None
        self._monitor = PositionMonitor(
            broker=self._binance,
            repo=repo,
            ws_manager=self._ws,
            check_interval=self.settings.crypto_monitor_interval,
            trailing_stop_activation_pct=self.settings.crypto_trailing_stop_activation_pct,
            trailing_stop_distance_pct=self.settings.crypto_trailing_stop_distance_pct,
            notifier=notifier_for_monitor,
            retrainer=self._retrainer,
            exiting_pairs=self._exiting_pairs,
        )
        await self._monitor.start()

        # News event reactor (real-time CryptoPanic stream)
        from halal_trader.sentiment.events import NewsEventReactor

        self._news_reactor = NewsEventReactor(
            api_key=self.settings.cryptopanic_api_key,
            trading_pairs=self.settings.crypto_pairs,
            poll_interval_seconds=30,
            importance_filter="hot",
        )
        if self._news_reactor.enabled:
            self._news_reactor.on_event(self._on_news_event)
            await self._news_reactor.start()

        # Expose live components to the dashboard
        from halal_trader.web.app import app_state as _web_state

        _web_state["ws_manager"] = self._ws
        _web_state["sentiment_manager"] = self._sentiment_manager
        _web_state["exchange"] = self._binance
        _web_state["bot_running"] = True

        logger.info("Crypto trading bot initialized successfully")

    def _get_cycle_service(self) -> CryptoCycleService:
        if self._cycle_service is None:
            raise RuntimeError("CryptoTradingBot.initialize() must be called first")
        return self._cycle_service

    async def _on_news_event(self, event) -> None:
        """Handle a breaking news event by triggering an emergency mini-cycle."""
        logger.info(
            "News event trigger: %s (affects %s)",
            event.title[:60],
            ", ".join(event.affected_pairs) or "general market",
        )

        if self._notifier and self._notifier.enabled:
            try:
                await self._notifier.send(
                    f"\U0001f4f0 <b>Breaking News</b>\n"
                    f"{event.title}\n"
                    f"Source: {event.source}\n"
                    f"Sentiment: {event.sentiment}\n"
                    f"Affects: {', '.join(event.affected_pairs) or 'general'}\n"
                    f"<i>Bot is evaluating response...</i>"
                )
            except Exception:
                pass

        if not self._cycle_service:
            return

        try:
            await self._cycle_service.run_cycle()
            logger.info("Emergency mini-cycle completed for news event")
        except Exception as e:
            logger.error("Emergency mini-cycle failed: %s", e)

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
                except Exception as e:
                    logger.debug("Failed to send daily summary: %s", e)
        except Exception as e:
            logger.error("Crypto daily end failed: %s", e)

    async def _check_day_rollover(self) -> None:
        """Check if we've crossed into a new Eastern day and handle rollover."""
        today = today_eastern().isoformat()
        if self._last_day is not None and today != self._last_day:
            await self._daily_end()
            await self._daily_start()

    # ── Shutdown ───────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Clean up all resources."""
        logger.info("Shutting down crypto trading bot...")
        self._running = False

        from halal_trader.web.app import app_state as _web_state

        _web_state["bot_running"] = False

        components: list[tuple[str, object | None]] = [
            ("monitor", self._monitor),
            ("news_reactor", getattr(self, "_news_reactor", None)),
            ("sentiment", self._sentiment_manager),
            ("notifier", self._notifier),
            ("websocket", self._ws),
        ]
        for name, component in components:
            if component is None:
                continue
            try:
                if hasattr(component, "stop"):
                    await component.stop()
                elif hasattr(component, "close"):
                    await component.close()
            except Exception as e:
                logger.warning("Failed to stop %s: %s", name, e)

        try:
            await self._binance.disconnect()
        except Exception as e:
            logger.warning("Failed to disconnect Binance: %s", e)

        await super().shutdown()
        logger.info("Crypto trading bot shut down")

    # ── Main Loop ──────────────────────────────────────────────

    async def run(self) -> None:
        """Start the crypto trading bot with continuous 1-minute cycles."""
        await self.initialize()
        self._running = True

        # Register signal handlers for clean shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

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
                cycle_start = time.monotonic()

                await self._check_day_rollover()

                # Check if self-review should trigger (consecutive losses)
                if self._self_review:
                    try:
                        if await self._self_review.should_trigger_review():
                            logger.info("Consecutive losses detected — triggering self-review")
                            await self._self_review.review(lookback_days=1)
                    except Exception as e:
                        logger.debug("Self-review trigger check failed: %s", e)

                # Run trading cycle with timeout (2x interval)
                cycle_timeout = interval * 2
                try:
                    await asyncio.wait_for(
                        self._cycle_service.run_cycle(),
                        timeout=cycle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Trading cycle timed out after %ds — skipping to next",
                        cycle_timeout,
                    )
                    await self._alerts.notify(
                        "cycle.timeout",
                        f"Crypto trading cycle exceeded {cycle_timeout}s and was cancelled.",
                    )

                from datetime import datetime, timezone

                from halal_trader.web.app import app_state as _web_state

                _web_state["last_cycle"] = datetime.now(timezone.utc).isoformat()

                # Sleep for remaining interval time
                elapsed = time.monotonic() - cycle_start
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

    def _handle_signal(self) -> None:
        """Signal handler: set running flag to false for clean shutdown."""
        logger.info("Received shutdown signal")
        self._running = False
