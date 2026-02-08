"""Crypto trading bot — composition root and 24/7 asyncio scheduler."""

import asyncio
import logging
from datetime import UTC, datetime

from halal_trader.agent.llm import create_llm
from halal_trader.config import get_settings
from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository

logger = logging.getLogger(__name__)


class CryptoTradingBot:
    """Composition root and scheduler — wires crypto components and runs 24/7."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._binance = BinanceClient(
            api_key=self.settings.binance_api_key,
            secret_key=self.settings.binance_secret_key,
            testnet=self.settings.binance_testnet,
        )
        self._ws: BinanceWSManager | None = None
        self._screener: CryptoHalalScreener | None = None
        self._cycle_service: CryptoCycleService | None = None
        self._portfolio: CryptoPortfolioTracker | None = None
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
        )

        # Portfolio tracker
        self._portfolio = CryptoPortfolioTracker(
            self._binance,
            repo,
            daily_loss_limit=self.settings.crypto_daily_loss_limit,
        )

        # Cycle service
        self._cycle_service = CryptoCycleService(
            broker=self._binance,
            screener=self._screener,
            strategy=strategy,
            executor=executor,
            portfolio=self._portfolio,
            ws_manager=self._ws,
            configured_pairs=self.settings.crypto_pairs,
        )

        logger.info("Crypto trading bot initialized successfully")

    async def shutdown(self) -> None:
        """Clean up all resources."""
        logger.info("Shutting down crypto trading bot...")
        self._running = False
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
            self._last_day = datetime.now(UTC).strftime("%Y-%m-%d")
            logger.info("Crypto daily start complete")
        except Exception as e:
            logger.error("Crypto daily start failed: %s", e)

    async def _daily_end(self) -> None:
        """Daily end routine: record P&L snapshot."""
        logger.info("=== CRYPTO DAILY END ===")
        try:
            summary = await self._portfolio.record_day_end()
            logger.info("Crypto daily summary: %s", summary)
        except Exception as e:
            logger.error("Crypto daily end failed: %s", e)

    async def _check_day_rollover(self) -> None:
        """Check if we've crossed into a new UTC day and handle rollover."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
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
