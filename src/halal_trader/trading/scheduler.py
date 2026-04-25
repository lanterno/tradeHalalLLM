"""APScheduler trading loop — pre-market, intraday, and end-of-day jobs."""

import asyncio
import fcntl
import logging
import os
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from halal_trader.core.llm import create_llm
from halal_trader.core.scheduler import BaseTradingBot
from halal_trader.domain.ports import Broker, ComplianceScreener
from halal_trader.halal.cache import HalalScreener
from halal_trader.halal.zoya import ZoyaClient
from halal_trader.market_hours import (
    MARKET_TZ,
    effective_close_time,
    is_trading_day,
    now_eastern,
    today_eastern,
)
from halal_trader.mcp.client import AlpacaMCPClient
from halal_trader.trading.cycle import TradingCycleService
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.portfolio import PortfolioTracker
from halal_trader.trading.sentiment import SentimentAnalyzer
from halal_trader.trading.strategy import TradingStrategy

logger = logging.getLogger(__name__)


_PID_FILE = Path("halal_trader.pid")


class TradingBot(BaseTradingBot):
    """Composition root and scheduler — wires components and runs cron jobs."""

    def __init__(self) -> None:
        super().__init__()
        self._mcp_client = AlpacaMCPClient()
        self.broker: Broker = self._mcp_client
        self.screener: ComplianceScreener | None = None
        self.executor: TradeExecutor | None = None
        self.portfolio: PortfolioTracker | None = None
        self.cycle_service: TradingCycleService | None = None
        self.scheduler = AsyncIOScheduler()
        self._lock_file: int | None = None

    async def _create_components(self) -> None:
        """Create stock-specific trading components."""
        logger.info("Initializing trading bot...")

        repo = self._repo
        assert repo is not None

        # Broker connection (Alpaca via MCP)
        await self._mcp_client.connect()

        # LLM
        llm = create_llm(self.settings)

        # Halal screener
        zoya = None
        if self.settings.zoya_api_key:
            zoya = ZoyaClient(
                api_key=self.settings.zoya_api_key,
                use_sandbox=self.settings.zoya_use_sandbox,
            )
        self.screener = HalalScreener(repo, zoya)

        # Strategy & executor
        strategy = TradingStrategy(
            llm,
            repo,
            llm_provider_name=self.settings.llm_provider.value,
            max_position_pct=self.settings.max_position_pct,
            daily_loss_limit=self.settings.daily_loss_limit,
            daily_return_target=self.settings.daily_return_target,
            max_simultaneous_positions=self.settings.max_simultaneous_positions,
        )
        self.executor = TradeExecutor(
            self.broker,
            repo,
            max_position_pct=self.settings.max_position_pct,
            max_simultaneous_positions=self.settings.max_simultaneous_positions,
        )
        self.portfolio = PortfolioTracker(
            self.broker,
            repo,
            daily_loss_limit=self.settings.daily_loss_limit,
        )

        # Sentiment analyzer (supplementary — gracefully degrades if deps missing)
        sentiment = SentimentAnalyzer()

        # Cycle service — owns the intraday trading logic
        self.cycle_service = TradingCycleService(
            broker=self.broker,
            screener=self.screener,
            strategy=strategy,
            executor=self.executor,
            portfolio=self.portfolio,
            sentiment=sentiment,
        )

        logger.info("Trading bot initialized successfully")

    def _get_cycle_service(self) -> TradingCycleService:
        _, _, _, cs = self._require_initialized()
        return cs

    async def _daily_start(self) -> None:
        await self.pre_market()

    async def _daily_end(self) -> None:
        await self.end_of_day()

    # ── PID Lock ─────────────────────────────────────────────────

    def _acquire_lock(self) -> None:
        """Acquire a PID file lock to prevent duplicate bot instances."""
        try:
            self._lock_file = os.open(str(_PID_FILE), os.O_CREAT | os.O_RDWR)
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(self._lock_file, str(os.getpid()).encode())
            os.ftruncate(self._lock_file, len(str(os.getpid())))
            logger.info("Acquired PID lock (pid=%d)", os.getpid())
        except OSError:
            try:
                with open(_PID_FILE) as f:
                    other_pid = f.read().strip()
            except Exception:
                other_pid = "unknown"
            raise RuntimeError(
                f"Another trading bot instance is already running (pid={other_pid}). "
                f"Remove {_PID_FILE} if the previous instance crashed."
            )

    def _release_lock(self) -> None:
        """Release the PID file lock."""
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                os.close(self._lock_file)
            except OSError:
                pass
            self._lock_file = None
            try:
                _PID_FILE.unlink(missing_ok=True)
            except OSError:
                pass
            logger.info("Released PID lock")

    # ── Shutdown ─────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Clean up all resources."""
        logger.info("Shutting down trading bot...")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        await self._mcp_client.disconnect()
        self._release_lock()
        await super().shutdown()
        logger.info("Trading bot shut down")

    def _require_initialized(
        self,
    ) -> tuple[ComplianceScreener, TradeExecutor, PortfolioTracker, TradingCycleService]:
        """Return initialized components; raise if the bot hasn't been initialized yet."""
        if (
            self.screener is None
            or self.executor is None
            or self.portfolio is None
            or self.cycle_service is None
        ):
            missing = [
                n
                for n, v in (
                    ("screener", self.screener),
                    ("executor", self.executor),
                    ("portfolio", self.portfolio),
                    ("cycle_service", self.cycle_service),
                )
                if v is None
            ]
            raise RuntimeError(
                "TradingBot.initialize() must be called before using: " + ", ".join(missing)
            )
        return (
            self.screener,
            self.executor,
            self.portfolio,
            self.cycle_service,
        )

    # ── Scheduled Jobs ──────────────────────────────────────────

    async def pre_market(self) -> None:
        """Pre-market job: refresh halal cache, record day start."""
        now = now_eastern()
        logger.info(
            "=== PRE-MARKET ROUTINE === (current time: %s ET)", now.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Skip entirely on market holidays (cron fires Mon-Fri regardless)
        if not is_trading_day(now.date()):
            logger.info("Today is not a trading day (holiday), skipping pre-market routine")
            return

        screener, _, portfolio, _ = self._require_initialized()
        try:
            # Check if market will open today
            clock = await self.broker.get_clock()
            logger.info(
                "Market clock: is_open=%s next_open='%s' next_close='%s'",
                clock.is_open,
                clock.next_open,
                clock.next_close,
            )

            close = effective_close_time(now.date())
            logger.info(
                "Market schedule: close at %s ET%s",
                close.strftime("%H:%M"),
                " (early close)" if close.hour < 16 else "",
            )

            # Log upcoming trading calendar
            try:
                calendar = await self.broker.get_calendar()
                if isinstance(calendar, list) and calendar:
                    next_days = calendar[:5]
                    logger.info(
                        "Upcoming trading days: %s",
                        [d.get("date", d) if isinstance(d, dict) else d for d in next_days],
                    )
            except Exception as e:
                logger.debug("Could not fetch market calendar: %s", e)

            # Refresh halal stock cache
            await screener.ensure_cache()

            # Record starting equity
            await portfolio.record_day_start()

            logger.info("Pre-market routine complete")
        except Exception as e:
            logger.error("Pre-market routine failed: %s", e)

    async def trading_cycle(self) -> None:
        """Intraday trading cycle — delegates to TradingCycleService."""
        _, _, _, cycle_service = self._require_initialized()
        await cycle_service.run_cycle()

    async def end_of_day(self) -> None:
        """End-of-day job: close all positions, record P&L."""
        logger.info("=== END OF DAY ROUTINE ===")
        _, executor, portfolio, _ = self._require_initialized()
        try:
            # Close all positions
            close_result = await executor.close_all()
            logger.info("Close all positions result: %s", close_result)

            # Wait for positions to close
            await asyncio.sleep(5)

            # Record daily P&L
            summary = await portfolio.record_day_end()
            logger.info("Day summary: %s", summary)

        except Exception as e:
            logger.error("End of day routine failed: %s", e)

    async def _early_close_eod(self) -> None:
        """End-of-day for early-close days (market closes at 1:00 PM ET).

        Fires at 12:50 PM ET every weekday, but only runs the EOD logic
        when today is actually an early-close day.
        """
        today = today_eastern()
        if effective_close_time(today).hour >= 16:
            return
        logger.info("=== EARLY CLOSE END OF DAY ROUTINE === (market closes at 1:00 PM ET)")
        await self.end_of_day()

    # ── Main Loop ───────────────────────────────────────────────

    async def run(self) -> None:
        """Start the trading bot with scheduled jobs."""
        self._acquire_lock()
        await self.initialize()
        try:
            self._running = True

            interval = self.settings.trading_interval_minutes

            # Schedule pre-market at 9:00 AM ET (Mon-Fri).
            # The job itself checks is_trading_day() and skips on holidays.
            # Allow up to 30 min grace period so the job still runs if the system
            # wakes from sleep after 09:00 ET (e.g. laptop lid open at 09:25).
            self.scheduler.add_job(
                self.pre_market,
                CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=MARKET_TZ),
                id="pre_market",
                replace_existing=True,
                misfire_grace_time=1800,
            )

            # Schedule trading cycles every N minutes during market hours (9:30 - 15:45 ET).
            # The cycle's run_cycle() performs its own is_market_open_local() check,
            # so holidays and early-close days are handled even though cron fires.
            # Use hour 10-15 for the bulk, plus a separate job for the 9:30-9:45 window.
            self.scheduler.add_job(
                self.trading_cycle,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour="10-15",
                    minute=f"*/{interval}",
                    timezone=MARKET_TZ,
                ),
                id="trading_cycle",
                replace_existing=True,
                misfire_grace_time=900,
            )
            # Cover the first 30 minutes after market open (9:30, 9:45)
            self.scheduler.add_job(
                self.trading_cycle,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour=9,
                    minute="30,45",
                    timezone=MARKET_TZ,
                ),
                id="trading_cycle_open",
                replace_existing=True,
                misfire_grace_time=900,
            )

            # Schedule end-of-day at 3:50 PM ET (before regular 4:00 close).
            # On early-close days (1:00 PM close), the trading cycle's local
            # market-open check will already stop trades after 1:00 PM, and the
            # EOD job will still fire at 3:50 PM to reconcile P&L.
            # A separate early-close EOD job fires at 12:50 PM ET to close
            # positions before the 1:00 PM early close.
            self.scheduler.add_job(
                self.end_of_day,
                CronTrigger(day_of_week="mon-fri", hour=15, minute=50, timezone=MARKET_TZ),
                id="end_of_day",
                replace_existing=True,
            )
            self.scheduler.add_job(
                self._early_close_eod,
                CronTrigger(day_of_week="mon-fri", hour=12, minute=50, timezone=MARKET_TZ),
                id="early_close_eod",
                replace_existing=True,
            )

            self.scheduler.start()
            logger.info(
                "Trading bot started — interval: %d min, target: %.1f%%, loss limit: %.1f%%",
                interval,
                self.settings.daily_return_target * 100,
                self.settings.daily_loss_limit * 100,
            )

            # Run pre-market once at startup to ensure cache and equity are
            # initialized even if the scheduled job was missed (e.g. system sleep).
            try:
                await self.pre_market()
            except Exception as e:
                logger.warning("Startup pre-market failed (will retry at scheduled time): %s", e)

            # Keep running until interrupted
            while self._running:
                await asyncio.sleep(1)

        except KeyboardInterrupt, asyncio.CancelledError:
            logger.info("Bot interrupted")
        finally:
            await self.shutdown()
