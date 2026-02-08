"""APScheduler trading loop — pre-market, intraday, and end-of-day jobs."""

import asyncio
import logging
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from halal_trader.agent.llm import create_llm
from halal_trader.agent.sentiment import SentimentAnalyzer
from halal_trader.agent.strategy import TradingStrategy
from halal_trader.config import get_settings
from halal_trader.db.models import init_db
from halal_trader.db.repository import Repository
from halal_trader.domain.ports import Broker, ComplianceScreener, LLMProvider, TradeRepository
from halal_trader.halal.cache import HalalScreener
from halal_trader.halal.zoya import ZoyaClient
from halal_trader.mcp.client import AlpacaMCPClient
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)


class TradingBot:
    """The main trading bot that orchestrates all components."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._mcp_client = AlpacaMCPClient()
        self.broker: Broker = self._mcp_client
        self.llm: LLMProvider | None = None
        self.repo: TradeRepository | None = None
        self.screener: ComplianceScreener | None = None
        self.strategy: TradingStrategy | None = None
        self.executor: TradeExecutor | None = None
        self.portfolio: PortfolioTracker | None = None
        self.sentiment: SentimentAnalyzer | None = None
        self.scheduler = AsyncIOScheduler()
        self._running = False

    async def initialize(self) -> None:
        """Set up all components."""
        logger.info("Initializing trading bot...")

        # Database
        db = await init_db(str(self.settings.db_path))
        self.repo = Repository(db)

        # Broker connection (Alpaca via MCP)
        await self._mcp_client.connect()

        # LLM
        self.llm = create_llm(self.settings)

        # Halal screener
        zoya = None
        if self.settings.zoya_api_key:
            zoya = ZoyaClient(
                api_key=self.settings.zoya_api_key,
                use_sandbox=self.settings.zoya_use_sandbox,
            )
        self.screener = HalalScreener(self.repo, zoya)

        # Strategy & executor
        self.strategy = TradingStrategy(self.llm, self.repo)
        self.executor = TradeExecutor(self.broker, self.repo)
        self.portfolio = PortfolioTracker(self.broker, self.repo)

        # Sentiment analyzer (supplementary — gracefully degrades if deps missing)
        self.sentiment = SentimentAnalyzer()

        logger.info("Trading bot initialized successfully")

    async def shutdown(self) -> None:
        """Clean up all resources."""
        logger.info("Shutting down trading bot...")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        await self._mcp_client.disconnect()
        self._running = False
        logger.info("Trading bot shut down")

    # ── Scheduled Jobs ──────────────────────────────────────────

    async def pre_market(self) -> None:
        """Pre-market job: refresh halal cache, record day start."""
        logger.info("=== PRE-MARKET ROUTINE ===")
        try:
            # Check if market will open today
            clock = await self.broker.get_clock()
            logger.info("Market clock: %s", clock)

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
            await self.screener.ensure_cache()

            # Record starting equity
            await self.portfolio.record_day_start()

            logger.info("Pre-market routine complete")
        except Exception as e:
            logger.error("Pre-market routine failed: %s", e)

    async def trading_cycle(self) -> None:
        """Intraday trading cycle — runs every N minutes during market hours."""
        logger.info("=== TRADING CYCLE ===")
        try:
            # Check market status
            clock = await self.broker.get_clock()
            is_open = clock.get("is_open", False) if isinstance(clock, dict) else False

            if not is_open:
                logger.info("Market is closed, skipping trading cycle")
                return

            # Check daily loss limit
            if await self.portfolio.should_halt_trading():
                logger.warning("Daily loss limit reached — halting trades")
                return

            # Gather data
            account = await self.broker.get_account_info()
            positions_raw = await self.broker.get_all_positions()
            positions = positions_raw if isinstance(positions_raw, list) else []

            halal_symbols = await self.screener.get_halal_symbols()
            if not halal_symbols:
                logger.warning("No halal symbols available, skipping cycle")
                return

            # Get market data for halal symbols (batch in groups)
            snapshots: dict[str, Any] = {}
            bars: dict[str, Any] = {}
            for sym in halal_symbols[:20]:  # Cap to avoid rate limits
                try:
                    snap = await self.broker.get_stock_snapshot(sym)
                    snapshots[sym] = snap
                except Exception as e:
                    logger.debug("Failed to get snapshot for %s: %s", sym, e)
                try:
                    bar = await self.broker.get_stock_bars(sym, days=5, timeframe="1Day")
                    bars[sym] = bar
                except Exception as e:
                    logger.debug("Failed to get bars for %s: %s", sym, e)

            today_pnl = await self.portfolio.get_current_pnl()

            # Run sentiment analysis (supplementary signal)
            sentiment_text = "Sentiment data: not available"
            if self.sentiment:
                try:
                    sentiment_scores = await self.sentiment.analyze_batch(halal_symbols[:10])
                    sentiment_text = self.sentiment.format_for_prompt(sentiment_scores)
                except Exception as e:
                    logger.debug("Sentiment analysis skipped: %s", e)

            # Run LLM analysis
            plan = await self.strategy.analyze(
                account=account if isinstance(account, dict) else {},
                positions=positions,
                halal_symbols=halal_symbols,
                snapshots=snapshots,
                bars=bars,
                today_pnl=today_pnl,
                sentiment_text=sentiment_text,
            )

            logger.info(
                "Trading plan: %s | %d buys, %d sells",
                plan.market_outlook[:80],
                len(plan.buys),
                len(plan.sells),
            )

            # Execute decisions
            if plan.decisions:
                results = await self.executor.execute_plan(plan)
                for r in results:
                    logger.info("Execution result: %s", r)
            else:
                logger.info("No trades to execute this cycle")

        except Exception as e:
            logger.error("Trading cycle failed: %s", e, exc_info=True)

    async def end_of_day(self) -> None:
        """End-of-day job: close all positions, record P&L."""
        logger.info("=== END OF DAY ROUTINE ===")
        try:
            # Close all positions
            close_result = await self.executor.close_all()
            logger.info("Close all positions result: %s", close_result)

            # Wait for positions to close
            await asyncio.sleep(5)

            # Record daily P&L
            summary = await self.portfolio.record_day_end()
            logger.info("Day summary: %s", summary)

        except Exception as e:
            logger.error("End of day routine failed: %s", e)

    # ── Main Loop ───────────────────────────────────────────────

    async def run(self) -> None:
        """Start the trading bot with scheduled jobs."""
        await self.initialize()
        self._running = True

        interval = self.settings.trading_interval_minutes

        # Schedule pre-market at 9:00 AM ET (Mon-Fri)
        self.scheduler.add_job(
            self.pre_market,
            CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone="US/Eastern"),
            id="pre_market",
            replace_existing=True,
        )

        # Schedule trading cycles every N minutes during market hours (9:30 - 15:45 ET)
        self.scheduler.add_job(
            self.trading_cycle,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute=f"*/{interval}",
                timezone="US/Eastern",
            ),
            id="trading_cycle",
            replace_existing=True,
        )

        # Schedule end-of-day at 3:50 PM ET (before 4:00 close)
        self.scheduler.add_job(
            self.end_of_day,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=50, timezone="US/Eastern"),
            id="end_of_day",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(
            "Trading bot started — interval: %d min, target: %.1f%%, loss limit: %.1f%%",
            interval,
            self.settings.daily_return_target * 100,
            self.settings.daily_loss_limit * 100,
        )

        # Keep running until interrupted
        try:
            while self._running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Bot interrupted")
        finally:
            await self.shutdown()

    async def run_once(self) -> None:
        """Run a single trading cycle (useful for testing)."""
        await self.initialize()
        try:
            await self.pre_market()
            await self.trading_cycle()
        finally:
            await self.shutdown()
