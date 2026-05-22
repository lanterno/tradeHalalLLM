"""APScheduler trading loop — pre-market, intraday, and end-of-day jobs."""

import asyncio
import fcntl
import logging
import os
from pathlib import Path
from typing import Any

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
from halal_trader.trading.catalysts import StockCatalystFeed
from halal_trader.trading.cycle import TradingCycleService
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.portfolio import PortfolioTracker
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
        # Lazy-built in ``_create_components``; closed in ``shutdown``.
        self._stocks_news: Any | None = None
        # Stocks-side self-review — wired in ``_create_components`` and
        # called from ``end_of_day``. ``Any`` because the bot's typed
        # ``self_review`` slot on ``TradingCycleService`` is also ``Any``.
        self._self_review: Any | None = None

    async def _create_components(self) -> None:
        """Create stock-specific trading components."""
        logger.info("Initializing trading bot...")

        # Live-mode token check: refuse to start without a daily confirmation.
        from halal_trader.core.safeguards import LiveModeChecker, check_live_mode_token

        check_live_mode_token(self.settings, market="stocks")
        self._live_mode_checker = LiveModeChecker(settings=self.settings, market="stocks")

        repo = self._repo
        assert repo is not None

        # Telegram notifier + rate-limited error sink (shared across cycle and EOD)
        from halal_trader.notifications.telegram import AlertSink, TelegramNotifier

        self._notifier = TelegramNotifier(
            bot_token=self.settings.telegram.bot_token,
            chat_id=self.settings.telegram.chat_id,
        )
        self._alerts = AlertSink(self._notifier)

        # Broker connection (Alpaca via MCP)
        await self._mcp_client.connect()

        # LLM
        llm = create_llm(self.settings)

        # Halal screener
        zoya = None
        if self.settings.zoya.api_key:
            zoya = ZoyaClient(
                api_key=self.settings.zoya.api_key,
                use_sandbox=self.settings.zoya.use_sandbox,
            )
        self.screener = HalalScreener(repo, zoya)

        # Optional adversarial co-bot for stocks (mirrors crypto). Off
        # by default; flipped on via LLM_ADVERSARIAL_ENABLED.
        attacker_llm = None
        if getattr(self.settings.llm, "adversarial_enabled", False):
            try:
                attacker_llm = create_llm(self.settings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("stocks adversarial LLM init failed: %s — disabling", exc)
                attacker_llm = None

        # Strategy & executor
        strategy = TradingStrategy(
            llm,
            repo,
            llm_provider_name=self.settings.llm.provider.value,
            max_position_pct=self.settings.stocks.max_position_pct,
            daily_loss_limit=self.settings.stocks.daily_loss_limit,
            daily_return_target=self.settings.stocks.daily_return_target,
            max_simultaneous_positions=self.settings.stocks.max_simultaneous_positions,
            attacker_llm=attacker_llm,
            # Wave H follow-up: stocks-side agentic mode. Default off;
            # ``agentic_hub`` stays None until the stocks bot grows its
            # own InsightsHub (RAG + regime memory are DB-backed, so
            # adding the hub is purely a composition-root change).
            # When enabled with no hub, the handlers degrade gracefully
            # to "not wired" messages.
            agentic_enabled=self.settings.stocks.agentic_enabled,
            agentic_max_turns=self.settings.stocks.agentic_max_turns,
            agentic_max_seconds=self.settings.stocks.agentic_max_seconds,
            agentic_hub=None,
        )
        self.executor = TradeExecutor(
            self.broker,
            repo,
            max_position_pct=self.settings.stocks.max_position_pct,
            max_simultaneous_positions=self.settings.stocks.max_simultaneous_positions,
        )
        self.portfolio = PortfolioTracker(
            self.broker,
            repo,
            daily_loss_limit=self.settings.stocks.daily_loss_limit,
        )

        # Catalyst feed — wires whichever sources are configured. The
        # FRED feed pulls scheduled CPI/FOMC/NFP/GDP release dates so
        # CatalystRiskPolicy can shrink position sizing in the 4h
        # window before each. EDGAR streams 8-K material events the
        # SEC publishes within minutes of the filing. Empty keys
        # disable each source cleanly.
        catalyst_sources: list[Any] = []
        if self.settings.fred.api_key:
            from halal_trader.trading.fred_catalysts import (
                FREDReleaseCalendarSource,
            )

            catalyst_sources.append(FREDReleaseCalendarSource(api_key=self.settings.fred.api_key))
        if self.settings.edgar.user_agent:
            from halal_trader.trading.edgar_catalysts import (
                EDGAREightKSource,
            )

            catalyst_sources.append(EDGAREightKSource(user_agent=self.settings.edgar.user_agent))

        # Options IV + Fed-speak are always-on (no key required).
        from halal_trader.trading.fed_speak_adapter import FedSpeakCatalystSource
        from halal_trader.trading.options_catalyst_adapter import (
            OptionsIVCatalystSource,
        )

        catalyst_sources.append(OptionsIVCatalystSource())
        catalyst_sources.append(FedSpeakCatalystSource())

        catalyst_feed = StockCatalystFeed(sources=catalyst_sources) if catalyst_sources else None

        # Stocks-side rolling-performance analytics — same surface as
        # the crypto cycle (``BuildPerformanceStage`` reads
        # ``compute_stats`` + ``format_for_prompt``). Built once here so
        # the cycle's stage list can stamp ``state.performance_text``
        # on each pass without a per-cycle constructor.
        from halal_trader.core.analytics import CrossAssetAnalytics
        from halal_trader.sentiment.stocks_news import (
            FinnhubNewsCollector,
            StockNewsCollector,
        )
        from halal_trader.trading.self_improve import StockTradeSelfReview

        repo = self._repo
        assert repo is not None  # populated by BaseTradingBot.initialize()
        bundle = self._bundle
        assert bundle is not None  # built alongside repo by initialize()
        stocks_analytics = CrossAssetAnalytics(repo, asset_class="stock")
        # Yahoo Finance — no API key, 15-min cache inside the collector.
        # Closed in :meth:`shutdown` so the underlying ``httpx`` client
        # doesn't leak past process exit.
        # Prefer Finnhub (free-tier 60 req/min, no IP rate-limit issues
        # like Yahoo's search endpoint) when a key is configured. Falls
        # back to Yahoo's `query2.finance.yahoo.com/v1/finance/search`
        # which has its own circuit breaker for the now-typical 429
        # storms. Operator drops `FINNHUB_API_KEY=…` in `.env` to switch.
        finnhub_key = getattr(self.settings, "finnhub", None)
        finnhub_key = getattr(finnhub_key, "api_key", "") if finnhub_key else ""
        if finnhub_key:
            self._stocks_news = FinnhubNewsCollector(api_key=finnhub_key)
            logger.info("StockNews backend: Finnhub")
        else:
            self._stocks_news = StockNewsCollector()
            logger.info("StockNews backend: Yahoo (fallback — FINNHUB_API_KEY not set)")

        # Stocks-side self-review — reviews closed Trade round-trips and
        # suggests bounded knob overrides (``max_position_pct``,
        # ``daily_loss_limit``). Mirrors crypto's wiring but with a
        # smaller knob menu because ``TradingStrategy`` doesn't carry
        # global SL/TP fallbacks. ``load_from_db`` restores any prior
        # adjustments so they survive a process restart.
        stocks_self_review = StockTradeSelfReview(
            llm,
            strategy_adjustments=bundle.strategy_adjustments,
            trades=bundle.trades,
            strategy=strategy,
        )
        await stocks_self_review.load_from_db()
        self._self_review = stocks_self_review

        # Cycle service — owns the intraday trading logic
        self.cycle_service = TradingCycleService(
            broker=self.broker,
            screener=self.screener,
            strategy=strategy,
            executor=self.executor,
            portfolio=self.portfolio,
            alerts=self._alerts,
            engine=self._engine,
            live_mode_checker=self._live_mode_checker,
            catalyst_feed=catalyst_feed,
            analytics=stocks_analytics,
            self_review=stocks_self_review,
            news_collector=self._stocks_news,
        )

        logger.info("Trading bot initialized successfully")

    def _get_cycle_service(self) -> TradingCycleService:
        _, _, _, cs = self._require_initialized()
        return cs

    async def _daily_start(self) -> None:
        await self.pre_market()

    async def _daily_end(self) -> None:
        await self.end_of_day()
        await self._prune_audit_log()

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
        if self._stocks_news is not None:
            try:
                await self._stocks_news.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stock news collector close failed: %s", exc)
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

        # Skip entirely on weekends and market holidays. APScheduler's
        # cron only fires Mon-Fri so weekends shouldn't normally hit
        # here, but the cycle can also be invoked manually (--once) or
        # during startup orchestration, so the check stays.
        if not is_trading_day(now.date()):
            day_kind = "weekend" if now.weekday() >= 5 else "market holiday"
            logger.info(
                "Today is not a trading day (%s), skipping pre-market routine",
                day_kind,
            )
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

            # Log upcoming trading calendar. The Alpaca MCP server wraps
            # its response as ``{"result": [...]}``; unwrap before
            # iterating. Decorative-only — failures are debug-logged.
            try:
                calendar = await self.broker.get_calendar()
                rows = (
                    calendar.get("result", [])
                    if isinstance(calendar, dict)
                    else (calendar if isinstance(calendar, list) else [])
                )
                if rows:
                    next_days = rows[:5]
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
            if self._alerts is not None:
                await self._alerts.notify(
                    "stock.pre_market.failed",
                    f"{type(e).__name__}: {e}",
                    market="stocks",
                    severity="error",
                )

    async def trading_cycle(self) -> None:
        """Intraday trading cycle — delegates to TradingCycleService.

        Before each cycle, check whether the self-review wants to fire
        an emergency review (3 consecutive losses or 10 exec failures).
        Mirrors ``crypto/scheduler.py:_run_cycle_loop`` lines 451-456.
        Failures degrade silently — the cycle must run regardless.
        """
        _, _, _, cycle_service = self._require_initialized()

        if self._self_review is not None:
            try:
                if await self._self_review.should_trigger_review():
                    logger.info("Consecutive stock losses detected — triggering self-review")
                    await self._self_review.review(lookback_days=1)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stocks self-review trigger check failed: %s", exc)

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

            # Enrich with market tag + date + LLM cost/calls. Mirrors
            # the crypto path in crypto/scheduler.py:_daily_end so the
            # richer Telegram summary fields fire.
            summary["market"] = "stocks"
            from datetime import UTC, datetime

            summary["date"] = datetime.now(UTC).date().isoformat()
            if self._engine is not None:
                try:
                    from sqlalchemy import text
                    from sqlalchemy.ext.asyncio import AsyncSession

                    async with AsyncSession(self._engine) as session:
                        row = (
                            await session.execute(
                                text(
                                    "SELECT COUNT(*)::int, "
                                    "COALESCE(SUM(cost_usd), 0)::float "
                                    "FROM llm_decisions "
                                    "WHERE timestamp::date = CURRENT_DATE"
                                )
                            )
                        ).first()
                        if row:
                            summary["llm_calls"] = int(row[0] or 0)
                            summary["llm_cost_usd"] = float(row[1] or 0.0)
                except Exception as exc:
                    logger.debug("Failed to enrich stocks daily summary with LLM cost: %s", exc)

            logger.info("Day summary: %s", summary)

            # End-of-day self-review — mirrors crypto/_daily_end. Pulls
            # the day's closed round-trips, asks the LLM what patterns
            # to learn, persists bounded knob overrides. Failures
            # (network, LLM down, no trades) degrade silently — the
            # rest of the daily-end routine must still run.
            if self._self_review:
                try:
                    review = await self._self_review.review(lookback_days=1)
                    if review.observations:
                        logger.info(
                            "Stocks self-review observations: %s",
                            "; ".join(review.observations[:3]),
                        )
                except Exception as exc:
                    logger.debug("Stocks self-review failed: %s", exc)

            # Send daily summary via Telegram — mirrors crypto/_daily_end.
            if self._notifier and self._notifier.enabled:
                try:
                    await self._notifier.notify_daily_summary(summary or {})
                except Exception as exc:
                    logger.debug("Failed to send stocks daily summary: %s", exc)

        except Exception as e:
            logger.error("End of day routine failed: %s", e)
            if self._alerts is not None:
                await self._alerts.notify(
                    "stock.end_of_day.failed",
                    f"{type(e).__name__}: {e}",
                    market="stocks",
                    severity="error",
                )

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

    async def _heartbeat(self) -> None:
        """Hourly heartbeat — proves the bot is alive even when no cycle
        is due. Fires regardless of market state so a silent log file is
        always an alarm, not just "the market is closed."
        """
        from halal_trader.market_hours import is_market_open_local

        now = now_eastern()
        try:
            clock = await self.broker.get_clock()
            next_open = clock.next_open
            next_close = clock.next_close
        except Exception:
            next_open = None
            next_close = None
        logger.info(
            "Stocks bot heartbeat — market_open=%s next_open=%s next_close=%s (now: %s ET)",
            is_market_open_local(),
            next_open or "unknown",
            next_close or "unknown",
            now.strftime("%Y-%m-%d %H:%M"),
        )

    # ── Main Loop ───────────────────────────────────────────────

    async def run(self) -> None:
        """Start the trading bot with scheduled jobs."""
        self._acquire_lock()
        await self.initialize()
        try:
            self._running = True

            interval = self.settings.stocks.trading_interval_minutes

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
                misfire_grace_time=1800,
                coalesce=True,
            )
            self.scheduler.add_job(
                self._early_close_eod,
                CronTrigger(day_of_week="mon-fri", hour=12, minute=50, timezone=MARKET_TZ),
                id="early_close_eod",
                replace_existing=True,
                misfire_grace_time=1800,
                coalesce=True,
            )

            # Idle-period heartbeat — proves the bot is alive on
            # weekends, holidays, and outside trading hours. Without
            # this, the stocks-bot container is silent for ~63h from
            # Friday 4pm ET to Monday 9am ET — indistinguishable from
            # a hung process.
            self.scheduler.add_job(
                self._heartbeat,
                CronTrigger(minute=0, timezone=MARKET_TZ),  # top of every hour
                id="heartbeat",
                replace_existing=True,
                misfire_grace_time=900,
                coalesce=True,
            )

            self.scheduler.start()
            logger.info(
                "Trading bot started — interval: %d min, target: %.1f%%, loss limit: %.1f%%",
                interval,
                self.settings.stocks.daily_return_target * 100,
                self.settings.stocks.daily_loss_limit * 100,
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
