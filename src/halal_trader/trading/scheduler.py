"""APScheduler trading loop — pre-market, intraday, and end-of-day jobs."""

import asyncio
import fcntl
import html
import logging
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from halal_trader.config import get_settings
from halal_trader.core import events
from halal_trader.core.heartbeat import (
    CORE_TRADE,
    DAILY_JOBS,
    MARKET_SNAPSHOT,
    RECOMMENDATION,
    RESEARCH,
    STOCK_CYCLE,
    STOCK_EOD,
    STOCK_LEDGER,
    STOCK_MONITOR,
    STOCK_PROCESS,
    WEEKLY_DIGEST,
    Beat,
    beat,
    read_beats,
    scheduled_at,
)
from halal_trader.core.llm import create_llm
from halal_trader.core.observability import job_context
from halal_trader.db.models import init_db
from halal_trader.db.repos import RepoBundle
from halal_trader.db.repository import Repository
from halal_trader.domain.ports import Broker, ComplianceScreener
from halal_trader.domain.status import EntryType
from halal_trader.halal.cache import HalalScreener
from halal_trader.market_hours import (
    MARKET_TZ,
    effective_close_time,
    is_trading_day,
    now_eastern,
    today_eastern,
)
from halal_trader.mcp.client import AlpacaMCPClient
from halal_trader.notifications.telegram import AlertSink, TelegramNotifier
from halal_trader.trading.catalysts import StockCatalystFeed
from halal_trader.trading.cycle import TradingCycleService
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.portfolio import PortfolioTracker
from halal_trader.trading.strategy import TradingStrategy
from halal_trader.trading.timeframes import StockTimeframeAnalyzer

logger = logging.getLogger(__name__)

# pg advisory-lock key for "one stock bot per database" (ASCII 'HALALSTK').
_TRADING_LOCK_KEY = 0x48414C414C53544B


_PID_FILE = Path("halal_trader.pid")

_RECOMMEND_RETRY_AFTER = timedelta(minutes=15)
# The core trades with market orders; a catch-up run after a restart must
# leave them time to fill before the close.
_CORE_CATCH_UP_CUTOFF = timedelta(minutes=5)
# Daily jobs older than this are not caught up: their moment has passed.
_CATCH_UP_HORIZON = timedelta(days=4)


def plan_catch_up(
    now: datetime,
    beats: dict[str, Beat],
    *,
    core_due: bool,
) -> list[tuple[str, date]]:
    """The daily jobs a bot starting at ``now`` missed and may still run.

    The scheduler's job store is in memory, so a restart near a job's time
    silently skipped it (~30 restarts in 4.5 days, 2026-10). Each job's last
    success is its heartbeat; each job has its own "too late" rule:

    * ``recommend`` (09:05) and ``end_of_day`` (15:50, 12:50 on early-close
      days) -- today's run, and only while the session is still open: an
      end-of-day flatten after the close would queue market orders for the
      next open.
    * ``core_trade`` (15:40) -- today's, only if ``core_due`` (enabled, keyed
      and not yet run today) and at least 5 minutes before the close, so its
      market orders can fill.
    * ``sync_broker_ledger`` (16:30) and ``research_daily`` (20:30) -- the
      newest missed trading day within a few days, run for THAT day.

    Returns ``(job, day)`` pairs, in the order to run them.
    """
    et = now.astimezone(MARKET_TZ)
    today = et.date()
    plan: list[tuple[str, date]] = []
    trading_today = is_trading_day(today)
    close = datetime.combine(today, effective_close_time(today), MARKET_TZ)

    def missed(component: str, at: datetime) -> bool:
        b = beats.get(component)
        return b is None or b.beat_at < at

    if trading_today and now < close:
        rec_at = scheduled_at(DAILY_JOBS[RECOMMENDATION], today)
        if now >= rec_at and missed(RECOMMENDATION, rec_at):
            plan.append(("recommend", today))
        core_at = scheduled_at(DAILY_JOBS[CORE_TRADE], today)  # 12:40 on a 13:00 close
        if core_due and core_at <= now < close - _CORE_CATCH_UP_CUTOFF:
            plan.append(("core_trade", today))
        eod_at = scheduled_at(DAILY_JOBS[STOCK_EOD], today)
        if now >= eod_at and missed(STOCK_EOD, eod_at):
            plan.append(("end_of_day", today))

    for component, job in ((STOCK_LEDGER, "sync_broker_ledger"), (RESEARCH, "research_daily")):
        day = today
        while today - day <= _CATCH_UP_HORIZON:
            if is_trading_day(day):
                at = scheduled_at(DAILY_JOBS[component], day)
                if at <= now:
                    if missed(component, at):
                        plan.append((job, day))
                    break
            day -= timedelta(days=1)
    return plan


@dataclass(frozen=True, slots=True)
class Skipped:
    """A job's run that had nothing to do (not a trading day, not its run today)."""

    reason: str


# What a job's body returns: its heartbeat's detail when it finished, or Skipped.
JobOutcome = dict[str, Any] | Skipped | None

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _off_schedule(component: str, scheduled: time | None) -> Skipped | None:
    """Skipped when ``scheduled`` (the cron time that fired) is not today's run time
    of ``component``: on a 13:00-close day only the early time runs, else only the
    regular one. A catch-up run passes no time and always runs."""
    if scheduled is None:
        return None
    due = scheduled_at(DAILY_JOBS[component], today_eastern()).time()
    return None if scheduled == due else Skipped(f"today's run is at {due:%H:%M}")


class TradingBot:
    """Composition root and scheduler — wires components and runs cron jobs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._running = False
        self._engine: AsyncEngine | None = None
        self._repo: Repository | None = None
        self._bundle: RepoBundle | None = None
        self._broker_client = AlpacaMCPClient()
        self.broker: Broker = self._broker_client
        self.screener: ComplianceScreener | None = None
        self.executor: TradeExecutor | None = None
        self.portfolio: PortfolioTracker | None = None
        self.cycle_service: TradingCycleService | None = None
        # Advisory daily "stock of the day" engine — built in
        # ``_create_components``, run by the pre-market ``recommend`` job.
        # Never trades.
        self._recommendation: Any | None = None
        # Defaults for every job: a job delayed past its time (a busy event
        # loop) still runs once rather than being dropped or run twice. Each
        # job may widen its own grace. A restart skips jobs regardless -- the
        # job store is in memory -- which _catch_up_missed_jobs handles.
        self.scheduler = AsyncIOScheduler(
            job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300}
        )
        self._lock_file: int | None = None
        # The session holding the database trading lock (_acquire_trading_lock).
        self._trading_lock_conn: AsyncConnection | None = None
        # Lazy-built in ``_create_components``; closed in ``shutdown``.
        self._stocks_news: Any | None = None
        # Stocks-side self-review — wired in ``_create_components`` and
        # called from ``end_of_day``. ``Any`` because the bot's typed
        # ``self_review`` slot on ``TradingCycleService`` is also ``Any``.
        self._self_review: Any | None = None
        # News-momentum reactor + background task — wired in
        # ``_create_components`` (when FINNHUB_API_KEY is set), spawned
        # in ``run()``, cancelled in ``shutdown()``. The "fast in" half of
        # the fast-in/slow-out strategy.
        self._news_reactor: Any | None = None
        self._news_reactor_task: asyncio.Task[None] | None = None
        # Intra-cycle SL/TP + trailing-stop monitor — runs between the
        # 15-min LLM cycles. Critical for reactor positions, which are
        # locked from LLM exits and rely on the monitor's wide trailing
        # stop / trend-break as their only rule-based exit.
        self._monitor: Any | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        # Telegram, and the rate-limited alert sink on it: always present (a
        # disabled notifier makes both no-ops), so no call site checks for None.
        self._notifier = TelegramNotifier(
            bot_token=self.settings.telegram.bot_token,
            chat_id=self.settings.telegram.chat_id,
        )
        self._alerts = AlertSink(self._notifier)
        # The live core's dated token, checked once at start (core_start_check).
        self._core_token_problem: str | None = "the live token was not checked at start"

    async def initialize(self) -> None:
        """Set up the database, then build every trading component."""
        engine = await init_db(self.settings.database_url)
        self._engine = engine
        self._repo = Repository(engine)
        # Typed per-table bundle (the repository's own) -- components can take
        # narrow protocols instead of the full ``Repository``.
        self._bundle = self._repo.bundle
        await self._create_components()

    async def _create_components(self) -> None:
        """Create the trading components (broker, strategy, executor, etc.)."""
        logger.info("Initializing trading bot...")

        # Live-mode token check: refuse to start without a daily confirmation.
        from halal_trader.core.safeguards import LiveModeChecker, check_live_mode_token

        check_live_mode_token(self.settings, market="stocks")
        self._live_mode_checker = LiveModeChecker(settings=self.settings, market="stocks")

        repo = self._repo
        assert repo is not None

        # One daily LLM spend total for every process on the key
        # (core/llm/spend.py); every GLM call below reports into it.
        from halal_trader.core.llm import spend

        spend.install(
            spend.SpendMeter(
                self._engine,
                consumer="stock",
                cap_usd=self.settings.llm.daily_usd_cap,
                enforce=True,
                alert=self._alerts.notify,
                monthly_cap_usd=spend.monthly_cap_for(
                    "stock",
                    live_usd=self.settings.llm.monthly_live_usd,
                    research_usd=self.settings.llm.monthly_research_usd,
                ),
            )
            if self._engine is not None
            else None
        )

        # Broker connection (the alpaca-mcp-server subprocess)
        await self._broker_client.connect()

        # LLM
        llm = create_llm(self.settings)

        # Halal screener
        # The strict in-house screen decides (fails closed when stale). Its
        # universe is rebuilt now, so the reactor's watchlist below and the
        # shadow (which reads the same cache) never start from an old one.
        self.screener = HalalScreener(repo, engine=self._engine)
        try:
            await self.screener.ensure_cache(force=True)
        except Exception as exc:  # noqa: BLE001 -- pre-market retries; the gate reads the screen
            logger.warning("halal universe refresh at startup failed: %r", exc)

        # Strategy & executor
        strategy = TradingStrategy(
            llm,
            repo,
            llm_provider_name="glm",
            max_position_pct=self.settings.stocks.max_position_pct,
            daily_loss_limit=self.settings.stocks.daily_loss_limit,
            daily_return_target=self.settings.stocks.daily_return_target,
            max_simultaneous_positions=self.settings.stocks.max_simultaneous_positions,
        )
        # Operator alert on strategy-LLM credit exhaustion (rate-limited
        # by the sink). The classifier already had this; the strategy
        # path stayed silent through the 2026-06 quota storm.
        strategy.attach_alert_sink(self._alerts)
        self.executor = TradeExecutor(
            self.broker,
            repo,
            max_position_pct=self.settings.stocks.max_position_pct,
            max_simultaneous_positions=self.settings.stocks.max_simultaneous_positions,
            recent_close_cooldown_minutes=(self.settings.stocks.recent_close_cooldown_minutes),
            stop_loss_reentry_cooldown_minutes=(
                self.settings.stocks.stop_loss_reentry_cooldown_minutes
            ),
            reactor_entry_size_fraction=self.settings.stocks.reactor_entry_size_fraction,
            reactor_entry_min_intraday_change_pct=(
                self.settings.stocks.reactor_entry_min_intraday_change_pct
            ),
            reactor_trailing_stop_distance_pct=(
                self.settings.stocks.reactor_trailing_stop_distance_pct
            ),
            screener=self.screener,
        )
        self.portfolio = PortfolioTracker(
            self.broker,
            repo,
            daily_loss_limit=self.settings.stocks.daily_loss_limit,
        )

        # Intra-cycle position monitor — enforces SL/TP + trailing stops
        # between the 15-min cycles. The reactor's slow-out positions
        # depend on its wide trailing stop as their only rule-based exit.
        from halal_trader.trading.monitor import StockPositionMonitor

        # Post-close fan-out: without this the stocks side never wrote the
        # rag_rationales store, so the retrieval corpus (setup→outcome
        # memory; halabot slice-2 grounding + query_rag) stayed EMPTY
        # (found 2026-07-03: store size 0 after weeks of stocks closes).
        # Recording only — the monitor wraps record_close in try/except,
        # so a recorder failure can never affect an exit.
        close_recorders = None
        if self._engine is not None:
            from halal_trader.core.llm.rag_db import DBRationaleStore
            from halal_trader.core.post_close import CloseRecorders

            close_recorders = CloseRecorders(rag_store=DBRationaleStore(self._engine))

        self._monitor = StockPositionMonitor(
            mcp=self.broker,
            repo=repo,
            close_recorders=close_recorders,
            check_interval=self.settings.stocks.monitor_interval_seconds,
            trailing_stop_activation_pct=self.settings.stocks.trailing_stop_activation_pct,
            trailing_stop_distance_pct=self.settings.stocks.trailing_stop_distance_pct,
            reactor_trailing_stop_distance_pct=(
                self.settings.stocks.reactor_trailing_stop_distance_pct
            ),
            trend_break_ma_period=self.settings.stocks.trend_break_ma_period,
            trend_break_timeframe=self.settings.stocks.trend_break_timeframe,
            on_tick=lambda detail: beat(self._engine, STOCK_MONITOR, detail),
            notifier=self._notifier,
        )

        # Catalyst feed — wires whichever sources are configured; the
        # cycle renders them into the prompt's RECENT CATALYSTS block.
        # FRED supplies scheduled CPI/FOMC/NFP/GDP release dates, EDGAR
        # the 8-K material events the SEC publishes within minutes of
        # the filing. Empty keys disable each source cleanly. (Nothing
        # sizes positions off catalysts.)
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

        # Fed-speak is always-on (no key required).
        # A Yahoo options-IV source lived here until Yahoo's auth change
        # 401'd every symbol; it was deleted on 2026-10-01 (the last tree
        # with it is 8b4be75: trading/options_iv.py and
        # options_catalyst_adapter.py). Start from there for a new IV provider.
        from halal_trader.trading.fed_speak_adapter import FedSpeakCatalystSource

        catalyst_sources.append(FedSpeakCatalystSource())

        catalyst_feed = StockCatalystFeed(sources=catalyst_sources) if catalyst_sources else None

        # Stocks-side rolling-performance analytics (``BuildPerformanceStage``
        # reads ``compute_stats`` + ``format_for_prompt``). Built once here so
        # the cycle's stage list can stamp ``state.performance_text``
        # on each pass without a per-cycle constructor.
        from halal_trader.portfolio.analytics import PerformanceAnalytics
        from halal_trader.sentiment.stocks_news import (
            FinnhubNewsCollector,
            StockNewsCollector,
        )
        from halal_trader.trading.self_improve import StockTradeSelfReview

        repo = self._repo
        assert repo is not None  # populated by initialize()
        bundle = self._bundle
        assert bundle is not None  # built alongside repo by initialize()
        stocks_analytics = PerformanceAnalytics(repo)
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
        # ``daily_loss_limit``). A small knob menu because
        # ``TradingStrategy`` doesn't carry global SL/TP fallbacks.
        # ``load_from_db`` restores any prior adjustments so they survive
        # a process restart.
        stocks_self_review = StockTradeSelfReview(
            llm,
            strategy_adjustments=bundle.strategy_adjustments,
            trades=bundle.trades,
            strategy=strategy,
        )
        await stocks_self_review.load_from_db()
        self._self_review = stocks_self_review

        # News-momentum reactor. Off when Finnhub key is unset — the
        # cron cycle keeps working. When enabled, every ~60s the reactor
        # polls Finnhub per halal symbol, classifies each new headline
        # through the dedicated classifier chain, and fires
        # ``_on_news_event`` on score >= threshold (0.85). The callback
        # places a half-size, price-confirmed paper entry (the "fast in"
        # side); the position
        # monitor then manages the slow-out exit.
        finnhub_cfg = getattr(self.settings, "finnhub", None)
        finnhub_key = getattr(finnhub_cfg, "api_key", "") if finnhub_cfg else ""
        stocks_cfg = self.settings.stocks
        use_alpaca = bool(self.settings.alpaca.api_key and self.settings.alpaca.secret_key)
        if finnhub_key or use_alpaca:
            from halal_trader.sentiment.stocks_events import (
                AlpacaNewsSource,
                FallbackNewsSource,
                FinnhubNewsSource,
                GPTHeadlineClassifier,
                StockNewsEventReactor,
            )

            # Reuse the strategy's LLM for classification — same key,
            # same provider, same rate budget. GLM-5.2 at ~$0.001
            # per headline stays well under the operator daily cap.
            try:
                watchlist = await self.screener.get_halal_symbols()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "halal symbols unavailable for reactor watchlist (%r) — reactor disabled",
                    exc,
                )
                watchlist = []
            if watchlist:
                # Dedicated classifier GLM stack (separate instances from the
                # strategy LLM) so backoff state and usage accounting don't
                # bleed between the two workloads. See create_classifier_llm.
                from halal_trader.core.llm import create_classifier_llm

                classifier = GPTHeadlineClassifier(
                    create_classifier_llm(self.settings),
                    alert_sink=self._alerts,
                    daily_classify_cap=self.settings.stocks.reactor_daily_classify_cap,
                )
                # Gate the reactor's sweep on the kill-switch so it stops
                # burning classifier/Finnhub calls while halted (entries are
                # blocked downstream regardless).
                from halal_trader.core.halt import is_halted

                # News feed: Alpaca (the feed event research is backtested on),
                # Finnhub for the trading watchlist if Alpaca fails.
                source: Any
                if use_alpaca:
                    from halal_trader.data.alpaca_market import AlpacaMarketData

                    source = AlpacaNewsSource(AlpacaMarketData.from_settings(self.settings))
                    if finnhub_key:
                        source = FallbackNewsSource(
                            source, FinnhubNewsSource(finnhub_key), watchlist
                        )
                else:
                    source = FinnhubNewsSource(finnhub_key)
                recorder = None
                observe = None
                if self._engine is not None:
                    from halal_trader.events.store import EventRecorder

                    recorder = EventRecorder(self._engine)
                    if stocks_cfg.reactor_observe_size > 0:
                        observe = self._observe_symbols
                self._news_reactor = StockNewsEventReactor(
                    api_key=finnhub_key,
                    symbols=watchlist,
                    classifier=classifier,
                    state_path=self.settings.resolve_data_dir() / "reactor_state.json",
                    halt_check=lambda: is_halted(self._engine),
                    source=source,
                    recorder=recorder,
                    observe_provider=observe,
                    max_headline_age_s=stocks_cfg.reactor_max_headline_age_s,
                    observe_daily_classify_cap=stocks_cfg.reactor_observe_daily_classify_cap,
                )
                self._news_reactor.on_event(self._on_news_event)
                logger.info(
                    "StockNewsEventReactor wired (source=%s, observe=%d, max_age=%.0fs)",
                    source.name,
                    stocks_cfg.reactor_observe_size if observe else 0,
                    stocks_cfg.reactor_max_headline_age_s,
                )
                logger.info(
                    "StockNewsEventReactor wired (%d symbols, threshold=%.2f, "
                    "daily_classify_cap=%d, entries of %.0f%% of cap)",
                    len(watchlist),
                    StockNewsEventReactor._DEFAULT_SCORE_THRESHOLD,
                    self.settings.stocks.reactor_daily_classify_cap,
                    self.settings.stocks.reactor_entry_size_fraction * 100,
                )

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
            # Multi-timeframe trend alignment (1H/1D/1W via Alpaca) — the strongest
            # signal in halabot's per-source attribution. Uses the shared
            # signals.timeframes math; errors degrade to an empty block (cycle-safe).
            timeframe_analyzer=StockTimeframeAnalyzer(self.broker),
            # Telegram alert per fill. The cycle always supported it; the
            # composition root never passed the notifier.
            notifier=self._notifier,
        )

        # Advisory daily halal recommendation engine (never trades).
        from halal_trader.recommendation.engine import DailyRecommendationEngine

        self._recommendation = DailyRecommendationEngine(
            broker=self.broker, repo=self._repo, settings=self.settings, engine=self._engine
        )

        logger.info("Trading bot initialized successfully")

    async def recommend(self, *, is_retry: bool = False) -> None:
        """Pre-market job: generate the advisory halal stock-of-the-day.

        Best-effort and non-fatal — a failure here must never affect trading.
        A failed run is retried once, 15 minutes later (on 2026-10-05 one
        transient failure left the day without a pick); only a failed retry
        alerts.
        """
        done = await self._job(
            "recommend",
            self._recommend,
            alert="recommendation.failed" if is_retry else None,
            beat_as=RECOMMENDATION,
        )
        if not done and not is_retry and self.scheduler.running:
            from apscheduler.triggers.date import DateTrigger

            self.scheduler.add_job(
                self.recommend,
                DateTrigger(run_date=datetime.now(UTC) + _RECOMMEND_RETRY_AFTER),
                kwargs={"is_retry": True},
                id="daily_recommendation_retry",
                replace_existing=True,
                misfire_grace_time=900,
            )
            logger.info("Daily recommendation: retrying in %s", _RECOMMEND_RETRY_AFTER)

    async def _recommend(self) -> JobOutcome:
        if not is_trading_day(today_eastern()):
            return Skipped("not a trading day")
        if self._recommendation is None:
            return Skipped("no recommendation engine")
        rec = await self._recommendation.generate()
        logger.info(
            "Daily recommendation ready: %s (conviction %.2f)",
            rec.get("symbol"),
            float(rec.get("conviction") or 0.0),
        )
        # Label matured past picks with forward returns (honest scorecard).
        from halal_trader.recommendation.scorecard import backfill_outcomes

        res = await backfill_outcomes(self.broker, self._repo)
        if res.get("updated") or res.get("skipped"):
            logger.info(
                "Recommendation scorecard backfill: %d updated, %d scored, %d skipped",
                res.get("updated", 0),
                res.get("scored", 0),
                res.get("skipped", 0),
            )
        # Adaptive-conformal band maintenance: matured candidate outcomes nudge
        # the band multiplier; a Kupiec coverage-drift failure is alerted.
        from halal_trader.quant.conformal import update_band_conformal

        aci = await update_band_conformal(self._repo)
        if aci.get("drift"):
            logger.warning(
                "Band coverage drift: trailing breach rate fails Kupiec "
                "(p=%.4f, n=%d); ACI alpha now %.4f (z_eff %.3f)",
                aci.get("drift_p") or 0.0,
                aci.get("trailing_n", 0),
                aci.get("alpha") or 0.0,
                aci.get("effective_z") or 0.0,
                extra={"event": events.BAND_COVERAGE_DRIFT},
            )
            await self._alerts.notify(
                "band.coverage_drift",
                f"5d band coverage drifted (Kupiec p={aci.get('drift_p'):.4f}, "
                f"n={aci.get('trailing_n')}); ACI widened to z={aci.get('effective_z')}",
            )
        return {"symbol": rec.get("symbol")}

    def _get_cycle_service(self) -> TradingCycleService:
        _, _, _, cs = self._require_initialized()
        return cs

    async def _prune_audit_log(self) -> None:
        """Delete ``web_actions`` rows older than the retention window.

        Run by ``end_of_day``. A retention of ``0`` disables the prune.
        """
        retention = int(getattr(self.settings.web, "audit_retention_days", 0) or 0)
        if retention <= 0 or self._bundle is None:
            return
        try:
            deleted = await self._bundle.web_audit.delete_old_web_actions(
                older_than=timedelta(days=retention)
            )
            if deleted:
                logger.info("Pruned %d web_actions row(s) older than %d days", deleted, retention)
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_actions prune failed: %r", exc)

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
            ) from None

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
        # Cancel the reactor BEFORE the MCP disconnect so its in-flight
        # callback can't reference a torn-down broker.
        if self._news_reactor_task is not None and not self._news_reactor_task.done():
            if self._news_reactor is not None:
                await self._news_reactor.stop()
            self._news_reactor_task.cancel()
            try:
                await self._news_reactor_task
            except asyncio.CancelledError, Exception:  # noqa: BLE001
                pass
            self._news_reactor_task = None
        # Stop the position monitor before MCP teardown for the same
        # reason — an in-flight exit must not touch a dead broker.
        if self._monitor_task is not None and not self._monitor_task.done():
            if self._monitor is not None:
                await self._monitor.stop()
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError, Exception:  # noqa: BLE001
                pass
            self._monitor_task = None
        if self._stocks_news is not None:
            try:
                await self._stocks_news.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stock news collector close failed: %r", exc)
        await self._broker_client.disconnect()
        self._release_lock()
        await self._release_trading_lock()
        self._running = False
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
        logger.info("Trading bot shut down")

    async def _observe_symbols(self) -> list[str]:
        """The largest halal names of the newest in-house screen: the reactor
        scores and records their news for research, and never trades them."""
        from halal_trader.halal.strict import halal_universe

        if self._engine is None:
            return []
        _, symbols = await halal_universe(
            self._engine, today=today_eastern(), limit=self.settings.stocks.reactor_observe_size
        )
        return symbols

    async def _on_news_event(self, event: Any) -> None:
        """Reactor callback — the "fast in" half of the strategy.

        Logs the scored catalyst, then places a half-size paper BUY through
        ``executor.execute_reactor_entry`` — gated on market being open, the
        kill-switch being clear, and the executor's own price-confirmation +
        risk gates. A closed market or a halt falls back to observation-only.
        """
        cls = event.classification
        logger.info(
            "Stocks news-momentum event: [%s score=%.2f tag=%s] %s — %s",
            event.symbol,
            cls.score,
            cls.tag,
            event.title[:80],
            cls.rationale[:120],
        )

        result, status_note = await self._maybe_execute_reactor_entry(event)

        if self._notifier and self._notifier.enabled:
            try:
                # Headlines and LLM rationales are third-party / model text:
                # escape them so they can't inject Telegram HTML (links,
                # formatting) into an alert the operator acts on.
                esc = html.escape
                await self._notifier.send(
                    f"\U0001f4f0 <b>Stocks news-momentum event</b>\n"
                    f"<b>{esc(event.symbol)}</b> · score {cls.score:.2f} · {esc(str(cls.tag))}\n"
                    f"{esc(event.title)}\n"
                    f"<i>{esc(cls.rationale[:160])}</i>\n"
                    f"Source: {esc(str(event.source))}\n"
                    f"<i>{esc(status_note)}</i>"
                )
            except Exception:  # noqa: BLE001
                pass
        if result is not None:
            logger.info(
                "Reactor entry result: %s → %s%s",
                event.symbol,
                result.get("status"),
                f" ({result.get('reason')})" if result.get("reason") else "",
                extra={
                    "event": events.REACTOR_ENTRY,
                    "symbol": event.symbol,
                    "status": result.get("status"),
                    "entry_type": EntryType.REACTOR_MOMENTUM,
                },
            )

    async def _maybe_execute_reactor_entry(self, event: Any) -> tuple[dict[str, Any] | None, str]:
        """Decide whether to place a reactor entry and do so.

        Returns ``(result, status_note)`` where ``result`` is the
        executor's result dict (or None when no order was attempted) and
        ``status_note`` is a short human string for the Telegram card.
        """
        if self.executor is None:
            return None, "Observation only — executor not initialized"

        # Kill-switch: never open new risk while halted.
        from halal_trader.core.halt import is_halted

        try:
            if await is_halted(self._engine):
                return None, "Observation only — kill-switch engaged"
        except Exception as exc:  # noqa: BLE001
            logger.debug("reactor entry halt check failed: %r", exc)
            return None, "Observation only — halt state unknown"

        # Only trade a live session — the reactor polls around the clock.
        try:
            clock = await self.broker.get_clock()
            if not getattr(clock, "is_open", False):
                return None, "Observation only — market closed"
        except Exception as exc:  # noqa: BLE001
            logger.debug("reactor entry market-clock check failed: %r", exc)
            return None, "Observation only — market state unknown"

        # The same entry gates the scheduled cycle applies, which this path
        # used to skip ("fast in" was also "unguarded in"): the daily loss
        # limit and the risk engine's last verdict. Both fail CLOSED -- a
        # reactor entry is optional, so "can't tell" means "don't".
        if self.portfolio is not None:
            try:
                if await self.portfolio.should_halt_trading():
                    return None, "Observation only — daily loss limit reached"
            except Exception as exc:  # noqa: BLE001
                logger.warning("reactor entry loss-limit check failed: %r", exc)
                return None, "Observation only — daily P&L unknown"
        risk_halt = getattr(self.cycle_service, "last_risk_halt", None)
        if risk_halt:
            return None, f"Observation only — risk engine halt: {risk_halt}"

        # Positions feed the max-positions cap and the per-name/sector caps.
        # An empty list on a failed read used to make every cap see an
        # empty book; refuse instead.
        try:
            positions = await self.broker.get_all_positions()
        except Exception as exc:  # noqa: BLE001
            logger.warning("reactor entry positions read failed: %r", exc)
            return None, "Observation only — positions unknown"

        cls = event.classification
        try:
            result = await self.executor.execute_reactor_entry(
                event.symbol,
                score=cls.score,
                reasoning=f"{cls.tag}: {event.title[:120]} — {cls.rationale[:120]}",
                positions=positions,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("reactor entry execution failed for %s: %r", event.symbol, exc)
            return None, "Entry attempt errored — see logs"

        status = str(result.get("status", ""))
        if status in ("filled", "partially_filled"):
            qty = result.get("quantity", "?")
            note = f"✅ Reactor entry: BUY {qty} {event.symbol} (half-size, slow-out lockout)"
        elif status == "skipped":
            note = f"Skipped: {result.get('reason', 'gate declined')}"
        elif status == "rejected":
            note = f"Rejected: {result.get('reason', 'risk gate')}"
        else:
            note = f"Entry status: {status}"
        return result, note

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

    async def _job(
        self,
        name: str,
        body: Callable[[], Awaitable[JobOutcome]],
        *,
        alert: str | None,
        beat_as: str | None = None,
    ) -> bool:
        """Run one scheduled job the way every job runs.

        Under one ``job_id`` (core/observability.py) its start, end or skip is
        logged as a structured event. A failure is logged with its traceback
        and alerted as ``alert`` (None: logged only), and never escapes: a job
        must not take the bot down. ``beat_as`` is written only when the body
        finished, with what it returned as the beat's detail, so a crashed
        run leaves no beat and the watchdog sees it. Returns whether it finished.
        """
        with job_context(name):
            logger.info("job %s started", name, extra={"event": events.JOB_START, "job": name})
            try:
                outcome = await body()
            except Exception as exc:  # noqa: BLE001 -- see the docstring
                logger.exception(
                    "job %s failed", name, extra={"event": events.JOB_FAILED, "job": name}
                )
                if alert is not None:
                    await self._alerts.notify(alert, f"{name}: {exc!r}", severity="error")
                return False
            if isinstance(outcome, Skipped):
                logger.info(
                    "job %s skipped: %s",
                    name,
                    outcome.reason,
                    extra={"event": events.JOB_SKIPPED, "job": name},
                )
                return True
            if beat_as is not None and self._engine is not None:
                await beat(self._engine, beat_as, outcome)
            logger.info("job %s finished", name, extra={"event": events.JOB_COMPLETE, "job": name})
            return True

    async def pre_market(self) -> None:
        """Pre-market job: check the clock, refresh the halal cache, record day start."""
        await self._job("pre_market", self._pre_market, alert="stock.pre_market.failed")

    async def _pre_market(self) -> JobOutcome:
        now = now_eastern()
        if not is_trading_day(now.date()):
            return Skipped("weekend" if now.weekday() >= 5 else "market holiday")
        screener, _, portfolio, _ = self._require_initialized()
        clock = await self.broker.get_clock()
        close = effective_close_time(now.date())
        logger.info(
            "Market clock: is_open=%s next_open=%s next_close=%s; closes at %s ET%s",
            clock.is_open,
            clock.next_open,
            clock.next_close,
            close.strftime("%H:%M"),
            " (early close)" if close.hour < 16 else "",
        )
        await screener.ensure_cache()
        await portfolio.record_day_start()
        return None

    async def trading_cycle(self) -> None:
        """Intraday trading cycle — delegates to TradingCycleService.

        Before each cycle, check whether the self-review wants to fire
        an emergency review (3 consecutive losses or 10 exec failures).
        Failures degrade silently — the cycle must run regardless.
        """
        _, _, _, cycle_service = self._require_initialized()

        if self._self_review is not None:
            try:
                if await self._self_review.should_trigger_review():
                    logger.info("Consecutive stock losses detected — triggering self-review")
                    await self._self_review.review(lookback_days=1)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stocks self-review trigger check failed: %r", exc)

        await cycle_service.run_cycle()
        # The cycle's latest risk read rides along with its heartbeat: that is
        # how the dashboard (another process) gets heat and drawdown.
        risk = getattr(cycle_service, "last_risk_snapshot", None)
        await beat(self._engine, STOCK_CYCLE, {"risk": risk} if risk else None)

    async def sync_broker_ledger(self, *, day: date | None = None) -> None:
        """After-close job: copy Alpaca's record into the ledger and check ours.

        The broker's fills and equity are the books of truth
        (execution/ledger.py). The day's fills (``day``, default today; a
        catch-up run after a restart passes the day it missed) are reconciled
        against the fills this bot recorded; any difference alerts, because a
        fill the bot does not know about is a position it is not managing.
        """
        await self._job(
            "broker_ledger",
            lambda: self._sync_broker_ledger(day or today_eastern()),
            alert="ledger.sync_failed",
            beat_as=STOCK_LEDGER,
        )

    async def _sync_broker_ledger(self, day: date) -> JobOutcome:
        from halal_trader.execution.ledger import reconcile_fills, sync_accounts
        from halal_trader.portfolio.core_account import broker_accounts

        if self._engine is None:
            return Skipped("no database")
        await sync_accounts(self._engine, broker_accounts(self.settings))
        rec = await reconcile_fills(self._engine, day)
        if rec.clean:
            logger.info("broker ledger: %d fills on %s, books agree", rec.broker_fills, rec.day)
        else:
            lines = [
                f"{d.symbol} {d.side}: broker {d.broker_qty:g} vs recorded {d.recorded_qty:g}"
                for d in rec.drifts
            ]
            logger.warning("broker ledger drift on %s: %s", rec.day, "; ".join(lines))
            await self._alerts.notify("ledger.fill_drift", f"{rec.day}: " + "; ".join(lines))
        return {"broker_fills": rec.broker_fills}

    async def research_daily(self, *, day: date | None = None) -> None:
        """Evening job: top up bars, re-screen weekly, advance the forward books.

        Research only -- nothing here places an order. It runs after the
        extended session so the day's bars are final. ``day`` defaults to
        today; a catch-up run after a restart passes the trading day whose
        run was missed, so the books never advance into a session that has
        not happened. A step that failed inside a finished run is alerted
        too, and counted in the beat.
        """
        await self._job(
            "research_daily",
            lambda: self._research_daily(day or today_eastern()),
            alert="research.failed",
            beat_as=RESEARCH,
        )

    async def _research_daily(self, day: date) -> JobOutcome:
        from halal_trader.research.daily import run_research

        if self._engine is None:
            return Skipped("no database")
        run = await run_research(self._engine, self.settings, today=day)
        if run.errors:
            await self._alerts.notify("research.failed", "; ".join(run.errors))
        if run.core_ready_now:
            await self._alerts.notify(
                "core.ready",
                "The core portfolio passed its live-money gate (20+ trading days on paper, a "
                "monthly rebalance, tracking its forward book, no refusals). To go live: put the "
                "live account's keys in CORE_ALPACA_API_KEY/SECRET and set CORE_PAPER=false. "
                "Details: halal-trader core readiness.",
            )
        if run.zakat:
            await self._alerts.notify(
                "zakat.due",
                "Zakat assessed for this year's hawl (both of Dar al-Ifta's methods, the "
                "higher taken): "
                + ", ".join(f"{account} ${amount:,.2f}" for account, amount in run.zakat.items())
                + ". Details: halal-trader zakat assess.",
            )
        logger.info(
            "research run: books %s, screened %s, event labels %s",
            run.books,
            run.screened,
            run.event_labels,
        )
        return {
            "books": run.books,
            "screened": run.screened,
            "rescreen_for": run.rescreen_for,
            "event_labels": run.event_labels,
            "event_refresh": run.event_refresh,
            "errors": len(run.errors),
        }

    def _add_daily(
        self,
        method: Callable[..., Awaitable[Any]],
        component: str,
        job_id: str,
        *,
        grace_s: int,
    ) -> None:
        """Schedule a daily job at its DAILY_JOBS time, and at its 13:00-close
        time where it has one; each run is told which time fired (``scheduled``)
        so only today's runs (see _off_schedule)."""
        job = DAILY_JOBS[component]
        day_of_week = "mon-fri" if job.weekday is None else _WEEKDAYS[job.weekday]
        times = [(job_id, job.at)]
        if job.early_close_at is not None:
            times.append((f"{job_id}_early_close", job.early_close_at))
        for name, at in times:
            self.scheduler.add_job(
                method,
                CronTrigger(
                    day_of_week=day_of_week, hour=at.hour, minute=at.minute, timezone=MARKET_TZ
                ),
                kwargs={"scheduled": at} if job.early_close_at is not None else {},
                id=name,
                replace_existing=True,
                misfire_grace_time=grace_s,
            )

    def _schedule_cycles(self, interval: int) -> None:
        """The day-trader's cycles: every ``interval`` minutes 10:00-15:45, plus 9:30/9:45."""
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

    async def weekly_digest(self) -> None:
        """Send the week's summary to Telegram (notifications/digest.py)."""
        await self._job(
            "weekly_digest", self._weekly_digest, alert="digest.failed", beat_as=WEEKLY_DIGEST
        )

    async def _weekly_digest(self) -> JobOutcome:
        from halal_trader.notifications.digest import build

        if self._engine is None:
            return Skipped("no database")
        message = await build(self._engine, self.settings, today=today_eastern())
        if not await self._notifier.send(message):
            # Not sent is not done: no beat, so the watchdog reports it.
            raise RuntimeError("Telegram did not take the weekly digest")
        return None

    async def market_snapshot(self) -> None:
        """Each minute of the session: both accounts' values and the benchmarks'
        prices, for the dashboard's home page (portfolio/snapshots.py).

        Read-only. Each account is taken on its own, so one refusing does not
        blank the other; failures are logged, not alerted (account_watch
        already alerts on access, and a missed minute heals itself).
        """
        from halal_trader.execution.alpaca_broker import AlpacaRestBroker
        from halal_trader.portfolio import snapshots
        from halal_trader.portfolio.core_account import broker_accounts

        if self._engine is None:
            return
        taken = []
        for i, a in enumerate(broker_accounts(self.settings)):
            if not (a.api_key and a.secret_key):
                continue
            broker = AlpacaRestBroker(a.api_key, a.secret_key, paper=a.paper)
            try:
                await snapshots.snapshot_account(self._engine, a.name, broker)
                if i == 0:
                    await snapshots.snapshot_quotes(self._engine, broker)
                taken.append(a.name)
            except Exception as exc:  # noqa: BLE001 -- a missed minute heals itself
                logger.warning("market snapshot of %s failed: %r", a.name, exc)
            finally:
                await broker.disconnect()
        await beat(self._engine, MARKET_SNAPSHOT, {"accounts": taken})

    async def account_watch(self) -> list[str]:
        """Check every configured Alpaca account answers its keys; alert on any that don't.

        Returns the failures (for tests). Transport hiccups are retried by the
        broker; a refusal (401/403) is reported at once, since it never heals.
        """
        from halal_trader.execution.alpaca_broker import AlpacaRestBroker
        from halal_trader.portfolio.core_account import broker_accounts

        failures = []
        for a in broker_accounts(self.settings):
            if not (a.api_key and a.secret_key):
                continue
            broker = AlpacaRestBroker(a.api_key, a.secret_key, paper=a.paper)
            try:
                account = await broker.get_account_info()
                if account.status.upper() != "ACTIVE":
                    failures.append(f"{a.label}: account status {account.status}")
            except Exception as exc:  # noqa: BLE001 -- the point is to report it
                failures.append(f"{a.label}: {exc!r}"[:200])
            finally:
                await broker.disconnect()
        if failures:
            logger.error("account watch: %s", "; ".join(failures))
            await self._alerts.notify(
                "broker.access_failed",
                "Alpaca account check failed -- "
                + "; ".join(failures)
                + ". Regenerated keys revoke the old ones: update .env and recreate the bot.",
            )
        return failures

    async def core_trade(self, *, scheduled: time | None = None) -> None:
        """Trade the strict-halal core portfolio on its own account (portfolio/core_executor.py).

        Monthly rebalance on the first run of a month that trades, sells of
        screen failures otherwise; skipped on a closed market and when the
        core's keys are missing. The run itself (preflight, plan, orders,
        record) is core_executor.run, shared with `halal-trader core run`.

        With the kill-switch engaged it still runs, restricted to forced sales:
        a holding the screen no longer passes is sold even in an emergency
        stop (holding it is the thing the stop must not prolong), nothing is
        bought, the run is recorded, and the operator is told.

        Live money (CORE_PAPER=false) also needs the dated token, checked once
        when the bot starts (core_start_check); every other gate is checked on
        every run.
        """
        await self._job(
            "core_trade",
            lambda: self._core_trade(scheduled),
            alert="core.failed",
            beat_as=CORE_TRADE,
        )

    async def _core_trade(self, scheduled: time | None) -> JobOutcome:
        from halal_trader.execution.alpaca_broker import AlpacaRestBroker
        from halal_trader.portfolio import core_executor as ce

        core = self.settings.core
        if self._engine is None or not core.enabled:
            return Skipped("the core has no keys")
        if off := _off_schedule(CORE_TRADE, scheduled):
            return off
        if not core.paper and self._core_token_problem:
            logger.error(
                "core trade refused: live money without its token (%s)", self._core_token_problem
            )
            await self._alerts.notify(
                "core.refused", f"live core not armed: {self._core_token_problem}"
            )
            return Skipped("live money without its token")
        broker = AlpacaRestBroker(core.alpaca_api_key, core.alpaca_secret_key, paper=core.paper)
        try:
            outcome = await ce.run(
                self._engine,
                broker,
                self.settings,
                today=today_eastern(),
                execute_orders=True,
                check_token=False,  # checked at start: a dated token matches one day only
            )
        finally:
            await broker.disconnect()
        if outcome.market_closed:
            return Skipped("the market is closed")
        # The run finished (traded, held, halted or refused: the last two also
        # alert below). A crash leaves no beat, so the watchdog sees it.
        detail = {"account": outcome.account, "refused": bool(outcome.refused)}
        if outcome.refused:
            logger.error("core trade refused: %s", "; ".join(outcome.refused))
            await self._alerts.notify("core.refused", "core not run: " + "; ".join(outcome.refused))
            return detail
        plan = outcome.plan
        assert plan is not None
        refused = outcome.rejected
        if outcome.kill_switch:
            sold = [f"{r['s']} ${r['n']:,.2f}" for r in outcome.submitted]
            await self._alerts.notify(
                "core.halted_sells",
                "Kill-switch engaged: the core bought nothing and sold only what the screen "
                "no longer holds halal: " + (", ".join(sold) if sold else "nothing to sell") + ".",
            )
        if plan.halted or refused:
            await self._alerts.notify(
                "core.attention",
                (f"halted: {plan.halted}. " if plan.halted else "")
                + (f"{len(refused)} order(s) refused." if refused else ""),
            )
        logger.info(
            "core trade (%s): %s, %d order(s), %d refused%s",
            outcome.account,
            "monthly" if plan.monthly else "sells only",
            len(outcome.results),
            len(refused),
            "; " + "; ".join(plan.notes) if plan.notes else "",
        )
        return detail

    async def core_start_check(self) -> list[str]:
        """At bot start: the core's dated live token, then every other preflight gate.

        The token is remembered for the process (core_trade refuses live money
        without it); the other gates are only reported here, and enforced on
        every run, so a broker blip at start cannot disarm the core for good.
        Returns the problems found (for tests and the log).
        """
        from halal_trader.core.safeguards import (
            core_preflight,
            core_token_problem,
            day_trader_account,
        )
        from halal_trader.execution.alpaca_broker import AlpacaRestBroker

        core = self.settings.core
        self._core_token_problem = core_token_problem(self.settings)
        problems = [self._core_token_problem] if self._core_token_problem else []
        if self._engine is None or not core.enabled:
            return problems
        broker = AlpacaRestBroker(core.alpaca_api_key, core.alpaca_secret_key, paper=core.paper)
        try:
            problems += await core_preflight(
                self.settings,
                engine=self._engine,
                core=await broker.get_account_info(),
                day_trader=day_trader_account,
                today=today_eastern(),
                check_token=False,
            )
        except Exception as exc:  # noqa: BLE001 -- reported; every run re-checks
            problems.append(f"core preflight could not run: {exc!r}"[:300])
        finally:
            await broker.disconnect()
        if problems:
            logger.error("core start check: %s", "; ".join(problems))
            await self._alerts.notify("core.refused", "core at start: " + "; ".join(problems))
        return problems

    async def end_of_day(self, *, scheduled: time | None = None) -> None:
        """End-of-day job: close all positions, record the day's P&L, review, report.

        On a 13:00-close day it runs at 12:50 only (``scheduled`` names the cron
        time that fired; a catch-up passes none). The audit-log prune runs
        whatever happened.
        """
        await self._job(
            "end_of_day",
            lambda: self._end_of_day(scheduled),
            alert="stock.end_of_day.failed",
            beat_as=STOCK_EOD,
        )
        await self._prune_audit_log()

    async def _end_of_day(self, scheduled: time | None) -> JobOutcome:
        if off := _off_schedule(STOCK_EOD, scheduled):
            return off
        _, executor, portfolio, _ = self._require_initialized()
        close_result = await executor.close_all()
        logger.info("Close all positions result: %s", close_result)
        await asyncio.sleep(5)  # let the closes fill before the day's P&L is read
        summary = await portfolio.record_day_end()
        summary["market"] = "Day-trader"  # the summary's header
        summary["date"] = today_eastern().isoformat()
        summary.update(await self._llm_spend_today())
        summary.update(await self._classifier_rollup())
        logger.info("Day summary: %s", summary)

        # End-of-day self-review: the day's closed round-trips, the LLM's
        # lessons, bounded knob overrides. It must not stop the report.
        if self._self_review:
            try:
                review = await self._self_review.review(lookback_days=1)
                if review.observations:
                    logger.info(
                        "Stocks self-review observations: %s",
                        "; ".join(review.observations[:3]),
                    )
            except Exception as exc:  # noqa: BLE001 -- see above
                logger.warning("Stocks self-review failed: %r", exc)
        await self._notifier.notify_daily_summary(summary)
        return {"date": summary["date"]}

    async def _llm_spend_today(self) -> dict[str, Any]:
        """Today's LLM calls and cost (New York day), for the daily summary."""
        from sqlalchemy import text

        from halal_trader.market_hours import trading_day_end_utc, trading_day_start_utc

        if self._engine is None:
            return {}
        today = today_eastern()
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT count(*)::int AS n, coalesce(sum(cost_usd), 0)::float AS usd "
                        "FROM llm_decisions WHERE timestamp >= :a AND timestamp < :b"
                    ),
                    {"a": trading_day_start_utc(today), "b": trading_day_end_utc(today)},
                )
            ).one()
        return {"llm_calls": row.n, "llm_cost_usd": row.usd}

    async def _classifier_rollup(self) -> dict[str, Any]:
        """The reactor classifier's cumulative calls and cost since the process
        started; a day ending in quota exhaustion is alerted as critical."""
        classifier = getattr(self._news_reactor, "classifier", None)
        if classifier is None or not hasattr(classifier, "get_telemetry"):
            return {}
        telem = classifier.get_telemetry()
        if telem.get("quota_exhausted"):
            await self._alerts.notify(
                "classifier.quota_exhausted.eod",
                "Reactor classifier ended the day in quota-exhausted state. Top up the LLM "
                "provider before the next session or the news-momentum reactor stays offline.",
                severity="critical",
            )
        return {
            "classifier_calls": telem.get("total_calls", 0),
            "classifier_calls_today": telem.get("calls_today", 0),
            "classifier_cost_usd": telem.get("cost_usd_total", 0.0),
            "classifier_quota_exhausted": telem.get("quota_exhausted", False),
            "classifier_by_provider": telem.get("calls_by_provider", {}),
        }

    # ── Main Loop ───────────────────────────────────────────────

    async def _acquire_trading_lock(self) -> None:
        """Hold a Postgres advisory lock for the life of this process.

        The PID file lock is per filesystem: a bot started on the host and
        the bot in the container never saw each other's lock, so two bots
        could trade one account at once (assessment infra-tooling#8). The
        database is the one thing every copy shares. Session-level advisory
        locks die with the connection, so a crashed bot cannot leave it stuck.

        The connection is AUTOCOMMIT. A plain connection auto-begins a
        transaction on its first statement, and this one is never committed:
        the bot held a transaction open for its whole life (seen idle in
        transaction for 10 h), which pins the database's xmin horizon so
        vacuum reclaims nothing anywhere. A session-level advisory lock does
        not need a transaction; it lives as long as the session.
        """
        from sqlalchemy import text

        if self._engine is None:
            return
        conn = await self._engine.connect()
        try:
            conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            got = (
                await conn.execute(
                    text("SELECT pg_try_advisory_lock(:k)"), {"k": _TRADING_LOCK_KEY}
                )
            ).scalar()
        except BaseException:
            await conn.close()
            raise
        if not got:
            await conn.close()
            raise RuntimeError(
                "another stock bot holds the trading lock on this database -- "
                "refusing to start a second one (stop the other first)"
            )
        self._trading_lock_conn = conn
        logger.info("Acquired the database trading lock")

    async def _holds_trading_lock(self) -> bool:
        """Does this process's lock session still exist and still hold the lock?

        False on any error: a dropped connection (a Postgres restart, a
        network cut) took the session -- and with it the lock -- away.
        """
        from sqlalchemy import text

        conn = getattr(self, "_trading_lock_conn", None)
        if conn is None:
            return False
        try:
            held = (
                await conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                        "AND granted AND pid = pg_backend_pid() "
                        "AND ((classid::bigint << 32) | objid::bigint) = :k)"
                    ),
                    {"k": _TRADING_LOCK_KEY},
                )
            ).scalar()
        except Exception as exc:  # noqa: BLE001 -- any failure means "not proven held"
            logger.warning("trading lock check failed: %r", exc)
            return False
        return bool(held)

    async def _ensure_trading_lock(self) -> None:
        """Keep the single-instance guarantee from lapsing silently.

        Called from the run loop. If the lock's session is gone, take the
        lock again on a fresh connection; if that fails -- another bot took
        it meanwhile, or the database is still down -- raise, so the process
        exits and docker restarts it into the normal startup path, which
        refuses to run beside another bot.
        """
        if self._engine is None or await self._holds_trading_lock():
            return
        logger.error("the database trading lock was lost -- re-acquiring it")
        stale = getattr(self, "_trading_lock_conn", None)
        self._trading_lock_conn = None
        if stale is not None:
            try:
                await stale.invalidate()
            except Exception as exc:  # noqa: BLE001 -- it is already broken
                logger.debug("dropping the lost lock connection failed: %r", exc)
        try:
            await self._acquire_trading_lock()
        except Exception as exc:
            try:
                await self._alerts.notify(
                    "stock.trading_lock_lost",
                    f"The stock bot lost its database trading lock and could not take it back "
                    f"({exc!r}); it is exiting so docker restarts it.",
                    severity="critical",
                )
            except Exception as alert_err:  # noqa: BLE001
                logger.warning("lock-loss alert failed: %r", alert_err)
            raise RuntimeError(f"trading lock lost and not re-acquired: {exc!r}") from exc
        logger.warning("re-acquired the database trading lock")

    async def _release_trading_lock(self) -> None:
        conn = getattr(self, "_trading_lock_conn", None)
        if conn is None:
            return
        self._trading_lock_conn = None
        from sqlalchemy import text

        try:
            # Unlock explicitly: close() only returns a pooled connection to
            # the pool, and the lock would ride along with it.
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _TRADING_LOCK_KEY})
            await conn.close()
        except Exception as exc:  # noqa: BLE001 -- drop the session; that frees the lock too
            logger.debug("trading lock release failed (%r); invalidating the connection", exc)
            await conn.invalidate()

    async def _core_due_today(self, today: date) -> bool:
        """Is the core enabled, keyed, and without a run (or an order) today?"""
        from sqlalchemy import text

        core = self.settings.core
        if self._engine is None or not (core.enabled):
            return False
        async with self._engine.connect() as conn:
            ran = (
                await conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM core_runs WHERE run_on = :d AND executed) "
                        "OR EXISTS (SELECT 1 FROM core_orders WHERE "
                        "(submitted_at AT TIME ZONE 'America/New_York')::date = :d)"
                    ),
                    {"d": today},
                )
            ).scalar()
        return not ran

    async def _catch_up_missed_jobs(self, now: datetime | None = None) -> list[tuple[str, date]]:
        """Queue every daily job this (re)start missed and may still run (plan_catch_up).

        Each runs as a one-off scheduler job, so startup is not held up and
        the scheduler's one-instance rule still applies. Returns the plan.
        """
        if self._engine is None:
            return []
        now = now or datetime.now(UTC)
        try:
            beats = await read_beats(self._engine)
            core_due = await self._core_due_today(now.astimezone(MARKET_TZ).date())
        except Exception as exc:  # noqa: BLE001 -- the scheduled runs still happen
            logger.warning("missed-job catch-up skipped: %r", exc)
            return []
        plan = plan_catch_up(now, beats, core_due=core_due)
        for job, day in plan:
            func = getattr(self, job)
            kwargs: dict[str, Any] = (
                {"day": day} if job in ("sync_broker_ledger", "research_daily") else {}
            )
            self.scheduler.add_job(
                func,
                "date",
                run_date=now,
                kwargs=kwargs,
                id=f"catch_up_{job}",
                replace_existing=True,
                misfire_grace_time=600,
            )
            logger.warning(
                "catching up %s for %s (its scheduled run was missed)",
                job,
                day,
                extra={"event": events.JOB_CATCH_UP, "job": job},
            )
        return plan

    async def run_once(self) -> None:
        """One pre-market check and one trading cycle, then exit.

        Never the end-of-day routine: that flattens every position on the
        account, so `halal-trader start --once` used as a smoke test would
        close the live bot's book. It also takes the single-instance lock, so
        it refuses to run beside a live bot rather than racing it.
        """
        self._acquire_lock()
        await self.initialize()
        await self._acquire_trading_lock()
        try:
            await self.pre_market()
            await self._get_cycle_service().run_cycle()
        finally:
            await self.shutdown()

    async def _supervise(
        self,
        name: str,
        run: Callable[[], Awaitable[None]],
        *,
        backoff_s: float = 5.0,
    ) -> None:
        """Run a long-lived component; restart it if it crashes or returns.

        The SL/TP monitor and the news reactor used to be bare create_task()s:
        an exception that escaped either one ended it silently while the bot
        kept trading -- with no stop-loss enforcement in the monitor's case.
        Cancellation (shutdown) passes through; anything else alerts and
        restarts after a short backoff.
        """
        while self._running:
            try:
                await run()
                if not self._running:
                    return
                reason = "returned unexpectedly"
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- the whole point is to survive it
                reason = f"crashed: {type(exc).__name__}: {exc}"
                logger.exception("%s %s -- restarting in %.0fs", name, reason, backoff_s)
            else:
                logger.error("%s %s -- restarting in %.0fs", name, reason, backoff_s)
            try:
                await self._alerts.notify(
                    f"supervisor.{name.replace(' ', '_')}", f"Stock bot {name} {reason}"
                )
            except Exception as alert_err:  # noqa: BLE001
                logger.warning("supervisor alert failed: %r", alert_err)
            await asyncio.sleep(backoff_s)

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Turn SIGTERM / SIGINT into a clean stop of the run loop.

        In docker this process is PID 1, and PID 1 ignores any signal it has
        no handler for: every `docker stop` / redeploy waited out the grace
        period and ended in SIGKILL, so shutdown() -- cancel the monitor and
        reactor, disconnect the broker, release the lock -- never ran.
        Clearing _running lets run()'s finally block do all of that.
        """

        def request_stop(sig: signal.Signals) -> None:
            logger.info("Received %s -- stopping after the current step", sig.name)
            self._running = False

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, request_stop, sig)
            except NotImplementedError, RuntimeError:  # non-main thread / platform without it
                logger.debug("signal handler for %s not installed", sig.name)

    async def run(self) -> None:
        """Start the trading bot with scheduled jobs."""
        from halal_trader.core.observability import set_service

        set_service("stock")  # tag this process's logs
        self._acquire_lock()
        await self.initialize()
        await self._acquire_trading_lock()
        try:
            self._running = True
            self._install_signal_handlers(asyncio.get_running_loop())

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
            self._schedule_cycles(interval)

            # The daily jobs, at the times DAILY_JOBS gives (core/heartbeat.py:
            # the watchdog judges them by the same table), each with its
            # 13:00-close time where it has one.
            # The advisory stock of the day (never trades), after the cache refresh.
            self._add_daily(self.recommend, RECOMMENDATION, "daily_recommendation", grace_s=3600)
            self._add_daily(self.end_of_day, STOCK_EOD, "end_of_day", grace_s=1800)
            self._add_daily(self.sync_broker_ledger, STOCK_LEDGER, "broker_ledger", grace_s=3600)
            self._add_daily(self.research_daily, RESEARCH, "research_daily", grace_s=3600)
            self._add_daily(self.weekly_digest, WEEKLY_DIGEST, "weekly_digest", grace_s=3600)
            if self.settings.core.enabled:
                # The core's gates, once at start: the live token is held for
                # the process, the rest are alerted now and re-checked per run.
                await self.core_start_check()
                self._add_daily(self.core_trade, CORE_TRADE, "core_trade", grace_s=600)

            # Both Alpaca accounts' keys still work: a revoked key otherwise
            # leaves a "healthy" bot that cannot trade (2026-10-04: regenerating
            # keys on the day-trader's account locked it out, unnoticed).
            # The home page's live figures: every minute from the pre-market
            # to an hour after the close, and once at start so a page opened
            # at the weekend still has the last values.
            self.scheduler.add_job(
                self.market_snapshot,
                CronTrigger(day_of_week="mon-fri", hour="9-16", minute="*", timezone=MARKET_TZ),
                id="market_snapshot",
                replace_existing=True,
                misfire_grace_time=50,
                next_run_time=datetime.now(UTC),
            )

            self.scheduler.add_job(
                self.account_watch,
                CronTrigger(minute="5,35", timezone=MARKET_TZ),
                id="account_watch",
                replace_existing=True,
                misfire_grace_time=900,
            )

            self.scheduler.start()
            # One line naming every job and its next run, so a job that should be
            # there (core_trade once the core has keys) can be checked from the logs.
            logger.info(
                "Scheduled jobs: %s",
                ", ".join(
                    f"{job.id} @ {job.next_run_time:%a %H:%M %Z}" if job.next_run_time else job.id
                    for job in self.scheduler.get_jobs()
                ),
            )
            logger.info(
                "Trading bot started — interval: %d min, target: %.1f%%, loss limit: %.1f%%",
                interval,
                self.settings.stocks.daily_return_target * 100,
                self.settings.stocks.daily_loss_limit * 100,
            )

            # Spawn the news-momentum reactor as a long-lived task so
            # callbacks fire on the event loop alongside the scheduler.
            # ``shutdown()`` cancels this cleanly. The reactor's own
            # ``enabled`` check short-circuits the loop when disabled.
            if self._news_reactor is not None and self._news_reactor.enabled:
                self._news_reactor_task = asyncio.create_task(
                    self._supervise("news reactor", self._news_reactor.run),
                    name="stocks-news-reactor",
                )

            # Spawn the intra-cycle position monitor (SL/TP + trailing
            # stops). Runs alongside the scheduler; cancelled in shutdown.
            if self._monitor is not None:
                self._monitor_task = asyncio.create_task(
                    self._supervise("position monitor", self._monitor.run),
                    name="stock-position-monitor",
                )
                logger.info(
                    "Stock position monitor spawned (check every %.0fs)",
                    self.settings.stocks.monitor_interval_seconds,
                )

            # Run pre-market once at startup to ensure cache and equity are
            # initialized even if the scheduled job was missed (e.g. system sleep).
            try:
                await self.pre_market()
            except Exception as e:
                logger.warning("Startup pre-market failed (will retry at scheduled time): %r", e)

            # Then any daily job this (re)start skipped and may still run.
            await self._catch_up_missed_jobs()

            # Keep running until interrupted
            # The run loop doubles as the process heartbeat: every 60 s it
            # records that the bot is alive, so a hung or dead process shows
            # up as a stale stock.process row in /api/health/bot. It also
            # checks the single-instance lock is still held (raises if it was
            # lost and cannot be taken back, so docker restarts the bot).
            loop = asyncio.get_running_loop()
            last_beat = float("-inf")
            while self._running:
                if loop.time() - last_beat >= 60:
                    await self._ensure_trading_lock()
                    await beat(
                        self._engine, STOCK_PROCESS, detail={"core": self.settings.core.enabled}
                    )
                    last_beat = loop.time()
                await asyncio.sleep(1)

        except KeyboardInterrupt, asyncio.CancelledError:
            logger.info("Bot interrupted")
        finally:
            await self.shutdown()
