"""Crypto trading bot — composition root and 24/7 asyncio scheduler."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any

from halal_trader.core.scheduler import BaseTradingBot
from halal_trader.crypto.cadence import select_interval
from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.monitor import PositionMonitor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.market_hours import today_eastern

logger = logging.getLogger(__name__)


class CryptoTradingBot(BaseTradingBot):
    """Composition root and scheduler — wires crypto components and runs 24/7."""

    def __init__(self) -> None:
        super().__init__()
        self._binance = BinanceClient(
            api_key=self.settings.binance.api_key,
            secret_key=self.settings.binance.secret_key,
            testnet=self.settings.binance.testnet,
            configured_pairs=self.settings.crypto.pairs,
        )
        self._ws: BinanceWSManager | None = None
        self._screener: CryptoHalalScreener | None = None
        self._cycle_service: CryptoCycleService | None = None
        self._portfolio: CryptoPortfolioTracker | None = None
        self._monitor: PositionMonitor | None = None
        self._sentiment_manager: Any = None
        self._self_review: Any = None
        self._notifier: Any = None
        self._news_reactor: Any = None
        self._last_day: str | None = None
        self._exiting_pairs: set[str] = set()
        self._reconcile_task: asyncio.Task[None] | None = None

    async def _create_components(self) -> None:
        """Build the full crypto wiring via :mod:`crypto.components`."""
        logger.info("Initializing crypto trading bot...")
        repo = self._repo
        assert repo is not None

        from halal_trader.crypto.components import build_components

        comps = await build_components(
            settings=self.settings,
            repo=repo,
            engine=self._engine,
            binance=self._binance,
            exiting_pairs=self._exiting_pairs,
        )

        # Hand long-lived components back to the scheduler.
        self._live_mode_checker = comps.live_mode_checker
        self._ws = comps.ws
        self._screener = comps.screener
        self._portfolio = comps.portfolio
        self._notifier = comps.notifier
        self._alerts = comps.alerts
        self._sentiment_manager = comps.sentiment_manager
        self._self_review = comps.self_review
        self._retrainer = comps.retrainer
        self._monitor = comps.monitor
        self._news_reactor = comps.news_reactor

        # Cycle service holds many of the optional components by reference.
        self._cycle_service = CryptoCycleService(
            broker=self._binance,
            screener=comps.screener,
            strategy=comps.strategy,
            executor=comps.executor,
            portfolio=comps.portfolio,
            ws_manager=comps.ws,
            configured_pairs=self.settings.crypto.pairs,
            analytics=comps.analytics,
            sentiment_manager=comps.sentiment_manager,
            timeframe_analyzer=comps.timeframe_analyzer,
            regime_detector=comps.regime_detector,
            ml_forecaster=comps.ml_forecaster,
            ml_anomaly_detector=comps.ml_anomaly,
            ml_signal_classifier=comps.ml_signal,
            self_review=comps.self_review,
            notifier=comps.notifier if comps.notifier.enabled else None,
            risk_engine=comps.risk_engine,
            news_feed=comps.news_feed,
            alerts=comps.alerts,
            engine=self._engine,
            live_mode_checker=comps.live_mode_checker,
            shadow_runner=comps.shadow_runner,
            whale_flow_source=comps.whale_flow_source,
            reddit_fetcher=comps.reddit_fetcher,
            hub=comps.hub,
            bus=comps.bus,
        )

        # Wave C: the bot's run() loop wires every long-lived task into
        # a single TaskSupervisor scope. Wire only the news-event
        # callback here; the actual poll loop registration happens in
        # run() so a supervisor crash propagates as expected.
        if self._news_reactor.enabled:
            self._news_reactor.on_event(self._on_news_event)

        # Wave A: build the bot's typed BotContext + populate the
        # runtime view the dashboard polls. Replaces the previous
        # write-to-app_state-dict pattern.
        from datetime import UTC, datetime

        from halal_trader.core.context import BotContext

        self._runtime.bot_running = True
        self._runtime.started_at = datetime.now(UTC)
        self._runtime.ws_manager = self._ws
        self._runtime.sentiment_manager = self._sentiment_manager
        self._runtime.crypto_broker = self._binance
        assert self._engine is not None  # populated by BaseTradingBot.initialize()
        self._ctx = BotContext(
            engine=self._engine,
            repo=repo,
            hub=comps.hub,
            analytics=comps.analytics,
            settings=self.settings,
            bus=comps.bus,
            runtime=self._runtime,
        )

        logger.info("Crypto trading bot initialized successfully")

    def _get_cycle_service(self) -> CryptoCycleService:
        if self._cycle_service is None:
            raise RuntimeError("CryptoTradingBot.initialize() must be called first")
        return self._cycle_service

    async def _reconcile_loop(self) -> None:
        """Run reconciliation every 5 minutes while the bot is alive."""
        from halal_trader.core.reconcile import reconcile_crypto

        interval = 300
        while self._running:
            try:
                if self._engine is not None:
                    await reconcile_crypto(
                        engine=self._engine,
                        broker=self._binance,
                        alerts=self._alerts,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Reconciliation pass failed: %s", e)

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def _on_news_event(self, event: Any) -> None:
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
            assert self._screener is not None and self._portfolio is not None
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
            assert self._portfolio is not None
            summary = await self._portfolio.record_day_end()

            # Enrich the summary with ops/spend stats from the DB so the
            # richer Telegram daily-summary template prints them. These
            # are best-effort — failures here shouldn't break the rest of
            # the daily-end routine.
            summary["market"] = "crypto"
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
                                    "SELECT COUNT(*)::int, COALESCE(SUM(cost_usd), 0)::float "
                                    "FROM llm_decisions WHERE timestamp::date = CURRENT_DATE"
                                )
                            )
                        ).first()
                        if row:
                            summary["llm_calls"] = int(row[0] or 0)
                            summary["llm_cost_usd"] = float(row[1] or 0.0)
                except Exception as exc:
                    logger.debug("Failed to enrich daily summary with LLM cost: %s", exc)

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

        await self._prune_audit_log()
        await self._nightly_prompt_evolve()

    async def _nightly_prompt_evolve(self) -> None:
        """Wave F: run one GA sweep over recent replay snapshots.

        Best-effort. Failures (no engine, no snapshots, GA crash) log at
        debug and never block the daily-end routine. Operator can also
        trigger this manually via ``halal-trader prompts evolve``.
        """
        if self._engine is None:
            return
        try:
            from halal_trader.core.llm.prompt_evo_runner import evolve_with_replay
            from halal_trader.crypto.prompt_fitness import replay_pnl_fitness
            from halal_trader.crypto.prompts import crypto_allele_pool

            result = await evolve_with_replay(
                engine=self._engine,
                name="crypto.strategy.system",
                pool=crypto_allele_pool(),
                evaluator=replay_pnl_fitness,
                generations=self.settings.crypto.prompt_evo_generations,
                population_size=self.settings.crypto.prompt_evo_population,
                snapshot_limit=self.settings.crypto.prompt_evo_snapshots,
            )
            logger.info(
                "Nightly prompt-evolve: best fitness %+.4f over %d snapshots",
                result.best.fitness,
                result.n_snapshots,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("nightly prompt-evolve failed: %s", exc)

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
        self._runtime.bot_running = False

        # Cancel the reconcile background task before disposing the engine.
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except Exception:
                pass
            self._reconcile_task = None

        components: list[tuple[str, object | None]] = [
            ("monitor", self._monitor),
            ("news_reactor", self._news_reactor),
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

        # Cancel any open orders BEFORE we disconnect — leaving them
        # sitting on the book during a restart leads to phantom positions
        # the next process can't reconcile.
        try:
            from halal_trader.core.shutdown import cancel_all_open_orders

            await cancel_all_open_orders(self._binance)
        except Exception as e:
            logger.warning("Failed to cancel open orders during shutdown: %s", e)

        try:
            await self._binance.disconnect()
        except Exception as e:
            logger.warning("Failed to disconnect Binance: %s", e)

        await super().shutdown()
        logger.info("Crypto trading bot shut down")

    # ── Main Loop ──────────────────────────────────────────────

    async def run(self) -> None:
        """Start the crypto trading bot with continuous 1-minute cycles.

        Wave C: every long-lived task (monitor, ws, news_reactor,
        sentiment_manager, reconcile loop) is registered with a single
        :class:`TaskSupervisor`. A crash in any ``CRASH_BOT``-policy
        task propagates back here with the original traceback rather
        than silently leaving a stale subsystem behind.
        """
        from halal_trader.core.supervisor import RestartPolicy, TaskSupervisor

        await self.initialize()
        self._running = True

        # Register signal handlers for clean shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

        try:
            async with TaskSupervisor() as sup:
                self._supervisor = sup
                # Long-lived tasks, restart policies tuned per subsystem:
                #   monitor     — CRASH_BOT (closing risk > running half-blind)
                #   ws          — CRASH_BOT (stale prices break SL/TP enforcement)
                #   news_reactor — RESTART  (external API flake shouldn't kill bot)
                #   sentiment   — RESTART  (best-effort context)
                #   reconcile   — RESTART  (cosmetic drift, retry on hiccup)
                if self._monitor is not None:
                    sup.start("monitor", self._monitor.run, policy=RestartPolicy.CRASH_BOT)
                if self._ws is not None:
                    sup.start("ws", self._ws.run, policy=RestartPolicy.CRASH_BOT)
                if self._news_reactor is not None and self._news_reactor.enabled:
                    sup.start("news_reactor", self._news_reactor.run, policy=RestartPolicy.RESTART)
                if self._sentiment_manager is not None and getattr(
                    self._sentiment_manager, "enabled", False
                ):
                    sup.start(
                        "sentiment_manager",
                        self._sentiment_manager.run,
                        policy=RestartPolicy.RESTART,
                    )
                sup.start("reconcile", self._reconcile_loop, policy=RestartPolicy.RESTART)

                await self._run_cycle_loop()
                # Cycle loop returned cleanly (running=False). Cancel
                # the supervised tasks so the ``async with`` exit isn't
                # blocked by long-running loops still running off the
                # ``_running`` flag.
                sup.cancel()
        finally:
            # Resource teardown (broker disconnect, open-order cancel,
            # daily-end summary) runs *after* the supervisor scope exits
            # so subsystems aren't still holding the broker handle when
            # we close it.
            try:
                await self._daily_end()
            except Exception as exc:  # noqa: BLE001
                logger.debug("daily_end failed during shutdown: %s", exc)
            await self.shutdown()

    async def _run_cycle_loop(self) -> None:
        """The main cycle loop — extracted so ``run()`` can wrap it in
        a supervisor scope without changing the inner logic."""
        # Initial daily start
        await self._daily_start()

        interval = self.settings.crypto.trading_interval_seconds

        logger.info(
            "Crypto bot started — interval: %ds, target: %.1f%%, loss limit: %.1f%%, "
            "pairs: %s, testnet: %s",
            interval,
            self.settings.crypto.daily_return_target * 100,
            self.settings.crypto.daily_loss_limit * 100,
            ", ".join(self.settings.crypto.pairs),
            self.settings.binance.testnet,
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
                    cycle_service = self._get_cycle_service()
                    await asyncio.wait_for(
                        cycle_service.run_cycle(),
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
                        market="crypto",
                        severity="error",
                    )

                from datetime import UTC, datetime

                self._runtime.last_cycle = {
                    "completed_at": datetime.now(UTC).isoformat(),
                    "market": "crypto",
                }

                # Adaptive cadence — high-vol regimes shorten the next
                # cycle; chop lengthens it. Falls back to the configured
                # interval when no indicators are available yet (cold start
                # or a cycle that returned early).
                next_interval = interval
                if cycle_service.last_indicators_cache:
                    decision = select_interval(
                        indicators_cache=cycle_service.last_indicators_cache,
                        base_interval=interval,
                        atr_baseline=self.settings.crypto.atr_baseline,
                    )
                    next_interval = decision.interval_seconds
                    # Log only on regime change. Volatility regimes tend
                    # to persist for many cycles; the prior code fired the
                    # same line every cycle and dominated the log output.
                    if decision.regime != "normal" and decision.regime != getattr(
                        self, "_last_cadence_regime", None
                    ):
                        logger.info(
                            "Adaptive cadence: %s regime (median ATR %.4f, ratio %.2f) → %ds",
                            decision.regime,
                            decision.median_atr,
                            decision.ratio,
                            next_interval,
                        )
                    self._last_cadence_regime = decision.regime  # type: ignore[attr-defined]

                # Sleep for remaining interval time
                elapsed = time.monotonic() - cycle_start
                sleep_time = max(0, next_interval - elapsed)
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
                        next_interval,
                    )

        except KeyboardInterrupt, asyncio.CancelledError:
            logger.info("Crypto bot interrupted")
            # daily_end + shutdown are owned by ``run()``'s outer
            # ``finally`` block (Wave C) so subsystems aren't still
            # running off the broker handle when we close it.

    def _handle_signal(self) -> None:
        """Signal handler: set running flag to false for clean shutdown."""
        logger.info("Received shutdown signal")
        self._running = False
