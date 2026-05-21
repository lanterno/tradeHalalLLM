"""Intraday trading cycle — gathers market data, analyzes, and executes."""

import logging
from typing import Any

from halal_trader.core.cycle import BaseCycleService
from halal_trader.core.observability import cycle_id_var
from halal_trader.domain.ports import Broker, ComplianceScreener
from halal_trader.market_hours import is_market_open_local, now_eastern
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.portfolio import PortfolioTracker
from halal_trader.trading.strategy import TradingStrategy

logger = logging.getLogger(__name__)

# Maximum number of symbols to fetch market data for per cycle.
_MAX_SYMBOLS_PER_CYCLE = 20


class TradingCycleService(BaseCycleService):
    """Runs a single intraday trading cycle: gather data, analyze, execute.

    Extracted from the scheduler so the cycle logic is independently testable
    and the scheduler stays a thin scheduling layer.
    """

    def __init__(
        self,
        broker: Broker,
        screener: ComplianceScreener,
        strategy: TradingStrategy,
        executor: TradeExecutor,
        portfolio: PortfolioTracker,
        catalyst_feed: Any = None,
        alerts=None,
        engine=None,
        live_mode_checker=None,
        shadow_runner: Any = None,
        regime_detector: Any = None,
        ml_anomaly_detector: Any = None,
        ml_signal_classifier: Any = None,
        timeframe_analyzer: Any = None,
        insights_hub: Any = None,
        notifier: Any = None,
        analytics: Any = None,
        self_review: Any = None,
        news_collector: Any = None,
    ) -> None:
        super().__init__(alerts=alerts, engine=engine)
        self._live_mode_checker = live_mode_checker
        self._broker = broker
        self._screener = screener
        self._strategy = strategy
        self._executor = executor
        self._portfolio = portfolio
        # Optional StockCatalystFeed (Phase 3.5) — gives the LLM live news,
        # earnings, insider activity. Cycle proceeds normally if absent.
        self._catalyst_feed = catalyst_feed
        # Optional shadow runner — runs a frozen-prompt strategy on the
        # same per-cycle inputs and records a divergence row to the
        # shadow ledger. Mirrors the crypto wiring; off when disabled.
        self._shadow_runner = shadow_runner
        # Optional regime detector — classifies each symbol's market state
        # (trending / ranging / high-vol) from indicators and surfaces a
        # tactical-instruction block to the LLM prompt. Mirrors crypto.
        self._regime_detector = regime_detector
        # Optional ML inference path — anomaly detector flags abnormal
        # indicator vectors; signal classifier converts the same vector
        # into a buy/hold/sell confidence. Both consume the per-symbol
        # indicator dict already computed for risk/regime; no extra
        # bar fetch. Forecaster is intentionally omitted — daily bars
        # are too sparse for Chronos's 96-step minimum.
        self._ml_anomaly = ml_anomaly_detector
        self._ml_signal = ml_signal_classifier
        # Optional multi-timeframe analyzer — pulls hourly/daily/weekly
        # bars and surfaces a trend-alignment score per symbol. Same
        # interface as the crypto analyzer.
        self._timeframes = timeframe_analyzer
        # Optional insights hub — when wired, the cycle pushes the
        # structured risk state into ``hub.runtime.risk_state`` so the
        # dashboard's risk panel renders the stocks bot's heat /
        # drawdown / correlation. Mirrors crypto's pattern.
        self._hub = insights_hub
        # Optional Telegram notifier — fires `notify_trade` on filled
        # buys/sells so the operator gets the same Telegram alerts on
        # stocks fills they already get for crypto.
        self._notifier = notifier
        # Stocks-side parity (round-7 follow-up): rolling-window
        # performance summary + active self-improve adjustments. The
        # analytics impl just needs ``compute_stats`` + ``format_for_prompt``
        # (``CrossAssetAnalytics(repo, asset_class="stock")`` satisfies);
        # ``self_review`` just needs ``format_adjustments_for_prompt()``.
        # Both default to None — stage emits an empty block.
        self._analytics = analytics
        self._self_review = self_review
        # Stocks-side equities news source — Yahoo Finance search
        # endpoint (see :mod:`sentiment.stocks_news`). When None the
        # ``FetchStockNewsStage`` is a no-op and ``state.news_text``
        # stays empty.
        self._news_collector = news_collector

    async def _pre_cycle_checks(self) -> bool:
        now = now_eastern()
        logger.info(
            "=== TRADING CYCLE === (current time: %s ET)", now.strftime("%Y-%m-%d %H:%M:%S")
        )

        if not is_market_open_local():
            logger.info("Market is closed (local check), skipping trading cycle")
            return False

        clock = await self._broker.get_clock()
        logger.info(
            "Market clock: is_open=%s next_open='%s' next_close='%s'",
            clock.is_open,
            clock.next_open,
            clock.next_close,
        )
        if not clock.is_open:
            logger.info("Market is closed (broker API), skipping trading cycle")
            return False

        return True

    async def _should_halt(self) -> bool:
        if await self._portfolio.should_halt_trading():
            logger.warning("Daily loss limit reached — halting trades")
            return True
        return False

    async def _post_cycle(self) -> None:
        """Run a reconciliation pass after each cycle (cheap; cycle is 15-min)."""
        if self._engine is None:
            return
        try:
            from halal_trader.core.reconcile import reconcile_stocks

            await reconcile_stocks(
                engine=self._engine,
                broker=self._broker,
                alerts=self._alerts,
            )
        except Exception as exc:
            import logging as _logging

            _logging.getLogger(__name__).debug("Stock reconcile failed: %s", exc)

    async def _run_cycle_impl(self) -> None:
        account = await self._broker.get_account_info()

        if self._live_mode_checker is not None and self._live_mode_checker.active:
            safe = await self._live_mode_checker.assert_safe(
                account_balance=account.effective_equity,
                engine=self._engine,
                alerts=self._alerts,
            )
            if not safe:
                logger.error("Stock live-mode safeguard tripped — refusing to trade.")
                return

        positions = await self._broker.get_all_positions()

        halal_symbols = await self._screener.get_halal_symbols()
        if not halal_symbols:
            logger.warning("No halal symbols available, skipping cycle")
            return

        snapshots, bars = await self._fetch_market_data(halal_symbols)
        today_pnl = await self._portfolio.get_current_pnl()

        # ── Wave B: drive a single CycleState through the stage list ──
        from halal_trader.core.cycle_pipeline import CycleState, run_stages
        from halal_trader.core.cycle_stages import (
            ApplyRegimeGateStage,
            BuildActiveAdjustmentsStage,
            BuildCatalystsStage,
            BuildMlSignalsStage,
            BuildPerformanceStage,
            BuildRegimeStage,
            BuildStockRiskStage,
            BuildTimeframeStage,
            FetchStockNewsStage,
        )

        state = CycleState(
            account=account,
            halal_pairs=halal_symbols[:_MAX_SYMBOLS_PER_CYCLE],
            open_positions=positions,
            today_pnl=today_pnl,
            snapshots=snapshots,
            bars=bars,
        )
        await run_stages(
            state,
            [
                BuildStockRiskStage(),
                BuildRegimeStage(self._regime_detector),
                BuildMlSignalsStage(
                    anomaly_detector=self._ml_anomaly,
                    signal_classifier=self._ml_signal,
                ),
                BuildTimeframeStage(self._timeframes),
                BuildCatalystsStage(self._catalyst_feed),
                # 7-day lookback — stocks cycle is daily-ish (15min cron,
                # but trades close intraday). Matches the crypto 24h
                # window in spirit (one trading day's worth of round-trips).
                BuildPerformanceStage(self._analytics, lookback_days=7),
                BuildActiveAdjustmentsStage(self._self_review),
                # Equities news pull (Yahoo Finance) — 15-min cadence
                # absorbs the per-symbol HTTP latency. Cached for the
                # same TTL inside the collector.
                FetchStockNewsStage(self._news_collector),
            ],
            stop_on_halt=True,
        )
        # Push the structured risk state into the hub's runtime view so
        # the dashboard's /api/risk/state can render the stocks bot's
        # heat / drawdown / correlation. Best-effort: no hub → no push.
        runtime = getattr(self._hub, "runtime", None)
        rs = state.risk_state
        if runtime is not None and rs is not None:
            from datetime import UTC
            from datetime import datetime as _dt

            runtime.risk_state = {
                "market": "stocks",
                "is_halted": getattr(rs, "is_halted", False),
                "halt_reason": getattr(rs, "halt_reason", ""),
                "portfolio_heat_pct": getattr(rs, "portfolio_heat_pct", None),
                "drawdown_pct": getattr(rs, "drawdown_pct", None),
                "avg_correlation": getattr(rs, "avg_correlation", None),
                "summary": state.risk_text,
                "pushed_at": _dt.now(UTC).isoformat(),
            }

        if state.halt:
            logger.warning(
                "Stocks risk engine halt: %s",
                getattr(rs, "halt_reason", "unspecified"),
            )
            return

        analyze_kwargs = dict(
            account=account,
            positions=positions,
            halal_symbols=halal_symbols,
            snapshots=snapshots,
            bars=bars,
            today_pnl=today_pnl,
            risk_text=state.risk_text,
            regime_text=state.regime_text,
            ml_signals_text=state.ml_signals_text,
            timeframe_text=state.timeframe_text,
            catalysts_text=state.catalysts_text,
            performance_text=state.performance_text,
            active_adjustments=state.active_adjustments,
            news_text=state.news_text,
        )
        plan = await self._strategy.analyze(**analyze_kwargs)

        # Mirror the crypto cycle's post-analyze regime gate: strip
        # BUYs for any symbol that the rule-based detector classifies
        # as a confirmed downtrend at ≥0.6 confidence.
        state.plan = plan
        await run_stages(state, [ApplyRegimeGateStage(self._regime_detector)])

        logger.info(
            "Trading plan: %s | %d buys, %d sells",
            plan.market_outlook[:80],
            len(plan.buys),
            len(plan.sells),
        )

        if plan.decisions:
            # Pass current positions so the executor can apply the
            # per-sector halal allocation cap on each candidate buy.
            results = await self._executor.execute_plan(plan, bars=bars, positions=positions)
            await self._handle_execution_results(results)
        else:
            logger.info("No trades to execute this cycle")

        if self._shadow_runner is not None:
            try:
                latest_prices = _extract_latest_prices(snapshots)
                await self._shadow_runner.observe_cycle(
                    cycle_id=cycle_id_var.get() or "cycle-unknown",
                    live_equity=account.effective_equity or account.equity or 0.0,
                    latest_prices=latest_prices,
                    analyze_kwargs=analyze_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("stocks shadow runner observe_cycle failed: %s", exc)

    # ── Private helpers ──────────────────────────────────────────

    async def _handle_execution_results(self, results: list[dict[str, Any]]) -> None:
        """Process executor results: notify on fills, record failures.

        Mirrors the crypto cycle's :class:`ExecuteAndNotifyStage` post-
        execute loop but kept inline because the stocks cycle doesn't
        yet drive the post-LLM block through a stage list.

        Failures (``status`` in ``"error"`` / ``"rejected"``) bump the
        self-review's per-symbol counter so the 10-failures trigger
        can fire on persistent failure modes. Filled trades fire a
        Telegram notification. Both side effects are best-effort —
        the cycle never aborts on a missing collaborator or a
        downstream blow-up.

        If the cycle proposed actions but every result was rejected /
        errored / skipped, emit a ``cycle.no_action`` event so we can
        grep wasted-cycle frequency from the JSON log. (Empty plans
        are silently no-op — the LLM had nothing to do.)
        """
        from halal_trader.core import events as _events

        no_action_statuses = {"error", "rejected", "skipped"}
        total = len(results)
        rejected_count = 0
        rejection_reasons: list[str] = []
        for r in results:
            logger.info("Execution result: %s", r)
            status = r.get("status")
            if status in no_action_statuses:
                rejected_count += 1
                reason = str(r.get("reason", status))[:120]
                rejection_reasons.append(f"{r.get('symbol', '?')}:{reason}")
            if status in ("error", "rejected") and self._self_review is not None:
                try:
                    self._self_review.record_execution_failure(
                        r.get("symbol", "?"),
                        str(r.get("reason", status)),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to record execution failure: %s", exc)
            if self._notifier and status in ("submitted", "filled"):
                try:
                    await self._notifier.notify_trade(
                        pair=r.get("symbol", ""),
                        side=r.get("action", ""),
                        quantity=r.get("quantity", 0),
                        price=r.get("price", 0),
                        market="stocks",
                        order_id=str(r.get("order_id", "")),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to send trade notification: %s", exc)

        # If the LLM proposed work AND every result was rejected/errored/
        # skipped, the cycle was a no-op despite a full LLM call. Emit
        # a structured event so we can grep wasted-cycle frequency and
        # tune the system prompt (e.g. position-cap awareness).
        if total > 0 and rejected_count == total:
            logger.warning(
                "Cycle proposed %d action(s) but every one was rejected/errored: %s",
                total,
                "; ".join(rejection_reasons[:5]),
                extra={
                    "event": _events.CYCLE_NO_ACTION,
                    "proposed": total,
                    "rejected": rejected_count,
                    "reasons": rejection_reasons[:5],
                },
            )

    async def _fetch_market_data(
        self, halal_symbols: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch snapshots and bars for halal symbols, capped to avoid rate limits."""
        snapshots: dict[str, Any] = {}
        bars: dict[str, Any] = {}
        for sym in halal_symbols[:_MAX_SYMBOLS_PER_CYCLE]:
            try:
                snap = await self._broker.get_stock_snapshot(sym)
                snapshots[sym] = snap
            except Exception as e:
                logger.debug("Failed to get snapshot for %s: %s", sym, e)
            try:
                bar = await self._broker.get_stock_bars(sym, days=5, timeframe="1Day")
                bars[sym] = bar
            except Exception as e:
                logger.debug("Failed to get bars for %s: %s", sym, e)
        return snapshots, bars


def _extract_latest_prices(snapshots: dict[str, Any]) -> dict[str, float]:
    """Best-effort latest-price map for the shadow simulator.

    Anything that can't be priced is silently dropped — the simulator
    skips positions it can't value.
    """
    from halal_trader.trading.bars import extract_last_price

    prices: dict[str, float] = {}
    for sym, snap in snapshots.items():
        price = extract_last_price(snap, sym)
        if price is not None:
            prices[sym] = price
    return prices
