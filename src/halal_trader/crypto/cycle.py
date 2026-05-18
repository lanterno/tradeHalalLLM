"""Crypto trading cycle — Wave B stage-pipeline composition.

``_run_cycle_impl`` is now a thin orchestration layer: it builds a
fresh :class:`CycleState`, threads it through three stage lists
(pre-LLM data gathering → prompt-context build → post-LLM execution +
analytics), and short-circuits on early-out conditions (no pairs,
all-pairs-flat skip, live-mode tripwire, low-USDT guard). Every
data-gathering and side-effect block now lives as a
:class:`core.cycle_stages.CycleStage` subclass — adding a new
prompt-context source = one new file + one new line on the stage list.
"""

from __future__ import annotations

import logging
from datetime import date

from halal_trader.config import get_settings
from halal_trader.core.cycle import BaseCycleService
from halal_trader.core.cycle_pipeline import CycleState, run_stages, stage
from halal_trader.core.cycle_stages import (
    ApplyRegimeGateStage,
    AugmentMicrostructureWithBasisStage,
    AugmentMicrostructureWithWhaleFlowsStage,
    AugmentRegimeWithMemoryStage,
    AugmentRegimeWithRagStage,
    BuildActiveAdjustmentsStage,
    BuildCryptoRiskStage,
    BuildExchangeRulesStage,
    BuildForecastsStage,
    BuildMicrostructureStage,
    BuildMlSignalsStage,
    BuildNewsStage,
    BuildPerformanceStage,
    BuildRegimeStage,
    BuildSentimentStage,
    BuildSlippageTextStage,
    BuildTimeframeStage,
    ComputeIndicatorsStage,
    ExecuteAndNotifyStage,
    FetchKlinesStage,
    FetchOrderbooksStage,
    GetTradeablePairsStage,
    RecordCycleAnalyticsStage,
)
from halal_trader.core.event_bus import EventBus
from halal_trader.core.insights_hub import InsightsHub
from halal_trader.core.tracing import tracer
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.risk import PortfolioRiskEngine
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.websocket import BinanceWSManager

logger = logging.getLogger(__name__)


class CryptoCycleService(BaseCycleService):
    """Runs a single crypto trading cycle: gather data, analyze, execute.

    Crypto markets are 24/7 — there is no market-hours check.
    """

    def __init__(
        self,
        broker: BinanceClient,
        screener: CryptoHalalScreener,
        strategy: CryptoTradingStrategy,
        executor: CryptoExecutor,
        portfolio: CryptoPortfolioTracker,
        ws_manager: BinanceWSManager | None = None,
        configured_pairs: list[str] | None = None,
        analytics: PerformanceAnalytics | None = None,
        sentiment_manager=None,
        timeframe_analyzer=None,
        regime_detector=None,
        ml_forecaster=None,
        ml_anomaly_detector=None,
        ml_signal_classifier=None,
        self_review=None,
        notifier=None,
        risk_engine: PortfolioRiskEngine | None = None,
        news_feed=None,
        alerts=None,
        engine=None,
        live_mode_checker=None,
        shadow_runner=None,
        whale_flow_source=None,
        reddit_fetcher=None,
        hub: InsightsHub | None = None,
        bus: "EventBus | None" = None,
    ) -> None:
        super().__init__(alerts=alerts, engine=engine)
        self._hub = hub or InsightsHub()
        self._bus = bus
        self._live_mode_checker = live_mode_checker
        self._broker = broker
        self._screener = screener
        self._strategy = strategy
        self._executor = executor
        self._portfolio = portfolio
        self._ws = ws_manager
        self._configured_pairs = configured_pairs or []
        self._analytics = analytics
        self._sentiment = sentiment_manager
        self._timeframes = timeframe_analyzer
        self._regime = regime_detector
        self._ml_forecaster = ml_forecaster
        self._ml_anomaly = ml_anomaly_detector
        self._ml_signal = ml_signal_classifier
        self._self_review = self_review
        self._notifier = notifier
        self._risk_engine = risk_engine
        self._news_feed = news_feed
        self._shadow_runner = shadow_runner
        self._whale_flow_source = whale_flow_source
        self._reddit_fetcher = reddit_fetcher
        self._consecutive_flat_skips = 0
        self._settings = get_settings()
        # Replay snapshot store — DB-backed when an engine is available.
        if engine is not None:
            from halal_trader.core.replay import ReplayStore

            self._replay_store: "ReplayStore | None" = ReplayStore(engine=engine)
        else:
            self._replay_store = None
        # The scheduler reads this after each cycle to drive the
        # adaptive cadence selector. None until the first successful
        # cycle has populated the indicator cache.
        self.last_indicators_cache: dict[str, dict] | None = None
        # First-cycle-of-the-day marker — kept on the service for
        # legacy test compatibility; the actual snapshot decision lives
        # on the analytics stage.
        self._last_regime_snapshot_date: date | None = None
        # Analytics + execute stages carry cross-cycle state (treasury
        # de-dup key, daily regime-snapshot date), so they're created
        # once here and reused on every cycle.
        self._analytics_stage = RecordCycleAnalyticsStage(
            hub=self._hub,
            shadow_runner=self._shadow_runner,
            replay_store=self._replay_store,
        )

    async def _pre_cycle_checks(self) -> bool:
        return True  # Crypto markets are 24/7

    async def _should_halt(self) -> bool:
        if await self._portfolio.should_halt_trading():
            logger.warning("Crypto daily loss limit reached — halting trades")
            return True
        return False

    async def _run_cycle_impl(self) -> None:
        await self._broker.refresh_symbol_filters_if_stale()
        state = CycleState()

        # ── Pre-LLM data gathering ────────────────────────────────
        await run_stages(
            state,
            [
                GetTradeablePairsStage(
                    screener=self._screener,
                    portfolio=self._portfolio,
                    configured_pairs=self._configured_pairs,
                    max_pairs=self._settings.crypto.max_pairs_per_cycle,
                ),
                FetchKlinesStage(broker=self._broker, ws_manager=self._ws),
                ComputeIndicatorsStage(),
            ],
            bus=self._bus,
        )
        if not state.halal_pairs:
            logger.warning("No halal crypto pairs available, skipping cycle")
            return
        # Scheduler reads this to drive the adaptive-cadence selector.
        self.last_indicators_cache = state.indicators_cache

        # ── Open positions + WS prices (cheap, no stage needed) ───
        open_trades = None
        current_prices: dict[str, float] = {}
        try:
            open_trades = await self._portfolio.get_open_trades()
            if self._ws:
                for pair in self._configured_pairs:
                    p = self._ws.get_latest_price(pair)
                    if p is not None:
                        current_prices[pair] = p
        except Exception as exc:
            logger.debug("Failed to fetch open trades: %s", exc)
        has_open_positions = bool(open_trades)
        state.current_prices = current_prices

        # ── All-pairs-flat skip guard (mutates counter on self) ───
        if not has_open_positions and self._should_skip_llm(state.indicators_cache):
            self._consecutive_flat_skips += 1
            if self._consecutive_flat_skips < self._settings.crypto.max_consecutive_flat_skips:
                logger.info(
                    "All pairs flat — skipping LLM analysis (%d/%d)",
                    self._consecutive_flat_skips,
                    self._settings.crypto.max_consecutive_flat_skips,
                )
                return
            logger.info(
                "All pairs flat but reached max consecutive skips (%d) — forcing LLM",
                self._consecutive_flat_skips,
            )
        self._consecutive_flat_skips = 0

        # ── Orderbooks + account + balances ───────────────────────
        async with tracer.aspan("cycle.fetch_orderbooks", pair_count=len(state.halal_pairs)):
            await run_stages(state, [FetchOrderbooksStage(broker=self._broker)], bus=self._bus)
        async with tracer.aspan("cycle.fetch_account"):
            account = await self._broker.get_account()
            balances = await self._broker.get_balances()
        state.account = account

        # ── Live-mode safeguard (kill-switch on first violation) ──
        if self._live_mode_checker is not None and self._live_mode_checker.active:
            safe = await self._live_mode_checker.assert_safe(
                account_balance=account.total_balance_usdt,
                engine=self._engine,
                alerts=self._alerts,
            )
            if not safe:
                logger.error("Crypto live-mode safeguard tripped — refusing to trade.")
                return

        # ── Position summary + low-USDT guard ─────────────────────
        positions_text = self._portfolio.format_positions_for_prompt(
            balances,
            configured_pairs=self._configured_pairs,
            open_trades=open_trades,
            current_prices=current_prices,
        )
        state.today_pnl = await self._portfolio.get_current_pnl(account=account)
        open_position_count = self._count_open_positions(balances)
        if not self._has_enough_usdt(account, balances):
            return

        # ── Prompt-context build (14 stages, halts on risk veto) ──
        await run_stages(
            state,
            [
                # 24h lookback — see commit history for the rationale.
                BuildPerformanceStage(self._analytics, lookback_days=1),
                BuildSentimentStage(
                    sentiment_manager=self._sentiment,
                    reddit_fetcher=self._reddit_fetcher,
                    hub=self._hub,
                    notifier=self._notifier,
                ),
                BuildTimeframeStage(self._timeframes),
                BuildRegimeStage(self._regime),
                BuildForecastsStage(self._ml_forecaster),
                BuildMlSignalsStage(
                    anomaly_detector=self._ml_anomaly,
                    signal_classifier=self._ml_signal,
                ),
                BuildCryptoRiskStage(
                    risk_engine=self._risk_engine,
                    broker=self._broker,
                    open_trades=open_trades or [],
                ),
                BuildActiveAdjustmentsStage(self._self_review),
                BuildExchangeRulesStage(self._broker),
                BuildMicrostructureStage(),
                AugmentMicrostructureWithBasisStage(
                    broker=self._broker,
                    basis_tracker=getattr(self._hub, "basis", None),
                ),
                AugmentMicrostructureWithWhaleFlowsStage(
                    whale_flow_source=self._whale_flow_source,
                    hub=self._hub,
                ),
                BuildNewsStage(self._news_feed),
                BuildSlippageTextStage(
                    slippage_model=getattr(self._executor, "_slippage_model", None),
                    max_position_pct=self._settings.crypto.max_position_pct,
                ),
                AugmentRegimeWithRagStage(getattr(self._hub, "rag", None)),
                AugmentRegimeWithMemoryStage(getattr(self._hub, "regime", None)),
            ],
            bus=self._bus,
            stop_on_halt=True,
        )

        # Push the structured risk state into the hub's runtime view.
        self._publish_risk_runtime(state)
        if state.halt:
            logger.warning(
                "Crypto risk engine halt: %s",
                getattr(state.risk_state, "halt_reason", "unspecified"),
            )
            return

        # ── LLM analyze (swallow=False; must propagate) ───────────
        analyze_kwargs = dict(
            account=account,
            positions_text=positions_text,
            halal_pairs=state.halal_pairs,
            klines_by_symbol=state.klines_by_symbol,
            orderbooks=state.orderbooks,
            today_pnl=state.today_pnl,
            performance_text=state.performance_text,
            sentiment_text=state.sentiment_text,
            timeframe_text=state.timeframe_text,
            ml_signals_text=state.ml_signals_text,
            regime_text=state.regime_text,
            active_adjustments=state.active_adjustments,
            exchange_rules_text=state.exchange_rules_text,
            indicators_cache=state.indicators_cache,
            open_position_count=open_position_count,
            risk_text=state.risk_text,
            microstructure_text=state.microstructure_text,
            news_text=state.news_text,
            slippage_text=state.slippage_text,
        )
        async with stage(
            self._bus,
            "strategy_analyze",
            swallow=False,
            pair_count=len(state.halal_pairs),
            open_positions=open_position_count,
        ):
            async with tracer.aspan(
                "cycle.strategy_analyze",
                pair_count=len(state.halal_pairs),
                open_positions=open_position_count,
            ):
                state.plan = await self._strategy.analyze(**analyze_kwargs)

        logger.info(
            "Crypto plan: %s | %d buys, %d sells",
            state.plan.market_outlook[:80] if state.plan.market_outlook else "N/A",
            len(state.plan.buys),
            len(state.plan.sells),
        )

        # ── Post-analyze gate → analytics → execute + notify ──────
        await run_stages(
            state,
            [
                ApplyRegimeGateStage(self._regime),
                self._analytics_stage,
                ExecuteAndNotifyStage(
                    executor=self._executor,
                    portfolio=self._portfolio,
                    notifier=self._notifier,
                    shadow_runner=self._shadow_runner,
                    shadow_kwargs_builder=lambda _s: analyze_kwargs,
                ),
            ],
            bus=self._bus,
        )
        # Mirror the analytics-stage's daily-regime tracker onto the
        # service so legacy tests asserting on ``_last_regime_snapshot_date``
        # still pass.
        self._last_regime_snapshot_date = self._analytics_stage._last_regime_snapshot_date

    # ── Cycle-control helpers (state-on-self, not state-on-CycleState) ──

    def _count_open_positions(self, balances) -> int:
        """Count balances whose USDT value is ≥ $5 (dust-aware)."""
        tracked_bases = {
            p.upper().removesuffix("USDT").removesuffix("BUSD") for p in self._configured_pairs
        }
        n = 0
        for b in balances:
            if b.asset in tracked_bases and b.free > 0:
                price = self._broker.get_cached_price(f"{b.asset}USDT")
                if price and b.free * price < 5.0:
                    continue
                n += 1
        return n

    def _has_enough_usdt(self, account, balances) -> bool:
        """Skip the cycle when USDT < $5 AND no open positions remain."""
        usdt_free = account.usdt_free
        if usdt_free >= 5.0:
            return True
        tracked_bases = {
            p.upper().removesuffix("USDT").removesuffix("BUSD") for p in self._configured_pairs
        }
        has_positions = any(b.asset in tracked_bases and b.free > 0 for b in balances)
        if not has_positions:
            logger.info(
                "Available USDT ($%.2f) below $5 and no open positions — skipping",
                usdt_free,
            )
            return False
        logger.info(
            "Low USDT ($%.2f) but have open positions — LLM may recommend sells",
            usdt_free,
        )
        return True

    def _should_skip_llm(self, indicators_cache: dict[str, dict]) -> bool:
        """Skip LLM if all pairs are flat with no directional signal."""
        if not indicators_cache:
            return True
        s = self._settings
        for _symbol, indicators in indicators_cache.items():
            if indicators.get("error"):
                continue
            price_change_5m = abs(indicators.get("price_change_5m", 0))
            rsi = indicators.get("rsi_14", 50)
            vol_ratio = indicators.get("volume_ratio", 1.0)
            if (
                price_change_5m > s.crypto.flat_price_threshold
                or rsi < s.crypto.flat_rsi_lower
                or rsi > s.crypto.flat_rsi_upper
                or vol_ratio > s.crypto.flat_vol_threshold
            ):
                return False
        return True

    def _publish_risk_runtime(self, state: CycleState) -> None:
        """Mirror the risk-stage output into ``hub.runtime`` for /api/risk/state."""
        runtime = getattr(self._hub, "runtime", None)
        rs = state.risk_state
        if runtime is None or rs is None:
            return
        from datetime import UTC
        from datetime import datetime as _dt

        runtime.risk_state = {
            "market": "crypto",
            "is_halted": getattr(rs, "is_halted", False),
            "halt_reason": getattr(rs, "halt_reason", ""),
            "portfolio_heat_pct": getattr(rs, "portfolio_heat_pct", None),
            "drawdown_pct": getattr(rs, "drawdown_pct", None),
            "avg_correlation": getattr(rs, "avg_correlation", None),
            "summary": state.risk_text,
            "pushed_at": _dt.now(UTC).isoformat(),
        }
