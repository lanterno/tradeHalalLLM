"""Crypto trading cycle — gathers data, computes indicators, analyzes, and executes."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

from binance import BinanceAPIException

from halal_trader.config import get_settings
from halal_trader.core.cycle import BaseCycleService
from halal_trader.core.cycle_pipeline import stage
from halal_trader.core.event_bus import EventBus
from halal_trader.core.insights_hub import InsightsHub
from halal_trader.core.observability import cycle_id_var
from halal_trader.core.tracing import tracer
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.indicators import compute_all
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.risk import PortfolioRiskEngine
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.domain.models import Kline

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
        # Captures the full LLM-prompt context for each cycle so prompt
        # changes can be honestly scored against historical inputs.
        if engine is not None:
            from halal_trader.core.replay import ReplayStore

            self._replay_store: "ReplayStore | None" = ReplayStore(engine=engine)
        else:
            self._replay_store = None
        # The scheduler reads this after each cycle to drive the
        # adaptive cadence selector. None until the first successful
        # cycle has populated the indicator cache.
        self.last_indicators_cache: dict[str, dict] | None = None
        # First-cycle-of-the-day marker for daily regime snapshots.
        self._last_regime_snapshot_date: date | None = None

    async def _pre_cycle_checks(self) -> bool:
        return True  # Crypto markets are 24/7

    async def _should_halt(self) -> bool:
        if await self._portfolio.should_halt_trading():
            logger.warning("Crypto daily loss limit reached — halting trades")
            return True
        return False

    async def _run_cycle_impl(self) -> None:
        await self._broker.refresh_symbol_filters_if_stale()

        halal_pairs = await self._get_tradeable_pairs()
        if not halal_pairs:
            logger.warning("No halal crypto pairs available, skipping cycle")
            return

        klines_by_symbol = await self._fetch_klines(halal_pairs)

        indicators_cache: dict[str, dict] = {}
        for symbol, klines in klines_by_symbol.items():
            indicators_cache[symbol] = compute_all(klines)
        # Snapshot for the scheduler's adaptive-cadence selector.
        self.last_indicators_cache = indicators_cache

        open_trades = None
        current_prices: dict[str, float] = {}
        try:
            open_trades = await self._portfolio.get_open_trades()
            if self._ws:
                for pair in self._configured_pairs:
                    p = self._ws.get_latest_price(pair)
                    if p is not None:
                        current_prices[pair] = p
        except Exception as e:
            logger.debug("Failed to fetch open trades: %s", e)

        has_open_positions = bool(open_trades)

        if not has_open_positions and self._should_skip_llm(indicators_cache):
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

        async with tracer.aspan("cycle.fetch_orderbooks", pair_count=len(halal_pairs)):
            orderbooks = await self._fetch_orderbooks(halal_pairs)

        async with tracer.aspan("cycle.fetch_account"):
            account = await self._broker.get_account()
            balances = await self._broker.get_balances()

        # Live-mode safeguards run on every cycle — they short-circuit cheaply
        # when the bot is on testnet, and trip the kill-switch on first violation.
        if self._live_mode_checker is not None and self._live_mode_checker.active:
            safe = await self._live_mode_checker.assert_safe(
                account_balance=account.total_balance_usdt,
                engine=self._engine,
                alerts=self._alerts,
            )
            if not safe:
                logger.error("Crypto live-mode safeguard tripped — refusing to trade.")
                return

        positions_text = self._portfolio.format_positions_for_prompt(
            balances,
            configured_pairs=self._configured_pairs,
            open_trades=open_trades,
            current_prices=current_prices,
        )
        today_pnl = await self._portfolio.get_current_pnl(account=account)

        tracked_bases = {
            p.upper().removesuffix("USDT").removesuffix("BUSD") for p in self._configured_pairs
        }
        open_position_count = 0
        for b in balances:
            if b.asset in tracked_bases and b.free > 0:
                price = self._broker.get_cached_price(f"{b.asset}USDT")
                if price and b.free * price < 5.0:
                    continue
                open_position_count += 1

        usdt_free = account.usdt_free
        if usdt_free < 5.0:
            tracked_bases = {
                p.upper().removesuffix("USDT").removesuffix("BUSD") for p in self._configured_pairs
            }
            has_positions = any(b.asset in tracked_bases and b.free > 0 for b in balances)
            if not has_positions:
                logger.info(
                    "Available USDT ($%.2f) below $5 and no open positions — skipping",
                    usdt_free,
                )
                return
            logger.info(
                "Low USDT ($%.2f) but have open positions — LLM may recommend sells",
                usdt_free,
            )

        # ── Wave B: a single CycleState carries the full prompt-context ──
        from halal_trader.core.cycle_pipeline import CycleState, run_stages
        from halal_trader.core.cycle_stages import (
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
            BuildTimeframeStage,
        )

        # Indicators are scoped to symbols we actually have klines for.
        scoped_indicators = {p: indicators_cache.get(p, {}) for p in klines_by_symbol}

        state = CycleState(
            account=account,
            halal_pairs=halal_pairs,
            today_pnl=today_pnl,
            klines_by_symbol=klines_by_symbol,
            indicators_cache=scoped_indicators,
            orderbooks=orderbooks,
            current_prices=current_prices,
        )
        # ── Wave B: one stage list drives the full prompt-context build ──
        # The risk stage may set ``state.halt = True``; ``stop_on_halt``
        # short-circuits the rest of the chain so we don't pay for
        # augmenter work the cycle is about to discard.
        await run_stages(
            state,
            [
                # Use a 24h lookback so the LLM evaluates the *current*
                # strategy on its own merits — a 7-day window can include
                # a now-resolved era of buggy prompts / wrong models that
                # bias the LLM into permanent risk-off mode ("I see 8%
                # win rate, no edge → hold forever"). One day is enough
                # signal for a 24/7 scalper.
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
                AugmentRegimeWithRagStage(getattr(self._hub, "rag", None)),
                AugmentRegimeWithMemoryStage(getattr(self._hub, "regime", None)),
            ],
            bus=self._bus,
            stop_on_halt=True,
        )

        # Push the structured risk state into the hub's runtime view so
        # the dashboard's /api/risk/state can render it. Falls back
        # silently when no runtime view or no risk state is present.
        runtime = getattr(self._hub, "runtime", None)
        rs = state.risk_state
        if runtime is not None and rs is not None:
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
        if state.halt:
            logger.warning(
                "Crypto risk engine halt: %s",
                getattr(rs, "halt_reason", "unspecified"),
            )
            return

        async with stage(
            self._bus,
            "strategy_analyze",
            swallow=False,
            pair_count=len(halal_pairs),
            open_positions=open_position_count,
        ):
            async with tracer.aspan(
                "cycle.strategy_analyze",
                pair_count=len(halal_pairs),
                open_positions=open_position_count,
            ):
                # Build the kwargs dict once — the shadow runner needs
                # the same shape, so duplicating it inline drifts.
                analyze_kwargs = dict(
                    account=account,
                    positions_text=positions_text,
                    halal_pairs=halal_pairs,
                    klines_by_symbol=klines_by_symbol,
                    orderbooks=orderbooks,
                    today_pnl=state.today_pnl,
                    performance_text=state.performance_text,
                    sentiment_text=state.sentiment_text,
                    timeframe_text=state.timeframe_text,
                    ml_signals_text=state.ml_signals_text,
                    regime_text=state.regime_text,
                    active_adjustments=state.active_adjustments,
                    exchange_rules_text=state.exchange_rules_text,
                    indicators_cache=indicators_cache,
                    open_position_count=open_position_count,
                    risk_text=state.risk_text,
                    microstructure_text=state.microstructure_text,
                    news_text=state.news_text,
                )
                plan = await self._strategy.analyze(**analyze_kwargs)

        from halal_trader.core.cycle_stages import ApplyRegimeGateStage

        state.plan = plan
        await run_stages(state, [ApplyRegimeGateStage(self._regime)], bus=self._bus)

        logger.info(
            "Crypto plan: %s | %d buys, %d sells",
            plan.market_outlook[:80] if plan.market_outlook else "N/A",
            len(plan.buys),
            len(plan.sells),
        )

        # Analytics: record this cycle's equity to the shadow ledger
        # and snapshot inputs for replay. Best-effort — never blocks.
        await self._record_cycle_analytics(
            account=account,
            klines_by_symbol=klines_by_symbol,
            indicators_cache=indicators_cache,
            halal_pairs=halal_pairs,
            today_pnl=state.today_pnl,
            sentiment_text=state.sentiment_text,
            regime_text=state.regime_text,
            risk_text=state.risk_text,
            microstructure_text=state.microstructure_text,
        )

        await self._execute_and_notify(
            plan,
            account=account,
            indicators_cache=indicators_cache,
            klines_by_symbol=klines_by_symbol,
            shadow_kwargs=analyze_kwargs,
        )

    # ── Plan post-processing + execution ───────────────────────

    async def _execute_and_notify(
        self,
        plan,
        *,
        account,
        indicators_cache: dict[str, dict],
        klines_by_symbol: dict,
        shadow_kwargs: dict,
    ) -> None:
        """Execute the plan, drive the shadow runner, and notify on fills."""
        results: list = []
        if plan.decisions:
            async with tracer.aspan("cycle.execute_plan", decision_count=len(plan.decisions)):
                results = await self._executor.execute_plan(plan, account=account)
        else:
            logger.info("No crypto trades to execute this cycle")

        # Shadow-runner observation is best-effort and independent of
        # whether the executor produced any fills — the runner records
        # divergence even on hold-only cycles.
        if self._shadow_runner is not None:
            try:
                latest_prices = {
                    pair: (klines[-1].close if klines else 0.0)
                    for pair, klines in klines_by_symbol.items()
                }
                await self._shadow_runner.observe_cycle(
                    cycle_id=cycle_id_var.get() or "cycle-unknown",
                    live_equity=account.total_balance_usdt or 0.0,
                    latest_prices=latest_prices,
                    analyze_kwargs=shadow_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("shadow runner observe_cycle failed: %s", exc)

        # Per-fill bookkeeping must run regardless of shadow runner —
        # the ML retraining loop depends on the indicator snapshot, and
        # the operator depends on the trade notification.
        for r in results:
            logger.info("Crypto execution: %s", r)
            status = r.get("status")
            if status in ("submitted", "filled") and r.get("action") == "buy":
                trade_id = r.get("trade_id")
                symbol = r.get("symbol", "")
                if trade_id and symbol in indicators_cache:
                    try:
                        await self._portfolio.record_indicator_snapshot(
                            trade_id=trade_id,
                            pair=symbol,
                            indicators=indicators_cache[symbol],
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Failed to record indicator snapshot: %s", exc)
            if self._notifier and status in ("submitted", "filled"):
                try:
                    await self._notifier.notify_trade(
                        pair=r.get("symbol", ""),
                        side=r.get("action", ""),
                        quantity=r.get("quantity", 0),
                        price=r.get("price", 0),
                        market="crypto",
                        order_id=str(r.get("order_id", "")),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to send trade notification: %s", exc)

    # ── Prompt-fragment builders ───────────────────────────────
    # Performance / sentiment / timeframe / regime / ML-signals are
    # all driven by the Wave B stage list in ``_run_cycle_impl`` (see
    # :func:`core.cycle_pipeline.run_stages`). The legacy methods that
    # used to live here are gone; their behaviour is preserved by the
    # stage classes under :mod:`core.cycle_stages` plus the per-stage
    # tests in ``tests/test_cycle_stages.py``.

    # ── Private helpers ────────────────────────────────────────

    async def _record_cycle_analytics(
        self,
        *,
        account,
        klines_by_symbol,
        indicators_cache,
        halal_pairs,
        today_pnl,
        sentiment_text,
        regime_text,
        risk_text,
        microstructure_text,
    ) -> None:
        """Per-cycle bookkeeping: shadow ledger + replay store + regime memory.

        Each step is best-effort; failures are debug-logged but never
        bubble up to the cycle loop. Wired here (not in the executor)
        so paper / live / dry-run cycles all populate analytics
        identically.
        """
        cycle_id = cycle_id_var.get() or "cycle-unknown"
        equity = account.total_balance_usdt or 0.0

        # Without a shadow runner, record a placeholder row so the
        # ledger has a per-cycle entry; with one, defer the record
        # to the runner's observe_cycle (driven from after-execute).
        if self._shadow_runner is None:
            try:
                self._hub.shadow.record(
                    cycle_id=cycle_id,
                    live_equity=equity,
                    shadow_equity=equity,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("shadow ledger record failed: %s", exc)

        replay_store = self._replay_store
        if replay_store is not None:
            try:
                from halal_trader.core.replay import (
                    CycleSnapshot,
                    record_snapshot,
                )

                snap = CycleSnapshot.from_inputs(
                    cycle_id=cycle_id,
                    market="crypto",
                    klines_by_symbol=klines_by_symbol,
                    indicators_cache=indicators_cache,
                    halal_pairs=list(halal_pairs),
                    today_pnl=today_pnl,
                    sentiment_text=sentiment_text,
                    regime_text=regime_text,
                    risk_text=risk_text,
                    microstructure_text=microstructure_text,
                    account={
                        "total_balance_usdt": equity,
                        "available_balance_usdt": account.available_balance_usdt,
                        "in_order_usdt": account.in_order_usdt,
                    },
                )
                await record_snapshot(replay_store, snap)
            except Exception as exc:  # noqa: BLE001
                logger.debug("replay snapshot failed: %s", exc)

        # Daily regime snapshot — only on first cycle of the day.
        try:
            from datetime import UTC
            from datetime import datetime as _dt

            today = _dt.now(UTC).date()
            if self._last_regime_snapshot_date != today:
                feats = self._build_regime_features(
                    indicators_cache=indicators_cache,
                    today_pnl=today_pnl,
                    equity=equity,
                )
                regime_mem = self._hub.regime
                if feats is not None and regime_mem is not None:
                    await regime_mem.add_today(feats, today=today)
                    self._last_regime_snapshot_date = today
        except Exception as exc:  # noqa: BLE001
            logger.debug("regime memory snapshot failed: %s", exc)

        # Idle-cash treasury policy log — emit a plan when fully flat.
        try:
            self._log_treasury_plan(account=account)
        except Exception as exc:  # noqa: BLE001
            logger.debug("treasury plan failed: %s", exc)

    def _build_regime_features(self, *, indicators_cache: dict, today_pnl: float, equity: float):
        """Thin wrapper over :func:`ml.regime_memory.build_regime_features`."""
        from halal_trader.ml.regime_memory import build_regime_features

        return build_regime_features(
            indicators_cache=indicators_cache,
            today_pnl=today_pnl,
            equity=equity,
        )

    def _log_treasury_plan(self, *, account) -> None:
        """Emit a treasury plan log line when the bot is fully flat —
        only when the plan changes vs the last cycle, otherwise the same
        line fires every 90s when the bot is idle and clutters the
        operator's view."""
        if account.in_order_usdt > 0 or account.available_balance_usdt < 50:
            return
        try:
            from halal_trader.core.treasury import (
                TreasuryPolicy,
                plan_idle_cash,
            )

            plan = plan_idle_cash(
                cash_balance=account.available_balance_usdt,
                positions_value=account.total_balance_usdt - account.available_balance_usdt,
                current_treasury_value=0.0,
                policy=TreasuryPolicy(),
            )
            if plan.is_noop:
                return
            # Dedupe: round amount to nearest $10 so trivial drift doesn't
            # re-fire the log. Same action+instrument+rounded-amount → skip.
            key = (plan.action, plan.instrument, round(plan.amount_usd / 10) * 10)
            last_key = getattr(self, "_last_treasury_key", None)
            if key == last_key:
                return
            self._last_treasury_key = key  # type: ignore[attr-defined]
            logger.info(
                "treasury: %s $%.2f into %s — %s",
                plan.action,
                plan.amount_usd,
                plan.instrument,
                plan.reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("treasury plan computation failed: %s", exc)

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

    async def _get_tradeable_pairs(self) -> list[str]:
        """Get the intersection of configured pairs and halal-screened pairs.

        Operator-paused pairs (via the dashboard's POST /api/admin/pair/.../pause)
        are filtered out at this layer so a pause takes effect on the
        very next cycle without needing a bot restart.
        """
        max_pairs = self._settings.crypto.max_pairs_per_cycle
        halal_symbols = await self._screener.get_halal_pairs()

        # Pull paused pairs once per cycle.
        paused: set[str] = set()
        try:
            paused = await self._portfolio.get_paused_pairs()
        except Exception as e:
            logger.debug("Could not fetch paused pairs: %s", e)

        if not halal_symbols:
            logger.info("No halal cache — using configured pairs: %s", self._configured_pairs)
            return [p for p in self._configured_pairs if p.upper() not in paused][:max_pairs]

        halal_set = {s.upper() for s in halal_symbols}
        tradeable = []
        for pair in self._configured_pairs:
            upper_pair = pair.upper()
            if upper_pair in paused:
                logger.info("Pair %s is paused by operator — skipping", pair)
                continue
            for suffix in ("USDT", "BUSD"):
                if upper_pair.endswith(suffix):
                    base = upper_pair.removesuffix(suffix)
                    break
            else:
                base = upper_pair

            if upper_pair in halal_set or base in halal_set:
                tradeable.append(pair)

        seen: set[str] = set()
        unique: list[str] = []
        for p in tradeable:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        if unique:
            return unique[:max_pairs]
        return [p for p in self._configured_pairs if p.upper() not in paused][:max_pairs]

    async def _fetch_klines(self, pairs: list[str]) -> dict[str, list[Kline]]:
        """Fetch klines from WebSocket buffer or REST fallback (throttled)."""
        sem = asyncio.Semaphore(5)

        async def _get_klines(pair: str) -> tuple[str, list[Kline]]:
            if self._ws:
                ws_klines = self._ws.get_klines(pair, limit=100)
                if len(ws_klines) >= 20:
                    return pair, ws_klines
            async with sem:
                klines = await self._broker.get_klines(pair, interval="1m", limit=100)
                return pair, klines

        results = await asyncio.gather(*[_get_klines(p) for p in pairs], return_exceptions=True)
        klines_by_symbol: dict[str, list[Kline]] = {}
        for result in results:
            if isinstance(result, Exception):
                if isinstance(result, BinanceAPIException) and result.code == -1003:
                    logger.warning("Rate limited fetching klines, backing off")
                    await asyncio.sleep(30)
                else:
                    logger.debug("Failed to get klines: %s", result)
                continue
            pair, klines = result
            klines_by_symbol[pair] = klines
        return klines_by_symbol

    async def _fetch_orderbooks(self, pairs: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch order book depth for each pair (throttled)."""
        sem = asyncio.Semaphore(5)

        async def _get_book(pair: str) -> tuple[str, dict[str, Any]]:
            async with sem:
                book = await self._broker.get_order_book(pair, limit=10)
                return pair, book

        results = await asyncio.gather(*[_get_book(p) for p in pairs], return_exceptions=True)
        orderbooks: dict[str, dict[str, Any]] = {}
        for result in results:
            if isinstance(result, Exception):
                if isinstance(result, BinanceAPIException) and result.code == -1003:
                    logger.warning("Rate limited fetching orderbooks, backing off")
                    await asyncio.sleep(30)
                else:
                    logger.debug("Failed to get order book: %s", result)
                continue
            pair, book = result
            orderbooks[pair] = book
        return orderbooks

    # Microstructure + news are now driven by ``BuildMicrostructureStage``
    # and ``BuildNewsStage`` from :mod:`core.cycle_stages` (called via
    # the secondary stage list in ``_run_cycle_impl``). The crypto-only
    # microstructure augmentations (basis + whale flows) chain after.
