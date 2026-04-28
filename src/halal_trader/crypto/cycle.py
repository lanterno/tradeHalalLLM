"""Crypto trading cycle — gathers data, computes indicators, analyzes, and executes."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

from binance import BinanceAPIException

from halal_trader.config import get_settings
from halal_trader.core.cycle import BaseCycleService
from halal_trader.core.insights_hub import hub as insights_hub
from halal_trader.core.observability import cycle_id_var
from halal_trader.core.tracing import tracer
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.indicators import compute_all
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.regime import MarketRegime
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
    ) -> None:
        super().__init__(alerts=alerts, engine=engine)
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

        performance_text = await self._build_performance_text()
        sentiment_text = await self._build_sentiment_text(halal_pairs)
        timeframe_text = await self._build_timeframe_text(halal_pairs)
        regime_text = self._build_regime_text(klines_by_symbol, indicators_cache)
        ml_signals_text = self._build_ml_signals_text(klines_by_symbol, indicators_cache)

        risk_text, halt = self._evaluate_portfolio_risk(
            account=account,
            klines_by_symbol=klines_by_symbol,
            indicators_cache=indicators_cache,
            open_trades=open_trades,
            current_prices=current_prices,
        )
        if halt:
            return

        active_adjustments = ""
        if self._self_review:
            active_adjustments = self._self_review.format_adjustments_for_prompt()

        exchange_rules_text = self._broker.format_filters_for_prompt()

        microstructure_text = self._build_microstructure_text(orderbooks)
        microstructure_text = await self._augment_with_basis(
            microstructure_text, halal_pairs, klines_by_symbol
        )
        microstructure_text = await self._augment_with_whale_flows(
            microstructure_text, klines_by_symbol
        )
        news_text = self._build_news_text(halal_pairs)

        regime_text = await self._augment_regime_with_rag(
            regime_text, indicators_cache, sentiment_text
        )
        regime_text = await self._augment_regime_with_memory(
            regime_text,
            indicators_cache=indicators_cache,
            today_pnl=today_pnl,
            equity=account.total_balance_usdt or 0.0,
        )

        async with tracer.aspan(
            "cycle.strategy_analyze",
            pair_count=len(halal_pairs),
            open_positions=open_position_count,
        ):
            plan = await self._strategy.analyze(
                account=account,
                positions_text=positions_text,
                halal_pairs=halal_pairs,
                klines_by_symbol=klines_by_symbol,
                orderbooks=orderbooks,
                today_pnl=today_pnl,
                performance_text=performance_text,
                sentiment_text=sentiment_text,
                timeframe_text=timeframe_text,
                ml_signals_text=ml_signals_text,
                regime_text=regime_text,
                active_adjustments=active_adjustments,
                exchange_rules_text=exchange_rules_text,
                indicators_cache=indicators_cache,
                open_position_count=open_position_count,
                risk_text=risk_text,
                microstructure_text=microstructure_text,
                news_text=news_text,
            )

        self._apply_regime_gate(plan, klines_by_symbol, indicators_cache)

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
            today_pnl=today_pnl,
            sentiment_text=sentiment_text,
            regime_text=regime_text,
            risk_text=risk_text,
            microstructure_text=microstructure_text,
        )

        await self._execute_and_notify(
            plan,
            account=account,
            indicators_cache=indicators_cache,
            klines_by_symbol=klines_by_symbol,
            shadow_kwargs=dict(
                account=account,
                positions_text=positions_text,
                halal_pairs=halal_pairs,
                klines_by_symbol=klines_by_symbol,
                orderbooks=orderbooks,
                today_pnl=today_pnl,
                performance_text=performance_text,
                sentiment_text=sentiment_text,
                timeframe_text=timeframe_text,
                ml_signals_text=ml_signals_text,
                regime_text=regime_text,
                active_adjustments=active_adjustments,
                exchange_rules_text=exchange_rules_text,
                indicators_cache=indicators_cache,
                open_position_count=open_position_count,
                risk_text=risk_text,
                microstructure_text=microstructure_text,
                news_text=news_text,
            ),
        )

    # ── Plan post-processing + execution ───────────────────────

    def _apply_regime_gate(
        self,
        plan,
        klines_by_symbol: dict,
        indicators_cache: dict[str, dict],
    ) -> None:
        """Strip BUYs in confirmed downtrend pairs from the plan in place."""
        if not (self._regime and plan.buys):
            return
        downtrend_pairs: set[str] = set()
        for pair in klines_by_symbol:
            indicators = indicators_cache.get(pair, {})
            if not indicators or "error" in indicators:
                continue
            regime, confidence, _ = self._regime.detect(indicators)
            if regime == MarketRegime.TRENDING_DOWN and confidence >= 0.6:
                downtrend_pairs.add(pair)
        if not downtrend_pairs:
            return
        blocked = [d for d in plan.buys if d.symbol in downtrend_pairs]
        if not blocked:
            return
        for d in blocked:
            plan.decisions.remove(d)
        logger.warning(
            "Regime gate blocked %d BUY(s) in downtrend: %s",
            len(blocked),
            ", ".join(d.symbol for d in blocked),
        )

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

        if self._shadow_runner is None:
            if not plan.decisions:
                logger.info("No crypto trades to execute this cycle")
            return

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

        for r in results:
            logger.info("Crypto execution: %s", r)
            if r.get("status") in ("submitted", "filled") and r.get("action") == "buy":
                trade_id = r.get("trade_id")
                symbol = r.get("symbol", "")
                if trade_id and symbol in indicators_cache:
                    try:
                        await self._portfolio._repo.record_indicator_snapshot(
                            trade_id=trade_id,
                            pair=symbol,
                            indicators=indicators_cache[symbol],
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Failed to record indicator snapshot: %s", exc)
            if self._notifier and r.get("status") in ("submitted", "filled"):
                try:
                    await self._notifier.notify_trade(
                        pair=r.get("symbol", ""),
                        side=r.get("action", ""),
                        quantity=r.get("quantity", 0),
                        price=r.get("price", 0),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to send trade notification: %s", exc)

    # ── Prompt-fragment builders (each best-effort) ────────────

    async def _build_performance_text(self) -> str:
        if not self._analytics:
            return ""
        try:
            stats = await self._analytics.compute_stats(lookback_days=7)
            return self._analytics.format_for_prompt(stats)
        except Exception as e:  # noqa: BLE001
            logger.debug("Performance stats unavailable: %s", e)
            return ""

    async def _build_sentiment_text(self, halal_pairs: list[str]) -> str:
        """CryptoPanic + Reddit composite sentiment + mention-velocity."""
        sentiment_text = ""
        if self._sentiment and self._sentiment.enabled:
            try:
                from halal_trader.sentiment.scoring import format_sentiment_for_prompt

                signals = self._sentiment.latest_signals
                if signals:
                    sentiment_text = format_sentiment_for_prompt(signals)
                    if self._notifier:
                        for pair, sig in signals.items():
                            if sig.buzz >= 3.0:
                                try:
                                    await self._notifier.notify_buzz(pair, sig.buzz, sig.score)
                                except Exception as exc:  # noqa: BLE001
                                    logger.debug("Failed to send buzz alert: %s", exc)
            except Exception as e:  # noqa: BLE001
                logger.debug("Sentiment data unavailable: %s", e)

        if self._reddit_fetcher is not None:
            try:
                from halal_trader.sentiment.velocity import (
                    compute_velocity,
                    format_velocity_for_prompt,
                )

                bases = sorted(
                    {p.upper().removesuffix("USDT").removesuffix("BUSD") for p in halal_pairs}
                )
                mentions = await self._reddit_fetcher.fetch_for_symbols(bases)
                if mentions:
                    velocity = compute_velocity(mentions)
                    insights_hub.velocity = velocity
                    velocity_block = format_velocity_for_prompt(velocity)
                    if velocity_block:
                        sentiment_text = (
                            sentiment_text + "\n\n" + velocity_block
                            if sentiment_text
                            else velocity_block
                        )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Reddit velocity fetch failed: %s", exc)
        return sentiment_text

    async def _build_timeframe_text(self, halal_pairs: list[str]) -> str:
        if not self._timeframes:
            return ""
        try:
            from halal_trader.crypto.timeframes import format_timeframes_for_prompt

            tf_results = await self._timeframes.analyze(halal_pairs)
            if tf_results:
                return format_timeframes_for_prompt(tf_results)
        except Exception as e:  # noqa: BLE001
            logger.debug("Multi-timeframe analysis unavailable: %s", e)
        return ""

    def _build_regime_text(self, klines_by_symbol: dict, indicators_cache: dict[str, dict]) -> str:
        if not self._regime:
            return ""
        try:
            from halal_trader.crypto.regime import format_regime_for_prompt

            regimes = {}
            for pair in klines_by_symbol:
                indicators = indicators_cache.get(pair, {})
                if not indicators or "error" in indicators:
                    continue
                regimes[pair] = self._regime.detect(indicators)
            if regimes:
                return format_regime_for_prompt(regimes)
        except Exception as e:  # noqa: BLE001
            logger.debug("Regime detection unavailable: %s", e)
        return ""

    def _build_ml_signals_text(
        self, klines_by_symbol: dict, indicators_cache: dict[str, dict]
    ) -> str:
        if not (self._ml_forecaster or self._ml_anomaly or self._ml_signal):
            return ""
        try:
            from halal_trader.ml.anomaly import format_ml_signals_for_prompt
            from halal_trader.ml.forecaster import format_forecasts_for_prompt

            forecasts: dict = {}
            anomalies: dict = {}
            ml_confidence: dict = {}

            for pair, klines in klines_by_symbol.items():
                if self._ml_forecaster and len(klines) >= 20:
                    closes = [k.close for k in klines]
                    fc = self._ml_forecaster.forecast(pair, closes)
                    if fc:
                        forecasts[pair] = fc

                indicators = indicators_cache.get(pair, {})
                if not indicators or "error" in indicators:
                    continue
                if self._ml_anomaly:
                    self._ml_anomaly.add_sample(indicators)
                    anomalies[pair] = self._ml_anomaly.detect(indicators)
                if self._ml_signal:
                    conf = self._ml_signal.predict_confidence(indicators)
                    if conf is not None:
                        ml_confidence[pair] = conf

            forecasts_text = format_forecasts_for_prompt(forecasts)
            return format_ml_signals_for_prompt(
                forecasts_text, anomalies or None, ml_confidence or None
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("ML signals unavailable: %s", e)
        return ""

    def _evaluate_portfolio_risk(
        self,
        *,
        account,
        klines_by_symbol: dict,
        indicators_cache: dict[str, dict],
        open_trades,
        current_prices: dict[str, float],
    ) -> tuple[str, bool]:
        """Run the portfolio-risk engine. Returns (risk_text, should_halt)."""
        if not self._risk_engine:
            return "", False
        try:
            open_pos_value: dict[str, float] = {}
            unrealized_pnl: dict[str, float] = {}
            for t in open_trades or []:
                price = current_prices.get(t.pair) or self._broker.get_cached_price(t.pair)
                if price and t.entry_price:
                    open_pos_value[t.pair] = t.quantity * price
                    unrealized_pnl[t.pair] = (price - t.entry_price) * t.quantity

            risk_state = self._risk_engine.evaluate(
                klines_by_symbol=klines_by_symbol,
                indicators_cache=indicators_cache,
                open_positions_value=open_pos_value,
                unrealized_pnl=unrealized_pnl,
                total_equity=account.total_balance_usdt,
            )
            risk_text = self._risk_engine.format_for_prompt(risk_state)

            try:
                from halal_trader.web.app import app_state as _web_state

                _web_state["risk_state"] = {
                    "is_halted": risk_state.is_halted,
                    "halt_reason": risk_state.halt_reason,
                    "portfolio_heat_pct": getattr(risk_state, "portfolio_heat_pct", None),
                    "drawdown_pct": getattr(risk_state, "drawdown_pct", None),
                    "avg_correlation": getattr(risk_state, "avg_correlation", None),
                    "summary": risk_text,
                }
            except Exception:  # noqa: BLE001
                pass

            if risk_state.is_halted:
                logger.warning("Risk engine halt: %s", risk_state.halt_reason)
                return risk_text, True
            return risk_text, False
        except Exception as e:  # noqa: BLE001
            logger.debug("Risk engine evaluation failed: %s", e)
            return "", False

    async def _augment_regime_with_rag(
        self,
        regime_text: str,
        indicators_cache: dict[str, dict],
        sentiment_text: str,
    ) -> str:
        """Append RAG hits over the closed-trade rationale store."""
        try:
            rag_store = insights_hub.rag
            if rag_store is None:
                return regime_text
            from halal_trader.core.llm.rag import format_rag_for_prompt

            size = await rag_store.size()
            if size <= 0:
                return regime_text
            rag_query = self._build_rag_query(
                indicators_cache=indicators_cache,
                sentiment_text=sentiment_text,
                regime_text=regime_text,
            )
            hits = await rag_store.query(rag_query, k=5, min_similarity=0.0)
            rag_text = format_rag_for_prompt(hits)
            if rag_text:
                return regime_text + "\n\n" + rag_text if regime_text else rag_text
        except Exception as exc:  # noqa: BLE001
            logger.debug("RAG query failed: %s", exc)
        return regime_text

    async def _augment_regime_with_memory(
        self,
        regime_text: str,
        *,
        indicators_cache: dict[str, dict],
        today_pnl: float,
        equity: float,
    ) -> str:
        """Append the top-K analogous past regimes to the regime block."""
        try:
            from halal_trader.ml.regime_memory import format_for_prompt

            features = self._build_regime_features(
                indicators_cache=indicators_cache,
                today_pnl=today_pnl,
                equity=equity,
            )
            regime_mem = insights_hub.regime
            if features is None or regime_mem is None:
                return regime_text
            if await regime_mem.size() <= 0:
                return regime_text
            hits = await regime_mem.query(features, k=3)
            analog_text = format_for_prompt(features, hits)
            if analog_text and "No analogous" not in analog_text:
                return regime_text + "\n\n" + analog_text if regime_text else analog_text
        except Exception as exc:  # noqa: BLE001
            logger.debug("regime memory query failed: %s", exc)
        return regime_text

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
                insights_hub.shadow.record(
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
                regime_mem = insights_hub.regime
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

    def _build_rag_query(
        self,
        *,
        indicators_cache: dict,
        sentiment_text: str,
        regime_text: str,
    ) -> str:
        """Render a short text query summarising the current setup.

        The query is hashed by HashingEmbedder, so we want token overlap
        with what the LLM typically writes in its ``reasoning`` field —
        keep it terse, technical-language-flavoured.
        """
        parts: list[str] = []
        for pair, inds in (indicators_cache or {}).items():
            if not inds or "error" in inds:
                continue
            rsi = inds.get("rsi_14")
            macd = inds.get("macd_histogram")
            bb = inds.get("bb_position")
            ev: list[str] = [pair]
            if rsi is not None:
                if rsi < 35:
                    ev.append("rsi oversold")
                elif rsi > 65:
                    ev.append("rsi overbought")
                else:
                    ev.append("rsi neutral")
            if macd is not None:
                ev.append("macd bullish" if macd > 0 else "macd bearish")
            if bb is not None:
                if bb < 0.2:
                    ev.append("bb lower")
                elif bb > 0.8:
                    ev.append("bb upper")
            parts.append(" ".join(ev))
        if regime_text:
            parts.append(regime_text[:80])
        if sentiment_text:
            parts.append(sentiment_text[:80])
        return " | ".join(parts)[:600]

    async def _augment_with_whale_flows(
        self,
        microstructure_text: str,
        klines_by_symbol: dict,
    ) -> str:
        """Pull on-chain whale-flow signals from Etherscan and append them
        to the microstructure block. Best-effort — silently degrade when
        the source is unconfigured or fails.

        The watched ERC-20s (USDT, USDC, DAI, WETH) are universe-wide
        signals: their flows apply to every pair regardless of which
        symbols the bot is currently trading.
        """
        if self._whale_flow_source is None:
            return microstructure_text
        try:
            from halal_trader.crypto.onchain import (
                TOKENS,
                format_whale_flows_for_prompt,
            )

            prices: dict[str, float] = {}
            eth_klines = klines_by_symbol.get("ETHUSDT") or []
            if eth_klines:
                prices["WETH"] = float(eth_klines[-1].close)

            symbols_to_watch = list(TOKENS.keys())
            signals = await self._whale_flow_source.fetch(symbols_to_watch, prices=prices)
            insights_hub.whale_flows = signals
            block = format_whale_flows_for_prompt(signals)
            if not block:
                return microstructure_text
            return microstructure_text + "\n\n" + block if microstructure_text else block
        except Exception as exc:  # noqa: BLE001
            logger.debug("whale-flow augmentation failed: %s", exc)
            return microstructure_text

    async def _augment_with_basis(
        self,
        microstructure_text: str,
        halal_pairs,
        klines_by_symbol,
    ) -> str:
        """Compute spot-perp basis features and append them to the
        microstructure block. Best-effort — skip silently if the broker
        doesn't expose ``get_funding_signal`` or any pair query fails.
        """
        if not halal_pairs:
            return microstructure_text
        if not hasattr(self._broker, "get_funding_signal"):
            return microstructure_text
        try:
            from halal_trader.crypto.basis import format_basis_for_prompt
        except Exception:  # noqa: BLE001
            return microstructure_text

        features = {}
        for pair in halal_pairs:
            try:
                sig = await self._broker.get_funding_signal(pair)
            except Exception:  # noqa: BLE001
                continue
            if not sig:
                continue
            try:
                spot_klines = klines_by_symbol.get(pair) or []
                spot_price = (
                    spot_klines[-1].close if spot_klines else self._broker.get_cached_price(pair)
                )
                if not spot_price:
                    continue
                features[pair] = insights_hub.basis.observe(
                    pair=pair,
                    spot_price=float(spot_price),
                    perp_price=float(sig.get("mark_price", spot_price)),
                    funding_rate_pct=float(sig.get("funding_rate", 0.0)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("basis observe failed for %s: %s", pair, exc)

        if not features:
            return microstructure_text
        basis_text = format_basis_for_prompt(features)
        if not basis_text:
            return microstructure_text
        return microstructure_text + "\n\n" + basis_text if microstructure_text else basis_text

    def _build_regime_features(self, *, indicators_cache: dict, today_pnl: float, equity: float):
        """Aggregate per-pair indicators into a daily ``RegimeFeatures``."""
        from halal_trader.ml.regime_memory import RegimeFeatures

        if not indicators_cache:
            return None
        atrs: list[float] = []
        rsis: list[float] = []
        for inds in indicators_cache.values():
            if not inds or "error" in inds:
                continue
            price = inds.get("current_price") or 0
            atr = inds.get("atr_14") or 0
            if price > 0 and atr > 0:
                atrs.append(atr / price)
            rsi = inds.get("rsi_14")
            if rsi is not None:
                rsis.append(rsi)
        if not atrs and not rsis:
            return None
        avg_atr = sum(atrs) / len(atrs) if atrs else 0.0
        avg_rsi = sum(rsis) / len(rsis) if rsis else 50.0
        # Conservative defaults for fields we don't track day-of-cycle yet.
        return RegimeFeatures(
            volatility=avg_atr,
            trend=0.0,
            breadth=0.0,
            sentiment=0.0,
            drawdown=0.0,
            volume_ratio=1.0,
            correlation=0.0,
            realized_return_1d=(today_pnl / equity) if equity else 0.0,
            rsi=avg_rsi,
            spread_bps=0.0,
        )

    def _log_treasury_plan(self, *, account) -> None:
        """Emit a treasury plan log line when the bot is fully flat."""
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
            if not plan.is_noop:
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
            paused = await self._portfolio._repo.get_paused_pairs()
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

    def _build_microstructure_text(self, orderbooks: dict[str, dict[str, Any]]) -> str:
        """Format depth-imbalance / spread features per pair for the LLM.

        Funding signal needs an extra REST call (perp endpoint) and we
        already throttle aggressively per-cycle, so we lean on the
        already-fetched spot orderbook here. A future task can fan
        funding fetches alongside orderbook fetches if signal value
        justifies the cost.
        """
        from halal_trader.crypto.microstructure import (
            format_microstructure_for_prompt,
            orderbook_features,
        )

        lines: list[str] = []
        for pair, book in sorted(orderbooks.items()):
            feats = orderbook_features(book)
            line = format_microstructure_for_prompt(pair=pair, book=feats)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _build_news_text(self, halal_pairs: list[str]) -> str:
        """Pull a snapshot from the bounded RecentNewsFeed, if one is wired."""
        if self._news_feed is None:
            return ""
        try:
            from halal_trader.sentiment.feed import format_news_for_prompt

            events = self._news_feed.snapshot()
            return format_news_for_prompt(events, pair_filter=halal_pairs)
        except Exception as e:
            logger.debug("News feed unavailable: %s", e)
            return ""
