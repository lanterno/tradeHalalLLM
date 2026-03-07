"""Crypto trading cycle — gathers data, computes indicators, analyzes, and executes."""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)

# Maximum pairs to analyze per cycle
_MAX_PAIRS_PER_CYCLE = 10


class CryptoCycleService:
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
    ) -> None:
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

    async def run_cycle(self) -> None:
        """Execute one complete crypto trading cycle.

        Steps:
        1. Check if daily loss limit has been breached.
        2. Determine tradeable halal pairs.
        3. Gather market data (klines from WS buffer or REST fallback).
        4. Fetch order books.
        5. Run LLM strategy analysis.
        6. Execute resulting trades.
        """
        logger.info("=== CRYPTO TRADING CYCLE ===")
        try:
            # 1. Check daily loss limit
            if await self._portfolio.should_halt_trading():
                logger.warning("Crypto daily loss limit reached — halting trades")
                return

            # 2. Get halal pairs (intersect configured pairs with screened halal ones)
            halal_pairs = await self._get_tradeable_pairs()
            if not halal_pairs:
                logger.warning("No halal crypto pairs available, skipping cycle")
                return

            # 3. Gather kline data
            klines_by_symbol = await self._fetch_klines(halal_pairs)

            # 4. Fetch order books
            orderbooks = await self._fetch_orderbooks(halal_pairs)

            # 5. Gather portfolio state
            account = await self._broker.get_account()
            balances = await self._broker.get_balances()
            positions_text = self._portfolio.format_positions_for_prompt(
                balances, configured_pairs=self._configured_pairs
            )
            today_pnl = await self._portfolio.get_current_pnl()

            # 5b. Skip LLM call if USDT balance is too low to place any order
            usdt_free = account.usdt_free
            if usdt_free < 5.0:
                logger.info(
                    "Available USDT ($%.2f) below $5 minimum — skipping LLM analysis",
                    usdt_free,
                )
                return

            # 5c. Compute performance stats for the LLM prompt
            performance_text = ""
            if self._analytics:
                try:
                    stats = await self._analytics.compute_stats(lookback_days=7)
                    performance_text = self._analytics.format_for_prompt(stats)
                except Exception as e:
                    logger.debug("Performance stats unavailable: %s", e)

            # 5d. Gather sentiment data
            sentiment_text = ""
            if self._sentiment and self._sentiment.enabled:
                try:
                    from halal_trader.sentiment.scoring import format_sentiment_for_prompt
                    signals = self._sentiment.latest_signals
                    if signals:
                        sentiment_text = format_sentiment_for_prompt(signals)

                        # Send buzz alerts via Telegram
                        if self._notifier:
                            for pair, sig in signals.items():
                                if sig.buzz >= 3.0:
                                    try:
                                        await self._notifier.notify_buzz(
                                            pair, sig.buzz, sig.score
                                        )
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.debug("Sentiment data unavailable: %s", e)

            # 5e. Multi-timeframe analysis
            timeframe_text = ""
            if self._timeframes:
                try:
                    from halal_trader.crypto.timeframes import format_timeframes_for_prompt
                    tf_results = await self._timeframes.analyze(halal_pairs)
                    if tf_results:
                        timeframe_text = format_timeframes_for_prompt(tf_results)
                except Exception as e:
                    logger.debug("Multi-timeframe analysis unavailable: %s", e)

            # 5f. Market regime detection
            regime_text = ""
            if self._regime:
                try:
                    from halal_trader.crypto.indicators import compute_all
                    from halal_trader.crypto.regime import format_regime_for_prompt
                    regimes = {}
                    for pair, klines in klines_by_symbol.items():
                        if len(klines) >= 30:
                            indicators = compute_all(klines)
                            if "error" not in indicators:
                                regimes[pair] = self._regime.detect(indicators)
                    if regimes:
                        regime_text = format_regime_for_prompt(regimes)
                except Exception as e:
                    logger.debug("Regime detection unavailable: %s", e)

            # 5g. ML model signals
            ml_signals_text = ""
            if self._ml_forecaster or self._ml_anomaly or self._ml_signal:
                try:
                    from halal_trader.crypto.indicators import compute_all
                    from halal_trader.ml.anomaly import format_ml_signals_for_prompt
                    from halal_trader.ml.forecaster import format_forecasts_for_prompt

                    forecasts = {}
                    anomalies = {}
                    ml_confidence = {}

                    for pair, klines in klines_by_symbol.items():
                        if self._ml_forecaster and len(klines) >= 20:
                            closes = [k.close for k in klines]
                            fc = self._ml_forecaster.forecast(pair, closes)
                            if fc:
                                forecasts[pair] = fc

                        if len(klines) >= 30:
                            indicators = compute_all(klines)
                            if "error" not in indicators:
                                if self._ml_anomaly:
                                    self._ml_anomaly.add_sample(indicators)
                                    anomalies[pair] = self._ml_anomaly.detect(indicators)
                                if self._ml_signal:
                                    conf = self._ml_signal.predict_confidence(indicators)
                                    if conf is not None:
                                        ml_confidence[pair] = conf

                    forecasts_text = format_forecasts_for_prompt(forecasts)
                    ml_signals_text = format_ml_signals_for_prompt(
                        forecasts_text, anomalies or None, ml_confidence or None
                    )
                except Exception as e:
                    logger.debug("ML signals unavailable: %s", e)

            # 5h. Self-improvement adjustments
            active_adjustments = ""
            if self._self_review:
                active_adjustments = self._self_review.format_adjustments_for_prompt()

            # 6. LLM analysis
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
            )

            logger.info(
                "Crypto plan: %s | %d buys, %d sells",
                plan.market_outlook[:80] if plan.market_outlook else "N/A",
                len(plan.buys),
                len(plan.sells),
            )

            # 7. Execute decisions
            if plan.decisions:
                results = await self._executor.execute_plan(plan)
                for r in results:
                    logger.info("Crypto execution: %s", r)
                    if self._notifier and r.get("status") == "submitted":
                        try:
                            await self._notifier.notify_trade(
                                pair=r.get("symbol", ""),
                                side=r.get("action", ""),
                                quantity=r.get("quantity", 0),
                                price=r.get("price", 0),
                            )
                        except Exception:
                            pass
            else:
                logger.info("No crypto trades to execute this cycle")

        except Exception as e:
            logger.error("Crypto trading cycle failed: %s", e, exc_info=True)

    # ── Private helpers ────────────────────────────────────────

    async def _get_tradeable_pairs(self) -> list[str]:
        """Get the intersection of configured pairs and halal-screened pairs."""
        halal_symbols = await self._screener.get_halal_pairs()

        if not halal_symbols:
            # If no cache yet, use configured pairs (first run)
            logger.info("No halal cache — using configured pairs: %s", self._configured_pairs)
            return self._configured_pairs[:_MAX_PAIRS_PER_CYCLE]

        halal_set = {s.upper() for s in halal_symbols}
        tradeable = []
        for pair in self._configured_pairs:
            upper_pair = pair.upper()
            # Extract base asset by removing known quote suffixes
            for suffix in ("USDT", "BUSD"):
                if upper_pair.endswith(suffix):
                    base = upper_pair.removesuffix(suffix)
                    break
            else:
                base = upper_pair

            if upper_pair in halal_set or base in halal_set:
                tradeable.append(pair)

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for p in tradeable:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        if unique:
            return unique[:_MAX_PAIRS_PER_CYCLE]
        return self._configured_pairs[:_MAX_PAIRS_PER_CYCLE]

    async def _fetch_klines(self, pairs: list[str]) -> dict[str, list[Kline]]:
        """Fetch klines from WebSocket buffer or REST fallback."""
        klines_by_symbol: dict[str, list[Kline]] = {}

        for pair in pairs:
            # Try WebSocket buffer first
            if self._ws:
                ws_klines = self._ws.get_klines(pair, limit=100)
                if len(ws_klines) >= 20:
                    klines_by_symbol[pair] = ws_klines
                    continue

            # REST fallback
            try:
                klines = await self._broker.get_klines(pair, interval="1m", limit=100)
                klines_by_symbol[pair] = klines
            except Exception as e:
                logger.debug("Failed to get klines for %s: %s", pair, e)

        return klines_by_symbol

    async def _fetch_orderbooks(self, pairs: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch order book depth for each pair."""
        orderbooks: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            try:
                book = await self._broker.get_order_book(pair, limit=10)
                orderbooks[pair] = book
            except Exception as e:
                logger.debug("Failed to get order book for %s: %s", pair, e)
        return orderbooks
