"""Crypto trading cycle — gathers data, computes indicators, analyzes, and executes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from binance import BinanceAPIException

from halal_trader.config import get_settings
from halal_trader.core.cycle import BaseCycleService
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
        alerts=None,
    ) -> None:
        super().__init__(alerts=alerts)
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
        self._consecutive_flat_skips = 0
        self._settings = get_settings()

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
            if self._consecutive_flat_skips < self._settings.crypto_max_consecutive_flat_skips:
                logger.info(
                    "All pairs flat — skipping LLM analysis (%d/%d)",
                    self._consecutive_flat_skips,
                    self._settings.crypto_max_consecutive_flat_skips,
                )
                return
            logger.info(
                "All pairs flat but reached max consecutive skips (%d) — forcing LLM",
                self._consecutive_flat_skips,
            )

        self._consecutive_flat_skips = 0

        orderbooks = await self._fetch_orderbooks(halal_pairs)

        account = await self._broker.get_account()
        balances = await self._broker.get_balances()

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

        performance_text = ""
        if self._analytics:
            try:
                stats = await self._analytics.compute_stats(lookback_days=7)
                performance_text = self._analytics.format_for_prompt(stats)
            except Exception as e:
                logger.debug("Performance stats unavailable: %s", e)

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
                                except Exception as exc:
                                    logger.debug("Failed to send buzz alert: %s", exc)
            except Exception as e:
                logger.debug("Sentiment data unavailable: %s", e)

        timeframe_text = ""
        if self._timeframes:
            try:
                from halal_trader.crypto.timeframes import format_timeframes_for_prompt

                tf_results = await self._timeframes.analyze(halal_pairs)
                if tf_results:
                    timeframe_text = format_timeframes_for_prompt(tf_results)
            except Exception as e:
                logger.debug("Multi-timeframe analysis unavailable: %s", e)

        regime_text = ""
        if self._regime:
            try:
                from halal_trader.crypto.regime import format_regime_for_prompt

                regimes = {}
                for pair in klines_by_symbol:
                    indicators = indicators_cache.get(pair, {})
                    if not indicators or "error" in indicators:
                        continue
                    regimes[pair] = self._regime.detect(indicators)
                if regimes:
                    regime_text = format_regime_for_prompt(regimes)
            except Exception as e:
                logger.debug("Regime detection unavailable: %s", e)

        ml_signals_text = ""
        if self._ml_forecaster or self._ml_anomaly or self._ml_signal:
            try:
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
                ml_signals_text = format_ml_signals_for_prompt(
                    forecasts_text, anomalies or None, ml_confidence or None
                )
            except Exception as e:
                logger.debug("ML signals unavailable: %s", e)

        risk_text = ""
        if self._risk_engine:
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

                if risk_state.is_halted:
                    logger.warning("Risk engine halt: %s", risk_state.halt_reason)
                    return
            except Exception as e:
                logger.debug("Risk engine evaluation failed: %s", e)

        active_adjustments = ""
        if self._self_review:
            active_adjustments = self._self_review.format_adjustments_for_prompt()

        exchange_rules_text = self._broker.format_filters_for_prompt()

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
        )

        if self._regime and plan.buys:
            downtrend_pairs: set[str] = set()
            for pair in klines_by_symbol:
                indicators = indicators_cache.get(pair, {})
                if not indicators or "error" in indicators:
                    continue
                regime, confidence, _ = self._regime.detect(indicators)
                if regime == MarketRegime.TRENDING_DOWN and confidence >= 0.6:
                    downtrend_pairs.add(pair)

            if downtrend_pairs:
                blocked = [d for d in plan.buys if d.symbol in downtrend_pairs]
                if blocked:
                    for d in blocked:
                        plan.decisions.remove(d)
                    logger.warning(
                        "Regime gate blocked %d BUY(s) in downtrend: %s",
                        len(blocked),
                        ", ".join(d.symbol for d in blocked),
                    )

        logger.info(
            "Crypto plan: %s | %d buys, %d sells",
            plan.market_outlook[:80] if plan.market_outlook else "N/A",
            len(plan.buys),
            len(plan.sells),
        )

        if plan.decisions:
            results = await self._executor.execute_plan(plan, account=account)
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
                        except Exception as exc:
                            logger.debug("Failed to record indicator snapshot: %s", exc)
                if self._notifier and r.get("status") in ("submitted", "filled"):
                    try:
                        await self._notifier.notify_trade(
                            pair=r.get("symbol", ""),
                            side=r.get("action", ""),
                            quantity=r.get("quantity", 0),
                            price=r.get("price", 0),
                        )
                    except Exception as exc:
                        logger.debug("Failed to send trade notification: %s", exc)
        else:
            logger.info("No crypto trades to execute this cycle")

    # ── Private helpers ────────────────────────────────────────

    def _should_skip_llm(self, indicators_cache: dict[str, dict]) -> bool:
        """Skip LLM if all pairs are flat with no directional signal."""
        if not indicators_cache:
            return True
        s = self._settings
        for symbol, indicators in indicators_cache.items():
            if indicators.get("error"):
                continue
            price_change_5m = abs(indicators.get("price_change_5m", 0))
            rsi = indicators.get("rsi_14", 50)
            vol_ratio = indicators.get("volume_ratio", 1.0)
            if (
                price_change_5m > s.crypto_flat_price_threshold
                or rsi < s.crypto_flat_rsi_lower
                or rsi > s.crypto_flat_rsi_upper
                or vol_ratio > s.crypto_flat_vol_threshold
            ):
                return False
        return True

    async def _get_tradeable_pairs(self) -> list[str]:
        """Get the intersection of configured pairs and halal-screened pairs."""
        max_pairs = self._settings.crypto_max_pairs_per_cycle
        halal_symbols = await self._screener.get_halal_pairs()

        if not halal_symbols:
            logger.info("No halal cache — using configured pairs: %s", self._configured_pairs)
            return self._configured_pairs[:max_pairs]

        halal_set = {s.upper() for s in halal_symbols}
        tradeable = []
        for pair in self._configured_pairs:
            upper_pair = pair.upper()
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
        return self._configured_pairs[:max_pairs]

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
