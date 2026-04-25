"""Intraday trading cycle — gathers market data, analyzes, and executes."""

import logging
from typing import Any

from halal_trader.core.cycle import BaseCycleService
from halal_trader.domain.ports import Broker, ComplianceScreener
from halal_trader.market_hours import is_market_open_local, now_eastern
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.portfolio import PortfolioTracker
from halal_trader.trading.sentiment import SentimentAnalyzer
from halal_trader.trading.strategy import TradingStrategy

logger = logging.getLogger(__name__)

# Maximum number of symbols to fetch market data for per cycle.
_MAX_SYMBOLS_PER_CYCLE = 20

# Maximum number of symbols to run sentiment analysis on.
_MAX_SENTIMENT_SYMBOLS = 10


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
        sentiment: SentimentAnalyzer | None = None,
        alerts=None,
        engine=None,
        live_mode_checker=None,
    ) -> None:
        super().__init__(alerts=alerts, engine=engine)
        self._live_mode_checker = live_mode_checker
        self._broker = broker
        self._screener = screener
        self._strategy = strategy
        self._executor = executor
        self._portfolio = portfolio
        self._sentiment = sentiment

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

        sentiment_text = await self._gather_sentiment(halal_symbols)
        risk_text = self._build_risk_text(bars, positions, account.effective_equity)

        plan = await self._strategy.analyze(
            account=account,
            positions=positions,
            halal_symbols=halal_symbols,
            snapshots=snapshots,
            bars=bars,
            today_pnl=today_pnl,
            sentiment_text=sentiment_text,
            risk_text=risk_text,
        )

        logger.info(
            "Trading plan: %s | %d buys, %d sells",
            plan.market_outlook[:80],
            len(plan.buys),
            len(plan.sells),
        )

        if plan.decisions:
            results = await self._executor.execute_plan(plan, bars=bars)
            for r in results:
                logger.info("Execution result: %s", r)
        else:
            logger.info("No trades to execute this cycle")

    # ── Private helpers ──────────────────────────────────────────

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

    def _build_risk_text(self, bars: dict[str, Any], positions: list[Any], equity: float) -> str:
        """Run the shared portfolio-risk engine and return a prompt-ready string."""
        try:
            from halal_trader.config import get_settings
            from halal_trader.trading.risk import evaluate_stock_risk

            output = evaluate_stock_risk(
                settings=get_settings(),
                bars_by_symbol=bars,
                positions=positions,
                total_equity=equity,
            )
        except Exception as exc:
            logger.debug("Stock risk engine evaluation failed: %s", exc)
            return ""
        return output.risk_text

    async def _gather_sentiment(self, halal_symbols: list[str]) -> str:
        """Run sentiment analysis if available, returning formatted text."""
        if not self._sentiment:
            return "Sentiment data: not available"
        try:
            scores = await self._sentiment.analyze_batch(halal_symbols[:_MAX_SENTIMENT_SYMBOLS])
            return self._sentiment.format_for_prompt(scores)
        except Exception as e:
            logger.debug("Sentiment analysis skipped: %s", e)
            return "Sentiment data: not available"
