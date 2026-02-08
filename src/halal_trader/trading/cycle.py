"""Intraday trading cycle — gathers market data, analyzes, and executes."""

import logging
from typing import Any

from halal_trader.agent.sentiment import SentimentAnalyzer
from halal_trader.agent.strategy import TradingStrategy
from halal_trader.domain.ports import Broker, ComplianceScreener
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)

# Maximum number of symbols to fetch market data for per cycle.
_MAX_SYMBOLS_PER_CYCLE = 20

# Maximum number of symbols to run sentiment analysis on.
_MAX_SENTIMENT_SYMBOLS = 10


class TradingCycleService:
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
    ) -> None:
        self._broker = broker
        self._screener = screener
        self._strategy = strategy
        self._executor = executor
        self._portfolio = portfolio
        self._sentiment = sentiment

    async def run_cycle(self) -> None:
        """Execute one complete trading cycle.

        Steps:
        1. Check that the market is open.
        2. Check whether the daily loss limit has been breached.
        3. Gather account, position, and market data.
        4. Run sentiment analysis (optional).
        5. Run LLM strategy analysis.
        6. Execute resulting trades.
        """
        logger.info("=== TRADING CYCLE ===")
        try:
            # 1. Check market status
            clock = await self._broker.get_clock()
            if not clock.is_open:
                logger.info("Market is closed, skipping trading cycle")
                return

            # 2. Check daily loss limit
            if await self._portfolio.should_halt_trading():
                logger.warning("Daily loss limit reached — halting trades")
                return

            # 3. Gather data
            account = await self._broker.get_account_info()
            positions = await self._broker.get_all_positions()

            halal_symbols = await self._screener.get_halal_symbols()
            if not halal_symbols:
                logger.warning("No halal symbols available, skipping cycle")
                return

            snapshots, bars = await self._fetch_market_data(halal_symbols)
            today_pnl = await self._portfolio.get_current_pnl()

            # 4. Sentiment analysis (supplementary signal)
            sentiment_text = await self._gather_sentiment(halal_symbols)

            # 5. LLM analysis
            plan = await self._strategy.analyze(
                account=account,
                positions=positions,
                halal_symbols=halal_symbols,
                snapshots=snapshots,
                bars=bars,
                today_pnl=today_pnl,
                sentiment_text=sentiment_text,
            )

            logger.info(
                "Trading plan: %s | %d buys, %d sells",
                plan.market_outlook[:80],
                len(plan.buys),
                len(plan.sells),
            )

            # 6. Execute decisions
            if plan.decisions:
                results = await self._executor.execute_plan(plan)
                for r in results:
                    logger.info("Execution result: %s", r)
            else:
                logger.info("No trades to execute this cycle")

        except Exception as e:
            logger.error("Trading cycle failed: %s", e, exc_info=True)

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
