"""Crypto trading cycle — gathers data, computes indicators, analyzes, and executes."""

import logging
from typing import Any

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
    ) -> None:
        self._broker = broker
        self._screener = screener
        self._strategy = strategy
        self._executor = executor
        self._portfolio = portfolio
        self._ws = ws_manager
        self._configured_pairs = configured_pairs or []

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
            positions_text = self._portfolio.format_positions_for_prompt(balances)
            today_pnl = await self._portfolio.get_current_pnl()

            # 6. LLM analysis
            plan = await self._strategy.analyze(
                account=account,
                positions_text=positions_text,
                halal_pairs=halal_pairs,
                klines_by_symbol=klines_by_symbol,
                orderbooks=orderbooks,
                today_pnl=today_pnl,
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

        # Map halal symbols to configured pair format (e.g., BTC -> BTCUSDT)
        halal_set = {s.upper() for s in halal_symbols}
        tradeable = []
        for pair in self._configured_pairs:
            # Extract base asset from pair (e.g., BTCUSDT -> BTC)
            base = pair.replace("USDT", "").replace("BUSD", "").replace("BTC", "")
            # If the pair itself or the base asset is in the halal set, include it
            if pair.upper() in halal_set or base.upper() in halal_set:
                tradeable.append(pair)
            # Also try the full base (e.g., for BTCUSDT, check "BTC")
            pair_base = pair.upper().rstrip("USDT")
            if pair_base in halal_set:
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
