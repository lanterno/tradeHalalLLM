"""Crypto order execution — translates LLM decisions into Binance orders."""

import logging
from typing import Any

from halal_trader.crypto.exchange import BinanceClient
from halal_trader.domain.models import CryptoTradeDecision, CryptoTradingPlan
from halal_trader.domain.ports import TradeRepository

logger = logging.getLogger(__name__)

_MIN_NOTIONAL_USDT = 5.0


class CryptoExecutor:
    """Executes crypto trading decisions via the Binance client."""

    def __init__(
        self,
        broker: BinanceClient,
        repo: TradeRepository,
        *,
        max_position_pct: float,
        max_simultaneous_positions: int,
        configured_pairs: list[str] | None = None,
    ) -> None:
        self._broker = broker
        self._repo = repo
        self._max_position_pct = max_position_pct
        self._max_simultaneous_positions = max_simultaneous_positions
        self._tracked_bases = {
            p.upper().removesuffix("USDT").removesuffix("BUSD")
            for p in (configured_pairs or [])
        }

    async def execute_plan(self, plan: CryptoTradingPlan) -> list[dict[str, Any]]:
        """Execute all decisions in a CryptoTradingPlan."""
        results: list[dict[str, Any]] = []

        # Execute sells first (free up capital)
        for decision in plan.sells:
            result = await self._execute_sell(decision)
            results.append(result)

        # Then execute buys (respecting max simultaneous positions).
        # Only count balances in configured trading pairs (ignore unrelated testnet coins).
        balances = await self._broker.get_balances()
        if self._tracked_bases:
            open_count = sum(
                1 for b in balances
                if b.asset in self._tracked_bases and b.free > 0
            )
        else:
            open_count = sum(1 for b in balances if b.asset != "USDT" and b.free > 0)

        for decision in plan.buys:
            if open_count >= self._max_simultaneous_positions:
                msg = (
                    f"Max simultaneous positions ({self._max_simultaneous_positions}) "
                    f"reached — skipping BUY {decision.symbol}"
                )
                logger.warning(msg)
                results.append(
                    {
                        "symbol": decision.symbol,
                        "action": "buy",
                        "status": "rejected",
                        "reason": msg,
                    }
                )
                continue
            result = await self._execute_buy(decision)
            if result.get("status") == "submitted":
                open_count += 1
            results.append(result)

        return results

    def _validate_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> str | None:
        """Pre-validate an order against Binance filters. Returns error message or None."""
        notional = quantity * price
        if notional < _MIN_NOTIONAL_USDT:
            return (
                f"Order too small for {symbol}: "
                f"${notional:.2f} < ${_MIN_NOTIONAL_USDT} minimum notional"
            )
        return None

    async def _execute_buy(self, decision: CryptoTradeDecision) -> dict[str, Any]:
        """Execute a crypto buy order."""
        account = await self._broker.get_account()
        price = await self._broker.get_ticker_price(decision.symbol)
        estimated_cost = price * decision.quantity

        usdt_available = (
            account.usdt_free if account.usdt_free > 0
            else account.available_balance_usdt
        )
        if estimated_cost > usdt_available:
            msg = (
                f"Insufficient USDT for {decision.symbol}: "
                f"need ${estimated_cost:,.2f}, have ${usdt_available:,.2f} USDT"
            )
            logger.warning(msg)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": msg}

        total = account.total_balance_usdt
        if total > 0 and (estimated_cost / total) > self._max_position_pct:
            msg = f"Position size for {decision.symbol} exceeds {self._max_position_pct:.0%} limit"
            logger.warning(msg)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": msg}

        if err := self._validate_order(decision.symbol, "BUY", decision.quantity, price):
            logger.info("Skipping BUY %s: %s", decision.symbol, err)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": err}

        try:
            order_result = await self._broker.place_order(
                symbol=decision.symbol,
                side="BUY",
                quantity=decision.quantity,
                order_type="MARKET",
            )
            logger.info(
                "Crypto BUY placed: %s qty=%s — orderId=%s",
                decision.symbol,
                decision.quantity,
                order_result.get("orderId"),
            )

            order_id = str(order_result.get("orderId", ""))
            fill_price = self._extract_fill_price(order_result) or price

            trade_id = await self._repo.record_crypto_trade(
                pair=decision.symbol,
                side="buy",
                quantity=decision.quantity,
                price=fill_price,
                order_id=order_id,
                status="submitted",
                llm_reasoning=decision.reasoning,
                entry_price=fill_price,
                stop_loss=decision.stop_loss,
                target_price=decision.target_price,
            )

            return {
                "symbol": decision.symbol,
                "action": "buy",
                "quantity": decision.quantity,
                "price": fill_price,
                "status": "submitted",
                "order_id": order_id,
                "trade_id": trade_id,
            }
        except Exception as e:
            logger.error("Failed to place crypto BUY for %s: %s", decision.symbol, e)
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "error",
                "reason": str(e),
            }

    async def _execute_sell(self, decision: CryptoTradeDecision) -> dict[str, Any]:
        """Execute a crypto sell order."""
        try:
            price = await self._broker.get_ticker_price(decision.symbol)
            if err := self._validate_order(decision.symbol, "SELL", decision.quantity, price):
                logger.info("Skipping SELL %s: %s", decision.symbol, err)
                return {
                    "symbol": decision.symbol, "action": "sell",
                    "status": "rejected", "reason": err,
                }

            order_result = await self._broker.place_order(
                symbol=decision.symbol,
                side="SELL",
                quantity=decision.quantity,
                order_type="MARKET",
            )
            logger.info(
                "Crypto SELL placed: %s qty=%s — orderId=%s",
                decision.symbol,
                decision.quantity,
                order_result.get("orderId"),
            )

            order_id = str(order_result.get("orderId", ""))
            fill_price = self._extract_fill_price(order_result)

            await self._repo.record_crypto_trade(
                pair=decision.symbol,
                side="sell",
                quantity=decision.quantity,
                price=fill_price,
                order_id=order_id,
                status="submitted",
                llm_reasoning=decision.reasoning,
            )

            return {
                "symbol": decision.symbol,
                "action": "sell",
                "quantity": decision.quantity,
                "price": fill_price,
                "status": "submitted",
                "order_id": order_id,
            }
        except Exception as e:
            logger.error("Failed to place crypto SELL for %s: %s", decision.symbol, e)
            return {
                "symbol": decision.symbol,
                "action": "sell",
                "status": "error",
                "reason": str(e),
            }

    def _extract_fill_price(self, order_result: dict[str, Any]) -> float | None:
        """Extract the average fill price from an order result."""
        # Binance returns fills array for market orders
        fills = order_result.get("fills", [])
        if fills:
            total_qty = sum(float(f.get("qty", 0)) for f in fills)
            total_cost = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
            if total_qty > 0:
                return total_cost / total_qty

        # Fallback to cumulativeQuoteQty / executedQty
        exec_qty = float(order_result.get("executedQty", 0))
        cumulative = float(order_result.get("cumulativeQuoteQty", 0))
        if exec_qty > 0 and cumulative > 0:
            return cumulative / exec_qty

        return None
