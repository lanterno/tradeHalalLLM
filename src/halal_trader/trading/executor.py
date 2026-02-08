"""Order execution logic — translates LLM decisions into Alpaca MCP orders."""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.agent.decision import TradeDecision, TradingPlan
from halal_trader.config import get_settings
from halal_trader.db.repository import Repository
from halal_trader.mcp.client import AlpacaMCPClient

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trading decisions via the Alpaca MCP server."""

    def __init__(self, mcp: AlpacaMCPClient, repo: Repository) -> None:
        self._mcp = mcp
        self._repo = repo

    async def execute_plan(self, plan: TradingPlan) -> list[dict[str, Any]]:
        """Execute all decisions in a TradingPlan, returning execution results."""
        results = []

        # Execute sells first (free up capital)
        for decision in plan.sells:
            result = await self._execute_sell(decision)
            results.append(result)

        # Then execute buys
        for decision in plan.buys:
            result = await self._execute_buy(decision)
            results.append(result)

        return results

    async def _execute_buy(self, decision: TradeDecision) -> dict[str, Any]:
        """Execute a buy order."""
        settings = get_settings()

        # Validate: check buying power before placing order
        account = await self._mcp.get_account_info()
        buying_power = float(account.get("buying_power", 0)) if isinstance(account, dict) else 0

        # Get current price estimate
        snapshot = await self._mcp.get_stock_snapshot(decision.symbol)
        estimated_price = self._extract_price(snapshot, decision.symbol)
        estimated_cost = estimated_price * decision.quantity

        if estimated_cost > buying_power:
            msg = (
                f"Insufficient buying power for {decision.symbol}: "
                f"need ${estimated_cost:,.2f}, have ${buying_power:,.2f}"
            )
            logger.warning(msg)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": msg}

        # Check position size limit
        portfolio_value = (
            float(account.get("portfolio_value", 0)) if isinstance(account, dict) else 0
        )
        if portfolio_value > 0 and (estimated_cost / portfolio_value) > settings.max_position_pct:
            msg = (
                f"Position size for {decision.symbol} exceeds {settings.max_position_pct:.0%} limit"
            )
            logger.warning(msg)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": msg}

        # Place the order
        try:
            order_result = await self._mcp.place_order(
                symbol=decision.symbol,
                side="buy",
                quantity=decision.quantity,
                order_type="market",
                time_in_force="day",
            )
            logger.info(
                "BUY order placed: %s x%d — %s",
                decision.symbol,
                decision.quantity,
                order_result,
            )

            order_id = order_result.get("id", "") if isinstance(order_result, dict) else ""
            await self._repo.record_trade(
                symbol=decision.symbol,
                side="buy",
                quantity=decision.quantity,
                price=estimated_price,
                order_id=order_id,
                status="submitted",
                llm_reasoning=decision.reasoning,
            )

            return {
                "symbol": decision.symbol,
                "action": "buy",
                "quantity": decision.quantity,
                "status": "submitted",
                "order": order_result,
            }
        except Exception as e:
            logger.error("Failed to place BUY order for %s: %s", decision.symbol, e)
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "error",
                "reason": str(e),
            }

    async def _execute_sell(self, decision: TradeDecision) -> dict[str, Any]:
        """Execute a sell order (close or reduce position)."""
        try:
            if decision.quantity == 0:
                # Close entire position
                result = await self._mcp.close_position(decision.symbol)
            else:
                result = await self._mcp.place_order(
                    symbol=decision.symbol,
                    side="sell",
                    quantity=decision.quantity,
                    order_type="market",
                    time_in_force="day",
                )

            logger.info(
                "SELL order placed: %s x%d — %s",
                decision.symbol,
                decision.quantity,
                result,
            )

            order_id = result.get("id", "") if isinstance(result, dict) else ""
            await self._repo.record_trade(
                symbol=decision.symbol,
                side="sell",
                quantity=decision.quantity,
                order_id=order_id,
                status="submitted",
                llm_reasoning=decision.reasoning,
            )

            return {
                "symbol": decision.symbol,
                "action": "sell",
                "quantity": decision.quantity,
                "status": "submitted",
                "order": result,
            }
        except Exception as e:
            logger.error("Failed to place SELL order for %s: %s", decision.symbol, e)
            return {
                "symbol": decision.symbol,
                "action": "sell",
                "status": "error",
                "reason": str(e),
            }

    async def close_all(self) -> Any:
        """Close all open positions (end of day)."""
        logger.info("Closing all positions (end of day)")
        return await self._mcp.close_all_positions()

    def _extract_price(self, snapshot: Any, symbol: str) -> float:
        """Extract a usable price from a snapshot response."""
        if isinstance(snapshot, dict):
            # May be nested under the symbol key
            data = snapshot.get(symbol, snapshot)
            if isinstance(data, dict):
                trade = data.get("latest_trade", {})
                if isinstance(trade, dict):
                    price = trade.get("price", 0)
                    if price:
                        return float(price)
                # Try daily bar close
                bar = data.get("daily_bar", {})
                if isinstance(bar, dict):
                    close = bar.get("close", 0)
                    if close:
                        return float(close)
        return 0.0
