"""Order execution logic — translates LLM decisions into broker orders."""

import logging
from datetime import UTC, datetime
from typing import Any

from halal_trader.core import events
from halal_trader.core.executor import BaseExecutor
from halal_trader.core.fills import confirm_alpaca
from halal_trader.db.repos import TradeRepo
from halal_trader.domain.models import TradingPlan
from halal_trader.domain.ports import Broker

logger = logging.getLogger(__name__)

_FILL_TIMEOUT = 30.0
_FILL_POLL_INTERVAL = 2.0


class TradeExecutor(BaseExecutor):
    """Executes stock trading decisions via the broker."""

    def __init__(
        self,
        broker: Broker,
        repo: TradeRepo,
        *,
        max_position_pct: float,
        max_simultaneous_positions: int,
        max_sector_pct: float = 0.40,
    ) -> None:
        super().__init__(
            max_position_pct=max_position_pct,
            max_simultaneous_positions=max_simultaneous_positions,
        )
        self._repo = repo
        self._broker = broker
        # 0 disables the sector check; keep the default at 40% so even
        # an operator who hasn't tuned this gets a sane diversification
        # floor on day one.
        self._max_sector_pct = max_sector_pct

    async def execute_plan(
        self,
        plan: TradingPlan,
        *,
        bars: dict[str, Any] | None = None,
        positions: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute all decisions in a TradingPlan, returning execution results.

        ``bars`` is the per-symbol bar payload from the cycle. When passed,
        every successful BUY records a stock-side IndicatorSnapshot for the
        shared retrainer. ``positions`` (current open positions) feeds the
        sector-rotation halal cap.
        """
        return await self._execute_plan_common(plan, bars=bars or {}, positions=positions or [])

    def _get_sells(self, plan: Any) -> list[Any]:
        return plan.sells

    def _get_buys(self, plan: Any) -> list[Any]:
        return plan.buys

    async def _get_current_position_count(self, **_kwargs: Any) -> int:
        current_positions = await self._broker.get_all_positions()
        return len(current_positions)

    async def _execute_buy(self, decision: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute a buy order."""
        account = await self._broker.get_account_info()

        snapshot = await self._broker.get_stock_snapshot(decision.symbol)
        estimated_price = self._extract_price(snapshot, decision.symbol)
        estimated_cost = estimated_price * decision.quantity

        if estimated_cost > account.buying_power:
            msg = (
                f"Insufficient buying power for {decision.symbol}: "
                f"need ${estimated_cost:,.2f}, have ${account.buying_power:,.2f}"
            )
            logger.warning(msg)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": msg}

        if (
            account.portfolio_value > 0
            and (estimated_cost / account.portfolio_value) > self._max_position_pct
        ):
            msg = f"Position size for {decision.symbol} exceeds {self._max_position_pct:.0%} limit"
            logger.warning(msg)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": msg}

        # Halal sector-rotation cap — refuse buys that would push a single
        # sector past its share of equity. Pull existing exposure from the
        # broker positions we already had to fetch above (in kwargs/account).
        sector_reject = await self._check_sector_limit(
            symbol=decision.symbol,
            notional_usd=estimated_cost,
            equity_usd=account.portfolio_value,
            positions=kwargs.get("positions") or [],
        )
        if sector_reject is not None:
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "rejected",
                "reason": sector_reject,
            }

        try:
            submitted_at = datetime.now(UTC)
            order_result = await self._broker.place_order(
                symbol=decision.symbol,
                side="buy",
                quantity=decision.quantity,
                order_type="market",
                time_in_force="day",
            )
            order_id = order_result.get("id", "") if isinstance(order_result, dict) else ""
            fill = await self._confirm_fill(order_id, submitted_at)

            logger.info(
                "BUY order placed: %s x%d — orderId=%s status=%s filled=%s",
                decision.symbol,
                decision.quantity,
                order_id,
                fill.status,
                fill.filled_quantity,
                extra={
                    "event": events.TRADE_BUY_PLACED,
                    "symbol": decision.symbol,
                    "order_id": order_id,
                    "status": fill.status,
                    "filled_quantity": fill.filled_quantity,
                    "filled_price": fill.filled_price,
                },
            )

            trade_id = await self._repo.record_trade(
                symbol=decision.symbol,
                side="buy",
                quantity=decision.quantity,
                price=fill.filled_price or estimated_price,
                order_id=order_id,
                status=fill.status,
                llm_reasoning=decision.reasoning,
                submitted_at=fill.submitted_at,
                filled_at=fill.filled_at,
                filled_price=fill.filled_price,
                filled_quantity=fill.filled_quantity,
            )

            # Stock-side ML snapshot — best-effort, never aborts the buy.
            bars_for_symbol = (kwargs.get("bars") or {}).get(decision.symbol)
            if fill.status in ("filled", "partially_filled") and bars_for_symbol:
                from halal_trader.trading.snapshots import record_stock_snapshot

                await record_stock_snapshot(
                    repo=self._repo,
                    trade_id=trade_id,
                    symbol=decision.symbol,
                    bars=bars_for_symbol,
                )

            return {
                "symbol": decision.symbol,
                "action": "buy",
                "quantity": decision.quantity,
                "status": fill.status,
                "order": order_result,
                "trade_id": trade_id,
            }
        except Exception as e:
            logger.error("Failed to place BUY order for %s: %s", decision.symbol, e)
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "error",
                "reason": str(e),
            }

    async def _execute_sell(self, decision: Any, **_kwargs: Any) -> dict[str, Any]:
        """Execute a sell order (close or reduce position)."""
        try:
            submitted_at = datetime.now(UTC)
            if decision.quantity == 0:
                result = await self._broker.close_position(decision.symbol)
            else:
                result = await self._broker.place_order(
                    symbol=decision.symbol,
                    side="sell",
                    quantity=decision.quantity,
                    order_type="market",
                    time_in_force="day",
                )

            order_id = result.get("id", "") if isinstance(result, dict) else ""
            fill = await self._confirm_fill(order_id, submitted_at)

            logger.info(
                "SELL order placed: %s x%d — orderId=%s status=%s filled=%s",
                decision.symbol,
                decision.quantity,
                order_id,
                fill.status,
                fill.filled_quantity,
                extra={
                    "event": events.TRADE_SELL_PLACED,
                    "symbol": decision.symbol,
                    "order_id": order_id,
                    "status": fill.status,
                    "filled_quantity": fill.filled_quantity,
                    "filled_price": fill.filled_price,
                },
            )

            await self._repo.record_trade(
                symbol=decision.symbol,
                side="sell",
                quantity=decision.quantity,
                price=fill.filled_price,
                order_id=order_id,
                status=fill.status,
                llm_reasoning=decision.reasoning,
                submitted_at=fill.submitted_at,
                filled_at=fill.filled_at,
                filled_price=fill.filled_price,
                filled_quantity=fill.filled_quantity,
            )

            return {
                "symbol": decision.symbol,
                "action": "sell",
                "quantity": decision.quantity,
                "status": fill.status,
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

    async def _confirm_fill(self, order_id: str, submitted_at: datetime) -> Any:
        """Poll the broker for fill state, returning a FillResult.

        If the order_id is empty (e.g. close_position returned a non-dict
        response), fall back to a "pending" FillResult so the trade is still
        recorded with the submission timestamp.
        """
        from halal_trader.core.fills import FillResult

        if not order_id:
            return FillResult(
                status="pending",
                order_id="",
                filled_quantity=0.0,
                filled_price=None,
                submitted_at=submitted_at,
                filled_at=None,
                raw={},
            )

        return await confirm_alpaca(
            poll=lambda: self._broker.get_order_by_id(order_id),
            order_id=order_id,
            submitted_at=submitted_at,
            timeout=_FILL_TIMEOUT,
            interval=_FILL_POLL_INTERVAL,
        )

    async def close_all(self) -> Any:
        """Close all open positions (end of day)."""
        logger.info("Closing all positions (end of day)")
        return await self._broker.close_all_positions()

    async def _check_sector_limit(
        self,
        *,
        symbol: str,
        notional_usd: float,
        equity_usd: float,
        positions: list[Any],
    ) -> str | None:
        """Return a rejection reason if the buy would breach the sector cap."""
        if equity_usd <= 0 or self._max_sector_pct <= 0:
            return None
        from halal_trader.halal.sector_limits import (
            check_buy_against_limits,
            compute_allocation,
        )

        positions_value = {
            p.symbol: float(p.qty) * float(p.current_price or p.avg_entry_price) for p in positions
        }
        allocation = compute_allocation(positions_value, total_equity=equity_usd)
        ok, reason = check_buy_against_limits(
            symbol=symbol,
            notional_usd=notional_usd,
            allocation=allocation,
            max_sector_pct=self._max_sector_pct,
        )
        if not ok:
            logger.warning("Sector cap rejection for %s: %s", symbol, reason)
            return reason
        return None

    def _extract_price(self, snapshot: Any, symbol: str) -> float:
        """Extract a usable price from a snapshot response."""
        if isinstance(snapshot, dict):
            data = snapshot.get(symbol, snapshot)
            if isinstance(data, dict):
                trade = data.get("latest_trade", {})
                if isinstance(trade, dict):
                    price = trade.get("price", 0)
                    if price:
                        return float(price)
                bar = data.get("daily_bar", {})
                if isinstance(bar, dict):
                    close = bar.get("close", 0)
                    if close:
                        return float(close)
        return 0.0
