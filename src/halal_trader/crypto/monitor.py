"""Live position monitor — watches open trades against SL/TP using WebSocket prices."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from halal_trader.crypto.exchange import BinanceClient, extract_fill_price
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.db.models import CryptoTrade
from halal_trader.db.repository import Repository

if TYPE_CHECKING:
    from halal_trader.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

_FALLBACK_MIN_NOTIONAL = 5.0


class PositionMonitor:
    """Watches open positions against their stop-loss and take-profit levels.

    Runs as a background async task alongside the trading cycle.  Uses the
    existing WebSocket price feed so there are no extra API calls.
    """

    def __init__(
        self,
        broker: BinanceClient,
        repo: Repository,
        ws_manager: BinanceWSManager,
        *,
        check_interval: float = 2.0,
        trailing_stop_activation_pct: float | None = None,
        trailing_stop_distance_pct: float = 0.003,
        notifier: TelegramNotifier | None = None,
    ) -> None:
        self._broker = broker
        self._repo = repo
        self._ws = ws_manager
        self._check_interval = check_interval
        self._trailing_activation_pct = trailing_stop_activation_pct
        self._trailing_distance_pct = trailing_stop_distance_pct
        self._notifier = notifier
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._high_water: dict[int, float] = {}

    async def start(self) -> None:
        """Start the monitor as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="position-monitor")
        logger.info("Position monitor started (check every %.1fs)", self._check_interval)

    async def stop(self) -> None:
        """Stop the monitor gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Position monitor stopped")

    async def _run_loop(self) -> None:
        """Main loop: poll open trades and check prices against SL/TP."""
        while self._running:
            try:
                open_trades = await self._repo.get_open_crypto_trades()
                for trade in open_trades:
                    if not trade.stop_loss and not trade.target_price:
                        continue
                    price = self._ws.get_latest_price(trade.pair)
                    if price is None:
                        continue
                    await self._check_trade(trade, price)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Position monitor error: %s", e, exc_info=True)

            await asyncio.sleep(self._check_interval)

    async def _check_trade(self, trade: CryptoTrade, price: float) -> None:
        """Check a single trade against its SL/TP levels."""
        trade_id = trade.id
        if trade_id is None:
            return

        if self._trailing_activation_pct and trade.entry_price and trade.stop_loss:
            await self._update_trailing_stop(trade, price)

        if trade.stop_loss and price <= trade.stop_loss:
            logger.warning(
                "STOP-LOSS triggered for %s (trade #%d): price $%.2f <= SL $%.2f",
                trade.pair, trade_id, price, trade.stop_loss,
            )
            await self._exit_position(trade, price, "stop_loss")

        elif trade.target_price and price >= trade.target_price:
            logger.info(
                "TAKE-PROFIT triggered for %s (trade #%d): price $%.2f >= TP $%.2f",
                trade.pair, trade_id, price, trade.target_price,
            )
            await self._exit_position(trade, price, "take_profit")

    async def _exit_position(self, trade: CryptoTrade, current_price: float, reason: str) -> None:
        """Place a market sell to exit the position and record the closure."""
        trade_id = trade.id
        if trade_id is None:
            return

        sf = self._broker.get_symbol_filter(trade.pair)
        min_notional = sf.min_notional if sf else _FALLBACK_MIN_NOTIONAL
        notional = trade.quantity * current_price
        if notional < min_notional:
            logger.info(
                "Skipping auto-exit for %s #%d: notional $%.2f below minimum",
                trade.pair, trade_id, notional,
            )
            await self._repo.close_crypto_trade(trade_id, current_price, f"{reason}_too_small")
            return

        try:
            order_result = await self._broker.place_order(
                symbol=trade.pair,
                side="SELL",
                quantity=trade.quantity,
                order_type="MARKET",
            )
            fill_price = extract_fill_price(order_result) or current_price
            order_status = order_result.get("status", "")
            db_status = "filled" if order_status == "FILLED" else "submitted"

            await self._repo.record_crypto_trade(
                pair=trade.pair,
                side="sell",
                quantity=trade.quantity,
                price=fill_price,
                order_id=str(order_result.get("orderId", "")),
                status=db_status,
                llm_reasoning=f"Auto {reason}: price ${current_price:.2f}",
            )

            await self._repo.close_crypto_trade(trade_id, fill_price, reason)

            pnl = (fill_price - (trade.entry_price or 0)) * trade.quantity
            logger.info(
                "Auto-%s exit for %s #%d: sold %.6f @ $%.2f (P&L: $%+.2f)",
                reason, trade.pair, trade_id, trade.quantity, fill_price, pnl,
            )

            if self._notifier and self._notifier.enabled:
                try:
                    await self._notifier.notify_sl_tp(
                        pair=trade.pair,
                        exit_reason=reason,
                        entry_price=trade.entry_price or 0,
                        exit_price=fill_price,
                        pnl=pnl,
                    )
                except Exception as e:
                    logger.debug("Failed to send SL/TP notification: %s", e)

        except Exception as e:
            logger.error(
                "Failed to auto-exit %s #%d (%s): %s",
                trade.pair, trade_id, reason, e,
            )

        self._high_water.pop(trade_id, None)

    async def _update_trailing_stop(self, trade: CryptoTrade, price: float) -> None:
        """Ratchet the stop-loss up when price moves favourably."""
        trade_id = trade.id
        if trade_id is None or trade.entry_price is None or self._trailing_activation_pct is None:
            return

        activation_price = trade.entry_price * (1 + self._trailing_activation_pct)
        if price < activation_price:
            return

        high = self._high_water.get(trade_id, price)
        if price > high:
            self._high_water[trade_id] = price
            high = price

        new_sl = high * (1 - self._trailing_distance_pct)
        current_sl = trade.stop_loss or 0
        if new_sl > current_sl:
            trade.stop_loss = new_sl
            await self._repo.update_crypto_trade_stop_loss(trade_id, new_sl)
            logger.debug(
                "Trailing stop updated for %s #%d: SL $%.2f -> $%.2f (high $%.2f)",
                trade.pair, trade_id, current_sl, new_sl, high,
            )
