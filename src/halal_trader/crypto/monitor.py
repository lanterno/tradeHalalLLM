"""Live position monitor — watches open trades against SL/TP using WebSocket prices."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from binance import BinanceAPIException

from halal_trader.core import events
from halal_trader.core.observability import monitor_context
from halal_trader.crypto.exchange import (
    DUST_NOTIONAL_USD,
    BinanceClient,
    extract_fill_price,
)
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.db.models import CryptoTrade
from halal_trader.db.repository import Repository

if TYPE_CHECKING:
    from halal_trader.ml.retrainer import RetrainingScheduler
    from halal_trader.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

_FALLBACK_MIN_NOTIONAL = DUST_NOTIONAL_USD
_DUST_NOTIONAL_THRESHOLD = DUST_NOTIONAL_USD
_MAX_EXIT_FAILURES = 3


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
        retrainer: RetrainingScheduler | None = None,
        exiting_pairs: set[str] | None = None,
        close_recorders: object | None = None,
    ) -> None:
        self._broker = broker
        self._repo = repo
        self._ws = ws_manager
        self._check_interval = check_interval
        self._trailing_activation_pct = trailing_stop_activation_pct
        self._trailing_distance_pct = trailing_stop_distance_pct
        self._notifier = notifier
        self._retrainer = retrainer
        self._running = False
        self._task: asyncio.Task[None] | None = None
        # NOTE: the dicts and the shared `exiting_pairs` set below are
        # single-asyncio-loop only — not thread-safe. Mutations to
        # `_exiting_pairs` are guarded by `_exit_lock` so the executor and
        # monitor can't race on the same pair.
        self._high_water: dict[int, float] = {}
        self._exit_failures: dict[int, int] = {}
        self._exiting_pairs: set[str] = exiting_pairs if exiting_pairs is not None else set()
        self._exit_lock = asyncio.Lock()
        # Optional fan-out for post-close analytics (drift, thesis,
        # regret, purification). When present, the close path calls
        # ``record_close`` after a successful SL/TP exit. When ``None``,
        # the monitor behaves exactly as before (back-compat).
        self._close_recorders = close_recorders

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

        if trade.pair in self._exiting_pairs:
            return

        if self._trailing_activation_pct and trade.entry_price and trade.stop_loss:
            await self._update_trailing_stop(trade, price)

        if trade.stop_loss and price <= trade.stop_loss:
            logger.warning(
                "STOP-LOSS triggered for %s (trade #%d): price $%.2f <= SL $%.2f",
                trade.pair,
                trade_id,
                price,
                trade.stop_loss,
                extra={
                    "event": events.TRADE_EXIT_SL,
                    "trade_id": trade_id,
                    "pair": trade.pair,
                    "price": price,
                    "stop_loss": trade.stop_loss,
                },
            )
            await self._exit_position(trade, price, "stop_loss")

        elif trade.target_price and price >= trade.target_price:
            logger.info(
                "TAKE-PROFIT triggered for %s (trade #%d): price $%.2f >= TP $%.2f",
                trade.pair,
                trade_id,
                price,
                trade.target_price,
                extra={
                    "event": events.TRADE_EXIT_TP,
                    "trade_id": trade_id,
                    "pair": trade.pair,
                    "price": price,
                    "target_price": trade.target_price,
                },
            )
            await self._exit_position(trade, price, "take_profit")

    async def _exit_position(self, trade: CryptoTrade, current_price: float, reason: str) -> None:
        """Place a market sell to exit the position and record the closure."""
        trade_id = trade.id
        if trade_id is None:
            return

        async with self._exit_lock:
            if trade.pair in self._exiting_pairs:
                return
            self._exiting_pairs.add(trade.pair)

        with monitor_context() as mid:
            logger.debug(
                "Monitor exit started for %s #%d (%s)",
                trade.pair,
                trade_id,
                mid,
                extra={"trade_id": trade_id, "pair": trade.pair, "reason": reason},
            )
            try:
                await self._exit_position_inner(trade, current_price, reason)
            finally:
                async with self._exit_lock:
                    self._exiting_pairs.discard(trade.pair)

    async def _exit_position_inner(
        self, trade: CryptoTrade, current_price: float, reason: str
    ) -> None:
        """Internal exit logic — caller manages the exiting_pairs lock."""
        trade_id = trade.id
        assert trade_id is not None

        base_asset = trade.pair.upper().removesuffix("USDT").removesuffix("BUSD")
        balances = await self._broker.get_balances()
        actual_free = next((b.free for b in balances if b.asset == base_asset), 0.0)

        if actual_free <= 0 or actual_free * current_price < _DUST_NOTIONAL_THRESHOLD:
            logger.warning(
                "No %s balance for %s #%d (free=%.8f) — closing as balance_exhausted",
                base_asset,
                trade.pair,
                trade_id,
                actual_free,
            )
            await self._repo.close_crypto_trade(trade_id, current_price, "balance_exhausted")
            self._exit_failures.pop(trade_id, None)
            self._high_water.pop(trade_id, None)
            return

        quantity = min(trade.quantity, actual_free)
        quantity = self._broker.round_quantity(trade.pair, quantity)

        if quantity <= 0:
            await self._repo.close_crypto_trade(trade_id, current_price, "balance_exhausted")
            self._exit_failures.pop(trade_id, None)
            self._high_water.pop(trade_id, None)
            return

        sf = self._broker.get_symbol_filter(trade.pair)
        min_notional = sf.min_notional if sf else _FALLBACK_MIN_NOTIONAL
        notional = quantity * current_price
        if notional < min_notional:
            logger.info(
                "Skipping auto-exit for %s #%d: notional $%.2f below minimum",
                trade.pair,
                trade_id,
                notional,
            )
            await self._repo.close_crypto_trade(trade_id, current_price, f"{reason}_too_small")
            self._exit_failures.pop(trade_id, None)
            self._high_water.pop(trade_id, None)
            return

        try:
            order_result = await self._broker.place_order(
                symbol=trade.pair,
                side="SELL",
                quantity=quantity,
                order_type="MARKET",
            )
            fill_price = extract_fill_price(order_result) or current_price
            order_status = order_result.get("status", "")
            db_status = "filled" if order_status == "FILLED" else "submitted"

            await self._repo.record_crypto_trade(
                pair=trade.pair,
                side="sell",
                quantity=quantity,
                price=fill_price,
                order_id=str(order_result.get("orderId", "")),
                status=db_status,
                llm_reasoning=f"Auto {reason}: price ${current_price:.2f}",
            )

            await self._repo.close_crypto_trade(trade_id, fill_price, reason)
            self._exit_failures.pop(trade_id, None)

            consolidated = await self._repo.close_open_crypto_trades_for_pair(
                trade.pair,
                fill_price,
                "consolidated",
                exclude_id=trade_id,
            )
            if consolidated > 0:
                logger.info(
                    "Consolidated %d ghost trade(s) for %s after %s exit",
                    consolidated,
                    trade.pair,
                    reason,
                )

            pnl = (fill_price - (trade.entry_price or 0)) * quantity
            entry = trade.entry_price or 0
            return_pct = (fill_price - entry) / entry if entry > 0 else 0
            logger.info(
                "Auto-%s exit for %s #%d: sold %.6f @ $%.2f (P&L: $%+.2f)",
                reason,
                trade.pair,
                trade_id,
                quantity,
                fill_price,
                pnl,
            )

            if self._retrainer:
                try:
                    await self._retrainer.on_trade_closed(trade_id, return_pct)
                except Exception as e:
                    logger.debug("Retrainer callback failed: %s", e)

            if self._close_recorders is not None:
                try:
                    from halal_trader.core.post_close import (
                        CloseEvent,
                        record_close,
                    )

                    hold_seconds = 0
                    if trade.timestamp:
                        from datetime import UTC
                        from datetime import datetime as _dt

                        now_ts = _dt.now(UTC)
                        if trade.timestamp.tzinfo is None:
                            trade_ts = trade.timestamp.replace(tzinfo=UTC)
                        else:
                            trade_ts = trade.timestamp
                        hold_seconds = max(0, int((now_ts - trade_ts).total_seconds()))

                    record_close(
                        CloseEvent(
                            trade_id=str(trade_id),
                            symbol=trade.pair,
                            side=trade.side,
                            entry_price=entry,
                            exit_price=fill_price,
                            exit_reason=reason,
                            realized_pnl_usd=pnl,
                            return_pct=return_pct,
                            quantity=quantity,
                            hold_seconds=hold_seconds,
                            reasoning=trade.llm_reasoning or "",
                        ),
                        self._close_recorders,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("post-close recorder failed: %s", e)

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

        except BinanceAPIException as e:
            if e.code == -2010:
                logger.warning(
                    "Insufficient balance for %s #%d — force-closing trade: %s",
                    trade.pair,
                    trade_id,
                    e,
                )
                await self._repo.close_crypto_trade(
                    trade_id, current_price, f"{reason}_insufficient_balance"
                )
                self._exit_failures.pop(trade_id, None)
                self._high_water.pop(trade_id, None)
                return
            await self._handle_exit_failure(trade_id, trade.pair, reason, e)

        except Exception as e:
            await self._handle_exit_failure(trade_id, trade.pair, reason, e)

        self._high_water.pop(trade_id, None)

    async def _handle_exit_failure(
        self, trade_id: int, pair: str, reason: str, error: Exception
    ) -> None:
        """Track exit failures and force-close after max retries."""
        failures = self._exit_failures.get(trade_id, 0) + 1
        self._exit_failures[trade_id] = failures

        if failures >= _MAX_EXIT_FAILURES:
            logger.error(
                "Max exit retries (%d) for %s #%d — force-closing: %s",
                _MAX_EXIT_FAILURES,
                pair,
                trade_id,
                error,
            )
            price = self._ws.get_latest_price(pair) or 0.0
            await self._repo.close_crypto_trade(trade_id, price, f"{reason}_max_retries")
            self._exit_failures.pop(trade_id, None)
        else:
            logger.error(
                "Failed to auto-exit %s #%d (%s), attempt %d/%d: %s",
                pair,
                trade_id,
                reason,
                failures,
                _MAX_EXIT_FAILURES,
                error,
            )

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
                trade.pair,
                trade_id,
                current_sl,
                new_sl,
                high,
            )
