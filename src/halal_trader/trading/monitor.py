"""Intra-cycle stock position monitor — enforces SL/TP between LLM cycles.

The crypto cycle runs every 60s; the stock cycle runs every 15 minutes.
Without a monitor, a stock position can breach its stop and bleed out
for a full quarter-hour before the next analysis fires. This module
fills that gap:

* Polls open stock positions every ``check_interval`` seconds.
* Pulls the latest snapshot from Alpaca (via the same MCP client the
  cycle uses — no extra credentials, no extra rate buckets).
* Closes the position via market order when the latest price crosses
  the recorded SL or TP.
* Honours market hours (no point polling at 4 AM ET) and the global
  kill-switch (the monitor still *closes* — that's reducing risk —
  but doesn't open new positions; that's the executor's job).

Halal note: closes only. The monitor never opens, leverages, or shorts.
A halt or compliance failure means we *exit* the position; we never
hedge it with derivatives.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from halal_trader.db.repository import Repository
from halal_trader.market_hours import is_market_open_local
from halal_trader.mcp.client import AlpacaMCPClient

logger = logging.getLogger(__name__)


class StockPositionMonitor:
    """Watches open stock trades against their SL / TP between LLM cycles."""

    def __init__(
        self,
        mcp: AlpacaMCPClient,
        repo: Repository,
        *,
        check_interval: float = 60.0,
        trailing_stop_activation_pct: float | None = None,
        trailing_stop_distance_pct: float = 0.005,
        retrainer: Any = None,
        close_recorders: object | None = None,
    ) -> None:
        self._mcp = mcp
        self._repo = repo
        self._check_interval = check_interval
        self._trailing_activation_pct = trailing_stop_activation_pct
        self._trailing_distance_pct = trailing_stop_distance_pct
        # Optional stock-namespaced RetrainingScheduler — same shape the
        # crypto monitor uses, so closed stock trades feed the ML loop
        # without us re-implementing the labeling pipeline.
        self._retrainer = retrainer
        # Optional post-close fan-out — drift / thesis / regret /
        # purification dispatch via core.post_close.record_close.
        self._close_recorders = close_recorders
        self._running = False
        self._task: asyncio.Task[None] | None = None
        # Per-trade-id high water mark for trailing-stop ratchet.
        self._high_water: dict[int, float] = {}

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="stock-position-monitor")
        logger.info("Stock position monitor started (check every %.0fs)", self._check_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stock position monitor stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                if not is_market_open_local():
                    # Sleep longer outside hours so we're not spinning a
                    # tight loop overnight; one minute is plenty for the
                    # 9:30 ET re-open detection.
                    await asyncio.sleep(60)
                    continue

                open_trades = await self._repo.get_open_trades()
                for trade in open_trades:
                    if not trade.stop_loss and not trade.target_price:
                        continue
                    price = await self._latest_price(trade.symbol)
                    if price is None:
                        continue
                    await self._check_trade(trade, price)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — never let monitor crash the bot
                logger.warning("Stock monitor loop error: %s", e)

            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break

    async def _latest_price(self, symbol: str) -> float | None:
        """Pull the latest snapshot price for ``symbol``, or None on failure."""
        try:
            snap = await self._mcp.get_stock_snapshot(symbol)
        except Exception as e:
            logger.debug("Snapshot fetch failed for %s: %s", symbol, e)
            return None
        return _extract_last_price(snap, symbol)

    async def _check_trade(self, trade: Any, price: float) -> None:
        """Close the trade if SL or TP triggered, otherwise update trailing stop."""
        if trade.stop_loss is not None and price <= trade.stop_loss:
            await self._exit(trade, price, "stop_loss")
            return
        if trade.target_price is not None and price >= trade.target_price:
            await self._exit(trade, price, "take_profit")
            return
        await self._update_trailing_stop(trade, price)

    async def _update_trailing_stop(self, trade: Any, price: float) -> None:
        """Ratchet the SL up once the position is comfortably in profit."""
        if self._trailing_activation_pct is None:
            return
        entry = trade.filled_price or trade.price
        if not entry:
            return
        gain = (price - entry) / entry
        if gain < self._trailing_activation_pct:
            return
        prior_high = self._high_water.get(trade.id, entry)
        if price > prior_high:
            self._high_water[trade.id] = price
        new_stop = self._high_water[trade.id] * (1 - self._trailing_distance_pct)
        if trade.stop_loss is None or new_stop > trade.stop_loss:
            await self._repo.update_stock_trade_stop_loss(trade.id, new_stop)
            logger.info(
                "Trailing stop on %s: SL %.2f → %.2f (high %.2f, price %.2f)",
                trade.symbol,
                trade.stop_loss or 0.0,
                new_stop,
                self._high_water[trade.id],
                price,
            )

    async def _exit(self, trade: Any, price: float, reason: str) -> None:
        """Submit a market sell to close the position; record exit on success."""
        try:
            result = await self._mcp.place_order(
                symbol=trade.symbol,
                side="sell",
                quantity=trade.quantity,
                order_type="market",
                time_in_force="day",
            )
        except Exception as e:
            logger.warning("Stock exit failed for %s (%s): %s", trade.symbol, reason, e)
            return

        # The MCP shape varies; treat anything truthy with a non-error as success.
        if isinstance(result, dict) and result.get("error"):
            logger.warning("Alpaca rejected exit for %s: %s", trade.symbol, result.get("error"))
            return

        await self._repo.close_trade(trade.id, exit_price=price, exit_reason=reason)
        self._high_water.pop(trade.id, None)
        logger.info("Closed %s on %s at %.2f", trade.symbol, reason, price)

        if self._retrainer is not None:
            entry = trade.filled_price or trade.price
            return_pct = (price - entry) / entry if entry else 0.0
            try:
                await self._retrainer.on_trade_closed(trade.id, return_pct)
            except Exception as e:  # noqa: BLE001 — retrain failure must not abort exit path
                logger.debug("retrainer.on_trade_closed failed for %s: %s", trade.symbol, e)

        if self._close_recorders is not None:
            try:
                from datetime import UTC
                from datetime import datetime as _dt

                from halal_trader.core.post_close import (
                    CloseEvent,
                    record_close,
                )

                entry = trade.filled_price or trade.price or 0.0
                return_pct = (price - entry) / entry if entry else 0.0
                pnl_usd = (price - entry) * (trade.filled_quantity or trade.quantity or 0)
                hold_seconds = 0
                if trade.timestamp:
                    now_ts = _dt.now(UTC)
                    ts = (
                        trade.timestamp
                        if trade.timestamp.tzinfo
                        else trade.timestamp.replace(tzinfo=UTC)
                    )
                    hold_seconds = max(0, int((now_ts - ts).total_seconds()))

                await record_close(
                    CloseEvent(
                        trade_id=str(trade.id),
                        symbol=trade.symbol,
                        side=trade.side,
                        entry_price=entry,
                        exit_price=price,
                        exit_reason=reason,
                        realized_pnl_usd=pnl_usd,
                        return_pct=return_pct,
                        quantity=trade.filled_quantity or trade.quantity or 0,
                        hold_seconds=hold_seconds,
                        reasoning=trade.llm_reasoning or "",
                    ),
                    self._close_recorders,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("stocks post-close recorder failed: %s", e)


def _extract_last_price(snap: Any, symbol: str) -> float | None:
    """Best-effort dig through Alpaca snapshot shapes for the latest price.

    Alpaca returns either a flat dict or a nested ``{symbol: {...}}``
    depending on whether one or many symbols were requested. Inside
    each entry, the latest trade lives under ``latestTrade.p`` (or
    ``latest_trade.price`` in some SDK versions).
    """
    if not isinstance(snap, dict):
        return None
    payload = snap.get(symbol) or snap.get(symbol.upper()) or snap
    if not isinstance(payload, dict):
        return None
    for path in (
        ("latestTrade", "p"),
        ("latestTrade", "price"),
        ("latest_trade", "p"),
        ("latest_trade", "price"),
        ("trade", "p"),
        ("trade", "price"),
    ):
        node: Any = payload
        ok = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if ok and node:
            try:
                return float(node)
            except Exception:
                continue
    return None
