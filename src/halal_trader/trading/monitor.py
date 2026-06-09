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

from halal_trader.db.repos import TradeRepo
from halal_trader.market_hours import is_market_open_local
from halal_trader.mcp.client import AlpacaMCPClient
from halal_trader.trading.bars import bars_to_klines
from halal_trader.trading.bars import extract_last_price as _extract_last_price

logger = logging.getLogger(__name__)

# Throttle bar-based trend-break checks so a tight monitor loop doesn't
# refetch bars every tick — structural breaks evolve over minutes, not
# seconds. Per-trade timestamp gate in ``_maybe_trend_break_exit``.
_TREND_BREAK_MIN_INTERVAL_S = 300.0


class StockPositionMonitor:
    """Watches open stock trades against their SL / TP between LLM cycles."""

    def __init__(
        self,
        mcp: AlpacaMCPClient,
        repo: TradeRepo,
        *,
        check_interval: float = 60.0,
        trailing_stop_activation_pct: float | None = None,
        trailing_stop_distance_pct: float = 0.005,
        reactor_trailing_stop_distance_pct: float = 0.08,
        trend_break_enabled: bool = True,
        trend_break_ma_period: int = 20,
        trend_break_timeframe: str = "1Hour",
        retrainer: Any = None,
        close_recorders: object | None = None,
        notifier: Any = None,
    ) -> None:
        self._mcp = mcp
        self._repo = repo
        self._check_interval = check_interval
        self._trailing_activation_pct = trailing_stop_activation_pct
        self._trailing_distance_pct = trailing_stop_distance_pct
        # Trend-break exit (reactor positions only) — see
        # ``_maybe_trend_break_exit``.
        self._trend_break_enabled = trend_break_enabled
        self._trend_break_ma_period = trend_break_ma_period
        self._trend_break_timeframe = trend_break_timeframe
        # Per-trade-id last trend-break check timestamp (monotonic).
        self._last_trend_check: dict[int, float] = {}
        # Reactor (news-momentum) positions are locked from LLM exits, so
        # the trailing stop is their main exit. It's WIDE (~8%) and
        # activates immediately (no activation gate) so a winner runs but
        # is never left unprotected — the "slow out" half of the strategy.
        self._reactor_trailing_distance_pct = reactor_trailing_stop_distance_pct
        # Optional stock-namespaced RetrainingScheduler — same shape the
        # crypto monitor uses, so closed stock trades feed the ML loop
        # without us re-implementing the labeling pipeline.
        self._retrainer = retrainer
        # Optional post-close fan-out — drift / thesis / regret /
        # purification dispatch via core.post_close.record_close.
        self._close_recorders = close_recorders
        # Optional Telegram notifier — fires `notify_sl_tp` on each
        # SL/TP exit (parity with the crypto position monitor).
        self._notifier = notifier
        self._running = False
        self._task: asyncio.Task[None] | None = None
        # Per-trade-id high water mark for trailing-stop ratchet.
        self._high_water: dict[int, float] = {}

    async def start(self) -> None:
        """Legacy entry point — kept for tests. Bot prefers :meth:`run`."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="stock-position-monitor")
        logger.info("Stock position monitor started (check every %.0fs)", self._check_interval)

    async def run(self) -> None:
        """Supervisor entry point — runs the SL/TP loop until cancelled."""
        self._running = True
        logger.info("Stock position monitor started (check every %.0fs)", self._check_interval)
        try:
            await self._run_loop()
        finally:
            self._running = False

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
        """Close the trade if SL/TP/trend-break triggered, else trail."""
        if trade.stop_loss is not None and price <= trade.stop_loss:
            await self._exit(trade, price, "stop_loss")
            return
        if trade.target_price is not None and price >= trade.target_price:
            await self._exit(trade, price, "take_profit")
            return
        if await self._maybe_trend_break_exit(trade, price):
            return
        await self._update_trailing_stop(trade, price)

    async def _maybe_trend_break_exit(self, trade: Any, price: float) -> bool:
        """Exit a *winning* reactor position on a structural trend break.

        The wide trailing stop alone gives back ~8% before exiting; this
        locks in gains sooner when the price structure actually reverses
        — defined as the latest price closing below an SMA of recent
        bars. Scoped to reactor-momentum positions (the slow-out ones)
        and only when already in profit, so a fresh entry dipping below
        its MA on noise isn't force-closed (the hard stop covers losers).

        Returns True when it closed the position. Best-effort: any bar /
        broker failure returns False so the trailing stop still governs.
        """
        if not self._trend_break_enabled:
            return False
        if str(getattr(trade, "entry_type", "") or "") != "reactor_momentum":
            return False
        entry = trade.filled_price or trade.price
        if not entry or price <= entry:
            return False  # only lock in winners; losers ride the hard stop

        import time as _t

        now = _t.monotonic()
        last = self._last_trend_check.get(trade.id, 0.0)
        if (now - last) < _TREND_BREAK_MIN_INTERVAL_S:
            return False
        self._last_trend_check[trade.id] = now

        ma = await self._trend_reference(trade.symbol)
        if ma is None or price >= ma:
            return False
        logger.info(
            "Trend-break exit on %s: price %.2f < SMA%d %.2f (entry %.2f)",
            trade.symbol,
            price,
            self._trend_break_ma_period,
            ma,
            entry,
        )
        await self._exit(trade, price, "trend_break")
        return True

    async def _trend_reference(self, symbol: str) -> float | None:
        """SMA of the last ``ma_period`` closes, or None if unavailable."""
        try:
            raw = await self._mcp.get_stock_bars(
                symbol,
                days=max(self._trend_break_ma_period, 5),
                timeframe=self._trend_break_timeframe,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("trend-break bars fetch failed for %s: %s", symbol, exc)
            return None
        klines = bars_to_klines(raw)
        closes = [float(k.close) for k in klines if getattr(k, "close", None)]
        if len(closes) < self._trend_break_ma_period:
            return None
        window = closes[-self._trend_break_ma_period :]
        return sum(window) / len(window)

    async def _update_trailing_stop(self, trade: Any, price: float) -> None:
        """Ratchet the SL up as the position runs.

        Reactor-momentum positions trail wide (~8%) and activate
        immediately — the trailing stop is their only rule-based exit
        (the LLM is locked out). Ordinary cycle positions keep the
        opt-in activation gate + tighter distance.
        """
        is_reactor = str(getattr(trade, "entry_type", "") or "") == "reactor_momentum"
        if is_reactor:
            distance_pct = self._reactor_trailing_distance_pct
            activation_pct = 0.0  # trail from the first tick in profit
        else:
            if self._trailing_activation_pct is None:
                return
            distance_pct = self._trailing_distance_pct
            activation_pct = self._trailing_activation_pct

        entry = trade.filled_price or trade.price
        if not entry:
            return
        gain = (price - entry) / entry
        if gain < activation_pct:
            return
        prior_high = self._high_water.get(trade.id, entry)
        if price > prior_high:
            self._high_water[trade.id] = price
        new_stop = self._high_water[trade.id] * (1 - distance_pct)
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

    async def _place_exit_order(self, trade: Any, reason: str) -> dict[str, Any] | None:
        """Submit the market sell that flattens ``trade``.

        Returns the raw MCP result as a dict, or ``None`` if the call raised
        (already logged). Non-dict MCP shapes are wrapped so callers can probe
        ``.get("error")`` uniformly.
        """
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
            return None
        return result if isinstance(result, dict) else {"_raw": result}

    @staticmethod
    def _wash_trade_conflict_id(result: dict[str, Any]) -> str | None:
        """Extract the resting-order id from an Alpaca wash-trade rejection.

        A market sell is rejected with code 40310000 ("potential wash trade
        detected") when an order on the same symbol is already resting (e.g. a
        leftover protective stop). The rejection carries that order's id in
        ``error.detail.existing_order_id`` — return it so the caller can cancel
        the blocker and retry, instead of re-submitting the same doomed sell on
        every monitor tick.
        """
        err = result.get("error")
        if not isinstance(err, dict):
            return None
        detail = err.get("detail")
        if not isinstance(detail, dict) or detail.get("code") != 40310000:
            return None
        oid = detail.get("existing_order_id")
        return str(oid) if oid else None

    async def _exit(self, trade: Any, price: float, reason: str) -> None:
        """Submit a market sell to close the position; record exit on success."""
        result = await self._place_exit_order(trade, reason)
        if result is None:
            return

        # The MCP shape varies; treat anything truthy with a non-error as success.
        if result.get("error"):
            # Wash-trade rejection: a resting order on this symbol is blocking
            # the sell. Cancel that order and retry ONCE — otherwise the monitor
            # loops a bare market sell every tick forever (seen re-submitting
            # every ~30 s against a leftover stop). Other rejections just abort.
            conflict_id = self._wash_trade_conflict_id(result)
            if conflict_id is None:
                logger.warning(
                    "Alpaca rejected exit for %s: %s", trade.symbol, result.get("error")
                )
                return
            logger.warning(
                "Exit on %s blocked by wash trade vs order %s — cancelling it and retrying",
                trade.symbol,
                conflict_id,
            )
            try:
                await self._mcp.cancel_order(trade.symbol, conflict_id)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "cancel_order(%s) failed for %s: %s", conflict_id, trade.symbol, e
                )
                return
            result = await self._place_exit_order(trade, reason)
            if result is None:
                return
            if result.get("error"):
                logger.warning(
                    "Exit retry after wash-trade cancel still rejected for %s: %s",
                    trade.symbol,
                    result.get("error"),
                )
                return

        await self._repo.close_trade(trade.id, exit_price=price, exit_reason=reason)
        self._high_water.pop(trade.id, None)
        self._last_trend_check.pop(trade.id, None)
        logger.info("Closed %s on %s at %.2f", trade.symbol, reason, price)

        if self._notifier and getattr(self._notifier, "enabled", False):
            try:
                entry_price = trade.filled_price or trade.price or 0.0
                pnl = (price - entry_price) * (trade.quantity or 0.0)
                await self._notifier.notify_sl_tp(
                    pair=trade.symbol,
                    exit_reason=reason,
                    entry_price=float(entry_price),
                    exit_price=float(price),
                    pnl=float(pnl),
                    quantity=float(trade.quantity or 0.0),
                    market="stocks",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("notify_sl_tp failed for %s: %s", trade.symbol, e)

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


__all__ = ["StockPositionMonitor", "_extract_last_price"]
