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
from halal_trader.domain.status import TradeStatus

logger = logging.getLogger(__name__)

_FILL_TIMEOUT = 30.0
_FILL_POLL_INTERVAL = 2.0


def _extract_order_id(order_result: Any) -> str:
    """Return the broker's order id, or ``""`` for malformed responses.

    Accepts both the bare order dict (``{"id": "...", ...}``) and the
    newer ``{"result": {"id": "...", ...}}`` envelope upstream Alpaca
    MCP wraps replies in (same shape change ``get_all_positions``
    saw). A non-dict / id-less response returns ``""`` so callers can
    treat the order as rejected instead of carrying it as a phantom
    ``pending`` row forever.
    """
    if not isinstance(order_result, dict):
        return ""
    raw = order_result.get("id", "")
    if not raw:
        wrapped = order_result.get("result")
        if isinstance(wrapped, dict):
            raw = wrapped.get("id", "")
    return str(raw) if raw else ""


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
        recent_close_cooldown_minutes: int = 30,
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
        # Hard gate that refuses BUYs for any symbol with a `closed_at`
        # within the last N minutes. Backs up the prompt-level
        # RECENTLY CLOSED warning + system-prompt rule 8 because the
        # LLM was visibly ignoring both on 2026-05-21 (CSCO ping-pong:
        # 4 transactions in 90 min on the same symbol despite three
        # escalating prompt warnings). Set to 0 to disable.
        self._recent_close_cooldown_minutes = recent_close_cooldown_minutes

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
        # Hard re-entry cooldown — refuse a BUY for any symbol the bot
        # closed within the last ``recent_close_cooldown_minutes``. Sits
        # ABOVE the broker calls so a blocked re-entry costs nothing
        # (no MCP round-trip, no snapshot fetch). The prompt-level
        # RECENTLY CLOSED warning + rule 8 stay in place to teach the
        # LLM; this gate catches the residual misses.
        cooldown_reject = await self._check_recent_close_cooldown(decision.symbol)
        if cooldown_reject is not None:
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "rejected",
                "reason": cooldown_reject,
            }

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
            order_id = _extract_order_id(order_result)
            # The broker call "succeeded" (no exception) but returned a
            # malformed payload — almost certainly a validation error
            # string from upstream. Persist as rejected with a synthetic
            # reason so the reconciler ignores the row instead of
            # carrying it as a phantom position.
            if not order_id:
                broker_msg = (
                    str(order_result)[:300] if order_result is not None else "no response"
                )
                logger.error(
                    "BUY order rejected by broker (no order id): %s — %s",
                    decision.symbol,
                    broker_msg,
                    extra={
                        "event": events.TRADE_BUY_PLACED,
                        "symbol": decision.symbol,
                        "order_id": "",
                        "status": TradeStatus.REJECTED.value,
                        "filled_quantity": 0.0,
                        "filled_price": None,
                    },
                )
                trade_id = await self._repo.record_trade(
                    symbol=decision.symbol,
                    side="buy",
                    quantity=decision.quantity,
                    price=estimated_price,
                    order_id="",
                    status=TradeStatus.REJECTED.value,
                    llm_reasoning=decision.reasoning,
                    submitted_at=submitted_at,
                    filled_at=None,
                    filled_price=None,
                    filled_quantity=0.0,
                )
                return {
                    "symbol": decision.symbol,
                    "action": "buy",
                    "status": TradeStatus.REJECTED.value,
                    "reason": broker_msg,
                    "order": order_result,
                    "trade_id": trade_id,
                }

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
            is_close_position = decision.quantity == 0
            # For ``close_position`` we snapshot the position size BEFORE
            # firing the broker call. The Alpaca MCP ``close_position``
            # tool routinely returns a payload with no ``id`` field, so
            # without the pre-call snapshot we'd have no way to record a
            # meaningful ``filled_quantity`` and the underlying BUY would
            # stay perpetually "open" in the DB → reconciler drift.
            pre_close_qty = 0.0
            if is_close_position:
                pre_close_qty = await self._fetch_position_qty(decision.symbol)
                if pre_close_qty <= 0:
                    logger.info(
                        "SELL close_position skipped: %s has no open position",
                        decision.symbol,
                    )
                    return {
                        "symbol": decision.symbol,
                        "action": "sell",
                        "status": "skipped",
                        "reason": "no open position",
                    }
                result = await self._broker.close_position(decision.symbol)
            else:
                result = await self._broker.place_order(
                    symbol=decision.symbol,
                    side="sell",
                    quantity=decision.quantity,
                    order_type="market",
                    time_in_force="day",
                )

            order_id = _extract_order_id(result)

            # Path A: sized SELL with malformed response → rejected.
            if not order_id and not is_close_position:
                broker_msg = str(result)[:300] if result is not None else "no response"
                logger.error(
                    "SELL order rejected by broker (no order id): %s — %s",
                    decision.symbol,
                    broker_msg,
                    extra={
                        "event": events.TRADE_SELL_PLACED,
                        "symbol": decision.symbol,
                        "order_id": "",
                        "status": TradeStatus.REJECTED.value,
                        "filled_quantity": 0.0,
                        "filled_price": None,
                    },
                )
                await self._repo.record_trade(
                    symbol=decision.symbol,
                    side="sell",
                    quantity=decision.quantity,
                    price=None,
                    order_id="",
                    status=TradeStatus.REJECTED.value,
                    llm_reasoning=decision.reasoning,
                    submitted_at=submitted_at,
                    filled_at=None,
                    filled_price=None,
                    filled_quantity=0.0,
                )
                return {
                    "symbol": decision.symbol,
                    "action": "sell",
                    "status": TradeStatus.REJECTED.value,
                    "reason": broker_msg,
                    "order": result,
                }

            # Path B: close_position with no order id → verify by polling
            # the broker for the post-call position. If the position
            # shrank we trust the close and record a filled SELL with the
            # actual closed quantity. If it didn't change, mark rejected.
            if not order_id and is_close_position:
                fill = await self._synthesize_close_fill(
                    symbol=decision.symbol,
                    pre_close_qty=pre_close_qty,
                    submitted_at=submitted_at,
                    raw=result if isinstance(result, dict) else {},
                )
            else:
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

            # Close the open BUY(s) for this symbol so the DB reflects
            # the exit. Without this, ``closed_at`` stays NULL on the
            # BUY, which (a) makes recent-exit queries miss LLM sells
            # and (b) leaves the symbol "open" for the cooldown gate.
            # Best-effort: a fill that succeeded shouldn't be rolled
            # back if the close-out write fails.
            if fill.status in ("filled", "partially_filled") and fill.filled_price:
                try:
                    closer = getattr(
                        self._repo, "close_open_trades_for_symbol", None
                    )
                    if closer is not None:
                        closed_n = await closer(
                            decision.symbol,
                            float(fill.filled_price),
                            "llm_sell" if not is_close_position else "llm_close_position",
                        )
                        if closed_n:
                            logger.info(
                                "Closed %d open BUY(s) for %s on SELL fill",
                                closed_n,
                                decision.symbol,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "close_open_trades_for_symbol failed for %s: %s",
                        decision.symbol,
                        exc,
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

    async def _check_recent_close_cooldown(self, symbol: str) -> str | None:
        """Return a rejection reason if ``symbol`` was closed within the
        cooldown window. Returns ``None`` to allow the BUY.

        A repo error degrades to "allow" — we never block a legitimate
        trade on a DB blip. Cooldown <= 0 disables the check entirely
        (operator escape hatch).
        """
        if self._recent_close_cooldown_minutes <= 0:
            return None
        try:
            rows = await self._repo.get_recently_closed(
                minutes=self._recent_close_cooldown_minutes
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "recent-close cooldown lookup failed for %s: %s — allowing trade",
                symbol,
                exc,
            )
            return None

        from datetime import UTC, datetime

        wanted = symbol.upper()
        now = datetime.now(UTC)
        latest_close = None
        for row in rows:
            if str(row.get("symbol") or "").upper() != wanted:
                continue
            closed_at_raw = row.get("closed_at")
            if isinstance(closed_at_raw, str):
                try:
                    closed_at = datetime.fromisoformat(
                        closed_at_raw.replace("Z", "+00:00")
                    )
                except ValueError:
                    continue
            elif isinstance(closed_at_raw, datetime):
                closed_at = closed_at_raw
            else:
                continue
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=UTC)
            if latest_close is None or closed_at > latest_close:
                latest_close = closed_at

        if latest_close is None:
            return None

        gap_min = (now - latest_close).total_seconds() / 60.0
        if gap_min >= self._recent_close_cooldown_minutes:
            return None

        reason = (
            f"recent-close cooldown: {symbol} closed {gap_min:.0f} min ago "
            f"(cooldown {self._recent_close_cooldown_minutes} min). "
            "Wait for the cooldown to elapse or pick a different symbol."
        )
        logger.warning(
            "BUY rejected by recent-close cooldown: %s (closed %.0f min ago, "
            "cooldown %d min)",
            symbol,
            gap_min,
            self._recent_close_cooldown_minutes,
        )
        return reason

    async def _fetch_position_qty(self, symbol: str) -> float:
        """Look up the broker's current open quantity for ``symbol``.

        Used by the close_position path to snapshot pre- and post-close
        state. Returns 0.0 if the position doesn't exist or the broker
        call fails — close_position is a best-effort path so a transient
        broker error shouldn't crash the cycle.
        """
        try:
            positions = await self._broker.get_all_positions()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_all_positions failed during close-position lookup for %s: %s",
                symbol,
                exc,
            )
            return 0.0
        wanted = symbol.upper()
        for pos in positions:
            if str(getattr(pos, "symbol", "")).upper() == wanted:
                try:
                    return float(getattr(pos, "qty", 0) or 0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    async def _synthesize_close_fill(
        self,
        *,
        symbol: str,
        pre_close_qty: float,
        submitted_at: datetime,
        raw: dict[str, Any],
    ) -> Any:
        """Construct a :class:`FillResult` for a close_position call that
        didn't return an order id.

        Verifies the close by re-fetching the position; if it shrank,
        record the delta as filled. Otherwise mark rejected so the
        reconciler doesn't carry the underlying BUY as still-open.
        """
        from halal_trader.core.fills import FillResult

        post_close_qty = await self._fetch_position_qty(symbol)
        closed_qty = max(pre_close_qty - post_close_qty, 0.0)
        if closed_qty > 0:
            logger.info(
                "close_position succeeded for %s (no order id; verified via "
                "position delta %.4f → %.4f, closed %.4f)",
                symbol,
                pre_close_qty,
                post_close_qty,
                closed_qty,
            )
            return FillResult(
                status=TradeStatus.FILLED.value,
                order_id="",
                filled_quantity=closed_qty,
                filled_price=None,
                submitted_at=submitted_at,
                filled_at=datetime.now(UTC),
                raw=raw,
            )
        logger.warning(
            "close_position returned no order id and position unchanged for %s "
            "(pre=%.4f, post=%.4f) — marking rejected",
            symbol,
            pre_close_qty,
            post_close_qty,
        )
        return FillResult(
            status=TradeStatus.REJECTED.value,
            order_id="",
            filled_quantity=0.0,
            filled_price=None,
            submitted_at=submitted_at,
            filled_at=None,
            raw=raw,
        )

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
