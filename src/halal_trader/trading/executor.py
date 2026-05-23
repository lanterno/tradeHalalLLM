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


def _compute_slippage_pct(
    *, side: str, estimated_price: float | None, filled_price: float | None
) -> float | None:
    """Return the per-trade paper-slippage as a signed fraction.

    Convention: positive value = adverse slippage to the bot.
      * BUY filled higher than estimated → positive (paid more)
      * SELL filled lower than estimated → positive (received less)
    Returns ``None`` when either input is missing or non-positive.
    """
    if (
        estimated_price is None
        or filled_price is None
        or estimated_price <= 0
        or filled_price <= 0
    ):
        return None
    delta = float(filled_price) - float(estimated_price)
    if side.lower() == "buy":
        return delta / float(estimated_price)
    return -delta / float(estimated_price)


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
        min_hold_minutes: int = 30,
        no_new_positions_minutes_before_close: int = 30,
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
        # Symmetric hold-time gate on the SELL side: refuse an LLM-
        # initiated SELL when the youngest open BUY for the symbol is
        # younger than this. Monitor-driven SL/TP exits go through a
        # separate code path (``close_trade``) so they're unaffected.
        # Observed 2026-05-21 15:30 ET (cycle-166585c8): LLM sold AVGO
        # 15 min after opening it. Set to 0 to disable.
        self._min_hold_minutes = min_hold_minutes
        # End-of-day lockout: refuse new BUYs in the last N min before
        # market close. Stops the bot from opening positions it can't
        # manage through close — those become forced-exit at the EOD
        # routine and lose to slippage. SELLs are always allowed (the
        # operator + monitor still need to be able to close anything).
        # Set to 0 to disable.
        self._no_new_positions_minutes_before_close = (
            no_new_positions_minutes_before_close
        )

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
        # Last-N-min-before-close lockout. Refuses NEW BUYs late in
        # the session because positions opened in the final minutes
        # can't be managed and become forced exits at EOD.
        close_lockout_reject = self._check_market_close_lockout()
        if close_lockout_reject is not None:
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "rejected",
                "reason": close_lockout_reject,
            }

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
                paper_slippage_pct=_compute_slippage_pct(
                    side="buy",
                    estimated_price=estimated_price,
                    filled_price=fill.filled_price,
                ),
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
        # Hold-time gate — refuse LLM-initiated SELLs of positions
        # younger than ``min_hold_minutes``. Monitor-driven SL/TP exits
        # don't go through this path so genuine stop-outs still fire.
        hold_reject = await self._check_min_hold(decision.symbol)
        if hold_reject is not None:
            return {
                "symbol": decision.symbol,
                "action": "sell",
                "status": "rejected",
                "reason": hold_reject,
            }
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
        """Return a rejection reason if ``symbol`` had any exit (closed
        BUY OR recent SELL) within the cooldown window. ``None`` allows
        the BUY.

        Looks at TWO sources because LLM-initiated SELLs may not have
        stamped ``closed_at`` on the original BUY (legacy rows from
        before the close-on-sell fix, or in-flight lag): ``get_recently_closed``
        catches closed BUYs, ``get_recent_sells`` catches SELL events
        regardless of the BUY's state. The latest timestamp of either
        wins.

        A repo error degrades to "allow" — we never block a legitimate
        trade on a DB blip. Cooldown <= 0 disables the check entirely
        (operator escape hatch).
        """
        if self._recent_close_cooldown_minutes <= 0:
            return None
        from datetime import UTC, datetime

        wanted = symbol.upper()
        now = datetime.now(UTC)
        latest_exit: datetime | None = None
        exit_kind = ""

        async def _scan(
            rows: list[dict[str, Any]], ts_key: str
        ) -> tuple[datetime | None, str]:
            best: datetime | None = None
            for row in rows:
                if str(row.get("symbol") or "").upper() != wanted:
                    continue
                ts_raw = row.get(ts_key)
                if isinstance(ts_raw, str):
                    try:
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                elif isinstance(ts_raw, datetime):
                    ts = ts_raw
                else:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if best is None or ts > best:
                    best = ts
            return best, ts_key

        try:
            closed_rows = await self._repo.get_recently_closed(
                minutes=self._recent_close_cooldown_minutes
            )
            closed_ts, _ = await _scan(closed_rows, "closed_at")
            if closed_ts is not None:
                latest_exit = closed_ts
                exit_kind = "closed"
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "recent-close cooldown closed-lookup failed for %s: %s",
                symbol,
                exc,
            )

        # Also check raw SELL events — covers LLM sells where closed_at
        # didn't get stamped on the BUY (pre-fix data or transient lag).
        sells_method = getattr(self._repo, "get_recent_sells", None)
        if sells_method is not None:
            try:
                sell_rows = await sells_method(
                    minutes=self._recent_close_cooldown_minutes
                )
                sell_ts, _ = await _scan(sell_rows, "timestamp")
                if sell_ts is not None and (
                    latest_exit is None or sell_ts > latest_exit
                ):
                    latest_exit = sell_ts
                    exit_kind = "sold"
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "recent-close cooldown sells-lookup failed for %s: %s",
                    symbol,
                    exc,
                )

        if latest_exit is None:
            return None

        gap_min = (now - latest_exit).total_seconds() / 60.0
        if gap_min >= self._recent_close_cooldown_minutes:
            return None

        reason = (
            f"recent-close cooldown: {symbol} {exit_kind} {gap_min:.0f} min ago "
            f"(cooldown {self._recent_close_cooldown_minutes} min). "
            "Wait for the cooldown to elapse or pick a different symbol."
        )
        logger.warning(
            "BUY rejected by recent-close cooldown: %s (%s %.0f min ago, "
            "cooldown %d min)",
            symbol,
            exit_kind,
            gap_min,
            self._recent_close_cooldown_minutes,
        )
        return reason

    def _check_market_close_lockout(self) -> str | None:
        """Refuse new BUYs in the last ``no_new_positions_minutes_before_close``
        minutes before market close (4:00 PM ET).

        Lets EOD reconciliation close the day cleanly without fresh
        positions in flight. ``None`` allows the BUY. Set to 0 to disable.
        """
        if self._no_new_positions_minutes_before_close <= 0:
            return None
        from halal_trader.market_hours import effective_close_time, now_eastern

        try:
            now_et = now_eastern()
        except Exception:  # noqa: BLE001
            return None
        close_t = effective_close_time(now_et.date())
        close_dt = now_et.replace(
            hour=close_t.hour, minute=close_t.minute, second=0, microsecond=0
        )
        minutes_to_close = (close_dt - now_et).total_seconds() / 60.0
        if minutes_to_close > self._no_new_positions_minutes_before_close:
            return None
        if minutes_to_close < 0:
            # Already past close — separate concern; let other gates handle.
            return None
        reason = (
            f"market-close lockout: {minutes_to_close:.0f} min to close "
            f"(lockout window {self._no_new_positions_minutes_before_close} min). "
            "New BUYs blocked — only SELLs / closes from here."
        )
        logger.info("BUY rejected by close lockout: %s", reason)
        return reason

    async def _check_min_hold(self, symbol: str) -> str | None:
        """Return rejection reason if any open BUY for ``symbol`` is
        younger than ``min_hold_minutes`` OR carries the
        ``reactor_momentum`` entry-type. ``None`` allows the SELL.

        Two flavours of lockout share this gate:

        * **Time-window lockout** for scheduled-cycle entries
          (default 30 min) — protects against the LLM second-guessing
          fresh positions on noise. Past the window, SELLs proceed.
        * **Permanent lockout** for reactor-driven momentum entries
          (any open BUY with ``entry_type='reactor_momentum'``) —
          the operator's "slow out" discipline (memory:
          strategy-fast-in-slow-out). LLM CANNOT close these; only the
          monitor's rule-based exit (trailing stop / trend break) can.

        Operator escape hatch: ``min_hold_minutes <= 0`` disables the
        time-window check but does NOT disable the reactor-momentum
        lockout — that's a deliberate policy gate, not a tunable.
        Repo errors degrade to "allow".
        """
        try:
            opens = await self._repo.get_open_trades()
        except Exception as exc:  # noqa: BLE001
            logger.debug("min-hold lookup failed for %s: %s — allowing sell", symbol, exc)
            return None
        from datetime import UTC, datetime

        wanted = symbol.upper()
        now = datetime.now(UTC)
        youngest_age: float | None = None
        has_reactor_momentum = False
        for trade in opens:
            t_sym = str(getattr(trade, "symbol", "") or "").upper()
            if t_sym != wanted:
                continue
            if str(getattr(trade, "entry_type", "") or "") == "reactor_momentum":
                has_reactor_momentum = True
            ts = getattr(trade, "timestamp", None)
            if not isinstance(ts, datetime):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            age_min = (now - ts).total_seconds() / 60.0
            if youngest_age is None or age_min < youngest_age:
                youngest_age = age_min

        # Permanent lockout — overrides any time-window relaxation.
        if has_reactor_momentum:
            reason = (
                f"reactor-momentum lockout: {symbol} has an open BUY tagged "
                "'reactor_momentum'. LLM-initiated SELLs on these positions "
                "are permanently blocked — exits go through the monitor's "
                "rule-based trailing stop / trend-break detector. "
                "(Operator memory: strategy-fast-in-slow-out — don't second-guess "
                "winners on noise.)"
            )
            logger.warning(
                "SELL rejected by reactor-momentum lockout: %s",
                symbol,
            )
            return reason

        if self._min_hold_minutes <= 0:
            return None
        if youngest_age is None or youngest_age >= self._min_hold_minutes:
            return None

        reason = (
            f"min-hold gate: {symbol} youngest BUY is only {youngest_age:.0f} min "
            f"old (min hold {self._min_hold_minutes} min). LLM-initiated SELL "
            "blocked to prevent whipsaw. Stop-loss / take-profit exits via the "
            "monitor are unaffected; wait for the hold window to elapse, or fire "
            "the monitor's SL if the position is genuinely broken."
        )
        logger.warning(
            "SELL rejected by min-hold gate: %s (youngest BUY %.0f min, min %d)",
            symbol,
            youngest_age,
            self._min_hold_minutes,
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

    @staticmethod
    def _eod_exit_price(
        sym: str, opens: list[Any], price_by_symbol: dict[str, float]
    ) -> float | None:
        """Best usable exit price for an EOD close, or None.

        Resolution order, each gated on ``> 0``:

        1. pre-close broker snapshot (``current_price`` of the live
           position) — the truth when the broker still holds it;
        2. the open BUY's ``filled_price`` — the confirmed entry fill;
        3. the open BUY's estimated ``price`` — a breakeven fallback so
           an unconfirmed (``filled_price`` NULL) position closes flat
           rather than at a fabricated $0.

        Returns None only when none of these yields a positive price,
        which the caller treats as "don't fabricate a SELL".
        """
        snap = price_by_symbol.get(sym)
        if snap is not None and snap > 0:
            return float(snap)
        fill_fallback: float | None = None
        entry_fallback: float | None = None
        for trade in opens:
            if str(getattr(trade, "symbol", "") or "").upper() != sym:
                continue
            fp = getattr(trade, "filled_price", None)
            if fill_fallback is None and fp is not None and float(fp) > 0:
                fill_fallback = float(fp)
            ep = getattr(trade, "price", None)
            if entry_fallback is None and ep is not None and float(ep) > 0:
                entry_fallback = float(ep)
        if fill_fallback is not None:
            return fill_fallback
        if entry_fallback is not None:
            return entry_fallback
        return None

    async def close_all(self) -> Any:
        """Close all open positions (end of day).

        Fires the broker-side close-all, then walks the DB and stamps
        ``closed_at`` on every open BUY so the next morning's reconcile
        doesn't show drift on positions that were genuinely closed.

        Without the DB cleanup, the previous EOD-close pattern left
        orphan BUYs that surfaced as 100% reconcile drift the next day
        (observed 2026-05-22 10:30 ET on SHOP/NOW/MSFT — all closed at
        EOD on 2026-05-21 but DB still showed them open).

        Pre-fetches the broker's pre-close positions so we know which
        exit price to stamp on each BUY. If the pre-fetch fails (broker
        flake), falls back to the BUY's last-known filled_price so the
        row is at least closed cleanly.
        """
        logger.info("Closing all positions (end of day)")

        # Snapshot per-symbol prices BEFORE the broker call so the DB
        # rows get a meaningful exit_price. close_all_positions is
        # destructive; positions are gone after.
        price_by_symbol: dict[str, float] = {}
        try:
            pre_positions = await self._broker.get_all_positions()
            for p in pre_positions:
                sym = str(getattr(p, "symbol", "") or "").upper()
                cp = float(getattr(p, "current_price", 0) or 0)
                if sym and cp > 0:
                    price_by_symbol[sym] = cp
        except Exception as exc:  # noqa: BLE001
            logger.debug("EOD pre-position snapshot failed: %s", exc)

        result = await self._broker.close_all_positions()

        # Stamp closed_at on every open BUY. We don't know per-symbol
        # exit_price for sure (broker may have filled at a slightly
        # different price than the pre-snapshot), but the snapshot is
        # accurate within the second. close_open_trades_for_symbol
        # already handles "no open BUYs" → noop.
        try:
            opens = await self._repo.get_open_trades()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EOD close-all: get_open_trades failed (%s) — DB orphans may "
                "show as drift on tomorrow's first cycle",
                exc,
            )
            return result

        symbols_to_close: set[str] = set()
        for trade in opens:
            sym = str(getattr(trade, "symbol", "") or "").upper()
            if sym:
                symbols_to_close.add(sym)

        closer = getattr(self._repo, "close_open_trades_for_symbol", None)
        if closer is None:
            logger.debug("EOD close-all: repo has no close_open_trades_for_symbol")
            return result

        # Pre-compute total open qty per symbol — needed for the
        # synthetic SELL trade we'll record so the reconciler's signed-
        # net math nets to zero.
        open_qty_by_sym: dict[str, float] = {}
        for trade in opens:
            sym = str(getattr(trade, "symbol", "") or "").upper()
            if not sym:
                continue
            q = getattr(trade, "filled_quantity", None) or getattr(trade, "quantity", 0)
            try:
                open_qty_by_sym[sym] = open_qty_by_sym.get(sym, 0.0) + float(q or 0)
            except (TypeError, ValueError):
                continue

        total_closed = 0
        from datetime import datetime as _dt

        now_utc = _dt.now(UTC)
        for sym in symbols_to_close:
            exit_price = self._eod_exit_price(sym, opens, price_by_symbol)
            if exit_price is None:
                # No usable exit price anywhere: not in the pre-close
                # snapshot (broker didn't report the position — a
                # reverse-orphan / never-filled phantom), no fill price,
                # and no entry estimate. Recording a $0 synthetic SELL
                # here is worse than doing nothing: it stamps exit_price=0
                # on the BUY, which shows up as a fabricated −100% loss in
                # P&L analytics (observed 2026-05-22 on CSCO). Leave the
                # row for ``fix_stocks_orphans`` to resolve from broker
                # truth instead of inventing a price.
                logger.warning(
                    "EOD close-all: no usable exit price for %s — skipping "
                    "synthetic SELL (leaving the orphan for fix_stocks_orphans "
                    "rather than fabricating a $0 close)",
                    sym,
                )
                continue
            try:
                n = await closer(sym, exit_price, "eod_close_all")
                total_closed += n
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EOD close-all: failed to stamp closed_at on %s: %s", sym, exc
                )
                continue

            # Record a synthetic SELL Trade row so the reconciler's
            # signed-net math (sum of BUY filled_qty minus SELL
            # filled_qty) cancels for this symbol. Without this, the
            # next morning's reconcile shows db=<original-qty>
            # broker=0 → 100% drift (observed 2026-05-22 on
            # SHOP/NOW/MSFT after the EOD on 2026-05-21).
            total_qty = open_qty_by_sym.get(sym, 0.0)
            if total_qty <= 0:
                continue
            try:
                await self._repo.record_trade(
                    symbol=sym,
                    side="sell",
                    quantity=total_qty,
                    price=exit_price,
                    order_id="",
                    status="filled",
                    llm_reasoning="EOD close-all (synthetic exit)",
                    submitted_at=now_utc,
                    filled_at=now_utc,
                    filled_price=exit_price,
                    filled_quantity=total_qty,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EOD close-all: failed to record synthetic SELL for %s: %s",
                    sym,
                    exc,
                )
        if total_closed:
            logger.info("EOD close-all: stamped closed_at on %d open BUY(s)", total_closed)
        return result

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
