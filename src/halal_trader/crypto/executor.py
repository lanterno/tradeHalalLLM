"""Crypto order execution — translates LLM decisions into Binance orders."""

import logging
import math
import time
from datetime import UTC, datetime
from typing import Any

from binance import BinanceAPIException

from halal_trader.core import events
from halal_trader.core.executor import BaseExecutor
from halal_trader.core.fills import confirm_binance
from halal_trader.crypto.exchange import DUST_NOTIONAL_USD, BinanceClient
from halal_trader.db.repository import Repository
from halal_trader.domain.models import (
    CryptoAccount,
    CryptoTradingPlan,
)

logger = logging.getLogger(__name__)

_DUST_NOTIONAL_THRESHOLD = DUST_NOTIONAL_USD
_MIN_BUY_NOTIONAL = 50.0
_MAX_SLIPPAGE_PCT = 0.005  # 0.5%


class CryptoExecutor(BaseExecutor):
    """Executes crypto trading decisions via the Binance client."""

    def __init__(
        self,
        broker: BinanceClient,
        repo: Repository,
        *,
        max_position_pct: float,
        max_simultaneous_positions: int,
        configured_pairs: list[str] | None = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_window: int = 600,
        circuit_breaker_cooldown: int = 1800,
        exiting_pairs: set[str] | None = None,
    ) -> None:
        super().__init__(
            repo,
            max_position_pct=max_position_pct,
            max_simultaneous_positions=max_simultaneous_positions,
        )
        self._broker = broker
        self._tracked_bases = {
            p.upper().removesuffix("USDT").removesuffix("BUSD") for p in (configured_pairs or [])
        }
        # NOTE: `_pair_errors` and `_exiting_pairs` are single-asyncio-loop only —
        # not thread-safe. The monitor and executor share the SAME `_exiting_pairs`
        # set; mutations there are guarded by `PositionMonitor._exit_lock`.
        self._pair_errors: dict[str, list[float]] = {}
        self._cb_threshold = circuit_breaker_threshold
        self._cb_window = circuit_breaker_window
        self._cb_cooldown = circuit_breaker_cooldown
        self._exiting_pairs: set[str] = exiting_pairs if exiting_pairs is not None else set()

    def is_pair_blocked(self, symbol: str) -> bool:
        """Check if a pair is temporarily blocked due to repeated errors."""
        errors = self._pair_errors.get(symbol.upper(), [])
        if not errors:
            return False
        now = time.monotonic()
        recent = [t for t in errors if now - t < self._cb_window]
        self._pair_errors[symbol.upper()] = recent
        if len(recent) >= self._cb_threshold:
            oldest = min(recent)
            if now - oldest < self._cb_cooldown:
                return True
        return False

    def _record_pair_error(self, symbol: str) -> None:
        sym = symbol.upper()
        errors = self._pair_errors.setdefault(sym, [])
        errors.append(time.monotonic())
        now = time.monotonic()
        kept = [t for t in errors if now - t < self._cb_cooldown]
        self._pair_errors[sym] = kept
        # Threshold-cross emits a structured event so operators can grep
        # for it; the AlertSink wiring lives in the cycle (this module is
        # broker-agnostic and shouldn't import notification code).
        if len(kept) == self._cb_threshold:
            logger.error(
                "Circuit breaker tripped for %s after %d errors in %ds",
                sym,
                self._cb_threshold,
                self._cb_window,
                extra={
                    "event": "executor.circuit_breaker.tripped",
                    "pair": sym,
                    "errors": self._cb_threshold,
                    "window_s": self._cb_window,
                },
            )

    async def execute_plan(
        self,
        plan: CryptoTradingPlan,
        account: CryptoAccount | None = None,
    ) -> list[dict[str, Any]]:
        """Execute all decisions in a CryptoTradingPlan."""
        if account is None:
            account = await self._broker.get_account()
        return await self._execute_plan_common(plan, account=account)

    def _get_sells(self, plan: Any) -> list[Any]:
        return plan.sells

    def _get_buys(self, plan: Any) -> list[Any]:
        return plan.buys

    async def _get_current_position_count(self, **_kwargs: Any) -> int:
        balances = await self._broker.get_balances()
        open_count = 0
        if self._tracked_bases:
            for b in balances:
                if b.asset in self._tracked_bases and b.free > 0:
                    price = self._broker.get_cached_price(f"{b.asset}USDT")
                    if price and b.free * price < _DUST_NOTIONAL_THRESHOLD:
                        continue
                    open_count += 1
        else:
            open_count = sum(1 for b in balances if b.asset != "USDT" and b.free > 0)
        return open_count

    def _validate_order(self, symbol: str, _side: str, quantity: float, price: float) -> str | None:
        """Pre-validate an order against Binance filters. Returns error message or None."""
        sf = self._broker.get_symbol_filter(symbol)
        if sf is None:
            notional = quantity * price
            if notional < 5.0:
                return f"Order too small for {symbol}: ${notional:.2f} < $5.00 minimum"
            return None

        notional = quantity * price
        if notional < sf.min_notional:
            return (
                f"Order too small for {symbol}: "
                f"${notional:.2f} < ${sf.min_notional:.2f} minimum notional"
            )
        if quantity < sf.min_qty:
            return f"Qty {quantity} below minimum {sf.min_qty} for {symbol}"
        if sf.max_qty > 0 and quantity > sf.max_qty:
            return f"Qty {quantity} above maximum {sf.max_qty} for {symbol}"
        if sf.step_size > 0:
            remainder = quantity % sf.step_size
            precision = max(0, int(round(-math.log10(sf.step_size))))
            remainder = round(remainder, precision + 2)
            if remainder > 0 and abs(remainder - sf.step_size) > 10 ** -(precision + 2):
                return f"Qty {quantity} not aligned to step size {sf.step_size} for {symbol}"
        return None

    async def _execute_buy(self, decision: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute a crypto buy order."""
        if self.is_pair_blocked(decision.symbol):
            logger.info("Skipping BUY %s — pair temporarily blocked", decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "rejected",
                "reason": "circuit breaker active",
            }

        if decision.symbol in self._exiting_pairs:
            logger.info("Skipping BUY %s — exit in progress", decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "rejected",
                "reason": "exit in progress for this pair",
            }

        existing_open = await self._repo.get_open_crypto_trades_for_pair(decision.symbol)
        if existing_open:
            logger.info(
                "Skipping BUY %s — already have %d open trade(s) for this pair",
                decision.symbol,
                len(existing_open),
            )
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "rejected",
                "reason": f"already have {len(existing_open)} open trade(s)",
            }

        account: CryptoAccount = kwargs["account"]
        price = await self._broker.get_ticker_price(decision.symbol)
        quantity = self._broker.round_quantity(decision.symbol, decision.quantity)
        estimated_cost = price * quantity

        if price > 0 and estimated_cost < _MIN_BUY_NOTIONAL:
            min_qty = self._broker.round_quantity(decision.symbol, _MIN_BUY_NOTIONAL / price)
            if min_qty > 0:
                logger.info(
                    "Scaling up BUY %s from %s ($%.2f) to %s ($%.2f) — below $%.0f floor",
                    decision.symbol,
                    quantity,
                    estimated_cost,
                    min_qty,
                    min_qty * price,
                    _MIN_BUY_NOTIONAL,
                )
                quantity = min_qty
                estimated_cost = price * quantity

        usdt_available = (
            account.usdt_free if account.usdt_free > 0 else account.available_balance_usdt
        )
        if estimated_cost > usdt_available:
            if price > 0 and usdt_available >= _DUST_NOTIONAL_THRESHOLD:
                clamped_qty = self._broker.round_quantity(
                    decision.symbol, (usdt_available * 0.995) / price
                )
                if clamped_qty > 0 and clamped_qty * price >= _DUST_NOTIONAL_THRESHOLD:
                    logger.info(
                        "Clamping BUY %s qty from %s to %s to fit available USDT ($%.2f)",
                        decision.symbol,
                        quantity,
                        clamped_qty,
                        usdt_available,
                    )
                    quantity = clamped_qty
                    estimated_cost = price * quantity
                else:
                    msg = (
                        f"Insufficient USDT for {decision.symbol}: "
                        f"need ${estimated_cost:,.2f}, have ${usdt_available:,.2f} USDT"
                    )
                    logger.warning(msg)
                    return {
                        "symbol": decision.symbol,
                        "action": "buy",
                        "status": "rejected",
                        "reason": msg,
                    }
            else:
                msg = (
                    f"Insufficient USDT for {decision.symbol}: "
                    f"need ${estimated_cost:,.2f}, have ${usdt_available:,.2f} USDT"
                )
                logger.warning(msg)
                return {
                    "symbol": decision.symbol,
                    "action": "buy",
                    "status": "rejected",
                    "reason": msg,
                }

        total = account.total_balance_usdt
        if total > 0 and (estimated_cost / total) > self._max_position_pct:
            msg = f"Position size for {decision.symbol} exceeds {self._max_position_pct:.0%} limit"
            logger.warning(msg)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": msg}

        if err := self._validate_order(decision.symbol, "BUY", quantity, price):
            logger.info("Skipping BUY %s: %s", decision.symbol, err)
            return {"symbol": decision.symbol, "action": "buy", "status": "rejected", "reason": err}

        try:
            submitted_at = datetime.now(UTC)
            order_result = await self._broker.place_order(
                symbol=decision.symbol,
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
            )
            fill = confirm_binance(order_result, submitted_at)
            order_id = fill.order_id
            fill_price = fill.filled_price or price
            db_status = fill.status

            logger.info(
                "Crypto BUY placed: %s qty=%s — orderId=%s status=%s filled=%s",
                decision.symbol,
                quantity,
                order_id,
                db_status,
                fill.filled_quantity,
                extra={
                    "event": events.TRADE_BUY_PLACED,
                    "pair": decision.symbol,
                    "order_id": order_id,
                    "status": db_status,
                    "filled_quantity": fill.filled_quantity,
                    "filled_price": fill.filled_price,
                },
            )

            if db_status == "partially_filled":
                logger.warning(
                    "Partial fill on BUY %s: filled %s of %s @ $%.2f",
                    decision.symbol,
                    fill.filled_quantity,
                    quantity,
                    fill_price,
                    extra={
                        "event": events.TRADE_FILL_PARTIAL,
                        "pair": decision.symbol,
                        "order_id": order_id,
                        "filled_quantity": fill.filled_quantity,
                        "requested_quantity": quantity,
                    },
                )

            slippage = abs(fill_price - price) / price if price > 0 else 0
            if slippage > _MAX_SLIPPAGE_PCT:
                logger.warning(
                    "High slippage on BUY %s: expected $%.2f, filled $%.2f (%.2f%%)",
                    decision.symbol,
                    price,
                    fill_price,
                    slippage * 100,
                )

            trade_id = await self._repo.record_crypto_trade(
                pair=decision.symbol,
                side="buy",
                quantity=quantity,
                price=fill_price,
                order_id=order_id,
                status=db_status,
                llm_reasoning=decision.reasoning,
                entry_price=fill_price,
                stop_loss=decision.stop_loss,
                target_price=decision.target_price,
                submitted_at=fill.submitted_at,
                filled_at=fill.filled_at,
                filled_price=fill.filled_price,
                filled_quantity=fill.filled_quantity,
            )

            return {
                "symbol": decision.symbol,
                "action": "buy",
                "quantity": quantity,
                "price": fill_price,
                "status": db_status,
                "order_id": order_id,
                "trade_id": trade_id,
            }
        except BinanceAPIException as e:
            if e.code in (-1013, -2010):
                logger.warning("BUY %s rejected by exchange: %s", decision.symbol, e)
                return {
                    "symbol": decision.symbol,
                    "action": "buy",
                    "status": "rejected",
                    "reason": str(e),
                }
            logger.error("Failed to place crypto BUY for %s: %s", decision.symbol, e)
            self._record_pair_error(decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "error",
                "reason": str(e),
            }
        except Exception as e:
            logger.error("Failed to place crypto BUY for %s: %s", decision.symbol, e)
            self._record_pair_error(decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "buy",
                "status": "error",
                "reason": str(e),
            }

    async def _execute_sell(self, decision: Any, **_kwargs: Any) -> dict[str, Any]:
        """Execute a crypto sell order, clamping quantity to actual holdings."""
        if self.is_pair_blocked(decision.symbol):
            logger.info("Skipping SELL %s — pair temporarily blocked", decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "sell",
                "status": "rejected",
                "reason": "circuit breaker active",
            }

        if decision.symbol in self._exiting_pairs:
            logger.info("Skipping SELL %s — exit already in progress", decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "sell",
                "status": "rejected",
                "reason": "exit in progress for this pair",
            }

        try:
            base_asset = decision.symbol.upper().removesuffix("USDT").removesuffix("BUSD")
            balances = await self._broker.get_balances()
            actual_free = next((b.free for b in balances if b.asset == base_asset), 0.0)

            quantity = min(decision.quantity, actual_free)
            quantity = self._broker.round_quantity(decision.symbol, quantity)

            if quantity <= 0:
                msg = f"No {base_asset} balance to sell for {decision.symbol}"
                logger.info("Skipping SELL %s: %s", decision.symbol, msg)
                return {
                    "symbol": decision.symbol,
                    "action": "sell",
                    "status": "rejected",
                    "reason": msg,
                }

            price = await self._broker.get_ticker_price(decision.symbol)
            if err := self._validate_order(decision.symbol, "SELL", quantity, price):
                logger.info("Skipping SELL %s: %s", decision.symbol, err)
                return {
                    "symbol": decision.symbol,
                    "action": "sell",
                    "status": "rejected",
                    "reason": err,
                }

            submitted_at = datetime.now(UTC)
            order_result = await self._broker.place_order(
                symbol=decision.symbol,
                side="SELL",
                quantity=quantity,
                order_type="MARKET",
            )
            fill = confirm_binance(order_result, submitted_at)
            order_id = fill.order_id
            fill_price = fill.filled_price
            db_status = fill.status

            logger.info(
                "Crypto SELL placed: %s qty=%s — orderId=%s status=%s filled=%s",
                decision.symbol,
                quantity,
                order_id,
                db_status,
                fill.filled_quantity,
                extra={
                    "event": events.TRADE_SELL_PLACED,
                    "pair": decision.symbol,
                    "order_id": order_id,
                    "status": db_status,
                    "filled_quantity": fill.filled_quantity,
                    "filled_price": fill.filled_price,
                },
            )

            if db_status == "partially_filled":
                logger.warning(
                    "Partial fill on SELL %s: filled %s of %s",
                    decision.symbol,
                    fill.filled_quantity,
                    quantity,
                    extra={
                        "event": events.TRADE_FILL_PARTIAL,
                        "pair": decision.symbol,
                        "order_id": order_id,
                        "filled_quantity": fill.filled_quantity,
                        "requested_quantity": quantity,
                    },
                )

            if fill_price and price > 0:
                slippage = abs(fill_price - price) / price
                if slippage > _MAX_SLIPPAGE_PCT:
                    logger.warning(
                        "High slippage on SELL %s: expected $%.2f, filled $%.2f (%.2f%%)",
                        decision.symbol,
                        price,
                        fill_price,
                        slippage * 100,
                    )

            await self._repo.record_crypto_trade(
                pair=decision.symbol,
                side="sell",
                quantity=quantity,
                price=fill_price,
                order_id=order_id,
                status=db_status,
                llm_reasoning=decision.reasoning,
                submitted_at=fill.submitted_at,
                filled_at=fill.filled_at,
                filled_price=fill.filled_price,
                filled_quantity=fill.filled_quantity,
            )

            return {
                "symbol": decision.symbol,
                "action": "sell",
                "quantity": quantity,
                "price": fill_price,
                "status": db_status,
                "order_id": order_id,
            }
        except BinanceAPIException as e:
            if e.code in (-1013, -2010):
                logger.warning("SELL %s rejected by exchange: %s", decision.symbol, e)
                return {
                    "symbol": decision.symbol,
                    "action": "sell",
                    "status": "rejected",
                    "reason": str(e),
                }
            logger.error("Failed to place crypto SELL for %s: %s", decision.symbol, e)
            self._record_pair_error(decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "sell",
                "status": "error",
                "reason": str(e),
            }
        except Exception as e:
            logger.error("Failed to place crypto SELL for %s: %s", decision.symbol, e)
            self._record_pair_error(decision.symbol)
            return {
                "symbol": decision.symbol,
                "action": "sell",
                "status": "error",
                "reason": str(e),
            }

    @staticmethod
    def _map_order_status(order_result: dict[str, Any]) -> str:
        """Map Binance order status to our internal status."""
        binance_status = order_result.get("status", "")
        if binance_status == "FILLED":
            return "filled"
        if binance_status in ("CANCELED", "REJECTED", "EXPIRED"):
            return "rejected"
        if binance_status == "PARTIALLY_FILLED":
            return "partial"
        return "submitted"
