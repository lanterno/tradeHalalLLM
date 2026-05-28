"""Executor (REARCHITECTURE L6) — sells-first order orchestration. DORMANT.

Turns ``TradeProposal``s into venue orders: process **sells first** (free capital
before buys), feasibility-gate each, place via the :class:`Venue`, confirm the
fill, and emit ``order.submitted`` / ``order.filled`` / ``order.rejected`` with
the ``belief_version`` + ``engine_owner`` for reconcile scoping (INV-8, fix R-02).

Safety:
* INV-7 defense-in-depth — a BUY re-checks the belief's halal gate in the order
  path, not just the policy path.
* INV-2 — a venue error never fabricates a fill; the asset is recorded against
  the per-asset breaker and skipped.
* NEVER instantiated by ``app.build_engine`` — this is dormant until Phase-4.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from halabot.execution.breaker import PerAssetBreaker
from halabot.execution.feasibility import FeasibilityConfig, feasible_buy, feasible_sell
from halabot.execution.venue import Order, OrderResult, Venue, VenueError
from halabot.platform.bus import EventBus
from halabot.platform.clock import Clock
from halabot.platform.events import EventType, new_event
from halabot.policy.policy import TradeProposal

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    equity: float  # account equity → notional = |weight_delta| × equity
    buying_power: float
    position_qty: Callable[[str], float]  # current broker qty for an asset
    halal_ok: Callable[[str], bool] | None = None  # INV-7 defense-in-depth on buys
    engine_owner: str = "belief"


class Executor:
    def __init__(
        self,
        *,
        venue: Venue,
        bus: EventBus,
        clock: Clock,
        breaker: PerAssetBreaker | None = None,
        feasibility: FeasibilityConfig | None = None,
    ) -> None:
        self._venue = venue
        self._bus = bus
        self._clock = clock
        self._breaker = breaker or PerAssetBreaker()
        self._cfg = feasibility or FeasibilityConfig()

    async def execute(
        self, proposals: list[TradeProposal], ctx: ExecutionContext
    ) -> list[OrderResult]:
        """Place orders sells-first; return the fills (rejections excluded)."""
        sells = [p for p in proposals if p.side == "sell"]
        buys = [p for p in proposals if p.side == "buy"]
        results: list[OrderResult] = []
        for p in sells:
            r = await self._execute_one(p, ctx)
            if r is not None:
                results.append(r)
        for p in buys:
            r = await self._execute_one(p, ctx)
            if r is not None:
                results.append(r)
        return results

    async def _execute_one(self, p: TradeProposal, ctx: ExecutionContext) -> OrderResult | None:
        now = self._clock.now()
        if self._breaker.is_open(p.asset, now):
            logger.warning("breaker open for %s — skipping %s", p.asset, p.side)
            return None

        if p.side == "buy" and ctx.halal_ok is not None and not ctx.halal_ok(p.asset):
            logger.warning("INV-7 order-path halal gate blocked buy %s", p.asset)
            return None

        # Sizing (a buy fetches a quote — may raise) and placement share one
        # error path so a venue glitch records the breaker + emits a rejection,
        # while a cold-start "no quote" (rejection=True) is a clean skip.
        try:
            order = await self._size_buy(p, ctx) if p.side == "buy" else self._size_sell(p, ctx)
            if order is None:
                return None  # infeasible — not an error
            await self._publish(EventType.ORDER_SUBMITTED, p, ctx, qty=order.quantity)
            result = await self._venue.place(order)
        except VenueError as exc:
            opened = self._breaker.record_error(p.asset, now, rejection=exc.rejection)
            logger.warning(
                "order error %s %s: %r (breaker_opened=%s)", p.side, p.asset, exc, opened
            )
            await self._publish(EventType.ORDER_REJECTED, p, ctx, qty=0.0, detail=str(exc))
            return None

        self._breaker.record_success(p.asset)
        await self._publish_fill(result, p, ctx)
        return result

    async def _size_buy(self, p: TradeProposal, ctx: ExecutionContext) -> Order | None:
        quote = await self._venue.snapshot(p.asset)  # VenueError handled by caller
        notional = abs(p.weight_delta) * ctx.equity
        feas = feasible_buy(notional, quote.price, buying_power=ctx.buying_power, cfg=self._cfg)
        if not feas.ok:
            logger.info("buy %s infeasible: %s", p.asset, feas.reason)
            return None
        return Order(
            asset=p.asset,
            side="buy",
            quantity=feas.quantity,
            client_id=f"{p.asset}-{p.belief_version}-buy",
            belief_version=p.belief_version,
        )

    def _size_sell(self, p: TradeProposal, ctx: ExecutionContext) -> Order | None:
        held = ctx.position_qty(p.asset)
        feas = feasible_sell(held, cfg=self._cfg)
        if not feas.ok:
            logger.info("sell %s infeasible: %s", p.asset, feas.reason)
            return None
        return Order(
            asset=p.asset,
            side="sell",
            quantity=feas.quantity,
            client_id=f"{p.asset}-{p.belief_version}-sell",
            belief_version=p.belief_version,
        )

    async def _publish(
        self, t: EventType, p: TradeProposal, ctx: ExecutionContext, *, qty: float, detail: str = ""
    ) -> None:
        await self._bus.publish(
            new_event(
                self._clock,
                t,
                source="execution.orders",
                asset=p.asset,
                payload={
                    "side": p.side,
                    "quantity": qty,
                    "belief_version": p.belief_version,
                    "engine_owner": ctx.engine_owner,
                    "detail": detail,
                },
            )
        )

    async def _publish_fill(
        self, r: OrderResult, p: TradeProposal, ctx: ExecutionContext
    ) -> None:
        slippage = None
        if r.filled_price and r.filled_price > 0:
            try:
                decision = (await self._venue.snapshot(p.asset)).price
                if decision > 0:
                    slippage = (r.filled_price - decision) / decision
            except VenueError:
                slippage = None
        await self._bus.publish(
            new_event(
                self._clock,
                EventType.ORDER_FILLED,
                source="execution.orders",
                asset=r.asset,
                payload={
                    "side": r.side,
                    "quantity": r.requested_qty,
                    "filled_price": r.filled_price,
                    "filled_quantity": r.filled_qty,
                    "order_id": r.order_id,
                    "belief_version": p.belief_version,
                    "engine_owner": ctx.engine_owner,
                    "slippage_pct": slippage,
                },
            )
        )
