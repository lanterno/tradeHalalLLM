"""Unified exit authority (REARCHITECTURE Appendix H, L6). DORMANT.

A SINGLE monitor owns every exit — there is no LLM-initiated exit and no fixed
take-profit (a winner is only cut on compliance, a structural break, or genuine
conviction decay). ``decide_exit`` encodes the precedence ladder (first match
wins), evaluated each monitor tick + on relevant events:

  1. risk halt / kill-switch         → flatten (allowed even when halted)
  2. compliance lapsed (real not_halal/doubtful on a held name) → exit, ANY P&L
  3. belief.invalidated              → exit (thesis dead)
  4. hard stop (price <= stop)        → exit (stop_loss)
  5. trend-break (winner, close<SMA)  → exit (trend_break)
  6. trailing ratchet                 → tighten stop (no exit; slow-out)
  7. policy target == 0               → exit (target_zero)
  else                                → hold

Compliance at rung 2 (above all P&L) because halal compliance is non-negotiable
(INV-7). NEVER instantiated by ``app.build_engine`` — dormant until Phase-4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from halabot.execution.venue import OrderResult, Venue
from halabot.platform.bus import EventBus
from halabot.platform.clock import Clock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HoldContext:
    asset: str
    price: float
    stop: float | None = None
    invalidation: float | None = None
    compliance_lapsed: bool = False  # real (non-transient) not_halal/doubtful
    belief_invalidated: bool = False  # price broke the invalidation level
    sma: float | None = None  # trend-break reference (rung 5)
    is_winner: bool = False  # price > entry → trend-break is armed
    trailing_high: float = 0.0  # high-water mark seen so far
    trailing_pct: float = 0.0  # 0 disables the trailing ratchet
    target_weight: float = 0.0  # policy target; <= 0 → conviction decayed
    risk_halted: bool = False
    kill_switch: bool = False


@dataclass(frozen=True)
class ExitDecision:
    action: Literal["exit", "tighten", "hold"]
    reason: str
    new_stop: float | None = None


def decide_exit(ctx: HoldContext) -> ExitDecision:
    """Apply the Appendix-H ladder; first match wins."""
    if ctx.risk_halted or ctx.kill_switch:
        return ExitDecision("exit", "risk_halt")
    if ctx.compliance_lapsed:
        return ExitDecision("exit", "compliance_lapsed")  # INV-7, ANY P&L
    if ctx.belief_invalidated:
        return ExitDecision("exit", "belief_invalidated")
    if ctx.stop is not None and ctx.price <= ctx.stop:
        return ExitDecision("exit", "stop_loss")
    if ctx.is_winner and ctx.sma is not None and ctx.price < ctx.sma:
        return ExitDecision("exit", "trend_break")
    # Trailing ratchet: tighten (never loosen) the stop toward the high-water
    # mark — but ONLY while the policy still wants the position. A fully
    # decayed-conviction position (target <= 0) must fall through to the
    # target_zero exit rather than ratchet forever on new highs (audit #2).
    if ctx.trailing_pct > 0.0 and ctx.target_weight > 0.0:
        high = max(ctx.trailing_high, ctx.price)
        candidate = high * (1.0 - ctx.trailing_pct)
        if ctx.stop is None or candidate > ctx.stop:
            return ExitDecision("tighten", "trailing_stop", new_stop=candidate)
    if ctx.target_weight <= 0.0:
        return ExitDecision("exit", "target_zero")
    return ExitDecision("hold", "hold")


class PositionMonitor:
    """Applies :func:`decide_exit` to held positions: closes on ``exit``, updates
    the stored stop on ``tighten``. The only exit authority. DORMANT."""

    def __init__(self, *, venue: Venue, bus: EventBus, clock: Clock) -> None:
        self._venue = venue
        self._bus = bus
        self._clock = clock
        self._stops: dict[str, float] = {}  # ratcheted stops, persisted per position

    def stop_for(self, asset: str) -> float | None:
        return self._stops.get(asset)

    async def evaluate(self, ctx: HoldContext) -> ExitDecision:
        """Decide + act for one held position. Returns the decision taken."""
        # Use the ratcheted stop if it's tighter than the incoming one.
        stored = self._stops.get(ctx.asset)
        effective = ctx
        if stored is not None and (ctx.stop is None or stored > ctx.stop):
            effective = HoldContext(**{**ctx.__dict__, "stop": stored})
        decision = decide_exit(effective)
        if decision.action == "tighten" and decision.new_stop is not None:
            self._stops[ctx.asset] = decision.new_stop
        elif decision.action == "exit":
            await self._close(ctx.asset, decision.reason)
            self._stops.pop(ctx.asset, None)
        return decision

    async def _close(self, asset: str, reason: str) -> OrderResult | None:
        from halabot.execution.venue import VenueError

        try:
            result = await self._venue.close(asset)
        except VenueError as exc:
            # INV-2: never fabricate a $0 close; log + leave the position for the
            # next tick / reconcile rather than inventing an exit price.
            logger.warning("close %s (%s) failed, will retry: %r", asset, reason, exc)
            return None
        logger.info("EXIT %s (%s) @ %s", asset, reason, result.filled_price)
        return result
