"""Pre-trade gates (REARCHITECTURE B.6) — the shadow-relevant subset.

Each gate returns a rejection reason or ``None`` to pass. BUY proposals run the
full chain; SELL proposals (exits) bypass the buy-blocking gates because
risk-reducing exits are always allowed. The execution-only gates (market-close
lockout, min-notional, lot/step, buying-power, per-asset breaker) are added in
Phase 4 where an order is actually placed; the gates here are the ones with
meaning for a *proposal* — and halal is enforceable now because the belief
carries its compliance verdict (INV-7).
"""

from __future__ import annotations

from dataclasses import dataclass

from halabot.belief.schema import BeliefState, Direction
from halabot.risk.engine import RiskState


@dataclass(frozen=True)
class GateContext:
    belief: BeliefState
    side: str  # "buy" | "sell"
    target_weight: float
    current_weight: float
    risk: RiskState
    kill_switch: bool = False


def evaluate_gates(ctx: GateContext) -> str | None:
    """Return the first rejection reason, or None if the proposal may proceed."""
    if ctx.kill_switch:
        return "kill-switch engaged"
    # Exits (reducing risk) are always allowed past the buy-blocking gates.
    if ctx.side == "sell":
        return None
    if ctx.risk.halted:
        return f"risk halt: {ctx.risk.reason}"
    if ctx.belief.direction != Direction.LONG_BIAS:
        return "belief not long-biased"
    if not _is_tradeable(ctx.belief):
        return f"halal gate: {_halal_reason(ctx.belief)}"
    return None


def _is_tradeable(b: BeliefState) -> bool:
    v = b.halal
    return v is not None and v.status == "halal" and not v.transient_error


def _halal_reason(b: BeliefState) -> str:
    if b.halal is None:
        return "no verdict"
    if b.halal.transient_error:
        return "screening transient error"
    return f"status={b.halal.status}"
