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
from datetime import datetime, timedelta

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
    # INV-7 entry-freshness: a positive verdict is only tradeable while fresh. When
    # both are supplied the halal gate also rejects a verdict older than the TTL
    # (fail-closed). Left None in unit fixtures that aren't exercising freshness.
    now: datetime | None = None
    compliance_ttl: timedelta | None = None


def evaluate_gates(ctx: GateContext) -> str | None:
    """Return the first rejection reason, or None if the proposal may proceed."""
    # Exits (reducing risk) are ALWAYS allowed — even under the kill-switch or a
    # risk halt (Appendix H rung 1: risk-reducing is always permitted). The
    # kill-switch + halt checks sit BELOW this so they never suppress a sell.
    if ctx.side == "sell":
        return None
    if ctx.kill_switch:
        return "kill-switch engaged"
    if ctx.risk.halted:
        return f"risk halt: {ctx.risk.reason}"
    if ctx.belief.direction != Direction.LONG_BIAS:
        return "belief not long-biased"
    if not _is_tradeable(ctx.belief, now=ctx.now, ttl=ctx.compliance_ttl):
        return f"halal gate: {_halal_reason(ctx.belief, now=ctx.now, ttl=ctx.compliance_ttl)}"
    return None


def _is_tradeable(
    b: BeliefState, *, now: datetime | None = None, ttl: timedelta | None = None
) -> bool:
    v = b.halal
    if v is None or v.status != "halal" or v.transient_error:
        return False
    if ttl is not None and now is not None:
        if v.screened_at is None or (now - v.screened_at) > ttl:
            return False  # stale verdict → fail-closed (INV-7 entry freshness)
    return True


def _halal_reason(
    b: BeliefState, *, now: datetime | None = None, ttl: timedelta | None = None
) -> str:
    if b.halal is None:
        return "no verdict"
    if b.halal.transient_error:
        return "screening transient error"
    if b.halal.status == "halal" and ttl is not None and now is not None:
        if b.halal.screened_at is None or (now - b.halal.screened_at) > ttl:
            return "verdict stale"
    return f"status={b.halal.status}"
