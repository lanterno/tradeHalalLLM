"""Policy: belief vector → target weights → trade deltas (REARCHITECTURE L5).

``targets`` maps every belief to a per-asset weight, then **normalizes the whole
vector** so total gross exposure never exceeds ``max_gross_exposure`` — no
implicit leverage when many assets convict at once (R-03, INV-10). ``deltas``
emits a trade only when the gap between target and *effective* weight clears the
rebalance threshold (anti-churn) and the gates pass (exits always allowed).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from halabot.belief.schema import BeliefState
from halabot.policy.gates import GateContext, evaluate_gates
from halabot.policy.portfolio import PortfolioState
from halabot.policy.sizing import PolicyConfig, target_weight
from halabot.risk.engine import RiskState


@dataclass(frozen=True)
class TargetWeight:
    asset: str
    weight: float
    reason: str
    belief_version: int


@dataclass(frozen=True)
class TradeProposal:
    asset: str
    side: str  # "buy" | "sell"
    target_weight: float
    current_weight: float
    weight_delta: float
    reason: str
    belief_version: int
    # Carries the decision's correlation_id across the execution boundary so
    # order.* events stay on the same causal chain (INV-5 replay).
    correlation_id: UUID | None = None


class Policy:
    def __init__(self, cfg: PolicyConfig | None = None) -> None:
        self._cfg = cfg or PolicyConfig()

    @property
    def config(self) -> PolicyConfig:
        return self._cfg

    def targets(
        self, beliefs: list[BeliefState], portfolio: PortfolioState, risk: RiskState
    ) -> list[TargetWeight]:
        # A risk halt overrides all conviction: every target collapses to 0 so the
        # book de-risks and the telemetry reflects zero intended exposure while
        # halted (spec L5/L7; exits are still allowed via deltas' sell bypass).
        if risk.halted:
            reason = f"risk halt: {risk.reason}"
            return [TargetWeight(b.asset, 0.0, reason, b.version) for b in beliefs]
        raw: dict[str, float] = {}
        version: dict[str, int] = {}
        for b in beliefs:
            raw[b.asset] = target_weight(b, risk, held=portfolio.holds(b.asset), cfg=self._cfg)
            version[b.asset] = b.version
        gross = sum(raw.values())
        # No implicit leverage: scale the WHOLE vector down if it wants more
        # than the gross cap (R-03). Never scales up.
        if gross > self._cfg.max_gross_exposure and gross > 0:
            factor = self._cfg.max_gross_exposure / gross
            raw = {a: w * factor for a, w in raw.items()}
            reason = "conviction (gross-normalized)"
        else:
            reason = "conviction"
        return [TargetWeight(a, w, reason, version[a]) for a, w in raw.items()]

    def deltas(
        self,
        targets: list[TargetWeight],
        portfolio: PortfolioState,
        *,
        beliefs_by_asset: dict[str, BeliefState],
        risk: RiskState,
        kill_switch: bool = False,
        now: datetime | None = None,
        compliance_ttl: timedelta | None = None,
        market_risk_off: bool = False,
        on_reject: Callable[[str, str], None] | None = None,
    ) -> list[TradeProposal]:
        out: list[TradeProposal] = []
        # Concurrent-position cap (anti-over-diversification): count currently-held
        # names, then refuse a BUY that would OPEN a new one beyond the cap.
        open_count = sum(1 for t in targets if portfolio.holds(t.asset))
        cap = self._cfg.max_open_positions
        for t in targets:
            cur = portfolio.effective_weight(t.asset)  # filled + pending (R-14)
            gap = t.weight - cur
            if abs(gap) < self._cfg.target_rebalance_threshold:
                continue  # belief didn't move the target enough → NO TRADE (anti-churn)
            if cap and gap > 0 and not portfolio.holds(t.asset) and open_count >= cap:
                continue  # at the max-open-positions cap → don't open a new name
            if portfolio.has_open_order(t.asset):
                continue  # one working order per asset; reconcile next tick (R-14)
            side = "buy" if gap > 0 else "sell"
            belief = beliefs_by_asset.get(t.asset)
            if belief is None:
                # Fail-closed (INV-7): with no belief there is no halal verdict, so
                # a BUY must never proceed. A SELL is a risk-reducing exit → allowed.
                if side == "buy":
                    if on_reject is not None:
                        on_reject(t.asset, "no belief (fail-closed)")
                    continue
            else:
                gate_reason = evaluate_gates(
                    GateContext(
                        belief=belief,
                        side=side,
                        target_weight=t.weight,
                        current_weight=cur,
                        risk=risk,
                        kill_switch=kill_switch,
                        now=now,
                        compliance_ttl=compliance_ttl,
                        relstrength_gate=self._cfg.relstrength_gate,
                        market_risk_off=market_risk_off,
                    )
                )
                if gate_reason is not None:
                    # Surface WHY a buy was suppressed (telemetry) — gated proposals
                    # are otherwise invisible (no event emitted).
                    if on_reject is not None:
                        on_reject(t.asset, gate_reason)
                    continue
            if side == "buy" and not portfolio.holds(t.asset):
                open_count += 1  # this buy opens a new name — counts toward the cap
            out.append(
                TradeProposal(
                    asset=t.asset,
                    side=side,
                    target_weight=t.weight,
                    current_weight=cur,
                    weight_delta=gap,
                    reason=t.reason,
                    belief_version=t.belief_version,
                )
            )
        return out
