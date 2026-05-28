"""Policy: belief vector → target weights → trade deltas (REARCHITECTURE L5).

``targets`` maps every belief to a per-asset weight, then **normalizes the whole
vector** so total gross exposure never exceeds ``max_gross_exposure`` — no
implicit leverage when many assets convict at once (R-03, INV-10). ``deltas``
emits a trade only when the gap between target and *effective* weight clears the
rebalance threshold (anti-churn) and the gates pass (exits always allowed).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

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


class Policy:
    def __init__(self, cfg: PolicyConfig | None = None) -> None:
        self._cfg = cfg or PolicyConfig()

    @property
    def config(self) -> PolicyConfig:
        return self._cfg

    def targets(
        self, beliefs: list[BeliefState], portfolio: PortfolioState, risk: RiskState
    ) -> list[TargetWeight]:
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
    ) -> list[TradeProposal]:
        out: list[TradeProposal] = []
        for t in targets:
            cur = portfolio.effective_weight(t.asset)  # filled + pending (R-14)
            gap = t.weight - cur
            if abs(gap) < self._cfg.target_rebalance_threshold:
                continue  # belief didn't move the target enough → NO TRADE (anti-churn)
            if portfolio.has_open_order(t.asset):
                continue  # one working order per asset; reconcile next tick (R-14)
            side = "buy" if gap > 0 else "sell"
            belief = beliefs_by_asset.get(t.asset)
            if belief is not None:
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
                    )
                )
                if gate_reason is not None:
                    continue  # gated out (logged by the shadow runner via no proposal)
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
