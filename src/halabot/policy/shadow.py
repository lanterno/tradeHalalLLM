"""ShadowPolicyRunner — Phase-3 log-only policy (REARCHITECTURE Part IV Phase 3).

Subscribes to ``belief.updated``; on each, recomputes the whole intended
portfolio (targets → deltas) against a hypothetical shadow book and emits
``policy.trade_proposed`` events. **It never executes** — it exists to A/B the
low-churn conviction behavior against the live cycle's churn on real sessions
before any live flip. The shadow book moves only on actual proposals, so a
stable belief produces no proposal (the anti-churn property, observable).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Protocol

from halabot.belief.store import BeliefStore
from halabot.platform.bus import EventBus, Subscription
from halabot.platform.clock import Clock
from halabot.platform.events import Event, EventType, new_event
from halabot.policy.policy import Policy, TradeProposal
from halabot.policy.portfolio import ShadowPortfolio
from halabot.risk.engine import PortfolioSnapshot, RiskEngine


class PriceSource(Protocol):
    def last_price(self, asset: str) -> float | None: ...

logger = logging.getLogger(__name__)


class ShadowPolicyRunner:
    def __init__(
        self,
        *,
        bus: EventBus,
        store: BeliefStore,
        policy: Policy,
        portfolio: ShadowPortfolio,
        risk_engine: RiskEngine,
        clock: Clock,
        prices: PriceSource | None = None,
        nominal_equity: float = 100_000.0,
        compliance_ttl: timedelta | None = None,
    ) -> None:
        self._bus = bus
        self._store = store
        self._policy = policy
        self._portfolio = portfolio
        self._risk = risk_engine
        self._clock = clock
        self._prices = prices
        self._nominal = nominal_equity
        self._compliance_ttl = compliance_ttl
        self._subs: list[Subscription] = []
        self.proposals_count = 0  # for the A/B (proposed trades over a session)
        self.last_proposals: list[TradeProposal] = []

    def start(self) -> None:
        self._subs.append(self._bus.subscribe({EventType.BELIEF_UPDATED}, self._on_belief))
        # Force-exits (INV-7 compliance_lapsed, price-break invalidation) bypass
        # the conviction path — a held belief turning invalid must close now.
        self._subs.append(
            self._bus.subscribe({EventType.BELIEF_INVALIDATED}, self._on_invalidated)
        )

    def stop(self) -> None:
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()

    def _snapshot(self) -> PortfolioSnapshot:
        # Pure shadow: no realized/unrealized P&L tracked, so risk halts don't
        # fire here (they're unit-tested in risk/). Gross comes from the book.
        return PortfolioSnapshot(
            equity=self._nominal,
            peak_equity=self._nominal,
            unrealized_pnl=0.0,
            realized_pnl_today=0.0,
            starting_equity_today=self._nominal,
            gross_exposure=self._portfolio.gross_exposure(),
        )

    async def _on_invalidated(self, event: Event) -> None:
        """A held belief was invalidated → propose a force-exit to weight 0.

        Models the live monitor's force-exit (Appendix H rung 1–2) in the
        shadow book: bypasses conviction/halal gates because exits always
        reduce risk. No-op if the shadow book doesn't hold the asset.
        """
        asset = event.asset
        if asset is None or not self._portfolio.holds(asset):
            return
        cur = self._portfolio.weight(asset)
        reason = str(event.payload.get("reason", "invalidated"))
        self._portfolio.set_weight(asset, 0.0)
        self.proposals_count += 1
        price = self._prices.last_price(asset) if self._prices is not None else None
        await self._bus.publish(
            new_event(
                self._clock,
                EventType.POLICY_TRADE_PROPOSED,
                source="policy.shadow",
                asset=asset,
                payload={
                    "side": "sell",
                    "target_weight": 0.0,
                    "current_weight": round(cur, 4),
                    "weight_delta": round(-cur, 4),
                    "price": price,
                    "reason": reason,
                    "belief_version": int(event.payload.get("version", 0)),
                    "shadow": True,
                    "forced_exit": True,
                },
                causation=event,
            )
        )
        logger.info("SHADOW force-exit: sell %s → 0 (%s)", asset, reason)

    async def _on_belief(self, event: Event) -> None:
        beliefs = await self._store.all_active()
        by_asset = {b.asset: b for b in beliefs}
        risk = self._risk.evaluate(self._snapshot())
        targets = self._policy.targets(beliefs, self._portfolio, risk)
        proposals = self._policy.deltas(
            targets,
            self._portfolio,
            beliefs_by_asset=by_asset,
            risk=risk,
            now=event.ts,
            compliance_ttl=self._compliance_ttl,
        )
        self.last_proposals = proposals
        for p in proposals:
            self._portfolio.set_weight(p.asset, p.target_weight)  # hypothetical book moves
            self.proposals_count += 1
            price = self._prices.last_price(p.asset) if self._prices is not None else None
            await self._bus.publish(
                new_event(
                    self._clock,
                    EventType.POLICY_TRADE_PROPOSED,
                    source="policy.shadow",
                    asset=p.asset,
                    payload={
                        "side": p.side,
                        "target_weight": round(p.target_weight, 4),
                        "current_weight": round(p.current_weight, 4),
                        "weight_delta": round(p.weight_delta, 4),
                        "price": price,  # decision/fill price for hypothetical P&L
                        "reason": p.reason,
                        "belief_version": p.belief_version,
                        "shadow": True,  # never executed
                    },
                    causation=event,
                )
            )
            logger.info(
                "SHADOW proposal: %s %s Δw=%+.3f → %.3f (%s)",
                p.side,
                p.asset,
                p.weight_delta,
                p.target_weight,
                p.reason,
            )
