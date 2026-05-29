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
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Protocol

from halabot.belief.store import BeliefStore
from halabot.cognition.structure import sma_trend_state
from halabot.platform.bus import EventBus, Subscription
from halabot.platform.clock import Clock
from halabot.platform.events import Event, EventType, new_event
from halabot.policy.policy import Policy, TargetWeight, TradeProposal
from halabot.policy.portfolio import ShadowPortfolio
from halabot.risk.engine import PortfolioSnapshot, RiskEngine, RiskState


class PriceSource(Protocol):
    def last_price(self, asset: str) -> float | None: ...


class PriceHistory(Protocol):
    """Timestamped closing-price history per asset (for the risk correlation pass)."""

    def timestamped_closes(self, asset: str) -> list[tuple[datetime, float]]: ...


logger = logging.getLogger(__name__)


def _returns(closes: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """Consecutive-close returns stamped with the LATER bar's time, so correlation
    can inner-join series on shared timestamps rather than positionally."""
    out: list[tuple[datetime, float]] = []
    for i in range(1, len(closes)):
        prev_c = closes[i - 1][1]
        if prev_c > 0:
            out.append((closes[i][0], (closes[i][1] - prev_c) / prev_c))
    return out


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
        history: PriceHistory | None = None,
        nominal_equity: float = 100_000.0,
        compliance_ttl: timedelta | None = None,
        halt_check: Callable[[], Awaitable[bool]] | None = None,
        benchmark: str | None = None,
        market_gate: bool = False,
        market_sma_window: int = 50,
    ) -> None:
        self._bus = bus
        self._store = store
        self._policy = policy
        self._portfolio = portfolio
        self._risk = risk_engine
        self._clock = clock
        self._prices = prices
        self._history = history
        self._halt_check = halt_check  # operator kill-switch (hb_control via API)
        # Market-regime ("don't fight the tape") gate: when enabled, BUYs are
        # blocked while the benchmark is below its SMA. Off by default.
        self._benchmark = benchmark
        self._market_gate = market_gate
        self._market_sma_window = market_sma_window
        self._halted = False  # for risk.halt edge-emission
        self._nominal = nominal_equity
        self._compliance_ttl = compliance_ttl
        self._subs: list[Subscription] = []
        self._last_target: dict[str, float] = {}  # for target_changed dedup
        self.proposals_count = 0  # for the A/B (proposed trades over a session)
        self.last_proposals: list[TradeProposal] = []
        self.last_rejections: list[tuple[str, str]] = []  # (asset, gate reason) last cycle

    def start(self) -> None:
        self._subs.append(self._bus.subscribe({EventType.BELIEF_UPDATED}, self._on_belief))
        # Force-exits (INV-7 compliance_lapsed, price-break invalidation) bypass
        # the conviction path — a held belief turning invalid must close now.
        self._subs.append(
            self._bus.subscribe({EventType.BELIEF_INVALIDATED}, self._on_invalidated)
        )
        # One recompute per heartbeat covers the decay-only updates (which skip
        # their per-asset recompute) so a tick is O(N), not O(N^2).
        self._subs.append(self._bus.subscribe({EventType.SYSTEM_HEARTBEAT}, self._recompute))

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

    async def _emit_risk_state(self, risk: RiskState, event: Event) -> None:
        """Publish risk.state each cycle (telemetry, INV-5) + risk.halt on the
        edge into a halted state (so the dashboard/operator sees the trigger)."""
        await self._bus.publish(
            new_event(
                self._clock,
                EventType.RISK_STATE,
                source="risk.shadow",
                payload={
                    "portfolio_heat_pct": round(risk.portfolio_heat_pct, 6),
                    "drawdown_pct": round(risk.drawdown_pct, 6),
                    "realized_loss_today_pct": round(risk.realized_loss_today_pct, 6),
                    "gross_exposure": round(risk.gross_exposure, 6),
                    "halted": risk.halted,
                    "reason": risk.reason,
                },
                causation=event,
            )
        )
        if risk.halted and not self._halted:
            await self._bus.publish(
                new_event(
                    self._clock,
                    EventType.RISK_HALT,
                    source="risk.shadow",
                    payload={"reason": risk.reason},
                    causation=event,
                )
            )
            logger.warning("SHADOW risk halt: %s", risk.reason)
        self._halted = risk.halted

    async def _emit_target_changes(self, targets: list[TargetWeight], event: Event) -> None:
        """Emit ``policy.target_changed`` for each target that moved materially —
        the policy output history (telemetry), independent of whether the change
        clears the rebalance threshold to actually trade."""
        for t in targets:
            prev = self._last_target.get(t.asset, 0.0)
            if abs(t.weight - prev) < 1e-6:
                continue
            self._last_target[t.asset] = t.weight
            await self._bus.publish(
                new_event(
                    self._clock,
                    EventType.POLICY_TARGET_CHANGED,
                    source="policy.shadow",
                    asset=t.asset,
                    payload={
                        "target_weight": round(t.weight, 4),
                        "current_weight": round(self._portfolio.effective_weight(t.asset), 4),
                        "reason": t.reason,
                        "belief_version": t.belief_version,
                    },
                    causation=event,
                )
            )

    async def _on_belief(self, event: Event) -> None:
        # A decay-only (heartbeat / no-new-evidence) update skips the per-asset
        # whole-portfolio recompute; the SYSTEM_HEARTBEAT handler recomputes ONCE
        # for the whole decay tick (avoids O(N^2) per heartbeat — perf fix).
        if event.payload.get("decay_only"):
            return
        await self._recompute(event)

    def _market_is_risk_off(self) -> bool:
        """True when the market gate is enabled and the benchmark sits below its
        SMA (risk-off). Reads the benchmark's closing history; safe (False) when
        the gate is off, no benchmark is set, or there is too little history."""
        if not self._market_gate or self._benchmark is None or self._history is None:
            return False
        closes = [c for _, c in self._history.timestamped_closes(self._benchmark)]
        return sma_trend_state(closes, self._market_sma_window) == "below"

    async def _recompute(self, event: Event) -> None:
        beliefs = await self._store.all_active()
        by_asset = {b.asset: b for b in beliefs}
        returns = None
        if self._history is not None:
            returns = {
                b.asset: _returns(self._history.timestamped_closes(b.asset)) for b in beliefs
            }
        risk = self._risk.evaluate(self._snapshot(), beliefs=beliefs, returns_by_asset=returns)
        await self._emit_risk_state(risk, event)
        targets = self._policy.targets(beliefs, self._portfolio, risk)
        await self._emit_target_changes(targets, event)
        kill_switch = await self._halt_check() if self._halt_check is not None else False
        rejections: list[tuple[str, str]] = []
        proposals = self._policy.deltas(
            targets,
            self._portfolio,
            beliefs_by_asset=by_asset,
            risk=risk,
            now=event.ts,
            compliance_ttl=self._compliance_ttl,
            kill_switch=kill_switch,
            market_risk_off=self._market_is_risk_off(),
            on_reject=lambda asset, reason: rejections.append((asset, reason)),
        )
        self.last_rejections = rejections
        if rejections:
            # Gated buys are otherwise invisible — log a per-cycle summary by reason
            # so the operator can SEE the gates (e.g. the market-regime gate) work.
            by_reason: dict[str, list[str]] = {}
            for asset, reason in rejections:
                by_reason.setdefault(reason, []).append(asset)
            summary = "; ".join(
                f"{reason}: {', '.join(sorted(assets))}" for reason, assets in by_reason.items()
            )
            logger.info("SHADOW gated %d buy(s) — %s", len(rejections), summary)
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
                        # Entry regime (telemetry + backtest per-regime P&L
                        # segmentation): which regime the engine believed it was
                        # entering into, so we can measure whether the labels
                        # correlate with realized edge.
                        "regime": str(by_asset[p.asset].regime)
                        if p.asset in by_asset
                        else "unknown",
                        # Evidence sources present at entry (telemetry + backtest
                        # per-source attribution): which signals were behind this
                        # entry, so we can measure which interpreters predict wins.
                        "sources": sorted({e.source for e in by_asset[p.asset].evidence})
                        if p.asset in by_asset
                        else [],
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
