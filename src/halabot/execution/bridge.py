"""Live trade bridge (REARCHITECTURE Phase-4 wiring). BUILT ONLY WHEN ARMED.

Subscribes to ``policy.trade_proposed`` and routes each proposal to the
:class:`Executor` against a real venue — the single point where shadow proposals
become live (paper) orders. Enforces the LiveModeChecker's un-loosenable caps
(INV-9): every buy's buying power is clamped to ``max_order_usd``, and buys are
refused once account exposure reaches ``max_account_usd``.

``wire_live_execution`` REFUSES to build the bridge unless the LiveModeChecker
returned ``armed=True`` — so an un-armed (default) engine can never construct the
path that places orders. Tested with a FakeVenue; never a real broker here.
"""

from __future__ import annotations

import logging
from typing import Protocol

from halabot.execution.live_mode import LiveModeDecision
from halabot.execution.orders import ExecutionContext, Executor
from halabot.platform.bus import EventBus, Subscription
from halabot.platform.events import Event, EventType
from halabot.policy.policy import TradeProposal

logger = logging.getLogger(__name__)


class AccountInfo(Protocol):
    """Broker-truth account snapshot the bridge reads each routed proposal."""

    def equity(self) -> float: ...
    def buying_power(self) -> float: ...
    def gross_exposure_usd(self) -> float: ...
    def position_qty(self, asset: str) -> float: ...
    def halal_ok(self, asset: str) -> bool: ...


class LiveTradeBridge:
    def __init__(
        self,
        *,
        bus: EventBus,
        executor: Executor,
        decision: LiveModeDecision,
        account: AccountInfo,
    ) -> None:
        if not decision.armed:
            raise RuntimeError("LiveTradeBridge requires an ARMED LiveModeDecision (INV-9)")
        self._bus = bus
        self._executor = executor
        self._decision = decision
        self._account = account
        self._subs: list[Subscription] = []
        self.routed = 0

    def start(self) -> None:
        self._subs.append(
            self._bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, self._on_proposal)
        )

    def stop(self) -> None:
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()

    async def _on_proposal(self, event: Event) -> None:
        if event.payload.get("shadow") and not self._decision.armed:
            return  # belt-and-suspenders: never route shadow-tagged proposals unarmed
        asset = event.asset
        if asset is None:
            return
        p = TradeProposal(
            asset=asset,
            side=str(event.payload.get("side", "")),
            target_weight=float(event.payload.get("target_weight", 0.0)),
            current_weight=float(event.payload.get("current_weight", 0.0)),
            weight_delta=float(event.payload.get("weight_delta", 0.0)),
            reason=str(event.payload.get("reason", "")),
            belief_version=int(event.payload.get("belief_version", 0)),
            correlation_id=event.correlation_id,  # keep the order on the decision chain
        )
        # INV-9 account-exposure ceiling: refuse new buys past the SAFEGUARD cap.
        if p.side == "buy" and self._account.gross_exposure_usd() >= self._decision.max_account_usd:
            logger.warning(
                "live buy %s refused — account exposure at SAFEGUARD cap $%.0f",
                asset,
                self._decision.max_account_usd,
            )
            return
        ctx = ExecutionContext(
            equity=self._account.equity(),
            # Per-order buying power clamped to the un-loosenable order ceiling.
            buying_power=min(self._account.buying_power(), self._decision.max_order_usd),
            position_qty=self._account.position_qty,
            halal_ok=self._account.halal_ok,
            engine_owner="belief",
        )
        await self._executor.execute([p], ctx)
        self.routed += 1


def wire_live_execution(
    *,
    bus: EventBus,
    executor: Executor,
    decision: LiveModeDecision,
    account: AccountInfo,
) -> LiveTradeBridge:
    """Build + start the live bridge. Raises unless live mode is ARMED."""
    if not decision.armed:
        raise RuntimeError(f"refusing to wire live execution: {decision.reason}")
    bridge = LiveTradeBridge(bus=bus, executor=executor, decision=decision, account=account)
    bridge.start()
    logger.warning("LIVE EXECUTION ARMED for %s — orders will be placed", decision.market)
    return bridge
