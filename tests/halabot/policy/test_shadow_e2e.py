"""ShadowPolicyRunner — belief.updated → log-only proposals, anti-churn, no execution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.belief.schema import BeliefState, ComplianceVerdict, Direction
from halabot.belief.store import InMemoryBeliefStore
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType, new_event
from halabot.platform.session import is_regular_session
from halabot.policy.policy import Policy
from halabot.policy.portfolio import ShadowPortfolio
from halabot.policy.shadow import ShadowPolicyRunner
from halabot.policy.sizing import PolicyConfig
from halabot.risk.engine import BasicRiskEngine

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


def _bullish(asset="NVDA", conviction=0.9, *, status="halal", direction=Direction.LONG_BIAS):
    return BeliefState(
        asset=asset,
        direction=direction,
        conviction=conviction,
        halal=ComplianceVerdict(asset, status),  # type: ignore[arg-type]
    )


async def _build():
    store = InMemoryBeliefStore()
    bus = InProcessEventBus(InMemoryEventLog())
    runner = ShadowPolicyRunner(
        bus=bus,
        store=store,
        policy=Policy(PolicyConfig()),
        portfolio=ShadowPortfolio(),
        risk_engine=BasicRiskEngine(),
        clock=CLOCK,
    )
    runner.start()
    proposed: list[Event] = []
    bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, lambda e: _cap(proposed, e))
    return store, bus, runner, proposed


async def _cap(sink, e):
    sink.append(e)


async def _signal_belief_update(bus, store, belief):
    await store.put(belief)
    await bus.publish(new_event(CLOCK, EventType.BELIEF_UPDATED, source="test", asset=belief.asset))


@pytest.mark.asyncio
async def test_decay_only_update_skips_recompute_heartbeat_does_it_once():
    # Perf fix: a decay_only belief.updated must NOT trigger a per-asset recompute;
    # a single SYSTEM_HEARTBEAT recomputes the whole book once.
    store, bus, runner, proposed = await _build()
    await store.put(_bullish())
    await bus.publish(
        new_event(
            CLOCK,
            EventType.BELIEF_UPDATED,
            source="belief.updater",
            asset="NVDA",
            payload={"decay_only": True},
        )
    )
    assert runner.proposals_count == 0  # decay_only → skipped, no proposal
    await bus.publish(new_event(CLOCK, EventType.SYSTEM_HEARTBEAT, source="heartbeat"))
    assert runner.proposals_count == 1  # one recompute on the heartbeat → the buy


@pytest.mark.asyncio
async def test_bullish_halal_belief_yields_a_buy_proposal():
    store, bus, runner, proposed = await _build()
    await _signal_belief_update(bus, store, _bullish())
    assert runner.proposals_count == 1
    assert len(proposed) == 1
    assert proposed[0].payload["side"] == "buy"
    assert proposed[0].payload["shadow"] is True  # never executed
    assert runner._portfolio.weight("NVDA") > 0  # hypothetical book moved


@pytest.mark.asyncio
async def test_stable_belief_produces_no_second_proposal():
    """The anti-churn property, observable: re-asserting the same belief moves
    nothing because the shadow book is already at target."""
    store, bus, runner, proposed = await _build()
    await _signal_belief_update(bus, store, _bullish())
    await _signal_belief_update(bus, store, _bullish())  # same conviction again
    assert runner.proposals_count == 1  # no churn


@pytest.mark.asyncio
async def test_non_halal_belief_yields_no_proposal():
    store, bus, runner, proposed = await _build()
    await _signal_belief_update(bus, store, _bullish(status="not_halal"))
    assert runner.proposals_count == 0  # INV-7


@pytest.mark.asyncio
async def test_conviction_decay_to_neutral_proposes_an_exit():
    store, bus, runner, proposed = await _build()
    await _signal_belief_update(bus, store, _bullish())  # buy in
    assert runner._portfolio.weight("NVDA") > 0
    # belief turns neutral (conviction gone) → target 0 → exit proposal
    await _signal_belief_update(bus, store, _bullish(conviction=0.0, direction=Direction.NEUTRAL))
    assert any(p.side == "sell" for p in runner.last_proposals)
    assert runner._portfolio.weight("NVDA") == 0.0  # flattened in the shadow book


@pytest.mark.asyncio
async def test_belief_invalidated_forces_exit_of_held_position():
    """INV-7: a compliance_lapsed invalidation on a held name flattens it,
    bypassing the conviction path (the shadow analogue of the monitor's exit)."""
    store, bus, runner, proposed = await _build()
    await _signal_belief_update(bus, store, _bullish())  # buy in
    assert runner._portfolio.weight("NVDA") > 0
    await bus.publish(
        new_event(
            CLOCK,
            EventType.BELIEF_INVALIDATED,
            source="belief.compliance",
            asset="NVDA",
            payload={"reason": "compliance_lapsed", "version": 2},
        )
    )
    assert runner._portfolio.weight("NVDA") == 0.0  # force-exited
    sells = [p for p in proposed if p.payload["side"] == "sell"]
    assert sells and sells[-1].payload["reason"] == "compliance_lapsed"
    assert sells[-1].payload["forced_exit"] is True


@pytest.mark.asyncio
async def test_belief_invalidated_on_unheld_asset_is_noop():
    store, bus, runner, proposed = await _build()
    await bus.publish(
        new_event(
            CLOCK,
            EventType.BELIEF_INVALIDATED,
            source="belief.compliance",
            asset="TSLA",
            payload={"reason": "compliance_lapsed", "version": 1},
        )
    )
    assert runner.proposals_count == 0  # nothing held → nothing to exit


@pytest.mark.asyncio
async def test_risk_state_emitted_on_belief_update():
    store, bus, runner, _ = await _build()
    risk_states: list[Event] = []
    bus.subscribe({EventType.RISK_STATE}, lambda e: _cap(risk_states, e))
    await _signal_belief_update(bus, store, _bullish())
    assert risk_states  # risk telemetry published each cycle (INV-5)
    assert "gross_exposure" in risk_states[0].payload


@pytest.mark.asyncio
async def test_runner_only_emits_policy_events_never_orders():
    store, bus, runner, _ = await _build()
    all_events: list[Event] = []
    bus.subscribe(set(EventType), lambda e: _cap(all_events, e))
    await _signal_belief_update(bus, store, _bullish())
    kinds = {e.type for e in all_events}
    assert EventType.ORDER_SUBMITTED not in kinds
    assert EventType.ORDER_FILLED not in kinds
    assert EventType.POLICY_TRADE_PROPOSED in kinds


# ── when a hypothetical fill could happen ──
SATURDAY = datetime(2026, 10, 3, 15, 0, tzinfo=UTC)
MONDAY_OPEN = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)  # 10:00 EDT


class _History:
    """The latest bar time per asset (one close each)."""

    def __init__(self, latest: dict[str, datetime]):
        self.latest = latest

    def timestamped_closes(self, asset: str) -> list[tuple[datetime, float]]:
        ts = self.latest.get(asset)
        return [(ts, 100.0)] if ts is not None else []


class _Prices:
    def last_price(self, asset: str) -> float | None:
        return 100.0


async def _gated(clock, *, history=None, max_price_age=None):
    store = InMemoryBeliefStore()
    bus = InProcessEventBus(InMemoryEventLog())
    runner = ShadowPolicyRunner(
        bus=bus,
        store=store,
        policy=Policy(PolicyConfig()),
        portfolio=ShadowPortfolio(),
        risk_engine=BasicRiskEngine(),
        clock=clock,
        prices=_Prices(),
        history=history,
        session=is_regular_session,
        max_price_age=max_price_age,
    )
    runner.start()
    proposed: list[Event] = []
    bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, lambda e: _cap(proposed, e))
    return store, bus, runner, proposed


async def _update(bus, store, clock, belief):
    await store.put(belief)
    await bus.publish(new_event(clock, EventType.BELIEF_UPDATED, source="test", asset=belief.asset))


@pytest.mark.asyncio
async def test_no_proposal_while_the_market_is_closed():
    # Weekend round trips at Friday's last close were booked as 0% losses.
    clock = FakeClock(SATURDAY)
    store, bus, runner, proposed = await _gated(clock)
    await _update(bus, store, clock, _bullish())
    assert proposed == [] and runner._portfolio.weight("NVDA") == 0.0
    clock.set(MONDAY_OPEN)
    await bus.publish(new_event(clock, EventType.SYSTEM_HEARTBEAT, source="heartbeat"))
    assert [p.payload["side"] for p in proposed] == ["buy"]  # proposed again once it can fill


@pytest.mark.asyncio
async def test_stale_price_holds_the_proposal_until_a_fresh_bar():
    clock = FakeClock(MONDAY_OPEN)
    history = _History({"NVDA": MONDAY_OPEN - timedelta(hours=60)})  # Friday's last bar
    store, bus, runner, proposed = await _gated(
        clock, history=history, max_price_age=timedelta(hours=2)
    )
    await _update(bus, store, clock, _bullish())
    assert proposed == []
    history.latest["NVDA"] = MONDAY_OPEN - timedelta(hours=1)
    await bus.publish(new_event(clock, EventType.SYSTEM_HEARTBEAT, source="heartbeat"))
    assert [p.payload["side"] for p in proposed] == ["buy"]


@pytest.mark.asyncio
async def test_a_held_back_sell_holds_the_buys_of_its_rebalance():
    clock = FakeClock(MONDAY_OPEN)
    fresh = MONDAY_OPEN - timedelta(minutes=30)
    history = _History({"NVDA": fresh, "MSFT": fresh})
    store, bus, runner, proposed = await _gated(
        clock, history=history, max_price_age=timedelta(hours=2)
    )
    await _update(bus, store, clock, _bullish())  # hold NVDA
    assert runner._portfolio.weight("NVDA") > 0
    history.latest["NVDA"] = MONDAY_OPEN - timedelta(hours=60)  # NVDA's price goes stale
    await store.put(_bullish("MSFT"))
    await _update(bus, store, clock, _bullish(conviction=0.0, direction=Direction.NEUTRAL))
    assert [p.payload["side"] for p in proposed] == ["buy"]  # nothing after the first buy


@pytest.mark.asyncio
async def test_force_exit_waits_for_the_session():
    clock = FakeClock(MONDAY_OPEN)
    store, bus, runner, proposed = await _gated(clock)
    await _update(bus, store, clock, _bullish())  # hold NVDA
    clock.set(SATURDAY + timedelta(days=7))
    await bus.publish(
        new_event(
            clock,
            EventType.BELIEF_INVALIDATED,
            source="belief.compliance",
            asset="NVDA",
            payload={"reason": "compliance_lapsed", "version": 2},
        )
    )
    assert runner._portfolio.weight("NVDA") > 0  # nothing can fill on a Saturday
    clock.set(MONDAY_OPEN + timedelta(days=7))
    await store.put(_bullish(status="not_halal"))
    await bus.publish(new_event(clock, EventType.SYSTEM_HEARTBEAT, source="heartbeat"))
    sells = [p for p in proposed if p.payload["side"] == "sell"]
    assert len(sells) == 1 and sells[0].payload["forced_exit"] is True
    assert runner._portfolio.weight("NVDA") == 0.0
