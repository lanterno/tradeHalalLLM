"""BeliefUpdater — the heart. Deterministic-first, LLM-guarded, replay-safe."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.belief.evidence import ContinuousCalendar
from halabot.belief.schema import EvidenceItem, Levels, Regime
from halabot.belief.store import InMemoryBeliefStore
from halabot.belief.updater import BeliefUpdater, UpdaterConfig
from halabot.conviction.raw import IdentityCalibrator
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


# ── fakes for the injected collaborators ──
class FakeRegime:
    def __init__(self, regime=Regime.TRENDING_UP, conf=0.9):
        self.regime, self.conf = regime, conf

    def classify(self, evidence):
        return self.regime, self.conf


class FakeLevels:
    def __init__(self, levels=None):
        self.levels = levels or Levels()

    async def levels_for(self, asset, prev):
        return self.levels


class FakeThesis:
    def __init__(self):
        self.calls = 0

    async def write(self, belief):
        self.calls += 1
        return f"thesis v{self.calls}"


class FakePrices:
    def __init__(self, prices=None):
        self.prices = prices or {}

    def last_price(self, asset):
        return self.prices.get(asset)


class FakePositions:
    def __init__(self, held=()):
        self.held = set(held)

    def has_position(self, asset):
        return asset in self.held


class FakeLLM:
    def __init__(self, available=True, breaker=False):
        self._available, self._breaker = available, breaker

    def available(self):
        return self._available

    def breaker_open(self):
        return self._breaker


def _ev(direction, weight=1.0, *, source="news", ts=T0, directional=True):
    return EvidenceItem(
        source=source, direction=direction, weight=weight, ts=ts, directional=directional
    )


def _build(*, regime=None, levels=None, thesis=None, prices=None, positions=None, llm=None,
           config=None):
    store = InMemoryBeliefStore()
    log = InMemoryEventLog()
    bus = InProcessEventBus(log)
    clock = FakeClock(T0)
    captured: list[Event] = []
    bus.subscribe(
        {EventType.BELIEF_UPDATED, EventType.BELIEF_INVALIDATED},
        lambda e: _append(captured, e),
    )
    updater = BeliefUpdater(
        store=store,
        bus=bus,
        clock=clock,
        calendar=ContinuousCalendar(),
        regime=regime or FakeRegime(),
        levels=levels or FakeLevels(),
        calibrator=IdentityCalibrator(),
        thesis_writer=thesis or FakeThesis(),
        prices=prices or FakePrices(),
        positions=positions or FakePositions(),
        llm=llm or FakeLLM(),
        config=config or UpdaterConfig(),
    )
    return updater, store, captured


async def _append(sink, e):
    sink.append(e)


# ── deterministic update + INV-1 ──
@pytest.mark.asyncio
async def test_evidence_produces_long_bias_and_conviction():
    updater, store, events = _build()
    b = await updater.apply_evidence("NVDA", [_ev(1.0), _ev(1.0)], T0)
    assert b.direction.value == "long_bias"
    assert b.conviction > 0.0
    assert b.version == 1
    assert any(e.type == EventType.BELIEF_UPDATED for e in events)


@pytest.mark.asyncio
async def test_belief_updates_fully_when_llm_unavailable():
    """INV-1: an LLM outage stales the thesis but never the beliefs."""
    thesis = FakeThesis()
    updater, store, _ = _build(thesis=thesis, llm=FakeLLM(available=False))
    b = await updater.apply_evidence("NVDA", [_ev(1.0), _ev(1.0)], T0)
    assert b.conviction > 0.0       # deterministic fields still computed
    assert b.regime == Regime.TRENDING_UP
    assert thesis.calls == 0        # LLM never called


@pytest.mark.asyncio
async def test_thesis_skipped_when_breaker_open():
    thesis = FakeThesis()
    updater, _, _ = _build(thesis=thesis, llm=FakeLLM(available=True, breaker=True))
    await updater.apply_evidence("NVDA", [_ev(1.0)], T0)
    assert thesis.calls == 0        # quota/circuit breaker blocks the call (R-15)


# ── material_shift (R-11) ──
@pytest.mark.asyncio
async def test_thesis_written_on_first_material_evidence():
    thesis = FakeThesis()
    updater, _, _ = _build(thesis=thesis)
    await updater.apply_evidence("NVDA", [_ev(1.0), _ev(1.0)], T0)
    # regime flips RANGING(neutral seed)→TRENDING_UP and raw crosses a band → material
    assert thesis.calls == 1


@pytest.mark.asyncio
async def test_thesis_not_rewritten_when_nothing_material_changes():
    """Second identical update: same regime, same raw band → no LLM spend."""
    thesis = FakeThesis()
    updater, _, _ = _build(thesis=thesis)
    await updater.apply_evidence("NVDA", [_ev(1.0), _ev(1.0)], T0)
    calls_after_first = thesis.calls
    await updater.apply_evidence("NVDA", [_ev(1.0), _ev(1.0)], T0 + timedelta(minutes=1))
    assert thesis.calls == calls_after_first  # no extra LLM call


@pytest.mark.asyncio
async def test_regime_flip_detected_against_prev_not_self():
    """R-11: material_shift compares the PREVIOUS persisted belief, so a genuine
    regime flip on the second update triggers a thesis refresh."""
    thesis = FakeThesis()
    regime = FakeRegime(regime=Regime.TRENDING_UP, conf=0.9)
    updater, _, _ = _build(thesis=thesis, regime=regime)
    await updater.apply_evidence("NVDA", [_ev(1.0)], T0)        # → TRENDING_UP (call 1)
    regime.regime = Regime.VOLATILE                            # flip
    await updater.apply_evidence("NVDA", [_ev(1.0)], T0 + timedelta(minutes=1))
    assert thesis.calls == 2                                   # flip detected vs prev


# ── drift/anomaly flags wired (R-12) ──
@pytest.mark.asyncio
async def test_anomaly_flag_lowers_conviction():
    plain, _, _ = _build()
    b_plain = await plain.apply_evidence("NVDA", [_ev(1.0)], T0)

    flagged, _, _ = _build()
    b_flag = await flagged.apply_evidence(
        "NVDA",
        [_ev(1.0), _ev(0.0, source="anomaly", directional=False)],
        T0,
    )
    assert b_flag.conviction < b_plain.conviction  # flag reached conviction (R-12)


# ── invalidation + replay suppression ──
@pytest.mark.asyncio
async def test_invalidation_fires_when_price_breaks_level():
    updater, _, events = _build(
        levels=FakeLevels(Levels(invalidation=95.0, stop=95.0)),
        prices=FakePrices({"NVDA": 90.0}),  # below invalidation
    )
    await updater.apply_evidence("NVDA", [_ev(1.0)], T0)
    assert any(e.type == EventType.BELIEF_INVALIDATED for e in events)


@pytest.mark.asyncio
async def test_invalidation_not_fired_when_price_above_level():
    updater, _, events = _build(
        levels=FakeLevels(Levels(invalidation=95.0, stop=95.0)),
        prices=FakePrices({"NVDA": 100.0}),
    )
    await updater.apply_evidence("NVDA", [_ev(1.0)], T0)
    assert not any(e.type == EventType.BELIEF_INVALIDATED for e in events)


@pytest.mark.asyncio
async def test_replay_suppresses_invalidation():
    """Bootstrap replay warms beliefs without firing exits vs historical prices."""
    updater, _, events = _build(
        levels=FakeLevels(Levels(invalidation=95.0, stop=95.0)),
        prices=FakePrices({"NVDA": 90.0}),
    )
    await updater.apply_evidence("NVDA", [_ev(1.0)], T0, is_replay=True)
    assert not any(e.type == EventType.BELIEF_INVALIDATED for e in events)
    assert any(e.type == EventType.BELIEF_UPDATED for e in events)  # belief still warmed


# ── heartbeat decay-only (R-08) ──
@pytest.mark.asyncio
async def test_decay_only_update_fades_conviction():
    """An empty-items update (the heartbeat tick) decays evidence so conviction
    fades on the passage of time alone — the market-data-outage de-risk path."""
    updater, store, _ = _build(config=UpdaterConfig(evidence_decay_halflife_min=60))
    await updater.apply_evidence("NVDA", [_ev(1.0), _ev(1.0)], T0)
    before = (await store.get("NVDA")).conviction
    # 120 min later, no new evidence → two half-lives of decay
    await updater.apply_evidence("NVDA", [], T0 + timedelta(minutes=120))
    after = (await store.get("NVDA")).conviction
    assert after < before


# ── compliance ingestion (INV-2 / INV-7) ──
@pytest.mark.asyncio
async def test_set_compliance_stamps_verdict():
    from halabot.belief.schema import ComplianceVerdict

    updater, store, _ = _build()
    await updater.set_compliance(
        "NVDA", ComplianceVerdict("NVDA", "halal", screening_id=7, screened_at=T0), T0
    )
    b = await store.get("NVDA")
    assert b is not None and b.halal is not None
    assert b.halal.status == "halal" and b.halal.screening_id == 7


@pytest.mark.asyncio
async def test_transient_error_does_not_overwrite_good_verdict():
    """INV-2: a screening outage must never flip a real prior verdict."""
    from halabot.belief.schema import ComplianceVerdict

    updater, store, _ = _build()
    await updater.set_compliance("NVDA", ComplianceVerdict("NVDA", "halal", screened_at=T0), T0)
    await updater.set_compliance(
        "NVDA",
        ComplianceVerdict("NVDA", "doubtful", transient_error=True, screened_at=T0),
        T0 + timedelta(minutes=1),
    )
    b = await store.get("NVDA")
    assert b is not None and b.halal is not None
    assert b.halal.status == "halal"  # prior good verdict preserved


@pytest.mark.asyncio
async def test_real_not_halal_verdict_does_overwrite():
    from halabot.belief.schema import ComplianceVerdict

    updater, store, _ = _build()
    await updater.set_compliance("NVDA", ComplianceVerdict("NVDA", "halal", screened_at=T0), T0)
    await updater.set_compliance(
        "NVDA", ComplianceVerdict("NVDA", "not_halal", screened_at=T0), T0 + timedelta(minutes=1)
    )
    b = await store.get("NVDA")
    assert b is not None and b.halal is not None
    assert b.halal.status == "not_halal"  # a REAL verdict does update


# ── lapsed compliance on HELD positions (INV-7, fix R-05) ──
@pytest.mark.asyncio
async def test_lapsed_compliance_on_held_position_forces_exit():
    from halabot.belief.schema import ComplianceVerdict

    updater, _, events = _build(positions=FakePositions(held={"NVDA"}))
    await updater.set_compliance(
        "NVDA", ComplianceVerdict("NVDA", "not_halal", screened_at=T0), T0
    )
    inval = [e for e in events if e.type == EventType.BELIEF_INVALIDATED]
    assert len(inval) == 1
    assert inval[0].payload["reason"] == "compliance_lapsed"


@pytest.mark.asyncio
async def test_lapsed_compliance_not_held_does_not_force_exit():
    from halabot.belief.schema import ComplianceVerdict

    updater, _, events = _build(positions=FakePositions(held=set()))  # not held
    await updater.set_compliance(
        "NVDA", ComplianceVerdict("NVDA", "not_halal", screened_at=T0), T0
    )
    assert not any(e.type == EventType.BELIEF_INVALIDATED for e in events)


@pytest.mark.asyncio
async def test_transient_error_on_held_does_not_force_exit():
    """INV-2: a screening outage on a held name never triggers a forced exit."""
    from halabot.belief.schema import ComplianceVerdict

    updater, _, events = _build(positions=FakePositions(held={"NVDA"}))
    await updater.set_compliance("NVDA", ComplianceVerdict("NVDA", "halal", screened_at=T0), T0)
    await updater.set_compliance(
        "NVDA",
        ComplianceVerdict("NVDA", "doubtful", transient_error=True, screened_at=T0),
        T0 + timedelta(minutes=1),
    )
    assert not any(e.type == EventType.BELIEF_INVALIDATED for e in events)
