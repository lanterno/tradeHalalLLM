"""BeliefState schema — neutral seed, band_index, catalyst timing, evidence scaling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halabot.belief.schema import (
    BeliefState,
    Catalyst,
    Direction,
    EvidenceItem,
    Regime,
    band_index,
)

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def test_neutral_seed_is_opinion_free():
    b = BeliefState.neutral("NVDA")
    assert b.asset == "NVDA"
    assert b.direction == Direction.NEUTRAL
    assert b.conviction == 0.0
    assert b.conviction_raw == 0.0
    assert b.version == 0
    assert b.halal is not None and b.halal.status == "doubtful"


def test_band_index_buckets():
    assert band_index(0.0) == 0
    assert band_index(0.2) == 0
    assert band_index(0.3) == 1     # boundary lands in the next band
    assert band_index(0.5) == 1
    assert band_index(0.6) == 2
    assert band_index(0.8) == 3
    assert band_index(1.0) == 3     # top inclusive (no IndexError)


def test_band_index_clamps_out_of_range():
    assert band_index(-0.5) == 0
    assert band_index(1.5) == 3


def test_catalyst_is_imminent_window():
    c = Catalyst(kind="earnings", scheduled_for=T0 + timedelta(minutes=10), expected_impact=0.9)
    assert c.is_imminent(T0, within_minutes=30)
    assert not c.is_imminent(T0 - timedelta(hours=2), within_minutes=30)  # 2h early
    # Just-passed catalysts are still "imminent" within the window (news lands late).
    assert c.is_imminent(T0 + timedelta(minutes=15), within_minutes=30)


def test_evidence_scaled_multiplies_weight():
    e = EvidenceItem(source="news", direction=1.0, weight=0.8, ts=T0)
    assert e.scaled(0.5).weight == 0.4
    assert e.scaled(0.5).direction == 1.0  # only weight changes


def test_regime_enum_values():
    assert Regime.TRENDING_UP == "trending_up"
