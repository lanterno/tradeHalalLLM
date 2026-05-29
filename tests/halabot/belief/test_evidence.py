"""Evidence algorithms — decay (trading-time), merge (dedup), signed vector."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from halabot.belief.evidence import (
    ContinuousCalendar,
    decay,
    fraction_same_sign,
    has_flag,
    merge,
    weighted_sum,
)
from halabot.belief.schema import EvidenceItem

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
CAL = ContinuousCalendar()


def _ev(source, direction, weight, *, ts=T0, event_id=None, directional=True):
    return EvidenceItem(
        source=source,
        direction=direction,
        weight=weight,
        ts=ts,
        event_id=event_id,
        directional=directional,
    )


# ── decay ──
def test_decay_halves_weight_at_one_halflife():
    items = [_ev("news", 1.0, 1.0, ts=T0)]
    out = decay(items, T0 + timedelta(minutes=60), halflife_min=60, calendar=CAL)
    assert out[0].weight == 0.5


def test_decay_prunes_fully_decayed_evidence():
    items = [_ev("news", 1.0, 1.0, ts=T0)]
    out = decay(items, T0 + timedelta(hours=48), halflife_min=60, calendar=CAL)
    assert out == []  # weight fell below the prune epsilon


def test_decay_continuous_calendar_uses_wall_clock():
    items = [_ev("news", 1.0, 1.0, ts=T0)]
    out = decay(items, T0 + timedelta(minutes=120), halflife_min=60, calendar=CAL)
    assert out[0].weight == 0.25  # two half-lives


def test_regular_hours_calendar_freezes_weekend():
    from halabot.belief.evidence import RegularHoursCalendar

    cal = RegularHoursCalendar()
    # Fri 16:00 ET (= 20:00 UTC during EDT) → Mon 09:30 ET: zero RTH minutes.
    fri_close = datetime(2026, 5, 29, 20, 0, tzinfo=UTC)
    mon_open = datetime(2026, 6, 1, 13, 30, tzinfo=UTC)  # Mon 09:30 EDT
    assert cal.minutes_between(fri_close, mon_open) == 0.0
    # A 1-hour RTH stretch on a single weekday counts ~60 minutes.
    wed_10 = datetime(2026, 5, 27, 14, 0, tzinfo=UTC)  # 10:00 EDT
    wed_11 = datetime(2026, 5, 27, 15, 0, tzinfo=UTC)  # 11:00 EDT
    assert cal.minutes_between(wed_10, wed_11) == pytest.approx(60.0)


def test_decay_trading_calendar_freezes_closed_gaps():
    """A trading-aware calendar that returns 0 minutes (market closed all
    weekend) must NOT decay evidence — the R-09 weekend-mass-exit fix."""

    class FrozenWeekend:
        def minutes_between(self, start, end):
            return 0.0  # market was closed the whole interval

    items = [_ev("news", 1.0, 1.0, ts=T0)]
    out = decay(items, T0 + timedelta(hours=63), halflife_min=60, calendar=FrozenWeekend())
    assert out[0].weight == 1.0  # undecayed across the closed gap


# ── merge ──
def test_merge_dedups_by_event_id():
    eid = uuid4()
    existing = [_ev("news", 1.0, 1.0, event_id=eid)]
    fresh = [_ev("news", 1.0, 1.0, event_id=eid)]  # same event redelivered
    out = merge(existing, fresh)
    assert len(out) == 1  # not double-counted (R, idempotency)


def test_merge_dedups_duplicate_event_id_within_fresh_batch():
    # Regression: the coalescing worker concatenates items across coalesced jobs,
    # so a redelivered event's two copies can land in the SAME fresh batch. Both
    # must NOT survive (else conviction's mass factor double-counts).
    eid = uuid4()
    dup = _ev("news", 1.0, 1.0, event_id=eid)
    out = merge([], [dup, dup])
    assert len(out) == 1


def test_merge_keeps_distinct_event_ids():
    out = merge(
        [_ev("news", 1.0, 1.0, event_id=uuid4())],
        [_ev("news", 1.0, 1.0, event_id=uuid4())],
    )
    assert len(out) == 2


def test_merge_keeps_items_without_event_id():
    out = merge([_ev("x", 1.0, 1.0)], [_ev("x", 1.0, 1.0)])
    assert len(out) == 2  # no event_id → always kept


def test_merge_caps_per_source_keeping_newest():
    existing = [
        _ev("news", 1.0, 1.0, ts=T0 + timedelta(minutes=m), event_id=uuid4())
        for m in range(5)
    ]
    out = merge(existing, [], cap_per_source=3)
    kept_minutes = sorted(e.ts.minute for e in out)
    assert kept_minutes == [2, 3, 4]  # the 3 newest


# ── weighted_sum / agreement / flags ──
def test_weighted_sum_normalizes_to_unit_interval():
    out = weighted_sum([_ev("a", 1.0, 1.0), _ev("b", 1.0, 1.0)])
    assert out == 1.0


def test_weighted_sum_nets_opposing_evidence():
    out = weighted_sum([_ev("a", 1.0, 1.0), _ev("b", -1.0, 1.0)])
    assert out == 0.0


def test_weighted_sum_excludes_flag_sources():
    # An anomaly flag must not bias the signed vector.
    out = weighted_sum(
        [_ev("a", 1.0, 1.0), _ev("anomaly", -1.0, 1.0, directional=False)]
    )
    assert out == 1.0


def test_fraction_same_sign_full_agreement():
    assert fraction_same_sign([_ev("a", 1.0, 1.0), _ev("b", 0.5, 1.0)]) == 1.0


def test_fraction_same_sign_split():
    out = fraction_same_sign(
        [_ev("a", 1.0, 1.0), _ev("b", 1.0, 1.0), _ev("c", -1.0, 0.1)]
    )
    assert out == 2 / 3


def test_has_flag_detects_nondirectional_source():
    items = [_ev("a", 1.0, 1.0), _ev("drift", 0.0, 1.0, directional=False)]
    assert has_flag(items, "drift")
    assert not has_flag(items, "anomaly")
