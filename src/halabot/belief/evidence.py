"""Evidence aggregation: decay, merge, and the signed-vector summary.

These are the deterministic, LLM-free core of belief formation (REARCHITECTURE
B.1/B.2). Two correctness fixes from the spec review are baked in here:

* **Trading-time decay (R-09):** evidence ages by *trading* minutes, not
  wall-clock, via an injectable :class:`Calendar`. A weekend/overnight gap must
  not annihilate evidence and force a Monday-open mass exit. For 24/7 venues
  the :class:`ContinuousCalendar` makes trading-time == wall-clock.
* **event_id dedup in merge (R, idempotency):** an at-least-once redelivery or
  a bootstrap-replay overlap must not double-count the same evidence.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Protocol
from zoneinfo import ZoneInfo

from halabot.belief.schema import EvidenceItem

# Sources that act as conviction *flags* (down-weight) rather than directional
# evidence. They are excluded from the signed vector and surfaced via has_flag.
FLAG_SOURCES: frozenset[str] = frozenset({"anomaly", "drift"})

_EPS_PRUNE = 1e-4  # drop evidence whose decayed weight falls below this
_CAP_PER_SOURCE = 3  # bound retained items per source


class Calendar(Protocol):
    """Maps a (start, end) interval to the number of *trading* minutes in it."""

    def minutes_between(self, start: datetime, end: datetime) -> float: ...


class ContinuousCalendar:
    """24/7 calendar — trading-time equals wall-clock (crypto, or the
    ``evidence_decay_trading_time=False`` setting)."""

    def minutes_between(self, start: datetime, end: datetime) -> float:
        return max(0.0, (end - start).total_seconds() / 60.0)


_ET = ZoneInfo("America/New_York")
_RTH_OPEN = dt_time(9, 30)
_RTH_CLOSE = dt_time(16, 0)


class RegularHoursCalendar:
    """US equity regular trading hours (Mon–Fri 09:30–16:00 ET), DST-aware via
    zoneinfo. This is the ``evidence_decay_trading_time=True`` calendar for stocks:
    a Friday-close belief does NOT decay over the weekend/overnight, so the engine
    doesn't force a Monday-open mass-exit (R-09). Market holidays are ignored — a
    minor over-count of a few closed days, far safer than the weekend annihilation
    a continuous calendar causes."""

    def minutes_between(self, start: datetime, end: datetime) -> float:
        if end <= start:
            return 0.0
        s = start.astimezone(_ET)
        e = end.astimezone(_ET)
        total = 0.0
        day = s.date()
        while day <= e.date():
            if day.weekday() < 5:  # Mon–Fri
                open_dt = datetime.combine(day, _RTH_OPEN, tzinfo=_ET)
                close_dt = datetime.combine(day, _RTH_CLOSE, tzinfo=_ET)
                lo = max(s, open_dt)
                hi = min(e, close_dt)
                if hi > lo:
                    total += (hi - lo).total_seconds() / 60.0
            day += timedelta(days=1)
        return total


def decay(
    items: list[EvidenceItem],
    now: datetime,
    *,
    halflife_min: float,
    calendar: Calendar,
) -> list[EvidenceItem]:
    """Exponentially decay each item's weight by its *trading-time* age.

    Items whose ``ts`` is None are treated as ageless (no decay) — defensive,
    though live evidence always carries a ``ts``. Fully-decayed items are pruned.
    """
    out: list[EvidenceItem] = []
    for it in items:
        if it.ts is None:
            out.append(it)
            continue
        age = calendar.minutes_between(it.ts, now)
        factor = 0.5 ** (age / halflife_min) if halflife_min > 0 else 1.0
        if it.weight * factor < _EPS_PRUNE:
            continue  # fully decayed — drop
        out.append(it.scaled(factor))
    return out


def merge(
    existing: list[EvidenceItem],
    fresh: list[EvidenceItem],
    *,
    cap_per_source: int = _CAP_PER_SOURCE,
) -> list[EvidenceItem]:
    """Combine evidence, deduping by ``event_id`` and bounding per source.

    Dedup (R, idempotency): a fresh item whose ``event_id`` is already held is
    dropped, so a redelivered/replayed event cannot double-count. Items without
    an ``event_id`` are always kept (synthetic/test evidence).

    Bounding: keep the ``cap_per_source`` newest items per source. All retained
    items still contribute to :func:`weighted_sum` — decay (not replacement) is
    what fades old ones.
    """
    # Dedup fresh against BOTH existing AND already-kept fresh items, in one pass —
    # so a redelivered event whose two copies land in the SAME batch (the coalescing
    # worker concatenates items across coalesced jobs) cannot both survive and
    # double-count conviction's mass factor (fix, idempotency within a batch).
    seen = {it.event_id for it in existing if it.event_id is not None}
    deduped_fresh: list[EvidenceItem] = []
    for it in fresh:
        if it.event_id is not None:
            if it.event_id in seen:
                continue
            seen.add(it.event_id)
        deduped_fresh.append(it)

    by_source: dict[str, list[EvidenceItem]] = defaultdict(list)
    for it in [*existing, *deduped_fresh]:
        by_source[it.source].append(it)

    out: list[EvidenceItem] = []
    for items in by_source.values():
        # Newest first; ts=None sorts oldest so real-dated items win the cap.
        items.sort(key=lambda x: (x.ts is not None, x.ts or datetime.min), reverse=True)
        out.extend(items[:cap_per_source])
    return out


def _directional(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [e for e in items if e.directional and e.source not in FLAG_SOURCES]


def weighted_sum(items: list[EvidenceItem]) -> float:
    """Normalized signed evidence ∈ [-1, +1] — the single ``signed`` value both
    ``direction`` and ``conviction_raw`` derive from, so they never disagree
    (REARCHITECTURE B.2, fix R consistency)."""
    directional = _directional(items)
    if not directional:
        return 0.0
    w = sum(e.weight for e in directional) or 1.0
    return sum(e.direction * e.weight for e in directional) / w


def fraction_same_sign(items: list[EvidenceItem]) -> float:
    """Agreement ∈ [0, 1]: the fraction of directional evidence whose sign
    matches the net direction (a dispersion penalty for conviction)."""
    directional = [e for e in _directional(items) if e.direction != 0.0]
    if not directional:
        return 0.0
    net = weighted_sum(items)
    if net == 0.0:
        return 0.0
    net_sign = 1.0 if net > 0 else -1.0
    agree = sum(1 for e in directional if (e.direction > 0) == (net_sign > 0))
    return agree / len(directional)


def has_flag(items: list[EvidenceItem], source: str) -> bool:
    """True when a non-directional flag from ``source`` (e.g. "anomaly",
    "drift") is present in the evidence."""
    return any(
        e.source == source and (not e.directional or e.source in FLAG_SOURCES) for e in items
    )
