"""The simulator's clock: when a bar is visible, when an order works, what runs first.

A bar's ``ts`` is the **start** of its minute; the bar is complete at
``ts + 60 s`` and visible to a playbook at ``ts + 60 s + feed.bar_lag``. A
decision at ``t`` makes a market order active at ``t + ORDER_LAG``; it fills
on the first bar with ``ts >= active_at`` (``exchange.py``).

Worked example (``SIP_RT``): bar 10:14 is visible at 10:15:05, the order is
active at 10:15:08, and it fills on the 10:16 bar. Under ``SIP_DELAYED`` the
same bar is visible at 10:32:00, the order is active at 10:32:03, and it
fills on the 10:33 bar.

Every simulated story has one :class:`EventHeap`, keyed ``(time, priority,
seq)``: ties in time go by :class:`Priority`, then by insertion order, so a
run is deterministic. Times inside the simulator are integer epoch
**microseconds** (``to_us``/``from_us``); bar timestamps are epoch seconds.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import IntEnum
from typing import Final

US: Final = 1_000_000  # microseconds per second
BAR: Final = timedelta(seconds=60)  # a bar spans [ts, ts + 60 s)
FILL_ACK: Final = timedelta(seconds=1)  # a FillIn arrives 1 s after the filling bar ends


@dataclass(frozen=True, slots=True)
class FeedProfile:
    """How late a complete bar reaches the playbook."""

    name: str
    bar_lag: timedelta


SIP_RT: Final = FeedProfile("sip-rt", timedelta(seconds=5))  # H1
SIP_DELAYED: Final = FeedProfile("sip-delayed", timedelta(minutes=17))  # implementability trial
HARNESS: Final = FeedProfile("harness", timedelta(0))  # reactor gate only

ORDER_LAG: Final = timedelta(seconds=3)  # decision to working order
GAP: Final = timedelta(minutes=5)  # a gap this long makes the next fill the bar's open
PRE_OPEN: Final = time(9, 20)  # ET: the compliance check before each held session

# Session markers (SessionIn), relative to the session's open and effective close.
ENTRY_START_AFTER_OPEN: Final = timedelta(minutes=20)
ENTRY_CUTOFF_BEFORE_CLOSE: Final = timedelta(minutes=60)
FLATTEN_BEFORE_CLOSE: Final = timedelta(minutes=5)


class Priority(IntEnum):
    """Order of events that fall at the same instant."""

    EXCHANGE = 0  # working orders tested against a complete bar, at ts + 60 s
    FILL = 1  # FillIn (and OrderClosedIn), at the end of the filling bar + 1 s
    SESSION = 2  # pre_open, open, entry_start, entry_cutoff, flatten, close
    SPY_BAR = 3
    BAR = 4  # the symbol's GapIn (if any), then its BarIn
    NEWS = 5
    TIMER = 6


def to_us(t: datetime) -> int:
    """Epoch microseconds of an aware datetime (exact)."""
    delta = t - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * US + delta.microseconds


def from_us(us: int) -> datetime:
    """The UTC datetime of epoch microseconds (exact)."""
    return _EPOCH + timedelta(microseconds=us)


def span_us(d: timedelta) -> int:
    """A timedelta in microseconds (exact)."""
    return (d.days * 86_400 + d.seconds) * US + d.microseconds


def visible_at(ts: int, feed: FeedProfile) -> int:
    """Epoch second at which the bar starting at ``ts`` (epoch seconds) is visible."""
    return ts + 60 + int(feed.bar_lag.total_seconds())


_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


class EventHeap[T]:
    """A min-heap of events keyed ``(time_us, priority, seq)``.

    ``seq`` is the insertion counter, so two events at the same time and
    priority come out in the order they went in, and the payloads are never
    compared.
    """

    __slots__ = ("_heap", "_seq")

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, T]] = []
        self._seq = 0

    def push(self, at_us: int, priority: int, item: T) -> None:
        heapq.heappush(self._heap, (at_us, priority, self._seq, item))
        self._seq += 1

    def pop(self) -> tuple[int, int, T]:
        at_us, priority, _, item = heapq.heappop(self._heap)
        return at_us, priority, item

    def peek_time(self) -> int | None:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


__all__ = [
    "BAR",
    "ENTRY_CUTOFF_BEFORE_CLOSE",
    "ENTRY_START_AFTER_OPEN",
    "FILL_ACK",
    "FLATTEN_BEFORE_CLOSE",
    "GAP",
    "HARNESS",
    "ORDER_LAG",
    "PRE_OPEN",
    "SIP_DELAYED",
    "SIP_RT",
    "US",
    "EventHeap",
    "FeedProfile",
    "Priority",
    "from_us",
    "span_us",
    "to_us",
    "visible_at",
]
