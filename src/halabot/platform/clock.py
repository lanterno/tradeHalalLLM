"""Injectable clock.

Every component that needs "now" takes a :class:`Clock` rather than calling
``datetime.now()`` directly. This buys two things the re-architecture spec
requires:

* **Test determinism (INV-6).** Tests inject :class:`FakeClock` and advance
  time explicitly — no sleeps, no wall-clock flakiness.
* **Replay correctness (Appendix F).** Belief bootstrap replays historical
  observations with the clock set to each event's ``ts`` (event-time, not
  wall-time), so evidence decays relative to *then*. A wall-clock here would
  age every replayed observation to ~zero and defeat the warm-up (fix R,
  bootstrap time-base).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """The single source of "now" in the engine."""

    def now(self) -> datetime: ...


class SystemClock:
    """Real wall-clock time, always timezone-aware UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    """Manually-advanced clock for tests and replay.

    Holds a fixed instant until :meth:`advance` or :meth:`set` moves it.
    Stored value is always coerced to UTC so comparisons against
    ``SystemClock`` output never raise naive-vs-aware errors.
    """

    def __init__(self, start: datetime) -> None:
        self._now = _as_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        """Move time forward (or back, with a negative delta) and return the new now."""
        self._now += delta
        return self._now

    def set(self, instant: datetime) -> datetime:
        """Jump to an absolute instant (used by replay to pin event-time)."""
        self._now = _as_utc(instant)
        return self._now


def _as_utc(value: datetime) -> datetime:
    """Coerce to a tz-aware UTC datetime (assume UTC if naive)."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def parse_iso(value: object) -> datetime | None:
    """Parse an ISO-8601 string to a datetime, or None for anything unparseable.

    Returns None for non-strings, empty strings, and malformed values — so a
    missing/garbage ``bar_ts`` on an observation becomes a clean fallback rather
    than a ValueError that the bus's handler-isolation would silently swallow
    (dropping the bar). Used by the cognition router and the belief serde."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
