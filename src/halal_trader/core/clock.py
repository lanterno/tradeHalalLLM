"""Deterministic time + UUID injection — Round-5 Wave 0.D.

The bot has built up rich replay primitives — `core/replay.py` snapshots
cycle inputs, `ml/fingerprint.py` pins model versions, `core/cycle_timeline.py`
re-derives stage timelines from event logs. The blocker for *bit-perfect*
replay is the still-implicit dependence on wall-clock + ``uuid.uuid4()``.
A cycle that calls ``datetime.now()`` or ``uuid.uuid4()`` directly produces
different outputs on every run — by definition.

This module ships two protocols + their production + test impls:

- :class:`Clock` — ``now() -> datetime`` + ``today() -> date`` +
  ``monotonic() -> float``. Production impl is :class:`SystemClock`.
  Test impl is :class:`FrozenClock` — frozen at construction time,
  optionally advanceable via ``advance(timedelta)``.
- :class:`IdSource` — ``uuid4() -> UUID`` + ``random_token(nbytes) -> str``.
  Production impl is :class:`SystemIdSource`. Test impl is
  :class:`SeededIdSource` — re-derives UUIDs from a seed so two runs
  with the same seed produce identical UUIDs.

Pinned semantics:

- **Both protocols are runtime-checkable** — ducktyping callers can
  pass any object with the required methods.
- **``FrozenClock`` is mutation-only via ``advance()``.** No
  ``set_now()`` — the only way to move the clock is to advance it,
  matching how a replay reconstructs a wall-clock timeline.
- **``SeededIdSource`` is deterministic across runs.** Test ensures
  two instances with the same seed return identical UUIDs in the
  same order.
- **Production impls use the system clock + ``os.urandom``** —
  cryptographically secure where needed.
- **No global singleton.** Every caller takes a Clock + IdSource
  via DI, mirroring how `Settings` already flows.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for wall-clock + monotonic-clock access.

    Production code should accept a ``Clock`` rather than calling
    ``datetime.now()`` directly. Tests pass a :class:`FrozenClock`.
    """

    def now(self) -> datetime:  # pragma: no cover - protocol
        ...

    def today(self) -> date:  # pragma: no cover - protocol
        ...

    def monotonic(self) -> float:  # pragma: no cover - protocol
        ...


@runtime_checkable
class IdSource(Protocol):
    """Protocol for UUID + random-token generation.

    Production code should accept an ``IdSource`` rather than calling
    ``uuid.uuid4()`` directly. Tests pass a :class:`SeededIdSource`.
    """

    def uuid4(self) -> uuid.UUID:  # pragma: no cover - protocol
        ...

    def random_token(self, nbytes: int = 16) -> str:  # pragma: no cover - protocol
        ...


# --- Production impls --------------------------------------------------------


class SystemClock:
    """Wall-clock-backed Clock impl. UTC for ``now``."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def today(self) -> date:
        return self.now().date()

    def monotonic(self) -> float:
        return time.monotonic()


class SystemIdSource:
    """``uuid.uuid4()`` + ``os.urandom``-backed IdSource impl."""

    def uuid4(self) -> uuid.UUID:
        return uuid.uuid4()

    def random_token(self, nbytes: int = 16) -> str:
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        return os.urandom(nbytes).hex()


# --- Test / replay impls -----------------------------------------------------


class FrozenClock:
    """Clock frozen at construction time. Advance via ``advance``."""

    __slots__ = ("_now", "_monotonic")

    def __init__(self, *, now: datetime, monotonic_start: float = 0.0) -> None:
        if now.tzinfo is None:
            raise ValueError("FrozenClock now must be timezone-aware")
        self._now = now
        self._monotonic = monotonic_start

    def now(self) -> datetime:
        return self._now

    def today(self) -> date:
        return self._now.date()

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, delta: timedelta) -> None:
        """Advance the clock by ``delta``. Negative deltas rejected."""
        if delta < timedelta(0):
            raise ValueError("FrozenClock.advance does not support negative deltas")
        self._now = self._now + delta
        # Monotonic advances by the same number of seconds.
        self._monotonic += delta.total_seconds()


class SeededIdSource:
    """Deterministic IdSource — UUIDs derived from an integer seed.

    Two instances seeded the same way return identical UUIDs in the
    same order. Used in replay tests to make UUID-bearing cycle
    snapshots bit-identical across runs.

    Implementation is UUID5 over a deterministic namespace + counter,
    so the output is a real :class:`uuid.UUID` (not a fake string).
    """

    _NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")

    __slots__ = ("_seed", "_counter")

    def __init__(self, *, seed: int = 0) -> None:
        self._seed = seed
        self._counter = 0

    def uuid4(self) -> uuid.UUID:
        # Compose name from seed + counter; UUID5 makes it deterministic
        # while still passing as a UUID. We use uuid4() name to keep the
        # call site's "I want a uuid4" semantics — caller doesn't care
        # about the version, only that it's a UUID.
        name = f"{self._seed}:{self._counter}"
        self._counter += 1
        return uuid.uuid5(self._NAMESPACE, name)

    def random_token(self, nbytes: int = 16) -> str:
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        # Hex over a deterministic byte stream from seed+counter.
        out = bytearray()
        while len(out) < nbytes:
            chunk = uuid.uuid5(
                self._NAMESPACE, f"token:{self._seed}:{self._counter}"
            ).bytes
            self._counter += 1
            out.extend(chunk)
        return out[:nbytes].hex()


__all__ = [
    "Clock",
    "IdSource",
    "SystemClock",
    "SystemIdSource",
    "FrozenClock",
    "SeededIdSource",
]
