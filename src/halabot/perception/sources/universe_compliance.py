"""Halal-universe compliance source — keeps the shadow's verdicts fresh (INV-7).

The shadow treats membership in the legacy screened universe
(``Repository.get_halal_symbols``, i.e. ``compliance='halal'`` rows) as its
halal verdict. That used to be a one-off seed at startup. With the policy's
24h entry-freshness TTL, every verdict went stale a day into a container's
life, after which every shadow BUY failed closed ("verdict stale"), and a
name that LEFT the universe was never told so, which left the INV-7 lapse-exit
path unexercised.

This source re-reads the universe on a cadence well inside the TTL and emits:

* ``halal`` for every current member (refreshes ``screened_at``), and
* ``doubtful`` for every name that was a member at the previous poll and no
  longer is. That is a real, non-transient verdict, so a held name is
  force-exited (lapse) and new entries are blocked.

An EMPTY universe is treated as a transient read failure, not as "everything
lapsed": the legacy cache returns [] when it is empty or unreachable, and
mass-exiting the whole book on a DB hiccup would be wrong. In that case
nothing is emitted, prior verdicts age out on their own, and entries fail
closed through the staleness gate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from halabot.perception.poll import PollingSource
from halabot.platform.clock import Clock
from halabot.platform.events import Event, EventType, new_event

UniverseProvider = Callable[[], Awaitable[list[str]]]
# Hourly: an order of magnitude inside the 24h entry-freshness TTL, and the
# read is one indexed DB query, so the cadence costs nothing.
_DEFAULT_INTERVAL_S = 3600.0


class UniverseComplianceSource(PollingSource):
    def __init__(
        self,
        universe: UniverseProvider,
        clock: Clock,
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
        exclude: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> None:
        super().__init__("halal-universe", interval_s=interval_s, **kwargs)
        self._universe = universe
        self._clock = clock
        # Symbols fed for context only (the relative-strength benchmark) must
        # never be stamped halal; the policy's halal gate is what keeps them
        # untradeable.
        self._exclude = exclude
        self._members: set[str] = set()

    async def fetch(self) -> list[dict[str, Any]]:
        current = {s for s in await self._universe() if s not in self._exclude}
        if not current:
            return []  # transient: see module docstring
        departed = self._members - current
        self._members = current
        return [{"_asset": s, "member": True} for s in sorted(current)] + [
            {"_asset": s, "member": False} for s in sorted(departed)
        ]

    def to_event(self, raw: dict[str, Any]) -> Event | None:
        asset = raw.get("_asset")
        if not asset:
            return None
        member = bool(raw.get("member"))
        return new_event(
            self._clock,
            EventType.COMPLIANCE_VERDICT,
            source="halal-universe",
            asset=asset,
            payload={
                "status": "halal" if member else "doubtful",
                "detail": "halal universe member" if member else "left the halal universe",
                "screening_id": None,
                "transient_error": False,
            },
        )

    # No dedup: re-emitting each cadence is what refreshes screened_at.
