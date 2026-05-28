"""Per-asset circuit breaker (REARCHITECTURE L6, carried from the executors).

N consecutive *unexpected* order errors on one asset within a window open a
per-asset breaker for a cooldown, quarantining a malfunctioning symbol (bad
filter, venue glitch) instead of letting the continuous target loop retry it
forever. Clean rejections (bad quantity / insufficient funds — Binance -1013 /
-2010) are NOT breaker trips; they're expected outcomes that reset nothing.

Clock-injected: callers pass ``now`` so the breaker is deterministic + testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class _State:
    consecutive: int = 0
    open_until: datetime | None = None


@dataclass
class PerAssetBreaker:
    threshold: int = 3
    cooldown_s: float = 900.0
    _state: dict[str, _State] = field(default_factory=dict)

    def _st(self, asset: str) -> _State:
        return self._state.setdefault(asset, _State())

    def is_open(self, asset: str, now: datetime) -> bool:
        st = self._state.get(asset)
        if st is None or st.open_until is None:
            return False
        if now >= st.open_until:
            st.open_until = None  # cooldown elapsed → close
            st.consecutive = 0
            return False
        return True

    def record_success(self, asset: str) -> None:
        st = self._st(asset)
        st.consecutive = 0
        st.open_until = None

    def record_error(self, asset: str, now: datetime, *, rejection: bool = False) -> bool:
        """Record an order error. ``rejection`` (bad qty/funds) does NOT count.
        Returns True if this error opened the breaker."""
        if rejection:
            return False
        st = self._st(asset)
        st.consecutive += 1
        if st.consecutive >= self.threshold:
            st.open_until = now + timedelta(seconds=self.cooldown_s)
            return True
        return False
