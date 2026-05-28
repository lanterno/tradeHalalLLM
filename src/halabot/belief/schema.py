"""The canonical ``BeliefState`` and its parts (REARCHITECTURE.md §6).

Beliefs are *versioned* — each persisted mutation is a new row — so the system
can reconstruct "what did we believe at T and why" (INV-5) and link every open
position to the exact belief version that opened it (INV-8).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID


class Regime(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"


class Direction(StrEnum):
    LONG_BIAS = "long_bias"
    NEUTRAL = "neutral"  # long-only universe (halal) — no SHORT


class Horizon(StrEnum):
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


@dataclass(frozen=True, slots=True)
class Levels:
    """Key price levels. ``invalidation`` is the structural level that, if lost,
    kills the long thesis; ``stop`` mirrors it for the monitor. A ``None``
    invalidation (cold-start asset with no swing structure yet) is valid and
    means "no position until the level engine can set one" (fix R, all-None)."""

    support: float | None = None
    resistance: float | None = None
    stop: float | None = None
    invalidation: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One signed, sourced, decaying contribution to a belief — the "why".

    ``directional`` items (indicators, news, forecaster) carry a signed
    ``direction`` in [-1, +1] that nets into conviction. Non-directional items
    (``anomaly``, ``drift``) carry ``directional=False`` and act as *flags* that
    down-weight conviction without biasing its sign (REARCHITECTURE B.2).
    """

    source: str
    direction: float  # [-1, +1]; ignored when not directional
    weight: float  # 0..1 reliability/recency weight
    detail: str = ""
    ts: datetime | None = None
    event_id: UUID | None = None  # provenance into the event log (INV-5)
    directional: bool = True

    def scaled(self, factor: float) -> "EvidenceItem":
        """Return a copy with ``weight`` multiplied by ``factor`` (decay)."""
        return replace(self, weight=self.weight * factor)


@dataclass(frozen=True, slots=True)
class Catalyst:
    """A scheduled or expected market-moving event with timing + impact."""

    kind: str  # "earnings" | "FOMC" | "CPI" | ...
    scheduled_for: datetime
    expected_impact: float  # 0..1
    detail: str = ""

    def is_imminent(self, now: datetime, *, within_minutes: float = 30.0) -> bool:
        """True when the catalyst is scheduled within ``within_minutes`` of now
        (and not already long past)."""
        delta_min = (self.scheduled_for - now).total_seconds() / 60.0
        return -within_minutes <= delta_min <= within_minutes


@dataclass(frozen=True, slots=True)
class ComplianceVerdict:
    """Cached, transient-error-safe halal screening result (INV-2, INV-7).

    ``transient_error=True`` is a NO-VERDICT (API/transport failure) that must
    never be persisted as a real verdict nor flip a belief — see the cache rule
    in REARCHITECTURE Appendix D.
    """

    asset: str
    status: Literal["halal", "not_halal", "doubtful"]
    detail: str = ""
    screened_at: datetime | None = None
    screening_id: int | None = None  # FK target for the trade row (INV-7)
    transient_error: bool = False

    @classmethod
    def unknown(cls, asset: str) -> "ComplianceVerdict":
        """A doubtful placeholder for a never-screened asset."""
        return cls(asset=asset, status="doubtful", detail="not screened")


@dataclass
class BeliefState:
    """The per-asset world model (mutable in-flight; persisted as versioned rows)."""

    asset: str
    regime: Regime = Regime.RANGING
    regime_confidence: float = 0.0
    direction: Direction = Direction.NEUTRAL
    conviction: float = 0.0  # 0..1, CALIBRATED — policy sizing reads this
    conviction_raw: float = 0.0  # 0..1, PRE-calibration — material_shift's raw-vs-raw test
    horizon: Horizon = Horizon.SWING
    thesis: str = ""
    levels: Levels = field(default_factory=Levels)
    catalysts_pending: list[Catalyst] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    halal: ComplianceVerdict | None = None
    opened_trade_ids: list[int] = field(default_factory=list)
    last_updated: datetime | None = None
    # None until the first thesis is written (material_shift guards on this).
    last_thesis_refresh: datetime | None = None
    version: int = 0

    @classmethod
    def neutral(cls, asset: str) -> "BeliefState":
        """A fresh, opinion-free belief — the cold-start / bootstrap seed."""
        return cls(asset=asset, halal=ComplianceVerdict.unknown(asset))


# Conviction bands for material_shift's LLM-spend throttle (REARCHITECTURE B.3).
# These bucket the RAW score; small wiggles within a band don't trigger an LLM
# thesis refresh. Distinct from the policy's calibrated entry/exit bands.
_CONVICTION_BANDS: tuple[float, ...] = (0.0, 0.30, 0.55, 0.75, 1.0)


def band_index(raw: float) -> int:
    """Bucket a raw conviction score into a band index (0..len-2)."""
    x = max(0.0, min(1.0, raw))
    for i in range(len(_CONVICTION_BANDS) - 1):
        # Upper-inclusive on the final band so 1.0 lands in the top bucket.
        upper = _CONVICTION_BANDS[i + 1]
        if x < upper or (i == len(_CONVICTION_BANDS) - 2 and x <= upper):
            return i
    return len(_CONVICTION_BANDS) - 2
