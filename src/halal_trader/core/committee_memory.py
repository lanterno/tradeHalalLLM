"""Committee memory of decisions — Round-5 Wave 8.C.

The multi-agent committee (`core/llm_committee.py`) makes a verdict
per trade idea. Without memory, the committee re-makes the same
mistake on the same setup type — Bull always argues for momentum,
Bear always argues against, they reach the same compromise, and the
operator pays the same loss every time.

This module is the **memory layer**. Setups are fingerprinted; each
fingerprint accumulates verdict + outcome history; future queries
return a calibrated bias on the committee's prior accuracy for this
setup. The committee can use the prior to either:
- adjust per-role weights (the 8.G meta-learner does this), or
- abstain when the past shows consistently bad performance on this
  setup (e.g. last 5 momentum-low-vol BUYs lost money → SKIP).

The persistent backend is abstracted via `MemoryStore` Protocol so
the production deployment can use Postgres + pgvector while tests
use the in-memory implementation. This module ships the protocol +
in-memory backend + setup-fingerprint + decay logic.

Pinned semantics:

- **Closed-set OutcomeLabel ladder**: WIN / LOSS / FLAT / OPEN.
- **Setup fingerprint is a hash over closed-set fields** — regime,
  rsi-bucket, macd-sign, volume-bucket, trend-tag. Pure-Python; no
  embedding model needed for the fingerprint itself.
- **Recency decay** — half-life 60 days by default. Older entries
  weigh less than recent ones.
- **`recall(fp, k=20)` is deterministic** — sorts by recency.
- **Bias output** is a `MemoryBias` dataclass with WIN_RATE / AVG_RETURN /
  N_EFFECTIVE — the consumer decides how to apply.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — operator IDs masked.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

from halal_trader.core.llm_committee import Stance


class OutcomeLabel(str, Enum):
    """Closed-set outcome ladder."""

    WIN = "win"
    LOSS = "loss"
    FLAT = "flat"
    OPEN = "open"


class RegimeTag(str, Enum):
    """Closed-set regime tag — used in the fingerprint."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    UNKNOWN = "unknown"


class RSIBucket(str, Enum):
    OVERSOLD = "oversold"  # < 30
    NEUTRAL_LOW = "neutral_low"  # 30–45
    NEUTRAL = "neutral"  # 45–55
    NEUTRAL_HIGH = "neutral_high"  # 55–70
    OVERBOUGHT = "overbought"  # > 70


def rsi_to_bucket(rsi: float) -> RSIBucket:
    if rsi < 30:
        return RSIBucket.OVERSOLD
    if rsi < 45:
        return RSIBucket.NEUTRAL_LOW
    if rsi <= 55:
        return RSIBucket.NEUTRAL
    if rsi <= 70:
        return RSIBucket.NEUTRAL_HIGH
    return RSIBucket.OVERBOUGHT


class VolumeBucket(str, Enum):
    LOW = "low"  # < 0.7× avg
    NORMAL = "normal"  # 0.7–1.3× avg
    HIGH = "high"  # > 1.3× avg


def volume_to_bucket(volume_ratio: float) -> VolumeBucket:
    if volume_ratio < 0.7:
        return VolumeBucket.LOW
    if volume_ratio <= 1.3:
        return VolumeBucket.NORMAL
    return VolumeBucket.HIGH


@dataclass(frozen=True)
class SetupFingerprint:
    """A hash-stable fingerprint over closed-set setup features."""

    regime: RegimeTag
    rsi_bucket: RSIBucket
    macd_positive: bool
    volume_bucket: VolumeBucket
    side: Stance  # BUY / SELL — abstaining setups get HOLD/SKIP
    sector: str = "unknown"

    def __post_init__(self) -> None:
        if not self.sector or not self.sector.strip():
            raise ValueError("sector must be non-empty")
        if self.side not in (Stance.BUY, Stance.SELL, Stance.HOLD, Stance.SKIP):
            raise ValueError("side must be a Stance member")

    def digest(self) -> str:
        """Stable 16-char hex digest (sha256 truncated)."""
        key = (
            f"{self.regime.value}|{self.rsi_bucket.value}|"
            f"{1 if self.macd_positive else 0}|{self.volume_bucket.value}|"
            f"{self.side.value}|{self.sector}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MemoryEntry:
    """One persisted committee decision + outcome."""

    fingerprint: SetupFingerprint
    decision_date: date
    stance: Stance
    confidence: float
    outcome: OutcomeLabel = OutcomeLabel.OPEN
    return_pct: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not -1.0 <= self.return_pct <= 5.0:
            raise ValueError("return_pct outside reasonable bounds")
        if self.outcome is OutcomeLabel.OPEN and self.return_pct != 0.0:
            raise ValueError("OPEN entries must have return_pct=0")
        if self.outcome is OutcomeLabel.WIN and self.return_pct <= 0:
            raise ValueError("WIN entry must have positive return_pct")
        if self.outcome is OutcomeLabel.LOSS and self.return_pct >= 0:
            raise ValueError("LOSS entry must have negative return_pct")


@dataclass(frozen=True)
class MemoryBias:
    """Output of `bias_for_fingerprint`."""

    fingerprint_digest: str
    n_total: int
    n_effective: float
    """Decay-weighted count — recent entries weigh more."""
    win_rate: float
    avg_return: float
    last_seen: date | None
    n_open: int
    """Entries still in OPEN state — operator should re-check before
    deciding to abstain."""

    def is_significant(self, min_n_effective: float = 3.0) -> bool:
        return self.n_effective >= min_n_effective


class MemoryStore(Protocol):
    """Protocol for the persistent backend."""

    def insert(self, entry: MemoryEntry) -> None: ...

    def update_outcome(
        self,
        fingerprint_digest: str,
        decision_date: date,
        outcome: OutcomeLabel,
        return_pct: float,
    ) -> int:
        """Mark all entries matching (digest, decision_date) with the
        new outcome. Returns the number of rows updated."""

    def query(self, fingerprint_digest: str, k: int = 20) -> tuple[MemoryEntry, ...]:
        """Return up to k entries for this fingerprint, newest first."""

    def all_entries(self) -> tuple[MemoryEntry, ...]: ...


class InMemoryStore:
    """Simple list-backed store. The production replacement uses
    Postgres + pgvector for similar-setup retrieval; this one matches
    on digest exactly."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def insert(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def update_outcome(
        self,
        fingerprint_digest: str,
        decision_date: date,
        outcome: OutcomeLabel,
        return_pct: float,
    ) -> int:
        updated = 0
        for i, e in enumerate(self._entries):
            if e.fingerprint.digest() == fingerprint_digest and e.decision_date == decision_date:
                # Validate the new state.
                replaced = MemoryEntry(
                    fingerprint=e.fingerprint,
                    decision_date=e.decision_date,
                    stance=e.stance,
                    confidence=e.confidence,
                    outcome=outcome,
                    return_pct=return_pct,
                    notes=e.notes,
                )
                self._entries[i] = replaced
                updated += 1
        return updated

    def query(self, fingerprint_digest: str, k: int = 20) -> tuple[MemoryEntry, ...]:
        matches = [e for e in self._entries if e.fingerprint.digest() == fingerprint_digest]
        matches.sort(key=lambda e: e.decision_date, reverse=True)
        return tuple(matches[:k])

    def all_entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)


def bias_for_fingerprint(
    store: MemoryStore,
    fingerprint: SetupFingerprint,
    *,
    today: date,
    half_life_days: int = 60,
    k: int = 20,
) -> MemoryBias:
    """Compute the decay-weighted bias for a setup fingerprint."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    digest = fingerprint.digest()
    entries = store.query(digest, k=k)
    if not entries:
        return MemoryBias(
            fingerprint_digest=digest,
            n_total=0,
            n_effective=0.0,
            win_rate=0.0,
            avg_return=0.0,
            last_seen=None,
            n_open=0,
        )
    n_open = sum(1 for e in entries if e.outcome is OutcomeLabel.OPEN)
    closed = [e for e in entries if e.outcome is not OutcomeLabel.OPEN]
    if not closed:
        return MemoryBias(
            fingerprint_digest=digest,
            n_total=len(entries),
            n_effective=0.0,
            win_rate=0.0,
            avg_return=0.0,
            last_seen=entries[0].decision_date,
            n_open=n_open,
        )
    weights: list[float] = []
    win_w = 0.0
    ret_w = 0.0
    for e in closed:
        days = max(0, (today - e.decision_date).days)
        w = 0.5 ** (days / half_life_days)
        weights.append(w)
        if e.outcome is OutcomeLabel.WIN:
            win_w += w
        ret_w += e.return_pct * w
    total_w = sum(weights)
    win_rate = win_w / total_w if total_w > 0 else 0.0
    avg_ret = ret_w / total_w if total_w > 0 else 0.0
    return MemoryBias(
        fingerprint_digest=digest,
        n_total=len(entries),
        n_effective=total_w,
        win_rate=win_rate,
        avg_return=avg_ret,
        last_seen=entries[0].decision_date,
        n_open=n_open,
    )


def render_bias(bias: MemoryBias) -> str:
    """Operator-readable summary."""
    if bias.n_total == 0:
        return f"🧠 Memory: no prior entries for {bias.fingerprint_digest}."
    last_str = bias.last_seen.isoformat() if bias.last_seen else "never"
    return (
        f"🧠 Memory[{bias.fingerprint_digest}]: "
        f"n={bias.n_total} (eff={bias.n_effective:.2f}, open={bias.n_open}), "
        f"win_rate={bias.win_rate * 100:.2f}%, "
        f"avg_return={bias.avg_return * 100:+.2f}%, "
        f"last={last_str}"
    )
