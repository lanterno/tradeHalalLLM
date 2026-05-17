"""Anti-frontrunning protection — Round-5 Wave 12.G.

Detects routing patterns that leave the parent order vulnerable to
frontrunning / MEV (Miner / Maximal Extractable Value) on both
traditional exchanges and on-chain venues.

This module ships the **detection engine + mitigation policy
recommendations**. Actual mitigation (reroute, randomise, batch with
peers) lives one layer up.

Pinned semantics:

- **Closed-set FrontrunRisk ladder** — LOW / MEDIUM / HIGH / CRITICAL.
- **Closed-set FrontrunSignal ladder** — predictable-cadence,
  large-relative-size, public-mempool, repeat-counterparty,
  same-block-collision.
- **Closed-set Mitigation ladder** — rerouted via private mempool,
  randomised cadence, batched-peers, broken-up.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class FrontrunRisk(str, Enum):
    """Closed-set frontrun-risk ladder."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FrontrunSignal(str, Enum):
    """Closed-set detection signals."""

    PREDICTABLE_CADENCE = "predictable_cadence"
    LARGE_RELATIVE_SIZE = "large_relative_size"
    PUBLIC_MEMPOOL = "public_mempool"
    REPEAT_COUNTERPARTY = "repeat_counterparty"
    SAME_BLOCK_COLLISION = "same_block_collision"


class Mitigation(str, Enum):
    """Closed-set mitigation strategies."""

    PRIVATE_MEMPOOL = "private_mempool"
    RANDOMISED_CADENCE = "randomised_cadence"
    BATCHED_WITH_PEERS = "batched_with_peers"
    SLICED_SMALLER = "sliced_smaller"


@dataclass(frozen=True)
class FrontrunPolicy:
    """Operator-tunable detection + mitigation thresholds."""

    cadence_cv_threshold: float = 0.10  # coefficient of variation
    large_size_pct_of_volume: float = 0.05  # 5% of recent volume = "large"
    repeat_counterparty_threshold: int = 3  # ≥3 prior interactions

    def __post_init__(self) -> None:
        if not 0.0 < self.cadence_cv_threshold < 1.0:
            raise ValueError("cadence_cv_threshold must be in (0, 1)")
        if not 0.0 < self.large_size_pct_of_volume <= 1.0:
            raise ValueError("large_size_pct_of_volume must be in (0, 1]")
        if self.repeat_counterparty_threshold <= 0:
            raise ValueError("repeat_counterparty_threshold must be positive")


@dataclass(frozen=True)
class OrderSignal:
    """Inputs for frontrun detection on a parent order."""

    parent_id: str
    submission_times: tuple[datetime, ...]
    parent_quantity: float
    recent_volume: float
    venue_is_public_mempool: bool
    counterparty_repeat_count: int = 0
    same_block_neighbours: int = 0

    def __post_init__(self) -> None:
        if not self.parent_id or not self.parent_id.strip():
            raise ValueError("parent_id must be non-empty")
        if self.parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        if self.recent_volume < 0:
            raise ValueError("recent_volume must be non-negative")
        if self.counterparty_repeat_count < 0:
            raise ValueError("counterparty_repeat_count must be non-negative")
        if self.same_block_neighbours < 0:
            raise ValueError("same_block_neighbours must be non-negative")
        for t in self.submission_times:
            if t.tzinfo is None:
                raise ValueError("submission_times must be timezone-aware")


@dataclass(frozen=True)
class FrontrunAssessment:
    """Result of running frontrun detection."""

    parent_id: str
    risk: FrontrunRisk
    signals: frozenset[FrontrunSignal]
    recommended_mitigations: frozenset[Mitigation]


def _cadence_cv(times: Sequence[datetime]) -> float:
    """Coefficient of variation of inter-submission gaps."""
    if len(times) < 3:
        return 1.0  # too few to be predictable; high CV
    sorted_t = sorted(times)
    gaps = [(sorted_t[i + 1] - sorted_t[i]).total_seconds() for i in range(len(sorted_t) - 1)]
    mean = statistics.mean(gaps)
    if mean == 0:
        return 0.0
    stdev = statistics.pstdev(gaps)
    return stdev / mean


def assess(signal: OrderSignal, *, policy: FrontrunPolicy | None = None) -> FrontrunAssessment:
    """Run the detection engine and return an assessment + mitigations."""
    pol = policy if policy is not None else FrontrunPolicy()
    signals: set[FrontrunSignal] = set()
    mitigations: set[Mitigation] = set()

    # 1. Cadence predictability
    cv = _cadence_cv(signal.submission_times)
    if cv < pol.cadence_cv_threshold and len(signal.submission_times) >= 3:
        signals.add(FrontrunSignal.PREDICTABLE_CADENCE)
        mitigations.add(Mitigation.RANDOMISED_CADENCE)

    # 2. Large relative size
    if signal.recent_volume > 0:
        size_ratio = signal.parent_quantity / signal.recent_volume
        if size_ratio >= pol.large_size_pct_of_volume:
            signals.add(FrontrunSignal.LARGE_RELATIVE_SIZE)
            mitigations.add(Mitigation.SLICED_SMALLER)

    # 3. Public mempool
    if signal.venue_is_public_mempool:
        signals.add(FrontrunSignal.PUBLIC_MEMPOOL)
        mitigations.add(Mitigation.PRIVATE_MEMPOOL)

    # 4. Repeat counterparty
    if signal.counterparty_repeat_count >= pol.repeat_counterparty_threshold:
        signals.add(FrontrunSignal.REPEAT_COUNTERPARTY)
        mitigations.add(Mitigation.BATCHED_WITH_PEERS)

    # 5. Same-block collision
    if signal.same_block_neighbours > 0:
        signals.add(FrontrunSignal.SAME_BLOCK_COLLISION)
        mitigations.add(Mitigation.RANDOMISED_CADENCE)

    # Risk classification — score by signal count.
    n = len(signals)
    if n == 0:
        risk = FrontrunRisk.LOW
    elif n == 1:
        risk = FrontrunRisk.MEDIUM
    elif n in (2, 3):
        risk = FrontrunRisk.HIGH
    else:
        risk = FrontrunRisk.CRITICAL

    return FrontrunAssessment(
        parent_id=signal.parent_id,
        risk=risk,
        signals=frozenset(signals),
        recommended_mitigations=frozenset(mitigations),
    )


def render_assessment(a: FrontrunAssessment) -> str:
    emoji = {
        FrontrunRisk.LOW: "🟢",
        FrontrunRisk.MEDIUM: "🟡",
        FrontrunRisk.HIGH: "🟠",
        FrontrunRisk.CRITICAL: "🔴",
    }[a.risk]
    head = f"{emoji} {a.parent_id} risk={a.risk.value}"
    lines = [head]
    for s in sorted(a.signals, key=lambda x: x.value):
        lines.append(f"  • signal: {s.value}")
    for m in sorted(a.recommended_mitigations, key=lambda x: x.value):
        lines.append(f"  → mitigation: {m.value}")
    return "\n".join(lines)
