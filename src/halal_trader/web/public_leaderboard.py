"""Public anonymized performance leaderboard — Round-5 Wave 17.A.

Round-4 Wave 3.H shipped `web/leaderboard.py` — the platform-internal
ranking + anonymisation engine. Round-5 Wave 17.A adds the **public
community-facing variant**: time-windowed (weekly / monthly / yearly),
category-based (risk bucket + asset class), with k-anonymity guards
to preserve privacy when an individual operator could otherwise be
identified by performance pattern.

Pinned semantics:

- **Closed-set TimeWindow ladder** (WEEKLY / MONTHLY / QUARTERLY /
  YEARLY / ALL_TIME).
- **Closed-set RiskBucket ladder** (CONSERVATIVE / MODERATE /
  AGGRESSIVE).
- **k-anonymity guard**: a (window, bucket) cell with fewer than
  ``min_cell_size`` participants returns no entries (the cell is
  marked ``CELL_TOO_SMALL`` to avoid surfacing a single operator's
  performance).
- **Display name** is the anonymous handle from `web/leaderboard.py`;
  this module never accepts or renders a real-name field.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum


class TimeWindow(str, Enum):
    """Closed-set leaderboard time windows."""

    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


class RiskBucket(str, Enum):
    """Closed-set risk buckets."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class LeaderboardPolicy:
    """Operator-tunable policy."""

    min_cell_size: int = 5  # k-anonymity threshold
    top_n_per_cell: int = 10
    confidence_required: bool = True  # rank only confirmed-publishing operators

    def __post_init__(self) -> None:
        if self.min_cell_size <= 0:
            raise ValueError("min_cell_size must be positive")
        if self.top_n_per_cell <= 0:
            raise ValueError("top_n_per_cell must be positive")


@dataclass(frozen=True)
class OperatorEntry:
    """One operator's performance entry in the eligible-set."""

    handle: str  # public anonymous handle
    window: TimeWindow
    bucket: RiskBucket
    return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    n_trades: int
    consents_to_publish: bool
    last_updated: date

    def __post_init__(self) -> None:
        if not self.handle or not self.handle.strip():
            raise ValueError("handle must be non-empty")
        if "@" in self.handle:
            raise ValueError("handle must be anonymous, not an email")
        if self.n_trades < 0:
            raise ValueError("n_trades must be non-negative")
        if self.max_drawdown_pct < 0 or self.max_drawdown_pct > 1.0:
            raise ValueError("max_drawdown_pct must be in [0, 1]")


@dataclass(frozen=True)
class LeaderboardCell:
    """A single (window, bucket) leaderboard cell."""

    window: TimeWindow
    bucket: RiskBucket
    entries: tuple[OperatorEntry, ...]
    cell_too_small: bool

    def __post_init__(self) -> None:
        # If cell_too_small, no entries should be exposed
        if self.cell_too_small and self.entries:
            raise ValueError("cell_too_small=True but entries non-empty")


def build_cell(
    eligible: Iterable[OperatorEntry],
    *,
    window: TimeWindow,
    bucket: RiskBucket,
    policy: LeaderboardPolicy | None = None,
) -> LeaderboardCell:
    """Build a single leaderboard cell, applying k-anonymity guard."""
    pol = policy if policy is not None else LeaderboardPolicy()

    # Filter to the cell + consent
    candidates = [
        e
        for e in eligible
        if e.window is window
        and e.bucket is bucket
        and (e.consents_to_publish or not pol.confidence_required)
    ]

    if len(candidates) < pol.min_cell_size:
        return LeaderboardCell(
            window=window,
            bucket=bucket,
            entries=(),
            cell_too_small=True,
        )

    # Rank by return_pct descending; tie-break by sharpe.
    candidates.sort(key=lambda e: (-e.return_pct, -e.sharpe_ratio))
    top = tuple(candidates[: pol.top_n_per_cell])

    return LeaderboardCell(
        window=window,
        bucket=bucket,
        entries=top,
        cell_too_small=False,
    )


def build_grid(
    eligible: Iterable[OperatorEntry],
    *,
    policy: LeaderboardPolicy | None = None,
) -> tuple[LeaderboardCell, ...]:
    """Build the full grid of (window, bucket) cells."""
    eligible_t = tuple(eligible)
    cells: list[LeaderboardCell] = []
    for window in TimeWindow:
        for bucket in RiskBucket:
            cells.append(build_cell(eligible_t, window=window, bucket=bucket, policy=policy))
    return tuple(cells)


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "real_name",
    "address",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_cell(cell: LeaderboardCell) -> str:
    head = f"Leaderboard {cell.window.value}/{cell.bucket.value}"
    if cell.cell_too_small:
        return _scrub(f"{head}: ⚠ insufficient participants for ranking")
    lines = [f"{head}: {len(cell.entries)} entries"]
    for i, e in enumerate(cell.entries, start=1):
        lines.append(
            f"  {i:>2d}. {e.handle:24s} ret={e.return_pct * 100:+.2f}% "
            f"sharpe={e.sharpe_ratio:+.2f} dd={e.max_drawdown_pct * 100:.2f}% "
            f"n={e.n_trades}"
        )
    return _scrub("\n".join(lines))


def render_grid(cells: Iterable[LeaderboardCell]) -> str:
    parts = [render_cell(c) for c in cells]
    return "\n\n".join(parts)
