"""Mention-velocity feature for social/news streams.

Raw mention *counts* (e.g. "$BTC mentioned 312 times today on Reddit")
are uninformative — popular tickers always trend high, unpopular ones
always trend low. The signal is in the *rate of change* and the
*novelty* of attention.

This module computes two cheap, robust features per symbol:

* **Mention velocity** — recent-window mentions / older-window mentions.
  >1.5 means attention is accelerating; <0.7 means decaying.
* **Novelty score** — fraction of total all-time mentions that occurred
  in the recent window. ≈0 for permanently-popular tickers; ≈1 for
  brand-new attention.

Both are computed from raw timestamped mentions, which keeps the
abstraction free of any specific Reddit/Twitter SDK. Plug PRAW or any
async client by handing it a ``Sequence[Mention]`` here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# ── Mention ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mention:
    """One observed mention of a symbol on a social source."""

    symbol: str
    timestamp: datetime
    source: str = ""
    score: float = 0.0  # upvotes / engagement signal — optional


# ── Velocity result ──────────────────────────────────────────────


@dataclass
class VelocityResult:
    """Computed velocity / novelty features for one symbol."""

    symbol: str
    n_recent: int = 0
    n_older: int = 0
    n_total: int = 0
    velocity: float = 1.0  # recent / older (1.0 = neutral)
    novelty: float = 0.0  # recent / total
    label: str = "neutral"  # "surge" | "decay" | "neutral"
    notes: list[str] = field(default_factory=list)


def _classify(velocity: float, novelty: float) -> str:
    if velocity >= 2.0 or (velocity >= 1.5 and novelty >= 0.5):
        return "surge"
    if velocity <= 0.5:
        return "decay"
    return "neutral"


# ── Compute ──────────────────────────────────────────────────────


def compute_velocity(
    mentions: Iterable[Mention],
    *,
    now: datetime | None = None,
    recent_window_hours: float = 6.0,
    older_window_hours: float = 24.0,
) -> dict[str, VelocityResult]:
    """Per-symbol mention velocity + novelty over the given windows.

    ``recent_window_hours`` is the leading window we expect to be
    elevated when something is breaking. ``older_window_hours`` is the
    baseline window — typically several × recent.

    Symbols with zero mentions in the older window get ``velocity =
    n_recent`` (so a brand-new spike has the most extreme velocity)
    capped at a sane upper bound to avoid divide-by-near-zero noise.
    """
    now = now or datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=recent_window_hours)
    older_cutoff = now - timedelta(hours=older_window_hours)

    by_symbol: dict[str, list[Mention]] = {}
    for m in mentions:
        ts = m.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < older_cutoff:
            continue
        sym = m.symbol.upper()
        by_symbol.setdefault(sym, []).append(
            Mention(symbol=sym, timestamp=ts, source=m.source, score=m.score)
        )

    out: dict[str, VelocityResult] = {}
    for sym, items in by_symbol.items():
        n_recent = sum(1 for m in items if m.timestamp >= recent_cutoff)
        n_older = sum(1 for m in items if m.timestamp < recent_cutoff)
        n_total = len(items)
        if n_older > 0:
            velocity = n_recent / n_older
        else:
            # No older mentions in the window — pure novelty.
            velocity = float(min(10.0, n_recent))
        novelty = n_recent / n_total if n_total else 0.0
        result = VelocityResult(
            symbol=sym,
            n_recent=n_recent,
            n_older=n_older,
            n_total=n_total,
            velocity=velocity,
            novelty=novelty,
            label=_classify(velocity, novelty),
        )
        out[sym] = result
    return out


# ── Prompt formatter ─────────────────────────────────────────────


def format_velocity_for_prompt(
    results: dict[str, VelocityResult],
    *,
    min_recent: int = 3,
    limit: int = 5,
) -> str:
    """Compact one-block string of the most actionable surges.

    Filters tickers with fewer than ``min_recent`` mentions to suppress
    noise from one-off shoutouts; sorts by velocity desc; trims to
    ``limit`` rows so the prompt stays cheap.
    """
    surges = [r for r in results.values() if r.n_recent >= min_recent and r.label == "surge"]
    if not surges:
        return ""
    surges.sort(key=lambda r: r.velocity, reverse=True)
    lines = ["Mention surges (recent / older window):"]
    for r in surges[:limit]:
        lines.append(
            f"  {r.symbol:<10} velocity={r.velocity:.1f}× "
            f"novelty={r.novelty:.0%} ({r.n_recent} recent / {r.n_older} older)"
        )
    return "\n".join(lines)


# ── Filter helper ────────────────────────────────────────────────


def filter_halal_mentions(
    mentions: Sequence[Mention], halal_symbols: Iterable[str]
) -> list[Mention]:
    """Drop mentions for non-halal symbols.

    Cheap helper for callers that scrape a broad universe and need to
    narrow before running ``compute_velocity``.
    """
    allow = {s.upper() for s in halal_symbols}
    return [m for m in mentions if m.symbol.upper() in allow]
