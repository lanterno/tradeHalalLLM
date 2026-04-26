"""Embedding-based regime memory.

Once you have a few months of trades you have something most retail bots
don't: a labelled history of how *your* strategy behaved in each regime.
The cheap way to exploit it is per-cycle retrieval — compute a small
feature vector summarising "what kind of day is this?", look up the K
most similar past days, and surface their P&L in the prompt.

This module keeps the abstraction tiny:

* :class:`RegimeFeatures` — the day's snapshot vector (volatility, trend,
  breadth, sentiment, drawdown, etc.). Hand-engineered, not learned —
  the goal is interpretability and stability, not raw accuracy.
* :class:`RegimeMemory` — a fixed-capacity in-process store; ``add()``
  appends a snapshot + outcome, ``query()`` returns top-K cosine matches.
* :func:`format_for_prompt` — render a query result into the plain
  text block the LLM reads.

No external embedding model. Cosine over the standardised feature vector
is enough to find genuinely similar days, and the feature vector is
already what the bot computes per cycle anyway.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

# ── Feature vector ────────────────────────────────────────────────


@dataclass
class RegimeFeatures:
    """Daily market-state snapshot used as the embedding input.

    All fields default to 0; missing inputs degrade gracefully (the
    cosine similarity just treats absent dimensions as neutral). Keep
    the shape stable — adding a field invalidates older snapshots.
    """

    volatility: float = 0.0  # ATR / price (decimal)
    trend: float = 0.0  # -1..+1 multi-tf alignment
    breadth: float = 0.0  # -1..+1 share of universe up
    sentiment: float = 0.0  # -1..+1 composite news/social
    drawdown: float = 0.0  # 0..1 from peak
    volume_ratio: float = 1.0  # current / 20d avg
    correlation: float = 0.0  # 0..1 portfolio internal corr
    realized_return_1d: float = 0.0  # past day's market return
    rsi: float = 50.0  # 0..100
    spread_bps: float = 0.0  # microstructure cost proxy

    def to_vector(self) -> list[float]:
        return [
            self.volatility * 50.0,  # scale ATR/price (~0.02) to ~1
            self.trend,
            self.breadth,
            self.sentiment,
            self.drawdown * 5.0,  # 0..1 -> 0..5
            self.volume_ratio - 1.0,  # center
            self.correlation,
            self.realized_return_1d * 50.0,  # ~5% ⇒ ~2.5
            (self.rsi - 50.0) / 50.0,  # -1..+1
            self.spread_bps / 10.0,
        ]


# ── Snapshot record ───────────────────────────────────────────────


@dataclass
class RegimeSnapshot:
    """One day's regime + downstream outcome."""

    date: str  # ISO date "YYYY-MM-DD"
    features: RegimeFeatures
    outcome_pnl_pct: float = 0.0
    outcome_win_rate: float = 0.0
    outcome_n_trades: int = 0
    note: str = ""

    def label(self) -> str:
        sign = "+" if self.outcome_pnl_pct >= 0 else ""
        return (
            f"{self.date}: P&L {sign}{self.outcome_pnl_pct:.2%}, "
            f"win {self.outcome_win_rate:.0%} on {self.outcome_n_trades} trades"
        )


# ── Memory store ──────────────────────────────────────────────────


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


@dataclass
class RegimeMemory:
    """Fixed-capacity store of daily regime snapshots.

    Writes are O(1); queries are O(N) — N is small (months of days),
    so this is fine. Persistence is JSON for ops simplicity; if the
    store grows past ~10k rows you'd swap to sqlite or an actual vector
    DB without changing the interface.
    """

    capacity: int = 730  # ~2 trading years
    snapshots: list[RegimeSnapshot] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.snapshots)

    def add(self, snapshot: RegimeSnapshot) -> None:
        # de-dupe by date; replace prior entry if same date
        for i, s in enumerate(self.snapshots):
            if s.date == snapshot.date:
                self.snapshots[i] = snapshot
                return
        self.snapshots.append(snapshot)
        # FIFO trim
        if len(self.snapshots) > self.capacity:
            self.snapshots = self.snapshots[-self.capacity :]

    def add_today(
        self,
        features: RegimeFeatures,
        *,
        today: str | date | None = None,
        outcome_pnl_pct: float = 0.0,
        outcome_win_rate: float = 0.0,
        outcome_n_trades: int = 0,
        note: str = "",
    ) -> RegimeSnapshot:
        if today is None:
            today = date.today()
        date_str = today.isoformat() if isinstance(today, date) else str(today)
        snap = RegimeSnapshot(
            date=date_str,
            features=features,
            outcome_pnl_pct=outcome_pnl_pct,
            outcome_win_rate=outcome_win_rate,
            outcome_n_trades=outcome_n_trades,
            note=note,
        )
        self.add(snap)
        return snap

    def query(
        self, features: RegimeFeatures, *, k: int = 5, min_similarity: float = 0.0
    ) -> list[tuple[RegimeSnapshot, float]]:
        """Top-K similar snapshots by cosine of feature vectors."""
        if not self.snapshots:
            return []
        q = features.to_vector()
        scored = [(s, _cosine(q, s.features.to_vector())) for s in self.snapshots]
        scored.sort(key=lambda p: p[1], reverse=True)
        out = [p for p in scored if p[1] >= min_similarity]
        return out[:k]

    def aggregate_outcome(
        self, hits: Iterable[tuple[RegimeSnapshot, float]]
    ) -> dict[str, float]:
        """Similarity-weighted outcome aggregate from a query result."""
        hits = list(hits)
        if not hits:
            return {"weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0, "n": 0}
        total_w = 0.0
        weighted_pnl = 0.0
        weighted_win = 0.0
        for s, sim in hits:
            w = max(0.0, sim)
            total_w += w
            weighted_pnl += w * s.outcome_pnl_pct
            weighted_win += w * s.outcome_win_rate
        if total_w == 0:
            return {"weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0, "n": len(hits)}
        return {
            "weighted_pnl_pct": weighted_pnl / total_w,
            "weighted_win_rate": weighted_win / total_w,
            "n": len(hits),
        }

    # ── persistence ──────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        data = {
            "capacity": self.capacity,
            "snapshots": [
                {**asdict(s), "features": asdict(s.features)} for s in self.snapshots
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "RegimeMemory":
        raw = json.loads(Path(path).read_text())
        snaps: list[RegimeSnapshot] = []
        for s in raw.get("snapshots", []):
            f = s.pop("features", {})
            snaps.append(RegimeSnapshot(features=RegimeFeatures(**f), **s))
        return cls(capacity=int(raw.get("capacity", 730)), snapshots=snaps)


# ── Prompt rendering ──────────────────────────────────────────────


def format_for_prompt(
    today: RegimeFeatures,
    hits: Sequence[tuple[RegimeSnapshot, float]],
    *,
    max_lines: int = 5,
) -> str:
    """Render the top-K analogous days as a compact prompt block.

    Empty result returns a stable placeholder string the prompt
    builder can recognise and elide.
    """
    if not hits:
        return "No analogous past regime data."
    lines = [
        "Top historical regimes most similar to today",
        f"  (today: vol={today.volatility:.4f}, trend={today.trend:+.2f}, "
        f"sent={today.sentiment:+.2f}, dd={today.drawdown:.2%})",
    ]
    for snap, sim in hits[:max_lines]:
        lines.append(f"  · sim={sim:+.2f} — {snap.label()}")
        if snap.note:
            lines.append(f"      note: {snap.note[:100]}")
    return "\n".join(lines)
