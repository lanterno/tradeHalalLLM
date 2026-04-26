"""Shadow-bot divergence detector.

Walk-forward backtests catch overfitting at decision time, but they
don't catch *gradual model decay* — the slow drift where a strategy
that was profitable six weeks ago is no longer profitable today,
because the market structure shifted.

The cheap fix is a *frozen* shadow strategy running in pure paper
alongside the live one. Same inputs every cycle, deliberately frozen
prompts/models — and we track whether the *live* P&L curve is keeping
up with the shadow's. Two regimes to alert on:

* Live ≪ shadow → the live changes (new prompts, new ML weights) are
  hurting. Roll back.
* Both falling together → the *underlying market* changed; both stale.
  Force retrain + tighten risk.
* Live ≫ shadow → live changes are working. Promote.

This module is the bookkeeping layer:

* :class:`ShadowLedger` records (cycle_id, ts, live_equity, shadow_equity).
* :func:`divergence_metrics` computes paired-difference stats on the
  cumulative-return series.
* :func:`shadow_alert_state` returns the alert state given a config.

The actual *running* of the shadow plan is the operator's call (separate
process, separate replica) — this module is the comparison logic.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


AlertLevel = Literal["ok", "watch", "diverged"]


# ── Ledger entry ──────────────────────────────────────────────────


@dataclass
class LedgerEntry:
    cycle_id: str
    ts: str  # ISO timestamp
    live_equity: float
    shadow_equity: float

    @property
    def diff(self) -> float:
        """Live minus shadow — positive when live is winning."""
        return self.live_equity - self.shadow_equity


@dataclass
class ShadowLedger:
    """In-process append-only ledger of (live, shadow) equity samples."""

    entries: list[LedgerEntry] = field(default_factory=list)
    capacity: int = 5_000

    def record(
        self,
        *,
        cycle_id: str,
        live_equity: float,
        shadow_equity: float,
        ts: str | None = None,
    ) -> LedgerEntry:
        ts = ts or datetime.now(UTC).isoformat()
        e = LedgerEntry(
            cycle_id=cycle_id,
            ts=ts,
            live_equity=live_equity,
            shadow_equity=shadow_equity,
        )
        self.entries.append(e)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity :]
        return e

    @property
    def size(self) -> int:
        return len(self.entries)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "capacity": self.capacity,
                    "entries": [asdict(e) for e in self.entries],
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path | str) -> "ShadowLedger":
        raw = json.loads(Path(path).read_text())
        return cls(
            capacity=int(raw.get("capacity", 5_000)),
            entries=[LedgerEntry(**e) for e in raw.get("entries", [])],
        )


# ── Metrics ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DivergenceMetrics:
    """Snapshot of how much live and shadow have drifted apart."""

    n: int
    mean_diff_pct: float  # average (live - shadow) / shadow
    last_diff_pct: float
    max_drawdown_diff: float  # worst (live - shadow) / shadow ever
    paired_t_score: float  # ~ z-score of mean diff under H0: mean=0
    direction: Literal["live_better", "live_worse", "even"]


def divergence_metrics(entries: Sequence[LedgerEntry]) -> DivergenceMetrics | None:
    """Paired-difference stats. Returns None if too few samples."""
    if len(entries) < 5:
        return None
    diffs_pct: list[float] = []
    for e in entries:
        if e.shadow_equity <= 0:
            continue
        diffs_pct.append((e.live_equity - e.shadow_equity) / e.shadow_equity)
    n = len(diffs_pct)
    if n < 5:
        return None
    mean = sum(diffs_pct) / n
    var = sum((d - mean) ** 2 for d in diffs_pct) / max(1, n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    t = mean / se if se > 0 else 0.0
    direction: Literal["live_better", "live_worse", "even"]
    if abs(t) < 1.5:
        direction = "even"
    elif mean > 0:
        direction = "live_better"
    else:
        direction = "live_worse"
    return DivergenceMetrics(
        n=n,
        mean_diff_pct=mean,
        last_diff_pct=diffs_pct[-1],
        max_drawdown_diff=min(diffs_pct),
        paired_t_score=t,
        direction=direction,
    )


# ── Alert policy ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ShadowAlertConfig:
    """Thresholds for promoting drift into operator alerts.

    Defaults are starting points; tune on live data.
    """

    watch_drawdown_pct: float = 0.02  # 2% cumulative live-vs-shadow underperformance
    diverge_drawdown_pct: float = 0.05  # 5% — operator alert
    min_t_score: float = 1.5  # require some statistical signal
    min_samples: int = 30


def shadow_alert_state(
    metrics: DivergenceMetrics | None, config: ShadowAlertConfig | None = None
) -> AlertLevel:
    """Map metrics → alert level using ``ShadowAlertConfig``."""
    cfg = config or ShadowAlertConfig()
    if metrics is None or metrics.n < cfg.min_samples:
        return "ok"
    # Only alert when live is *underperforming* the shadow with statistical signal.
    if metrics.direction == "live_worse" and abs(metrics.paired_t_score) >= cfg.min_t_score:
        if abs(metrics.max_drawdown_diff) >= cfg.diverge_drawdown_pct:
            return "diverged"
        if abs(metrics.max_drawdown_diff) >= cfg.watch_drawdown_pct:
            return "watch"
    # Catastrophic single-day divergence even without sustained signal
    if abs(metrics.last_diff_pct) >= cfg.diverge_drawdown_pct:
        return "diverged"
    return "ok"


def render_status(metrics: DivergenceMetrics | None, level: AlertLevel) -> str:
    if metrics is None:
        return "shadow status: warming up (insufficient samples)"
    sign = "+" if metrics.mean_diff_pct >= 0 else ""
    return (
        f"shadow status: {level.upper()} — n={metrics.n}, "
        f"mean diff {sign}{metrics.mean_diff_pct:.2%}, "
        f"last {metrics.last_diff_pct:+.2%}, "
        f"worst {metrics.max_drawdown_diff:+.2%}, t={metrics.paired_t_score:+.2f}, "
        f"direction={metrics.direction}"
    )


# ── Compare-cycles helper ─────────────────────────────────────────


def diff_plans(live_plan: Any, shadow_plan: Any) -> dict[str, int]:
    """Compute simple structural diff between two cycle plans.

    Pure helper for the operator: how many decisions disagree by
    (action, symbol)? Useful as a per-cycle divergence proxy without
    waiting for the equity curve to bend.
    """

    def _decisions(plan: Any) -> list[Any]:
        return list(getattr(plan, "decisions", []) or [])

    def _key(d: Any) -> tuple[str, str]:
        a = getattr(d, "action", "")
        a = a.value if hasattr(a, "value") else str(a)
        return (a.lower(), getattr(d, "symbol", ""))

    live_keys = {_key(d) for d in _decisions(live_plan)}
    shadow_keys = {_key(d) for d in _decisions(shadow_plan)}
    only_live = live_keys - shadow_keys
    only_shadow = shadow_keys - live_keys
    shared = live_keys & shadow_keys
    return {
        "shared": len(shared),
        "only_live": len(only_live),
        "only_shadow": len(only_shadow),
    }


def aggregate_plan_diffs(diffs: Iterable[dict[str, int]]) -> dict[str, float]:
    """Roll many per-cycle diffs into one fraction-disagreed summary."""
    diffs = list(diffs)
    if not diffs:
        return {"n": 0, "frac_disagreed": 0.0}
    total = 0
    disagreed = 0
    for d in diffs:
        s = d.get("shared", 0) + d.get("only_live", 0) + d.get("only_shadow", 0)
        total += s
        disagreed += d.get("only_live", 0) + d.get("only_shadow", 0)
    return {
        "n": len(diffs),
        "frac_disagreed": disagreed / total if total else 0.0,
    }
