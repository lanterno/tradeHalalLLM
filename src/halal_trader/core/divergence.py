"""Paper-vs-live slippage divergence tracker.

The fundamental question every paper-trading bot has to answer before
it goes live with real capital: *are my paper fills representative of
what I'd actually get?* Slippage is the cleanest signal here — if our
backtester models 5bps and the live broker keeps giving us 25bps, the
"profitable" backtest is a fiction.

This module:

* Computes per-trade slippage vs the LLM's intended entry/exit prices.
* Aggregates the live-vs-paper distribution into a single divergence
  report (mean, p95, count).
* Surfaces a clear "exceeds threshold" flag the operator can act on
  (raise paper slippage, switch strategies, halt).

Pure functions over the trade rows; no DB writes happen here — the
caller is expected to persist via the new ``paper_slippage_pct`` /
``live_slippage_pct`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TradeSlippage:
    """Per-trade slippage record (signed, in fractions of intended price)."""

    trade_id: int
    paper_slippage_pct: float | None
    live_slippage_pct: float | None

    @property
    def divergence(self) -> float | None:
        if self.paper_slippage_pct is None or self.live_slippage_pct is None:
            return None
        return self.live_slippage_pct - self.paper_slippage_pct


@dataclass
class DivergenceReport:
    """Roll-up of paper-vs-live divergence across a trade sample."""

    sample_size: int
    mean_divergence_bps: float
    p95_divergence_bps: float  # right tail = live worse than paper
    mean_paper_bps: float
    mean_live_bps: float
    exceeds_threshold: bool
    threshold_bps: float


def compute_slippage(
    *,
    intended_price: float,
    actual_fill_price: float,
    side: str,
) -> float | None:
    """Return signed slippage as a fraction of intended price.

    For a buy, slippage is positive when the actual fill is *worse* than
    intended (fill > intended). For a sell it's the reverse — positive
    means we sold for less than intended. Same sign convention either
    way: positive = "cost us money."
    """
    if intended_price <= 0 or actual_fill_price <= 0:
        return None
    if side not in ("buy", "sell"):
        return None
    raw = (actual_fill_price - intended_price) / intended_price
    return raw if side == "buy" else -raw


def build_report(
    samples: Sequence[TradeSlippage],
    *,
    threshold_bps: float = 10.0,
) -> DivergenceReport:
    """Aggregate per-trade slippage into a single divergence report.

    ``threshold_bps`` is the operator-set "alert level" — when the mean
    divergence is worse than this we flag it. 10bps is a sane default;
    for small-cap or low-liquidity venues the operator may set higher.
    """
    pairs = [
        (s.paper_slippage_pct, s.live_slippage_pct)
        for s in samples
        if s.paper_slippage_pct is not None and s.live_slippage_pct is not None
    ]
    if not pairs:
        return DivergenceReport(
            sample_size=0,
            mean_divergence_bps=0.0,
            p95_divergence_bps=0.0,
            mean_paper_bps=0.0,
            mean_live_bps=0.0,
            exceeds_threshold=False,
            threshold_bps=threshold_bps,
        )

    paper = np.array([p for p, _ in pairs])
    live = np.array([liv for _, liv in pairs])
    div = live - paper

    mean_div_bps = float(np.mean(div) * 10_000)
    p95_div_bps = float(np.percentile(div, 95) * 10_000)
    mean_paper_bps = float(np.mean(paper) * 10_000)
    mean_live_bps = float(np.mean(live) * 10_000)

    return DivergenceReport(
        sample_size=len(pairs),
        mean_divergence_bps=mean_div_bps,
        p95_divergence_bps=p95_div_bps,
        mean_paper_bps=mean_paper_bps,
        mean_live_bps=mean_live_bps,
        exceeds_threshold=mean_div_bps > threshold_bps,
        threshold_bps=threshold_bps,
    )


def format_report(report: DivergenceReport) -> str:
    """Human-readable one-paragraph summary for logs and the dashboard."""
    if report.sample_size == 0:
        return "No paper/live slippage samples yet — divergence unknown."
    flag = " ⚠ EXCEEDS THRESHOLD" if report.exceeds_threshold else ""
    return (
        f"Paper-vs-live slippage divergence over {report.sample_size} trades: "
        f"mean {report.mean_divergence_bps:+.2f}bps, "
        f"p95 {report.p95_divergence_bps:+.2f}bps "
        f"(paper {report.mean_paper_bps:+.2f}bps vs live {report.mean_live_bps:+.2f}bps; "
        f"threshold {report.threshold_bps:.1f}bps){flag}"
    )
