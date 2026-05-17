"""Insider trading pattern detector — Round-5 Wave 11.F.

Insider transactions become signals when they cluster in unusual
ways: many insiders buying simultaneously, sale immediately before
material news, etc. This module ships the **pattern detector** that
flags suspicious clusters from the Form-4 stream.

Pinned semantics:

- **Closed-set Direction ladder** (BUY / SELL).
- **Closed-set ClusterPattern ladder** (CLUSTER_BUY / CLUSTER_SELL /
  PRE_NEWS_SALE / NORMAL).
- **Cluster threshold** — operator-tunable; default ≥ 3 transactions
  by ≥ 2 distinct insiders within a 5-day window.
- **No-secret-leak pin** — never includes insider names directly;
  uses operator-supplied handles only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class Direction(str, Enum):
    """Closed-set direction."""

    BUY = "buy"
    SELL = "sell"


class ClusterPattern(str, Enum):
    """Closed-set cluster patterns."""

    NORMAL = "normal"
    CLUSTER_BUY = "cluster_buy"
    CLUSTER_SELL = "cluster_sell"
    PRE_NEWS_SALE = "pre_news_sale"


@dataclass(frozen=True)
class InsiderTrade:
    """A single insider trade event."""

    trade_id: str
    insider_handle: str
    symbol: str
    direction: Direction
    shares: float
    trade_date: date

    def __post_init__(self) -> None:
        if not self.trade_id or not self.trade_id.strip():
            raise ValueError("trade_id must be non-empty")
        if not self.insider_handle.strip():
            raise ValueError("insider_handle must be non-empty")
        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.shares <= 0:
            raise ValueError("shares must be positive")


@dataclass(frozen=True)
class DetectorPolicy:
    """Operator-tunable thresholds."""

    cluster_window_days: int = 5
    min_trades_for_cluster: int = 3
    min_distinct_insiders: int = 2
    pre_news_window_days: int = 7

    def __post_init__(self) -> None:
        if self.cluster_window_days <= 0:
            raise ValueError("cluster_window_days must be positive")
        if self.min_trades_for_cluster < 2:
            raise ValueError("min_trades_for_cluster must be >= 2")
        if self.min_distinct_insiders < 2:
            raise ValueError("min_distinct_insiders must be >= 2")
        if self.pre_news_window_days <= 0:
            raise ValueError("pre_news_window_days must be positive")


@dataclass(frozen=True)
class PatternDetection:
    """Result of running the pattern detector for a symbol."""

    symbol: str
    pattern: ClusterPattern
    n_trades: int
    n_distinct_insiders: int
    direction_skew: float  # +1 = all buys, -1 = all sells
    triggering_window_start: date | None
    triggering_window_end: date | None


def _trades_in_window(trades: list[InsiderTrade], start: date, end: date) -> list[InsiderTrade]:
    return [t for t in trades if start <= t.trade_date <= end]


def detect(
    symbol: str,
    trades: Iterable[InsiderTrade],
    *,
    policy: DetectorPolicy | None = None,
    upcoming_news_date: date | None = None,
) -> PatternDetection:
    """Detect cluster + pre-news patterns in insider trade flow."""
    if not symbol.strip():
        raise ValueError("symbol must be non-empty")
    pol = policy if policy is not None else DetectorPolicy()
    relevant = sorted([t for t in trades if t.symbol == symbol], key=lambda t: t.trade_date)

    if not relevant:
        return PatternDetection(
            symbol=symbol,
            pattern=ClusterPattern.NORMAL,
            n_trades=0,
            n_distinct_insiders=0,
            direction_skew=0.0,
            triggering_window_start=None,
            triggering_window_end=None,
        )

    # Find the densest window
    best_pattern = ClusterPattern.NORMAL
    best_count = 0
    best_insiders = 0
    best_skew = 0.0
    best_start: date | None = None
    best_end: date | None = None

    for anchor in relevant:
        window_start = anchor.trade_date
        window_end = anchor.trade_date + timedelta(days=pol.cluster_window_days)
        window_trades = _trades_in_window(relevant, window_start, window_end)
        n = len(window_trades)
        insiders = {t.insider_handle for t in window_trades}
        n_insiders = len(insiders)
        if n >= pol.min_trades_for_cluster and n_insiders >= pol.min_distinct_insiders:
            buys = sum(1 for t in window_trades if t.direction is Direction.BUY)
            sells = n - buys
            skew = (buys - sells) / n
            if n > best_count:
                best_count = n
                best_insiders = n_insiders
                best_skew = skew
                best_start = window_start
                best_end = window_end
                if skew > 0.5:
                    best_pattern = ClusterPattern.CLUSTER_BUY
                elif skew < -0.5:
                    best_pattern = ClusterPattern.CLUSTER_SELL

    # Pre-news sale check
    if upcoming_news_date is not None:
        pre_window_start = upcoming_news_date - timedelta(days=pol.pre_news_window_days)
        pre_trades = _trades_in_window(
            relevant, pre_window_start, upcoming_news_date - timedelta(days=1)
        )
        pre_sells = [t for t in pre_trades if t.direction is Direction.SELL]
        pre_insiders = {t.insider_handle for t in pre_sells}
        if (
            len(pre_sells) >= pol.min_trades_for_cluster
            and len(pre_insiders) >= pol.min_distinct_insiders
        ):
            best_pattern = ClusterPattern.PRE_NEWS_SALE
            best_count = len(pre_sells)
            best_insiders = len(pre_insiders)
            best_skew = -1.0
            best_start = pre_window_start
            best_end = upcoming_news_date - timedelta(days=1)

    if best_pattern is ClusterPattern.NORMAL:
        # Even when no cluster, summarise overall flow
        buys = sum(1 for t in relevant if t.direction is Direction.BUY)
        sells = len(relevant) - buys
        best_count = len(relevant)
        best_insiders = len({t.insider_handle for t in relevant})
        best_skew = (buys - sells) / len(relevant) if relevant else 0.0

    return PatternDetection(
        symbol=symbol,
        pattern=best_pattern,
        n_trades=best_count,
        n_distinct_insiders=best_insiders,
        direction_skew=best_skew,
        triggering_window_start=best_start,
        triggering_window_end=best_end,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "SSN",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_detection(d: PatternDetection) -> str:
    emoji = {
        ClusterPattern.NORMAL: "⚪",
        ClusterPattern.CLUSTER_BUY: "🟢",
        ClusterPattern.CLUSTER_SELL: "🔴",
        ClusterPattern.PRE_NEWS_SALE: "🚨",
    }[d.pattern]
    win = ""
    if d.triggering_window_start and d.triggering_window_end:
        win = f" ({d.triggering_window_start.isoformat()}→{d.triggering_window_end.isoformat()})"
    return _scrub(
        f"{emoji} {d.symbol}: {d.pattern.value} "
        f"n_trades={d.n_trades} n_insiders={d.n_distinct_insiders} "
        f"skew={d.direction_skew:+.2f}{win}"
    )
