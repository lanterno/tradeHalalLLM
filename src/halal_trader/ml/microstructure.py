"""Order-book microstructure feature extractor.

Round-4 wave 4.E: the bot doesn't yet ingest L2 tick data, but
when it does (Coinbase Advanced Trade adapter / Binance L2 stream
/ exchange-specific WebSocket), the strategy needs a small set of
proven microstructure features to filter or downsize LLM picks.
This module is the feature layer — given a single L2 snapshot, it
produces a `MicrostructureFeatures` block ready to feed the model
or the prompt.

Features computed (all bounded so a glitched feed can't explode
the downstream signal):

* **Imbalance** — `(bid_volume - ask_volume) / (bid_volume +
  ask_volume)` over the top-N levels. Range [-1, 1]; positive
  means more buying pressure on the book, negative more selling.
* **Micro-price** — `(best_bid × ask_size + best_ask × bid_size)
  / (bid_size + ask_size)`. The volume-weighted "true mid" that
  the literature shows leads the simple mid by a few seconds.
  Returned as both an absolute price and a deviation from
  the simple mid (in basis points).
* **Spread** — best_ask - best_bid, both absolute and in
  basis points of the mid.
* **Depth-decay** — how quickly volume falls off as you move
  away from the inside. A flat order book (operator can size
  large) vs a thin book (small operator size before slippage).
  Computed as the slope of `log(volume)` against level index;
  steeper negative = thinner book.
* **Top-of-book skew** — best-bid-size / best-ask-size,
  log-transformed. Positive log-skew means the bid side has
  more size at the touch.

Why a feature module rather than a model directly:

* The order-book ML wave (when tick data lands) trains the
  model. The features it consumes are stable across model
  iterations — pin them once here.
* Operators can also feed the features into the LLM prompt as
  raw values ("bid imbalance: +0.45; micro-price 0.7bp above
  mid") for explainability without an in-house model.
* Pure-numpy keeps the extractor testable without a real
  exchange connection.

Halal alignment: read-only signal computation. Never opens a
position. The L2 snapshot is operator-fetched (the bot already
respects exchange terms in `crypto/exchange.py`) and
`MicrostructureFeatures` is just numbers.

Pure-numpy; no scipy / DB / async.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ── Inputs ────────────────────────────────────────────────


@dataclass(frozen=True)
class OrderBookLevel:
    """One price level in the L2 book.

    ``price`` is the level's price; ``size`` is the resting volume
    at that price (in base currency units). The extractor doesn't
    care about per-level identity — it operates on the aggregated
    snapshot."""

    price: float
    size: float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"price must be positive; got {self.price}")
        if self.size < 0:
            raise ValueError(f"size must be >= 0; got {self.size}")


@dataclass(frozen=True)
class OrderBookSnapshot:
    """One L2 snapshot.

    ``bids`` are the buy-side levels sorted **descending** by
    price (best bid first). ``asks`` are sell-side levels sorted
    **ascending** by price (best ask first).

    The extractor doesn't sort — pin the contract so the caller's
    L2 adapter (the future Binance / Coinbase stream wiring)
    matches exchange convention without an extra sort cost on
    every snapshot.
    """

    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]

    def __post_init__(self) -> None:
        if not self.bids:
            raise ValueError("snapshot must have at least one bid level")
        if not self.asks:
            raise ValueError("snapshot must have at least one ask level")
        # Pin: the caller's adapter must order bids desc + asks asc.
        # We surface a clear error if it didn't, rather than silently
        # mis-computing the spread.
        for i in range(len(self.bids) - 1):
            if self.bids[i].price < self.bids[i + 1].price:
                raise ValueError(
                    f"bids must be sorted descending; got "
                    f"{self.bids[i].price} → {self.bids[i + 1].price}"
                )
        for i in range(len(self.asks) - 1):
            if self.asks[i].price > self.asks[i + 1].price:
                raise ValueError(
                    f"asks must be sorted ascending; got "
                    f"{self.asks[i].price} → {self.asks[i + 1].price}"
                )
        if self.bids[0].price >= self.asks[0].price:
            raise ValueError(
                f"crossed book: best bid {self.bids[0].price} >= best ask {self.asks[0].price}"
            )


# ── Output ────────────────────────────────────────────────


@dataclass(frozen=True)
class MicrostructureFeatures:
    """The feature vector consumers feed to a model or prompt.

    Every field is bounded so a glitched feed can't explode the
    downstream signal:

    * ``imbalance`` ∈ [-1, 1]
    * ``micro_price_bp_dev`` clamped to ±500bp (±5%) — anything
      beyond is data corruption.
    * ``spread_bp`` clamped at 0 lower bound; no upper clamp
      because a wide-spread regime is a real feature.
    * ``depth_decay_slope`` clamped to [-5, 5] — past these
      bounds the slope estimation is unreliable on a thin book.
    * ``top_of_book_log_skew`` clamped to [-5, 5] for the same
      reason.
    """

    best_bid: float
    best_ask: float
    mid_price: float
    micro_price: float
    micro_price_bp_dev: float  # micro_price vs mid, in basis points
    spread_abs: float
    spread_bp: float
    imbalance: float
    depth_decay_slope: float
    top_of_book_log_skew: float
    levels_used: int


# ── Feature computations ──────────────────────────────────


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _imbalance(snapshot: OrderBookSnapshot, levels: int) -> float:
    """Top-N volume imbalance. Pin: aggregated over `levels` from
    the inside outward; returns 0.0 when both sides have zero
    volume (degenerate empty book)."""
    bid_vol = sum(level.size for level in snapshot.bids[:levels])
    ask_vol = sum(level.size for level in snapshot.asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def _micro_price(snapshot: OrderBookSnapshot) -> float:
    """Volume-weighted micro-price.

    ``(best_bid × ask_size + best_ask × bid_size) / (bid_size +
    ask_size)``. The literature (Cartea / Jaimungal) shows this
    leads the simple mid by a few seconds; the weighting puts
    more weight on the side with *less* size (the side about to
    move toward).
    """
    best_bid = snapshot.bids[0]
    best_ask = snapshot.asks[0]
    total = best_bid.size + best_ask.size
    if total == 0:
        return (best_bid.price + best_ask.price) / 2.0
    return (best_bid.price * best_ask.size + best_ask.price * best_bid.size) / total


def _depth_decay(snapshot: OrderBookSnapshot, levels: int) -> float:
    """Slope of log(volume) vs level index, averaged across both
    sides.

    A flat book has slope ≈ 0; a thinning book has negative
    slope. The slope is computed by simple linear regression
    against `[0, 1, 2, ..., n-1]`; we use np.polyfit because the
    extractor is pure-numpy already.

    Pin: zero-volume levels are skipped (log(0) is -inf); the
    regression runs on whatever levels remain. Below 3 valid
    levels per side, the slope is reported as 0 (insufficient
    data for a stable fit).
    """
    bid_slope = _slope_one_side(snapshot.bids, levels)
    ask_slope = _slope_one_side(snapshot.asks, levels)
    return (bid_slope + ask_slope) / 2.0


def _slope_one_side(levels_seq: Sequence[OrderBookLevel], levels: int) -> float:
    sample = levels_seq[:levels]
    log_volumes: list[float] = []
    indices: list[int] = []
    for i, level in enumerate(sample):
        if level.size > 0:
            log_volumes.append(math.log(level.size))
            indices.append(i)
    if len(log_volumes) < 3:
        return 0.0
    # polyfit returns [slope, intercept]; we want the slope.
    slope, _intercept = np.polyfit(indices, log_volumes, 1)
    return float(slope)


def _top_of_book_skew(snapshot: OrderBookSnapshot) -> float:
    """log(bid_size / ask_size) at the touch.

    Positive log-skew → bid side has more size than ask side.
    Returns 0 when either size is zero (no useful skew when one
    side is empty).
    """
    bid_size = snapshot.bids[0].size
    ask_size = snapshot.asks[0].size
    if bid_size == 0 or ask_size == 0:
        return 0.0
    return math.log(bid_size / ask_size)


# ── Extractor entry point ─────────────────────────────────


def extract(
    snapshot: OrderBookSnapshot,
    *,
    levels: int = 10,
) -> MicrostructureFeatures:
    """Compute all features from one L2 snapshot.

    ``levels`` controls how deep the imbalance / depth-decay
    metrics scan. Smaller = more sensitive to the inside (good
    for short-term price-movement prediction); larger = more
    stable but slower-responding.

    Pin: the caller's L2 adapter is responsible for delivering
    bids descending and asks ascending; the snapshot's
    `__post_init__` raises on a violation rather than silently
    mis-computing.
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1; got {levels}")

    best_bid = snapshot.bids[0].price
    best_ask = snapshot.asks[0].price
    mid_price = (best_bid + best_ask) / 2.0
    micro = _micro_price(snapshot)
    spread_abs = best_ask - best_bid
    spread_bp = (spread_abs / mid_price) * 1e4 if mid_price > 0 else 0.0
    micro_bp = ((micro - mid_price) / mid_price) * 1e4 if mid_price > 0 else 0.0
    imbalance = _imbalance(snapshot, levels)
    depth_slope = _depth_decay(snapshot, levels)
    skew = _top_of_book_skew(snapshot)
    levels_used = min(levels, max(len(snapshot.bids), len(snapshot.asks)))

    return MicrostructureFeatures(
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        micro_price=micro,
        micro_price_bp_dev=_clip(micro_bp, -500.0, 500.0),
        spread_abs=spread_abs,
        spread_bp=max(0.0, spread_bp),
        imbalance=_clip(imbalance, -1.0, 1.0),
        depth_decay_slope=_clip(depth_slope, -5.0, 5.0),
        top_of_book_log_skew=_clip(skew, -5.0, 5.0),
        levels_used=levels_used,
    )


# ── Render helper ─────────────────────────────────────────


def render_features(features: MicrostructureFeatures) -> str:
    """One-line operator-readable summary suitable for a Telegram /
    Slack alert or a per-cycle log entry."""
    sign = "+" if features.imbalance >= 0 else ""
    return (
        f"book imbalance {sign}{features.imbalance:.2f} · "
        f"spread {features.spread_bp:.1f}bp · "
        f"micro {'+' if features.micro_price_bp_dev >= 0 else ''}"
        f"{features.micro_price_bp_dev:.1f}bp · "
        f"depth slope {features.depth_decay_slope:+.2f}"
    )


__all__ = [
    "MicrostructureFeatures",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "extract",
    "render_features",
]
