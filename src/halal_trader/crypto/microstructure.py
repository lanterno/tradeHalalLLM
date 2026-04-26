"""Microstructure features — orderbook + funding signal extraction.

These features describe *short-horizon order flow* — what other market
participants are doing right now — that the spot OHLCV view doesn't
capture. The cycle today only sees price/volume; this module gives the
LLM (and the regime gate) signals like:

* **Depth imbalance** — heavier bids than asks → upward pressure now.
* **Cumulative delta** — net buy vs sell volume in recent trades.
* **Spread** — wider spreads = thin liquidity = sized entries will eat
  into book.
* **Funding rate** (perp) and **basis** (perp_mid − spot_mid) — paid
  positioning skew on the futures venue. Halal note: we *observe* perp
  data for signal, but **execute on spot only**, no leverage, no shorts.

All functions are pure: pass in the snapshot dicts the exchange client
already returns, get back features. No I/O. Wiring into the cycle
prompt happens in a follow-up; this PR ships the math + the test seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OrderbookFeatures:
    """Derived signals from a depth-of-book snapshot."""

    best_bid: float
    best_ask: float
    mid: float
    spread_bps: float  # basis points = 1/100th of a percent
    depth_imbalance: float  # in [-1, 1]; +1 = all bids, -1 = all asks
    bid_notional: float  # sum(price × size) on the bid side
    ask_notional: float


@dataclass(frozen=True)
class FundingFeatures:
    """Perp funding-rate / basis features (signal-only — we trade spot)."""

    funding_rate: float  # period rate, e.g. 0.0001 = 1bp every funding interval
    annualised_funding: float  # period rate × periods_per_year
    perp_mark: float
    spot_mid: float
    basis_bps: float  # (perp - spot) / spot in bps; positive = perp premium


def orderbook_features(orderbook: dict, *, top_n: int = 10) -> OrderbookFeatures | None:
    """Compute depth/imbalance features from an orderbook snapshot.

    ``orderbook`` is the shape ``BinanceClient.get_order_book`` already
    returns: ``{"bids": [[price, size], ...], "asks": [[price, size], ...]}``.
    Top-N controls how deep into the book we aggregate the imbalance —
    too shallow and a single resting order dominates; too deep and we
    pick up sleepy liquidity that won't actually move on a touch.

    Returns ``None`` if the book is empty/malformed — callers should
    treat absent features as "no signal" rather than crash the cycle.
    """
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        return None

    bid_levels = bids[:top_n]
    ask_levels = asks[:top_n]
    try:
        best_bid = float(bid_levels[0][0])
        best_ask = float(ask_levels[0][0])
    except Exception:
        return None
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return None

    bid_notional = sum(float(p) * float(s) for p, s in bid_levels)
    ask_notional = sum(float(p) * float(s) for p, s in ask_levels)
    total = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / total if total > 0 else 0.0

    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 10_000

    return OrderbookFeatures(
        best_bid=best_bid,
        best_ask=best_ask,
        mid=mid,
        spread_bps=spread_bps,
        depth_imbalance=imbalance,
        bid_notional=bid_notional,
        ask_notional=ask_notional,
    )


def cumulative_delta(trades: Sequence[dict]) -> float:
    """Net buyer-vs-seller signed quote-volume from a recent-trades stream.

    ``trades`` is the shape Binance's ``aggTrades`` / ``trades`` endpoints
    return — each entry has ``"price"``, ``"qty"`` and ``"isBuyerMaker"``.
    A buyer-maker trade is a sell-aggression hitting a resting bid, so we
    flip the sign for that case. Positive delta → net buying pressure.
    """
    delta = 0.0
    for trade in trades:
        try:
            price = float(trade.get("price") or trade.get("p") or 0)
            qty = float(trade.get("qty") or trade.get("q") or 0)
        except Exception:
            continue
        if price <= 0 or qty <= 0:
            continue
        notional = price * qty
        is_buyer_maker = bool(trade.get("isBuyerMaker", trade.get("m", False)))
        delta += -notional if is_buyer_maker else notional
    return delta


def funding_features(
    funding_rate: float,
    perp_mark: float,
    spot_mid: float,
    *,
    funding_periods_per_year: int = 365 * 3,
) -> FundingFeatures | None:
    """Combine a single funding-rate read + perp/spot midprices into features.

    Default ``funding_periods_per_year`` matches Binance USDT-margined
    perps: funding settles every 8 hours = 3 times a day. Override for
    venues with different cadences (some perps fund hourly).

    Returns ``None`` when perp/spot prices are unusable — callers should
    treat that as "perp signal unavailable for this pair."
    """
    if perp_mark <= 0 or spot_mid <= 0:
        return None
    annualised = funding_rate * funding_periods_per_year
    basis_bps = (perp_mark - spot_mid) / spot_mid * 10_000
    return FundingFeatures(
        funding_rate=funding_rate,
        annualised_funding=annualised,
        perp_mark=perp_mark,
        spot_mid=spot_mid,
        basis_bps=basis_bps,
    )


def format_microstructure_for_prompt(
    *,
    pair: str,
    book: OrderbookFeatures | None = None,
    funding: FundingFeatures | None = None,
    cum_delta: float | None = None,
) -> str:
    """One-line per-pair summary suitable for splicing into the LLM prompt.

    Empty when nothing useful was computed — the prompt template should
    omit the section rather than show "Microstructure: —".
    """
    parts: list[str] = []
    if book is not None:
        side = (
            "bid-heavy"
            if book.depth_imbalance > 0.1
            else ("ask-heavy" if book.depth_imbalance < -0.1 else "balanced")
        )
        parts.append(f"book {side} ({book.depth_imbalance:+.2f}), spread {book.spread_bps:.1f}bps")
    if cum_delta is not None and abs(cum_delta) > 0:
        parts.append(f"cum_delta ${cum_delta:+,.0f}")
    if funding is not None:
        # Annualised funding is the most operator-readable form: positive
        # means longs are paying — i.e. excess long demand on the perp.
        annual = funding.annualised_funding * 100
        parts.append(f"perp basis {funding.basis_bps:+.1f}bps, funding {annual:+.1f}%/yr")
    if not parts:
        return ""
    return f"  {pair}: " + "; ".join(parts)
