"""Spot-perp basis features.

The bot only trades spot (perpetual futures aren't permissible under
our halal interpretation), but the *information* in the perp market is
free for the taking. Persistent positive basis (perp price > spot)
implies leveraged longs are dominant; flipping basis often precedes
short-term spot moves the same way.

This module computes basis features for any pair, given fresh spot +
perp marks plus the latest funding rate. It is wire-format agnostic
on purpose: the caller picks the venue (Binance, Bybit, OKX) and
hands native floats in.

Three derived signals:

* ``basis_bps`` — (perp - spot) / spot, in basis points.
* ``funding_rate_pct`` — pass-through of the venue funding rate.
* ``basis_zscore`` — z-score of basis relative to its trailing window,
  so "unusually rich" moves stand out without hand-tuning thresholds.
* ``regime`` — categorical: ``"contango"``, ``"backwardation"``, or
  ``"neutral"``, based on basis_bps + funding sign.

All halal-permissible because we *observe* perp pricing, never trade
it. The risk policy that pairs with this should still refuse to size up
into a contango blow-off (the entire point).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

BasisRegime = Literal["contango", "backwardation", "neutral"]


# ── One-shot feature ─────────────────────────────────────────────


@dataclass(frozen=True)
class BasisFeatures:
    pair: str
    spot_price: float
    perp_price: float
    funding_rate_pct: float
    basis_bps: float
    basis_zscore: float = 0.0
    regime: BasisRegime = "neutral"


def compute_basis(
    *,
    pair: str,
    spot_price: float,
    perp_price: float,
    funding_rate_pct: float,
    basis_history: Iterable[float] | None = None,
    contango_bps: float = 25.0,
    backwardation_bps: float = -25.0,
) -> BasisFeatures:
    """Build a single :class:`BasisFeatures` snapshot.

    ``basis_history`` (most recent N values, in basis points) lets us
    compute a z-score; pass an empty iterable to skip — the field stays
    at 0.0 in that case.
    """
    if spot_price <= 0:
        return BasisFeatures(
            pair=pair,
            spot_price=spot_price,
            perp_price=perp_price,
            funding_rate_pct=funding_rate_pct,
            basis_bps=0.0,
        )
    basis_bps = (perp_price - spot_price) / spot_price * 10_000.0
    z = _zscore(basis_history, basis_bps) if basis_history else 0.0
    regime: BasisRegime
    if basis_bps >= contango_bps and funding_rate_pct > 0:
        regime = "contango"
    elif basis_bps <= backwardation_bps and funding_rate_pct < 0:
        regime = "backwardation"
    else:
        regime = "neutral"
    return BasisFeatures(
        pair=pair,
        spot_price=spot_price,
        perp_price=perp_price,
        funding_rate_pct=funding_rate_pct,
        basis_bps=basis_bps,
        basis_zscore=z,
        regime=regime,
    )


def _zscore(history: Iterable[float], current: float) -> float:
    xs = list(history)
    if len(xs) < 5:
        return 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
    if var == 0:
        return 0.0
    return (current - mu) / math.sqrt(var)


# ── Rolling tracker ──────────────────────────────────────────────


@dataclass
class BasisTracker:
    """Per-pair rolling history of basis observations.

    ``observe()`` returns the freshly computed :class:`BasisFeatures`
    so callers can both record and consume in one call.
    """

    window: int = 96  # ~24h of 15-min samples
    history_by_pair: dict[str, deque[float]] = field(default_factory=dict)

    def observe(
        self,
        *,
        pair: str,
        spot_price: float,
        perp_price: float,
        funding_rate_pct: float,
    ) -> BasisFeatures:
        hist = self.history_by_pair.setdefault(pair, deque(maxlen=self.window))
        feat = compute_basis(
            pair=pair,
            spot_price=spot_price,
            perp_price=perp_price,
            funding_rate_pct=funding_rate_pct,
            basis_history=list(hist),
        )
        hist.append(feat.basis_bps)
        return feat


# ── Prompt formatting ────────────────────────────────────────────


def format_basis_for_prompt(features: dict[str, BasisFeatures]) -> str:
    """One block of pair → basis lines for the LLM prompt.

    Always emit even when ``features`` is empty — the prompt builder
    will elide the section if the body is empty.
    """
    if not features:
        return ""
    lines = ["Spot/perp basis (read-only — execute SPOT only):"]
    for pair, f in sorted(features.items()):
        sign = "+" if f.basis_bps >= 0 else ""
        z_part = f" (z={f.basis_zscore:+.2f})" if abs(f.basis_zscore) >= 0.01 else ""
        lines.append(
            f"  {pair}: basis {sign}{f.basis_bps:.0f}bps{z_part}, "
            f"funding {f.funding_rate_pct:+.4%}, regime {f.regime}"
        )
    return "\n".join(lines)


# ── Risk policy hook ─────────────────────────────────────────────


@dataclass(frozen=True)
class BasisRiskPolicy:
    """Reduce position sizing in extreme contango / backwardation.

    Pure data — easy to swap. Maps a pair's :class:`BasisFeatures`
    to a sizing multiplier for *new buys*. Sells/exits are not modified.
    """

    extreme_contango_bps: float = 100.0
    extreme_backwardation_bps: float = -100.0
    extreme_size_multiplier: float = 0.5

    def buy_size_multiplier(self, f: BasisFeatures) -> float:
        if f.regime == "contango" and f.basis_bps >= self.extreme_contango_bps:
            return self.extreme_size_multiplier
        if (
            f.regime == "backwardation"
            and f.basis_bps <= self.extreme_backwardation_bps
        ):
            return self.extreme_size_multiplier
        return 1.0
