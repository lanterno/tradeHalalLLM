"""Risk state + engine (REARCHITECTURE L7).

``RiskState`` carries the three halt conditions and the size multipliers the
policy multiplies target weights by. This is a minimal, correct engine: heat /
drawdown / daily-loss halts are real; correlation/volatility multipliers default
to 1.0 with the structure to fold in ``halal_trader/crypto/risk.py``'s
correlation + ATR scaling later. ``gross_exposure`` feeds the policy's
no-leverage normalization (R-03).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from halabot.belief.schema import BeliefState

# A per-asset return series as (timestamp, return) pairs, so correlation aligns
# on shared bar times rather than positionally (fix: tail-index misalignment).
TimedReturns = list[tuple[datetime, float]]


@dataclass(frozen=True)
class RiskConfig:
    max_portfolio_heat_pct: float = 0.05  # unrealized loss on open positions
    max_drawdown_pct: float = 0.08  # peak-to-trough equity
    daily_loss_limit: float = 0.02  # realized intraday loss floor (R-10)
    correlation_threshold: float = 0.7  # |corr| above which two names are "clustered"
    volatile_size_mult: float = 0.6  # size haircut for a VOLATILE-regime name


@dataclass(frozen=True)
class RiskState:
    portfolio_heat_pct: float = 0.0
    drawdown_pct: float = 0.0
    realized_loss_today_pct: float = 0.0
    gross_exposure: float = 0.0
    halted: bool = False
    reason: str | None = None
    _correlation_mult: dict[str, float] = field(default_factory=dict)
    _volatility_mult: dict[str, float] = field(default_factory=dict)

    def correlation_multiplier(self, asset: str) -> float:
        return self._correlation_mult.get(asset, 1.0)

    def volatility_multiplier(self, asset: str) -> float:
        return self._volatility_mult.get(asset, 1.0)


@dataclass(frozen=True)
class PortfolioSnapshot:
    """What the risk engine evaluates: equity + open exposure (broker truth)."""

    equity: float
    peak_equity: float
    unrealized_pnl: float
    realized_pnl_today: float
    starting_equity_today: float
    gross_exposure: float  # Σ |position weight|


class RiskEngine(Protocol):
    def evaluate(
        self,
        snapshot: PortfolioSnapshot,
        *,
        beliefs: list[BeliefState] | None = None,
        returns_by_asset: dict[str, TimedReturns] | None = None,
    ) -> RiskState: ...


def _pearson(a: list[float], b: list[float]) -> float | None:
    """Pearson correlation of two equal-length aligned vectors, None if degenerate."""
    n = len(a)
    if n < 10 or len(b) != n:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return cov / math.sqrt(va * vb)


def _aligned(x: TimedReturns, y: TimedReturns) -> tuple[list[float], list[float]]:
    """Inner-join two timestamped return series on shared timestamps."""
    yb = dict(y)
    xs: list[float] = []
    ys: list[float] = []
    for ts, rx in x:
        if ts in yb:
            xs.append(rx)
            ys.append(yb[ts])
    return xs, ys


def correlation_multipliers(
    returns_by_asset: dict[str, TimedReturns], *, threshold: float
) -> dict[str, float]:
    """Per-asset size haircut for correlation clustering: an asset correlated
    (|ρ| > threshold) with k other names is scaled by 1/√(1+k), so a cluster of
    co-moving bets carries roughly √k risk instead of k (diversification-aware).

    Series are inner-joined on TIMESTAMP before correlating, so unequal-length or
    gapped histories never pair returns from different times (fix R, alignment)."""
    assets = list(returns_by_asset)
    peers: dict[str, int] = dict.fromkeys(assets, 0)
    for i, x in enumerate(assets):
        for y in assets[i + 1 :]:
            ax, ay = _aligned(returns_by_asset[x], returns_by_asset[y])
            rho = _pearson(ax, ay)
            if rho is not None and abs(rho) > threshold:
                peers[x] += 1
                peers[y] += 1
    return {a: 1.0 / math.sqrt(1 + k) for a, k in peers.items()}


class BasicRiskEngine:
    """Heat / drawdown / daily-loss halts + belief-aware size multipliers.

    Multipliers (applied to target weight by ``policy.sizing``): a VOLATILE-regime
    name is haircut by ``volatile_size_mult``; correlated clusters are haircut by
    :func:`correlation_multipliers` when per-asset returns are supplied."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._cfg = config or RiskConfig()

    def evaluate(
        self,
        snapshot: PortfolioSnapshot,
        *,
        beliefs: list[BeliefState] | None = None,
        returns_by_asset: dict[str, TimedReturns] | None = None,
    ) -> RiskState:
        eq = snapshot.equity or 1.0
        heat = max(0.0, -snapshot.unrealized_pnl) / eq  # only losses count as heat
        drawdown = (
            max(0.0, snapshot.peak_equity - snapshot.equity) / snapshot.peak_equity
            if snapshot.peak_equity > 0
            else 0.0
        )
        start = snapshot.starting_equity_today or 1.0
        realized_loss = max(0.0, -snapshot.realized_pnl_today) / start

        halted, reason = False, None
        if heat > self._cfg.max_portfolio_heat_pct:
            halted = True
            reason = f"portfolio heat {heat:.1%} > {self._cfg.max_portfolio_heat_pct:.1%}"
        elif drawdown > self._cfg.max_drawdown_pct:
            halted = True
            reason = f"drawdown {drawdown:.1%} > {self._cfg.max_drawdown_pct:.1%}"
        elif realized_loss >= self._cfg.daily_loss_limit:
            halted = True
            reason = f"daily realized loss {realized_loss:.1%} >= {self._cfg.daily_loss_limit:.1%}"

        # Belief-aware size haircuts (sizing reads these per asset).
        vol_mult: dict[str, float] = {}
        if beliefs:
            from halabot.belief.schema import Regime

            for b in beliefs:
                if b.regime == Regime.VOLATILE:
                    vol_mult[b.asset] = self._cfg.volatile_size_mult
        corr_mult: dict[str, float] = {}
        if returns_by_asset:
            corr_mult = correlation_multipliers(
                returns_by_asset, threshold=self._cfg.correlation_threshold
            )

        return RiskState(
            portfolio_heat_pct=heat,
            drawdown_pct=drawdown,
            realized_loss_today_pct=realized_loss,
            gross_exposure=snapshot.gross_exposure,
            halted=halted,
            reason=reason,
            _correlation_mult=corr_mult,
            _volatility_mult=vol_mult,
        )
