"""Risk state + engine (REARCHITECTURE L7).

``RiskState`` carries the three halt conditions and the size multipliers the
policy multiplies target weights by. This is a minimal, correct engine: heat /
drawdown / daily-loss halts are real; correlation/volatility multipliers default
to 1.0 with the structure to fold in ``halal_trader/crypto/risk.py``'s
correlation + ATR scaling later. ``gross_exposure`` feeds the policy's
no-leverage normalization (R-03).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RiskConfig:
    max_portfolio_heat_pct: float = 0.05  # unrealized loss on open positions
    max_drawdown_pct: float = 0.08  # peak-to-trough equity
    daily_loss_limit: float = 0.02  # realized intraday loss floor (R-10)


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
    def evaluate(self, snapshot: PortfolioSnapshot) -> RiskState: ...


class BasicRiskEngine:
    """Heat / drawdown / daily-loss halts; identity size multipliers (for now)."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._cfg = config or RiskConfig()

    def evaluate(self, snapshot: PortfolioSnapshot) -> RiskState:
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

        return RiskState(
            portfolio_heat_pct=heat,
            drawdown_pct=drawdown,
            realized_loss_today_pct=realized_loss,
            gross_exposure=snapshot.gross_exposure,
            halted=halted,
            reason=reason,
        )
