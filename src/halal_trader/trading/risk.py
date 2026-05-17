"""Stock portfolio-risk adapter — feeds Alpaca bars to the shared risk engine.

The crypto engine ( :class:`crypto.risk.PortfolioRiskEngine` ) is already
broker-agnostic: it just needs a ``klines_by_symbol`` mapping plus
indicator + position context. This module wires Alpaca bars into that
engine; bar-shape coercion and indicator computation live in
:mod:`halal_trader.trading.bars`.

Round-4 wave 0.C moved the risk knobs (``atr_baseline``,
``max_portfolio_heat_pct``, ``max_drawdown_pct``,
``high_correlation_threshold``, ``correlation_reduction_factor``) onto
:class:`StockSettings` so stocks + crypto can be tuned independently
(daily equity bars vs. 1-minute crypto klines have very different
volatility regimes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from halal_trader.config import Settings
from halal_trader.crypto.risk import PortfolioRiskEngine, PortfolioRiskState
from halal_trader.domain.models import Position
from halal_trader.trading.bars import bars_to_klines, compute_indicators_by_symbol

logger = logging.getLogger(__name__)


# Backwards-compatible alias — the helper now lives in trading.bars but
# a few call sites still import this private name.
_bars_to_klines = bars_to_klines


@dataclass
class StockRiskOutput:
    state: PortfolioRiskState
    risk_text: str
    # Per-symbol indicator dict (rsi_14, atr_14, ema_9, ema_21, vwap, …).
    # Surfaced so the cycle can hand it to regime / ML stages without
    # re-parsing the bars payload.
    indicators_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)


def evaluate_stock_risk(
    *,
    settings: Settings,
    bars_by_symbol: dict[str, Any],
    positions: list[Position],
    total_equity: float,
) -> StockRiskOutput:
    """Run the shared risk engine against an Alpaca bars payload.

    Returns the ``PortfolioRiskState`` plus a prompt-ready string and
    the per-symbol indicator cache used along the way.
    """
    klines_by_symbol, indicators_cache = compute_indicators_by_symbol(bars_by_symbol)

    open_positions_value: dict[str, float] = {
        p.symbol: float(p.qty) * float(p.current_price or p.avg_entry_price) for p in positions
    }
    unrealized_pnl: dict[str, float] = {p.symbol: float(p.unrealized_pl) for p in positions}

    engine = PortfolioRiskEngine(
        base_max_position_pct=settings.stocks.max_position_pct,
        max_portfolio_heat_pct=settings.stocks.max_portfolio_heat_pct,
        max_drawdown_pct=settings.stocks.max_drawdown_pct,
        high_correlation_threshold=settings.stocks.high_correlation_threshold,
        correlation_reduction_factor=settings.stocks.correlation_reduction_factor,
        atr_baseline=settings.stocks.atr_baseline,
    )
    state = engine.evaluate(
        klines_by_symbol=klines_by_symbol,
        indicators_cache=indicators_cache,
        open_positions_value=open_positions_value,
        unrealized_pnl=unrealized_pnl,
        total_equity=total_equity,
    )
    text = engine.format_for_prompt(state)
    return StockRiskOutput(state=state, risk_text=text, indicators_by_symbol=indicators_cache)
