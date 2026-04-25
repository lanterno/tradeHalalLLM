"""Stock portfolio-risk adapter — feeds Alpaca bars to the shared risk engine.

The crypto engine ( :class:`crypto.risk.PortfolioRiskEngine` ) is already
broker-agnostic: it just needs a ``klines_by_symbol`` mapping plus
indicator + position context. This adapter:

  1. Normalises Alpaca's bars-by-symbol response into the
     :class:`domain.models.Kline` shape.
  2. Computes the same per-symbol indicators the crypto cycle does
     (``rsi_14``, ``atr_14``, ``ema_9``, ``ema_21``, ``vwap``, …).
  3. Runs ``PortfolioRiskEngine.evaluate`` and returns a
     ``(state, risk_text)`` pair the stock cycle can splice into the
     LLM prompt.

The atr_baseline default (2%) is reasonable for daily stock bars; the
operator can override via ``Settings.crypto.atr_baseline`` if they want
a different floor.  A future split would add stock-specific settings,
but they aren't load-bearing today.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from halal_trader.config import Settings
from halal_trader.crypto.indicators import compute_all
from halal_trader.crypto.risk import PortfolioRiskEngine, PortfolioRiskState
from halal_trader.domain.models import Kline, Position

logger = logging.getLogger(__name__)


@dataclass
class StockRiskOutput:
    state: PortfolioRiskState
    risk_text: str


def _bars_to_klines(bars_for_symbol: Any) -> list[Kline]:
    """Best-effort coercion of Alpaca's ``get_stock_bars`` response into Klines.

    Alpaca returns a list of dicts with ``t``, ``o``, ``h``, ``l``, ``c``, ``v``
    keys (or sometimes nested under a ``bars`` field). We tolerate both
    shapes and unfamiliar bar entries.
    """
    if not bars_for_symbol:
        return []
    raw_bars: list[dict[str, Any]]
    if isinstance(bars_for_symbol, dict):
        raw_bars = bars_for_symbol.get("bars") or bars_for_symbol.get("data") or []
    elif isinstance(bars_for_symbol, list):
        raw_bars = bars_for_symbol
    else:
        return []

    out: list[Kline] = []
    for i, bar in enumerate(raw_bars):
        if not isinstance(bar, dict):
            continue
        try:
            o = float(bar.get("o", bar.get("open", 0)))
            h = float(bar.get("h", bar.get("high", 0)))
            low = float(bar.get("l", bar.get("low", 0)))
            c = float(bar.get("c", bar.get("close", 0)))
            v = float(bar.get("v", bar.get("volume", 0)))
        except TypeError, ValueError:
            continue
        if c <= 0:
            continue
        # Synthetic monotonic times (in ms) — the engine only uses
        # close prices for correlation, so the precise timestamp doesn't
        # matter as long as ordering is preserved.
        ts = i * 60_000
        out.append(
            Kline(
                open_time=ts,
                open=o,
                high=h,
                low=low,
                close=c,
                volume=v,
                close_time=ts + 60_000,
            )
        )
    return out


def evaluate_stock_risk(
    *,
    settings: Settings,
    bars_by_symbol: dict[str, Any],
    positions: list[Position],
    total_equity: float,
) -> StockRiskOutput:
    """Run the shared risk engine against an Alpaca bars payload.

    Returns the ``PortfolioRiskState`` plus a prompt-ready string.
    """
    klines_by_symbol: dict[str, list[Kline]] = {}
    indicators_cache: dict[str, dict] = {}

    for symbol, raw in bars_by_symbol.items():
        klines = _bars_to_klines(raw)
        if not klines:
            continue
        klines_by_symbol[symbol] = klines
        indicators_cache[symbol] = compute_all(klines)

    open_positions_value: dict[str, float] = {
        p.symbol: float(p.qty) * float(p.current_price or p.avg_entry_price) for p in positions
    }
    unrealized_pnl: dict[str, float] = {p.symbol: float(p.unrealized_pl) for p in positions}

    engine = PortfolioRiskEngine(
        base_max_position_pct=settings.stocks.max_position_pct,
        max_portfolio_heat_pct=settings.crypto.max_portfolio_heat_pct,
        max_drawdown_pct=settings.crypto.max_drawdown_pct,
        high_correlation_threshold=settings.crypto.high_correlation_threshold,
        correlation_reduction_factor=settings.crypto.correlation_reduction_factor,
        atr_baseline=settings.crypto.atr_baseline,
    )
    state = engine.evaluate(
        klines_by_symbol=klines_by_symbol,
        indicators_cache=indicators_cache,
        open_positions_value=open_positions_value,
        unrealized_pnl=unrealized_pnl,
        total_equity=total_equity,
    )
    text = engine.format_for_prompt(state)
    return StockRiskOutput(state=state, risk_text=text)
