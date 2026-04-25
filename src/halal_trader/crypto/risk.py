"""Portfolio-level risk engine — correlation, volatility-adjusted sizing, heat, drawdown."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)

_MIN_KLINES_FOR_CORRELATION = 30


@dataclass
class PortfolioRiskState:
    """Snapshot of portfolio-level risk metrics for a single cycle."""

    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    avg_correlation: float = 0.0
    adjusted_position_pcts: dict[str, float] = field(default_factory=dict)
    portfolio_heat: float = 0.0
    portfolio_heat_pct: float = 0.0
    drawdown_from_peak: float = 0.0
    drawdown_pct: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""


class PortfolioRiskEngine:
    """Enforces portfolio-level risk constraints beyond individual SL/TP.

    - Correlation matrix: reduces exposure when positions are highly correlated
    - Volatility-adjusted sizing: scales position size inversely with ATR
    - Portfolio heat: blocks new entries when aggregate unrealized P&L is too negative
    - Drawdown circuit breaker: halts trading on deep peak-to-trough drawdown
    """

    def __init__(
        self,
        *,
        base_max_position_pct: float = 0.25,
        max_portfolio_heat_pct: float = 0.05,
        max_drawdown_pct: float = 0.08,
        high_correlation_threshold: float = 0.7,
        correlation_reduction_factor: float = 0.5,
        atr_baseline: float = 0.02,
    ) -> None:
        self._base_max_position_pct = base_max_position_pct
        self._max_heat_pct = max_portfolio_heat_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._high_corr_threshold = high_correlation_threshold
        self._corr_reduction = correlation_reduction_factor
        self._atr_baseline = atr_baseline
        self._equity_peak: float = 0.0

    def evaluate(
        self,
        klines_by_symbol: dict[str, list[Kline]],
        indicators_cache: dict[str, dict],
        open_positions_value: dict[str, float],
        unrealized_pnl: dict[str, float],
        total_equity: float,
    ) -> PortfolioRiskState:
        """Run all risk checks and return the current risk state."""
        state = PortfolioRiskState()

        if total_equity > self._equity_peak:
            self._equity_peak = total_equity

        state.correlation_matrix = self._compute_correlations(klines_by_symbol)
        state.avg_correlation = self._average_correlation(state.correlation_matrix)

        state.adjusted_position_pcts = self._compute_adjusted_sizing(
            indicators_cache, state.correlation_matrix, list(open_positions_value.keys())
        )

        total_unrealized = sum(unrealized_pnl.values())
        state.portfolio_heat = total_unrealized
        state.portfolio_heat_pct = total_unrealized / total_equity if total_equity > 0 else 0

        if self._equity_peak > 0:
            state.drawdown_from_peak = self._equity_peak - total_equity
            state.drawdown_pct = state.drawdown_from_peak / self._equity_peak
        else:
            state.drawdown_from_peak = 0.0
            state.drawdown_pct = 0.0

        if state.drawdown_pct >= self._max_drawdown_pct:
            state.is_halted = True
            state.halt_reason = (
                f"Drawdown circuit breaker: {state.drawdown_pct:.1%} "
                f"exceeds {self._max_drawdown_pct:.1%} limit"
            )
            logger.warning(state.halt_reason)

        if state.portfolio_heat_pct < -self._max_heat_pct:
            state.is_halted = True
            state.halt_reason = (
                f"Portfolio heat limit: unrealized P&L {state.portfolio_heat_pct:.1%} "
                f"exceeds -{self._max_heat_pct:.1%} threshold"
            )
            logger.warning(state.halt_reason)

        return state

    def get_adjusted_max_position_pct(self, symbol: str, state: PortfolioRiskState) -> float:
        """Get the risk-adjusted max position % for a symbol."""
        return state.adjusted_position_pcts.get(symbol, self._base_max_position_pct)

    def _compute_correlations(
        self, klines_by_symbol: dict[str, list[Kline]]
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise Pearson correlation of returns."""
        symbols = sorted(klines_by_symbol.keys())
        if len(symbols) < 2:
            return {}

        returns_by_sym: dict[str, np.ndarray] = {}
        for sym in symbols:
            klines = klines_by_symbol[sym]
            if len(klines) < _MIN_KLINES_FOR_CORRELATION:
                continue
            closes = np.array([k.close for k in klines])
            returns = np.diff(closes) / closes[:-1]
            returns_by_sym[sym] = returns

        valid_symbols = sorted(returns_by_sym.keys())
        if len(valid_symbols) < 2:
            return {}

        min_len = min(len(r) for r in returns_by_sym.values())
        matrix: dict[str, dict[str, float]] = {}

        for i, sym_a in enumerate(valid_symbols):
            matrix[sym_a] = {}
            for j, sym_b in enumerate(valid_symbols):
                if i == j:
                    matrix[sym_a][sym_b] = 1.0
                elif j < i:
                    matrix[sym_a][sym_b] = matrix[sym_b][sym_a]
                else:
                    r_a = returns_by_sym[sym_a][-min_len:]
                    r_b = returns_by_sym[sym_b][-min_len:]
                    corr = float(np.corrcoef(r_a, r_b)[0, 1])
                    matrix[sym_a][sym_b] = corr if np.isfinite(corr) else 0.0

        return matrix

    def _average_correlation(self, corr_matrix: dict[str, dict[str, float]]) -> float:
        """Compute average off-diagonal correlation."""
        if not corr_matrix:
            return 0.0
        values = []
        symbols = list(corr_matrix.keys())
        for i, a in enumerate(symbols):
            for j, b in enumerate(symbols):
                if i < j:
                    values.append(corr_matrix[a][b])
        return float(np.mean(values)) if values else 0.0

    def _compute_adjusted_sizing(
        self,
        indicators_cache: dict[str, dict],
        corr_matrix: dict[str, dict[str, float]],
        open_symbols: list[str],
    ) -> dict[str, float]:
        """Compute risk-adjusted position size for each symbol."""
        result: dict[str, float] = {}

        for symbol, indicators in indicators_cache.items():
            if indicators.get("error"):
                result[symbol] = self._base_max_position_pct
                continue

            pct = self._base_max_position_pct

            atr_pct = indicators.get("atr_pct", indicators.get("atr_14", 0))
            if atr_pct > 0 and self._atr_baseline > 0:
                vol_scale = min(2.0, max(0.3, self._atr_baseline / atr_pct))
                pct *= vol_scale

            if corr_matrix and open_symbols:
                max_corr_with_open = 0.0
                for open_sym in open_symbols:
                    if open_sym in corr_matrix and symbol in corr_matrix.get(open_sym, {}):
                        c = abs(corr_matrix[open_sym][symbol])
                        max_corr_with_open = max(max_corr_with_open, c)

                if max_corr_with_open >= self._high_corr_threshold:
                    pct *= self._corr_reduction

            result[symbol] = max(0.05, min(pct, self._base_max_position_pct))

        return result

    def format_for_prompt(self, state: PortfolioRiskState) -> str:
        """Format the risk state as text for the LLM prompt."""
        lines = []

        if state.avg_correlation > 0:
            corr_label = (
                "HIGH"
                if state.avg_correlation > 0.7
                else ("MODERATE" if state.avg_correlation > 0.4 else "LOW")
            )
            lines.append(
                f"Portfolio Correlation: {state.avg_correlation:.2f} ({corr_label}) — "
                + (
                    "reduce exposure, positions move together"
                    if corr_label == "HIGH"
                    else "diversification is adequate"
                )
            )

        if state.portfolio_heat != 0:
            lines.append(
                f"Portfolio Heat: ${state.portfolio_heat:+,.2f} "
                f"({state.portfolio_heat_pct:+.1%} of equity)"
            )

        if state.drawdown_pct > 0:
            lines.append(
                f"Drawdown from Peak: {state.drawdown_pct:.1%} "
                f"(max allowed: {self._max_drawdown_pct:.1%})"
            )

        if state.adjusted_position_pcts:
            adjusted = []
            for sym, pct in sorted(state.adjusted_position_pcts.items()):
                if abs(pct - self._base_max_position_pct) > 0.01:
                    adjusted.append(f"{sym}: {pct:.0%}")
            if adjusted:
                lines.append("Risk-Adjusted Position Limits: " + ", ".join(adjusted))

        if state.is_halted:
            lines.append(f"⚠ RISK HALT: {state.halt_reason}")

        return "\n".join(lines) if lines else ""
