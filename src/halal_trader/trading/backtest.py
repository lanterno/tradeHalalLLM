"""Stock backtester — replays a rule-based strategy against Alpaca bars.

Pattern parallels :mod:`halal_trader.crypto.backtest`. Differences:

* Input bars come from Alpaca ``get_stock_bars`` (a different shape than
  Binance klines). We use :func:`trading.risk._bars_to_klines` to coerce.
* Sharpe annualisation is daily, not minute (252 trading days, not
  24×60 minutes).
* Trade fees default to zero — Alpaca commission-free; backtester users
  can override if they're modelling SOR/PFOF impact later.

The engine reuses ``crypto.backtest.SimulatedExecutor`` because the
executor is asset-class-agnostic (the slippage model is vol-aware
already, which works for any underlying with an ATR signal). Stock
backtests share the same ``BacktestResult`` shape so analytics can roll
both up uniformly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from halal_trader.crypto.backtest import BacktestResult, SimulatedExecutor
from halal_trader.crypto.indicators import compute_all
from halal_trader.domain.models import Kline
from halal_trader.trading.bars import bars_to_klines

logger = logging.getLogger(__name__)


class StockBacktestEngine:
    """Replays Alpaca stock bars through a rule-based strategy."""

    def __init__(
        self,
        *,
        initial_balance: float = 10000.0,
        slippage_pct: float = 0.0005,
        fee_pct: float = 0.0,  # Alpaca = commission-free
        max_position_pct: float = 0.10,
        rsi_buy: float = 35.0,
        rsi_sell: float = 65.0,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
        atr_baseline: float = 0.02,
    ) -> None:
        self._initial_balance = initial_balance
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct
        self._max_position_pct = max_position_pct
        self._rsi_buy = rsi_buy
        self._rsi_sell = rsi_sell
        self._sl_pct = sl_pct
        self._tp_pct = tp_pct
        self._atr_baseline = atr_baseline

    async def run(
        self,
        symbol: str,
        bars: Any,
        *,
        window_size: int = 50,
    ) -> BacktestResult:
        """Run the backtest on Alpaca historical bars.

        ``bars`` is whatever ``AlpacaMCPClient.get_stock_bars`` returned —
        the helper handles both flat-list and ``{"bars": [...]}`` shapes.
        """
        klines = bars_to_klines(bars)
        if len(klines) <= window_size:
            logger.warning(
                "Not enough bars for backtest (%d <= window=%d) — returning empty result",
                len(klines),
                window_size,
            )
            return BacktestResult(
                pair=symbol,
                start_date="",
                end_date="",
                initial_balance=self._initial_balance,
                final_balance=self._initial_balance,
                equity_curve=[self._initial_balance],
            )

        executor = SimulatedExecutor(
            initial_balance=self._initial_balance,
            slippage_pct=self._slippage_pct,
            fee_pct=self._fee_pct,
            max_position_pct=self._max_position_pct,
            atr_baseline=self._atr_baseline,
        )

        for i in range(window_size, len(klines)):
            window = klines[i - window_size : i]
            current = klines[i]

            executor.check_sl_tp(current)

            indicators = compute_all(window)
            if "error" in indicators:
                executor.update_equity(current.close)
                continue

            rsi = indicators.get("rsi_14")
            macd_hist = indicators.get("macd_histogram")
            bb_pos = indicators.get("bb_position")
            atr_pct = indicators.get("atr_pct", 0.0) or 0.0

            if executor.position is None and rsi is not None:
                buy_signal = (
                    rsi < self._rsi_buy
                    and (macd_hist is not None and macd_hist > 0)
                    and (bb_pos is not None and bb_pos < 0.3)
                )
                if buy_signal:
                    sl = current.close * (1 - self._sl_pct)
                    tp = current.close * (1 + self._tp_pct)
                    executor.buy(
                        symbol,
                        current.close,
                        current.close_time,
                        stop_loss=sl,
                        target_price=tp,
                        reasoning=f"RSI={rsi:.0f}, MACD_hist={macd_hist:.6f}, BB={bb_pos:.2f}",
                        atr_pct=atr_pct,
                    )
            elif executor.position is not None and rsi is not None:
                sell_signal = rsi > self._rsi_sell and (macd_hist is not None and macd_hist < 0)
                if sell_signal:
                    executor.sell(current.close, current.close_time, "signal", atr_pct=atr_pct)

            executor.update_equity(current.close)

        if executor.position is not None:
            executor.sell(klines[-1].close, klines[-1].close_time, "backtest_end")

        return self._compute_results(symbol, klines, executor)

    def _compute_results(
        self, symbol: str, klines: list[Kline], executor: SimulatedExecutor
    ) -> BacktestResult:
        """Mirror crypto BacktestEngine._compute_results, daily annualisation."""
        trades = executor.closed_trades
        equity = executor.equity_curve

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = sum(abs(t.pnl) for t in losses)

        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        if len(equity) > 1:
            returns = np.diff(equity) / np.array(equity[:-1])
            # Daily bars → annualise by sqrt(252).
            annual = np.sqrt(252)
            std_r = float(np.std(returns))
            sharpe = float(np.mean(returns) / std_r * annual) if std_r > 0 else 0.0
            downside = returns[returns < 0]
            std_d = float(np.std(downside)) if len(downside) > 0 else 0.0
            sortino = float(np.mean(returns) / std_d * annual) if std_d > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        # avg hold time in candles (i.e. days for a daily-bar backtest).
        durations = [
            (t.exit_timestamp - t.timestamp) / 60_000  # synthetic ts in ms
            for t in trades
            if t.exit_timestamp is not None
        ]
        avg_hold = float(sum(durations) / len(durations)) if durations else 0.0

        final_balance = equity[-1] if equity else self._initial_balance
        total_pnl = final_balance - self._initial_balance

        return BacktestResult(
            pair=symbol,
            start_date="",
            end_date="",
            initial_balance=self._initial_balance,
            final_balance=final_balance,
            total_return_pct=(total_pnl / self._initial_balance if self._initial_balance else 0.0),
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0.0,
            profit_factor=gross_wins / gross_losses if gross_losses > 0 else 0.0,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_hold_candles=avg_hold,
            trades=trades,
            equity_curve=equity,
        )
