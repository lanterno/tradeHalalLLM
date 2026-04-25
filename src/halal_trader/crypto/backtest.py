"""Backtesting engine — replays historical data through the strategy pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from halal_trader.crypto.indicators import compute_all
from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTrade:
    """A trade generated during backtesting."""

    pair: str
    side: str
    quantity: float
    price: float
    timestamp: int
    reasoning: str = ""
    stop_loss: float | None = None
    target_price: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_timestamp: int | None = None
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtesting results."""

    pair: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_return_pct: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_hold_candles: float = 0.0
    trades: list[SimulatedTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class SimulatedExecutor:
    """Simulates trade execution with slippage and fees."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        *,
        slippage_pct: float = 0.0005,
        fee_pct: float = 0.001,
        max_position_pct: float = 0.25,
    ) -> None:
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct
        self._max_position_pct = max_position_pct
        self.position: SimulatedTrade | None = None
        self.closed_trades: list[SimulatedTrade] = []
        self.equity_curve: list[float] = [initial_balance]

    def buy(
        self,
        pair: str,
        price: float,
        timestamp: int,
        *,
        stop_loss: float | None = None,
        target_price: float | None = None,
        reasoning: str = "",
    ) -> bool:
        """Simulate a buy order."""
        if self.position is not None:
            return False

        fill_price = price * (1 + self._slippage_pct)
        max_spend = self.balance * self._max_position_pct
        quantity = max_spend / fill_price
        cost = quantity * fill_price * (1 + self._fee_pct)

        if cost > self.balance:
            return False

        self.balance -= cost
        self.position = SimulatedTrade(
            pair=pair,
            side="buy",
            quantity=quantity,
            price=fill_price,
            timestamp=timestamp,
            stop_loss=stop_loss,
            target_price=target_price,
            reasoning=reasoning,
        )
        return True

    def sell(self, price: float, timestamp: int, reason: str = "signal") -> SimulatedTrade | None:
        """Simulate a sell order, closing the current position."""
        if self.position is None:
            return None

        fill_price = price * (1 - self._slippage_pct)
        proceeds = self.position.quantity * fill_price * (1 - self._fee_pct)
        self.balance += proceeds

        self.position.exit_price = fill_price
        self.position.exit_timestamp = timestamp
        self.position.exit_reason = reason
        self.position.pnl = (fill_price - self.position.price) * self.position.quantity

        trade = self.position
        self.closed_trades.append(trade)
        self.position = None
        return trade

    def check_sl_tp(self, kline: Kline) -> SimulatedTrade | None:
        """Check stop-loss and take-profit against a candle."""
        if self.position is None:
            return None

        if self.position.stop_loss and kline.low <= self.position.stop_loss:
            return self.sell(self.position.stop_loss, kline.close_time, "stop_loss")

        if self.position.target_price and kline.high >= self.position.target_price:
            return self.sell(self.position.target_price, kline.close_time, "take_profit")

        return None

    def update_equity(self, current_price: float) -> None:
        """Record current equity for the equity curve."""
        equity = self.balance
        if self.position is not None:
            equity += self.position.quantity * current_price
        self.equity_curve.append(equity)


class BacktestEngine:
    """Replays historical klines through a rule-based strategy."""

    def __init__(
        self,
        *,
        initial_balance: float = 10000.0,
        slippage_pct: float = 0.0005,
        fee_pct: float = 0.001,
        max_position_pct: float = 0.25,
        rsi_buy: float = 35.0,
        rsi_sell: float = 65.0,
        sl_pct: float = 0.01,
        tp_pct: float = 0.015,
    ) -> None:
        self._initial_balance = initial_balance
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct
        self._max_position_pct = max_position_pct
        self._rsi_buy = rsi_buy
        self._rsi_sell = rsi_sell
        self._sl_pct = sl_pct
        self._tp_pct = tp_pct

    async def run(
        self,
        pair: str,
        klines: list[Kline],
        *,
        window_size: int = 100,
    ) -> BacktestResult:
        """Run the backtest on historical klines."""
        executor = SimulatedExecutor(
            initial_balance=self._initial_balance,
            slippage_pct=self._slippage_pct,
            fee_pct=self._fee_pct,
            max_position_pct=self._max_position_pct,
        )

        for i in range(window_size, len(klines)):
            window = klines[i - window_size : i]
            current = klines[i]

            # Check SL/TP first
            executor.check_sl_tp(current)

            # Compute indicators
            indicators = compute_all(window)
            if "error" in indicators:
                executor.update_equity(current.close)
                continue

            rsi = indicators.get("rsi_14")
            macd_hist = indicators.get("macd_histogram")
            bb_pos = indicators.get("bb_position")

            # Rule-based entry signals
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
                        pair,
                        current.close,
                        current.close_time,
                        stop_loss=sl,
                        target_price=tp,
                        reasoning=f"RSI={rsi:.0f}, MACD_hist={macd_hist:.6f}, BB_pos={bb_pos:.2f}",
                    )

            # Rule-based exit signals
            elif executor.position is not None and rsi is not None:
                sell_signal = rsi > self._rsi_sell and (macd_hist is not None and macd_hist < 0)
                if sell_signal:
                    executor.sell(current.close, current.close_time, "signal")

            executor.update_equity(current.close)

        # Close any remaining position
        if executor.position is not None:
            executor.sell(klines[-1].close, klines[-1].close_time, "backtest_end")

        return self._compute_results(pair, klines, executor)

    def _compute_results(
        self, pair: str, klines: list[Kline], executor: SimulatedExecutor
    ) -> BacktestResult:
        """Compute performance metrics from the executor state."""
        trades = executor.closed_trades
        equity = executor.equity_curve

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = sum(abs(t.pnl) for t in losses)

        # Max drawdown
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe & Sortino
        if len(equity) > 1:
            returns = np.diff(equity) / np.array(equity[:-1])
            annual = np.sqrt(252 * 24 * 60)
            std_r = float(np.std(returns))
            sharpe = float(np.mean(returns) / std_r * annual) if std_r > 0 else 0.0
            downside = returns[returns < 0]
            std_d = float(np.std(downside)) if len(downside) > 0 else 0.0
            sortino = float(np.mean(returns) / std_d * annual) if std_d > 0 else 0.0
        else:
            sharpe = sortino = 0.0

        # Avg hold time
        hold_candles = []
        for t in trades:
            if t.exit_timestamp and t.timestamp:
                hold_candles.append((t.exit_timestamp - t.timestamp) / 60000)

        start_ts = klines[0].open_time if klines else 0
        end_ts = klines[-1].close_time if klines else 0

        from datetime import datetime
        start_date = (
            datetime.fromtimestamp(start_ts / 1000).strftime("%Y-%m-%d")
            if start_ts else ""
        )
        end_date = (
            datetime.fromtimestamp(end_ts / 1000).strftime("%Y-%m-%d")
            if end_ts else ""
        )

        return BacktestResult(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            initial_balance=executor.initial_balance,
            final_balance=executor.balance,
            total_return_pct=(
                (executor.balance - executor.initial_balance) / executor.initial_balance
            ),
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0.0,
            profit_factor=gross_wins / gross_losses if gross_losses > 0 else float("inf"),
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_hold_candles=sum(hold_candles) / len(hold_candles) if hold_candles else 0.0,
            trades=trades,
            equity_curve=equity,
        )


class LLMBacktestEngine:
    """Replays historical klines through the full LLM strategy pipeline.

    Caches LLM responses keyed by a hash of the prompt inputs so that
    repeated runs with the same data don't re-invoke the LLM — useful
    for prompt engineering iterations.
    """

    def __init__(
        self,
        llm,
        *,
        initial_balance: float = 10000.0,
        slippage_pct: float = 0.0005,
        fee_pct: float = 0.001,
        max_position_pct: float = 0.25,
        sl_pct: float = 0.01,
        tp_pct: float = 0.015,
        cache_dir: str | None = None,
    ) -> None:
        self._llm = llm
        self._initial_balance = initial_balance
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct
        self._max_position_pct = max_position_pct
        self._sl_pct = sl_pct
        self._tp_pct = tp_pct
        self._cache: dict[str, dict] = {}
        self._cache_dir = cache_dir

        if cache_dir:
            self._load_cache()

    def _load_cache(self) -> None:
        import json
        from pathlib import Path
        cache_path = Path(self._cache_dir) / "llm_backtest_cache.json"
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
                logger.info("Loaded %d cached LLM responses", len(self._cache))
            except Exception:
                pass

    def _save_cache(self) -> None:
        if not self._cache_dir:
            return
        import json
        from pathlib import Path
        cache_path = Path(self._cache_dir) / "llm_backtest_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(self._cache))

    def _cache_key(self, pair: str, step: int, indicators: dict) -> str:
        import hashlib
        import json
        data = json.dumps({"pair": pair, "step": step, "ind": indicators}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def run(
        self,
        pair: str,
        klines: list[Kline],
        *,
        window_size: int = 100,
        cycle_interval: int = 5,
    ) -> BacktestResult:
        """Run the LLM backtest on historical klines.

        Args:
            pair: Trading pair (e.g. BTCUSDT).
            klines: Historical candlestick data.
            window_size: Number of candles to include in each analysis window.
            cycle_interval: Run LLM every N candles (to avoid excessive API calls).
        """
        from halal_trader.crypto.indicators import compute_all, format_indicators_for_prompt

        executor = SimulatedExecutor(
            initial_balance=self._initial_balance,
            slippage_pct=self._slippage_pct,
            fee_pct=self._fee_pct,
            max_position_pct=self._max_position_pct,
        )

        llm_calls = 0

        for i in range(window_size, len(klines)):
            current = klines[i]

            executor.check_sl_tp(current)

            if (i - window_size) % cycle_interval != 0:
                executor.update_equity(current.close)
                continue

            window = klines[i - window_size : i]
            indicators = compute_all(window)
            if "error" in indicators:
                executor.update_equity(current.close)
                continue

            cache_key = self._cache_key(pair, i, indicators)
            if cache_key in self._cache:
                raw = self._cache[cache_key]
            else:
                balance = executor.balance
                if executor.position:
                    balance += executor.position.quantity * current.close

                prompt = (
                    f"Pair: {pair}\nPrice: ${current.close:,.2f}\n"
                    f"Balance: ${balance:,.2f}\n"
                    f"Position: {'LONG' if executor.position else 'NONE'}\n"
                    f"Indicators:\n{format_indicators_for_prompt(pair, indicators)}\n\n"
                    f"Should I buy, sell, or hold? Respond with JSON: "
                    f'{{"action": "buy"|"sell"|"hold", "confidence": 0.0-1.0, '
                    f'"reasoning": "..."}}'
                )
                system = (
                    "You are a crypto scalping AI analyzing 1-minute candle data. "
                    "Respond ONLY with valid JSON."
                )
                try:
                    raw = await self._llm.generate_json(prompt, system=system)
                    self._cache[cache_key] = raw
                    llm_calls += 1
                except Exception as e:
                    logger.debug("LLM backtest call failed at step %d: %s", i, e)
                    executor.update_equity(current.close)
                    continue

            action = raw.get("action", "hold").lower()
            confidence = float(raw.get("confidence", 0.5))
            reasoning = raw.get("reasoning", "")

            if action == "buy" and executor.position is None and confidence >= 0.5:
                sl = current.close * (1 - self._sl_pct)
                tp = current.close * (1 + self._tp_pct)
                executor.buy(
                    pair, current.close, current.close_time,
                    stop_loss=sl, target_price=tp, reasoning=reasoning,
                )
            elif action == "sell" and executor.position is not None:
                executor.sell(current.close, current.close_time, "llm_sell")

            executor.update_equity(current.close)

        if executor.position is not None:
            executor.sell(klines[-1].close, klines[-1].close_time, "backtest_end")

        self._save_cache()
        logger.info("LLM backtest complete: %d LLM calls, %d cached hits",
                     llm_calls, len(self._cache) - llm_calls)

        result = BacktestEngine(
            initial_balance=self._initial_balance,
            slippage_pct=self._slippage_pct,
            fee_pct=self._fee_pct,
            max_position_pct=self._max_position_pct,
        )._compute_results(pair, klines, executor)
        return result


async def fetch_historical_klines(
    broker, pair: str, interval: str = "1m", limit: int = 1000
) -> list[Kline]:
    """Fetch historical klines from the exchange for backtesting."""
    return await broker.get_klines(pair, interval=interval, limit=min(limit, 1000))
