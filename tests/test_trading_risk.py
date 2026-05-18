"""Tests for the stock portfolio-risk adapter (trading/risk.py)."""

from __future__ import annotations

from halal_trader.config import Settings
from halal_trader.domain.models import Position
from halal_trader.trading.bars import bars_to_klines
from halal_trader.trading.risk import evaluate_stock_risk


def _bar(o: float, h: float, low: float, c: float, v: float = 1_000.0) -> dict:
    return {"o": o, "h": h, "l": low, "c": c, "v": v}


def _series(start: float, n: int, step: float) -> list[dict]:
    out = []
    price = start
    for _ in range(n):
        out.append(_bar(price, price + 0.5, price - 0.5, price + step))
        price += step
    return out


def _settings() -> Settings:
    """Settings stub for the stocks-side risk adapter.

    Round-4 wave 0.C moved the risk knobs from CryptoSettings to
    StockSettings — pydantic's `extra="ignore"` swallowed the legacy
    `crypto_*` kwargs silently before, so the test was using defaults
    by accident. Now we construct StockSettings explicitly with the
    intended values."""
    from halal_trader.config import StockSettings

    return Settings(
        stocks=StockSettings(
            max_position_pct=0.20,
            max_portfolio_heat_pct=0.05,
            max_drawdown_pct=0.08,
            high_correlation_threshold=0.7,
            correlation_reduction_factor=0.5,
            atr_baseline=0.02,
        ),
    )


def test_bars_to_klines_handles_list_form():
    raw = [_bar(100, 101, 99, 100.5) for _ in range(5)]
    klines = bars_to_klines(raw)
    assert len(klines) == 5
    assert klines[0].close == 100.5


def test_bars_to_klines_handles_alpaca_dict_form():
    raw = {"bars": [_bar(50, 51, 49, 50.5) for _ in range(3)]}
    klines = bars_to_klines(raw)
    assert len(klines) == 3
    assert klines[-1].close == 50.5


def test_bars_to_klines_skips_garbage_entries():
    raw = [_bar(1, 2, 0.5, 1.5), "not a dict", {}, _bar(2, 3, 1, 2.5)]
    klines = bars_to_klines(raw)
    assert len(klines) == 2  # garbage rows filtered


def test_bars_to_klines_skips_zero_close():
    raw = [_bar(1, 2, 0.5, 0)]
    assert bars_to_klines(raw) == []


def test_evaluate_stock_risk_clean_portfolio():
    bars = {
        "AAPL": _series(180.0, 50, 0.5),
        "MSFT": _series(420.0, 50, 0.4),
    }
    positions: list[Position] = []
    out = evaluate_stock_risk(
        settings=_settings(),
        bars_by_symbol=bars,
        positions=positions,
        total_equity=100_000,
    )
    assert out.state.is_halted is False
    assert out.risk_text  # non-empty (correlation should be present)


def test_evaluate_stock_risk_drawdown_halt():
    bars = {"AAPL": _series(100.0, 50, 0.0)}
    settings = _settings()
    # Equity has crashed below the (synthetic) peak the engine sees.
    out = evaluate_stock_risk(
        settings=settings,
        bars_by_symbol=bars,
        positions=[],
        total_equity=100_000,
    )
    assert out.state.is_halted is False  # first call sets the peak

    # Second call with a much lower equity should trip the drawdown.
    # The engine state is local to each call (new instance), so we
    # construct one and prime its peak via two calls.
    from halal_trader.crypto.risk import PortfolioRiskEngine

    engine = PortfolioRiskEngine(
        base_max_position_pct=0.20,
        max_portfolio_heat_pct=0.05,
        max_drawdown_pct=0.08,
        high_correlation_threshold=0.7,
        correlation_reduction_factor=0.5,
        atr_baseline=0.02,
    )
    engine.evaluate({}, {}, {}, {}, total_equity=200_000)
    state = engine.evaluate({}, {}, {}, {}, total_equity=180_000)  # 10% drawdown
    assert state.is_halted
    assert "Drawdown" in state.halt_reason


def test_evaluate_stock_risk_handles_empty_bars():
    out = evaluate_stock_risk(
        settings=_settings(),
        bars_by_symbol={},
        positions=[],
        total_equity=100_000,
    )
    assert out.state.is_halted is False
    # No bars → no correlations → no risk text (empty is acceptable).
    assert isinstance(out.risk_text, str)


def test_evaluate_stock_risk_with_open_positions():
    bars = {
        "AAPL": _series(180.0, 50, 0.5),
        "MSFT": _series(420.0, 50, 0.4),
    }
    positions = [
        Position(
            symbol="AAPL",
            qty=10,
            avg_entry_price=180.0,
            current_price=185.0,
            unrealized_pl=50.0,
            unrealized_plpc=0.028,
        )
    ]
    out = evaluate_stock_risk(
        settings=_settings(),
        bars_by_symbol=bars,
        positions=positions,
        total_equity=10_000,
    )
    # Heat is positive so no halt; the engine should still compute correlations.
    assert out.state.is_halted is False
    assert out.state.portfolio_heat == 50.0
