"""Backward-compatible re-export — strategy moved to halal_trader.trading.strategy."""

from halal_trader.trading.strategy import TradingStrategy  # noqa: F401

__all__ = ["TradingStrategy"]
