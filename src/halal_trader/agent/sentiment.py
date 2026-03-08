"""Backward-compatible re-export — sentiment moved to halal_trader.trading.sentiment."""

from halal_trader.trading.sentiment import SentimentAnalyzer  # noqa: F401

__all__ = ["SentimentAnalyzer"]
