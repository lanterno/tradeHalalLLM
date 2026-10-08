"""Halal Trader - LLM-powered halal day-trading bot."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("halal-trader")  # pyproject's, the one place it is set
except PackageNotFoundError:  # running from a tree that was never installed
    __version__ = "0+unknown"
