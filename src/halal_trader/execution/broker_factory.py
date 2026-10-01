"""Which Alpaca adapter the bot runs on (``ALPACA_BROKER_ADAPTER``)."""

from __future__ import annotations

from halal_trader.config import Settings
from halal_trader.execution.alpaca_broker import AlpacaRestBroker
from halal_trader.mcp.client import AlpacaMCPClient


def create_broker(settings: Settings) -> AlpacaMCPClient | AlpacaRestBroker:
    """The configured adapter, not yet connected."""
    alpaca = settings.alpaca
    if alpaca.broker_adapter == "rest":
        return AlpacaRestBroker(alpaca.api_key, alpaca.secret_key, paper=alpaca.paper_trade)
    return AlpacaMCPClient()
