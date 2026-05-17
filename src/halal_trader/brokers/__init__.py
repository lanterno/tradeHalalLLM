"""Broker plugin registry — pluggable adapters per asset class.

Round-4 wave 1.A introduces this package. Existing adapters
(``mcp.client.AlpacaMCPClient`` for stocks, ``crypto.exchange.BinanceClient``
for crypto) keep working unchanged; the registry is additive — operators
opt into new adapters by name in ``Settings``.

Adding a new adapter is three steps:

1. Implement the relevant ``domain.ports`` Protocol (``Broker`` for
   stocks, ``CryptoBroker`` for crypto).
2. Register the implementation under a stable name via
   :func:`register_stock_broker` or :func:`register_crypto_broker`.
3. Set ``STOCK_BROKER=<name>`` (default: ``alpaca``) or
   ``CRYPTO_BROKER=<name>`` (default: ``binance``) in ``.env``.

The registry is lazy — each adapter's heavy imports (e.g. ``ib_insync``
for IBKR) are deferred to first lookup. Tests can register stub
adapters via the public API without touching the production list.
"""

from __future__ import annotations

from halal_trader.brokers.aggregator import (
    AggregatedPortfolio,
    BrokerHealth,
    PortfolioAggregator,
)
from halal_trader.brokers.registry import (
    KNOWN_CRYPTO_BROKERS,
    KNOWN_STOCK_BROKERS,
    BrokerNotConfiguredError,
    get_crypto_broker_factory,
    get_stock_broker_factory,
    register_crypto_broker,
    register_stock_broker,
)

__all__ = [
    "AggregatedPortfolio",
    "BrokerHealth",
    "BrokerNotConfiguredError",
    "KNOWN_CRYPTO_BROKERS",
    "KNOWN_STOCK_BROKERS",
    "PortfolioAggregator",
    "get_crypto_broker_factory",
    "get_stock_broker_factory",
    "register_crypto_broker",
    "register_stock_broker",
]
