"""Broker registry — name → factory map for stock + crypto adapters.

Adapters are registered with a callable that builds them from
``Settings`` rather than instantiated eagerly. The factory is a
``Callable[[Settings], Broker | CryptoBroker]``; the trading bot calls
the factory at composition time. This means:

* The registry stays lightweight — no broker SDK imports happen until
  the operator picks an adapter.
* Tests can register a stub factory without monkey-patching the
  production list (use a unique name).
* Configuration drift (typo in env var) raises a precise
  :class:`BrokerNotConfiguredError` at startup, not silently mid-cycle.

Default registrations live in this file's ``_register_defaults`` so
the existing operator workflow keeps working: ``STOCK_BROKER=alpaca``
maps to :class:`mcp.client.AlpacaMCPClient`, ``CRYPTO_BROKER=binance``
maps to :class:`crypto.exchange.BinanceClient`. Anything else has to
be added explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from halal_trader.config import Settings
    from halal_trader.domain.ports import Broker, CryptoBroker


StockBrokerFactory = Callable[["Settings"], "Broker"]
CryptoBrokerFactory = Callable[["Settings"], "CryptoBroker"]


class BrokerNotConfiguredError(RuntimeError):
    """Raised when an unknown broker name is requested.

    The error message lists the registered names so the operator can
    fix the typo without grepping the source.
    """

    def __init__(self, asset_class: str, name: str, known: list[str]) -> None:
        super().__init__(
            f"{asset_class} broker {name!r} is not registered. "
            f"Known broker names: {', '.join(sorted(known)) if known else '(none)'}."
        )
        self.asset_class = asset_class
        self.name = name
        self.known = sorted(known)


_STOCK_BROKERS: dict[str, StockBrokerFactory] = {}
_CRYPTO_BROKERS: dict[str, CryptoBrokerFactory] = {}


def register_stock_broker(name: str, factory: StockBrokerFactory) -> None:
    """Register a stock broker factory under ``name`` (case-insensitive)."""
    _STOCK_BROKERS[name.lower()] = factory


def register_crypto_broker(name: str, factory: CryptoBrokerFactory) -> None:
    """Register a crypto broker factory under ``name`` (case-insensitive)."""
    _CRYPTO_BROKERS[name.lower()] = factory


def get_stock_broker_factory(name: str) -> StockBrokerFactory:
    """Resolve a stock broker name to its factory.

    Raises :class:`BrokerNotConfiguredError` when the name is unknown.
    """
    key = name.lower()
    if key not in _STOCK_BROKERS:
        raise BrokerNotConfiguredError("stock", name, list(_STOCK_BROKERS))
    return _STOCK_BROKERS[key]


def get_crypto_broker_factory(name: str) -> CryptoBrokerFactory:
    """Resolve a crypto broker name to its factory.

    Raises :class:`BrokerNotConfiguredError` when the name is unknown.
    """
    key = name.lower()
    if key not in _CRYPTO_BROKERS:
        raise BrokerNotConfiguredError("crypto", name, list(_CRYPTO_BROKERS))
    return _CRYPTO_BROKERS[key]


def KNOWN_STOCK_BROKERS() -> list[str]:  # noqa: N802 - constant-style accessor
    """Snapshot of registered stock broker names (sorted)."""
    return sorted(_STOCK_BROKERS)


def KNOWN_CRYPTO_BROKERS() -> list[str]:  # noqa: N802 - constant-style accessor
    """Snapshot of registered crypto broker names (sorted)."""
    return sorted(_CRYPTO_BROKERS)


# ── Default registrations ─────────────────────────────────────


def _alpaca_factory(_settings: "Settings") -> Any:
    from halal_trader.mcp.client import AlpacaMCPClient

    return AlpacaMCPClient()


def _binance_factory(settings: "Settings") -> Any:
    from halal_trader.crypto.exchange import BinanceClient

    return BinanceClient(
        api_key=settings.binance.api_key,
        secret_key=settings.binance.secret_key,
        testnet=settings.binance.testnet,
        configured_pairs=settings.crypto.pairs,
    )


def _register_defaults() -> None:
    """Register the adapters that ship with the bot today.

    Idempotent — calling twice is a no-op. The factories use lazy
    imports so adding a new default doesn't pull in heavy SDKs at
    package-import time.

    Round-4 wave 1.A also ships the SDK-free ``paper`` adapters so
    a fresh checkout can run without configuring real broker keys
    (onboarding + CI integration tests + scholar / auditor demos).
    """
    register_stock_broker("alpaca", _alpaca_factory)
    register_crypto_broker("binance", _binance_factory)

    # Paper-trading adapters — pure Python, no SDK.
    from halal_trader.brokers.paper import _paper_crypto_factory, _paper_stock_factory

    register_stock_broker("paper", _paper_stock_factory)
    register_crypto_broker("paper", _paper_crypto_factory)


_register_defaults()
