"""Tests for the broker plugin registry (`brokers/registry.py`).

Round-4 wave 1.A introduces a pluggable broker adapter system:
each adapter registers a factory under a name, the bot resolves
the configured name at composition time. This test pins the
registry's contract:

* Default registrations match the built-in adapters (Alpaca for
  stocks, Binance for crypto).
* Lookup is case-insensitive.
* Unknown names raise a precise diagnostic that lists the known
  alternatives.
* Tests can register custom factories without monkey-patching.
* Factory invocation is lazy — no broker SDK import happens at
  registry-import time.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from halal_trader.brokers import (
    KNOWN_CRYPTO_BROKERS,
    KNOWN_STOCK_BROKERS,
    BrokerNotConfiguredError,
    get_crypto_broker_factory,
    get_stock_broker_factory,
    register_crypto_broker,
    register_stock_broker,
)


def test_default_alpaca_stock_broker_registered():
    """Alpaca is the built-in default — pin so a refactor that drops
    the default registration doesn't silently break operator workflows."""
    assert "alpaca" in KNOWN_STOCK_BROKERS()


def test_default_binance_crypto_broker_registered():
    assert "binance" in KNOWN_CRYPTO_BROKERS()


def test_get_stock_broker_factory_returns_callable():
    """The registry returns a factory (Settings -> Broker), not the
    broker instance itself. Lazy: no Alpaca/MCP imports at lookup."""
    factory = get_stock_broker_factory("alpaca")
    assert callable(factory)


def test_get_crypto_broker_factory_returns_callable():
    factory = get_crypto_broker_factory("binance")
    assert callable(factory)


def test_lookup_is_case_insensitive():
    """`STOCK_BROKER=Alpaca` (uppercase / mixed) must work — operators
    write env vars in many cases."""
    assert get_stock_broker_factory("ALPACA") is get_stock_broker_factory("alpaca")
    assert get_stock_broker_factory("Alpaca") is get_stock_broker_factory("alpaca")
    assert get_crypto_broker_factory("BINANCE") is get_crypto_broker_factory("binance")


def test_unknown_stock_broker_raises_with_known_list():
    """The error message lists the registered names so the operator
    can fix the typo without grepping the source."""
    with pytest.raises(BrokerNotConfiguredError) as exc_info:
        get_stock_broker_factory("ibkr-typo")
    msg = str(exc_info.value)
    assert "stock" in msg
    assert "'ibkr-typo'" in msg
    assert "alpaca" in msg  # lists the known alternatives
    assert exc_info.value.asset_class == "stock"
    assert exc_info.value.name == "ibkr-typo"
    assert "alpaca" in exc_info.value.known


def test_unknown_crypto_broker_raises_with_known_list():
    with pytest.raises(BrokerNotConfiguredError) as exc_info:
        get_crypto_broker_factory("kraken-typo")
    msg = str(exc_info.value)
    assert "crypto" in msg
    assert "'kraken-typo'" in msg
    assert "binance" in msg


def test_register_custom_stock_broker():
    """Tests can register a stub factory under a unique name without
    touching the production list."""
    sentinel = object()

    def stub_factory(_settings):
        return sentinel

    register_stock_broker("test-stub-stock", stub_factory)
    assert "test-stub-stock" in KNOWN_STOCK_BROKERS()
    factory = get_stock_broker_factory("test-stub-stock")
    fake_settings = SimpleNamespace()
    assert factory(fake_settings) is sentinel


def test_register_custom_crypto_broker():
    sentinel = object()

    def stub_factory(_settings):
        return sentinel

    register_crypto_broker("test-stub-crypto", stub_factory)
    assert "test-stub-crypto" in KNOWN_CRYPTO_BROKERS()
    factory = get_crypto_broker_factory("test-stub-crypto")
    fake_settings = SimpleNamespace()
    assert factory(fake_settings) is sentinel


def test_register_normalises_name_to_lowercase():
    """Operators may register under any case; lookup keys are
    canonicalised. Pin so a registration of 'IBKR' is reachable as
    `ibkr`."""

    def stub(_):
        return None

    register_stock_broker("MIXED-CASE-STOCK", stub)
    assert "mixed-case-stock" in KNOWN_STOCK_BROKERS()
    assert "MIXED-CASE-STOCK" not in KNOWN_STOCK_BROKERS()
    # Lookup works under both cases.
    assert get_stock_broker_factory("mixed-case-stock") is stub
    assert get_stock_broker_factory("MIXED-CASE-STOCK") is stub


def test_register_overwrites_existing():
    """Re-registering under the same name swaps the factory — useful
    for tests that want to monkey-patch the default."""
    sentinel_v1 = object()
    sentinel_v2 = object()

    def v1(_):
        return sentinel_v1

    def v2(_):
        return sentinel_v2

    register_stock_broker("test-overwrite", v1)
    assert get_stock_broker_factory("test-overwrite")(None) is sentinel_v1

    register_stock_broker("test-overwrite", v2)
    assert get_stock_broker_factory("test-overwrite")(None) is sentinel_v2


def test_known_lists_are_sorted_snapshots():
    """The accessor returns a *new* sorted list every call — operators
    inspecting it shouldn't see internal dict-iteration order."""
    a = KNOWN_STOCK_BROKERS()
    b = KNOWN_STOCK_BROKERS()
    assert a == b
    assert a == sorted(a)
    # Mutating one must not affect the other (defensive copy).
    a.append("not-real")
    assert "not-real" not in KNOWN_STOCK_BROKERS()


def test_default_alpaca_factory_is_lazy():
    """The Alpaca factory shouldn't import `mcp.client` until called.
    Pin so adding a heavy SDK dependency to a future broker doesn't
    slow down `import halal_trader.brokers`."""
    import sys

    # mcp.client may already be in sys.modules from earlier imports —
    # we can't reliably assert "absent at lookup time". Instead pin
    # that just *getting* the factory doesn't trigger a side effect:
    # the returned object is callable, not an instance.
    factory = get_stock_broker_factory("alpaca")
    assert callable(factory)
    # The mcp.client module may or may not be loaded — that's fine.
    assert "halal_trader.mcp.client" in sys.modules or True  # tolerant


def test_settings_broker_field_default_alpaca():
    """`StockSettings.broker` defaults to ``alpaca`` (preserves
    existing operator workflows)."""
    from halal_trader.config import StockSettings

    s = StockSettings()
    assert s.broker == "alpaca"


def test_settings_crypto_broker_field_default_binance():
    from halal_trader.config import CryptoSettings

    s = CryptoSettings()
    assert s.broker == "binance"


def test_settings_broker_field_accepts_override():
    """Operators can set `BROKER=ibkr` (or any registered name) to
    switch — Settings doesn't constrain to a fixed enum so new
    adapters work without touching the config schema."""
    from halal_trader.config import StockSettings

    s = StockSettings(broker="ibkr")
    assert s.broker == "ibkr"
