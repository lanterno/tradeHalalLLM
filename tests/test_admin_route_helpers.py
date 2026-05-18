"""Tests for the small helpers in :mod:`web.routes.admin`.

The route handlers themselves need a FastAPI test client + DB; this
file pins the pure pair → base helper used by force-close / cancel-
orders endpoints.
"""

from __future__ import annotations

from halal_trader.web.routes.admin import _crypto_base


def test_strips_usdt_suffix():
    assert _crypto_base("BTCUSDT") == "BTC"


def test_strips_busd_suffix():
    assert _crypto_base("ETHBUSD") == "ETH"


def test_returns_uppercased_input_when_no_suffix():
    """A pair that doesn't end in a known stablecoin returns the
    pair name verbatim (uppercased) — defensive: caller may pass
    a base asset directly."""
    assert _crypto_base("BTC") == "BTC"


def test_normalises_lowercase_input():
    """Operator might paste a lowercased pair from a UI; the helper
    uppercases before matching the suffix."""
    assert _crypto_base("btcusdt") == "BTC"


def test_mixed_case_input():
    assert _crypto_base("BtcUsdt") == "BTC"


def test_empty_string_returns_empty():
    """Defensive: empty pair returns empty (no crash)."""
    assert _crypto_base("") == ""


def test_partial_match_is_kept_intact():
    """`BTCUSD` (no T) doesn't match the USDT/BUSD suffixes, so it's
    returned as-is. Avoids accidentally truncating a mismatched
    suffix."""
    assert _crypto_base("BTCUSD") == "BTCUSD"
