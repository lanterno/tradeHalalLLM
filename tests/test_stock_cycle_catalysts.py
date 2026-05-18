"""Stock cycle's ``_extract_latest_prices`` helper.

The catalyst-feed wrapper tests previously lived here; that wiring is
now driven by ``BuildCatalystsStage`` and tested in
``test_cycle_stages.py``. What remains here is the snapshot →
``{symbol: price}`` walker used by the shadow runner.
"""

from __future__ import annotations

from halal_trader.trading.cycle import _extract_latest_prices

# ── _extract_latest_prices (shadow runner price feed) ─────────────


def test_extract_prices_flat_dict():
    snaps = {"AAPL": {"latestTrade": {"p": 150.5}}}
    assert _extract_latest_prices(snaps) == {"AAPL": 150.5}


def test_extract_prices_alt_keys():
    snaps = {
        "MSFT": {"latest_trade": {"price": 410.25}},
        "GOOG": {"trade": {"p": 175.0}},
    }
    out = _extract_latest_prices(snaps)
    assert out == {"MSFT": 410.25, "GOOG": 175.0}


def test_extract_prices_nested_by_symbol():
    snaps = {"AAPL": {"AAPL": {"latestTrade": {"p": 200.0}}}}
    assert _extract_latest_prices(snaps) == {"AAPL": 200.0}


def test_extract_prices_drops_unparseable_entries():
    snaps = {
        "AAPL": {"latestTrade": {"p": "not-a-number"}},
        "MSFT": "not-a-dict",
        "TSLA": {"latestTrade": {"p": 240.5}},
    }
    assert _extract_latest_prices(snaps) == {"TSLA": 240.5}
