"""Tests for the pure-ish helpers in :mod:`crypto.exchange`.

`test_extract_fill_price.py` covers `extract_fill_price`. This file
pins the small surface around symbol-filter parsing + lot-size /
tick-size rounding that's only reached today through the executor's
order-placement path.

We construct a `BinanceClient` without calling `connect()` — the
helpers don't touch the underlying SDK, so the unconnected client is
fine for testing pure math.
"""

from __future__ import annotations

import pytest

from halal_trader.crypto.exchange import BinanceClient, SymbolFilter


def _client(filters: dict[str, SymbolFilter] | None = None) -> BinanceClient:
    """Construct a non-connected client with optional pre-loaded filters."""
    c = BinanceClient(api_key="x", secret_key="y", testnet=True)
    if filters:
        c._symbol_filters = filters
    return c


async def test_connect_raises_on_empty_credentials():
    """An unconfigured client must fail fast at connect() rather than let the
    bot dead-loop a signed-endpoint failure (cycle.failed + alert) every 60s.
    See crypto.exchange.connect."""
    c = BinanceClient(api_key="", secret_key="", testnet=True)
    with pytest.raises(RuntimeError, match="credentials"):
        await c.connect()


def _filter(
    *,
    min_qty: float = 0.00001,
    max_qty: float = 9000.0,
    step_size: float = 0.00001,
    min_notional: float = 10.0,
    tick_size: float = 0.01,
) -> SymbolFilter:
    return SymbolFilter(
        min_qty=min_qty,
        max_qty=max_qty,
        step_size=step_size,
        min_notional=min_notional,
        tick_size=tick_size,
        base_asset_precision=8,
        quote_asset_precision=8,
    )


# ── _parse_symbol_filters ──────────────────────────────────


def test_parse_symbol_filters_full_payload():
    """Real-shaped Binance ``get_symbol_info`` response — all three
    filter types present."""
    info = {
        "baseAssetPrecision": 8,
        "quoteAssetPrecision": 8,
        "filters": [
            {
                "filterType": "LOT_SIZE",
                "minQty": "0.00001",
                "maxQty": "9000.00",
                "stepSize": "0.00001",
            },
            {"filterType": "MIN_NOTIONAL", "minNotional": "10.00"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        ],
    }
    sf = BinanceClient._parse_symbol_filters(info)
    assert sf is not None
    assert sf.min_qty == 0.00001
    assert sf.max_qty == 9000.0
    assert sf.step_size == 0.00001
    assert sf.min_notional == 10.0
    assert sf.tick_size == 0.01


def test_parse_symbol_filters_returns_none_when_no_lot_size():
    """``step_size <= 0`` (i.e. no LOT_SIZE filter) → return None.
    Without LOT_SIZE we can't safely round quantities, so the executor
    will refuse to place orders for the symbol. Pin this defensive
    return so downstream code can branch on it."""
    info = {"filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}]}
    sf = BinanceClient._parse_symbol_filters(info)
    assert sf is None


def test_parse_symbol_filters_accepts_notional_alias():
    """Newer Binance API uses ``NOTIONAL``; older returns ``MIN_NOTIONAL``.
    Both must populate `min_notional`."""
    info_new = {
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0", "maxQty": "9000"},
            {"filterType": "NOTIONAL", "minNotional": "20.00"},
        ]
    }
    info_old = {
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0", "maxQty": "9000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "20.00"},
        ]
    }
    assert BinanceClient._parse_symbol_filters(info_new).min_notional == 20.0
    assert BinanceClient._parse_symbol_filters(info_old).min_notional == 20.0


def test_parse_symbol_filters_uses_defaults_for_missing_blocks():
    """LOT_SIZE present but the others missing → use the in-code
    defaults (`min_notional=5.0`, `tick_size=0.01`)."""
    info = {
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0", "maxQty": "100"},
        ]
    }
    sf = BinanceClient._parse_symbol_filters(info)
    assert sf is not None
    assert sf.min_notional == 5.0  # default
    assert sf.tick_size == 0.01  # default


def test_parse_symbol_filters_uses_default_precisions():
    """Missing precision fields → 8 (Binance's typical default)."""
    info = {
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0", "maxQty": "100"},
        ]
    }
    sf = BinanceClient._parse_symbol_filters(info)
    assert sf.base_asset_precision == 8
    assert sf.quote_asset_precision == 8


def test_parse_symbol_filters_no_filters_at_all_returns_none():
    """Defensive: empty filters list → None (no LOT_SIZE step)."""
    assert BinanceClient._parse_symbol_filters({"filters": []}) is None


def test_parse_symbol_filters_handles_string_inputs():
    """All numeric fields come from JSON as strings — pin the cast."""
    info = {
        "filters": [
            {
                "filterType": "LOT_SIZE",
                "minQty": "1.5",
                "maxQty": "200",
                "stepSize": "0.5",
            },
        ]
    }
    sf = BinanceClient._parse_symbol_filters(info)
    assert isinstance(sf.min_qty, float)
    assert sf.step_size == 0.5


# ── round_quantity ─────────────────────────────────────────


def test_round_quantity_no_filter_passes_through():
    """An unknown symbol has no filter cached — return the qty as-is
    (caller decides how to handle the missing filter; we don't truncate)."""
    c = _client()
    assert c.round_quantity("UNKNOWN", 1.234567) == 1.234567


def test_round_quantity_floors_to_step_size():
    """Step 0.001 → precision 3 → floor to 3 decimals (Binance's
    semantics: never round up past a step boundary, that would
    overspend)."""
    c = _client(filters={"BTCUSDT": _filter(step_size=0.001, min_qty=0, max_qty=9000)})
    assert c.round_quantity("BTCUSDT", 0.123456) == 0.123  # floored


def test_round_quantity_clamps_to_min_qty():
    """A qty below `min_qty` gets bumped up — Binance rejects below-min
    orders, but the caller may want to retry with min_qty."""
    c = _client(filters={"BTCUSDT": _filter(step_size=0.001, min_qty=0.5, max_qty=9000)})
    assert c.round_quantity("BTCUSDT", 0.001) == 0.5


def test_round_quantity_clamps_to_max_qty():
    """A qty above `max_qty` gets capped — defensive against
    a hallucinated giant size from the LLM."""
    c = _client(filters={"BTCUSDT": _filter(step_size=0.001, min_qty=0, max_qty=10)})
    assert c.round_quantity("BTCUSDT", 99999) == 10


def test_round_quantity_case_insensitive_symbol():
    """Filters are keyed UPPER but caller may pass any case."""
    c = _client(filters={"BTCUSDT": _filter(step_size=0.001, min_qty=0, max_qty=9000)})
    assert c.round_quantity("btcusdt", 0.5) == 0.5
    assert c.round_quantity("BtcUsdt", 0.5) == 0.5


def test_round_quantity_zero_step_passes_through():
    """Defensive: step_size=0 (a malformed filter) → return qty as-is
    rather than divide by zero in the `log10`."""
    c = _client(filters={"BTCUSDT": _filter(step_size=0.0, min_qty=0, max_qty=100)})
    assert c.round_quantity("BTCUSDT", 1.5) == 1.5


# ── round_price ────────────────────────────────────────────


def test_round_price_no_filter_passes_through():
    c = _client()
    assert c.round_price("UNKNOWN", 100.123456) == 100.123456


def test_round_price_rounds_to_tick_size():
    """Tick 0.01 → 2 decimals (cent precision for USD-quoted markets)."""
    c = _client(filters={"BTCUSDT": _filter(tick_size=0.01)})
    assert c.round_price("BTCUSDT", 100.456789) == 100.46


def test_round_price_finer_tick():
    """A symbol with a 0.0001 tick (some BUSD pairs) → 4 decimals."""
    c = _client(filters={"BTCUSDT": _filter(tick_size=0.0001)})
    assert c.round_price("BTCUSDT", 0.12345678) == round(0.12345678, 4)


def test_round_price_zero_tick_passes_through():
    c = _client(filters={"BTCUSDT": _filter(tick_size=0.0)})
    assert c.round_price("BTCUSDT", 100.123456) == 100.123456


def test_round_price_case_insensitive():
    c = _client(filters={"BTCUSDT": _filter(tick_size=0.01)})
    assert c.round_price("btcusdt", 100.456789) == 100.46


# ── get_symbol_filter / get_cached_price ───────────────────


def test_get_symbol_filter_returns_none_for_unknown():
    c = _client()
    assert c.get_symbol_filter("UNKNOWN") is None


def test_get_symbol_filter_case_insensitive():
    flt = _filter()
    c = _client(filters={"BTCUSDT": flt})
    assert c.get_symbol_filter("btcusdt") is flt
    assert c.get_symbol_filter("BTCUSDT") is flt


def test_get_cached_price_returns_none_when_missing():
    c = _client()
    assert c.get_cached_price("BTCUSDT") is None


def test_get_cached_price_case_insensitive():
    c = _client()
    c._latest_price_cache["BTCUSDT"] = 50_000.0
    assert c.get_cached_price("btcusdt") == 50_000.0
    assert c.get_cached_price("BTCUSDT") == 50_000.0


# ── format_filters_for_prompt ──────────────────────────────


def test_format_filters_for_prompt_empty_returns_sentinel():
    """No symbols loaded → human-readable sentinel rather than empty."""
    c = _client()
    out = c.format_filters_for_prompt()
    assert "No exchange trading rules" in out


def test_format_filters_for_prompt_lists_each_symbol():
    """Each symbol gets one line with its filter values rendered."""
    c = _client(
        filters={
            "BTCUSDT": _filter(min_qty=0.00001, step_size=0.00001, min_notional=10, tick_size=0.01),
            "ETHUSDT": _filter(min_qty=0.0001, step_size=0.0001, min_notional=10, tick_size=0.01),
        }
    )
    out = c.format_filters_for_prompt()
    assert "BTCUSDT" in out
    assert "ETHUSDT" in out
    assert "$10.00" in out  # min_notional formatted as currency


def test_format_filters_for_prompt_sorted_alphabetically():
    """Output is sorted by symbol — pin so the same prompt always
    renders the same shape (cache-friendliness, deterministic
    LlmDecision rows)."""
    c = _client(
        filters={
            "ZRXUSDT": _filter(),
            "ATOMUSDT": _filter(),
            "BTCUSDT": _filter(),
        }
    )
    out = c.format_filters_for_prompt()
    a_pos = out.index("ATOMUSDT")
    b_pos = out.index("BTCUSDT")
    z_pos = out.index("ZRXUSDT")
    assert a_pos < b_pos < z_pos


# ── invalidate_account_cache ───────────────────────────────


def test_invalidate_account_cache_clears_state():
    """Pin the cache-clear behaviour — after a buy, the executor calls
    this to ensure the next `get_account()` reflects the new balance."""
    c = _client()
    c._account_cache = (123.0, "fake-account")  # type: ignore[assignment]
    c.invalidate_account_cache()
    assert c._account_cache is None
