"""Tests for :class:`crypto.websocket.BinanceWSManager` — buffer + health.

These cover the pure synchronous surface — buffer mutation, kline
parsing, health status — without spinning up a real Binance socket.
The async stream loops (`_combined_kline_stream`, `_kline_stream`) are
deliberately untested here; they're integration-tested at runtime.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.domain.models import Kline


def _make_mgr(symbols: list[str]) -> BinanceWSManager:
    """Construct a manager with a stub client — init doesn't touch it."""
    return BinanceWSManager(MagicMock(), symbols)


def _msg(open_time: int = 1, close: float = 100.0, *, closed: bool = False) -> dict:
    """A Binance kline payload minus the multiplex `data` wrapper."""
    return {
        "e": "kline",
        "s": "BTCUSDT",
        "k": {
            "t": open_time,
            "T": open_time + 60_000,
            "o": close,
            "h": close,
            "l": close,
            "c": str(close),
            "v": "1.0",
            "x": closed,
        },
    }


# ── construction / buffer keys ──────────────────────────────


def test_init_uppercases_buffer_keys():
    """Symbols come in any case; buffers must key on UPPER for lookup."""
    mgr = _make_mgr(["btcusdt", "ETHUSDT"])
    assert set(mgr.buffer_sizes.keys()) == {"BTCUSDT", "ETHUSDT"}


def test_initial_buffers_are_empty():
    mgr = _make_mgr(["btcusdt"])
    assert mgr.buffer_sizes == {"BTCUSDT": 0}


# ── get_klines ──────────────────────────────────────────────


def test_get_klines_unknown_symbol_returns_empty():
    """Unknown symbol returns [] rather than raising — defensive read."""
    mgr = _make_mgr(["btcusdt"])
    assert mgr.get_klines("DOGEUSDT") == []


def test_get_klines_case_insensitive():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(closed=True))
    assert len(mgr.get_klines("btcusdt")) == 1
    assert len(mgr.get_klines("BTCUSDT")) == 1


def test_get_klines_respects_limit():
    """Limit truncates from the *back* (most recent kept)."""
    mgr = _make_mgr(["btcusdt"])
    for i in range(5):
        mgr._process_kline_msg("BTCUSDT", _msg(open_time=i, closed=True))
    out = mgr.get_klines("BTCUSDT", limit=3)
    assert len(out) == 3
    assert [k.open_time for k in out] == [2, 3, 4]


def test_get_klines_limit_above_buffer_returns_all():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(closed=True))
    assert len(mgr.get_klines("BTCUSDT", limit=999)) == 1


# ── get_latest_price ────────────────────────────────────────


def test_get_latest_price_none_until_message():
    mgr = _make_mgr(["btcusdt"])
    assert mgr.get_latest_price("BTCUSDT") is None


def test_get_latest_price_tracks_close():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(close=42_000.5))
    assert mgr.get_latest_price("BTCUSDT") == 42_000.5


def test_get_latest_price_case_insensitive():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(close=99.0))
    assert mgr.get_latest_price("btcusdt") == 99.0


# ── _process_kline_msg buffer semantics ─────────────────────


def test_closed_kline_appended():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(open_time=1, closed=True))
    assert len(mgr.get_klines("BTCUSDT")) == 1


def test_in_progress_kline_replaces_last_when_same_open_time():
    """Receiving an updated tick for the same minute replaces the last
    entry — so the buffer always holds the latest version of an
    in-flight candle, not duplicates."""
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(open_time=1, close=100.0, closed=False))
    mgr._process_kline_msg("BTCUSDT", _msg(open_time=1, close=110.0, closed=False))
    bars = mgr.get_klines("BTCUSDT")
    assert len(bars) == 1
    assert bars[0].close == 110.0


def test_in_progress_kline_appends_when_new_open_time():
    """An in-progress tick with a *new* open_time means the prior minute
    closed without a final 'closed' message — append rather than mutate."""
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(open_time=1, closed=False))
    mgr._process_kline_msg("BTCUSDT", _msg(open_time=2, closed=False))
    assert [k.open_time for k in mgr.get_klines("BTCUSDT")] == [1, 2]


def test_skip_when_event_is_not_kline():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", {"e": "trade", "k": {}})
    assert mgr.get_klines("BTCUSDT") == []


def test_skip_when_kline_dict_missing():
    """Defensive — a malformed message with no `k` key is silently dropped."""
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", {"e": "kline"})
    assert mgr.get_klines("BTCUSDT") == []


def test_skip_when_close_is_zero():
    """A zero close is a Binance edge-case (rare but seen on exchange
    issues) — skip rather than poisoning the buffer."""
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(close=0.0, closed=True))
    assert mgr.get_klines("BTCUSDT") == []
    assert mgr.get_latest_price("BTCUSDT") is None


def test_unknown_symbol_does_not_create_buffer():
    """A message for a symbol the manager wasn't constructed for must
    not silently create a new buffer entry — keeps memory bounded."""
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("DOGEUSDT", _msg(closed=True))
    assert "DOGEUSDT" not in mgr.buffer_sizes


def test_kline_fields_round_trip():
    """The parsed Kline carries the wire fields verbatim."""
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg(open_time=1234, close=100.0, closed=True))
    k = mgr.get_klines("BTCUSDT")[0]
    assert isinstance(k, Kline)
    assert k.open_time == 1234
    assert k.close_time == 1234 + 60_000
    assert k.open == 100.0 and k.close == 100.0


# ── buffer_sizes ────────────────────────────────────────────


def test_buffer_sizes_increment_on_appends():
    mgr = _make_mgr(["btcusdt"])
    for i in range(3):
        mgr._process_kline_msg("BTCUSDT", _msg(open_time=i, closed=True))
    assert mgr.buffer_sizes == {"BTCUSDT": 3}


# ── health_status / check_health ────────────────────────────


def test_health_status_inf_when_never_seen():
    """A symbol that's never received a message has `inf` staleness —
    distinguishable from "0 seconds since the last one"."""
    mgr = _make_mgr(["btcusdt"])
    assert mgr.health_status() == {"BTCUSDT": float("inf")}


def test_health_status_recent_after_message():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg())
    staleness = mgr.health_status()["BTCUSDT"]
    assert staleness < 1.0  # just received


def test_check_health_flags_stale_symbols():
    """A symbol older than the threshold lands in the returned list."""
    mgr = _make_mgr(["btcusdt", "ethusdt"])
    # BTC: just arrived. ETH: pre-date by 200s.
    mgr._process_kline_msg("BTCUSDT", _msg())
    mgr._last_message_time["ETHUSDT"] = time.monotonic() - 200.0
    stale = mgr.check_health(stale_threshold=120.0)
    assert "ETHUSDT" in stale
    assert "BTCUSDT" not in stale


def test_check_health_returns_empty_when_all_fresh():
    mgr = _make_mgr(["btcusdt"])
    mgr._process_kline_msg("BTCUSDT", _msg())
    assert mgr.check_health(stale_threshold=120.0) == []


def test_check_health_treats_never_seen_as_stale():
    """A never-seen symbol has `inf` staleness — must be flagged."""
    mgr = _make_mgr(["btcusdt"])
    assert mgr.check_health(stale_threshold=120.0) == ["BTCUSDT"]
