"""Tests for the prompt-block formatters in :mod:`trading.strategy`.

These three functions (``_format_positions``, ``_format_snapshots``,
``_format_bars``) shape what the LLM sees each cycle. They have no
side effects and are pure dict-walkers — easy to lock in.
"""

from unittest.mock import MagicMock

from halal_trader.trading.strategy import (
    _format_bars,
    _format_positions,
    _format_snapshots,
)

# ── _format_positions ──────────────────────────────────────────


def test_format_positions_empty_returns_sentinel():
    assert _format_positions([]) == "No open positions."


def test_format_positions_renders_per_position_line():
    p = MagicMock(
        symbol="AAPL",
        qty=10,
        avg_entry_price=180.0,
        current_price=182.5,
        unrealized_pl=25.0,
        unrealized_plpc=0.0139,
    )
    out = _format_positions([p])
    assert "AAPL" in out
    assert "180.00" in out
    assert "182.50" in out
    assert "+$25.00" in out or "$+25.00" in out


def test_format_positions_negative_plpc_renders_minus_sign():
    """Sign is meaningful — the prompt uses it to nudge the LLM."""
    p = MagicMock(
        symbol="AAPL",
        qty=10,
        avg_entry_price=180.0,
        current_price=170.0,
        unrealized_pl=-100.0,
        unrealized_plpc=-0.0556,
    )
    out = _format_positions([p])
    assert "-$100.00" in out or "$-100.00" in out
    assert "-5" in out  # -5.56%


# ── _format_snapshots / _format_bars, on RECORDED live payloads ─────
#
# The old fixtures used a snake_case shape (latest_trade.price, ...) that the
# live server never sends, which is why the prompt showed "Price=$N/A" for
# every symbol while these tests stayed green. The fixtures below are real
# alpaca-mcp-server 2.3.2 responses (tests/fixtures/alpaca_mcp_2_3_2/).

import json  # noqa: E402
from pathlib import Path  # noqa: E402

_FIXTURES = Path(__file__).parent / "fixtures" / "alpaca_mcp_2_3_2"


def _recorded(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text())


def test_format_snapshots_empty_returns_sentinel():
    assert _format_snapshots({}) == "No snapshot data available."


def test_format_snapshots_reads_the_live_shape():
    out = _format_snapshots({"AAPL": _recorded("snapshot_AAPL.json")})
    assert "N/A" not in out
    assert "last $329.17" in out
    assert "-1.21% vs prev close $333.21" in out
    assert "bid $329.13 ask $329.19" in out


def test_format_snapshots_survives_a_data_envelope():
    out = _format_snapshots({"AAPL": {"data": _recorded("snapshot_AAPL.json")}})
    assert "last $329.17" in out


def test_format_snapshots_says_so_when_there_is_no_price():
    out = _format_snapshots({"AAPL": {"AAPL": {"latestQuote": {"bp": 1, "ap": 2}}}})
    assert out == "  AAPL: no last price in snapshot"


def test_format_bars_empty_returns_sentinel():
    assert _format_bars({}) == "No bar data available."


def test_format_bars_renders_the_last_five_dated_bars_from_the_live_shape():
    out = _format_bars({"AAPL": _recorded("bars_AAPL.json")})
    lines = out.splitlines()
    assert lines[0] == "  AAPL:"
    assert len(lines) == 6  # header + 5 bars, not the whole 60-day envelope
    assert lines[-1].startswith("    2026-10-01: O=330.00")
    assert "{" not in out  # no raw dict dump
