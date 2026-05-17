"""Tests for :meth:`CryptoPortfolioTracker.format_positions_for_prompt`.

This is the per-position rendering the LLM sees in every cycle prompt:
quantity + entry + unrealized P&L + hold duration. Pure formatting,
no DB / broker calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.db.models import CryptoTrade
from halal_trader.domain.models import CryptoBalance


def _tracker() -> CryptoPortfolioTracker:
    return CryptoPortfolioTracker(
        broker=MagicMock(),
        repo=MagicMock(),
        daily_loss_limit=0.05,
    )


def _trade(
    *,
    pair: str = "BTCUSDT",
    entry_price: float = 40_000.0,
    when: datetime | None = None,
) -> CryptoTrade:
    return CryptoTrade(
        id=1,
        pair=pair,
        side="buy",
        quantity=0.001,
        price=entry_price,
        entry_price=entry_price,
        timestamp=when or datetime.now(UTC),
    )


# ── Empty / sentinel paths ───────────────────────────────────


def test_empty_balances_returns_sentinel():
    out = _tracker().format_positions_for_prompt([])
    assert out == "No open positions."


def test_zero_balance_assets_skipped():
    """A row with free=0 and locked=0 isn't a position — drop it."""
    balances = [CryptoBalance(asset="BTC", free=0.0, locked=0.0)]
    out = _tracker().format_positions_for_prompt(balances)
    assert out == "No open positions."


# ── USDT cash row ────────────────────────────────────────────


def test_usdt_renders_as_cash_line():
    balances = [CryptoBalance(asset="USDT", free=1234.56, locked=10.0)]
    out = _tracker().format_positions_for_prompt(balances)
    assert "USDT (cash)" in out
    assert "1234.56" in out
    assert "10.00" in out  # locked


# ── Configured-pair filter ───────────────────────────────────


def test_only_relevant_assets_rendered_when_configured_pairs_given():
    """Filter limits the prompt to assets we actually trade."""
    balances = [
        CryptoBalance(asset="BTC", free=0.001),
        CryptoBalance(asset="ETH", free=0.01),
        CryptoBalance(asset="DOGE", free=1000.0),  # not configured
    ]
    out = _tracker().format_positions_for_prompt(balances, configured_pairs=["BTCUSDT", "ETHUSDT"])
    assert "BTC" in out
    assert "ETH" in out
    assert "DOGE" not in out


def test_unfiltered_when_no_configured_pairs():
    balances = [
        CryptoBalance(asset="BTC", free=0.001),
        CryptoBalance(asset="DOGE", free=1000.0),
    ]
    out = _tracker().format_positions_for_prompt(balances)
    assert "BTC" in out
    assert "DOGE" in out


# ── With open trades + current prices ────────────────────────


def test_renders_entry_and_unrealized_pnl_with_prices():
    """Entry + current price → unrealized P&L row."""
    balances = [CryptoBalance(asset="BTC", free=0.001)]
    trade = _trade(pair="BTCUSDT", entry_price=40_000.0)
    out = _tracker().format_positions_for_prompt(
        balances,
        open_trades=[trade],
        current_prices={"BTCUSDT": 42_000.0},
    )
    assert "entry: $40,000" in out
    # 0.001 * (42000 - 40000) = $2.00 unrealized
    assert "+$2.00" in out or "$+2.00" in out
    # 5% gain
    assert "+5.0%" in out


def test_renders_negative_pnl_for_losing_position():
    balances = [CryptoBalance(asset="BTC", free=0.001)]
    trade = _trade(pair="BTCUSDT", entry_price=40_000.0)
    out = _tracker().format_positions_for_prompt(
        balances,
        open_trades=[trade],
        current_prices={"BTCUSDT": 38_000.0},
    )
    assert "-$2.00" in out or "$-2.00" in out
    assert "-5.0%" in out


def test_skips_pnl_when_no_current_price():
    """Missing current price → no unrealized line, but entry still shows."""
    balances = [CryptoBalance(asset="BTC", free=0.001)]
    trade = _trade(pair="BTCUSDT", entry_price=40_000.0)
    out = _tracker().format_positions_for_prompt(
        balances,
        open_trades=[trade],
        current_prices={},
    )
    assert "entry:" in out
    assert "unrealized" not in out


# ── Hold duration ────────────────────────────────────────────


def test_held_minutes_formatted_when_under_an_hour():
    balances = [CryptoBalance(asset="BTC", free=0.001)]
    fifteen_minutes_ago = datetime.now(UTC) - timedelta(minutes=15)
    trade = _trade(pair="BTCUSDT", entry_price=40_000.0, when=fifteen_minutes_ago)
    out = _tracker().format_positions_for_prompt(
        balances,
        open_trades=[trade],
        current_prices={"BTCUSDT": 40_000.0},
    )
    # Should include "held: 15m" or similar
    assert "held: 15m" in out


def test_held_hours_formatted_when_over_an_hour():
    balances = [CryptoBalance(asset="BTC", free=0.001)]
    three_hours_ago = datetime.now(UTC) - timedelta(hours=3)
    trade = _trade(pair="BTCUSDT", entry_price=40_000.0, when=three_hours_ago)
    out = _tracker().format_positions_for_prompt(
        balances,
        open_trades=[trade],
        current_prices={"BTCUSDT": 40_000.0},
    )
    assert "held: 3.0h" in out


def test_naive_timestamp_assumed_utc():
    """Defensive: a tz-naive trade timestamp shouldn't crash the format —
    the formatter promotes it to UTC."""
    balances = [CryptoBalance(asset="BTC", free=0.001)]
    naive = datetime.now() - timedelta(minutes=10)  # no tzinfo
    trade = _trade(pair="BTCUSDT", entry_price=40_000.0, when=naive)
    out = _tracker().format_positions_for_prompt(
        balances,
        open_trades=[trade],
        current_prices={"BTCUSDT": 40_000.0},
    )
    assert "held:" in out


# ── Locked balance display when no trade matched ─────────────


def test_locked_only_shows_when_no_trade_match():
    """Lone balance with no matching open-trade row but a locked
    portion → renders the locked qty so the operator sees the
    in-flight order."""
    balances = [CryptoBalance(asset="BTC", free=0.001, locked=0.0005)]
    out = _tracker().format_positions_for_prompt(balances)
    assert "BTC" in out
    assert "locked" in out
