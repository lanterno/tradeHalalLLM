"""INVARIANT: an LLM cannot sell a news-momentum position, even when the DB hiccups.

"Slow out": positions opened by the news reactor are closed only by the
monitor's rule-based exits. The lockout used to fail OPEN on a repo error.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from halal_trader.domain.status import EntryType
from halal_trader.trading.executor import TradeExecutor


def _executor(repo: MagicMock) -> TradeExecutor:
    return TradeExecutor(MagicMock(), repo, max_position_pct=0.2, max_simultaneous_positions=5)


async def test_an_open_momentum_position_blocks_llm_sells() -> None:
    repo = MagicMock()
    repo.get_open_trades = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="NVDA", entry_type=EntryType.REACTOR_MOMENTUM, timestamp=None)
        ]
    )

    reason = await _executor(repo)._check_min_hold("NVDA")

    assert reason is not None and "momentum" in reason


async def test_an_unreadable_lockout_refuses_the_sell() -> None:
    repo = MagicMock()
    repo.get_open_trades = AsyncMock(side_effect=ConnectionError("db gone"))

    reason = await _executor(repo)._check_min_hold("NVDA")

    assert reason is not None and "unverifiable" in reason


def test_the_tag_is_the_stored_string() -> None:
    """trades.entry_type holds the string; the enum must keep matching rows
    written before it existed."""
    assert EntryType.REACTOR_MOMENTUM == "reactor_momentum"
