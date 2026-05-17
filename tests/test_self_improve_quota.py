"""Tests for `TradeSelfReview.review()`'s insufficient_quota handling.

The 1-hour backoff fix from 4db7e21 keeps the bot from re-attempting the
LLM every 5 min when the provider account is out of credits. This pins
that behavior so a future cleanup doesn't accidentally restore the old
loop.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.crypto.self_improve import TradeSelfReview


def _review(llm_error: Exception | None = None) -> TradeSelfReview:
    llm = AsyncMock()
    if llm_error is not None:
        llm.generate_json.side_effect = llm_error

    crypto_trades = AsyncMock()
    crypto_trades.get_completed_round_trips.return_value = [
        {
            "pair": "BTCUSDT",
            "buy_price": 50000,
            "sell_price": 49500,
            "pnl": -10.0,
            "pnl_pct": -0.01,
            "duration_minutes": 5.0,
            "buy_at": "2026-05-17T00:00:00Z",
            "sell_at": "2026-05-17T00:05:00Z",
        }
    ]

    return TradeSelfReview(
        llm=llm,
        strategy_adjustments=MagicMock(),
        crypto_trades=crypto_trades,
    )


@pytest.mark.asyncio
async def test_insufficient_quota_pushes_next_review_an_hour_out():
    """Pin: a review that hits insufficient_quota backs the
    `_last_review_time` an hour past the normal cooldown, so the next
    `should_trigger_review` check returns False for ~1 hour.
    """
    review = _review(
        llm_error=RuntimeError(
            "Error code: 429 - {'message': 'You exceeded your current quota', "
            "'code': 'insufficient_quota'}"
        )
    )

    await review.review(lookback_days=1)

    # After backoff, last_review_time should be > now + cooldown,
    # such that should_trigger_review returns False right away even
    # if a trigger condition fires.
    now = time.monotonic()
    assert review._last_review_time > now + review._review_cooldown


@pytest.mark.asyncio
async def test_generic_error_uses_normal_cooldown():
    """Pin: ordinary errors (not insufficient_quota) keep the normal
    5-min cooldown — they're transient and worth retrying.
    """
    review = _review(llm_error=RuntimeError("Connection reset by peer"))

    before = time.monotonic()
    await review.review(lookback_days=1)
    after = time.monotonic()

    # last_review_time was set to "now" inside review() — should NOT
    # be pushed an hour out. Allow a small slack window for test
    # scheduling jitter.
    assert before - 0.1 <= review._last_review_time <= after + 0.1
