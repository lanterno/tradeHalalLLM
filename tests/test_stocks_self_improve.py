"""Tests for the stocks-side :class:`StockTradeSelfReview`.

Mirrors ``test_self_improve_*.py`` on the crypto side. The
asset-agnostic orchestration (cooldown, exec-failure tracking,
prompt assembly, parse/clamp/apply) is exercised by the crypto
suite against the base class; this file pins the stocks-specific
bits: knob menu shape, repo wiring, prompt asset label, and the
stocks-only ``daily_loss_limit`` knob clamping correctly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.self_improve import (
    ReviewResult,
    StrategyAdjustment,
    TradeSelfReviewBase,
)
from halal_trader.trading.self_improve import (
    _STOCK_SAFE_BOUNDS,
    _STOCK_STRATEGY_PARAM_MAP,
    StockTradeSelfReview,
)


def _review(
    *,
    llm_response: dict | None = None,
    llm_error: Exception | None = None,
    strategy: object | None = None,
) -> tuple[StockTradeSelfReview, MagicMock, MagicMock, MagicMock]:
    """Build a review with mocked repos + LLM. Returns (review, llm,
    strategy_adjustments_repo, trades_repo) so each test can assert
    on the seams it cares about."""
    llm = MagicMock()
    if llm_error is not None:
        llm.generate_json = AsyncMock(side_effect=llm_error)
    else:
        llm.generate_json = AsyncMock(
            return_value=llm_response
            or {
                "observations": [],
                "parameter_adjustments": {},
                "pairs_to_avoid": [],
                "strategy_notes": "",
            }
        )
    strategy_adjustments = MagicMock()
    strategy_adjustments.get_latest_strategy_adjustments = AsyncMock(return_value={})
    strategy_adjustments.record_strategy_adjustment = AsyncMock()
    trades = MagicMock()
    trades.get_completed_stock_round_trips = AsyncMock(return_value=[])

    review = StockTradeSelfReview(
        llm=llm,
        strategy_adjustments=strategy_adjustments,
        trades=trades,
        strategy=strategy,
    )
    return review, llm, strategy_adjustments, trades


# ── Class-attr config ─────────────────────────────────────────


def test_class_attrs_match_stocks_menu():
    """Pin: only 2 knobs, both tied to the strategy's existing instance
    attributes. Adding a third knob requires (1) a new bounds entry,
    (2) a new STRATEGY_PARAM_MAP entry, AND (3) updating the JSON
    schema in the system prompt — all three must stay in sync."""
    assert set(_STOCK_SAFE_BOUNDS) == {"max_position_pct", "daily_loss_limit"}
    assert set(_STOCK_STRATEGY_PARAM_MAP) == {"max_position_pct", "daily_loss_limit"}
    # Strategy attribute names match the leading-underscore convention
    # the base class's setattr loop expects.
    assert _STOCK_STRATEGY_PARAM_MAP["max_position_pct"] == "_max_position_pct"
    assert _STOCK_STRATEGY_PARAM_MAP["daily_loss_limit"] == "_daily_loss_limit"


def test_subclass_inherits_base_orchestration():
    """Sanity: the stocks subclass is-a base class. The asset-agnostic
    methods (load_from_db, format_adjustments_for_prompt, etc.) live
    on the base; this is the pin that they remain reachable."""
    assert issubclass(StockTradeSelfReview, TradeSelfReviewBase)


def test_system_prompt_mentions_stock_trading():
    """The system prompt must say "stock" — the LLM uses the asset
    label to anchor its review (mentioning the wrong asset would
    produce nonsense suggestions like "RSI threshold" knobs that
    don't exist on the stocks strategy)."""
    review, *_ = _review()
    assert "stock trading decisions" in review._SYSTEM_PROMPT
    # And the JSON schema only lists the 2 stocks knobs — no crypto
    # leftovers like rsi_buy_threshold or stop_loss_pct.
    assert "max_position_pct" in review._SYSTEM_PROMPT
    assert "daily_loss_limit" in review._SYSTEM_PROMPT
    assert "rsi_buy_threshold" not in review._SYSTEM_PROMPT
    assert "stop_loss_pct" not in review._SYSTEM_PROMPT


# ── Repo wiring ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_round_trips_calls_stock_repo_method():
    """The subclass dispatches to ``TradeRepo.get_completed_stock_round_trips``
    — NOT to ``CryptoTradeRepo.get_completed_round_trips``. Wrong
    repo method = silently empty review for stocks."""
    review, _, _, trades = _review()
    await review._fetch_round_trips(limit=10, lookback_days=1)
    trades.get_completed_stock_round_trips.assert_awaited_once_with(limit=10, lookback_days=1)


@pytest.mark.asyncio
async def test_load_from_db_filters_to_stocks_safe_bounds():
    """A leftover crypto adjustment row in the DB (e.g. ``stop_loss_pct``)
    must NOT load onto the stocks review state — the param-name space
    is asset-specific and feeding a crypto knob to a stocks strategy
    would either no-op or worse, write to the wrong attribute."""
    review, _, strategy_adjustments, _ = _review()
    # Simulate a DB with mixed crypto + stocks rows (in practice the
    # query is scoped, but the filter is defense-in-depth).
    strategy_adjustments.get_latest_strategy_adjustments = AsyncMock(
        return_value={
            "max_position_pct": 0.22,  # stocks knob — keep
            "daily_loss_limit": 0.025,  # stocks knob — keep
            "stop_loss_pct": 0.008,  # crypto-only — DROP
            "rsi_buy_threshold": 35.0,  # crypto-only — DROP
        }
    )
    await review.load_from_db()
    assert set(review._active_adjustments) == {"max_position_pct", "daily_loss_limit"}
    assert review._active_adjustments["max_position_pct"] == 0.22
    assert review._active_adjustments["daily_loss_limit"] == 0.025


# ── Bounds clamping (stocks-specific values) ─────────────────


@pytest.mark.asyncio
async def test_review_clamps_max_position_pct_to_stocks_bounds():
    """LLM suggests an out-of-bounds value; ``_parse_review`` clamps
    to the stocks-side (0.05, 0.30) window — NOT the crypto window
    (0.10, 0.30). Subtle: a 0.07 suggestion is valid for stocks
    (in [0.05, 0.30]) but would clamp to 0.10 on the crypto side."""
    trades_response = [
        {
            "pair": "AAPL",
            "buy_price": 180.0,
            "sell_price": 175.0,
            "pnl": -50.0,
            "pnl_pct": -0.028,
            "duration_minutes": 30.0,
            "exit_reason": "stop_loss",
        }
    ]
    review, llm, strategy_adjustments, trades = _review(
        llm_response={
            "observations": ["stops too tight"],
            "parameter_adjustments": {
                "max_position_pct": 0.50,  # over the high bound — clamp to 0.30
                "daily_loss_limit": 0.001,  # under the low bound — clamp to 0.005
            },
            "pairs_to_avoid": [],
            "strategy_notes": "",
        },
    )
    trades.get_completed_stock_round_trips = AsyncMock(return_value=trades_response)

    result = await review.review(lookback_days=1)

    persisted = {
        c.kwargs["parameter"]: c.kwargs["new_value"]
        for c in strategy_adjustments.record_strategy_adjustment.await_args_list
    }
    assert persisted == {"max_position_pct": 0.30, "daily_loss_limit": 0.005}
    assert {a.parameter for a in result.adjustments} == {
        "max_position_pct",
        "daily_loss_limit",
    }


@pytest.mark.asyncio
async def test_review_drops_crypto_only_knobs_silently():
    """LLM (perhaps confused by a stale prompt) emits crypto knobs
    on a stocks review. The base's ``param not in _SAFE_BOUNDS``
    guard drops them — must not appear in persisted adjustments,
    must not be applied to the strategy."""
    strategy = MagicMock(spec=["_max_position_pct", "_daily_loss_limit"])
    review, _, strategy_adjustments, trades = _review(
        llm_response={
            "observations": [],
            "parameter_adjustments": {
                "stop_loss_pct": 0.012,  # crypto-only — DROP
                "rsi_buy_threshold": 30.0,  # crypto-only — DROP
                "max_position_pct": 0.18,  # valid stocks knob
            },
            "pairs_to_avoid": [],
            "strategy_notes": "",
        },
        strategy=strategy,
    )
    trades.get_completed_stock_round_trips = AsyncMock(
        return_value=[
            {
                "pair": "MSFT",
                "buy_price": 400,
                "sell_price": 395,
                "pnl": -5,
                "pnl_pct": -0.0125,
                "duration_minutes": 45.0,
                "exit_reason": "stop_loss",
            }
        ]
    )

    result = await review.review(lookback_days=1)

    persisted = {
        c.kwargs["parameter"]
        for c in strategy_adjustments.record_strategy_adjustment.await_args_list
    }
    assert persisted == {"max_position_pct"}  # the crypto-only knobs never made it
    assert {a.parameter for a in result.adjustments} == {"max_position_pct"}
    # And the strategy didn't sprout a phantom stop_loss_pct attribute.
    assert not hasattr(strategy, "_stop_loss_pct")


# ── Strategy mutation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_mutates_strategy_max_position_pct():
    """Happy path: the live strategy's ``_max_position_pct`` is
    overwritten in-place so the next analyze() call uses the
    adjusted value."""
    strategy = MagicMock(spec=["_max_position_pct", "_daily_loss_limit"])
    strategy._max_position_pct = 0.20
    strategy._daily_loss_limit = 0.02
    review, _, _, trades = _review(
        llm_response={
            "observations": [],
            "parameter_adjustments": {"max_position_pct": 0.15},
            "pairs_to_avoid": [],
            "strategy_notes": "",
        },
        strategy=strategy,
    )
    trades.get_completed_stock_round_trips = AsyncMock(
        return_value=[
            {
                "pair": "TSLA",
                "buy_price": 200,
                "sell_price": 195,
                "pnl": -5,
                "pnl_pct": -0.025,
                "duration_minutes": 10.0,
                "exit_reason": "stop_loss",
            }
        ]
    )

    await review.review(lookback_days=1)

    assert strategy._max_position_pct == 0.15  # mutated in-place
    assert strategy._daily_loss_limit == 0.02  # untouched


# ── format_adjustments_for_prompt (stocks-flavored) ─────────


def test_format_adjustments_uses_stocks_friendly_labels():
    """Active adjustments render as plain ``- key: value`` lines.
    Same surface as crypto (the base class owns this), but pinning
    here to catch a future refactor that breaks the contract."""
    review, *_ = _review()
    review._active_adjustments = {"max_position_pct": 0.18, "daily_loss_limit": 0.015}
    review._pairs_to_avoid = ["GME"]
    out = review.format_adjustments_for_prompt()
    assert "max_position_pct: 0.18" in out
    assert "daily_loss_limit: 0.015" in out
    assert "GME" in out


# ── Misc smoke ────────────────────────────────────────────────


def test_dataclasses_reachable_from_core():
    """Pin: the dataclasses moved to ``core.self_improve`` but the
    crypto module re-exports them for back-compat. Stocks tests
    can import either path; canonicalise on ``core`` for new code."""
    assert ReviewResult().observations == []
    assert (
        StrategyAdjustment(parameter="x", old_value=None, new_value=0.1, reasoning="").parameter
        == "x"
    )
