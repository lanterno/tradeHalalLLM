"""Tests for the Wave B stage classes.

Each stage is a thin wrapper that takes a :class:`CycleState`, mutates
one field, returns it. The underlying helpers are already covered by
:mod:`tests.test_cycle_shared_helpers` — these tests exercise the
state-mutation contract.
"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle_pipeline import CycleState
from halal_trader.core.cycle_stages import (
    BuildActiveAdjustmentsStage,
    BuildCatalystsStage,
    BuildPerformanceStage,
    BuildStockRiskStage,
    BuildTimeframeStage,
)

# ── BuildTimeframeStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeframe_stage_no_analyzer_leaves_text_empty():
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildTimeframeStage(analyzer=None).run(state)
    assert out.timeframe_text == ""


@pytest.mark.asyncio
async def test_timeframe_stage_calls_analyzer_and_formats():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(
        return_value={
            "AAPL": {
                "alignment_score": 0.72,
                "per_tf": {"1Day": "RSI=58, MACD=bullish"},
                "support_resistance": [],
            }
        }
    )
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildTimeframeStage(analyzer=analyzer).run(state)
    assert "AAPL" in out.timeframe_text
    assert "BULLISH" in out.timeframe_text


@pytest.mark.asyncio
async def test_timeframe_stage_has_stable_name():
    assert BuildTimeframeStage(analyzer=None).name == "build_timeframe_text"


# ── BuildPerformanceStage ────────────────────────────────────────


@pytest.mark.asyncio
async def test_performance_stage_no_analytics_leaves_text_empty():
    state = CycleState()
    out = await BuildPerformanceStage(analytics=None).run(state)
    assert out.performance_text == ""


@pytest.mark.asyncio
async def test_performance_stage_calls_compute_and_format():
    analytics = MagicMock()
    analytics.compute_stats = AsyncMock(return_value="<stats>")
    analytics.format_for_prompt = MagicMock(return_value="Win rate: 55%")
    state = CycleState()
    out = await BuildPerformanceStage(analytics=analytics, lookback_days=14).run(state)
    analytics.compute_stats.assert_awaited_once_with(lookback_days=14)
    analytics.format_for_prompt.assert_called_once_with("<stats>")
    assert out.performance_text == "Win rate: 55%"


@pytest.mark.asyncio
async def test_performance_stage_swallows_failure():
    analytics = MagicMock()
    analytics.compute_stats = AsyncMock(side_effect=RuntimeError("db down"))
    state = CycleState()
    out = await BuildPerformanceStage(analytics=analytics).run(state)
    assert out.performance_text == ""


@pytest.mark.asyncio
async def test_performance_stage_has_stable_name():
    assert BuildPerformanceStage(analytics=None).name == "build_performance_text"


# ── BuildActiveAdjustmentsStage ──────────────────────────────────


@pytest.mark.asyncio
async def test_active_adjustments_stage_no_reviewer_leaves_text_empty():
    state = CycleState()
    out = await BuildActiveAdjustmentsStage(self_review=None).run(state)
    assert out.active_adjustments == ""


@pytest.mark.asyncio
async def test_active_adjustments_stage_calls_formatter():
    reviewer = MagicMock()
    reviewer.format_adjustments_for_prompt.return_value = "- max_position_pct: 0.10"
    state = CycleState()
    out = await BuildActiveAdjustmentsStage(self_review=reviewer).run(state)
    assert out.active_adjustments == "- max_position_pct: 0.10"


@pytest.mark.asyncio
async def test_active_adjustments_stage_swallows_failure():
    reviewer = MagicMock()
    reviewer.format_adjustments_for_prompt.side_effect = RuntimeError("boom")
    state = CycleState()
    out = await BuildActiveAdjustmentsStage(self_review=reviewer).run(state)
    assert out.active_adjustments == ""


@pytest.mark.asyncio
async def test_active_adjustments_stage_has_stable_name():
    assert BuildActiveAdjustmentsStage(self_review=None).name == "build_active_adjustments"


# ── BuildCatalystsStage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_catalysts_stage_no_feed_leaves_text_empty():
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildCatalystsStage(feed=None).run(state)
    assert out.catalysts_text == ""


@pytest.mark.asyncio
async def test_catalysts_stage_no_symbols_leaves_text_empty():
    feed = MagicMock()
    feed.fetch_all = AsyncMock(return_value=[])
    state = CycleState()  # empty halal_pairs
    out = await BuildCatalystsStage(feed=feed).run(state)
    assert out.catalysts_text == ""
    assert feed.fetch_all.call_count == 0


@pytest.mark.asyncio
async def test_catalysts_stage_calls_feed_and_formats():
    from datetime import datetime

    from halal_trader.trading.catalysts import Catalyst

    feed = MagicMock()
    feed.fetch_all = AsyncMock(
        return_value=[
            Catalyst(
                symbol="AAPL",
                kind="news",
                title="Apple beats Q1",
                timestamp=datetime.now(UTC),
                sentiment="positive",
                source="Bloomberg",
            )
        ]
    )
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildCatalystsStage(feed=feed).run(state)
    assert "AAPL" in out.catalysts_text
    assert "Apple beats Q1" in out.catalysts_text


@pytest.mark.asyncio
async def test_catalysts_stage_swallows_failure():
    feed = MagicMock()
    feed.fetch_all = AsyncMock(side_effect=RuntimeError("alpaca down"))
    state = CycleState(halal_pairs=["AAPL"])
    out = await BuildCatalystsStage(feed=feed).run(state)
    assert out.catalysts_text == ""


@pytest.mark.asyncio
async def test_catalysts_stage_has_stable_name():
    assert BuildCatalystsStage(feed=None).name == "build_catalysts_text"


# ── BuildStockRiskStage ──────────────────────────────────────────


def _bar(o: float, h: float, low: float, c: float, v: float = 1_000.0) -> dict:
    return {"o": o, "h": h, "l": low, "c": c, "v": v}


def _stock_series(start: float, n: int, step: float = 0.5) -> list[dict]:
    out = []
    price = start
    for _ in range(n):
        out.append(_bar(price, price + 0.5, price - 0.5, price + step))
        price += step
    return out


@pytest.mark.asyncio
async def test_stock_risk_stage_no_bars_leaves_text_empty():
    state = CycleState()
    out = await BuildStockRiskStage().run(state)
    assert out.risk_text == ""
    assert out.indicators_cache == {}


@pytest.mark.asyncio
async def test_stock_risk_stage_populates_risk_and_indicators():
    """Happy path: bars in, risk_text + indicators_cache out."""
    account = MagicMock()
    account.effective_equity = 100_000.0
    account.equity = 100_000.0
    state = CycleState(
        bars={"AAPL": _stock_series(180.0, 50)},
        open_positions=[],
        account=account,
    )
    out = await BuildStockRiskStage().run(state)
    # The risk engine produces a non-empty text block + populates the
    # indicator cache that downstream stages (regime, ML) consume.
    assert isinstance(out.risk_text, str)
    assert "AAPL" in out.indicators_cache


@pytest.mark.asyncio
async def test_stock_risk_stage_swallows_failure():
    """A buggy bars payload mustn't break the cycle."""
    account = MagicMock()
    account.effective_equity = 100_000.0
    state = CycleState(
        bars={"AAPL": "not a bars payload"},  # triggers parse failure
        open_positions=[],
        account=account,
    )
    out = await BuildStockRiskStage().run(state)
    # Risk text empty; indicators cache stays empty.
    assert out.risk_text == ""


@pytest.mark.asyncio
async def test_stock_risk_stage_threads_halt_signal():
    """When the risk engine returns ``is_halted=True``, the stage must
    set ``state.halt`` so the cycle short-circuits before the LLM call.
    """
    from unittest.mock import patch

    halted_state = MagicMock()
    halted_state.is_halted = True
    halted_state.halt_reason = "drawdown_breach"
    output = MagicMock(
        state=halted_state,
        risk_text="HALT: drawdown breach",
        indicators_by_symbol={},
    )
    account = MagicMock()
    account.effective_equity = 100_000.0
    state = CycleState(
        bars={"AAPL": [_bar(180.0, 181.0, 179.5, 180.5)]},
        open_positions=[],
        account=account,
    )
    with patch("halal_trader.trading.risk.evaluate_stock_risk", return_value=output):
        out = await BuildStockRiskStage().run(state)
    assert out.halt is True
    assert out.risk_state is halted_state
    assert "HALT" in out.risk_text


@pytest.mark.asyncio
async def test_stock_risk_stage_no_bars_leaves_halt_false():
    """Empty-bars early return must not leave a stale halt from prior cycle."""
    state = CycleState(halt=True)  # simulate stale halt flag
    out = await BuildStockRiskStage().run(state)
    assert out.halt is False


@pytest.mark.asyncio
async def test_stock_risk_stage_has_stable_name():
    assert BuildStockRiskStage().name == "evaluate_stock_risk"
