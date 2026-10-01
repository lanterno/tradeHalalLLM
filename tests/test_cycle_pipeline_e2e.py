"""End-to-end stage-pipeline smoke test.

Each stage is tested in isolation elsewhere. This file proves the
stock cycle's stages compose: a single :class:`CycleState` flowing
through a real stage list produces every prompt-context field.

We don't drive a real cycle here — just the stage list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle_pipeline import CycleState
from halal_trader.core.cycle_stages import (
    BuildActiveAdjustmentsStage,
    BuildCatalystsStage,
    BuildMlSignalsStage,
    BuildPerformanceStage,
    BuildTimeframeStage,
    FetchStockNewsStage,
)


@pytest.mark.asyncio
async def test_full_stage_pipeline_populates_every_text_field():
    """Drive the stock prompt-context stages over one state."""
    anomaly = MagicMock()
    anomaly.detect.return_value = (False, 0.1)
    signal = MagicMock()
    signal.predict_confidence.return_value = 0.55

    timeframe = MagicMock()
    timeframe.analyze = AsyncMock(
        return_value={
            "AAPL": {
                "alignment_score": 0.7,
                "per_tf": {"1Day": "RSI=58, MACD=bullish"},
                "support_resistance": [],
            }
        }
    )

    analytics = MagicMock()
    analytics.compute_stats = AsyncMock(return_value="<stats>")
    analytics.format_for_prompt = MagicMock(return_value="Win rate: 60%")

    self_review = MagicMock()
    self_review.format_adjustments_for_prompt.return_value = "- max_position_pct: 0.10"

    # Use a real Catalyst so the formatter accepts it.
    from halal_trader.trading.catalysts import Catalyst

    feed = MagicMock()
    feed.fetch_all = AsyncMock(
        return_value=[
            Catalyst(
                symbol="AAPL",
                kind="news",
                title="AAPL beats earnings",
                timestamp=datetime.now(timezone.utc),
                sentiment="positive",
                source="Bloomberg",
            )
        ]
    )

    news_collector = MagicMock()
    news_collector.fetch_for_symbols = AsyncMock(return_value=[])  # empty path

    state = CycleState(
        account=MagicMock(),
        halal_pairs=["AAPL"],
        indicators_cache={"AAPL": {"rsi_14": 60, "ema_9": 100, "ema_21": 99}},
    )

    stages = [
        BuildMlSignalsStage(anomaly_detector=anomaly, signal_classifier=signal),
        BuildTimeframeStage(analyzer=timeframe),
        BuildCatalystsStage(feed=feed),
        BuildPerformanceStage(analytics=analytics),
        BuildActiveAdjustmentsStage(self_review=self_review),
        FetchStockNewsStage(news_collector=news_collector),
    ]
    for stage in stages:
        out = await stage.run(state)
        # Every stage returns the state in place (the contract).
        assert out is state

    # ML stage emits a confidence section even with no anomalies.
    assert "ML confidence" in state.ml_signals_text
    assert "AAPL" in state.timeframe_text
    assert "BULLISH" in state.timeframe_text  # alignment 0.7 → BULLISH bucket
    assert "AAPL" in state.catalysts_text
    assert "earnings" in state.catalysts_text
    assert state.performance_text == "Win rate: 60%"
    assert state.active_adjustments == "- max_position_pct: 0.10"
    # Empty news fetch → empty block.
    assert state.news_text == ""


@pytest.mark.asyncio
async def test_pipeline_runs_with_no_deps_wired():
    """Every stage's no-op path: empty state in, empty state out."""
    state = CycleState()
    stages = [
        BuildMlSignalsStage(),
        BuildTimeframeStage(analyzer=None),
        BuildCatalystsStage(feed=None),
        BuildPerformanceStage(analytics=None),
        BuildActiveAdjustmentsStage(self_review=None),
        FetchStockNewsStage(news_collector=None),
    ]
    for stage in stages:
        await stage.run(state)
    # Every text field is still the empty default.
    assert state.regime_text == ""
    assert state.ml_signals_text == ""
    assert state.forecasts_text == ""
    assert state.timeframe_text == ""
    assert state.performance_text == ""
    assert state.active_adjustments == ""
    assert state.catalysts_text == ""
    assert state.news_text == ""
    assert state.risk_text == ""
    assert state.halt is False


@pytest.mark.asyncio
async def test_run_stages_drives_list_and_returns_state():
    """``run_stages`` runs each stage in order and returns the mutated state."""
    from halal_trader.core.cycle_pipeline import run_stages

    analytics = MagicMock()
    analytics.compute_stats = AsyncMock(return_value="<stats>")
    analytics.format_for_prompt = MagicMock(return_value="Win rate: 55%")
    state = CycleState(halal_pairs=["AAPL"])
    out = await run_stages(state, [BuildPerformanceStage(analytics=analytics)])
    assert out is state
    assert out.performance_text == "Win rate: 55%"


@pytest.mark.asyncio
async def test_run_stages_short_circuits_on_halt():
    """``stop_on_halt=True`` stops the chain after a stage sets ``state.halt``."""
    from halal_trader.core.cycle_pipeline import run_stages

    class _Halt:
        name = "halt_setter"

        async def run(self, state):  # noqa: ANN001
            state.halt = True
            return state

    class _After:
        name = "after_halt"

        def __init__(self):
            self.ran = False

        async def run(self, state):  # noqa: ANN001
            self.ran = True
            return state

    after = _After()
    state = CycleState()
    await run_stages(state, [_Halt(), after], stop_on_halt=True)
    assert state.halt is True
    assert after.ran is False  # didn't run because halt short-circuited


@pytest.mark.asyncio
async def test_run_stages_runs_all_when_stop_on_halt_default():
    """Default behavior (``stop_on_halt=False``) runs every stage even if halt is set."""
    from halal_trader.core.cycle_pipeline import run_stages

    class _Halt:
        name = "halt_setter"

        async def run(self, state):  # noqa: ANN001
            state.halt = True
            return state

    class _After:
        name = "after_halt"

        def __init__(self):
            self.ran = False

        async def run(self, state):  # noqa: ANN001
            self.ran = True
            return state

    after = _After()
    state = CycleState()
    await run_stages(state, [_Halt(), after])  # default stop_on_halt=False
    assert state.halt is True
    assert after.ran is True  # ran despite halt — opt-in semantics


@pytest.mark.asyncio
async def test_run_stages_publishes_per_stage_events():
    """Each stage runs inside the instrumentation context — bus sees stage.start/end."""
    from halal_trader.core.cycle_pipeline import run_stages

    bus = MagicMock()
    bus.publish = AsyncMock()
    state = CycleState()
    await run_stages(
        state,
        [BuildTimeframeStage(analyzer=None), BuildMlSignalsStage()],
        bus=bus,
    )
    # 2 stages × (start + end) = 4 publishes.
    assert bus.publish.await_count == 4
    topics = [call.args[0] for call in bus.publish.await_args_list]
    assert topics == [
        "cycle.stage.start",
        "cycle.stage.end",
        "cycle.stage.start",
        "cycle.stage.end",
    ]
