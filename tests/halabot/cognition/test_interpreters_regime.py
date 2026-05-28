"""Regime classifier + cheap interpreters."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.belief.schema import EvidenceItem, Regime
from halabot.cognition.bars import Bar, BarBuffer
from halabot.cognition.interpreters import (
    AnomalyInterpreter,
    IndicatorInterpreter,
    NewsLexiconInterpreter,
    RsiInterpreter,
    TrendAlignmentInterpreter,
)
from halabot.cognition.regime import EvidenceRegimeClassifier
from halabot.platform.clock import FakeClock
from halabot.platform.events import EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
CLOCK = FakeClock(T0)


def _ev(direction, weight=1.0, *, source="x", directional=True):
    return EvidenceItem(
        source=source, direction=direction, weight=weight, ts=T0, directional=directional
    )


# ── regime ──
def test_regime_empty_is_ranging():
    assert EvidenceRegimeClassifier().classify([])[0] == Regime.RANGING


def test_regime_strong_bullish_is_trending_up():
    regime, conf = EvidenceRegimeClassifier().classify([_ev(1.0), _ev(0.9)])
    assert regime == Regime.TRENDING_UP
    assert conf > 0


def test_regime_strong_bearish_is_trending_down():
    regime, _ = EvidenceRegimeClassifier().classify([_ev(-1.0), _ev(-0.8)])
    assert regime == Regime.TRENDING_DOWN


def test_regime_weak_signal_is_ranging():
    regime, _ = EvidenceRegimeClassifier().classify([_ev(0.1, 0.2)])
    assert regime == Regime.RANGING


def test_regime_anomaly_flag_is_volatile():
    regime, _ = EvidenceRegimeClassifier().classify(
        [_ev(1.0), _ev(0.0, source="anomaly", directional=False)]
    )
    assert regime == Regime.VOLATILE


# ── IndicatorInterpreter ──
@pytest.mark.asyncio
async def test_indicator_interpreter_emits_on_uptrend():
    buf = BarBuffer()
    for i in range(30):
        c = 100 + i
        buf.append("NVDA", Bar(o=c, h=c + 1, low=c - 1, c=c, v=1.0, ts=T0))
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await IndicatorInterpreter(buf).interpret(obs)
    assert len(out) == 1
    assert out[0].source == "indicator.momentum"
    assert out[0].direction > 0
    assert out[0].event_id == obs.id  # provenance


@pytest.mark.asyncio
async def test_indicator_interpreter_silent_without_history():
    buf = BarBuffer()
    buf.append("NVDA", Bar(o=1, h=1, low=1, c=1, v=1.0, ts=T0))
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await IndicatorInterpreter(buf).interpret(obs) == []


# ── RsiInterpreter ──
def _fill(buf, asset, closes):
    for i, c in enumerate(closes):
        buf.append(asset, Bar(o=c, h=c + 1, low=c - 1, c=c, v=1.0, ts=T0))


@pytest.mark.asyncio
async def test_rsi_interpreter_bullish_on_rising():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(20)])  # all gains → RSI ~100
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await RsiInterpreter(buf).interpret(obs)
    assert len(out) == 1 and out[0].source == "indicator.rsi" and out[0].direction > 0


@pytest.mark.asyncio
async def test_rsi_interpreter_silent_without_history():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100, 101, 102])
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await RsiInterpreter(buf).interpret(obs) == []


# ── TrendAlignmentInterpreter ──
@pytest.mark.asyncio
async def test_alignment_bullish_when_both_horizons_up():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(50)])  # steadily up
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await TrendAlignmentInterpreter(buf).interpret(obs)
    assert len(out) == 1 and out[0].source == "indicator.alignment" and out[0].direction > 0


@pytest.mark.asyncio
async def test_alignment_silent_when_horizons_disagree():
    buf = BarBuffer()
    # long-window up overall, but last 10 bars down → mixed → no signal
    closes = [100 + i for i in range(40)] + [139 - i for i in range(10)]
    _fill(buf, "NVDA", closes)
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await TrendAlignmentInterpreter(buf).interpret(obs) == []


# ── AnomalyInterpreter ──
@pytest.mark.asyncio
async def test_anomaly_flag_on_volatility_spike():
    buf = BarBuffer()
    calm = [100 + 0.1 * i for i in range(40)]          # low-vol drift
    spike = [104, 96, 108, 92, 110]                     # sudden chaos
    _fill(buf, "NVDA", calm + spike)
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await AnomalyInterpreter(buf).interpret(obs)
    assert len(out) == 1
    assert out[0].source == "anomaly"
    assert out[0].directional is False  # a flag, not a directional vote


@pytest.mark.asyncio
async def test_anomaly_silent_on_calm_tape():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + 0.1 * i for i in range(45)])  # steady, no spike
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await AnomalyInterpreter(buf).interpret(obs) == []


# ── NewsLexiconInterpreter ──
@pytest.mark.asyncio
async def test_news_interpreter_emits_polarity_evidence():
    obs = new_event(
        CLOCK, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
        payload={"lexicon_polarity": 0.8, "headline": "blowout earnings"},
    )
    out = await NewsLexiconInterpreter().interpret(obs)
    assert len(out) == 1
    assert out[0].source == "news"
    assert out[0].direction == 0.8


@pytest.mark.asyncio
async def test_news_interpreter_silent_when_lexicon_abstains():
    obs = new_event(
        CLOCK, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
        payload={"lexicon_polarity": None, "headline": "ambiguous"},
    )
    assert await NewsLexiconInterpreter().interpret(obs) == []
