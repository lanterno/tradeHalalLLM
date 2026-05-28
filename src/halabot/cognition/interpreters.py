"""Cheap, continuous interpreters (REARCHITECTURE L2) — all LLM-free.

Bar-driven (momentum, RSI, multi-horizon trend alignment) read the rolling
buffer the router fills; the news interpreter uses the cheap lexicon polarity
perception attached. Multiple bar interpreters from the same series are
correlated, so they don't add *independent* information — but they raise
conviction mass when they agree and, via the agreement/dispersion penalty,
*dampen* it when they disagree (e.g. price momentum up but RSI rolling over).
The fitted calibrator (L8) learns each source's true weight once outcomes exist.
"""

from __future__ import annotations

import statistics

from halabot.belief.schema import EvidenceItem
from halabot.cognition.bars import BarBuffer, momentum_signal, rsi
from halabot.platform.events import Event, EventType

# News evidence decays fast; one headline shouldn't dominate for long. The
# decay half-life is global, so we encode "freshness" via a modest base weight.
_NEWS_WEIGHT = 0.8
# Secondary confirmations carry less weight than the primary trend signal.
_RSI_WEIGHT = 0.5
_ALIGN_WEIGHT = 0.6


class IndicatorInterpreter:
    """Emits a trend-momentum :class:`EvidenceItem` from the bar buffer."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(self, buffer: BarBuffer) -> None:
        self._buffer = buffer

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        direction, weight = momentum_signal(self._buffer.closes(asset))
        if weight <= 0.0:
            return []  # insufficient history — neutral, weightless
        return [
            EvidenceItem(
                source="indicator.momentum",
                direction=direction,
                weight=weight,
                detail=f"fast/slow EMA momentum {direction:+.2f}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class RsiInterpreter:
    """Emits an RSI momentum-confirmation signal (RSI>50 bullish, <50 bearish)."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(self, buffer: BarBuffer, *, period: int = 14) -> None:
        self._buffer = buffer
        self._period = period

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        r = rsi(self._buffer.closes(asset), period=self._period)
        if r is None:
            return []
        direction = max(-1.0, min(1.0, (r - 50.0) / 25.0))  # 50→0, 75→+1, 25→-1
        if abs(direction) < 0.04:
            return []  # ~neutral — no signal
        return [
            EvidenceItem(
                source="indicator.rsi",
                direction=direction,
                weight=_RSI_WEIGHT,
                detail=f"RSI {r:.0f}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class TrendAlignmentInterpreter:
    """Multi-horizon alignment: short- AND long-window returns agreeing is a
    stronger directional signal than either alone; mixed → no signal."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(self, buffer: BarBuffer, *, short: int = 10, long: int = 40) -> None:
        self._buffer = buffer
        self._short = short
        self._long = long

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        if len(closes) < self._long or closes[-self._short] <= 0 or closes[-self._long] <= 0:
            return []
        short_ret = (closes[-1] - closes[-self._short]) / closes[-self._short]
        long_ret = (closes[-1] - closes[-self._long]) / closes[-self._long]
        if short_ret > 0 and long_ret > 0:
            direction = 0.8
        elif short_ret < 0 and long_ret < 0:
            direction = -0.8
        else:
            return []  # horizons disagree → no alignment signal
        return [
            EvidenceItem(
                source="indicator.alignment",
                direction=direction,
                weight=_ALIGN_WEIGHT,
                detail=f"short {short_ret:+.2%} / long {long_ret:+.2%}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class AnomalyInterpreter:
    """Emits a non-directional ``anomaly`` flag when short-window return
    volatility spikes above its baseline — wiring up conviction_raw's anomaly
    down-weight (×0.6), so the engine trusts a signal less in chaotic tape."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self, buffer: BarBuffer, *, short: int = 5, baseline: int = 30, mult: float = 2.0
    ) -> None:
        self._buffer = buffer
        self._short = short
        self._baseline = baseline
        self._mult = mult

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        if len(closes) < self._baseline + 1:
            return []
        rets = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if len(rets) < self._baseline:
            return []
        short_vol = statistics.pstdev(rets[-self._short :])
        base_vol = statistics.pstdev(rets[-self._baseline :])
        if base_vol <= 0 or short_vol <= self._mult * base_vol:
            return []
        return [
            EvidenceItem(
                source="anomaly",
                direction=0.0,
                weight=1.0,
                detail=f"vol spike {short_vol / base_vol:.1f}x baseline",
                ts=observation.ts,
                event_id=observation.id,
                directional=False,  # a flag, not a directional vote
            )
        ]


class NewsLexiconInterpreter:
    """Emits polarity evidence from the cheap lexicon score on a news observation."""

    consumes = frozenset({EventType.OBSERVATION_NEWS})

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        polarity = observation.payload.get("lexicon_polarity")
        if asset is None or polarity is None:
            return []  # lexicon abstained — no evidence (LLM scoring is the sparse path)
        direction = max(-1.0, min(1.0, float(polarity)))
        if direction == 0.0:
            return []
        headline = str(observation.payload.get("headline", ""))[:80]
        return [
            EvidenceItem(
                source="news",
                direction=direction,
                weight=_NEWS_WEIGHT,
                detail=f"news polarity {direction:+.2f}: {headline}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]
