"""Cheap, continuous interpreters: bar momentum + news lexicon (REARCHITECTURE L2).

Both are LLM-free. The momentum interpreter reads the rolling buffer (the router
appends the new bar before invoking it); the news interpreter uses the cheap
lexicon polarity that perception already attached, leaving LLM headline scoring
to the sparse cognition path.
"""

from __future__ import annotations

from halabot.belief.schema import EvidenceItem
from halabot.cognition.bars import BarBuffer, momentum_signal
from halabot.platform.events import Event, EventType

# News evidence decays fast; one headline shouldn't dominate for long. The
# decay half-life is global, so we encode "freshness" via a modest base weight.
_NEWS_WEIGHT = 0.8


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
