"""Sparse LLM thesis writer + gate (REARCHITECTURE L3 step 4, INV-1).

The ONLY LLM touch in the belief loop, and it's triple-guarded by the updater
(``llm_thesis_enabled`` + ``available()`` + ``not breaker_open()``) and fired only
on a *material shift* — so an LLM outage stales the narrative, never the beliefs
(INV-1). The writer turns the deterministic belief state into a short, human
rationale ("the why"); it never feeds back into conviction or direction.

Decoupled from the legacy LLM via a structural ``Generator`` protocol, so it's
testable with a fake and the real ``halal_trader.core.llm`` backend wires in only
when the operator enables it (it costs money — OFF by default)."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from halabot.belief.schema import BeliefState

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a terse trading analyst. Given a market belief, write ONE sentence "
    "(max 40 words) explaining the thesis: why this direction, the key evidence, "
    "and what would invalidate it. No preamble, no disclaimers, no markdown."
)


class Generator(Protocol):
    async def generate(self, prompt: str, system: str | None = None) -> str: ...


def _prompt(b: BeliefState) -> str:
    ev = ", ".join(
        f"{e.source}{e.direction:+.2f}"
        for e in sorted(b.evidence, key=lambda e: -abs(e.direction * e.weight))[:5]
    )
    return (
        f"Asset {b.asset}: regime={b.regime.value}, direction={b.direction.value}, "
        f"conviction={b.conviction:.2f}, evidence=[{ev}], "
        f"invalidation={b.levels.invalidation}."
    )


class LlmThesisWriter:
    """Writes a concise thesis via an injected LLM. An LLM error propagates — the
    BeliefUpdater wraps the write() call in try/except so a failure stales the
    narrative but never the belief (INV-1). Output is length-bounded."""

    def __init__(self, llm: Generator, *, max_chars: int = 400) -> None:
        self._llm = llm
        self._max = max_chars

    async def write(self, belief: BeliefState) -> str:
        text = await self._llm.generate(_prompt(belief), _SYSTEM)
        return text.strip()[: self._max]


_SCORE_SYSTEM = (
    "You score a financial headline's directional impact on the named stock. "
    "Reply with ONLY a number in [-1, 1]: -1 very bearish, 0 neutral, +1 very "
    "bullish. No words, no explanation."
)


class LlmHeadlineScorer:
    """Scores a headline to a polarity via the LLM (the sparse news path). Returns
    None on an unparseable reply (abstain rather than fabricate a signal)."""

    def __init__(self, llm: Generator) -> None:
        self._llm = llm

    async def score(self, headline: str) -> float | None:
        reply = await self._llm.generate(headline, _SCORE_SYSTEM)
        return _parse_polarity(reply)


def _parse_polarity(text: str) -> float | None:
    import re

    m = re.search(r"-?\d*\.?\d+", text or "")
    if m is None:
        return None
    try:
        return max(-1.0, min(1.0, float(m.group())))
    except ValueError:
        return None


class LlmGate:
    """LLM health gate the updater consults. ``breaker_open`` defers to the
    backend's circuit breaker when it exposes one, else reports closed."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def available(self) -> bool:
        return self._llm is not None

    def breaker_open(self) -> bool:
        breaker = getattr(self._llm, "breaker_open", None)
        if callable(breaker):
            try:
                return bool(breaker())
            except Exception:  # noqa: BLE001 — a flaky breaker check defaults to "open" (safe)
                return True
        return bool(breaker) if isinstance(breaker, bool) else False
