"""Active-learning hard-case selector.

Round-4 wave 6.H: not every closed trade is equally informative
for the next training run. A confident BUY that became a 2σ loss
teaches the model far more than a routine winner-as-expected. The
operator's review time is finite, so the bot should rank trades
by **labeling priority** — surface the trades whose mismatch
between predicted-and-actual is most diagnostic.

A `TradeCase` carries the fields the scorer cares about (predicted
return, actual return, confidence, optional indicator outlier
score). A `Scorer` weighs four contributions:

* **Confidence × error** — a high-confidence prediction that
  missed by a large amount is a 2σ teaching moment.
* **Sign disagreement** — the model said BUY (positive predicted)
  and the trade lost money: the bigger the magnitude, the worse
  the disagreement.
* **Outlier indicator** — when the entry indicator vector was an
  outlier (e.g. RSI in a band the model rarely sees), the trade is
  a coverage gap worth labelling.
* **Recency** — newer trades teach the most about the *current*
  regime; older ones decay exponentially.

The scorer returns a `Priority` ranked list. Pin: ranks
deterministic given the inputs — no random tie-breaks. The
dashboard surfaces the top-N for operator review; labelled cases
feed back into the retraining queue.

Halal alignment: the selector is a *triage* — it surfaces cases
for human review, never auto-labels and never opens a position.
The operator's review is the only authority that produces a label
that flows into training.

Pure-Python; no numpy / scipy / DB / async.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ── Inputs ────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeCase:
    """One closed trade as the selector sees it.

    ``predicted_return`` and ``actual_return`` are unit fractions
    (0.05 = 5%). ``confidence`` is in [0, 1] from the strategy /
    LLM. ``indicator_outlier_score`` is optional and free-form —
    the scorer treats higher = more outlier; pass 0.0 (or None)
    when no outlier signal is available.

    ``trade_id`` is the operator's identifier; passed through
    unchanged so the dashboard can deep-link the case.

    ``age_seconds`` is how long ago the trade closed; the recency
    weight in the score uses this.
    """

    trade_id: str
    pair: str
    predicted_return: float
    actual_return: float
    confidence: float
    age_seconds: float
    indicator_outlier_score: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]; got {self.confidence}")
        if self.age_seconds < 0:
            raise ValueError(f"age_seconds must be >= 0; got {self.age_seconds}")


# ── Scoring config ───────────────────────────────────────


@dataclass(frozen=True)
class ScorerWeights:
    """Per-contribution weights.

    Defaults sum to ~1.0 so the raw priority score is roughly
    comparable across runs. Operators can tilt the mix toward
    sign-disagreement (a strategy that's been embarrassed by
    confident wrong-direction calls) or recency (a regime change
    where only the last week matters).

    ``recency_half_life_seconds`` controls how quickly old cases
    decay; default 7 days means a 14-day-old case contributes
    25% of a fresh one's weight.
    """

    confidence_error: float = 0.40
    sign_disagreement: float = 0.30
    outlier: float = 0.15
    recency: float = 0.15
    recency_half_life_seconds: float = 7 * 24 * 3600.0

    def __post_init__(self) -> None:
        for name in (
            "confidence_error",
            "sign_disagreement",
            "outlier",
            "recency",
        ):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} weight must be non-negative; got {v}")
        if self.recency_half_life_seconds <= 0:
            raise ValueError(
                f"recency_half_life_seconds must be positive; got {self.recency_half_life_seconds}"
            )


# ── Output ────────────────────────────────────────────────


@dataclass(frozen=True)
class Priority:
    """One case's ranked priority + the per-component breakdown.

    ``score`` is the operator-visible total. ``components``
    explains *why* the case ranked where it did — the dashboard
    renders these as a stacked bar so the operator can see "this
    case bubbled up because it was a high-confidence sign
    disagreement", not just "the algorithm said so"."""

    case: TradeCase
    score: float
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""


# ── Scoring ───────────────────────────────────────────────


def _confidence_error_score(case: TradeCase) -> float:
    """Confidence × |predicted - actual|. A high-confidence
    prediction that missed by a large margin scores high.

    Pin: capped at 2.0 so a single freak outlier (5σ event) doesn't
    wash out the rest of the score components."""
    err = abs(case.predicted_return - case.actual_return)
    return min(2.0, case.confidence * err * 10.0)


def _sign_disagreement_score(case: TradeCase) -> float:
    """Penalise sign mismatch scaled by magnitude.

    Pin: when both predicted and actual are zero, score is zero
    (no information). When signs disagree, score = min(2.0,
    |predicted| + |actual|) × 5 — pushes confident wrong-direction
    calls to the top."""
    p = case.predicted_return
    a = case.actual_return
    if p == 0.0 or a == 0.0:
        return 0.0
    if (p > 0) == (a > 0):
        return 0.0
    raw = min(2.0, abs(p) + abs(a)) * 5.0
    return min(2.0, raw)


def _outlier_score(case: TradeCase) -> float:
    """Indicator outlier passes through if available. Pin: clamped
    to [0, 1] — operators may pass un-normalised z-scores; the
    selector won't let one outlier dominate."""
    if case.indicator_outlier_score is None:
        return 0.0
    return max(0.0, min(1.0, case.indicator_outlier_score))


def _recency_score(case: TradeCase, half_life: float) -> float:
    """Exponential decay of priority by age. Pin: half_life is
    passed in so the operator can override per-call (e.g. "this
    week only" sweeps use a 1-day half-life)."""
    return math.exp(-math.log(2) * case.age_seconds / half_life)


_REASON_TEMPLATES: dict[str, str] = {
    "confidence_error": "high-confidence prediction missed by a wide margin",
    "sign_disagreement": "predicted direction was wrong",
    "outlier": "entry indicators were an outlier",
    "recency": "recent trade — informative for the current regime",
}


def _explain(components: dict[str, float]) -> str:
    """One-line operator-readable summary of the dominant
    contribution. Pin: picks the largest non-zero component."""
    if not components or max(components.values()) == 0:
        return "no notable signal"
    dominant = max(components, key=lambda k: components[k])
    return _REASON_TEMPLATES.get(dominant, dominant)


def score_case(
    case: TradeCase,
    *,
    weights: ScorerWeights | None = None,
) -> Priority:
    """Score one case. Returns a `Priority` with the per-component
    breakdown so the dashboard can render the explanation."""
    w = weights or ScorerWeights()
    components = {
        "confidence_error": w.confidence_error * _confidence_error_score(case),
        "sign_disagreement": w.sign_disagreement * _sign_disagreement_score(case),
        "outlier": w.outlier * _outlier_score(case),
        "recency": w.recency * _recency_score(case, w.recency_half_life_seconds),
    }
    total = sum(components.values())
    return Priority(
        case=case,
        score=total,
        components=components,
        reason=_explain(components),
    )


def select_top_n(
    cases: Iterable[TradeCase],
    *,
    n: int,
    weights: ScorerWeights | None = None,
) -> list[Priority]:
    """Score every case, return the top-N by descending score.

    Pin: stable sort — when two cases have equal scores, the
    earlier-listed one wins. Operators feed cases in chronological
    order; the stable sort means the older of two identically-
    scored cases gets reviewed first."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    scored = [score_case(c, weights=weights) for c in cases]
    # Sort descending by score; stable so original order breaks ties.
    scored.sort(key=lambda p: p.score, reverse=True)
    return scored[:n]


def render_queue(priorities: Sequence[Priority]) -> str:
    """CLI / Slack-ready text payload for the triage queue.

    Visual layout matches the other Round-4 render helpers — short
    one-line entries with the top-priority case first."""
    if not priorities:
        return "=== Active-learning queue ===\n(empty)"
    lines = ["=== Active-learning queue ==="]
    for i, p in enumerate(priorities, start=1):
        c = p.case
        lines.append(
            f"  {i:>2}. {c.trade_id} ({c.pair}) "
            f"score={p.score:.3f}  pred={c.predicted_return:+.2%} "
            f"actual={c.actual_return:+.2%}  conf={c.confidence:.0%}"
        )
        lines.append(f"      → {p.reason}")
    return "\n".join(lines)


__all__ = [
    "Priority",
    "ScorerWeights",
    "TradeCase",
    "render_queue",
    "score_case",
    "select_top_n",
]
