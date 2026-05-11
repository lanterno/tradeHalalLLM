"""Risk-tolerance self-assessment — Round-5 Wave 20.G.

Multi-question quiz that maps user responses to a `RiskProfile`
(CONSERVATIVE / BALANCED / AGGRESSIVE) and recommends an equity/sukuk
allocation band consistent with that profile.

Pinned semantics:

- **Closed-set RiskProfile ladder** — CONSERVATIVE / BALANCED /
  AGGRESSIVE.
- **Closed-set QuestionAxis** — DRAWDOWN_TOLERANCE / TIME_HORIZON /
  INCOME_DEPENDENCE / VOLATILITY_REACTION / KNOWLEDGE.
- **Each question has options with a closed-set score in {0, 1, 2, 3, 4}.**
- **Weights are operator-tunable** per axis; total weighted score
  maps to profile via fixed thresholds (≤0.33 → CONSERVATIVE,
  ≤0.66 → BALANCED, else AGGRESSIVE).
- **Recommended allocations are pinned per profile** — operators can
  override, but defaults match the standard halal-wealth-management
  literature (CONSERVATIVE 30/70, BALANCED 60/40, AGGRESSIVE 80/20
  for equity/sukuk).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum


class RiskProfile(str, Enum):
    """Closed-set risk profile ladder."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class QuestionAxis(str, Enum):
    """Closed-set axis ladder."""

    DRAWDOWN_TOLERANCE = "drawdown_tolerance"
    TIME_HORIZON = "time_horizon"
    INCOME_DEPENDENCE = "income_dependence"
    VOLATILITY_REACTION = "volatility_reaction"
    KNOWLEDGE = "knowledge"


_VALID_SCORES = {0, 1, 2, 3, 4}


@dataclass(frozen=True)
class Option:
    """One answer option for a quiz question."""

    key: str
    label: str
    score: int
    """Closed-set in {0, 1, 2, 3, 4}; higher = more risk-tolerant."""

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("key must be non-empty")
        if not self.label or not self.label.strip():
            raise ValueError("label must be non-empty")
        if self.score not in _VALID_SCORES:
            raise ValueError(f"score must be in {sorted(_VALID_SCORES)}")


@dataclass(frozen=True)
class Question:
    """One quiz question."""

    question_id: str
    axis: QuestionAxis
    prompt: str
    options: tuple[Option, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not self.options:
            raise ValueError("options must be non-empty")
        keys = [o.key for o in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError("option keys must be unique")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        # Each question must have at least one option per low / mid / high
        # band so the scoring axis is informative.
        scores = sorted({o.score for o in self.options})
        if len(scores) < 2:
            raise ValueError("question must span ≥ 2 distinct scores to be informative")


@dataclass(frozen=True)
class Response:
    """A user's response to one question."""

    question_id: str
    option_key: str


def _question_score(question: Question, response: Response) -> int:
    opt = next((o for o in question.options if o.key == response.option_key), None)
    if opt is None:
        raise ValueError(
            f"option_key {response.option_key!r} not in question {question.question_id}"
        )
    return opt.score


def _max_score(question: Question) -> int:
    return max(o.score for o in question.options)


@dataclass(frozen=True)
class AllocationBand:
    """Recommended equity/sukuk allocation."""

    equity_min: float
    equity_max: float
    sukuk_min: float
    sukuk_max: float

    def __post_init__(self) -> None:
        for name, lo, hi in (
            ("equity", self.equity_min, self.equity_max),
            ("sukuk", self.sukuk_min, self.sukuk_max),
        ):
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError(f"{name} band must satisfy 0 ≤ min ≤ max ≤ 1")
        # Feasibility: there must exist (e, s) with e+s=1 that satisfies
        # both bands. That requires:
        #   max(equity_min, 1 - sukuk_max) ≤ min(equity_max, 1 - sukuk_min)
        feasible_lo = max(self.equity_min, 1.0 - self.sukuk_max)
        feasible_hi = min(self.equity_max, 1.0 - self.sukuk_min)
        if feasible_lo > feasible_hi + 1e-9:
            raise ValueError("equity + sukuk bands do not admit any allocation summing to 1")


_DEFAULT_ALLOCATIONS: dict[RiskProfile, AllocationBand] = {
    RiskProfile.CONSERVATIVE: AllocationBand(
        equity_min=0.20,
        equity_max=0.40,
        sukuk_min=0.60,
        sukuk_max=0.80,
    ),
    RiskProfile.BALANCED: AllocationBand(
        equity_min=0.40,
        equity_max=0.70,
        sukuk_min=0.30,
        sukuk_max=0.60,
    ),
    RiskProfile.AGGRESSIVE: AllocationBand(
        equity_min=0.70,
        equity_max=0.90,
        sukuk_min=0.10,
        sukuk_max=0.30,
    ),
}


@dataclass(frozen=True)
class AssessmentResult:
    """Output of `assess`."""

    candidate_id: str
    assessment_date: date
    raw_score: float
    """Sum of (weight × option_score)."""
    max_score: float
    """Sum of (weight × max_option_score per question)."""
    normalised: float
    """raw / max ∈ [0, 1]."""
    profile: RiskProfile
    allocation: AllocationBand
    axis_scores: dict[QuestionAxis, float]
    """Per-axis normalised score ∈ [0, 1]."""


def _profile_from_normalised(score: float) -> RiskProfile:
    if score <= 0.33:
        return RiskProfile.CONSERVATIVE
    if score <= 0.66:
        return RiskProfile.BALANCED
    return RiskProfile.AGGRESSIVE


def assess(
    questions: Sequence[Question],
    responses: Iterable[Response],
    *,
    candidate_id: str,
    assessment_date: date,
    allocation_overrides: dict[RiskProfile, AllocationBand] | None = None,
) -> AssessmentResult:
    """Compute a risk-tolerance assessment.

    Pinned:
    - Every question must have exactly one matching response.
    - Unknown response question_ids raise.
    """
    if not candidate_id or not candidate_id.strip():
        raise ValueError("candidate_id must be non-empty")
    if not questions:
        raise ValueError("questions must be non-empty")
    response_list = list(responses)
    if len(response_list) != len(questions):
        raise ValueError("must respond to every question exactly once")
    by_qid = {r.question_id: r for r in response_list}
    if len(by_qid) != len(response_list):
        raise ValueError("duplicate response for the same question")
    raw_total = 0.0
    max_total = 0.0
    by_axis_raw: dict[QuestionAxis, float] = {}
    by_axis_max: dict[QuestionAxis, float] = {}
    for q in questions:
        r = by_qid.get(q.question_id)
        if r is None:
            raise ValueError(f"missing response for question {q.question_id}")
        s = _question_score(q, r)
        mx = _max_score(q)
        raw_total += q.weight * s
        max_total += q.weight * mx
        by_axis_raw[q.axis] = by_axis_raw.get(q.axis, 0.0) + q.weight * s
        by_axis_max[q.axis] = by_axis_max.get(q.axis, 0.0) + q.weight * mx
    # Check for orphan responses.
    q_ids = {q.question_id for q in questions}
    for r in response_list:
        if r.question_id not in q_ids:
            raise ValueError(f"response refers to unknown question {r.question_id}")
    norm = raw_total / max_total if max_total > 0 else 0.0
    profile = _profile_from_normalised(norm)
    allocations = allocation_overrides if allocation_overrides is not None else _DEFAULT_ALLOCATIONS
    if profile not in allocations:
        raise ValueError(f"missing allocation override for {profile.value}")
    axis_scores = {
        axis: (by_axis_raw[axis] / by_axis_max[axis]) if by_axis_max[axis] > 0 else 0.0
        for axis in by_axis_raw
    }
    return AssessmentResult(
        candidate_id=candidate_id,
        assessment_date=assessment_date,
        raw_score=raw_total,
        max_score=max_total,
        normalised=norm,
        profile=profile,
        allocation=allocations[profile],
        axis_scores=axis_scores,
    )


def default_quiz() -> tuple[Question, ...]:
    """A 5-question default quiz spanning all axes."""
    return (
        Question(
            question_id="Q-DD",
            axis=QuestionAxis.DRAWDOWN_TOLERANCE,
            prompt="Your portfolio drops 20% in a month. What do you do?",
            options=(
                Option(key="a", label="Sell everything", score=0),
                Option(key="b", label="Sell half", score=1),
                Option(key="c", label="Hold and watch", score=2),
                Option(key="d", label="Buy more on dip", score=4),
            ),
        ),
        Question(
            question_id="Q-TH",
            axis=QuestionAxis.TIME_HORIZON,
            prompt="When will you need this money?",
            options=(
                Option(key="a", label="< 1 year", score=0),
                Option(key="b", label="1-3 years", score=1),
                Option(key="c", label="3-7 years", score=2),
                Option(key="d", label="7+ years", score=4),
            ),
        ),
        Question(
            question_id="Q-ID",
            axis=QuestionAxis.INCOME_DEPENDENCE,
            prompt="Do you depend on this portfolio for living expenses?",
            options=(
                Option(key="a", label="Yes, primarily", score=0),
                Option(key="b", label="Partially", score=2),
                Option(key="c", label="No", score=4),
            ),
        ),
        Question(
            question_id="Q-VR",
            axis=QuestionAxis.VOLATILITY_REACTION,
            prompt="A volatile day (5%+ moves). How do you feel?",
            options=(
                Option(key="a", label="Anxious; can't sleep", score=0),
                Option(key="b", label="Concerned but watching", score=2),
                Option(key="c", label="Comfortable — it's normal", score=4),
            ),
        ),
        Question(
            question_id="Q-KN",
            axis=QuestionAxis.KNOWLEDGE,
            prompt="How familiar are you with halal market structure?",
            options=(
                Option(key="a", label="New to it", score=0),
                Option(key="b", label="Some basic knowledge", score=2),
                Option(key="c", label="Practiced, comfortable with screening", score=3),
                Option(key="d", label="Advanced, including sukuk + structured", score=4),
            ),
        ),
    )


_PROFILE_EMOJI: dict[RiskProfile, str] = {
    RiskProfile.CONSERVATIVE: "🛡️",
    RiskProfile.BALANCED: "⚖️",
    RiskProfile.AGGRESSIVE: "🚀",
}


def _mask(candidate_id: str) -> str:
    if len(candidate_id) <= 4:
        return "***"
    return candidate_id[:2] + "…" + candidate_id[-2:]


def render_result(result: AssessmentResult) -> str:
    head = (
        f"{_PROFILE_EMOJI[result.profile]} {result.profile.value} "
        f"(score {result.normalised * 100:.1f}%, "
        f"raw {result.raw_score:.1f}/{result.max_score:.1f})\n"
        f"  Candidate {_mask(result.candidate_id)} on "
        f"{result.assessment_date.isoformat()}\n"
        f"  Allocation: equity {result.allocation.equity_min * 100:.0f}–"
        f"{result.allocation.equity_max * 100:.0f}% / sukuk "
        f"{result.allocation.sukuk_min * 100:.0f}–"
        f"{result.allocation.sukuk_max * 100:.0f}%"
    )
    return head
