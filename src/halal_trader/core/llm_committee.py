"""Multi-agent LLM trading committee — Round-5 Wave 8.A.

The bot already has a single-LLM cycle plus a Round-4 Wave 4.J
``core/committee.py`` that aggregates ensemble decisions of equivalent
prompts. Round-5 Wave 8.A adds a different cut: a **role-specialized
committee** where each agent has a distinct prompt persona (Bull /
Bear / Quant / Halal-judge / Macro). Each role argues its case; an
aggregator weighs them under operator-tunable rules to produce the
committee verdict.

This module is the **structural aggregator**, not the LLM client. It
takes a tuple of :class:`AgentVote` objects + a policy and produces
a :class:`CommitteeVerdict`. Sourcing the votes (calling each LLM
with its specialized prompt) lives in the cycle layer.

Pinned semantics:

- **Closed-set AgentRole ladder.** Adding a role is a code review
  change.
- **Closed-set Stance ladder** (BUY / HOLD / SELL / SKIP).
- **Halal-judge VETO.** A SKIP vote from the HALAL_JUDGE role
  trumps every other vote — the committee verdict is SKIP regardless
  of weighted aggregation. Pinned in tests.
- **Tied weighted votes resolve to HOLD** (conservative default).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(str, Enum):
    """Closed-set agent roles."""

    BULL = "bull"
    BEAR = "bear"
    QUANT = "quant"
    HALAL_JUDGE = "halal_judge"
    MACRO = "macro"
    OPERATOR_OVERRIDE = "operator_override"


class Stance(str, Enum):
    """Closed-set stance ladder."""

    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    SKIP = "skip"


@dataclass(frozen=True)
class AgentVote:
    """A single agent's vote on the committee."""

    role: AgentRole
    stance: Stance
    confidence: float  # in [0, 1]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class CommitteePolicy:
    """Operator-tunable aggregation policy."""

    role_weights: Mapping[AgentRole, float] = field(
        default_factory=lambda: {
            AgentRole.BULL: 1.0,
            AgentRole.BEAR: 1.0,
            AgentRole.QUANT: 1.5,
            AgentRole.HALAL_JUDGE: 2.0,
            AgentRole.MACRO: 1.0,
            AgentRole.OPERATOR_OVERRIDE: 5.0,
        }
    )
    halal_judge_veto_on_skip: bool = True
    require_quorum: int = 3  # min votes for a non-SKIP verdict

    def __post_init__(self) -> None:
        for role, w in self.role_weights.items():
            if w < 0:
                raise ValueError(f"weight for {role.value} must be non-negative")
        if self.require_quorum <= 0:
            raise ValueError("require_quorum must be positive")


@dataclass(frozen=True)
class CommitteeVerdict:
    """The committee's aggregated decision."""

    stance: Stance
    confidence: float
    votes: tuple[AgentVote, ...]
    veto_invoked: bool
    weighted_scores: Mapping[Stance, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


def aggregate(
    votes: Iterable[AgentVote],
    *,
    policy: CommitteePolicy | None = None,
) -> CommitteeVerdict:
    """Aggregate role-specialized votes into a committee verdict."""
    pol = policy if policy is not None else CommitteePolicy()
    votes_t = tuple(votes)

    # Quorum check first
    if len(votes_t) < pol.require_quorum:
        return CommitteeVerdict(
            stance=Stance.SKIP,
            confidence=0.0,
            votes=votes_t,
            veto_invoked=False,
            weighted_scores={s: 0.0 for s in Stance},
        )

    # Halal-judge veto check
    if pol.halal_judge_veto_on_skip:
        for v in votes_t:
            if v.role is AgentRole.HALAL_JUDGE and v.stance is Stance.SKIP:
                return CommitteeVerdict(
                    stance=Stance.SKIP,
                    confidence=v.confidence,
                    votes=votes_t,
                    veto_invoked=True,
                    weighted_scores={s: 0.0 for s in Stance},
                )

    # Weighted aggregation: each vote contributes role_weight * confidence to its stance.
    scores: dict[Stance, float] = {s: 0.0 for s in Stance}
    for v in votes_t:
        weight = pol.role_weights.get(v.role, 1.0)
        scores[v.stance] += weight * v.confidence

    # Pick the highest-scoring stance; tie → HOLD (conservative).
    sorted_stances = sorted(scores.items(), key=lambda kv: -kv[1])
    top_stance, top_score = sorted_stances[0]
    if len(sorted_stances) > 1 and abs(sorted_stances[1][1] - top_score) < 1e-9:
        top_stance = Stance.HOLD

    # Confidence: top score normalised by sum of all positive weights, scaled.
    total_weight = sum(pol.role_weights.get(v.role, 1.0) for v in votes_t)
    confidence = top_score / total_weight if total_weight > 0 else 0.0
    confidence = max(0.0, min(1.0, confidence))

    return CommitteeVerdict(
        stance=top_stance,
        confidence=confidence,
        votes=votes_t,
        veto_invoked=False,
        weighted_scores=dict(scores),
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_verdict(verdict: CommitteeVerdict) -> str:
    flag = " [HALAL VETO]" if verdict.veto_invoked else ""
    head = (
        f"Committee verdict: {verdict.stance.value.upper()} "
        f"(conf={verdict.confidence:.2f}){flag}"
    )
    lines = [head]
    for v in verdict.votes:
        lines.append(
            f"  • {v.role.value:18s} → {v.stance.value:5s} "
            f"conf={v.confidence:.2f}"
        )
    return _scrub("\n".join(lines))
