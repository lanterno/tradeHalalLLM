"""Adversarial red-team agent — Round-5 Wave 8.B.

Complements the multi-agent committee (Wave 8.A) with a dedicated
**adversarial agent** whose role is to find counter-arguments to the
committee's proposed decision. The red-team agent never adds capital
to a stance; instead, it surfaces:

- Plausible reasons the committee's verdict could be wrong.
- Historical analogues where similar reasoning failed.
- Tail-risk scenarios the committee under-weighted.

This module ships the **red-team verdict aggregator** that sits
between the committee output + the executor: if the red team produces
strong counter-arguments and the committee's confidence is low, the
red team can VETO; otherwise it produces a structured caution note.

Pinned semantics:

- **Closed-set Concern ladder** (LOGICAL_FALLACY / OVERCONFIDENCE /
  HISTORICAL_ANALOGUE / TAIL_RISK / DATA_GAP).
- **Closed-set Stance** (PROCEED / CAUTION / VETO).
- **VETO threshold** is operator-tunable; defaults pin a high bar
  (committee confidence < 0.5 + concern severity > 0.7).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Concern(str, Enum):
    """Closed-set concern types."""

    LOGICAL_FALLACY = "logical_fallacy"
    OVERCONFIDENCE = "overconfidence"
    HISTORICAL_ANALOGUE = "historical_analogue"
    TAIL_RISK = "tail_risk"
    DATA_GAP = "data_gap"


class RedTeamStance(str, Enum):
    """Closed-set red-team stances."""

    PROCEED = "proceed"
    CAUTION = "caution"
    VETO = "veto"


@dataclass(frozen=True)
class RedTeamArgument:
    """A single counter-argument from the red team."""

    concern: Concern
    severity: float  # 0..1
    summary: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")


@dataclass(frozen=True)
class RedTeamPolicy:
    """Operator-tunable thresholds."""

    veto_severity_threshold: float = 0.7
    veto_committee_confidence_max: float = 0.5
    caution_severity_threshold: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 < self.caution_severity_threshold <= self.veto_severity_threshold <= 1.0:
            raise ValueError("0 < caution_severity_threshold <= veto_severity_threshold <= 1")
        if not 0.0 <= self.veto_committee_confidence_max <= 1.0:
            raise ValueError("veto_committee_confidence_max must be in [0, 1]")


@dataclass(frozen=True)
class RedTeamVerdict:
    """Aggregated red-team verdict on a committee decision."""

    stance: RedTeamStance
    arguments: tuple[RedTeamArgument, ...]
    max_severity: float
    committee_confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_severity <= 1.0:
            raise ValueError("max_severity must be in [0, 1]")
        if not 0.0 <= self.committee_confidence <= 1.0:
            raise ValueError("committee_confidence must be in [0, 1]")


def aggregate_redteam(
    arguments: Iterable[RedTeamArgument],
    *,
    committee_confidence: float,
    policy: RedTeamPolicy | None = None,
) -> RedTeamVerdict:
    """Aggregate red-team arguments into a verdict."""
    if not 0.0 <= committee_confidence <= 1.0:
        raise ValueError("committee_confidence must be in [0, 1]")
    pol = policy if policy is not None else RedTeamPolicy()
    args_t = tuple(arguments)

    max_severity = max((a.severity for a in args_t), default=0.0)

    if (
        max_severity >= pol.veto_severity_threshold
        and committee_confidence <= pol.veto_committee_confidence_max
    ):
        stance = RedTeamStance.VETO
    elif max_severity >= pol.caution_severity_threshold:
        stance = RedTeamStance.CAUTION
    else:
        stance = RedTeamStance.PROCEED

    return RedTeamVerdict(
        stance=stance,
        arguments=args_t,
        max_severity=max_severity,
        committee_confidence=committee_confidence,
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


def render_verdict(v: RedTeamVerdict) -> str:
    emoji = {
        RedTeamStance.PROCEED: "✅",
        RedTeamStance.CAUTION: "⚠️",
        RedTeamStance.VETO: "⛔",
    }[v.stance]
    head = (
        f"{emoji} red-team: {v.stance.value} "
        f"(max_severity={v.max_severity:.2f}, "
        f"committee_conf={v.committee_confidence:.2f})"
    )
    lines = [head]
    for arg in sorted(v.arguments, key=lambda a: -a.severity):
        lines.append(f"  • {arg.concern.value} (severity={arg.severity:.2f}): {arg.summary}")
    return _scrub("\n".join(lines))
