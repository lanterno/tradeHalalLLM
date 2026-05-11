"""Cross-agent contradiction detector — Round-5 Wave 8.F.

The multi-agent committee (`core/llm_committee.py`) aggregates votes
from Bull / Bear / Quant / Halal-judge / Macro / OperatorOverride.
Aggregation alone is insufficient: an operator deserves a heads-up
when *the votes themselves contradict each other in structurally
significant ways*. A 3-2 vote where the Quant says "high-conviction
SELL" while the Bull says "high-conviction BUY" should not silently
resolve to HOLD without the operator seeing the conflict.

This module is the **contradiction surfacer**. It does not change the
verdict; it returns a structured report the operator (and downstream
post-mortem) can use to understand *why* the committee was conflicted.

Contradiction classes:

- **STANCE**: two agents on opposite sides (BUY vs SELL); either vote
  has confidence ≥ `min_confidence_for_contradiction`.
- **CONFIDENCE_OUTLIER**: one agent's confidence is more than
  `outlier_factor` standard deviations above/below the rest.
- **HALAL_DISSENT**: the halal-judge votes SKIP while every other
  agent votes BUY/SELL — surfaces compliance disagreement.
- **QUANT_FUNDAMENTAL_GAP**: Quant says BUY but Bull/Bear/Macro
  collectively SKIP (or vice versa) — number-driven vs narrative
  disagreement.
- **CONFIDENCE_MISALIGNMENT**: weighted score's top stance is also the
  one that drew the highest *individual* confidence vote against it.

Pinned semantics:

- **Pure-functional.** Inputs in → report out; no state, no side-effects.
- **Closed-set ContradictionType ladder.** Adding a new type is
  intentional + tested.
- **Severity ladder**: NOTE / WARN / BLOCK. BLOCK is reserved for the
  HALAL_DISSENT case (compliance-significant); WARN for stance/quant
  gaps; NOTE for confidence outliers + misalignment.
- **No-secret-leak pin** on render — votes' rationale is not echoed
  verbatim (avoid prompt-injection bleed).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from halal_trader.core.llm_committee import AgentRole, AgentVote, Stance


class ContradictionType(str, Enum):
    """Closed-set ladder of detected contradiction types."""

    STANCE = "stance"
    CONFIDENCE_OUTLIER = "confidence_outlier"
    HALAL_DISSENT = "halal_dissent"
    QUANT_FUNDAMENTAL_GAP = "quant_fundamental_gap"
    CONFIDENCE_MISALIGNMENT = "confidence_misalignment"


class Severity(str, Enum):
    """Closed-set severity ladder."""

    NOTE = "note"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class Contradiction:
    """One detected contradiction between two or more agents."""

    type: ContradictionType
    severity: Severity
    summary: str
    involved_roles: tuple[AgentRole, ...]


@dataclass(frozen=True)
class ContradictionReport:
    """Output of `detect`."""

    contradictions: tuple[Contradiction, ...]

    def has_block(self) -> bool:
        return any(c.severity is Severity.BLOCK for c in self.contradictions)

    def has_warn(self) -> bool:
        return any(c.severity is Severity.WARN for c in self.contradictions)

    def by_severity(self, sev: Severity) -> tuple[Contradiction, ...]:
        return tuple(c for c in self.contradictions if c.severity is sev)


_OPPOSITES: dict[Stance, Stance] = {
    Stance.BUY: Stance.SELL,
    Stance.SELL: Stance.BUY,
}


def _detect_stance_conflicts(
    votes: Sequence[AgentVote], min_confidence: float
) -> list[Contradiction]:
    """Surface BUY vs SELL pairs where either side has high confidence."""
    out: list[Contradiction] = []
    seen: set[tuple[AgentRole, AgentRole]] = set()
    for i, a in enumerate(votes):
        for j, b in enumerate(votes):
            if i >= j:
                continue
            if (
                a.stance in _OPPOSITES
                and b.stance == _OPPOSITES[a.stance]
                and (a.confidence >= min_confidence or b.confidence >= min_confidence)
            ):
                sorted_pair = sorted((a.role, b.role), key=lambda r: r.value)
                key: tuple[AgentRole, AgentRole] = (sorted_pair[0], sorted_pair[1])
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Contradiction(
                        type=ContradictionType.STANCE,
                        severity=Severity.WARN,
                        summary=(
                            f"{a.role.value}({a.stance.value} c={a.confidence:.2f}) "
                            f"vs {b.role.value}({b.stance.value} c={b.confidence:.2f})"
                        ),
                        involved_roles=(a.role, b.role),
                    )
                )
    return out


def _detect_confidence_outliers(votes: Sequence[AgentVote], factor: float) -> list[Contradiction]:
    """Surface votes whose confidence is `factor`σ from the mean.

    Requires ≥ 3 votes (σ undefined / unstable below that)."""
    if len(votes) < 3:
        return []
    confs = [v.confidence for v in votes]
    mean = sum(confs) / len(confs)
    var = sum((c - mean) ** 2 for c in confs) / max(1, len(confs) - 1)
    std = var**0.5
    if std < 1e-6:
        return []
    out: list[Contradiction] = []
    for v in votes:
        z = (v.confidence - mean) / std
        if abs(z) >= factor:
            out.append(
                Contradiction(
                    type=ContradictionType.CONFIDENCE_OUTLIER,
                    severity=Severity.NOTE,
                    summary=(
                        f"{v.role.value} confidence {v.confidence:.2f} "
                        f"({z:+.2f}σ from mean {mean:.2f})"
                    ),
                    involved_roles=(v.role,),
                )
            )
    return out


def _detect_halal_dissent(
    votes: Sequence[AgentVote],
) -> list[Contradiction]:
    """Halal-judge SKIPs while ≥ 1 other agent votes BUY/SELL → BLOCK."""
    halal = [v for v in votes if v.role is AgentRole.HALAL_JUDGE]
    if not halal:
        return []
    halal_skips = [v for v in halal if v.stance is Stance.SKIP]
    if not halal_skips:
        return []
    other_active = [
        v
        for v in votes
        if v.role is not AgentRole.HALAL_JUDGE and v.stance in (Stance.BUY, Stance.SELL)
    ]
    if not other_active:
        return []
    return [
        Contradiction(
            type=ContradictionType.HALAL_DISSENT,
            severity=Severity.BLOCK,
            summary=(f"halal-judge SKIPs while {len(other_active)} other(s) want BUY/SELL"),
            involved_roles=tuple(v.role for v in halal_skips + other_active),
        )
    ]


def _detect_quant_fundamental_gap(
    votes: Sequence[AgentVote],
) -> list[Contradiction]:
    """Quant says BUY/SELL but Bull/Bear/Macro collectively dissent.

    The "fundamental" group = Bull, Bear, Macro. If the Quant is
    decisive (BUY or SELL with conf > 0.5) while the other narrative
    agents are all in HOLD/SKIP, surface as WARN.
    """
    quant_votes = [v for v in votes if v.role is AgentRole.QUANT]
    if not quant_votes:
        return []
    quant = quant_votes[0]
    if quant.stance not in (Stance.BUY, Stance.SELL):
        return []
    if quant.confidence <= 0.5:
        return []
    fund_roles = {AgentRole.BULL, AgentRole.BEAR, AgentRole.MACRO}
    fund_votes = [v for v in votes if v.role in fund_roles]
    if not fund_votes:
        return []
    if all(v.stance in (Stance.HOLD, Stance.SKIP) for v in fund_votes):
        return [
            Contradiction(
                type=ContradictionType.QUANT_FUNDAMENTAL_GAP,
                severity=Severity.WARN,
                summary=(
                    f"quant decisive {quant.stance.value} c={quant.confidence:.2f} "
                    f"vs fundamental agents all HOLD/SKIP"
                ),
                involved_roles=(quant.role, *(v.role for v in fund_votes)),
            )
        ]
    return []


def _detect_confidence_misalignment(
    votes: Sequence[AgentVote],
) -> list[Contradiction]:
    """If the *highest individual confidence* vote disagrees with the
    plurality stance, surface a NOTE.

    Captures cases where many low-conviction votes overwhelm one
    high-conviction dissent.
    """
    if not votes:
        return []
    plurality_count: dict[Stance, int] = {s: 0 for s in Stance}
    for v in votes:
        plurality_count[v.stance] += 1
    plurality = max(plurality_count.items(), key=lambda kv: kv[1])[0]
    top_vote = max(votes, key=lambda v: v.confidence)
    if top_vote.stance is plurality:
        return []
    if top_vote.confidence < 0.7:
        return []
    return [
        Contradiction(
            type=ContradictionType.CONFIDENCE_MISALIGNMENT,
            severity=Severity.NOTE,
            summary=(
                f"plurality={plurality.value} but highest-confidence vote "
                f"({top_vote.role.value} c={top_vote.confidence:.2f}) "
                f"says {top_vote.stance.value}"
            ),
            involved_roles=(top_vote.role,),
        )
    ]


def detect(
    votes: Iterable[AgentVote],
    *,
    min_confidence_for_stance_conflict: float = 0.5,
    confidence_outlier_factor: float = 2.0,
) -> ContradictionReport:
    """Run all detectors over a vote set and return the consolidated report.

    Detectors are independent + side-effect-free; the union of their
    outputs is the report. Order is deterministic (per detector).
    """
    if not 0.0 <= min_confidence_for_stance_conflict <= 1.0:
        raise ValueError("min_confidence_for_stance_conflict must be in [0, 1]")
    if confidence_outlier_factor <= 0:
        raise ValueError("confidence_outlier_factor must be positive")
    votes_t = tuple(votes)
    if not votes_t:
        return ContradictionReport(contradictions=tuple())
    contradictions: list[Contradiction] = []
    contradictions.extend(_detect_halal_dissent(votes_t))
    contradictions.extend(_detect_stance_conflicts(votes_t, min_confidence_for_stance_conflict))
    contradictions.extend(_detect_quant_fundamental_gap(votes_t))
    contradictions.extend(_detect_confidence_outliers(votes_t, confidence_outlier_factor))
    contradictions.extend(_detect_confidence_misalignment(votes_t))
    return ContradictionReport(contradictions=tuple(contradictions))


_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.NOTE: "ℹ️",
    Severity.WARN: "⚠️",
    Severity.BLOCK: "🛑",
}


def render_report(report: ContradictionReport) -> str:
    """Operator-readable summary of the report.

    The vote rationales are *not* echoed verbatim — only the structured
    summary lines are emitted. This keeps the renderer prompt-injection
    safe even when LLM-generated rationales are pasted upstream.
    """
    if not report.contradictions:
        return "✅ No contradictions detected."
    lines = [f"🔍 {len(report.contradictions)} contradiction(s) detected:"]
    for c in report.contradictions:
        emoji = _SEVERITY_EMOJI.get(c.severity, "•")
        lines.append(f"  {emoji} [{c.severity.value.upper()}] {c.type.value}: {c.summary}")
    return "\n".join(lines)
