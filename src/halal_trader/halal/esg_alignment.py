"""ESG + halal alignment scorer — Round-5 Wave 11.H.

Halal screening + ESG (environmental / social / governance) screening
have substantial overlap (no tobacco, no weapons, etc.) but diverge
on some axes (Sharia screen does not consider carbon emissions; ESG
typically tolerates conventional banks). This module ships the
**alignment scorer** that produces a unified score reflecting both.

Pinned semantics:

- **Closed-set Pillar ladder** (ENVIRONMENTAL / SOCIAL / GOVERNANCE /
  HALAL).
- **Composite score** is a weighted average of pillar scores.
- **Closed-set AlignmentTier ladder** (LEADER / ALIGNED / NEUTRAL /
  MISALIGNED).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum


class Pillar(str, Enum):
    """Closed-set ESG + halal pillars."""

    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    HALAL = "halal"


class AlignmentTier(str, Enum):
    """Closed-set alignment tiers."""

    LEADER = "leader"
    ALIGNED = "aligned"
    NEUTRAL = "neutral"
    MISALIGNED = "misaligned"


@dataclass(frozen=True)
class AlignmentPolicy:
    """Operator-tunable weights per pillar."""

    weights: Mapping[Pillar, float] = field(
        default_factory=lambda: {
            Pillar.ENVIRONMENTAL: 0.25,
            Pillar.SOCIAL: 0.25,
            Pillar.GOVERNANCE: 0.20,
            Pillar.HALAL: 0.30,
        }
    )
    leader_threshold: float = 0.85
    aligned_threshold: float = 0.65
    neutral_threshold: float = 0.40

    def __post_init__(self) -> None:
        for p, w in self.weights.items():
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"weight for {p.value} must be in [0, 1]")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("pillar weights must sum to 1.0")
        if not 0.0 < self.neutral_threshold < self.aligned_threshold < self.leader_threshold < 1.0:
            raise ValueError(
                "thresholds must be strictly increasing in (0, 1)"
            )


@dataclass(frozen=True)
class PillarScore:
    """Score for a single pillar in [0, 1]."""

    pillar: Pillar
    score: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")


@dataclass(frozen=True)
class AlignmentReport:
    """Composite alignment report."""

    issuer: str
    pillar_scores: tuple[PillarScore, ...]
    composite_score: float
    tier: AlignmentTier

    def __post_init__(self) -> None:
        if not self.issuer or not self.issuer.strip():
            raise ValueError("issuer must be non-empty")
        if not 0.0 <= self.composite_score <= 1.0:
            raise ValueError("composite_score must be in [0, 1]")


def score_alignment(
    issuer: str,
    pillar_scores: Iterable[PillarScore],
    *,
    policy: AlignmentPolicy | None = None,
) -> AlignmentReport:
    """Compute composite alignment + tier."""
    if not issuer.strip():
        raise ValueError("issuer must be non-empty")
    pol = policy if policy is not None else AlignmentPolicy()
    scores_t = tuple(pillar_scores)

    if not scores_t:
        return AlignmentReport(
            issuer=issuer,
            pillar_scores=(),
            composite_score=0.0,
            tier=AlignmentTier.NEUTRAL,
        )

    # All pillars must be unique
    if len({s.pillar for s in scores_t}) != len(scores_t):
        raise ValueError("each pillar may appear at most once")

    composite = 0.0
    weight_used = 0.0
    for s in scores_t:
        w = pol.weights.get(s.pillar, 0.0)
        composite += s.score * w
        weight_used += w

    if weight_used > 0:
        composite = composite / weight_used  # normalise to total weight present

    if composite >= pol.leader_threshold:
        tier = AlignmentTier.LEADER
    elif composite >= pol.aligned_threshold:
        tier = AlignmentTier.ALIGNED
    elif composite >= pol.neutral_threshold:
        tier = AlignmentTier.NEUTRAL
    else:
        tier = AlignmentTier.MISALIGNED

    return AlignmentReport(
        issuer=issuer,
        pillar_scores=scores_t,
        composite_score=composite,
        tier=tier,
    )


def render_report(r: AlignmentReport) -> str:
    emoji = {
        AlignmentTier.LEADER: "🌟",
        AlignmentTier.ALIGNED: "✅",
        AlignmentTier.NEUTRAL: "⚪",
        AlignmentTier.MISALIGNED: "⚠️",
    }[r.tier]
    head = (
        f"{emoji} {r.issuer}: {r.tier.value} "
        f"(composite={r.composite_score:.2f})"
    )
    lines = [head]
    for s in sorted(r.pillar_scores, key=lambda x: x.pillar.value):
        lines.append(
            f"  • {s.pillar.value}: {s.score:.2f}"
            + (f" — {s.rationale}" if s.rationale else "")
        )
    return "\n".join(lines)
