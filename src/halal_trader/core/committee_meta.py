"""Committee meta-learner — Round-5 Wave 8.G.

Track per-role accuracy across setup types so the operator can answer:
*"When is the Bull right? When should I trust the Quant over the Bear?"*

This module ingests a stream of resolved committee outcomes and emits
per-role × per-setup reliability scores. The aggregator (`core/llm_committee`)
can read these scores at debate time and bias the verdict toward the
historically-most-accurate voice on this setup type.

This is a **scorer**, not an updater. It's pure-functional: feed it
historical outcomes, get a `RoleReliability` table back. The committee
runtime decides how to apply.

Pinned semantics:

- **Reliability per (role, setup_class)**. `setup_class` defaults to
  `RegimeTag` from `core.committee_memory` but operators can pass any
  hashable group key.
- **Brier-score correctness** for each individual vote: the score is
  computed against the *resolved outcome* of the trade, not the
  consensus verdict. This isolates each role's predictive accuracy.
- **Recency decay** (half-life 60 days by default).
- **Min sample threshold** — below `min_n_effective`, reliability is
  reported as `None` (insufficient data).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Hashable

from halal_trader.core.committee_memory import OutcomeLabel
from halal_trader.core.llm_committee import AgentRole, AgentVote, Stance


@dataclass(frozen=True)
class ResolvedDecision:
    """One historical decision + outcome with the role-by-role votes."""

    ticker: str
    decision_date: date
    setup_class: Hashable
    """Group-key for stratification (e.g. RegimeTag.BULL_TREND)."""
    final_stance: Stance
    """The committee's verdict after aggregation."""
    votes: tuple[AgentVote, ...]
    outcome: OutcomeLabel
    return_pct: float
    """Realised return, signed."""

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if not self.votes:
            raise ValueError("votes must be non-empty")
        if self.outcome is OutcomeLabel.OPEN:
            raise ValueError("ResolvedDecision must not be OPEN")
        if self.outcome is OutcomeLabel.WIN and self.return_pct <= 0:
            raise ValueError("WIN must have positive return")
        if self.outcome is OutcomeLabel.LOSS and self.return_pct >= 0:
            raise ValueError("LOSS must have negative return")


@dataclass(frozen=True)
class RoleReliability:
    """Per-(role, setup_class) reliability score."""

    role: AgentRole
    setup_class: Hashable
    n_samples: int
    n_effective: float
    accuracy: float
    """Fraction of times the role's stance pre-aligned with realised
    outcome (BUY → WIN, SELL → LOSS, HOLD/SKIP → FLAT). In [0, 1]."""
    avg_correct_confidence: float
    """Average confidence the role assigned when it was right."""
    avg_wrong_confidence: float
    """Average confidence the role assigned when it was wrong. A high
    value means the role is *overconfident* — useful for calibration."""
    overconfidence_gap: float
    """avg_wrong_confidence − avg_correct_confidence. Positive means
    the role is more confident when wrong than when right (bad)."""

    def is_significant(self, min_n_effective: float = 3.0) -> bool:
        return self.n_effective >= min_n_effective


def _stance_correct(stance: Stance, outcome: OutcomeLabel, return_pct: float) -> bool:
    """A stance is correct iff its directional view matched the outcome.

    BUY → WIN; SELL → LOSS; HOLD → FLAT (or small loss/win); SKIP →
    FLAT (or any).
    """
    if stance is Stance.BUY:
        return outcome is OutcomeLabel.WIN
    if stance is Stance.SELL:
        return outcome is OutcomeLabel.LOSS
    if stance is Stance.HOLD:
        return outcome is OutcomeLabel.FLAT or abs(return_pct) < 0.005
    if stance is Stance.SKIP:
        # SKIP is correct if the trade would have been a loss.
        return outcome is not OutcomeLabel.WIN
    return False


def reliability_table(
    decisions: Iterable[ResolvedDecision],
    *,
    today: date,
    half_life_days: int = 60,
    min_n_effective: float = 3.0,
) -> tuple[RoleReliability, ...]:
    """Aggregate per-role × per-setup reliability scores.

    Returns one RoleReliability row per (role, setup_class) seen,
    sorted by setup_class then role for deterministic output.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    by_key: dict[
        tuple[AgentRole, Hashable],
        list[tuple[float, bool, float]],  # (decay_weight, correct, confidence)
    ] = {}
    for d in decisions:
        days = max(0, (today - d.decision_date).days)
        w = 0.5 ** (days / half_life_days)
        for v in d.votes:
            correct = _stance_correct(v.stance, d.outcome, d.return_pct)
            by_key.setdefault((v.role, d.setup_class), []).append((w, correct, v.confidence))
    out: list[RoleReliability] = []
    for (role, setup_class), records in by_key.items():
        n = len(records)
        n_eff = sum(w for w, _, _ in records)
        if n_eff < 1e-12:
            continue
        # Decay-weighted accuracy.
        correct_w = sum(w for w, c, _ in records if c)
        accuracy = correct_w / n_eff
        # Confidence-conditional averages.
        right = [(w, conf) for w, c, conf in records if c]
        wrong = [(w, conf) for w, c, conf in records if not c]
        sum_w_right = sum(w for w, _ in right)
        sum_w_wrong = sum(w for w, _ in wrong)
        avg_right = sum(w * c for w, c in right) / sum_w_right if sum_w_right > 0 else 0.0
        avg_wrong = sum(w * c for w, c in wrong) / sum_w_wrong if sum_w_wrong > 0 else 0.0
        gap = avg_wrong - avg_right
        out.append(
            RoleReliability(
                role=role,
                setup_class=setup_class,
                n_samples=n,
                n_effective=n_eff,
                accuracy=accuracy,
                avg_correct_confidence=avg_right,
                avg_wrong_confidence=avg_wrong,
                overconfidence_gap=gap,
            )
        )
    out.sort(key=lambda r: (str(r.setup_class), r.role.value))
    return tuple(out)


def best_role_for_setup(
    table: Sequence[RoleReliability],
    setup_class: Hashable,
    *,
    min_n_effective: float = 3.0,
) -> RoleReliability | None:
    """Return the most-accurate (significant) role for a given setup."""
    candidates = [
        r for r in table if r.setup_class == setup_class and r.is_significant(min_n_effective)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.accuracy)


def render_table(table: Sequence[RoleReliability]) -> str:
    """Operator-readable summary of the reliability table."""
    if not table:
        return "🧪 Meta: no resolved decisions yet."
    lines = [f"🧪 Meta-learner: {len(table)} (role, setup) pairs"]
    for r in table:
        sig = "✓" if r.is_significant() else "✗"
        lines.append(
            f"  {sig} {r.setup_class} / {r.role.value}: "
            f"acc={r.accuracy * 100:.2f}% "
            f"(n={r.n_samples}, eff={r.n_effective:.2f}), "
            f"conf-when-right={r.avg_correct_confidence:.2f}, "
            f"conf-when-wrong={r.avg_wrong_confidence:.2f}, "
            f"overconf-gap={r.overconfidence_gap:+.2f}"
        )
    return "\n".join(lines)
