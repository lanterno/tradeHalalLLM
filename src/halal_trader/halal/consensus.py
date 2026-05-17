"""Multi-source halal-screening consensus.

Round-4 wave 2.B: generalise the two-source corroborator
(`halal/corroborate.py`) into an N-source consensus aggregator that
also handles the three-state decision (``halal`` / ``not_halal`` /
``doubtful``) the existing :class:`HalalScreening` audit row uses.

Why this is its own module rather than an extension of
`corroborate.py`:

* `corroborate.py` is binary (halal vs not-halal) and assumes
  exactly two sources. Generalising it would change its public
  shape; keeping it stable preserves callers that already adopt it.
* The consensus logic here is *opinion-shaped* — it operates on a
  list of dataclass records, not on the screener Protocols. That
  makes it independently testable without async / network mocks
  (just feed it ``[ScreeningOpinion(…)]``) and lets a future SQL-
  side aggregator pull cached opinions out of `HalalScreening`
  rows without re-querying the providers.

Three resolution policies:

* :class:`ConsensusPolicy.STRICT` (default) — any ``not_halal``
  forces ``not_halal``; any ``doubtful`` (and no ``not_halal``)
  forces ``doubtful``. Only an unanimous ``halal`` cohort yields
  ``halal``. This matches the **conservative-default** stance the
  rest of the bot takes.
* :class:`ConsensusPolicy.MAJORITY` — most common decision wins;
  ties resolve to the more conservative side
  (``not_halal > doubtful > halal``).
* :class:`ConsensusPolicy.WEIGHTED` — each opinion carries a
  ``weight`` (defaults to 1.0 if unspecified). Weights sum per
  decision and the largest sum wins; ties resolve to the more
  conservative side.

Every result records the dissent: which sources said what, and a
human-readable reason string. The audit row's ``criteria`` JSONB
field can stash the consensus payload verbatim.

Halal alignment: the module *defaults* to STRICT on purpose — the
"any provider rejecting → reject" rule is the safest interpretation
when scholars themselves can disagree. An operator who explicitly
opts into MAJORITY / WEIGHTED takes responsibility for the choice
(and can record their reasoning in the audit trail).

Pure-Python; no DB, no async. The aggregator is a single function
on a list of dataclasses — easy to unit-test, easy to call from
either the cycle (after fan-out) or a backfill script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ── Decision vocabulary ───────────────────────────────────


class Decision(str, Enum):
    """The three-state decision the audit row uses.

    Ordering is from most-permissive (HALAL) to most-restrictive
    (NOT_HALAL). The tiebreak rule "more conservative wins" walks
    in the opposite direction (high → low).
    """

    HALAL = "halal"
    DOUBTFUL = "doubtful"
    NOT_HALAL = "not_halal"


# Conservatism rank — higher = more restrictive. Used as the
# tiebreaker when two decisions share the same vote count.
_CONSERVATISM_RANK: dict[Decision, int] = {
    Decision.HALAL: 0,
    Decision.DOUBTFUL: 1,
    Decision.NOT_HALAL: 2,
}


def _coerce_decision(value: str | Decision) -> Decision:
    """Accept either the enum or the underlying string. Pin so a
    caller passing a JSON-loaded dict (which loses the enum) can
    still feed it in without conversion friction."""
    if isinstance(value, Decision):
        return value
    try:
        return Decision(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown halal decision {value!r}; expected one of {[d.value for d in Decision]}"
        ) from exc


# ── Inputs / Outputs ──────────────────────────────────────


@dataclass(frozen=True)
class ScreeningOpinion:
    """One provider's read on a single symbol.

    ``source`` is the audit-row provenance label (e.g. ``"zoya"``,
    ``"musaffa"``, ``"idealratings"``, ``"manual_override"``).
    ``decision`` is the three-state value. ``weight`` is only used
    by ``WEIGHTED`` policy; ignored otherwise.
    ``criteria`` is the per-provider raw payload — debt ratio,
    income ratio, sector, etc. — passed through unchanged so the
    consensus result can echo it for the audit trail.
    """

    source: str
    decision: Decision | str
    weight: float = 1.0
    criteria: dict | None = None


@dataclass(frozen=True)
class ConsensusDecision:
    """Final decision plus enough provenance to defend it later.

    ``decision`` is the resolved three-state value.
    ``policy`` is the rule that produced it.
    ``opinions`` is the unmodified list of inputs (in input order).
    ``reason`` is a one-sentence operator-readable summary
    suitable for the dashboard's compliance tile or a notifier
    payload.
    """

    decision: Decision
    policy: "ConsensusPolicy"
    opinions: list[ScreeningOpinion]
    reason: str
    dissenters: list[str] = field(default_factory=list)


class ConsensusPolicy(str, Enum):
    """How to combine multiple opinions into a single decision."""

    STRICT = "strict"
    MAJORITY = "majority"
    WEIGHTED = "weighted"


# ── Resolution ────────────────────────────────────────────


def _resolve_strict(opinions: list[ScreeningOpinion]) -> tuple[Decision, str]:
    """STRICT: any not_halal → not_halal; any doubtful → doubtful;
    else halal. Pin the precedence so a refactor can't accidentally
    soften the conservative default."""
    decisions = [_coerce_decision(o.decision) for o in opinions]
    if Decision.NOT_HALAL in decisions:
        haram = [o.source for o, d in zip(opinions, decisions) if d == Decision.NOT_HALAL]
        return (
            Decision.NOT_HALAL,
            f"strict: {', '.join(haram)} flagged not_halal — rejecting",
        )
    if Decision.DOUBTFUL in decisions:
        doubt = [o.source for o, d in zip(opinions, decisions) if d == Decision.DOUBTFUL]
        return (
            Decision.DOUBTFUL,
            f"strict: {', '.join(doubt)} flagged doubtful — defaulting to doubtful",
        )
    return Decision.HALAL, "strict: all sources concurred halal"


def _resolve_majority(opinions: list[ScreeningOpinion]) -> tuple[Decision, str]:
    """MAJORITY: most common decision wins; ties → more conservative."""
    decisions = [_coerce_decision(o.decision) for o in opinions]
    counts: dict[Decision, int] = {d: 0 for d in Decision}
    for d in decisions:
        counts[d] += 1
    return _pick_top(counts, len(opinions), policy_label="majority")


def _resolve_weighted(opinions: list[ScreeningOpinion]) -> tuple[Decision, str]:
    """WEIGHTED: per-decision weight sums; largest wins; ties →
    more conservative. A non-positive weight is normalised to 0
    (degenerates to abstention) so a buggy config can't silently
    flip a result."""
    decisions = [_coerce_decision(o.decision) for o in opinions]
    sums: dict[Decision, float] = {d: 0.0 for d in Decision}
    total = 0.0
    for opinion, d in zip(opinions, decisions):
        w = max(0.0, opinion.weight)
        sums[d] += w
        total += w
    return _pick_top(sums, total, policy_label="weighted")


def _pick_top(
    scores: dict[Decision, float], total: float, *, policy_label: str
) -> tuple[Decision, str]:
    """Pick the highest-scoring decision; on ties, pick the most
    conservative. Empty / zero-total inputs default to DOUBTFUL —
    "no opinions" is not "approved"."""
    if total <= 0:
        return (
            Decision.DOUBTFUL,
            f"{policy_label}: no opinions / zero weight — defaulting to doubtful",
        )
    max_score = max(scores.values())
    candidates = [d for d, s in scores.items() if s == max_score]
    # Tiebreak: most conservative.
    winner = max(candidates, key=lambda d: _CONSERVATISM_RANK[d])
    if len(candidates) > 1:
        reason = (
            f"{policy_label}: tie at {max_score} between "
            f"{[c.value for c in candidates]}; "
            f"resolving to {winner.value} (more conservative)"
        )
    else:
        reason = f"{policy_label}: {winner.value} won with score {max_score}"
    return winner, reason


_RESOLVERS = {
    ConsensusPolicy.STRICT: _resolve_strict,
    ConsensusPolicy.MAJORITY: _resolve_majority,
    ConsensusPolicy.WEIGHTED: _resolve_weighted,
}


def consensus(
    opinions: list[ScreeningOpinion],
    *,
    policy: ConsensusPolicy = ConsensusPolicy.STRICT,
) -> ConsensusDecision:
    """Resolve a list of provider opinions into one decision.

    Empty input returns a ``DOUBTFUL`` consensus rather than raising
    — the caller's contract should be "no opinions = unattested =
    refuse to trade", which the upstream gate already treats as
    blocking. Raising here would force every caller to handle the
    edge separately.
    """
    if not opinions:
        return ConsensusDecision(
            decision=Decision.DOUBTFUL,
            policy=policy,
            opinions=[],
            reason="no opinions provided — defaulting to doubtful",
            dissenters=[],
        )

    decision, reason = _RESOLVERS[policy](opinions)

    dissenters = [o.source for o in opinions if _coerce_decision(o.decision) != decision]

    return ConsensusDecision(
        decision=decision,
        policy=policy,
        opinions=list(opinions),
        reason=reason,
        dissenters=dissenters,
    )


__all__ = [
    "ConsensusDecision",
    "ConsensusPolicy",
    "Decision",
    "ScreeningOpinion",
    "consensus",
]
