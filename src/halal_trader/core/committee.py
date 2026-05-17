"""Multi-variant decision aggregator for the shadow committee.

Round-4 wave 4.J: the existing `core/shadow_runner.py` runs one
*alternate* strategy in parallel. This module is the next step:
multiple strategy variants (GPT-4o + Claude + Llama + RL policy +
GA-evolved) all run in shadow per cycle, and a committee vote
decides the live trade.

The committee is **pure aggregation** — it doesn't run the
variants, doesn't call any model, doesn't open positions. It
takes a list of `VariantDecision`s per pair and returns a
`CommitteeVerdict` with the aggregated action plus per-variant
attribution. The dashboard renders the attribution as "this trade
was 4-for-1 BUY (GPT-4o + Claude + Llama + GA voted BUY; RL
voted HOLD)" so operators can audit the committee's behaviour
over time.

Three voting policies share the same conservative-tiebreak
philosophy used elsewhere in the codebase (Wave 2.B / 4.F):

* **Majority** — most-common action wins; ties resolve to the
  more conservative side (HOLD > SELL > BUY — pin: HOLD is the
  "do nothing" safe default; refusing a BUY when variants split
  evenly avoids low-conviction entries).
* **Weighted** — each variant has a weight (typically tied to
  its rolling Sharpe via the dashboard's accuracy ledger); sums
  per action; largest sum wins; ties → conservative.
* **Unanimous** — every variant must agree on a non-HOLD action,
  else HOLD. The strictest mode; useful when an operator wants
  the highest-conviction trades only.

Each verdict carries:

* The chosen action + the per-variant breakdown.
* The aggregate confidence — for BUYs, the *minimum* confidence
  across the agreeing variants (not the max — pin: a 3-of-5 BUY
  where the lowest agreeing variant says 0.4 should size off
  0.4, not 0.9).
* The committee's quantity — median across the agreeing
  variants (more robust to one variant proposing an outsized
  bet than mean).

Halal alignment: voting is informational. The committee never
opens a position or screens an asset; the strategy / executor
downstream are unchanged. Halal screening still gates every
candidate before the committee even sees it.

Pure-Python; no NumPy / DB / async / LLM. Operates on plain
dataclasses so the committee can be tested without the full
strategy stack.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum

# ── Vocabulary ────────────────────────────────────────────


class Action(str, Enum):
    """The three trade actions a variant can recommend.

    Ordering pinned for the conservative tiebreak: HOLD is the
    safest "do nothing"; SELL closes risk; BUY opens new risk.
    `_CONSERVATISM_RANK` below uses this ordering.
    """

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


# Rank used for tiebreaks. Higher rank = more conservative.
# Pin: HOLD wins over SELL wins over BUY when variants tie —
# refusing to open new risk on a split is the safer default.
_CONSERVATISM_RANK: dict[Action, int] = {
    Action.BUY: 0,
    Action.SELL: 1,
    Action.HOLD: 2,
}


class VotingPolicy(str, Enum):
    """How variant decisions combine into a committee verdict."""

    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"


# ── Inputs ────────────────────────────────────────────────


@dataclass(frozen=True)
class VariantDecision:
    """One variant's recommendation for one pair.

    ``variant`` is the variant's stable identifier
    (`"gpt-4o"`, `"claude-3-5-sonnet"`, `"llama-3-70b"`,
    `"rl-policy-v3"`, `"ga-evolved-7f3a"`). The dashboard
    correlates per-variant attribution by this name.

    ``confidence`` in [0, 1] is the variant's own self-reported
    confidence in the decision. The committee uses the *minimum*
    confidence across agreeing variants for BUY sizing — pin so
    a single low-confidence outlier in the agreeing cohort can't
    sneak through on the others' high confidence.

    ``quantity`` is the variant's proposed quantity (in base
    currency, e.g. BTC). The committee picks the median across
    agreeing variants — robust to one outsized bet.
    """

    variant: str
    action: Action | str
    confidence: float
    quantity: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]; got {self.confidence}")
        if self.quantity < 0:
            raise ValueError(f"quantity must be >= 0; got {self.quantity}")


def _coerce_action(value: Action | str) -> Action:
    if isinstance(value, Action):
        return value
    try:
        return Action(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown action {value!r}; expected one of {[a.value for a in Action]}"
        ) from exc


# ── Output ────────────────────────────────────────────────


@dataclass(frozen=True)
class VariantAttribution:
    """One variant's contribution to the verdict.

    ``agreed_with_committee`` is True iff this variant's action
    matches the committee's chosen action — operators want to
    spot which variants are reliably aligned with the consensus
    vs the contrarian dissenters."""

    variant: str
    action: Action
    confidence: float
    quantity: float
    weight: float
    agreed_with_committee: bool


@dataclass(frozen=True)
class CommitteeVerdict:
    """Aggregated decision with per-variant attribution.

    ``confidence`` is the *minimum* confidence across agreeing
    variants for a BUY (most conservative); the *mean* for SELL
    and HOLD (a closing decision benefits from confidence
    averaging across the cohort). Pin: the asymmetry is
    intentional — opening risk needs the strictest test.

    ``quantity`` is the median across agreeing variants for BUY;
    0.0 for SELL / HOLD (the executor handles the close).

    ``reason`` is a one-line operator-readable summary suitable
    for a notification or dashboard tile."""

    action: Action
    policy: VotingPolicy
    confidence: float
    quantity: float
    attributions: list[VariantAttribution] = field(default_factory=list)
    reason: str = ""

    @property
    def agreement_count(self) -> int:
        return sum(1 for a in self.attributions if a.agreed_with_committee)

    @property
    def total_variants(self) -> int:
        return len(self.attributions)


# ── Voting policies ───────────────────────────────────────


def _resolve_majority(
    decisions: list[VariantDecision],
) -> tuple[Action, str]:
    """Most-common action wins; conservative tiebreak."""
    actions = [_coerce_action(d.action) for d in decisions]
    counts: dict[Action, int] = {a: 0 for a in Action}
    for a in actions:
        counts[a] += 1
    return _pick_top(counts, len(decisions), policy_label="majority")


def _resolve_weighted(
    decisions: list[VariantDecision],
    weights: dict[str, float],
) -> tuple[Action, str]:
    """Per-variant weight sums per action; largest wins;
    conservative tiebreak. Variants without an explicit weight
    default to 1.0; a non-positive weight clamps to 0 (variant
    abstains).
    """
    actions = [_coerce_action(d.action) for d in decisions]
    sums: dict[Action, float] = {a: 0.0 for a in Action}
    total = 0.0
    for d, action in zip(decisions, actions):
        w = max(0.0, weights.get(d.variant, 1.0))
        sums[action] += w
        total += w
    return _pick_top(sums, total, policy_label="weighted")


def _resolve_unanimous(
    decisions: list[VariantDecision],
) -> tuple[Action, str]:
    """Pin: every non-HOLD vote must agree. If any variant says
    HOLD, the committee says HOLD. If variants split between BUY
    and SELL, HOLD. Only when every variant agrees on the same
    non-HOLD action does the committee echo that."""
    if not decisions:
        return Action.HOLD, "unanimous: no variants — defaulting to HOLD"
    actions = [_coerce_action(d.action) for d in decisions]
    if all(a == actions[0] for a in actions):
        if actions[0] == Action.HOLD:
            return (
                Action.HOLD,
                "unanimous: every variant said HOLD",
            )
        return (
            actions[0],
            f"unanimous: every variant said {actions[0].value.upper()}",
        )
    # Disagreement → conservative HOLD.
    distinct = sorted({a.value for a in actions})
    return (
        Action.HOLD,
        f"unanimous: variants split across {distinct} — defaulting to HOLD",
    )


def _pick_top(
    scores: dict[Action, float],
    total: float,
    *,
    policy_label: str,
) -> tuple[Action, str]:
    if total <= 0:
        return (
            Action.HOLD,
            f"{policy_label}: no votes / zero weight — defaulting to HOLD",
        )
    max_score = max(scores.values())
    candidates = [a for a, s in scores.items() if s == max_score]
    winner = max(candidates, key=lambda a: _CONSERVATISM_RANK[a])
    if len(candidates) > 1:
        reason = (
            f"{policy_label}: tie at {max_score} between "
            f"{[c.value for c in candidates]}; resolved to "
            f"{winner.value} (more conservative)"
        )
    else:
        reason = f"{policy_label}: {winner.value} won with score {max_score}"
    return winner, reason


# ── Aggregation ──────────────────────────────────────────


def _aggregate_buy_confidence_and_qty(
    agreeing: list[VariantDecision],
) -> tuple[float, float]:
    """Pin: BUY confidence = MIN across agreeing variants
    (a 3-of-5 BUY where the lowest agreeing variant says 0.4
    should size off 0.4, not 0.9 — refusing to extrapolate from
    the most-confident outlier is the safer default).

    Quantity = median across agreeing variants — robust to one
    variant proposing an outsized bet."""
    if not agreeing:
        return 0.0, 0.0
    confidences = [d.confidence for d in agreeing]
    quantities = [d.quantity for d in agreeing]
    return min(confidences), float(statistics.median(quantities))


def _aggregate_close_confidence(agreeing: list[VariantDecision]) -> float:
    """Pin: SELL/HOLD confidence = MEAN across agreeing variants.
    A closing decision benefits from confidence averaging — the
    asymmetry with BUY is intentional (opening risk needs the
    strictest test, closing risk the typical one)."""
    if not agreeing:
        return 0.0
    return statistics.mean(d.confidence for d in agreeing)


# ── Committee driver ──────────────────────────────────────


def vote(
    decisions: list[VariantDecision],
    *,
    policy: VotingPolicy = VotingPolicy.MAJORITY,
    weights: dict[str, float] | None = None,
) -> CommitteeVerdict:
    """Aggregate ``decisions`` into a single committee verdict.

    Empty input returns a HOLD verdict — pin: a no-variants
    cycle (every model failed) shouldn't accidentally promote a
    blank vote into BUY/SELL.
    """
    if not decisions:
        return CommitteeVerdict(
            action=Action.HOLD,
            policy=policy,
            confidence=0.0,
            quantity=0.0,
            attributions=[],
            reason="no variants — defaulting to HOLD",
        )

    if policy == VotingPolicy.MAJORITY:
        winner, reason = _resolve_majority(decisions)
    elif policy == VotingPolicy.WEIGHTED:
        winner, reason = _resolve_weighted(decisions, weights or {})
    elif policy == VotingPolicy.UNANIMOUS:
        winner, reason = _resolve_unanimous(decisions)
    else:
        raise ValueError(f"unknown voting policy {policy!r}")

    # Build per-variant attribution.
    weights = weights or {}
    attributions: list[VariantAttribution] = []
    agreeing: list[VariantDecision] = []
    for d in decisions:
        d_action = _coerce_action(d.action)
        agreed = d_action == winner
        if agreed:
            agreeing.append(d)
        attributions.append(
            VariantAttribution(
                variant=d.variant,
                action=d_action,
                confidence=d.confidence,
                quantity=d.quantity,
                weight=max(0.0, weights.get(d.variant, 1.0)),
                agreed_with_committee=agreed,
            )
        )

    # Aggregate confidence + quantity.
    if winner == Action.BUY:
        confidence, quantity = _aggregate_buy_confidence_and_qty(agreeing)
    else:
        # SELL or HOLD: average confidence across agreeing variants;
        # quantity is determined by the executor (it knows the
        # operator's open position) so the committee returns 0.
        confidence = _aggregate_close_confidence(agreeing)
        quantity = 0.0

    return CommitteeVerdict(
        action=winner,
        policy=policy,
        confidence=confidence,
        quantity=quantity,
        attributions=attributions,
        reason=reason,
    )


# ── Render helper ─────────────────────────────────────────


def render_verdict(verdict: CommitteeVerdict) -> str:
    """One-line operator-readable summary for logs / Slack /
    Telegram. Visual layout mirrors the other Round-4 render
    helpers — emoji severity prefix, action, vote tally, top
    contributing variants."""
    emoji_map = {
        Action.BUY: "🟢",
        Action.SELL: "🔴",
        Action.HOLD: "🟡",
    }
    emoji = emoji_map[verdict.action]
    lines = [
        f"{emoji} Committee {verdict.action.value.upper()} "
        f"({verdict.agreement_count}/{verdict.total_variants} variants)"
    ]
    lines.append(verdict.reason)
    if verdict.action == Action.BUY:
        lines.append(f"Confidence: {verdict.confidence:.0%}  Qty: {verdict.quantity:g}")
    elif verdict.action == Action.SELL:
        lines.append(f"Confidence: {verdict.confidence:.0%}")
    if verdict.attributions:
        lines.append("")
        for a in verdict.attributions:
            mark = "✓" if a.agreed_with_committee else "·"
            lines.append(
                f"  {mark} {a.variant:<22} {a.action.value:<5} "
                f"conf={a.confidence:.0%} qty={a.quantity:g}"
            )
    return "\n".join(lines)


__all__ = [
    "Action",
    "CommitteeVerdict",
    "VariantAttribution",
    "VariantDecision",
    "VotingPolicy",
    "render_verdict",
    "vote",
]
