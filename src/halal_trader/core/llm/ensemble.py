"""Ensemble LLM judge — fan out, agree, size by consensus.

A single LLM call is a noisy estimator. The cheap fix is parallel calls
with deliberate variation (different providers / temperatures / prompt
phrasings) plus a tiny arbiter that:

1. Counts agreement across variants on each (symbol, action).
2. Returns the consensus plan with a sizing multiplier driven by
   *how unanimous* the variants were.

This is intentionally a separate primitive from
``halal_trader.core.llm.adversarial`` — the adversarial co-bot tries to
break a plan, the ensemble judge tries to corroborate it. Both are
useful; combine them by running the ensemble first, then the adversary
on the consensus plan.

Cost notes:
* If the endpoint caches the shared prompt prefix and only the
  user-prompt suffix varies, K parallel calls cost roughly
  ``input_tokens × cache_price + K × output_tokens × output_price``.
* For 3 GLM-5.2 variants this is on the order of a cent a cycle.

The interface is deliberately data-shape-agnostic so it works for both
stock ``TradingPlan`` and crypto ``CryptoTradingPlan`` — anything with
``decisions[].action / .symbol / .quantity / .confidence`` works.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


async def wrap_existing(plan: Any) -> Any:
    """Trivial awaitable that returns ``plan`` — feeds the primary into
    :func:`run_ensemble` as one of the variants without re-calling the LLM.

    Shared by both ``crypto/strategy.py`` and ``trading/strategy.py``;
    historically each module had its own private copy. Single home now.
    """
    return plan


# ── Variant definition ───────────────────────────────────────────


@dataclass(frozen=True)
class EnsembleVariant:
    """One member of the ensemble.

    ``name`` shows up in logs and the audit trail. ``call`` is an awaitable
    returning a plan-shaped object — typically a small lambda over an
    existing ``CryptoTradingStrategy.analyze`` with different temperature
    or prompt-version overrides.
    """

    name: str
    call: Callable[[], Awaitable[Any]]


# ── Verdict ──────────────────────────────────────────────────────


@dataclass
class EnsembleVerdict:
    """The result of consensus aggregation."""

    consensus_plan: Any
    agreement_score: float  # 0 (everyone disagrees) .. 1 (unanimous)
    sizing_multiplier: float  # 0..1 driven by agreement
    per_variant: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────


def _action_str(decision: Any) -> str:
    a = getattr(decision, "action", "")
    return (a.value if hasattr(a, "value") else str(a)).lower()


def _decisions(plan: Any) -> list[Any]:
    return list(getattr(plan, "decisions", []) or [])


def _key(decision: Any) -> tuple[str, str]:
    return (_action_str(decision), getattr(decision, "symbol", ""))


# ── Aggregation ──────────────────────────────────────────────────


def aggregate_plans(
    plans_by_variant: dict[str, Any],
    *,
    quorum: int = 2,
    skip_quorum_at: float | None = None,
) -> EnsembleVerdict:
    """Roll N variant plans into one consensus plan.

    * A (action, symbol) pair survives the consensus if at least
      ``quorum`` variants emitted it. The kept decision uses the median
      quantity and median confidence across the agreeing variants.
    * Agreement score = sum of agreeing-variant counts / total decisions
      (1.0 when every variant emits the same set; 0 when they all
      disagree).
    * ``sizing_multiplier`` scales linearly from 0.5 → 1.0 as agreement
      goes from quorum → unanimity. Divergent ensembles size small.
    """
    if not plans_by_variant:
        raise ValueError("aggregate_plans needs at least one plan")
    n_variants = len(plans_by_variant)
    quorum = max(1, min(quorum, n_variants))

    # bucket each (action, symbol) → list of decisions across variants
    buckets: dict[tuple[str, str], list[Any]] = {}
    for variant_name, plan in plans_by_variant.items():
        for d in _decisions(plan):
            buckets.setdefault(_key(d), []).append(d)

    counts: dict[str, dict[str, int]] = {}
    surviving: list[Any] = []
    surviving_strengths: list[float] = []
    for (action, symbol), variants in buckets.items():
        n = len(variants)
        counts.setdefault(symbol, {})[action] = n
        if n >= quorum:
            surviving.append(_consensus_decision(variants))
            surviving_strengths.append(n / n_variants)

    # Agreement = how concentrated the surviving buckets are. A perfectly
    # unanimous ensemble has every surviving bucket at strength 1.0;
    # a fragmented one (each variant emits a different decision) has
    # buckets at strength 1/n_variants.
    if surviving_strengths:
        agreement = sum(surviving_strengths) / len(surviving_strengths)
    else:
        agreement = 0.0
    multiplier = _multiplier_from_agreement(
        agreement, quorum=quorum, n_variants=n_variants, skip_at=skip_quorum_at
    )

    base_plan = next(iter(plans_by_variant.values()))
    consensus = _build_consensus_plan(base_plan, surviving)
    return EnsembleVerdict(
        consensus_plan=consensus,
        agreement_score=agreement,
        sizing_multiplier=multiplier,
        per_variant=plans_by_variant,
        counts=counts,
        notes=[
            f"variants={n_variants} quorum={quorum} kept={len(surviving)} agreement={agreement:.2f}"
        ],
    )


def _consensus_decision(variants: Sequence[Any]) -> Any:
    """Median quantity + median confidence across agreeing variants."""
    base = variants[0]
    quantities = sorted(float(getattr(v, "quantity", 0)) for v in variants)
    confidences = sorted(float(getattr(v, "confidence", 0)) for v in variants)
    median_qty = quantities[len(quantities) // 2]
    median_conf = confidences[len(confidences) // 2]
    if hasattr(base, "model_copy"):
        return base.model_copy(update={"quantity": median_qty, "confidence": median_conf})
    return base


def _build_consensus_plan(base_plan: Any, decisions: list[Any]) -> Any:
    if hasattr(base_plan, "model_copy"):
        return base_plan.model_copy(
            update={
                "decisions": decisions,
                "risk_notes": (
                    (getattr(base_plan, "risk_notes", "") or "") + " | ensemble consensus"
                ).strip(" |"),
            }
        )
    # Plain object fallback — mutate
    try:
        base_plan.decisions = decisions
    except Exception:
        pass
    return base_plan


def _multiplier_from_agreement(
    agreement: float,
    *,
    quorum: int,
    n_variants: int,
    skip_at: float | None = None,
) -> float:
    """Map agreement → sizing multiplier in [0, 1].

    * ``skip_at`` (optional, in [0,1]): below this score the ensemble
      returns 0 (skip the cycle).
    * Otherwise interpolate linearly between ``quorum_floor`` (= 0.5) at
      the quorum threshold and 1.0 at unanimity.
    """
    if skip_at is not None and agreement < skip_at:
        return 0.0
    if n_variants <= 1:
        return 1.0
    quorum_floor = 0.5
    quorum_share = quorum / n_variants
    if agreement <= quorum_share:
        return quorum_floor
    if agreement >= 1.0:
        return 1.0
    span = 1.0 - quorum_share
    if span <= 0:
        return 1.0
    t = (agreement - quorum_share) / span
    return quorum_floor + t * (1.0 - quorum_floor)


# ── Driver ────────────────────────────────────────────────────────


async def run_ensemble(
    variants: Sequence[EnsembleVariant],
    *,
    quorum: int = 2,
    skip_quorum_at: float | None = None,
    timeout_s: float = 30.0,
) -> EnsembleVerdict:
    """Run all variants concurrently, then aggregate.

    Variant errors degrade gracefully — that variant simply doesn't vote.
    If *every* variant fails, raises the last error so the cycle can
    surface the failure cleanly.
    """
    if not variants:
        raise ValueError("ensemble needs at least one variant")

    async def _safe(variant: EnsembleVariant) -> tuple[str, Any | Exception]:
        try:
            res = await asyncio.wait_for(variant.call(), timeout=timeout_s)
            return variant.name, res
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensemble variant %s failed: %s", variant.name, exc)
            return variant.name, exc

    results = await asyncio.gather(*[_safe(v) for v in variants])
    plans = {name: plan for name, plan in results if not isinstance(plan, Exception)}
    if not plans:
        # all failed — surface the last error
        raise next(plan for _, plan in results if isinstance(plan, Exception))
    return aggregate_plans(plans, quorum=quorum, skip_quorum_at=skip_quorum_at)
