"""Adversarial co-bot — a cheap second LLM call that attacks the primary plan.

The primary strategy LLM is wired to *find trades*, which is exactly the
optimization target you want at most steps but a slightly bad one for the
worst tail of cycles (chasing a blow-off, ignoring an imminent catalyst,
sizing into a regime that just flipped). The cheap fix is an *adversarial
critic*: a small follow-up call whose only job is to surface the single
strongest counter-thesis to whatever the primary just produced.

The contract is intentionally minimal:

    review = await critique_plan(attacker_llm, decisions=plan.decisions,
                                 market_outlook=plan.market_outlook,
                                 context_excerpt=user_prompt[:1200])

    if review.recommendation == "skip":      # severity >= skip_at
        plan.decisions = []
    elif review.recommendation == "downsize":  # severity >= downsize_at
        plan.decisions = apply_review_to_buys(plan.decisions, review)

The attacker emits ``{"severity": 0..1, "counter_thesis": "..."}`` — that's
it. Severity thresholds drive sizing, not the LLM. The strategy stays in
charge of *what* to do; the attacker only flags *how confident to be*.

Sells and holds are never downsized (those are risk-reducing) — only buys.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


_ATTACKER_SYSTEM = """\
You are a skeptical adversarial reviewer of a trading bot's plan. Your only
job is to surface the single STRONGEST reason the proposed trades could go
wrong RIGHT NOW. You do not propose alternatives — you only attack.

Score severity 0.0 (no real concern) to 1.0 (this looks like a clear trap).
Be ruthless but realistic. Most plans should score 0.2–0.5; reserve 0.7+ for
genuinely dangerous setups (overextended, against a violent regime, ignoring
an imminent catalyst, chasing a blow-off, fading a strong trend).

Output ONLY this JSON, nothing else:
{"severity": 0.0, "counter_thesis": "<= 30 words"}
"""

_ATTACKER_USER_TEMPLATE = """\
Proposed plan:
{plan_summary}

Recent context the primary saw:
{context_excerpt}

Respond with the JSON above. Be terse.
"""


@dataclass(frozen=True)
class AdversarialReview:
    """Verdict from the adversarial critic."""

    severity: float
    counter_thesis: str
    recommendation: str  # "proceed" | "downsize" | "skip"
    elapsed_ms: int = 0
    cost_usd: float = 0.0

    @property
    def sizing_multiplier(self) -> float:
        if self.recommendation == "skip":
            return 0.0
        if self.recommendation == "downsize":
            return 0.5
        return 1.0


def _action_str(decision: Any) -> str:
    action = getattr(decision, "action", "")
    return action.value if hasattr(action, "value") else str(action)


def _summarize_plan(decisions: Iterable[Any], market_outlook: str) -> str:
    lines: list[str] = []
    for d in decisions:
        action = _action_str(d).upper()
        sym = getattr(d, "symbol", "?")
        qty = getattr(d, "quantity", "?")
        conf = getattr(d, "confidence", "?")
        why = (getattr(d, "reasoning", "") or "")[:80]
        lines.append(f"- {action} {sym} qty={qty} conf={conf} :: {why}")
    if market_outlook:
        lines.append(f"outlook: {market_outlook[:160]}")
    return "\n".join(lines) if lines else "(empty plan)"


def _classify(severity: float, *, downsize_at: float, skip_at: float) -> str:
    if severity >= skip_at:
        return "skip"
    if severity >= downsize_at:
        return "downsize"
    return "proceed"


async def critique_plan(
    attacker: LLMBackend,
    *,
    decisions: Sequence[Any],
    market_outlook: str = "",
    context_excerpt: str = "",
    downsize_at: float = 0.45,
    skip_at: float = 0.75,
    context_chars: int = 1200,
) -> AdversarialReview:
    """Run the adversarial critic and return its verdict.

    Network/parse errors degrade gracefully to ``proceed`` — the attacker
    is advisory only. Never block trading because the attacker fell over.
    """
    has_buy = any(_action_str(d).lower() == "buy" for d in decisions)
    if not decisions or not has_buy:
        return AdversarialReview(
            severity=0.0,
            counter_thesis="(no buys to attack)",
            recommendation="proceed",
        )

    plan_summary = _summarize_plan(decisions, market_outlook)
    user_prompt = _ATTACKER_USER_TEMPLATE.format(
        plan_summary=plan_summary,
        context_excerpt=(context_excerpt or "(none)")[:context_chars],
    )

    t0 = time.monotonic()
    try:
        raw = await attacker.generate_json(user_prompt, system=_ATTACKER_SYSTEM)
    except Exception as exc:  # noqa: BLE001 — attacker is advisory
        logger.warning("adversarial attacker failed: %s — defaulting to proceed", exc)
        return AdversarialReview(
            severity=0.0,
            counter_thesis=f"attacker-error: {exc}",
            recommendation="proceed",
        )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    try:
        severity = float(raw.get("severity", 0.0))
    except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
        severity = 0.0
    severity = max(0.0, min(1.0, severity))
    counter = str(raw.get("counter_thesis", ""))[:240]

    usage = getattr(attacker, "last_usage", None)
    try:
        cost = float(getattr(usage, "cost_usd", 0) or 0)
    except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
        cost = 0.0

    rec = _classify(severity, downsize_at=downsize_at, skip_at=skip_at)
    review = AdversarialReview(
        severity=severity,
        counter_thesis=counter,
        recommendation=rec,
        elapsed_ms=elapsed_ms,
        cost_usd=cost,
    )
    logger.info(
        "adversarial review: severity=%.2f rec=%s thesis=%r (cost=$%.4f, %dms)",
        severity,
        rec,
        counter,
        cost,
        elapsed_ms,
    )
    return review


def apply_review_to_buys(decisions: Sequence[Any], review: AdversarialReview) -> list[Any]:
    """Return new decisions with BUY quantities scaled by ``review.sizing_multiplier``.

    SELL and HOLD are passed through unchanged — exits are risk-reducing
    and shouldn't be skipped on a counter-thesis to a *buy*.

    Pydantic models are copied via ``model_copy``; plain objects fall
    through unchanged when they don't expose a copy hook.
    """
    if review.recommendation == "proceed":
        return list(decisions)
    multiplier = review.sizing_multiplier

    out: list[Any] = []
    for d in decisions:
        is_buy = _action_str(d).lower() == "buy"
        if not is_buy:
            out.append(d)
            continue
        if multiplier == 0.0:
            # skip: drop the buy entirely
            continue
        new_qty = float(getattr(d, "quantity", 0)) * multiplier
        if hasattr(d, "model_copy"):
            out.append(d.model_copy(update={"quantity": new_qty}))
        else:
            try:
                d.quantity = new_qty  # type: ignore[attr-defined]
            except Exception:
                pass
            out.append(d)
    return out
