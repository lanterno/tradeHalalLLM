"""Counter-factual / hindsight regret accounting.

A trading bot's quality has two cheap-to-fool measures: P&L and win-rate.
Both miss the most useful question: *for the trades the bot did make, how
close to the best-feasible action was each one given what we now know?*

This module gives the rest of the system a deterministic regret signal
for every closed trade, plus an aggregator and a small async hook for
optionally running a counter-factual LLM "would you do it again?" pass.

Regret is bounded in [0, 1]:

* 0  = bot's action equals the hindsight-optimal action.
* 1  = bot's action was the worst plausible choice given the outcome.

The default rule is intentionally simple — *if the actual P&L was
positive, the optimal action was to take the trade at full size; if
negative, the optimal was to skip*. This is hindsight, not foresight,
so it answers "how much edge did we leave on the table" not "what was
predictable".

Run aggregated regret weekly. A rising trend tells you the prompt /
model is getting worse at sizing, even if P&L hasn't dropped yet.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


# ── Hindsight (deterministic) regret ──────────────────────────────


@dataclass(frozen=True)
class ClosedTradeView:
    """Minimal view a regret computation needs.

    ``action_size_pct`` is the fraction of max-position-size the bot
    actually used (0..1). Skipping = 0. ``pnl_pct`` is the realized
    return relative to the position notional.
    """

    trade_id: str
    symbol: str
    action_size_pct: float
    pnl_pct: float
    confidence: float = 0.0
    setup_type: str | None = None


@dataclass(frozen=True)
class RegretRecord:
    trade_id: str
    symbol: str
    regret: float  # 0..1
    optimal_size_pct: float
    actual_size_pct: float
    pnl_pct: float
    note: str = ""


def hindsight_optimal_size(pnl_pct: float) -> float:
    """Optimal *post-hoc* size for one trade.

    Win → 1.0 (full size); flat-or-loss → 0.0 (skip).
    """
    return 1.0 if pnl_pct > 0 else 0.0


def hindsight_regret(trade: ClosedTradeView) -> RegretRecord:
    """Deterministic regret for one closed trade.

    Regret = absolute distance between actual size and the post-hoc
    optimal size. Both lie in [0, 1] so regret does too.
    """
    optimal = hindsight_optimal_size(trade.pnl_pct)
    actual = max(0.0, min(1.0, trade.action_size_pct))
    regret = abs(optimal - actual)
    if trade.pnl_pct > 0 and actual < 0.5:
        note = "missed-edge: small size on a winner"
    elif trade.pnl_pct < 0 and actual > 0.5:
        note = "tail-loss: large size on a loser"
    else:
        note = ""
    return RegretRecord(
        trade_id=trade.trade_id,
        symbol=trade.symbol,
        regret=regret,
        optimal_size_pct=optimal,
        actual_size_pct=actual,
        pnl_pct=trade.pnl_pct,
        note=note,
    )


# ── Aggregation ───────────────────────────────────────────────────


@dataclass
class RegretSummary:
    """Aggregated regret stats across a window of trades."""

    n: int = 0
    mean_regret: float = 0.0
    median_regret: float = 0.0
    pct_high_regret: float = 0.0  # share with regret >= 0.7
    missed_edge_count: int = 0
    tail_loss_count: int = 0
    by_symbol: dict[str, float] = field(default_factory=dict)
    by_setup: dict[str, float] = field(default_factory=dict)


def aggregate_regret(
    records: Sequence[RegretRecord], setup_lookup: dict[str, str | None] | None = None
) -> RegretSummary:
    """Roll a list of records into a summary suitable for the dashboard."""
    n = len(records)
    if n == 0:
        return RegretSummary()
    regrets = sorted(r.regret for r in records)
    mean = sum(regrets) / n
    median = regrets[n // 2] if n % 2 else 0.5 * (regrets[n // 2 - 1] + regrets[n // 2])
    high = sum(1 for r in regrets if r >= 0.7) / n
    missed = sum(1 for r in records if "missed-edge" in r.note)
    tail = sum(1 for r in records if "tail-loss" in r.note)

    by_symbol: dict[str, list[float]] = {}
    for r in records:
        by_symbol.setdefault(r.symbol, []).append(r.regret)
    by_symbol_mean = {sym: sum(v) / len(v) for sym, v in by_symbol.items()}

    by_setup_buckets: dict[str, list[float]] = {}
    if setup_lookup:
        for r in records:
            setup = setup_lookup.get(r.trade_id) or "unknown"
            by_setup_buckets.setdefault(setup, []).append(r.regret)
    by_setup_mean = {k: sum(v) / len(v) for k, v in by_setup_buckets.items()}

    return RegretSummary(
        n=n,
        mean_regret=mean,
        median_regret=median,
        pct_high_regret=high,
        missed_edge_count=missed,
        tail_loss_count=tail,
        by_symbol=by_symbol_mean,
        by_setup=by_setup_mean,
    )


# ── Optional counter-factual LLM pass ─────────────────────────────


_CF_SYSTEM = """\
You will be given a closed trade plus the context the original strategy
saw at the time of decision. Tell me, with the benefit of seeing only
the snapshot (NOT the outcome), whether you would now have taken the
same action, a smaller version, a different action, or no action.

Output ONLY this JSON, nothing else:
{"would_repeat": true|false, "regret": 0.0, "alt_action": "buy|sell|hold|skip"}

`regret` is your assessment in [0, 1] of how confident you are that
the original action was wrong relative to a hypothetical best action
on this snapshot — NOT a P&L number. Be calibrated, not punitive.
"""


_CF_USER_TEMPLATE = """\
Snapshot the original strategy saw:
{context_excerpt}

Original action:
- {action} {symbol} size_pct={size_pct} confidence={confidence}

Respond with the JSON in the system prompt.
"""


@dataclass(frozen=True)
class CounterFactualVerdict:
    would_repeat: bool
    regret: float
    alt_action: str
    elapsed_ms: int = 0
    cost_usd: float = 0.0


async def counter_factual_review(
    llm: LLMBackend,
    *,
    context_excerpt: str,
    action: str,
    symbol: str,
    size_pct: float,
    confidence: float,
    context_chars: int = 1500,
) -> CounterFactualVerdict:
    """One LLM call asking whether the original action holds up.

    *No outcome is shown to the LLM* — that's the whole point of the
    counter-factual: we want its judgement on the snapshot, not on the
    realised P&L. Pair the result with :func:`hindsight_regret` to get
    both views per trade.
    """
    user_prompt = _CF_USER_TEMPLATE.format(
        context_excerpt=(context_excerpt or "(none)")[:context_chars],
        action=action,
        symbol=symbol,
        size_pct=size_pct,
        confidence=confidence,
    )
    t0 = time.monotonic()
    try:
        raw = await llm.generate_json(user_prompt, system=_CF_SYSTEM)
    except Exception as exc:  # noqa: BLE001
        logger.warning("counter-factual call failed: %s", exc)
        return CounterFactualVerdict(would_repeat=True, regret=0.0, alt_action="unknown")
    elapsed = int((time.monotonic() - t0) * 1000)

    would = bool(raw.get("would_repeat", True))
    try:
        regret = float(raw.get("regret", 0.0))
    except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
        regret = 0.0
    regret = max(0.0, min(1.0, regret))
    alt = str(raw.get("alt_action", "")).lower() or "unknown"

    usage = getattr(llm, "last_usage", None)
    try:
        cost = float(getattr(usage, "cost_usd", 0) or 0)
    except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
        cost = 0.0

    return CounterFactualVerdict(
        would_repeat=would,
        regret=regret,
        alt_action=alt,
        elapsed_ms=elapsed,
        cost_usd=cost,
    )


# ── Stream helper ─────────────────────────────────────────────────


async def review_closed_trades(
    closed: Iterable[ClosedTradeView],
    *,
    fetch_context: Callable[[str], Awaitable[str]] | None = None,
    llm: LLMBackend | None = None,
) -> list[tuple[RegretRecord, CounterFactualVerdict | None]]:
    """Run hindsight regret on every closed trade and (optionally) a
    counter-factual review when ``llm`` and ``fetch_context`` are wired.

    Returns parallel records so callers can persist either or both.
    """
    out: list[tuple[RegretRecord, CounterFactualVerdict | None]] = []
    for trade in closed:
        rec = hindsight_regret(trade)
        cf: CounterFactualVerdict | None = None
        if llm is not None and fetch_context is not None:
            ctx = await fetch_context(trade.trade_id)
            cf = await counter_factual_review(
                llm,
                context_excerpt=ctx,
                action=("buy" if trade.action_size_pct > 0 else "skip"),
                symbol=trade.symbol,
                size_pct=trade.action_size_pct,
                confidence=trade.confidence,
            )
        out.append((rec, cf))
    return out
