"""Post-trade thesis tagging + P&L attribution.

Today we slice analytics by symbol and exit reason, which answers
"which symbols pay" but not "which *thesis* pays". Two trades on the
same symbol can be totally different bets — one a momentum chase, one
a mean-revert. Without that decomposition you can't kill the losing
thesis without killing the winning one.

This module gives every closed trade a small categorical tag derived
from its *post-trade* context (entry indicators, regime, exit reason,
realised return) and rolls those tags into an attribution view.

The taxonomy is intentionally tiny:

    trend_follow | mean_revert | breakout | news_react | range_fade | scalp

Few enough buckets that 50 trades is enough to spot which one is decaying.

Persistence: a JSON sidecar (``thesis_tags.json`` in the data dir). No
schema migration required, so this lands behind a feature flag without
touching any other module's wire format. When the column lands, the
sidecar is the source of truth for backfill.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


THESIS_TAGS = (
    "trend_follow",
    "mean_revert",
    "breakout",
    "news_react",
    "range_fade",
    "scalp",
    "unknown",
)


# ── Trade view ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaggedTradeContext:
    """Just enough about a closed trade to assign a thesis tag."""

    trade_id: str
    symbol: str
    side: str  # "buy" | "sell"
    entry_price: float
    exit_price: float | None
    exit_reason: str | None
    pnl_pct: float
    hold_seconds: int
    setup_type: str | None = None  # forward-looking guess from decision time
    indicators: dict[str, float] = field(default_factory=dict)
    regime: str | None = None
    news_blob: str = ""
    reasoning: str = ""


# ── Heuristic tagger (no LLM) ─────────────────────────────────────


def heuristic_tag(ctx: TaggedTradeContext) -> str:
    """Cheap deterministic classifier based on indicators + reasoning.

    Use as the default tagger so the system always has a tag without
    LLM cost. The LLM tagger (below) refines tags asynchronously.
    """
    text = f"{ctx.reasoning} {ctx.news_blob} {ctx.exit_reason or ''}".lower()

    if "news" in text or "headline" in text or "8-k" in text or "fomc" in text:
        return "news_react"

    if ctx.setup_type:
        s = ctx.setup_type.lower()
        if "breakout" in s:
            return "breakout"
        if "mean_rev" in s or "mean-rev" in s:
            return "mean_revert"
        if "momentum" in s or "trend" in s:
            return "trend_follow"
        if "range" in s:
            return "range_fade"

    rsi = ctx.indicators.get("rsi_14")
    macd = ctx.indicators.get("macd_histogram")
    bb_pos = ctx.indicators.get("bb_position")  # 0..1

    if rsi is not None and macd is not None:
        if rsi > 65 and macd > 0:
            return "trend_follow"
        if rsi < 35 and (bb_pos is None or bb_pos < 0.2):
            return "mean_revert"

    if ctx.hold_seconds < 600:  # < 10min
        return "scalp"

    return "unknown"


# ── LLM tagger ────────────────────────────────────────────────────


_LLM_SYSTEM = """\
You assign ONE thesis tag to a closed trade. Choose exactly one from this list:
trend_follow, mean_revert, breakout, news_react, range_fade, scalp, unknown.

Pick the tag that best describes WHAT KIND OF EDGE the trade was attempting
based on the entry context, indicators, and reasoning. Do not pick based on
whether it won or lost — that's a different analysis.

Output ONLY this JSON, nothing else:
{"tag": "trend_follow", "confidence": 0.7, "reason": "<= 15 words"}
"""

_LLM_USER_TEMPLATE = """\
Symbol: {symbol}    Side: {side}    Hold: {hold_seconds}s
Entry: {entry_price}    Exit: {exit_price}    Exit reason: {exit_reason}
Realized: {pnl_pct:+.2%}
Indicators (entry): {indicators}
Regime: {regime}
News blob: {news_blob}
Reasoning at decision time: {reasoning}

Reply with the JSON above.
"""


@dataclass(frozen=True)
class TagVerdict:
    tag: str
    confidence: float
    reason: str
    elapsed_ms: int = 0
    cost_usd: float = 0.0


async def llm_tag(llm: LLMBackend, ctx: TaggedTradeContext) -> TagVerdict:
    """One LLM call to refine the heuristic tag."""
    user_prompt = _LLM_USER_TEMPLATE.format(
        symbol=ctx.symbol,
        side=ctx.side,
        hold_seconds=ctx.hold_seconds,
        entry_price=ctx.entry_price,
        exit_price=ctx.exit_price,
        exit_reason=ctx.exit_reason,
        pnl_pct=ctx.pnl_pct,
        indicators=", ".join(f"{k}={v:.2f}" for k, v in ctx.indicators.items()) or "(none)",
        regime=ctx.regime or "unknown",
        news_blob=(ctx.news_blob or "(none)")[:200],
        reasoning=(ctx.reasoning or "(none)")[:200],
    )
    t0 = time.monotonic()
    try:
        raw = await llm.generate_json(user_prompt, system=_LLM_SYSTEM)
    except Exception as exc:  # noqa: BLE001
        logger.warning("thesis-tag LLM failed: %s — falling back to heuristic", exc)
        return TagVerdict(tag=heuristic_tag(ctx), confidence=0.3, reason="llm-error")
    elapsed = int((time.monotonic() - t0) * 1000)

    raw_tag = str(raw.get("tag", "unknown")).lower()
    tag = raw_tag if raw_tag in THESIS_TAGS else "unknown"
    try:
        conf = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
        conf = 0.5
    reason = str(raw.get("reason", ""))[:200]

    usage = getattr(llm, "last_usage", None)
    try:
        cost = float(getattr(usage, "cost_usd", 0) or 0)
    except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
        cost = 0.0
    return TagVerdict(tag=tag, confidence=conf, reason=reason, elapsed_ms=elapsed, cost_usd=cost)


# ── Sidecar persistence ───────────────────────────────────────────


@dataclass
class ThesisTagStore:
    """JSON-on-disk store of trade_id → tag.

    Schema-free, ops-simple, swappable for a real column when one lands.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict):
                return {}
            return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("thesis tag sidecar unreadable, starting fresh: %s", exc)
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def get(self, trade_id: str) -> str | None:
        data = self._load()
        rec = data.get(str(trade_id))
        return rec.get("tag") if rec else None

    def set(
        self,
        trade_id: str,
        tag: str,
        *,
        confidence: float = 1.0,
        reason: str = "",
        method: str = "heuristic",
    ) -> None:
        if tag not in THESIS_TAGS:
            tag = "unknown"
        data = self._load()
        data[str(trade_id)] = {
            "tag": tag,
            "confidence": confidence,
            "reason": reason,
            "method": method,
        }
        self._save(data)

    def all(self) -> dict[str, str]:
        return {tid: rec.get("tag", "unknown") for tid, rec in self._load().items()}


# ── Attribution analytics ─────────────────────────────────────────


@dataclass
class AttributionRow:
    tag: str
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    sum_pnl_pct: float = 0.0  # sum of pnl_pct across trades

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else 0.0

    @property
    def avg_pnl_pct(self) -> float:
        return self.sum_pnl_pct / self.n_trades if self.n_trades else 0.0


def attribute_pnl_by_thesis(
    trades: Iterable[TaggedTradeContext],
    tag_lookup: dict[str, str] | None = None,
) -> dict[str, AttributionRow]:
    """Group closed trades by their (assigned) thesis tag.

    Falls back to :func:`heuristic_tag` when a trade isn't in the lookup
    so callers can attribute even before the LLM tagger has run.
    """
    rows: dict[str, AttributionRow] = {tag: AttributionRow(tag=tag) for tag in THESIS_TAGS}
    lookup = tag_lookup or {}
    for t in trades:
        tag = lookup.get(t.trade_id) or heuristic_tag(t)
        if tag not in rows:
            tag = "unknown"
        row = rows[tag]
        row.n_trades += 1
        row.sum_pnl_pct += t.pnl_pct
        if t.pnl_pct > 0:
            row.wins += 1
        elif t.pnl_pct < 0:
            row.losses += 1
    return {k: v for k, v in rows.items() if v.n_trades > 0}


def deprecated_thesis_kill_list(
    rows: dict[str, AttributionRow],
    *,
    min_trades: int = 30,
    min_avg_pnl_pct: float = 0.0,
) -> list[str]:
    """Tags whose realised attribution suggests killing the playbook.

    Default rule: tag has at least ``min_trades`` samples and average P&L
    at-or-below ``min_avg_pnl_pct``.
    """
    return sorted(
        tag
        for tag, row in rows.items()
        if row.n_trades >= min_trades and row.avg_pnl_pct <= min_avg_pnl_pct
    )


def render_attribution(rows: Sequence[AttributionRow]) -> str:
    """Multi-line operator-friendly attribution table."""
    lines = ["thesis            n   win%    avg pnl"]
    lines.append("-" * 38)
    for row in sorted(rows, key=lambda r: -r.n_trades):
        lines.append(
            f"{row.tag:<16} {row.n_trades:>3}  {row.win_rate * 100:>4.0f}%  {row.avg_pnl_pct:+.2%}"
        )
    return "\n".join(lines)
