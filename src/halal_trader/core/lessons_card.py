"""Auto-generated "lessons learned" card for a closed trade.

Round-4 wave 5.D: when a trade closes — especially a loser — surface
a structured post-mortem the operator can scan in 10 seconds.
Sections:

* **Summary.** Pair, side, qty, entry/exit prices, return %, exit
  reason. Fast triage.
* **Entry rationale.** The LLM's reasoning at entry, lightly
  truncated. The "why we took this".
* **Entry indicators.** RSI, MACD, vol ratio, ATR, BB position at
  entry — the feature vector the strategy saw.
* **What changed.** When an exit snapshot is available, delta each
  indicator. Reveals "RSI was 28 at entry, 72 at exit — we caught
  the snap-back".
* **Verdict + lessons.** A small heuristic classifier labels the
  trade (`winner_thesis_intact`, `winner_lucky`,
  `loser_thesis_invalidated`, `loser_noise`, `winner`, `loser`) and
  emits 0–3 lesson bullets — e.g. "RSI was already extended at
  entry; consider a smaller size for confidence levels above 0.7
  when RSI > 70".

Why we own this in-house instead of asking the LLM at close-time:

* The card must render without an LLM call so the dashboard isn't
  bottlenecked behind the per-trade-close queue. The LLM can still
  be layered on top later (a richer "expert commentary" field).
* Heuristics are deterministic — easier to audit, test, and pin in
  regression. Two traders with identical inputs will see identical
  cards.
* Halal alignment: the card is informational only. It never feeds
  back into a sizing or entry decision.

Pure-Python; no NumPy, no SciPy, no DB. The renderer takes a
``LessonCardInput`` dataclass and returns a ``LessonCard`` — the
caller is responsible for fetching the trade row, the entry
indicator snapshot, and (optionally) the exit indicator snapshot
from the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Inputs ────────────────────────────────────────────────


@dataclass(frozen=True)
class IndicatorVector:
    """Compact indicator snapshot the renderer cares about.

    Mirrors the subset of `IndicatorSnapshot` fields that matter for
    the lessons-learned heuristics. Nullable across the board so the
    renderer degrades gracefully when a row is partial (legacy trades
    pre-snapshot, broker-side data gaps, etc.).
    """

    rsi_14: float | None = None
    macd_histogram: float | None = None
    volume_ratio: float | None = None
    atr_14: float | None = None
    bb_position: float | None = None  # 0..1, where 0=lower band, 1=upper band


@dataclass(frozen=True)
class LessonCardInput:
    """All the data the renderer needs to produce one card.

    Field naming matches the underlying DB schema closely so the
    SQL adapter is a thin map (no business-logic in the join).
    """

    pair: str
    side: str  # "buy" / "sell" — for crypto always "buy" (longs only)
    quantity: float
    entry_price: float | None
    exit_price: float | None
    return_pct: float | None
    exit_reason: str | None  # "stop_loss" | "take_profit" | "trailing_stop" | …
    llm_reasoning: str | None
    confidence: float | None = None
    entry_indicators: IndicatorVector | None = None
    exit_indicators: IndicatorVector | None = None


# ── Outputs ───────────────────────────────────────────────


@dataclass(frozen=True)
class LessonCard:
    """Structured post-mortem suitable for JSON / dashboard / email.

    ``verdict`` is a stable enum-like string the dashboard groups on.
    ``lessons`` is a 0–3 element list of one-sentence operator nudges.
    ``markdown`` is a pre-rendered, paste-into-Slack representation
    so the notifier doesn't have to re-walk the structure.
    """

    pair: str
    side: str
    quantity: float
    entry_price: float | None
    exit_price: float | None
    return_pct: float | None
    exit_reason: str | None
    rationale: str | None
    confidence: float | None
    entry_indicators: dict[str, float | None]
    exit_indicators: dict[str, float | None] | None
    indicator_deltas: dict[str, float] | None
    verdict: str
    lessons: list[str] = field(default_factory=list)
    markdown: str = ""


# ── Heuristics ────────────────────────────────────────────

_RSI_OVERBOUGHT = 70.0
_RSI_OVERSOLD = 30.0
_BB_NEAR_UPPER = 0.85
_BB_NEAR_LOWER = 0.15
_HIGH_CONFIDENCE = 0.70
_HIGH_VOL_RATIO = 2.0  # 2× normal volume


def _classify(input: LessonCardInput) -> str:
    """Bucket the trade into one of six verdicts.

    Pin: the classifier prefers explicit "thesis" verdicts when an
    exit-side indicator snapshot is available, falling back to plain
    winner/loser when it isn't — operators who haven't backfilled
    exit snapshots still get a useful card.
    """
    rp = input.return_pct
    if rp is None:
        return "unknown"
    won = rp > 0

    entry = input.entry_indicators
    exit_ind = input.exit_indicators
    has_both = entry is not None and exit_ind is not None

    if not has_both:
        return "winner" if won else "loser"

    # When we have both, classify by whether the indicators moved
    # the way the entry implied.
    if won:
        # A winner whose entry was clearly extended (RSI > 70 or
        # BB near upper) is "lucky" — the bot bought near a top
        # and got bailed out.
        if entry is not None and (
            (entry.rsi_14 is not None and entry.rsi_14 >= _RSI_OVERBOUGHT)
            or (entry.bb_position is not None and entry.bb_position >= _BB_NEAR_UPPER)
        ):
            return "winner_lucky"
        return "winner_thesis_intact"
    else:
        # A loser whose exit indicator vector materially diverges
        # from entry is "thesis invalidated" — the regime changed.
        # A loser whose vector barely moved is "noise" — random
        # walk killed the trade.
        if entry is not None and exit_ind is not None:
            rsi_delta = _abs_delta(entry.rsi_14, exit_ind.rsi_14)
            macd_delta = _abs_delta(entry.macd_histogram, exit_ind.macd_histogram)
            if rsi_delta > 15 or macd_delta > 0.005:
                return "loser_thesis_invalidated"
            return "loser_noise"
        return "loser"


def _abs_delta(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    return abs(b - a)


def _lessons_for(input: LessonCardInput, verdict: str) -> list[str]:
    """Generate up to three operator-readable lessons.

    The lessons aren't predictions — they're prompts to *re-examine*
    a hypothesis with the data this trade contributed. Each is
    keyed off a verifiable input (RSI value, exit reason, vol ratio)
    so the operator can quickly disagree if the heuristic mis-fired.
    """
    lessons: list[str] = []
    entry = input.entry_indicators
    rp = input.return_pct
    confidence = input.confidence

    # Lesson 1: high-confidence loss with no clear thesis change
    if verdict == "loser_noise" and confidence is not None and confidence >= _HIGH_CONFIDENCE:
        lessons.append(
            f"High-confidence loss ({confidence:.0%}) with no clear regime change — "
            "review whether the conviction signal is over-weighting on your edge."
        )

    # Lesson 2: extended entry that "worked"
    if verdict == "winner_lucky" and entry is not None and entry.rsi_14 is not None:
        lessons.append(
            f"Won despite buying at RSI {entry.rsi_14:.0f} — extended entries "
            "should size smaller; this winner is more luck than thesis."
        )

    # Lesson 3: stop-loss with already-overbought entry
    if (
        input.exit_reason
        and input.exit_reason.lower() in ("stop_loss", "stoploss")
        and entry is not None
        and entry.rsi_14 is not None
        and entry.rsi_14 >= _RSI_OVERBOUGHT
    ):
        lessons.append(
            "Stop-loss exit on an entry that was already overbought (RSI ≥ 70). "
            "Filter buys above RSI 70 for the next 20 trades and re-evaluate."
        )

    # Lesson 4: trailing-stop winner — celebrate, but don't size up
    if (
        input.exit_reason
        and input.exit_reason.lower() in ("trailing_stop", "trailing")
        and rp is not None
        and rp > 0
    ):
        lessons.append(
            "Trailing-stop exit captured the profitable run — keep the existing "
            "trailing-stop distance; don't loosen it without a backtest."
        )

    # Lesson 5: low-volume entry that lost
    if (
        verdict.startswith("loser")
        and entry is not None
        and entry.volume_ratio is not None
        and entry.volume_ratio < 0.5
    ):
        lessons.append(
            f"Loser entered on volume {entry.volume_ratio:.1f}× normal — "
            "low-volume regime tends to chop; consider gating entries on volume ≥ 0.8×."
        )

    # Lesson 6: high-volume win — reinforces the thesis
    if (
        verdict == "winner_thesis_intact"
        and entry is not None
        and entry.volume_ratio is not None
        and entry.volume_ratio >= _HIGH_VOL_RATIO
    ):
        lessons.append(
            f"Winner entered on volume {entry.volume_ratio:.1f}× normal — "
            "high-volume confirmations remain a useful filter."
        )

    return lessons[:3]


# ── Renderer ──────────────────────────────────────────────


def _vec_to_dict(v: IndicatorVector | None) -> dict[str, float | None]:
    if v is None:
        return {}
    return {
        "rsi_14": v.rsi_14,
        "macd_histogram": v.macd_histogram,
        "volume_ratio": v.volume_ratio,
        "atr_14": v.atr_14,
        "bb_position": v.bb_position,
    }


def _deltas(
    entry: IndicatorVector | None, exit_v: IndicatorVector | None
) -> dict[str, float] | None:
    if entry is None or exit_v is None:
        return None
    out: dict[str, float] = {}
    for key in ("rsi_14", "macd_histogram", "volume_ratio", "atr_14", "bb_position"):
        a = getattr(entry, key)
        b = getattr(exit_v, key)
        if a is None or b is None:
            continue
        out[key] = b - a
    return out


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.2%}" if v is not None else "n/a"


def _fmt_money(v: float | None) -> str:
    return f"${v:,.2f}" if v is not None else "n/a"


def _render_markdown(card_data: dict[str, Any], lessons: list[str]) -> str:
    """Build a paste-ready markdown rendering. Used by the Slack /
    Discord notifier to emit a single card per close event."""

    pair = card_data["pair"]
    side = card_data["side"].upper()
    rp = card_data["return_pct"]
    verdict = card_data["verdict"]
    emoji = "🟢" if (rp or 0) > 0 else "🔴"

    lines = [
        f"{emoji} **Lessons-learned · `{pair}` {side}**",
        f"Return: **{_fmt_pct(rp)}** · Exit: `{card_data['exit_reason'] or '—'}` "
        f"· Verdict: `{verdict}`",
        f"Entry: {_fmt_money(card_data['entry_price'])} → "
        f"Exit: {_fmt_money(card_data['exit_price'])}",
    ]
    if card_data.get("rationale"):
        rationale = card_data["rationale"]
        if len(rationale) > 200:
            rationale = rationale[:197] + "…"
        lines.append(f"> {rationale}")
    if card_data.get("entry_indicators"):
        ei = card_data["entry_indicators"]
        ind_parts = []
        if ei.get("rsi_14") is not None:
            ind_parts.append(f"RSI {ei['rsi_14']:.0f}")
        if ei.get("macd_histogram") is not None:
            ind_parts.append(f"MACD {ei['macd_histogram']:+.4f}")
        if ei.get("volume_ratio") is not None:
            ind_parts.append(f"Vol {ei['volume_ratio']:.1f}×")
        if ind_parts:
            lines.append(f"Entry indicators: {' · '.join(ind_parts)}")
    if lessons:
        lines.append("")
        for lesson in lessons:
            lines.append(f"• {lesson}")
    return "\n".join(lines)


def render(input: LessonCardInput) -> LessonCard:
    """Produce the lessons-learned card for one closed trade.

    Pure function — no DB, no LLM. Safe to call from any thread,
    stage, or notifier callback.
    """
    verdict = _classify(input)
    lessons = _lessons_for(input, verdict)

    entry_dict = _vec_to_dict(input.entry_indicators)
    exit_dict = _vec_to_dict(input.exit_indicators) if input.exit_indicators else None
    deltas = _deltas(input.entry_indicators, input.exit_indicators)

    card_data = {
        "pair": input.pair,
        "side": input.side,
        "quantity": input.quantity,
        "entry_price": input.entry_price,
        "exit_price": input.exit_price,
        "return_pct": input.return_pct,
        "exit_reason": input.exit_reason,
        "rationale": input.llm_reasoning,
        "confidence": input.confidence,
        "entry_indicators": entry_dict,
        "exit_indicators": exit_dict,
        "indicator_deltas": deltas,
        "verdict": verdict,
    }
    md = _render_markdown(card_data, lessons)

    return LessonCard(
        pair=input.pair,
        side=input.side,
        quantity=input.quantity,
        entry_price=input.entry_price,
        exit_price=input.exit_price,
        return_pct=input.return_pct,
        exit_reason=input.exit_reason,
        rationale=input.llm_reasoning,
        confidence=input.confidence,
        entry_indicators=entry_dict,
        exit_indicators=exit_dict,
        indicator_deltas=deltas,
        verdict=verdict,
        lessons=lessons,
        markdown=md,
    )
