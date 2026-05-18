"""Per-position "story view" aggregator.

Round-4 wave 5.H: when an operator clicks a position in the
dashboard, they should see the entire story of the trade on one
page — the entry rationale (what the LLM said and why), the
indicator vector at entry, every indicator observation since,
every news event the symbol triggered, the price track, the
unrealized P&L tracked over time, and the active SL/TP levels.

This module is the *aggregator* — it composes already-collected
data into one structured `PositionStory`. The renderer at the
end produces a markdown narrative suitable for the dashboard's
detail panel or a Slack / email digest. SQL fan-in (subscribe to
the trade row + IndicatorSnapshot + news feed + price ticks) is
deferred to a follow-up; this module operates on plain
dataclasses so it's testable in isolation.

Pure-Python; no NumPy, no DB, no async. The aggregator is a single
function on a `PositionStoryInput` and returns a `PositionStory`
that can be JSON-serialised for the API or rendered to markdown
for the operator. Why an aggregator-only design (no SQL): the data
already exists across `crypto_trades`, `indicator_snapshots`,
`news_events`, `kline_ticks`, and `llm_decisions` — duplicating
that logic in a "story service" would just make a fragile second
copy. The aggregator's job is presentation, not retrieval.

Halal alignment: the story is informational only — never feeds
back into a sizing or entry decision. The aggregator never calls a
broker or screener; it operates on data the cycle already
collected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

# ── Inputs ────────────────────────────────────────────────


@dataclass(frozen=True)
class IndicatorObservation:
    """One indicator vector at one point in time.

    Mirrors the subset of `IndicatorSnapshot` fields that matter
    for the story timeline. Values are nullable so a partial row
    (legacy crypto trade pre-snapshot, gappy data feed) doesn't
    break the renderer.
    """

    at: datetime
    rsi_14: float | None = None
    macd_histogram: float | None = None
    volume_ratio: float | None = None
    atr_14: float | None = None
    bb_position: float | None = None  # 0..1 between Bollinger bands


@dataclass(frozen=True)
class PriceObservation:
    """One price tick. The aggregator computes mark-to-market P&L
    from these against the entry."""

    at: datetime
    price: float


@dataclass(frozen=True)
class NewsEvent:
    """One news / catalyst event keyed to the symbol.

    ``score`` is a fraction in [-1, 1] — negative = bearish,
    positive = bullish — same convention CryptoPanic / Reddit
    feeds use elsewhere.
    """

    at: datetime
    headline: str
    source: str
    score: float = 0.0
    url: str | None = None


@dataclass(frozen=True)
class PositionStoryInput:
    """All the data the aggregator needs to produce one story.

    Field naming matches the underlying DB schema closely (mirrors
    `Trade` / `CryptoTrade` columns + `IndicatorSnapshot`) so the
    SQL fan-in adapter is a thin map.

    ``pair`` is symbol-level — `"BTCUSDT"` for crypto, `"AAPL"`
    for stocks; the renderer doesn't care which.
    """

    pair: str
    side: str
    quantity: float
    entry_price: float
    entry_at: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_pct: float | None = None
    llm_reasoning: str | None = None
    confidence: float | None = None
    prompt_version: str | None = None
    entry_indicators: IndicatorObservation | None = None
    indicator_timeline: list[IndicatorObservation] = field(default_factory=list)
    price_timeline: list[PriceObservation] = field(default_factory=list)
    news: list[NewsEvent] = field(default_factory=list)


# ── Outputs ───────────────────────────────────────────────


@dataclass(frozen=True)
class PnlPoint:
    """One unrealized-P&L snapshot. Fraction-based; the renderer
    formats as percent."""

    at: datetime
    price: float
    unrealized_pct: float
    unrealized_usd: float


@dataclass(frozen=True)
class IndicatorDelta:
    """How an indicator moved between entry and the latest
    observation. Used in the `summary` section so the operator can
    see at-a-glance which features changed."""

    name: str
    entry_value: float | None
    latest_value: float | None
    delta: float | None  # latest - entry, None when either side missing


@dataclass(frozen=True)
class PositionStory:
    """Composed view ready for rendering.

    ``deltas`` summarises the indicator drift between entry and
    latest observation. ``pnl_curve`` is the time series of
    unrealized P&L derived from `price_timeline`. ``news_count`` is
    a fast cardinality for the dashboard tile; the full list lives
    in the input.

    ``markdown`` is a pre-rendered narrative the dashboard /
    notifier can ship straight; saves them re-walking the structure.
    """

    pair: str
    side: str
    quantity: float
    entry_price: float
    entry_at: datetime
    stop_loss: float | None
    take_profit: float | None
    trailing_stop_pct: float | None
    rationale: str | None
    confidence: float | None
    prompt_version: str | None
    current_price: float | None
    current_unrealized_pct: float | None
    current_unrealized_usd: float | None
    deltas: list[IndicatorDelta]
    pnl_curve: list[PnlPoint]
    news_count: int
    bullish_news: int
    bearish_news: int
    markdown: str = ""


# ── Helpers ───────────────────────────────────────────────


def _pnl_curve(
    *, entry_price: float, quantity: float, prices: Sequence[PriceObservation]
) -> list[PnlPoint]:
    """Compute the unrealized P&L track from entry against each
    observed price. Returns an empty list when no observations are
    supplied (the dashboard renders a "no data yet" state in that
    case). Pin: the curve preserves the input ordering — caller
    is responsible for chronological order, since the data layer
    already orders by timestamp."""
    out: list[PnlPoint] = []
    if entry_price <= 0:
        return out
    for obs in prices:
        pct = (obs.price - entry_price) / entry_price
        usd = (obs.price - entry_price) * quantity
        out.append(
            PnlPoint(
                at=obs.at,
                price=obs.price,
                unrealized_pct=float(pct),
                unrealized_usd=float(usd),
            )
        )
    return out


def _indicator_deltas(
    entry: IndicatorObservation | None, latest: IndicatorObservation | None
) -> list[IndicatorDelta]:
    """Per-indicator delta from entry to latest. Skips fields that
    are None on either side rather than emitting a `delta=None` row
    — the dashboard renders fewer rows when data is partial, which
    is more useful than a wall of "n/a"s."""
    deltas: list[IndicatorDelta] = []
    fields = ("rsi_14", "macd_histogram", "volume_ratio", "atr_14", "bb_position")
    if entry is None and latest is None:
        return deltas
    e = entry
    latest_obs = latest
    for name in fields:
        e_val = getattr(e, name, None) if e else None
        l_val = getattr(latest_obs, name, None) if latest_obs else None
        if e_val is None and l_val is None:
            continue
        if e_val is None or l_val is None:
            deltas.append(
                IndicatorDelta(name=name, entry_value=e_val, latest_value=l_val, delta=None)
            )
            continue
        deltas.append(
            IndicatorDelta(name=name, entry_value=e_val, latest_value=l_val, delta=l_val - e_val)
        )
    return deltas


def _classify_news(news: Sequence[NewsEvent]) -> tuple[int, int]:
    """Bucket news into (bullish, bearish) counts. Ignores neutral
    (score == 0) — they'd skew the dashboard tile if counted as
    bearish *or* bullish."""
    bullish = sum(1 for n in news if n.score > 0)
    bearish = sum(1 for n in news if n.score < 0)
    return bullish, bearish


def _format_pct(value: float | None) -> str:
    return f"{value:+.2%}" if value is not None else "n/a"


def _format_usd(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "n/a"


def _format_price(value: float | None) -> str:
    return f"${value:,.4f}" if value is not None else "n/a"


def _render_markdown(story: PositionStory, news: Sequence[NewsEvent]) -> str:
    """Build the operator-facing narrative."""
    pnl_emoji = "🟢" if (story.current_unrealized_pct or 0) >= 0 else "🔴"
    side = story.side.upper()
    lines = [
        f"# {pnl_emoji} `{story.pair}` {side} · {story.quantity:g} units",
        "",
        f"**Entry:** {_format_price(story.entry_price)} at {story.entry_at:%Y-%m-%d %H:%M UTC}",
        f"**Current:** {_format_price(story.current_price)} "
        f"({_format_pct(story.current_unrealized_pct)} / "
        f"{_format_usd(story.current_unrealized_usd)})",
    ]
    risk_parts = []
    if story.stop_loss is not None:
        risk_parts.append(f"SL {_format_price(story.stop_loss)}")
    if story.take_profit is not None:
        risk_parts.append(f"TP {_format_price(story.take_profit)}")
    if story.trailing_stop_pct is not None:
        risk_parts.append(f"trail {story.trailing_stop_pct:.2%}")
    if risk_parts:
        lines.append("**Risk levels:** " + " · ".join(risk_parts))
    if story.confidence is not None:
        conf_line = f"**LLM confidence:** {story.confidence:.0%}"
        if story.prompt_version:
            conf_line += f" · prompt `{story.prompt_version}`"
        lines.append(conf_line)
    if story.rationale:
        rationale = story.rationale
        if len(rationale) > 400:
            rationale = rationale[:397] + "…"
        lines.append("")
        lines.append("## Why we entered")
        lines.append("")
        lines.append(f"> {rationale}")

    if story.deltas:
        lines.append("")
        lines.append("## Indicator drift since entry")
        lines.append("")
        lines.append("| Indicator | At entry | Latest | Δ |")
        lines.append("| --- | --- | --- | --- |")
        for d in story.deltas:
            entry_str = f"{d.entry_value:.4f}" if d.entry_value is not None else "n/a"
            latest_str = f"{d.latest_value:.4f}" if d.latest_value is not None else "n/a"
            delta_str = f"{d.delta:+.4f}" if d.delta is not None else "n/a"
            lines.append(f"| {d.name} | {entry_str} | {latest_str} | {delta_str} |")

    if news:
        lines.append("")
        lines.append(
            f"## News ({story.news_count} total · {story.bullish_news} bullish / "
            f"{story.bearish_news} bearish)"
        )
        lines.append("")
        # Most recent first — already in input order, but the
        # database fan-in will likely give them ASCending; reverse
        # here so the operator sees the latest at the top.
        for n in sorted(news, key=lambda n: n.at, reverse=True)[:5]:
            score_emoji = "▲" if n.score > 0 else ("▼" if n.score < 0 else "·")
            link = f" [↗]({n.url})" if n.url else ""
            lines.append(
                f"- {n.at:%Y-%m-%d %H:%M} {score_emoji} **{n.source}**: {n.headline}{link}"
            )

    if story.pnl_curve:
        first = story.pnl_curve[0]
        worst = min(story.pnl_curve, key=lambda p: p.unrealized_pct)
        best = max(story.pnl_curve, key=lambda p: p.unrealized_pct)
        lines.append("")
        lines.append("## P&L track")
        lines.append("")
        lines.append(f"- Started: {_format_pct(first.unrealized_pct)} at {first.at:%Y-%m-%d %H:%M}")
        lines.append(f"- Trough: {_format_pct(worst.unrealized_pct)} at {worst.at:%Y-%m-%d %H:%M}")
        lines.append(f"- Peak: {_format_pct(best.unrealized_pct)} at {best.at:%Y-%m-%d %H:%M}")
        lines.append(
            f"- Latest: {_format_pct(story.current_unrealized_pct)} at "
            f"{story.pnl_curve[-1].at:%Y-%m-%d %H:%M}"
        )

    return "\n".join(lines)


# ── Aggregator ────────────────────────────────────────────


def build_story(input: PositionStoryInput) -> PositionStory:
    """Compose the structured story from the input bundle.

    Pure function — no DB, no LLM. Safe to call from any thread,
    notifier callback, or HTTP request handler.
    """
    pnl_curve = _pnl_curve(
        entry_price=input.entry_price,
        quantity=input.quantity,
        prices=input.price_timeline,
    )
    if pnl_curve:
        latest_pnl = pnl_curve[-1]
        current_price = latest_pnl.price
        current_pct = latest_pnl.unrealized_pct
        current_usd = latest_pnl.unrealized_usd
    else:
        current_price = None
        current_pct = None
        current_usd = None

    latest_indicators = input.indicator_timeline[-1] if input.indicator_timeline else None
    deltas = _indicator_deltas(input.entry_indicators, latest_indicators)
    bullish, bearish = _classify_news(input.news)

    story = PositionStory(
        pair=input.pair,
        side=input.side,
        quantity=input.quantity,
        entry_price=input.entry_price,
        entry_at=input.entry_at,
        stop_loss=input.stop_loss,
        take_profit=input.take_profit,
        trailing_stop_pct=input.trailing_stop_pct,
        rationale=input.llm_reasoning,
        confidence=input.confidence,
        prompt_version=input.prompt_version,
        current_price=current_price,
        current_unrealized_pct=current_pct,
        current_unrealized_usd=current_usd,
        deltas=deltas,
        pnl_curve=pnl_curve,
        news_count=len(input.news),
        bullish_news=bullish,
        bearish_news=bearish,
    )
    md = _render_markdown(story, input.news)
    return PositionStory(
        pair=story.pair,
        side=story.side,
        quantity=story.quantity,
        entry_price=story.entry_price,
        entry_at=story.entry_at,
        stop_loss=story.stop_loss,
        take_profit=story.take_profit,
        trailing_stop_pct=story.trailing_stop_pct,
        rationale=story.rationale,
        confidence=story.confidence,
        prompt_version=story.prompt_version,
        current_price=story.current_price,
        current_unrealized_pct=story.current_unrealized_pct,
        current_unrealized_usd=story.current_unrealized_usd,
        deltas=story.deltas,
        pnl_curve=story.pnl_curve,
        news_count=story.news_count,
        bullish_news=story.bullish_news,
        bearish_news=story.bearish_news,
        markdown=md,
    )


__all__ = [
    "IndicatorDelta",
    "IndicatorObservation",
    "NewsEvent",
    "PnlPoint",
    "PositionStory",
    "PositionStoryInput",
    "PriceObservation",
    "build_story",
]
