"""Scenario simulator — project current positions through a synthetic
kline path and report fills, drawdowns, and equity impact.

Round-4 wave 5.F: lets an operator ask "what happens to my open
positions if the Fed surprises hawkish tomorrow?" by piping a stress
scenario from ``crypto/stress.py`` through the same SL/TP logic the
position monitor uses, then summarising the result. The point is
*manual de-risking before known events* — not a replacement for the
position monitor, which keeps running on real prices regardless.

Design:

* **Symbol-agnostic.** Operates on a `SimulatedPosition` dataclass
  carrying entry/qty/SL/TP. The caller is responsible for mapping
  live `CryptoTrade` rows or `BrokerPosition` records into this
  shape — keeps the simulator free of DB / domain imports.
* **Path-aware fills.** SL/TP are checked bar-by-bar against the
  bar's high/low (not just close), matching the real monitor's
  semantic — a wick that pierces the SL fills the order even if
  the close recovers. Pin: when both SL and TP could fire in the
  same bar (rare; happens on volatile gap bars), SL wins, since
  the wire-up assumes worst-case execution.
* **Trailing-stop aware.** Optionally supplies a
  `trailing_stop_pct` per position — re-prices the SL on each new
  high while the position is open, mirroring `core/sl_tp.py`'s
  trailing logic.
* **Halal alignment.** Simulates only the close path of an existing
  position. The simulator never opens a new trade and never does
  anything destructive — pure projection / what-if.

Pure-Python; no NumPy / DB / async. The simulator is fully
deterministic given identical inputs, so a test can pin both the
fill price and the drawdown trough with exact equality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from halal_trader.domain.models import Kline


@dataclass(frozen=True)
class SimulatedPosition:
    """One open long position about to be put through the scenario.

    All prices are quote-currency (USD/USDT). ``stop_loss`` and
    ``take_profit`` are absolute prices, not percentages — the
    simulator doesn't know whether the operator computes them
    from ATR / fixed pct / etc.
    """

    pair: str
    quantity: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_pct: float | None = None  # e.g. 0.005 = 0.5% trail


@dataclass(frozen=True)
class PositionProjection:
    """Per-position outcome after walking the scenario klines.

    ``filled`` is True iff the position closed during the scenario
    via SL / TP / trailing stop. ``fill_reason`` is one of
    ``"stop_loss"`` / ``"take_profit"`` / ``"trailing_stop"`` /
    ``""`` (still open).

    ``min_equity`` / ``max_equity`` are the lowest / highest
    *position-level* mark-to-market values seen during the walk;
    ``min_equity`` − ``starting_equity`` is the worst drawdown the
    operator would observe before the close (or before scenario
    end if the position survives).

    ``end_equity`` is the position's M2M at the last bar — equal
    to the fill notional if it closed, else qty × last_close.
    """

    pair: str
    starting_equity: float
    end_equity: float
    min_equity: float
    max_equity: float
    filled: bool
    fill_reason: str
    fill_bar_index: int | None
    fill_price: float | None


@dataclass(frozen=True)
class ScenarioReport:
    """Aggregate result over every position in the scenario."""

    total_starting_equity: float
    total_end_equity: float
    portfolio_pnl: float
    portfolio_pnl_pct: float
    portfolio_drawdown: float  # most negative trough across all positions, summed
    projections: list[PositionProjection] = field(default_factory=list)


def _project_one(position: SimulatedPosition, klines: Sequence[Kline]) -> PositionProjection:
    """Walk one position through the scenario and produce a projection."""
    starting = position.entry_price * position.quantity
    if not klines:
        return PositionProjection(
            pair=position.pair,
            starting_equity=starting,
            end_equity=starting,
            min_equity=starting,
            max_equity=starting,
            filled=False,
            fill_reason="",
            fill_bar_index=None,
            fill_price=None,
        )

    sl = position.stop_loss
    tp = position.take_profit
    trail_pct = position.trailing_stop_pct
    high_water = position.entry_price  # for trailing-stop ratchet
    min_equity = starting
    max_equity = starting

    # Track whether the current SL value came from the trailing
    # ratchet (used for the fill-reason label when SL fires).
    sl_is_trailing = False

    for i, bar in enumerate(klines):
        # Pin: SL/TP fills are checked against the trail level *as
        # of the previous bar*, not this bar's high. Ratcheting on
        # the same-bar high and then checking the same-bar low
        # against the freshly-ratcheted trail leads to phantom
        # trailing-stop fills (the high triggers the ratchet inside
        # the very bar that recovers). The trail update is applied
        # AFTER the in-bar SL/TP check so it only affects subsequent
        # bars. Matches `core/sl_tp.py`'s "trail observed at end of
        # bar, fired next bar" semantic.

        # Pin: when both SL and TP could fill in the same bar, SL
        # wins. Worst-case execution is the safer projection for
        # an operator deciding whether to de-risk.
        sl_hit = sl is not None and bar.low <= sl
        tp_hit = tp is not None and bar.high >= tp

        # Track per-bar mark-to-market range.
        bar_min_equity = bar.low * position.quantity
        bar_max_equity = bar.high * position.quantity
        min_equity = min(min_equity, bar_min_equity)
        max_equity = max(max_equity, bar_max_equity)

        if sl_hit:
            fill_price = sl  # type: ignore[assignment]
            return PositionProjection(
                pair=position.pair,
                starting_equity=starting,
                end_equity=fill_price * position.quantity,
                min_equity=min_equity,
                max_equity=max_equity,
                filled=True,
                fill_reason="trailing_stop" if sl_is_trailing else "stop_loss",
                fill_bar_index=i,
                fill_price=float(fill_price),
            )
        if tp_hit:
            fill_price = tp  # type: ignore[assignment]
            return PositionProjection(
                pair=position.pair,
                starting_equity=starting,
                end_equity=fill_price * position.quantity,
                min_equity=min_equity,
                max_equity=max_equity,
                filled=True,
                fill_reason="take_profit",
                fill_bar_index=i,
                fill_price=float(fill_price),
            )

        # End-of-bar trailing-stop ratchet — only affects subsequent
        # bars (see comment above re: same-bar phantom-fill bug).
        if trail_pct is not None and bar.high > high_water:
            high_water = bar.high
            new_trail = high_water * (1.0 - trail_pct)
            if sl is None or new_trail > sl:
                sl = new_trail
                sl_is_trailing = True

    # Survived the scenario.
    last_close = klines[-1].close
    return PositionProjection(
        pair=position.pair,
        starting_equity=starting,
        end_equity=last_close * position.quantity,
        min_equity=min_equity,
        max_equity=max_equity,
        filled=False,
        fill_reason="",
        fill_bar_index=None,
        fill_price=None,
    )


def simulate(positions: Sequence[SimulatedPosition], klines: Sequence[Kline]) -> ScenarioReport:
    """Project every position through the same kline path.

    Returns a :class:`ScenarioReport` that the dashboard / CLI can
    render directly. The function is total — empty positions or
    empty klines both produce a sensible zero-everywhere report.
    """
    if not positions:
        return ScenarioReport(
            total_starting_equity=0.0,
            total_end_equity=0.0,
            portfolio_pnl=0.0,
            portfolio_pnl_pct=0.0,
            portfolio_drawdown=0.0,
            projections=[],
        )

    projections = [_project_one(pos, klines) for pos in positions]

    total_start = sum(p.starting_equity for p in projections)
    total_end = sum(p.end_equity for p in projections)
    pnl = total_end - total_start
    pnl_pct = pnl / total_start if total_start > 0 else 0.0
    # Portfolio-level drawdown = sum of the worst trough across
    # positions. This is conservative (assumes troughs align in
    # time, which they may not) — but for an operator deciding
    # whether to de-risk before an event, the conservative number
    # is the right one to show.
    drawdown = sum(p.min_equity - p.starting_equity for p in projections)

    return ScenarioReport(
        total_starting_equity=total_start,
        total_end_equity=total_end,
        portfolio_pnl=pnl,
        portfolio_pnl_pct=pnl_pct,
        portfolio_drawdown=drawdown,
        projections=projections,
    )


# ── Convenience ──────────────────────────────────────────


def render_report(report: ScenarioReport) -> str:
    """Pretty multi-line text suitable for CLI / Slack / Telegram.

    The format matches `crypto/stress.render_report` for visual
    consistency — operators running the stress harness and the
    scenario simulator see similar output shapes."""
    lines = ["=== Scenario projection ==="]
    lines.append(
        f"Portfolio: ${report.total_starting_equity:,.2f} → "
        f"${report.total_end_equity:,.2f} "
        f"({report.portfolio_pnl:+,.2f} / {report.portfolio_pnl_pct:+.2%})"
    )
    lines.append(f"Worst-case drawdown: ${report.portfolio_drawdown:+,.2f}")
    lines.append("")
    for p in report.projections:
        if p.filled:
            line = (
                f"  {p.pair:<12} CLOSED at bar {p.fill_bar_index} "
                f"via {p.fill_reason} @ ${p.fill_price:,.4f} → "
                f"${p.end_equity:,.2f} (start ${p.starting_equity:,.2f})"
            )
        else:
            line = (
                f"  {p.pair:<12} OPEN at scenario end → "
                f"${p.end_equity:,.2f} (start ${p.starting_equity:,.2f}, "
                f"trough ${p.min_equity:,.2f})"
            )
        lines.append(line)
    return "\n".join(lines)
