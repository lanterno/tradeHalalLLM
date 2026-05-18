"""Post-trade analytics — slippage + market impact + cost — Round-5 Wave 12.H.

After a parent order completes, the bot needs to grade the execution
quality: how much slippage vs. the arrival-price benchmark, how much
market-impact cost did the order's footprint produce, and how does
this compare to TWAP / VWAP benchmarks. This is the **execution-
quality scorecard** that feeds back into the routing-algo selection
logic (TWAP / VWAP / iceberg / smart-router).

Pinned semantics:

- **Closed-set Benchmark ladder** — ARRIVAL / TWAP / VWAP / CLOSE.
- **Slippage convention** — for BUY, slippage = (avg_fill -
  benchmark) / benchmark; for SELL, sign is flipped so that *positive
  slippage = bad* in both cases.
- **Market-impact cost** is the difference between the trade's price
  trajectory and a no-impact baseline price (typically arrival
  price); we expose the simple version: ``avg_fill - arrival_price``
  signed so positive = harmful for buy.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from halal_trader.trading.twap import Side


class Benchmark(str, Enum):
    """Closed-set execution benchmarks."""

    ARRIVAL = "arrival"
    TWAP = "twap"
    VWAP = "vwap"
    CLOSE = "close"


@dataclass(frozen=True)
class Fill:
    """A single fill received from the broker."""

    fill_id: str
    quantity: float
    price: float
    fill_time: datetime

    def __post_init__(self) -> None:
        if not self.fill_id or not self.fill_id.strip():
            raise ValueError("fill_id must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.fill_time.tzinfo is None:
            raise ValueError("fill_time must be timezone-aware")


@dataclass(frozen=True)
class ExecutionInputs:
    """Inputs for execution-quality analysis."""

    parent_id: str
    symbol: str
    side: Side
    fills: tuple[Fill, ...]
    arrival_price: float
    twap_price: float | None = None
    vwap_price: float | None = None
    close_price: float | None = None

    def __post_init__(self) -> None:
        if not self.parent_id or not self.parent_id.strip():
            raise ValueError("parent_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.fills:
            raise ValueError("fills must be non-empty")
        if self.arrival_price <= 0:
            raise ValueError("arrival_price must be positive")
        for name, val in (
            ("twap_price", self.twap_price),
            ("vwap_price", self.vwap_price),
            ("close_price", self.close_price),
        ):
            if val is not None and val <= 0:
                raise ValueError(f"{name}, if set, must be positive")


@dataclass(frozen=True)
class ExecutionReport:
    """Execution-quality report."""

    parent_id: str
    side: Side
    total_quantity: float
    average_fill_price: float
    arrival_slippage_bps: float
    twap_slippage_bps: float | None
    vwap_slippage_bps: float | None
    close_slippage_bps: float | None
    market_impact_pct: float
    fill_duration_seconds: float

    def __post_init__(self) -> None:
        if self.total_quantity <= 0:
            raise ValueError("total_quantity must be positive")
        if self.average_fill_price <= 0:
            raise ValueError("average_fill_price must be positive")
        if self.fill_duration_seconds < 0:
            raise ValueError("fill_duration_seconds must be non-negative")


def _avg_price(fills: Sequence[Fill]) -> tuple[float, float]:
    """Return (total_quantity, volume-weighted-average-price)."""
    total_qty = sum(f.quantity for f in fills)
    if total_qty == 0:
        return 0.0, 0.0
    avg = sum(f.quantity * f.price for f in fills) / total_qty
    return total_qty, avg


def _slippage_bps(side: Side, avg_fill: float, benchmark: float) -> float:
    """Compute slippage in basis points — positive = bad in both directions."""
    if benchmark <= 0:
        return 0.0
    raw = (avg_fill - benchmark) / benchmark * 10000.0
    if side is Side.SELL:
        raw = -raw
    return raw


def analyze(inputs: ExecutionInputs) -> ExecutionReport:
    """Run post-trade analytics on a completed parent order."""
    total_qty, avg_fill = _avg_price(inputs.fills)
    arrival_slip = _slippage_bps(inputs.side, avg_fill, inputs.arrival_price)
    twap_slip = (
        _slippage_bps(inputs.side, avg_fill, inputs.twap_price)
        if inputs.twap_price is not None
        else None
    )
    vwap_slip = (
        _slippage_bps(inputs.side, avg_fill, inputs.vwap_price)
        if inputs.vwap_price is not None
        else None
    )
    close_slip = (
        _slippage_bps(inputs.side, avg_fill, inputs.close_price)
        if inputs.close_price is not None
        else None
    )

    # Market impact in pct: signed so positive = harmful for buy
    impact_raw = (avg_fill - inputs.arrival_price) / inputs.arrival_price
    impact_pct = -impact_raw if inputs.side is Side.SELL else impact_raw

    sorted_fills = sorted(inputs.fills, key=lambda f: f.fill_time)
    duration = (sorted_fills[-1].fill_time - sorted_fills[0].fill_time).total_seconds()

    return ExecutionReport(
        parent_id=inputs.parent_id,
        side=inputs.side,
        total_quantity=total_qty,
        average_fill_price=avg_fill,
        arrival_slippage_bps=arrival_slip,
        twap_slippage_bps=twap_slip,
        vwap_slippage_bps=vwap_slip,
        close_slippage_bps=close_slip,
        market_impact_pct=impact_pct,
        fill_duration_seconds=duration,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_report(report: ExecutionReport) -> str:
    head = (
        f"Post-trade {report.parent_id} {report.side.value}: "
        f"{report.total_quantity:.2f} avg=${report.average_fill_price:.4f} "
        f"in {report.fill_duration_seconds:.0f}s"
    )
    lines = [
        head,
        f"  arrival slippage: {report.arrival_slippage_bps:+.2f} bps",
    ]
    if report.twap_slippage_bps is not None:
        lines.append(f"  TWAP slippage:    {report.twap_slippage_bps:+.2f} bps")
    if report.vwap_slippage_bps is not None:
        lines.append(f"  VWAP slippage:    {report.vwap_slippage_bps:+.2f} bps")
    if report.close_slippage_bps is not None:
        lines.append(f"  close slippage:   {report.close_slippage_bps:+.2f} bps")
    lines.append(f"  market impact:    {report.market_impact_pct * 100:+.4f}%")
    return _scrub("\n".join(lines))
