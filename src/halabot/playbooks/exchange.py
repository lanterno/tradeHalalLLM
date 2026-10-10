"""The exchange: working orders, the market-order fill rule and costs (pure).

**Market order.** It fills whole on the first bar ``i`` of its session with
``ts_i >= active_at``; there is no participation cap, and the trade records
``participation = (reference_notional / price) / v_i``.

**Price.** The bar's open ``o_i`` (flag ``gap_fill``) when any of these
holds, otherwise ``clamp(vw_i, l_i, h_i)`` (``o_i`` when vw is null):

* the order waited 5 minutes or more for a print: ``ts_i - active_at >= 5 min``;
* the bar ends a gap: ``ts_i - ts_{i-1} >= 5 min`` inside the session;
* ``i`` is the session's first bar and ``ts_i >= open + 5 min`` (also
  flagged ``late_open``).

**SPY legs** use the identical rule at the identical bar start, with no
cost: SPY's open when the stock filled at its open, else SPY's clamped
VWAP. Fallback legs (official closes) are the simulator's.

**An exit unfilled at the deadline's close** falls back (``unfilled_exit``)
to the official close, else the next session's market, else the last
trade; the trade is always kept.

**Costs** (``one_way_bps``): ``study.cost_bps(rank)`` per side, or 1.5x,
2x, or the "surcharge" reporting model (half-spread doubled in the first 15
minutes after the session's first bar or 5 minutes after a gap, plus a
square-root impact).

**Fill models** (:class:`FillModel`, ``SimConfig.fill``). Everything above is
the D.5 market rule, :data:`MARKET_FILL`, the only model a trial uses. The
gate-only models that replicate the legacy studies live in ``legacy.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import numpy as np

from halabot.playbooks.clock import US, span_us
from halabot.playbooks.types import BarSeries, CostMode, Execution, SimConfig, WorkingOrder

SURCHARGE_AFTER_FIRST = 15 * 60  # seconds after the session's first bar
SURCHARGE_AFTER_GAP = 5 * 60  # seconds after a gap ends
GAP_FOR_SURCHARGE = 5 * 60


def first_eligible(bars: BarSeries, active_at_us: int, k: int) -> int | None:
    """Index of the first bar of path session ``k`` with ``ts >= active_at``, or None."""
    t = -(-active_at_us // US)  # ts (whole seconds) >= active_at  <=>  ts >= ceil(active_at)
    i = int(np.searchsorted(bars.ts, t, side="left"))
    if i < len(bars) and int(bars.k[i]) == k:
        return i
    return None


def market_fill(
    bars: BarSeries, i: int, active_at_us: int, *, gap_us: int, session_open_us: int
) -> tuple[float, tuple[str, ...]]:
    """(raw price, flags) of a market order active at ``active_at`` filling on bar ``i``."""
    ts_us = int(bars.ts[i]) * US
    first = i == 0 or int(bars.k[i - 1]) != int(bars.k[i])
    late = first and ts_us >= session_open_us + gap_us
    waited = ts_us - active_at_us >= gap_us
    after_gap = not first and ts_us - int(bars.ts[i - 1]) * US >= gap_us
    flags: tuple[str, ...] = ()
    if waited or after_gap or late:
        flags = ("gap_fill", "late_open") if late else ("gap_fill",)
        return float(bars.o[i]), flags
    vw = float(bars.vw[i])
    if math.isnan(vw):
        return float(bars.o[i]), flags
    return min(max(vw, float(bars.l[i])), float(bars.h[i])), flags


def spy_fill(spy: BarSeries, ts: int, k: int, *, use_open: bool) -> tuple[float, tuple[str, ...]]:
    """SPY's price at the bar starting at ``ts`` (epoch s) of path session ``k``.

    The same rule as the stock: the open when ``use_open``, else the clamped
    VWAP. If SPY has no bar with that start, its last print before it in the
    session (a close), else its first print after (an open), flagged
    ``spy_proxy``; NaN when SPY has no bar in the session at all.
    """
    j = int(np.searchsorted(spy.ts, ts, side="left"))
    if j < len(spy) and int(spy.ts[j]) == ts:
        vw = float(spy.vw[j])
        if use_open or math.isnan(vw):
            return float(spy.o[j]), ()
        return min(max(vw, float(spy.l[j])), float(spy.h[j])), ()
    if j > 0 and int(spy.k[j - 1]) == k:
        return float(spy.c[j - 1]), ("spy_proxy",)
    if j < len(spy) and int(spy.k[j]) == k:
        return float(spy.o[j]), ("spy_proxy",)
    return math.nan, ("spy_proxy",)


def in_surcharge_window(bars: BarSeries, i: int) -> bool:
    """Whether bar ``i`` is within 15 min of its session's first bar or 5 min of a gap."""
    k = int(bars.k[i])
    first = bars.session_start(k)
    ts = int(bars.ts[i])
    if ts - int(bars.ts[first]) < SURCHARGE_AFTER_FIRST:
        return True
    j = i
    while j > first and ts - int(bars.ts[j]) < SURCHARGE_AFTER_GAP:
        if int(bars.ts[j]) - int(bars.ts[j - 1]) >= GAP_FOR_SURCHARGE:
            return True
        j -= 1
    return False


def one_way_bps(
    base_bps: float,
    mode: CostMode,
    *,
    rank: int,
    surcharge_window: bool = False,
    sigma: float = math.nan,
    adv20_usd: float = math.nan,
    notional: float = 10_000.0,
) -> float:
    """One side's cost in bps under ``mode`` (``base_bps`` is ``study.cost_bps(rank)``)."""
    if mode == "study":
        return base_bps
    if mode == "study_x1.5":
        return 1.5 * base_bps
    if mode == "study_x2":
        return 2.0 * base_bps
    half_spread = 2.0 if 0 <= rank < 300 else 10.0
    impact = 5.0
    if sigma > 0 and adv20_usd > 0:
        impact = max(5.0, sigma * 1e4 * math.sqrt(notional / adv20_usd))
    return half_spread * (2.0 if surcharge_window else 1.0) + impact


@dataclass(frozen=True, slots=True)
class FallbackFill:
    """How an exit still working (or a position still held) at the deadline's close is filled.

    ``close_fallback``, ``no_market`` and ``unresolved`` are the market
    rule's (:func:`unfilled_exit`); ``last_close`` is the gate-only models'
    exit at the session's last bar (``legacy.py``).
    """

    rule: Literal["close_fallback", "no_market", "unresolved", "last_close"]
    price: float  # raw
    spy_price: float
    bar_ts: int | None = None  # the filling bar (no_market: the spare's; last_close: S's), epoch s
    i: int = -1  # its index in its series
    flags: tuple[str, ...] = ()


def unfilled_exit(
    *,
    official_close: float | None,
    spy_close: float,
    spare: BarSeries | None,
    spy_spare: BarSeries | None,
    spare_open_us: int | None,
    lag_us: int,
    gap_us: int,
    last_close: float,
    spy_last: tuple[float, tuple[str, ...]],
) -> FallbackFill:
    """The fallback for an exit unfilled at the deadline session's close; the trade is kept.

    * the session's official close, with SPY's (``close_fallback``);
    * else, the stock having no daily bar that session, the next session in
      the window by the market rule: its first bar at or after open +
      ``ORDER_LAG`` (``no_market``; ``spare`` is that session's bars);
    * else the last trade, and SPY's close of the same minute (``unresolved``).
    """
    if official_close is not None:
        return FallbackFill("close_fallback", official_close, spy_close, flags=("close_fallback",))
    if spare is not None and spy_spare is not None and spare_open_us is not None:
        active = spare_open_us + lag_us
        i = first_eligible(spare, active, 0)
        if i is not None:
            price, flags = market_fill(
                spare, i, active, gap_us=gap_us, session_open_us=spare_open_us
            )
            ts = int(spare.ts[i])
            spy_px, spy_flags = spy_fill(spy_spare, ts, 0, use_open="gap_fill" in flags)
            return FallbackFill(
                "no_market", price, spy_px, ts, i, ("no_market", *flags, *spy_flags)
            )
    spy_px, spy_flags = spy_last
    return FallbackFill("unresolved", last_close, spy_px, flags=("unresolved", *spy_flags))


# ── fill models ───────────────────────────────────────────────


class FillModel(Protocol):
    """Which bar fills a working order, at what price, and how a held position ends.

    The simulator asks it for every order (``SimConfig.fill``; None is
    :data:`MARKET_FILL`). A model never fills on a bar that starts before
    the order is active.
    """

    @property
    def name(self) -> str:
        """Recorded in the run's ``config["sim"]["fill"]``."""
        ...

    @property
    def gate_only(self) -> bool:
        """True for a model that replicates a legacy study (``legacy.py``).

        The simulator then admits a buy without the screen or the entry
        window (still inside the session, one position at a time) and skips
        the deadline's flatten; ``sim.run`` accepts it only under a gate
        unlock. Never True for a trial.
        """
        ...

    def due(self, bars: BarSeries, order: WorkingOrder) -> int | None:
        """The bar ``order`` fills on: one of session ``order.k`` with ``ts >= active_at``.

        None while no such bar exists, or when the model fills this order
        some other way (a gate model's exits: :meth:`at_close`).
        """
        ...

    def price(
        self, bars: BarSeries, i: int, order: WorkingOrder, *, gap_us: int, session_open_us: int
    ) -> tuple[float, tuple[str, ...]]:
        """(raw price, flags) of ``order`` filling on bar ``i``."""
        ...

    def spy_price(self, spy: BarSeries, ex: Execution) -> tuple[float, tuple[str, ...]]:
        """SPY's price (raw) and flags for the leg that goes with the stock fill ``ex``."""
        ...

    def at_close(self, bars: BarSeries, spy: BarSeries, k: int) -> FallbackFill | None:
        """How an exit still open at path session ``k``'s close (the deadline) is filled.

        None: the market rule's fallback (:func:`unfilled_exit`: the
        official close, the next session, or the last trade).
        """
        ...


class MarketFill:
    """The D.5 market rule (module docstring): the only fill model a trial uses."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return "market"

    @property
    def gate_only(self) -> bool:
        return False

    def due(self, bars: BarSeries, order: WorkingOrder) -> int | None:
        return first_eligible(bars, order.active_at_us, order.k)

    def price(
        self, bars: BarSeries, i: int, order: WorkingOrder, *, gap_us: int, session_open_us: int
    ) -> tuple[float, tuple[str, ...]]:
        return market_fill(
            bars, i, order.active_at_us, gap_us=gap_us, session_open_us=session_open_us
        )

    def spy_price(self, spy: BarSeries, ex: Execution) -> tuple[float, tuple[str, ...]]:
        assert ex.bar_ts is not None
        return spy_fill(spy, ex.bar_ts, ex.k, use_open="gap_fill" in ex.flags)

    def at_close(self, bars: BarSeries, spy: BarSeries, k: int) -> FallbackFill | None:
        return None


MARKET_FILL: Final[FillModel] = MarketFill()


def fill_model(cfg: SimConfig) -> FillModel:
    """The fill model ``cfg`` names: its ``fill``, else the market rule."""
    return cfg.fill if cfg.fill is not None else MARKET_FILL


class Exchange:
    """Working orders and their fills; pure, so a live driver can reuse it on live bars."""

    def __init__(self, cfg: SimConfig) -> None:
        self._cfg = cfg
        self._fill = fill_model(cfg)
        self._gap_us = span_us(cfg.gap)
        self._working: dict[str, WorkingOrder] = {}

    @property
    def fill(self) -> FillModel:
        return self._fill

    @property
    def working(self) -> tuple[WorkingOrder, ...]:
        """Working orders in submission order."""
        return tuple(self._working.values())

    def get(self, order_id: str) -> WorkingOrder | None:
        return self._working.get(order_id)

    def submit(self, order: WorkingOrder) -> None:
        if order.order_id in self._working:
            raise ValueError(f"order {order.order_id} is already working")
        self._working[order.order_id] = order

    def replace(self, order: WorkingOrder) -> None:
        """Re-time a working order (an exit carried to the next session's open)."""
        if order.order_id not in self._working:
            raise KeyError(order.order_id)
        self._working[order.order_id] = order

    def cancel(self, order_id: str) -> WorkingOrder | None:
        return self._working.pop(order_id, None)

    def due(self, bars: BarSeries, order: WorkingOrder) -> int | None:
        """The bar ``order`` will fill on, by the fill model (None: not on a bar yet)."""
        return self._fill.due(bars, order)

    def on_bar(self, i: int, bars: BarSeries, *, session_open_us: int) -> list[Execution]:
        """Fill every working order the fill model makes due on bar ``i`` (sells first).

        Whether ``i`` is its session's first bar comes from ``bars.k``; a
        session's first bar is never "after a gap" from the previous session,
        only late (``ts_i >= open + 5 min``).
        """
        k = int(bars.k[i])
        ts = int(bars.ts[i])
        out: list[Execution] = []
        orders = sorted(self._working.values(), key=lambda o: o.side != "sell")
        for order in orders:
            if order.k != k or ts * US < order.active_at_us:
                continue
            due = self._fill.due(bars, order)
            if due is None:
                continue  # the model fills it otherwise (a gate model's exit, at the close)
            if due != i:
                raise AssertionError(f"{order.order_id} was due on bar {due}, tested on {i}")
            assert ts * US >= order.active_at_us  # every fill bar starts at or after active_at
            price, flags = self._fill.price(
                bars, i, order, gap_us=self._gap_us, session_open_us=session_open_us
            )
            ref_qty = self._cfg.reference_notional / price
            out.append(
                Execution(
                    order=order,
                    k=k,
                    i=i,
                    bar_ts=ts,
                    price=price,
                    filled_at_us=(ts + 60) * US,
                    flags=flags,
                    participation=ref_qty / float(bars.v[i]) if bars.v[i] > 0 else math.inf,
                )
            )
            del self._working[order.order_id]
        return out

    def expire(self, k: int) -> list[WorkingOrder]:
        """Remove and return the orders working in path session ``k`` (day orders, at the close)."""
        gone = [o for o in self._working.values() if o.k == k]
        for o in gone:
            del self._working[o.order_id]
        return gone


__all__ = [
    "MARKET_FILL",
    "Exchange",
    "FallbackFill",
    "FillModel",
    "MarketFill",
    "fill_model",
    "first_eligible",
    "in_surcharge_window",
    "market_fill",
    "one_way_bps",
    "spy_fill",
    "unfilled_exit",
]
