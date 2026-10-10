"""The overreaction bounce: H1's playbook (spec §G.3-G.4; constants frozen).

After a large drop on non-structural negative news (NSN_CORE: an analyst
downgrade or an earnings miss), buy once the selling is exhausted, and exit
at a 50% retrace target, a close below the capitulation low, a structural
abort, a compliance exit or the time stop. It is a pure state machine over
the simulator's inputs (the contract at the top of ``playbook.py``).

**References** (``start``, S units). ``at_news`` is the ``at`` of the item
that made the story NSN, ``start_case`` the story's:

* "in" (news inside S's session): P0 is the close of the last bar of S with
  ``ts + 60 s <= at_news``, SPY0 SPY's close at the same rule
  (``spy_prev_close_s`` if none), and the anchor is the first bar with
  ``ts >= floor_minute(at_news)``. When the stock has no such bar, P0 is
  ``prev_close_s`` and SPY0 ``spy_prev_close_s`` whatever SPY printed: D
  then measures both from the previous closes, never the stock's overnight
  move against SPY's intraday one;
* "out": P0 = ``prev_close_s``, SPY0 = ``spy_prev_close_s``, the anchor is
  S's first bar.

``thr = max(k_sigma * sigma, floor)``, σ from ``PreEvent`` (sessions closed
strictly before ``at_news``). P0 and SPY0 of an "in" story are read when the
anchor bar is visible: every bar before it is visible by then.

**Per bar t of S** from the anchor (visible bars only; S's own prices, A = 1):

* ``D_t = (c_t/P0 - 1) - (spy_c_t/SPY0 - 1)``, with ``spy_c_t`` SPY's bar of
  the same start or its latest with start <= ``ts_t`` (SPY0 before any);
* ``L_t = min l`` over [anchor, t], ``t_L`` the latest bar attaining it;
* ``AVWAP_t = sum(w v) / sum(v)`` over [anchor, t], ``w = clamp(vw, l, h)``,
  or ``(h + l + c)/3`` when vw is null.

**States** (:class:`BounceState`):

1. DETECTED -> WATCHING at the start (``max(nsn_at, open(S))``) if eligible,
   else DISMISSED with the eligibility's reason. ``blocked_open`` is the
   simulator's: it never builds a playbook for a blocked story.
2. Triggered (a ``Transition(state, "triggered")``) once some bar of
   [anchor, t] has ``D <= -thr``.
3. WATCHING -> ARMED when triggered and ``ts_t - ts_{t_L} >= quiet``;
   ARMED -> WATCHING when a new low (or an equal one) resets that clock.
4. ARMED -> ENTERING at the visible time T of bar t (the ``BarIn``) when
   E1 ``T in [open + 20 min, close - 60 min]``, E2 ``c_t > AVWAP_t``,
   E3 ``c_t - L_t <= 0.25 (P0 - L_t)``, E4 no SPY close of S so far
   ``<= 0.98 spy_prev_close_s``, E5 ``card_at(T).family == "NSN_CORE"``:
   a market buy with :class:`~halabot.playbooks.types.TradeFacts`. ``L* =
   L_t`` and ``TGT = L* + 0.5 (P0 - L*)`` are frozen there.
5. ENTERING -> ENTERED on the fill; EXPIRED (``entry_unfilled``) if the buy
   expires or is cancelled at a close or a flatten, ``entry_rejected:<why>``
   if the simulator refuses it. A buy filled at or after the deadline
   session's flatten (the exchange fills a bar before the flatten marker of
   the same instant) goes straight on to EXITING (``time_stop``): the
   simulator's flatten already sells it, and no exit is judged.
6. ENTERED -> EXITING, first of: X3 ``abort`` > X1 ``stop`` (a bar close
   below L*, in S units) > X2 ``target`` (a close >= TGT). Exits are judged
   on every bar from the entry's fill bar on, X3 also at the fill and on
   each item. X5 ``compliance`` and X4 ``time_stop`` are the simulator's
   sells (its pre-open screen check and its flatten at close - 5 min of the
   deadline session); the playbook follows them (``ComplianceIn``, the
   ``flatten`` ``SessionIn``).
7. WATCHING / ARMED -> EXPIRED at the entry cutoff (``cutoff``, a timer at
   ``close - 60 min``, after any bar visible at that instant), on a SPY close
   of S <= ``(1 + market_break) * spy_prev_close_s`` (``market_break``), or
   on a ``veto``: the card is no longer NSN_CORE, or another story of the
   symbol delivers a structural item (below).
8. One entry per story: every terminal state (EXITED, EXPIRED, DISMISSED)
   ends with ``Finish``.

**Structural items** (spec §B.1: "veto before entry, abort after") reach
the playbook from the story's own card and, for the symbol's other stories
(a blocked story, or the next session's), only as ``NewsIn``. An item
*brings a structural item* when it is the story's own and adds a structural
type to its card's vetoes, or it is another story's and that story's card
is structural (every such item counts, not only the first).

* H1 (``require_family``): any structural item known while WATCHING or
  ARMED ends the story EXPIRED (``veto``), the own one through E5's family
  check and another story's on its ``NewsIn``. Another story's item usable
  at the entry decision's own instant vetoes it too (the bar comes before
  the news at one instant, and E5 would have read an own item then): the
  ``Finish`` cancels the buy before it works. After the decision, X3 aborts
  on any structural item of the story's card or one brought since.
* The atlas (``require_family=False``, spec §F) drops E5 and the vetoes, so
  a structural item already known at the entry decision does not abort (the
  story's own type may be structural): X3 aborts only on an item that
  brings a structural item later than the decision.

Prices on later path sessions are put in S units with
``MarketView.to_s_units`` (splits and dividends inside the path are not
moves).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Final, Protocol

import numpy as np

from halabot.playbooks.clock import (
    ENTRY_CUTOFF_BEFORE_CLOSE,
    ENTRY_START_AFTER_OPEN,
    FLATTEN_BEFORE_CLOSE,
    US,
    from_us,
    to_us,
)
from halabot.playbooks.interfaces import StoryView
from halabot.playbooks.playbook import TRIGGERED, Ctx
from halabot.playbooks.types import (
    BarIn,
    BarSeries,
    ComplianceIn,
    FillIn,
    Finish,
    Input,
    Intent,
    NewsIn,
    OrderClosedIn,
    SessionIn,
    SetTimer,
    Submit,
    TimerIn,
    TradeFacts,
    Transition,
)
from halal_trader.events.taxonomy import STRUCTURAL

NAME: Final = "overreaction_bounce"
VERSION: Final = "1"
FAMILY: Final = "NSN_CORE"
SPY: Final = "SPY"
CUTOFF_TIMER: Final = "bounce.cutoff"
NO_PRE_EVENT: Final = "no_pre_event"
NO_AT_NEWS: Final = "no_at_news"
# The rule's words in the pre-registration's ``playbook`` block (spec §G.14).
RECLAIM: Final = "close > AVWAP from anchor"
STOP: Final = "bar close < L*"
ABORT: Final = "structural item"
PRIORITY: Final = ("abort", "stop", "target")
_TICK: Final = timedelta(microseconds=1)  # a card at ``t - _TICK`` lacks the items of ``t``


def _minutes(d: timedelta) -> int | float:
    """A span in minutes: an int when whole (the pre-registration writes ``20``, not ``20.0``)."""
    whole, rest = divmod(d, timedelta(minutes=1))
    return whole if not rest else d.total_seconds() / 60.0


def _structural(vetoes: Iterable[str]) -> frozenset[str]:
    return frozenset(v for v in vetoes if v in STRUCTURAL)


@dataclass(frozen=True, slots=True)
class BounceParams:
    """The rule's constants (spec §G.4), frozen by the pre-registration.

    ``require_family=False`` lets the atlas run the rule on other negative
    types (no E5, no veto expiry). The entry window must lie inside the
    simulator's admission window and the flatten must be the simulator's:
    both are enforced there (``rules.admit_buy``, ``sim``), not here.

    :meth:`as_config` is the pre-registration's ``playbook`` block;
    :meth:`run_config` holds what a run sets beside it (the hold, which the
    cell names, and the atlas's family switch).
    """

    k_sigma: float = 2.0
    floor: float = 0.03
    quiet: timedelta = timedelta(minutes=20)
    entry_start_after_open: timedelta = timedelta(minutes=20)
    entry_cutoff_before_close: timedelta = timedelta(minutes=60)
    max_retrace_at_entry: float = 0.25
    target_retrace: float = 0.50
    market_break: float = -0.02
    flatten_before_close: timedelta = timedelta(minutes=5)
    hold_sessions: int = 1  # 1 = ID, 3 = MD3
    require_family: bool = True

    def __post_init__(self) -> None:
        if self.hold_sessions < 1:
            raise ValueError("hold_sessions must be at least 1")
        if self.flatten_before_close != FLATTEN_BEFORE_CLOSE:
            raise ValueError("the simulator flattens at close - 5 min; the playbook cannot move it")
        if (
            self.entry_start_after_open < ENTRY_START_AFTER_OPEN
            or self.entry_cutoff_before_close < ENTRY_CUTOFF_BEFORE_CLOSE
        ):
            raise ValueError("the entry window must lie inside the simulator's buy window")

    @property
    def variant(self) -> str:
        """``"ID"`` for one session, ``"MD<n>"`` otherwise (``"MD3"``)."""
        return "ID" if self.hold_sessions == 1 else f"MD{self.hold_sessions}"

    def as_config(self) -> dict[str, object]:
        """The pre-registration's ``playbook`` block (spec §G.14), key for key and type for type.

        Spans are in whole minutes as ints, so ``config_hash`` of a PREREG
        built from it equals the spec's literal one.
        """
        return {
            "k_sigma": self.k_sigma,
            "floor": self.floor,
            "quiet_min": _minutes(self.quiet),
            "entry_start_min": _minutes(self.entry_start_after_open),
            "entry_cutoff_min": _minutes(self.entry_cutoff_before_close),
            "max_retrace_at_entry": self.max_retrace_at_entry,
            "target_retrace": self.target_retrace,
            "market_break": self.market_break,
            "flatten_min": _minutes(self.flatten_before_close),
            "reclaim": RECLAIM,
            "stop": STOP,
            "abort": ABORT,
            "priority": list(PRIORITY),
            "compliance_exit": True,
        }

    def run_config(self) -> dict[str, object]:
        """What a run sets beside :meth:`as_config`: the hold and the family check."""
        return {
            "hold_sessions": self.hold_sessions,
            "variant": self.variant,
            "require_family": self.require_family,
        }


class BounceState(StrEnum):
    DETECTED = "DETECTED"
    WATCHING = "WATCHING"
    ARMED = "ARMED"
    ENTERING = "ENTERING"
    ENTERED = "ENTERED"
    EXITING = "EXITING"
    EXITED = "EXITED"
    EXPIRED = "EXPIRED"
    DISMISSED = "DISMISSED"


LIVE: Final = frozenset(
    {BounceState.WATCHING, BounceState.ARMED, BounceState.ENTERING, BounceState.ENTERED}
)
_WAITING: Final = frozenset({BounceState.WATCHING, BounceState.ARMED})


class PreEventLike(Protocol):
    """What the rule reads of ``context.PreEvent`` (levels in S units)."""

    @property
    def prev_close_s(self) -> float: ...
    @property
    def spy_prev_close_s(self) -> float: ...
    @property
    def sigma(self) -> float: ...
    @property
    def beta(self) -> float: ...
    @property
    def adv20_usd(self) -> float: ...


class EligibilityLike(Protocol):
    """What the rule reads of ``context.Eligibility``."""

    @property
    def eligible(self) -> bool: ...
    @property
    def reason(self) -> str: ...
    @property
    def liquidity_rank(self) -> int | None: ...
    @property
    def cost_bps(self) -> float: ...
    @property
    def tech(self) -> bool: ...


def _w(h: float, low: float, c: float, vw: float) -> float:
    """A bar's AVWAP weight price: its VWAP clamped to [l, h], else (h + l + c)/3."""
    if math.isnan(vw):
        return (h + low + c) / 3.0
    return min(max(vw, low), h)


class OverreactionBounce:
    """One story's H1 state machine (module docstring); built by :class:`BounceFactory`."""

    name: Final = NAME
    version: Final = VERSION

    def __init__(
        self,
        story: StoryView,
        pre: PreEventLike | None,
        elig: EligibilityLike,
        params: BounceParams | None = None,
    ) -> None:
        self.params = params if params is not None else BounceParams()
        self._symbol = story.symbol
        self._pre = pre
        self._elig = elig
        self._state = BounceState.DETECTED
        # references, set at the start (P0/SPY0 of an "in" story at the anchor)
        self._in = False
        self._at_news_us = 0
        self._anchor_from = 0  # epoch s: the anchor is the first bar of S at or after it
        self._anchor_ts: int | None = None
        self._p0 = math.nan
        self._spy0 = math.nan
        self._thr = math.nan
        self._break_level = math.nan
        self._entry_from: datetime | None = None
        self._entry_until: datetime | None = None
        # running quantities over [anchor, t]
        self._next_i = 0  # the next bar of the symbol to read
        self._spy_next = 0  # the next SPY bar to test for a market break
        self._low = math.inf
        self._low_ts = 0
        self._wv = 0.0
        self._v = 0.0
        self._last_close = math.nan
        self._last_ts = 0
        self._triggered = False
        self._broken = False
        # the entry and the exits
        self._decided_at: datetime | None = None
        self._low_star = math.nan
        self._target = math.nan
        self._entry_bar_ts: int | None = None
        self._exit_next = 0
        self._structural_at: datetime | None = None  # the last item that brought a structural one

    # ── the protocol ──

    @property
    def path_sessions(self) -> int:
        return self.params.hold_sessions

    def state(self) -> str:
        return str(self._state)

    def live(self) -> bool:
        return self._state in LIVE

    def start(self, ctx: Ctx) -> list[Intent]:
        p = self.params
        if not self._elig.eligible:
            return self._end(BounceState.DISMISSED, str(self._elig.reason))
        pre = self._pre
        if pre is None:
            return self._end(BounceState.DISMISSED, NO_PRE_EVENT)
        story = ctx.story
        at_news = story.at_news()
        if at_news is None:
            return self._end(BounceState.DISMISSED, NO_AT_NEWS)
        s = ctx.session
        self._in = story.start_case() == "in"
        self._at_news_us = to_us(at_news)
        self._thr = max(p.k_sigma * pre.sigma, p.floor)
        self._break_level = (1.0 + p.market_break) * pre.spy_prev_close_s
        self._entry_from = s.open + p.entry_start_after_open
        self._entry_until = s.close - p.entry_cutoff_before_close
        if self._in:
            self._anchor_from = self._at_news_us // (60 * US) * 60  # floor_minute(at_news)
        else:
            self._anchor_from = to_us(s.open) // US
            self._p0 = pre.prev_close_s
            self._spy0 = pre.spy_prev_close_s

        out: list[Intent] = [
            self._go(BounceState.WATCHING, "detected"),
            SetTimer(self._entry_until, CUTOFF_TIMER),
        ]
        if p.require_family and story.card_at(ctx.now).family != FAMILY:
            return out + self._end(BounceState.EXPIRED, "veto")
        if self._spy_breaks(ctx):
            return out + self._end(BounceState.EXPIRED, "market_break")
        return out + self._watch(ctx, decide=False)

    def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
        st = self._state
        if isinstance(ev, BarIn):
            if ev.symbol == SPY:
                if st in _WAITING and self._spy_breaks(ctx):
                    return self._end(BounceState.EXPIRED, "market_break")
                return []
            if ev.symbol != self._symbol:
                return []
            if st in _WAITING:
                return self._watch(ctx, decide=True)
            if st is BounceState.ENTERED:
                return self._exits(ctx)
            return []
        if isinstance(ev, NewsIn):
            return self._news(ev, ctx)
        if isinstance(ev, FillIn):
            if ev.side == "buy" and st is BounceState.ENTERING:
                if ev.bar_ts is None:
                    raise ValueError(f"{ev.order_id}: an entry fill without its bar")
                self._entry_bar_ts = to_us(ev.bar_ts) // US
                filled = self._go(BounceState.ENTERED, "filled")
                if ctx.now >= ctx.sessions[-1].flatten:
                    # Filled at the deadline's flatten: the simulator's time stop sells it.
                    return [filled, self._go(BounceState.EXITING, "time_stop")]
                return [filled, *self._exits(ctx)]
            if ev.side == "sell":
                reason = ev.reason or "unspecified"
                return self._end(BounceState.EXITED, reason)
            return []
        if isinstance(ev, OrderClosedIn):
            if ev.side == "buy" and st is BounceState.ENTERING:
                why = "entry_unfilled" if ev.status != "rejected" else f"entry_rejected:{ev.reason}"
                return self._end(BounceState.EXPIRED, why)
            return []  # a refused sell: the simulator's flatten still exits
        if isinstance(ev, ComplianceIn):
            if st is BounceState.ENTERED:  # the simulator sells at the open
                return [self._go(BounceState.EXITING, "compliance")]
            return []
        if isinstance(ev, SessionIn):
            if ev.kind == "flatten" and ev.k == len(ctx.sessions) - 1 and st is BounceState.ENTERED:
                return [self._go(BounceState.EXITING, "time_stop")]  # the simulator's sell
            return []
        if isinstance(ev, TimerIn):
            if ev.tag == CUTOFF_TIMER and st in _WAITING:
                return self._end(BounceState.EXPIRED, "cutoff")
            return []
        return []

    # ── transitions ──

    def _go(self, to: BounceState, reason: str) -> Transition:
        self._state = to
        return Transition(str(to), reason)

    def _end(self, to: BounceState, reason: str) -> list[Intent]:
        return [self._go(to, reason), Finish(reason)]

    # ── watching ──

    def _references(self, ctx: Ctx, bars: BarSeries, anchor_i: int) -> None:
        """P0 and SPY0 of an "in" story: the last closes of S with ``ts + 60 s <= at_news``.

        Without such a bar of the stock both are the previous closes, so the
        two legs of D share one baseline.
        """
        pre = self._pre
        assert pre is not None
        limit = (self._at_news_us - 60 * US) // US  # ts + 60 s <= at_news  <=>  ts <= limit
        j = int(np.searchsorted(bars.ts[:anchor_i], limit, side="right")) - 1
        if j < 0 or int(bars.k[j]) != 0:
            self._p0, self._spy0 = pre.prev_close_s, pre.spy_prev_close_s
            return
        self._p0 = float(bars.c[j])
        spy = ctx.market.bars(SPY)
        j = int(np.searchsorted(spy.ts, limit, side="right")) - 1
        self._spy0 = float(spy.c[j]) if j >= 0 and int(spy.k[j]) == 0 else pre.spy_prev_close_s

    def _spy_close(self, spy: BarSeries, ts: int) -> float:
        """SPY's close of the bar starting at ``ts``, else its latest before; SPY0 before any."""
        j = int(np.searchsorted(spy.ts, ts, side="right")) - 1
        return float(spy.c[j]) if j >= 0 and int(spy.k[j]) == 0 else self._spy0

    def _spy_breaks(self, ctx: Ctx) -> bool:
        """Whether some visible SPY close of S is at or below the market-break level."""
        if not self._broken:
            spy = ctx.market.bars(SPY)
            for j in range(self._spy_next, len(spy)):
                if int(spy.k[j]) != 0:
                    break
                if float(spy.c[j]) <= self._break_level:
                    self._broken = True
                    break
            self._spy_next = len(spy)
        return self._broken

    def _watch(self, ctx: Ctx, *, decide: bool) -> list[Intent]:
        """Read the newly visible bars of S, then arm, disarm or enter on the last one."""
        bars = ctx.market.bars(self._symbol)
        out: list[Intent] = []
        spy: BarSeries | None = None
        read = False
        n = len(bars)
        i = self._next_i
        while i < n and int(bars.k[i]) == 0:
            ts = int(bars.ts[i])
            if ts >= self._anchor_from:
                if self._anchor_ts is None:
                    self._anchor_ts = ts
                    if self._in:
                        self._references(ctx, bars, i)
                if spy is None:
                    spy = ctx.market.bars(SPY)
                h, low, c, v, vw = (
                    float(bars.h[i]),
                    float(bars.l[i]),
                    float(bars.c[i]),
                    float(bars.v[i]),
                    float(bars.vw[i]),
                )
                self._wv += _w(h, low, c, vw) * v
                self._v += v
                if low <= self._low:  # t_L is the latest bar attaining the low
                    self._low, self._low_ts = low, ts
                self._last_close, self._last_ts = c, ts
                d = (c / self._p0 - 1.0) - (self._spy_close(spy, ts) / self._spy0 - 1.0)
                if not self._triggered and d <= -self._thr:
                    self._triggered = True
                    out.append(Transition(str(self._state), TRIGGERED))
                read = True
            i += 1
        self._next_i = i
        if not read:
            return out

        quiet_s = self.params.quiet.total_seconds()
        quiet = self._triggered and (self._last_ts - self._low_ts) >= quiet_s
        if self._state is BounceState.WATCHING and quiet:
            out.append(self._go(BounceState.ARMED, "quiet"))
        elif self._state is BounceState.ARMED and not quiet:
            out.append(self._go(BounceState.WATCHING, "new_low"))
        if decide and self._state is BounceState.ARMED and self._entry_ok(ctx):
            out += self._enter(ctx)
        return out

    def _avwap(self) -> float:
        return self._wv / self._v if self._v > 0 else math.nan

    def _entry_ok(self, ctx: Ctx) -> bool:
        """E1-E5 at the visible time of the last bar read (``ctx.now``)."""
        p = self.params
        now = ctx.now
        assert self._entry_from is not None and self._entry_until is not None
        c, low = self._last_close, self._low
        return (
            self._entry_from <= now <= self._entry_until  # E1
            and c > self._avwap()  # E2
            and (c - low) <= p.max_retrace_at_entry * (self._p0 - low)  # E3
            and not self._spy_breaks(ctx)  # E4
            and (not p.require_family or ctx.story.card_at(now).family == FAMILY)  # E5
        )

    def _enter(self, ctx: Ctx) -> list[Intent]:
        p = self.params
        pre, elig = self._pre, self._elig
        assert pre is not None and self._anchor_ts is not None
        card = ctx.story.card_at(ctx.now)
        self._decided_at = ctx.now
        self._low_star = self._low
        self._target = self._low_star + p.target_retrace * (self._p0 - self._low_star)
        family = FAMILY if p.require_family else card.type
        facts = TradeFacts(
            family_type=card.type,
            cell=f"{family}/{p.variant}",
            variant=p.variant,
            cost_bps=elig.cost_bps,
            rank=elig.liquidity_rank if elig.liquidity_rank is not None else -1,
            tech=elig.tech,
            beta=pre.beta,
            p0=self._p0,
            spy0=self._spy0,
            sigma=pre.sigma,
            thr=self._thr,
            low_star=self._low_star,
            target=self._target,
            anchor_ts=from_us(self._anchor_ts * US),
            adv20_usd=pre.adv20_usd,
        )
        return [self._go(BounceState.ENTERING, "entry"), Submit("buy", facts=facts)]

    # ── holding ──

    def _news(self, ev: NewsIn, ctx: Ctx) -> list[Intent]:
        st = self._state
        strict = self.params.require_family
        if self._brings_structural(ev):
            self._structural_at = ev.at
            at_decision = st is BounceState.ENTERING and ev.at == self._decided_at
            if strict and (st in _WAITING or at_decision):
                return self._end(BounceState.EXPIRED, "veto")  # the Finish cancels a buy
        if ev.own and strict and st in _WAITING and ctx.story.card_at(ctx.now).family != FAMILY:
            return self._end(BounceState.EXPIRED, "veto")
        if st is BounceState.ENTERED and self._aborts(ctx):
            return self._exit("abort")
        return []

    @staticmethod
    def _brings_structural(ev: NewsIn) -> bool:
        """Whether the item(s) usable at ``ev.at`` bring a structural item (module docstring).

        The story's own: a structural type its card's vetoes lacked just
        before. Another story's: any item once that story's card is
        structural.
        """
        after = ev.story.card_at(ev.at)
        if not after.structural:
            return False
        if not ev.own:
            return True
        before = ev.story.card_at(ev.at - _TICK)
        return bool(_structural(after.vetoes) - _structural(before.vetoes))

    def _aborts(self, ctx: Ctx) -> bool:
        """X3: an item brought a structural item after the decision; under H1 also any
        structural item on the story's card (none can be known at the decision there)."""
        later = (
            self._structural_at is not None
            and self._decided_at is not None
            and self._structural_at > self._decided_at
        )
        return later or (self.params.require_family and ctx.story.card_at(ctx.now).structural)

    def _s_units(self, ctx: Ctx, bars: BarSeries, u: int) -> float:
        """Bar ``u``'s close in session-S units (``MarketView.to_s_units`` after S)."""
        c = float(bars.c[u])
        k = int(bars.k[u])
        if k == 0:
            return c
        day: date = ctx.sessions[k].day
        try:
            return ctx.market.to_s_units(c, day)
        except KeyError:  # no A-factor that day: the simulator's own fallback scale
            return c * float(bars.scale[u])

    def _exits(self, ctx: Ctx) -> list[Intent]:
        """X3 > X1 > X2, on each visible bar not yet judged from the entry's fill bar on."""
        if self._aborts(ctx):
            return self._exit("abort")
        assert self._entry_bar_ts is not None
        bars = ctx.market.bars(self._symbol)
        for u in range(self._exit_next, len(bars)):
            self._exit_next = u + 1
            if int(bars.ts[u]) < self._entry_bar_ts:
                continue
            c = self._s_units(ctx, bars, u)
            if c < self._low_star:
                return self._exit("stop")
            if c >= self._target:
                return self._exit("target")
        self._exit_next = max(self._exit_next, len(bars))
        return []

    def _exit(self, reason: str) -> list[Intent]:
        return [self._go(BounceState.EXITING, reason), Submit("sell", reason=reason)]


@dataclass(frozen=True, slots=True)
class BounceFactory:
    """The H1 runner's :class:`~halabot.playbooks.playbook.PlaybookFactory`.

    ``context`` maps a story id to its ``(PreEvent, Eligibility)``, computed
    by the runner from ``context.PitContext`` with the story's ``at_news``,
    so the simulator never imports the context. ``PreEvent`` may be None (no
    σ or previous close): an eligible story then ends DISMISSED
    (``no_pre_event``). A story missing from ``context`` is a runner bug: a
    ``KeyError``. Plain data, so a spawn pool can pickle it.
    """

    context: Mapping[str, tuple[PreEventLike | None, EligibilityLike]] = field(repr=False)
    params: BounceParams = BounceParams()
    name: str = NAME
    version: str = VERSION

    @property
    def path_sessions(self) -> int:
        return self.params.hold_sessions

    def __call__(self, story: StoryView) -> OverreactionBounce:
        try:
            pre, elig = self.context[story.story_id]
        except KeyError:
            raise KeyError(f"no pre-event context for story {story.story_id}") from None
        return OverreactionBounce(story, pre, elig, self.params)


__all__ = [
    "ABORT",
    "CUTOFF_TIMER",
    "FAMILY",
    "LIVE",
    "NAME",
    "NO_AT_NEWS",
    "NO_PRE_EVENT",
    "PRIORITY",
    "RECLAIM",
    "STOP",
    "VERSION",
    "BounceFactory",
    "BounceParams",
    "BounceState",
    "EligibilityLike",
    "OverreactionBounce",
    "PreEventLike",
]
