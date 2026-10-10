"""The Playbook protocol, its factory, and the views a playbook decides from.

**The contract a playbook implements** (``bounce.py`` and every later one):

* It is a pure state machine. The driver (the simulator here, a live
  driver in Phase 4) calls :meth:`Playbook.start` once, then
  :meth:`Playbook.on` for every input, and applies the intents each call
  returns, in order. Nothing else reaches it: no clock, no database, no
  daily bar of the current or a later session.
* :meth:`Playbook.live` is True while the playbook holds its symbol:
  watching, armed, entering or entered (spec §D.10). While one playbook on
  a symbol is live, a story starting on that symbol is ``blocked_open``
  (its items still reach the live playbook as ``NewsIn``). The simulator
  asks ``live()`` after every call; it never reads state names for this.
* :meth:`Playbook.state` is the state's name, for the records only
  (``StoryOutcome.terminal_state``, the transitions).
* **The buy** is ``Submit("buy", facts=TradeFacts(...))``: the trade record
  is built from those facts (P0, levels, cost, rank, beta). A buy without
  facts is rejected (``no_facts``).
* **Every sell** carries its exit reason in ``Submit.reason`` (``target``,
  ``stop``, ``abort``, ``compliance``, ``time_stop``); a sell without one
  is recorded as ``"unspecified"``. Sells are whole: a partial one is
  rejected (``partial_exit``).
* **State changes** are reported with ``Transition(to, reason)``. The first
  one whose ``reason`` is :data:`TRIGGERED` (``Transition(state,
  reason="triggered")``, usually a self-transition) sets ``triggered_at``;
  the first into a state named ``armed`` (any case: :data:`ARMED`) sets
  ``armed_at``.
* **A terminal state ends with** ``Finish(reason)`` (exited, expired,
  dismissed). Without it the story runs to the deadline session's close.
  A ``Finish`` while holding makes the simulator sell (``time_stop``).
* The simulator itself enforces the compliance exit (``ComplianceIn`` tells
  the playbook), the flatten at close - 5 min of the deadline session, day
  orders, and the admission rules (``rules.py``).

**The factory** (:class:`PlaybookFactory`) builds one playbook per started
story and says, before any is built, how many sessions its paths need
(``path_sessions``) and the playbook's name and version (the run row). It
receives the story as the simulator's point-in-time view (``sim.PitStory``),
clamped to the simulated ``now``.

**What a playbook sees** comes through :class:`Ctx`:

* ``market`` shows only bars already **visible** at ``now`` (``bars``
  end at ``searchsorted(visible_at, now, 'right')``, copied, so no later
  bar is reachable); prices are raw, and ``to_s_units`` (or
  ``BarSeries.in_s_units``) puts them in session-S units with the
  A-ratios.
* ``story.card_at(now)`` labels the story from items available by ``now``.
* ``position`` is the story's own position and working orders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Protocol

from halabot.playbooks.interfaces import CardView, ContextView, StoryView
from halabot.playbooks.types import BarSeries, Input, Intent, Session, WorkingOrder

TRIGGERED: Final = "triggered"
ARMED: Final = "armed"  # compared case-insensitively: "ARMED" and StrEnum auto() both count


class MarketView(Protocol):
    """Visible bars of the current path, for the story's symbol and SPY."""

    def bars(self, symbol: str) -> BarSeries:
        """Visible bars of ``symbol`` (the story's symbol or ``"SPY"``), raw prices."""
        ...

    def last(self, symbol: str) -> float | None:
        """Close of the last visible bar (raw), None before the first."""
        ...

    def to_s_units(self, price: float, day: date, *, symbol: str | None = None) -> float:
        """``price * A(day) / A(S)`` for the story's symbol (or ``symbol``, e.g. SPY)."""
        ...


class PositionView(Protocol):
    """The story's own position and working orders."""

    @property
    def held(self) -> bool: ...
    @property
    def qty(self) -> float: ...
    @property
    def entry_price(self) -> float | None:
        """Raw entry fill price."""
        ...

    @property
    def entry_price_s(self) -> float | None:
        """Entry fill price in session-S units."""
        ...

    @property
    def entry_at(self) -> datetime | None:
        """When the entry filled (the end of its bar)."""
        ...

    @property
    def working(self) -> tuple[WorkingOrder, ...]: ...


@dataclass(frozen=True, slots=True)
class Ctx:
    """What a playbook sees at ``now``."""

    now: datetime
    story: StoryView
    sessions: tuple[Session, ...]  # the path: S .. S + path_sessions - 1
    k: int  # the current path session: the last whose pre_open is <= now (0 before)
    market: MarketView
    position: PositionView

    @property
    def session(self) -> Session:
        """S, the reaction session."""
        return self.sessions[0]


class Playbook(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def path_sessions(self) -> int:
        """1 (intraday) or 3 (multi-day): the sessions the simulator loads and runs."""
        ...

    def start(self, ctx: Ctx) -> list[Intent]: ...
    def on(self, ev: Input, ctx: Ctx) -> list[Intent]: ...

    def state(self) -> str:
        """The current state's name, for the records."""
        ...

    def live(self) -> bool:
        """Watching, armed, entering or entered: the symbol is taken (``blocked_open``)."""
        ...


class PlaybookFactory(Protocol):
    """Builds a story's playbook; knows its paths' length and its name before building one."""

    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def path_sessions(self) -> int: ...

    def __call__(self, story: StoryView) -> Playbook:
        """The playbook of ``story`` (the simulator's point-in-time view of it)."""
        ...


@dataclass(frozen=True, slots=True)
class Factory:
    """A :class:`PlaybookFactory` around a function.

    A spawn process pool pickles the factory, so ``make`` must then be a
    module-level function or class, not a closure (``sim.run``).
    """

    make: Callable[[StoryView], Playbook]
    name: str
    version: str
    path_sessions: int

    def __call__(self, story: StoryView) -> Playbook:
        return self.make(story)


__all__ = [
    "ARMED",
    "TRIGGERED",
    "CardView",
    "ContextView",
    "Ctx",
    "Factory",
    "MarketView",
    "Playbook",
    "PlaybookFactory",
    "PositionView",
    "StoryView",
]
