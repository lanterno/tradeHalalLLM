"""The Playbook protocol and the views a playbook decides from.

A playbook is a pure state machine: the driver (the simulator here, a live
driver in Phase 4) calls :meth:`Playbook.start` once and :meth:`Playbook.on`
for every input, and applies the intents it returns. Everything it may look
at comes through :class:`Ctx`:

* ``market`` shows only bars already **visible** at ``now`` (``bars``
  end at ``searchsorted(visible_at, now, 'right')``); prices are raw, and
  ``to_s_units`` (or ``BarSeries.in_s_units``) puts them in session-S units
  with the A-ratios. No daily bar of the current or a later session is
  reachable.
* ``story.card_at(now)`` labels the story from items available by ``now``.
* ``position`` is the story's own position and working orders.

Conventions the simulator reads from the intents:

* ``Transition(to, reason)`` records a state change; the first transition to
  ``"ARMED"`` sets ``armed_at``, and the first with ``reason == TRIGGERED``
  (usually a self-transition) sets ``triggered_at``.
* A playbook is **live** while its state is in :data:`LIVE_STATES` and it has
  not finished; a story starting on the same symbol meanwhile is
  ``blocked_open``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Protocol

from halabot.playbooks.interfaces import CardView, ContextView, StoryView
from halabot.playbooks.types import BarSeries, Input, Intent, Session, WorkingOrder

TRIGGERED: Final = "triggered"
LIVE_STATES: Final = frozenset({"WATCHING", "ARMED", "ENTERING", "ENTERED"})


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
    def state(self) -> str: ...


PlaybookFactory = Callable[[StoryView], Playbook]

__all__ = [
    "LIVE_STATES",
    "TRIGGERED",
    "CardView",
    "ContextView",
    "Ctx",
    "MarketView",
    "Playbook",
    "PlaybookFactory",
    "PositionView",
    "StoryView",
]
