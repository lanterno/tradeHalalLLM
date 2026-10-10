"""What the simulator reads from the story builder and the point-in-time context.

The simulator never imports ``halal_trader.events.stories`` or
``halal_trader.events.context``: it depends on these Protocols, which
``stories.Story``, ``taxonomy.StoryCard`` and ``context.PitContext``
satisfy. Every member is declared as a read-only property, so a frozen
dataclass satisfies it as well as a plain attribute does.

Everything here is point in time by construction: ``card_at(t)`` and
``nsn_at(cutoff)`` read only items with ``available_at <= t`` (or ``<=
cutoff``), and the playbook passes its own ``now``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Protocol


class CardView(Protocol):
    """A story's label at one moment (``taxonomy.StoryCard``)."""

    @property
    def family(self) -> str | None: ...
    @property
    def type(self) -> str: ...
    @property
    def structural(self) -> bool:
        """Any structural item known: a held position aborts on it."""
        ...

    @property
    def vetoes(self) -> tuple[str, ...]: ...


class StoryView(Protocol):
    """One (symbol, reaction session) story (``stories.Story``)."""

    @property
    def story_id(self) -> str: ...
    @property
    def symbol(self) -> str: ...
    @property
    def session(self) -> date:
        """S, the reaction session."""
        ...

    def card_at(self, t: datetime) -> CardView:
        """The label from the items with ``available_at <= t`` only."""
        ...

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        """The first item time ``<= cutoff`` at which the card is NSN_CORE, else None."""
        ...

    def at_news(self) -> datetime | None:
        """``at`` (public time) of the item that first made the story NSN_CORE."""
        ...

    def start_case(self) -> str:
        """``"in"`` (news inside S's session) or ``"out"``."""
        ...

    def news_times(self) -> Sequence[datetime]:
        """``available_at`` of every item; the simulator sends a ``NewsIn`` at each."""
        ...


class DailyPointLike(Protocol):
    """One raw daily bar: the official open and close of a session.

    ``context.DailyPoint`` (``day, open, high, low, close, volume, adj``)
    satisfies it; the simulator reads only the official prices, for its
    close fallbacks, the daily legs' marks and the SPY legs that go with
    them. A-factors come from ``ContextView.adj``.
    """

    @property
    def open(self) -> float: ...
    @property
    def close(self) -> float: ...


class ContextView(Protocol):
    """The point-in-time context the simulator reads (``context.PitContext``)."""

    @property
    def sessions(self) -> Sequence[date]:
        """SPY's raw daily-bar sessions, sorted."""
        ...

    def adj(self, symbol: str, day: date) -> float | None:
        """A(day) = close_all / close_raw; None without both bars."""
        ...

    def screen_verdict(self, symbol: str, day: date) -> str:
        """The verdict of the newest screen with ``as_of < day``; ``'no_screen'`` if none."""
        ...

    def daily(self, symbol: str, day: date) -> DailyPointLike | None: ...


__all__ = ["CardView", "ContextView", "DailyPointLike", "StoryView"]
