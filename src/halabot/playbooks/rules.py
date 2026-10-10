"""Order admission, applied when an order is submitted (the same rules in sim and live).

**Buys** (fail closed, like ``TradeExecutor._execute_buy``):

* the point-in-time screen for (symbol, S) says ``halal``; otherwise
  ``not_halal``, or ``no_screen`` when there is no screen before S;
* submitted inside S's entry window, ``[open + 20 min, close - 60 min]``;
* no position held on the symbol, and no buy already working;
* active inside S's session ``[open, close)``; a buy is never queued.

**Sells** are clamped to the position less the sells already working
(long-only); a sell with nothing to sell is rejected, and so is one for
part of the position (a trade record holds one exit). A sell decided while
the market is shut (overnight, or in the last seconds of a session) is
active at the next path session's open + ``ORDER_LAG``, the rule for an
exit pending overnight; with no session left it is rejected
(``market_closed``).

Rejections are reasons, not exceptions: the driver turns one into an
``OrderClosedIn(status="rejected")``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from halabot.playbooks.clock import to_us
from halabot.playbooks.types import OrderKind, Session

NOT_HALAL: Final = "not_halal"
NO_SCREEN: Final = "no_screen"
OUTSIDE_ENTRY_WINDOW: Final = "outside_entry_window"
POSITION_HELD: Final = "position_held"
NO_POSITION: Final = "no_position"
MARKET_CLOSED: Final = "market_closed"
KIND_NOT_ALLOWED: Final = "kind_not_allowed"
NO_FACTS: Final = "no_facts"
PARTIAL_EXIT: Final = "partial_exit"


def screen_reason(verdict: str) -> str | None:
    """None when the screen admits a buy, else the rejection reason (fail closed)."""
    if verdict == "halal":
        return None
    return NO_SCREEN if verdict == "no_screen" else NOT_HALAL


def kind_reason(kind: OrderKind, allowed: frozenset[OrderKind]) -> str | None:
    return None if kind in allowed else KIND_NOT_ALLOWED


def admit_buy(*, verdict: str, decided_us: int, session: Session, held: bool) -> str | None:
    """None when a buy decided at ``decided_us`` for reaction session ``session`` is admitted."""
    if (why := screen_reason(verdict)) is not None:
        return why
    if held:
        return POSITION_HELD
    if not to_us(session.entry_start) <= decided_us <= to_us(session.entry_cutoff):
        return OUTSIDE_ENTRY_WINDOW
    return None


def buy_active_at(decided_us: int, lag_us: int, session: Session) -> int | None:
    """When a buy decided at ``decided_us`` works, or None if that is outside the session."""
    active = decided_us + lag_us
    return active if to_us(session.open) <= active < to_us(session.close) else None


def sell_active_at(
    decided_us: int, lag_us: int, sessions: Sequence[Session]
) -> tuple[int, int] | None:
    """(path session, active_at) of a sell decided at ``decided_us``, or None (market closed).

    Inside a session it works ``lag`` later; decided while the market is shut
    it works at the next session's open + ``lag``.
    """
    active = decided_us + lag_us
    for k, s in enumerate(sessions):
        open_us, close_us = to_us(s.open), to_us(s.close)
        if active < open_us:
            return k, open_us + lag_us
        if active < close_us:
            return k, active
    return None


def clamp_sell(requested: float | None, position: float, working_sells: float) -> float:
    """The quantity a sell may have: at most the position not already being sold."""
    free = max(position - working_sells, 0.0)
    return free if requested is None else max(min(requested, free), 0.0)


__all__ = [
    "KIND_NOT_ALLOWED",
    "MARKET_CLOSED",
    "NOT_HALAL",
    "NO_FACTS",
    "NO_POSITION",
    "NO_SCREEN",
    "PARTIAL_EXIT",
    "OUTSIDE_ENTRY_WINDOW",
    "POSITION_HELD",
    "admit_buy",
    "buy_active_at",
    "clamp_sell",
    "kind_reason",
    "screen_reason",
    "sell_active_at",
]
