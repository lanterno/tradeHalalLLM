"""Order admission: the halal gate fails closed, the entry window, long-only, session hours."""

from __future__ import annotations

from halabot.playbooks import rules
from halabot.playbooks.clock import US, to_us
from halabot.playbooks.types import OrderKind, Session
from tests.halabot.playbooks._support import HALF, MON, TUE, et

LAG = 3 * US


def test_the_screen_admits_only_halal() -> None:
    assert rules.screen_reason("halal") is None
    assert rules.screen_reason("no_screen") == "no_screen"
    for verdict in ("not_halal", "doubtful", "absent", ""):
        assert rules.screen_reason(verdict) == "not_halal"


def test_buy_admission() -> None:
    s = Session.of(MON)

    def admit(verdict: str = "halal", at: tuple[int, ...] = (10, 0), held: bool = False):  # type: ignore[no-untyped-def]
        return rules.admit_buy(
            verdict=verdict, decided_us=to_us(et(MON, *at)), session=s, held=held
        )

    assert admit() is None
    assert admit(at=(9, 50)) is None and admit(at=(15, 0)) is None  # both ends inclusive
    assert admit(at=(9, 49, 59)) == "outside_entry_window"
    assert admit(at=(15, 0, 1)) == "outside_entry_window"
    assert admit(held=True) == "position_held"
    assert admit("not_halal", held=True) == "not_halal"  # the screen is checked first
    h = Session.of(HALF)
    assert rules.admit_buy(
        verdict="halal", decided_us=to_us(et(HALF, 12, 0, 1)), session=h, held=False
    ) == ("outside_entry_window")


def test_a_buy_is_active_only_inside_the_session() -> None:
    s = Session.of(MON)
    assert rules.buy_active_at(to_us(et(MON, 10, 0)), LAG, s) == to_us(et(MON, 10, 0, 3))
    assert rules.buy_active_at(to_us(et(MON, 15, 59, 57)), LAG, s) is None
    assert rules.buy_active_at(to_us(et(MON, 9, 29)), LAG, s) is None


def test_a_sell_decided_while_shut_works_at_the_next_open() -> None:
    sessions = [Session.of(MON), Session.of(TUE)]
    assert rules.sell_active_at(to_us(et(MON, 11, 0)), LAG, sessions) == (
        0,
        to_us(et(MON, 11, 0, 3)),
    )
    assert rules.sell_active_at(to_us(et(MON, 18, 0)), LAG, sessions) == (
        1,
        to_us(et(TUE, 9, 30, 3)),
    )
    assert rules.sell_active_at(to_us(et(TUE, 9, 20)), LAG, sessions) == (
        1,
        to_us(et(TUE, 9, 30, 3)),
    )
    assert rules.sell_active_at(to_us(et(MON, 15, 59, 58)), LAG, sessions) == (
        1,
        to_us(et(TUE, 9, 30, 3)),
    )
    assert rules.sell_active_at(to_us(et(TUE, 16, 0)), LAG, sessions) is None


def test_sells_are_clamped_to_the_position() -> None:
    assert rules.clamp_sell(None, 10.0, 0.0) == 10.0
    assert rules.clamp_sell(4.0, 10.0, 0.0) == 4.0
    assert rules.clamp_sell(15.0, 10.0, 0.0) == 10.0
    assert rules.clamp_sell(None, 10.0, 10.0) == 0.0  # already being sold
    assert rules.clamp_sell(None, 0.0, 0.0) == 0.0


def test_only_allowed_kinds() -> None:
    allowed = frozenset({OrderKind.MARKET})
    assert rules.kind_reason(OrderKind.MARKET, allowed) is None
    for kind in (OrderKind.LIMIT, OrderKind.MOO, OrderKind.MOC):
        assert rules.kind_reason(kind, allowed) == "kind_not_allowed"
