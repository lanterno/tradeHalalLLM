"""The day trader's risk meter (core/risk_meter.py)."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.config import StockSettings
from halal_trader.core import risk_meter
from halal_trader.core.risk_meter import LEVELS, MeterRefused
from halal_trader.trading.strategy import SYSTEM_PROMPT

FRIDAY = date(2026, 10, 23)
MONDAY = date(2026, 10, 26)


@pytest.fixture
def all_levels_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(risk_meter, "SETTABLE", frozenset(LEVELS))


async def test_an_unset_meter_is_level_3(engine) -> None:
    s = await risk_meter.get_status(engine, today=FRIDAY)

    assert (s.level, s.chosen, s.effective_from, s.set_by) == (3, 3, None, None)
    assert s.limits.name == "Standard"


async def test_a_set_level_persists_with_who_and_why(engine) -> None:
    await risk_meter.set_level(engine, 3, reason="hold until the review", set_by="op", today=FRIDAY)

    s = await risk_meter.get_status(engine, today=FRIDAY)
    assert (s.level, s.reason, s.set_by) == (3, "hold until the review", "op")
    assert s.set_at is not None


@pytest.mark.parametrize(
    ("level", "why"),
    [(1, "11-02 review"), (2, "11-02 review"), (4, "unlock test"), (5, "unlock test")],
)
async def test_a_level_that_is_not_open_is_refused_and_not_stored(engine, level, why) -> None:
    with pytest.raises(MeterRefused, match=why):
        await risk_meter.set_level(engine, level, reason="r", set_by="op", today=FRIDAY)

    assert (await risk_meter.get_status(engine, today=FRIDAY)).set_by is None


@pytest.mark.parametrize("level", [0, 6])
async def test_a_level_outside_1_to_5_is_refused(engine, level) -> None:
    with pytest.raises(MeterRefused, match="the levels are 1 to 5"):
        await risk_meter.set_level(engine, level, reason="r", set_by="op", today=FRIDAY)


@pytest.mark.usefixtures("all_levels_open")
async def test_a_raise_waits_for_the_next_trading_day(engine) -> None:
    s = await risk_meter.set_level(engine, 4, reason="r", set_by="op", today=FRIDAY)

    assert (s.level, s.chosen, s.effective_from) == (3, 4, MONDAY)
    assert "level 4 (Bold) from 2026-10-26" in risk_meter.describe(s)
    later = await risk_meter.get_status(engine, today=MONDAY)
    assert (later.level, later.effective_from) == (4, None)


@pytest.mark.usefixtures("all_levels_open")
async def test_a_cut_applies_at_once(engine) -> None:
    s = await risk_meter.set_level(engine, 2, reason="r", set_by="op", today=FRIDAY)

    assert (s.level, s.chosen, s.effective_from) == (2, 2, None)


@pytest.mark.usefixtures("all_levels_open")
async def test_a_cut_while_a_raise_waits_applies_at_once(engine) -> None:
    await risk_meter.set_level(engine, 5, reason="r", set_by="op", today=FRIDAY)
    s = await risk_meter.set_level(engine, 2, reason="r", set_by="op", today=FRIDAY)

    assert (s.level, s.effective_from) == (2, None)
    assert (await risk_meter.get_status(engine, today=MONDAY)).level == 2


def test_level_3_is_the_configured_settings() -> None:
    s, lim = StockSettings(), LEVELS[3]

    assert lim.max_position_pct == s.max_position_pct
    assert lim.max_positions == s.max_simultaneous_positions
    assert lim.daily_loss_limit == s.daily_loss_limit
    assert lim.max_drawdown_pct == s.max_drawdown_pct
    assert lim.max_portfolio_heat_pct == s.max_portfolio_heat_pct
    assert lim.max_stop_distance_pct is None
    assert "Default to 8–15% of portfolio_value" in SYSTEM_PROMPT


def test_each_level_allows_at_least_as_much_risk_as_the_one_below() -> None:
    for low, high in zip(list(LEVELS.values()), list(LEVELS.values())[1:], strict=False):
        assert high.max_position_pct > low.max_position_pct
        assert high.typical_position_pct > low.typical_position_pct
        assert high.typical_position_pct[1] <= high.max_position_pct
        assert high.max_positions > low.max_positions
        assert high.daily_loss_limit > low.daily_loss_limit
        assert high.max_drawdown_pct > low.max_drawdown_pct
        assert high.max_portfolio_heat_pct > low.max_portfolio_heat_pct


def test_the_level_list_says_which_can_be_chosen() -> None:
    rows = {r["level"]: r for r in risk_meter.levels_json()}

    assert rows[3]["refusal"] is None
    assert all(rows[n]["refusal"] for n in (1, 2, 4, 5))
    assert rows[1]["max_stop_distance_pct"] == 0.02
