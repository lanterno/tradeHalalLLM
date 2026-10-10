"""`halal-trader data minutes --plan h1 [--part P]... [--dry-run] [--rate N] [--force]`."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Collection
from datetime import UTC, date, datetime, time
from typing import Any

import pytest
from click.testing import CliRunner
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.cli import cli
from halal_trader.cli.data import MINUTE_PARTS
from halal_trader.data import minutes
from halal_trader.data.alpaca_market import AlpacaMarketData
from halal_trader.events import units
from halal_trader.events.units import UnitPlan
from halal_trader.market_hours import MARKET_TZ

D1, D2 = date(2016, 3, 1), date(2016, 3, 2)
PLAN = UnitPlan(
    {
        "spy": frozenset({("SPY", D1), ("SPY", D2)}),
        "gate_g1": frozenset({("AAPL", D1), ("SPY", D1)}),
    }
)


def test_the_commands_parts_are_the_planners() -> None:
    assert MINUTE_PARTS == units.PARTS


class FakeMarket:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], date]] = []
        self.closed = False

    async def minute_bars_many(self, symbols, *, start, end):  # type: ignore[no-untyped-def]
        day = start.astimezone(MARKET_TZ).date()
        self.calls.append((tuple(symbols), day))
        at = datetime.combine(day, time(9, 30), MARKET_TZ).astimezone(UTC)
        stamp = at.isoformat().replace("+00:00", "Z")
        bar = {"t": stamp, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0, "v": 1.0, "vw": 10.0}
        return {s: [bar] for s in symbols}

    async def aclose(self) -> None:
        self.closed = True


def _plan(monkeypatch: pytest.MonkeyPatch) -> list[Collection[str] | None]:
    """``units.h1_plan`` returns PLAN (restricted to the parts asked); returns the parts asked."""
    asked: list[Collection[str] | None] = []

    async def h1_plan(
        engine: Any, *, parts: Collection[str] | None = None, counts: Counter[str] | None = None
    ) -> UnitPlan:
        asked.append(parts)
        if counts is not None:
            counts["g1.stories"] = 1
        return UnitPlan({p: u for p, u in PLAN.parts.items() if parts is None or p in parts})

    monkeypatch.setattr(units, "h1_plan", h1_plan)
    return asked


def _market(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeMarket, list[dict[str, Any]]]:
    market = FakeMarket()
    made: list[dict[str, Any]] = []

    def from_settings(cls: type, settings: Any, **kwargs: Any) -> FakeMarket:
        made.append(kwargs)
        return market

    monkeypatch.setattr(AlpacaMarketData, "from_settings", classmethod(from_settings))
    return market, made


def _done(database_url: str) -> set[str]:
    async def go() -> set[str]:
        engine: AsyncEngine = create_async_engine(database_url)
        try:
            return await minutes.done_units(engine)
        finally:
            await engine.dispose()

    return asyncio.run(go())


def test_the_dry_run_prints_each_part_and_the_estimate(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _plan(monkeypatch)
    market, made = _market(monkeypatch)

    result = CliRunner().invoke(cli, ["data", "minutes", "--plan", "h1", "--dry-run"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[1].split() == [
        "part",
        "units",
        "new",
        "done",
        "fetch",
        "sessions",
        "requests",
        "first",
        "last",
        "sha",
    ]
    assert lines[2].split() == ["spy", "2", "2", "0", "2", "2", "2", "2016-03-01", "2016-03-02"] + [
        PLAN.sha("spy")[:12]
    ]
    assert lines[3].split()[:7] == ["gate_g1", "2", "1", "0", "1", "1", "1"]
    assert "unique units 3; to fetch 3; about 3 request(s), 0.0 h at 100 a minute" in lines[4]
    assert "selection: g1.stories 1" in result.output
    assert "SUE complement" not in result.output  # no gate_sue part
    assert asked == [None] and made == [] and market.calls == []  # nothing fetched


def test_the_dry_run_says_how_many_sue_events_meet_h1s_units(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sue = UnitPlan({"gate_sue": frozenset({("A", D1)})})

    async def h1_plan(engine: Any, *, parts: Any = None, counts: Counter[str]) -> UnitPlan:
        counts.update({"sue.h1_overlap": 7, "sue.complement": 40})
        return sue

    monkeypatch.setattr(units, "h1_plan", h1_plan)

    result = CliRunner().invoke(cli, ["data", "minutes", "--plan", "h1", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert (
        "gate_sue: 7 SUE complement event(s) dropped because their units meet the train or "
        "validation part" in result.output.replace("\n", " ")
    )


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_plan_that_fails_its_check_is_refused_before_anything_is_fetched(
    database_url: str, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    async def h1_plan(engine: Any, **kwargs: Any) -> UnitPlan:
        UnitPlan({"gate_reactor": frozenset({("A", date(2025, 6, 2))})}).check()
        raise AssertionError("check() passed")

    monkeypatch.setattr(units, "h1_plan", h1_plan)
    monkeypatch.setattr(units, "busy", lambda now: None)
    market, made = _market(monkeypatch)

    args = ["data", "minutes", "--plan", "h1", *(["--dry-run"] if dry_run else [])]
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    assert "plan h1 refused, nothing fetched: gate_reactor: 1 unit(s) outside" in " ".join(
        result.output.split()
    )
    assert "Traceback" not in result.output
    assert made == [] and market.calls == [] and _done(database_url) == set()


def test_the_dry_run_takes_the_parts_asked(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _plan(monkeypatch)
    args = ["data", "minutes", "--plan", "h1", "--dry-run", "--part", "spy", "--part", "train"]

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert asked == [("spy", "train")]
    assert "gate_g1" not in result.output


def test_a_run_fetches_every_part_at_the_rate(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(monkeypatch)
    market, made = _market(monkeypatch)
    monkeypatch.setattr(units, "busy", lambda now: None)

    result = CliRunner().invoke(cli, ["data", "minutes", "--plan", "h1", "--rate", "50"])

    assert result.exit_code == 0, result.output
    assert made == [{"min_interval_s": 60.0 / 50}] and market.closed
    assert "spy: 2 unit(s) fetched, 2 bar(s) stored" in result.output
    assert "gate_g1: 1 unit(s) fetched, 1 bar(s) stored" in result.output
    assert "done: 3 unit(s) fetched, 3 bar(s) stored" in result.output
    assert _done(database_url) == {minutes.unit(*u) for u in PLAN.all()}


def test_a_run_refuses_market_hours_unless_forced(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(monkeypatch)
    market, made = _market(monkeypatch)
    monkeypatch.setattr(units, "busy", lambda now: "US market hours (09:30-16:00 ET)")

    refused = CliRunner().invoke(cli, ["data", "minutes", "--plan", "h1"])

    assert refused.exit_code == 1
    assert "not fetching during US market hours" in refused.output
    assert "Traceback" not in refused.output and made == []
    forced = CliRunner().invoke(cli, ["data", "minutes", "--plan", "h1", "--force"])
    assert forced.exit_code == 0, forced.output
    assert len(market.calls) == 3


def test_a_run_that_reaches_market_hours_stops_and_says_how_to_resume(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(monkeypatch)
    market, _ = _market(monkeypatch)
    answers = iter([None, None, "the research job's window (20:30-23:30 ET)"])
    monkeypatch.setattr(units, "busy", lambda now: next(answers))

    result = CliRunner().invoke(cli, ["data", "minutes", "--plan", "h1"])

    assert result.exit_code == 1
    assert "stopped at the research job's window" in result.output
    assert "run the command again later to resume" in result.output
    assert len(market.calls) == 2  # spy's one batch, then the check before gate_g1's
    assert _done(database_url) == {"SPY:2016-03-01", "SPY:2016-03-02"}


def test_the_command_needs_a_plan() -> None:
    result = CliRunner().invoke(cli, ["data", "minutes", "--dry-run"])
    assert result.exit_code == 2 and "--plan" in result.output
