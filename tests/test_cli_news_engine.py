"""`halal-trader events renames {seed,backfill}` and `events aliases build`."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.cli import cli
from halal_trader.data.alpaca_market import AlpacaMarketData
from tests._renames import NewsMarket, seed_candidates_data


def _run(database_url: str, work: Callable[[AsyncEngine], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        engine = create_async_engine(database_url)
        try:
            return await work(engine)
        finally:
            await engine.dispose()

    return asyncio.run(go())


def test_the_seed_command_lists_candidates_and_their_mapping(database_url: str) -> None:
    _run(database_url, seed_candidates_data)
    result = CliRunner().invoke(cli, ["events", "renames", "seed"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("3 candidate(s)")
    assert "LATE" in lines[1] and "unmapped" in lines[1]
    assert "META" in lines[2] and "FB (to 2022-06-08)" in lines[2]
    assert "SILENT" in lines[3] and "no news" in lines[3]
    tail = " ".join(lines[4:])
    assert "also mapped (not flagged by the rule):" in tail and "XYZ" in tail


@pytest.mark.usefixtures("small_map")
def test_the_backfill_command_paces_the_client_at_the_rate(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[dict[str, Any]] = []

    def from_settings(cls: type, settings: Any, **kwargs: Any) -> NewsMarket:
        made.append(kwargs)
        return NewsMarket()

    monkeypatch.setattr(AlpacaMarketData, "from_settings", classmethod(from_settings))
    result = CliRunner().invoke(cli, ["events", "renames", "backfill", "--rate", "6000000"])
    assert result.exit_code == 0, result.output
    assert "renamed-ticker news: 4 new event(s)" in result.output
    assert made == [{"min_interval_s": 60.0 / 6_000_000}]


def test_the_alias_build_command_stores_and_pins_the_set(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        AlpacaMarketData, "from_settings", classmethod(lambda cls, settings, **kw: NewsMarket())
    )
    result = CliRunner().invoke(cli, ["events", "aliases", "build"])
    assert result.exit_code == 0, result.output

    async def stored(engine: AsyncEngine) -> int:
        async with engine.connect() as conn:
            return int((await conn.execute(text("SELECT count(*) FROM story_aliases"))).scalar())

    rows = _run(database_url, stored)
    assert rows > 0  # the renamed tickers' current symbols, at least
    assert f"{rows} alias row(s) stored; alias_sha " in result.output


def test_the_new_groups_are_on_the_events_command() -> None:
    result = CliRunner().invoke(cli, ["events", "--help"])
    assert result.exit_code == 0
    assert "renames" in result.output and "aliases" in result.output
