"""`halal-trader events stories {build,counts}`."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from click.testing import CliRunner
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.cli import cli
from halal_trader.events import stories
from halal_trader.events.stories import build_range
from tests._renames import mark_renamed_news_done
from tests._stories import (
    ALIAS_ROWS,
    FRI,
    MON,
    WEEK,
    add_aliases,
    fake_context,
    seed_week,
    store,
)

WEEK_ARGS = ["--start", "2024-05-06", "--end", "2024-05-10"]


def _run(database_url: str, work: Callable[[AsyncEngine], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        engine = create_async_engine(database_url)
        try:
            return await work(engine)
        finally:
            await engine.dispose()

    return asyncio.run(go())


@pytest.mark.usefixtures("small_map")
def test_the_build_command_reports_counts_and_pins(database_url: str) -> None:
    _run(database_url, seed_week)
    result = CliRunner().invoke(cli, ["events", "stories", "build", *WEEK_ARGS])
    assert result.exit_code == 0, result.output
    assert "7 story(ies) stored for 2024-05-06..2024-05-10" in result.output
    assert "admitted 9" in result.output and "entity 1" in result.output
    assert f"stories_sha {stories.STORIES_SHA}" in result.output
    assert "renames_sha" in result.output and "alias_sha" in result.output


@pytest.mark.usefixtures("small_map")
def test_the_build_command_refuses_without_aliases(database_url: str) -> None:
    async def seed(engine: AsyncEngine) -> None:
        await store(engine, WEEK)
        await mark_renamed_news_done(engine)

    _run(database_url, seed)
    args = ["events", "stories", "build", *WEEK_ARGS]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert "no story aliases" in result.output and "(or pass --force)" in result.output
    assert "Traceback" not in result.output
    forced = CliRunner().invoke(cli, [*args, "--force"])
    assert forced.exit_code == 0, forced.output
    assert "forced: the range is not marked complete" in forced.output


def test_the_build_command_needs_its_range() -> None:
    result = CliRunner().invoke(cli, ["events", "stories", "build", "--start", "2024-05-06"])
    assert result.exit_code == 2 and "--end" in result.output


@pytest.mark.usefixtures("small_map")
def test_the_counts_command_prints_each_universe(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_context(monkeypatch)

    async def seed(engine: AsyncEngine) -> None:
        await seed_week(engine)
        await build_range(engine, start=MON, end=FRI)

    _run(database_url, seed)
    result = CliRunner().invoke(cli, ["events", "stories", "counts", *WEEK_ARGS])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "== all stories ==" in lines and "== Technology (PRIMARY and in Technology) ==" in lines
    primary = lines.index(next(x for x in lines if x.startswith("== PRIMARY")))
    nsn = next(x for x in lines[primary:] if x.startswith("NSN_CORE"))
    assert nsn.split()[-4:] == ["1", "0", "1", "1"]  # 2024, train, validation, total
    assert lines[-2] == "eligibility (every story): ok 5, rank 2"
    assert lines[-1] == "eligibility (NSN_CORE stories): ok 1, rank 1"


@pytest.mark.usefixtures("small_map")
def test_the_counts_command_refuses_a_range_no_complete_build_covers(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_context(monkeypatch)

    async def seed(engine: AsyncEngine) -> None:
        await seed_week(engine)
        await build_range(engine, start=MON, end=FRI)

    _run(database_url, seed)
    args = ["events", "stories", "counts", "--start", "2024-05-06", "--end", "2024-05-13"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert "1 session(s) in 2024-05-06..2024-05-13" in result.output
    assert "first: 2024-05-13" in result.output and "Traceback" not in result.output


@pytest.mark.usefixtures("small_map")
def test_the_build_command_refuses_unparsed_news(database_url: str) -> None:
    async def seed(engine: AsyncEngine) -> None:
        await store(engine, WEEK, facts=False)
        await add_aliases(engine, ALIAS_ROWS)
        await mark_renamed_news_done(engine)

    _run(database_url, seed)
    result = CliRunner().invoke(cli, ["events", "stories", "build", *WEEK_ARGS])
    assert result.exit_code == 1
    assert "9 news event(s)" in result.output and "events extract" in result.output
    assert "(or pass --force)" in result.output
