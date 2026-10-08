"""The CLI's shared database runner and error style (cli/_run.py)."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_env(database_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    import halal_trader.config as config

    monkeypatch.setattr(config, "_settings", None)
    return database_url


def test_run_db_hands_over_a_checked_engine_and_disposes_it(cli_env: str) -> None:
    from sqlalchemy import text

    from halal_trader.cli._run import run_db

    seen = {}

    async def work(engine, settings):  # type: ignore[no-untyped-def]
        seen["engine"] = engine
        async with engine.connect() as conn:
            return (await conn.execute(text("SELECT 1"))).scalar(), settings.database_url

    assert run_db(work) == (1, cli_env)
    assert seen["engine"].pool.checkedout() == 0


def test_fail_ends_the_command_with_status_one() -> None:
    from halal_trader.cli._run import fail

    @click.command()
    def broken() -> None:
        fail("no screen yet")

    result = CliRunner().invoke(broken)
    assert result.exit_code == 1
    assert "Error: no screen yet" in result.output


def test_halal_explain_reaches_the_database(cli_env: str) -> None:
    """It used to import init_db from a package that does not export it."""
    from halal_trader.cli import cli

    result = CliRunner().invoke(cli, ["halal", "explain", "424242"])
    assert result.exit_code == 1
    assert "trade 424242 not found" in result.output
