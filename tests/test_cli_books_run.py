"""`halal-trader books run`: the evening run by hand keeps superseded facts unless asked."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from click.testing import CliRunner

from halal_trader.cli import cli
from halal_trader.events import daily
from halal_trader.research import daily as research_daily
from halal_trader.research.daily import ResearchRun


def test_books_run_drops_superseded_facts_only_when_told_to(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    drops: list[bool] = []

    async def run_research(engine: Any, settings: Any, *, today: date) -> ResearchRun:
        # What the event refresh's facts step would read.
        drops.append(daily._drop_superseded.get())
        return ResearchRun()

    monkeypatch.setattr(research_daily, "run_research", run_research)

    kept = CliRunner().invoke(cli, ["books", "run"])
    assert kept.exit_code == 0, kept.output
    assert "superseded parsers' facts kept" in kept.output

    dropped = CliRunner().invoke(cli, ["books", "run", "--drop-superseded"])
    assert dropped.exit_code == 0, dropped.output
    assert "kept" not in dropped.output

    assert drops == [False, True]
    # Neither run leaves its choice behind.
    assert daily._drop_superseded.get() is True


def test_books_run_help_says_the_drop_deletes_what_the_bot_reads() -> None:
    result = CliRunner().invoke(cli, ["books", "run", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "--drop-superseded" in text
    assert "keeps the facts of superseded parsers unless --drop-superseded" in text
    assert "deletes the facts the bot reads" in text
